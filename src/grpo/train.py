import argparse
import copy
import dataclasses
import inspect
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
import tiktoken
import torch
import torch.nn as nn
import torch.optim as optim
from rich.console import Console
from rich.table import Table
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset

from grpo.common import (
    DEMO_TRAINING_DEFAULTS,
    GuessingGameEnvironment,
    RewardType,
    RolloutTrace,
    RolloutTurn,
    TrainingProgress,
    compute_direction_accuracy_stats,
    format_compact_trace,
    plot_metrics,
    save_plot,
)

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"

GPU_DEVICE = torch.device("cuda", 0)
CPU_DEVICE = torch.device("cpu")
GPU_DEVICE = CPU_DEVICE

DATA_DIR: Path = Path(__file__).resolve().parent.parent.parent / "data"

MODEL_PATH: Path = DATA_DIR / "weights.pt"
assert MODEL_PATH.exists(), f"Model path {MODEL_PATH} does not exist"

VOCAB_PATH: Path = DATA_DIR / "vocab.json"
assert VOCAB_PATH.exists(), f"Vocab path {VOCAB_PATH} does not exist"

with VOCAB_PATH.open("r") as f:
    VOCAB: dict[str, object] = json.load(f)

DEFAULT_CHECKPOINT_PATH = str(MODEL_PATH)
_DEFAULT_CHECKPOINT = object()
EXPERIENCE_BATCH_FIELDS = (
    "sequences",
    "action_log_probs",
    "log_probs_ref",
    "returns",
    "advantages",
    "action_mask",
    "kl",
)
CONSOLE = Console()


@dataclass
class GuessCandidate:
    guess_text: str
    token_ids: tuple[int, ...]


@dataclass
class RolloutSummaryRow:
    game: int
    rollout: int
    target: int
    reward: float
    turns: int
    trace: str


class CustomVocabTokenizer:
    """Tokenizer wrapper that handles the custom vocabulary mapping."""

    def __init__(self, vocab_data=VOCAB):
        """Initialize with vocabulary mapping."""
        # Load the original GPT-2 tokenizer
        self.base_tokenizer = tiktoken.get_encoding("gpt2")

        # Create mappings (convert string keys to int)
        self.old_to_new = {int(k): v for k, v in vocab_data["old_to_new"].items()}
        self.new_to_old = {int(k): v for k, v in vocab_data["new_to_old"].items()}
        self.vocab_size = vocab_data["vocab_size"]

        # Token 0 is reserved for padding
        self.padding_token = 0

    def _normalize_tokens(self, tokens: list[int] | torch.Tensor) -> list[int]:
        if isinstance(tokens, torch.Tensor):
            return tokens.tolist()
        return tokens

    def encode(self, text: str) -> list[int]:
        """Encode text to tokens."""
        return [self.old_to_new.get(token, self.padding_token) for token in self.base_tokenizer.encode(text)]

    def decode(self, tokens: list[int] | torch.Tensor) -> str:
        """Decode tokens back to text."""
        old_tokens = [
            self.new_to_old[token]
            for token in self._normalize_tokens(tokens)
            if token != self.padding_token and token in self.new_to_old
        ]
        return self.base_tokenizer.decode(old_tokens)

    def decode_single_token_bytes(self, token: int) -> bytes:
        """Decode a single token to bytes."""
        if token == self.padding_token:
            return b""  # Return empty bytes for padding token
        return self.base_tokenizer.decode_single_token_bytes(self.new_to_old[token])

    def decode_single_token(self, token: int) -> str:
        """Decode a single token to string."""
        return self.decode_single_token_bytes(token).decode("utf-8", errors="ignore")

    def _batched_decode_to_token_bytes(
        self, tokens_batch: torch.Tensor, strip_padding: bool = True
    ) -> list[list[bytes]]:
        results = [[self.decode_single_token_bytes(x.item()) for x in seq] for seq in tokens_batch]
        if not strip_padding:
            return results

        trimmed_results = []
        for seq, orig_seq in zip(results, tokens_batch, strict=True):
            orig_list = orig_seq.tolist()
            while orig_list and orig_list[-1] == self.padding_token:
                orig_list.pop()
            trimmed_results.append(seq[: len(orig_list)])
        return trimmed_results

    def batched_decode_to_token_strs(self, tokens_batch: torch.Tensor, strip_padding: bool = True) -> list[list[str]]:
        """Decode a batch of tokens to a list of list of strings representing each token."""
        return [
            [x.decode("utf-8", errors="ignore") for x in seq]
            for seq in self._batched_decode_to_token_bytes(tokens_batch, strip_padding=strip_padding)
        ]


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print("warning: using slow attention. flash attention requires pytorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True,
            )
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        self.transformer.wte.weight = self.lm_head.weight  # https://paperswithcode.com/method/weight-tying

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fm" % (self.get_num_params() / 1e6,))

    def get_num_params(self, non_embedding=True):
        """Return the number of parameters in the model."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, output_all_logits=False):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        )
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)

        # forward the GPT model itself
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None or output_all_logits:
            # if we are given some desired targets also calculate the loss
            # or if we explicitly want all logits
            logits = self.lm_head(x)
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            else:
                loss = None
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, "bias"):
                block.attn.bias = block.attn.bias[:, :, :block_size, :block_size]

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused adamw: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS"""
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd // cfg.n_head, cfg.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0 / dt)  # per second
        flops_promised = 312e12  # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Generate max_new_tokens by feeding predictions back into the model."""
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


def init_model(
    checkpoint_path: str | Path | None | object = _DEFAULT_CHECKPOINT,
    config: GPTConfig = GPTConfig(
        n_layer=2,
        n_head=2,
        n_embd=64,
        vocab_size=50257,
        block_size=1024,
        bias=True,
    ),
    device: torch.device = torch.device("cuda"),
) -> tuple[GPT, CustomVocabTokenizer]:
    """Initialize or load a GPT model."""
    tokenizer = CustomVocabTokenizer()
    if checkpoint_path is _DEFAULT_CHECKPOINT:
        path = MODEL_PATH
    elif checkpoint_path is None:
        path = None
    else:
        path = Path(checkpoint_path)

    if path is not None and path.exists():
        # Load from checkpoint (saved by scripts/download_weights.py)
        checkpoint = torch.load(path, map_location=device)
        gptconf = GPTConfig(**checkpoint["model_args"])
        model = GPT(gptconf)
        state_dict = checkpoint["model"]
        # Remove unwanted prefix if present
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        print(f"loaded model from {path}")
    else:
        # Initialize new model (run scripts/download_weights.py to fetch pretrained weights)
        model = GPT(config)
        if path is None:
            print("initialized new model")
        else:
            print(f"weights not found at {path}; initialized new model.")

    return model, tokenizer


def init_rng(seed: int) -> random.Random:
    """Initialize random number generators."""
    random.seed(seed)
    try:
        torch.manual_seed(seed)
    except Exception as e:
        print(f"warning: failed to set manual seed: {e}, ignoring")
    return random.Random(seed)


@dataclass
class TrainingRunResult:
    stats: pd.DataFrame
    checkpoint_path: Path | None = None


def save_checkpoint(
    model: "GPT",
    path: str | Path,
    stats: pd.DataFrame,
    step: int,
) -> Path:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_name(f"{checkpoint_path.name}.tmp")

    checkpoint = {
        "model": model.state_dict(),
        "model_args": dataclasses.asdict(model.config),
        "optimizer": None,
        "stats": stats.to_dict(orient="records"),
        "step": step,
    }
    torch.save(checkpoint, tmp_path)
    tmp_path.replace(checkpoint_path)
    return checkpoint_path


def summarize_rollouts(
    game_idx: int,
    env: GuessingGameEnvironment,
    traces: list[RolloutTrace],
    returns: torch.Tensor,
) -> list[RolloutSummaryRow]:
    summaries: list[RolloutSummaryRow] = []
    if game_idx < 2:
        for i, (trace, reward) in enumerate(zip(traces[:2], returns[:2], strict=True)):  # Show first 2 rollouts
            compact = format_compact_trace(trace.turns) or "(no guesses)"
            summaries.append(
                RolloutSummaryRow(
                    game=game_idx + 1,
                    rollout=i + 1,
                    target=env.target,
                    reward=reward.item(),
                    turns=trace.turn_count,
                    trace=compact,
                )
            )
    return summaries


def render_rollout_summary(step: int, total_steps: int, rollout_summaries: list[RolloutSummaryRow]) -> None:
    if not rollout_summaries:
        return

    table = Table()
    table.add_column("game", no_wrap=True, justify="right")
    table.add_column("rollout", no_wrap=True, justify="right")
    table.add_column("target", no_wrap=True, justify="right")
    table.add_column("reward", no_wrap=True, justify="right")
    table.add_column("turns", no_wrap=True, justify="right")
    table.add_column("trace")

    for row in rollout_summaries:
        table.add_row(
            str(row.game),
            str(row.rollout),
            str(row.target),
            f"{row.reward:.2f}",
            str(row.turns),
            row.trace,
        )

    CONSOLE.print(table)


def render_stats_table(df: pd.DataFrame) -> None:
    """Render a pandas DataFrame as a rich table."""
    if df.empty:
        return
    table = Table()
    for col in df.columns:
        table.add_column(str(col), no_wrap=True, justify="right")
    for _, row in df.iterrows():
        cells: list[str] = []
        for col, v in zip(df.columns, row, strict=True):
            if pd.isna(v):
                cells.append("n/a")
            elif col == "step":
                cells.append(str(int(v)))
            elif col == "loss":
                cells.append(f"{v:.8f}")
            elif isinstance(v, float):
                cells.append(f"{v:.4f}")
            else:
                cells.append(str(v))
        table.add_row(*cells)
    CONSOLE.print(table)


@dataclasses.dataclass
class Experience:
    """Container for a batch of rollout experiences."""

    sequences: torch.Tensor
    action_log_probs: torch.Tensor
    log_probs_ref: torch.Tensor
    returns: torch.Tensor | None
    advantages: torch.Tensor | None
    action_mask: torch.Tensor
    kl: torch.Tensor | None = None

    def to(self, device: torch.device) -> "Experience":
        """Move all tensors to specified device."""
        members = {}
        for field in dataclasses.fields(self):
            v = getattr(self, field.name)
            if isinstance(v, torch.Tensor):
                v = v.to(device=device)
            members[field.name] = v
        return Experience(**members)


def _pad_tensor_batch(
    tensors: list[torch.Tensor],
    side: str = "right",
    pad_value: int | float | bool = 0,
    stack: bool = True,
) -> torch.Tensor | list[torch.Tensor]:
    assert side in ("left", "right")
    max_len = max(tensor.size(-1) for tensor in tensors)
    padded_tensors = []
    for tensor in tensors:
        pad_len = max_len - tensor.size(-1)
        padding = (pad_len, 0) if side == "left" else (0, pad_len)
        padded_tensors.append(F.pad(tensor, padding, value=pad_value) if pad_len > 0 else tensor)
    return torch.stack(padded_tensors, dim=0) if stack else padded_tensors


def zero_pad_sequences(sequences: list[torch.Tensor], side: str = "left") -> torch.Tensor:
    """Pad sequences to same length with zeros."""
    return _pad_tensor_batch(sequences, side=side, pad_value=0)


def split_experience_batch(experience: Experience) -> list[Experience]:
    """Split a batched experience into individual experiences."""
    batch_size = experience.sequences.size(0)
    batch_data = [{} for _ in range(batch_size)]
    for key in EXPERIENCE_BATCH_FIELDS:
        value = getattr(experience, key)
        vals = [None] * batch_size if value is None else torch.unbind(value)
        assert batch_size == len(vals)
        for i, v in enumerate(vals):
            batch_data[i][key] = v

    return [Experience(**data) for data in batch_data]


def join_experience_batch(items: list[Experience]) -> Experience:
    """Join individual experiences into a batched experience."""
    batch_data = {}
    for key in EXPERIENCE_BATCH_FIELDS:
        vals = [getattr(item, key) for item in items]
        if all(v is not None for v in vals):
            data = _pad_tensor_batch(vals, side="left", pad_value=False if key == "action_mask" else 0)
        else:
            data = None
        batch_data[key] = data
    return Experience(**batch_data)


class ReplayBuffer(Dataset):
    """Experience replay buffer for GRPO training."""

    def __init__(self, limit: int = 0) -> None:
        """Initialize replay buffer."""
        self.limit = limit
        self.items: list[Experience] = []

    def append(self, experience: Experience) -> None:
        """Add new experience to buffer."""
        items = split_experience_batch(experience)
        self.items.extend(items)
        if self.limit > 0:
            samples_to_remove = len(self.items) - self.limit
            if samples_to_remove > 0:
                self.items = self.items[samples_to_remove:]

    def clear(self) -> None:
        """Clear all experiences from buffer."""
        self.items.clear()

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Experience:
        return self.items[idx]


@torch.no_grad()
def rollout(
    env: GuessingGameEnvironment,
    model: GPT,
    tokenizer: CustomVocabTokenizer,
    max_len: int,
    reward_type: RewardType = "binary",
    device: torch.device = GPU_DEVICE,
) -> tuple[torch.Tensor, float, torch.Tensor, RolloutTrace]:
    """Generate a single rollout of the guessing game."""
    model.eval()

    prompt = env.get_initial_prompt()
    sequence = tokenizer.encode(prompt)
    action_mask = [0 for _ in range(len(sequence))]
    turns: list[RolloutTurn] = []

    is_done = False
    while not is_done and len(sequence) < max_len:
        prev_length = len(sequence)
        guess_text = sample_guess(
            model=model,
            tokenizer=tokenizer,
            sequence=sequence,
            max_len=max_len,
        )
        new_tokens = len(sequence) - prev_length
        action_mask.extend([1 for _ in range(new_tokens)])

        if guess_text is None:
            break

        response = env.process_guess(guess_text)
        is_done = response.done
        turns.append(
            RolloutTurn(
                guess_text=guess_text,
                numeric_guess=response.numeric_guess,
                feedback=response.feedback,
            )
        )

        response_tokens = tokenizer.encode(response.response_text)
        action_mask.extend([0 for _ in range(len(response_tokens))])
        sequence.extend(response_tokens)

    reward = env.compute_reward(reward_type)

    sequence_tensor = torch.tensor(sequence, dtype=torch.long, device=device)
    action_mask = torch.tensor(action_mask, dtype=torch.bool, device=device)
    assert len(sequence_tensor) == len(action_mask)

    trace = RolloutTrace(
        transcript=tokenizer.decode(sequence),
        turns=tuple(turns),
        success=env.success,
    )

    return sequence_tensor, reward, action_mask, trace


def sample_guess(
    model: GPT,
    tokenizer: CustomVocabTokenizer,
    sequence: list[int],
    max_len: int,
    top_k: int = 50,
    temperature: float = 1.0,
    device: torch.device = GPU_DEVICE,
) -> str | None:
    """Generate a single guess from the model, mutating sequence in place."""
    input_ids = torch.tensor(
        sequence,
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    added_tokens: list[int] = []

    recent_text = ""
    while "</guess>" not in recent_text and len(input_ids[0]) < max_len:
        logits, _ = model(input_ids)
        logits = logits[:, -1, :] / temperature

        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("Inf")

        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        next_int = int(next_token.item())
        sequence.append(next_int)
        added_tokens.append(next_int)
        input_ids = torch.cat([input_ids, next_token], dim=1)

        recent_text = tokenizer.decode(added_tokens)
        if "</guess>" in recent_text:
            break

    full_text = tokenizer.decode(added_tokens)
    if "</guess>" in full_text:
        guess_text = full_text.split("<guess>")[-1].split("</guess>")[0]
        return guess_text
    return None


@torch.no_grad()
def guessing_game_rollouts(
    model: GPT,
    tokenizer: CustomVocabTokenizer,
    env: GuessingGameEnvironment,
    num_rollouts: int,
    max_len: int,
    reward_type: RewardType = "binary",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[RolloutTrace], torch.Tensor]:
    """Generate `num_rollouts` rollouts of the guessing game."""
    device = next(model.parameters()).device

    all_sequences = []
    all_action_masks = []
    all_returns = []
    all_traces = []
    all_successes = []

    for _ in range(num_rollouts):
        this_env = env.copy()
        sequence, reward, action_mask, trace = rollout(
            env=this_env,
            model=model,
            tokenizer=tokenizer,
            max_len=max_len,
            reward_type=reward_type,
        )

        all_sequences.append(sequence)
        all_action_masks.append(action_mask)
        all_returns.append(reward)
        all_traces.append(trace)
        all_successes.append(trace.success)

    sequences = _pad_tensor_batch(all_sequences, side="right", pad_value=0).to(device)
    action_masks = _pad_tensor_batch(all_action_masks, side="right", pad_value=False).to(device)
    returns = torch.tensor(all_returns, dtype=torch.float, device=device).unsqueeze(1)
    successes = torch.tensor(all_successes, dtype=torch.bool, device=device).unsqueeze(1)

    return sequences, returns, action_masks, all_traces, successes


def kl_divergence(
    log_probs: torch.Tensor, log_probs_ref: torch.Tensor, action_mask: torch.Tensor | None
) -> torch.Tensor:
    """Compute the k3 estimator of KL(pi_theta, pi_ref)."""
    log_ratio = log_probs_ref.float() - log_probs.float()
    if action_mask is not None:
        log_ratio = log_ratio * action_mask[:, 1:]

    return log_ratio.exp() - log_ratio - 1


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor | None, dim: int | None = None) -> torch.Tensor:
    """Compute mean of tensor with optional masking."""
    if mask is None:
        return tensor.mean(dim=dim)
    return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim)


def summarize_return_metrics(
    rollout_returns: list[torch.Tensor], rollout_successes: list[torch.Tensor]
) -> tuple[float, float]:
    """Compute mean return and success rate for a training step."""
    all_returns = torch.cat([r.flatten() for r in rollout_returns])
    all_successes = torch.cat([s.flatten() for s in rollout_successes])
    return all_returns.mean().item(), all_successes.float().mean().item()


def group_advantages(returns: torch.Tensor) -> torch.Tensor:
    """Normalize advantages within a group of rollouts."""
    return (returns - returns.mean()) / (returns.std() + 1e-6)


def sequences_log_probs(model: GPT, sequence_ids: torch.Tensor) -> torch.Tensor:
    """Compute logprobs for sequences."""
    logits, _ = model(sequence_ids, output_all_logits=True)

    all_log_probs = F.log_softmax(logits[:, :-1], dim=-1)
    log_probs = all_log_probs.gather(dim=-1, index=sequence_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return log_probs


class GRPOLoss(nn.Module):
    def __init__(self, clip_eps: float, kl_weight: float) -> None:
        super().__init__()
        self.clip_eps = clip_eps
        self.kl_weight = kl_weight

    def forward(self, log_probs: torch.Tensor, experience: Experience) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute GRPO loss."""
        old_log_probs = experience.action_log_probs
        log_probs_ref = experience.log_probs_ref
        action_mask = experience.action_mask
        advantages = experience.advantages

        kl = kl_divergence(
            log_probs=log_probs,
            log_probs_ref=log_probs_ref,
            action_mask=action_mask,
        )

        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        loss = -torch.min(surr1, surr2) + self.kl_weight * kl

        loss = masked_mean(loss, action_mask[:, 1:], dim=-1).mean()
        return loss, kl.mean()


@torch.no_grad()
def _collect_training_rollout_groups(
    model: GPT,
    tokenizer: CustomVocabTokenizer,
    c: "Config",
    rng: random.Random,
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[torch.Tensor],
    list[RolloutSummaryRow],
    float | None,
]:
    rollout_returns: list[torch.Tensor] = []
    rollout_successes: list[torch.Tensor] = []
    all_sequences: list[torch.Tensor] = []
    all_returns: list[torch.Tensor] = []
    all_action_masks: list[torch.Tensor] = []
    all_traces: list[list[RolloutTrace]] = []
    rollout_summaries: list[RolloutSummaryRow] = []

    for game_idx in range(c.rollouts_per_step):
        env = GuessingGameEnvironment.random(rng)
        sequence_ids, returns, action_mask, traces, successes = guessing_game_rollouts(
            model=model,
            tokenizer=tokenizer,
            env=env,
            num_rollouts=c.group_size,
            max_len=c.max_length,
            reward_type=c.reward_type,
        )
        rollout_returns.append(returns.cpu())
        rollout_successes.append(successes.cpu())
        all_sequences.append(sequence_ids)
        all_returns.append(returns)
        all_action_masks.append(action_mask)
        all_traces.append(traces)

        if game_idx < 2:
            rollout_summaries.extend(summarize_rollouts(game_idx, env, traces, returns))

    flat_traces = [trace for group in all_traces for trace in group]
    direction_acc_value, _ = compute_direction_accuracy_stats(flat_traces)
    return (
        rollout_returns,
        rollout_successes,
        all_sequences,
        all_returns,
        all_action_masks,
        rollout_summaries,
        direction_acc_value,
    )


def _pad_rollout_groups(
    sequences: list[torch.Tensor],
    action_masks: list[torch.Tensor],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    return (
        _pad_tensor_batch(sequences, side="right", pad_value=0, stack=False),
        _pad_tensor_batch(action_masks, side="right", pad_value=False, stack=False),
    )


def _append_replay_experiences(
    replay_buffer: ReplayBuffer,
    padded_sequences: list[torch.Tensor],
    returns_by_group: list[torch.Tensor],
    padded_action_masks: list[torch.Tensor],
    all_log_probs: torch.Tensor,
    all_log_probs_ref: torch.Tensor,
) -> None:
    start_idx = 0
    for sequence_ids, returns, action_mask in zip(
        padded_sequences,
        returns_by_group,
        padded_action_masks,
        strict=True,
    ):
        batch_size = sequence_ids.shape[0]
        end_idx = start_idx + batch_size
        log_probs = all_log_probs[start_idx:end_idx]
        log_probs_ref = all_log_probs_ref[start_idx:end_idx]
        kl = kl_divergence(
            log_probs=log_probs,
            log_probs_ref=log_probs_ref,
            action_mask=action_mask,
        )
        replay_buffer.append(
            Experience(
                sequences=sequence_ids,
                action_log_probs=log_probs,
                log_probs_ref=log_probs_ref,
                returns=returns,
                advantages=group_advantages(returns),
                action_mask=action_mask,
                kl=kl,
            ).to(CPU_DEVICE)
        )
        start_idx = end_idx


def _run_training_epoch(
    model: GPT,
    replay_buffer: ReplayBuffer,
    optimizer: optim.Optimizer,
    objective: GRPOLoss,
    c: "Config",
) -> tuple[float, float] | None:
    step_losses: list[float] = []
    step_kls: list[float] = []

    experience_sampler = DataLoader(
        replay_buffer,
        batch_size=c.train_batch_size,
        shuffle=True,
        drop_last=False,
        collate_fn=join_experience_batch,
    )

    for batch_idx, exp in enumerate(experience_sampler):
        exp = exp.to(GPU_DEVICE)

        optimizer.zero_grad()
        log_probs = sequences_log_probs(model, sequence_ids=exp.sequences)
        loss, kl = objective(log_probs=log_probs, experience=exp)
        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=c.max_norm)
        optimizer.step()

        step_losses.append(loss.item())
        step_kls.append(kl.item())

    if not step_losses:
        return None

    return sum(step_losses) / len(step_losses), sum(step_kls) / len(step_kls)


def train(
    model: GPT,
    tokenizer: CustomVocabTokenizer,
    c: "Config",
    rng: random.Random,
    progress_callback: Callable[[TrainingProgress], None] | None = None,
) -> pd.DataFrame:
    reference_model = copy.deepcopy(model)
    reference_model.eval()
    model.eval()

    optimizer = optim.Adam(model.parameters(), lr=c.lr)

    stats = []
    replay_buffer = ReplayBuffer()
    objective = GRPOLoss(clip_eps=c.clip_eps, kl_weight=c.kl_weight)

    for k in range(c.num_steps):
        replay_buffer.clear()

        with torch.no_grad():
            (
                rollout_returns,
                rollout_successes,
                all_sequences,
                all_returns,
                all_action_masks,
                rollout_summaries,
                direction_acc_value,
            ) = _collect_training_rollout_groups(
                model=model,
                tokenizer=tokenizer,
                c=c,
                rng=rng,
            )
            padded_sequences, padded_action_masks = _pad_rollout_groups(all_sequences, all_action_masks)
            flat_sequences = torch.cat(padded_sequences, dim=0)
            all_log_probs = sequences_log_probs(model=model, sequence_ids=flat_sequences)
            all_log_probs_ref = sequences_log_probs(model=reference_model, sequence_ids=flat_sequences)
            _append_replay_experiences(
                replay_buffer,
                padded_sequences=padded_sequences,
                returns_by_group=all_returns,
                padded_action_masks=padded_action_masks,
                all_log_probs=all_log_probs,
                all_log_probs_ref=all_log_probs_ref,
            )

        episode_mean_return, success_rate = summarize_return_metrics(rollout_returns, rollout_successes)
        direction_acc = direction_acc_value if direction_acc_value is not None else math.nan
        epoch_metrics = _run_training_epoch(
            model=model,
            replay_buffer=replay_buffer,
            optimizer=optimizer,
            objective=objective,
            c=c,
        )
        if epoch_metrics is None:
            continue
        loss, kl = epoch_metrics

        stats.append(
            {
                "step": int(k),
                "loss": loss,
                "kl": kl,
                "mean_return": episode_mean_return,
                "success_rate": success_rate,
                "direction_acc": direction_acc,
            }
        )
        render_rollout_summary(
            step=k + 1,
            total_steps=c.num_steps,
            rollout_summaries=rollout_summaries,
        )
        direction_acc_msg = "n/a (0 pairs)" if direction_acc_value is None else f"{direction_acc_value:.4f}"
        print(
            f"step {k + 1}/{c.num_steps} - loss: {loss:.8f}, kl: {kl:.4f}, "
            f"return: {episode_mean_return:.4f}, "
            f"direction accuracy: {direction_acc_msg}",
        )
        if progress_callback is not None:
            progress_callback(
                TrainingProgress(
                    current_step=k + 1,
                    total_steps=c.num_steps,
                    loss=loss,
                    kl=kl,
                    mean_return=episode_mean_return,
                    success_rate=success_rate,
                    direction_acc=direction_acc_value,
                )
            )

    return pd.DataFrame(stats)


def run_training(
    config: "Config",
    progress_callback: Callable[[TrainingProgress], None] | None = None,
    device: torch.device = GPU_DEVICE,
) -> TrainingRunResult:
    rng = init_rng(config.seed)
    model, tokenizer = init_model(checkpoint_path=config.checkpoint_path, device=device)
    model = model.to(device)

    results = train(
        model=model,
        tokenizer=tokenizer,
        c=config,
        rng=rng,
        progress_callback=progress_callback,
    )
    checkpoint_path = None
    if config.output_checkpoint_path is not None:
        checkpoint_path = save_checkpoint(
            model=model,
            path=config.output_checkpoint_path,
            stats=results,
            step=len(results),
        )

    return TrainingRunResult(stats=results, checkpoint_path=checkpoint_path)


def train_torch(config: "Config"):
    result = run_training(config)
    render_stats_table(result.stats)
    return result.stats


@dataclass
class Config:
    seed: int = 42
    num_steps: int = 20

    checkpoint_path: str | None = DEFAULT_CHECKPOINT_PATH
    output_checkpoint_path: str | None = None
    checkpoint_interval: int = 20
    train_batch_size: int = 32
    lr: float = 3e-5
    kl_weight: float = 0.01
    clip_eps: float = 0.2

    rollouts_per_step: int = 8
    group_size: int = 4
    max_norm: float = 1.0

    max_length: int = 230
    reward_type: RewardType = "binary"


def build_demo_training_config(
    checkpoint_path: str | None = DEFAULT_CHECKPOINT_PATH,
    output_checkpoint_path: str | None = None,
    num_steps: int | None = None,
    seed: int | None = None,
    reward_type: RewardType | None = None,
    kl_weight: float | None = None,
) -> Config:
    return Config(
        checkpoint_path=checkpoint_path,
        output_checkpoint_path=output_checkpoint_path,
        num_steps=DEMO_TRAINING_DEFAULTS.num_steps if num_steps is None else num_steps,
        seed=DEMO_TRAINING_DEFAULTS.seed if seed is None else seed,
        reward_type=DEMO_TRAINING_DEFAULTS.reward_type if reward_type is None else reward_type,
        kl_weight=DEMO_TRAINING_DEFAULTS.kl_weight if kl_weight is None else kl_weight,
        rollouts_per_step=DEMO_TRAINING_DEFAULTS.rollouts_per_step,
        group_size=DEMO_TRAINING_DEFAULTS.group_size,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reward-type", choices=["binary", "dense"], default="binary")
    parser.add_argument("--steps", type=int, default=None, help="Number of training steps")
    return parser.parse_args(argv)


def main(config: Config | None = None) -> None:
    if config is None:
        args = parse_args()
        config = build_demo_training_config(
            reward_type=args.reward_type,
            num_steps=args.steps,
        )
    cfg = config
    result = run_training(cfg)
    render_stats_table(result.stats)
    images_dir = Path("images")
    images_dir.mkdir(parents=True, exist_ok=True)
    fig = plot_metrics(result.stats)
    if fig is not None:
        save_plot(fig, str(images_dir / "metrics.png"))


if __name__ == "__main__":
    main()
