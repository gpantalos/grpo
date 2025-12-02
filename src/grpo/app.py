import sys
import time
from datetime import datetime
from pathlib import Path
from typing import cast

import pandas as pd
import streamlit as st
import torch

from grpo.common import (
    DEMO_TRAINING_DEFAULTS,
    EVAL_DEFAULTS,
    GuessingGameEnvironment,
    RewardType,
    RolloutTrace,
    TrainingProgress,
    compute_direction_accuracy_stats,
    format_compact_trace,
    plot_metrics,
)
from grpo.train import (
    build_demo_training_config,
    guessing_game_rollouts,
    init_model,
    run_training,
)

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR.parent.parent / "data"
TRAINING_LOG_ROWS_LIMIT = 16

type RolloutResults = tuple[list[float], list[int], list[RolloutTrace], list[bool], int]


def make_rollout_environment(target: int) -> GuessingGameEnvironment:
    return GuessingGameEnvironment(min_val=EVAL_DEFAULTS.min_val, max_val=EVAL_DEFAULTS.max_val, target=target)


def discover_weights() -> list[tuple[str, str]]:
    return [
        ("pre-trained" if path.name == "weights.pt" else f"post-trained (`{path.name}`)", str(path))
        for path in sorted(DATA_DIR.glob("*.pt"), key=lambda path: (path.name != "weights.pt", path.name))
    ]


def _clear_session(*keys: str) -> None:
    for key in keys:
        st.session_state.pop(key, None)


def _init_caches() -> None:
    if "rollout_cache" not in st.session_state:
        st.session_state["rollout_cache"] = {}
    if "training_cache" not in st.session_state:
        st.session_state["training_cache"] = {}


def _clear_caches_for_path(path: str) -> None:
    st.session_state.get("rollout_cache", {}).pop(path, None)
    st.session_state.get("training_cache", {}).pop(path, None)


@st.cache_resource(show_spinner=False)
def load_model(weights_path: str) -> tuple:
    device = torch.device("cpu")
    model, tokenizer = init_model(checkpoint_path=weights_path, device=device)
    return model.to(device).eval(), tokenizer


def _fmt_pct(val: float | None) -> str:
    return "N/A" if val is None or pd.isna(val) else f"{val * 100:.0f}%"


def _progress_row(p: TrainingProgress) -> dict[str, str]:
    return {
        "step": f"{p.current_step}/{p.total_steps}",
        "loss": f"{p.loss:.8f}",
        "KL divergence": f"{p.kl:.4f}",
        "success rate": _fmt_pct(p.success_rate),
        "reward": f"{p.mean_return:.4f}" if p.mean_return > 0 else "—",
        "direction accuracy": _fmt_pct(p.direction_acc),
    }


def _resolve_default_weight_index(paths: tuple[str, ...], selected_path: str | None) -> int:
    return next((i for i, path in enumerate(paths) if path == selected_path), 0)


def _build_rollout_metrics(
    rewards: list[float],
    guess_counts: list[int],
    traces: list[RolloutTrace],
    successes: list[bool],
) -> list[tuple[str, str, str]]:
    direction_acc, _ = compute_direction_accuracy_stats(traces)
    successful_guess_counts = [count for count, success in zip(guess_counts, successes, strict=True) if success]
    avg_guesses = f"{sum(successful_guess_counts) / len(successful_guess_counts):.1f}" if successful_guess_counts else "N/A"
    return [
        (
            "Success rate",
            f"{sum(successes)}/{len(traces)}",
            "Number of rollouts where the model correctly guessed the target number.",
        ),
        (
            "Direction accuracy",
            _fmt_pct(direction_acc),
            "Fraction of consecutive guess pairs where the model correctly followed higher/lower feedback.",
        ),
        (
            "Average guesses",
            avg_guesses,
            "Average number of turns (guesses + hints) per successful rollout.",
        ),
    ]


def _build_rollout_rows(
    traces: list[RolloutTrace],
    guess_counts: list[int],
    rollout_target: int,
) -> list[dict[str, object]]:
    return [
        {"target": rollout_target, "turns": count, "trace": format_compact_trace(trace.turns) or "(no guesses)"}
        for trace, count in zip(traces, guess_counts, strict=True)
    ]


def _render_training_charts(stats: pd.DataFrame, log_rows: list[dict[str, str]]) -> None:
    if log_rows:
        st.table(pd.DataFrame(log_rows))
    fig = plot_metrics(stats)
    if fig is None:
        return
    st.pyplot(fig, clear_figure=True)


def _select_weights() -> str:
    weight_options = discover_weights()
    if not weight_options:
        st.error(f"No .pt files found in {DATA_DIR}.")
        st.stop()

    labels, paths = zip(*weight_options, strict=True)
    weights_col, delete_col = st.columns([3, 1])
    with weights_col:
        selected_label = st.radio(
            "Weights",
            options=list(labels),
            index=_resolve_default_weight_index(paths, st.session_state.get("selected_weights_path")),
            key="weights_selector",
            horizontal=True,
        )
    weights_path = dict(zip(labels, paths, strict=True))[selected_label]
    st.session_state["selected_weights_path"] = weights_path

    with delete_col:
        is_pretrained = Path(weights_path).name == "weights.pt"
        if (
            st.button(
                "Delete",
                disabled=is_pretrained,
                help="Delete the selected weights file (pre-trained weights cannot be deleted)",
                key="delete_weights",
            )
            and not is_pretrained
        ):
            Path(weights_path).unlink(missing_ok=True)
            load_model.clear()
            _clear_caches_for_path(weights_path)
            _clear_session("weights_selector", "selected_weights_path")
            st.rerun()
    return weights_path


def _render_rollouts_panel(weights_path: str) -> None:
    st.subheader("Rollouts")
    target_col, rollout_col = st.columns(2)
    with target_col:
        target = st.number_input(
            "Target",
            EVAL_DEFAULTS.min_val,
            EVAL_DEFAULTS.max_val,
            EVAL_DEFAULTS.target,
            1,
            key="eval_target",
        )
    with rollout_col:
        num_rollouts = st.number_input("Rollouts", 1, 500, EVAL_DEFAULTS.num_rollouts, 5, key="eval_rollouts")

    if st.button("Run"):
        with st.spinner("Running rollouts..."):
            model, tokenizer = load_model(weights_path)
            _, returns, _, traces, successes = guessing_game_rollouts(
                model=model,
                tokenizer=tokenizer,
                env=make_rollout_environment(target),
                num_rollouts=num_rollouts,
                max_len=200,
                reward_type=EVAL_DEFAULTS.reward_type,
            )
        _init_caches()
        st.session_state["rollout_cache"][weights_path] = (
            returns.squeeze(1).tolist(),
            [trace.turn_count for trace in traces],
            traces,
            successes.squeeze(1).tolist(),
            target,
        )

    _init_caches()
    rollout_data: RolloutResults | None = st.session_state["rollout_cache"].get(weights_path)
    if not rollout_data:
        return

    rewards, guess_counts, traces = rollout_data[:3]
    if len(rollout_data) == 5:
        successes, rollout_target = rollout_data[3], rollout_data[4]
    else:
        successes = [reward > 0 for reward in rewards]
        rollout_target = rollout_data[3] if len(rollout_data) == 4 else target
    for col, (label, value, help_text) in zip(
        st.columns(3),
        _build_rollout_metrics(rewards, guess_counts, traces, successes),
        strict=True,
    ):
        with col:
            st.metric(label, value, help=help_text)
    st.table(pd.DataFrame(_build_rollout_rows(traces, guess_counts, rollout_target)))


def _render_training_panel(weights_path: str) -> None:
    st.subheader("Training")
    tp1, tp2, tp3, tp4 = st.columns(4)
    with tp1:
        train_steps = st.number_input("Steps", 1, 1000, DEMO_TRAINING_DEFAULTS.num_steps, 1, key="train_steps")
    with tp2:
        train_reward = st.radio(
            "Reward Function",
            ["binary", "dense"],
            ["binary", "dense"].index(DEMO_TRAINING_DEFAULTS.reward_type),
            key="train_reward",
            horizontal=True,
            help="binary: 1.0 on success, 0.0 on failure. dense: success bonus scaled by guess count (fewer guesses = higher reward).",
        )
    with tp3:
        train_kl_weight = st.number_input(
            "KL weight",
            0.0,
            1.0,
            DEMO_TRAINING_DEFAULTS.kl_weight,
            0.001,
            key="train_kl_weight",
            help="Penalty coefficient for KL divergence from reference policy.",
        )
    with tp4:
        train_seed = st.number_input("Seed", 0, 2**31 - 1, DEMO_TRAINING_DEFAULTS.seed, 1, key="train_seed")

    if st.button("Train"):
        config = build_demo_training_config(
            checkpoint_path=weights_path,
            output_checkpoint_path=str(DATA_DIR / f"weights-{datetime.now().strftime('%Y%m%d-%H%M%S')}.pt"),
            num_steps=train_steps,
            seed=train_seed,
            reward_type=cast(RewardType, train_reward),
            kl_weight=train_kl_weight,
        )
        eta_ph, progress_bar, log_ph = st.empty(), st.progress(0.0), st.empty()
        log_rows: list[dict[str, str]] = []
        start = time.time()

        def on_progress(progress: TrainingProgress) -> None:
            eta = (
                (time.time() - start) / progress.current_step * (progress.total_steps - progress.current_step)
                if progress.current_step
                else None
            )
            eta_ph.caption(f"ETA: {int(eta)}s" if eta else "ETA: —")
            progress_bar.progress(progress.current_step / progress.total_steps)
            log_rows.append(_progress_row(progress))
            with log_ph.container():
                st.table(pd.DataFrame(log_rows[-TRAINING_LOG_ROWS_LIMIT:]))

        try:
            with st.spinner("Training..."):
                result = run_training(config, progress_callback=on_progress, device=torch.device("cpu"))
        except Exception as exc:
            eta_ph.empty()
            progress_bar.empty()
            st.error(str(exc))
        else:
            progress_bar.progress(1.0)
            saved = str(result.checkpoint_path) if result.checkpoint_path else weights_path
            _init_caches()
            st.session_state["training_cache"][saved] = {
                "stats": result.stats.to_dict(orient="records"),
                "log": log_rows,
            }
            st.session_state["selected_weights_path"] = saved
            _clear_session("weights_selector")
            st.rerun()

    _init_caches()
    cached = st.session_state["training_cache"].get(weights_path)
    if cached:
        _render_training_charts(pd.DataFrame(cached["stats"]), cached["log"])


def main() -> None:
    st.set_page_config(page_title="Guessing Game", layout="wide")
    _init_caches()
    st.markdown(
        "A small language model learns to guess a hidden number. It hears higher or lower after each guess, "
        "and can ask for an even/odd hint (🅴/🅾) along the way."
    )
    weights_path = _select_weights()
    rollout_col, training_col = st.columns(2)
    with rollout_col:
        _render_rollouts_panel(weights_path)
    with training_col:
        _render_training_panel(weights_path)


def run() -> None:
    """Launch the Streamlit app. Used by the `uv run app` entry point."""
    from streamlit.web.cli import main as stcli_main

    sys.argv = ["streamlit", "run", str(APP_DIR / "app.py")]
    stcli_main(prog_name="streamlit")


if __name__ == "__main__":
    main()
