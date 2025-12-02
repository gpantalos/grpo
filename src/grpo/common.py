import copy
import random
from dataclasses import dataclass
from typing import Literal

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")
plt.rcParams["font.family"] = "Helvetica"

type RewardType = Literal["binary", "dense"]
type FeedbackKind = Literal["correct", "higher", "lower", "invalid", "even", "odd"]


@dataclass
class EvalDefaults:
    min_val: int
    max_val: int
    target: int
    num_rollouts: int
    reward_type: RewardType


@dataclass
class DemoTrainingDefaults:
    num_steps: int
    rollouts_per_step: int
    group_size: int
    seed: int
    reward_type: RewardType
    kl_weight: float


EVAL_DEFAULTS = EvalDefaults(min_val=1, max_val=10, target=5, num_rollouts=5, reward_type="binary")

DEMO_TRAINING_DEFAULTS = DemoTrainingDefaults(
    num_steps=10,
    rollouts_per_step=4,
    group_size=4,
    seed=2,
    reward_type="binary",
    kl_weight=0.01,
)

METRIC_LABELS = {
    "mean_return": "mean reward",
    "success_rate": "success rate",
    "direction_acc": "direction accuracy",
}
FEEDBACK_RESPONSE_TEXT: dict[FeedbackKind, str] = {
    "correct": " Env: correct",
    "higher": " Env: higher Model: <guess>",
    "lower": " Env: lower Model: <guess>",
    "invalid": " Env: invalid Model: <guess>",
    "even": " Env: even Model: <guess>",
    "odd": " Env: odd Model: <guess>",
}
COMPACT_TRACE_SYMBOLS: dict[FeedbackKind, str] = {
    "correct": "✅",
    "higher": "⬆️",
    "lower": "⬇️",
    "invalid": "❌",
    "even": "🅴",
    "odd": "🅾",
}


@dataclass
class RolloutTurn:
    guess_text: str
    numeric_guess: int | None
    feedback: FeedbackKind


@dataclass
class RolloutTrace:
    transcript: str
    turns: tuple[RolloutTurn, ...]
    success: bool

    @property
    def numeric_guess_count(self) -> int:
        return sum(turn.numeric_guess is not None for turn in self.turns)

    @property
    def turn_count(self) -> int:
        """Number of turns (guesses + hints)."""
        return len(self.turns)


def render_feedback_response(feedback: FeedbackKind) -> str:
    return FEEDBACK_RESPONSE_TEXT[feedback]


def format_compact_trace(turns: tuple[RolloutTurn, ...]) -> str:
    """Format turns as compact guess sequence, e.g. '6 ↓ 6 ↓ 5 ✓'."""
    return " ".join(
        f"{turn.numeric_guess if turn.numeric_guess is not None else 'H' if turn.guess_text.lower() == 'hint' else turn.guess_text} "
        f"{COMPACT_TRACE_SYMBOLS[turn.feedback]}"
        for turn in turns
    )


def compute_direction_accuracy_stats(traces: list[RolloutTrace]) -> tuple[float | None, int]:
    """Return (accuracy, count) for higher/lower-following transitions."""
    correct = 0
    total = 0

    for trace in traces:
        for current_turn, next_turn in zip(trace.turns, trace.turns[1:], strict=False):
            if current_turn.feedback not in ("higher", "lower"):
                continue
            if current_turn.numeric_guess is None or next_turn.numeric_guess is None:
                continue

            total += 1
            if current_turn.feedback == "higher" and next_turn.numeric_guess > current_turn.numeric_guess:
                correct += 1
            elif current_turn.feedback == "lower" and next_turn.numeric_guess < current_turn.numeric_guess:
                correct += 1

    if total == 0:
        return None, 0
    return correct / total, total


@dataclass
class GuessResponse:
    feedback: FeedbackKind
    response_text: str
    done: bool
    numeric_guess: int | None


class GuessingGameEnvironment:
    def __init__(self, min_val: int, max_val: int, target: int):
        self.min_val = min_val
        self.max_val = max_val
        self.target = target

        self.success = False
        self.num_guesses = 0

    def _hint_feedback(self) -> FeedbackKind:
        return "even" if self.target % 2 == 0 else "odd"

    def _numeric_feedback(self, guess: int) -> tuple[FeedbackKind, bool]:
        if guess == self.target:
            self.success = True
            return "correct", True
        return ("higher", False) if guess < self.target else ("lower", False)

    def _response(self, feedback: FeedbackKind, done: bool, numeric_guess: int | None) -> GuessResponse:
        return GuessResponse(
            feedback=feedback,
            response_text=render_feedback_response(feedback),
            done=done,
            numeric_guess=numeric_guess,
        )

    def process_guess(self, guess_text: str) -> GuessResponse:
        """Process a guess from the model."""
        if "hint" in guess_text.lower():
            return self._response(self._hint_feedback(), done=False, numeric_guess=None)

        guess_text = guess_text.strip()
        try:
            guess = int(guess_text)
        except ValueError:
            return self._response("invalid", done=False, numeric_guess=None)

        self.num_guesses += 1
        feedback, done = self._numeric_feedback(guess)
        return self._response(feedback, done=done, numeric_guess=guess)

    def compute_reward(self, reward_type: RewardType = "binary") -> float:
        if reward_type == "dense":
            if not self.success:
                return 0.0
            return max(0.1, 1.0 - 0.1 * (self.num_guesses - 1))
        return 1.0 if self.success else 0.0

    def get_initial_prompt(self) -> str:
        return f"System: [{self.min_val}, {self.max_val}] Model: <guess>"

    def valid_guess_texts(self) -> tuple[str, ...]:
        return tuple(str(value) for value in range(self.min_val, self.max_val + 1)) + ("hint",)

    def copy(self) -> "GuessingGameEnvironment":
        return copy.deepcopy(self)

    @classmethod
    def random(cls, rng: random.Random | None) -> "GuessingGameEnvironment":
        rng = rng or random.Random()
        min_val, max_val = rng.choice([(EVAL_DEFAULTS.min_val, EVAL_DEFAULTS.max_val)])
        target = rng.randint(min_val, max_val)
        return GuessingGameEnvironment(min_val=min_val, max_val=max_val, target=target)


@dataclass
class TrainingProgress:
    current_step: int
    total_steps: int
    loss: float
    kl: float
    mean_return: float
    success_rate: float
    direction_acc: float | None


def plot_metrics(stats: pd.DataFrame) -> plt.Figure | None:
    columns = [column for column in METRIC_LABELS if column in stats.columns]
    if not columns:
        return None

    fig, ax = plt.subplots(figsize=(8, 3))
    for column in columns:
        ax.plot(stats["step"], stats[column], label=METRIC_LABELS[column], linewidth=0.5, marker="o", markersize=3)
    ax.set_ylim(0, 1)
    ax.set_xlabel("step")
    ax.set_title("" if len(columns) > 1 else "Return / Direction accuracy")
    for spine in ax.spines.values():
        spine.set_linewidth(0.25)
    ax.tick_params(axis="both", length=0)
    ax.grid(True, alpha=0.3)
    if len(ax.lines) > 1:
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), ncol=len(ax.lines), frameon=False)
    fig.tight_layout()
    return fig


def save_plot(fig: plt.Figure, save_path: str) -> None:
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
