"""Intra-game switchpoint detection: did accuracy change abruptly mid-game?

The forensic premise
--------------------
Honest players maintain roughly consistent accuracy (adjusted for position
difficulty) throughout a game.  Cheaters who activate assistance mid-game show
a step change: the first N moves are played at their real strength, then
accuracy jumps to engine level.  Two common patterns:

1. **Losing trigger**: the player starts cheating when the eval turns against
   them.  The switchpoint coincides with a negative eval swing.
2. **From the start**: the player cheats from move 1.  No switchpoint within
   the game, but uniformly high accuracy is caught by the existing match-rate
   tests.

Detection method
----------------
Scan every possible split point k (minimum segment on each side).  At each k,
fit a two-rate model (match rate before vs after) and compare its log-
likelihood to the single-rate model.  The best split's LLR is tested against
a permutation null -- the scan over k inflates the nominal chi-squared
p-value, and the permutation corrects for it exactly.

The eval at the detected switchpoint is recorded and contributes to the
signal: a jump in accuracy right when the player was losing is far stronger
evidence than a jump at an arbitrary point.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


MIN_SEGMENT = 8   # minimum scoreable moves on each side of a split
MIN_MOVES = 20    # fewer than this and there is no room to detect anything
RATE_JUMP_THRESH = 0.18  # minimum improvement in match rate to flag


@dataclass
class SwitchpointResult:
    """Result of intra-game switchpoint analysis for one player."""

    detected: bool
    reason: str = ""

    # Best split point (index within the scoreable-move arrays)
    split_index: int | None = None
    split_ply: int | None = None
    split_move_number: int | None = None

    # Match rates
    rate_before: float | None = None
    rate_after: float | None = None
    rate_jump: float | None = None

    # Statistical strength
    log_lr: float = 0.0
    p_value: float = 1.0

    # Context at the switch
    eval_at_switch: float | None = None   # centipawns, player's perspective
    losing_at_switch: bool = False
    eval_trend_before: float | None = None  # eval change in the 5 moves before

    # ACPL before and after
    acpl_before: float | None = None
    acpl_after: float | None = None

    n_before: int = 0
    n_after: int = 0
    n_total: int = 0

    caveats: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


# ---------------------------------------------------------------------------
# core algorithm
# ---------------------------------------------------------------------------


def _two_rate_llr(matched: np.ndarray, k: int) -> float:
    """Log-likelihood ratio: two-rate model at split k vs single-rate."""
    n = len(matched)
    eps = 1e-12

    s_all = float(matched.sum())
    p_all = max(eps, min(1 - eps, s_all / n))

    s1 = float(matched[:k].sum())
    s2 = float(matched[k:].sum())
    n1, n2 = k, n - k
    p1 = max(eps, min(1 - eps, s1 / n1))
    p2 = max(eps, min(1 - eps, s2 / n2))

    ll_null = s_all * math.log(p_all) + (n - s_all) * math.log(1 - p_all)
    ll_alt = (
        s1 * math.log(p1) + (n1 - s1) * math.log(1 - p1)
        + s2 * math.log(p2) + (n2 - s2) * math.log(1 - p2)
    )
    return ll_alt - ll_null


def _best_split(matched: np.ndarray) -> tuple[int | None, float]:
    """Find the split point k that maximises the two-rate LLR."""
    n = len(matched)
    best_llr = 0.0
    best_k = None
    for k in range(MIN_SEGMENT, n - MIN_SEGMENT + 1):
        llr = _two_rate_llr(matched, k)
        if llr > best_llr:
            best_llr = llr
            best_k = k
    return best_k, best_llr


def detect_switchpoint(
    matched_t1: np.ndarray,
    delta_cp: np.ndarray,
    eval_before: np.ndarray,
    ply_numbers: np.ndarray,
    move_numbers: np.ndarray,
    color: str,
    n_perm: int = 2000,
    seed: int = 42,
) -> SwitchpointResult:
    """Detect an abrupt accuracy change within a single game.

    All arrays must be aligned to the player's scoreable moves.

    Parameters
    ----------
    matched_t1 : binary, 1 = played the engine's top move
    delta_cp : centipawn loss per move
    eval_before : eval (cp) from the player's perspective before each move
    ply_numbers : absolute ply numbers (for display)
    move_numbers : move numbers (for display)
    color : "white" or "black"
    n_perm : permutations for the p-value
    """
    n = len(matched_t1)
    if n < MIN_MOVES:
        return SwitchpointResult(
            detected=False,
            reason=f"only {n} scoreable moves, need at least {MIN_MOVES}",
            n_total=n,
        )

    matched = np.asarray(matched_t1, dtype=float)
    cpl = np.asarray(delta_cp, dtype=float)
    evals = np.asarray(eval_before, dtype=float)

    best_k, best_llr = _best_split(matched)

    if best_k is None or best_llr < 0.1:
        return SwitchpointResult(
            detected=False,
            reason="no split point improves the fit",
            n_total=n,
        )

    rate_before = float(matched[:best_k].mean())
    rate_after = float(matched[best_k:].mean())
    rate_jump = rate_after - rate_before

    # Only care about accuracy *improving* (the suspicious direction).
    if rate_jump < RATE_JUMP_THRESH:
        return SwitchpointResult(
            detected=False,
            reason=(
                f"best split at move {int(move_numbers[best_k])} shows "
                f"{rate_before:.0%} -> {rate_after:.0%}, below threshold"
            ),
            n_total=n,
        )

    # Permutation test: how often does a random shuffle beat this LLR?
    rng = np.random.default_rng(seed)
    count_ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(matched)
        _, perm_llr = _best_split(perm)
        if perm_llr >= best_llr:
            count_ge += 1
    p_value = (count_ge + 1) / (n_perm + 1)

    # Context at the switch
    eval_at = float(evals[best_k]) if best_k < len(evals) else None
    losing = eval_at is not None and eval_at < -100

    # Eval trend in the moves leading up to the switch
    lookback = min(5, best_k)
    if lookback >= 2:
        recent = evals[best_k - lookback: best_k]
        eval_trend = float(recent[-1] - recent[0])
    else:
        eval_trend = None

    # ACPL before and after
    c1 = cpl[:best_k]
    c2 = cpl[best_k:]
    c1_valid = c1[np.isfinite(c1)]
    c2_valid = c2[np.isfinite(c2)]
    acpl_before = float(c1_valid.mean()) if len(c1_valid) else None
    acpl_after = float(c2_valid.mean()) if len(c2_valid) else None

    result = SwitchpointResult(
        detected=p_value < 0.05,
        split_index=best_k,
        split_ply=int(ply_numbers[best_k]),
        split_move_number=int(move_numbers[best_k]),
        rate_before=rate_before,
        rate_after=rate_after,
        rate_jump=rate_jump,
        log_lr=best_llr,
        p_value=p_value,
        eval_at_switch=eval_at,
        losing_at_switch=losing,
        eval_trend_before=eval_trend,
        acpl_before=acpl_before,
        acpl_after=acpl_after,
        n_before=best_k,
        n_after=n - best_k,
        n_total=n,
    )

    caveats = result.caveats
    if losing:
        caveats.append(
            f"accuracy jumped from {rate_before:.0%} to {rate_after:.0%} at "
            f"move {int(move_numbers[best_k])}, when the player was losing "
            f"({eval_at:+.0f}cp). This pattern is consistent with activating "
            f"engine assistance under pressure."
        )
    if eval_trend is not None and eval_trend < -150:
        caveats.append(
            f"eval dropped {eval_trend:+.0f}cp in the {lookback} moves before "
            f"the switchpoint, suggesting the accuracy change followed a setback"
        )
    if acpl_before is not None and acpl_after is not None and acpl_before > 0:
        ratio = acpl_after / acpl_before
        if ratio < 0.35:
            caveats.append(
                f"ACPL dropped from {acpl_before:.0f}cp to {acpl_after:.0f}cp "
                f"({1 - ratio:.0%} reduction) across the switchpoint"
            )

    return result


# ---------------------------------------------------------------------------
# signal for score.combine
# ---------------------------------------------------------------------------


def switchpoint_signal(result: SwitchpointResult) -> dict | None:
    """Convert a detected switchpoint into a signal for evidence combination.

    Returns None if no switchpoint was detected.  The z-score comes from the
    probit transform of the permutation p-value, boosted when the switch
    coincides with a losing position.
    """
    if not result.detected or result.p_value >= 0.05:
        return None

    from scipy import stats

    z = float(stats.norm.ppf(1 - result.p_value))

    # A switchpoint that coincides with losing is much more damning: it rules
    # out the innocent explanation ("they warmed up / the position simplified").
    if result.losing_at_switch:
        z *= 1.3

    parts = [
        f"match rate {result.rate_before:.0%} -> {result.rate_after:.0%}",
        f"at move {result.split_move_number}",
        f"(p={result.p_value:.3f})",
    ]
    if result.losing_at_switch and result.eval_at_switch is not None:
        parts.append(f"losing at {result.eval_at_switch:+.0f}cp")

    return {
        "name": "switchpoint",
        "label": "mid-game accuracy jump",
        "z": z,
        "effect": 2.0,
        "weight": 1.0,
        "observed": result.rate_after,
        "expected": result.rate_before,
        "detail": ", ".join(parts),
    }
