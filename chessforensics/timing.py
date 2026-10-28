"""Move-time forensics.

Runs only when clock data exists. ``analyse_timing`` returns a result whose
``available`` flag is False otherwise, so the UI can render an explicit "no
clock data" state instead of a page of zeros.

The honest caveat, encoded rather than disclaimed
-------------------------------------------------
In bullet and blitz, premoves, mouse-slips and network lag make near-instant
moves extremely common and entirely innocent. A player who premoves a
recapture shows a 0.0s "think time" that looks robotic and is not. So the whole
module carries a time-control-dependent weight: full strength in classical,
reduced in rapid, heavily reduced in blitz, and essentially nil in bullet. That
weight is returned alongside the statistics and is applied in score.py, and the
reason is shown in the UI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

# Multiplier applied to all timing evidence, by speed bucket. These are
# judgement calls about measurement validity, not fitted quantities, so they
# live here as named constants rather than in the baseline file.
SPEED_WEIGHT = {
    "ultrabullet": 0.0,
    "bullet": 0.05,
    "blitz": 0.30,
    "rapid": 0.75,
    "classical": 1.0,
    "correspondence": 0.0,  # unlimited thinking time; the signal is meaningless
}
DEFAULT_SPEED_WEIGHT = 0.5

MIN_MOVES = 10


@dataclass
class TimingResult:
    available: bool
    reason: str = ""
    n: int = 0
    weight: float = 0.0
    speed: str | None = None

    # correlation of think time with position difficulty
    rho: float | None = None
    rho_p: float | None = None

    # robotic uniformity
    cv_raw: float | None = None
    cv_residual: float | None = None

    # forced-move behaviour
    forced_mean: float | None = None
    forced_n: int = 0
    unforced_mean: float | None = None

    # distribution comparison against the band baseline
    ks_stat: float | None = None
    ks_p: float | None = None

    # cadence
    cadence_fraction: float | None = None
    cadence_value: float | None = None
    instant_fraction: float | None = None

    caveats: list[str] = field(default_factory=list)
    per_move: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d.pop("per_move", None)
        return d


def speed_weight(speed: str | None) -> float:
    if speed is None:
        return DEFAULT_SPEED_WEIGHT
    return SPEED_WEIGHT.get(speed.lower(), DEFAULT_SPEED_WEIGHT)


def _residual_cv(times: np.ndarray, complexity: np.ndarray) -> float | None:
    """Coefficient of variation of think times after removing the complexity trend.

    Raw CV is a poor uniformity measure because a legitimate player's times vary
    largely *because* position difficulty varies. Residualising first asks the
    sharper question: once you account for how hard each position was, is what
    remains implausibly uniform?
    """
    if len(times) < MIN_MOVES or np.ptp(complexity) < 1e-9:
        return None
    # Work in log space: think times are strongly right-skewed.
    logt = np.log(np.maximum(times, 0.1))
    try:
        res = stats.linregress(complexity, logt)
    except ValueError:
        return None
    resid = logt - (res.intercept + res.slope * complexity)
    # Back to a multiplicative scale so the number is interpretable as a CV.
    return float(np.std(resid))


def _detect_cadence(times: np.ndarray, tol: float = 0.35) -> tuple[float, float]:
    """Largest share of moves clustered on one value, and that value.

    Catches both increment-hugging (every move costs exactly the increment) and
    a constant artificial cadence. Operates on rounded seconds and asks for the
    modal bucket's share.
    """
    if len(times) < MIN_MOVES:
        return 0.0, 0.0
    finite = times[np.isfinite(times)]
    if len(finite) == 0:
        return 0.0, 0.0
    best_share, best_val = 0.0, 0.0
    for centre in np.unique(np.round(finite, 1)):
        share = float(np.mean(np.abs(finite - centre) <= tol))
        if share > best_share:
            best_share, best_val = share, float(centre)
    return best_share, best_val


def analyse_timing(
    df,
    color: str,
    speed: str | None = None,
    increment: int | None = None,
    baseline_times: list[float] | None = None,
) -> TimingResult:
    """Run the timing battery for one player.

    ``df`` is the feature frame from features.build_frame. Think times that are
    None (unknown, e.g. the first move) are excluded -- never coerced to zero,
    which would fabricate instant-move behaviour.
    """
    w = speed_weight(speed)

    if df is None or len(df) == 0:
        return TimingResult(False, "no analysed moves", speed=speed, weight=w)

    sub = df[df["color"] == color]
    if len(sub) == 0:
        return TimingResult(False, "no moves for this player", speed=speed, weight=w)

    has_any = sub["think_time"].notna().any()
    if not has_any:
        return TimingResult(
            False,
            "this game carries no clock data, so no timing analysis is possible",
            speed=speed,
            weight=w,
        )

    known = sub[sub["think_time"].notna()]
    scoreable = known[known["scoreable"] == 1]

    times_all = known["think_time"].to_numpy(dtype=float)
    if len(scoreable) < MIN_MOVES:
        return TimingResult(
            False,
            f"only {len(scoreable)} moves have both clock data and a scoreable "
            f"position; too few for timing inference",
            n=len(scoreable),
            speed=speed,
            weight=w,
        )

    times = scoreable["think_time"].to_numpy(dtype=float)
    cx = scoreable["complexity"].to_numpy(dtype=float)

    res = TimingResult(True, n=len(times), speed=speed, weight=w)

    # 1. Does the player think longer in harder positions? Spearman, because
    #    the relationship is monotone but very far from linear.
    if np.ptp(cx) > 1e-9 and np.ptp(times) > 1e-9:
        rho, p = stats.spearmanr(times, cx)
        if np.isfinite(rho):
            res.rho = float(rho)
            # Suspicion is the *absence* of correlation, so one-sided below.
            res.rho_p = float(p / 2 if rho > 0 else 1 - p / 2)

    # 2. Uniformity.
    mean_t = float(np.mean(times))
    res.cv_raw = float(np.std(times) / mean_t) if mean_t > 0 else None
    res.cv_residual = _residual_cv(times, cx)

    # 3. Forced moves should be near-instant for everyone.
    forced = known[known["is_forced"] == 1]["think_time"].to_numpy(dtype=float)
    res.forced_n = len(forced)
    if len(forced):
        res.forced_mean = float(np.mean(forced))
    res.unforced_mean = mean_t

    # 4. Distribution shape against the band baseline.
    if baseline_times and len(baseline_times) >= 50:
        try:
            ks = stats.ks_2samp(times, np.asarray(baseline_times, dtype=float))
            res.ks_stat = float(ks.statistic)
            res.ks_p = float(ks.pvalue)
        except ValueError:
            pass

    # 5. Cadence / increment hugging.
    share, val = _detect_cadence(times_all)
    res.cadence_fraction = share
    res.cadence_value = val
    res.instant_fraction = float(np.mean(times_all <= 0.5))

    # ---- caveats --------------------------------------------------------
    if w <= 0.05:
        res.caveats.append(
            f"time control is {speed}: premoves and network lag dominate move "
            f"times at this speed, so timing evidence is given effectively no "
            f"weight (x{w:.2f})."
        )
    elif w < 1.0:
        res.caveats.append(
            f"time control is {speed}: premoves and lag make fast moves common "
            f"and innocent, so timing evidence is down-weighted (x{w:.2f})."
        )
    if increment and res.cadence_value and abs(res.cadence_value - increment) < 0.6:
        res.caveats.append(
            f"the modal move time ({res.cadence_value:.1f}s) sits on the "
            f"increment ({increment}s). This is ordinary clock management in "
            f"increment games, not by itself a red flag."
        )
    if res.forced_n and res.forced_mean is not None and res.forced_mean > 3.0:
        res.caveats.append(
            f"the player averaged {res.forced_mean:.1f}s on moves with only one "
            f"legal reply, which is the opposite of engine-like"
        )
    if len(times) < 20:
        res.caveats.append(
            f"only {len(times)} usable timed moves; all timing statistics here "
            f"are very noisy"
        )

    res.per_move = [
        {
            "ply": int(r["ply"]),
            "move_number": int(r["move_number"]),
            "san": r["san"],
            "think_time": float(r["think_time"]),
            "complexity": float(r["complexity"]),
            "is_forced": int(r["is_forced"]),
        }
        for _, r in known.iterrows()
    ]
    return res


@dataclass
class TimingAccuracyCross:
    """Cross-signal: how fast does the player find hard moves?

    Among high-complexity positions where the player matched the engine's top
    choice, what was their average think time?  A player who consistently finds
    hard moves fast is suspicious in a way that neither timing nor accuracy
    alone captures.
    """

    available: bool = False
    mean_time_hard_correct: float | None = None
    mean_time_hard_all: float | None = None
    expected: float | None = None
    z: float = 0.0
    n_hard_correct: int = 0
    n_hard: int = 0

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def analyse_timing_accuracy(
    df,
    color: str,
    speed: str | None = None,
    band_timing: dict | None = None,
) -> TimingAccuracyCross:
    """Cross timing with accuracy on hard positions for one player.

    ``df`` is the full feature frame from features.build_frame.
    """
    if df is None or len(df) == 0:
        return TimingAccuracyCross()

    sub = df[(df["color"] == color) & (df["scoreable"] == 1)]
    known = sub[sub["think_time"].notna()]
    if len(known) < 10:
        return TimingAccuracyCross()

    times = known["think_time"].to_numpy(dtype=float)
    matched = known["matched_t1"].to_numpy(dtype=float).astype(bool)
    cx = known["complexity"].to_numpy(dtype=float)

    # "Hard" = top 40% of complexity within this game.
    cx_threshold = float(np.percentile(cx, 60))
    hard = cx >= cx_threshold
    hard_correct = hard & matched

    if hard_correct.sum() < 3 or hard.sum() < 5:
        return TimingAccuracyCross()

    mean_hc = float(np.mean(times[hard_correct]))
    mean_ha = float(np.mean(times[hard]))

    ref = band_timing or {}
    expected = ref.get("hard_correct_mean", mean_ha)
    expected_sd = ref.get("hard_correct_sd", max(expected * 0.4, 1.0))

    # Work in log space -- think times are right-skewed.
    log_obs = math.log(max(mean_hc, 0.1))
    log_exp = math.log(max(expected, 0.1))
    log_sd = max(0.2, expected_sd / max(expected, 0.1))

    # Faster is suspicious.
    z = -(log_obs - log_exp) / log_sd

    # Speed weighting is applied via Signal.weight in analyse_player, not here.
    # Multiplying z here AND passing the weight to Signal would double-discount
    # (z * w in the signal, then effective_effect = effect * w in the LR).

    return TimingAccuracyCross(
        available=True,
        mean_time_hard_correct=mean_hc,
        mean_time_hard_all=mean_ha,
        expected=expected,
        z=z,
        n_hard_correct=int(hard_correct.sum()),
        n_hard=int(hard.sum()),
    )


def timing_signals(res: TimingResult, baseline: dict | None) -> list[dict]:
    """Convert the timing battery into z-scores against band references.

    Each signal is returned with its own weight already folded in, so score.py
    does not need to know anything about time controls.
    """
    if not res.available:
        return []

    ref = baseline or {}
    out: list[dict] = []

    def add(name: str, label: str, value: float | None, mu: float, sd: float, sign: float):
        if value is None:
            return
        sd = max(sd, 1e-6)
        z = sign * (value - mu) / sd
        out.append(
            {
                "name": name,
                "label": label,
                "value": value,
                "expected": mu,
                "sd": sd,
                "z": float(z),
                "weight": res.weight,
            }
        )

    # Low correlation between time and difficulty is the suspicious direction.
    add(
        "time_complexity_rho",
        "think time vs complexity correlation",
        res.rho,
        ref.get("rho_mean", 0.32),
        ref.get("rho_sd", 0.18),
        sign=-1.0,
    )
    # Low residual variability = robotic uniformity.
    add(
        "time_uniformity",
        "think-time variability (complexity-adjusted)",
        res.cv_residual,
        ref.get("cv_residual_mean", 0.95),
        ref.get("cv_residual_sd", 0.30),
        sign=-1.0,
    )
    # A high modal share means a constant cadence.
    add(
        "time_cadence",
        "share of moves on a single cadence",
        res.cadence_fraction,
        ref.get("cadence_mean", 0.22),
        ref.get("cadence_sd", 0.12),
        sign=1.0,
    )
    return out
