"""Regan-style multinomial choice model, MLE fitting, and Poisson-binomial tests.

The model
---------
A player facing a position with candidate moves of eval loss ``delta_i`` (in
pawns, ``delta_1 = 0`` for the engine's best move) plays move ``i`` with

    P(i)  proportional to  exp( -(delta_i / s) ** c )

* ``s`` -- sensitivity: how much eval loss the player tolerates. **Smaller s
  means stronger**, because the exponent grows and probability mass piles onto
  the best move.
* ``c`` -- consistency: how sharply they discriminate between a small error and
  a large one.

Two uses: fit per rating band on the corpus to get a reference, and fit on the
player under test to get their intrinsic performance rating (IPR).

Why Poisson-binomial
--------------------
Expected top-1 match count is ``sum(p_i)`` over the positions *actually played*,
and its variance is ``sum(p_i * (1 - p_i))``. Positions within one game range
from forced recaptures (``p ~ 0.99``) to genuinely hard choices (``p ~ 0.25``),
so this heterogeneity is the dominant feature of the null distribution.

Two distinct errors are possible here, and they pull in opposite directions:

1. **Wrong expectation (this is the false-positive generator.)** Taking the
   band's *overall* average match rate as the expectation ignores which
   positions this game actually contained. A game full of forced recaptures
   and one-move-only tactics should have a high expected match rate; score it
   against a global average and the player looks superhuman for playing the
   obvious. Conditioning ``sum(p_i)`` on the real positions is what fixes this,
   and it is the single most important thing in this module.

2. **Wrong variance.** Pooling to ``n * p_bar * (1 - p_bar)`` differs from the
   true ``sum(p_i * (1 - p_i))`` by exactly ``sum((p_i - p_bar)**2)``, so the
   pooled form is always *larger*. Pooling the variance is therefore
   conservative, not anti-conservative -- it loses power rather than inventing
   guilt. We still use the exact form, because losing real power on the only
   test that matters is not a virtue, but the honest statement is that error
   (1) is the dangerous one.

The PMF is computed exactly by dynamic programming rather than normal
approximation: n is a few dozen per game, so exactness is free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, stats

# Parameter bounds. Wide enough not to bind in practice, tight enough to stop
# the optimiser wandering into numerically degenerate corners.
S_BOUNDS = (0.005, 3.0)
C_BOUNDS = (0.15, 4.0)

MAX_DELTA_PAWNS = 6.0  # deltas are clipped here; see features.DECIDED_CP

# The MultiPV width the s -> Elo calibration was fitted at. Comparing a subject
# fitted at a different width to this curve reintroduces the truncation bias
# documented in _extrapolate_tail, so the mismatch is surfaced, not ignored.
BASELINE_MULTIPV = 5


def check_multipv_consistency(subject_multipv: int, baseline_multipv: int) -> str | None:
    """Return a warning string if the subject and baseline widths disagree."""
    if subject_multipv == baseline_multipv:
        return None
    return (
        f"candidate window mismatch: this game was analysed at MultiPV "
        f"{subject_multipv} but the baseline was fitted at MultiPV "
        f"{baseline_multipv}. Sensitivity estimates are biased low by a "
        f"width-dependent amount, so the IPR and any test derived from it are "
        f"not directly comparable."
    )


@dataclass
class Position:
    """One decision, reduced to what the choice model needs.

    ``deltas`` are in pawns, ascending, with ``deltas[0] == 0``.
    ``played`` indexes the move actually played, or None if it fell outside the
    MultiPV window and could not be placed.
    """

    deltas: np.ndarray
    played: int | None
    legal: int
    complexity: float = 0.5
    weight: float = 1.0

    @property
    def k(self) -> int:
        return len(self.deltas)


def _extrapolate_tail(deltas: np.ndarray, legal: int) -> np.ndarray:
    """Append plausible deltas for legal moves outside the MultiPV window.

    The engine only reports the top k lines, but the choice model's normaliser
    runs over everything the player could have played. Dropping the unseen
    moves makes the denominator too small and inflates every candidate's
    probability -- which biases the fitted ``s`` and, worse, biases the expected
    top-1 match rate *upward*, exactly the quantity we are testing.

    The unseen moves all have delta >= deltas[-1]. We extend the observed
    falloff linearly, which is crude but monotone: it puts a non-trivial amount
    of mass outside the window without pretending to know where exactly.

    Measured bias (see tests/test_stats.py): fitting from a top-5 window
    recovers ``s`` 3-23% *low* across the plausible range, i.e. it makes players
    look slightly stronger than they are. That is the unsafe direction for this
    application, so it must not be left to cancel by luck. It cancels by
    construction instead: the s -> Elo calibration curve in the baseline file is
    fitted through this identical code path at the same MultiPV width, so the
    subject and the reference are measured with the same distorted ruler.
    ``BASELINE_MULTIPV`` records that width and ``check_multipv_consistency``
    refuses to compare across mismatched widths.
    """
    n_missing = max(0, legal - len(deltas))
    if n_missing == 0:
        return deltas
    if len(deltas) >= 2:
        step = max(0.05, float(deltas[-1] - deltas[-2]))
    else:
        step = 0.5
    tail = deltas[-1] + step * np.arange(1, n_missing + 1, dtype=float)
    return np.concatenate([deltas, np.clip(tail, None, MAX_DELTA_PAWNS)])


def move_probs(deltas: np.ndarray, s: float, c: float) -> np.ndarray:
    """Choice probabilities for one position. Numerically guarded."""
    d = np.clip(np.asarray(deltas, dtype=float), 0.0, MAX_DELTA_PAWNS)
    with np.errstate(over="ignore", invalid="ignore"):
        expo = np.power(d / max(s, 1e-9), c)
    expo = np.nan_to_num(expo, nan=700.0, posinf=700.0)
    # Subtracting the min keeps exp() in range; it cancels in the normalisation.
    expo = expo - expo.min()
    w = np.exp(-np.clip(expo, 0.0, 700.0))
    total = w.sum()
    if not np.isfinite(total) or total <= 0:
        out = np.zeros_like(w)
        out[0] = 1.0
        return out
    return w / total


@dataclass
class ChoiceModel:
    """A fitted (or assumed) (s, c) pair."""

    s: float
    c: float
    se_s: float | None = None
    se_c: float | None = None
    loglik: float | None = None
    n: int = 0
    converged: bool = True

    def probs(self, pos: Position, with_tail: bool = True) -> np.ndarray:
        d = _extrapolate_tail(pos.deltas, pos.legal) if with_tail else pos.deltas
        return move_probs(d, self.s, self.c)

    def p_top1(self, pos: Position) -> float:
        return float(self.probs(pos)[0])

    def p_topn(self, pos: Position, n: int) -> float:
        p = self.probs(pos)
        return float(p[: min(n, len(p))].sum())

    def expected_delta(self, pos: Position) -> tuple[float, float]:
        """(expected eval loss, variance of eval loss) in pawns."""
        d = _extrapolate_tail(pos.deltas, pos.legal)
        p = move_probs(d, self.s, self.c)
        mean = float((p * d).sum())
        var = float((p * d * d).sum() - mean**2)
        return mean, max(0.0, var)

    def to_dict(self) -> dict:
        return {
            "s": self.s,
            "c": self.c,
            "se_s": self.se_s,
            "se_c": self.se_c,
            "loglik": self.loglik,
            "n": self.n,
            "converged": self.converged,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ChoiceModel":
        return cls(
            s=float(d["s"]),
            c=float(d["c"]),
            se_s=d.get("se_s"),
            se_c=d.get("se_c"),
            loglik=d.get("loglik"),
            n=int(d.get("n", 0)),
            converged=bool(d.get("converged", True)),
        )


# --------------------------------------------------------------------------
# maximum likelihood
# --------------------------------------------------------------------------


class _Design:
    """Padded (n_positions x max_candidates) delta matrix for vectorised fitting.

    The per-position Python loop was the bottleneck: fitting a rating band on a
    corpus means tens of thousands of positions times several optimiser
    restarts times a numerical Hessian. Building the ragged delta lists into one
    masked array up front turns the whole likelihood into a handful of NumPy
    calls.
    """

    __slots__ = ("deltas", "mask", "played", "weights", "n")

    def __init__(self, positions: list[Position]):
        usable = [p for p in positions if p.played is not None]
        full = [_extrapolate_tail(p.deltas, p.legal) for p in usable]
        self.n = len(usable)
        width = max((len(f) for f in full), default=1)
        self.deltas = np.zeros((self.n, width))
        self.mask = np.zeros((self.n, width), dtype=bool)
        self.played = np.zeros(self.n, dtype=int)
        self.weights = np.ones(self.n)
        for i, (pos, f) in enumerate(zip(usable, full)):
            k = len(f)
            self.deltas[i, :k] = np.clip(f, 0.0, MAX_DELTA_PAWNS)
            self.mask[i, :k] = True
            self.played[i] = min(int(pos.played), k - 1)
            self.weights[i] = pos.weight
        # Padding sits at a delta large enough that masking it out is exact.
        self.deltas[~self.mask] = MAX_DELTA_PAWNS

    def neg_loglik(self, s: float, c: float) -> float:
        if self.n == 0:
            return 0.0
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            expo = np.power(self.deltas / max(s, 1e-9), c)
        expo = np.nan_to_num(expo, nan=700.0, posinf=700.0)
        expo = expo - expo.min(axis=1, keepdims=True)
        w = np.exp(-np.clip(expo, 0.0, 700.0)) * self.mask
        total = w.sum(axis=1)
        chosen = w[np.arange(self.n), self.played]
        ll = np.log(np.maximum(chosen, 1e-300)) - np.log(np.maximum(total, 1e-300))
        return float(-(self.weights * ll).sum())


def _neg_loglik(theta: np.ndarray, design: "_Design") -> float:
    return design.neg_loglik(math.exp(theta[0]), math.exp(theta[1]))


def fit_choice_model(
    positions: list[Position],
    init: tuple[float, float] = (0.10, 0.50),
) -> ChoiceModel:
    """Fit (s, c) by maximum likelihood over observed move choices.

    Optimised in log-space so both parameters stay positive. Standard errors
    come from a numerical Hessian of the negative log-likelihood (observed
    Fisher information) -- and are reported as None when that matrix is not
    positive definite, rather than quietly handing back a meaningless number.
    """
    usable = [p for p in positions if p.played is not None and p.k >= 2]
    if len(usable) < 8:
        # Not enough decisions to identify two parameters. Say so.
        return ChoiceModel(s=init[0], c=init[1], n=len(usable), converged=False)

    design = _Design(usable)
    lo = np.log([S_BOUNDS[0], C_BOUNDS[0]])
    hi = np.log([S_BOUNDS[1], C_BOUNDS[1]])

    best = None
    # Multi-start: the likelihood has a mild ridge in (s, c) and a single start
    # occasionally settles on it rather than the mode.
    for s0, c0 in ((init[0], init[1]), (0.04, 0.4), (0.25, 0.8), (0.6, 1.4)):
        x0 = np.clip(np.log([s0, c0]), lo, hi)
        res = optimize.minimize(
            _neg_loglik,
            x0,
            args=(design,),
            method="L-BFGS-B",
            bounds=list(zip(lo, hi)),
            options={"maxiter": 400, "ftol": 1e-12},
        )
        if best is None or res.fun < best.fun:
            best = res

    assert best is not None
    s, c = float(math.exp(best.x[0])), float(math.exp(best.x[1]))

    se_s = se_c = None
    try:
        hess = _numerical_hessian(lambda t: _neg_loglik(t, design), best.x)
        cov = np.linalg.inv(hess)
        if np.all(np.diag(cov) > 0):
            # Delta method back from log-space to the natural scale.
            se_s = float(math.sqrt(cov[0, 0]) * s)
            se_c = float(math.sqrt(cov[1, 1]) * c)
    except (np.linalg.LinAlgError, ValueError):
        pass

    return ChoiceModel(
        s=s,
        c=c,
        se_s=se_s,
        se_c=se_c,
        loglik=float(-best.fun),
        n=len(usable),
        converged=bool(best.success),
    )


def _numerical_hessian(f, x: np.ndarray, eps: float = 1e-4) -> np.ndarray:
    n = len(x)
    h = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            ei = np.zeros(n)
            ej = np.zeros(n)
            ei[i] = eps
            ej[j] = eps
            val = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (
                4 * eps * eps
            )
            h[i, j] = h[j, i] = val
    return h


# --------------------------------------------------------------------------
# Poisson-binomial
# --------------------------------------------------------------------------


def poisson_binomial_pmf(probs: np.ndarray) -> np.ndarray:
    """Exact PMF of the number of successes among independent Bernoulli(p_i).

    O(n^2) dynamic programming, verified against brute-force enumeration and
    Monte Carlo in the test suite.
    """
    p = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    pmf = np.zeros(len(p) + 1)
    pmf[0] = 1.0
    for i, pi in enumerate(p):
        # Walk downwards so each slot is updated from the previous iteration.
        pmf[1 : i + 2] = pmf[1 : i + 2] * (1 - pi) + pmf[0 : i + 1] * pi
        pmf[0] *= 1 - pi
    return pmf


@dataclass
class CountTest:
    """Observed vs expected success count under a Poisson-binomial null."""

    observed: int
    expected: float
    variance: float
    n: int
    z: float
    p_one_sided: float  # P(X >= observed), exact
    p_mid: float  # mid-p variant, less conservative for discrete data
    ci_low: float
    ci_high: float

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def count_test(probs: np.ndarray, observed: int) -> CountTest:
    """Test an observed match count against per-position probabilities."""
    p = np.clip(np.asarray(probs, dtype=float), 1e-12, 1 - 1e-12)
    n = len(p)
    if n == 0:
        return CountTest(observed, 0.0, 0.0, 0, 0.0, 1.0, 1.0, 0.0, 0.0)

    expected = float(p.sum())
    variance = float((p * (1 - p)).sum())  # the Poisson-binomial variance
    sd = math.sqrt(variance) if variance > 0 else 0.0
    z = (observed - expected) / sd if sd > 0 else 0.0

    pmf = poisson_binomial_pmf(p)
    k = max(0, min(observed, n))
    p_ge = float(pmf[k:].sum())
    p_mid = float(pmf[k + 1 :].sum() + 0.5 * pmf[k])

    # Central 95% interval of the null distribution, for plotting.
    cdf = np.cumsum(pmf)
    ci_low = float(np.searchsorted(cdf, 0.025))
    ci_high = float(np.searchsorted(cdf, 0.975))

    return CountTest(
        observed=observed,
        expected=expected,
        variance=variance,
        n=n,
        z=float(z),
        p_one_sided=min(1.0, p_ge),
        p_mid=min(1.0, max(0.0, p_mid)),
        ci_low=ci_low,
        ci_high=ci_high,
    )


@dataclass
class MeanTest:
    """Observed vs expected mean of a continuous per-move quantity (e.g. ACPL)."""

    observed: float
    expected: float
    sd: float
    n: int
    z: float
    p_one_sided: float
    ci_low: float
    ci_high: float

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def mean_test(
    per_move_mean: np.ndarray,
    per_move_var: np.ndarray,
    observed_mean: float,
    lower_is_suspicious: bool = True,
) -> MeanTest:
    """Test an observed mean against a sum of independent, non-identical terms.

    Same principle as the count test: each position has its own mean and
    variance under the reference model, so the variance of the average is
    ``sum(var_i) / n^2`` -- not ``var_pooled / n``.
    """
    mu = np.asarray(per_move_mean, dtype=float)
    var = np.asarray(per_move_var, dtype=float)
    n = len(mu)
    if n == 0:
        return MeanTest(observed_mean, 0.0, 0.0, 0, 0.0, 1.0, 0.0, 0.0)

    expected = float(mu.mean())
    sd = float(math.sqrt(max(var.sum(), 0.0)) / n)
    z = (observed_mean - expected) / sd if sd > 0 else 0.0
    # Suspicion runs in the direction of *too few* errors.
    p = float(stats.norm.cdf(z)) if lower_is_suspicious else float(stats.norm.sf(z))
    return MeanTest(
        observed=float(observed_mean),
        expected=expected,
        sd=sd,
        n=n,
        z=float(z),
        p_one_sided=p,
        ci_low=expected - 1.96 * sd,
        ci_high=expected + 1.96 * sd,
    )


# --------------------------------------------------------------------------
# intrinsic performance rating
# --------------------------------------------------------------------------


@dataclass
class IPR:
    """Intrinsic performance rating derived from a fitted (s, c)."""

    elo: float | None
    low: float | None
    high: float | None
    s: float
    c: float
    n: int
    extrapolated: bool = False
    note: str = ""

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def ipr_from_model(
    model: ChoiceModel,
    calibration: list[tuple[float, float]],
) -> IPR:
    """Map a fitted ``s`` to an Elo via the per-band calibration curve.

    ``calibration`` is [(elo_mid, s_for_that_band), ...]. Interpolation is on
    log(s), which is close to linear in Elo over the practical range, and the
    interval comes from propagating the standard error of ``s``.
    """
    if not calibration or len(calibration) < 2:
        return IPR(None, None, None, model.s, model.c, model.n, note="no calibration curve")
    if model.n < 8:
        return IPR(
            None, None, None, model.s, model.c, model.n, note="too few decisions to fit"
        )

    # Ascending in s, which is descending in Elo. Bands the corpus could not
    # separate arrive as exact ties (isotonic pooling in
    # Baseline.calibration_curve collapses reversals), so each tie is reduced to
    # its mean Elo. That keeps the mapping a well-defined function instead of
    # something that depends on sort order, and a wide tie is an honest
    # statement that the corpus cannot tell those ratings apart.
    grouped: dict[float, list[float]] = {}
    for elo, s in calibration:
        grouped.setdefault(round(math.log(max(s, 1e-9)), 9), []).append(elo)
    pts = sorted((ls, sum(es) / len(es)) for ls, es in grouped.items())
    if len(pts) < 2:
        return IPR(
            None, None, None, model.s, model.c, model.n,
            note="the calibration curve has no resolution -- every rating band in "
                 "the corpus fitted the same sensitivity, so no rating can be inferred",
        )
    log_s = np.array([p[0] for p in pts])
    elos = np.array([p[1] for p in pts])

    def interp(s_val: float) -> tuple[float, bool]:
        ls = math.log(max(min(s_val, S_BOUNDS[1]), 1e-9))
        outside = ls < log_s.min() or ls > log_s.max()
        # Stronger play = smaller s, so Elo decreases with log(s).
        return float(np.interp(ls, log_s, elos)), outside

    elo, extrap = interp(model.s)
    low = high = None
    if model.se_s:
        hi_s = model.s + 1.96 * model.se_s
        lo_s = max(S_BOUNDS[0], model.s - 1.96 * model.se_s)
        e_hi, _ = interp(lo_s)  # smaller s -> higher Elo
        e_lo, _ = interp(hi_s)
        low, high = min(e_lo, e_hi), max(e_lo, e_hi)

    note = ""
    if extrap:
        note = (
            "fitted sensitivity lies outside the calibrated range; "
            "the Elo figure is an extrapolation"
        )
    if model.n < 25:
        note = (note + " " if note else "") + (
            f"only {model.n} scoreable decisions -- a single-game IPR is very noisy"
        )
    return IPR(elo, low, high, model.s, model.c, model.n, extrap, note.strip())


# --------------------------------------------------------------------------
# glue: build Positions from the feature frame
# --------------------------------------------------------------------------


def positions_from_frame(df, ga, color: str) -> list[Position]:
    """Build model Positions for one player's scoreable moves.

    Deltas come from the MultiPV window of the best available analysis. When
    the played move is outside that window we still include it, placed at its
    measured eval loss, so its probability is well defined -- otherwise every
    bad move would silently drop out of the likelihood and the fit would think
    the player never erred.
    """
    out: list[Position] = []
    if df is None or len(df) == 0:
        return out

    sub = df[(df["color"] == color) & (df["scoreable"] == 1)]
    for _, row in sub.iterrows():
        pa = ga.at(int(row["ply"]))
        if pa is None or not pa.candidates:
            continue
        deltas = [max(0.0, d) / 100.0 for d in pa.deltas()]
        if not deltas:
            continue
        played_idx: int | None = None
        rank = row["rank"]
        if rank is not None and not (isinstance(rank, float) and math.isnan(rank)):
            played_idx = int(rank) - 1
        else:
            d_played = row["delta_cp"]
            if d_played is None or (isinstance(d_played, float) and math.isnan(d_played)):
                continue
            # Splice the played move into the delta list at its true position.
            dv = min(float(d_played) / 100.0, MAX_DELTA_PAWNS)
            insert_at = int(np.searchsorted(np.array(deltas), dv))
            deltas.insert(insert_at, dv)
            played_idx = insert_at

        arr = np.clip(np.array(deltas, dtype=float), 0.0, MAX_DELTA_PAWNS)
        legal = max(int(row["legal_moves"]), len(arr))
        out.append(
            Position(
                deltas=arr,
                played=played_idx,
                legal=legal,
                complexity=float(row["complexity"]),
            )
        )
    return out


@dataclass
class TailTest:
    """Test whether the player's error distribution lacks the expected tail.

    Humans occasionally blunder (CPL > 100).  Engine-assisted players have a
    truncated distribution: everything clusters near zero.  This tests the
    blunder rate and the 90th percentile of CPL against the reference model's
    position-conditional expectations.
    """

    observed_p90: float
    expected_p90: float
    observed_blunder_rate: float  # fraction of moves with CPL > 100
    expected_blunder_rate: float
    z: float
    p_one_sided: float
    n: int

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def tail_test(
    observed_cpl: np.ndarray,
    per_move_mean: np.ndarray,
    per_move_var: np.ndarray,
    blunder_cp: float = 100.0,
) -> TailTest | None:
    """Test whether the player's CPL tail is suspiciously thin.

    For each position the reference model gives an expected mean and variance
    of centipawn loss.  The probability of a blunder (CPL > ``blunder_cp``) at
    each position is ``1 - Phi((blunder_cp - mu_i) / sd_i)``.  The observed
    blunder rate is tested against this position-conditional expectation using
    the same Poisson-binomial logic as the match-rate tests.

    Returns None when there are too few moves.
    """
    if len(observed_cpl) < 15:
        return None

    cpl = np.asarray(observed_cpl, dtype=float)
    mu = np.asarray(per_move_mean, dtype=float)
    var = np.asarray(per_move_var, dtype=float)
    n = len(cpl)

    obs_p90 = float(np.percentile(cpl, 90))

    sd = np.sqrt(np.maximum(var, 1.0))
    # Expected 90th percentile: approx mean + 1.28 * sd, averaged across positions.
    exp_p90 = float(np.mean(mu + 1.28 * sd))

    # Per-position probability of a blunder under the reference model.
    p_blunder = 1.0 - stats.norm.cdf((blunder_cp - mu) / sd)
    exp_blunder_rate = float(p_blunder.mean())
    obs_blunder_rate = float(np.mean(cpl > blunder_cp))

    # Variance of the observed blunder rate: Poisson-binomial style.
    blunder_var = float((p_blunder * (1 - p_blunder)).sum()) / (n * n)
    blunder_sd = math.sqrt(max(blunder_var, 1e-12))

    # Lower blunder rate is the suspicious direction.
    z = -(obs_blunder_rate - exp_blunder_rate) / blunder_sd if blunder_sd > 0 else 0.0
    p = float(stats.norm.cdf(-z))  # one-sided: fewer blunders than expected

    return TailTest(
        observed_p90=obs_p90,
        expected_p90=exp_p90,
        observed_blunder_rate=obs_blunder_rate,
        expected_blunder_rate=exp_blunder_rate,
        z=z,
        p_one_sided=p,
        n=n,
    )


def expected_rates(model: ChoiceModel, positions: list[Position]) -> dict:
    """Reference-model expectations for exactly the positions in this game.

    This is the whole point of the exercise: not "what does a 1500 average", but
    "what would a 1500 have done *in these specific positions*".
    """
    if not positions:
        return {"p_t1": [], "p_t2": [], "p_t3": [], "e_delta": [], "v_delta": []}
    p_t1, p_t2, p_t3, e_d, v_d = [], [], [], [], []
    for pos in positions:
        p = model.probs(pos)
        p_t1.append(float(p[0]))
        p_t2.append(float(p[: min(2, len(p))].sum()))
        p_t3.append(float(p[: min(3, len(p))].sum()))
        m, v = model.expected_delta(pos)
        e_d.append(m * 100.0)  # back to centipawns
        v_d.append(v * 100.0 * 100.0)
    return {"p_t1": p_t1, "p_t2": p_t2, "p_t3": p_t3, "e_delta": e_d, "v_delta": v_d}
