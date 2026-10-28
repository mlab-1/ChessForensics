"""Accuracy versus complexity: the strongest single discriminator.

The idea
--------
Regress whether the player found the top move on how hard the position was:

    P(match top1) = logistic(b0 + b1 * complexity)

A human's ``b1`` is clearly negative -- they get things right in easy positions
and go wrong in hard ones. An engine does not find hard positions hard, so
assistance flattens the slope towards zero. The attraction of this test is that
it needs no absolute calibration of skill: it is a statement about *shape*, so
it survives not knowing how strong the player really is.

Three things that will bite if ignored
--------------------------------------
1. **Strong players have naturally flatter slopes.** A 2600 is accurate almost
   everywhere, so their curve is flat-ish for entirely innocent reasons. The
   observed slope therefore has to be compared against the distribution of
   slopes for *that player's band*, never against zero.

2. **Rating inflation destroys the reference.** If the account's rating is
   already inflated by earlier cheating, "that player's band" is the wrong
   band, and the test loses power exactly where it is needed. When the fitted
   IPR greatly exceeds the nominal rating we say so rather than pretending the
   comparison is clean.

3. **One game is ~30 decisions.** The standard error on a logistic slope from
   30 points is large. It is reported, and the UI leans on it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

MIN_POINTS = 12  # below this a slope is not worth quoting


@dataclass
class SlopeFit:
    """A fitted accuracy-vs-complexity slope with honest uncertainty."""

    beta0: float
    beta1: float
    se_beta1: float | None
    n: int
    converged: bool
    separable: bool = False  # all matches or all misses: slope unidentified
    note: str = ""

    @property
    def flatness(self) -> float | None:
        """|beta1| in standard errors; small = suspiciously flat."""
        if self.se_beta1 is None or self.se_beta1 <= 0:
            return None
        return abs(self.beta1) / self.se_beta1

    def observed_or_zero(self) -> float:
        """Treat an unidentified slope as 'no deviation' rather than as evidence."""
        if self.separable or not self.converged:
            return 0.0
        return self.beta1

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["flatness"] = self.flatness
        return d


def fit_logistic_slope(
    complexity: np.ndarray,
    matched: np.ndarray,
    l2: float = 0.5,
    max_iter: int = 100,
) -> SlopeFit:
    """Ridge-penalised logistic regression of match on complexity.

    The small L2 penalty is not cosmetic: with ~30 points it is common for a
    player to match on every easy position and miss every hard one, which makes
    the unpenalised MLE diverge to an infinite slope. The penalty keeps the
    estimate finite, and the separable case is flagged so a wildly large slope
    is never read as a precise measurement.
    """
    x = np.asarray(complexity, dtype=float)
    y = np.asarray(matched, dtype=float)
    n = len(x)
    if n < MIN_POINTS:
        return SlopeFit(0.0, 0.0, None, n, False, note=f"only {n} scoreable decisions")
    if y.min() == y.max():
        # Constant outcome: the slope carries no information at all.
        return SlopeFit(
            0.0,
            0.0,
            None,
            n,
            False,
            separable=True,
            note="player matched on every decision or none of them; slope undefined",
        )

    X = np.column_stack([np.ones(n), x])
    beta = np.zeros(2)
    penalty = np.array([0.0, l2])
    converged = False

    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        grad = X.T @ (p - y) + penalty * beta
        w = np.clip(p * (1 - p), 1e-8, None)
        hess = (X * w[:, None]).T @ X + np.diag(penalty)
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        new = beta - step
        if not np.all(np.isfinite(new)):
            break
        if np.max(np.abs(new - beta)) < 1e-9:
            beta = new
            converged = True
            break
        beta = new

    se = None
    try:
        eta = np.clip(X @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-eta))
        w = np.clip(p * (1 - p), 1e-8, None)
        cov = np.linalg.inv((X * w[:, None]).T @ X + np.diag(penalty))
        if cov[1, 1] > 0:
            se = float(math.sqrt(cov[1, 1]))
    except np.linalg.LinAlgError:
        pass

    note = ""
    if n < 25:
        note = f"{n} decisions only; this slope is very imprecisely measured"
    return SlopeFit(float(beta[0]), float(beta[1]), se, n, converged, note=note)


def fit_linear_slope(
    complexity: np.ndarray, cpl: np.ndarray
) -> tuple[float, float, float | None, int]:
    """OLS of centipawn loss on complexity: (intercept, slope, se, n).

    A second, correlated view of the same phenomenon. Honest play loses more
    eval in harder positions, so the slope should be clearly positive. This is
    deliberately *not* treated as independent evidence downstream.
    """
    x = np.asarray(complexity, dtype=float)
    y = np.asarray(cpl, dtype=float)
    n = len(x)
    if n < MIN_POINTS or np.ptp(x) < 1e-9:
        return 0.0, 0.0, None, n
    res = stats.linregress(x, y)
    return float(res.intercept), float(res.slope), float(res.stderr), n


@dataclass
class SlopeTest:
    """Observed slope against the band-conditional reference distribution."""

    observed: float
    band_mean: float
    band_sd: float
    z: float  # positive z = flatter than the band (suspicious direction)
    p_one_sided: float
    n: int
    se_observed: float | None
    total_sd: float
    caveats: list[str]

    def to_dict(self) -> dict:
        return dict(self.__dict__)


def test_slope_against_band(
    fit: SlopeFit,
    band_mean: float,
    band_sd: float,
    ipr_elo: float | None = None,
    nominal_elo: float | None = None,
    inflation_threshold: float = 300.0,
) -> SlopeTest:
    """Compare an observed slope to the corpus distribution for that band.

    The reference spread and the subject's own measurement error are combined
    in quadrature. Skipping that second term is a classic way to produce
    confident nonsense: with 30 decisions the slope's own standard error is
    usually *larger* than the between-player spread, so it dominates.
    """
    caveats: list[str] = []
    band_sd = max(band_sd, 1e-6)
    se_obs = fit.se_beta1 or 0.0
    total_sd = math.sqrt(band_sd**2 + se_obs**2)

    # Suspicion is "flatter than expected", i.e. beta1 closer to zero / above
    # the band mean, since band means are negative.
    z = (fit.observed_or_zero() - band_mean) / total_sd if total_sd > 0 else 0.0
    p = float(stats.norm.sf(z))

    if fit.separable:
        caveats.append(
            "the player's results were all-or-nothing across complexity, "
            "so the slope is undefined; this test is uninformative here"
        )
    if fit.n < 25:
        caveats.append(
            f"only {fit.n} scoreable decisions -- the slope's own standard error "
            f"({se_obs:.2f}) is comparable to or larger than the between-player "
            f"spread ({band_sd:.2f})"
        )
    if ipr_elo is not None and nominal_elo is not None:
        if ipr_elo - nominal_elo > inflation_threshold:
            caveats.append(
                f"intrinsic performance ({ipr_elo:.0f}) far exceeds the account's "
                f"rating ({nominal_elo:.0f}). If that rating is itself the product "
                f"of earlier assistance, the band reference used here is the wrong "
                f"one and this test is biased towards innocence."
            )
    return SlopeTest(
        observed=fit.beta1,
        band_mean=band_mean,
        band_sd=band_sd,
        z=float(z),
        p_one_sided=p,
        n=fit.n,
        se_observed=fit.se_beta1,
        total_sd=total_sd,
        caveats=caveats,
    )


def binned_curve(
    complexity: np.ndarray, matched: np.ndarray, bins: int = 5
) -> list[dict]:
    """Observed match rate per complexity bin, with Wilson intervals.

    Used for the chart. Wilson rather than normal intervals because bins hold
    only a handful of moves and rates sit near 0 or 1, where the normal
    interval runs outside [0, 1] and looks absurd.
    """
    x = np.asarray(complexity, dtype=float)
    y = np.asarray(matched, dtype=float)
    if len(x) == 0:
        return []
    edges = np.quantile(x, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    if len(edges) < 2:
        edges = np.array([x.min() - 1e-9, x.max() + 1e-9])
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)

    out = []
    for b in range(len(edges) - 1):
        sel = idx == b
        k = int(y[sel].sum())
        n = int(sel.sum())
        if n == 0:
            continue
        rate = k / n
        z = 1.96
        denom = 1 + z * z / n
        centre = (rate + z * z / (2 * n)) / denom
        half = z * math.sqrt(rate * (1 - rate) / n + z * z / (4 * n * n)) / denom
        out.append(
            {
                "bin": b,
                "complexity_low": float(edges[b]),
                "complexity_high": float(edges[b + 1]),
                "complexity_mid": float(x[sel].mean()),
                "n": n,
                "matched": k,
                "rate": rate,
                "lo": max(0.0, centre - half),
                "hi": min(1.0, centre + half),
            }
        )
    return out
