"""Evidence combination: module z-scores to a calibrated posterior.

How the combination works
-------------------------
Each module yields a z-score, signed so that positive means "more suspicious",
standardised against the band-conditional null. Under H0 (honest) the vector of
z-scores is multivariate normal with mean 0 and correlation ``R``. Under H1
(assisted) it has mean ``mu``, the per-signal effect size measured from the
positive-control experiments. With a shared covariance the log-likelihood ratio
is exactly

    log LR = mu' R^-1 z  -  0.5 * mu' R^-1 mu

This is the whole point. **The signals are nowhere near independent**: a high
top-1 match rate, a flat complexity slope and a low ACPL are three views of one
underlying thing. Summing their individual log-LRs -- naive Bayes -- triple
counts that shared component and produces absurd certainty. The ``R^-1`` term
does the deduplication properly, and it is a derivation rather than a fudge
factor.

``R`` is estimated from the corpus and shrunk towards the identity, because a
noisily estimated correlation matrix can be near-singular and would then blow
up ``R^-1``. The shrinkage intensity scales with how little corpus data backs
the estimate.

Honesty requirements this module enforces
-----------------------------------------
* The prior is explicit and defaults to a 1% base rate. At that base rate even
  a likelihood ratio of 20:1 only reaches ~17% posterior, which is exactly the
  point users need to see.
* A sensitivity band is always returned, derived by Monte Carlo over the
  uncertainty in the baseline parameters. A bare point estimate is not offered.
  This is not a Bayesian credible interval; it shows how the point estimate
  moves when the reference model is perturbed.
* ``games_curve`` reports how the evidence would accumulate over N comparable
  games, so the weakness of single-game evidence is visible rather than stated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import nnls
from scipy.special import logsumexp

DEFAULT_PRIOR = 0.01  # base rate of engine assistance in the population
DEFAULT_SHRINKAGE = 0.25
MAX_ABS_Z = 8.0  # z-scores beyond this are model failure, not evidence

# Assistance-rate grid for the mixture model. DEFAULT_EFFECTS are calibrated
# for rate 1.0 (full engine assistance). mu(r) = r * e.
MIXTURE_RATES = np.array([0.10, 0.25, 0.50, 1.0])
MIXTURE_LOG_PRIOR = np.full(len(MIXTURE_RATES), -np.log(len(MIXTURE_RATES)))

# Empirical null: per-signal bias and overdispersion measured on the corpus
# (calibrate_signals.py, 2026-10-22). The null is NOT N(0,1) -- it has
# mean != 0 and std > 1 due to model misspecification (2-candidate window,
# within-game dependence, etc.). Correcting for this is essential to avoid
# both false positives (bias > 0) and false negatives (bias < 0).
#
# Format: {signal_name: (bias, std)} where corrected_z = (raw_z - bias) / std
#
# IMPORTANT: only include signals where the calibration was measured through
# the SAME code path as the production scorer. Signals measured with a
# simplified formula (slope, cpl_slope, cpl_tail) produced unreliable
# calibrations and are excluded until they can be measured properly.
# The match-rate signals (t1/t2/t3) use Poisson-binomial z in both paths,
# so their calibration is trustworthy.
EMPIRICAL_NULL = {
    "t1_match":           (-0.51, 1.66),
    "t2_match":           (-0.21, 1.77),
    "t3_match":           (-0.09, 1.77),
    "acpl":               (-0.21, 1.91),
    "time_complexity_rho": (+0.02, 1.17),
    "time_uniformity":    (-0.09, 1.40),
    # NOT included (calibration code path mismatch):
    # slope:     measured with linear regression, production uses logistic
    # cpl_slope: measured with linear, production uses different normalization
    # cpl_tail:  measured with global blunder rate, production uses per-position
}
# Signals not listed use no correction (identity transform).

# Effect sizes (in z units) for each signal under H1.
#
# KNOWN WEAKNESS: these are hand-set, not measured from positive controls.
# Proper estimation requires either (a) games from known cheaters (Lichess
# engine-ban data), or (b) synthetic engine-play simulation at various
# assistance rates. The values below are deliberately modest: overstating
# them inflates every posterior.
#
# Corpus tail analysis (calibrate_signals.py, 2026-10-22) found that p99 of
# the honest z-distribution is around +3 to +4 for match-rate signals. True
# cheater effect sizes would be larger, so these are conservative lower bounds.
DEFAULT_EFFECTS = {
    "t1_match": 2.2,
    "t2_match": 1.6,
    "t3_match": 1.3,
    "acpl": 1.8,
    "slope": 1.9,
    "cpl_slope": 1.2,
    "ipr_gap": 2.0,
    "time_complexity_rho": 1.4,
    "time_uniformity": 1.1,
    "time_cadence": 1.0,
    "switchpoint": 2.0,
    "cpl_tail": 1.6,
    "timing_accuracy_cross": 1.5,
    "deep_only_match": 2.4,
    "engine_fingerprint": 2.5,
}


@dataclass
class Signal:
    """One piece of evidence, already standardised."""

    name: str
    label: str
    z: float
    effect: float  # expected z under H1
    weight: float = 1.0  # measurement-validity discount (e.g. bullet timing)
    observed: float | None = None
    expected: float | None = None
    detail: str = ""

    @property
    def effective_effect(self) -> float:
        return self.effect * self.weight


@dataclass
class EvidenceRow:
    name: str
    label: str
    observed: float | None
    expected: float | None
    z: float
    weight: float
    decibans_naive: float  # what this signal would contribute alone
    decibans_adjusted: float  # its share after correlation adjustment
    detail: str = ""


@dataclass
class Verdict:
    posterior: float
    posterior_low: float
    posterior_high: float
    prior: float
    log_lr: float
    decibans: float
    decibans_naive: float
    redundancy: float  # 1 - adjusted/naive: how much was double counting
    rows: list[EvidenceRow] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    n_signals: int = 0
    uncalibrated: bool = False
    log_lrs_by_rate: list[float] = field(default_factory=list)  # per-rate LRs for multi-game aggregation

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "rows"}
        d["rows"] = [r.__dict__ for r in self.rows]
        return d


def _logit(p: float) -> float:
    p = min(max(p, 1e-12), 1 - 1e-12)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    """Numerically stable logistic. Both branches use exp(-|x|)."""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-min(x, 500.0)))
    e = math.exp(max(x, -500.0))
    return e / (1.0 + e)


def _nnls_weights(R: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Non-negative signal weights for the projected LR.

    Returns w >= 0 that minimise 0.5 w'Rw - mu'w. When R^{-1}mu is already
    non-negative, returns R^{-1}mu exactly (the standard Mahalanobis weights).
    """
    Rinv_mu = np.linalg.solve(R, mu)
    if np.all(Rinv_mu >= -1e-12):
        return np.maximum(Rinv_mu, 0.0)
    try:
        L = np.linalg.cholesky(R)
    except np.linalg.LinAlgError:
        L = np.linalg.cholesky(R + 1e-6 * np.eye(len(R)))
    w, _ = nnls(L.T, np.linalg.solve(L, mu))
    return w


def _projected_llr(
    z: np.ndarray, R: np.ndarray, mu: np.ndarray, w: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Log-likelihood ratio with non-negative signal weights.

    The standard Mahalanobis LR uses weights w = R^{-1} mu, which can be
    negative when correlated signals have effect sizes that pull in opposite
    directions under the fitted R. A negative weight means "a more suspicious
    z-score lowers the evidence", which is logically backwards and produces
    absurd results (e.g. -150 dB from one game).

    The fix: project mu onto the non-negative cone of the R-inner-product
    space via NNLS. The score T = w'z is then used as a one-dimensional
    sufficient statistic, with its own exact LR.

    Pass ``w`` to reuse pre-computed weights (for the MC loop).
    """
    if w is None:
        w = _nnls_weights(R, mu)

    m = float(w @ mu)       # E[T] under H1
    s2 = float(w @ R @ w)   # Var[T]
    if m <= 0 or s2 <= 0:
        return 0.0, np.zeros_like(z)

    a = m / s2
    t = float(w @ z)
    log_lr = a * t - 0.5 * a * m

    # Per-signal attribution that sums exactly to the total
    contrib = a * w * z - (0.5 * a * m) * (w * mu / max(m, 1e-12))
    return log_lr, contrib


def repair_corr(R: np.ndarray, floor: float = 0.05) -> np.ndarray:
    """Make a correlation matrix positive-definite by flooring eigenvalues.

    Unlike shrinkage towards identity, this preserves measured correlations
    (e.g. t1-t2 = 0.85) while fixing near-singularity from unmeasured pairs
    whose family defaults may clash with corpus values. The floor should be
    well above machine epsilon — 0.05 keeps the condition number ≤ 20.
    """
    from numpy.linalg import eigh
    vals, vecs = eigh(R)
    vals = np.maximum(vals, floor)
    R2 = vecs @ np.diag(vals) @ vecs.T
    # Re-normalise to unit diagonal (eigenvalue floor may shift it).
    d = np.sqrt(np.diag(R2))
    R2 = R2 / np.outer(d, d)
    return R2


def shrink_correlation(R: np.ndarray, intensity: float) -> np.ndarray:
    """Blend a correlation matrix towards the identity.

    Without this, a correlation estimated from a modest corpus can be
    near-singular; inverting it then amplifies noise into enormous spurious
    evidence. Shrinkage trades a little power for not producing nonsense.
    """
    k = R.shape[0]
    lam = min(max(intensity, 0.0), 1.0)
    out = (1 - lam) * R + lam * np.eye(k)
    # Guarantee positive definiteness even if R arrived malformed.
    eig = np.linalg.eigvalsh(out)
    if eig.min() < 1e-6:
        out = out + (1e-6 - eig.min()) * np.eye(k)
        d = np.sqrt(np.diag(out))
        out = out / np.outer(d, d)
    return out


def _family_default(fam_a: str | None, fam_b: str | None) -> float:
    """Conservative correlation default for a signal pair based on families.

    These are deliberately high: understating correlation overstates evidence,
    which fails towards false accusation. repair_corr(floor=0.05) handles any
    ill-conditioning from mixing these with lower corpus-measured values.
    """
    if fam_a and fam_a == fam_b:
        return 0.75  # same family (e.g. deep_only ↔ t1, both accuracy)
    if {fam_a, fam_b} <= {"accuracy", "shape"}:
        return 0.45  # accuracy ↔ shape
    return 0.15  # cross-family (accuracy ↔ time, etc.)


# Family assignments: signals that measure the same underlying thing
# must not be treated as independent even when the corpus has no direct
# correlation estimate for the pair.
SIGNAL_FAMILY = {
    "t1_match": "accuracy",
    "t2_match": "accuracy",
    "t3_match": "accuracy",
    "acpl": "accuracy",
    "ipr_gap": "ipr",
    "deep_only_match": "accuracy",
    "engine_fingerprint": "accuracy",
    "cpl_tail": "accuracy",
    "slope": "shape",
    "cpl_slope": "shape",
    "switchpoint": "shape",
    "time_complexity_rho": "time",
    "time_uniformity": "time",
    "time_cadence": "time",
    "timing_accuracy_cross": "time",
}


def build_correlation(
    signals: list[Signal], corpus_R: dict | None, shrinkage: float
) -> np.ndarray:
    """Assemble the correlation sub-matrix for the signals actually present.

    For pairs covered by ``corpus_R``, use the measured value. For pairs NOT
    in the corpus (e.g. newer signals like cpl_tail, switchpoint), fall back
    to conservative family defaults rather than 0 (independence). Then apply
    light shrinkage and ``repair_corr`` with eigenvalue floor 0.05 to
    guarantee positive-definiteness without pulling measured correlations
    towards zero.
    """
    k = len(signals)
    R = np.eye(k)

    if corpus_R:
        names = [s.name for s in signals]
        for i, a in enumerate(names):
            for j, b in enumerate(names):
                if i >= j:
                    continue
                v = corpus_R.get(a, {}).get(b)
                if v is None:
                    v = corpus_R.get(b, {}).get(a)
                if v is not None:
                    R[i, j] = R[j, i] = float(v)
                else:
                    # Unmeasured pair: family default, not 0
                    fa = SIGNAL_FAMILY.get(a)
                    fb = SIGNAL_FAMILY.get(b)
                    R[i, j] = R[j, i] = _family_default(fa, fb)
    else:
        # No corpus estimate: use family-based priors throughout.
        for i, a in enumerate(signals):
            for j, b in enumerate(signals):
                if i >= j:
                    continue
                fa = SIGNAL_FAMILY.get(a.name)
                fb = SIGNAL_FAMILY.get(b.name)
                if fa and fa == fb:
                    R[i, j] = R[j, i] = 0.75
                elif {fa, fb} <= {"accuracy", "shape"}:
                    R[i, j] = R[j, i] = 0.45
                else:
                    R[i, j] = R[j, i] = 0.15

    # Light shrinkage (preserves structure), then eigenvalue floor to fix
    # any ill-conditioning from mixing corpus values with family defaults.
    R = shrink_correlation(R, shrinkage)
    R = repair_corr(R, floor=0.05)
    return R


def combine(
    signals: list[Signal],
    prior: float = DEFAULT_PRIOR,
    corpus_R: dict | None = None,
    shrinkage: float = DEFAULT_SHRINKAGE,
    extra_caveats: list[str] | None = None,
    uncalibrated: bool = False,
    n_mc: int = 4000,
    baseline_n_games: int = 0,
    seed: int = 0,
) -> Verdict:
    """Combine signals into a posterior with a sensitivity band."""
    caveats = list(extra_caveats or [])
    usable = [s for s in signals if s.weight > 0 and math.isfinite(s.z)]

    if not usable:
        caveats.append("no usable signals; the posterior is just the prior")
        return Verdict(
            posterior=prior,
            posterior_low=prior,
            posterior_high=prior,
            prior=prior,
            log_lr=0.0,
            decibans=0.0,
            decibans_naive=0.0,
            redundancy=0.0,
            rows=[],
            caveats=caveats,
            n_signals=0,
            uncalibrated=uncalibrated,
            log_lrs_by_rate=[0.0] * len(MIXTURE_RATES),
        )

    clipped = [s for s in usable if abs(s.z) > MAX_ABS_Z]
    z_raw = np.array([max(-MAX_ABS_Z, min(MAX_ABS_Z, s.z)) for s in usable])
    mu = np.array([s.effective_effect for s in usable])

    # Null calibration is now applied in _signal() (per-band when available,
    # pooled fallback otherwise). The z-scores arriving here are already
    # corrected, so no further adjustment is needed. The EMPIRICAL_NULL dict
    # is retained as documentation of the measured values.
    z = z_raw
    if clipped:
        caveats.append(
            "z-score(s) for "
            + ", ".join(s.label for s in clipped)
            + f" exceeded +/-{MAX_ABS_Z:g} and were clipped. A deviation that "
            "extreme usually means the reference model does not fit this game, "
            "not that the evidence is overwhelming."
        )

    R = build_correlation(usable, corpus_R, shrinkage)

    # Mixture over assistance rates. DEFAULT_EFFECTS are calibrated for
    # rate 1.0 (full assistance), so mu(r) = r * e. The per-rate log-LRs
    # are stored on the Verdict so that multi-game aggregation can sum
    # them per rate and logsumexp once (one rate per player, not per game).
    per_rate_lls: list[float] = []
    contribs_by_rate = []
    for rate in MIXTURE_RATES:
        mu_r = mu * rate
        ll_r, c_r = _projected_llr(z, R, mu_r)
        per_rate_lls.append(ll_r)
        contribs_by_rate.append(c_r)

    log_lr_arr = np.array(per_rate_lls)
    # Per-game marginal LR (used for single-game display)
    log_lr_mix = float(logsumexp(log_lr_arr + MIXTURE_LOG_PRIOR))
    # For attribution, use the MAP rate's contributions
    best_rate_idx = int(np.argmax(log_lr_arr + MIXTURE_LOG_PRIOR))
    contrib = contribs_by_rate[best_rate_idx]

    log_lr = log_lr_mix

    # Naive (independence-assuming) total, for the redundancy diagnostic.
    naive_each = mu * z - 0.5 * mu * mu
    log_lr_naive = float(naive_each.sum())

    db = 10.0 * log_lr / math.log(10)
    db_naive = 10.0 * log_lr_naive / math.log(10)
    redundancy = 0.0
    if abs(db_naive) > 1e-9:
        redundancy = max(0.0, 1.0 - db / db_naive) if db_naive > 0 else 0.0

    rows = [
        EvidenceRow(
            name=s.name,
            label=s.label,
            observed=s.observed,
            expected=s.expected,
            z=float(z[i]),
            weight=s.weight,
            decibans_naive=float(10.0 * naive_each[i] / math.log(10)),
            decibans_adjusted=float(10.0 * contrib[i] / math.log(10)),
            detail=s.detail,
        )
        for i, s in enumerate(usable)
    ]
    rows.sort(key=lambda r: -abs(r.decibans_adjusted))

    posterior = _sigmoid(_logit(prior) + log_lr)

    # ---- sensitivity band ------------------------------------------------
    # The dominant uncertainty is in the baseline itself: band means, spreads
    # and effect sizes are all estimated from a finite corpus. Monte Carlo over
    # that uncertainty to show how sensitive the point estimate is to the
    # reference model -- this is NOT a Bayesian credible interval.
    rng = np.random.default_rng(seed)
    if baseline_n_games >= 50:
        z_jitter = 1.0 / math.sqrt(baseline_n_games)
        mu_rel = 0.20
    else:
        # Shipped defaults: the reference model is essentially a guess, and the
        # interval has to be wide enough to admit that.
        z_jitter = 0.35
        mu_rel = 0.40

    # Precompute NNLS weights once per rate; reuse for all MC draws.
    w_bases = [_nnls_weights(R, mu * r) for r in MIXTURE_RATES]
    draws = np.empty(n_mc)
    for i in range(n_mc):
        z_i = z + rng.normal(0.0, z_jitter * np.maximum(1.0, np.abs(z)))
        mu_i = mu * (1.0 + rng.normal(0.0, mu_rel, size=len(mu)))
        mu_i = np.maximum(mu_i, 0.05)
        ll_rates = []
        for ri, rate in enumerate(MIXTURE_RATES):
            mu_r = mu_i * rate
            ll_r, _ = _projected_llr(z_i, R, mu_r, w=w_bases[ri])
            ll_rates.append(ll_r)
        ll_mix = float(logsumexp(np.array(ll_rates) + MIXTURE_LOG_PRIOR))
        draws[i] = _sigmoid(_logit(prior) + ll_mix)

    lo, hi = np.percentile(draws, [5.0, 95.0])

    if uncalibrated:
        caveats.insert(
            0,
            "running on shipped default baselines, not a fitted corpus. Treat "
            "the posterior as indicative only -- the reference distributions "
            "have not been validated against real games on this machine.",
        )
    if len(usable) < 3:
        caveats.append(
            f"only {len(usable)} signal(s) available; the combination has little "
            "to work with"
        )
    if redundancy > 0.45:
        caveats.append(
            f"{redundancy:.0%} of the raw evidence was double counting between "
            f"correlated signals and has been removed"
        )

    return Verdict(
        posterior=posterior,
        posterior_low=float(lo),
        posterior_high=float(hi),
        prior=prior,
        log_lr=log_lr,
        decibans=db,
        decibans_naive=db_naive,
        redundancy=redundancy,
        rows=rows,
        caveats=caveats,
        n_signals=len(usable),
        uncalibrated=uncalibrated,
        log_lrs_by_rate=per_rate_lls,
    )


def games_curve(
    verdict: Verdict,
    prior: float | None = None,
    max_games: int = 40,
) -> list[dict]:
    """Posterior as a function of how many comparable games were analysed.

    For N independent games showing the same per-game deviation, the sufficient
    statistic is the sum of z-scores, and the algebra collapses to log LR
    scaling linearly in N. So this is an honest extrapolation of "what if we saw
    this much evidence N times", not a projection of certainty.
    """
    p0 = prior if prior is not None else verdict.prior
    base = verdict.log_lr
    out = []
    for n in range(1, max_games + 1):
        out.append(
            {
                "games": n,
                "log_lr": base * n,
                "decibans": 10.0 * base * n / math.log(10),
                "posterior": _sigmoid(_logit(p0) + base * n),
            }
        )
    return out


def games_to_threshold(
    verdict: Verdict, threshold: float = 0.95, prior: float | None = None
) -> int | None:
    """How many comparable games before the posterior would cross a threshold.

    Returns None when the per-game evidence points away from guilt, in which
    case no amount of repetition gets there.
    """
    p0 = prior if prior is not None else verdict.prior
    if verdict.log_lr <= 0:
        return None
    need = _logit(threshold) - _logit(p0)
    return max(1, math.ceil(need / verdict.log_lr))


# ======================================================================
# Per-move log-LR scoring (M5)
# ======================================================================

import json as _json
from pathlib import Path as _Path

_P1_MODEL_CACHE: dict | None = None


def _load_p1_model() -> dict:
    """Load the per-band logistic regression for position-specific p1."""
    global _P1_MODEL_CACHE
    if _P1_MODEL_CACHE is not None:
        return _P1_MODEL_CACHE
    p = _Path(__file__).resolve().parent.parent / "cache" / "p1_logistic_model.json"
    if p.exists():
        _P1_MODEL_CACHE = _json.loads(p.read_text())
    else:
        _P1_MODEL_CACHE = {}
    return _P1_MODEL_CACHE


def predict_p1(move: dict, band_name: str) -> float:
    """Position-specific probability of matching engine top-1.

    Uses a per-band logistic regression fitted on the training corpus
    (AUC ~0.75, perfectly calibrated per band).

    ``move`` must contain: gap_cp, legal_moves, complexity,
    f_spread, f_gap_small, f_only_move.
    """
    model = _load_p1_model()
    bm = model.get(band_name)
    if bm is None:
        # Fallback: try nearest band by name, or return a safe default
        for candidate in sorted(model.keys()):
            bm = model[candidate]  # use any available
        if bm is None:
            return 0.40
    x = np.array([
        math.log1p(max(move.get("gap_cp") or 0, 0) / 100.0),
        math.log(max(move.get("legal_moves") or move.get("legal", 30), 1)),
        move.get("complexity", move.get("cx", 0.5)),
        move.get("f_spread", move.get("spread", 0.5)),
        move.get("f_gap_small", move.get("gap_small", 0.5)),
        move.get("f_only_move", move.get("only_move", 0.0)),
    ])
    logit = bm["intercept"] + float(x @ np.array(bm["coefs"]))
    p = 1.0 / (1.0 + math.exp(-logit))
    return max(0.01, min(0.99, p))


def combine_permove(
    moves: list[dict],
    band_name: str,
    prior: float = DEFAULT_PRIOR,
    signal_rows: list[EvidenceRow] | None = None,
    extra_caveats: list[str] | None = None,
    uncalibrated: bool = False,
    n_mc: int = 4000,
    baseline_n_games: int = 0,
    seed: int = 0,
    baseline: "Baseline | None" = None,
) -> Verdict:
    """Score a game using per-move log-likelihood ratios.

    For each move, computes a binary log-LR using position-specific p1
    from the fitted logistic model, then sums across moves and applies
    the mixture over assistance rates.

    When a fitted baseline with per-band permove null calibration is
    available, the raw dB score is z-normalised against the band's
    honest distribution and rescaled to the pooled reference.  This
    ensures uniform FPR across rating bands.

    This replaces combine() as the primary scoring function.
    The old signal-based rows can still be passed for display purposes.
    """
    caveats = list(extra_caveats or [])

    if not moves or len(moves) < 8:
        caveats.append("too few scoreable moves for per-move scoring")
        return Verdict(
            posterior=prior, posterior_low=prior, posterior_high=prior,
            prior=prior, log_lr=0.0, decibans=0.0, decibans_naive=0.0,
            redundancy=0.0, rows=signal_rows or [], caveats=caveats,
            n_signals=0, uncalibrated=uncalibrated,
            log_lrs_by_rate=[0.0] * len(MIXTURE_RATES),
        )

    model = _load_p1_model()
    if not model.get(band_name):
        caveats.append(
            f"no per-move p1 model for band {band_name}; falling back to "
            f"flat p1=0.35"
        )

    n_rates = len(MIXTURE_RATES)
    per_rate_ll = np.zeros(n_rates)

    for m in moves:
        t1 = m.get("matched_t1", m.get("t1", 0))
        p1 = predict_p1(m, band_name)

        for i, r in enumerate(MIXTURE_RATES):
            if t1:
                per_rate_ll[i] += math.log(r + (1 - r) * p1) - math.log(p1)
            else:
                per_rate_ll[i] += math.log(max(1 - r, 1e-10))

    log_lr = float(logsumexp(per_rate_ll + MIXTURE_LOG_PRIOR))
    db = 10.0 * log_lr / math.log(10)

    # --- Per-band null calibration ---
    # Correct the raw dB so every band's honest distribution has the same
    # location and scale.  Without this, bands with systematically higher
    # raw scores (e.g. 2000-2200) get inflated FPR at a global threshold.
    _band_cal = None
    _pooled_cal = None
    if baseline is not None:
        ncb = getattr(baseline, "null_calibration_by_band", None)
        nc = getattr(baseline, "null_calibration", None)
        if callable(ncb):
            ncb = ncb()
        if callable(nc):
            nc = nc()
        if ncb and nc:
            _band_cal = (ncb.get(band_name) or {}).get("permove")
            _pooled_cal = nc.get("permove")

    if _band_cal and _pooled_cal and _band_cal["n"] >= 30:
        band_bias = _band_cal["bias"]
        band_scale = max(_band_cal["scale"], 0.1)
        pool_bias = _pooled_cal["bias"]
        pool_scale = max(_pooled_cal["scale"], 0.1)
        db = (db - band_bias) / band_scale * pool_scale + pool_bias
        log_lr = db * math.log(10) / 10.0
        # Also calibrate per-rate LLs proportionally so multi-game
        # aggregation benefits from the same correction.
        raw_db_rates = 10.0 * per_rate_ll / math.log(10)
        cal_db_rates = (raw_db_rates - band_bias) / band_scale * pool_scale + pool_bias
        per_rate_ll = cal_db_rates * math.log(10) / 10.0

    posterior = _sigmoid(_logit(prior) + log_lr)

    # Sensitivity band: jitter the p1 predictions
    if n_mc > 0:
        rng = np.random.default_rng(seed)
        p1_jitter = 0.05 if baseline_n_games >= 50 else 0.10
        draws = np.empty(n_mc)
        for d in range(n_mc):
            ll_mc = np.zeros(n_rates)
            for m in moves:
                t1 = m.get("matched_t1", m.get("t1", 0))
                p1 = predict_p1(m, band_name)
                p1 = max(0.01, min(0.99, p1 + rng.normal(0, p1_jitter * p1)))
                for i, r in enumerate(MIXTURE_RATES):
                    if t1:
                        ll_mc[i] += math.log(r + (1 - r) * p1) - math.log(p1)
                    else:
                        ll_mc[i] += math.log(max(1 - r, 1e-10))
            ll_mix = float(logsumexp(ll_mc + MIXTURE_LOG_PRIOR))
            if _band_cal and _pooled_cal and _band_cal["n"] >= 30:
                db_mc = 10.0 * ll_mix / math.log(10)
                db_mc = (db_mc - band_bias) / band_scale * pool_scale + pool_bias
                ll_mix = db_mc * math.log(10) / 10.0
            draws[d] = _sigmoid(_logit(prior) + ll_mix)
        lo, hi = float(np.percentile(draws, 5.0)), float(np.percentile(draws, 95.0))
    else:
        lo, hi = posterior, posterior

    if uncalibrated:
        caveats.insert(
            0,
            "running on shipped default baselines, not a fitted corpus. Treat "
            "the posterior as indicative only.",
        )
    if len(moves) < 25:
        caveats.append(
            f"only {len(moves)} scoreable decisions; single-game evidence is "
            "intrinsically weak and the credible interval reflects that"
        )

    return Verdict(
        posterior=posterior,
        posterior_low=lo,
        posterior_high=hi,
        prior=prior,
        log_lr=log_lr,
        decibans=db,
        decibans_naive=db,  # no naive/adjusted distinction for per-move
        redundancy=0.0,
        rows=signal_rows or [],
        caveats=caveats,
        n_signals=len(moves),
        uncalibrated=uncalibrated,
        log_lrs_by_rate=per_rate_ll.tolist(),
    )


def interpret(verdict: Verdict) -> str:
    """A short, deliberately unexciting verbal summary.

    Negative decibans are evidence *against* assistance, which is the common
    case and must read as such -- lumping it in with "no evidence" throws away
    the most reassuring result the tool can produce.
    """
    db = verdict.decibans
    if db <= -10:
        phrase = "strong evidence against engine assistance"
    elif db <= -3:
        phrase = "evidence against engine assistance"
    elif db < 3:
        phrase = "no meaningful evidence either way"
    elif db < 6:
        phrase = "weak evidence of engine assistance"
    elif db < 10:
        phrase = "moderate evidence of engine assistance"
    elif db < 20:
        phrase = "strong evidence of engine assistance"
    else:
        phrase = "very strong evidence of engine assistance"
    return (
        f"{phrase} ({db:+.1f} decibans; posterior {verdict.posterior:.2%}, "
        f"90% sensitivity band {verdict.posterior_low:.2%}-{verdict.posterior_high:.2%} "
        f"at a {verdict.prior:.1%} base rate)"
    )
