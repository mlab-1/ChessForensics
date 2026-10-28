"""Loading and querying the baseline (reference-model) file.

Separate from baseline.py, which *builds* the file. This module only reads it,
so the app has no dependency on the corpus machinery.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASELINE_DIR = ROOT / "baselines"
FITTED_PATH = BASELINE_DIR / "model_v1.json"
DEFAULT_PATH = BASELINE_DIR / "default.json"

# Band edges used throughout. Bands are 200 Elo wide through the main range,
# with wider catch-all bins at the extremes (beginners and super-GMs).
BAND_EDGES = [0, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400, 2600, 2800, 4000]

# Below this many corpus games a fitted file is treated as *provisional*: it is
# still loaded, because even a small real fit gets the scale of ``s`` roughly
# right and the shipped defaults do not, but it must not be presented as a
# calibrated reference. Without this gate a 15-game corpus silently removes the
# "uncalibrated" banner, which is exactly the kind of quiet overconfidence this
# project exists to avoid.
MIN_CREDIBLE_GAMES = 300


def band_for(elo: int | float | None) -> str:
    """Name the rating band an Elo falls into."""
    if elo is None:
        return "unknown"
    for lo, hi in zip(BAND_EDGES, BAND_EDGES[1:]):
        if lo <= elo < hi:
            return f"{lo}-{hi}"
    return f"{BAND_EDGES[-2]}-{BAND_EDGES[-1]}"


def band_midpoint(band: str) -> float | None:
    """Representative Elo for a band.

    The outermost bands are open-ended (``0-1000``, ``2400-4000``) and their
    arithmetic midpoints -- 500 and 3200 -- are nowhere near where their players
    actually sit. Feeding 3200 into the IPR interpolation would stretch the
    s -> Elo curve over 900 Elo of empty space and badly distort every strong
    player's estimate, so the open ends are clamped to a realistic offset.
    """
    try:
        lo_s, hi_s = band.split("-")
        lo, hi = float(lo_s), float(hi_s)
    except (ValueError, AttributeError):
        return None
    if hi - lo > 400:
        if lo <= BAND_EDGES[0]:
            return hi - 150.0
        return lo + 150.0
    return (lo + hi) / 2


@dataclass
class Baseline:
    """The reference model: expected behaviour per rating band."""

    data: dict
    path: Path | None
    is_default: bool

    @property
    def provenance(self) -> dict:
        return self.data.get("provenance", {})

    @property
    def n_games(self) -> int:
        return int(self.provenance.get("games", 0))

    @property
    def multipv(self) -> int:
        return int(self.provenance.get("multipv", 5))

    @property
    def bands(self) -> dict:
        return self.data.get("bands", {})

    def band(self, elo: int | float | None) -> dict:
        """Reference statistics for a rating band, with nearest-band fallback."""
        name = band_for(elo)
        bands = self.bands
        if name in bands:
            return bands[name]
        # Fall back to the nearest band that actually has data, so an unusual
        # rating does not silently get an empty reference.
        mid = band_midpoint(name)
        candidates = [(b, band_midpoint(b)) for b in bands]
        candidates = [(b, m) for b, m in candidates if m is not None]
        if not candidates or mid is None:
            return {}
        nearest = min(candidates, key=lambda t: abs(t[1] - mid))
        out = dict(bands[nearest[0]])
        out["_fallback_from"] = nearest[0]
        return out

    def calibration_curve(self) -> list[tuple[float, float]]:
        """[(elo_midpoint, fitted s)] across bands, for the IPR mapping.

        The curve is forced to be non-increasing in Elo. That is not cosmetic:
        the IPR mapping inverts this curve, so it has to be invertible, and a
        per-band fit on a finite corpus is noisy enough to produce reversals
        (a 2300 band landing above a 2100 band on four player-games, say). An
        un-monotone curve silently scrambles the Elo axis and collapses every
        IPR onto one value.

        Monotonicity is a structural fact rather than an assumption -- a
        stronger player tolerates less eval loss, so ``s`` must fall with
        rating -- which makes isotonic regression the right way to impose it:
        reversals get pooled into their weighted mean and genuine signal is
        left alone.

        When the fitted model doesn't cover the full rating range, the curve
        is extended by log-linear extrapolation from the last two fitted points.
        This is more accurate than merging in default values, which may be on a
        different scale.
        """
        import math as _math

        pts = []
        for name, stats in self.bands.items():
            mid = band_midpoint(name)
            s = (stats.get("choice_model") or {}).get("s")
            if mid is not None and s:
                w = float(stats.get("player_games") or 1)
                pts.append((mid, float(s), max(w, 1.0)))
        # Extend coverage by extrapolating the fitted trend (only when the
        # fitted curve has enough bands for a reliable slope estimate)
        if not self.is_default and len(pts) >= 4:
            pts_sorted = sorted(pts)
            # High-end: extrapolate two bands above the highest fitted point
            e1, s1, _ = pts_sorted[-2]
            e2, s2, _ = pts_sorted[-1]
            if s1 > 0 and s2 > 0 and e2 > e1:
                ratio = s2 / s1  # per-(e2-e1) Elo step
                step = e2 - e1
                for i in range(1, 3):
                    ext_elo = e2 + step * i
                    ext_s = s2 * (ratio ** i)
                    pts.append((ext_elo, ext_s, 1.0))
            # Low-end: extrapolate one band below the lowest fitted point
            e1, s1, _ = pts_sorted[0]
            e2, s2, _ = pts_sorted[1]
            if s1 > 0 and s2 > 0 and e2 > e1:
                ratio = s1 / s2  # going weaker -> higher s
                step = e2 - e1
                ext_elo = e1 - step
                ext_s = s1 * ratio
                if ext_elo >= 0:
                    pts.append((ext_elo, ext_s, 1.0))
        pts.sort()
        if len(pts) < 2:
            return [(e, s) for e, s, _ in pts]

        # Pool-adjacent-violators on s, descending in Elo.
        blocks: list[tuple[float, float, list[float]]] = []  # (sum w*s, sum w, elos)
        for elo, s, w in pts:
            blocks.append((w * s, w, [elo]))
            while len(blocks) >= 2 and (
                blocks[-2][0] / blocks[-2][1] < blocks[-1][0] / blocks[-1][1]
            ):
                a, b = blocks[-2], blocks[-1]
                blocks[-2:] = [(a[0] + b[0], a[1] + b[1], a[2] + b[2])]

        out: list[tuple[float, float]] = []
        for num, den, elos in blocks:
            level = num / den
            for elo in elos:
                out.append((elo, level))
        return out

    # The builder writes an explicit ``null`` for anything it could not
    # estimate, so ``.get(key, default)`` is not enough -- the key is present
    # and holds None. Callers treat {} and None very differently (fall back to
    # defaults vs crash), so normalise here rather than at every use site.
    @property
    def correlation(self) -> dict | None:
        # "signal_correlations" (from controls.py z-score measurement) takes
        # precedence over "signal_correlation" (from corpus fit_all).
        return (self.data.get("signal_correlations")
                or self.data.get("signal_correlation")
                or None)

    @property
    def effects(self) -> dict:
        return self.data.get("signal_effects") or {}

    @property
    def complexity_weights(self) -> dict | None:
        return self.data.get("complexity_weights") or None

    @property
    def null_calibration(self) -> dict | None:
        """Per-signal null-distribution calibration: {name: {bias, scale, n}}.

        Measured from a held-out negative control. ``bias`` is the mean z-score
        under H0 (should be 0 if the model is perfect). ``scale`` is the
        standard deviation (should be 1). Subtracting the bias and dividing by
        scale centres the null so that the posterior is not systematically
        pushed one way by model misspecification.
        """
        return self.data.get("null_calibration") or None

    @property
    def null_calibration_by_band(self) -> dict | None:
        """Per-band, per-signal null calibration: {band: {name: {bias, scale, n}}}.

        Allows band-specific correction so that e.g. super-GM games aren't
        corrected with biases measured at 1400-1600.
        """
        return self.data.get("null_calibration_by_band") or None

    @property
    def provisional(self) -> bool:
        """Fitted, but on too little data to be relied on."""
        return not self.is_default and self.n_games < MIN_CREDIBLE_GAMES

    @property
    def trustworthy(self) -> bool:
        return not self.is_default and not self.provisional

    def shrinkage(self) -> float:
        """Shrinkage intensity for the correlation matrix, from corpus size."""
        n = self.n_games
        if n <= 0:
            return 0.60  # defaults: shrink hard, we know nothing
        if n < 200:
            return 0.45
        if n < 1000:
            return 0.30
        if n < 3000:
            return 0.20
        return 0.12

    def describe(self) -> str:
        if self.is_default:
            return "shipped defaults (uncalibrated)"
        p = self.provenance
        base = (
            f"fitted on {p.get('games', '?')} games "
            f"({p.get('player_games', '?')} player-games, "
            f"{p.get('positions', '?')} positions), depth {p.get('depth', '?')}, "
            f"built {p.get('date', '?')}"
        )
        if self.provisional:
            return (
                f"PROVISIONAL -- {base}. Below the {MIN_CREDIBLE_GAMES}-game "
                f"threshold for a usable reference."
            )
        return base


_defaults_cache: Baseline | None = None


def _load_defaults() -> Baseline | None:
    """Cached loader for the shipped default baseline."""
    global _defaults_cache
    if _defaults_cache is not None:
        return _defaults_cache
    if DEFAULT_PATH.exists():
        try:
            data = json.loads(DEFAULT_PATH.read_text())
            _defaults_cache = Baseline(data, DEFAULT_PATH, is_default=True)
            return _defaults_cache
        except (json.JSONDecodeError, OSError):
            pass
    return None


def load_baseline(prefer_fitted: bool = True) -> Baseline:
    """Load the fitted baseline if present, else the shipped defaults."""
    if prefer_fitted and FITTED_PATH.exists():
        try:
            data = json.loads(FITTED_PATH.read_text())
            if data.get("bands"):
                return Baseline(data, FITTED_PATH, is_default=False)
        except (json.JSONDecodeError, OSError):
            pass
    if DEFAULT_PATH.exists():
        data = json.loads(DEFAULT_PATH.read_text())
        return Baseline(data, DEFAULT_PATH, is_default=True)
    return Baseline({"bands": {}, "provenance": {}}, None, is_default=True)
