"""V1/V2 threshold transfer analysis.

Methodology (no data snooping):
  1. Load V1 (calibrated) and V2 (raw) per-game results.
  2. Re-calibrate V2 scores offline using the same per-band null
     calibration parameters that V1 used, so both sets are on the
     same dB scale.
  3. Within each rating band, find the dB thresholds on V1
     corresponding to target FPRs (5%, 1%, 0.1%).
  4. Apply those exact band-specific thresholds to V2 and report
     the observed FPR with Clopper-Pearson exact 95% CIs.
  5. Also report global (un-stratified) results for comparison,
     to show the band-composition artifact.

V1 is the "validation" set (threshold selection).
V2 is the "test" set (held-out, touched once).

Usage:
    py -3 -u -X utf8 scripts/v1v2_threshold_analysis.py
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats as sp_stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

V1_RESULTS = ROOT / "cache" / "oos_chesscom_results.jsonl"
V2_RESULTS = ROOT / "cache" / "oos_chesscom_v2_results.jsonl"
OUTPUT_FILE = ROOT / "cache" / "v1v2_threshold_analysis.json"

TARGET_FPRS = [0.05, 0.01, 0.001]  # 5%, 1%, 0.1%
BANDS_WITH_DATA = ["2000-2200", "2200-2400"]  # bands with n>=100 in both sets
MIN_BAND_N = 100


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Exact binomial CI (Clopper-Pearson). Returns (lo, hi)."""
    if n == 0:
        return (0.0, 1.0)
    if k == 0:
        lo = 0.0
    else:
        lo = sp_stats.beta.ppf(alpha / 2, k, n - k + 1)
    if k == n:
        hi = 1.0
    else:
        hi = sp_stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return (float(lo), float(hi))


def calibrate_db(raw_db: float, band: str, cal_by_band: dict, cal_pooled: dict) -> float:
    """Apply per-band null calibration: dB_cal = (dB_raw - b) / s * s_pool + b_pool."""
    bc = (cal_by_band.get(band) or {}).get("permove")
    pc = cal_pooled.get("permove")
    if bc and pc and bc.get("n", 0) >= 30:
        return (raw_db - bc["bias"]) / max(bc["scale"], 0.1) * max(pc["scale"], 0.1) + pc["bias"]
    return raw_db


def db_to_posterior(db: float, prior: float = 0.01) -> float:
    log_lr = db * math.log(10) / 10.0
    logit_prior = math.log(prior / (1 - prior))
    return 1.0 / (1.0 + math.exp(-(logit_prior + log_lr)))


def empirical_threshold(values: np.ndarray, target_fpr: float) -> float:
    """(1 - target_fpr) quantile — the dB above which fraction target_fpr lies."""
    return float(np.quantile(values, 1.0 - target_fpr))


def fpr_at_threshold(values: np.ndarray, threshold: float) -> tuple[int, int, float]:
    """Count exceedances and return (k, n, fpr)."""
    n = len(values)
    k = int(np.sum(values > threshold))
    return k, n, k / n if n > 0 else 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    from chessforensics.baselines import load_baseline

    baseline = load_baseline()
    ncb = baseline.null_calibration_by_band
    nc = baseline.null_calibration
    if callable(ncb):
        ncb = ncb()
    if callable(nc):
        nc = nc()

    # --- Load ---
    print("Loading V1 and V2 results...")
    v1_raw = load_jsonl(V1_RESULTS)
    v2_raw = load_jsonl(V2_RESULTS)

    v1_players = set(r["username"] for r in v1_raw)
    v2_players = set(r["username"] for r in v2_raw)
    overlap = v1_players & v2_players
    print(f"  V1: {len(v1_raw)} games, {len(v1_players)} players")
    print(f"  V2: {len(v2_raw)} games, {len(v2_players)} players")
    print(f"  Player overlap: {len(overlap)}")

    # --- V1 already calibrated; re-calibrate V2 ---
    print("\nRe-calibrating V2 with per-band null calibration...")
    v1_db = np.array([r["decibans"] for r in v1_raw])

    v2_db_cal = np.array([
        calibrate_db(r["decibans"], r["band"], ncb, nc) for r in v2_raw
    ])
    # Store calibrated dB back into rows for band-level access
    for r, db in zip(v2_raw, v2_db_cal):
        r["db_cal"] = db

    print(f"  V1 median dB: {np.median(v1_db):+.2f}")
    print(f"  V2 median dB (calibrated): {np.median(v2_db_cal):+.2f}")

    # Band composition
    from collections import Counter
    v1_bands = Counter(r["band"] for r in v1_raw)
    v2_bands = Counter(r["band"] for r in v2_raw)
    print("\n  Band composition:")
    for b in sorted(set(list(v1_bands) + list(v2_bands))):
        n1, n2 = v1_bands.get(b, 0), v2_bands.get(b, 0)
        p1 = n1 / len(v1_raw) * 100
        p2 = n2 / len(v2_raw) * 100
        print(f"    {b}: V1={n1:>6} ({p1:>5.1f}%)  V2={n2:>6} ({p2:>5.1f}%)")

    results = {}

    # =======================================================================
    # SECTION 1: Band-stratified per-game thresholds (PRIMARY result)
    # =======================================================================
    print(f"\n{'='*72}")
    print("BAND-STRATIFIED PER-GAME THRESHOLD TRANSFER (primary analysis)")
    print(f"{'='*72}")
    print("For each band with n>=100 in both sets, the threshold is the")
    print("empirical quantile from V1. V2 FPR is the held-out estimate.\n")

    band_strat = {}
    for target in TARGET_FPRS:
        band_strat[f"fpr_{target}"] = {}
        for band in BANDS_WITH_DATA:
            v1_band = np.array([r["decibans"] for r in v1_raw if r["band"] == band])
            v2_band = np.array([r["db_cal"] for r in v2_raw if r["band"] == band])

            if len(v1_band) < MIN_BAND_N or len(v2_band) < MIN_BAND_N:
                continue

            thr = empirical_threshold(v1_band, target)
            k1, n1, fpr1 = fpr_at_threshold(v1_band, thr)
            k2, n2, fpr2 = fpr_at_threshold(v2_band, thr)
            ci_lo, ci_hi = clopper_pearson(k2, n2)

            entry = {
                "band": band,
                "v1_threshold_db": round(thr, 3),
                "v1_n": n1, "v1_k": k1, "v1_fpr": round(fpr1, 6),
                "v2_n": n2, "v2_k": k2, "v2_fpr": round(fpr2, 6),
                "v2_ci95_lo": round(ci_lo, 6), "v2_ci95_hi": round(ci_hi, 6),
            }
            band_strat[f"fpr_{target}"][band] = entry

            flag = ""
            if ci_hi < target:
                flag = "  << conservative (CI entirely below target)"
            elif ci_lo > target:
                flag = "  ** inflated (CI entirely above target)"

            print(f"  Target {target:.1%}, band {band}:")
            print(f"    V1 threshold: {thr:+.2f} dB  |  V1: {k1}/{n1} = {fpr1:.4%}")
            print(f"    V2 held-out:  {k2}/{n2} = {fpr2:.4%}  "
                  f"95% CI [{ci_lo:.4%}, {ci_hi:.4%}]{flag}")

    results["band_stratified_per_game"] = band_strat

    # =======================================================================
    # SECTION 2: Global per-game thresholds (shows band-composition artifact)
    # =======================================================================
    print(f"\n{'='*72}")
    print("GLOBAL PER-GAME THRESHOLD TRANSFER (for comparison)")
    print(f"{'='*72}")
    print("Single threshold from V1's marginal distribution. Expected to")
    print("show inflated V2 FPR due to different band composition.\n")

    global_pg = []
    for target in TARGET_FPRS:
        thr = empirical_threshold(v1_db, target)
        k1, n1, fpr1 = fpr_at_threshold(v1_db, thr)
        k2, n2, fpr2 = fpr_at_threshold(v2_db_cal, thr)
        ci_lo, ci_hi = clopper_pearson(k2, n2)

        entry = {
            "target_fpr": target,
            "v1_threshold_db": round(thr, 3),
            "v1_n": n1, "v1_k": k1, "v1_fpr": round(fpr1, 6),
            "v2_n": n2, "v2_k": k2, "v2_fpr": round(fpr2, 6),
            "v2_ci95_lo": round(ci_lo, 6), "v2_ci95_hi": round(ci_hi, 6),
        }
        global_pg.append(entry)

        print(f"  Target {target:.1%}:")
        print(f"    V1 threshold: {thr:+.2f} dB  |  V1: {k1}/{n1} = {fpr1:.4%}")
        print(f"    V2 held-out:  {k2}/{n2} = {fpr2:.4%}  "
              f"95% CI [{ci_lo:.4%}, {ci_hi:.4%}]")

    results["global_per_game"] = global_pg

    # =======================================================================
    # SECTION 3: Band-stratified per-player thresholds
    # =======================================================================
    print(f"\n{'='*72}")
    print("BAND-STRATIFIED PER-PLAYER THRESHOLD TRANSFER")
    print(f"{'='*72}")
    print("Players assigned to their modal band. Sum dB across games.\n")

    def player_agg_by_band(rows, db_key):
        """Aggregate per-player dB, assign to modal band."""
        by_player = defaultdict(lambda: {"dbs": [], "bands": []})
        for r in rows:
            by_player[r["username"]]["dbs"].append(r[db_key])
            by_player[r["username"]]["bands"].append(r["band"])

        result = {}  # band -> list of player total dBs
        for username, info in by_player.items():
            total_db = sum(info["dbs"])
            # Modal band (most frequent)
            band_counts = Counter(info["bands"])
            modal_band = band_counts.most_common(1)[0][0]
            result.setdefault(modal_band, []).append(total_db)
        return {b: np.array(v) for b, v in result.items()}

    v1_player_bands = player_agg_by_band(v1_raw, "decibans")
    v2_player_bands = player_agg_by_band(v2_raw, "db_cal")

    band_strat_player = {}
    for target in TARGET_FPRS:
        band_strat_player[f"fpr_{target}"] = {}
        for band in BANDS_WITH_DATA:
            v1p = v1_player_bands.get(band)
            v2p = v2_player_bands.get(band)
            if v1p is None or v2p is None:
                continue
            if len(v1p) < MIN_BAND_N or len(v2p) < MIN_BAND_N:
                continue

            thr = empirical_threshold(v1p, target)
            k1, n1, fpr1 = fpr_at_threshold(v1p, thr)
            k2, n2, fpr2 = fpr_at_threshold(v2p, thr)
            ci_lo, ci_hi = clopper_pearson(k2, n2)

            entry = {
                "band": band,
                "v1_threshold_db": round(thr, 3),
                "v1_n": n1, "v1_k": k1, "v1_fpr": round(fpr1, 6),
                "v2_n": n2, "v2_k": k2, "v2_fpr": round(fpr2, 6),
                "v2_ci95_lo": round(ci_lo, 6), "v2_ci95_hi": round(ci_hi, 6),
            }
            band_strat_player[f"fpr_{target}"][band] = entry

            print(f"  Target {target:.1%}, band {band}:")
            print(f"    V1 threshold: {thr:+.2f} dB  |  V1: {k1}/{n1} = {fpr1:.4%}")
            print(f"    V2 held-out:  {k2}/{n2} = {fpr2:.4%}  "
                  f"95% CI [{ci_lo:.4%}, {ci_hi:.4%}]")

    results["band_stratified_per_player"] = band_strat_player

    # =======================================================================
    # SECTION 4: Summary table for paper
    # =======================================================================
    print(f"\n{'='*72}")
    print("SUMMARY FOR PAPER (band-stratified, primary result)")
    print(f"{'='*72}")
    print(f"\n{'Band':<12} {'Target':>7} {'V1 thr (dB)':>12} "
          f"{'V2 FPR':>8} {'95% CI':>20}")
    print("-" * 65)
    for target in TARGET_FPRS:
        key = f"fpr_{target}"
        for band in BANDS_WITH_DATA:
            e = band_strat.get(key, {}).get(band)
            if e is None:
                continue
            ci_str = f"[{e['v2_ci95_lo']:.4%}, {e['v2_ci95_hi']:.4%}]"
            print(f"  {band:<12} {target:>6.1%} {e['v1_threshold_db']:>+11.2f} "
                  f"{e['v2_fpr']:>8.4%} {ci_str:>20}")

    # --- Save ---
    OUTPUT_FILE.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
