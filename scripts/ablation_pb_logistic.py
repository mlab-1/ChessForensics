r"""Ablation: Poisson-binomial z-score using M5's logistic p_m.

Isolates the contribution of per-move log-LR scoring vs improved
probability estimates.  Compares three methods on identical data:

  M1: PB z-score with band-average p  (choice model, current baseline)
  M5: Per-move log-LR with logistic p_m  (paper's primary model)
  M6: PB z-score with logistic p_m  (this ablation)

If M6 >> M1, the logistic p_m estimates are better than band averages.
If M5 >> M6, the log-LR scoring mechanism contributes beyond better p_m.
If M6 ~ M5, the advantage comes entirely from better p_m.

Usage:
    py -3.12 -X utf8 scripts/ablation_pb_logistic.py
"""
from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.special import logsumexp

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chessforensics.score import MIXTURE_RATES, MIXTURE_LOG_PRIOR, DEFAULT_EFFECTS
from chessforensics.baselines import load_baseline
from scripts.regan_comparison import (
    load_corpus_frames,
    make_honest_summary,
    inject_engine_moves,
    method1_t1_only,
    method5_permove_lr,
    _predict_p1,
    find_threshold,
    mcnemar_test,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# =====================================================================
# M6: Poisson-binomial z-score from logistic p_m
# =====================================================================

def method6_pb_logistic(row: dict, baseline) -> dict | None:
    """Poisson-binomial z-score using M5's per-position logistic p_m.

    For each scoreable position, predict p_m from the same logistic
    regression used by M5.  Then compute the Poisson-binomial z-score:

        z = (X - sum(p_m)) / sqrt(sum(p_m * (1 - p_m)))

    where X = number of top-1 matches.  This is the SAME z-score
    formula as M1 (Definition 1 in the paper), but with position-specific
    p_m from logistic regression instead of the band-average p from the
    choice model.

    The z-score is converted to a log-LR using a fixed effect size
    (same as M1), then the mixture over assistance rates is applied
    (same as M5).
    """
    band_name = row.get("band")
    if not band_name:
        return None
    frame = row.get("frame")
    if not frame or len(frame) < 8:
        return None

    # Compute per-position p_m from logistic regression (same as M5)
    probs = []
    matches = 0
    for m in frame:
        p = _predict_p1(m, band_name)
        p = max(0.01, min(0.99, p))
        probs.append(p)
        matches += int(m["t1"])

    probs = np.array(probs)
    n = len(probs)

    # Poisson-binomial z-score (same formula as Definition 1)
    expected = probs.sum()
    variance = (probs * (1 - probs)).sum()
    if variance <= 0:
        return None
    z = (matches - expected) / math.sqrt(variance)

    # Convert to log-LR using fixed effect size (same as M1)
    effect = float(baseline.effects.get("t1_match",
                   DEFAULT_EFFECTS.get("t1_match", 2.2)))
    log_lr = effect * z - 0.5 * effect ** 2
    db = 10 * log_lr / math.log(10)

    # Apply per-band null calibration (same as M5)
    ncb = baseline.null_calibration_by_band
    nc = baseline.null_calibration
    if ncb and nc:
        # Use t1_match calibration since this is a t1 z-score
        bc = (ncb.get(band_name) or {}).get("t1_match")
        pc = nc.get("t1_match")
        if bc and pc and bc.get("n", 0) >= 30:
            b_bias, b_scale = bc["bias"], max(bc["scale"], 0.1)
            p_bias, p_scale = pc["bias"], max(pc["scale"], 0.1)
            db = (db - b_bias) / b_scale * p_scale + p_bias
            log_lr = db * math.log(10) / 10.0

    return {"z": z, "log_lr": log_lr, "db": db}


def method6_pb_logistic_mixture(row: dict, baseline) -> dict | None:
    """M6b: PB z-score with logistic p_m + mixture over rates.

    Same as M6 but instead of a single fixed-effect log-LR, we compute
    z-scores under each assistance rate (adjusting expected counts) and
    apply the mixture.  This isolates the scoring mechanism (z vs log-LR)
    while keeping everything else identical to M5.
    """
    band_name = row.get("band")
    if not band_name:
        return None
    frame = row.get("frame")
    if not frame or len(frame) < 8:
        return None

    probs = []
    t1_vec = []
    for m in frame:
        p = _predict_p1(m, band_name)
        p = max(0.01, min(0.99, p))
        probs.append(p)
        t1_vec.append(int(m["t1"]))

    probs = np.array(probs)
    t1_vec = np.array(t1_vec)
    matches = t1_vec.sum()
    n = len(probs)

    # For each assistance rate, compute z-score under that alternative
    # Under H1(r): each move matched with prob r + (1-r)*p_m
    # We still test the OBSERVED count against the NULL expectation
    # (this is what a z-score approach would do)
    rates = MIXTURE_RATES
    per_rate_ll = np.zeros(len(rates))
    effect = float(baseline.effects.get("t1_match",
                   DEFAULT_EFFECTS.get("t1_match", 2.2)))

    for i, r in enumerate(rates):
        # Z-score: observed vs null expectation (same formula, always)
        expected = probs.sum()
        variance = (probs * (1 - probs)).sum()
        if variance <= 0:
            continue
        z = (matches - expected) / math.sqrt(variance)
        # Scale effect by rate (higher rate -> more signal)
        per_rate_ll[i] = effect * z - 0.5 * effect ** 2

    log_lr = float(logsumexp(per_rate_ll + MIXTURE_LOG_PRIOR))
    db = 10.0 * log_lr / math.log(10)

    # Per-band null calibration
    ncb = baseline.null_calibration_by_band
    nc = baseline.null_calibration
    if ncb and nc:
        bc = (ncb.get(band_name) or {}).get("t1_match")
        pc = nc.get("t1_match")
        if bc and pc and bc.get("n", 0) >= 30:
            b_bias, b_scale = bc["bias"], max(bc["scale"], 0.1)
            p_bias, p_scale = pc["bias"], max(pc["scale"], 0.1)
            db = (db - b_bias) / b_scale * p_scale + p_bias
            log_lr = db * math.log(10) / 10.0

    return {"z": None, "log_lr": log_lr, "db": db,
            "log_lrs_by_rate": per_rate_ll.tolist()}


# =====================================================================
# Main experiment
# =====================================================================

METHODS = {
    "M1_t1_only": method1_t1_only,
    "M5_permove": method5_permove_lr,
    "M6_pb_logistic": method6_pb_logistic,
}

DISPLAY = {
    "M1_t1_only":      "M1 (PB z, band-avg p)",
    "M5_permove":      "M5 (per-move log-LR, logistic p)",
    "M6_pb_logistic":  "M6 (PB z, logistic p)  [ABLATION]",
}


def main():
    print("=" * 70)
    print("ABLATION: Poisson-binomial z with logistic p_m")
    print("=" * 70)

    baseline = load_baseline()
    all_bands = ("1000-1200", "1200-1400", "1400-1600", "1600-1800",
                 "1800-2000", "2000-2200", "2200-2400")
    frames = load_corpus_frames(n_target=5000, bands=all_bands, split="test")
    if not frames:
        print("ERROR: no data")
        return

    print(f"  Loaded {len(frames)} player-games")
    band_dist = Counter(f["band"] for f in frames)
    print(f"  Bands: {dict(sorted(band_dist.items()))}")

    # ---- Score honest games ----
    print("\n--- Scoring honest games ---")
    honest_dbs = {m: [] for m in METHODS}

    for f in frames:
        row = make_honest_summary(f["frame"])
        row["band"] = f["band"]
        row["elo"] = f["elo"]
        for name, fn in METHODS.items():
            try:
                r = fn(row, baseline)
                if r is not None:
                    honest_dbs[name].append(r["db"])
            except Exception:
                pass

    for m in METHODS:
        arr = np.array(honest_dbs[m])
        print(f"  {DISPLAY[m]:45s} n={len(arr):5d}  "
              f"median={np.median(arr):+7.1f}  p99={np.percentile(arr, 99):+7.1f}")

    # ---- Set thresholds ----
    thresholds = {}
    for m in METHODS:
        arr = np.array(honest_dbs[m])
        thresholds[m] = {
            "fpr_1pct": find_threshold(arr, 0.01),
            "fpr_01pct": find_threshold(arr, 0.001),
        }

    print("\n  Thresholds (dB):")
    for m in METHODS:
        print(f"    {DISPLAY[m]:45s}  1%={thresholds[m]['fpr_1pct']:+7.1f}  "
              f"0.1%={thresholds[m]['fpr_01pct']:+7.1f}")

    # ---- Synthetic cheaters ----
    print("\n--- Synthetic cheater TPR ---")
    rates = [0.10, 0.25, 0.50, 1.0]
    results = {}

    for rate in rates:
        method_dbs = {m: [] for m in METHODS}
        for f in frames:
            inj = inject_engine_moves(f["frame"], rate, "uniform")
            if inj is None:
                continue
            inj["band"] = f["band"]
            inj["elo"] = f["elo"]
            for name, fn in METHODS.items():
                try:
                    r = fn(inj, baseline)
                    if r is not None:
                        method_dbs[name].append(r["db"])
                except Exception:
                    pass

        print(f"\n  r = {rate:.0%}:")
        for fpr_name, fpr_label in [("fpr_1pct", "1%"), ("fpr_01pct", "0.1%")]:
            tprs = {}
            for m in METHODS:
                arr = np.array(method_dbs[m])
                t = thresholds[m][fpr_name]
                tpr = float((arr >= t).mean()) if len(arr) > 0 else 0.0
                tprs[m] = tpr
            print(f"    TPR at {fpr_label:4s} FPR: ", end="")
            for m in METHODS:
                print(f" {DISPLAY[m].split('(')[0].strip():3s}={tprs[m]:5.1%}", end="")
            print()

            results[f"r{rate}_{fpr_name}"] = tprs

        # McNemar: M5 vs M6
        arr5 = np.array(method_dbs["M5_permove"])
        arr6 = np.array(method_dbs["M6_pb_logistic"])
        n_common = min(len(arr5), len(arr6))
        if n_common > 0:
            for fpr_name, fpr_label in [("fpr_01pct", "0.1%")]:
                flags5 = arr5[:n_common] >= thresholds["M5_permove"][fpr_name]
                flags6 = arr6[:n_common] >= thresholds["M6_pb_logistic"][fpr_name]
                mn = mcnemar_test(flags5, flags6)
                results[f"r{rate}_mcnemar_{fpr_name}"] = mn
                if mn["a_only"] + mn["b_only"] > 0:
                    print(f"    McNemar M5 vs M6 at {fpr_label} FPR: "
                          f"OR={mn['odds_ratio']:.2f}, p={mn['p_value']:.2e}, "
                          f"M5-only={mn['a_only']}, M6-only={mn['b_only']}")

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    # Compute key comparisons
    m1_med = np.median(honest_dbs["M1_t1_only"])
    m5_med = np.median(honest_dbs["M5_permove"])
    m6_med = np.median(honest_dbs["M6_pb_logistic"])

    print(f"\n  Honest median dB:  M1={m1_med:+.1f}  M6={m6_med:+.1f}  M5={m5_med:+.1f}")
    print()
    print("  If M6 >> M1: logistic p_m is better than band-average p")
    print("  If M5 >> M6: log-LR scoring adds value beyond better p_m")
    print("  If M6 ~ M5:  advantage comes entirely from better p_m")

    # Save results
    out_path = ROOT / "cache" / "ablation_pb_logistic.json"
    out_data = {
        "honest_medians": {m: float(np.median(honest_dbs[m])) for m in METHODS},
        "honest_p99": {m: float(np.percentile(honest_dbs[m], 99)) for m in METHODS},
        "thresholds": thresholds,
        "results": {k: v for k, v in results.items()
                    if not k.endswith("mcnemar_fpr_01pct")},
    }
    # Serialize McNemar results separately (they have non-serializable types)
    mcnemar_results = {}
    for k, v in results.items():
        if "mcnemar" in k and isinstance(v, dict):
            mcnemar_results[k] = {kk: float(vv) if isinstance(vv, (int, float, np.integer, np.floating)) else vv
                                  for kk, vv in v.items()}
    out_data["mcnemar"] = mcnemar_results

    out_path.write_text(json.dumps(out_data, indent=2, default=str))
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    main()
