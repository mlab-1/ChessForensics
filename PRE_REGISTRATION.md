# Pre-Registration: M5 vs Regan-Style Cheat Detection Comparison

**Date:** 2026-11-03
**Authors:** Alberto (ChessForensics)
**Status:** LOCKED — do not modify after first commit

---

## 1. Definition of "Better"

A method A is **better** than method B if and only if:

> At the **same false-positive rate**, A detects more cheaters (higher TPR).

Equivalently: A's ROC curve dominates B's in the low-FPR region that matters
for deployment (FPR < 1%).

We do NOT define "better" as higher AUC across the full ROC — we only care
about the left edge where FPR is small enough for real-world use.

### Primary metric (single number)

**TPR at FPR = 0.1%** (1 in 1,000 games).

This is the threshold we would actually deploy at.  A clean player with 10
games has < 1% chance of a single game crossing it, and multi-game aggregation
drives the per-player false alarm rate far below that.

### Secondary metrics

| Metric | Definition |
|--------|-----------|
| TPR at FPR = 1% | Power at a less conservative threshold |
| TPR at FPR = 0.01% | Power at a very conservative threshold (1 in 10,000) |
| Partial AUC (FPR 0-1%) | Area under ROC in the only region that matters |
| Per-player FPR at 50% posterior | After multi-game aggregation, fraction of clean players flagged |
| Games-to-detect at FPR = 0.1% | Median N games needed to reach 80% TPR per-player, by assistance rate |

### What we will NOT use as primary evidence

- Full AUC (dominated by the high-FPR region nobody uses)
- Accuracy at 50% threshold (meaningless for rare-event detection)
- Any metric computed on training data

---

## 2. Methods Under Comparison

All methods receive **identical inputs**: the same games, same Stockfish
version and depth, same position filters (forced/decided/book excluded), same
feature frame.

### M2: Regan-Style Baseline

Reimplementation of the IPR/z-score approach from Regan & Haworth (2011) and
Regan (2014):

1. For each position, compute P(match top-1) from a fitted choice model
   (function of eval gap, legal move count).
2. Count observed top-1 matches, compute expected count and variance under
   the binomial model.
3. z = (observed - expected) / sqrt(variance).
4. Optionally add ACPL z-score (analogous construction).
5. Multi-game: Stouffer's method — sum(z) / sqrt(N).

**What this is NOT:** We do not have Regan's proprietary software or his
exact fitted parameters. We reimplement from his published papers. This is a
faithful reimplementation of his *published method*, not a strawman. We use
the same depth-12 MultiPV-5 Stockfish analysis he recommends.

**What we acknowledge:** Regan's actual deployed system may have unpublished
improvements. Our comparison is against his *published scientific method*, not
his operational tool.

### M3: Regan-Style + Per-Band Calibration

Same as M2 but with per-band null calibration (subtract band mean, divide by
band std). This isolates the contribution of calibration.

### M5: Per-Move Log-Likelihood Ratio (our method)

1. For each position, compute P(match top-1 | honest) = p1 via a logistic
   regression model fitted per rating band.  Features: log(gap), log(legal),
   complexity, spread, gap_small, only_move.
2. For each assistance rate r in {0.10, 0.25, 0.50, 1.00}:
   - If move matched top-1: log_lr = log(r + (1-r)*p1) - log(p1)
   - If not: log_lr = log(1-r) (clamped at log(1e-10) for r=1)
3. Sum log_lr across moves within a game, for each rate.
4. Mixture: logsumexp across rates with uniform prior.
5. Multi-game: sum per-rate log-LRs across games, then logsumexp once.

### M1: Raw Top-1 Match Rate (sanity baseline)

Just the z-score of the raw top-1 match rate against band average, with no
choice model. Included to show the floor — any real method must beat this.

---

## 3. Data

### 3a. Clean players (null distribution / FPR estimation)

**Source:** Chess.com titled players + rapid-rated >= 2100 accounts.

**Size:** ~16,000 players, 2 games each, ~28,000 games.

**Justification for "clean":** These are FIDE-titled or high-rated players
on a platform with active cheat detection. Some may still be cheaters — this
is a *conservative* estimate of FPR (true FPR is lower).

**OOS requirement:** The p1 logistic model was fitted on a separate training
corpus (Lichess classical games). The Chess.com OOS data was NEVER used for
model fitting.

### 3b. Synthetic cheaters (TPR estimation)

**Source:** Clean games from the training corpus held-out split (by player).

**Injection:** For each game, replace a fraction r of the player's moves with
Stockfish top-1 move. Rates: r in {0.10, 0.25, 0.50, 1.00}.

**Size:** 5,000 games per rate = 20,000 synthetic cheater games.

**Why synthetic:** Real cheaters don't come with ground truth for *which*
moves were assisted or at *what rate*. Synthetic injection gives us the true
r, enabling power curves by rate.

### 3c. Confirmed real cases (external validation, NOT used for thresholds)

Cases where cheating was established by evidence independent of any
statistical detection method:

| Case | Evidence type | Games |
|------|--------------|-------|
| Sebastien Feller (2010 Olympiad) | Co-conspirator testimony (Marzolo) identifying 8 of 9 games | 9 |
| Gaioz Nigalidze (2015 Dubai) | Phone found in toilet stall during game | 6 |
| Igors Rausis (2019) | Caught with phone by arbiter, photo evidence | TBD |
| Lichess ToS-banned accounts | Platform ban (independent of our method) | 50+ |

**These cases are used ONLY for illustration** (where do they fall in the
score distribution?). They are NOT used to set thresholds or compute TPR.
A method that flags Feller but misses low-rate cheaters at acceptable FPR
has proven nothing.

---

## 4. Experimental Protocol

### Step 1: Threshold setting (on clean data only)

For each method, compute scores on all clean OOS games. Set thresholds at
FPR = {0.01%, 0.1%, 1%} from the empirical distribution.

### Step 2: Power estimation (on synthetic cheaters only)

Apply thresholds from Step 1 to synthetic cheater games. Report TPR at each
FPR level, broken down by assistance rate.

### Step 3: Paired comparison

On the **same** synthetic cheater games, for each game record whether M5
flags it and whether M2 flags it (at the same FPR threshold). Report:

- McNemar's test for paired proportions
- Bootstrap 95% CI on the difference in TPR (resampled by player, not game)
- Bootstrap 95% CI on the difference in partial AUC (FPR 0-1%)

### Step 4: Multi-game aggregation

For each clean player, aggregate across their 2 games. For synthetic players,
aggregate across N = {1, 2, 4, 8} games. Report:

- Per-player FPR at 50% posterior threshold
- Games-to-detect curves: median N to reach 80% per-player TPR

### Step 5: Real case illustration

Score each confirmed case with all methods. Report where each falls in the
clean distribution (percentile rank). This is descriptive, not inferential.

### Step 6: Stress tests

See Section 6.

---

## 5. Statistical Tests

### Paired test for single-game TPR

McNemar's test on the 2x2 table of (M5 flags, M2 flags) at each FPR level.
Report exact p-value and odds ratio with 95% CI.

### Bootstrap for confidence intervals

- Resample **by player** (not by game) to respect within-player correlation.
- 10,000 bootstrap replicates.
- Report 95% percentile CIs.

### Multiple comparisons

We compare 4 methods. Apply Bonferroni correction to the 6 pairwise tests,
or report unadjusted p-values with a note.

---

## 6. Stress Tests and Failure Modes

### 6a. Opening preparation

**Test:** Take 50 games with deep home preparation (>= 15 book moves). Score
with all methods. A good method should NOT flag these — the book moves are
excluded before scoring, so preparation should be invisible.

**Report:** Max score among these 50 games, and the percentile in the clean
distribution.

### 6b. Short games

**Test:** Score games with < 15 scoreable moves separately. Report TPR and
FPR for this subset. Short games have less signal — power should drop, but
FPR must not increase.

### 6c. Forced sequences and tablebase positions

**Test:** These are excluded by the `is_forced` and `is_decided` filters.
Verify that the filters work: count how many forced/decided positions remain
in the scored set. Should be zero.

### 6d. Engine version drift

**Test:** If feasible, re-analyse a subset (100 games) with Stockfish 16 vs
17. Compare the score distributions. Report the rank correlation between
scores from the two versions.

**Regan's advantage:** He has dealt with version drift for 15+ years and
recalibrates periodically. We acknowledge this and report sensitivity.

### 6e. Rating band edge effects

**Test:** Plot FPR by rating band (200-point buckets). A well-calibrated
method should have roughly uniform FPR across bands. If 2600+ players have
systematically higher scores (because the p1 model is poorly calibrated at
the top), this must be flagged.

### 6f. Time control sensitivity

**Test:** Report TPR and FPR separately for rapid vs classical. The p1 model
was fitted on classical; if rapid games have different characteristics, this
should be visible.

---

## 7. What Would Falsify Our Claim?

We commit in advance: if any of the following occur, **M5 is NOT better**
and we will say so in the paper:

1. M2 (Regan-style) has higher TPR than M5 at FPR = 0.1% for r >= 0.25,
   with the difference surviving the bootstrap CI.

2. M5's FPR exceeds the nominal level by more than 2x in any rating band
   (e.g., nominal 0.1% but observed 0.3% in the 2600+ band).

3. M5 flags any confirmed clean GM (top-100 player, no cheating allegations)
   above the 99.9th percentile of the clean distribution.

4. The stress tests in Section 6 reveal a systematic failure mode (e.g.,
   all deep-prep games score above the 95th percentile).

---

## 8. Reporting Commitments

1. All code, data generation scripts, and analysis notebooks will be public.
2. We will report ALL metrics listed above, not cherry-pick favorable ones.
3. If M5 wins on single-game power but loses on multi-game aggregation (or
   vice versa), we report both honestly.
4. If the improvement is statistically significant but practically small
   (e.g., 1% TPR difference), we will say so.
5. We will not claim "better than Regan" without specifying *which* metric,
   *at which FPR*, *at which assistance rate*, with confidence intervals.

---

## 9. Execution Checklist

- [ ] Commit this document to git (immutable timestamp)
- [ ] Finish OOS validation run (18k players, 2100+ only)
- [ ] Compute clean null distributions for all 4 methods
- [ ] Run synthetic cheater experiments with M5 (already done for M1-M4)
- [ ] Compute all primary and secondary metrics with bootstrap CIs
- [ ] Run paired McNemar tests
- [ ] Run all 6 stress tests
- [ ] Score confirmed cases (Feller, Nigalidze, Rausis, Lichess bans)
- [ ] Generate comparison plots (ROC, power-vs-rate, games-to-detect)
- [ ] Write up results section
- [ ] If any falsification criterion is triggered, update claims accordingly
