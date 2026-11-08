---
title: Chess Forensics
emoji: "&#9812;"
colorFrom: indigo
colorTo: cyan
sdk: docker
app_port: 10000
---

# Chess Forensics

**Bayesian detection of engine assistance in competitive chess.**

A statistical framework that computes a per-move log-likelihood ratio for every position in a game, integrates the evidence over a four-rate mixture model, and produces a calibrated posterior probability of engine assistance. Built on the foundational work of Prof. Kenneth Regan (University at Buffalo), extended with per-move Bernoulli scoring and position-specific probability calibration.

**Live at [chessforensics.org](https://chessforensics.org)**

## How it works

1. **Engine analysis** -- Three-pass Stockfish analysis (depth 4/MultiPV 3, depth 12/MultiPV 5, depth 22/MultiPV 5). An adaptive priority ranker allocates ~40% of the deep-analysis budget to critical positions.
2. **Position filtering** -- Book positions, forced moves, and positions with evaluations beyond +/-6 pawns are excluded, retaining only positions where the choice model has discriminative power. Minimum 8 scoreable positions per game.
3. **Per-move scoring (M5)** -- For each scoreable position, a per-band logistic regression predicts the probability p_m that an honest player of the appropriate rating would match the engine's top choice, given six position features (evaluation gap, legal move count, complexity, spread, gap-small indicator, only-move indicator). Each move is scored as a Bernoulli trial, and a binary log-likelihood ratio is computed: l_m(r) = log(r + (1-r)*p_m) - log(p_m) if top-1 match, else log(1-r). A separate logistic regression is fitted for each of 12 rating bands on the 2.7M-game training corpus (AUC ~ 0.75).
4. **Four-rate mixture model** -- Per-move log-LRs are summed across all scoreable positions at each of four assistance rates r in {10%, 25%, 50%, 100%} with a uniform prior, then integrated via logsumexp to produce a single marginal log-LR. This enables detection of partial or intermittent engine use without requiring the assistance rate to be specified in advance.
5. **Per-band null calibration** -- The raw dB score is z-normalised against each rating band's honest distribution, ensuring that a single global threshold achieves uniform false-positive rates across all rating levels. A three-tier calibration scheme (per-band, pooled, empirical defaults) ensures graceful degradation when band-specific data is sparse.
6. **Posterior and sensitivity band** -- The calibrated log-LR is converted to a posterior probability (1% base-rate prior). A Monte Carlo procedure (4,000 draws) perturbs the per-move p_m estimates and recomputes the posterior, yielding a 5th--95th percentile sensitivity band.
7. **Multi-game aggregation** -- For player scans, per-rate log-LRs are summed across games and the mixture penalty is applied once per player, dramatically improving detection of partial assistance spread across multiple games.

## Corpus

The reference corpus comprises 3,400,451 rated rapid and classical games (6.78 million player-games, 229.6 million positions) drawn from the Lichess open database and TWIC OTB tournament archives. A deep engine pass (Stockfish 17, depth 22, MultiPV 5) was applied to 41,698 games (70,906 player-games, 3.36 million positions). The corpus is partitioned into a training set and a held-out test set of 682,952 games, with strict train/test independence.

## Validation

Validation is conducted along five axes: held-out false-positive evaluation, out-of-sample Chess.com validation, titled-player games, super-GM stress tests, and sanctioned FIDE cheating cases.

| Category | Result |
|---|---|
| Held-out FPR at 50% posterior | **0%** (682,952 games) |
| Held-out FPR at 5% posterior | **0.97%** |
| Out-of-sample V1 (threshold selection) | **28,670 games** from 15,586 players (Chess.com) |
| Out-of-sample V2 (held-out test) | **22,932 games** from 7,505 players (zero overlap with V1) |
| V2 FPR at 1% target (2000--2200) | **0.47%** [95% CI: 0.38--0.58%] |
| V2 FPR at 1% target (2200--2400) | **0.59%** [95% CI: 0.35--0.94%] |
| Titled-player games (Lichess) | **0% FPR** at 50% threshold (5,560 games, 92 titled players: 64 GM, 24 IM, 4 WGM) |
| FIDE sanctioned cheating cases | **10/10 positive** cumulative evidence (+4.7 to +55.6 dB) |
| Strong detections (>+10 dB) | **7/10**; 4/10 exceed +20 dB |
| Kasparov--Topalov 1999 | **correctly cleared** (-4.9 dB); two-signal baseline would falsely flag (+8.4 dB) |
| Fischer--Spassky 1972 Game 6 | **correctly cleared** (+3.5 dB, below any reasonable threshold) |

### Baseline comparison (M5 vs four baselines)

At the operationally critical regime (r = 25%, N = 4 games, 1% FPR), M5 detects **87%** of cheaters vs **64%** for the two-signal Regan-style baseline (M2). At 0.1% FPR the gap widens: **83% vs 58%**. Single-game detection at r = 50%: M5 achieves **57.5% TPR** at 0.1% FPR vs 41.0% for M2 (McNemar p < 10^-137). All comparisons use the same positions, engine output, and calibration data.

### Sanctioned cheating cases

| Player | Sanction | Games | +Games | Sum dB |
|---|---|---|---|---|
| Borislav Ivanov | Bulgarian CF ban, FM stripped | 9 | 6 | +55.6 |
| Marc Morgunov | FIDE 24-month ban | 7 | 6 | +50.5 |
| Joeri Harreman | FIDE 36-month ban | 4 | 3 | +33.4 |
| Kirill Shevchenko | FIDE 3-year ban, GM stripped | 2 | 2 | +21.5 |
| Sebastien Feller | FIDE 2y9m suspension | 9 | 7 | +14.5 |
| Igors Rausis | FIDE 6-year ban, GM stripped | 6 | 4 | +12.4 |
| Siddharth Banik | FIDE 30-month ban | 6 | 4 | +10.3 |
| Pang Khai Jie Ryan | FIDE 2-year ban | 8 | 5 | +9.7 |
| Patrycja Waszczuk | Polish CF + FIDE 2-year ban | 4 | 2 | +5.0 |
| Matija Ivic | FIDE 24-month ban | 10 | 4 | +4.7 |

Training and evaluation data are strictly separated. All cheater PGNs come from independent sources (TWIC archives, FIDE tournament bulletins) with no overlap with the calibration corpus. Out-of-sample validation uses two non-overlapping Chess.com datasets with zero player or platform overlap with the training data.

## Features

- **Single game analysis** -- paste PGN, upload a file, or fetch from Chess.com / Lichess / URL
- **Player scan** -- analyse multiple games across a player's history and aggregate the evidence
- **Game import** -- fetch directly from Chess.com or Lichess by username, or from any PGN URL
- **Case studies** -- documented cases with interactive board replay (Ivanov, Shevchenko, Rausis, Feller, Nigalidze, Niemann)
- **Sensitivity band** -- Monte Carlo uncertainty quantification (4,000 draws)
- **Per-game flags** -- individual suspicious games highlighted within multi-game scans
- **Fifteen supplementary signals** -- match rates, ACPL, accuracy-complexity slopes, CPL tail, timing signals, switchpoint detection, IPR gap, engine fingerprinting

## Architecture

```
server.py          FastAPI backend (async job queue, SSE progress, worker threads)
web/               Vanilla JS + CSS frontend (dark cosmic theme, Plotly charts, cm-chessboard)
chessforensics/
  engine.py        EnginePool (N Stockfish instances), 3-pass analysis, SQLite cache
  features.py      Per-move DataFrame: match rates, CPL, complexity, timing, book detection
  model.py         Multinomial choice model P(i) ~ exp(-(delta/s)^c), MLE, Poisson-binomial
  score.py         Bayesian evidence: per-move log-LR, logistic regression p_m, mixture model
  analyse.py       Orchestration: game -> per-move scoring -> calibrated posterior
  scan.py          Multi-game per-player aggregation, convergence, changepoint detection
  baselines.py     Rating-band model loader (12 bands, 200 Elo wide)
  ingest.py        PGN parsing, Chess.com API, Lichess API (ndjson), clock/eval parsing
baselines/
  model_v1.json    Fitted model (12 rating bands)
paper/
  paper.pdf        Full academic paper (November 2026)
```

## Running locally

```bash
pip install -r requirements.txt
python setup_engine.py          # downloads Stockfish 18
python server.py                # http://127.0.0.1:8000
```

Or with Docker:

```bash
docker build -t chessforensics .
docker run -p 8000:10000 chessforensics
```

## Paper

The full paper is at [`paper/paper.pdf`](paper/paper.pdf):

> **ChessForensics: Bayesian Detection of Engine Assistance in Competitive Chess**
> Alberto Mirandola, November 2026

## License

The analysis pipeline is original work. Stockfish is licensed under GPL v3.
