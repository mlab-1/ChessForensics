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

A statistical framework that analyses chess games through fifteen independent signals, combines them via correlation-adjusted log-likelihood ratios, and produces a calibrated posterior probability of engine assistance. Built on the research of Prof. Kenneth Regan (University at Buffalo), extended with empirical calibration from a 3.4-million-game corpus (Lichess + TWIC).

**Live at [chessforensics.org](https://chessforensics.org)**

## How it works

1. **Engine analysis** -- Multi-pass Stockfish (depth 4/12/22, MultiPV 5) evaluates every position
2. **Signal extraction** -- Fifteen signals: three match-rate z-scores, ACPL, two accuracy-vs-complexity slopes, CPL tail test, switchpoint detection, IPR gap, four timing signals, and two engine-fingerprint signals
3. **Null calibration** -- Each signal is corrected for rating-dependent bias via a three-tier scheme (per-band, pooled, empirical fallback) fitted on the corpus
4. **Bayesian combination** -- Correlation-adjusted multivariate log-likelihood ratio with a four-rate mixture model (10%, 25%, 50%, 100% assistance) and NNLS-projected non-negative weights
5. **Verdict** -- Posterior probability with Monte Carlo sensitivity band. Multi-game scans use per-player aggregation that charges the mixture penalty once

## Validation

| Category | Sample | Result |
|---|---|---|
| Held-out FPR @ 50% | 682,952 games | **0%** |
| Held-out FPR @ 5% | 682,952 games | **0.97%** |
| OOS titled players @ 50% | 5,560 games (92 GM/IM/WGM) | **0%** |
| FIDE cheating cases | 10 sanctioned cases | **10/10 positive** (+4.7 to +55.6 dB) |
| Strong detections (>+10 dB) | 10 cases | **8/10** |
| Fischer 1972, Kasparov 1999 | single games | **1.70%, 0.28%** |
| Carlsen-Niemann 2022 | single game | **0.82%, 1.07%** |
| Full assist detection | synthetic | **2-4 games to 50% posterior** |

Training and evaluation data are strictly separated. All cheater PGNs come from independent sources (FIDE tournament bulletins) with no overlap with the calibration corpus.

## Features

- **Single game analysis** -- paste PGN, upload a file, or fetch from Chess.com / Lichess / URL
- **Player scan** -- analyse multiple games across a player's history and aggregate the evidence
- **Game import** -- fetch directly from Chess.com or Lichess by username, or from any PGN URL
- **Case studies** -- six documented cases with interactive board replay (Ivanov, Shevchenko, Rausis, Feller, Nigalidze, Niemann)
- **Sensitivity band** -- Monte Carlo uncertainty quantification (4,000 draws)
- **Per-game flags** -- individual suspicious games highlighted within multi-game scans

## Architecture

```
server.py          FastAPI backend (async job queue, SSE progress, worker threads)
web/               Vanilla JS + CSS frontend (dark cosmic theme, Plotly charts, cm-chessboard)
chessforensics/
  engine.py        EnginePool (N Stockfish instances), 3-pass analysis, SQLite cache
  features.py      Per-move DataFrame: match rates, CPL, complexity, timing, book detection
  model.py         Multinomial choice model P(i) ~ exp(-(delta/s)^c), MLE, Poisson-binomial
  score.py         Bayesian evidence: correlation matrix, NNLS projection, mixture model
  analyse.py       Orchestration: game -> 15 signals -> calibrated posterior
  scan.py          Multi-game per-player aggregation, convergence, changepoint detection
  baselines.py     Rating-band model loader (12 bands, 200 Elo wide)
  ingest.py        PGN parsing, Chess.com API, Lichess API (ndjson), clock/eval parsing
baselines/
  model_v1.json    Fitted model (3.4M games, 229.6M positions, 12 bands)
paper/
  paper.pdf        Full academic paper (October 2026)
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

## The corpus

3,400,451 rated rapid and classical games (6.78M player-games, 229.6M positions) drawn from two sources: the Lichess open database (bulk of games below Elo 2200) and TWIC (The Week in Chess) OTB tournament archives (majority of games in the 2200+ bands). Collection filters:

- Minimum 20 plies per game
- Maximum 300 Elo gap between opponents
- Rated games only; no correspondence
- Accounts flagged for ToS violations excluded (0.73% contamination rate verified via API audit)

Twelve rating bands from <800 through 2800+, each with its own fitted choice model and null calibration parameters.

## Signals

| Signal | Family | Test | Effect size |
|---|---|---|---|
| t1_match | accuracy | Poisson-binomial | 3.3 |
| t2_match | accuracy | Poisson-binomial | 2.2 |
| t3_match | accuracy | Poisson-binomial | 2.1 |
| acpl | accuracy | mean test | 1.9 |
| cpl_tail | shape | tail rate test | 2.2 |
| slope | shape | logistic fit | 0.3 |
| cpl_slope | shape | linear fit | 0.0 |
| ipr_gap | strategic | IPR - Elo | 2.0 |
| switchpoint | strategic | permutation LRT | 2.0 |
| time_uniformity | timing | residual CV | 1.3 |
| time_complexity | timing | Spearman rho | 0.7 |
| time_cadence | timing | modal cluster | 0.5 |
| timing x accuracy | timing | log z-test | 0.0 |
| deep_only | engine | match rate | 2.4 |
| engine_fp | engine | match rate | 2.5 |

Effect sizes measured from 55 cheater player-games (87 Lichess-banned accounts, posterior >5%). Signals with fewer than 5 observations retain prior estimates.

## Paper

The full paper is at [`paper/paper.pdf`](paper/paper.pdf):

> **ChessForensics: Bayesian Detection of Engine Assistance in Competitive Chess**
> Alberto Mirandola, October 2026

## License

The analysis pipeline is original work. Stockfish is licensed under GPL v3.
