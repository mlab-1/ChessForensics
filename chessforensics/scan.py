"""Multi-game player scan: aggregate evidence across many games.

A single game has ~30 scoreable positions — the posterior is inherently wide.
Analysing 10-30 games gives 300-900 positions, and the evidence accumulates
because games are approximately independent draws from the same player
behaviour.

Aggregation uses a per-player mixture over assistance rates: each game produces
a log-likelihood-ratio at each rate, and the rates are summed across games
before marginalising.  This assumes one constant rate per player (not a rate
that changes from game to game), and charges the mixture penalty (log K) once,
recovering evidence that a per-game mixture would lose.

The convergence curve (posterior after 1 game, 2 games, ..., N games) is the
primary output for the UI: watching the posterior converge (or not) is both
visually compelling and statistically honest.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.special import logsumexp

from . import score
from .analyse import Report, analyse
from .baselines import Baseline, load_baseline
from .engine import AnalysisCache, EnginePool
from .ingest import Game

ScanProgressFn = Callable[[int, int, int, int, str], None]
# (game_done, game_total, move_done, move_total, label)


@dataclass
class GameResult:
    """Per-game summary within a scan."""

    game_index: int
    game_info: dict
    color: str  # which side is the target player
    log_lr: float
    decibans: float
    posterior_cumulative: float  # posterior after this game + all prior
    individual_posterior: float  # this game alone against the prior
    n_scoreable: int
    t1_rate: float | None
    acpl: float | None
    slope_beta1: float | None
    verdict_interpretation: str
    insufficient: bool = False
    speed: str | None = None
    ipr_elo: float | None = None
    switchpoint_detected: bool = False
    raw_pgn: str = ""
    log_lrs_by_rate: list[float] = field(default_factory=list)  # per-rate for mixture aggregation


@dataclass
class TemporalChangepoint:
    """Result of inter-game changepoint detection."""

    detected: bool = False
    split_game: int | None = None
    db_before: float = 0.0
    db_after: float = 0.0
    p_value: float = 1.0
    detail: str = ""


@dataclass
class SpeedBreakdown:
    """Evidence split by time control."""

    speed: str
    n_games: int
    total_db: float
    mean_t1: float | None
    mean_acpl: float | None


@dataclass
class ScanResult:
    """Aggregated result across all games."""

    username: str
    source: str  # "chesscom" | "lichess"
    n_games: int
    n_games_scored: int  # games with enough data
    total_positions: int

    # Aggregated evidence
    total_log_lr: float
    total_decibans: float
    posterior: float
    posterior_low: float
    posterior_high: float
    prior: float

    interpretation: str

    # Per-game breakdown
    games: list[GameResult] = field(default_factory=list)

    # Convergence curve: posterior after each game
    convergence: list[dict] = field(default_factory=list)

    # Per-signal aggregated z-scores
    signal_summary: list[dict] = field(default_factory=list)

    # Temporal changepoint: did evidence concentrate in a time window?
    temporal: TemporalChangepoint | None = None

    # Per-game individual flags: games suspicious on their own
    flagged_games: list[int] = field(default_factory=list)
    peak_game_db: float = 0.0
    peak_game_posterior: float = 0.0

    # Variance analysis: is the evidence uniformly distributed or spiked?
    evidence_variance: float | None = None
    outlier_games: list[int] = field(default_factory=list)

    # Per time-control breakdown
    speed_breakdown: list[dict] = field(default_factory=list)

    # IPR trajectory
    ipr_trajectory: list[dict] = field(default_factory=list)

    caveats: list[str] = field(default_factory=list)


def _find_player_color(game: Game, username: str) -> str | None:
    """Determine which color the target player is in this game."""
    u = username.lower()
    if game.players["white"].name.lower() == u:
        return "white"
    if game.players["black"].name.lower() == u:
        return "black"
    return None


def scan_player(
    games: list[Game],
    username: str,
    prior: float = score.DEFAULT_PRIOR,
    pool_size: int = 10,
    deep_pass: bool = False,
    progress: ScanProgressFn | None = None,
    cancel_fn: Callable[[], bool] | None = None,
    run_l0: bool = True,
) -> ScanResult:
    """Analyse multiple games for one player and aggregate evidence.

    ``deep_pass`` defaults to False for scans because the per-game precision
    matters less when you have many games, and skipping the deep pass roughly
    halves the analysis time per game.
    """
    baseline = load_baseline()
    cache = AnalysisCache()
    pool = EnginePool(size=pool_size)

    game_results: list[GameResult] = []
    cumulative_log_lr = 0.0
    total_positions = 0
    per_signal_z: dict[str, list[float]] = {}  # signal_name -> list of z per game

    try:
        for gi, game in enumerate(games):
            if cancel_fn and cancel_fn():
                break

            color = _find_player_color(game, username)
            if color is None:
                # Player not found in this game — skip
                continue

            # Progress callback bridging
            def on_move_progress(done: int, total: int, label: str) -> None:
                if progress:
                    progress(gi, len(games), done, total, label)

            if progress:
                progress(gi, len(games), 0, 0, f"analysing game {gi + 1}/{len(games)}")

            report = analyse(
                game,
                pool=pool,
                cache=cache,
                baseline=baseline,
                prior=prior,
                progress=on_move_progress,
                deep_pass=deep_pass,
                pool_size=pool_size,
                run_l0=run_l0,
            )

            player = report.players[color]

            n_rates = len(score.MIXTURE_RATES)
            zero_rates = [0.0] * n_rates

            if player.insufficient:
                game_results.append(
                    GameResult(
                        game_index=gi,
                        game_info=report.game_info,
                        color=color,
                        log_lr=0.0,
                        decibans=0.0,
                        posterior_cumulative=score._sigmoid(
                            score._logit(prior) + cumulative_log_lr
                        ),
                        individual_posterior=prior,
                        n_scoreable=0,
                        t1_rate=None,
                        acpl=None,
                        slope_beta1=None,
                        verdict_interpretation="insufficient data",
                        insufficient=True,
                        raw_pgn=game.raw_pgn or "",
                        log_lrs_by_rate=zero_rates,
                    )
                )
                continue

            v = player.verdict or {}
            game_log_lr = v.get("log_lr", 0.0)
            game_db = v.get("decibans", 0.0)
            n_sc = (player.summary or {}).get("n_scoreable", 0)
            game_rate_lls = v.get("log_lrs_by_rate", zero_rates)

            cumulative_log_lr += game_log_lr
            total_positions += n_sc

            cumulative_posterior = score._sigmoid(
                score._logit(prior) + cumulative_log_lr
            )

            # Collect per-signal z-scores for aggregation
            for row in v.get("rows", []):
                name = row.get("name", "")
                z = row.get("z")
                if name and z is not None and math.isfinite(z):
                    per_signal_z.setdefault(name, []).append(z)

            # Per-game individual posterior (this game alone)
            individual_post = score._sigmoid(
                score._logit(prior) + game_log_lr
            )

            game_results.append(
                GameResult(
                    game_index=gi,
                    game_info=report.game_info,
                    color=color,
                    log_lr=game_log_lr,
                    decibans=game_db,
                    posterior_cumulative=cumulative_posterior,
                    individual_posterior=individual_post,
                    n_scoreable=n_sc,
                    t1_rate=(player.summary or {}).get("t1"),
                    acpl=(player.summary or {}).get("acpl"),
                    slope_beta1=(player.slope_fit or {}).get("beta1"),
                    verdict_interpretation=v.get("interpretation", ""),
                    speed=report.game_info.get("speed"),
                    ipr_elo=(player.ipr or {}).get("elo"),
                    switchpoint_detected=bool(
                        (player.switchpoint or {}).get("detected")
                    ),
                    raw_pgn=game.raw_pgn or "",
                    log_lrs_by_rate=game_rate_lls,
                )
            )
    finally:
        pool.close()

    # --- Aggregate with per-player mixture ---
    # One rate per player across all games: logsumexp_r(log pi_r + sum_g log Lambda_g(r))
    # This charges the mixture penalty (log K) once, not per game.
    scored = [g for g in game_results if not g.insufficient]
    n_scored = len(scored)
    n_rates = len(score.MIXTURE_RATES)

    if scored:
        # Sum per-rate log-LRs across games
        rate_sums = np.zeros(n_rates)
        for g in scored:
            if g.log_lrs_by_rate and len(g.log_lrs_by_rate) == n_rates:
                rate_sums += np.array(g.log_lrs_by_rate)
        # Per-player marginal: logsumexp(log_prior_r + sum_g log_lr_g(r))
        total_log_lr = float(logsumexp(rate_sums + score.MIXTURE_LOG_PRIOR))
    else:
        rate_sums = np.zeros(n_rates)
        total_log_lr = 0.0

    total_db = 10.0 * total_log_lr / math.log(10) if total_log_lr else 0.0
    final_posterior = score._sigmoid(score._logit(prior) + total_log_lr)

    # Recompute convergence curve with per-player mixture
    running_rate_sums = np.zeros(n_rates)
    for g in game_results:
        if not g.insufficient and g.log_lrs_by_rate and len(g.log_lrs_by_rate) == n_rates:
            running_rate_sums += np.array(g.log_lrs_by_rate)
        running_ll = float(logsumexp(running_rate_sums + score.MIXTURE_LOG_PRIOR))
        g.posterior_cumulative = score._sigmoid(score._logit(prior) + running_ll)

    # Sensitivity band via Monte Carlo on per-game per-rate log_lr
    rng = np.random.default_rng(42)
    n_mc = 5000
    draws = np.empty(n_mc)
    if scored:
        # Build matrix [n_games x n_rates]
        rate_matrix = np.array([
            g.log_lrs_by_rate if len(g.log_lrs_by_rate) == n_rates
            else [0.0] * n_rates
            for g in scored
        ])
        for i in range(n_mc):
            jittered = rate_matrix.copy()
            for j, g in enumerate(scored):
                n_j = max(g.n_scoreable, 10)
                noise = rng.normal(0, 0.3 / math.sqrt(n_j), size=n_rates)
                jittered[j] += noise + rng.normal(0, np.abs(rate_matrix[j]) * 0.15)
            sums = jittered.sum(axis=0)
            draws[i] = score._sigmoid(
                score._logit(prior) + float(logsumexp(sums + score.MIXTURE_LOG_PRIOR))
            )
    else:
        draws[:] = prior

    lo, hi = float(np.percentile(draws, 5.0)), float(np.percentile(draws, 95.0))

    # Convergence curve
    convergence = []
    running_rate_sums2 = np.zeros(n_rates)
    for g in game_results:
        if not g.insufficient and g.log_lrs_by_rate and len(g.log_lrs_by_rate) == n_rates:
            running_rate_sums2 += np.array(g.log_lrs_by_rate)
        cum_ll = float(logsumexp(running_rate_sums2 + score.MIXTURE_LOG_PRIOR))
        convergence.append({
            "game": g.game_index + 1,
            "posterior": g.posterior_cumulative,
            "decibans_cumulative": 10.0 * cum_ll / math.log(10),
            "label": f"{g.game_info.get('white', '?')} vs {g.game_info.get('black', '?')}",
            "insufficient": g.insufficient,
        })

    # Per-signal summary
    signal_summary = []
    for name, zs in per_signal_z.items():
        if not zs:
            continue
        arr = np.array(zs)
        signal_summary.append({
            "name": name,
            "mean_z": float(arr.mean()),
            "std_z": float(arr.std()) if len(arr) > 1 else 0.0,
            "n_games": len(arr),
            "combined_z": float(arr.sum() / math.sqrt(len(arr))),
        })
    signal_summary.sort(key=lambda s: -abs(s["combined_z"]))

    # --- Per-game individual flags ---
    # Games that are individually suspicious (posterior > 50% on their own)
    FLAG_THRESHOLD = 0.50
    flagged_games = [
        g.game_index for g in scored
        if g.individual_posterior > FLAG_THRESHOLD
    ]
    peak_game_db = max((g.decibans for g in scored), default=0.0)
    peak_game_posterior = max(
        (g.individual_posterior for g in scored), default=prior
    )

    # Interpretation
    if total_db <= -10:
        interp = "Strong evidence against engine assistance across these games."
    elif total_db <= -3:
        interp = "Evidence against engine assistance."
    elif total_db < 3:
        interp = "No meaningful evidence either way."
    elif total_db < 10:
        interp = "Weak evidence of engine assistance."
    elif total_db < 20:
        interp = "Moderate evidence of engine assistance."
    elif total_db < 35:
        interp = "Strong evidence of engine assistance."
    else:
        interp = "Very strong evidence of engine assistance."

    # Amend interpretation when individual games are flagged
    if flagged_games and total_db < 10:
        game_nums = ", ".join(str(i + 1) for i in flagged_games)
        interp += (
            f" However, {len(flagged_games)} individual game(s) "
            f"({game_nums}) exceed the suspicion threshold on their own."
        )

    caveats = []
    if flagged_games:
        game_nums = ", ".join(str(i + 1) for i in flagged_games)
        caveats.append(
            f"{len(flagged_games)} game(s) ({game_nums}) individually exceed "
            f"50% posterior — these warrant single-game review regardless of "
            f"the aggregate verdict."
        )
    if n_scored < 5:
        caveats.append(
            f"Only {n_scored} games with enough data. Scan at least 10-20 games "
            f"for a meaningful aggregate verdict."
        )
    if not baseline.trustworthy:
        caveats.append(
            "The reference model is not fully calibrated. Treat the aggregate "
            "posterior as approximate."
        )

    # ---- temporal changepoint: did evidence concentrate in a window? ------
    temporal = _temporal_changepoint(scored)
    if temporal.detected:
        caveats.append(temporal.detail)

    # ---- variance analysis: spiked vs uniform evidence --------------------
    evidence_variance = None
    outlier_games = []
    if n_scored >= 5:
        dbs = np.array([g.decibans for g in scored])
        evidence_variance = float(dbs.var())
        # Outlier games: individual dB more than 2 sd above the mean.
        mu, sd_db = float(dbs.mean()), float(dbs.std())
        if sd_db > 0:
            outlier_games = [
                g.game_index for g in scored
                if g.decibans > mu + 2 * sd_db
            ]
        if len(outlier_games) > 0 and len(outlier_games) <= 3:
            caveats.append(
                f"evidence is concentrated in {len(outlier_games)} outlier game(s) "
                f"({', '.join(str(i + 1) for i in outlier_games)}); the remaining "
                f"games are unremarkable"
            )

    # ---- speed breakdown --------------------------------------------------
    speed_breakdown = _speed_breakdown(scored)

    if len(speed_breakdown) >= 2:
        sorted_speeds = sorted(speed_breakdown, key=lambda s: -s["total_db"])
        top = sorted_speeds[0]
        if top["total_db"] > 5 and top["n_games"] >= 3:
            caveats.append(
                f"evidence is strongest in {top['speed']} games "
                f"({top['total_db']:+.1f} dB from {top['n_games']} games)"
            )

    # ---- IPR trajectory ---------------------------------------------------
    ipr_trajectory = [
        {"game": g.game_index + 1, "ipr": g.ipr_elo,
         "rated": g.game_info.get("white_elo") if g.color == "white"
                  else g.game_info.get("black_elo")}
        for g in game_results
        if g.ipr_elo is not None
    ]

    return ScanResult(
        username=username,
        source="scan",
        n_games=len(game_results),
        n_games_scored=n_scored,
        total_positions=total_positions,
        total_log_lr=total_log_lr,
        total_decibans=total_db,
        posterior=final_posterior,
        posterior_low=float(lo),
        posterior_high=float(hi),
        prior=prior,
        interpretation=interp,
        games=game_results,
        convergence=convergence,
        signal_summary=signal_summary,
        temporal=temporal,
        flagged_games=flagged_games,
        peak_game_db=peak_game_db,
        peak_game_posterior=peak_game_posterior,
        evidence_variance=evidence_variance,
        outlier_games=outlier_games,
        speed_breakdown=speed_breakdown,
        ipr_trajectory=ipr_trajectory,
        caveats=caveats,
    )


# ---------------------------------------------------------------------------
# inter-game temporal changepoint
# ---------------------------------------------------------------------------


def _temporal_changepoint(
    scored: list[GameResult],
    min_segment: int = 3,
    n_perm: int = 1000,
) -> TemporalChangepoint:
    """Detect whether evidence concentrates in a contiguous block of games.

    Same maximum-LLR scan as the intra-game switchpoint, applied to the
    per-game deciban sequence.  A detected changepoint means "the player was
    unremarkable for games 1..k and then evidence starts accumulating from
    k+1 onward" (or vice versa).
    """
    if len(scored) < 2 * min_segment:
        return TemporalChangepoint()

    dbs = np.array([g.decibans for g in scored])
    n = len(dbs)

    # Use a simple mean-shift model: fit two group means separated at k.
    def _mse_llr(vals: np.ndarray, k: int) -> float:
        g1, g2 = vals[:k], vals[k:]
        ss_null = float(((vals - vals.mean()) ** 2).sum())
        ss_alt = float(((g1 - g1.mean()) ** 2).sum() + ((g2 - g2.mean()) ** 2).sum())
        if ss_null <= 0:
            return 0.0
        return max(0.0, (ss_null - ss_alt) / max(ss_null, 1e-12) * n)

    best_stat, best_k = 0.0, None
    for k in range(min_segment, n - min_segment + 1):
        stat = _mse_llr(dbs, k)
        if stat > best_stat:
            best_stat, best_k = stat, k

    if best_k is None:
        return TemporalChangepoint()

    # Permutation p-value.
    rng = np.random.default_rng(99)
    count_ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(dbs)
        perm_best = 0.0
        for k in range(min_segment, n - min_segment + 1):
            s = _mse_llr(perm, k)
            if s > perm_best:
                perm_best = s
        if perm_best >= best_stat:
            count_ge += 1
    p = (count_ge + 1) / (n_perm + 1)

    db_before = float(dbs[:best_k].mean())
    db_after = float(dbs[best_k:].mean())

    detected = p < 0.05 and abs(db_after - db_before) > 2.0
    detail = ""
    if detected:
        direction = "increases" if db_after > db_before else "decreases"
        detail = (
            f"evidence {direction} sharply at game {best_k + 1}: "
            f"mean {db_before:+.1f} dB/game before vs {db_after:+.1f} dB/game "
            f"after (p={p:.3f})"
        )

    return TemporalChangepoint(
        detected=detected,
        split_game=best_k,
        db_before=db_before,
        db_after=db_after,
        p_value=p,
        detail=detail,
    )


# ---------------------------------------------------------------------------
# speed breakdown
# ---------------------------------------------------------------------------


def _speed_breakdown(scored: list[GameResult]) -> list[dict]:
    """Aggregate evidence by time control."""
    by_speed: dict[str, list[GameResult]] = {}
    for g in scored:
        sp = g.speed or "unknown"
        by_speed.setdefault(sp, []).append(g)

    out = []
    for sp, games in sorted(by_speed.items()):
        total_db = sum(g.decibans for g in games)
        t1s = [g.t1_rate for g in games if g.t1_rate is not None]
        acpls = [g.acpl for g in games if g.acpl is not None]
        out.append({
            "speed": sp,
            "n_games": len(games),
            "total_db": total_db,
            "mean_db": total_db / len(games),
            "mean_t1": float(np.mean(t1s)) if t1s else None,
            "mean_acpl": float(np.mean(acpls)) if acpls else None,
        })
    return out
