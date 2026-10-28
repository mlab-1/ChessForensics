"""Top-level orchestration: game in, per-player report out.

This is the only module the UI needs to call. It owns the order of operations
and, importantly, the decision about which signals are admissible for a given
game -- a 15-move bullet game cannot support most of the battery, and the right
response is to drop those signals rather than compute them on inadequate data
and present the result as if it meant something.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import insights as insights_mod
from . import score, slope, timing
from . import switchpoint as sp_mod
from .baselines import Baseline, band_for, load_baseline
from .complexity import ComplexityWeights
from .engine import (
    PASS0_DEPTH,
    PASS1_DEPTH,
    PASS1_MULTIPV,
    PASS2_DEPTH,
    PASS2_MULTIPV,
    AnalysisCache,
    EnginePool,
    GameAnalysis,
    analyse_game,
)
from .features import build_frame, detect_book_plies, frame_for, summarise
from .ingest import Color, Game
from .model import (
    BASELINE_MULTIPV,
    ChoiceModel,
    check_multipv_consistency,
    count_test,
    expected_rates,
    fit_choice_model,
    ipr_from_model,
    mean_test,
    positions_from_frame,
    tail_test,
)

# Minimum scoreable decisions before we are willing to report a posterior.
# Below this a game carries so little information that any number is theatre.
MIN_SCOREABLE = 8

ProgressFn = Callable[[int, int, str], None]


@dataclass
class PlayerReport:
    color: str
    name: str
    elo: int | None
    band: str

    summary: dict = field(default_factory=dict)
    expected: dict = field(default_factory=dict)

    t1_test: dict | None = None
    t2_test: dict | None = None
    t3_test: dict | None = None
    acpl_test: dict | None = None

    slope_fit: dict | None = None
    slope_test: dict | None = None
    cpl_slope: dict | None = None
    curve: list[dict] = field(default_factory=list)

    ipr: dict | None = None
    fitted_model: dict | None = None

    timing: dict | None = None
    timing_per_move: list[dict] = field(default_factory=list)
    timing_accuracy: dict | None = None

    switchpoint: dict | None = None
    tail_test: dict | None = None

    # Second engine / L0
    deep_only_match_rate: float | None = None
    deep_only_test: dict | None = None
    second_engine_tag: str = ""
    sf_only_match_rate: float | None = None
    sf_vs_second_detail: dict | None = None

    verdict: dict | None = None
    games_curve: list[dict] = field(default_factory=list)
    games_to_95: int | None = None

    caveats: list[str] = field(default_factory=list)
    insufficient: bool = False

    def to_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class Report:
    game_info: dict
    frame_records: list[dict]
    players: dict[str, PlayerReport]
    baseline_description: str
    uncalibrated: bool
    provisional: bool
    critical: dict[int, list[str]]
    eval_curve: list[dict]
    insights: dict = field(default_factory=dict)
    engine_settings: dict = field(default_factory=dict)


def _band_stats(baseline: Baseline, elo: int | None) -> dict:
    return baseline.band(elo) or {}


def _signal(
    name: str,
    label: str,
    z: float,
    baseline: Baseline,
    observed: float | None = None,
    expected: float | None = None,
    weight: float = 1.0,
    detail: str = "",
    band: str | None = None,
) -> score.Signal:
    effect = float(baseline.effects.get(name, score.DEFAULT_EFFECTS.get(name, 1.5)))

    # Per-band null calibration: correct z for the bias and overdispersion
    # measured in the negative controls for this rating band. Falls back to
    # the pooled (all-band) calibration, then to the hardcoded EMPIRICAL_NULL
    # in score.py (which was measured on the corpus and is better than nothing).
    cal = None
    by_band = baseline.null_calibration_by_band
    if by_band and band:
        cal = (by_band.get(band) or {}).get(name)
    if cal is None:
        pooled = baseline.null_calibration
        if pooled:
            cal = pooled.get(name)
    if cal is None:
        # Hardcoded fallback from score.EMPIRICAL_NULL (bias AND scale)
        empirical = score.EMPIRICAL_NULL.get(name)
        if empirical is not None:
            cal = {"bias": empirical[0], "scale": empirical[1]}
    if cal is not None:
        bias = cal.get("bias", 0.0)
        scale = cal.get("scale", 1.0)
        if scale > 0:
            z = (z - bias) / scale

    return score.Signal(
        name=name,
        label=label,
        z=z,
        effect=effect,
        weight=weight,
        observed=observed,
        expected=expected,
        detail=detail,
    )


def analyse_player(
    game: Game,
    ga: GameAnalysis,
    df,
    color: Color,
    baseline: Baseline,
    prior: float = score.DEFAULT_PRIOR,
) -> PlayerReport:
    """Run the full battery for one side."""
    player = game.players[color]
    elo = player.elo
    band_name = band_for(elo)
    rep = PlayerReport(color=color, name=player.name, elo=elo, band=band_name)

    rep.summary = summarise(df, color)
    sub = frame_for(df, color)
    n = len(sub)

    if elo is None:
        rep.caveats.append(
            "no rating in the game headers, so the reference band is a guess. "
            "Every observed-vs-expected comparison below is only as good as that "
            "guess -- supply a rating if you have one."
        )

    if n < MIN_SCOREABLE:
        rep.insufficient = True
        rep.caveats.append(
            f"only {n} scoreable decisions (book moves, forced replies and "
            f"already-decided positions are excluded). That is far too few to "
            f"support any inference, so no posterior is reported."
        )
        return rep

    band = _band_stats(baseline, elo)
    if band.get("_fallback_from"):
        rep.caveats.append(
            f"no corpus data for band {band_name}; using {band['_fallback_from']} "
            f"as the nearest available reference"
        )

    # ---- reference model expectations for THESE positions ---------------
    positions = positions_from_frame(df, ga, color)
    band_cm_raw = band.get("choice_model") or {}
    band_s = float(band_cm_raw.get("s", 0.18))
    band_c = float(band_cm_raw.get("c", 0.47))
    band_model = ChoiceModel(s=band_s, c=band_c)
    exp = expected_rates(band_model, positions)
    rep.expected = {
        "t1": float(np.mean(exp["p_t1"])) if exp["p_t1"] else None,
        "t2": float(np.mean(exp["p_t2"])) if exp["p_t2"] else None,
        "t3": float(np.mean(exp["p_t3"])) if exp["p_t3"] else None,
        "acpl": float(np.mean(exp["e_delta"])) if exp["e_delta"] else None,
        "band_s": band_model.s,
        "band_c": band_model.c,
    }

    signals: list[score.Signal] = []

    def sig(name, label, z, **kw) -> score.Signal:
        return _signal(name, label, z, baseline, band=band_name, **kw)

    # ---- engine correlation: match counts, Poisson-binomial -------------
    if exp["p_t1"] and len(exp["p_t1"]) == n:
        for key, probs_key, label in (
            ("t1", "p_t1", "top-1 match rate"),
            ("t2", "p_t2", "top-2 match rate"),
            ("t3", "p_t3", "top-3 match rate"),
        ):
            probs = np.array(exp[probs_key])
            observed = int(sub[f"matched_{key}"].sum())
            ct = count_test(probs, observed)
            setattr(rep, f"{key}_test", ct.to_dict())
            signals.append(
                sig(
                    f"{key}_match",
                    label,
                    ct.z,
                    observed=observed / n,
                    expected=ct.expected / n,
                    detail=(
                        f"{observed}/{n} vs {ct.expected:.1f} expected "
                        f"(exact p={ct.p_one_sided:.3g})"
                    ),
                )
            )

        # ACPL, with a proper sum-of-variances rather than a pooled estimate.
        mt = mean_test(
            np.array(exp["e_delta"]),
            np.array(exp["v_delta"]),
            float(sub["delta_cp"].mean()),
            lower_is_suspicious=True,
        )
        rep.acpl_test = mt.to_dict()
        # mean_test z is negative when the player lost less eval than expected,
        # which is the suspicious direction, so flip the sign for scoring.
        signals.append(
            sig(
                "acpl",
                "average centipawn loss",
                -mt.z,
                observed=mt.observed,
                expected=mt.expected,
                detail=f"{mt.observed:.0f}cp vs {mt.expected:.0f}cp expected",
            )
        )

    # ---- intrinsic performance rating -----------------------------------
    fitted = fit_choice_model(positions)
    rep.fitted_model = fitted.to_dict()
    ipr = ipr_from_model(fitted, baseline.calibration_curve())
    rep.ipr = ipr.to_dict()

    mismatch = check_multipv_consistency(PASS2_MULTIPV, baseline.multipv)
    if mismatch:
        rep.caveats.append(mismatch)

    if ipr.elo is not None and elo is not None:
        gap = ipr.elo - elo
        # The IPR gap only becomes *evidence* if the s -> Elo curve it was read
        # off is itself trustworthy. The scale of ``s`` is specific to this
        # pipeline's depth and MultiPV width, so the shipped defaults -- shaped
        # from literature about a different pipeline -- put it in the wrong
        # place entirely: they map a genuinely 2785-strength performance to an
        # IPR near 1700. Fed into the posterior that does not merely add noise,
        # it adds a large *exculpating* term to strong play and understates
        # assisted play, which is the worst available failure mode. So on an
        # uncalibrated or provisional reference the IPR is still computed and
        # displayed, clearly labelled, but contributes no evidence.
        if baseline.trustworthy and not ipr.extrapolated:
            # Spread of single-game IPR estimates is genuinely large; 250 Elo is
            # a realistic standard deviation for ~30 decisions.
            ipr_sd = 250.0
            signals.append(
                sig(
                    "ipr_gap",
                    "intrinsic rating vs account rating",
                    gap / ipr_sd,
                    observed=ipr.elo,
                    expected=float(elo),
                    detail=f"IPR {ipr.elo:.0f} vs rated {elo} ({gap:+.0f})",
                )
            )
        else:
            rep.caveats.append(
                f"IPR {ipr.elo:.0f} (vs rated {elo}) is shown for information only "
                f"and contributes no evidence: the sensitivity-to-Elo curve comes "
                f"from an uncalibrated reference, so its scale is not established"
            )
        if ipr.note:
            rep.caveats.append(ipr.note)

    # ---- accuracy vs complexity -----------------------------------------
    cx = sub["complexity"].to_numpy(dtype=float)
    matched = sub["matched_t1"].to_numpy(dtype=float)
    sfit = slope.fit_logistic_slope(cx, matched)
    rep.slope_fit = sfit.to_dict()
    rep.curve = slope.binned_curve(cx, matched)

    band_slope = band.get("slope") or {}
    st = slope.test_slope_against_band(
        sfit,
        float(band_slope.get("mean", -2.4)),
        float(band_slope.get("sd", 1.0)),
        ipr_elo=ipr.elo,
        nominal_elo=float(elo) if elo is not None else None,
    )
    rep.slope_test = st.to_dict()
    rep.caveats.extend(st.caveats)
    if not sfit.separable and sfit.converged:
        signals.append(
            sig(
                "slope",
                "accuracy-vs-complexity slope",
                st.z,
                observed=sfit.beta1,
                expected=st.band_mean,
                detail=(
                    f"beta1={sfit.beta1:+.2f} "
                    f"(se {sfit.se_beta1:.2f}) vs band {st.band_mean:+.2f}"
                    if sfit.se_beta1
                    else f"beta1={sfit.beta1:+.2f} vs band {st.band_mean:+.2f}"
                ),
            )
        )

    # Second, correlated view: centipawn loss as a continuous outcome.
    cpl = sub["delta_cp"].to_numpy(dtype=float)
    icept, lslope, lse, ln = slope.fit_linear_slope(cx, cpl)
    rep.cpl_slope = {
        "intercept": icept,
        "slope": lslope,
        "se": lse,
        "n": ln,
    }
    band_cpl = band.get("cpl_slope") or {}
    if lse:
        cpl_mu = float(band_cpl.get("mean", 120.0))
        cpl_sd = float(band_cpl.get("sd", 75.0))
        total_sd = float(np.hypot(cpl_sd, lse))
        signals.append(
            sig(
                "cpl_slope",
                "centipawn-loss growth with complexity",
                -(lslope - cpl_mu) / total_sd,
                observed=lslope,
                expected=cpl_mu,
                detail=f"{lslope:+.0f}cp per unit complexity vs {cpl_mu:+.0f} expected",
            )
        )

    # ---- move-time forensics --------------------------------------------
    band_timing = band.get("timing") or {}
    tres = timing.analyse_timing(
        df, color, speed=game.speed, increment=game.increment,
        baseline_times=band.get("think_times"),
    )
    rep.timing = tres.to_dict()
    rep.timing_per_move = tres.per_move
    if tres.available:
        rep.caveats.extend(tres.caveats)
        for tsig in timing.timing_signals(tres, band_timing):
            signals.append(
                sig(
                    tsig["name"],
                    tsig["label"],
                    tsig["z"],
                    observed=tsig["value"],
                    expected=tsig["expected"],
                    weight=tsig["weight"],
                    detail=f"{tsig['value']:.2f} vs {tsig['expected']:.2f} expected",
                )
            )
    elif tres.reason:
        rep.caveats.append(tres.reason)

    # ---- switchpoint: mid-game accuracy jump ------------------------------
    if n >= 20:
        sp_matched = sub["matched_t1"].to_numpy(dtype=float)
        sp_cpl = sub["delta_cp"].to_numpy(dtype=float)
        # Eval from the player's perspective: positive = good for them.
        sp_eval = sub["eval_before"].to_numpy(dtype=float)
        if color == "black":
            sp_eval = -sp_eval
        sp_result = sp_mod.detect_switchpoint(
            sp_matched,
            sp_cpl,
            sp_eval,
            sub["ply"].to_numpy(),
            sub["move_number"].to_numpy(),
            color,
        )
        rep.switchpoint = sp_result.to_dict()
        rep.caveats.extend(sp_result.caveats)
        sp_sig = sp_mod.switchpoint_signal(sp_result)
        if sp_sig is not None:
            signals.append(
                sig(
                    sp_sig["name"],
                    sp_sig["label"],
                    sp_sig["z"],
                    observed=sp_sig["observed"],
                    expected=sp_sig["expected"],
                    weight=sp_sig.get("weight", 1.0),
                    detail=sp_sig["detail"],
                )
            )

    # ---- CPL tail test: truncated error distribution ---------------------
    if exp["e_delta"] and len(exp["e_delta"]) == n:
        tt = tail_test(
            sub["delta_cp"].to_numpy(dtype=float),
            np.array(exp["e_delta"]),
            np.array(exp["v_delta"]),
        )
        if tt is not None:
            rep.tail_test = tt.to_dict()
            signals.append(
                sig(
                    "cpl_tail",
                    "centipawn loss tail (blunder rate)",
                    tt.z,
                    observed=tt.observed_blunder_rate,
                    expected=tt.expected_blunder_rate,
                    detail=(
                        f"blunder rate {tt.observed_blunder_rate:.1%} vs "
                        f"{tt.expected_blunder_rate:.1%} expected, "
                        f"P90 {tt.observed_p90:.0f}cp vs {tt.expected_p90:.0f}cp"
                    ),
                )
            )

    # ---- timing x accuracy cross-signal ----------------------------------
    ta_cross = timing.analyse_timing_accuracy(
        df, color, speed=game.speed, band_timing=band.get("timing"),
    )
    if ta_cross.available:
        rep.timing_accuracy = ta_cross.to_dict()
        signals.append(
            sig(
                "timing_accuracy_cross",
                "think time on hard correct moves",
                ta_cross.z,
                observed=ta_cross.mean_time_hard_correct,
                expected=ta_cross.expected,
                weight=timing.speed_weight(game.speed),
                detail=(
                    f"{ta_cross.mean_time_hard_correct:.1f}s on {ta_cross.n_hard_correct} "
                    f"hard correct moves vs {ta_cross.expected:.1f}s expected"
                ),
            )
        )

    # ---- L0 deep-only match rate -----------------------------------------
    # Only run if L0 data was actually populated (at least some positions
    # have l0_top1_uci set). When run_l0=False the column exists but is all
    # None, which would make l0_agrees==0 everywhere and produce a
    # nonsensical exculpating signal.
    l0_populated = (
        "l0_top1_uci" in sub.columns
        and sub["l0_top1_uci"].notna().sum() > 0
    )
    if l0_populated and "deep_only_match" in sub.columns and "l0_agrees" in sub.columns:
        l0_disagree = sub[sub["l0_agrees"] == 0]
        if len(l0_disagree) >= 8:
            deep_only_count = int(l0_disagree["deep_only_match"].sum())
            deep_only_n = len(l0_disagree)
            deep_only_rate = deep_only_count / deep_only_n
            rep.deep_only_match_rate = deep_only_rate

            # Build per-position probabilities for a Poisson-binomial test on
            # the deep-only matches, conditioned on positions where L0 disagrees.
            l0_positions = [
                p for i, p in enumerate(positions)
                if i < len(sub) and int(sub.iloc[i].get("l0_agrees", 1)) == 0
            ]
            if l0_positions and len(l0_positions) == deep_only_n:
                l0_probs = np.array([band_model.p_top1(p) for p in l0_positions])
                do_test = count_test(l0_probs, deep_only_count)
                rep.deep_only_test = do_test.to_dict()
                signals.append(
                    sig(
                        "deep_only_match",
                        "deep-only match rate (L0 disagrees)",
                        do_test.z,
                        observed=deep_only_rate,
                        expected=do_test.expected / deep_only_n if deep_only_n else None,
                        detail=(
                            f"{deep_only_count}/{deep_only_n} where shallow disagrees "
                            f"(p={do_test.p_one_sided:.3g})"
                        ),
                    )
                )

    # ---- second engine fingerprint ---------------------------------------
    # Only run if second engine data was actually populated.
    sec_populated = (
        "second_top1_uci" in sub.columns
        and sub["second_top1_uci"].notna().sum() > 0
    )
    if sec_populated and "sf_only_match" in sub.columns and "second_agrees" in sub.columns:
        sec_disagree = sub[sub["second_agrees"] == 0]
        if len(sec_disagree) >= 8:
            sf_only_count = int(sec_disagree["sf_only_match"].sum())
            sec_only_count = int(
                (sec_disagree["matched_second"] == 1).sum()
                - (sec_disagree["matched_t1"] == 1).sum()
                + sf_only_count  # those who matched both are already excluded
            )
            sf_only_n = len(sec_disagree)
            sf_only_rate = sf_only_count / sf_only_n
            rep.sf_only_match_rate = sf_only_rate
            rep.second_engine_tag = ga.second_tag

            # On positions where the two engines disagree, how often does the
            # player follow Stockfish specifically?  Under H0 (no engine), they
            # should follow neither at a rate set by the reference model.
            # Under H1 (Stockfish user), they follow SF much more.
            # Expected SF-only match rate: very roughly p_top1 * (1 - agreement_rate)
            overall_agree = float(sub["second_agrees"].mean()) if len(sub) else 0.5
            exp_sf_only = float(
                np.mean([band_model.p_top1(p) for p in positions]) * (1 - overall_agree)
            ) if positions else 0.1
            sd_sf = max(0.05, math.sqrt(exp_sf_only * (1 - exp_sf_only) / max(sf_only_n, 1)))
            z_fp = (sf_only_rate - exp_sf_only) / sd_sf

            rep.sf_vs_second_detail = {
                "sf_only_rate": sf_only_rate,
                "sf_only_count": sf_only_count,
                "n_disagree": sf_only_n,
                "agreement_rate": float(sub["second_agrees"].mean()),
                "second_tag": ga.second_tag,
            }
            signals.append(
                sig(
                    "engine_fingerprint",
                    f"Stockfish-specific match rate (vs {ga.second_tag})",
                    z_fp,
                    observed=sf_only_rate,
                    expected=exp_sf_only,
                    detail=(
                        f"{sf_only_count}/{sf_only_n} SF-only matches where "
                        f"{ga.second_tag} disagrees"
                    ),
                )
            )

    # ---- combine ---------------------------------------------------------
    extra = []
    if baseline.provisional:
        extra.append(
            f"the reference model is provisional ({baseline.n_games} corpus games). "
            f"The fitted scale is probably about right, but each band's expected "
            f"rates and slope spread rest on very few games, so the z-scores "
            f"driving this posterior are themselves uncertain"
        )
    if n < 25:
        extra.append(
            f"only {n} scoreable decisions in this game; single-game evidence is "
            f"intrinsically weak and the credible interval reflects that"
        )
    verdict = score.combine(
        signals,
        prior=prior,
        corpus_R=baseline.correlation,
        shrinkage=baseline.shrinkage(),
        extra_caveats=extra,
        uncalibrated=baseline.is_default,
        baseline_n_games=baseline.n_games,
    )
    rep.verdict = verdict.to_dict()
    rep.verdict["interpretation"] = score.interpret(verdict)
    rep.games_curve = score.games_curve(verdict, prior=prior)
    rep.games_to_95 = score.games_to_threshold(verdict, 0.95, prior)
    return rep


def analyse(
    game: Game,
    pool: EnginePool | None = None,
    cache: AnalysisCache | None = None,
    baseline: Baseline | None = None,
    prior: float = score.DEFAULT_PRIOR,
    progress: ProgressFn | None = None,
    deep_pass: bool = True,
    pool_size: int | None = None,
    second_pool: EnginePool | None = None,
    second_cache: AnalysisCache | None = None,
    second_nodes: int | None = None,
    run_l0: bool = True,
) -> Report:
    """Analyse a whole game and produce reports for both players."""
    baseline = baseline or load_baseline()
    cache = cache if cache is not None else AnalysisCache()

    fens = [m.fen_before for m in game.moves]
    ucis = [m.uci for m in game.moves]

    owns_pool = pool is None
    pool = pool or EnginePool(size=pool_size)
    try:
        ga = analyse_game(
            fens, ucis, pool, cache, progress=progress, deep_pass=deep_pass,
            run_l0=run_l0,
            second_pool=second_pool, second_cache=second_cache,
            second_nodes=second_nodes,
        )
    finally:
        if owns_pool:
            pool.close()

    weights = ComplexityWeights.from_dict(baseline.complexity_weights)
    book_plies = detect_book_plies(game.moves)
    df = build_frame(game, ga, weights, book_plies=book_plies)

    players = {
        color: analyse_player(game, ga, df, color, baseline, prior)
        for color in ("white", "black")
    }

    eval_curve = []
    for mv in game.moves:
        pa = ga.at(mv.ply)
        if pa is None or not pa.candidates:
            continue
        # Normalise every eval to white's point of view for a readable graph.
        cp = pa.best_cp if mv.color == "white" else -pa.best_cp
        eval_curve.append(
            {
                "ply": mv.ply,
                "move_number": mv.move_number,
                "color": mv.color,
                "san": mv.san,
                "eval_white": cp / 100.0,
                "depth": pa.depth,
                "critical": mv.ply in ga.critical,
            }
        )
    # Add the final position eval (after the last move) so the eval bar
    # can show the correct value for the board's final state.
    if game.moves:
        last = game.moves[-1]
        final_ply = last.ply + 1
        pa_final = ga.at(final_ply)
        if pa_final and pa_final.candidates:
            final_color = "black" if last.color == "white" else "white"
            cp_f = pa_final.best_cp if final_color == "white" else -pa_final.best_cp
            eval_curve.append({
                "ply": final_ply,
                "move_number": last.move_number,
                "color": final_color,
                "san": "",
                "eval_white": cp_f / 100.0,
                "depth": pa_final.depth,
                "critical": False,
            })

    game_info = {
        "white": game.players["white"].name,
        "black": game.players["black"].name,
        "white_elo": game.players["white"].elo,
        "black_elo": game.players["black"].elo,
        "result": game.result,
        "time_control": game.time_control,
        "speed": game.speed,
        "source": game.source,
        "url": game.url,
        "moves": len(game.moves),
        "has_clocks": game.has_clocks,
        "event": game.headers.get("Event"),
        "date": game.headers.get("UTCDate") or game.headers.get("Date"),
        "opening": game.headers.get("Opening") or game.headers.get("ECO"),
    }

    return Report(
        game_info=game_info,
        frame_records=df.to_dict("records") if len(df) else [],
        players=players,
        baseline_description=baseline.describe(),
        uncalibrated=baseline.is_default,
        provisional=baseline.provisional,
        critical=ga.critical,
        eval_curve=eval_curve,
        insights=insights_mod.build(game, ga, df),
        engine_settings={
            "pass0_depth": PASS0_DEPTH,
            "pass1_depth": PASS1_DEPTH,
            "pass1_multipv": PASS1_MULTIPV,
            "pass2_depth": PASS2_DEPTH,
            "pass2_multipv": PASS2_MULTIPV,
            "deep_positions": len(ga.critical),
            "l0_positions": len(ga.l0),
            "second_engine": ga.second_tag or None,
            "second_positions": len(ga.second),
            "total_positions": len(fens),
            "baseline_multipv": baseline.multipv,
            "model_multipv": BASELINE_MULTIPV,
        },
    )
