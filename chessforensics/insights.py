"""Game insights: the statistics the engine pass produces as a by-product.

Nothing here is forensic. It is the "what actually happened in this game" view
that Game Review does not expose: win-probability swings, a luck-versus-skill
decomposition of the result, critical moments, only-moves found and missed, and
how sharp the game was over time.

Win probability uses the Lichess-style logistic map from centipawns, which is
fitted to real game outcomes and is far more interpretable than raw eval: the
difference between +0.3 and +0.6 matters enormously, between +6.0 and +9.0 not
at all.
"""

from __future__ import annotations

import math

import numpy as np

from .engine import GameAnalysis
from .ingest import Game

# Lichess' fitted constant for cp -> win probability.
WIN_PROB_K = 0.00368208


def win_prob(cp: float) -> float:
    """Probability the side to move wins, from centipawns."""
    return 1.0 / (1.0 + math.exp(-WIN_PROB_K * max(-1500.0, min(1500.0, cp))))


def win_prob_white(cp_white: float) -> float:
    return win_prob(cp_white)


def _accuracy_from_winprob_drop(drop: float) -> float:
    """Lichess-style per-move accuracy percentage from a win-probability drop.

    Included because it is the number users recognise from Game Review, so
    having it here makes the forensic figures easier to sanity-check against a
    familiar scale.
    """
    pct = 103.1668 * math.exp(-0.04354 * (drop * 100.0)) - 3.1669
    return max(0.0, min(100.0, pct))


def build(game: Game, ga: GameAnalysis, df) -> dict:
    """Assemble the insights payload for one game."""
    if df is None or len(df) == 0:
        return {"available": False}

    rows = []
    for _, r in df.iterrows():
        cp_white = float(r["eval_before"]) * (1 if r["color"] == "white" else -1)
        wp = win_prob_white(cp_white)
        after = r["eval_after"]
        if after is None or (isinstance(after, float) and math.isnan(after)):
            wp_after = wp
        else:
            cp_after_white = float(after) * (1 if r["color"] == "white" else -1)
            wp_after = win_prob_white(cp_after_white)

        # Swing from the mover's point of view: positive = they gained.
        swing = (wp_after - wp) * (1 if r["color"] == "white" else -1)
        drop = max(0.0, -swing)
        rows.append(
            {
                "ply": int(r["ply"]),
                "move_number": int(r["move_number"]),
                "color": r["color"],
                "san": r["san"],
                "wp_white_before": wp,
                "wp_white_after": wp_after,
                "swing": swing,
                "drop": drop,
                "accuracy": _accuracy_from_winprob_drop(drop),
                "complexity": float(r["complexity"]),
                "delta_cp": None if r["delta_cp"] is None else float(r["delta_cp"]),
                "gap_cp": None if r["gap_cp"] is None else float(r["gap_cp"]),
                "matched_t1": int(r["matched_t1"]),
                "is_forced": int(r["is_forced"]),
                "scoreable": int(r["scoreable"]),
                "critical_reasons": r["critical_reasons"],
            }
        )

    per_player = {}
    for color in ("white", "black"):
        mine = [r for r in rows if r["color"] == color]
        if not mine:
            continue
        drops = np.array([r["drop"] for r in mine])
        accs = np.array([r["accuracy"] for r in mine])
        scoreable = [r for r in mine if r["scoreable"]]

        # Only-moves: positions where one move stood far above the rest.
        only_moves = [r for r in scoreable if (r["gap_cp"] or 0) > 150]
        only_found = [r for r in only_moves if r["matched_t1"]]

        # Luck / skill split. "Skill" is what the player gave away through
        # their own inaccuracies; "luck" is what the opponent handed them.
        # They are not symmetric and should not be read as a single number.
        theirs = [r for r in rows if r["color"] != color]
        gifted = sum(r["drop"] for r in theirs)
        spilled = float(drops.sum())

        per_player[color] = {
            "accuracy": float(accs.mean()) if len(accs) else None,
            "total_wp_lost": spilled,
            "total_wp_gifted": float(gifted),
            "net_wp": float(gifted - spilled),
            "biggest_error": max(mine, key=lambda r: r["drop"]) if len(mine) else None,
            "best_move_played": (
                max(
                    (r for r in mine if r["swing"] > 0),
                    key=lambda r: r["swing"],
                    default=None,
                )
            ),
            "only_moves_total": len(only_moves),
            "only_moves_found": len(only_found),
            "only_move_rate": (
                len(only_found) / len(only_moves) if only_moves else None
            ),
            "mean_complexity_faced": (
                float(np.mean([r["complexity"] for r in scoreable])) if scoreable else None
            ),
        }

    # Critical moments: the plies where the win probability moved most.
    critical = sorted(rows, key=lambda r: -r["drop"])[:8]
    critical = [r for r in critical if r["drop"] > 0.08]

    # Sharpness over time: how much was at stake on each move.
    sharpness = [
        {
            "ply": r["ply"],
            "move_number": r["move_number"],
            "complexity": r["complexity"],
            "gap_cp": r["gap_cp"],
            "wp_white": r["wp_white_before"],
        }
        for r in rows
    ]

    swings = np.array([abs(r["swing"]) for r in rows])
    return {
        "available": True,
        "moves": rows,
        "per_player": per_player,
        "critical_moments": critical,
        "sharpness": sharpness,
        "game_volatility": float(swings.mean()) if len(swings) else 0.0,
        "max_swing": float(swings.max()) if len(swings) else 0.0,
        "lead_changes": _lead_changes(rows),
    }


def _lead_changes(rows: list[dict]) -> int:
    """How many times the game crossed from one side being better to the other."""
    changes = 0
    prev_side = 0
    for r in rows:
        wp = r["wp_white_after"]
        side = 1 if wp > 0.55 else (-1 if wp < 0.45 else 0)
        if side != 0 and prev_side != 0 and side != prev_side:
            changes += 1
        if side != 0:
            prev_side = side
    return changes
