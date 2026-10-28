"""Per-move feature frame: the single table everything else reads from.

The one genuinely delicate thing in here is centipawn loss. There are two ways
to get it and they are not interchangeable:

1. If the played move appears in the MultiPV list, ``delta = top1_cp -
   played_cp`` with both numbers from the *same search*. Clean.
2. Otherwise we fall back on the next ply: ``delta = best_cp(t) +
   best_cp(t+1)`` (the signs work out because each eval is from the point of
   view of whoever is to move).

Route 2 is only valid if both plies were searched to the same depth. Because
the adaptive schedule deepens some plies and not others, naively mixing a
depth-22 eval with a depth-12 one produces a systematic bias -- typically
inventing 10-30cp of "loss" that is really just the deeper search seeing
further. So route 2 deliberately re-reads both plies from the shallow pass, and
the depth actually used is recorded in ``delta_depth`` so the bias is visible
rather than silent.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd
import requests

from .complexity import ComplexityWeights, complexity_of
from .engine import GameAnalysis, PositionAnalysis
from .ingest import Color, Game

# Beyond this, the position is decided and "accuracy" stops meaning anything:
# throwing away 300cp when you are already +900 is not a blunder in any sense
# that distinguishes players.
DECIDED_CP = 600.0


@dataclass
class MoveFeature:
    ply: int
    move_number: int
    color: Color
    san: str
    uci: str
    fen_before: str
    legal_moves: int
    complexity: float
    delta_cp: float | None  # centipawn loss of the played move
    delta_depth: int  # search depth the delta came from
    delta_source: str  # "multipv" or "nextply"
    best_uci: str | None
    best_cp: float | None
    gap_cp: float | None
    rank: int | None  # rank of played move in MultiPV, None = outside top-k
    matched_t1: int
    matched_t2: int
    matched_t3: int
    is_forced: int  # only one legal move
    is_decided: int  # position already won/lost by a wide margin
    played_capture: int
    played_check: int
    best_quiet: int
    best_forcing: int
    phase: float
    phase_name: str  # "opening", "middlegame", "endgame"
    think_time: float | None
    depth_used: int
    critical_reasons: str
    eval_before: float | None
    eval_after: float | None
    # L0 shallow-engine features
    l0_top1_uci: str | None = None
    l0_agrees: int = 0       # 1 if L0 depth-4 top-1 == deep top-1
    matched_l0: int = 0      # 1 if the played move == L0 top-1
    deep_only_match: int = 0  # 1 if matched_t1 AND NOT l0_agrees
    # Second engine features (e.g. Lc0)
    second_top1_uci: str | None = None
    second_agrees: int = 0    # 1 if second engine top-1 == Stockfish top-1
    matched_second: int = 0   # 1 if the played move == second engine top-1
    sf_only_match: int = 0    # 1 if matched SF top-1 but NOT second engine top-1


def _consistent_delta(
    ga: GameAnalysis, ply: int, uci: str
) -> tuple[float | None, int, str]:
    """Centipawn loss for the move played at ``ply``.

    Returns (delta_cp, depth_used, source). See the module docstring for why
    the depth bookkeeping matters.
    """
    best_pa = ga.at(ply)
    if best_pa is None or not best_pa.candidates:
        return None, 0, "none"

    # Route 1: the played move was inside the MultiPV window, so both numbers
    # come from one search at one depth.
    played_cp = best_pa.cp_of(uci)
    if played_cp is not None:
        return max(0.0, best_pa.best_cp - played_cp), best_pa.depth, "multipv"

    # Route 2: next-ply differencing, forced onto a single consistent depth.
    this_shallow = ga.shallow_at(ply)
    next_shallow = ga.shallow_at(ply + 1)
    if this_shallow is not None and next_shallow is not None and next_shallow.candidates:
        delta = this_shallow.best_cp + next_shallow.best_cp
        return max(0.0, delta), this_shallow.depth, "nextply"

    # Last ply of the game, or a gap in the analysis. A checkmate or stalemate
    # delivered by the played move is not an error.
    next_pa = ga.at(ply + 1)
    if next_pa is not None and next_pa.legal_moves == 0:
        return 0.0, best_pa.depth, "terminal"
    return None, best_pa.depth, "none"


_BOOK_CACHE: dict[str, list[str] | None] = {}


def detect_book_plies(moves, min_games: int = 50) -> set[int] | None:
    """Detect book plies via the Lichess opening explorer.

    Returns ply indices where the played move appears in >= ``min_games``
    database games, or ``None`` if the API was completely unreachable (so the
    caller can fall back to a heuristic).

    Tracks each colour independently and stops querying once either side
    leaves book.  Results are cached so repeated openings across a scan are
    essentially free.
    """
    book_plies: set[int] = set()
    done = {"white": False, "black": False}
    any_success = False

    for m in moves:
        if done[m.color]:
            if done["white"] and done["black"]:
                break
            continue

        fen = m.fen_before
        if fen not in _BOOK_CACHE:
            try:
                resp = requests.get(
                    "https://explorer.lichess.ovh/lichess",
                    params={
                        "fen": fen,
                        "speeds": "rapid,classical,correspondence",
                        "ratings": "1600,1800,2000,2200,2500",
                    },
                    timeout=5,
                )
                if resp.ok:
                    any_success = True
                    data = resp.json()
                    _BOOK_CACHE[fen] = [
                        bm["san"]
                        for bm in data.get("moves", [])
                        if (
                            bm.get("white", 0)
                            + bm.get("draws", 0)
                            + bm.get("black", 0)
                        )
                        >= min_games
                    ]
                else:
                    _BOOK_CACHE[fen] = None
                time.sleep(0.08)
            except Exception:
                _BOOK_CACHE[fen] = None

        cached = _BOOK_CACHE.get(fen)
        if cached is None:
            done[m.color] = True
            continue

        if m.san in cached:
            any_success = True  # cache hit from a prior successful call
            book_plies.add(m.ply)
        else:
            done[m.color] = True

    return book_plies if any_success else None


def build_frame(
    game: Game,
    ga: GameAnalysis,
    weights: ComplexityWeights | None = None,
    book_plies: set[int] | None = None,
) -> pd.DataFrame:
    """Build the per-move feature frame for a whole game."""
    import chess

    rows: list[MoveFeature] = []
    for mv in game.moves:
        pa: PositionAnalysis | None = ga.at(mv.ply)
        if pa is None or not pa.candidates:
            continue

        prev_uci = game.moves[mv.ply - 1].uci if mv.ply > 0 else None
        cx, feats = complexity_of(pa, weights, prev_uci)
        delta, delta_depth, delta_source = _consistent_delta(ga, mv.ply, mv.uci)
        rank = pa.rank_of(mv.uci)

        board = chess.Board(mv.fen_before)
        try:
            played = chess.Move.from_uci(mv.uci)
            played_ok = played in board.legal_moves
        except ValueError:
            played, played_ok = None, False

        next_pa = ga.at(mv.ply + 1)
        eval_after = -next_pa.best_cp if next_pa is not None and next_pa.candidates else None

        best_uci = pa.best.uci if pa.best else None

        # L0 shallow-engine columns
        l0_pa = ga.l0_at(mv.ply)
        l0_top1 = l0_pa.best.uci if l0_pa and l0_pa.best else None
        l0_agrees = int(l0_top1 is not None and best_uci is not None and l0_top1 == best_uci)
        matched_l0 = int(l0_top1 is not None and mv.uci == l0_top1)
        matched_t1_val = int(rank == 1)
        deep_only = int(matched_t1_val == 1 and l0_agrees == 0) if l0_top1 else 0

        # Second engine columns
        sec_pa = ga.second_at(mv.ply)
        sec_top1 = sec_pa.best.uci if sec_pa and sec_pa.best else None
        sec_agrees = int(sec_top1 is not None and best_uci is not None and sec_top1 == best_uci)
        matched_sec = int(sec_top1 is not None and mv.uci == sec_top1)
        sf_only = int(matched_t1_val == 1 and matched_sec == 0) if sec_top1 else 0

        rows.append(
            MoveFeature(
                ply=mv.ply,
                move_number=mv.move_number,
                color=mv.color,
                san=mv.san,
                uci=mv.uci,
                fen_before=mv.fen_before,
                legal_moves=pa.legal_moves,
                complexity=cx,
                delta_cp=delta,
                delta_depth=delta_depth,
                delta_source=delta_source,
                best_uci=best_uci,
                best_cp=pa.best_cp,
                gap_cp=pa.gap(),
                rank=rank,
                matched_t1=matched_t1_val,
                matched_t2=int(rank is not None and rank <= 2),
                matched_t3=int(rank is not None and rank <= 3),
                is_forced=int(pa.legal_moves <= 1),
                is_decided=int(abs(pa.best_cp) > DECIDED_CP),
                played_capture=int(played_ok and board.is_capture(played)),
                played_check=int(played_ok and board.gives_check(played)),
                best_quiet=int(feats["best_quiet"] > 0),
                best_forcing=int(feats["best_forcing"] > 0),
                phase=feats["phase"],
                phase_name=feats["phase_name"],
                think_time=mv.think_time,
                depth_used=pa.depth,
                critical_reasons="|".join(ga.critical.get(mv.ply, [])),
                eval_before=pa.best_cp,
                eval_after=eval_after,
                l0_top1_uci=l0_top1,
                l0_agrees=l0_agrees,
                matched_l0=matched_l0,
                deep_only_match=deep_only,
                second_top1_uci=sec_top1,
                second_agrees=sec_agrees,
                matched_second=matched_sec,
                sf_only_match=sf_only,
            )
        )

    df = pd.DataFrame([r.__dict__ for r in rows])
    if df.empty:
        return df

    # Positions where the player had no real decision to make carry no signal
    # about their strength, and including them inflates every match rate.
    df["usable"] = (
        (df["is_forced"] == 0)
        & (df["is_decided"] == 0)
        & df["delta_cp"].notna()
        & (df["legal_moves"] > 1)
    ).astype(int)

    # Book moves: the opening is memorised, not calculated. Matching Stockfish
    # on move 3 of the Italian says nothing at all.
    if book_plies is not None:
        df["in_book"] = df["ply"].isin(book_plies).astype(int)
    else:
        df["in_book"] = (df["move_number"] <= 8).astype(int)
    df["scoreable"] = (df["usable"] == 1) & (df["in_book"] == 0)
    df["scoreable"] = df["scoreable"].astype(int)

    return df


def frame_for(df: pd.DataFrame, color: Color, scoreable_only: bool = True) -> pd.DataFrame:
    """Slice the frame to one player's moves."""
    if df.empty:
        return df
    out = df[df["color"] == color]
    if scoreable_only:
        out = out[out["scoreable"] == 1]
    return out.reset_index(drop=True)


def summarise(df: pd.DataFrame, color: Color) -> dict:
    """Headline descriptive statistics for one player (no inference here)."""
    sub = frame_for(df, color)
    allsub = df[df["color"] == color] if not df.empty else df
    if sub.empty:
        return {
            "n_scoreable": 0,
            "n_moves": int(len(allsub)),
            "t1": None,
            "t2": None,
            "t3": None,
            "acpl": None,
            "median_cpl": None,
            "mean_complexity": None,
            "blunders": 0,
            "mistakes": 0,
            "inaccuracies": 0,
        }
    delta = sub["delta_cp"].astype(float)
    return {
        "n_scoreable": int(len(sub)),
        "n_moves": int(len(allsub)),
        "t1": float(sub["matched_t1"].mean()),
        "t2": float(sub["matched_t2"].mean()),
        "t3": float(sub["matched_t3"].mean()),
        "acpl": float(delta.mean()),
        "median_cpl": float(delta.median()),
        "mean_complexity": float(sub["complexity"].mean()),
        "blunders": int((delta >= 200).sum()),
        "mistakes": int(((delta >= 100) & (delta < 200)).sum()),
        "inaccuracies": int(((delta >= 50) & (delta < 100)).sum()),
    }
