"""Chess Forensics -- HTTP backend for the browser front end.

Replaces the Streamlit app. The split is clean because every statistical module
is already a plain library and ``Report`` already serialises to JSON, so this
file only has to do three things: expose game import, run an analysis, and
stream its progress.

Why a job queue rather than a plain blocking endpoint: a full adaptive analysis
is minutes of engine time, which no browser will wait on without feedback. So
``POST /api/analyse`` returns immediately with a job id, the work proceeds on a
worker thread, and the page follows along over server-sent events. That also
makes the progress reporting genuinely live, which Streamlit's rerun model
could not do.

Run with: ``python server.py``
"""

from __future__ import annotations

import json
import math
import os
import queue
import sys
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from chessforensics import ingest, score  # noqa: E402
from chessforensics.analyse import analyse  # noqa: E402
from chessforensics.baselines import load_baseline  # noqa: E402
from chessforensics.engine import (  # noqa: E402
    AnalysisCache,
    EnginePool,
    SecondEngineConfig,
)
from chessforensics.scan import scan_player  # noqa: E402

WEB = ROOT / "web"

app = FastAPI(title="Chess Forensics")

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# second engine (Lc0) setup
# ---------------------------------------------------------------------------

def _load_second_engine_config() -> SecondEngineConfig:
    cfg_path = ROOT / "config.json"
    if cfg_path.exists():
        try:
            raw = json.loads(cfg_path.read_text()).get("second_engine")
            return SecondEngineConfig.from_dict(raw)
        except (json.JSONDecodeError, KeyError):
            pass
    return SecondEngineConfig()


def _make_second_engine() -> tuple["EnginePool | None", "AnalysisCache | None", int | None]:
    """Return (pool, cache, nodes) for the second engine, or (None, None, None)."""
    cfg = _load_second_engine_config()
    if not cfg.enabled:
        return None, None, None
    try:
        pool = EnginePool.for_lc0(cfg)
        cache = AnalysisCache(ROOT / "cache" / f"analysis_{cfg.tag}.sqlite")
        return pool, cache, cfg.nodes
    except RuntimeError:
        return None, None, None


# --------------------------------------------------------------------------
# payload
# --------------------------------------------------------------------------


def jsonable(obj: Any) -> Any:
    """Coerce a report into types a JSON encoder and a browser both accept.

    Two distinct hazards, both of which reach the page as a blank screen rather
    than as an error, so they are dealt with here once instead of per endpoint:

    * The statistical modules compute in numpy, so the report is littered with
      ``numpy.bool_`` and ``numpy.float64``. Pydantic refuses to serialise them
      outright. Stringifying them would be worse than refusing: ``"False"`` is
      truthy in JavaScript, which would silently invert every boolean flag.
    * Non-finite floats are a legitimate result here -- an unfittable slope, a
      z-score with zero variance -- and ``NaN``/``Infinity`` are not valid JSON.
      They become ``null``, which the front end already renders as "not
      measured" rather than as zero.
    """
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, (int, str)) or obj is None:
        return obj
    # numpy scalars, and anything else exposing the same unwrapping protocol.
    item = getattr(obj, "item", None)
    if callable(item) and getattr(obj, "ndim", None) == 0:
        return jsonable(item())
    if hasattr(obj, "tolist"):
        return jsonable(obj.tolist())
    return str(obj)


def report_payload(report, game) -> dict:
    """Flatten a ``Report`` plus its source game into one JSON object.

    ``moves`` carries ``fen_before`` because the board replay reconstructs
    positions client-side; without it the browser would have to replay the move
    list itself and could drift out of step with the server's ply numbering.
    """
    # Compute final FEN so the client can display the position after the last move
    import chess
    final_fen = None
    if game.moves:
        last = game.moves[-1]
        try:
            board = chess.Board(last.fen_before)
            board.push(chess.Move.from_uci(last.uci))
            final_fen = board.fen()
        except Exception:
            pass

    return jsonable({
        "game_info": report.game_info,
        "frame_records": report.frame_records,
        "players": {k: v.to_dict() for k, v in report.players.items()},
        "baseline_description": report.baseline_description,
        "uncalibrated": report.uncalibrated,
        "provisional": report.provisional,
        "critical": {str(k): v for k, v in report.critical.items()},
        "eval_curve": report.eval_curve,
        "insights": report.insights,
        "engine_settings": report.engine_settings,
        "final_fen": final_fen,
        "moves": [
            {
                "ply": m.ply,
                "move_number": m.move_number,
                "color": m.color,
                "san": m.san,
                "uci": m.uci,
                "fen_before": m.fen_before,
            }
            for m in game.moves
        ],
    })


# --------------------------------------------------------------------------
# jobs
# --------------------------------------------------------------------------


@dataclass
class Job:
    id: str
    state: str = "running"  # running | done | error
    done: int = 0
    total: int = 0
    label: str = "starting engines"
    report: dict | None = None
    error: str | None = None
    trace: str | None = None
    # Progress is consumed by an SSE stream rather than polled, so each update
    # is pushed onto a queue. ``maxsize`` is unbounded on purpose: dropping
    # updates would make the bar jump backwards.
    cancelled: bool = False
    events: queue.Queue = field(default_factory=queue.Queue)


JOBS: dict[str, Job] = {}
JOBS_LOCK = threading.Lock()
# One analysis at a time. The engine pool already saturates every core, so
# running two jobs concurrently would make both slower and oversubscribe the
# CPU; it would also corrupt nothing but would thrash the shared SQLite cache.
RUN_LOCK = threading.Lock()


class AnalyseRequest(BaseModel):
    pgn: str
    game_index: int = 0
    prior: float = Field(default=score.DEFAULT_PRIOR, gt=0.0, le=0.5)
    deep_pass: bool = True
    pool_size: int = Field(default=10, ge=1, le=20)


def _work(job: Job, req: AnalyseRequest) -> None:
    try:
        games = ingest.games_from_pgn_text(req.pgn)
        if not games:
            raise ValueError("no parsable games in that PGN")
        game = games[min(max(req.game_index, 0), len(games) - 1)]

        def on_progress(done: int, total: int, label: str) -> None:
            job.done, job.total, job.label = done, total, label
            job.events.put({"done": done, "total": total, "label": label})

        sec_pool, sec_cache, sec_nodes = _make_second_engine()
        try:
            with RUN_LOCK:
                report = analyse(
                    game,
                    prior=req.prior,
                    progress=on_progress,
                    deep_pass=req.deep_pass,
                    pool_size=req.pool_size,
                    second_pool=sec_pool,
                    second_cache=sec_cache,
                    second_nodes=sec_nodes,
                )
        finally:
            if sec_pool is not None:
                sec_pool.close()
        job.report = report_payload(report, game)
        job.state = "done"
        job.events.put({"state": "done"})
    except Exception as exc:  # surfaced to the page, not swallowed
        job.state = "error"
        job.error = str(exc) or exc.__class__.__name__
        job.trace = traceback.format_exc()
        job.events.put({"state": "error", "error": job.error})


@app.post("/api/analyse")
def start_analysis(req: AnalyseRequest) -> dict:
    job = Job(id=uuid.uuid4().hex[:12])
    with JOBS_LOCK:
        JOBS[job.id] = job
    threading.Thread(target=_work, args=(job, req), daemon=True).start()
    return {"job_id": job.id}


def _get(job_id: str) -> Job:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job")
    return job


@app.get("/api/jobs/{job_id}/events")
def job_events(job_id: str) -> StreamingResponse:
    job = _get(job_id)

    def stream() -> Iterator[str]:
        # Replay current state first so a page that connects late, or
        # reconnects, is not stuck showing zero progress.
        yield _sse({"done": job.done, "total": job.total, "label": job.label})
        if job.state != "running":
            yield _sse({"state": job.state, "error": job.error})
            return
        while True:
            try:
                msg = job.events.get(timeout=20.0)
            except queue.Empty:
                yield ": keepalive\n\n"
                continue
            yield _sse(msg)
            if msg.get("state") in ("done", "error"):
                return

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(obj: Any) -> str:
    return f"data: {json.dumps(obj)}\n\n"


@app.get("/api/jobs/{job_id}")
def job_result(job_id: str) -> dict:
    job = _get(job_id)
    if job.state == "error":
        return {"state": "error", "error": job.error, "trace": job.trace}
    if job.state != "done":
        return {"state": job.state, "done": job.done, "total": job.total,
                "label": job.label}
    return {"state": "done", "report": job.report}


@app.get("/api/jobs/{job_id}/frame.csv")
def job_frame_csv(job_id: str) -> PlainTextResponse:
    import pandas as pd

    job = _get(job_id)
    if job.state != "done" or job.report is None:
        raise HTTPException(status_code=409, detail="analysis not finished")
    csv = pd.DataFrame(job.report["frame_records"]).to_csv(index=False)
    return PlainTextResponse(
        csv,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="frame_{job_id}.csv"'},
    )


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict:
    job = _get(job_id)
    job.cancelled = True
    job.events.put({"state": "cancelled"})
    return {"status": "cancelled"}


@app.get("/api/jobs/{job_id}/report.json")
def job_report_json(job_id: str) -> PlainTextResponse:
    job = _get(job_id)
    if job.state != "done" or job.report is None:
        raise HTTPException(status_code=409, detail="analysis not finished")
    return PlainTextResponse(
        json.dumps(job.report, indent=2, default=str),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="report_{job_id}.json"'},
    )


# --------------------------------------------------------------------------
# scan (multi-game)
# --------------------------------------------------------------------------


class ScanRequest(BaseModel):
    mode: str  # chesscom | lichess
    username: str
    count: int = Field(default=20, ge=5, le=500)
    prior: float = Field(default=score.DEFAULT_PRIOR, gt=0.0, le=0.5)
    pool_size: int = Field(default=10, ge=1, le=20)
    speed: str = ""  # "" = all, or rapid/blitz/bullet/classical
    quick: bool = False  # skip L0 pre-pass for speed


def _scan_work(job: Job, req: ScanRequest) -> None:
    try:
        if req.mode == "chesscom":
            # Over-fetch when filtering by speed, but cap to avoid very slow fetches
            fetch_n = min(req.count * 3, req.count + 200) if req.speed else req.count
            games = ingest.chesscom_recent(req.username, max_games=fetch_n)
            if req.speed:
                speeds = set(req.speed.split(","))
                games = [g for g in games if g.speed in speeds][:req.count]
        elif req.mode == "lichess":
            perf = req.speed if req.speed else None
            games = list(
                ingest.lichess_user_games(
                    req.username, max_games=req.count, perf=perf
                )
            )
        else:
            raise ValueError(f"unknown mode {req.mode!r}")

        if not games:
            raise ValueError(f"no games found for {req.username!r}")

        # APIs return newest-first; reverse so game 1 = oldest (chronological order)
        games = list(reversed(games))

        def on_progress(
            game_done: int,
            game_total: int,
            move_done: int,
            move_total: int,
            label: str,
        ) -> None:
            job.done = game_done
            job.total = game_total
            job.label = label
            job.events.put({
                "game_done": game_done,
                "game_total": game_total,
                "move_done": move_done,
                "move_total": move_total,
                "label": label,
            })

        # Default to the corpus's own deep_pass setting so scans use the
        # same instrument as the reference. "quick" mode explicitly opts out.
        from chessforensics.baselines import load_baseline as _lb
        bl = _lb()
        use_deep = bl.data.get("provenance", {}).get("deep_pass", True)
        if req.quick:
            use_deep = False

        with RUN_LOCK:
            result = scan_player(
                games,
                username=req.username,
                prior=req.prior,
                pool_size=req.pool_size,
                deep_pass=use_deep,
                progress=on_progress,
                cancel_fn=lambda: job.cancelled,
                run_l0=not req.quick,
            )

        job.report = jsonable(_scan_payload(result))
        job.state = "done"
        job.events.put({"state": "done"})
    except Exception as exc:
        job.state = "error"
        job.error = str(exc) or exc.__class__.__name__
        job.trace = traceback.format_exc()
        job.events.put({"state": "error", "error": job.error})


def _scan_payload(result) -> dict:
    """Serialise a ScanResult for the frontend."""
    return {
        "type": "scan",
        "username": result.username,
        "source": result.source,
        "n_games": result.n_games,
        "n_games_scored": result.n_games_scored,
        "total_positions": result.total_positions,
        "total_log_lr": result.total_log_lr,
        "total_decibans": result.total_decibans,
        "posterior": result.posterior,
        "posterior_low": result.posterior_low,
        "posterior_high": result.posterior_high,
        "prior": result.prior,
        "interpretation": result.interpretation,
        "convergence": result.convergence,
        "signal_summary": result.signal_summary,
        "caveats": result.caveats,
        "temporal": result.temporal.__dict__ if result.temporal else None,
        "evidence_variance": result.evidence_variance,
        "flagged_games": result.flagged_games,
        "peak_game_db": result.peak_game_db,
        "peak_game_posterior": result.peak_game_posterior,
        "outlier_games": result.outlier_games,
        "speed_breakdown": result.speed_breakdown,
        "ipr_trajectory": result.ipr_trajectory,
        "games": [
            {
                "game_index": g.game_index,
                "game_info": g.game_info,
                "color": g.color,
                "log_lr": g.log_lr,
                "decibans": g.decibans,
                "posterior_cumulative": g.posterior_cumulative,
                "individual_posterior": g.individual_posterior,
                "n_scoreable": g.n_scoreable,
                "t1_rate": g.t1_rate,
                "acpl": g.acpl,
                "slope_beta1": g.slope_beta1,
                "verdict_interpretation": g.verdict_interpretation,
                "insufficient": g.insufficient,
                "speed": g.speed,
                "ipr_elo": g.ipr_elo,
                "switchpoint_detected": g.switchpoint_detected,
                "raw_pgn": g.raw_pgn,
            }
            for g in result.games
        ],
    }


@app.post("/api/scan")
def start_scan(req: ScanRequest) -> dict:
    job = Job(id=uuid.uuid4().hex[:12])
    with JOBS_LOCK:
        JOBS[job.id] = job
    threading.Thread(target=_scan_work, args=(job, req), daemon=True).start()
    return {"job_id": job.id}


# --------------------------------------------------------------------------
# import
# --------------------------------------------------------------------------


class ImportRequest(BaseModel):
    mode: str  # url | chesscom | lichess
    value: str
    count: int = Field(default=10, ge=1, le=30)
    speed: str = ""  # "" = all, or rapid/blitz/bullet/classical


def _label(g) -> str:
    w, b = g.players["white"], g.players["black"]
    return (
        f"{w.name} ({w.elo or '?'}) vs {b.name} ({b.elo or '?'})"
        f" - {g.result or '?'} [{g.speed or '?'}]"
    )


@app.post("/api/import")
def import_games(req: ImportRequest) -> dict:
    try:
        if req.mode == "url":
            games = [ingest.from_url(req.value)]
        elif req.mode == "chesscom":
            # chess.com doesn't support server-side speed filter; fetch more and filter
            fetch_n = req.count * 3 if req.speed else req.count
            games = ingest.chesscom_recent(req.value, max_games=fetch_n)
            if req.speed:
                speeds = set(req.speed.split(","))
                games = [g for g in games if g.speed in speeds][:req.count]
        elif req.mode == "lichess":
            perf = req.speed if req.speed else None
            games = list(
                ingest.lichess_user_games(
                    req.value, max_games=req.count, perf=perf
                )
            )
        else:
            raise HTTPException(status_code=400, detail=f"unknown mode {req.mode!r}")
    except HTTPException:
        raise
    except Exception as exc:
        # Import failures are overwhelmingly the remote API's doing (rate
        # limits, missing user, private game), so report the message rather
        # than a 500 with no explanation.
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    out = []
    for g in games:
        if not g.raw_pgn:
            continue
        out.append({"label": _label(g), "pgn": g.raw_pgn})
    if not out:
        raise HTTPException(
            status_code=502,
            detail="those games carried no PGN text, so there is nothing to analyse",
        )
    return {"games": out}


class CountRequest(BaseModel):
    pgn: str


@app.post("/api/pgn/inspect")
def inspect_pgn(req: CountRequest) -> dict:
    try:
        games = ingest.games_from_pgn_text(req.pgn)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"could not parse that PGN: {exc}")
    if not games:
        raise HTTPException(status_code=400, detail="no parsable games in that PGN")
    return {"count": len(games), "labels": [_label(g) for g in games]}


# --------------------------------------------------------------------------
# reference model
# --------------------------------------------------------------------------


@app.get("/api/corpus-status")
def corpus_status() -> dict:
    """Live corpus stats read directly from the SQLite database."""
    import sqlite3 as _sql
    corpus_path = ROOT / "cache" / "corpus.sqlite"
    cache_path = ROOT / "cache" / "analysis.sqlite"
    out: dict = {"games": 0, "positions": 0, "player_games": 0, "bands": {}, "cache_positions": 0}
    try:
        conn = _sql.connect(str(corpus_path), timeout=5)
        out["games"] = conn.execute("SELECT COUNT(*) FROM games WHERE status='ok'").fetchone()[0]
        out["positions"] = conn.execute("SELECT COUNT(*) FROM moves").fetchone()[0]
        out["player_games"] = conn.execute("SELECT COUNT(*) FROM player_games").fetchone()[0]
        for band, n in conn.execute("SELECT band, COUNT(*) FROM player_games GROUP BY band"):
            out["bands"][band] = n
        conn.close()
    except Exception:
        pass
    try:
        conn2 = _sql.connect(str(cache_path), timeout=5)
        out["cache_positions"] = conn2.execute("SELECT COUNT(*) FROM analysis").fetchone()[0]
        conn2.close()
    except Exception:
        pass
    # Lite corpus stats
    try:
        conn = _sql.connect(str(corpus_path), timeout=5)
        out["lite_games"] = conn.execute("SELECT COUNT(*) FROM games_lite").fetchone()[0]
        out["lite_player_games"] = conn.execute("SELECT COUNT(*) FROM player_games_lite").fetchone()[0]
        lite_bands = {}
        for band, n in conn.execute("SELECT band, COUNT(*) FROM player_games_lite GROUP BY band"):
            lite_bands[band] = n
        out["lite_bands"] = lite_bands
        conn.close()
    except Exception:
        out["lite_games"] = 0
        out["lite_player_games"] = 0
        out["lite_bands"] = {}
    return out


@app.get("/api/baseline")
def baseline() -> dict:
    b = load_baseline()
    prov = b.provenance
    return {
        "description": b.describe(),
        "games": b.n_games,
        "player_games": int(prov.get("player_games", 0)),
        "positions": int(prov.get("positions", 0)),
        "bands_fitted": len(b.bands),
        "is_default": b.is_default,
        "provisional": b.provisional,
        "trustworthy": b.trustworthy,
        "multipv": b.multipv,
        "default_prior": score.DEFAULT_PRIOR,
    }


# --------------------------------------------------------------------------
# case studies
# --------------------------------------------------------------------------

_CASE_STUDIES = [
    {
        "id": "feller",
        "type": "flagged",
        "title": "The Feller Relay Scheme",
        "subtitle": "Chess Olympiad, Khanty-Mansiysk 2010",
        "narrative": (
            "French IM S\u00e9bastien Feller conspired with Cyril Marzolo "
            "and team captain Arnaud Hauchard during the 2010 Chess Olympiad. "
            "Marzolo followed the games with an engine and texted moves to "
            "Hauchard, who signalled the recommended moves to Feller during play. "
            "The scheme was uncovered through analysis of game records and "
            "communication evidence."
        ),
        "method": (
            "SMS relay: Marzolo ran an engine and texted moves to Hauchard "
            "(team captain), who signalled them to Feller at the board"
        ),
        "outcome": (
            "FIDE Ethics Commission (1 July 2012): Hauchard banned 3 years, "
            "Feller 2 years 9 months, Marzolo 1 year 6 months (9 months suspended)."
        ),
        "highlight_index": 2,
        # Stats are computed dynamically from the analysis (see _load_case_stats)
        "stats": [],
        "pgn_file": "feller_olympiad_2010.pgn",
    },
    {
        "id": "nigalidze",
        "type": "flagged",
        "title": "The Bathroom Phone",
        "subtitle": "Dubai Open 2015",
        "narrative": (
            "Georgian GM Gaioz Nigalidze was caught during Round 6 of the "
            "2015 Dubai Open. His opponent, GM Tigran L. Petrosian, noticed "
            "Nigalidze\u2019s frequent and suspiciously regular bathroom "
            "visits \u2014 always to the same stall. Petrosian alerted the "
            "chief arbiter, who found a smartphone hidden inside a toilet "
            "paper holder with a chess engine application open, analysing "
            "the current game position."
        ),
        "method": (
            "Smartphone hidden in bathroom stall with a chess analysis app, "
            "consulted during frequent bathroom visits"
        ),
        "outcome": (
            "Expelled from the tournament on the spot. FIDE banned him for "
            "3 years and stripped his GM title (IM title retained)."
        ),
        "highlight_index": 0,
        "stats": [],
        "pgn_file": "nigalidze_dubai_2015.pgn",
    },
    {
        "id": "niemann",
        "type": "clean",
        "title": "The Niemann-Carlsen Case",
        "subtitle": "Sinquefield Cup, Saint Louis 2022",
        "narrative": (
            "Hans Niemann defeated Magnus Carlsen with the Black pieces in "
            "Round 3 of the 2022 Sinquefield Cup, prompting Carlsen to "
            "withdraw from the tournament. The incident sparked the biggest "
            "cheating controversy in modern chess history. However, "
            "statistical analysis of this game showed Niemann\u2019s play "
            "was consistent with his rating level \u2014 engine correlation, "
            "ACPL, and move complexity all fell within normal ranges for a "
            "2700-level player. This case illustrates why single-game "
            "analysis alone is rarely sufficient to draw conclusions."
        ),
        "method": (
            "Alleged over-the-board engine assistance (never proven for "
            "this specific game)"
        ),
        "outcome": (
            "FIDE (13 December 2023) found no evidence of over-the-board "
            "cheating. Carlsen was fined \u20ac10,000 for withdrawing. "
            "Carlsen and Niemann later settled their legal dispute."
        ),
        "highlight_index": 0,
        "stats": [],
        "pgn_file": "niemann_sinquefield_2022.pgn",
    },
]


@app.get("/api/case-studies")
def get_case_studies():
    """Parsed case study games for the showcase section."""
    import chess
    import chess.pgn
    import io as _io

    result = []
    for meta in _CASE_STUDIES:
        entry = {k: v for k, v in meta.items() if k != "pgn_file"}
        pgn_path = ROOT / "tests" / meta["pgn_file"]
        games = []
        raw_pgn = ""
        if pgn_path.exists():
            raw_pgn = pgn_path.read_text(encoding="utf-8")
            reader = _io.StringIO(raw_pgn)
            while True:
                g = chess.pgn.read_game(reader)
                if g is None:
                    break
                moves = []
                board = g.board()
                for node in g.mainline():
                    fen_before = board.fen()
                    san = board.san(node.move)
                    uci = node.move.uci()
                    board.push(node.move)
                    moves.append({
                        "ply": len(moves),
                        "san": san,
                        "uci": uci,
                        "fen_before": fen_before,
                    })
                games.append({
                    "white": g.headers.get("White", "?"),
                    "black": g.headers.get("Black", "?"),
                    "white_elo": g.headers.get("WhiteElo", ""),
                    "black_elo": g.headers.get("BlackElo", ""),
                    "round": g.headers.get("Round", ""),
                    "result": g.headers.get("Result", ""),
                    "eco": g.headers.get("ECO", ""),
                    "final_fen": board.fen(),
                    "moves": moves,
                })
        entry["games"] = games
        entry["raw_pgn"] = raw_pgn
        result.append(entry)

    return {"cases": result}


# --------------------------------------------------------------------------
# static
# --------------------------------------------------------------------------


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB / "index.html")


app.mount("/static", StaticFiles(directory=WEB), name="static")


def main() -> None:
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", 8000)))
    args = parser.parse_args()

    print(f"Chess Forensics -- http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
