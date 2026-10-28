"""Stockfish worker pool with an adaptive two-pass schedule and a SQLite cache.

Design notes
------------
*Many single-threaded engines, not one many-threaded engine.* Lazy SMP scales
sublinearly -- a 20-thread search is nowhere near 20x a 1-thread search -- but
positions are completely independent, so running N=10 one-thread instances in
parallel gets close to linear throughput on bulk work.

*Two passes.* A depth-12 MultiPV-3 sweep over every ply is cheap and tells us
where the interesting decisions were. Only flagged positions get the expensive
depth-22 MultiPV-5 treatment. Spending deep search on a forced recapture is
wasted compute.

*The cache is load-bearing.* Baseline building re-analyses an enormous number of
repeated opening and transposed positions, and UI reruns must be instant. Keyed
on (fen, depth, multipv) so a deeper or wider result never masks a shallower
request and vice versa.
"""

from __future__ import annotations

import json
import os
import queue
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import chess
import chess.engine

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "cache"
CACHE_PATH = CACHE_DIR / "analysis.sqlite"

# Pass parameters. Kept as module constants so the baseline builder and the app
# agree on cache keys without passing them around everywhere.
PASS1_DEPTH = 12
# Both passes use the same MultiPV width deliberately. The choice model's
# candidate window determines how much probability mass falls outside it, which
# biases the fitted sensitivity (see model._extrapolate_tail). That bias only
# cancels against the baseline if every position -- shallow, deep, subject and
# corpus -- is measured through the same width. A narrower pass 1 would make the
# effective window depend on whether a position happened to be promoted, which
# is not something the statistics can correct for after the fact.
PASS1_MULTIPV = 5
PASS2_DEPTH = 22
PASS2_MULTIPV = 5

# Shallow pass for "obvious move" detection. Depth 4 is near-instant and
# establishes what any basic search would find. Used to separate trivial
# agreement from meaningful engine correlation.
PASS0_DEPTH = 4
PASS0_MULTIPV = 3

MATE_SCORE = 10000  # centipawns used to stand in for mate when clamping


# ---------------------------------------------------------------------------
# second engine (Lc0) configuration
# ---------------------------------------------------------------------------


@dataclass
class SecondEngineConfig:
    """Configuration for a second analysis engine (typically Lc0).

    Lc0 uses nodes rather than depth, so ``nodes`` is the search budget.
    The ``backend`` controls GPU vs CPU: "cpu" works everywhere but is slower;
    "cuda" or "dx12" use the GPU when available.
    """

    path: str = ""
    weights: str = ""
    tag: str = "lc0"
    nodes: int = 1200
    multipv: int = 5
    backend: str = "cpu"
    threads: int = 1
    cache_size: int = 200000
    enabled: bool = False

    @classmethod
    def from_dict(cls, d: dict | None) -> "SecondEngineConfig":
        if not d:
            return cls()
        return cls(
            path=str(d.get("path", "")),
            weights=str(d.get("weights", "")),
            tag=str(d.get("tag", "lc0")),
            nodes=int(d.get("nodes", 1200)),
            multipv=int(d.get("multipv", 5)),
            backend=str(d.get("backend", "cpu")),
            threads=int(d.get("threads", 1)),
            cache_size=int(d.get("cache_size", 200000)),
            enabled=bool(d.get("enabled", False)),
        )

    def resolve(self) -> str | None:
        """Return a usable engine path, or None if not configured."""
        import shutil

        if self.path and Path(self.path).exists():
            return self.path
        env = os.environ.get("LC0_PATH")
        if env and Path(env).exists():
            return env
        engines = ROOT / "engines"
        if engines.exists():
            for name in ("lc0", "lc0.exe"):
                cands = sorted(engines.rglob(name), key=lambda p: len(p.parts))
                if cands:
                    return str(cands[0])
        found = shutil.which("lc0")
        return found

    def resolve_weights(self) -> str | None:
        """Return the weights file path, or None."""
        if self.weights and Path(self.weights).exists():
            return self.weights
        engines = ROOT / "engines"
        if engines.exists():
            for pat in ("*.pb.gz", "*.pb", "*.onnx"):
                hits = sorted(engines.rglob(pat), key=lambda p: -p.stat().st_size)
                if hits:
                    return str(hits[0])
        return None


@dataclass
class Candidate:
    """One MultiPV line, from the point of view of the side to move."""

    uci: str
    cp: float  # centipawns, positive = good for side to move
    rank: int  # 1 = engine's first choice
    mate: int | None = None
    pv: list[str] = field(default_factory=list)

    @property
    def is_mate(self) -> bool:
        return self.mate is not None


@dataclass
class PositionAnalysis:
    fen: str
    depth: int
    multipv: int
    candidates: list[Candidate]
    legal_moves: int
    from_cache: bool = False

    @property
    def best(self) -> Candidate | None:
        return self.candidates[0] if self.candidates else None

    @property
    def best_cp(self) -> float:
        b = self.best
        return b.cp if b else 0.0

    def gap(self) -> float | None:
        """top1 - top2 in centipawns, or None if only one line is known."""
        if len(self.candidates) < 2:
            return None
        return self.candidates[0].cp - self.candidates[1].cp

    def rank_of(self, uci: str) -> int | None:
        for c in self.candidates:
            if c.uci == uci:
                return c.rank
        return None

    def cp_of(self, uci: str) -> float | None:
        for c in self.candidates:
            if c.uci == uci:
                return c.cp
        return None

    def deltas(self) -> list[float]:
        """Eval loss of each candidate relative to the best, in centipawns."""
        if not self.candidates:
            return []
        top = self.candidates[0].cp
        return [top - c.cp for c in self.candidates]


def _clamp(cp: float) -> float:
    """Squash absurd evals.

    Once a position is +6 pawns it does not matter whether the engine says +6 or
    +60: the decision quality signal saturates. Leaving mate scores unclamped
    would let one won position dominate every average we compute.
    """
    return max(-MATE_SCORE, min(MATE_SCORE, cp))


# --------------------------------------------------------------------------
# cache
# --------------------------------------------------------------------------


class AnalysisCache:
    """Thread-safe SQLite cache of (fen, depth, multipv) -> candidate lines."""

    def __init__(self, path: Path | str = CACHE_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._write_lock = threading.Lock()
        self._init_schema()
        self.hits = 0
        self.misses = 0

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(str(self.path), timeout=60, isolation_level=None)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn = conn
        return conn

    def _init_schema(self) -> None:
        conn = self._conn()
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS analysis (
                fen      TEXT NOT NULL,
                depth    INTEGER NOT NULL,
                multipv  INTEGER NOT NULL,
                legal    INTEGER NOT NULL,
                lines    TEXT NOT NULL,
                engine   TEXT,
                created  REAL,
                PRIMARY KEY (fen, depth, multipv)
            );
            CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
            """
        )

    def get(self, fen: str, depth: int, multipv: int) -> PositionAnalysis | None:
        row = self._conn().execute(
            "SELECT legal, lines FROM analysis WHERE fen=? AND depth=? AND multipv=?",
            (fen, depth, multipv),
        ).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        legal, blob = row
        cands = [
            Candidate(
                uci=d["uci"], cp=d["cp"], rank=d["rank"], mate=d.get("mate"), pv=d.get("pv", [])
            )
            for d in json.loads(blob)
        ]
        return PositionAnalysis(fen, depth, multipv, cands, legal, from_cache=True)

    def get_many(
        self, keys: Sequence[tuple[str, int, int]]
    ) -> dict[tuple[str, int, int], PositionAnalysis]:
        """Batched lookup; one query per 500 keys instead of one per key."""
        out: dict[tuple[str, int, int], PositionAnalysis] = {}
        conn = self._conn()
        for start in range(0, len(keys), 500):
            chunk = keys[start : start + 500]
            placeholders = ",".join(["(?,?,?)"] * len(chunk))
            flat: list = []
            for fen, depth, mpv in chunk:
                flat.extend((fen, depth, mpv))
            rows = conn.execute(
                "SELECT fen, depth, multipv, legal, lines FROM analysis "
                f"WHERE (fen, depth, multipv) IN ({placeholders})",
                flat,
            ).fetchall()
            for fen, depth, mpv, legal, blob in rows:
                cands = [
                    Candidate(
                        uci=d["uci"],
                        cp=d["cp"],
                        rank=d["rank"],
                        mate=d.get("mate"),
                        pv=d.get("pv", []),
                    )
                    for d in json.loads(blob)
                ]
                out[(fen, depth, mpv)] = PositionAnalysis(
                    fen, depth, mpv, cands, legal, from_cache=True
                )
        self.hits += len(out)
        self.misses += len(keys) - len(out)
        return out

    def put(self, pa: PositionAnalysis, engine_id: str = "") -> None:
        blob = json.dumps(
            [
                {"uci": c.uci, "cp": c.cp, "rank": c.rank, "mate": c.mate, "pv": c.pv[:8]}
                for c in pa.candidates
            ],
            separators=(",", ":"),
        )
        with self._write_lock:
            self._conn().execute(
                "INSERT OR REPLACE INTO analysis "
                "(fen, depth, multipv, legal, lines, engine, created) VALUES (?,?,?,?,?,?,?)",
                (pa.fen, pa.depth, pa.multipv, pa.legal_moves, blob, engine_id, time.time()),
            )

    def put_many(self, items: Iterable[PositionAnalysis], engine_id: str = "") -> None:
        rows = []
        for pa in items:
            blob = json.dumps(
                [
                    {"uci": c.uci, "cp": c.cp, "rank": c.rank, "mate": c.mate, "pv": c.pv[:8]}
                    for c in pa.candidates
                ],
                separators=(",", ":"),
            )
            rows.append(
                (pa.fen, pa.depth, pa.multipv, pa.legal_moves, blob, engine_id, time.time())
            )
        if not rows:
            return
        with self._write_lock:
            conn = self._conn()
            conn.execute("BEGIN")
            conn.executemany(
                "INSERT OR REPLACE INTO analysis "
                "(fen, depth, multipv, legal, lines, engine, created) VALUES (?,?,?,?,?,?,?)",
                rows,
            )
            conn.execute("COMMIT")

    def stats(self) -> dict:
        n = self._conn().execute("SELECT COUNT(*) FROM analysis").fetchone()[0]
        size = self.path.stat().st_size if self.path.exists() else 0
        return {
            "positions": n,
            "size_mb": round(size / 1e6, 1),
            "hits": self.hits,
            "misses": self.misses,
        }


# --------------------------------------------------------------------------
# engine pool
# --------------------------------------------------------------------------


def resolve_engine_path() -> str:
    """Locate Stockfish via config.json, env, bundled engines/, or PATH."""
    import shutil

    cfg_path = ROOT / "config.json"
    if cfg_path.exists():
        try:
            path = json.loads(cfg_path.read_text()).get("engine_path")
            if path and Path(path).exists():
                return path
        except json.JSONDecodeError:
            pass
    env = os.environ.get("STOCKFISH_PATH")
    if env and Path(env).exists():
        return env
    engines = ROOT / "engines"
    if engines.exists():
        cands = sorted(
            (p for p in engines.rglob("stockfish*") if p.is_file() and p.suffix.lower() in {".exe", ""}),
            key=lambda p: len(p.parts),
        )
        if cands:
            return str(cands[0])
    found = shutil.which("stockfish")
    if found:
        return found
    raise RuntimeError("Stockfish not found. Run: python setup_engine.py")


class EnginePool:
    """A fixed pool of single-threaded Stockfish processes.

    Use as a context manager. Engines are created lazily on first checkout and
    reused; a crashed engine is replaced transparently on the next checkout.
    """

    def __init__(
        self,
        size: int | None = None,
        engine_path: str | None = None,
        hash_mb: int = 256,
        threads: int = 1,
        tag: str = "stockfish",
        extra_uci: dict | None = None,
    ):
        cores = os.cpu_count() or 4
        # Leave headroom so the UI and OS stay responsive during long runs.
        self.size = size if size is not None else max(1, min(10, cores - 2))
        self.engine_path = engine_path or resolve_engine_path()
        self.hash_mb = hash_mb
        self.threads = threads
        self.tag = tag
        self._extra_uci = extra_uci or {}
        self._pool: queue.LifoQueue = queue.LifoQueue()
        self._created = 0
        self._lock = threading.Lock()
        self._closed = False
        self.engine_id = ""

    def _spawn(self) -> chess.engine.SimpleEngine:
        eng = chess.engine.SimpleEngine.popen_uci(self.engine_path)
        cfg: dict = {"Threads": self.threads, "Hash": self.hash_mb}
        cfg.update(self._extra_uci)
        eng.configure(cfg)
        if not self.engine_id:
            self.engine_id = eng.id.get("name", self.tag)
        return eng

    def acquire(self) -> chess.engine.SimpleEngine:
        if self._closed:
            raise RuntimeError("pool is closed")
        try:
            return self._pool.get_nowait()
        except queue.Empty:
            pass
        with self._lock:
            if self._created < self.size:
                self._created += 1
                return self._spawn()
        return self._pool.get()

    def release(self, eng: chess.engine.SimpleEngine, broken: bool = False) -> None:
        if broken or self._closed:
            try:
                eng.close()
            except Exception:
                pass
            with self._lock:
                self._created -= 1
            return
        self._pool.put(eng)

    def close(self) -> None:
        self._closed = True
        while True:
            try:
                eng = self._pool.get_nowait()
            except queue.Empty:
                break
            try:
                eng.close()
            except Exception:
                pass

    def __enter__(self) -> "EnginePool":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @classmethod
    def for_lc0(cls, cfg: SecondEngineConfig, size: int | None = None) -> "EnginePool":
        """Create a pool of Lc0 processes from a SecondEngineConfig."""
        path = cfg.resolve()
        if not path:
            raise RuntimeError(
                "Lc0 not found. Place lc0(.exe) under engines/ or set LC0_PATH."
            )
        extra: dict = {}
        weights = cfg.resolve_weights()
        if weights:
            extra["WeightsFile"] = weights
        if cfg.backend:
            extra["Backend"] = cfg.backend
        if cfg.cache_size:
            extra["NNCacheSize"] = cfg.cache_size
        return cls(
            size=size or max(1, min(4, (os.cpu_count() or 4) // 2)),
            engine_path=path,
            hash_mb=0,
            threads=cfg.threads,
            tag=cfg.tag,
            extra_uci=extra,
        )

    # ---- analysis -------------------------------------------------------

    def analyse_one(
        self, fen: str, depth: int, multipv: int, nodes: int | None = None,
    ) -> PositionAnalysis:
        board = chess.Board(fen)
        legal = board.legal_moves.count()
        if legal == 0:
            return PositionAnalysis(fen, depth, multipv, [], 0)

        eng = self.acquire()
        broken = False
        try:
            if nodes is not None:
                limit = chess.engine.Limit(nodes=nodes)
            else:
                limit = chess.engine.Limit(depth=depth)
            infos = eng.analyse(
                board,
                limit,
                multipv=min(multipv, legal),
            )
        except chess.engine.EngineTerminatedError:
            broken = True
            raise
        finally:
            self.release(eng, broken)

        if isinstance(infos, dict):
            infos = [infos]
        cands: list[Candidate] = []
        for idx, info in enumerate(infos):
            pv = info.get("pv") or []
            if not pv:
                continue
            score = info["score"].relative
            mate = score.mate()
            cp = _clamp(float(score.score(mate_score=MATE_SCORE)))
            cands.append(
                Candidate(
                    uci=pv[0].uci(),
                    cp=cp,
                    rank=info.get("multipv", idx + 1),
                    mate=mate,
                    pv=[m.uci() for m in pv[:8]],
                )
            )
        cands.sort(key=lambda c: c.rank)
        return PositionAnalysis(fen, depth, multipv, cands, legal)


ProgressFn = Callable[[int, int, str], None]


def analyse_positions(
    fens: Sequence[str],
    depth: int,
    multipv: int,
    pool: EnginePool,
    cache: AnalysisCache | None = None,
    progress: ProgressFn | None = None,
    label: str = "analysing",
    nodes: int | None = None,
) -> dict[str, PositionAnalysis]:
    """Analyse many positions in parallel, consulting and filling the cache.

    Deduplicates FENs first: repetitions and transpositions are common and
    there is no reason to search the same position twice in one batch.

    When ``nodes`` is set (e.g. for Lc0), the search is bounded by node count
    rather than depth.  The cache key still uses ``depth`` -- for node-based
    engines, pass the node count as the depth parameter in the cache.
    """
    unique = list(dict.fromkeys(fens))
    results: dict[str, PositionAnalysis] = {}

    # For node-based engines the cache key uses nodes in the depth slot.
    cache_depth = nodes if nodes is not None else depth

    todo = unique
    if cache is not None:
        cached = cache.get_many([(f, cache_depth, multipv) for f in unique])
        for (fen, _, _), pa in cached.items():
            results[fen] = pa
        todo = [f for f in unique if f not in results]

    total = len(unique)
    done = total - len(todo)
    if progress:
        progress(done, total, f"{label}: {done}/{total} from cache")

    if not todo:
        return results

    from concurrent.futures import ThreadPoolExecutor, as_completed

    pending: list[PositionAnalysis] = []
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=pool.size) as ex:
        futures = {
            ex.submit(pool.analyse_one, fen, depth, multipv, nodes): fen
            for fen in todo
        }
        for fut in as_completed(futures):
            fen = futures[fut]
            try:
                pa = fut.result()
                # Normalise the stored depth for node-based engines.
                if nodes is not None:
                    pa.depth = nodes
            except Exception:
                # A single failed position must not sink the whole run; it is
                # simply absent from the feature frame downstream.
                continue
            with lock:
                results[fen] = pa
                pending.append(pa)
                done += 1
                if progress and (done % 5 == 0 or done == total):
                    progress(done, total, f"{label}: {done}/{total}")
                if cache is not None and len(pending) >= 64:
                    cache.put_many(pending, pool.engine_id)
                    pending = []

    if cache is not None and pending:
        cache.put_many(pending, pool.engine_id)
    if progress:
        progress(total, total, f"{label}: done")
    return results


# --------------------------------------------------------------------------
# adaptive two-pass schedule
# --------------------------------------------------------------------------

AMBIGUOUS_GAP = 30.0  # cp: below this the top two moves are near-equivalent
ONLY_MOVE_GAP = 150.0  # cp: above this there is effectively one move
SWING_THRESHOLD = 50.0  # cp: eval change from the previous ply
DEEP_BUDGET_FRACTION = 0.45  # cap on the share of plies promoted to pass 2

# Hard triggers always get a deep look; soft ones only compete for the budget.
# Measured on real games, "quiet move" fires on ~75% of plies and "gap < 30cp"
# on ~60% of quiet-opening plies, so treating either as a hard trigger promotes
# essentially the whole game and the two-pass schedule saves nothing.
_HARD_REASONS = frozenset({"only-move", "swing", "mate", "unstable"})


def critical_reasons(
    pa: PositionAnalysis,
    prev_best_cp: float | None,
    played_uci: str | None,
) -> list[str]:
    """Reasons this position might deserve a deep second pass.

    The reasons are kept for the UI so the compute budget is explainable rather
    than magic.
    """
    reasons: list[str] = []
    if not pa.candidates:
        return reasons

    gap = pa.gap()
    if gap is not None:
        if gap > ONLY_MOVE_GAP:
            # Only-move: the eval hinges entirely on finding it, so a shallow
            # score is untrustworthy.
            reasons.append("only-move")
        elif gap < AMBIGUOUS_GAP:
            # Genuinely ambiguous. At depth 12 the *ordering* of near-equal
            # moves is unstable, and match-rate statistics depend on which move
            # is top-1, so these are the highest-value deepening targets.
            reasons.append("ambiguous")

    if prev_best_cp is not None:
        # Evals alternate sign between plies (both are "good for side to move"),
        # so the comparable quantity is -prev_best_cp.
        if abs(pa.best_cp - (-prev_best_cp)) > SWING_THRESHOLD:
            reasons.append("swing")

    if pa.best is not None and pa.best.is_mate:
        reasons.append("mate")

    if played_uci:
        board = chess.Board(pa.fen)
        try:
            mv = chess.Move.from_uci(played_uci)
        except ValueError:
            mv = None
        if mv is not None and mv in board.legal_moves:
            if not board.is_capture(mv) and not board.gives_check(mv):
                # Quiet moves are where humans actually fail, so they carry the
                # most information -- but they are also the majority of moves,
                # hence soft.
                reasons.append("quiet")

    return reasons


def _deep_priority(pa: PositionAnalysis, reasons: Sequence[str]) -> float:
    """Higher = more worth spending a deep search on.

    Ranked by how much a deeper search is likely to *change* the numbers we
    report, not by how interesting the position is to a human.
    """
    score = 0.0
    if "only-move" in reasons:
        score += 3.0
    if "swing" in reasons:
        score += 2.5
    if "mate" in reasons:
        score += 2.0
    if "ambiguous" in reasons:
        gap = pa.gap()
        # A 2cp gap is far more order-unstable than a 28cp one.
        score += 2.0 - 1.5 * min(1.0, (gap or 0.0) / AMBIGUOUS_GAP)
    if "quiet" in reasons:
        score += 0.5
    # Wide-open positions have more scope for the ordering to shift.
    score += 0.3 * min(1.0, pa.legal_moves / 40.0)
    return score


@dataclass
class GameAnalysis:
    """Merged multi-pass analysis for one game, indexed by ply."""

    fens: list[str]
    shallow: dict[str, PositionAnalysis]
    deep: dict[str, PositionAnalysis]
    critical: dict[int, list[str]] = field(default_factory=dict)
    # L0 shallow pass (depth 4): what is "obvious" to any engine.
    l0: dict[str, PositionAnalysis] = field(default_factory=dict)
    # Second engine (e.g. Lc0) analysis, keyed by FEN.
    second: dict[str, PositionAnalysis] = field(default_factory=dict)
    second_tag: str = ""

    def at(self, ply: int) -> PositionAnalysis | None:
        """Best available analysis for a ply: deep if we have it, else shallow."""
        if ply < 0 or ply >= len(self.fens):
            return None
        fen = self.fens[ply]
        return self.deep.get(fen) or self.shallow.get(fen)

    def shallow_at(self, ply: int) -> PositionAnalysis | None:
        if ply < 0 or ply >= len(self.fens):
            return None
        return self.shallow.get(self.fens[ply])

    def l0_at(self, ply: int) -> PositionAnalysis | None:
        """Depth-4 analysis: the 'obvious move' baseline."""
        if ply < 0 or ply >= len(self.fens):
            return None
        return self.l0.get(self.fens[ply])

    def second_at(self, ply: int) -> PositionAnalysis | None:
        """Second engine's analysis for a ply."""
        if ply < 0 or ply >= len(self.fens):
            return None
        return self.second.get(self.fens[ply])

    def depth_at(self, ply: int) -> int:
        pa = self.at(ply)
        return pa.depth if pa else 0

    def __len__(self) -> int:
        return len(self.fens)


def analyse_game(
    fens: Sequence[str],
    played_ucis: Sequence[str],
    pool: EnginePool,
    cache: AnalysisCache | None = None,
    progress: ProgressFn | None = None,
    pass1_depth: int = PASS1_DEPTH,
    pass1_multipv: int = PASS1_MULTIPV,
    pass2_depth: int = PASS2_DEPTH,
    pass2_multipv: int = PASS2_MULTIPV,
    deep_pass: bool = True,
    run_l0: bool = True,
    second_pool: "EnginePool | None" = None,
    second_cache: "AnalysisCache | None" = None,
    second_nodes: int | None = None,
    second_multipv: int = PASS1_MULTIPV,
) -> GameAnalysis:
    """Run the adaptive multi-pass schedule over one game.

    When ``run_l0`` is True (default), a depth-4 pre-pass establishes which
    moves are "obvious" to any engine.  When ``second_pool`` is provided
    (e.g. an Lc0 pool), every position also gets a second-engine opinion.
    """
    # Pass 0: L0 shallow sweep (depth 4, near-instant).
    l0: dict[str, PositionAnalysis] = {}
    if run_l0:
        l0 = analyse_positions(
            fens, PASS0_DEPTH, PASS0_MULTIPV, pool, cache, progress,
            label="pass 0 (d%d)" % PASS0_DEPTH,
        )

    # Pass 1: regular shallow sweep.
    shallow = analyse_positions(
        fens, pass1_depth, pass1_multipv, pool, cache, progress,
        label="pass 1 (d%d)" % pass1_depth,
    )

    # Rank every ply, then promote hard triggers plus the best soft candidates
    # up to the budget. Without the cap, "quiet" and "ambiguous" between them
    # flag nearly every ply and pass 2 stops being a saving at all.
    scored: list[tuple[float, int, list[str]]] = []
    prev_best: float | None = None
    for ply, fen in enumerate(fens):
        pa = shallow.get(fen)
        if pa is None:
            prev_best = None
            continue
        played = played_ucis[ply] if ply < len(played_ucis) else None
        reasons = critical_reasons(pa, prev_best, played)
        if reasons:
            scored.append((_deep_priority(pa, reasons), ply, reasons))
        prev_best = pa.best_cp

    budget = max(8, int(round(DEEP_BUDGET_FRACTION * len(fens))))
    hard = [(p, ply, r) for p, ply, r in scored if _HARD_REASONS & set(r)]
    soft = [(p, ply, r) for p, ply, r in scored if not (_HARD_REASONS & set(r))]
    hard.sort(key=lambda t: -t[0])
    soft.sort(key=lambda t: -t[0])
    chosen = hard[:budget] + soft[: max(0, budget - len(hard))]
    critical: dict[int, list[str]] = {ply: r for _, ply, r in chosen}

    deep: dict[str, PositionAnalysis] = {}
    if deep_pass and critical:
        deep_fens = [fens[p] for p in sorted(critical)]
        deep = analyse_positions(
            deep_fens,
            pass2_depth,
            pass2_multipv,
            pool,
            cache,
            progress,
            label="pass 2 (d%d)" % pass2_depth,
        )

    # Second engine pass (e.g. Lc0).
    second: dict[str, PositionAnalysis] = {}
    second_tag = ""
    if second_pool is not None:
        second_tag = second_pool.tag
        second = analyse_positions(
            fens,
            0,  # depth is meaningless for node-based engines; cache uses nodes
            second_multipv,
            second_pool,
            second_cache,
            progress,
            label=f"{second_tag} ({second_nodes}n)" if second_nodes else second_tag,
            nodes=second_nodes,
        )

    return GameAnalysis(
        list(fens), shallow, deep, critical,
        l0=l0, second=second, second_tag=second_tag,
    )
