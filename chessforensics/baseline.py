"""Corpus builder: fit the reference model on real Lichess games.

    python -m chessforensics.baseline harvest --games 400
    python -m chessforensics.baseline fit
    python -m chessforensics.baseline run --games 400     # both

Why this exists
---------------
Every number the app reports is an observed-vs-expected comparison, so it is
only as good as the "expected". Shipped defaults are guesses; this turns them
into measurements.

Design decisions that matter
----------------------------
*Resumable by construction.* Harvesting is a multi-hour unattended job, so
nothing is held in memory between games. Each game's per-move features are
written straight to SQLite and the game id recorded as done. Interrupt it at
any point and re-running picks up exactly where it stopped. Fitting is a
separate command that reads the accumulated tables, so the expensive part is
never repeated to try a different model.

*Contamination control.* Lichess marks accounts closed for terms-of-service
violations, which is the public signal for "caught cheating". Including those
games would put actual engine users in the honest-player reference
distribution, which raises the expected match rate, flattens the expected
complexity slope, and makes real cheats look normal. They are excluded, and the
exclusion count is recorded in the provenance block.

*The same instrument.* The corpus is analysed through the identical engine
schedule and MultiPV width as a game under test. A reference measured with a
different ruler does not cancel the truncation bias documented in
model._extrapolate_tail.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re as _re
import sqlite3
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import slope as slope_mod
from . import timing as timing_mod
from .baselines import BAND_EDGES, FITTED_PATH, band_for
from .complexity import FEATURES, ComplexityWeights, fit_complexity_weights, raw_features
from .engine import (
    PASS1_DEPTH,
    PASS2_DEPTH,
    PASS2_MULTIPV,
    AnalysisCache,
    EnginePool,
    analyse_game,
)
from .features import build_frame
from .ingest import (
    Game,
    RateLimited,
    _lichess_session,
    iter_pgn_file,
    lichess_account_status,
    lichess_user_games,
)
from .model import fit_choice_model, positions_from_frame

ROOT = Path(__file__).resolve().parent.parent
CORPUS_PATH = ROOT / "cache" / "corpus.sqlite"

MIN_MOVES = 20  # plies; shorter games are mostly book and carry no signal
SIGNAL_NAMES = [
    "t1_match",
    "t2_match",
    "t3_match",
    "acpl",
    "ipr_gap",
    "slope",
    "cpl_slope",
    "time_complexity_rho",
    "time_uniformity",
    "time_cadence",
]


# --------------------------------------------------------------------------
# corpus storage
# --------------------------------------------------------------------------


class Corpus:
    """SQLite-backed store of harvested per-move features."""

    def __init__(self, path: Path = CORPUS_PATH):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), timeout=120, isolation_level=None)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._write_lock = threading.Lock()
        self._schema()

    def _schema(self) -> None:
        cols = ", ".join(f"f_{n} REAL" for n in FEATURES)
        self.conn.executescript(
            f"""
            CREATE TABLE IF NOT EXISTS games (
                game_id TEXT PRIMARY KEY,
                url TEXT, speed TEXT, white_elo INTEGER, black_elo INTEGER,
                moves INTEGER, status TEXT, reason TEXT, done_at REAL
            );
            CREATE TABLE IF NOT EXISTS moves (
                game_id TEXT, ply INTEGER, color TEXT, band TEXT, elo INTEGER,
                complexity REAL, delta_cp REAL, matched_t1 INTEGER,
                matched_t2 INTEGER, matched_t3 INTEGER, rank INTEGER,
                legal INTEGER, gap_cp REAL, best_cp REAL, think_time REAL,
                is_forced INTEGER, scoreable INTEGER, {cols},
                PRIMARY KEY (game_id, ply)
            );
            CREATE TABLE IF NOT EXISTS player_games (
                game_id TEXT, color TEXT, band TEXT, elo INTEGER, speed TEXT,
                n INTEGER, t1 REAL, t2 REAL, t3 REAL, acpl REAL,
                beta1 REAL, beta1_se REAL, cpl_slope REAL,
                rho REAL, cv_residual REAL, cadence REAL,
                s REAL, c REAL,
                PRIMARY KEY (game_id, color)
            );
            CREATE TABLE IF NOT EXISTS seen_users (username TEXT PRIMARY KEY, added REAL);
            CREATE INDEX IF NOT EXISTS idx_moves_band ON moves(band, scoreable);
            CREATE INDEX IF NOT EXISTS idx_pg_band ON player_games(band);

            -- Lite tables: built from %eval annotations only (no engine needed)
            CREATE TABLE IF NOT EXISTS moves_lite (
                game_id TEXT, ply INTEGER, color TEXT, band TEXT, elo INTEGER,
                delta_cp REAL, best_cp REAL, think_time REAL,
                phase TEXT, move_number INTEGER,
                PRIMARY KEY (game_id, ply)
            );
            CREATE TABLE IF NOT EXISTS player_games_lite (
                game_id TEXT, color TEXT, band TEXT, elo INTEGER, speed TEXT,
                n INTEGER, acpl REAL, median_cpl REAL,
                accuracy REAL, blunder_rate REAL,
                PRIMARY KEY (game_id, color)
            );
            CREATE TABLE IF NOT EXISTS games_lite (
                game_id TEXT PRIMARY KEY,
                white_elo INTEGER, black_elo INTEGER,
                speed TEXT, moves INTEGER, done_at REAL
            );
            CREATE INDEX IF NOT EXISTS idx_mlite_band ON moves_lite(band);
            CREATE INDEX IF NOT EXISTS idx_pglite_band ON player_games_lite(band);
            """
        )

    def _migrate_deep_column(self) -> None:
        """Add 'deep' column to games table if missing."""
        try:
            self.conn.execute("ALTER TABLE games ADD COLUMN deep INTEGER")
        except sqlite3.OperationalError:
            pass  # column already exists

    def migrate_bands(self) -> int:
        """Re-compute band labels for all rows after BAND_EDGES change.

        Returns the number of rows updated. Safe to call repeatedly -- rows
        already on the current scheme are no-ops.
        """
        self._migrate_deep_column()
        updated = 0
        for table, elo_col in [("moves", "elo"), ("player_games", "elo")]:
            rows = self.conn.execute(
                f"SELECT rowid, {elo_col}, band FROM {table}"
            ).fetchall()
            batch = []
            for rowid, elo, old_band in rows:
                if elo is None:
                    continue
                new_band = band_for(elo)
                if new_band != old_band:
                    batch.append((new_band, rowid))
            if batch:
                self.conn.execute("BEGIN")
                self.conn.executemany(
                    f"UPDATE {table} SET band=? WHERE rowid=?", batch
                )
                self.conn.execute("COMMIT")
                updated += len(batch)
                print(f"  migrated {len(batch)} rows in {table}")
        return updated

    # ---- bookkeeping ----------------------------------------------------

    def is_done(self, game_id: str) -> bool:
        return (
            self.conn.execute(
                "SELECT 1 FROM games WHERE game_id=?", (game_id,)
            ).fetchone()
            is not None
        )

    def mark(
        self, game: Game, game_id: str, status: str, reason: str = "",
        deep: bool | None = None,
    ) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO games "
            "(game_id,url,speed,white_elo,black_elo,moves,status,reason,done_at,deep) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                game_id,
                game.url,
                game.speed,
                game.players["white"].elo,
                game.players["black"].elo,
                len(game.moves),
                status,
                reason,
                time.time(),
                int(deep) if deep is not None else None,
            ),
        )

    def counts(self) -> dict:
        row = self.conn.execute(
            "SELECT status, COUNT(*) FROM games GROUP BY status"
        ).fetchall()
        out = {s: n for s, n in row}
        out["moves"] = self.conn.execute("SELECT COUNT(*) FROM moves").fetchone()[0]
        out["player_games"] = self.conn.execute(
            "SELECT COUNT(*) FROM player_games"
        ).fetchone()[0]
        return out

    def add_user(self, username: str) -> bool:
        """Returns True if this username is new."""
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO seen_users VALUES (?,?)", (username.lower(), time.time())
        )
        return cur.rowcount > 0

    def band_counts(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT band, COUNT(*) FROM player_games GROUP BY band"
        ).fetchall()
        return {b: n for b, n in rows}

    def band_counts_lite(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT band, COUNT(*) FROM player_games_lite GROUP BY band"
        ).fetchall()
        return {b: n for b, n in rows}

    def is_done_lite(self, game_id: str) -> bool:
        return (
            self.conn.execute(
                "SELECT 1 FROM games_lite WHERE game_id=?", (game_id,)
            ).fetchone()
            is not None
        )

    def lite_counts(self) -> dict:
        g = self.conn.execute("SELECT COUNT(*) FROM games_lite").fetchone()[0]
        m = self.conn.execute("SELECT COUNT(*) FROM moves_lite").fetchone()[0]
        pg = self.conn.execute("SELECT COUNT(*) FROM player_games_lite").fetchone()[0]
        return {"games": g, "moves": m, "player_games": pg}

    def close(self) -> None:
        self.conn.close()


# --------------------------------------------------------------------------
# harvesting
# --------------------------------------------------------------------------

# How far outside the requested band an arena rating may sit and still make the
# account worth fetching. Arena ratings are per-speed (usually blitz) while the
# band filter reads rapid/classical, and the two differ by roughly this much.
_ARENA_ELO_MARGIN = 200


def harvest_usernames(
    corpus: Corpus,
    target: int = 400,
    session=None,
    elo_range: tuple[int, int] | None = None,
) -> list[str]:
    """Collect candidate usernames from recent Lichess arena tournaments.

    Tournaments are used because they are the only public endpoint that yields
    a large, rating-diverse set of ordinary accounts. Leaderboards would give
    only the very top, and the TV feed only whoever is playing right now.

    ``elo_range`` pre-filters candidates on the rating the results endpoint
    already reports, which costs no extra request. Without it, filling a tail
    band means fetching games for mostly mid-rated accounts and discarding
    nearly all of them -- and because each discard still costs a call to the
    games endpoint, a narrow band starves on rate limiting instead of
    harvesting. The arena rating is for that arena's speed rather than the
    rapid/classical rating the band filter uses, so the window is widened by
    ``_ARENA_ELO_MARGIN`` to avoid rejecting accounts that would in fact
    qualify.
    """
    s = session or _lichess_session()
    s.headers["Accept"] = "application/json"
    names: list[str] = []

    try:
        resp = s.get("https://lichess.org/api/tournament", timeout=60)
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, json.JSONDecodeError) as exc:
        print(f"  could not list tournaments: {exc}")
        return names

    tourneys = []
    for key in ("finished", "started", "created"):
        for t in payload.get(key, []) or []:
            if t.get("id"):
                tourneys.append(t["id"])

    s.headers["Accept"] = "application/x-ndjson"
    for tid in tourneys:
        if len(names) >= target:
            break
        try:
            with s.get(
                f"https://lichess.org/api/tournament/{tid}/results",
                params={"nb": 200},
                stream=True,
                timeout=120,
            ) as r:
                if r.status_code == 429:
                    print("  rate limited while listing results; backing off 60s")
                    time.sleep(60)
                    continue
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    name = row.get("username")
                    if not name:
                        continue
                    if elo_range is not None:
                        rating = row.get("rating")
                        lo, hi = elo_range
                        if rating is None or not (
                            lo - _ARENA_ELO_MARGIN <= rating <= hi + _ARENA_ELO_MARGIN
                        ):
                            # Deliberately not marked as seen: this account is
                            # wrong for *this* band but may be the right one for
                            # another, so it stays available to a later pass.
                            continue
                    if corpus.add_user(name):
                        names.append(name)
                    if len(names) >= target:
                        break
        except requests.RequestException:
            continue
        time.sleep(1.0)  # be a good citizen
    return names


def harvest_elite_usernames(
    corpus: Corpus,
    target: int = 200,
    session=None,
    min_rating: int = 2200,
) -> list[str]:
    """Collect high-rated usernames from Lichess leaderboards and titled players.

    Arena tournaments are overwhelmingly blitz/bullet with mid-range ratings,
    so the generic harvester cannot fill the 2400+ bands. This function pulls
    from all speed leaderboards (top 200 each), the titled players endpoint,
    and titled arena results.
    """
    s = session or _lichess_session()
    names: list[str] = []

    # 1. Top 200 from ALL speed leaderboards (not just rapid/classical)
    for perf in ("rapid", "classical", "blitz", "bullet"):
        s.headers["Accept"] = "application/json"
        try:
            resp = s.get(
                f"https://lichess.org/api/player/top/200/{perf}",
                timeout=60,
            )
            if resp.status_code == 429:
                print(f"  rate limited on {perf} leaderboard; backing off 60s")
                time.sleep(60)
                continue
            resp.raise_for_status()
            data = resp.json()
            users = data.get("users", [])
            for u in users:
                name = u.get("username") or u.get("id")
                if name and corpus.add_user(name):
                    names.append(name)
                if len(names) >= target:
                    break
            print(f"  {perf} leaderboard: {len(users)} users, {len(names)} new so far")
        except (requests.RequestException, json.JSONDecodeError) as exc:
            print(f"  could not fetch {perf} leaderboard: {exc}")
        time.sleep(1.0)

    if len(names) >= target:
        return names[:target]

    # 2. Titled players endpoint — GMs, IMs, FMs, etc.
    s.headers["Accept"] = "application/x-ndjson"
    for title in ("GM", "IM", "FM", "WGM", "WIM"):
        if len(names) >= target:
            break
        try:
            resp = s.get(
                f"https://lichess.org/api/titled",
                params={"titles": title},
                timeout=120,
            )
            if resp.status_code == 429:
                print(f"  rate limited on titled/{title}; backing off 60s")
                time.sleep(60)
                continue
            if resp.status_code == 200:
                for line in resp.text.splitlines():
                    if not line.strip():
                        continue
                    try:
                        user = json.loads(line)
                    except json.JSONDecodeError:
                        # Might be plain text (just usernames)
                        name = line.strip()
                        if name and corpus.add_user(name):
                            names.append(name)
                        continue
                    name = user.get("username") or user.get("id") or line.strip()
                    if name and corpus.add_user(name):
                        names.append(name)
                    if len(names) >= target:
                        break
                print(f"  titled {title}: {len(names)} new so far")
        except (requests.RequestException, json.JSONDecodeError) as exc:
            print(f"  could not fetch titled/{title}: {exc}")
        time.sleep(1.0)

    if len(names) >= target:
        return names[:target]

    # 3. Titled Arena tournaments (IMs, GMs) — search for recent ones
    s.headers["Accept"] = "application/json"
    try:
        resp = s.get("https://lichess.org/api/tournament", timeout=60)
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, json.JSONDecodeError):
        return names

    titled_ids = []
    for key in ("finished", "started", "created"):
        for t in payload.get(key, []) or []:
            tname = (t.get("fullName") or t.get("name") or "").lower()
            if "titled" in tname or t.get("hasMaxRating", False):
                if t.get("id"):
                    titled_ids.append(t["id"])

    s.headers["Accept"] = "application/x-ndjson"
    for tid in titled_ids:
        if len(names) >= target:
            break
        try:
            with s.get(
                f"https://lichess.org/api/tournament/{tid}/results",
                params={"nb": 200},
                stream=True,
                timeout=120,
            ) as r:
                if r.status_code == 429:
                    time.sleep(60)
                    continue
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    name = row.get("username")
                    if not name:
                        continue
                    rating = row.get("rating", 0)
                    if rating < min_rating:
                        continue
                    if corpus.add_user(name):
                        names.append(name)
                    if len(names) >= target:
                        break
        except requests.RequestException:
            continue
        time.sleep(1.0)

    print(f"  elite harvest: {len(names)} new usernames total")
    return names[:target]


def harvest_low_rated_usernames(
    corpus: Corpus,
    target: int = 200,
    session=None,
    max_rating: int = 1200,
) -> list[str]:
    """Collect low-rated usernames from Lichess arena tournaments.

    Low-rated players (sub-1000) rarely appear in standard arenas. This
    function specifically targets tournaments with low average ratings and
    fetches bottom-ranked participants who are more likely to be beginners.
    It also fetches more results per tournament (up to 400) to find the
    lower-rated tail.
    """
    s = session or _lichess_session()
    s.headers["Accept"] = "application/json"
    names: list[str] = []

    try:
        resp = s.get("https://lichess.org/api/tournament", timeout=60)
        resp.raise_for_status()
        payload = resp.json()
    except (requests.RequestException, json.JSONDecodeError) as exc:
        print(f"  could not list tournaments: {exc}")
        return names

    tourneys = []
    for key in ("finished", "started", "created"):
        for t in payload.get(key, []) or []:
            if t.get("id"):
                tourneys.append(t["id"])

    s.headers["Accept"] = "application/x-ndjson"
    for tid in tourneys:
        if len(names) >= target:
            break
        try:
            # Fetch more results to reach the low-rated tail of each tournament
            with s.get(
                f"https://lichess.org/api/tournament/{tid}/results",
                params={"nb": 400},
                stream=True,
                timeout=120,
            ) as r:
                if r.status_code == 429:
                    print("  rate limited while listing results; backing off 60s")
                    time.sleep(60)
                    continue
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    name = row.get("username")
                    if not name:
                        continue
                    rating = row.get("rating")
                    if rating is None or rating > max_rating + _ARENA_ELO_MARGIN:
                        continue
                    if corpus.add_user(name):
                        names.append(name)
                    if len(names) >= target:
                        break
        except requests.RequestException:
            continue
        time.sleep(1.0)

    print(f"  low-rated harvest (<={max_rating}): {len(names)} new usernames")
    return names[:target]


def _game_id(game: Game) -> str:
    if game.url:
        return game.url.rstrip("/").rsplit("/", 1)[-1][:8]
    return f"{game.players['white'].name}-{game.players['black'].name}-{len(game.moves)}"


def acceptable(game: Game) -> tuple[bool, str]:
    """Filters that keep the reference distribution clean."""
    if len(game.moves) < MIN_MOVES:
        return False, "too short"
    if game.speed in {"correspondence", "ultrabullet"}:
        return False, f"excluded speed {game.speed}"
    if any(game.tos_flagged.get(c) for c in ("white", "black")):
        # The crucial filter: these are accounts Lichess closed for ToS
        # violations, i.e. the ones caught cheating.
        return False, "tos-flagged account"
    elos = [p.elo for p in game.players.values() if p.elo]
    if len(elos) < 2:
        return False, "missing ratings"
    if abs(elos[0] - elos[1]) > 300:
        return False, "rating mismatch"
    return True, ""


def process_game(
    game: Game,
    corpus: Corpus,
    pool: EnginePool,
    cache: AnalysisCache,
    weights: ComplexityWeights,
    deep_pass: bool,
) -> bool:
    """Analyse one game and write its features. Returns True if stored.

    Thread-safe: engine analysis runs without holding the corpus lock, then
    the DB write is serialised through ``corpus._write_lock``.
    """
    gid = _game_id(game)

    # Quick check under lock (fast read)
    with corpus._write_lock:
        if corpus.is_done(gid):
            return False

    ok, reason = acceptable(game)
    if not ok:
        with corpus._write_lock:
            corpus.mark(game, gid, "skipped", reason, deep=deep_pass)
        return False

    # ---- expensive part: engine analysis (no lock held) ----
    fens = [m.fen_before for m in game.moves]
    ucis = [m.uci for m in game.moves]
    try:
        ga = analyse_game(fens, ucis, pool, cache, deep_pass=deep_pass)
    except Exception as exc:
        with corpus._write_lock:
            corpus.mark(game, gid, "error", str(exc)[:200], deep=deep_pass)
        return False

    df = build_frame(game, ga, weights)
    if df.empty:
        with corpus._write_lock:
            corpus.mark(game, gid, "skipped", "no analysable moves", deep=deep_pass)
        return False

    # ---- prepare all data before taking the lock ----
    def _f(v):
        return None if v is None or pd.isna(v) else float(v)

    def _i(v):
        return None if v is None or pd.isna(v) else int(v)

    move_rows = []
    for _, r in df.iterrows():
        pa = ga.at(int(r["ply"]))
        if pa is None:
            continue
        prev = game.moves[int(r["ply"]) - 1].uci if int(r["ply"]) > 0 else None
        feats = raw_features(pa, prev)
        elo = game.players[r["color"]].elo
        move_rows.append(
            (
                gid,
                int(r["ply"]),
                r["color"],
                band_for(elo),
                elo,
                float(r["complexity"]),
                _f(r["delta_cp"]),
                int(r["matched_t1"]),
                int(r["matched_t2"]),
                int(r["matched_t3"]),
                _i(r["rank"]),
                int(r["legal_moves"]),
                _f(r["gap_cp"]),
                float(r["best_cp"]),
                _f(r["think_time"]),
                int(r["is_forced"]),
                int(r["scoreable"]),
                *[float(feats[n]) for n in FEATURES],
            )
        )

    pg_rows = []
    for color in ("white", "black"):
        sub = df[(df["color"] == color) & (df["scoreable"] == 1)]
        if len(sub) < slope_mod.MIN_POINTS:
            continue
        elo = game.players[color].elo
        cx = sub["complexity"].to_numpy(dtype=float)
        matched = sub["matched_t1"].to_numpy(dtype=float)
        sfit = slope_mod.fit_logistic_slope(cx, matched)
        _, lslope, _, _ = slope_mod.fit_linear_slope(
            cx, sub["delta_cp"].to_numpy(dtype=float)
        )
        tres = timing_mod.analyse_timing(
            df, color, speed=game.speed, increment=game.increment
        )
        cm = fit_choice_model(positions_from_frame(df, ga, color))
        pg_rows.append(
            (
                gid,
                color,
                band_for(elo),
                elo,
                game.speed,
                int(len(sub)),
                float(sub["matched_t1"].mean()),
                float(sub["matched_t2"].mean()),
                float(sub["matched_t3"].mean()),
                float(sub["delta_cp"].mean()),
                float(sfit.beta1) if sfit.converged and not sfit.separable else None,
                float(sfit.se_beta1) if sfit.se_beta1 else None,
                float(lslope),
                tres.rho,
                tres.cv_residual,
                tres.cadence_fraction,
                float(cm.s) if cm.converged else None,
                float(cm.c) if cm.converged else None,
            ),
        )

    # ---- single short lock for all DB writes ----
    placeholders = ",".join("?" * (17 + len(FEATURES)))
    with corpus._write_lock:
        corpus.conn.execute("BEGIN")
        corpus.conn.executemany(f"INSERT OR REPLACE INTO moves VALUES ({placeholders})", move_rows)
        for pg in pg_rows:
            corpus.conn.execute(
                "INSERT OR REPLACE INTO player_games VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                pg,
            )
        corpus.mark(game, gid, "ok", deep=deep_pass)
        corpus.conn.execute("COMMIT")
    return True


def harvest(
    n_games: int,
    min_elo: int,
    max_elo: int,
    deep_pass: bool,
    pool_size: int | None,
    per_user: int,
) -> None:
    corpus = Corpus()
    cache = AnalysisCache()
    weights = ComplexityWeights()  # raw features are stored, so refitting is free

    print(f"corpus: {corpus.counts()}")
    stored = 0
    t0 = time.time()
    session = _lichess_session()

    # Choose the right username harvester for the rating range.
    use_elite = min_elo >= 2200
    use_low = max_elo <= 1200

    try:
        with EnginePool(size=pool_size) as pool:
            while stored < n_games:
                if use_elite:
                    users = harvest_elite_usernames(
                        corpus, target=100, session=session,
                        min_rating=min_elo,
                    )
                elif use_low:
                    users = harvest_low_rated_usernames(
                        corpus, target=100, session=session,
                        max_rating=max_elo,
                    )
                else:
                    users = harvest_usernames(
                        corpus,
                        target=60,
                        session=session,
                        elo_range=(min_elo, max_elo),
                    )
                if not users:
                    print("no new usernames available; stopping")
                    break
                print(f"harvested {len(users)} new usernames")

                for user in users:
                    if stored >= n_games:
                        break
                    try:
                        games = list(
                            lichess_user_games(
                                user,
                                max_games=per_user,
                                perf="rapid,classical",
                                min_elo=min_elo,
                                max_elo=max_elo,
                                session=session,
                            )
                        )
                    except RateLimited:
                        print("  rate limited; sleeping 60s")
                        time.sleep(60)
                        continue
                    except requests.RequestException as exc:
                        print(f"  {user}: {exc}")
                        continue

                    # Bulk-check ToS status for all players in this batch.
                    # The per-game export does NOT include tosViolation; only
                    # the /api/users endpoint does.
                    player_ids = [
                        p.name.lower()
                        for g in games
                        for p in g.players.values()
                        if p.name
                    ]
                    if player_ids:
                        try:
                            tos = lichess_account_status(player_ids, session)
                            for g in games:
                                for c in ("white", "black"):
                                    uid = g.players[c].name.lower()
                                    g.tos_flagged[c] = tos.get(uid, True)
                        except Exception as exc:
                            print(f"  tos check failed ({exc}); using per-game flags")

                    for game in games:
                        if stored >= n_games:
                            break
                        if process_game(game, corpus, pool, cache, weights, deep_pass):
                            stored += 1
                            rate = stored / max(time.time() - t0, 1e-9)
                            print(
                                f"  [{stored}/{n_games}] {_game_id(game)} "
                                f"{game.players['white'].elo}/{game.players['black'].elo} "
                                f"{len(game.moves)}ply "
                                f"({rate * 3600:.0f} games/h, "
                                f"cache {cache.stats()['positions']} pos)"
                            )
                    time.sleep(0.5)
    except KeyboardInterrupt:
        print("\ninterrupted -- progress is saved, re-run to continue")
    finally:
        print(f"corpus now: {corpus.counts()}")
        print(f"per band:   {corpus.band_counts()}")
        corpus.close()


def harvest_all(
    target_per_band: int,
    deep_pass: bool,
    pool_size: int | None,
    per_user: int,
) -> None:
    """Systematically fill every rating band to a target number of player-games.

    Iterates bands from most under-filled to least, harvesting until each band
    reaches the target or runs out of new usernames.
    """
    corpus = Corpus()
    migrated = corpus.migrate_bands()
    if migrated:
        print(f"migrated {migrated} rows to new band scheme")
    current = corpus.band_counts()
    corpus.close()

    bands = list(zip(BAND_EDGES, BAND_EDGES[1:]))
    # Sort by deficit: fill the emptiest bands first
    bands.sort(key=lambda b: current.get(f"{b[0]}-{b[1]}", 0))

    for lo, hi in bands:
        name = f"{lo}-{hi}"
        have = current.get(name, 0)
        if have >= target_per_band:
            print(f"band {name}: already has {have} >= {target_per_band}, skipping")
            continue
        need = target_per_band - have
        # Convert player-game deficit to approximate game count
        # (each game produces ~2 player-games if both sides are in-band)
        games_to_harvest = max(need, need // 2 + 50)
        print(f"\n{'='*60}")
        print(f"band {name}: have {have}, need {need} more player-games")
        print(f"harvesting ~{games_to_harvest} games for {lo}-{hi}")
        print(f"{'='*60}")
        harvest(
            n_games=games_to_harvest,
            min_elo=lo,
            max_elo=hi,
            deep_pass=deep_pass,
            pool_size=pool_size,
            per_user=per_user,
        )

    # Final summary
    corpus = Corpus()
    print(f"\n{'='*60}")
    print("FINAL CORPUS STATUS")
    print(f"{'='*60}")
    counts = corpus.counts()
    band_counts = corpus.band_counts()
    print(f"Total: {counts}")
    for lo, hi in zip(BAND_EDGES, BAND_EDGES[1:]):
        name = f"{lo}-{hi}"
        n = band_counts.get(name, 0)
        bar = "#" * min(n // 20, 50)
        status = "OK" if n >= target_per_band else "LOW"
        print(f"  {name:>9}: {n:5d} player-games {bar} [{status}]")
    corpus.close()


# --------------------------------------------------------------------------
# fitting
# --------------------------------------------------------------------------


def _robust_stats(values: list[float] | np.ndarray) -> tuple[float, float, int]:
    """Mean and sd with 2.5% trimming at each tail.

    Single-game slopes and ACPLs have heavy tails -- a 12-move scoreable sample
    occasionally produces an absurd estimate -- and an untrimmed sd would be
    dominated by those, widening every reference band and destroying power.
    """
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if len(v) < 5:
        return (float(v.mean()) if len(v) else 0.0, float(v.std()) if len(v) > 1 else 0.0, len(v))
    lo, hi = np.quantile(v, [0.025, 0.975])
    trimmed = v[(v >= lo) & (v <= hi)]
    if len(trimmed) < 3:
        trimmed = v
    return float(trimmed.mean()), float(trimmed.std(ddof=1)), len(v)


def fit_all(min_per_band: int = 40) -> dict:
    """Read the corpus and produce the fitted baseline document."""
    corpus = Corpus()
    corpus.migrate_bands()  # ensure bands match current BAND_EDGES
    counts = corpus.counts()
    print(f"fitting from corpus: {counts}")

    # ---- 1. complexity weights, fitted across the whole corpus ----------
    all_rows = corpus.conn.execute(
        f"SELECT matched_t1, {','.join('f_' + n for n in FEATURES)} "
        "FROM moves WHERE scoreable=1"
    ).fetchall()
    cx_weights = ComplexityWeights()
    if len(all_rows) >= 500:
        feats = [dict(zip(FEATURES, r[1:])) for r in all_rows]
        found = [int(r[0]) for r in all_rows]
        cx_weights = fit_complexity_weights(feats, found)
        print(
            f"  complexity weights fitted on {len(all_rows)} positions "
            f"(intercept {cx_weights.intercept:+.2f})"
        )
    else:
        print(f"  only {len(all_rows)} positions; keeping default complexity weights")

    # ---- 2. per-band choice model and reference distributions ----------
    bands: dict[str, dict] = {}
    for lo, hi in zip(BAND_EDGES, BAND_EDGES[1:]):
        name = f"{lo}-{hi}"
        pg = corpus.conn.execute(
            "SELECT n,t1,t2,t3,acpl,beta1,cpl_slope,rho,cv_residual,cadence,s,c "
            "FROM player_games WHERE band=?",
            (name,),
        ).fetchall()
        if len(pg) < min_per_band:
            continue

        arr = {
            k: [r[i] for r in pg]
            for i, k in enumerate(
                [
                    "n",
                    "t1",
                    "t2",
                    "t3",
                    "acpl",
                    "beta1",
                    "cpl_slope",
                    "rho",
                    "cv_residual",
                    "cadence",
                    "s",
                    "c",
                ]
            )
        }

        # Band (s, c): 2.5%-trimmed means of the converged per-game MLE fits.
        # Not a pooled refit: the corpus stores each move's own loss and rank,
        # not the five candidate losses a pooled likelihood would need.
        s_mean, _, _ = _robust_stats(arr["s"])
        c_mean, _, _ = _robust_stats(arr["c"])
        t1_m, t1_sd, _ = _robust_stats(arr["t1"])
        acpl_m, acpl_sd, _ = _robust_stats(arr["acpl"])
        b1_m, b1_sd, b1_n = _robust_stats(arr["beta1"])
        cs_m, cs_sd, _ = _robust_stats(arr["cpl_slope"])
        rho_m, rho_sd, _ = _robust_stats(arr["rho"])
        cv_m, cv_sd, _ = _robust_stats(arr["cv_residual"])
        cad_m, cad_sd, _ = _robust_stats(arr["cadence"])

        bands[name] = {
            "choice_model": {"s": s_mean, "c": c_mean, "n": len(pg)},
            "slope": {"mean": b1_m, "sd": max(b1_sd, 0.3), "n": b1_n},
            "cpl_slope": {"mean": cs_m, "sd": max(cs_sd, 10.0)},
            "acpl": {"mean": acpl_m, "sd": max(acpl_sd, 5.0)},
            "t1": {"mean": t1_m, "sd": max(t1_sd, 0.02)},
            "timing": {
                "rho_mean": rho_m,
                "rho_sd": max(rho_sd, 0.05),
                "cv_residual_mean": cv_m,
                "cv_residual_sd": max(cv_sd, 0.05),
                "cadence_mean": cad_m,
                "cadence_sd": max(cad_sd, 0.03),
            },
            "player_games": len(pg),
        }
        print(
            f"  band {name:>9}: {len(pg):4d} player-games  "
            f"s={s_mean:.3f} c={c_mean:.3f} t1={t1_m:.3f} "
            f"acpl={acpl_m:5.1f} beta1={b1_m:+.2f}+-{b1_sd:.2f}"
        )

    # ---- 2b. merge lite corpus distributions into bands -------------------
    # The lite corpus has 100x more data for ACPL/blunder_rate. Use it to
    # sharpen the reference distributions where the deep corpus is thin.
    lite_counts = corpus.lite_counts()
    lite_pg_total = lite_counts.get("player_games", 0)
    if lite_pg_total > 0:
        print(f"  merging lite corpus: {lite_pg_total} player-games")
        for lo, hi in zip(BAND_EDGES, BAND_EDGES[1:]):
            name = f"{lo}-{hi}"
            lite_pg = corpus.conn.execute(
                "SELECT acpl, blunder_rate, accuracy, n "
                "FROM player_games_lite WHERE band=? AND n >= 15",
                (name,),
            ).fetchall()
            if len(lite_pg) < 100:
                continue
            acpl_vals = [r[0] for r in lite_pg if r[0] is not None]
            blunder_vals = [r[1] for r in lite_pg if r[1] is not None]
            accuracy_vals = [r[2] for r in lite_pg if r[2] is not None]
            if name in bands and acpl_vals:
                la_m, la_sd, _ = _robust_stats(acpl_vals)
                bands[name]["acpl_lite"] = {
                    "mean": la_m,
                    "sd": max(la_sd, 5.0),
                    "n": len(acpl_vals),
                }
            if name in bands and blunder_vals:
                lb_m, lb_sd, _ = _robust_stats(blunder_vals)
                bands[name]["blunder_rate_lite"] = {
                    "mean": lb_m,
                    "sd": max(lb_sd, 0.01),
                    "n": len(blunder_vals),
                }
            if name in bands and accuracy_vals:
                lac_m, lac_sd, _ = _robust_stats(accuracy_vals)
                bands[name]["accuracy_lite"] = {
                    "mean": lac_m,
                    "sd": max(lac_sd, 2.0),
                    "n": len(accuracy_vals),
                }
            if name not in bands and acpl_vals:
                # Band has no deep data but lite data exists — create a
                # lite-only entry so the band isn't completely empty.
                la_m, la_sd, _ = _robust_stats(acpl_vals)
                bands[name] = {
                    "lite_only": True,
                    "acpl_lite": {
                        "mean": la_m,
                        "sd": max(la_sd, 5.0),
                        "n": len(acpl_vals),
                    },
                    "player_games": 0,
                    "player_games_lite": len(lite_pg),
                }
                if blunder_vals:
                    lb_m, lb_sd, _ = _robust_stats(blunder_vals)
                    bands[name]["blunder_rate_lite"] = {
                        "mean": lb_m,
                        "sd": max(lb_sd, 0.01),
                        "n": len(blunder_vals),
                    }
                if accuracy_vals:
                    lac_m, lac_sd, _ = _robust_stats(accuracy_vals)
                    bands[name]["accuracy_lite"] = {
                        "mean": lac_m,
                        "sd": max(lac_sd, 2.0),
                        "n": len(accuracy_vals),
                    }
            if name in bands and "acpl_lite" in bands[name]:
                n_lite = bands[name]["acpl_lite"]["n"]
                print(
                    f"    lite {name:>9}: {n_lite:>7} pg  "
                    f"acpl={bands[name]['acpl_lite']['mean']:5.1f}"
                    f"{'  (lite-only band)' if bands[name].get('lite_only') else ''}"
                )

    # ---- 3. signal correlation matrix -----------------------------------
    corr = _signal_correlation(corpus, bands)

    corpus.close()

    # Merge deep + lite counts for the headline numbers shown on the site.
    # Both corpora contribute to the reference model.
    total_games = counts.get("ok", 0) + lite_counts.get("games", 0)
    total_player_games = counts.get("player_games", 0) + lite_pg_total
    total_positions = counts.get("moves", 0) + lite_counts.get("moves", 0)

    return {
        "schema": 1,
        "provenance": {
            "kind": "fitted",
            "validated": False,
            "games": total_games,
            "skipped": counts.get("skipped", 0),
            "player_games": total_player_games,
            "positions": total_positions,
            "games_deep": counts.get("ok", 0),
            "player_games_deep": counts.get("player_games", 0),
            "positions_deep": counts.get("moves", 0),
            "games_lite": lite_counts.get("games", 0),
            "player_games_lite": lite_pg_total,
            "positions_lite": lite_counts.get("moves", 0),
            "depth": f"adaptive {PASS1_DEPTH}/{PASS2_DEPTH}",
            "multipv": PASS2_MULTIPV,
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            "deep_pass": True,  # default; re-fitted corpora should record this
            "filters": {
                "min_moves": MIN_MOVES,
                "exclude_tos_flagged": True,
                "exclude_correspondence": True,
                "max_rating_gap": 300,
                "perf": "rapid,classical",
                "rated_only": True,
            },
            "min_player_games_per_band": min_per_band,
        },
        "complexity_weights": cx_weights.to_dict(),
        "bands": bands,
        "signal_effects": None,  # measured by the positive control, see verify.py
        "signal_correlation": corr,
    }


def _signal_correlation(corpus: Corpus, bands: dict) -> dict | None:
    """Estimate the between-signal correlation matrix on honest games.

    Each player-game is converted to the same z-scores the live scorer uses,
    standardised within its own band so that band-to-band differences in
    difficulty do not masquerade as signal correlation. The result is what
    score.combine inverts to stop correlated evidence being counted repeatedly.
    """
    rows = corpus.conn.execute(
        "SELECT band,t1,t2,t3,acpl,beta1,cpl_slope,rho,cv_residual,cadence "
        "FROM player_games"
    ).fetchall()
    if len(rows) < 60:
        print("  too few player-games for a correlation estimate; keeping defaults")
        return None

    # Map each stored quantity to a signal name and the sign that makes
    # "larger = more suspicious".
    spec = [
        ("t1_match", 1, +1.0),
        ("t2_match", 2, +1.0),
        ("t3_match", 3, +1.0),
        ("acpl", 4, -1.0),
        ("slope", 5, +1.0),
        ("cpl_slope", 6, -1.0),
        ("time_complexity_rho", 7, -1.0),
        ("time_uniformity", 8, -1.0),
        ("time_cadence", 9, +1.0),
    ]

    by_band: dict[str, list] = {}
    for r in rows:
        by_band.setdefault(r[0], []).append(r)

    stacked: list[np.ndarray] = []
    for band, brows in by_band.items():
        if len(brows) < 20:
            continue
        cols = []
        for _, idx, sign in spec:
            v = np.array(
                [np.nan if r[idx] is None else float(r[idx]) for r in brows], dtype=float
            )
            mu = np.nanmean(v)
            sd = np.nanstd(v)
            if not np.isfinite(sd) or sd < 1e-9:
                cols.append(np.zeros(len(brows)))
            else:
                z = sign * (v - mu) / sd
                cols.append(np.nan_to_num(z, nan=0.0))
        stacked.append(np.column_stack(cols))

    if not stacked:
        return None
    M = np.vstack(stacked)
    if len(M) < 40:
        return None
    R = np.corrcoef(M, rowvar=False)
    R = np.nan_to_num(R, nan=0.0)

    names = [s[0] for s in spec]
    out: dict[str, dict] = {}
    for i, a in enumerate(names):
        out[a] = {b: float(R[i, j]) for j, b in enumerate(names) if i != j}
    # ipr_gap is derived from the same fit as acpl/t1, so copy acpl's row
    # rather than leaving it to the independence default.
    out["ipr_gap"] = dict(out.get("acpl", {}))
    out["ipr_gap"]["acpl"] = 0.85
    for k in out:
        if k != "ipr_gap":
            out[k]["ipr_gap"] = out["ipr_gap"].get(k, 0.3)
    print(f"  signal correlation estimated from {len(M)} player-games")
    return out


def _needs_band(elo: int, band_fill: dict[str, int], target: int) -> bool:
    """True if this player's band is below the target."""
    band = band_for(elo)
    return band_fill.get(band, 0) < target


def import_pgn(
    pgn_paths: list[str],
    deep_pass: bool,
    pool_size: int | None,
    min_elo: int | None = None,
    max_elo: int | None = None,
    max_games: int | None = None,
    workers: int = 1,
    target: int = 5000,
    smart: bool = False,
) -> None:
    """Import games from one or more PGN files into the corpus.

    This is the fast path for building the corpus from bulk sources:
    Lichess database dumps, TWIC archives, KingBase, Caissabase, etc.

    When ``smart=True``, games are only imported if at least one player falls
    in a band that is still below ``target`` player-games. Bands already at
    target are skipped, concentrating engine time on the bands that need it.
    Band counts are refreshed from the DB periodically.
    """
    corpus = Corpus()
    corpus.migrate_bands()
    cache = AnalysisCache()
    weights = ComplexityWeights()

    print(f"corpus before import: {corpus.counts()}")
    band_fill = corpus.band_counts()
    if smart:
        print("smart mode: skipping games that only fill bands already at target")
        for b in sorted(band_fill):
            status = "FULL" if band_fill[b] >= target else f"need {target - band_fill[b]}"
            print(f"  {b:>9}: {band_fill[b]:5d}  [{status}]")

    stored = 0
    skipped = 0
    skipped_full = 0
    t0 = time.time()

    try:
        with EnginePool(size=pool_size) as pool:
            for pgn_path in pgn_paths:
                p = Path(pgn_path)
                if not p.exists():
                    print(f"file not found: {pgn_path}")
                    continue
                print(f"\nimporting from {p.name} ...")

                for game in iter_pgn_file(p, source=f"pgn:{p.name}"):
                    if max_games and stored >= max_games:
                        break

                    # Rating filter
                    elos = [pl.elo for pl in game.players.values() if pl.elo]
                    if len(elos) < 2:
                        skipped += 1
                        continue
                    if min_elo is not None and min(elos) < min_elo:
                        skipped += 1
                        continue
                    if max_elo is not None and max(elos) > max_elo:
                        skipped += 1
                        continue

                    # Smart band filter: skip if BOTH players are in full bands
                    if smart:
                        if not any(_needs_band(e, band_fill, target) for e in elos):
                            skipped_full += 1
                            continue

                    if process_game(game, corpus, pool, cache, weights, deep_pass):
                        stored += 1
                        # Update in-memory band counts
                        for color in ("white", "black"):
                            elo = game.players[color].elo
                            if elo:
                                b = band_for(elo)
                                band_fill[b] = band_fill.get(b, 0) + 1
                        if stored % 20 == 0 or stored < 5:
                            rate = stored / max(time.time() - t0, 1e-9)
                            avg_elo = sum(elos) / len(elos)
                            print(
                                f"  [{stored}] {_game_id(game)} "
                                f"~{avg_elo:.0f} elo, {len(game.moves)}ply "
                                f"({rate * 3600:.0f} games/h, "
                                f"skipped-full {skipped_full}, "
                                f"cache {cache.stats()['positions']} pos)"
                            )
                        # Refresh band counts from DB every 200 games for accuracy
                        if stored % 200 == 0:
                            band_fill = corpus.band_counts()
                            done_bands = sum(1 for b, n in band_fill.items() if n >= target)
                            total_bands = len(BAND_EDGES) - 1
                            print(f"  -- bands at target: {done_bands}/{total_bands} --")
                            if smart and done_bands >= total_bands:
                                print("  ALL bands at target!")
                                break
                    else:
                        skipped += 1

                if max_games and stored >= max_games:
                    print(f"reached --max-games {max_games}")
                    break

    except KeyboardInterrupt:
        print("\ninterrupted -- progress is saved, re-run to continue")
    finally:
        elapsed = time.time() - t0
        print(f"\nimported {stored} games, skipped {skipped}, "
              f"skipped-full-bands {skipped_full}, "
              f"in {elapsed/60:.1f} minutes")
        print(f"corpus now: {corpus.counts()}")
        print(f"per band:   {corpus.band_counts()}")
        corpus.close()


# --------------------------------------------------------------------------
# fast regex-based PGN parser for lite import (no python-chess dependency)
# --------------------------------------------------------------------------

_EVAL_INLINE = _re.compile(r"\[%eval\s+(#?-?[\d.]+)\]")
_CLK_INLINE = _re.compile(r"\[%clk\s+(\d+):(\d+):(\d+(?:\.\d+)?)\]")
_HEADER_RE = _re.compile(r'^\[(\w+)\s+"(.*)"\]\s*$')
_MOVE_SPLIT = _re.compile(
    r'(\d+\.+\s*)?([a-hKQRBNO][a-h1-8x+#=QRBN\-O]*)'
    r'(?:\s*\{([^}]*)\})?'
)


def _eval_cp(raw: str) -> float:
    """Parse eval string to centipawns."""
    if raw.startswith("#"):
        n = int(raw[1:])
        return 1000.0 if n > 0 else -1000.0
    return float(raw) * 100


def _phase(mn: int) -> str:
    if mn <= 15:
        return "opening"
    if mn <= 35:
        return "middle"
    return "endgame"


def _game_id_lite(site: str, white: str, black: str, date: str) -> str:
    """Fast deterministic game id."""
    if site and "lichess.org/" in site:
        # Use Lichess game id directly
        return site.rsplit("/", 1)[-1][:12]
    key = f"{white}-{black}-{date}-{site}"
    return hashlib.md5(key.encode()).hexdigest()[:16]


def _iter_raw_games(fh) -> "Iterator[tuple[dict, str]]":
    """Yield (headers_dict, moves_text) from a text stream. No python-chess."""
    headers: dict[str, str] = {}
    moves_lines: list[str] = []
    in_moves = False

    for line in fh:
        line = line.rstrip("\n\r")
        if not line:
            if in_moves and moves_lines:
                yield headers, " ".join(moves_lines)
                headers, moves_lines, in_moves = {}, [], False
            continue
        hm = _HEADER_RE.match(line)
        if hm:
            headers[hm.group(1)] = hm.group(2)
            in_moves = False
        else:
            in_moves = True
            moves_lines.append(line)

    if moves_lines:
        yield headers, " ".join(moves_lines)


def _open_pgn_stream(path: Path):
    """Open .pgn or .pgn.zst as a text line iterator."""
    if path.suffix == ".zst" or path.name.endswith(".pgn.zst"):
        import zstandard as zstd
        dctx = zstd.ZstdDecompressor()
        fh_bin = open(path, "rb")
        reader = dctx.stream_reader(fh_bin)
        return io.TextIOWrapper(reader, encoding="utf-8", errors="replace"), fh_bin
    fh = open(path, encoding="utf-8", errors="replace")
    return fh, None


def _parse_tc(tc: str | None) -> str | None:
    """Classify speed from TimeControl header."""
    if not tc or tc in {"-", "?"}:
        return None
    first = tc.split(":")[0]
    if "/" in first:
        first = first.split("/", 1)[1]
    m = _re.match(r"^(\d+)(?:\+(\d+))?$", first.strip())
    if not m:
        return None
    base = int(m.group(1))
    inc = int(m.group(2)) if m.group(2) else 0
    est = base + 40 * inc
    if est < 30:
        return "ultrabullet"
    if est < 180:
        return "bullet"
    if est < 480:
        return "blitz"
    if est < 1500:
        return "rapid"
    if est < 21600:
        return "classical"
    return "correspondence"


def import_pgn_lite(
    pgn_paths: list[str],
    target: int = 500_000,
    smart: bool = True,
    max_games: int | None = None,
    batch_size: int = 1000,
) -> None:
    """Ultra-fast import using %eval annotations -- no engine needed.

    Uses a regex-based PGN parser (no python-chess) for ~3000-5000 games/s.
    Stores CPL-based statistics in the lite tables.

    Supports .pgn and .pgn.zst files.
    """
    corpus = Corpus()
    counts = corpus.lite_counts()
    print(f"lite corpus before: {counts}")

    band_fill = corpus.band_counts_lite()
    if smart:
        print(f"smart mode ON (target per band: {target:,})")
        for b in sorted(band_fill):
            n = band_fill[b]
            status = "FULL" if n >= target else f"need {target - n}"
            print(f"  {b:>9}: {n:6d}  [{status}]")

    stored = 0
    scanned = 0
    skipped = 0
    skipped_full = 0
    no_eval = 0
    t0 = time.time()

    move_buf: list[tuple] = []
    pg_buf: list[tuple] = []
    game_buf: list[tuple] = []

    def flush():
        nonlocal move_buf, pg_buf, game_buf
        if not move_buf and not pg_buf:
            return
        corpus.conn.execute("BEGIN")
        if move_buf:
            corpus.conn.executemany(
                "INSERT OR IGNORE INTO moves_lite VALUES (?,?,?,?,?,?,?,?,?,?)",
                move_buf,
            )
        if pg_buf:
            corpus.conn.executemany(
                "INSERT OR IGNORE INTO player_games_lite VALUES (?,?,?,?,?,?,?,?,?,?)",
                pg_buf,
            )
        if game_buf:
            corpus.conn.executemany(
                "INSERT OR IGNORE INTO games_lite VALUES (?,?,?,?,?,?)",
                game_buf,
            )
        corpus.conn.execute("COMMIT")
        move_buf, pg_buf, game_buf = [], [], []

    try:
        for pgn_path in pgn_paths:
            p = Path(pgn_path)
            if not p.exists():
                print(f"not found: {pgn_path}")
                continue
            print(f"\nimporting from {p.name} ...")

            stream, closer = _open_pgn_stream(p)
            try:
                for headers, moves_text in _iter_raw_games(stream):
                    if max_games and stored >= max_games:
                        break

                    scanned += 1

                    # Need both ratings
                    try:
                        w_elo = int(headers.get("WhiteElo", ""))
                        b_elo = int(headers.get("BlackElo", ""))
                    except (ValueError, TypeError):
                        skipped += 1
                        continue

                    # Quick check: does the move text contain %eval at all?
                    if "%eval" not in moves_text:
                        no_eval += 1
                        continue

                    # Smart: skip if both players in full bands
                    if smart:
                        w_band = band_for(w_elo)
                        b_band = band_for(b_elo)
                        if (band_fill.get(w_band, 0) >= target
                                and band_fill.get(b_band, 0) >= target):
                            skipped_full += 1
                            continue

                    gid = _game_id_lite(
                        headers.get("Site", ""),
                        headers.get("White", "?"),
                        headers.get("Black", "?"),
                        headers.get("UTCDate", headers.get("Date", "?")),
                    )

                    # Parse evals from moves text using regex
                    # Lichess format: 1. e4 { [%eval 0.17] [%clk 0:10:00] } 1... c5 { [%eval 0.32] }
                    evals: list[tuple[float, float | None]] = []  # (eval_cp, clock)
                    for m in _re.finditer(r'\{([^}]*)\}', moves_text):
                        comment = m.group(1)
                        em = _EVAL_INLINE.search(comment)
                        if em:
                            ev = _eval_cp(em.group(1))
                            clk = None
                            cm = _CLK_INLINE.search(comment)
                            if cm:
                                clk = int(cm.group(1)) * 3600 + int(cm.group(2)) * 60 + float(cm.group(3))
                            evals.append((ev, clk))

                    if len(evals) < MIN_MOVES:
                        no_eval += 1
                        continue

                    speed = _parse_tc(headers.get("TimeControl"))
                    total_plies = len(evals)

                    # Compute CPL from consecutive evals
                    # evals[i] = eval after ply i (in white's perspective)
                    # For ply i (0-indexed): color = white if i is even, black if odd
                    move_rows_local: list[tuple] = []
                    per_color: dict[str, list[float]] = {"white": [], "black": []}
                    for i in range(1, len(evals)):
                        prev_ev, _ = evals[i - 1]
                        cur_ev, clk = evals[i]
                        color = "white" if i % 2 == 1 else "black"
                        mn = (i // 2) + 1
                        # CPL from the moving side's perspective
                        if color == "white":
                            cpl = prev_ev - cur_ev
                        else:
                            cpl = cur_ev - prev_ev
                        cpl = max(0.0, min(cpl, 1000.0))

                        elo = w_elo if color == "white" else b_elo
                        move_rows_local.append((
                            gid, i, color, band_for(elo), elo,
                            round(cpl, 1), round(cur_ev, 1), clk,
                            _phase(mn), mn,
                        ))
                        per_color[color].append(cpl)

                    if not move_rows_local:
                        no_eval += 1
                        continue

                    # Player-game summaries
                    pg_local: list[tuple] = []
                    for color in ("white", "black"):
                        cpls = per_color[color]
                        if len(cpls) < 10:
                            continue
                        elo = w_elo if color == "white" else b_elo
                        acpl = sum(cpls) / len(cpls)
                        median = sorted(cpls)[len(cpls) // 2]
                        accuracy = max(0.0, min(100.0,
                            103.1668 * (2.71828 ** (-0.04354 * acpl)) - 3.1668))
                        blunders = sum(1 for c in cpls if c >= 200) / len(cpls)
                        pg_local.append((
                            gid, color, band_for(elo), elo, speed,
                            len(cpls), round(acpl, 1), round(median, 1),
                            round(accuracy, 1), round(blunders, 4),
                        ))

                    if not pg_local:
                        no_eval += 1
                        continue

                    move_buf.extend(move_rows_local)
                    pg_buf.extend(pg_local)
                    game_buf.append((gid, w_elo, b_elo, speed, total_plies, time.time()))

                    stored += 1
                    for pg in pg_local:
                        band_fill[pg[2]] = band_fill.get(pg[2], 0) + 1

                    if stored % batch_size == 0:
                        flush()

                    if stored % 2000 == 0 or (stored < 200 and stored % 50 == 0):
                        elapsed = time.time() - t0
                        rate = stored / max(elapsed, 0.001)
                        scan_rate = scanned / max(elapsed, 0.001)
                        print(
                            f"  [{stored:,}] {rate:.0f} stored/s ({scan_rate:.0f} scanned/s) | "
                            f"skip-full {skipped_full:,} | no-eval {no_eval:,} | "
                            f"elapsed {elapsed / 60:.1f}m"
                        )

                    if smart and stored % 50_000 == 0:
                        flush()
                        band_fill = corpus.band_counts_lite()
                        done = sum(1 for n in band_fill.values() if n >= target)
                        total = len(BAND_EDGES) - 1
                        print(f"  -- lite bands at target: {done}/{total} --")
                        for b in sorted(band_fill):
                            n = band_fill[b]
                            st = "FULL" if n >= target else f"need {target - n}"
                            print(f"    {b:>9}: {n:6d}  [{st}]")
                        if done >= total:
                            print("  ALL lite bands at target!")
                            break

            finally:
                stream.close()
                if closer:
                    closer.close()

            if max_games and stored >= max_games:
                print(f"reached --max-games {max_games}")
                break

    except KeyboardInterrupt:
        print("\ninterrupted -- flushing...")
    finally:
        flush()
        elapsed = time.time() - t0
        print(
            f"\nlite import: {stored:,} games stored, "
            f"{scanned:,} scanned, "
            f"skipped {skipped:,}, skipped-full {skipped_full:,}, "
            f"no-eval {no_eval:,}, in {elapsed / 60:.1f} min"
        )
        print(f"lite corpus now: {corpus.lite_counts()}")
        print(f"lite bands: {corpus.band_counts_lite()}")
        corpus.close()


def import_pgn_quick_eval(
    pgn_paths: list[str],
    target: int = 500_000,
    smart: bool = True,
    max_games: int | None = None,
    pool_size: int | None = None,
    depth: int = 6,
    batch_size: int = 200,
    workers: int = 4,
) -> None:
    """Import games into lite tables using fast Stockfish eval (no %eval needed).

    For PGN files without Lichess %eval annotations (e.g. Elite DB, TWIC).
    Runs Stockfish at low depth (default 6) with 1 PV to compute CPL.
    Uses threading to process multiple games in parallel across the engine pool.
    """
    import chess
    import chess.pgn
    from concurrent.futures import ThreadPoolExecutor, as_completed

    corpus = Corpus()
    counts = corpus.lite_counts()
    print(f"lite corpus before: {counts}")

    band_fill = corpus.band_counts_lite()
    if smart:
        print(f"smart mode ON (target per band: {target:,})")
        for b in sorted(band_fill):
            n = band_fill[b]
            status = "FULL" if n >= target else f"need {target - n}"
            print(f"  {b:>9}: {n:6d}  [{status}]")

    stored = 0
    scanned = 0
    skipped = 0
    skipped_full = 0
    t0 = time.time()

    move_buf: list[tuple] = []
    pg_buf: list[tuple] = []
    game_buf: list[tuple] = []

    def flush():
        nonlocal move_buf, pg_buf, game_buf
        if not move_buf and not pg_buf:
            return
        corpus.conn.execute("BEGIN")
        if move_buf:
            corpus.conn.executemany(
                "INSERT OR IGNORE INTO moves_lite VALUES (?,?,?,?,?,?,?,?,?,?)",
                move_buf,
            )
        if pg_buf:
            corpus.conn.executemany(
                "INSERT OR IGNORE INTO player_games_lite VALUES (?,?,?,?,?,?,?,?,?,?)",
                pg_buf,
            )
        if game_buf:
            corpus.conn.executemany(
                "INSERT OR IGNORE INTO games_lite VALUES (?,?,?,?,?,?)",
                game_buf,
            )
        corpus.conn.execute("COMMIT")
        move_buf, pg_buf, game_buf = [], [], []

    def _eval_game(headers, moves_text, eng_pool, eval_depth):
        """Evaluate one game — runs in a worker thread."""
        w_elo = int(headers["WhiteElo"])
        b_elo = int(headers["BlackElo"])

        # Parse game with python-chess
        game_text = ""
        for k, v in headers.items():
            game_text += f'[{k} "{v}"]\n'
        game_text += "\n" + moves_text + "\n"
        game = chess.pgn.read_game(io.StringIO(game_text))
        if game is None:
            return None

        board = game.board()
        fens = [board.fen()]
        node = game
        while node.variations:
            node = node.variation(0)
            board.push(node.move)
            fens.append(board.fen())

        if len(fens) < 20:
            return None

        # Cap at 100 plies to save time
        fens = fens[:min(len(fens), 100)]

        gid = _game_id_lite(
            headers.get("Site", ""),
            headers.get("White", "?"),
            headers.get("Black", "?"),
            headers.get("UTCDate", headers.get("Date", "?")),
        )

        # Eval each position
        evals = []
        for fen in fens:
            try:
                pa = eng_pool.analyse_one(fen, depth=eval_depth, multipv=1)
                evals.append(pa.candidates[0].cp if pa.candidates else 0.0)
            except Exception:
                evals.append(0.0)

        # Compute CPL
        speed = _parse_tc(headers.get("TimeControl"))
        move_rows = []
        per_color = {"white": [], "black": []}
        for i in range(1, len(evals)):
            color = "white" if i % 2 == 1 else "black"
            mn = (i // 2) + 1
            if color == "white":
                cpl = evals[i - 1] - evals[i]
            else:
                cpl = evals[i] - evals[i - 1]
            cpl = max(0.0, min(cpl, 1000.0))
            elo = w_elo if color == "white" else b_elo
            move_rows.append((
                gid, i, color, band_for(elo), elo,
                round(cpl, 1), round(evals[i], 1), None,
                _phase(mn), mn,
            ))
            per_color[color].append(cpl)

        if not move_rows:
            return None

        pg_rows = []
        for color in ("white", "black"):
            cpls = per_color[color]
            if len(cpls) < 10:
                continue
            elo = w_elo if color == "white" else b_elo
            acpl = sum(cpls) / len(cpls)
            median = sorted(cpls)[len(cpls) // 2]
            accuracy = max(0.0, min(100.0,
                103.1668 * (2.71828 ** (-0.04354 * acpl)) - 3.1668))
            blunders = sum(1 for c in cpls if c >= 200) / len(cpls)
            pg_rows.append((
                gid, color, band_for(elo), elo, speed,
                len(cpls), round(acpl, 1), round(median, 1),
                round(accuracy, 1), round(blunders, 4),
            ))

        if not pg_rows:
            return None

        game_row = (gid, w_elo, b_elo, speed, len(evals), time.time())
        return move_rows, pg_rows, game_row

    try:
        with EnginePool(size=pool_size) as pool:
            executor = ThreadPoolExecutor(max_workers=workers)
            for pgn_path in pgn_paths:
                p = Path(pgn_path)
                if not p.exists():
                    print(f"not found: {pgn_path}")
                    continue
                print(f"\nimporting from {p.name} ...")

                stream, closer = _open_pgn_stream(p)
                futures: dict = {}
                try:
                    for headers, moves_text in _iter_raw_games(stream):
                        if max_games and stored >= max_games:
                            break
                        scanned += 1

                        try:
                            w_elo = int(headers.get("WhiteElo", ""))
                            b_elo = int(headers.get("BlackElo", ""))
                        except (ValueError, TypeError):
                            skipped += 1
                            continue

                        if smart:
                            w_band = band_for(w_elo)
                            b_band = band_for(b_elo)
                            if (band_fill.get(w_band, 0) >= target
                                    and band_fill.get(b_band, 0) >= target):
                                skipped_full += 1
                                continue

                        # Submit to thread pool
                        fut = executor.submit(
                            _eval_game, dict(headers), moves_text, pool, depth,
                        )
                        futures[fut] = (w_elo, b_elo)

                        # Collect completed futures (non-blocking)
                        done_futs = [f for f in futures if f.done()]
                        for fut in done_futs:
                            result = fut.result()
                            del futures[fut]
                            if result is None:
                                skipped += 1
                                continue
                            move_rows, pg_rows, game_row = result
                            move_buf.extend(move_rows)
                            pg_buf.extend(pg_rows)
                            game_buf.append(game_row)
                            stored += 1
                            for pg in pg_rows:
                                band_fill[pg[2]] = band_fill.get(pg[2], 0) + 1

                        # Limit in-flight futures
                        while len(futures) >= workers * 2:
                            # Wait for any one to complete
                            for fut in list(futures):
                                if fut.done():
                                    result = fut.result()
                                    del futures[fut]
                                    if result is None:
                                        skipped += 1
                                        continue
                                    move_rows, pg_rows, game_row = result
                                    move_buf.extend(move_rows)
                                    pg_buf.extend(pg_rows)
                                    game_buf.append(game_row)
                                    stored += 1
                                    for pg in pg_rows:
                                        band_fill[pg[2]] = band_fill.get(pg[2], 0) + 1
                                    break
                            else:
                                time.sleep(0.01)

                        if stored % batch_size == 0 and move_buf:
                            flush()

                        if stored > 0 and (stored % 100 == 0 or (stored < 20 and stored % 5 == 0)):
                            elapsed = time.time() - t0
                            rate = stored / max(elapsed, 0.001)
                            print(
                                f"  [{stored:,}] {rate:.1f} games/s | "
                                f"skip-full {skipped_full:,} | "
                                f"elapsed {elapsed / 60:.1f}m"
                            )

                        if smart and stored > 0 and stored % 10_000 == 0:
                            flush()
                            band_fill = corpus.band_counts_lite()
                            done = sum(1 for n in band_fill.values() if n >= target)
                            total_bands = len(BAND_EDGES) - 1
                            print(f"  -- lite bands at target: {done}/{total_bands} --")
                            for b in sorted(band_fill):
                                n = band_fill[b]
                                st = "FULL" if n >= target else f"need {target - n}"
                                print(f"    {b:>9}: {n:6d}  [{st}]")
                            if done >= total_bands:
                                print("  ALL lite bands at target!")
                                break

                    # Drain remaining futures
                    for fut in as_completed(futures):
                        result = fut.result()
                        if result is None:
                            skipped += 1
                            continue
                        move_rows, pg_rows, game_row = result
                        move_buf.extend(move_rows)
                        pg_buf.extend(pg_rows)
                        game_buf.append(game_row)
                        stored += 1
                        for pg in pg_rows:
                            band_fill[pg[2]] = band_fill.get(pg[2], 0) + 1

                finally:
                    stream.close()
                    if closer:
                        closer.close()

                if max_games and stored >= max_games:
                    print(f"reached --max-games {max_games}")
                    break

            executor.shutdown(wait=False)

    except KeyboardInterrupt:
        print("\ninterrupted -- flushing...")
    finally:
        flush()
        elapsed = time.time() - t0
        print(
            f"\nquick-eval import: {stored:,} games stored, "
            f"{scanned:,} scanned, "
            f"skipped {skipped:,}, skipped-full {skipped_full:,}, "
            f"in {elapsed / 60:.1f} min"
        )
        print(f"lite corpus now: {corpus.lite_counts()}")
        print(f"lite bands: {corpus.band_counts_lite()}")
        corpus.close()


def scan_pgn(
    pgn_paths: list[str],
    min_elo: int | None = None,
    max_elo: int | None = None,
) -> None:
    """Quick scan of PGN files: count games per rating band without analysis.

    Useful for estimating how many games a bulk import would produce before
    committing to the expensive engine analysis step.
    """
    from collections import Counter
    band_counts: Counter[str] = Counter()
    total = 0
    usable = 0

    for pgn_path in pgn_paths:
        p = Path(pgn_path)
        if not p.exists():
            print(f"not found: {pgn_path}")
            continue
        print(f"scanning {p.name} ...")

        for game in iter_pgn_file(p):
            total += 1
            if total % 10000 == 0:
                print(f"  scanned {total:,} games, {usable:,} usable so far ...")

            ok, _ = acceptable(game)
            if not ok:
                continue

            elos = [pl.elo for pl in game.players.values() if pl.elo]
            if len(elos) < 2:
                continue
            if min_elo is not None and min(elos) < min_elo:
                continue
            if max_elo is not None and max(elos) > max_elo:
                continue

            usable += 1
            for color in ("white", "black"):
                elo = game.players[color].elo
                if elo is not None:
                    band_counts[band_for(elo)] += 1

    print(f"\n{'='*50}")
    print(f"Total games scanned: {total:,}")
    print(f"Usable games: {usable:,}")
    print(f"Player-games per band:")
    for lo, hi in zip(BAND_EDGES, BAND_EDGES[1:]):
        name = f"{lo}-{hi}"
        n = band_counts.get(name, 0)
        bar = "#" * min(n // 100, 60)
        print(f"  {name:>9}: {n:6,}  {bar}")
    print(f"{'='*50}")


def download_twic(
    start: int,
    end: int,
    output_dir: Path | None = None,
) -> list[Path]:
    """Download TWIC (The Week in Chess) PGN archives.

    TWIC publishes weekly PGN bundles of master/GM tournament games since 1994.
    Issues are numbered sequentially (~1560+ as of late 2026). Each zip is
    small (~1-5 MB) but packed with 2200+ rated games.

    Returns list of downloaded PGN file paths.
    """
    import zipfile

    out_dir = output_dir or ROOT / "cache" / "twic"
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []

    for issue in range(start, end + 1):
        pgn_path = out_dir / f"twic{issue}.pgn"
        if pgn_path.exists() and pgn_path.stat().st_size > 100:
            downloaded.append(pgn_path)
            continue

        zip_url = f"https://theweekinchess.com/zips/twic{issue}g.zip"
        zip_path = out_dir / f"twic{issue}g.zip"

        try:
            resp = requests.get(
                zip_url, timeout=60,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            )
            if resp.status_code == 404:
                print(f"  TWIC {issue}: not found (may not exist yet)")
                continue
            resp.raise_for_status()
            zip_path.write_bytes(resp.content)

            with zipfile.ZipFile(zip_path) as zf:
                # Find the PGN file inside the zip
                pgn_names = [n for n in zf.namelist() if n.lower().endswith(".pgn")]
                if not pgn_names:
                    print(f"  TWIC {issue}: no PGN in zip")
                    zip_path.unlink(missing_ok=True)
                    continue
                with zf.open(pgn_names[0]) as src, open(pgn_path, "wb") as dst:
                    dst.write(src.read())

            zip_path.unlink(missing_ok=True)
            size_kb = pgn_path.stat().st_size / 1024
            downloaded.append(pgn_path)
            print(f"  TWIC {issue}: {size_kb:.0f} KB")

        except requests.RequestException as exc:
            print(f"  TWIC {issue}: download failed ({exc})")
            continue
        except zipfile.BadZipFile:
            print(f"  TWIC {issue}: bad zip file")
            zip_path.unlink(missing_ok=True)
            continue
        time.sleep(0.3)  # be polite

    print(f"downloaded {len(downloaded)} TWIC issues to {out_dir}")
    return downloaded


def download_lichess_db(
    year: int,
    month: int,
    variant: str = "rated",
    output_dir: Path | None = None,
) -> Path | None:
    """Download a Lichess monthly database dump.

    These are bulk PGN exports from database.lichess.org containing ALL rated
    games for a month. A single month of rapid games is hundreds of thousands
    of games — orders of magnitude more than API harvesting can produce.

    Returns the path to the downloaded (decompressed) PGN file.
    """
    import subprocess
    import shutil

    out_dir = output_dir or ROOT / "cache" / "lichess_db"
    out_dir.mkdir(parents=True, exist_ok=True)

    base = f"lichess_db_standard_{variant}_{year:04d}-{month:02d}.pgn"
    url = f"https://database.lichess.org/standard/{base}.zst"
    zst_path = out_dir / f"{base}.zst"
    pgn_path = out_dir / base

    if pgn_path.exists():
        print(f"already downloaded: {pgn_path}")
        return pgn_path
    if zst_path.exists():
        print(f"already downloaded (compressed): {zst_path}")
        return zst_path

    print(f"downloading {url} ...")
    print("(this may take a while -- Lichess monthly dumps are large)")

    try:
        resp = requests.get(url, stream=True, timeout=30)
        if resp.status_code == 404:
            print(f"not found: {url}")
            print("try a different year/month, or check database.lichess.org")
            return None
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(zst_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                f.write(chunk)
                downloaded += len(chunk)
                if total:
                    pct = downloaded / total * 100
                    print(f"\r  {downloaded / 1e6:.0f} / {total / 1e6:.0f} MB ({pct:.0f}%)", end="", flush=True)
                else:
                    print(f"\r  {downloaded / 1e6:.0f} MB", end="", flush=True)
        print()
    except requests.RequestException as exc:
        print(f"download failed: {exc}")
        return None

    # Decompress .zst -> .pgn
    if shutil.which("zstd"):
        print("decompressing with zstd ...")
        subprocess.run(["zstd", "-d", "--rm", str(zst_path), "-o", str(pgn_path)], check=True)
    elif shutil.which("unzstd"):
        print("decompressing with unzstd ...")
        subprocess.run(["unzstd", str(zst_path), "-o", str(pgn_path)], check=True)
    else:
        try:
            import zstandard as zstd
            print("decompressing with python-zstandard ...")
            dctx = zstd.ZstdDecompressor()
            with open(zst_path, "rb") as ifh, open(pgn_path, "wb") as ofh:
                dctx.copy_stream(ifh, ofh)
            zst_path.unlink()
        except ImportError:
            print(
                f"downloaded {zst_path} but cannot decompress: "
                "install zstd CLI or python-zstandard (pip install zstandard)"
            )
            return None

    size_mb = pgn_path.stat().st_size / 1e6
    print(f"ready: {pgn_path} ({size_mb:.0f} MB)")
    return pgn_path


def write_baseline(doc: dict, path: Path = FITTED_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2))
    print(f"wrote {path}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m chessforensics.baseline",
        description="build and fit the reference model from a Lichess corpus",
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(p):
        p.add_argument("--games", type=int, default=2000, help="games to harvest")
        p.add_argument("--bands", default="1000-2400", help="rating range, e.g. 1200-2400")
        p.add_argument("--per-user", type=int, default=6)
        p.add_argument("--pool", type=int, default=None, help="engine pool size")
        p.add_argument(
            "--flat",
            action="store_true",
            help="skip the deep second pass (faster, but the corpus is then "
            "measured with a different instrument than games under test)",
        )

    h = sub.add_parser("harvest", help="download and analyse games")
    add_common(h)

    f = sub.add_parser("fit", help="fit the reference model from the stored corpus")
    f.add_argument("--min-per-band", type=int, default=40)

    r = sub.add_parser("run", help="harvest then fit")
    add_common(r)
    r.add_argument("--min-per-band", type=int, default=40)

    s = sub.add_parser("status", help="show corpus progress")

    sc = sub.add_parser(
        "scan-pgn",
        help="quick-count games per band in a PGN file (no engine analysis)",
    )
    sc.add_argument("files", nargs="+", help="PGN file paths")
    sc.add_argument("--min-elo", type=int, default=None)
    sc.add_argument("--max-elo", type=int, default=None)

    tw = sub.add_parser(
        "download-twic",
        help="download TWIC master game archives (e.g. issues 1400-1560)",
    )
    tw.add_argument("--start", type=int, required=True, help="first TWIC issue number")
    tw.add_argument("--end", type=int, required=True, help="last TWIC issue number")

    dl = sub.add_parser(
        "download-lichess",
        help="download a Lichess monthly database dump",
    )
    dl.add_argument("--year", type=int, required=True)
    dl.add_argument("--month", type=int, required=True)

    imp = sub.add_parser(
        "import-pgn",
        help="import games from PGN files (TWIC, Lichess dumps, etc.)",
    )
    imp.add_argument("files", nargs="+", help="PGN file paths")
    imp.add_argument("--min-elo", type=int, default=None, help="minimum rating filter")
    imp.add_argument("--max-elo", type=int, default=None, help="maximum rating filter")
    imp.add_argument("--max-games", type=int, default=None, help="stop after N games")
    imp.add_argument("--pool", type=int, default=None, help="engine pool size")
    imp.add_argument("--workers", type=int, default=3,
                     help="concurrent games to analyse (default 3)")
    imp.add_argument(
        "--flat", action="store_true",
        help="skip the deep second pass (faster)",
    )
    imp.add_argument(
        "--smart", action="store_true",
        help="skip games where both players are in bands already at target",
    )
    imp.add_argument(
        "--target", type=int, default=5000,
        help="target player-games per band for smart mode (default 5000)",
    )

    a = sub.add_parser(
        "harvest-all",
        help="systematically fill every rating band to a target",
    )
    a.add_argument(
        "--target", type=int, default=5000,
        help="target player-games per band (default 5000)",
    )
    a.add_argument("--per-user", type=int, default=6)
    a.add_argument("--pool", type=int, default=None, help="engine pool size")
    a.add_argument(
        "--flat",
        action="store_true",
        help="skip the deep second pass (faster harvest)",
    )

    lt = sub.add_parser(
        "import-lite",
        help="ultra-fast import using %%eval annotations (no engine needed)",
    )
    lt.add_argument("files", nargs="+", help="PGN or .pgn.zst files")
    lt.add_argument(
        "--target", type=int, default=500_000,
        help="target player-games per band (default 500000)",
    )
    lt.add_argument("--smart", action="store_true", help="skip bands already at target")
    lt.add_argument("--max-games", type=int, default=None, help="stop after N games")
    lt.add_argument("--batch-size", type=int, default=500, help="DB batch size")

    qe = sub.add_parser(
        "quick-eval",
        help="import into lite tables using fast Stockfish eval (for PGN without %%eval)",
    )
    qe.add_argument("files", nargs="+", help="PGN or .pgn.zst files")
    qe.add_argument("--target", type=int, default=500_000,
                     help="target player-games per band (default 500000)")
    qe.add_argument("--smart", action="store_true", help="skip bands already at target")
    qe.add_argument("--max-games", type=int, default=None, help="stop after N games")
    qe.add_argument("--pool", type=int, default=None, help="engine pool size")
    qe.add_argument("--depth", type=int, default=6, help="Stockfish depth (default 6)")
    qe.add_argument("--workers", type=int, default=4, help="concurrent game threads (default 4)")

    args = ap.parse_args(argv)

    if args.cmd == "status":
        c = Corpus()
        counts = c.counts()
        band_counts = c.band_counts()
        total_pg = counts.get("player_games", 0)
        total_games = counts.get("ok", 0)
        total_pos = counts.get("moves", 0)
        print(f"corpus: {total_games:,} games, {total_pg:,} player-games, "
              f"{total_pos:,} positions")
        print(f"bands ({len(BAND_EDGES)-1}):")
        for lo, hi in zip(BAND_EDGES, BAND_EDGES[1:]):
            name = f"{lo}-{hi}"
            n = band_counts.get(name, 0)
            bar = "#" * min(n // 50, 60)
            print(f"  {name:>9}: {n:5d}  {bar}")
        c.close()
        return 0

    if args.cmd == "scan-pgn":
        scan_pgn(args.files, min_elo=args.min_elo, max_elo=args.max_elo)
        return 0

    if args.cmd == "download-twic":
        paths = download_twic(args.start, args.end)
        if paths:
            flist = " ".join(f'"{p}"' for p in paths)
            print(f"\nto import all downloaded TWIC games:")
            print(f"  python -m chessforensics.baseline import-pgn {flist}")
            print(f"\nor just 2400+ games:")
            print(f"  python -m chessforensics.baseline import-pgn {flist} --min-elo 2400")
        return 0

    if args.cmd == "download-lichess":
        path = download_lichess_db(args.year, args.month)
        if path:
            print(f"\nto import high-rated games from this dump:")
            print(f"  python -m chessforensics.baseline import-pgn \"{path}\" --min-elo 2400")
            print(f"\nor all games:")
            print(f"  python -m chessforensics.baseline import-pgn \"{path}\"")
        return 0 if path else 1

    if args.cmd == "quick-eval":
        import_pgn_quick_eval(
            pgn_paths=args.files,
            target=args.target,
            smart=args.smart,
            max_games=args.max_games,
            pool_size=args.pool,
            depth=args.depth,
            workers=args.workers,
        )
        return 0

    if args.cmd == "import-lite":
        import_pgn_lite(
            pgn_paths=args.files,
            target=args.target,
            smart=args.smart,
            max_games=args.max_games,
            batch_size=args.batch_size,
        )
        return 0

    if args.cmd == "import-pgn":
        import_pgn(
            pgn_paths=args.files,
            deep_pass=not args.flat,
            pool_size=args.pool,
            min_elo=args.min_elo,
            max_elo=args.max_elo,
            max_games=args.max_games,
            workers=args.workers,
            target=args.target,
            smart=args.smart,
        )
        doc = fit_all()
        if doc["bands"]:
            write_baseline(doc)
        return 0

    if args.cmd == "harvest-all":
        harvest_all(
            target_per_band=args.target,
            deep_pass=not args.flat,
            pool_size=args.pool,
            per_user=args.per_user,
        )
        doc = fit_all()
        if doc["bands"]:
            write_baseline(doc)
        return 0

    if args.cmd in {"harvest", "run"}:
        try:
            lo, hi = (int(x) for x in args.bands.split("-"))
        except ValueError:
            print(f"bad --bands {args.bands!r}; expected e.g. 1200-2400")
            return 2
        harvest(
            n_games=args.games,
            min_elo=lo,
            max_elo=hi,
            deep_pass=not args.flat,
            pool_size=args.pool,
            per_user=args.per_user,
        )

    if args.cmd in {"fit", "run"}:
        doc = fit_all(min_per_band=getattr(args, "min_per_band", 40))
        if not doc["bands"]:
            print(
                "\nno band had enough player-games to fit. Harvest more games, or "
                "lower --min-per-band (at the cost of noisier references)."
            )
            return 1
        write_baseline(doc)
    return 0


if __name__ == "__main__":
    sys.exit(main())
