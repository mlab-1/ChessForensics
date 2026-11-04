"""OOS V2 validation using FRESH Chess.com games — lean M5-only scorer.

Same as oos_chesscom_m5.py but reads from cache/chesscom_pgns_v2/ and
writes to cache/oos_chesscom_v2_results.jsonl. These players have ZERO
overlap with the V1 set used for calibration.

Usage:
    py -3 -u -X utf8 scripts/oos_chesscom_v2.py                # full run
    py -3 -u -X utf8 scripts/oos_chesscom_v2.py --workers 8
    py -3 -u -X utf8 scripts/oos_chesscom_v2.py --stats
"""
import argparse
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chessforensics import score
from chessforensics.baselines import Baseline, band_for, load_baseline
from chessforensics.complexity import ComplexityWeights
from chessforensics.engine import AnalysisCache, EnginePool, analyse_game
from chessforensics.features import build_frame, frame_for
from chessforensics.ingest import games_from_pgn_text

PGN_DIR = ROOT / "cache" / "chesscom_pgns_v2"
RESULTS_FILE = ROOT / "cache" / "oos_chesscom_v2_results.jsonl"
PROGRESS_FILE = ROOT / "cache" / "oos_chesscom_v2_progress.json"
SUMMARY_FILE = ROOT / "cache" / "oos_chesscom_v2_summary.json"

MAX_GAMES_PER_PLAYER = 2
MIN_SCOREABLE = 8
MIN_ELO = 2100  # Only score players rated 2100+ (credible clean baseline)
STOCKFISH_EXE = "stockfish-windows-x86-64-bmi2.exe"


# ---------------------------------------------------------------------------
# Watchdog: kill orphan Stockfish processes
# ---------------------------------------------------------------------------

def _get_stockfish_pids() -> list[int]:
    """Get PIDs of all running Stockfish processes."""
    try:
        out = subprocess.check_output(
            ["tasklist", "/FI", f"IMAGENAME eq {STOCKFISH_EXE}", "/FO", "CSV", "/NH"],
            text=True, stderr=subprocess.DEVNULL,
        )
        pids = []
        for line in out.strip().splitlines():
            if STOCKFISH_EXE.lower() in line.lower():
                # CSV format: "name","pid","session","session#","mem"
                parts = line.split(",")
                if len(parts) >= 2:
                    pids.append(int(parts[1].strip('"')))
        return pids
    except Exception:
        return []


# PIDs of engines we know are ours (populated by EnginePool)
_our_pids: set[int] = set()
_our_pids_lock = threading.Lock()


def _start_watchdog(max_expected: int, interval: int = 60):
    """Background thread that kills orphan Stockfish when count is excessive.

    Uses a high threshold (4x workers) so it only fires in true runaway
    scenarios, not during normal transient spawns.
    """
    threshold = max_expected * 4  # very conservative — only panic mode

    def _loop():
        while not _watchdog_stop.is_set():
            pids = _get_stockfish_pids()
            if len(pids) > threshold:
                print(f"  [watchdog] {len(pids)} Stockfish (limit {threshold}), killing all...",
                      flush=True)
                for pid in pids:
                    try:
                        subprocess.run(
                            ["taskkill", "/F", "/PID", str(pid)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                        )
                    except Exception:
                        pass
            _watchdog_stop.wait(interval)

    _watchdog_stop.clear()
    t = threading.Thread(target=_loop, daemon=True)
    t.start()
    return t


_watchdog_stop = threading.Event()


# ---------------------------------------------------------------------------
# Progress tracking (thread-safe)
# ---------------------------------------------------------------------------

class Progress:
    def __init__(self):
        self.lock = threading.Lock()
        if PROGRESS_FILE.exists():
            data = json.loads(PROGRESS_FILE.read_text())
            self.done_players = set(data.get("done_players", []))
            self.n_games = data.get("n_games", 0)
            self.n_players = data.get("n_players", 0)
        else:
            self.done_players = set()
            self.n_games = 0
            self.n_players = 0
        self._results_fh = open(RESULTS_FILE, "a", encoding="utf-8")

    def is_done(self, username: str) -> bool:
        return username in self.done_players

    def record(self, username: str, results: list[dict]):
        with self.lock:
            for r in results:
                self._results_fh.write(json.dumps(r) + "\n")
            self._results_fh.flush()
            self.done_players.add(username)
            self.n_games += len(results)
            self.n_players += 1
            if self.n_players % 50 == 0:
                self._save()

    def _save(self):
        PROGRESS_FILE.write_text(json.dumps({
            "done_players": list(self.done_players),
            "n_games": self.n_games,
            "n_players": self.n_players,
        }))

    def close(self):
        with self.lock:
            self._save()
            self._results_fh.close()


# ---------------------------------------------------------------------------
# Per-player analysis — lean M5 path
# ---------------------------------------------------------------------------

_thread_local = threading.local()

# Shared across all workers — thread-safe (SQLite WAL mode)
_shared_cache: AnalysisCache | None = None
_shared_baseline: Baseline | None = None

HASH_MB = 32  # 32MB per engine — 8 workers × 32MB = 256MB total


def _get_pool():
    if not hasattr(_thread_local, "pool"):
        _thread_local.pool = EnginePool(size=1, hash_mb=HASH_MB)
    return _thread_local.pool


def analyse_player_m5(pgn_path: Path) -> list[dict]:
    """Analyse up to MAX_GAMES_PER_PLAYER games with lean M5 scoring."""
    username = pgn_path.stem
    text = pgn_path.read_text(encoding="utf-8", errors="replace")
    if not text.strip():
        return []

    all_games = games_from_pgn_text(text, source="chesscom")
    if not all_games:
        return []

    # Pre-filter: only keep games where this player is rated >= MIN_ELO
    eligible = []
    for g in all_games:
        for color in ("white", "black"):
            p = g.players.get(color)
            if p and (p.name or "").lower() == username.lower() and p.elo >= MIN_ELO:
                eligible.append(g)
                break
    if not eligible:
        return []

    if len(eligible) > MAX_GAMES_PER_PLAYER:
        eligible = random.sample(eligible, MAX_GAMES_PER_PLAYER)

    pool = _get_pool()
    baseline = _shared_baseline
    cache = _shared_cache
    weights = ComplexityWeights.from_dict(baseline.complexity_weights)
    results = []

    for game in eligible:
        try:
            fens = [m.fen_before for m in game.moves]
            ucis = [m.uci for m in game.moves]

            # Stockfish analysis — no L0, no deep pass, no second engine
            ga = analyse_game(
                fens, ucis, pool, cache,
                deep_pass=False, run_l0=False,
            )

            # Build feature frame — skip Lichess API, use move_number heuristic
            df = build_frame(game, ga, weights, book_plies=None)
            if df.empty:
                continue

            for color in ("white", "black"):
                sub = frame_for(df, color)
                if len(sub) < MIN_SCOREABLE:
                    continue

                # Check if this side is our target player
                player = game.players.get(color)
                if player is None:
                    continue
                pname = (player.name or "").lower()
                if pname != username.lower():
                    continue

                elo = player.elo
                if elo < MIN_ELO:
                    continue
                band_name = band_for(elo)

                # Score with M5 directly — no old signals needed
                move_dicts = sub.to_dict("records")
                verdict = score.combine_permove(
                    move_dicts,
                    band_name=band_name,
                    prior=score.DEFAULT_PRIOR,
                    uncalibrated=baseline.is_default,
                    baseline_n_games=baseline.n_games,
                    n_mc=0,  # skip MC — only need point estimate for OOS
                    # NO baseline= here: we want RAW scores for calibration
                )

                results.append({
                    "username": username,
                    "color": color,
                    "elo": elo,
                    "band": band_name,
                    "n_scoreable": len(sub),
                    "posterior": verdict.posterior,
                    "decibans": verdict.decibans,
                    "log_lr": verdict.log_lr,
                    "speed": game.speed or "",
                    "log_lrs_by_rate": verdict.log_lrs_by_rate,
                })

        except Exception:
            continue

    return results


# ---------------------------------------------------------------------------
# Main parallel runner
# ---------------------------------------------------------------------------

def run(workers: int = 12):
    global _shared_cache, _shared_baseline

    # Shared resources — initialized once, used by all workers
    _shared_cache = AnalysisCache()
    _shared_baseline = load_baseline()

    pgn_files = sorted(PGN_DIR.glob("*.pgn"))
    print(f"\n=== OOS VALIDATION: Chess.com games (M5 lean) ===")
    print(f"  PGN files: {len(pgn_files)}")
    print(f"  Max games/player: {MAX_GAMES_PER_PLAYER}")
    print(f"  Workers: {workers}")
    print(f"  Hash/engine: {HASH_MB}MB (total ~{workers * HASH_MB}MB)")
    print(f"  Cache: {_shared_cache.path}")

    _start_watchdog(max_expected=workers)

    progress = Progress()
    remaining = [p for p in pgn_files if not progress.is_done(p.stem)]
    print(f"  Already done: {progress.n_players} players, {progress.n_games} games")
    print(f"  Remaining: {len(remaining)} players\n", flush=True)

    completed = 0
    t0 = time.time()

    BATCH = workers * 8
    for batch_start in range(0, len(remaining), BATCH):
        batch = remaining[batch_start:batch_start + BATCH]

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(analyse_player_m5, p): p for p in batch}

            for future in as_completed(futures):
                pgn_path = futures[future]
                username = pgn_path.stem
                completed += 1

                try:
                    results = future.result()
                except Exception:
                    results = []

                progress.record(username, results)

        elapsed = time.time() - t0
        rate = completed / elapsed if elapsed > 0 else 0
        eta_min = (len(remaining) - completed) / rate / 60 if rate > 0 else 0
        ch = _shared_cache.hits + _shared_cache.misses
        hit_pct = _shared_cache.hits / ch * 100 if ch > 0 else 0
        print(f"  [{completed}/{len(remaining)}] "
              f"players={progress.n_players} games={progress.n_games} "
              f"cache={hit_pct:.0f}% "
              f"rate={rate:.2f}/s ETA={eta_min:.0f}min", flush=True)

    progress.close()
    _watchdog_stop.set()
    elapsed = time.time() - t0
    print(f"\n=== DONE ({elapsed / 60:.1f} min) ===")
    print(f"  Players: {progress.n_players}")
    print(f"  Games scored: {progress.n_games}")
    compute_summary()


# ---------------------------------------------------------------------------
# Summary stats
# ---------------------------------------------------------------------------

def compute_summary():
    if not RESULTS_FILE.exists():
        print("No results file found.")
        return

    import numpy as np
    from collections import Counter, defaultdict

    rows = []
    with open(RESULTS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    if not rows:
        print("No results.")
        return

    posteriors = np.array([r["posterior"] for r in rows])
    dbs = np.array([r["decibans"] for r in rows])
    players = set(r["username"] for r in rows)

    # Per-player aggregation (sum dB per player)
    by_player = defaultdict(list)
    for r in rows:
        by_player[r["username"]].append(r)

    player_posteriors = []
    player_dbs = []
    for username, games in by_player.items():
        total_db = sum(g["decibans"] for g in games)
        log_lr = total_db * math.log(10) / 10.0
        prior = 0.01
        logit_prior = math.log(prior / (1 - prior))
        post = 1.0 / (1.0 + math.exp(-(logit_prior + log_lr)))
        player_posteriors.append(post)
        player_dbs.append(total_db)

    player_posteriors = np.array(player_posteriors)
    player_dbs = np.array(player_dbs)

    fpr_50 = float((posteriors > 0.50).mean())
    fpr_5 = float((posteriors > 0.05).mean())
    max_post = float(posteriors.max())
    median_db = float(np.median(dbs))

    pfpr_50 = float((player_posteriors > 0.50).mean())
    pfpr_5 = float((player_posteriors > 0.05).mean())
    pmax_post = float(player_posteriors.max())

    speeds = Counter(r.get("speed", "?") for r in rows)
    bands = Counter(r.get("band", "?") for r in rows)

    summary = {
        "n_games": len(rows),
        "n_players": len(players),
        "per_game": {
            "fpr_50pct": fpr_50,
            "fpr_5pct": fpr_5,
            "max_posterior": max_post,
            "median_db": median_db,
            "mean_db": float(dbs.mean()),
            "pct_negative_db": float((dbs < 0).mean()),
        },
        "per_player": {
            "fpr_50pct": pfpr_50,
            "fpr_5pct": pfpr_5,
            "max_posterior": pmax_post,
            "median_db": float(np.median(player_dbs)),
        },
        "speeds": dict(speeds.most_common()),
        "bands": dict(sorted(bands.items())),
    }

    SUMMARY_FILE.write_text(json.dumps(summary, indent=2))

    print(f"\n{'='*60}")
    print(f"OOS VALIDATION SUMMARY (M5 per-move LR)")
    print(f"{'='*60}")
    print(f"  Games:   {len(rows)}")
    print(f"  Players: {len(players)}")
    print(f"\n  --- Per-Game ---")
    print(f"  FPR at 50% threshold: {fpr_50:.4%}")
    print(f"  FPR at  5% threshold: {fpr_5:.4%}")
    print(f"  Max posterior:        {max_post:.4%}")
    print(f"  Median dB:           {median_db:+.1f}")
    print(f"  Pct negative dB:     {(dbs < 0).mean():.1%}")
    print(f"\n  --- Per-Player (aggregate) ---")
    print(f"  FPR at 50% threshold: {pfpr_50:.4%}")
    print(f"  FPR at  5% threshold: {pfpr_5:.4%}")
    print(f"  Max posterior:        {pmax_post:.4%}")
    print(f"  Median dB:           {np.median(player_dbs):+.1f}")
    print(f"\n  --- Speeds ---")
    for s, c in speeds.most_common():
        print(f"    {s}: {c}")
    print(f"\n  --- Bands ---")
    for b, c in sorted(bands.items()):
        print(f"    {b}: {c}")
    print(f"\n  95% CI upper bound on FPR (per-player):")
    print(f"    3/{len(players)} = {3/len(players):.6%}")


def print_stats():
    if PROGRESS_FILE.exists():
        d = json.loads(PROGRESS_FILE.read_text())
        print(f"Progress: {d['n_players']} players, {d['n_games']} games")
    compute_summary()


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="OOS validation — M5 lean")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    if args.stats:
        print_stats()
        return

    run(workers=args.workers)


if __name__ == "__main__":
    main()
