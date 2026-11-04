"""Fetch a FRESH set of Chess.com players for M5 calibration validation.

These players must NOT overlap with the existing OOS set (cache/chesscom_pgns/).
Output goes to cache/chesscom_pgns_v2/ so the two sets stay separate.

Strategy:
  1. Load the existing player list (already fetched/failed) as an exclusion set
  2. Re-discover titled players (the list may have grown)
  3. BFS from NEW titled players to find strong untitled opponents
  4. Fetch rapid games for the new players only

Usage:
    py -3 -u -X utf8 scripts/fetch_chesscom_v2.py                # full run
    py -3 -u -X utf8 scripts/fetch_chesscom_v2.py --discover      # discover only
    py -3 -u -X utf8 scripts/fetch_chesscom_v2.py --fetch          # fetch only
    py -3 -u -X utf8 scripts/fetch_chesscom_v2.py --stats
    py -3 -u -X utf8 scripts/fetch_chesscom_v2.py --target 5000
"""
import argparse
import json
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CACHE_DIR = ROOT / "cache"
CACHE_DIR.mkdir(exist_ok=True)

# V1 files (exclusion set)
V1_PROGRESS = CACHE_DIR / "chesscom_fetch_progress.json"
V1_PGN_DIR = CACHE_DIR / "chesscom_pgns"

# V2 files (new set)
V2_PLAYERS_FILE = CACHE_DIR / "chesscom_v2_players.json"
V2_PROGRESS_FILE = CACHE_DIR / "chesscom_v2_progress.json"
V2_PGN_DIR = CACHE_DIR / "chesscom_pgns_v2"

GAMES_PER_PLAYER = 15
MIN_ELO = 2100
TITLES = ["GM", "IM", "FM", "WGM", "WIM", "WFM", "CM", "WCM", "NM"]

BASE = "https://api.chess.com/pub"
_HEADERS = {
    "User-Agent": "ChessForensics OOS-v2 validation (academic research)",
    "Accept": "application/json",
}

SESSION = requests.Session()
SESSION.headers.update(_HEADERS)

_thread_local = threading.local()


def _get_session() -> requests.Session:
    if not hasattr(_thread_local, "session"):
        s = requests.Session()
        s.headers.update(_HEADERS)
        _thread_local.session = s
    return _thread_local.session


class RateLimiter:
    def __init__(self, calls_per_sec: float = 25.0):
        self.interval = 1.0 / calls_per_sec
        self.lock = threading.Lock()
        self.next_slot = time.monotonic()

    def wait(self):
        with self.lock:
            now = time.monotonic()
            if self.next_slot <= now:
                self.next_slot = now + self.interval
                return
            my_slot = self.next_slot
            self.next_slot = my_slot + self.interval
        delay = my_slot - time.monotonic()
        if delay > 0:
            time.sleep(delay)


RATE = RateLimiter(calls_per_sec=25.0)


def _load_v1_exclusion() -> set[str]:
    """Load all usernames from the V1 set (done + failed + existing PGNs)."""
    excluded = set()
    if V1_PROGRESS.exists():
        prog = json.loads(V1_PROGRESS.read_text())
        for u in prog.get("players_done", []):
            excluded.add(u.lower())
        for u in prog.get("players_failed", []):
            excluded.add(u.lower())
    if V1_PGN_DIR.exists():
        for p in V1_PGN_DIR.glob("*.pgn"):
            excluded.add(p.stem.lower())
    return excluded


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_new_players(target: int = 10000) -> list[dict]:
    """Find titled + strong untitled players NOT in V1."""
    excluded = _load_v1_exclusion()
    print(f"\n=== V2 DISCOVERY ===")
    print(f"  V1 exclusion set: {len(excluded)} players")

    # Phase 1: All titled players
    print(f"\n  Fetching titled players...")
    all_titled = []
    for title in TITLES:
        try:
            RATE.wait()
            r = SESSION.get(f"{BASE}/titled/{title}", timeout=30)
            r.raise_for_status()
            usernames = r.json().get("players", [])
            new = [u for u in usernames if u.lower() not in excluded]
            for u in new:
                all_titled.append({"username": u, "title": title})
            print(f"    {title}: {len(usernames)} total, {len(new)} new")
            time.sleep(0.3)
        except Exception as e:
            print(f"    {title}: FAILED {e}")
            time.sleep(2)

    print(f"  New titled players: {len(all_titled)}")

    # Phase 2: BFS from new titled players to discover strong untitled opponents
    print(f"\n  BFS for strong untitled opponents...")
    seen = excluded | {p["username"].lower() for p in all_titled}
    untitled = []

    seeds = [p["username"] for p in all_titled[:300]]
    queue = list(seeds)
    visited = set()
    wave = 0
    max_waves = 30

    while queue and wave < max_waves and (len(all_titled) + len(untitled)) < target:
        wave += 1
        batch = queue[:80]
        queue = queue[80:]
        new_this_wave = 0

        for username in batch:
            if username.lower() in visited:
                continue
            visited.add(username.lower())

            try:
                RATE.wait()
                r = SESSION.get(
                    f"{BASE}/player/{username}/games/archives", timeout=10)
                if not r.ok:
                    continue
                archives = r.json().get("archives", [])
                if not archives:
                    continue

                RATE.wait()
                r2 = SESSION.get(archives[-1], timeout=20)
                if not r2.ok:
                    continue

                for game in r2.json().get("games", []):
                    tc = game.get("time_class", "")
                    if tc != "rapid":
                        continue
                    for color in ["white", "black"]:
                        p = game.get(color, {})
                        u = p.get("username", "")
                        rating = p.get("rating", 0)
                        if (u and u.lower() not in seen and rating >= MIN_ELO):
                            seen.add(u.lower())
                            untitled.append({
                                "username": u, "title": "", "rating": rating})
                            if u.lower() not in visited:
                                queue.append(u)
                            new_this_wave += 1
            except Exception:
                continue

        print(f"    Wave {wave}: +{new_this_wave} "
              f"(total new: {len(all_titled) + len(untitled)}, "
              f"queue: {len(queue)})")

        if new_this_wave == 0 and wave > 5:
            break

    all_new = all_titled + untitled
    print(f"\n  === V2 DISCOVERY COMPLETE ===")
    print(f"  New titled: {len(all_titled)}")
    print(f"  New untitled (>= {MIN_ELO}): {len(untitled)}")
    print(f"  Total new: {len(all_new)}")

    V2_PLAYERS_FILE.write_text(json.dumps(all_new, indent=1))
    print(f"  Saved to {V2_PLAYERS_FILE}")
    return all_new


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def load_progress() -> dict:
    if V2_PROGRESS_FILE.exists():
        return json.loads(V2_PROGRESS_FILE.read_text())
    return {"players_done": [], "players_failed": [], "total_games": 0}


def save_progress(data: dict):
    V2_PROGRESS_FILE.write_text(json.dumps(data, indent=1))


def fetch_player_games(username: str, max_games: int = GAMES_PER_PLAYER) -> str:
    sess = _get_session()
    RATE.wait()
    r = sess.get(f"{BASE}/player/{username}/games/archives", timeout=15)
    if r.status_code in (404, 410):
        return ""
    if r.status_code == 429:
        time.sleep(5)
        RATE.wait()
        r = sess.get(f"{BASE}/player/{username}/games/archives", timeout=15)
    r.raise_for_status()
    archives = r.json().get("archives", [])
    if not archives:
        return ""

    pgn_parts = []
    game_count = 0

    for archive_url in reversed(archives[-6:]):
        if game_count >= max_games:
            break
        try:
            RATE.wait()
            r2 = sess.get(archive_url, timeout=30)
            if r2.status_code == 429:
                time.sleep(5)
                RATE.wait()
                r2 = sess.get(archive_url, timeout=30)
            if not r2.ok:
                continue

            for game in r2.json().get("games", []):
                if game_count >= max_games:
                    break
                tc = game.get("time_class", "")
                if tc != "rapid":
                    continue
                if not game.get("rated", True):
                    continue
                pgn = game.get("pgn", "")
                if pgn:
                    pgn_parts.append(pgn)
                    game_count += 1
        except Exception:
            continue

    return "\n\n".join(pgn_parts)


def _fetch_one(username: str) -> tuple[str, str, int]:
    for attempt in range(3):
        try:
            pgn_text = fetch_player_games(username)
            if not pgn_text.strip():
                return (username, "", 0)
            n_games = pgn_text.count("[Event ")
            return (username, pgn_text, n_games)
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower():
                time.sleep(15 * (attempt + 1))
            else:
                return (username, "", -1)
    return (username, "", -1)


def run_fetch(players: list[dict], workers: int = 15):
    V2_PGN_DIR.mkdir(exist_ok=True)
    print(f"\n=== V2 FETCH ({len(players)} players, {workers} workers) ===\n",
          flush=True)

    progress = load_progress()
    done = set(progress["players_done"])
    failed = set(progress["players_failed"])
    remaining = [p for p in players
                 if p["username"] not in done and p["username"] not in failed]

    print(f"  Already done: {len(done)}", flush=True)
    print(f"  Remaining: {len(remaining)}", flush=True)

    completed = 0
    t0 = time.time()
    BATCH = 80

    for batch_start in range(0, len(remaining), BATCH):
        batch = remaining[batch_start:batch_start + BATCH]

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_fetch_one, p["username"]) for p in batch]
            for future in as_completed(futures):
                username, pgn_text, n_games = future.result()
                completed += 1

                if n_games == -1:
                    progress["players_failed"].append(username)
                    failed.add(username)
                else:
                    progress["players_done"].append(username)
                    done.add(username)
                    if n_games > 0:
                        pgn_path = V2_PGN_DIR / f"{username}.pgn"
                        pgn_path.write_text(pgn_text, encoding="utf-8")
                        progress["total_games"] += n_games

        save_progress(progress)
        elapsed = time.time() - t0
        rate = completed / elapsed if elapsed > 0 else 0
        eta = (len(remaining) - completed) / rate / 60 if rate > 0 else 0
        print(f"  [{completed}/{len(remaining)}] "
              f"done={len(done)} games={progress['total_games']} "
              f"rate={rate:.1f}/s ETA={eta:.0f}min", flush=True)

    save_progress(progress)
    print(f"\n=== V2 FETCH COMPLETE ===", flush=True)
    print(f"  Players: {len(done)}", flush=True)
    print(f"  Games: {progress['total_games']}", flush=True)
    print(f"  PGN dir: {V2_PGN_DIR}", flush=True)


def print_stats():
    excluded = _load_v1_exclusion()
    print(f"V1 exclusion set: {len(excluded)}")

    if V2_PLAYERS_FILE.exists():
        players = json.loads(V2_PLAYERS_FILE.read_text())
        tc = Counter(p["title"] for p in players if p["title"])
        untitled = sum(1 for p in players if not p["title"])
        print(f"V2 discovered: {len(players)}")
        print(f"  Titled: {sum(tc.values())}")
        for t in TITLES:
            if tc[t]:
                print(f"    {t}: {tc[t]}")
        print(f"  Untitled (>= {MIN_ELO}): {untitled}")

    progress = load_progress()
    print(f"V2 fetch progress:")
    print(f"  Done: {len(progress['players_done'])}")
    print(f"  Failed: {len(progress['players_failed'])}")
    print(f"  Games: {progress['total_games']}")

    if V2_PGN_DIR.exists():
        print(f"  PGN files: {len(list(V2_PGN_DIR.glob('*.pgn')))}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch fresh Chess.com players for V2 calibration validation")
    parser.add_argument("--discover", action="store_true")
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--stats", action="store_true")
    parser.add_argument("--target", type=int, default=10000,
                        help="Target number of new players to discover")
    parser.add_argument("--workers", type=int, default=15)
    args = parser.parse_args()

    if args.stats:
        print_stats()
        return

    if args.discover:
        discover_new_players(target=args.target)
        return

    if args.fetch:
        if not V2_PLAYERS_FILE.exists():
            print("No V2 discovery file. Run --discover first.")
            return
        players = json.loads(V2_PLAYERS_FILE.read_text())
        run_fetch(players, workers=args.workers)
        return

    # Default: full pipeline
    players = discover_new_players(target=args.target)
    run_fetch(players, workers=args.workers)


if __name__ == "__main__":
    main()
