"""Normalise every game source into a single Game dataclass.

Sources: raw PGN text, PGN file, chess.com monthly archive API, Lichess user
export (ndjson), and a pasted game URL for either site.

The important subtlety here is clock handling. ``think_times`` is ``None`` when
the game carries no clock data at all, and individual entries are ``None`` when
a specific ply's think time cannot be computed (notably the first move of each
side, where there is no previous clock reading to subtract from). Downstream,
the timing module must be able to distinguish "no data" from "zero seconds" --
collapsing those two into 0.0 would invent robotic instant-move behaviour where
there is simply nothing recorded.
"""

from __future__ import annotations

import io
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Literal

import chess
import chess.pgn
import requests
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

USER_AGENT = (
    "ChessForensics/1.0 (local single-game analysis tool; "
    "contact: local user; https://github.com/local/chessforensics)"
)

Color = Literal["white", "black"]


class RateLimited(Exception):
    """HTTP 429 from an upstream API."""


class IngestError(Exception):
    """A game could not be fetched or parsed."""


@dataclass
class Player:
    name: str
    elo: int | None = None
    title: str | None = None


@dataclass
class Move:
    ply: int  # 0-based index into the move list
    san: str
    uci: str
    fen_before: str
    color: Color
    move_number: int  # standard chess move number (1-based)
    clock: float | None = None  # seconds remaining AFTER this move
    think_time: float | None = None  # seconds spent on this move, None = unknown
    eval_comment: float | None = None  # [%eval ...] if the source provided one


@dataclass
class Game:
    headers: dict[str, str]
    players: dict[Color, Player]
    moves: list[Move]
    time_control: str | None
    base_time: int | None  # seconds
    increment: int | None  # seconds
    source: str
    result: str | None = None
    url: str | None = None
    has_clocks: bool = False
    speed: str | None = None  # bullet / blitz / rapid / classical / correspondence
    raw_pgn: str | None = None
    tos_flagged: dict[Color, bool] = field(default_factory=dict)

    @property
    def think_times(self) -> dict[Color, list[float | None]] | None:
        """Per-colour think times in play order, or None if no clock data."""
        if not self.has_clocks:
            return None
        out: dict[Color, list[float | None]] = {"white": [], "black": []}
        for mv in self.moves:
            out[mv.color].append(mv.think_time)
        return out

    def moves_for(self, color: Color) -> list[Move]:
        return [m for m in self.moves if m.color == color]

    def __len__(self) -> int:
        return len(self.moves)


# --------------------------------------------------------------------------
# clock comment parsing
# --------------------------------------------------------------------------

_CLK_RE = re.compile(r"\[%clk\s+(\d+):(\d+):(\d+(?:\.\d+)?)\]")
_CLK_SHORT_RE = re.compile(r"\[%clk\s+(\d+):(\d+(?:\.\d+)?)\]")
_EVAL_RE = re.compile(r"\[%eval\s+(#?-?[\d.]+)\]")
_EMT_RE = re.compile(r"\[%emt\s+(\d+):(\d+):(\d+(?:\.\d+)?)\]")


def parse_clock(comment: str) -> float | None:
    """Extract seconds remaining from a ``[%clk h:mm:ss(.f)]`` comment.

    Also accepts the ``m:ss`` short form some exporters emit. Returns None when
    no clock annotation is present -- callers must not substitute 0.
    """
    if not comment:
        return None
    m = _CLK_RE.search(comment)
    if m:
        h, mm, ss = m.groups()
        return int(h) * 3600 + int(mm) * 60 + float(ss)
    m = _CLK_SHORT_RE.search(comment)
    if m:
        mm, ss = m.groups()
        return int(mm) * 60 + float(ss)
    return None


def parse_emt(comment: str) -> float | None:
    """Extract elapsed move time from a ``[%emt h:mm:ss]`` comment, if any."""
    if not comment:
        return None
    m = _EMT_RE.search(comment)
    if not m:
        return None
    h, mm, ss = m.groups()
    return int(h) * 3600 + int(mm) * 60 + float(ss)


def parse_eval(comment: str) -> float | None:
    """Extract a ``[%eval]`` score in pawns; mate scores map to +-10."""
    if not comment:
        return None
    m = _EVAL_RE.search(comment)
    if not m:
        return None
    raw = m.group(1)
    if raw.startswith("#"):
        try:
            n = int(raw[1:])
        except ValueError:
            return None
        return 10.0 if n > 0 else -10.0
    try:
        return float(raw)
    except ValueError:
        return None


def parse_time_control(tc: str | None) -> tuple[int | None, int | None]:
    """Parse a PGN TimeControl tag into (base_seconds, increment_seconds)."""
    if not tc or tc in {"-", "?"}:
        return None, None
    # Multi-stage controls (e.g. "40/7200:1800") -- use the first stage base.
    first = tc.split(":")[0]
    if "/" in first:
        first = first.split("/", 1)[1]
    m = re.match(r"^(\d+)(?:\+(\d+))?$", first.strip())
    if not m:
        return None, None
    base = int(m.group(1))
    inc = int(m.group(2)) if m.group(2) else 0
    return base, inc


def classify_speed(base: int | None, inc: int | None) -> str | None:
    """Lichess-style speed bucket from base + increment.

    Used to down-weight the timing module: in bullet and blitz, premoves and
    network lag make near-zero move times both common and innocent.
    """
    if base is None:
        return None
    estimate = base + 40 * (inc or 0)
    if estimate < 30:
        return "ultrabullet"
    if estimate < 180:
        return "bullet"
    if estimate < 480:
        return "blitz"
    if estimate < 1500:
        return "rapid"
    if estimate < 21600:
        return "classical"
    return "correspondence"


# --------------------------------------------------------------------------
# PGN
# --------------------------------------------------------------------------


def _compute_think_times(moves: list[Move], increment: int | None, base: int | None) -> None:
    """Fill ``think_time`` in place from consecutive clock readings.

    think_time[i] = clock[i-2] - clock[i] + increment   (same colour, so i-2)

    The first move of each colour has no preceding reading. If the game's base
    time is known we can still derive it (base - clock + increment); otherwise
    it stays None. An explicit ``[%emt]`` annotation always wins when present.
    """
    prev_clock: dict[Color, float | None] = {"white": None, "black": None}
    inc = increment or 0

    for mv in moves:
        if mv.think_time is not None:  # came from [%emt]
            prev_clock[mv.color] = mv.clock
            continue
        if mv.clock is None:
            prev_clock[mv.color] = None
            continue

        previous = prev_clock[mv.color]
        if previous is None:
            # First recorded move for this colour. Only derivable if we know
            # the starting time, and only trustworthy if the clock has not
            # already been through an unrecorded move.
            if base is not None and mv.move_number <= 1:
                spent = base - mv.clock + inc
                mv.think_time = max(0.0, spent)
        else:
            spent = previous - mv.clock + inc
            # Negative spend means the exporter's increment handling differs
            # from ours (some add the increment before writing the clock).
            # Retry without the increment before giving up on the value.
            if spent < 0:
                spent = previous - mv.clock
            mv.think_time = spent if spent >= 0 else None
        prev_clock[mv.color] = mv.clock


def game_from_pgn_node(node: chess.pgn.Game, source: str = "pgn") -> Game:
    """Convert a python-chess Game node into our Game dataclass."""
    headers = dict(node.headers)
    tc = headers.get("TimeControl")
    base, inc = parse_time_control(tc)

    def _elo(key: str) -> int | None:
        raw = headers.get(key, "")
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    players: dict[Color, Player] = {
        "white": Player(headers.get("White", "White"), _elo("WhiteElo"), headers.get("WhiteTitle")),
        "black": Player(headers.get("Black", "Black"), _elo("BlackElo"), headers.get("BlackTitle")),
    }

    moves: list[Move] = []
    board = node.board()
    ply = 0
    for child in node.mainline():
        move = child.move
        if move is None:
            continue
        fen_before = board.fen()
        try:
            san = board.san(move)
        except (ValueError, AssertionError):
            # Illegal move in the PGN: stop rather than corrupt the line.
            break
        color: Color = "white" if board.turn == chess.WHITE else "black"
        comment = child.comment or ""
        moves.append(
            Move(
                ply=ply,
                san=san,
                uci=move.uci(),
                fen_before=fen_before,
                color=color,
                move_number=board.fullmove_number,
                clock=parse_clock(comment),
                think_time=parse_emt(comment),
                eval_comment=parse_eval(comment),
            )
        )
        board.push(move)
        ply += 1

    has_clocks = any(m.clock is not None for m in moves) or any(
        m.think_time is not None for m in moves
    )
    _compute_think_times(moves, inc, base)

    # Lichess monthly dumps use "Site" for the game URL, while the JSON API
    # puts it in "Link". Accept either so PGN-imported games get a URL too.
    url = headers.get("Link")
    if not url:
        site = headers.get("Site", "")
        if "lichess.org/" in site:
            url = site

    return Game(
        headers=headers,
        players=players,
        moves=moves,
        time_control=tc,
        base_time=base,
        increment=inc,
        source=source,
        result=headers.get("Result"),
        url=url,
        has_clocks=has_clocks,
        speed=classify_speed(base, inc),
        raw_pgn=str(node),
    )


def games_from_pgn_text(text: str, source: str = "pgn") -> list[Game]:
    """Parse every game in a PGN string."""
    stream = io.StringIO(text)
    out: list[Game] = []
    while True:
        node = chess.pgn.read_game(stream)
        if node is None:
            break
        if not list(node.mainline_moves()):
            continue
        out.append(game_from_pgn_node(node, source))
    if not out:
        raise IngestError("no games with moves found in PGN")
    return out


def games_from_pgn_file(path: str | Path) -> list[Game]:
    data = Path(path).read_text(encoding="utf-8", errors="replace")
    return games_from_pgn_text(data, source=f"file:{Path(path).name}")


def iter_pgn_file(path: str | Path, source: str = "pgn") -> Iterator[Game]:
    """Stream games from a (possibly huge) PGN file without loading it all."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        while True:
            node = chess.pgn.read_game(fh)
            if node is None:
                break
            if not list(node.mainline_moves()):
                continue
            yield game_from_pgn_node(node, source)


# --------------------------------------------------------------------------
# HTTP plumbing
# --------------------------------------------------------------------------


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    return s


@retry(
    retry=retry_if_exception_type((RateLimited, requests.ConnectionError, requests.Timeout)),
    wait=wait_exponential(multiplier=4, min=4, max=90),
    stop=stop_after_attempt(5),
    reraise=True,
)
def _get(url: str, session: requests.Session, **kwargs) -> requests.Response:
    resp = session.get(url, timeout=kwargs.pop("timeout", 60), **kwargs)
    if resp.status_code == 429:
        # Lichess asks for a full minute of silence after a 429. Honour it:
        # hammering it earns a longer ban than waiting does.
        raise RateLimited(f"429 from {url}")
    if resp.status_code == 403:
        raise IngestError(
            f"403 from {url} -- the API rejected the request. chess.com requires a "
            "descriptive User-Agent header; check it is being sent."
        )
    resp.raise_for_status()
    return resp


# --------------------------------------------------------------------------
# chess.com
# --------------------------------------------------------------------------

CHESSCOM_API = "https://api.chess.com/pub"


def chesscom_archives(username: str, session: requests.Session | None = None) -> list[str]:
    s = session or _session()
    url = f"{CHESSCOM_API}/player/{username.lower()}/games/archives"
    return _get(url, s).json().get("archives", [])


def chesscom_month(
    username: str,
    year: int,
    month: int,
    session: requests.Session | None = None,
) -> list[Game]:
    """Fetch one monthly archive. chess.com 403s without a real User-Agent."""
    s = session or _session()
    url = f"{CHESSCOM_API}/player/{username.lower()}/games/{year:04d}/{month:02d}"
    payload = _get(url, s).json()
    out: list[Game] = []
    for entry in payload.get("games", []):
        pgn = entry.get("pgn")
        if not pgn:
            continue
        try:
            games = games_from_pgn_text(pgn, source="chess.com")
        except IngestError:
            continue
        for g in games:
            g.url = entry.get("url") or g.url
            g.speed = entry.get("time_class") or g.speed
            for color in ("white", "black"):
                side = entry.get(color) or {}
                if side.get("rating"):
                    g.players[color].elo = int(side["rating"])
                if side.get("username"):
                    g.players[color].name = side["username"]
            out.append(g)
    return out


def chesscom_recent(username: str, max_games: int = 20) -> list[Game]:
    """Most recent games, walking archives backwards from the newest month."""
    s = _session()
    archives = chesscom_archives(username, s)
    out: list[Game] = []
    for url in reversed(archives):
        m = re.search(r"/(\d{4})/(\d{2})$", url)
        if not m:
            continue
        out.extend(chesscom_month(username, int(m.group(1)), int(m.group(2)), s))
        if len(out) >= max_games:
            break
    out.sort(key=lambda g: g.headers.get("UTCDate", "") + g.headers.get("UTCTime", ""))
    return out[-max_games:][::-1]


def chesscom_game(game_id: str) -> Game:
    """Fetch a single chess.com game by numeric id.

    chess.com has no public single-game endpoint, so this resolves the game via
    its callback API to learn the players and month, then pulls that archive.
    """
    s = _session()
    resp = s.get(f"https://www.chess.com/callback/live/game/{game_id}", timeout=60)
    if resp.status_code != 200:
        resp = s.get(f"https://www.chess.com/callback/daily/game/{game_id}", timeout=60)
    if resp.status_code != 200:
        raise IngestError(
            f"could not resolve chess.com game {game_id}. "
            "Paste the PGN instead (Share -> PGN on the game page)."
        )
    data = resp.json()
    game = data.get("game", {})
    pgn = game.get("moveList") and data.get("pgnHeaders")
    players = data.get("players") or {}
    white = (players.get("top") or {}), (players.get("bottom") or {})
    username = None
    for cand in white:
        if cand.get("username"):
            username = cand["username"]
            break
    end = game.get("endTime") or game.get("lastMoveTime")
    if not username or not end:
        raise IngestError(
            f"chess.com game {game_id} returned an unexpected payload; paste the PGN instead"
        )
    import datetime as _dt

    when = _dt.datetime.fromtimestamp(end, _dt.timezone.utc)
    for y, mo in ((when.year, when.month), _prev_month(when.year, when.month)):
        for g in chesscom_month(username, y, mo, s):
            if g.url and g.url.rstrip("/").endswith(str(game_id)):
                return g
    raise IngestError(f"chess.com game {game_id} not found in {username}'s archives")


def _prev_month(year: int, month: int) -> tuple[int, int]:
    return (year - 1, 12) if month == 1 else (year, month - 1)


# --------------------------------------------------------------------------
# Lichess
# --------------------------------------------------------------------------

LICHESS_API = "https://lichess.org/api"

# Wall-clock budget for one user's game stream. A handful of games is seconds of
# traffic, so anything near this is a connection that has stopped making
# progress rather than a genuinely slow export.
_STREAM_BUDGET_S = 120.0


def _lichess_session(token: str | None = None) -> requests.Session:
    s = _session()
    s.headers["Accept"] = "application/x-ndjson"
    if token:
        s.headers["Authorization"] = f"Bearer {token}"
    return s


def lichess_user_games(
    username: str,
    max_games: int = 20,
    *,
    rated: bool = True,
    perf: str | None = "rapid,classical",
    token: str | None = None,
    min_elo: int | None = None,
    max_elo: int | None = None,
    since: int | None = None,
    session: requests.Session | None = None,
) -> Iterator[Game]:
    """Stream a user's games as ndjson+PGN. Rate-limit aware via tenacity."""
    s = session or _lichess_session(token)
    params: dict[str, object] = {
        "max": max_games,
        "clocks": "true",
        "evals": "false",
        "pgnInJson": "true",
        "rated": "true" if rated else "false",
        "sort": "dateDesc",
    }
    if perf:
        params["perfType"] = perf
    if since:
        params["since"] = since

    url = f"{LICHESS_API}/games/user/{username}"
    # ``timeout`` on a streamed response is a per-read deadline, not a total
    # one: as long as the far end dribbles a byte occasionally it never fires,
    # and an unattended harvest blocks forever on a single half-open
    # connection. Observed in practice as multi-hour gaps in the harvest log
    # with every engine sitting idle. So bound the connect and read phases
    # tightly *and* enforce a wall-clock budget for the whole stream.
    deadline = time.monotonic() + _STREAM_BUDGET_S
    with s.get(url, params=params, stream=True, timeout=(10, 60)) as r:
        if r.status_code == 429:
            raise RateLimited("429 from lichess game export")
        r.raise_for_status()
        for line in r.iter_lines(decode_unicode=True):
            if time.monotonic() > deadline:
                # Truncating is right: whatever arrived is still valid, and the
                # caller moves on to the next account instead of stalling. Say
                # so, because a silent truncation would look like a user who
                # simply had few games.
                print(
                    f"  {username}: game stream exceeded {_STREAM_BUDGET_S}s, "
                    f"moving on"
                )
                break
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            g = _game_from_lichess_json(payload)
            if g is None:
                continue
            if min_elo is not None or max_elo is not None:
                elos = [p.elo for p in g.players.values() if p.elo is not None]
                if len(elos) < 2:
                    continue
                if min_elo is not None and min(elos) < min_elo:
                    continue
                if max_elo is not None and max(elos) > max_elo:
                    continue
            yield g


def _game_from_lichess_json(payload: dict) -> Game | None:
    pgn = payload.get("pgn")
    if not pgn:
        return None
    try:
        games = games_from_pgn_text(pgn, source="lichess")
    except IngestError:
        return None
    g = games[0]
    g.url = f"https://lichess.org/{payload.get('id')}" if payload.get("id") else g.url
    g.speed = payload.get("speed") or g.speed
    clock = payload.get("clock") or {}
    if clock:
        g.base_time = clock.get("initial", g.base_time)
        g.increment = clock.get("increment", g.increment)
        # Recompute now that the authoritative base/increment are known.
        for mv in g.moves:
            mv.think_time = None
        _compute_think_times(g.moves, g.increment, g.base_time)
    for color in ("white", "black"):
        side = (payload.get("players") or {}).get(color) or {}
        user = side.get("user") or {}
        if side.get("rating"):
            g.players[color].elo = int(side["rating"])
        if user.get("name"):
            g.players[color].name = user["name"]
        if user.get("title"):
            g.players[color].title = user["title"]
        # Lichess marks accounts closed for ToS violations. Excluding them is
        # essential when building an "honest player" baseline -- otherwise the
        # reference distribution is contaminated with real cheaters and every
        # test loses power.
        g.tos_flagged[color] = bool(user.get("tosViolation"))
    return g


def lichess_account_status(
    user_ids: list[str],
    session: requests.Session | None = None,
) -> dict[str, bool]:
    """Check which Lichess accounts are flagged for ToS violation or closed.

    Returns ``{user_id: True}`` for accounts that should be excluded from any
    "honest player" corpus. Uses ``POST /api/users`` (up to 300 ids per
    request) which is the only endpoint that reliably exposes ``tosViolation``.
    The per-game export does NOT include that field.

    Accounts not returned by the API (deleted / non-existent) are treated as
    flagged -- better to lose a few honest games than to include a cheater.
    """
    s = session or _session()
    # The /api/users endpoint requires plain text, not JSON.
    s.headers["Content-Type"] = "text/plain"
    ids = sorted({u.lower() for u in user_ids})
    out: dict[str, bool] = {}
    for i in range(0, len(ids), 300):
        chunk = ids[i : i + 300]
        resp = s.post(
            f"{LICHESS_API}/users",
            data=",".join(chunk),
            headers={"Content-Type": "text/plain"},
            timeout=60,
        )
        if resp.status_code == 429:
            time.sleep(60)
            resp = s.post(
                f"{LICHESS_API}/users",
                data=",".join(chunk),
                headers={"Content-Type": "text/plain"},
                timeout=60,
            )
        resp.raise_for_status()
        for u in resp.json():
            out[u["id"]] = bool(u.get("tosViolation") or u.get("disabled"))
        # Accounts not in the response are gone -- treat as flagged.
        for uid in chunk:
            out.setdefault(uid, True)
        time.sleep(1.0)
    return out


def lichess_game(game_id: str, token: str | None = None) -> Game:
    s = _lichess_session(token)
    s.headers["Accept"] = "application/json"
    url = f"{LICHESS_API}/game/export/{game_id}"
    resp = _get(url, s, params={"clocks": "true", "pgnInJson": "true", "evals": "false"})
    g = _game_from_lichess_json(resp.json())
    if g is None:
        raise IngestError(f"lichess game {game_id} has no PGN")
    return g


# --------------------------------------------------------------------------
# URL dispatch
# --------------------------------------------------------------------------

_LICHESS_URL = re.compile(
    r"lichess\.org/(?:embed/)?(?:game/)?([a-zA-Z0-9]{8,12})(?:/(?:white|black))?", re.I
)
_CHESSCOM_URL = re.compile(
    r"chess\.com/(?:game|live|daily|analysis(?:/game)?)/(?:live/|daily/|game/)?(\d+)", re.I
)


def from_url(url: str, token: str | None = None) -> Game:
    """Dispatch a pasted game URL to the right export endpoint."""
    url = url.strip()
    m = _CHESSCOM_URL.search(url)
    if m:
        return chesscom_game(m.group(1))
    m = _LICHESS_URL.search(url)
    if m:
        # Lichess ids are 8 chars for games, 12 for the per-player variant.
        return lichess_game(m.group(1)[:8], token)
    raise IngestError(
        "unrecognised URL. Expected a chess.com game URL "
        "(chess.com/game/live/123456789) or a lichess one (lichess.org/abcd1234)."
    )


def load_any(text: str, token: str | None = None) -> list[Game]:
    """Best-effort entry point: accepts a URL, a file path, or raw PGN."""
    stripped = text.strip()
    if not stripped:
        raise IngestError("nothing to load")
    if stripped.startswith(("http://", "https://")) or "lichess.org/" in stripped[:80]:
        return [from_url(stripped, token)]
    candidate = Path(stripped)
    if len(stripped) < 300 and candidate.exists() and candidate.is_file():
        return games_from_pgn_file(candidate)
    return games_from_pgn_text(stripped)
