"""Per-position difficulty index.

Everything downstream is conditioned on this number, so it has to mean
something specific. Here, complexity is defined as **empirical human
difficulty**: the weights are fitted by logistic regression on a corpus of
ordinary games, predicting whether a normal player of the relevant rating band
found the engine's top move.

That definition is what makes the accuracy-vs-complexity test meaningful. If
complexity were hand-weighted by intuition it would measure "positions that
look hairy to me", and a flat slope against it would prove nothing.

Shipped defaults are literature-shaped (more legal moves / wider eval spread /
quiet best move = harder; forcing best move = easier) and are replaced by
fitted weights once ``baseline.py`` has run.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import chess

from .engine import PositionAnalysis

# Raw feature names, in a fixed order so weight vectors stay interpretable.
FEATURES = (
    "legal",  # breadth of choice
    "spread",  # sharpness: std dev of top-k evals
    "gap_small",  # top1 ~ top2: easy to go wrong, hard to pick THE move
    "only_move",  # top1 >> top2: a single move holds the position
    "best_quiet",  # best move is neither capture nor check
    "best_forcing",  # best move is a capture / check / recapture
    "phase",  # 0 = opening, 1 = endgame
    "imbalance",  # |material difference|, normalised
    "in_check",  # side to move is in check (narrows the tree)
)

# Literature-shaped fallback weights on the logit scale. Positive = harder.
#
# The *signs and relative sizes* are judgement; the *location and scale* are
# not. An earlier hand-picked intercept made the index saturate -- 65% of real
# positions scored above 0.9 -- which left the accuracy-vs-complexity
# regression with almost no dynamic range to work with. These values are the
# same shape rescaled so the index centres near 0.5 with sd ~0.29 on real
# positions. See tests/recenter_complexity.py, which recomputes the correction.
DEFAULT_WEIGHTS: dict[str, float] = {
    "intercept": -3.027,
    "legal": 1.751,
    "spread": 1.328,
    "gap_small": 1.570,
    "only_move": -0.423,
    "best_quiet": 1.147,
    "best_forcing": -1.509,
    "phase": -0.241,
    "imbalance": -0.543,
    "in_check": -1.087,
}

PIECE_VALUES = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.25,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}

# Total non-king material at the start, used to define game phase.
_START_MATERIAL = 2 * (8 * 1.0 + 2 * 3.0 + 2 * 3.25 + 2 * 5.0 + 9.0)


@dataclass
class ComplexityWeights:
    """Fitted (or default) logistic weights for the complexity index."""

    intercept: float = DEFAULT_WEIGHTS["intercept"]
    weights: dict[str, float] = field(
        default_factory=lambda: {k: v for k, v in DEFAULT_WEIGHTS.items() if k != "intercept"}
    )
    fitted: bool = False
    n_positions: int = 0

    @classmethod
    def from_dict(cls, d: dict | None) -> "ComplexityWeights":
        if not d:
            return cls()
        w = dict(DEFAULT_WEIGHTS)
        w.update(d.get("weights", {}))
        return cls(
            intercept=d.get("intercept", w["intercept"]),
            weights={k: w[k] for k in FEATURES},
            fitted=bool(d.get("fitted", False)),
            n_positions=int(d.get("n_positions", 0)),
        )

    def to_dict(self) -> dict:
        return {
            "intercept": self.intercept,
            "weights": dict(self.weights),
            "fitted": self.fitted,
            "n_positions": self.n_positions,
        }

    def score(self, feats: dict[str, float]) -> float:
        """Map a raw feature dict to a 0-1 complexity index."""
        z = self.intercept
        for name in FEATURES:
            z += self.weights.get(name, 0.0) * feats.get(name, 0.0)
        return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))


def material(board: chess.Board) -> tuple[float, float]:
    """(total non-king material, signed imbalance from white's perspective)."""
    total = 0.0
    signed = 0.0
    for piece_type, value in PIECE_VALUES.items():
        if piece_type == chess.KING:
            continue
        w = len(board.pieces(piece_type, chess.WHITE))
        b = len(board.pieces(piece_type, chess.BLACK))
        total += value * (w + b)
        signed += value * (w - b)
    return total, signed


def game_phase(board: chess.Board) -> str:
    """Classify the position as 'opening', 'middlegame', or 'endgame'.

    Uses the Lichess/chess-programming approach based on non-pawn material:
    - Endgame: both sides have no queens, or total non-pawn material is very
      low (queen + minor or less per side).
    - Opening: full move number <= 10 AND no pieces have left the back ranks
      significantly (proxy: total material still high, >= 90% of start).
    - Middlegame: everything else.

    The material thresholds are calibrated to match Lichess's phase detection.
    """
    # Count non-pawn, non-king material per side
    npm_white = 0.0
    npm_black = 0.0
    has_queen_w = bool(board.pieces(chess.QUEEN, chess.WHITE))
    has_queen_b = bool(board.pieces(chess.QUEEN, chess.BLACK))
    for pt in (chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN):
        val = PIECE_VALUES[pt]
        npm_white += val * len(board.pieces(pt, chess.WHITE))
        npm_black += val * len(board.pieces(pt, chess.BLACK))

    npm_total = npm_white + npm_black

    # Non-queen material per side
    minor_w = npm_white - (9.0 if has_queen_w else 0.0)
    minor_b = npm_black - (9.0 if has_queen_b else 0.0)

    # Endgame conditions (following Lichess/chess-programming conventions):
    # 1. No queens on either side
    if not has_queen_w and not has_queen_b:
        return "endgame"
    # 2. Low total non-pawn material (roughly rook + minor per side max)
    if npm_total <= 13.0:
        return "endgame"
    # 3. One side has queen but nothing else, other side has no queen
    if has_queen_w and minor_w == 0 and not has_queen_b:
        return "endgame"
    if has_queen_b and minor_b == 0 and not has_queen_w:
        return "endgame"
    # 4. Both have queens but no other pieces (pure queen endgame)
    if has_queen_w and has_queen_b and minor_w == 0 and minor_b == 0:
        return "endgame"
    # 5. Queen + minor vs queen (or similar minimal material with queens)
    if has_queen_w and has_queen_b and minor_w <= 3.25 and minor_b <= 3.25:
        return "endgame"

    # Opening: early moves with most material intact
    total_mat, _ = material(board)
    fullmove = board.fullmove_number
    if fullmove <= 12 and total_mat >= 0.88 * _START_MATERIAL:
        return "opening"

    return "middlegame"


def is_recapture(board: chess.Board, move: chess.Move, prev_move_uci: str | None) -> bool:
    """True if ``move`` captures on the square the opponent just captured on.

    Recaptures are the archetypal 'obvious' move: a player of any strength
    plays them, so a match there carries almost no information.
    """
    if prev_move_uci is None or not board.is_capture(move):
        return False
    try:
        prev = chess.Move.from_uci(prev_move_uci)
    except ValueError:
        return False
    return prev.to_square == move.to_square


def raw_features(
    pa: PositionAnalysis,
    prev_move_uci: str | None = None,
) -> dict[str, float]:
    """Extract the raw complexity features from a position's MultiPV data."""
    board = chess.Board(pa.fen)
    legal = pa.legal_moves or board.legal_moves.count()

    # Breadth of choice, log-scaled: going from 5 to 10 options matters far
    # more than going from 35 to 40.
    f_legal = min(1.0, math.log(max(legal, 1)) / math.log(45.0))

    cps = [c.cp for c in pa.candidates]
    if len(cps) >= 2:
        mean = sum(cps) / len(cps)
        var = sum((c - mean) ** 2 for c in cps) / len(cps)
        f_spread = min(1.0, math.sqrt(var) / 120.0)
    else:
        f_spread = 0.0

    gap = pa.gap()
    if gap is None:
        f_gap_small, f_only = (1.0, 0.0) if legal > 1 else (0.0, 1.0)
    else:
        gap = max(0.0, gap)
        f_gap_small = 1.0 - min(1.0, gap / 100.0)
        f_only = min(1.0, gap / 300.0)

    f_quiet = 0.0
    f_forcing = 0.0
    best = pa.best
    if best is not None:
        try:
            mv = chess.Move.from_uci(best.uci)
        except ValueError:
            mv = None
        if mv is not None and mv in board.legal_moves:
            capture = board.is_capture(mv)
            check = board.gives_check(mv)
            recap = is_recapture(board, mv, prev_move_uci)
            if capture or check or recap:
                f_forcing = 1.0
                if recap:
                    # A recapture is maximally obvious; count it extra.
                    f_forcing = 1.5
            else:
                f_quiet = 1.0

    total_mat, signed = material(board)
    f_phase = max(0.0, min(1.0, 1.0 - total_mat / _START_MATERIAL))
    f_imbalance = min(1.0, abs(signed) / 9.0)
    f_check = 1.0 if board.is_check() else 0.0

    return {
        "legal": f_legal,
        "spread": f_spread,
        "gap_small": f_gap_small,
        "only_move": f_only,
        "best_quiet": f_quiet,
        "best_forcing": f_forcing,
        "phase": f_phase,
        "phase_name": game_phase(board),
        "imbalance": f_imbalance,
        "in_check": f_check,
    }


def complexity_of(
    pa: PositionAnalysis,
    weights: ComplexityWeights | None = None,
    prev_move_uci: str | None = None,
) -> tuple[float, dict[str, float]]:
    """Return (0-1 complexity index, raw features)."""
    w = weights or ComplexityWeights()
    feats = raw_features(pa, prev_move_uci)
    return w.score(feats), feats


def forced(pa: PositionAnalysis) -> bool:
    """Single legal reply: a 'match' here is worth exactly nothing."""
    return pa.legal_moves <= 1


def fit_complexity_weights(
    feature_rows: list[dict[str, float]],
    found_top1: list[int],
    l2: float = 1.0,
    max_iter: int = 200,
) -> ComplexityWeights:
    """Fit the complexity weights on corpus data.

    Logistic regression of ``NOT found_top1`` (i.e. the player went wrong) on
    the raw features, so a higher index means a harder position. L2-regularised
    Newton-Raphson, implemented directly to avoid a scikit-learn dependency for
    one 10-parameter fit.
    """
    import numpy as np

    if len(feature_rows) < 50:
        return ComplexityWeights()

    X = np.array(
        [[1.0] + [row.get(name, 0.0) for name in FEATURES] for row in feature_rows],
        dtype=float,
    )
    # Target is "failed to find the top move" = the position was hard for them.
    y = 1.0 - np.asarray(found_top1, dtype=float)

    n, d = X.shape
    beta = np.zeros(d)
    penalty = np.full(d, l2)
    penalty[0] = 0.0  # never shrink the intercept

    for _ in range(max_iter):
        z = np.clip(X @ beta, -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        grad = X.T @ (p - y) + penalty * beta
        w = np.clip(p * (1 - p), 1e-6, None)
        hess = (X * w[:, None]).T @ X + np.diag(penalty)
        try:
            step = np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            break
        beta_new = beta - step
        if not np.all(np.isfinite(beta_new)):
            break
        if np.max(np.abs(beta_new - beta)) < 1e-8:
            beta = beta_new
            break
        beta = beta_new

    return ComplexityWeights(
        intercept=float(beta[0]),
        weights={name: float(beta[i + 1]) for i, name in enumerate(FEATURES)},
        fitted=True,
        n_positions=n,
    )
