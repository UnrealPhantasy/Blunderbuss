//! The search: pick a move for a position.
//!
//! First version: **negamax** with **alpha-beta** pruning, to a fixed depth,
//! guided by the material [`evaluate`](crate::evaluation::evaluate). No move
//! ordering yet, so the pruning is weak — that is expected; ordering is a later,
//! separately measured increment.

use crate::evaluation::evaluate;
use crate::position::{Move, Position};

/// A mate score, large enough to dominate any material balance. The distance to
/// mate (`ply`) is subtracted so the search prefers shorter mates, and so a mate
/// is always distinguishable from a mere material advantage.
pub const MATE: i32 = 30_000;

// A bound strictly above any reachable score (including `MATE`), used as the
// initial alpha/beta window.
const INF: i32 = 40_000;

/// The best move for `pos`, searched to `depth` plies, with its score (from the
/// side-to-move perspective). Returns `None` iff `pos` has no legal move — the
/// game is already over.
pub fn best_move(pos: &Position, depth: u32) -> Option<(Move, i32)> {
    let moves = pos.legal_moves();
    if moves.is_empty() {
        return None;
    }

    // `moves` is non-empty, so index 0 is a valid initial choice.
    let mut best_move = moves[0];
    let mut best_score = -INF;
    let mut alpha = -INF;

    for &mv in &moves {
        // Copy-make: `play` returns the child position; `pos` stays untouched.
        // The child is searched from the opponent's side, hence the negation and
        // the swapped/negated window.
        let score = -negamax(&pos.play(mv), depth.saturating_sub(1), -INF, -alpha, 1);
        if score > best_score {
            best_score = score;
            best_move = mv;
        }
        if score > alpha {
            alpha = score;
        }
    }
    Some((best_move, best_score))
}

/// Negamax with alpha-beta pruning. Returns the value of `pos` from the
/// side-to-move perspective. `ply` is the distance from the root, used only to
/// score mates (closer mates score higher).
fn negamax(pos: &Position, depth: u32, mut alpha: i32, beta: i32, ply: i32) -> i32 {
    let moves = pos.legal_moves();
    if moves.is_empty() {
        // Terminal: checkmate (the side to move has lost) or stalemate (draw).
        return if pos.in_check() { -(MATE - ply) } else { 0 };
    }
    if depth == 0 {
        return evaluate(pos);
    }

    let mut best = -INF;
    for &mv in &moves {
        let score = -negamax(&pos.play(mv), depth - 1, -beta, -alpha, ply + 1);
        if score > best {
            best = score;
        }
        if best > alpha {
            alpha = best;
        }
        if alpha >= beta {
            break; // Beta cutoff: this branch cannot improve the result.
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::{Color, Piece, Status};

    #[test]
    fn finds_mate_in_one() {
        // White: Ra1, Kh1. Black: Kg8, pawns f7/g7/h7. Ra8# is the mate in one.
        let p = Position::from_fen("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1").unwrap();
        let (mv, score) = best_move(&p, 3).expect("a move exists");
        assert_eq!(p.play(mv).status(), Status::Checkmate);
        assert!(score >= MATE - 100, "score {score} should be mate-level");
    }

    #[test]
    fn grabs_a_hanging_piece() {
        // Black's queen on d5 is undefended; White's rook on d1 takes it for free.
        let p = Position::from_fen("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1").unwrap();
        let (mv, _) = best_move(&p, 3).expect("a move exists");
        let after = p.play(mv);
        assert_eq!(
            after.count(Color::Black, Piece::Queen),
            0,
            "the engine should win the free queen"
        );
    }

    #[test]
    fn no_move_at_a_terminal_root() {
        // Checkmate (scholar's mate) and stalemate: nothing to search.
        let mate = Position::from_fen(
            "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4",
        )
        .unwrap();
        let stalemate = Position::from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1").unwrap();
        assert!(best_move(&mate, 3).is_none());
        assert!(best_move(&stalemate, 3).is_none());
    }

    #[test]
    fn plays_a_full_legal_game() {
        // The engine plays both sides. Every chosen move must be legal (accepted
        // by `try_play`) and nothing should panic. Shallow depth keeps it fast.
        let mut pos = Position::initial();
        for _ in 0..40 {
            match best_move(&pos, 2) {
                Some((mv, _)) => pos = pos.try_play(mv).expect("engine chose a legal move"),
                None => break, // terminal reached
            }
        }
    }
}
