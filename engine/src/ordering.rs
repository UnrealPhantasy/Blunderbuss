//! Move ordering — try the most promising moves first so alpha-beta prunes more.
//!
//! Ordering never changes *which* move is best (alpha-beta returns the same value
//! whatever the order); it only changes how fast the search proves it, by causing
//! earlier beta cutoffs. Fewer nodes at a fixed depth means deeper search in a
//! given time budget — that is where the strength ultimately comes from.
//!
//! First heuristic: **MVV-LVA** (Most Valuable Victim − Least Valuable Aggressor).
//! Try the most profitable captures first: taking a queen with a pawn before
//! taking a pawn with a queen.

use crate::position::{Move, Piece, Position};

// Piece values used *for ordering only* (not for evaluation): only their relative
// order matters here, so this table stays local to the module.
fn value(piece: Piece) -> i32 {
    match piece {
        Piece::Pawn => 100,
        Piece::Knight => 320,
        Piece::Bishop => 330,
        Piece::Rook => 500,
        Piece::Queen => 900,
        // A king is never actually captured in a legal position; the value is
        // only here to make the match exhaustive.
        Piece::King => 20_000,
    }
}

/// The MVV-LVA score of a move.
///
/// A capture scores `100 * victim - attacker`: the `100 *` makes a more valuable
/// victim always outrank a less valuable one, whatever the attacker; among equal
/// victims, a cheaper attacker (smaller `attacker`) ranks higher. Quiet moves
/// score `0`, so every capture sorts before every quiet move.
pub fn mvv_lva(pos: &Position, mv: Move) -> i32 {
    // The victim is the enemy piece standing on the destination square. A legal
    // move never lands on a friendly piece, so "occupied" means "capture".
    // (En-passant leaves the destination empty, so it reads as a quiet move here
    // — acceptable: ordering never affects correctness, only pruning.)
    match pos.piece_on(mv.to) {
        Some(victim) => {
            let attacker = pos.piece_on(mv.from).map_or(0, value);
            100 * value(victim) - attacker
        }
        None => 0,
    }
}

/// Sorts `moves` in place: best captures first, quiet moves last.
pub fn order_moves(pos: &Position, moves: &mut [Move]) {
    // Idiom: `sort_by_key` with `Reverse` sorts by the key in *descending* order,
    // so the highest MVV-LVA score comes first.
    moves.sort_by_key(|&mv| std::cmp::Reverse(mvv_lva(pos, mv)));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::Position;

    #[test]
    fn a_capture_scores_above_a_quiet_move() {
        // White to move: Rd1xd5 captures the queen; other rook/king moves are quiet.
        let p = Position::from_fen("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1").unwrap();
        let moves = p.legal_moves();
        let capture = moves.iter().copied().find(|&mv| p.piece_on(mv.to).is_some()).unwrap();
        let quiet = moves.iter().copied().find(|&mv| p.piece_on(mv.to).is_none()).unwrap();
        assert!(mvv_lva(&p, capture) > mvv_lva(&p, quiet));
    }

    #[test]
    fn cheap_attacker_on_rich_victim_ranks_highest() {
        // Both a pawn (d4) and the queen (e2) can capture the black rook on e5.
        // Pawn-takes-rook (dxe5) must outrank queen-takes-rook (Qxe5): same victim,
        // cheaper attacker.
        let p = Position::from_fen("4k3/8/8/4r3/3P4/8/4Q3/6K1 w - - 0 1").unwrap();
        let moves = p.legal_moves();
        let target = cozy_chess::Square::E5;
        let by_pawn = moves
            .iter()
            .copied()
            .find(|&mv| mv.to == target && p.piece_on(mv.from) == Some(Piece::Pawn))
            .unwrap();
        let by_queen = moves
            .iter()
            .copied()
            .find(|&mv| mv.to == target && p.piece_on(mv.from) == Some(Piece::Queen))
            .unwrap();
        assert!(mvv_lva(&p, by_pawn) > mvv_lva(&p, by_queen));
    }

    #[test]
    fn order_moves_puts_captures_first() {
        let p = Position::from_fen("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1").unwrap();
        let mut moves = p.legal_moves();
        order_moves(&p, &mut moves);
        // The first move after ordering must be a capture.
        assert!(p.piece_on(moves[0].to).is_some(), "a capture should sort first");
    }
}
