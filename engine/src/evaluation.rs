//! Static evaluation of a position.
//!
//! First version: **pure material**, in **centipawns** (1 pawn = 100), from the
//! side-to-move perspective — positive means the side to move is materially
//! ahead. That sign convention is exactly what a negamax search consumes.
//!
//! Integer centipawns (rather than whole pawns) keep the arithmetic fast and
//! exact while leaving room for later sub-pawn positional terms (piece-square
//! tables, etc.).

use crate::position::{Color, Piece, Position};

// Piece values in centipawns. The king is not scored: both sides always have
// exactly one, so it never shifts the balance.
const PAWN: i32 = 100;
const KNIGHT: i32 = 320;
const BISHOP: i32 = 330;
const ROOK: i32 = 500;
const QUEEN: i32 = 900;

// The scored piece types with their value. King is omitted on purpose.
const SCORED: [(Piece, i32); 5] = [
    (Piece::Pawn, PAWN),
    (Piece::Knight, KNIGHT),
    (Piece::Bishop, BISHOP),
    (Piece::Rook, ROOK),
    (Piece::Queen, QUEEN),
];

/// The material balance of `pos`, in centipawns, from the side-to-move
/// perspective.
pub fn evaluate(pos: &Position) -> i32 {
    let mut white_minus_black = 0;
    // Idiom: iterating an array by value (`for (piece, value) in SCORED`) works
    // because `Piece` and `i32` are `Copy` — each element is copied out.
    for (piece, value) in SCORED {
        let white = pos.count(Color::White, piece) as i32;
        let black = pos.count(Color::Black, piece) as i32;
        white_minus_black += value * (white - black);
    }
    // Flip to the side-to-move perspective: negamax wants "good for me" > 0.
    match pos.side_to_move() {
        Color::White => white_minus_black,
        Color::Black => -white_minus_black,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_position_is_balanced() {
        assert_eq!(evaluate(&Position::initial()), 0);
    }

    #[test]
    fn a_queen_up_is_plus_900_centipawns() {
        // Standard start with Black's queen removed; White to move is +900 cp
        // (i.e. +9.00 pawns as a GUI would show).
        let p = Position::from_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            .unwrap();
        assert_eq!(evaluate(&p), 900);
    }

    #[test]
    fn sign_flips_with_the_side_to_move() {
        // Same material imbalance, but Black to move: the side to move is a queen
        // down, so the score is negative.
        let p = Position::from_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")
            .unwrap();
        assert_eq!(evaluate(&p), -900);
    }
}
