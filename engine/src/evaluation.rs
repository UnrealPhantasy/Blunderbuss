//! Static evaluation of a position.
//!
//! Two terms, both in **centipawns** (1 pawn = 100) and both from the
//! side-to-move perspective:
//!
//! - **material** — how much each side has (pawn 100 … queen 900);
//! - **piece-square tables (PST)** — a bonus for *where* each piece sits, so the
//!   engine develops toward the centre, advances central pawns, and keeps its
//!   king safe instead of shuffling material-equal positions aimlessly.
//!
//! The PST values are the well-known standard "simplified" tables. They are ours
//! to keep and tune (and, one day, to learn).

use crate::position::{Color, Piece, Position, Square};

// Material values in centipawns. The king is not scored (both sides always have
// exactly one, so it never shifts the balance).
fn value(piece: Piece) -> i32 {
    match piece {
        Piece::Pawn => 100,
        Piece::Knight => 320,
        Piece::Bishop => 330,
        Piece::Rook => 500,
        Piece::Queen => 900,
        Piece::King => 0,
    }
}

/// The material + positional value of `pos`, in centipawns, from the side-to-move
/// perspective.
pub fn evaluate(pos: &Position) -> i32 {
    let mut white = 0;
    let mut black = 0;

    // Walk every square once; on each occupied square add material + PST from the
    // piece owner's point of view.
    for sq in Square::ALL {
        if let Some(piece) = pos.piece_on(sq) {
            let color = pos.color_on(sq).expect("an occupied square has a colour");
            // `relative_to(color)` orients the square to White's view (it flips the
            // rank for Black), so a single White-oriented table serves both sides.
            let bonus = PST[piece as usize][sq.relative_to(color) as usize];
            let score = value(piece) + bonus;
            match color {
                Color::White => white += score,
                Color::Black => black += score,
            }
        }
    }

    let balance = white - black;
    // Flip to the side-to-move perspective: negamax wants "good for me" > 0.
    match pos.side_to_move() {
        Color::White => balance,
        Color::Black => -balance,
    }
}

// --- Piece-square tables --------------------------------------------------
//
// One table per piece type, indexed by `Square as usize` (a1 = 0 … h8 = 63), so
// each table is laid out **rank 1 first** (the White home rank) up to rank 8.
// Values are from White's point of view; Black reads the rank-flipped square via
// `relative_to`. Order matches `Piece`: pawn, knight, bishop, rook, queen, king.

#[rustfmt::skip]
const PAWN: [i32; 64] = [
     0,  0,  0,  0,  0,  0,  0,  0, // rank 1
     5, 10, 10,-20,-20, 10, 10,  5,
     5, -5,-10,  0,  0,-10, -5,  5,
     0,  0,  0, 20, 20,  0,  0,  0,
     5,  5, 10, 25, 25, 10,  5,  5,
    10, 10, 20, 30, 30, 20, 10, 10,
    50, 50, 50, 50, 50, 50, 50, 50,
     0,  0,  0,  0,  0,  0,  0,  0, // rank 8
];

#[rustfmt::skip]
const KNIGHT: [i32; 64] = [
    -50,-40,-30,-30,-30,-30,-40,-50,
    -40,-20,  0,  5,  5,  0,-20,-40,
    -30,  5, 10, 15, 15, 10,  5,-30,
    -30,  0, 15, 20, 20, 15,  0,-30,
    -30,  5, 15, 20, 20, 15,  5,-30,
    -30,  0, 10, 15, 15, 10,  0,-30,
    -40,-20,  0,  0,  0,  0,-20,-40,
    -50,-40,-30,-30,-30,-30,-40,-50,
];

#[rustfmt::skip]
const BISHOP: [i32; 64] = [
    -20,-10,-10,-10,-10,-10,-10,-20,
    -10,  5,  0,  0,  0,  0,  5,-10,
    -10, 10, 10, 10, 10, 10, 10,-10,
    -10,  0, 10, 10, 10, 10,  0,-10,
    -10,  5,  5, 10, 10,  5,  5,-10,
    -10,  0,  5, 10, 10,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10,-10,-10,-10,-10,-20,
];

#[rustfmt::skip]
const ROOK: [i32; 64] = [
     0,  0,  0,  5,  5,  0,  0,  0,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
    -5,  0,  0,  0,  0,  0,  0, -5,
     5, 10, 10, 10, 10, 10, 10,  5,
     0,  0,  0,  0,  0,  0,  0,  0,
];

#[rustfmt::skip]
const QUEEN: [i32; 64] = [
    -20,-10,-10, -5, -5,-10,-10,-20,
    -10,  0,  5,  0,  0,  0,  0,-10,
    -10,  5,  5,  5,  5,  5,  0,-10,
      0,  0,  5,  5,  5,  5,  0, -5,
     -5,  0,  5,  5,  5,  5,  0, -5,
    -10,  0,  5,  5,  5,  5,  0,-10,
    -10,  0,  0,  0,  0,  0,  0,-10,
    -20,-10,-10, -5, -5,-10,-10,-20,
];

// King, middle-game only: reward the back rank / castled corner, punish walking
// into the centre. A separate end-game table (where the king should centralise)
// is a later refinement (tapered evaluation).
#[rustfmt::skip]
const KING: [i32; 64] = [
     20, 30, 10,  0,  0, 10, 30, 20, // rank 1: castled squares score highest
     20, 20,  0,  0,  0,  0, 20, 20,
    -10,-20,-20,-20,-20,-20,-20,-10,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
];

const PST: [[i32; 64]; 6] = [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_position_is_balanced() {
        // The start is perfectly symmetric, so material and PST both cancel.
        assert_eq!(evaluate(&Position::initial()), 0);
    }

    #[test]
    fn a_queen_up_is_worth_about_900() {
        // Standard start with Black's queen removed; White to move. Material is
        // +900; PST shifts it by a few centipawns, so we check a tight range
        // rather than an exact number.
        let p = Position::from_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
            .unwrap();
        let e = evaluate(&p);
        assert!((850..=950).contains(&e), "expected ~+900, got {e}");
    }

    #[test]
    fn sign_flips_with_the_side_to_move() {
        // Same board, opposite side to move → opposite score.
        let w = evaluate(
            &Position::from_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1").unwrap(),
        );
        let b = evaluate(
            &Position::from_fen("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1").unwrap(),
        );
        assert_eq!(w, -b);
    }

    #[test]
    fn a_central_knight_beats_a_rim_knight() {
        // Same material (K+N vs K); only the knight's square differs. The kings
        // sit symmetrically (e1/e8) so their PST cancels.
        let central = Position::from_fen("4k3/8/8/8/3N4/8/8/4K3 w - - 0 1").unwrap();
        let rim = Position::from_fen("4k3/8/8/8/8/8/8/N3K3 w - - 0 1").unwrap();
        assert!(evaluate(&central) > evaluate(&rim));
    }

    #[test]
    fn the_king_prefers_safety_in_the_middlegame() {
        // A castled/back-rank king outscores a king marching into the centre.
        let safe = Position::from_fen("4k3/8/8/8/8/8/8/6K1 w - - 0 1").unwrap(); // Kg1
        let exposed = Position::from_fen("4k3/8/8/8/4K3/8/8/8 w - - 0 1").unwrap(); // Ke4
        assert!(evaluate(&safe) > evaluate(&exposed));
    }

    #[test]
    fn evaluation_is_colour_symmetric() {
        // `b` is `a` mirrored: colours swapped, ranks flipped, side to move
        // swapped. The two must evaluate to the same number.
        let a = Position::from_fen("4k3/8/8/8/3N4/8/8/4K3 w - - 0 1").unwrap();
        let b = Position::from_fen("4k3/8/8/3n4/8/8/8/4K3 b - - 0 1").unwrap();
        assert_eq!(evaluate(&a), evaluate(&b));
    }
}
