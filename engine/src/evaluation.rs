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
//!
//! # Tapered evaluation
//!
//! One square is worth different things at different moments of a game, and the
//! king is the extreme case: sheltered in a corner while the queens are on, and
//! marching to the centre once they are off. A single table cannot say both, so
//! the king has two — [`KING_MIDDLEGAME`] and [`KING_ENDGAME`] — and the score is
//! **interpolated** between them according to how much material is left
//! ([`phase`]).
//!
//! Interpolating rather than switching at a threshold matters: a switch would make
//! the evaluation of one position jump by tens of centipawns the moment a single
//! capture crosses the boundary, and the engine would chase or avoid that capture
//! for a reason that has nothing to do with chess.

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

/// How much each piece contributes to the game phase.
///
/// Pawns count for nothing: a position with every pawn and no piece is an endgame,
/// which is exactly what these weights should say. Kings are always present, so they
/// would only add a constant.
fn phase_weight(piece: Piece) -> i32 {
    match piece {
        Piece::Knight | Piece::Bishop => 1,
        Piece::Rook => 2,
        Piece::Queen => 4,
        Piece::Pawn | Piece::King => 0,
    }
}

/// The phase at the initial position: 4 minors + 2 rooks + 1 queen, per side.
const MAX_PHASE: i32 = 24;

/// How far into the middlegame `pos` is: [`MAX_PHASE`] with every piece on the board,
/// `0` once only kings and pawns remain.
///
/// Clamped, because promotions can put more material on the board than the opening
/// had — three queens is unusual but perfectly legal, and an unclamped phase would
/// then weight the middlegame table by more than 100%.
pub fn phase(pos: &Position) -> i32 {
    let mut phase = 0;
    for sq in Square::ALL {
        if let Some(piece) = pos.piece_on(sq) {
            phase += phase_weight(piece);
        }
    }
    phase.min(MAX_PHASE)
}

/// The material + positional value of `pos`, in centipawns, from the side-to-move
/// perspective.
pub fn evaluate(pos: &Position) -> i32 {
    // Two running scores — one reading the middlegame tables, one the endgame tables
    // — plus the phase, all accumulated in the **same** pass over the board.
    //
    // The single pass is the point. `evaluate` is the hottest function in the engine,
    // called at every leaf and at every quiescence node; walking the 64 squares a
    // second time just to count material would double it. A king-safety term was
    // abandoned on this engine for costing 0.23 µs per node, which is a whole ply of
    // depth at a fixed time budget.
    let mut middlegame = 0;
    let mut endgame = 0;
    let mut phase = 0;

    for sq in Square::ALL {
        if let Some(piece) = pos.piece_on(sq) {
            let color = pos.color_on(sq).expect("an occupied square has a colour");
            // `relative_to(color)` orients the square to White's view (it flips the
            // rank for Black), so a single White-oriented table serves both sides.
            let square = sq.relative_to(color) as usize;
            let material = value(piece);
            let mg = material + PST_MIDDLEGAME[piece as usize][square];
            let eg = material + PST_ENDGAME[piece as usize][square];
            phase += phase_weight(piece);
            // Idiom: a `match` used as an expression — it yields +1 or -1, so the two
            // running scores are updated without branching into two near-identical
            // bodies. Both sides read the same tables; only the sign differs.
            let sign = match color {
                Color::White => 1,
                Color::Black => -1,
            };
            middlegame += sign * mg;
            endgame += sign * eg;
        }
    }

    let phase = phase.min(MAX_PHASE);
    // Weighted average of the two readings. With every piece on the board this is
    // exactly `middlegame`; with none, exactly `endgame`; in between it slides.
    let balance = (middlegame * phase + endgame * (MAX_PHASE - phase)) / MAX_PHASE;

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

// King in the middlegame: reward the back rank / castled corner, punish walking
// into the centre, where the enemy queen and rooks will find it.
#[rustfmt::skip]
const KING_MIDDLEGAME: [i32; 64] = [
     20, 30, 10,  0,  0, 10, 30, 20, // rank 1: castled squares score highest
     20, 20,  0,  0,  0,  0, 20, 20,
    -10,-20,-20,-20,-20,-20,-20,-10,
    -20,-30,-30,-40,-40,-30,-30,-20,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
    -30,-40,-40,-50,-50,-40,-40,-30,
];

// King in the endgame: the exact reversal. With the heavy pieces gone the king is
// a strong piece, and a pawn rarely promotes without one escorting it — so the
// centre is now worth +40 and the corner -50. Reading this table too early is how
// an engine walks its king into a mating net; reading the other one too late is
// how it draws a won endgame by shuffling on the back rank, which is the defect
// this table exists to fix.
#[rustfmt::skip]
const KING_ENDGAME: [i32; 64] = [
    -50,-30,-30,-30,-30,-30,-30,-50, // rank 1: the corner is now the worst place
    -30,-30,  0,  0,  0,  0,-30,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 30, 40, 40, 30,-10,-30,
    -30,-10, 20, 30, 30, 20,-10,-30,
    -30,-20,-10,  0,  0,-10,-20,-30,
    -50,-40,-30,-20,-20,-30,-40,-50, // rank 8
];

// Order matches `Piece`: pawn, knight, bishop, rook, queen, king. Only the king
// differs between the two sets — every other piece reads the same table in both,
// so adding a genuinely different endgame table for another piece is a separate,
// separately measured change.
const PST_MIDDLEGAME: [[i32; 64]; 6] = [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING_MIDDLEGAME];
const PST_ENDGAME: [[i32; 64]; 6] = [PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING_ENDGAME];

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

    // The same position but for where the White king stands: castled on g1, or out
    // on e4. Queens and rooks are still on, so this is a genuine middlegame — which
    // the previous version of this test was not: it used two bare kings, the purest
    // endgame there is, and asserted middlegame behaviour on it. With one king table
    // that passed; it was the position that was wrong, not the assertion.
    const MIDDLEGAME_KING_SAFE: &str = "r2qk2r/pppp1ppp/8/8/8/8/PPPP1PPP/R2Q1RK1 w kq - 0 1";
    const MIDDLEGAME_KING_OUT: &str = "r2qk2r/pppp1ppp/8/8/4K3/8/PPPP1PPP/R2Q1R2 w kq - 0 1";

    #[test]
    fn the_king_prefers_safety_in_the_middlegame() {
        // Pieces still on the board: the king belongs behind them.
        let safe = Position::from_fen(MIDDLEGAME_KING_SAFE).unwrap();
        let exposed = Position::from_fen(MIDDLEGAME_KING_OUT).unwrap();
        assert!(
            evaluate(&safe) > evaluate(&exposed),
            "castled {} should beat central {}",
            evaluate(&safe),
            evaluate(&exposed),
        );
    }

    #[test]
    fn the_king_prefers_the_centre_in_the_endgame() {
        // The reversal this whole change exists for. Same two king squares, but with
        // nothing left on the board: now the centre is where the king wins pawn races
        // and escorts a passer, and the corner is where games get drawn by shuffling.
        let corner = Position::from_fen("4k3/8/8/8/8/8/8/6K1 w - - 0 1").unwrap(); // Kg1
        let central = Position::from_fen("4k3/8/8/8/4K3/8/8/8 w - - 0 1").unwrap(); // Ke4
        assert!(
            evaluate(&central) > evaluate(&corner),
            "central {} should beat corner {}",
            evaluate(&central),
            evaluate(&corner),
        );
    }

    #[test]
    fn the_same_king_square_is_judged_differently_by_phase() {
        // Not two tables side by side, but one evaluation that changes its mind: the
        // central king is a liability with the queens on and an asset without them.
        // If this fails while the two tests above pass, the tables are right and the
        // interpolation is not wired to the phase.
        let mg_safe = evaluate(&Position::from_fen(MIDDLEGAME_KING_SAFE).unwrap());
        let mg_out = evaluate(&Position::from_fen(MIDDLEGAME_KING_OUT).unwrap());
        let eg_corner = evaluate(&Position::from_fen("4k3/8/8/8/8/8/8/6K1 w - - 0 1").unwrap());
        let eg_central = evaluate(&Position::from_fen("4k3/8/8/8/4K3/8/8/8 w - - 0 1").unwrap());
        assert!(mg_safe > mg_out && eg_central > eg_corner, "the verdict must flip with the phase");
    }

    #[test]
    fn the_phase_runs_from_full_material_to_bare_kings() {
        assert_eq!(phase(&Position::initial()), MAX_PHASE);
        // Kings and pawns only: pawns carry no phase weight, so this is a pure endgame
        // even with sixteen of them on the board.
        let pawns = Position::from_fen("4k3/pppppppp/8/8/8/8/PPPPPPPP/4K3 w - - 0 1").unwrap();
        assert_eq!(phase(&pawns), 0);
        // One queen each, nothing else: 4 + 4.
        let queens = Position::from_fen("3qk3/8/8/8/8/8/8/3QK3 w - - 0 1").unwrap();
        assert_eq!(phase(&queens), 8);
    }

    #[test]
    fn promotions_cannot_push_the_phase_past_its_maximum() {
        // Three queens a side is legal and does happen. Unclamped, the phase would
        // exceed MAX_PHASE, the middlegame table would be weighted by more than 100%
        // and the endgame one by a negative amount — an evaluation built from a
        // weighted average whose weights do not sum to one.
        let many = Position::from_fen("qqqqk3/8/8/8/8/8/8/QQQQK3 w - - 0 1").unwrap();
        // The precondition matters: with three queens a side the raw weight lands on
        // exactly MAX_PHASE, so the clamp would never be exercised and the test would
        // pass with the clamp removed. Four a side is what puts it over.
        let queens = many.count(Color::White, Piece::Queen) + many.count(Color::Black, Piece::Queen);
        assert!(
            queens as i32 * 4 > MAX_PHASE,
            "precondition: raw weight {} must exceed the cap {MAX_PHASE}",
            queens * 4,
        );
        assert_eq!(phase(&many), MAX_PHASE, "the phase must never exceed its maximum");
    }

    #[test]
    fn the_endgame_king_table_peaks_in_the_centre_and_bottoms_in_the_corners() {
        // A structural check on the table itself, not on a position. A table entered
        // upside down, or shifted by a rank, still produces a plausible-looking
        // evaluation — and this is the one property that says which way up it is.
        let at = |name: &str| {
            let sq: Square = name.parse().expect("a square name");
            KING_ENDGAME[sq as usize]
        };
        let centre = ["d4", "e4", "d5", "e5"].map(at);
        let corners = ["a1", "h1", "a8", "h8"].map(at);
        let best = *centre.iter().max().unwrap();
        let worst = *corners.iter().max().unwrap();
        assert!(best > worst, "centre {best} must beat corners {worst}");
        assert_eq!(best, *KING_ENDGAME.iter().max().unwrap(), "the peak is in the centre");
        assert_eq!(worst, *KING_ENDGAME.iter().min().unwrap(), "the floor is in the corners");

        // And the middlegame table says the exact opposite, which is the whole point
        // of having two.
        let mg_corner = KING_MIDDLEGAME["g1".parse::<Square>().unwrap() as usize];
        let mg_centre = KING_MIDDLEGAME["e4".parse::<Square>().unwrap() as usize];
        assert!(mg_corner > mg_centre, "middlegame: {mg_corner} must beat {mg_centre}");
    }

    #[test]
    fn the_phase_reads_material_and_nothing_else() {
        // Not the side to move, not where the kings stand — otherwise the phase would
        // wobble as the kings walk, and every evaluation with it.
        let a = Position::from_fen("4k3/8/8/8/8/8/8/R3K2R w - - 0 1").unwrap();
        let b = Position::from_fen("4k3/8/8/8/8/8/8/R3K2R b - - 0 1").unwrap();
        let c = Position::from_fen("1k6/8/8/8/8/8/8/R5KR w - - 0 1").unwrap();
        assert_eq!(phase(&a), 4, "two rooks");
        assert_eq!(phase(&a), phase(&b), "the side to move is not material");
        assert_eq!(phase(&a), phase(&c), "neither is where the kings stand");
    }

    #[test]
    fn the_interpolation_has_no_cliff() {
        // Removing one piece at a time from a middlegame down to bare kings must move
        // the evaluation gradually. A hard switch at a threshold would show up here as
        // one large jump — and in a game as an engine chasing or dodging the single
        // capture that flips the phase, for no chess reason at all.
        let steps = [
            "r2qk2r/pppp1ppp/8/8/4K3/8/PPPP1PPP/R2Q1R2 w kq - 0 1",
            "r2qk2r/pppp1ppp/8/8/4K3/8/PPPP1PPP/R2Q4 w kq - 0 1",
            "r2qk2r/pppp1ppp/8/8/4K3/8/PPPP1PPP/3Q4 w kq - 0 1",
            "r2qk2r/pppp1ppp/8/8/4K3/8/PPPP1PPP/8 w kq - 0 1",
            "r3k2r/pppp1ppp/8/8/4K3/8/PPPP1PPP/8 w kq - 0 1",
            "r3k3/pppp1ppp/8/8/4K3/8/PPPP1PPP/8 w q - 0 1",
            "4k3/pppp1ppp/8/8/4K3/8/PPPP1PPP/8 w - - 0 1",
        ];
        let scores: Vec<i32> = steps
            .iter()
            .map(|f| evaluate(&Position::from_fen(f).unwrap()))
            .collect();
        // Each step removes at most a queen (900) plus its square bonus, so any jump
        // far beyond that is the interpolation misbehaving rather than the material.
        for pair in scores.windows(2) {
            let jump = (pair[1] - pair[0]).abs();
            assert!(jump < 1_100, "jump of {jump} between phases: {scores:?}");
        }
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
