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

use std::sync::LazyLock;

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
/// For each colour and square, every square from which an enemy pawn could stop the pawn
/// standing there: its own file and the two beside it, on every rank ahead of it.
///
/// A pawn is passed exactly when this mask and the enemy pawns do not intersect — one `&`
/// instead of walking up to 21 squares. Computed once, because `evaluate` is the hottest
/// function in the engine and #29 died of costing 0.23 µs a node.
///
/// Idiom: `LazyLock` runs its closure on first access and hands out the same value
/// thereafter. Needed because the loops below are not `const`-evaluable in the form written
/// here, and a table computed per call would defeat the point of having one.
static PASSED_MASK: LazyLock<[[u64; 64]; 2]> = LazyLock::new(|| {
    let mut masks = [[0u64; 64]; 2];
    // Idiom: split the two colour rows first, so each inner loop writes through a plain
    // `&mut [u64; 64]` indexed by the loop variable — clippy flags `masks[c][square]` inside
    // a loop over `square` as a pattern better expressed by iterating.
    let (white, black) = masks.split_at_mut(1);
    let (white, black) = (&mut white[0], &mut black[0]);
    for square in 0..64usize {
        let (file, rank) = (square % 8, square / 8);
        for other in 0..64usize {
            let (other_file, other_rank) = (other % 8, other / 8);
            // Adjacent files include the pawn's own: a pawn on the same file ahead blocks
            // just as surely as one that can capture.
            if other_file.abs_diff(file) > 1 {
                continue;
            }
            // "Ahead" is the direction the pawn moves, so it flips with the colour.
            if other_rank > rank {
                white[square] |= 1u64 << other;
            }
            if other_rank < rank {
                black[square] |= 1u64 << other;
            }
        }
    }
    masks
});

/// What a passed pawn is worth, by the rank it has reached from its own side's view — index
/// 0 is the home rank, index 7 the promotion square.
///
/// Two schedules, read by the same phase interpolation as the piece-square tables, and the
/// endgame one is far steeper. That difference *is* the term: in a middlegame a passed pawn
/// is a long-term asset among many, while in an endgame it is often the whole position. The
/// growth is faster than linear because a pawn two squares from queening is not twice a pawn
/// four squares away — the defender's task changes in kind, not in degree.
///
/// Ranks 0 and 7 are zero on purpose: a pawn cannot stand on its own home rank, and one that
/// reaches the eighth is no longer a pawn.
const PASSED_MIDDLEGAME: [i32; 8] = [0, 5, 10, 18, 32, 55, 85, 0];
const PASSED_ENDGAME: [i32; 8] = [0, 8, 13, 21, 36, 60, 90, 0];

/// The square immediately in front of a pawn of `color` standing on `square`, or `None` if
/// there is none — which for a pawn can only mean the promotion rank, since a pawn never
/// stands on its own first rank.
///
/// "In front" is the direction of travel, so it flips with the colour: one rank up the board
/// for White, one down for Black. The board is laid out `a1` = 0 with eight squares per rank,
/// so that is ±8.
fn square_ahead(square: usize, color: Color) -> Option<usize> {
    match color {
        Color::White if square < 56 => Some(square + 8),
        Color::Black if square >= 8 => Some(square - 8),
        _ => None,
    }
}

/// Whether the pawn on `square` (given as an index, `a1` = 0) has no enemy pawn able to stop
/// it — nothing on its file or the two beside it, anywhere ahead.
fn is_passed(square: usize, color: Color, enemy_pawns: u64) -> bool {
    PASSED_MASK[color as usize][square] & enemy_pawns == 0
}

/// Centipawns per square commanded, by piece, in the middlegame and in the endgame.
///
/// **Why this term exists, and it was located rather than chosen from a list.** Over 29 276
/// positions from 6 400 anchored games, our search score was compared to Stockfish's on the same
/// position. Our error correlates with the mobility difference at **−0.31**, the strongest of the
/// eight features tested — and, unlike the others, it correlates *identically* in games we win
/// (−0.29) and games we lose (−0.31). A feature that only appeared inside losses would describe
/// what losing looks like; one that holds in both is a blind spot. The sign says we are optimistic
/// when our pieces are cramped.
///
/// **Why only the minor pieces, and this was measured rather than reasoned.** The first version
/// weighted rooks and queens too, and the tree **doubled** — 733 605 nodes against 371 194 at
/// depth 10 on the start position. A queen's mobility swings between five squares and
/// twenty-five, and feeding that swing into the evaluation moves scores past the futility margins
/// that decide whether a node is cut at all. The knights and bishops carry the signal; the heavy
/// pieces carried noise, and the noise was expensive. Weights at zero for pawns and the king for
/// the ordinary reasons — a pawn's "mobility" is two capture squares, and the king's is king
/// safety, a different term that measured neutral here (#29).
///
/// | weights (N,B,R,Q) | nodes, opening | nodes, ruy-lopez |
/// |---|---|---|
/// | 4,4,3,2 | ×1.98 | ×1.02 |
/// | 2,2,2,1 | ×1.28 | ×1.19 |
/// | 1,1,1,1 | ×1.28 | ×1.61 |
/// | **3,3,0,0** | **×0.98** | **×1.02** |
///
/// **Why tapered.** Everything in this evaluation is. A queen's freedom matters less in a
/// middlegame full of pieces than in an endgame where it decides; a rook's matters more once the
/// files open.
const MOBILITY_MIDDLEGAME: [i32; 6] = [0, 3, 3, 0, 0, 0];
const MOBILITY_ENDGAME: [i32; 6] = [0, 3, 4, 0, 0, 0];

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

    // Read once, outside the loop: both are needed for every pawn encountered, and they do
    // not change while the board is being walked.
    let pawns = [pos.pawns(Color::White), pos.pawns(Color::Black)];

    for sq in Square::ALL {
        if let Some(piece) = pos.piece_on(sq) {
            let color = pos.color_on(sq).expect("an occupied square has a colour");
            // `relative_to(color)` orients the square to White's view (it flips the
            // rank for Black), so a single White-oriented table serves both sides.
            let square = sq.relative_to(color) as usize;
            let material = value(piece);
            let mut mg = material + PST_MIDDLEGAME[piece as usize][square];
            let mut eg = material + PST_ENDGAME[piece as usize][square];
            // The passed-pawn bonus, added in this same pass rather than in a second walk
            // over the board — the cost of a second walk is what ended #29.
            //
            // `square` is already oriented to the pawn's own side, so `square / 8` is the
            // rank it has advanced to whatever its colour. The mask lookup, however, needs
            // the *absolute* square, since it is about where the enemy pawns really are.
            if piece == Piece::Pawn {
                let enemy = pawns[!color as usize];
                if is_passed(sq as usize, color, enemy) {
                    let rank = square / 8;
                    let (mut bonus_mg, mut bonus_eg) =
                        (PASSED_MIDDLEGAME[rank], PASSED_ENDGAME[rank]);
                    // A passed pawn with something standing on the square in front of it is
                    // not running anywhere. The schedule above prices a pawn by how far it
                    // has come; this asks whether it can still go further, which is the one
                    // piece of context that costs a single lookup.
                    //
                    // Halved rather than removed: a blockaded passer still ties down the
                    // piece blockading it, and the blockade can be broken.
                    if square_ahead(sq as usize, color)
                        .is_some_and(|front| pos.piece_on(Square::index(front)).is_some())
                    {
                        bonus_mg /= 2;
                        bonus_eg /= 2;
                    }
                    mg += bonus_mg;
                    eg += bonus_eg;
                }
            }
            // Mobility, riding the walk that is already happening rather than adding a second
            // one — the cost of a second pass over the board is what ended #29, and this term has
            // to be cheap enough not to buy accuracy with depth.
            //
            // **The zero-weight pieces are skipped, not multiplied by zero**, because the lookup
            // is the cost and the multiply is free. With rooks and queens at zero this saves the
            // two most expensive probes — a queen costs a rook lookup *and* a bishop lookup — on
            // every occupied square that holds one. Both arrays are `const`, so the test folds
            // away at compile time for every piece whose weights are zero.
            if MOBILITY_MIDDLEGAME[piece as usize] != 0 || MOBILITY_ENDGAME[piece as usize] != 0 {
                let mob = pos.mobility_from(sq, piece, color) as i32;
                mg += MOBILITY_MIDDLEGAME[piece as usize] * mob;
                eg += MOBILITY_ENDGAME[piece as usize] * mob;
            }
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

        // The same property with a passed pawn on the board, because the passed term is the
        // one that reads a *direction* — "ahead" flips with the colour, and a mask indexed
        // by the wrong side would pass every test above while scoring one colour's pawns as
        // permanently passed.
        //
        // The mirror is *computed*, not written out. Transcribing one by hand is how this
        // test first failed: rank 6 was mirrored to rank 4 instead of rank 3, and the
        // evaluation was blamed for a typo in its own test.
        for fen in [
            "4k3/8/8/3P4/8/8/8/4K3 w - - 0 1",   // lone passer
            "4k3/8/8/3P4/2p5/8/8/4K3 w - - 0 1", // enemy pawn behind: still passed
            "4k3/8/2p5/3P4/8/8/8/4K3 w - - 0 1", // enemy pawn ahead on an adjacent file
            "4k3/8/8/3P4/3p4/8/8/4K3 w - - 0 1", // blocked head on
        ] {
            let mirrored = mirror(fen);
            let (a, b) = (
                Position::from_fen(fen).unwrap(),
                Position::from_fen(&mirrored).unwrap(),
            );
            assert_eq!(evaluate(&a), evaluate(&b), "`{fen}` mirrors to `{mirrored}`");
        }
    }

    // The same position seen from the other side: ranks reversed, colours swapped, side to
    // move swapped. Only the board and the side-to-move fields matter here, and every test
    // position below is castling- and en-passant-free.
    fn mirror(fen: &str) -> String {
        let (board, rest) = fen.split_once(' ').expect("a board and a side to move");
        let flipped: Vec<String> = board
            .split('/')
            .rev() // rank 8 first becomes rank 1 first
            .map(|rank| {
                rank.chars()
                    .map(|c| {
                        if c.is_ascii_uppercase() {
                            c.to_ascii_lowercase()
                        } else if c.is_ascii_lowercase() {
                            c.to_ascii_uppercase()
                        } else {
                            c // a digit: a run of empty squares, unchanged
                        }
                    })
                    .collect()
            })
            .collect();
        let side = if rest.starts_with('w') { 'b' } else { 'w' };
        format!("{} {} {}", flipped.join("/"), side, &rest[2..])
    }

    // --- passed pawns ---------------------------------------------------------

    // Is the pawn on `square` passed, given the enemy pawns of the position in `fen`?
    //
    // Drives the same predicate the evaluation uses, rather than reading a score difference:
    // a score can move for a dozen reasons and would make a failure here ambiguous.
    fn passed_in(fen: &str, square: &str, color: Color) -> bool {
        let pos = Position::from_fen(fen).unwrap();
        let sq: Square = square.parse().expect("a square name");
        is_passed(sq as usize, color, pos.pawns(!color))
    }

    #[test]
    fn the_mask_rows_are_indexed_by_colour_the_way_the_lookup_assumes() {
        // `PASSED_MASK` is built by splitting the array in two and filling `[0]` with White's
        // masks, then read back as `PASSED_MASK[color as usize]`. That is only correct while
        // `White as usize == 0`, which is a fact about a borrowed crate rather than about this
        // file — so it is asserted here instead of assumed. If it ever flips, every pawn of
        // both colours is judged against the wrong direction, and the engine would still
        // compile and still play.
        assert_eq!(Color::White as usize, 0);
        assert_eq!(Color::Black as usize, 1);
        // And the rows really do differ, so a symmetric bug cannot hide behind the indices
        // being right: a pawn on d5 looks forwards for White and backwards for Black.
        let d5 = Square::D5 as usize;
        assert_ne!(
            PASSED_MASK[Color::White as usize][d5],
            PASSED_MASK[Color::Black as usize][d5],
        );
    }

    #[test]
    fn a_pawn_is_passed_when_no_enemy_pawn_can_stop_it() {
        // Every way a pawn can be stopped, and the two ways it cannot. Swept as a table so
        // that adding a case is one line, and so a failure names which relationship broke.
        // No type annotation at all: the literals below determine both the element type and
        // the length, so adding a row is one line. The original spelled the count into the
        // type (`; 8]`), which turns adding a row into a compile error to chase — friction on
        // exactly the action this table exists to make cheap, and part of why the row replaced
        // below sat here unexamined.
        //
        // Not `[_; _]` either, tempting as it reads: inferring an array length is
        // `generic_arg_infer`, stabilised well after the `rust-version = "1.85"` this workspace
        // declares. Omitting the annotation needs no such feature.
        let cases = [
            ("8/8/8/3P4/8/8/8/K6k w - - 0 1", "d5", Color::White, true,
             "nothing in front at all"),
            ("8/3p4/8/3P4/8/8/8/K6k w - - 0 1", "d5", Color::White, false,
             "enemy pawn on the same file ahead"),
            ("8/2p5/8/3P4/8/8/8/K6k w - - 0 1", "d5", Color::White, false,
             "enemy pawn on the file to the left, ahead"),
            ("8/4p3/8/3P4/8/8/8/K6k w - - 0 1", "d5", Color::White, false,
             "enemy pawn on the file to the right, ahead"),
            ("8/8/8/3P4/2p5/8/8/K6k w - - 0 1", "d5", Color::White, true,
             "enemy pawn adjacent but BEHIND — it can never come back"),
            // `f7` is exactly two files from `d5` and ahead of it — the first square
            // *outside* the window. The row it replaces used `a3`, which is three files away
            // *and* behind: excluded twice over, so it could not discriminate the file
            // boundary, while its comment claimed it did. Raised in review, and it is the
            // seventh comment in this repository describing a stronger check than the code
            // performs.
            ("8/5p2/8/3P4/8/8/8/K6k w - - 0 1", "d5", Color::White, true,
             "two files away and ahead — just outside the window"),
            // `c5` is on an adjacent file at the *same* rank. A pawn beside ours moves away
            // from us and can never come back, so it must not count as a stopper. This is the
            // rank boundary, and nothing tested it either.
            ("8/8/8/2pP4/8/8/8/K6k w - - 0 1", "d5", Color::White, true,
             "adjacent file, same rank — it moves away, it cannot stop us"),
            // Black pawns run the other way: the same geometry must flip.
            ("8/8/8/3p4/8/8/8/K6k w - - 0 1", "d5", Color::Black, true,
             "black pawn with nothing in front of it"),
            ("8/8/8/3p4/2P5/8/8/K6k w - - 0 1", "d5", Color::Black, false,
             "black pawn with a white pawn ahead on an adjacent file"),
        ];
        for (fen, square, color, expected, why) in cases {
            assert_eq!(
                passed_in(fen, square, color),
                expected,
                "{why} — `{fen}` square {square} for {color:?}",
            );
        }
    }

    #[test]
    fn a_blockaded_passer_is_worth_less_than_a_running_one() {
        // The context the bare schedule ignores: a passed pawn with something standing in
        // front of it is not going anywhere, yet a rank-6 passer blocked by a king was priced
        // exactly like one with an open road.
        //
        // Compared against its own baseline rather than against the free pawn directly: the
        // blocking piece carries its own material and square value, which would swamp the
        // difference being measured.
        let free = Position::from_fen("4k3/8/8/3P4/8/8/8/4K3 w - - 0 1").unwrap();
        let blocked = Position::from_fen("4k3/8/3n4/3P4/8/8/8/4K3 w - - 0 1").unwrap();
        let bare = Position::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let knight_only = Position::from_fen("4k3/8/3n4/8/8/8/8/4K3 w - - 0 1").unwrap();
        let running = evaluate(&free) - evaluate(&bare);
        let stopped = evaluate(&blocked) - evaluate(&knight_only);
        assert!(
            stopped < running,
            "a blockaded passer must be worth less: {stopped} against {running}",
        );
        // And still worth *more than a pawn that is not passed at all*, because halving rather
        // than removing is the design choice.
        //
        // Comparing `stopped > 0` would not test that: a pawn is worth 100 centipawns before
        // any bonus, so that assertion stays true with the bonus zeroed. Found by mutation —
        // "zero the bonus instead of halving it" broke nothing. The residual has to be
        // isolated against a pawn on the *same square* that is blocked *and* not passed.
        let blocked_not_passed =
            Position::from_fen("4k3/8/3p4/3P4/8/8/8/4K3 w - - 0 1").unwrap();
        let pawn_only = Position::from_fen("4k3/8/3p4/8/8/8/8/4K3 w - - 0 1").unwrap();
        let no_bonus_at_all = evaluate(&blocked_not_passed) - evaluate(&pawn_only);
        assert!(
            stopped > no_bonus_at_all,
            "a blockaded passer must keep part of its bonus: {stopped} against \
             {no_bonus_at_all} for a pawn that is blocked and not passed",
        );
    }

    #[test]
    fn the_square_ahead_follows_the_direction_of_travel() {
        // "In front" flips with the colour, and getting it backwards would halve the bonus of
        // every pawn with a piece *behind* it while leaving genuinely blockaded ones at full
        // value — a mistake that changes no test above and no compile.
        assert_eq!(square_ahead(Square::D5 as usize, Color::White), Some(Square::D6 as usize));
        assert_eq!(square_ahead(Square::D5 as usize, Color::Black), Some(Square::D4 as usize));
        // The promotion rank has nothing ahead of it. A pawn never stands there, but the
        // lookup must not wrap around the board into a square on the other side.
        assert_eq!(square_ahead(Square::D8 as usize, Color::White), None);
        assert_eq!(square_ahead(Square::D1 as usize, Color::Black), None);
        // Every other square yields a real neighbour on the same file.
        for sq in 8..56usize {
            for color in [Color::White, Color::Black] {
                let ahead = square_ahead(sq, color).expect("a square in the middle has one");
                assert_eq!(ahead % 8, sq % 8, "the file must not change");
                assert_eq!(ahead.abs_diff(sq), 8, "exactly one rank");
            }
        }
    }

    #[test]
    fn a_friendly_piece_blocks_just_as_an_enemy_one_does() {
        // A deliberate choice, recorded because the usual engines only count enemy blockers:
        // this one counts any piece. A pawn cannot advance through its own knight either, and
        // the simpler rule is the one being measured. If a later brick distinguishes the two,
        // this test is what will have to change, and on purpose.
        // The black king sits on a8, not e8: a white knight on d6 attacks e8, and a position
        // where the side *not* to move is in check is illegal — `from_fen` rejects it.
        let own = Position::from_fen("k7/8/3N4/3P4/8/8/8/4K3 w - - 0 1").unwrap();
        let own_baseline = Position::from_fen("k7/8/3N4/8/8/8/8/4K3 w - - 0 1").unwrap();
        let free = Position::from_fen("k7/8/8/3P4/8/8/8/4K3 w - - 0 1").unwrap();
        let bare = Position::from_fen("k7/8/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        assert!(
            evaluate(&own) - evaluate(&own_baseline) < evaluate(&free) - evaluate(&bare),
            "a pawn cannot advance through its own piece either",
        );
    }

    #[test]
    fn the_passed_bonus_grows_with_rank() {
        // Asserted as monotonicity over the whole schedule rather than as values, so that
        // retuning the numbers cannot fail this test for a reason unrelated to its name.
        // Ranks 0 and 7 are excluded: a pawn cannot stand on its home rank, and one on the
        // eighth is no longer a pawn — both are zero by construction.
        for table in [PASSED_MIDDLEGAME, PASSED_ENDGAME] {
            for rank in 1..6 {
                assert!(
                    table[rank + 1] > table[rank],
                    "rank {rank} -> {} went {} -> {}",
                    rank + 1,
                    table[rank],
                    table[rank + 1],
                );
            }
            assert_eq!(table[0], 0, "a pawn cannot stand on its own home rank");
            assert_eq!(table[7], 0, "a pawn on the eighth rank has already promoted");
        }
    }

    #[test]
    fn a_passed_pawn_is_worth_more_in_the_endgame() {
        // The whole reason this term is tapered rather than a single schedule: in a
        // middlegame a passed pawn is one asset among many, in an endgame it is often the
        // position. Checked at every rank a pawn can occupy, not at one chosen rank.
        for rank in 1..7 {
            assert!(
                PASSED_ENDGAME[rank] > PASSED_MIDDLEGAME[rank],
                "rank {rank}: endgame {} is not above middlegame {}",
                PASSED_ENDGAME[rank],
                PASSED_MIDDLEGAME[rank],
            );
        }
    }

    #[test]
    fn a_passed_pawn_scores_above_an_identical_blocked_one() {
        // The end-to-end check: the predicate and the schedule reaching `evaluate`. Two
        // positions differing only in whether one enemy pawn stands in the way.
        let passed = Position::from_fen("4k3/8/8/3P4/8/8/8/4K3 w - - 0 1").unwrap();
        let blocked = Position::from_fen("4k3/3p4/8/3P4/8/8/8/4K3 w - - 0 1").unwrap();
        // The blocked position also contains an extra enemy pawn, so compare each against
        // its own baseline rather than against the other directly.
        let bare = Position::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let with_enemy = Position::from_fen("4k3/3p4/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let gain_when_passed = evaluate(&passed) - evaluate(&bare);
        let gain_when_blocked = evaluate(&blocked) - evaluate(&with_enemy);
        assert!(
            gain_when_passed > gain_when_blocked,
            "a passed pawn must be worth more than a blocked one: {gain_when_passed} \
             against {gain_when_blocked}",
        );
    }
    // ------------------------------------------------------------------- mobility (#77)

    #[test]
    fn mobility_counts_only_squares_a_piece_could_move_to() {
        // **The definition of the term, and the one line nothing else asserts.** `mobility_from`
        // excludes squares occupied by one's own side; mutating it to count them left the entire
        // suite green, because the error is *symmetric* — the mirror test still reads zero — and
        // it preserves the ordering, so the cramped-versus-free comparison still holds. Every
        // other test here compares two positions, and this defect moves both.
        //
        // Asserted as a statement about chess rather than as a number: on the initial board, a
        // bishop behind its own pawns commands nothing at all, and a rook in the corner nothing
        // either. Counting own pieces would make them 2 each.
        let p = Position::initial();
        assert_eq!(
            p.mobility_from(Square::C1, Piece::Bishop, Color::White), 0,
            "a bishop on its initial square is blocked by its own pawn and its own knight",
        );
        assert_eq!(
            p.mobility_from(Square::A1, Piece::Rook, Color::White), 0,
            "a rook in the corner is blocked by its own pawn and its own knight",
        );
        assert_eq!(
            p.mobility_from(Square::D1, Piece::Queen, Color::White), 0,
            "a queen on its initial square commands nothing: every ray runs into its own side",
        );
        // The control that keeps the three zeros from proving nothing: the knight *can* jump over
        // its own pieces, and reads exactly the two squares it really has.
        assert_eq!(
            p.mobility_from(Square::B1, Piece::Knight, Color::White), 2,
            "a knight on b1 has exactly a3 and c3 — c1 and d2 are its own pieces",
        );
        // And once the board opens, the same bishop counts: without the control above, a term
        // that always returned zero would satisfy the assertions.
        let opened = Position::from_fen("rnbqkbnr/pppppppp/8/8/8/4P3/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
            .unwrap();
        assert!(
            opened.mobility_from(Square::F1, Piece::Bishop, Color::White) > 0,
            "precondition: with the e-pawn advanced the light bishop must see something",
        );
    }

    #[test]
    fn mobility_cancels_on_a_mirrored_position() {
        // The load-bearing control. A term that did not cancel on a symmetric position would be
        // a side-to-move bonus wearing a positional name — and it would show up as strength in
        // a duel against our own clone while being worth nothing at all.
        for fen in [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
            "4k3/8/8/3n4/3N4/8/8/4K3 w - - 0 1",
        ] {
            let p = Position::from_fen(fen).unwrap();
            assert_eq!(
                evaluate(&p), 0,
                "{fen}: a mirrored position must evaluate to exactly zero",
            );
        }
    }

    #[test]
    fn a_cramped_side_is_scored_below_a_free_one() {
        // The term's whole claim, on identical material. Both sides have a knight and a bishop;
        // White's stand in the centre, Black's are boxed into the corner behind their own pawns.
        let free = Position::from_fen("4k3/pppppppp/8/8/3NB3/8/PPPPPPPP/4K3 w - - 0 1").unwrap();
        let cramped = Position::from_fen("4k3/pppppppp/8/8/8/8/PPPPPPPP/N3K2B w - - 0 1").unwrap();
        // Material is identical by construction, so any difference is positional. The
        // piece-square tables also prefer the centre, which is why the assertion is on the
        // *difference between two versions of this evaluation* rather than on a raw number:
        // see `mobility_is_what_makes_the_difference` below.
        assert!(
            evaluate(&free) > evaluate(&cramped),
            "free pieces {} must beat cramped ones {}",
            evaluate(&free), evaluate(&cramped),
        );
    }

    #[test]
    fn mobility_is_what_makes_the_difference_not_the_tables() {
        // The control the test above needs: the piece-square tables also reward the centre, so
        // "free scores higher than cramped" does not by itself say *this* term did anything.
        //
        // The contribution is recomputed here from the same constants and the same lookup the
        // evaluation uses, rather than by re-walking the board in a second implementation — a
        // first version did that and was **inert**: the copy drifted from the original, so the
        // difference it measured was its own bug rather than the term. Zeroing the weights left
        // it green, which is how it was caught.
        let free = Position::from_fen("4k3/pppppppp/8/8/3NB3/8/PPPPPPPP/4K3 w - - 0 1").unwrap();
        let cramped = Position::from_fen("4k3/pppppppp/8/8/8/8/PPPPPPPP/N3K2B w - - 0 1").unwrap();
        assert!(
            mobility_contribution(&free) > mobility_contribution(&cramped),
            "the term itself must separate these two: {} against {}",
            mobility_contribution(&free), mobility_contribution(&cramped),
        );
    }

    // What the mobility term adds to White's side of the score, from the same constants and the
    // same lookup `evaluate` uses. Zero when the weights are zero, which is what makes the test
    // above sensitive to them.
    fn mobility_contribution(pos: &Position) -> i32 {
        let mut total = 0;
        for sq in Square::ALL {
            if let Some(piece) = pos.piece_on(sq) {
                if pos.color_on(sq) == Some(Color::White) {
                    total += MOBILITY_MIDDLEGAME[piece as usize]
                        * pos.mobility_from(sq, piece, Color::White) as i32;
                }
            }
        }
        total
    }

    #[test]
    fn mobility_is_tapered_like_everything_else() {
        // Every term in this evaluation is phase-dependent, and this one has to be too: a
        // bishop's freedom is worth more once the board empties. Asserted on the constants
        // rather than through a position, because a position mixes the taper with the tables.
        assert_ne!(
            MOBILITY_MIDDLEGAME, MOBILITY_ENDGAME,
            "the term must differ between phases, or it is not tapered",
        );
        for piece in [Piece::Knight, Piece::Bishop] {
            assert!(
                MOBILITY_ENDGAME[piece as usize] >= MOBILITY_MIDDLEGAME[piece as usize],
                "{piece:?} mobility must not be worth less in the endgame",
            );
        }
        // And the heavy pieces stay out, which is a measured decision rather than an oversight:
        // weighting them doubled the tree. See the constant's own comment for the sweep.
        for piece in [Piece::Rook, Piece::Queen, Piece::Pawn, Piece::King] {
            assert_eq!(
                (MOBILITY_MIDDLEGAME[piece as usize], MOBILITY_ENDGAME[piece as usize]),
                (0, 0),
                "{piece:?} must stay at zero: weighting it cost a factor of two in nodes",
            );
        }
    }

}
