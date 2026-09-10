//! Static evaluation of a position.
//!
//! Two terms, both in **centipawns** (1 pawn = 100) and both from the
//! side-to-move perspective:
//!
//! - **material** — how much each side has;
//! - **piece-square tables (PST)** — a bonus for *where* each piece sits, so the
//!   engine develops toward the centre, advances central pawns, and keeps its
//!   king safe instead of shuffling material-equal positions aimlessly.
//!
//! The two are **not separately identifiable**, and it is worth knowing before
//! reading either: `evaluate` adds `value(piece)` to every one of a table's 64
//! squares, so only their sum has meaning. Splitting it back out is a readability
//! choice — `value` carries the mean and the tables carry the deviation.
//!
//! The numbers were **fitted**, not chosen: the evaluation is linear in them, so
//! "are they good" is a convex problem rather than an opinion. Sparse logistic
//! regression against an unrestricted Stockfish, on quiescence-leaf positions drawn
//! at most one per game from this engine's own games. They started life as the
//! Chess Programming Wiki's "simplified" tables and no longer resemble them.
//!
//! # Tapered evaluation
//!
//! One square is worth different things at different moments of a game — a rook is
//! worth more once the files open, a pawn more the closer the endgame gets, and the
//! king is the extreme case: sheltered in a corner while the queens are on, and
//! marching to the centre once they are off. A single table cannot say both, so
//! **every piece has two** — `PAWN_MG` and `PAWN_EG`, and so on to [`KING_MG`]
//! and [`KING_EG`] — and the score is **interpolated** between them according to how
//! much material is left ([`phase`]).
//!
//! Until these tables were fitted, five of the six pieces read the *same* array in
//! both phases and only the king had two, so this whole mechanism moved nothing but
//! the king. Measured when it was fixed: of the 3.03 % of held-out cross-entropy the
//! refit bought, **0.51 points came from the taper and 2.52 from the levels**. The
//! shared tables were a real defect, and a smaller one than the numbers themselves.
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
        Piece::Pawn => 105,
        Piece::Knight => 298,
        Piece::Bishop => 299,
        Piece::Rook => 513,
        Piece::Queen => 956,
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

/// What being **passed** adjusts a pawn by, on top of its square, by the rank it has reached
/// from its own side's view — index 0 is the home rank, index 7 the promotion square.
///
/// Two schedules, read by the same phase interpolation as the piece-square tables, and both
/// are fitted rather than chosen. Ranks 0 and 7 are zero on purpose: a pawn cannot stand on its
/// own home rank, and one that reaches the eighth is no longer a pawn.
///
/// **The middlegame schedule is negative until the fifth rank, and that is a finding rather
/// than noise.** These are the best-observed columns in the whole fit — 54 213 net occurrences
/// at rank 2 against 17 376 at rank 7 — so the sign is not a small-sample artefact. A passed
/// pawn that has not moved is a pawn on a half-open file: the file is a highway for the enemy
/// rook, the pawn is a target, and the enemy pawns that are not in front of it are massed
/// somewhere else. What the schedule says is that a passer only becomes an asset once it is
/// close enough to run, which is the same statement the endgame schedule makes at every rank —
/// it is above the middlegame one throughout, by 34 to 92 cp.
///
/// **What this does to the halving below, and it is worth stating because the sign flipped
/// under it.** A blockaded passer has its adjustment divided by two. That was written when
/// every value here was positive and read as "a blockaded passer keeps part of its bonus"; with
/// negative ranks it now also reads as "a blockaded back passer is half as much of a liability".
/// Both are the same rule — being passed matters half as much when the pawn cannot advance —
/// and halving moves the adjustment toward zero whichever side of zero it starts on.
///
/// The values are forced **even** so that `bonus /= 2` on an `i32` is exact. An odd value would
/// truncate toward zero and cost half a centipawn that the fit's model does not know about.
const PASSED_MIDDLEGAME: [i32; 8] = [0, -30, -52, -50, -18, 64, 116, 0];
const PASSED_ENDGAME: [i32; 8] = [0, 20, 12, 42, 66, 110, 138, 0];

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
/// **Why only the minor pieces, and it was measured rather than reasoned (2026-08).** The first
/// version weighted rooks and queens too, and the tree **doubled** — 733 605 nodes against
/// 371 194 at depth 10 on the start position. A queen's mobility swings between five squares and
/// twenty-five, and feeding that swing into the evaluation moves scores past the futility margins
/// that decide whether a node is cut at all. The sweep that settled it, and its numbers are a
/// historical record of that day rather than a description of the weights below:
///
/// | weights tried (N,B,R,Q), 2026-08 | nodes, opening | nodes, ruy-lopez |
/// |---|---|---|
/// | 4,4,3,2 | ×1.98 | ×1.02 |
/// | 2,2,2,1 | ×1.28 | ×1.19 |
/// | 1,1,1,1 | ×1.28 | ×1.61 |
/// | 3,3,0,0 — the hand-picked pair the fit replaced | ×0.98 | ×1.02 |
///
/// **What that decision costs, priced in 2026-09.** On 265 248 quiescence leaves, after removing
/// the best *monotone* function of this evaluation — the part worth 0 Elo by construction — the
/// net rook-and-queen mobility is **the largest single blind spot left**, and by a distance: a
/// per-stratum constant on it removes **6.7 %** of the residual variance, against 0.6 % for the
/// knights and bishops that are scored here and 0.03 % for the game phase. Measured with the same
/// number of rooks and queens on both sides, so it is the pieces' freedom and not their count.
/// The decision above is not overturned by that figure — it was about the size of the tree, which
/// a static fit cannot see — but the trade is now priced, and a cheaper formulation deserves its
/// own issue.
///
/// **The weights below are fitted, not chosen**, and the two surprises in them are recorded
/// rather than smoothed. A **bishop**'s freedom is worth three times more with the pieces on than
/// in an endgame: its mobility is what separates a good bishop from a bad one, and that is a
/// question about a dense pawn structure. A **knight**'s is worth nothing at all in the
/// middlegame and something in the endgame — a knight's reach is nearly a function of its square
/// once own pieces are discounted, so the piece-square table already carries it, and only on an
/// emptier board does the residual variation say anything. Zero for pawns and the king for the
/// ordinary reasons: a pawn's "mobility" is two capture squares, and the king's is king safety, a
/// different term that measured neutral here (#29).
///
/// The mobility feature was **centred** before fitting — `weight × mobility` splits into
/// `weight × mean`, which is indistinguishable from material, plus `weight × (mobility − mean)`,
/// which is the part that varies — and the mean was folded back into [`value`]. Without that
/// split the fit moves piece value into this array and back out again at random.
const MOBILITY_MIDDLEGAME: [i32; 6] = [0, 0, 18, 0, 0, 0];
const MOBILITY_ENDGAME: [i32; 6] = [0, 8, 6, 0, 0, 0];

/// By how much a pawnless material edge is divided when it cannot mate.
///
/// **Not zero, and that is deliberate.** A drawn endgame is drawn, but returning exactly 0 would
/// make every such position identical and leave the search nothing to steer by — reaching K+N
/// against K is still better than being mated, and the engine has to be able to say so. Dividing
/// keeps the ordering while removing the illusion of a win.
const DRAWISH_DIVISOR: i32 = 8;

// The switch the tests use to hold the factor off, so a comparison is between two verdicts of
// **one** binary rather than between two builds. Compiled out of production entirely.
//
// Rust idiom: a thread-local `Cell` rather than a parameter, because `evaluate` is called from the
// hottest loop in the engine and threading a flag through every call site would change a signature
// the search, the quiescence and four tests all share. A plain comment and not a doc comment —
// `thread_local!` expands to several items, so rustdoc has nothing single to attach it to and
// clippy says so.
#[cfg(test)]
thread_local! {
    static SCALING: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}

// How many times `evaluate` has been called on this thread. **Tests only**, and it exists for one
// measurement: the share of a node's cost that this function represents (#92). That share is
// `calls_per_node * cost_per_call / cost_per_node`, and the first factor cannot be assumed to be
// one -- `evaluate` runs at every quiescence stand-pat but only at the interior nodes that can use
// a static score, so the ratio has to be read rather than guessed.
//
// A thread-local `Cell` and not an `AtomicU64` on purpose. The atomic would be a `lock xadd` in
// the hottest function of the engine, tens of cycles against the one or two a thread-local
// increment costs, and it would show up in the very figure being measured. The price of the
// choice is that only the calling thread's calls are visible, which is exactly right here: every
// instrument in this project runs on the single-threaded path, and `cost.rs` measures its own
// thread.
//
// The increment is still not free, so `cost.rs` prices it rather than assuming it away. There is no
// row of its own for it: what prices it is the `evaluate (production fn)` row against `full (copy)`,
// two names for one source where only the first carries this increment and the thread-local read of
// `SCALING`. Their difference is printed as the parenthesis under `-> evaluate is`.
#[cfg(test)]
thread_local! {
    static EVAL_CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// The measurement harness for this module's cost, run explicitly with `--ignored`.
#[cfg(test)]
mod cost;

/// Below this edge, a pawnless side cannot be assumed to be winning.
///
/// A rook is the threshold because K+R against K **is** a win and K+B or K+N against K is not.
/// Everything between — two minors at 640, for instance — needs its own condition rather than a
/// wider net, which is why two knights are named explicitly below and bishop-plus-knight is not:
/// B+N against a bare king is a genuine win, and flattening it would throw away real games.
const PAWNLESS_WIN_THRESHOLD: i32 = 500;

/// Whether `balance` describes a material edge that cannot be converted, so that calling it a win
/// is what makes the engine trade into it.
///
/// **Measured, not assumed.** Over 10 400 anchored games, 44% of our draws were positions we had
/// been winning by +300 or more; 23% of those ended with **no pawn at all** on our side, at a
/// median edge of +320 cp, and the most frequent final positions were a lone knight or a lone
/// bishop against a bare king. `evaluate` was returning about +320 for those, so the search walked
/// into them and then shuffled until the threefold repetition — 55% of all our draws.
fn cannot_mate(pos: &Position, pawns: [u64; 2]) -> bool {
    // **No pawn anywhere on the board**, and the restriction is measured rather than cautious.
    // A first version asked only that the *strong* side be pawnless, which also caught a rook
    // against three pawns — an edge of 200 cp that is nothing like drawn, and three existing
    // tests said so immediately. Every configuration this brick was built for is pawnless on
    // both sides: a lone knight, a lone bishop, two knights, rook against rook.
    //
    // The counts are already in hand: `evaluate` reads them before walking the board.
    if pawns[0] != 0 || pawns[1] != 0 {
        return false;
    }
    // **The edge in MATERIAL, and not the tapered balance the caller is holding.** The
    // threshold below is a statement about what the pieces can *do* — "a rook is the
    // threshold because K+R against K is a win" — while the caller's balance also carries
    // the two kings' square values and the phase interpolation. The two part company
    // exactly where it matters: measured on this file before the tables were retuned,
    // `8/8/8/3k4/8/8/8/K6R w` — a forced win — scored **52 cp**, because a bare king on d5
    // and ours in the corner pushed the balance under 500 and the divisor fired. Widening
    // the king tables made it fire on nearly every K+R against K, which is how it was
    // found; but it was already firing before, and on `main`.
    //
    // Rust idiom: `iter().map(..).sum()` over a fixed array rather than four additions, so
    // adding a piece type cannot be forgotten in one of the two places it appears. Pawns
    // are not in the list because this function has already returned unless the board is
    // pawnless.
    let material: i32 = [Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen]
        .iter()
        .map(|&piece| {
            value(piece)
                * (pos.count(Color::White, piece) as i32 - pos.count(Color::Black, piece) as i32)
        })
        .sum();
    let strong = if material > 0 { Color::White } else { Color::Black };
    // Rust idiom: `!color` is the `Not` operator cozy-chess implements on `Color`, so this reads as
    // "the other side" rather than as a match on two variants.
    let weak = !strong;
    // **The weak side may hold nothing more than a single minor**, and without this line the
    // function stops matching its name. Everything above is a statement about what the *strong
    // side's material* can do; the threshold below is a statement about the *edge between the two
    // sides*, and the two part company the moment the weak side holds something real. Measured on
    // this branch before the line existed, all pawnless and all divided by eight — and all
    // before mobility was merged, so the raw figures are a few centipawns higher today:
    //
    // | position       | theory           | scored |
    // |----------------|------------------|--------|
    // | K+Q vs K+R     | win              |     49 |
    // | Q+R+N vs Q+R   | clear advantage  |     35 |
    // | 2Q vs Q+R      | win              |     47 |
    //
    // The second is not even an endgame — a knight up with queens and rooks still on the board is
    // exactly the middlegame case this brick must never touch.
    //
    // Narrow on purpose: every true positive the rule was built for survives, because in each of
    // them the weak side has a bare king (a lone minor, two knights) or a single minor (rook
    // against knight, rook against bishop, both drawn). What stops firing is the whole family
    // where the weak side answers with a rook or a queen.
    //
    // One true positive goes with them, and it is a decision rather than an oversight: **R+N
    // against R is a theoretical draw** and was firing here by accident of the threshold, never
    // having been named in this brick's scope. It needs a rule of its own — the weak side holding
    // a rook is precisely what makes it drawn, so no widening of *this* condition can express it.
    if pos.count(weak, Piece::Rook) + pos.count(weak, Piece::Queen) > 0
        || pos.count(weak, Piece::Knight) + pos.count(weak, Piece::Bishop) > 1
    {
        return false;
    }
    if material.abs() < PAWNLESS_WIN_THRESHOLD {
        return true;
    }
    // Two knights and a bare king is the one drawn position above the threshold: 640 cp of
    // material that cannot force mate. Named rather than covered by a wider net, because the
    // configurations either side of it — B+N at 650, R at 500 — are wins.
    let knights = pos.count(strong, Piece::Knight);
    let others = pos.count(strong, Piece::Bishop)
        + pos.count(strong, Piece::Rook)
        + pos.count(strong, Piece::Queen);
    knights == 2 && others == 0
}

pub fn evaluate(pos: &Position) -> i32 {
    // Compiled out of production builds entirely -- see the thread-local above.
    #[cfg(test)]
    EVAL_CALLS.with(|n| n.set(n.get() + 1));
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
    let mut balance = (middlegame * phase + endgame * (MAX_PHASE - phase)) / MAX_PHASE;

    // The endgame scale, applied after the taper because it is about what the *remaining*
    // material can do, and the taper is what already knows how little is left.
    #[cfg(test)]
    let scaling_on = SCALING.with(|s| s.get());
    #[cfg(not(test))]
    let scaling_on = true;
    if scaling_on && cannot_mate(pos, pawns) {
        balance /= DRAWISH_DIVISOR;
    }

    // Flip to the side-to-move perspective: negamax wants "good for me" > 0.
    match pos.side_to_move() {
        Color::White => balance,
        Color::Black => -balance,
    }
}

// --- Piece-square tables --------------------------------------------------
//
// One table per piece type **and per phase**, indexed by `Square as usize`
// (a1 = 0 … h8 = 63), so each table is laid out **rank 1 first** (the White home
// rank) up to rank 8. Values are from White's point of view; Black reads the
// rank-flipped square via `relative_to`. Order matches `Piece`: pawn, knight,
// bishop, rook, queen, king.
//
// **Twelve tables, not six.** Until this file was tuned, five of the six pieces
// read the *same* array in both phases and only the king had two, so the 24-step
// taper documented at the top of the module moved nothing but the king. A pawn on
// the sixth rank was worth the same with both queens on the board as in a pure pawn
// endgame. The numbers below come from a fit, not from a hand adjustment: the
// evaluation is linear in them, so "are these good" is a convex problem, and they
// were fitted by sparse logistic regression against an unrestricted Stockfish on
// 265 248 quiescence-leaf positions drawn one per game from this engine's own games.
//
// Two directions of that fit are **unidentifiable** and had to be pinned, or the
// optimiser drifts along them on noise and the output stops being readable: the
// king's constant offset (see `KING_MG` below) and the overall scale, which is why
// the fit holds the sigmoid's `K` fixed at the value that was optimal for the
// hand-made tables. Ranks 1 and 8 of the pawn tables are frozen at zero — a pawn
// can never stand there, so their columns are empty in any corpus.


/// A pawn's square, and the two schedules are the whole reason this file was retuned:
/// a passed-looking central pawn is a middlegame asset, and an advanced pawn in an endgame
/// is a promotion in the making. One shared table could say only one of the two.
///
/// Fitted mean over the squares a pawn can occupy: **100 cp in the middlegame, 111 in the endgame**, of which 105 is carried by `value()`.
#[rustfmt::skip]
const PAWN_MG: [i32; 64] = [
      0,   0,   0,   0,   0,   0,   0,   0, // rank 1 (a pawn can never stand here)
    -99, -39, -97, -38, -20, -10,  60, -71,
    -80, -23, -23, -23,  -6,  -4,  44, -53,
    -77, -18, -11,   4,  -4,  -1,  14, -63,
    -50,  -6, -11, -14,   7,  25,  31, -38,
    -21,  36,  30,   4,  13,  61,  66,  16,
     36,  52,  36,  36,  25,  27,  43,  11,
      0,   0,   0,   0,   0,   0,   0,   0, // rank 8
];
#[rustfmt::skip]
const PAWN_EG: [i32; 64] = [
      0,   0,   0,   0,   0,   0,   0,   0, // rank 1 (a pawn can never stand here)
    -20,   1, -23, -19,   0,  28,  41, -23,
    -57,  -7, -24,  -5,  30,   9,  29, -48,
    -58,  -3, -22, -14,  -2,   7,  30, -39,
    -26,   8, -11, -23, -24,  18,  29,  -7,
     -6,  43,  32,  -2,   5,  56,  72,  22,
     35,  56,  36,  36,  24,  29,  46,   9,
      0,   0,   0,   0,   0,   0,   0,   0, // rank 8
];
/// A knight is a middlegame piece: it needs outposts and support points, and an endgame
/// with pawns on both wings is where it is worst. The two tables differ mostly in level.
///
/// Fitted mean over the squares a knight can occupy: **317 cp in the middlegame, 278 in the endgame**, of which 298 is carried by `value()`.
#[rustfmt::skip]
const KNIGHT_MG: [i32; 64] = [
    -36, -61, -46, -53, -56, -25, -55, -31, // rank 1
    -61, -17,   0,   1, -14,  -5,  -7, -29,
    -40, -26, -12,  21,  19,  13,   7, -25,
     -2,   7,  46,  25,  52,  43,  45,  31,
     31,  32,  76,  93,  78, 115,  84, 104,
     33,  60,  88,  98, 105, 117,  92,  39,
      0,  19,  45,  59,  55,  67,  43,  23,
    -56,  -5,   7,   9,  17,   4,  -8,  -7, // rank 8
];
#[rustfmt::skip]
const KNIGHT_EG: [i32; 64] = [
     -75, -101,  -84,  -91,  -96,  -64,  -95,  -70, // rank 1
    -101,  -54,  -41,  -48,  -56,  -47,  -47,  -68,
     -77,  -62,  -46,  -13,  -19,  -29,  -37,  -66,
     -36,  -30,   10,   -2,   18,    7,    5,   -9,
      -8,   -3,   38,   49,   50,   70,   46,   63,
      -4,   22,   50,   56,   63,   77,   52,   -2,
     -37,  -18,    8,   20,   16,   26,    5,  -15,
     -94,  -44,  -32,  -29,  -21,  -36,  -47,  -46, // rank 8
];
/// A bishop gains as the board empties, which is the classic bishop-versus-knight
/// asymmetry; here it is read off the corpus rather than asserted.
///
/// Fitted mean over the squares a bishop can occupy: **266 cp in the middlegame, 333 in the endgame**, of which 299 is carried by `value()`.
#[rustfmt::skip]
const BISHOP_MG: [i32; 64] = [
    -38, -48, -61, -61, -85, -67, -82, -49, // rank 1
    -21,  -5, -40, -59, -40, -64,  16, -72,
     11, -29, -36, -35, -51, -30, -49,  -2,
     -6, -37, -40,  -8, -14, -55, -31,  10,
    -43, -47, -25,   2,  -9, -16, -21, -26,
    -46, -17, -14, -15, -16,  49,   0,  57,
    -57, -50, -41, -53, -37, -27, -36, -65,
    -28, -47, -54, -33, -56, -48, -61, -49, // rank 8
];
#[rustfmt::skip]
const BISHOP_EG: [i32; 64] = [
     30,  20, -11,   9, -17, -17, -14,  22, // rank 1
     45,  45,  22,   5,  18,   2,  58,  -6,
     73,  36,  32,  33,  22,  33,  18,  61,
     61,  35,  36,  48,  42,  23,  36,  74,
     29,  30,  42,  63,  57,  51,  48,  44,
     29,  53,  56,  50,  51, 111,  68, 120,
     15,  21,  31,  17,  35,  43,  34,   8,
     43,  23,  17,  40,  14,  22,   7,  21, // rank 8
];
/// A rook wants open files in the middlegame and the seventh rank at any time; its
/// endgame level is the one that moves most against the hand-made table.
///
/// Fitted mean over the squares a rook can occupy: **511 cp in the middlegame, 515 in the endgame**, of which 513 is carried by `value()`.
#[rustfmt::skip]
const ROOK_MG: [i32; 64] = [
    -120,  -80,  -48,  -66,  -54,  -53,  -13, -111, // rank 1
    -115,  -53,  -67,  -82,  -82,  -52,  -35,  -67,
     -80,  -65,  -49,  -70,  -49,  -38,  -23,   -6,
     -51,  -13,  -25,  -23,   -8,   -2,    9,  -15,
      17,   20,   36,   37,   29,   34,   37,   49,
      45,   52,   40,   57,   52,   53,   48,   73,
      66,   44,   67,   62,   69,   66,   37,   77,
      33,   45,   43,   46,   47,   28,   35,   35, // rank 8
];
#[rustfmt::skip]
const ROOK_EG: [i32; 64] = [
     -82,  -68,  -54,  -48,  -46,  -46,  -25,  -87, // rank 1
    -106,  -52,  -66,  -78,  -78,  -53,  -38,  -68,
     -72,  -61,  -47,  -65,  -51,  -39,  -28,   -8,
     -41,   -9,  -20,  -21,   -6,   -2,   12,  -13,
      23,   25,   41,   41,   28,   35,   37,   49,
      55,   57,   43,   60,   54,   52,   50,   72,
      69,   50,   68,   62,   72,   65,   39,   72,
      39,   49,   48,   52,   50,   32,   38,   38, // rank 8
];
/// A queen is worth close to the same everywhere — which is itself worth knowing, since
/// it means the piece the taper cannot help is the one it was least needed for.
///
/// Fitted mean over the squares a queen can occupy: **954 cp in the middlegame, 958 in the endgame**, of which 956 is carried by `value()`.
#[rustfmt::skip]
const QUEEN_MG: [i32; 64] = [
     -62,  -83,  -78,  -38,  -64, -103,  -71,  -41, // rank 1
     -54,  -36,  -13,  -22,  -18,  -40,  -49,   -7,
     -36,  -52,  -10,  -24,  -18,  -10,  -10,   13,
     -44,  -14,  -10,   16,   24,   25,   30,   50,
      -5,   -5,   16,   32,   73,   67,   58,   89,
       4,    8,    9,   59,   57,  127,   85,  144,
     -12,  -33,   32,   25,   29,   66,    3,   60,
     -58,  -36,  -31,  -22,  -19,  -22,  -27,  -20, // rank 8
];
#[rustfmt::skip]
const QUEEN_EG: [i32; 64] = [
     -61,  -81,  -81,  -49,  -64, -104,  -71,  -40, // rank 1
     -54,  -33,  -19,  -25,  -25,  -41,  -50,   -7,
     -31,  -46,   -6,  -19,  -12,   -5,   -8,   14,
     -35,   -8,   -1,   27,   30,   32,   35,   57,
       1,    4,   23,   42,   82,   73,   65,   96,
       9,   13,   17,   66,   62,  132,   89,  146,
      -6,  -21,   41,   31,   36,   71,    8,   62,
     -59,  -37,  -30,  -20,  -18,  -20,  -27,  -24, // rank 8
];
/// The king is the piece the old code already tapered, and the only one whose table is
/// **zero-mean by construction**: both sides always have exactly one king, so adding a
/// constant to all 64 squares adds `+C` for White and `-C` for Black and cancels. The
/// loss is therefore exactly flat along that direction, and an ungauged fit drifts along
/// it on noise alone. Re-centring changes no evaluation and makes the table readable as
/// "this square against the average square".
///
/// Fitted mean over the squares a king can occupy: **-23 cp in the middlegame, -10 in the endgame** (the king carries no material value).
#[rustfmt::skip]
const KING_MG: [i32; 64] = [
     54,  68,  89, -48,  55, -27,  86,  89, // rank 1
     52,  26, -16, -18, -16,   7,  52,  66,
      4, -29, -58, -75, -65, -48,  -1,  24,
     -1, -20, -64, -76, -75, -46,  -8,  -3,
      2,   3, -25, -74, -59, -21,  14,  29,
     14,  37,   8,  -6,  -1,  25,  45,  39,
     -2,  10,   7,  -2,  -5,  30,  41,  16,
     -7,  -8, -11, -24, -30,  -9,  -6,  -8, // rank 8
];
#[rustfmt::skip]
const KING_EG: [i32; 64] = [
    -25, -12,  12, -77, -25, -61,  -8,  -9, // rank 1
    -12, -32, -34, -26, -22, -11,  -5,  -7,
    -29, -33, -33, -41, -27, -21,  -2,  -8,
    -24, -14, -20, -12,  -8,   1,   2, -22,
     -9,  23,  32,   0,  19,  38,  37,  21,
      4,  59,  58,  63,  70,  77,  69,  31,
    -14,  19,  26,  37,  34,  52,  54,   5,
    -39, -20, -13,  -6, -13, -11, -17, -41, // rank 8
];
// Order matches `Piece`: pawn, knight, bishop, rook, queen, king.
const PST_MIDDLEGAME: [[i32; 64]; 6] =
    [PAWN_MG, KNIGHT_MG, BISHOP_MG, ROOK_MG, QUEEN_MG, KING_MG];
const PST_ENDGAME: [[i32; 64]; 6] =
    [PAWN_EG, KNIGHT_EG, BISHOP_EG, ROOK_EG, QUEEN_EG, KING_EG];

#[cfg(test)]
mod tests {
    use super::*;

    // Runs `f` with the endgame scale held off, for the tests that isolate another term.
    //
    // Two of them use a lone knight against a bare king as their reference point — which is
    // exactly the position the scale divides — so leaving it on would make them measure the pair
    // rather than the term they name. Same reasoning `search_reducing` applies to the futility
    // cuts, one crate down.
    fn without_scaling<T>(f: impl FnOnce() -> T) -> T {
        SCALING.with(|s| s.set(false));
        let out = f();
        SCALING.with(|s| s.set(true));
        out
    }

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
        // The reversal this whole change exists for: with the pieces gone, the centre is
        // where the king wins pawn races and escorts a passer, and the shelter he wanted in
        // the middlegame is where games get drawn by shuffling.
        //
        // Three pawns a side, mirrored, and nothing else. The structure contributes exactly
        // zero to the balance, so the entire difference between these positions is the White
        // king's square; and pawns carry no phase weight, so this is still a pure endgame —
        // asserted below rather than assumed.
        //
        // The pawns are not decoration. They are the first of the two reasons this test no
        // longer uses the bare-king pair it was written with, `4k3/8/8/8/8/8/8/6K1` (Kg1)
        // against `4k3/8/8/8/4K3/8/8/8` (Ke4):
        //
        // 1. King against king is a **dead draw**, and now that `cannot_mate` reads material
        //    the evaluation says so: pawnless on both sides with no material edge, so the
        //    balance is divided by DRAWISH_DIVISOR. Refusing to have an opinion there is the
        //    correct behaviour, so a test of king activity must not be asking the one question
        //    the evaluation deliberately declines to answer — the difference it wanted to see
        //    was being truncated away by the divisor. (The sibling middlegame test above was
        //    moved off bare kings for the mirror-image reason: the position was wrong, not the
        //    claim.) Here the pawns keep the scale off, so the assertion runs the production
        //    path with nothing held back.
        // 2. e4 is not a square the fitted table pays for. Measured 2026-09-10 on KING_EG,
        //    e4 and g1 are both worth -8, so the old pair differed by 0 cp even before the
        //    divisor. The fit puts the endgame king's value on how far up the board he has
        //    come rather than on how near the middle files he stands: -8 on e4, +19 on e5,
        //    +70 on e6. So the centre squares used below are the ones he has actually walked
        //    to, which is what an active king is at the board — and the property under test is
        //    unchanged, an active centralised king outscoring one still tucked up on g1.
        let corner = Position::from_fen("4k3/5ppp/8/8/8/8/5PPP/6K1 w - - 0 1").unwrap(); // Kg1
        let centre_e5 = Position::from_fen("4k3/5ppp/8/4K3/8/8/5PPP/8 w - - 0 1").unwrap(); // Ke5
        let centre_d5 = Position::from_fen("4k3/5ppp/8/3K4/8/8/5PPP/8 w - - 0 1").unwrap(); // Kd5
        // All three hold the same material, so one reading covers them.
        assert_eq!(phase(&corner), 0, "precondition: a pure endgame, or the taper is not on");
        // Two centre squares and not one, so that the claim is about the centre rather than
        // about a single lucky entry in a fitted table.
        for (name, active) in [("e5", &centre_e5), ("d5", &centre_d5)] {
            assert!(
                evaluate(active) > evaluate(&corner),
                "K{name} {} should beat Kg1 {}",
                evaluate(active),
                evaluate(&corner),
            );
        }
    }

    #[test]
    fn the_same_king_square_is_judged_differently_by_phase() {
        // Not two tables side by side, but one evaluation that changes its mind: **one**
        // pair of king squares — castled on g1, or out on e5 — put to `evaluate` twice,
        // once with the queens and rooks still on and once with nothing left but a pawn
        // each, and the *sign* of its verdict has to reverse between the two readings. If
        // this fails while the two tests above pass, the tables are right and the
        // interpolation is not wired to the phase.
        //
        // **The out square is e5, and it was e4 until the tables were fitted.** The fitted
        // endgame table prices a king by how far up the board it has walked, not by how
        // near the middle it stands: down the e-file it reads -25, -22, -27, -8, +19, +70,
        // +34, -13 from e1 to e8, while the castled square g1 is worth -8 — the very same
        // -8 as e4. e4 against g1 is a dead tie in the endgame, and a tie shows no
        // reversal, so the pair moves one rank up, to the first square this table actually
        // pays for. What is lost with e4 is real, and it belongs to the test above, which
        // owns the claim that the centre beats the corner; what *this* test owns is that
        // the verdict turns over with the phase at all, and it still does.
        // (Figures read from `KING_EG` as fitted on 2026-09-10. The two assertions below
        // are what keep them honest: they go red the moment the reversal stops.)
        //
        // **A pawn each, and not two bare kings.** Pawnless king against king is exactly
        // what `cannot_mate` sends through `DRAWISH_DIVISOR`, so bare kings would put this
        // test's whole margin through an integer division by eight and leave it measuring
        // the drawish scale instead of the tables. The two pawns are mirror images on e2
        // and e7, so they cancel to the centipawn and the king square is all that is left
        // standing between the two positions of a pair.
        const MIDDLEGAME_KING_OUT_E5: &str = "r2qk2r/pppp1ppp/8/4K3/8/8/PPPP1PPP/R2Q1R2 w kq - 0 1";
        const ENDGAME_KING_SAFE: &str = "4k3/4p3/8/8/8/8/4P3/6K1 w - - 0 1";
        const ENDGAME_KING_OUT_E5: &str = "4k3/4p3/8/4K3/8/8/4P3/8 w - - 0 1";
        let at = |fen: &str| evaluate(&Position::from_fen(fen).expect("a legal position"));

        let middlegame = at(MIDDLEGAME_KING_SAFE) - at(MIDDLEGAME_KING_OUT_E5);
        let endgame = at(ENDGAME_KING_SAFE) - at(ENDGAME_KING_OUT_E5);
        assert!(middlegame > 0, "with the pieces on, g1 must beat e5, and leads by {middlegame}");
        assert!(endgame < 0, "with them gone, e5 must beat g1, and leads by {}", -endgame);
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
        //
        // The claim is now made over REGIONS rather than over the single best and single
        // worst square, because the fitted table places them differently from the
        // hand-made one it replaces. The hand-made table was drawn as concentric rings
        // around d4/e4/d5/e5, so "the maximum is a centre square" and "the minimum is a
        // corner" happened to hold exactly; the fit reads *activity* off the corpus, and
        // an active endgame king is an advanced one, so its peak sits one ring off the
        // middle and a couple of ranks up, and its floor on the back rank without being
        // in a corner (f6 and d1 in the tables fitted in September 2026). Neither
        // placement contradicts "high in the middle, low at the edges" — pinning the
        // property to one square did.
        //
        // Both tables are read through `relative_to(color)` in `evaluate`, so rank 1 is
        // the side's OWN home rank whatever its colour, and rank 8 the promotion rank.
        let at = |table: &[i32; 64], name: &str| {
            let sq: Square = name.parse().expect("a square name");
            table[sq as usize]
        };
        let mean = |squares: &[i32]| squares.iter().sum::<i32>() / squares.len() as i32;
        // Distance from the middle of the board in king moves: 0 on d4/e4/d5/e5, 3 on the
        // rim (files a/h and ranks 1/8).
        let ring = |square: usize| {
            let from_middle = |x: usize| (x as i32 - 3).abs().min((x as i32 - 4).abs());
            from_middle(square % 8).max(from_middle(square / 8))
        };

        let centre = ["d4", "e4", "d5", "e5"].map(|n| at(&KING_EG, n));
        let corners = ["a1", "h1", "a8", "h8"].map(|n| at(&KING_EG, n));
        let best = *centre.iter().max().unwrap();
        let worst = *corners.iter().max().unwrap();
        assert!(best > worst, "centre {best} must beat corners {worst}");
        assert!(
            mean(&centre) > mean(&corners),
            "centre {} must beat corners {} on average too",
            mean(&centre),
            mean(&corners),
        );
        // Centralisation over the whole board, which is what the two equalities on the
        // extremes used to stand in for: the 36 squares off the rim are worth more than
        // the 28 on it.
        let inside: Vec<i32> = (0..64).filter(|&sq| ring(sq) < 3).map(|sq| KING_EG[sq]).collect();
        let rim: Vec<i32> = (0..64).filter(|&sq| ring(sq) == 3).map(|sq| KING_EG[sq]).collect();
        assert!(
            mean(&inside) > mean(&rim),
            "off the rim {} must beat the rim {}",
            mean(&inside),
            mean(&rim),
        );

        // Which way up. Every comparison above survives a table flipped top to bottom —
        // the centre, the corners and the rim are all symmetric about the middle — so the
        // orientation has to be claimed on its own: the half of the board the endgame king
        // walks TOWARDS is worth more than the half it starts in, and the best square of
        // all lies in it, off the rim.
        assert!(
            mean(&KING_EG[32..]) > mean(&KING_EG[..32]),
            "the far half {} must beat the home half {}",
            mean(&KING_EG[32..]),
            mean(&KING_EG[..32]),
        );
        let peak = (0..64).max_by_key(|&sq| KING_EG[sq]).expect("64 squares");
        let floor = (0..64).min_by_key(|&sq| KING_EG[sq]).expect("64 squares");
        assert!(peak >= 32, "the peak {} must be in the far half", Square::index(peak));
        assert!(ring(peak) < 3, "the peak {} must be off the rim", Square::index(peak));
        assert!(ring(floor) == 3, "the floor {} must be on the rim", Square::index(floor));
        // A shift by one rank slides the whole profile and survives all of the above, so
        // pin the one step that carries a chess meaning: leaving the back rank is already
        // worth something. Shifted either way, rank 2's row lands on rank 1 and reverses
        // this comparison.
        let rank_mean = |rank: usize| mean(&KING_EG[rank * 8..rank * 8 + 8]);
        assert!(
            rank_mean(0) < rank_mean(1),
            "the home rank {} must be worse than the rank in front of it {}",
            rank_mean(0),
            rank_mean(1),
        );

        // And the middlegame table says the exact opposite, which is the whole point
        // of having two — stated over the same two regions, so that the contrast is
        // between the tables and not between two ways of measuring them.
        let mg_centre = ["d4", "e4", "d5", "e5"].map(|n| at(&KING_MG, n));
        let mg_corners = ["a1", "h1", "a8", "h8"].map(|n| at(&KING_MG, n));
        assert!(
            mean(&mg_corners) > mean(&mg_centre),
            "middlegame: corners {} must beat centre {}",
            mean(&mg_corners),
            mean(&mg_centre),
        );
        let castled = at(&KING_MG, "g1");
        let exposed = at(&KING_MG, "e4");
        assert!(castled > exposed, "middlegame: {castled} must beat {exposed}");
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
        let running = without_scaling(|| evaluate(&free)) - without_scaling(|| evaluate(&bare));
        let stopped = without_scaling(|| evaluate(&blocked)) - without_scaling(|| evaluate(&knight_only));
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
        let no_bonus_at_all = without_scaling(|| evaluate(&blocked_not_passed)) - without_scaling(|| evaluate(&pawn_only));
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
            without_scaling(|| evaluate(&own)) - without_scaling(|| evaluate(&own_baseline)) < without_scaling(|| evaluate(&free)) - without_scaling(|| evaluate(&bare)),
            "a pawn cannot advance through its own piece either",
        );
    }

    #[test]
    fn the_passed_bonus_grows_with_rank() {
        // Asserted as monotonicity rather than as values, so that retuning the numbers cannot
        // fail this test for a reason unrelated to its name.
        //
        // Indices 0 and 7 are excluded: a pawn cannot stand on the first rank, and one that
        // reaches the eighth is no longer a pawn — both are zero by construction.
        //
        // **Why the run starts at index 2 and not at index 1.** The two schedules are no
        // longer a curve chosen on their own: they are fitted jointly with PAWN_MG / PAWN_EG,
        // which are indexed by rank as well, so for a given square only the *sum* of the two
        // is identified and the split between them is a presentation choice. Index 1 — a pawn
        // still on its starting rank — is where that split is worst determined: such a pawn is
        // rarely already passed, and it is the one entry the term is not about, since it is
        // not running anywhere yet. As fitted in September 2026 the schedule steps *down*
        // there, in both phases, and that is the only downward step either of them has.
        // Everything from the third rank on — the whole range in which "closer to promotion"
        // means anything — still grows at every single step, which is the property this test
        // exists for. The exempted entry is not left unchecked: the three assertions after the
        // loop keep it from drifting into nonsense.
        for table in [PASSED_MIDDLEGAME, PASSED_ENDGAME] {
            for rank in 2..6 {
                assert!(
                    table[rank + 1] > table[rank],
                    "rank {rank} -> {} went {} -> {}",
                    rank + 1,
                    table[rank],
                    table[rank + 1],
                );
            }
            // A passer that has actually crossed to the middle of the board beats one that has
            // not moved, whatever the fit did to the step between the first two entries.
            assert!(
                table[4] > table[1],
                "a passer on the fifth rank ({}) must beat one still on its starting rank ({})",
                table[4],
                table[1],
            );
            // And that downward step stays smaller than what one rank of real progress buys at
            // the top of the schedule — noise at the least-observed entry, not a second slope.
            assert!(
                table[1] - table[2] < table[6] - table[5],
                "the dip off the starting rank ({} -> {}) is no longer smaller than the last \
                 stride ({} -> {})",
                table[1],
                table[2],
                table[5],
                table[6],
            );
            // The term is a *bonus*: one rank from promoting it has to be worth something in
            // both phases. Nothing else here asserts the sign, and a schedule turned wholesale
            // into a penalty would satisfy every check above.
            assert!(
                table[6] > 0,
                "a passer one rank from promoting is scored {}",
                table[6],
            );
            assert_eq!(table[0], 0, "a pawn cannot stand on the first rank");
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

        // **The other half of the definition, and it was the half nothing asserted.** Only *one's
        // own* side is excluded, so a square held by an enemy piece counts — it is a capture, which
        // is a move. Narrowing the mask to `!occupied()` — "empty squares only" — used to leave this
        // whole test green and reddened one transposition-table assertion by accident.
        //
        // Asserted as a **pair on the same square**, since that is what isolates the asymmetry: the
        // three zeros above and every comparison in this file move together when both sides are
        // excluded, and a single count proves nothing about which side was meant.
        //
        // Both families of piece, because they fail differently: a jumper simply includes or
        // excludes the square, while for a slider the enemy piece is the *last* square of the ray.
        for (enemy, ours, empty, sq, piece, name) in [
            (
                "4k3/8/8/8/8/2p5/8/1N2K3 w - - 0 1",
                "4k3/8/8/8/8/2P5/8/1N2K3 w - - 0 1",
                "4k3/8/8/8/8/8/8/1N2K3 w - - 0 1",
                Square::B1, Piece::Knight, "a knight on b1, c3 held",
            ),
            (
                "4k3/8/8/8/p7/8/8/R3K3 w - - 0 1",
                "4k3/8/8/8/P7/8/8/R3K3 w - - 0 1",
                "4k3/8/8/8/8/8/8/R3K3 w - - 0 1",
                Square::A1, Piece::Rook, "a rook on a1, a4 held",
            ),
        ] {
            let m = |fen: &str| {
                Position::from_fen(fen).unwrap().mobility_from(sq, piece, Color::White)
            };
            assert_eq!(
                m(enemy), m(empty).min(m(enemy)),
                "{name}: precondition, an enemy piece cannot add mobility",
            );
            assert!(
                m(enemy) > m(ours),
                "{name}: an enemy piece on a reachable square is a capture and must count, \
                 while one of ours must not — read {} against {}",
                m(enemy), m(ours),
            );
        }
        // And the slider case pinned exactly: the enemy pawn on a4 is counted *and* stops the ray,
        // so the a-file contributes a2, a3, a4 and nothing beyond. Measured before written — 6 with
        // the pawn there, 10 with the file clear.
        let blocked = Position::from_fen("4k3/8/8/8/p7/8/8/R3K3 w - - 0 1").unwrap();
        let clear = Position::from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1").unwrap();
        assert_eq!(
            blocked.mobility_from(Square::A1, Piece::Rook, Color::White), 6,
            "the enemy pawn counts and ends the ray: a2, a3, a4, then b1, c1, d1",
        );
        assert_eq!(
            clear.mobility_from(Square::A1, Piece::Rook, Color::White), 10,
            "control: with the file clear the same rook reads the whole of it",
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

        // **Why this test no longer compares the two prices per square.** It used to require
        // `ENDGAME >= MIDDLEGAME` for each minor, reading "a bishop's freedom is worth more once
        // the board empties" straight off these two arrays. That reading was only ever valid
        // because a bishop looked up the *same* square table in both phases, so the price per
        // square was the only place a bishop's phase information could live. It is not any more:
        // BISHOP_MG and BISHOP_EG are independent, and the fit may charge for a bishop up front,
        // in its table, or by the square, in these arrays. Trading between the two moves the
        // comparison without moving the chess, so the comparison stopped being a chess claim and
        // became a fact about one particular split. Freezing a split is not this test's job.
        //
        // The comparison rested on a second assumption too, and that one is measurably false: it
        // treats a square of scope as the same quantity in both phases. It is not, and the two
        // assertions below are the measurement rather than a recollection of one — a smaller
        // price on two or three times as many squares is not a smaller term.
        let middlegame =
            Position::from_fen("r2q1rk1/pp2bppp/2n1bn2/3p4/3P4/2N1BN2/PP2BPPP/R2Q1RK1 w - - 0 1")
                .unwrap();
        let endgame = Position::from_fen("4k3/5p2/8/3B4/8/8/5P2/4K3 w - - 0 1").unwrap();
        assert_eq!(
            middlegame.mobility_from(Square::E3, Piece::Bishop, Color::White), 5,
            "a developed middlegame bishop sees a handful of squares",
        );
        assert_eq!(
            endgame.mobility_from(Square::D5, Piece::Bishop, Color::White), 12,
            "the same piece sees a multiple of that once the board empties",
        );

        // What the two arrays can still honestly promise starts here.
        //
        // Freedom is never a liability, in either phase. This is the half of the old comparison
        // that was a chess invariant rather than an artefact of the split, and it is a live risk
        // now that the weights are fitted rather than chosen: a regression run over collinear
        // features can hand back a negative coefficient, and nothing else in this file looks.
        for piece in Piece::ALL {
            assert!(
                MOBILITY_MIDDLEGAME[piece as usize] >= 0 && MOBILITY_ENDGAME[piece as usize] >= 0,
                "{piece:?} mobility must never be priced below zero: {} middlegame, {} endgame",
                MOBILITY_MIDDLEGAME[piece as usize], MOBILITY_ENDGAME[piece as usize],
            );
        }
        // Both halves of the taper price mobility. A phase whose weights were all zero would
        // switch the term off for half the game with `assert_ne!` above still green, and that
        // hole is the one the old per-piece comparison happened to plug on its way past.
        for (phase, weights) in
            [("middlegame", MOBILITY_MIDDLEGAME), ("endgame", MOBILITY_ENDGAME)]
        {
            assert!(
                [Piece::Knight, Piece::Bishop].iter().any(|p| weights[*p as usize] > 0),
                "no minor is priced in the {phase}: the term is switched off for half the game",
            );
        }
        // And each minor is priced in at least one phase. The knights and the bishops are the
        // two pieces this term exists for, and one of them falling to zero in both phases would
        // be the term quietly losing half its subject.
        for piece in [Piece::Knight, Piece::Bishop] {
            assert!(
                MOBILITY_MIDDLEGAME[piece as usize] > 0 || MOBILITY_ENDGAME[piece as usize] > 0,
                "{piece:?} must be priced in at least one phase",
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



    // ---------------------------------------------------------- endgame scaling (#79)

    // **Every figure below was re-measured after mobility (#77) was merged into this branch**, and
    // this note exists because the merge could have broken the arm in silence. Mobility adds to the
    // balance the arm then compares against `PAWNLESS_WIN_THRESHOLD`, so a position could have
    // crossed it without a single test failing.
    //
    // **Each row carries its FEN**, and that is not decoration. A first version of this table named
    // only the material, and two rows could not be reproduced by a reader — `K+R vs K+N` read 203
    // instead of 176 — because with mobility counting, *where the defending minor stands* moves the
    // number by exactly that much. A table of measured values without the positions they were
    // measured on is not a measurement, it is a recollection.
    //
    // | position | FEN | before | after | verdict |
    // |---|---|---|---|---|
    // | K+N vs K | `4k3/8/8/8/8/8/8/3NK3 w` | 290 | **302** | still divided |
    // | K+B vs K | `4k3/8/8/8/8/8/8/3BK3 w` | 320 | **347** | still divided |
    // | K+N+N vs K | `4k3/8/8/8/8/8/8/1N1NK3 w` | 570 | **591** | still divided, two-knights clause |
    // | K+B+N vs K | `4k3/8/8/8/8/8/8/1B1NK3 w` | 610 | **649** | still a win, untouched |
    // | **K+R vs K** | `4k3/8/8/8/8/8/8/3RK3 w` | 505 | **505** | still a win, untouched |
    // | K+R vs K+N | `4n3/4k3/8/8/8/8/8/3RK3 w` | 188 | **176** | still divided |
    // | K+R vs K+B | `4b3/4k3/8/8/8/8/8/3RK3 w` | 158 | **131** | still divided |
    //
    // **The row that mattered is `K+R vs K`, and it did not move at all.** It sits five centipawns
    // above the threshold, so any addition to it would have turned a won ending into a scaled one —
    // and mobility weights rooks at **zero**, which is what keeps it still. Not luck: the weights
    // were set that way because pricing the heavy pieces doubled the tree (#77), and the reason
    // happens to protect this arm as well.
    //
    // The two rows that went *down* are the weak side gaining mobility, which shrinks our edge:
    // that is the arm being fed a smaller number and still reaching the same verdict. And they are
    // the two rows whose value depends on the defending minor's square — on `e8` beside its king
    // here, which is why the FEN had to be written down.


    #[test]
    fn a_lone_minor_against_a_bare_king_is_not_a_win() {
        // The measured failure this brick exists for: over 10 400 anchored games, 44% of our
        // draws were positions we had been winning, 23% of those ended with no pawn on our side,
        // and the two most frequent final positions were exactly these. `evaluate` was returning
        // about +330 for them, so the search traded into them and then shuffled to a repetition.
        for (fen, name) in [
            ("4k3/8/8/8/8/8/8/3NK3 w - - 0 1", "K+N against K"),
            ("4k3/8/8/8/8/8/8/3BK3 w - - 0 1", "K+B against K"),
            ("4k3/8/8/8/8/8/8/1N1NK3 w - - 0 1", "K+N+N against K"),
        ] {
            let p = Position::from_fen(fen).unwrap();
            assert!(
                evaluate(&p).abs() < 100,
                "{name} is drawn and must not read as a win: {} cp",
                evaluate(&p),
            );
            // From the other side too: the factor keys on the sign of the balance, and a version
            // that only handled White would pass the three assertions above.
            let mirrored = Position::from_fen(&fen.replace('N', "\u{1}").replace('n', "N")
                .replace('\u{1}', "n").replace('B', "\u{2}").replace('b', "B")
                .replace('\u{2}', "b").replace('K', "\u{3}").replace('k', "K")
                .replace('\u{3}', "k").replace(" w ", " b ")).unwrap();
            assert!(
                evaluate(&mirrored).abs() < 100,
                "{name} mirrored is drawn too: {} cp",
                evaluate(&mirrored),
            );
        }
    }

    #[test]
    fn the_endgames_that_do_win_keep_their_score() {
        // The control, and it is what stops the rule from being a wider net than the measurement
        // asked for. B+N against a bare king is a genuine win — long and technical, but a win —
        // and a factor that flattened it would throw away real games. Same for a rook.
        //
        // **The last three are where the rule used to break, and they broke on `main`.** The
        // verdict was read off the *tapered* balance, which carries the two kings' square
        // values as well as the material; a bare king in the centre and ours in a corner
        // pushed a rook's edge under the 500 cp threshold and the divisor fired on a forced
        // win. `8/8/8/3k4/8/8/8/K6R w` scored **52 cp** before `cannot_mate` was made to read
        // material instead. Found while retuning the tables — widening the king tables made
        // it fire on nearly every K+R against K rather than on the awkward ones — but the
        // defect predates the retuning, which is why these cases are pinned here and not in
        // the retuning's own tests.
        for (fen, name) in [
            ("4k3/8/8/8/8/8/8/1B1NK3 w - - 0 1", "K+B+N against K"),
            ("4k3/8/8/8/8/8/8/3RK3 w - - 0 1", "K+R against K"),
            ("4k3/8/8/8/8/8/8/3QK3 w - - 0 1", "K+Q against K"),
            ("8/8/8/3k4/8/8/8/K6R w - - 0 1", "K+R against a centralised bare king"),
            ("7k/8/8/8/8/8/8/R3K3 w - - 0 1", "K+R against K, kings far apart"),
            ("8/8/8/3k4/8/8/8/KQ6 w - - 0 1", "K+Q against a centralised bare king"),
        ] {
            let p = Position::from_fen(fen).unwrap();
            assert!(
                evaluate(&p) > 300,
                "{name} is a win and must keep its score: {} cp",
                evaluate(&p),
            );
        }
    }

    #[test]
    fn a_pawn_anywhere_switches_the_factor_off() {
        // Pawns mean a promotion to play for, so none of this applies — and the restriction is
        // measured rather than cautious. A first version asked only that the *strong* side be
        // pawnless, which also caught a rook against three pawns: an edge measured at 200 cp on
        // the hand-made tables and at 159 on the fitted ones (2026-09), nothing like drawn under
        // either. Two existing passed-pawn tests said so within the minute.
        let bare = Position::from_fen("4k3/8/8/8/8/8/8/3NK3 w - - 0 1").unwrap();
        let with_our_pawn = Position::from_fen("4k3/8/8/8/8/8/P7/3NK3 w - - 0 1").unwrap();
        let with_their_pawn = Position::from_fen("4k3/p7/8/8/8/8/8/3NK3 w - - 0 1").unwrap();
        assert!(
            evaluate(&bare).abs() < 100,
            "precondition: the pawnless version must be scaled, or the two below prove nothing",
        );
        // Asserted as inertness rather than against a number: what has to hold is that the factor
        // does not touch these positions at all. A threshold would have to be recalibrated every
        // time a piece-square table moves — a first draft of this test picked 200 cp for a knight
        // against a pawn that read 189 on the tables of the day (177 before mobility was merged),
        // which is why these two assertions are inertness and not a threshold.
        for (p, name) in [(&with_our_pawn, "a pawn of ours"), (&with_their_pawn, "an enemy pawn")] {
            assert_eq!(
                evaluate(p), without_scaling(|| evaluate(p)),
                "{name} must switch the factor off entirely",
            );
        }
        // And the enemy-pawn case is the one the first version of the rule got wrong: it asked
        // only that the *strong* side be pawnless, which also caught a rook against three pawns.
        // This assertion is what keeps the two above from protecting nothing: inertness is only
        // worth asserting where the factor firing would change the verdict, and here it would —
        // `DRAWISH_DIVISOR` would take this score down to a rounding error.
        //
        // **Stated against `value(Piece::Pawn)`, where it used to name `> 150`.** The prediction
        // in the comment above came true on the very next retuning: fitting the tables took this
        // position from 189 cp to 139, because a knight is now worth 278 in the endgame where one
        // shared table used to answer for both phases, a pawn 111, and a passed enemy pawn on its
        // second rank 20 instead of 8. The literal went red for a reason that has nothing to do
        // with the factor this test is about. "A knight against a lone pawn is worth more than a
        // pawn" says the same thing about chess and tracks the tables instead of dating from them.
        let edge = evaluate(&with_their_pawn);
        assert!(
            edge > value(Piece::Pawn),
            "a knight against a lone pawn is an edge, not a draw: {edge} cp against the {} \
             this engine pays for the pawn it is a piece up on",
            value(Piece::Pawn),
        );
    }

    #[test]
    fn the_factor_never_fires_while_the_board_is_full() {
        // It keys on pawns, not on the phase, so a middlegame with every piece present is
        // untouched by construction — but the property is worth an assertion rather than an
        // argument, since a later version keying on `phase` instead would pass every test above.
        let p = Position::from_fen(
            "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 3",
        )
        .unwrap();
        assert_eq!(
            evaluate(&p), without_scaling(|| evaluate(&p)),
            "the factor must be inert on a full board",
        );
        // And a crushing middlegame edge stays crushing: pawns are on the board, so nothing
        // scales, whatever the material difference.
        let crushing = Position::from_fen("4k3/pppppppp/8/8/8/8/PPPPPPPP/3QK3 w - - 0 1").unwrap();
        assert!(
            evaluate(&crushing) > 500,
            "a queen up with pawns on the board is a win: {} cp",
            evaluate(&crushing),
        );
    }


    #[test]
    fn a_pawnless_edge_the_weak_side_can_answer_is_not_a_draw() {
        // AC#4, and the case that had no test: the factor keys on pawns, so a **pawnless** board
        // still crowded with pieces reaches it, and the threshold it then applies is about the
        // *edge between the sides* rather than about what the strong side can mate with.
        //
        // Every position below was measured before the weak-side condition existed and was being
        // divided by eight. The middle one is not an endgame at all — a knight up with queens and
        // rooks on the board — which is exactly what AC#4 says must stay a win.
        for (fen, name, was) in [
            ("4k2r/8/8/8/8/8/8/3QK3 w - - 0 1", "K+Q against K+R", 49),
            ("r2qk3/8/8/8/8/8/8/RN1QK3 w - - 0 1", "Q+R+N against Q+R", 35),
            ("r2qk3/8/8/8/8/8/8/Q2QK3 w - - 0 1", "2Q against Q+R", 47),
        ] {
            let p = Position::from_fen(fen).unwrap();
            assert_eq!(
                evaluate(&p),
                without_scaling(|| evaluate(&p)),
                "{name}: the weak side answers with a rook or a queen, so the factor must not \
                 fire — it used to read {was} cp",
            );
            assert!(
                evaluate(&p) > 200,
                "{name}: and it must read as the advantage it is: {} cp",
                evaluate(&p),
            );
        }
    }

    #[test]
    fn a_single_minor_on_the_weak_side_still_scales() {
        // The other half of the condition, and it is what keeps it narrow. Rook against a lone
        // minor is drawn, the weak side holds one piece, and both must keep scaling — a fix
        // written as "the weak side must be bare" would have passed every assertion above while
        // throwing these two away.
        for (fen, name) in [
            ("4n3/4k3/8/8/8/8/8/3RK3 w - - 0 1", "K+R against K+N"),
            ("4b3/4k3/8/8/8/8/8/3RK3 w - - 0 1", "K+R against K+B"),
        ] {
            let p = Position::from_fen(fen).unwrap();
            assert!(
                evaluate(&p).abs() < 60,
                "{name} is drawn and must still be scaled: {} cp against {} raw",
                evaluate(&p),
                without_scaling(|| evaluate(&p)),
            );
        }
    }

}
