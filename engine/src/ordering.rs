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
//!
//! Second heuristic: **killer moves**. MVV-LVA has nothing to say about quiet
//! moves — it scores them all `0`, so they are tried in whatever order the
//! generator emitted them. Yet a quiet move is very often what refutes a branch,
//! and in most positions quiet moves are the majority. A killer is a quiet move
//! that already caused a cutoff at this same distance from the root: sibling
//! positions at one ply usually share the same threat, so the refutation that
//! worked next door tends to work here too.

use crate::position::{BitBoard, Color, Move, Piece, Position, Square};
use crate::search::MAX_DEPTH;

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


/// The material a capture wins or loses once **both sides have taken turns recapturing** on the
/// destination square — a static exchange evaluation.
///
/// # Why MVV-LVA is not enough
///
/// MVV-LVA ranks `pawn takes queen` above `queen takes pawn`, which is right, but it cannot see
/// what happens *next*. `Nxd5` when d5 is defended by a pawn is scored as winning a knight's
/// worth of material when it loses one. The search finds out — two plies later, having built the
/// whole subtree first. This function answers the same question before the move is searched, for
/// the cost of a few bitboard lookups.
///
/// # The algorithm, and the one thing that makes it non-trivial
///
/// Pieces capture on the square in turn, cheapest first, each side free to stop when continuing
/// would lose material. The gains are accumulated on a stack and then folded back with a
/// **negamax minimum**: a side only continues if doing so is better than standing pat.
///
/// What makes it more than a two-step calculation is **uncovering**. Removing the piece that just
/// captured can reveal a sliding attacker behind it — a rook behind a rook, a queen behind a
/// bishop. So the attacker set is recomputed at every step against the *current* hypothetical
/// occupation, which is exactly what [`Position::attackers`] takes an `occupied` argument for.
/// A version that computed the attackers once would misjudge every battery, and batteries are
/// what exchange sequences are made of.
///
/// # What it deliberately does not handle
///
/// **En passant and promotions return 0** — "no information" — rather than a wrong number. The
/// captured pawn of an en passant is not on the destination square, and a promotion changes the
/// value of the capturing piece mid-sequence; both need special cases that would earn their
/// complexity only if this function were also used for pruning. It is used for *ordering*, where
/// a missing verdict costs a few nodes and a wrong verdict costs a bad move order.
pub fn see(pos: &Position, mv: Move) -> i32 {
    let Some(victim) = pos.piece_on(mv.to) else {
        return 0; // quiet move, or en passant — nothing to weigh on this square
    };
    let Some(attacker) = pos.piece_on(mv.from) else {
        return 0;
    };
    if mv.promotion.is_some() {
        return 0; // the capturing piece changes value mid-sequence — out of scope
    }

    // `gain[d]` is the material balance for the side to move at depth `d`, assuming the exchange
    // stops there. Thirty-two is more than any legal sequence: every step removes a piece.
    let mut gain = [0i32; 32];
    gain[0] = value(victim);
    let mut on_square = value(attacker);
    let mut occupied = pos.occupied() ^ mv.from.bitboard();
    let mut side = !pos.side_to_move();
    let mut d = 0usize;

    loop {
        d += 1;
        gain[d] = on_square - gain[d - 1];
        let Some((square, piece)) = cheapest_attacker(pos, mv.to, side, occupied) else {
            break;
        };
        on_square = value(piece);
        occupied ^= square.bitboard();
        side = !side;
        if d + 1 >= gain.len() {
            break;
        }
    }
    // Fold back: at each step the side to move takes the exchange only if it beats stopping.
    while d > 1 {
        d -= 1;
        gain[d - 1] = -std::cmp::max(-gain[d - 1], gain[d]);
    }
    gain[0]
}

/// The least valuable piece of `color` attacking `square` under `occupied`, if any.
///
/// Cheapest first is not a heuristic here but the rule of the exchange: recapturing with the
/// queen when a pawn would do loses material the sequence would otherwise keep.
fn cheapest_attacker(
    pos: &Position,
    square: Square,
    color: Color,
    occupied: BitBoard,
) -> Option<(Square, Piece)> {
    let attackers = pos.attackers(square, color, occupied);
    if attackers.is_empty() {
        return None;
    }
    for piece in [Piece::Pawn, Piece::Knight, Piece::Bishop, Piece::Rook, Piece::Queen, Piece::King]
    {
        let of_type = attackers & pos.pieces_of(color, piece);
        if let Some(sq) = of_type.next_square() {
            return Some((sq, piece));
        }
    }
    None
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
    //
    // This function ranks captures against each other; it is *not* the test for
    // whether a move is one. En passant scores 0 here (the captured pawn is not on
    // `mv.to`) and a king capture scores negative (`value(King)` dwarfs any victim).
    // [`is_quiet`] is what answers that question — see [`score`].
    match pos.piece_on(mv.to) {
        Some(victim) => {
            let attacker = pos.piece_on(mv.from).map_or(0, value);
            100 * value(victim) - attacker
        }
        None => 0,
    }
}

/// Whether `mv` leaves the material on the board untouched.
///
/// Only quiet moves are worth remembering as killers. A capture is already ranked
/// by `mvv_lva`, and a promotion by the material it creates; spending one of the two
/// slots on either would evict the quiet move the heuristic exists to find.
///
/// The three ways a move touches material are tested separately because en passant
/// is the one that hides: it captures a pawn standing on *another* square than the
/// destination, so `piece_on(mv.to)` reads it as an empty square. A pawn changing
/// file onto an empty square can only be an en-passant capture.
pub fn is_quiet(pos: &Position, mv: Move) -> bool {
    if pos.piece_on(mv.to).is_some() || mv.promotion.is_some() {
        return false;
    }
    let en_passant =
        pos.piece_on(mv.from) == Some(Piece::Pawn) && mv.from.file() != mv.to.file();
    !en_passant
}

/// How many killer moves are kept per ply.
///
/// Two, which is the usual choice: one slot alone is overwritten by whichever
/// sibling cut off last, and a third adds candidates that match rarely enough not to
/// pay for the comparison.
const KILLER_SLOTS: usize = 2;

/// Killers of one node, ready for ordering: the freshest first.
///
/// Handed to [`order_moves`] rather than the whole table plus a ply, so that
/// out-of-range plies are resolved in exactly one place ([`Killers::at`]) instead of
/// at every call site. Quiescence, which has no killers of its own, passes
/// [`KillerSlots::none`].
///
/// Idiom: `Copy` because it is two `Option<Move>` — 8 bytes; passing it by value is
/// cheaper than the reference that would borrow the searcher while its move list is
/// being sorted.
#[derive(Clone, Copy, Default)]
pub struct KillerSlots([Option<Move>; KILLER_SLOTS]);

impl KillerSlots {
    /// No killers apply here.
    pub fn none() -> KillerSlots {
        KillerSlots::default()
    }

    /// Which slot holds `mv`, if any. Slot `0` is the most recent.
    fn slot_of(&self, mv: Move) -> Option<usize> {
        self.0.iter().position(|&killer| killer == Some(mv))
    }

    /// Whether `mv` is one of this node's killers.
    ///
    /// The search needs the question without the answer's detail: late move reductions
    /// exempt killers from being reduced, and for that "is it one" is enough — which of
    /// the two slots holds it only matters to the ordering.
    pub fn contains(&self, mv: Move) -> bool {
        self.slot_of(mv).is_some()
    }
}

/// The killer moves found so far, one set of slots per ply.
///
/// Lives for a whole search, so the deepening iterations share it: iteration N+1
/// starts already knowing what refuted what in iteration N.
pub struct Killers {
    /// Indexed by ply. Sized once at construction, so a ply beyond the table is a
    /// silent no-op rather than a panic — quiescence recurses past the nominal
    /// maximum depth, and a heuristic must never be the reason a search crashes.
    table: Vec<[Option<Move>; KILLER_SLOTS]>,
}

impl Killers {
    pub fn new() -> Killers {
        // One slot set per reachable ply: the root is ply 0 and no branch goes
        // deeper than `MAX_DEPTH` plies below it.
        Killers { table: vec![[None; KILLER_SLOTS]; MAX_DEPTH as usize + 1] }
    }

    /// The killers that apply at `ply` — none if `ply` is past the table.
    pub fn at(&self, ply: usize) -> KillerSlots {
        // Idiom: `get` returns `Option` instead of panicking on an out-of-range
        // index, so the bound is enforced here rather than trusted at every caller.
        self.table.get(ply).map(|&slots| KillerSlots(slots)).unwrap_or_default()
    }

    /// Remember `mv` as a killer at `ply`, pushing the previous one down a slot.
    ///
    /// A move that is not quiet is refused here rather than at the call site: the
    /// table's whole value is that its two slots hold moves nothing else ranks, and
    /// an invariant the caller has to remember is one a later caller will forget.
    pub fn record(&mut self, pos: &Position, ply: usize, mv: Move) {
        if !is_quiet(pos, mv) {
            return;
        }
        // Idiom: `let ... else` leaves the function when the pattern does not match.
        let Some(slots) = self.table.get_mut(ply) else { return };
        // Already the freshest killer: re-recording it would push a copy of itself
        // into the second slot and cost the only other candidate we keep.
        if slots[0] == Some(mv) {
            return;
        }
        slots[1] = slots[0];
        slots[0] = Some(mv);
    }
}

impl Default for Killers {
    fn default() -> Killers {
        Killers::new()
    }
}

// The ordering bands. They must not overlap: every capture is tried before every
// killer, and every killer before every remaining quiet move.
//
// The width of the capture band is set by `mvv_lva`, which spans **−10 000 to
// +89 900**. The top is a queen taken by a pawn. The bottom is a *king* taking a
// pawn: `value(King)` is 20 000, so `100 * 100 − 20 000` goes negative — which is
// why the base has to sit far enough above `KILLER_BASE` to absorb it. Worst
// capture 990 000 against best killer 900 002. A test pins the property, rather
// than trusting an eyeball on the constants.
const CAPTURE_BASE: i32 = 1_000_000;

/// Where a capture goes once the exchange evaluation says it loses material.
///
/// **Below the quiet moves**, which is the whole point. `Nxd5` into a defended square is not a
/// slightly worse capture to try fifth — it is a move that loses a knight, and it belongs after
/// every ordinary developing move. MVV-LVA cannot know that: it sees the victim and stops.
///
/// Far enough below zero that the worst losing capture (a queen for nothing, about -900) still
/// lands above nothing else's score, and losing captures stay ordered among themselves —
/// least catastrophic first.
const LOSING_CAPTURE_BASE: i32 = -1_000_000;
const KILLER_BASE: i32 = 900_000;

/// The ordering score of a move: captures, then killers, then the rest.
fn score(pos: &Position, mv: Move, killers: KillerSlots) -> i32 {
    // `is_quiet` decides what counts as a capture — not `mvv_lva(..) > 0`, which
    // answers a different question and gets two real captures wrong: a king capture
    // scores negative (see above) and en passant scores 0, so both would land among
    // the quiet moves and be tried *after* a killer. One predicate decides.
    if !is_quiet(pos, mv) {
        // The exchange evaluation is consulted before MVV-LVA gets to speak. A capture that
        // loses material is demoted below the quiet moves; everything else keeps the MVV-LVA
        // ordering, which is a good ranking among captures that are worth making.
        //
        // `see` answers 0 for en passant and for capturing promotions — "no information" rather
        // than a wrong number — so those keep their MVV-LVA place instead of being demoted on a
        // verdict that was never given.
        let exchange = see(pos, mv);
        if exchange < 0 {
            return LOSING_CAPTURE_BASE + exchange;
        }
        return CAPTURE_BASE + mvv_lva(pos, mv);
    }
    match killers.slot_of(mv) {
        // The most recent killer (slot 0) is tried before the older one.
        Some(slot) => KILLER_BASE + (KILLER_SLOTS - slot) as i32,
        None => 0,
    }
}

/// Sorts `moves` in place: best captures first, then the killers of this node,
/// then the remaining quiet moves.
pub fn order_moves(pos: &Position, moves: &mut [Move], killers: KillerSlots) {
    // Idiom: `sort_by_cached_key` computes each key **once** and sorts the keys, where
    // `sort_by_key` is free to recompute a key on every comparison. That distinction did not
    // matter while the key was a table lookup; it matters now that scoring a capture runs a
    // static exchange evaluation, which walks a sequence of bitboard lookups. Sorting 40 moves
    // could otherwise pay for the same exchange a dozen times.
    //
    // `Reverse` sorts by the key in *descending* order, so the highest score comes first.
    moves.sort_by_cached_key(|&mv| std::cmp::Reverse(score(pos, mv, killers)));
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
    fn most_valuable_victim_is_preferred() {
        // The core of MVV-LVA: a bigger victim outranks a smaller one. White can
        // take a queen with a pawn (dxe5) or a pawn with the queen (Qxb5); the
        // pawn-takes-queen must score far higher.
        let p = Position::from_fen("7k/8/8/1p2q3/Q2P4/8/8/6K1 w - - 0 1").unwrap();
        let moves = p.legal_moves();
        let pawn_takes_queen = moves
            .iter()
            .copied()
            .find(|&mv| p.piece_on(mv.from) == Some(Piece::Pawn) && p.piece_on(mv.to) == Some(Piece::Queen))
            .unwrap();
        let queen_takes_pawn = moves
            .iter()
            .copied()
            .find(|&mv| p.piece_on(mv.from) == Some(Piece::Queen) && p.piece_on(mv.to) == Some(Piece::Pawn))
            .unwrap();
        assert!(mvv_lva(&p, pawn_takes_queen) > mvv_lva(&p, queen_takes_pawn));
    }

    #[test]
    fn winning_captures_lead_and_stay_sorted_by_mvv_lva() {
        // White has captures of three different victims (queen e5, knight c5, pawn a6) plus
        // quiet moves. Of the three, **Rxa6 loses 400 centipawns** — the pawn is defended by
        // the knight — so it is no longer among the captures that lead.
        //
        // This test asserted "every capture comes before every quiet move" until the exchange
        // evaluation landed. That sentence was the contract of MVV-LVA alone; it is not the
        // contract any more, and the test says what replaced it instead of being weakened.
        let p = Position::from_fen("7k/8/p7/2n1q3/3P4/8/8/R5K1 w - - 0 1").unwrap();
        let mut moves = p.legal_moves();
        order_moves(&p, &mut moves, KillerSlots::none());

        let is_capture = |mv: Move| p.piece_on(mv.to).is_some();
        let winning = |mv: Move| is_capture(mv) && see(&p, mv) >= 0;
        let lead = moves.iter().copied().filter(|&mv| winning(mv)).count();
        assert_eq!(lead, 2, "expected exactly the two captures worth making");

        assert!(moves[..lead].iter().all(|&mv| winning(mv)), "winning captures must lead");
        assert!(
            moves[lead..].iter().all(|&mv| !winning(mv)),
            "and nothing worth making may follow the quiet moves",
        );

        // Among themselves, they keep the MVV-LVA order: the exchange evaluation decides
        // *whether* a capture is worth trying, MVV-LVA decides in *which order*.
        let scores: Vec<i32> = moves[..lead].iter().map(|&mv| mvv_lva(&p, mv)).collect();
        assert!(scores.windows(2).all(|w| w[0] >= w[1]), "captures not sorted: {scores:?}");
    }

    // Three captures of different value (dxe5 a queen, dxc5 a knight, Rxa6 a pawn)
    // and fourteen quiet moves: enough of both to tell the bands apart.
    pub(super) const CAPTURES_AND_QUIETS: &str = "7k/8/p7/2n1q3/3P4/8/8/R5K1 w - - 0 1";
    // Two quiet moves of that position, the last ones the generator emits — so a
    // killer promoting them has to cross the whole quiet band to reach the front.
    const QUIET_A: &str = "g1g2";
    const QUIET_B: &str = "a1a5";

    fn uci_move(pos: &Position, uci: &str) -> Move {
        pos.move_from_uci(uci).expect("a legal move")
    }

    #[test]
    fn a_killer_is_tried_before_the_other_quiet_moves() {
        let p = Position::from_fen(CAPTURES_AND_QUIETS).unwrap();
        let killer = uci_move(&p, QUIET_A);

        let mut moves = p.legal_moves();
        order_moves(&p, &mut moves, KillerSlots::none());
        let without = moves.iter().position(|&mv| mv == killer).unwrap();

        let mut killers = Killers::new();
        killers.record(&p, 3, killer);
        order_moves(&p, &mut moves, killers.at(3));
        let with = moves.iter().position(|&mv| mv == killer).unwrap();

        assert!(with < without, "the killer must move up: {without} -> {with}");
        // Ahead of *every* other quiet move, not merely a few places up.
        let quiet_ahead = moves[..with].iter().filter(|&&mv| is_quiet(&p, mv)).count();
        assert_eq!(quiet_ahead, 0, "no quiet move may be tried before the killer");
    }

    #[test]
    fn winning_captures_come_before_killers_and_losing_ones_after() {
        // The rule as the exchange evaluation leaves it, and this position states it perfectly.
        // Its three captures are dxc5 (SEE +220), dxe5 (+900) and **Rxa6 (SEE -400)** — the
        // rook takes a pawn defended by the knight on c5.
        //
        // Before the SEE this test asserted that *every* capture precedes the killer, and named
        // Rxa6 as "the least valuable capture there is, the one a killer would overtake if the
        // bands were too close". That was the intent then; the measurement says Rxa6 loses 400
        // centipawns and belongs after every ordinary quiet move, not fifth. The test is updated
        // rather than relaxed: it now pins **both** sides of the new rule.
        let p = Position::from_fen(CAPTURES_AND_QUIETS).unwrap();
        let killer = uci_move(&p, QUIET_A);
        let mut killers = Killers::new();
        killers.record(&p, 0, killer);

        let mut moves = p.legal_moves();
        order_moves(&p, &mut moves, killers.at(0));

        let killer_at = moves.iter().position(|&mv| mv == killer).unwrap();
        let winning_after = moves[killer_at..]
            .iter()
            .filter(|&&mv| !is_quiet(&p, mv) && see(&p, mv) >= 0)
            .count();
        assert_eq!(winning_after, 0, "no capture worth making may be tried after the killer");
        assert_eq!(killer_at, 2, "the two winning captures, then the killer");

        let losing = uci_move(&p, "a1a6");
        let losing_at = moves.iter().position(|&mv| mv == losing).unwrap();
        assert!(
            losing_at > killer_at,
            "a capture that loses material must be tried after the killer, not before",
        );
        let quiet_after_losing = moves[losing_at..]
            .iter()
            .filter(|&&mv| is_quiet(&p, mv) && Some(mv) != Some(killer))
            .count();
        assert_eq!(
            quiet_after_losing, 0,
            "and after *every* quiet move: {losing_at} of {}",
            moves.len(),
        );
    }

    #[test]
    fn the_score_bands_never_overlap() {
        // The property the constants exist for: the cheapest capture outranks the
        // best killer, which outranks any other quiet move.
        let p = Position::from_fen(CAPTURES_AND_QUIETS).unwrap();
        let mut killers = Killers::new();
        killers.record(&p, 0, uci_move(&p, QUIET_A));
        killers.record(&p, 0, uci_move(&p, QUIET_B));
        let slots = killers.at(0);

        let moves = p.legal_moves();
        let band = |mv: Move| score(&p, mv, slots);
        let captures: Vec<i32> =
            moves.iter().copied().filter(|&mv| !is_quiet(&p, mv)).map(band).collect();
        let killer_scores: Vec<i32> =
            moves.iter().copied().filter(|&mv| slots.slot_of(mv).is_some()).map(band).collect();
        let quiet: Vec<i32> = moves
            .iter()
            .copied()
            .filter(|&mv| is_quiet(&p, mv) && slots.slot_of(mv).is_none())
            .map(band)
            .collect();

        assert_eq!(killer_scores.len(), 2, "both killers should be legal here");
        let worst_capture = *captures.iter().min().expect("captures exist");
        let best_killer = *killer_scores.iter().max().unwrap();
        let worst_killer = *killer_scores.iter().min().unwrap();
        let best_quiet = *quiet.iter().max().expect("quiet moves exist");
        // Four bands now, not three. The captures split: those the exchange evaluation
        // endorses stay on top, those it condemns fall below everything.
        let winning: Vec<i32> = moves
            .iter()
            .copied()
            .filter(|&mv| !is_quiet(&p, mv) && see(&p, mv) >= 0)
            .map(band)
            .collect();
        let losing: Vec<i32> = moves
            .iter()
            .copied()
            .filter(|&mv| !is_quiet(&p, mv) && see(&p, mv) < 0)
            .map(band)
            .collect();
        assert!(!losing.is_empty(), "precondition: this position has a losing capture (Rxa6)");
        let worst_winning = *winning.iter().min().expect("winning captures exist");
        let best_losing = *losing.iter().max().unwrap();
        assert!(worst_winning > best_killer, "{worst_winning} vs {best_killer}");
        assert!(worst_killer > best_quiet, "{worst_killer} vs {best_quiet}");
        let worst_quiet = *quiet.iter().min().expect("quiet moves exist");
        assert!(
            worst_quiet > best_losing,
            "a losing capture must rank below the worst quiet move: {worst_quiet} vs {best_losing}",
        );
        let _ = worst_capture;

        // The extreme the bases were sized for, and the one this position does not
        // contain: a king taking a pawn is the lowest `mvv_lva` reachable (−10 000),
        // so it is the capture that comes closest to falling into the killer band.
        let kp = Position::from_fen("4k3/8/8/4p3/4K3/8/8/8 w - - 0 1").unwrap();
        let mut kk = Killers::new();
        kk.record(&kp, 0, uci_move(&kp, "e4d3"));
        let worst_possible = score(&kp, uci_move(&kp, "e4e5"), kk.at(0));
        let best_possible_killer = score(&kp, uci_move(&kp, "e4d3"), kk.at(0));
        assert_eq!(worst_possible, 990_000);
        assert_eq!(best_possible_killer, 900_002);
        assert!(worst_possible > best_possible_killer);
    }

    #[test]
    fn the_freshest_killer_is_tried_first() {
        let p = Position::from_fen(CAPTURES_AND_QUIETS).unwrap();
        let older = uci_move(&p, QUIET_A);
        let fresher = uci_move(&p, QUIET_B);
        let mut killers = Killers::new();
        killers.record(&p, 1, older);
        killers.record(&p, 1, fresher);

        let mut moves = p.legal_moves();
        order_moves(&p, &mut moves, killers.at(1));
        let at = |mv: Move| moves.iter().position(|&m| m == mv).unwrap();
        assert!(at(fresher) < at(older), "the most recent killer goes first");
    }

    #[test]
    fn a_second_killer_does_not_evict_the_first() {
        // Two slots, so the previous killer survives one replacement. A single slot
        // would keep only whichever sibling cut off last.
        let p = Position::from_fen(CAPTURES_AND_QUIETS).unwrap();
        let first = uci_move(&p, QUIET_A);
        let second = uci_move(&p, QUIET_B);
        let mut killers = Killers::new();
        killers.record(&p, 2, first);
        killers.record(&p, 2, second);

        let slots = killers.at(2);
        assert_eq!(slots.slot_of(second), Some(0));
        assert_eq!(slots.slot_of(first), Some(1));
    }

    #[test]
    fn recording_the_same_killer_twice_leaves_the_other_slot_free() {
        // Otherwise a move that cuts off repeatedly ends up occupying both slots,
        // and the second candidate is lost for nothing.
        let p = Position::from_fen(CAPTURES_AND_QUIETS).unwrap();
        let repeated = uci_move(&p, QUIET_A);
        let mut killers = Killers::new();
        killers.record(&p, 2, repeated);
        killers.record(&p, 2, repeated);

        assert_eq!(killers.table[2], [Some(repeated), None]);
    }

    #[test]
    fn a_king_capture_is_still_a_capture() {
        // The case `mvv_lva(pos, mv) > 0` got wrong: `value(King)` is 20 000, so
        // Kxe5 scores 100 * 100 − 20 000 = −10 000. Read as "not a capture", it was
        // ordered among the quiet moves, behind the killer.
        let p = Position::from_fen("4k3/8/8/4p3/4K3/8/8/8 w - - 0 1").unwrap();
        let capture = uci_move(&p, "e4e5");
        assert!(mvv_lva(&p, capture) < 0, "the trap: a king capture scores negative");

        let mut killers = Killers::new();
        killers.record(&p, 0, uci_move(&p, "e4d3"));
        let mut moves = p.legal_moves();
        order_moves(&p, &mut moves, killers.at(0));
        assert_eq!(moves[0], capture, "the capture must be tried first, before the killer");
    }

    #[test]
    fn an_en_passant_capture_is_still_a_capture() {
        // Same defect, other cause: the captured pawn stands on d5, not on the
        // destination d6, so `mvv_lva` reads an empty square and scores 0.
        let p = Position::from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2").unwrap();
        let capture = uci_move(&p, "e5d6");
        assert_eq!(mvv_lva(&p, capture), 0, "the trap: en passant scores as a quiet move");

        let mut killers = Killers::new();
        killers.record(&p, 0, uci_move(&p, "e5e6"));
        let mut moves = p.legal_moves();
        order_moves(&p, &mut moves, killers.at(0));
        assert_eq!(moves[0], capture, "the capture must be tried first, before the killer");
    }

    #[test]
    fn a_move_that_touches_material_is_never_recorded() {
        // The table refuses what `mvv_lva` already ranks. Each of the three forms is
        // offered, including the en passant that reads as quiet on the board alone.
        let p = Position::from_fen(CAPTURES_AND_QUIETS).unwrap();
        let mut killers = Killers::new();
        killers.record(&p, 0, uci_move(&p, "d4e5")); // capture
        assert_eq!(killers.at(0).slot_of(uci_move(&p, "d4e5")), None, "a capture");

        let promo = Position::from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let mut killers = Killers::new();
        killers.record(&promo, 0, uci_move(&promo, "a7a8q"));
        assert_eq!(killers.at(0).slot_of(uci_move(&promo, "a7a8q")), None, "a promotion");

        let ep = Position::from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2").unwrap();
        let mut killers = Killers::new();
        killers.record(&ep, 0, uci_move(&ep, "e5d6"));
        assert_eq!(killers.at(0).slot_of(uci_move(&ep, "e5d6")), None, "en passant");
        // …while the quiet push of the same pawn is accepted.
        killers.record(&ep, 0, uci_move(&ep, "e5e6"));
        assert_eq!(killers.at(0).slot_of(uci_move(&ep, "e5e6")), Some(0), "the push");
    }

    #[test]
    fn killers_do_not_leak_into_another_ply() {
        // A killer is a refutation *at this distance from the root*; elsewhere it is
        // just a quiet move, and promoting it there would order on noise.
        let p = Position::from_fen(CAPTURES_AND_QUIETS).unwrap();
        let killer = uci_move(&p, QUIET_A);
        let mut killers = Killers::new();
        killers.record(&p, 4, killer);

        assert_eq!(killers.at(4).slot_of(killer), Some(0));
        assert_eq!(killers.at(3).slot_of(killer), None);
        assert_eq!(killers.at(5).slot_of(killer), None);
    }

    #[test]
    fn a_ply_past_the_table_is_ignored_rather_than_a_panic() {
        // Quiescence recurses past the nominal maximum depth. It records no killers,
        // but the bound must hold by construction, not by the caller's good manners.
        let p = Position::from_fen(CAPTURES_AND_QUIETS).unwrap();
        let mv = uci_move(&p, QUIET_A);
        let mut killers = Killers::new();
        let beyond = killers.table.len() + 10;
        killers.record(&p, beyond, mv);
        assert_eq!(killers.at(beyond).slot_of(mv), None);
    }

    #[test]
    fn is_quiet_rejects_every_way_of_touching_material() {
        // A plain capture and an ordinary quiet move.
        let p = Position::from_fen("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1").unwrap();
        assert!(!is_quiet(&p, uci_move(&p, "d1d5")), "a capture is not quiet");
        assert!(is_quiet(&p, uci_move(&p, "d1d3")), "a rook step is quiet");

        // A promotion: no piece on the destination, yet it creates a queen.
        let promo = Position::from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        assert!(!is_quiet(&promo, uci_move(&promo, "a7a8q")), "a promotion is not quiet");

        // En passant: the captured pawn stands on d5, not on the destination d6.
        let ep = Position::from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2").unwrap();
        assert!(!is_quiet(&ep, uci_move(&ep, "e5d6")), "en passant is a capture");
        assert!(is_quiet(&ep, uci_move(&ep, "e5e6")), "the pawn push is quiet");
    }

    // ------------------------------------------------------- static exchange evaluation

    fn see_of(fen: &str, uci: &str) -> i32 {
        let p = Position::from_fen(fen).unwrap();
        let mv = p.move_from_uci(uci).unwrap();
        see(&p, mv)
    }

    // Every expected value below is computed **by hand from the rules**, not read off the
    // implementation. That direction matters: asserting what the code returns would pin a bug as
    // firmly as a feature, and an exchange evaluation is precisely the kind of code whose bugs
    // are invisible — a wrong sign costs nodes and a bad move order, never a crash.

    #[test]
    fn an_undefended_capture_is_worth_the_piece() {
        // Rd1xd5 takes a knight nobody defends: the black king on e8 is far away.
        // By hand: +320, the sequence ends immediately.
        assert_eq!(see_of("4k3/8/8/3n4/8/8/8/K2R4 w - - 0 1", "d1d5"), 320);
    }

    #[test]
    fn a_defended_capture_costs_the_difference() {
        // Same, but a pawn on c6 defends d5. Rxd5 wins a knight (320) and loses a rook (500)
        // to cxd5. By hand: 320 - 500 = **-180**. MVV-LVA scores this move *positively* — it
        // sees the knight and not the recapture, which is the whole reason this function exists.
        assert_eq!(see_of("4k3/8/2p5/3n4/8/8/8/K2R4 w - - 0 1", "d1d5"), -180);
    }

    #[test]
    fn an_even_trade_is_worth_nothing() {
        // dxc5 takes a pawn, bxc5 takes it back. By hand: 100 - 100 = 0. A zero here must not be
        // confused with "no information": the caller distinguishes them, and this is the value
        // that separates a losing capture from an acceptable one.
        assert_eq!(see_of("4k3/8/1p6/2p5/3P4/8/8/4K3 w - - 0 1", "d4c5"), 0);
    }

    #[test]
    fn a_battery_behind_the_capturing_piece_is_counted() {
        // **The case that makes this more than arithmetic.** White rooks on d1 and d2, black rook
        // on d5 defended by a knight on c6.
        //
        // Rd2xd5 (+500), Nxd5 (-500), and now the rook on d1 — which was *behind* the one that
        // captured and attacked nothing at the start — recaptures the knight (+320).
        // By hand: 500 - 500 + 320 = **+320**.
        //
        // An implementation that computed the attacker set once, before the sequence, would stop
        // after two steps and answer 0. That is why `Position::attackers` takes a hypothetical
        // occupation rather than reading the board.
        assert_eq!(see_of("4k3/8/1n6/3r4/8/8/3R4/3RK3 w - - 0 1", "d2d5"), 320);
    }

    #[test]
    fn the_cheapest_defender_recaptures_and_not_the_strongest() {
        // **The case that pins *which* defender recaptures**, where the battery above pins
        // *whether* an uncovered slider joins. Both are hand-computable and they fail apart:
        // reversing the piece order in `cheapest_attacker` leaves the battery test green.
        //
        // Black knight on d5, defended twice — a pawn on c6 and the queen on d8. White rook
        // battery on d1+d2.
        //
        // Rd2xd5 (+320 knight), cxd5 (-500 rook), Rxd5 (+100 pawn), Qxd5 (-500 rook),
        // and White is out of attackers. By hand, folding back: **-180**.
        //
        // Recapturing with the queen instead — the same sequence run cheapest-last — folds back
        // to **+220**: "losing, prune it" becomes "winning, keep it" on one position.
        assert_eq!(see_of("3q3k/8/2p5/3n4/8/8/3R4/3R3K w - - 0 1", "d2d5"), -180);
    }

    #[test]
    fn what_the_evaluation_declines_to_judge_returns_zero() {
        // En passant (the captured pawn is not on the destination square) and promotions (the
        // capturing piece changes value mid-sequence) return 0 — "no information" — rather than a
        // number that would be wrong. Both are documented as out of scope, and a test says so, so
        // that a future reader does not mistake the zero for "this capture is even".
        let ep = Position::from_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1").unwrap();
        let mv = ep.move_from_uci("e5d6").unwrap();
        assert_eq!(see(&ep, mv), 0, "en passant is out of scope");

        let promo = Position::from_fen("3r3k/4P3/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let mv = promo.move_from_uci("e7d8q").unwrap();
        assert_eq!(see(&promo, mv), 0, "a capturing promotion is out of scope");
    }

    #[test]
    fn a_quiet_move_has_no_exchange_to_evaluate() {
        assert_eq!(see_of("4k3/8/8/8/8/8/8/K2R4 w - - 0 1", "d1d5"), 0);
    }


}
