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

use crate::position::{Move, Piece, Position};
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
// killer, and every killer before every remaining quiet move. `mvv_lva` reaches
// 89 900 (a queen taken by a pawn), so the gap between the bases is what keeps the
// bands apart — hence a test pinning the property rather than an eyeball on the
// constants.
const CAPTURE_BASE: i32 = 1_000_000;
const KILLER_BASE: i32 = 900_000;

/// The ordering score of a move: captures, then killers, then the rest.
fn score(pos: &Position, mv: Move, killers: KillerSlots) -> i32 {
    let capture = mvv_lva(pos, mv);
    if capture > 0 {
        return CAPTURE_BASE + capture;
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
    // Idiom: `sort_by_key` with `Reverse` sorts by the key in *descending* order,
    // so the highest score comes first.
    moves.sort_by_key(|&mv| std::cmp::Reverse(score(pos, mv, killers)));
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
    fn order_moves_sorts_all_captures_first_by_mvv_lva() {
        // White has several captures of different victims (queen e5, knight c5,
        // pawn a6) plus quiet moves.
        let p = Position::from_fen("7k/8/p7/2n1q3/3P4/8/8/R5K1 w - - 0 1").unwrap();
        let mut moves = p.legal_moves();
        order_moves(&p, &mut moves, KillerSlots::none());

        let is_capture = |mv: Move| p.piece_on(mv.to).is_some();
        let captures = moves.iter().copied().filter(|&mv| is_capture(mv)).count();
        assert!(captures >= 3, "expected several captures, got {captures}");

        // Every capture comes before every quiet move.
        assert!(moves[..captures].iter().all(|&mv| is_capture(mv)), "captures must lead");
        assert!(moves[captures..].iter().all(|&mv| !is_capture(mv)), "quiet moves must follow");

        // Captures are sorted by non-increasing MVV-LVA score.
        let scores: Vec<i32> = moves[..captures].iter().map(|&mv| mvv_lva(&p, mv)).collect();
        assert!(scores.windows(2).all(|w| w[0] >= w[1]), "captures not sorted: {scores:?}");
    }

    // Three captures of different value (dxe5 a queen, dxc5 a knight, Rxa6 a pawn)
    // and fourteen quiet moves: enough of both to tell the bands apart.
    const CAPTURES_AND_QUIETS: &str = "7k/8/p7/2n1q3/3P4/8/8/R5K1 w - - 0 1";
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
    fn captures_still_come_before_killers() {
        // Including Rxa6, a pawn taken by a rook — the least valuable capture there
        // is, and the one a killer would overtake if the bands were too close.
        let p = Position::from_fen(CAPTURES_AND_QUIETS).unwrap();
        let killer = uci_move(&p, QUIET_A);
        let mut killers = Killers::new();
        killers.record(&p, 0, killer);

        let mut moves = p.legal_moves();
        order_moves(&p, &mut moves, killers.at(0));

        let killer_at = moves.iter().position(|&mv| mv == killer).unwrap();
        let captures_after =
            moves[killer_at..].iter().filter(|&&mv| !is_quiet(&p, mv)).count();
        assert_eq!(captures_after, 0, "every capture must be tried before the killer");
        assert_eq!(killer_at, 3, "the three captures, then the killer");
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
        assert!(worst_capture > best_killer, "{worst_capture} vs {best_killer}");
        assert!(worst_killer > best_quiet, "{worst_killer} vs {best_quiet}");
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
}

