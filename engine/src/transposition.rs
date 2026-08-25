//! The transposition table: what a position was worth, keyed by the position
//! itself rather than by the path that reached it.
//!
//! Two kinds of repeated work make this pay. Iterative deepening re-searches
//! every position at each depth, and *transpositions* — different move orders
//! arriving at the same position — are explored as if unrelated. Both disappear
//! when a node can answer "I already know this one".
//!
//! The subtlety is that a cached score is not always a score. Alpha-beta prunes,
//! so a search often finishes knowing only that the value is *at least* or *at
//! most* something. Storing that distinction is what makes the table correct
//! rather than merely fast — see [`Bound`].

use crate::position::{Move, Piece, Square};
use crate::search::MATE_THRESHOLD;

/// What a stored score tells us about the true value of a position.
///
/// Alpha-beta does not always compute an exact value: once it knows a node is
/// good enough to be cut off, it stops looking and the value it has is only a
/// bound. Reusing a bound as though it were exact is the classic way to make a
/// transposition table produce moves that cannot be explained.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Bound {
    /// The search completed inside its window: the score is the true value.
    Exact,
    /// A beta cutoff happened — the true value is **at least** the stored score.
    Lower,
    /// No move improved alpha — the true value is **at most** the stored score.
    Upper,
}

/// One cached position, packed into a single 64-bit word.
///
/// Packed rather than kept as a struct because the table has to be readable and writable from
/// several threads at once, and a struct of 24 bytes cannot be written in one instruction. Two
/// atomic words can — see [`Slot`], which is where the lock-free part actually lives.
///
/// The layout puts the move in the low bits, so packing it is self-contained, and the
/// `OCCUPIED` flag above the fields, so an all-zero word — what a freshly cleared table
/// holds — decodes as *empty* rather than as a legal entry with score zero.
///
/// | bits | field |
/// |---|---|
/// | 0..16 | the best move, or zero for none |
/// | 16..36 | the score, two's complement |
/// | 36..44 | the depth searched |
/// | 44..46 | the bound |
/// | 46 | occupied |
mod layout {
    pub const SCORE_SHIFT: u32 = 16;
    pub const SCORE_BITS: u32 = 20;
    pub const DEPTH_SHIFT: u32 = 36;
    pub const BOUND_SHIFT: u32 = 44;
    pub const OCCUPIED: u64 = 1 << 46;
}

/// Rust idiom: `Ordering::Relaxed` on every access below, aliased here because `Ordering` is
/// also the name of a chess concept in this crate. Relaxed orders nothing — it guarantees only
/// that a single word is not read or written half-way. That is exactly the guarantee wanted:
/// the table is a cache, no other data depends on having seen an entry, and a reader that
/// misses a store simply searches the node. Anything stronger would buy a synchronisation
/// nothing needs and pay for it on the hottest path in the engine.
use std::sync::atomic::{AtomicU64, Ordering as Atomicity};

/// One slot: two atomic words, no lock and no `unsafe`.
///
/// Rust idiom: a struct of two atomics rather than an atomic of a struct. There is no
/// `AtomicU128` on stable, so a 16-byte entry cannot be written in one go — and that is the
/// whole difficulty this type exists to solve.
///
/// **`check` is not the key.** It is `key ^ data`. Storing the key plainly would leave a torn
/// slot undetectable: two threads writing at once can leave the key of one beside the data of
/// the other, and the full-key comparison — the thing that makes an entry trustworthy — would
/// accept it.
///
/// With the XOR, a reader recovers `check ^ data` and compares *that* to the key it wants. A
/// torn read gives `check` from write A and `data` from write B, so the comparison sees
/// `key_A ^ data_A ^ data_B`, which equals the wanted key only when `data_A == data_B` — and in
/// that case the slot is coherent anyway. The scheme is therefore **exact rather than
/// probabilistic**, which is why it needs neither `unsafe` nor a lock. It is Hyatt's lockless
/// hashing, and it is the reason a shared table costs two relaxed loads instead of a mutex.
///
/// The residual risk is a false *accept*: `key_A ^ delta` coinciding with the wanted key. That
/// needs a 64-bit coincidence between an unrelated Zobrist key and a difference of two plausible
/// data words — negligible, though not zero, and worth stating rather than implying.
struct Slot {
    check: AtomicU64,
    data: AtomicU64,
}

impl Slot {
    fn empty() -> Slot {
        Slot { check: AtomicU64::new(0), data: AtomicU64::new(0) }
    }
}

/// A move as sixteen bits: one flag, two squares, a promotion.
///
/// Zero means "no move". The flag sits at the bottom rather than being implied by the squares
/// both reading zero: `a1a1` is not a legal move, but relying on that would smuggle a rule of
/// chess into a bit layout.
fn pack_move(best: Option<Move>) -> u64 {
    match best {
        None => 0,
        Some(mv) => {
            let promotion = mv.promotion.map_or(0, |p| p as u64 + 1);
            1 | ((mv.from as u64) << 1) | ((mv.to as u64) << 7) | (promotion << 13)
        }
    }
}

fn unpack_move(word: u64) -> Option<Move> {
    if word & 1 == 0 {
        return None;
    }
    let promotion = (word >> 13) & 0x7;
    Some(Move {
        from: Square::index(((word >> 1) & 0x3F) as usize),
        to: Square::index(((word >> 7) & 0x3F) as usize),
        promotion: (promotion > 0).then(|| Piece::index(promotion as usize - 1)),
    })
}

fn pack(depth: u32, score: i32, bound: Bound, best: Option<Move>) -> u64 {
    use layout::*;
    // Twenty bits hold ±524 287. The largest value the search ever stores is a mate score
    // adjusted by the ply, about 30 064 — four bits of headroom over what is reachable, and the
    // assertion states it rather than leaving it to be trusted.
    debug_assert!(score.unsigned_abs() < (1 << (SCORE_BITS - 1)), "score {score} does not fit");
    debug_assert!(depth < 256, "depth {depth} does not fit");
    let score_bits = (score as i64 as u64) & ((1 << SCORE_BITS) - 1);
    let bound_bits = match bound {
        Bound::Exact => 0u64,
        Bound::Lower => 1,
        Bound::Upper => 2,
    };
    OCCUPIED
        | (bound_bits << BOUND_SHIFT)
        | ((depth as u64) << DEPTH_SHIFT)
        | (score_bits << SCORE_SHIFT)
        | pack_move(best)
}

/// Rust idiom: sign extension by hand. Shifting a value up to the top of an `i64` and back down
/// copies the sign bit, which is what makes a 20-bit two's-complement field read back as a
/// negative `i32`. A plain mask would turn every negative score into a large positive one.
fn unpack(word: u64) -> Option<(u32, i32, Bound, Option<Move>)> {
    use layout::*;
    if word & OCCUPIED == 0 {
        return None;
    }
    let shift = 64 - SCORE_BITS;
    let score = ((((word >> SCORE_SHIFT) << shift) as i64) >> shift) as i32;
    let bound = match (word >> BOUND_SHIFT) & 0x3 {
        0 => Bound::Exact,
        1 => Bound::Lower,
        _ => Bound::Upper,
    };
    Some((((word >> DEPTH_SHIFT) & 0xFF) as u32, score, bound, unpack_move(word)))
}

/// What a probe found: enough to cut off, or just a move worth trying first.
pub struct Hit {
    /// The score, if the entry is deep enough *and* its bound settles the
    /// question for the current window. `None` means "search this node, but try
    /// `best` first".
    pub cutoff: Option<i32>,
    /// The best move recorded for this position, whatever the depth. Useful for
    /// ordering even when the entry is too shallow to cut off.
    pub best: Option<Move>,
    /// Whether an entry for *this* position was found at all, cutoff or not.
    ///
    /// Reported rather than inferred: an entry may legitimately hold `best: None`, so the
    /// caller cannot tell a miss from a match by looking at the other two fields. The rates
    /// that used to live on the table are computed from this, per searcher.
    pub matched: bool,
}

/// Number of entries. A power of two, so the index is a mask rather than a
/// modulo. 2^20 entries at 32 bytes each is about 33 MB — large enough to matter
/// at the depths this engine reaches, small enough to ignore on any machine.
const ENTRIES: usize = 1 << 20;

/// A fixed-size, single-threaded transposition table.
pub struct Table {
    /// Idiom: `Box<[T]>` is a heap slice of fixed length — a `Vec` without the
    /// capacity to grow, which is exactly what a table with a fixed mask wants.
    slots: Box<[Slot]>,
    // No counters here, and that absence is measured rather than tidy. A single
    // `fetch_add` per probe puts eight cores in a fight over one cache line: with the
    // counters in this struct, `go depth 11` took 4.01 s on eight threads and 1.01 s
    // without them. Counting is a diagnostic, so it lives per-searcher — see
    // `Searcher::table_probes` — where it costs a plain increment on a line nobody else
    // touches.
}

impl Default for Table {
    fn default() -> Table {
        Table::new()
    }
}

impl Table {
    pub fn new() -> Table {
        // `AtomicU64` is not `Clone`, so `vec![...; N]` is unavailable: each slot has to be
        // constructed. One allocation either way.
        Table {
            slots: (0..ENTRIES).map(|_| Slot::empty()).collect::<Vec<_>>().into_boxed_slice(),
        }
    }

    /// Forget everything, keeping the allocation.
    ///
    /// Called between **games**, not between moves. Within a game, carrying entries
    /// from one move to the next is the whole point — consecutive moves search
    /// largely the same tree. Across games it would only be noise: positions from a
    /// finished game competing for slots with the new one.
    ///
    /// Clearing rather than reallocating matters for the same reason the table is now
    /// kept at all: building a fresh one costs 7.3 ms, which is 7% of a late-game
    /// thinking budget. Overwriting in place costs a memset and no page faults.
    pub fn clear(&mut self) {
        for slot in self.slots.iter() {
            slot.check.store(0, Atomicity::Relaxed);
            slot.data.store(0, Atomicity::Relaxed);
        }
    }

    fn index(&self, key: u64) -> usize {
        // `ENTRIES` is a power of two, so this is the low bits of the key.
        (key as usize) & (ENTRIES - 1)
    }

    /// Look `key` up for a search of `depth` plies within `alpha..beta`.
    ///
    /// `ply` is needed because mate scores are stored relative to the node that
    /// found them — see [`Table::store`].
    pub fn probe(&self, key: u64, depth: u32, alpha: i32, beta: i32, ply: i32) -> Hit {
        let slot = &self.slots[self.index(key)];
        let data = slot.data.load(Atomicity::Relaxed);
        let check = slot.check.load(Atomicity::Relaxed);
        // An empty slot, another position at the same index, and a torn write are the same
        // answer here — and one comparison covers all three.
        if check ^ data != key {
            return Hit { cutoff: None, best: None, matched: false };
        }
        let Some((entry_depth, stored_score, bound, best)) = unpack(data) else {
            return Hit { cutoff: None, best: None, matched: false };
        };

        let score = from_table(stored_score, ply);
        // A shallower entry was searched less thoroughly than we are about to
        // search: its score is not trustworthy for this depth. Its *move* still
        // is — a good move stays a good move.
        let cutoff = if entry_depth < depth {
            None
        } else {
            match bound {
                Bound::Exact => Some(score),
                Bound::Lower if score >= beta => Some(score),
                Bound::Upper if score <= alpha => Some(score),
                _ => None,
            }
        };
        Hit { cutoff, best, matched: true }
    }

    /// Record what a search of `depth` plies concluded about `key`.
    ///
    /// Replacement is unconditional. Preferring deeper entries sounds better but
    /// lets a stale deep entry hold a slot for the whole search; always replacing
    /// keeps the table biased towards what the search is looking at now. Worth
    /// revisiting with a measurement rather than by intuition.
    pub fn store(&self, key: u64, depth: u32, score: i32, bound: Bound, best: Option<Move>) {
        self.store_at(key, depth, score, bound, best, 0)
    }

    /// As [`Table::store`], with the `ply` at which the score was found so mate
    /// scores can be made root-independent.
    pub fn store_at(
        &self,
        key: u64,
        depth: u32,
        score: i32,
        bound: Bound,
        best: Option<Move>,
        ply: i32,
    ) {
        let data = pack(depth, to_table(score, ply), bound, best);
        let slot = &self.slots[self.index(key)];
        // There is no write order that closes the window in which a reader sees one new word
        // beside one old one — which is precisely why the XOR is there instead of an order.
        slot.check.store(key ^ data, Atomicity::Relaxed);
        slot.data.store(data, Atomicity::Relaxed);
    }
}

/// Mate scores are the one value that cannot be cached as-is.
///
/// `negamax` returns `MATE - ply` for a mate, so the number depends on where the
/// node sits in *this* search. The same position reached at a different distance
/// from the root would read back a mate that is too near or too far — an engine
/// that announces mate in 3 and then fails to deliver it.
///
/// The fix is to store the distance from the *node*, not from the root: add `ply`
/// on the way in, subtract it on the way out. Non-mate scores pass through
/// untouched.
fn to_table(score: i32, ply: i32) -> i32 {
    if score >= MATE_THRESHOLD {
        score + ply
    } else if score <= -MATE_THRESHOLD {
        score - ply
    } else {
        score
    }
}

fn from_table(score: i32, ply: i32) -> i32 {
    if score >= MATE_THRESHOLD {
        score - ply
    } else if score <= -MATE_THRESHOLD {
        score + ply
    } else {
        score
    }
}

#[cfg(test)]
mod tests {
    use super::*;


    /// The property the whole lock-free design rests on, and the only one no *sequential* test
    /// can reach: a slot written by several threads at once must never be read as the pairing of
    /// one writer's key with another's data.
    ///
    /// Eight threads hammer **one** slot with distinct (key, data) pairs, chosen so the data
    /// encodes which key produced it. Every probe that succeeds is then checkable: it must hand
    /// back the data belonging to the key it asked for.
    ///
    /// This cannot prove tearing never happens — it is a race, so absence is not observable. It
    /// does catch the version that stores the key plainly, which is what matters: the argument in
    /// [`Slot`] says the XOR makes a torn pair fail the comparison, and this is that argument
    /// exposed to a real race rather than left as prose.
    #[test]
    fn a_slot_hammered_by_eight_threads_never_mixes_one_key_with_another_data() {
        use std::sync::atomic::{AtomicU64, Ordering as At};
        const THREADS: u64 = 8;
        const ROUNDS: u64 = 20_000;
        let table = Table::new();
        // All keys share their low 20 bits, so `index` sends every one of them to the same
        // slot: without that they would spread out and never collide.
        let key_of = |t: u64| (t << 32) | 0xABCD;
        let bad = AtomicU64::new(0);
        let seen = AtomicU64::new(0);
        std::thread::scope(|scope| {
            for t in 0..THREADS {
                let (table, bad, seen) = (&table, &bad, &seen);
                scope.spawn(move || {
                    // The depth carries the writer's identity, so a mismatched pair is visible.
                    let depth = (t + 1) as u32;
                    for _ in 0..ROUNDS {
                        table.store(key_of(t), depth, t as i32, Bound::Exact, None);
                        for other in 0..THREADS {
                            let hit = table.probe(key_of(other), 1, -30_000, 30_000, 0);
                            if let Some(score) = hit.cutoff {
                                seen.fetch_add(1, At::Relaxed);
                                if score != other as i32 {
                                    bad.fetch_add(1, At::Relaxed);
                                }
                            }
                        }
                    }
                });
            }
        });
        assert!(
            seen.load(At::Relaxed) > 0,
            "precondition: no probe ever matched, so the race was never exercised",
        );
        assert_eq!(
            bad.load(At::Relaxed),
            0,
            "{} of {} successful probes returned another writer's data",
            bad.load(At::Relaxed),
            seen.load(At::Relaxed),
        );
    }
    use crate::position::Position;
    use crate::search::MATE;

    fn a_move() -> Move {
        let p = Position::initial();
        p.move_from_uci("e2e4").unwrap()
    }

    #[test]
    fn an_exact_entry_is_returned() {
        let t = Table::new();
        t.store(1234, 4, 42, Bound::Exact, Some(a_move()));
        let hit = t.probe(1234, 4, -100, 100, 0);
        assert_eq!(hit.cutoff, Some(42));
        assert_eq!(hit.best, Some(a_move()));
    }

    #[test]
    fn a_shallower_entry_does_not_cut_off() {
        // Stored at depth 2, asked about depth 5: the score was obtained by a
        // less thorough search and must not stand in for a deeper one.
        let t = Table::new();
        t.store(1234, 2, 42, Bound::Exact, Some(a_move()));
        let hit = t.probe(1234, 5, -100, 100, 0);
        assert_eq!(hit.cutoff, None, "a shallow score must not be reused as deep");
        assert_eq!(hit.best, Some(a_move()), "but its move is still worth trying first");
    }

    #[test]
    fn a_key_mismatch_is_ignored() {
        // Two keys one table-size apart share an index — the collision case. This is
        // the test that guards the failure nobody would notice: one position's score
        // silently attributed to another.
        let t = Table::new();
        let key = 1234u64;
        let colliding = key + (ENTRIES as u64);
        assert_eq!(t.index(key), t.index(colliding), "the keys must actually collide");

        t.store(key, 9, 999, Bound::Exact, Some(a_move()));
        let hit = t.probe(colliding, 1, -100, 100, 0);
        assert_eq!(hit.cutoff, None, "another position's score must never be returned");
        assert_eq!(hit.best, None, "nor its move");
    }

    #[test]
    fn a_lower_bound_cuts_only_at_or_above_beta() {
        let t = Table::new();
        t.store(7, 4, 50, Bound::Lower, None);
        // The true value is >= 50. That settles a window whose beta is <= 50...
        assert_eq!(t.probe(7, 4, -100, 50, 0).cutoff, Some(50));
        // ...but says nothing when beta is above it: the value could be anywhere
        // from 50 upwards.
        assert_eq!(t.probe(7, 4, -100, 100, 0).cutoff, None);
    }

    #[test]
    fn an_upper_bound_cuts_only_at_or_below_alpha() {
        let t = Table::new();
        t.store(7, 4, 50, Bound::Upper, None);
        // The true value is <= 50, so a window already demanding 50 or more is
        // settled: this node cannot deliver.
        assert_eq!(t.probe(7, 4, 50, 100, 0).cutoff, Some(50));
        // With alpha below it, the value could still beat alpha — search it.
        assert_eq!(t.probe(7, 4, 0, 100, 0).cutoff, None);
    }

    #[test]
    fn a_mate_score_survives_a_round_trip_at_another_ply() {
        // A mate found 3 plies from the root scores `MATE - 3`. Cached and read
        // back 7 plies from the root, it must read as `MATE - 7` — the same mate,
        // the same number of moves away from *the node*, not from the root.
        let t = Table::new();
        t.store_at(99, 4, MATE - 3, Bound::Exact, None, 3);
        let hit = t.probe(99, 4, -MATE, MATE, 7);
        assert_eq!(hit.cutoff, Some(MATE - 7));
    }

    #[test]
    fn a_losing_mate_score_survives_too() {
        let t = Table::new();
        t.store_at(99, 4, -(MATE - 3), Bound::Exact, None, 3);
        assert_eq!(t.probe(99, 4, -MATE, MATE, 7).cutoff, Some(-(MATE - 7)));
    }

    #[test]
    fn an_ordinary_score_is_stored_verbatim() {
        // Only mate scores are ply-relative; a material evaluation must not be
        // shifted by the distance from the root.
        let t = Table::new();
        t.store_at(5, 3, 150, Bound::Exact, None, 6);
        assert_eq!(t.probe(5, 3, -1000, 1000, 11).cutoff, Some(150));
    }

    #[test]
    fn a_match_and_a_cutoff_are_two_different_answers() {
        // The distinction the two rates are built on, asserted where it actually lives now that
        // the counting moved to the searcher: `matched` says an entry for this position exists,
        // `cutoff` says it settles the current window. The middle case is the point — a key
        // match that buys move ordering only, and counting it as a cutoff would overstate what
        // the table buys in pruning.
        let t = Table::new();
        t.store(1, 5, 0, Bound::Exact, None);
        t.store(2, 1, 0, Bound::Exact, None);

        let deep = t.probe(1, 5, -1, 1, 0);
        assert!(deep.matched && deep.cutoff.is_some(), "deep enough: a match that cuts off");

        let shallow = t.probe(2, 5, -1, 1, 0);
        assert!(shallow.matched, "the entry is for this position, whatever its depth");
        assert!(shallow.cutoff.is_none(), "too shallow: ordering only, never a cutoff");

        let absent = t.probe(3, 5, -1, 1, 0);
        assert!(!absent.matched && absent.cutoff.is_none(), "no entry: neither");
    }
}
