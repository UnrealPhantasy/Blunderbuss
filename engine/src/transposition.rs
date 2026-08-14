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

use crate::position::Move;
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

/// One cached position.
///
/// `key` is the full Zobrist key, not the table index. Two positions can share
/// an index, so the key is what distinguishes "this entry is about the position
/// I am asking about" from "this entry belongs to someone else".
#[derive(Clone, Copy)]
struct Entry {
    key: u64,
    depth: u32,
    score: i32,
    bound: Bound,
    best: Option<Move>,
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
}

/// Number of entries. A power of two, so the index is a mask rather than a
/// modulo. 2^20 entries at 32 bytes each is about 33 MB — large enough to matter
/// at the depths this engine reaches, small enough to ignore on any machine.
const ENTRIES: usize = 1 << 20;

/// A fixed-size, single-threaded transposition table.
pub struct Table {
    /// Idiom: `Box<[T]>` is a heap slice of fixed length — a `Vec` without the
    /// capacity to grow, which is exactly what a table with a fixed mask wants.
    entries: Box<[Option<Entry>]>,
    hits: u64,
    probes: u64,
    cutoffs: u64,
}

impl Default for Table {
    fn default() -> Table {
        Table::new()
    }
}

impl Table {
    pub fn new() -> Table {
        Table { entries: vec![None; ENTRIES].into_boxed_slice(), hits: 0, probes: 0, cutoffs: 0 }
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
        self.entries.fill(None);
        self.hits = 0;
        self.probes = 0;
        self.cutoffs = 0;
    }

    /// Fraction of probes that found an entry for *this* position — a key match,
    /// whatever came of it.
    ///
    /// Worth reporting on its own: a table that is never read and a table whose
    /// every hit is rejected as a collision look identical from the node count, and
    /// only this rate separates a working table from a decorative one.
    ///
    /// Note what it is **not**: a matched entry may still be too shallow to cut off,
    /// in which case it contributes ordering only. Measured at depth 7, roughly half
    /// of the key matches fall in that case — see [`Table::cutoff_rate`].
    ///
    /// Measure it through [`search_timed`](crate::search::search_timed), never by
    /// driving `Searcher::root` in a loop: the real search feeds each iteration the
    /// previous one's best move, which changes the move order and therefore the mix
    /// of entries the table ends up holding.
    pub fn key_match_rate(&self) -> f64 {
        if self.probes == 0 {
            0.0
        } else {
            self.hits as f64 / self.probes as f64
        }
    }

    /// Fraction of probes that returned a score, i.e. that saved a whole subtree.
    ///
    /// The stricter of the two rates, and the one that measures what the table buys
    /// in pruning. The gap with [`Table::key_match_rate`] is the share of matches
    /// that were useful for move ordering but not deep enough to cut off. Measured
    /// through `search_timed(pos, Limits::depth(7))`, release build:
    ///
    /// | position | key match | cutoff |
    /// |---|---|---|
    /// | start position | 0.196 | 0.116 |
    /// | Kiwipete | 0.467 | 0.243 |
    /// | Ruy Lopez | 0.295 | 0.159 |
    pub fn cutoff_rate(&self) -> f64 {
        if self.probes == 0 {
            0.0
        } else {
            self.cutoffs as f64 / self.probes as f64
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
    pub fn probe(&mut self, key: u64, depth: u32, alpha: i32, beta: i32, ply: i32) -> Hit {
        self.probes += 1;
        let Some(entry) = self.entries[self.index(key)] else {
            return Hit { cutoff: None, best: None };
        };
        // A different position living at the same index. Its score says nothing
        // about ours.
        if entry.key != key {
            return Hit { cutoff: None, best: None };
        }
        self.hits += 1;

        let score = from_table(entry.score, ply);
        // A shallower entry was searched less thoroughly than we are about to
        // search: its score is not trustworthy for this depth. Its *move* still
        // is — a good move stays a good move.
        let cutoff = if entry.depth < depth {
            None
        } else {
            match entry.bound {
                Bound::Exact => Some(score),
                Bound::Lower if score >= beta => Some(score),
                Bound::Upper if score <= alpha => Some(score),
                _ => None,
            }
        };
        if cutoff.is_some() {
            self.cutoffs += 1;
        }
        Hit { cutoff, best: entry.best }
    }

    /// Record what a search of `depth` plies concluded about `key`.
    ///
    /// Replacement is unconditional. Preferring deeper entries sounds better but
    /// lets a stale deep entry hold a slot for the whole search; always replacing
    /// keeps the table biased towards what the search is looking at now. Worth
    /// revisiting with a measurement rather than by intuition.
    pub fn store(&mut self, key: u64, depth: u32, score: i32, bound: Bound, best: Option<Move>) {
        self.store_at(key, depth, score, bound, best, 0)
    }

    /// As [`Table::store`], with the `ply` at which the score was found so mate
    /// scores can be made root-independent.
    pub fn store_at(
        &mut self,
        key: u64,
        depth: u32,
        score: i32,
        bound: Bound,
        best: Option<Move>,
        ply: i32,
    ) {
        let i = self.index(key);
        self.entries[i] =
            Some(Entry { key, depth, score: to_table(score, ply), bound, best });
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
    use crate::position::Position;
    use crate::search::MATE;

    fn a_move() -> Move {
        let p = Position::initial();
        p.move_from_uci("e2e4").unwrap()
    }

    #[test]
    fn an_exact_entry_is_returned() {
        let mut t = Table::new();
        t.store(1234, 4, 42, Bound::Exact, Some(a_move()));
        let hit = t.probe(1234, 4, -100, 100, 0);
        assert_eq!(hit.cutoff, Some(42));
        assert_eq!(hit.best, Some(a_move()));
    }

    #[test]
    fn a_shallower_entry_does_not_cut_off() {
        // Stored at depth 2, asked about depth 5: the score was obtained by a
        // less thorough search and must not stand in for a deeper one.
        let mut t = Table::new();
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
        let mut t = Table::new();
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
        let mut t = Table::new();
        t.store(7, 4, 50, Bound::Lower, None);
        // The true value is >= 50. That settles a window whose beta is <= 50...
        assert_eq!(t.probe(7, 4, -100, 50, 0).cutoff, Some(50));
        // ...but says nothing when beta is above it: the value could be anywhere
        // from 50 upwards.
        assert_eq!(t.probe(7, 4, -100, 100, 0).cutoff, None);
    }

    #[test]
    fn an_upper_bound_cuts_only_at_or_below_alpha() {
        let mut t = Table::new();
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
        let mut t = Table::new();
        t.store_at(99, 4, MATE - 3, Bound::Exact, None, 3);
        let hit = t.probe(99, 4, -MATE, MATE, 7);
        assert_eq!(hit.cutoff, Some(MATE - 7));
    }

    #[test]
    fn a_losing_mate_score_survives_too() {
        let mut t = Table::new();
        t.store_at(99, 4, -(MATE - 3), Bound::Exact, None, 3);
        assert_eq!(t.probe(99, 4, -MATE, MATE, 7).cutoff, Some(-(MATE - 7)));
    }

    #[test]
    fn an_ordinary_score_is_stored_verbatim() {
        // Only mate scores are ply-relative; a material evaluation must not be
        // shifted by the distance from the root.
        let mut t = Table::new();
        t.store_at(5, 3, 150, Bound::Exact, None, 6);
        assert_eq!(t.probe(5, 3, -1000, 1000, 11).cutoff, Some(150));
    }

    #[test]
    fn the_two_rates_measure_different_things() {
        let mut t = Table::new();
        assert_eq!(t.key_match_rate(), 0.0, "no probe yet");
        assert_eq!(t.cutoff_rate(), 0.0, "no probe yet");

        // Three probes: a match that cuts off, a match too shallow to cut off, and
        // a miss. The point of the test is the middle one — it is a key match that
        // buys ordering only, and counting it as a cutoff would overstate the table.
        t.store(1, 5, 0, Bound::Exact, None);
        t.store(2, 1, 0, Bound::Exact, None);
        t.probe(1, 5, -1, 1, 0); // deep enough  -> match + cutoff
        t.probe(2, 5, -1, 1, 0); // too shallow  -> match, no cutoff
        t.probe(3, 5, -1, 1, 0); // absent       -> neither

        assert!((t.key_match_rate() - 2.0 / 3.0).abs() < 1e-9, "two of three matched");
        assert!((t.cutoff_rate() - 1.0 / 3.0).abs() < 1e-9, "only one settled the window");
    }
}
