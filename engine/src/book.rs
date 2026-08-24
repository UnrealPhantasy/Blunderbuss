//! The opening book: a move to play instantly, so the clock is never spent on a position whose
//! answer is already known.
//!
//! # Why a book is not a clock heuristic
//!
//! Two clock bricks were refused in this repository because moving a fixed budget between phases
//! is zero-sum by construction. A book does not move time, it **removes work**: the engine does
//! not search at all while the position is in the book, and the saving propagates by itself,
//! since the per-move budget is computed from a clock that was never spent. No time-management
//! code is involved.
//!
//! Where the saving lands is what makes it worth having. The depth gap against the anchor is
//! −5.5 plies in the opening and −1.8 plies at moves 31-50, so the time is taken from the phase
//! where half a ply changes nothing and given to the one where a ply is worth 28% of the gap.
//!
//! # Keyed by position, never by move sequence
//!
//! An entry is a Zobrist key. That is the whole design, and it answers the two questions a book
//! has to answer without any special case:
//!
//! - **Transpositions are free.** The London System reached by `1.d4 d5 2.Bf4` and by
//!   `1.d4 Nf6 2.Bf4 d5` is one key, so both orders get the same answer. A book keyed by move
//!   sequence would need every order enumerated.
//! - **Named variations need no names.** The Najdorf and the Dragon share nine plies and diverge
//!   on the tenth; they are simply two different keys. The engine never recognises an opening, it
//!   looks one up. Naming belongs to the game analyst, which is a different artefact.
//!
//! # The file, and why it is lines rather than entries
//!
//! The source is a list of **lines** — UCI moves from the start position, one opening per line —
//! and the table is built by replaying them. Authoring stays readable, and the transposition
//! property above becomes structural rather than something the author has to maintain: two lines
//! arriving at the same position both contribute to the same key.

use crate::position::{Move, Position};
use std::collections::HashMap;

/// What the book knows about one position: the moves played from it, with how many lines went
/// through each.
///
/// The weight is not a strength judgement — it comes from how the file is written. It exists so
/// a position offered by several lines is answered in proportion, and so the second stage of this
/// brick (statistics filtered by rating band) has somewhere to put its numbers.
type Entry = Vec<(Move, u32)>;

/// The most weight one move can accumulate, and it exists because of a defect a test found.
///
/// A weight counts the lines passing through a move. In a **generated** tree that number measures
/// the size of the subtree below it, not the quality of the move — an artefact of how the file
/// was produced. Merging a hand-written book of 27 lines with a generated one of 8 700 therefore
/// let the large file drown the small one: `c2c4`, offered by two authored lines against
/// thousands through `e2e4`, was picked roughly once in four thousand, and the merged book opened
/// three ways instead of four.
///
/// Capping keeps a small deliberate line from being erased by a large mechanical one, while
/// preserving the ordering the weights are for. The value is not tuned and does not need to be:
/// stage two of this brick replaces these counts with statistics filtered by rating band, and
/// this cap goes with them.
const WEIGHT_CAP: u32 = 8;

pub struct Book {
    entries: HashMap<u64, Entry>,
    /// Rust idiom: interior state on a `&self` method would need `Cell`. It is kept as a plain
    /// field instead and `pick` takes `&mut self`, because the caller owns the book anyway and an
    /// engine that cannot vary its openings is the one thing this brick exists to avoid.
    rng: u64,
}

impl Book {
    /// Build a book from lines of UCI moves. `#` starts a comment; blank lines are ignored.
    ///
    /// A line that contains a move which is not legal in the position it reached is **skipped
    /// entirely**, and the count of skipped lines is returned alongside. Silently ignoring a
    /// typo would leave a book smaller than its author believes, which is the same failure as a
    /// test that passes without testing anything.
    pub fn from_lines(text: &str) -> (Book, usize) {
        let mut entries: HashMap<u64, Entry> = HashMap::new();
        let mut skipped = 0;

        for line in text.lines() {
            let line = line.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            // Collected before recording anything: a line is accepted whole or not at all, so a
            // typo in the eighth move cannot leave the first seven half-recorded.
            let mut pos = Position::initial();
            let mut steps = Vec::new();
            let mut ok = true;
            for token in line.split_whitespace() {
                match pos.move_from_uci(token) {
                    Some(mv) if pos.legal_moves().contains(&mv) => {
                        steps.push((pos.hash(), mv));
                        pos = pos.play(mv);
                    }
                    _ => {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok {
                skipped += 1;
                continue;
            }
            for (key, mv) in steps {
                let slot = entries.entry(key).or_default();
                match slot.iter_mut().find(|(m, _)| *m == mv) {
                    Some((_, weight)) => *weight = (*weight + 1).min(WEIGHT_CAP),
                    None => slot.push((mv, 1)),
                }
            }
        }
        (Book { entries, rng: 0x9E37_79B9_7F4A_7C15 }, skipped)
    }

    /// How many positions the book answers. Used by callers to say "no book" rather than
    /// "an empty book", which behave the same and mean different things.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// A move for `pos`, or `None` if the book does not know it.
    ///
    /// Chosen among the entry's moves in proportion to their weights. Varying rather than always
    /// answering the same move is the point: an engine that replays one line every game hands the
    /// opponent a free preparation, and diversity is most of what a book buys against a human.
    pub fn pick(&mut self, pos: &Position) -> Option<Move> {
        let entry = self.entries.get(&pos.hash())?;
        let total: u32 = entry.iter().map(|(_, w)| w).sum();
        if total == 0 {
            return None;
        }
        // Xorshift64: three shifts, no state beyond the seed. Enough for choosing between a
        // handful of book moves, and deterministic for a given sequence of calls — so a test that
        // constructs a book and probes it twice gets a reproducible answer.
        self.rng ^= self.rng << 13;
        self.rng ^= self.rng >> 7;
        self.rng ^= self.rng << 17;
        let mut ticket = (self.rng % total as u64) as u32;
        for (mv, weight) in entry {
            if ticket < *weight {
                return Some(*mv);
            }
            ticket -= weight;
        }
        // Unreachable while the weights sum to `total`, and returning the last move rather than
        // panicking keeps a book bug out of the game.
        entry.last().map(|(mv, _)| *mv)
    }

    /// Start the selection sequence again, so two games from the same position do not have to
    /// follow the same line. Called at the start of a game, never between moves of one.
    pub fn reseed(&mut self, seed: u64) {
        // Zero is the one state a xorshift cannot leave, so it is mapped away rather than trusted
        // not to occur.
        self.rng = if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The shipped book, so a typo in it is a test failure rather than a smaller book than its
    /// author believes.
    const SHIPPED: &str = include_str!("../../book.txt");

    /// The generated book, whose lines nobody can check by eye — which is exactly why a test
    /// checks them. A corrupted regeneration has to be a test failure, not a discovery in a game.
    const GENERATED: &str = include_str!("../../book-genere.txt");

    #[test]
    fn every_line_of_both_books_is_legal() {
        let (hand, skipped) = Book::from_lines(SHIPPED);
        assert_eq!(skipped, 0, "{skipped} line(s) of book.txt contain an illegal move");
        assert!(hand.len() > 40, "the hand-written book answers only {} positions", hand.len());

        let (generated, skipped) = Book::from_lines(GENERATED);
        assert_eq!(skipped, 0, "{skipped} line(s) of book-genere.txt contain an illegal move");
        assert!(
            generated.len() > 5_000,
            "the generated book answers only {} positions",
            generated.len(),
        );
    }

    #[test]
    fn concatenating_the_two_books_merges_them_by_position() {
        // Why there are two files rather than a choice between them. They have opposite
        // strengths — the hand-written one is broader at the root and readable, the generated one
        // is far deeper and opaque — and the line format merges them for free, since a position
        // offered by both simply collects both continuations.
        let (hand, _) = Book::from_lines(SHIPPED);
        let (generated, _) = Book::from_lines(GENERATED);
        let (both, skipped) = Book::from_lines(&format!("{SHIPPED}\n{GENERATED}"));
        assert_eq!(skipped, 0);
        assert!(
            both.len() >= generated.len() && both.len() > hand.len(),
            "merged {} against {} generated and {} hand-written",
            both.len(), generated.len(), hand.len(),
        );

        // The property that matters more than the count: the union is broader at the root. The
        // hand-written file offers four first moves, the generated one three (Stockfish's top
        // three), and an engine that opens four ways is harder to prepare against than one that
        // opens three.
        let mut both = both;
        let start = Position::initial();
        let mut seen = std::collections::HashSet::new();
        for _ in 0..400 {
            if let Some(mv) = both.pick(&start) {
                seen.insert(format!("{mv}"));
            }
        }
        assert!(seen.len() >= 4, "the merged book opens only {} ways: {seen:?}", seen.len());
    }

    #[test]
    fn the_two_london_move_orders_are_one_key() {
        // The property that justifies keying on position rather than on the move sequence, and
        // the reason the file may be written in whichever order reads best. `1.d4 d5 2.Bf4 Nf6
        // 3.e3` and `1.d4 Nf6 2.Bf4 d5 3.e3` reach the identical position with Black to move.
        let walk = |line: &str| {
            let mut p = Position::initial();
            for t in line.split_whitespace() {
                p = p.play(p.move_from_uci(t).expect("a legal book move"));
            }
            p
        };
        let a = walk("d2d4 d7d5 c1f4 g8f6 e2e3");
        let b = walk("d2d4 g8f6 c1f4 d7d5 e2e3");
        assert_eq!(a.hash(), b.hash(), "precondition: the two move orders must transpose");

        let (mut book, _) = Book::from_lines(SHIPPED);
        // Both orders are in the file, so the key carries both continuations; whichever is
        // picked, it must be one the book knows for *that* position.
        let mv = book.pick(&a).expect("the book must answer the transposed position");
        assert!(a.legal_moves().contains(&mv), "the book answered an illegal move");
    }

    #[test]
    fn a_line_with_an_illegal_move_is_skipped_whole() {
        // Not half-recorded: a typo in the fourth move must not leave the first three in the
        // table, or the book would answer a position its author never meant to cover.
        let (book, skipped) = Book::from_lines("e2e4 e7e5 g1f3 z9z9
");
        assert_eq!(skipped, 1);
        assert!(book.is_empty(), "a rejected line left {} entries behind", book.len());
    }

    #[test]
    fn an_unknown_position_is_not_answered() {
        let (mut book, _) = Book::from_lines(SHIPPED);
        // Reached by 1.a3 a6 — the shape the measurement protocol actually plays, and precisely
        // what a book of mainlines does not cover.
        let mut p = Position::initial();
        for t in ["a2a3", "a7a6"] {
            p = p.play(p.move_from_uci(t).unwrap());
        }
        assert_eq!(book.pick(&p), None, "the book claimed to know an offbeat position");
    }

    #[test]
    fn weights_follow_how_many_lines_pass_through() {
        // Three lines share `1.e4 e5 2.Nf3 Nc6`, two of which continue `3.Bb5`. Over many picks
        // the split has to follow that, otherwise the weights are decorative and the second stage
        // of this brick has nowhere to put its statistics.
        let (mut book, _) = Book::from_lines(SHIPPED);
        let mut p = Position::initial();
        for t in ["e2e4", "e7e5", "g1f3", "b8c6"] {
            p = p.play(p.move_from_uci(t).unwrap());
        }
        let bb5 = p.move_from_uci("f1b5").unwrap();
        let picks = (0..600).filter(|_| book.pick(&p) == Some(bb5)).count();
        // Two of four lines from this position play Bb5, so ~50%. The bounds are wide because
        // this asserts that the weighting *happens*, not what the exact ratio is.
        assert!(
            (150..450).contains(&picks),
            "Bb5 chosen {picks} times in 600: the weights are not being read",
        );
    }
}
