//! Where the cost of a node goes -- the breakdown `evaluate` owes before anything tries to make it
//! cheaper (#92).
//!
//! # Why this exists at all
//!
//! Five bricks in a row returned zero Elo, and every one of them attacked the **number** of nodes.
//! A ply is worth 58 +- 7 Elo against the anchor, but that rule was measured by giving the engine
//! four times the *time* -- more nodes of the same kind -- so it prices time and throughput, and
//! overstates a pruning cut by about a factor of four. The one category it *can* price is the cost
//! of a node, and nothing has ever attacked it here. This module measures where that cost goes, so
//! that the brick which would try to reduce it can be sized before it is written rather than after.
//!
//! # The trap this module is shaped around
//!
//! The obvious way to price a term is to remove it and compare microseconds per node in a real
//! search. That measurement is **unattributable**: removing a term changes the score, which changes
//! the move ordering, which changes the cutoffs, which changes *which nodes are visited*. The
//! microseconds per node would then differ for two reasons at once and no arithmetic separates
//! them.
//!
//! So every timing here runs `evaluate` **outside any search**, over a fixed set of positions. A
//! node's cost is a sum, and the terms of a sum can be measured one at a time.
//!
//! # The second trap, found by this module's own blank
//!
//! Removing a term recompiles the function, and a recompiled function lands at another address with
//! another register allocation and another unrolling decision. That difference is worth **several
//! per cent** on its own, in either direction -- the first run of this harness had two variants
//! come out *slower* with a term removed, which is not a thing that can happen physically. That
//! signature is not only a symptom: it is turned into the measurement of the third trap below.
//!
//! So a variant is not one function here, it is [`REPLICAS`] compiled copies of the same source at
//! different addresses, and what the table reports is the median across them. The spread between
//! replicas of the *same* source is printed before anything else, because no share smaller than it
//! is a share.
//!
//! # The third trap, and it is the one the replicas do NOT close
//!
//! That blank bounds **one source compiled twice**. Every row of the table compares **two different
//! sources**, and a layout or inlining penalty attached to a body is systematic across all eight of
//! its copies -- so the replicas average over addresses and register allocations *within* a body
//! and cannot see it. The blank is therefore necessary and not sufficient: a lower bound on the
//! error the table actually makes.
//!
//! The harness measures that second floor instead of arguing about it, and the table's own
//! impossibilities are what measure it. **A negative share means removing code made the function
//! slower**, which cannot be true of the work done; so the largest negative share is a floor for
//! comparisons between bodies, produced by exactly the subtraction the real terms go through. On
//! this engine it runs at 3 to 6 %, and the endgame-scale row -- a test on two bitboards that
//! short-circuits on any position holding a pawn -- is negative in the lower bands on every run.
//! Mobility and passed pawns clear that floor by an order of magnitude and are unaffected; the two
//! small terms are **bounded, not priced**, and the report now says so on its own.
//!
//! Getting even that far took forcing the compiler's hand: with `lto = true` and
//! `codegen-units = 1`, LLVM merges functions with identical bodies, so the first blank compared a
//! function **with itself at one address** and read 0.04 %. A blank that cannot fail is the failure
//! mode this project has already paid for once, under another name. The `black_box(TAG)` at the top
//! of [`variant`] is what keeps the copies apart, and
//! `the_replicas_are_really_distinct_machine_code` is what keeps it that way.
//!
//! # How to read a figure out of it
//!
//! ```text
//! cargo test --release -p engine -- --ignored --test-threads=1 --nocapture cost::
//! ```
//!
//! `--release` is not optional -- a debug build is twenty to fifty times slower and measures a
//! different program. `--test-threads=1` matters as much: `cargo test` runs its tests in parallel
//! by default, so without it the harness competes with every other test on the machine.
//!
//! # What is deliberately not here
//!
//! No term is changed, tuned or removed for production. Every variant below is a **copy** built to
//! be timed and thrown away, and the copy is checked against the real function on both axes that
//! could make it lie: same score on every position of the set (asserted on every `cargo test`), and
//! same time to within the floor (printed).

use std::hint::black_box;
use std::time::{Duration, Instant};

use super::*;
use crate::search::{Limits, search_timed};

// ---------------------------------------------------------------- the position set
//
// Positions come from pseudo-random play at a fixed seed rather than from a hand-written list, for
// the reason the SEE sweep already gives: a list chosen by the author of the measurement is the
// one place its blind spots are least likely to be. The seed is in the source, so the set is
// reproducible to the position -- which is what AC#1 asks for.
//
// They are then **stratified by phase**, on the same four bands `banc.sh` uses, so a figure here
// can be laid beside a figure from the bench without a conversion. The stratification is not
// cosmetic: mobility is a lookup per minor piece and passed pawns are a mask per pawn, so both
// terms cost different amounts at different moments of a game, and an unstratified average would
// hide exactly the variation that decides whether the breakdown generalises.

/// How many pieces stand on the board. Reported beside the phase because the phase counts material
/// weight and the loop in `evaluate` counts *squares*: a board of eight pawns and two kings has a
/// phase of zero and still costs ten iterations.
fn occupied_count(pos: &Position) -> usize {
    Square::ALL.iter().filter(|&&sq| pos.piece_on(sq).is_some()).count()
}

/// The four phase bands, named as `banc.sh` names them, from a full board down to a thin endgame.
const STRATA: [(&str, i32, i32); 4] = [
    ("opening    (20-24)", 20, 24),
    ("middle high(16-19)", 16, 19),
    ("middle low (11-15)", 11, 15),
    ("endgame    ( 0-10)", 0, 10),
];

/// How many positions each band holds. 64 x 4 = 256 positions per timed pass, which is enough for
/// one pass to cost tens of microseconds -- far above the resolution of `Instant`, and small enough
/// that the whole set stays in cache, which is the regime a leaf evaluation actually runs in.
const PER_STRATUM: usize = 64;

/// A xorshift64, because the standard library has no random number generator and a measurement
/// that depends on a crate is a measurement that moves when the crate does.
///
/// Rust idiom: a tuple struct wrapping the state, so the generator is a value that can be handed
/// around rather than a global.
struct Xorshift(u64);

impl Xorshift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
}

/// The position set, one vector per stratum of [`STRATA`].
///
/// Built by playing pseudo-random games from the initial position and keeping positions as their
/// phase falls through the bands. Sampled sparsely (one candidate in eight, at most four per game
/// and band) because consecutive positions of one game differ by a single move and would make 64
/// positions carry the information of a handful.
fn position_set() -> Vec<Vec<Position>> {
    let mut rng = Xorshift(0x5EED_C057);
    let mut sets: Vec<Vec<Position>> = STRATA.iter().map(|_| Vec::new()).collect();
    let mut game = 0u32;
    while sets.iter().any(|s| s.len() < PER_STRATUM) {
        game += 1;
        // A bound rather than a `loop`: pseudo-random play reaches the endgame bands often but not
        // always, and a set that could not be filled must fail an assertion rather than hang.
        assert!(
            game < 20_000,
            "the position set could not be filled: {:?}",
            sets.iter().map(Vec::len).collect::<Vec<_>>(),
        );
        let mut pos = Position::initial();
        let mut taken = [0usize; STRATA.len()];
        for ply in 0..300 {
            let moves = pos.legal_moves();
            if moves.is_empty() {
                break;
            }
            pos = pos.play(moves[(rng.next() % moves.len() as u64) as usize]);
            // Skip the first few plies: they are the same handful of positions in every game, and
            // the opening band would otherwise be filled with near-duplicates of the start.
            if ply < 4 || rng.next() % 8 != 0 {
                continue;
            }
            if pos.legal_moves().is_empty() {
                break;
            }
            let ph = phase(&pos);
            let Some(band) = STRATA.iter().position(|&(_, lo, hi)| (lo..=hi).contains(&ph)) else {
                continue;
            };
            if taken[band] >= 4 || sets[band].len() >= PER_STRATUM {
                continue;
            }
            taken[band] += 1;
            sets[band].push(pos.clone());
        }
    }
    sets
}

// ---------------------------------------------------------------- the variants
//
// A **copy** of `evaluate`, parameterised by which terms it runs. A copy rather than a generic
// version of the real function, because the issue puts changing `evaluate` out of scope and a
// const-generic production signature would be a change to the hottest path in the engine for a
// measurement's convenience.
//
// A copy can lie in two ways, and both are checked rather than assumed:
//
//   * it can drift from the original, and then it prices a function nobody runs. Guarded by
//     `the_copy_scores_exactly_what_the_real_function_scores`, which is a plain `#[test]`: it runs
//     on every `cargo test`, so a change to `evaluate` that is not mirrored here goes red at once
//     rather than at the next measurement.
//   * it can cost something different from the original even while scoring the same. That one is
//     printed rather than asserted -- the `evaluate` row and the `full (copy)` row of the table are
//     the same source, and their difference is the price of the two things the *test* binary adds
//     to the real one: the call counter and the thread-local read of `SCALING`.

/// A set of named variants, each with its compiled copies, as [`measure`] takes them.
///
/// Rust idiom: a type alias, because the shape -- a slice of `(name, copies)` where a copy is a
/// **function pointer** rather than a closure -- appears in four signatures and clippy refuses the
/// repetition. `fn(&Position) -> i32` is a bare pointer and not `impl Fn`, which is what lets
/// copies of different monomorphisations sit in one `Vec`.
type Row<'a> = (&'a str, Vec<fn(&Position) -> i32>);
type Rows<'a> = [Row<'a>];

/// How many compiled copies of each variant are timed.
///
/// Not a repetition count -- these are distinct functions at distinct addresses, and the point is
/// that a body's cost depends on where the compiler put it and how it allocated its registers.
/// **Measured, on this machine and this position set: eight copies of one source spread over 4 to
/// 15 % of their own median**, so a single compilation of a variant says nothing about a term worth
/// 10 %. Eight is what makes the median stable enough to read such a term while keeping the whole
/// harness under half a minute, and it splits into two halves of four for the blank.
const REPLICAS: usize = 8;

/// A variant's [`REPLICAS`] compiled copies.
///
/// Rust idiom: a declarative macro, because the copies differ only by a **const generic** argument
/// and there is no way to build an array of monomorphisations from a loop -- each one is a distinct
/// type-level instantiation, resolved at compile time.
macro_rules! replicas {
    ($base:expr, $material_pst:expr, $passed:expr, $mobility:expr, $scale:expr) => {
        [
            variant::<{ $base }, $material_pst, $passed, $mobility, $scale>
                as fn(&Position) -> i32,
            variant::<{ $base + 1 }, $material_pst, $passed, $mobility, $scale>,
            variant::<{ $base + 2 }, $material_pst, $passed, $mobility, $scale>,
            variant::<{ $base + 3 }, $material_pst, $passed, $mobility, $scale>,
            variant::<{ $base + 4 }, $material_pst, $passed, $mobility, $scale>,
            variant::<{ $base + 5 }, $material_pst, $passed, $mobility, $scale>,
            variant::<{ $base + 6 }, $material_pst, $passed, $mobility, $scale>,
            variant::<{ $base + 7 }, $material_pst, $passed, $mobility, $scale>,
        ]
    };
}

fn full() -> [fn(&Position) -> i32; REPLICAS] {
    replicas!(0, true, true, true, true)
}
fn minus_mobility() -> [fn(&Position) -> i32; REPLICAS] {
    replicas!(10, true, true, false, true)
}
fn minus_passed() -> [fn(&Position) -> i32; REPLICAS] {
    replicas!(20, true, false, true, true)
}
fn minus_scale() -> [fn(&Position) -> i32; REPLICAS] {
    replicas!(30, true, true, true, false)
}
fn minus_material_pst() -> [fn(&Position) -> i32; REPLICAS] {
    replicas!(40, false, true, true, true)
}
fn bare_walk() -> [fn(&Position) -> i32; REPLICAS] {
    replicas!(50, false, false, false, false)
}

fn variant<
    const TAG: u8,
    const MATERIAL_PST: bool,
    const PASSED: bool,
    const MOBILITY: bool,
    const SCALE: bool,
>(
    pos: &Position,
) -> i32 {
    // **This line is the blank**, and without it there is no blank at all. `black_box` is opaque to
    // the optimiser, so a different `TAG` makes a different body and LLVM's function merging -- on
    // by default under this crate's `lto = true` -- can no longer collapse the copies into one
    // address. It costs one instruction, identically in every variant, so it cancels in every
    // difference the table reports.
    black_box(TAG);

    let mut middlegame = 0;
    let mut endgame = 0;
    let mut phase = 0;

    let pawns = [pos.pawns(Color::White), pos.pawns(Color::Black)];

    for sq in Square::ALL {
        if let Some(piece) = pos.piece_on(sq) {
            let color = pos.color_on(sq).expect("an occupied square has a colour");
            let square = sq.relative_to(color) as usize;
            // The material and the two table reads, together: they are the part an incremental
            // evaluation would maintain on `make_move` instead of recomputing here, so they are
            // switched as one block. What is left when they are off is the floor such a rewrite
            // would still have to pay at every node.
            let (mut mg, mut eg) = if MATERIAL_PST {
                let material = value(piece);
                (
                    material + PST_MIDDLEGAME[piece as usize][square],
                    material + PST_ENDGAME[piece as usize][square],
                )
            } else {
                (0, 0)
            };
            if PASSED && piece == Piece::Pawn {
                let enemy = pawns[!color as usize];
                if is_passed(sq as usize, color, enemy) {
                    let rank = square / 8;
                    let (mut bonus_mg, mut bonus_eg) =
                        (PASSED_MIDDLEGAME[rank], PASSED_ENDGAME[rank]);
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
            if MOBILITY
                && (MOBILITY_MIDDLEGAME[piece as usize] != 0
                    || MOBILITY_ENDGAME[piece as usize] != 0)
            {
                let mob = pos.mobility_from(sq, piece, color) as i32;
                mg += MOBILITY_MIDDLEGAME[piece as usize] * mob;
                eg += MOBILITY_ENDGAME[piece as usize] * mob;
            }
            phase += phase_weight(piece);
            let sign = match color {
                Color::White => 1,
                Color::Black => -1,
            };
            middlegame += sign * mg;
            endgame += sign * eg;
        }
    }

    let phase = phase.min(MAX_PHASE);
    let mut balance = (middlegame * phase + endgame * (MAX_PHASE - phase)) / MAX_PHASE;

    // No thread-local read here, unlike `evaluate` under `cfg(test)`: this copy mirrors the
    // **production** function, which reads a compile-time `true`. Mirroring the test build instead
    // would fold that read into the price of the endgame scale and inflate the one term it is
    // hardest to see.
    if SCALE && cannot_mate(pos, pawns, balance) {
        balance /= DRAWISH_DIVISOR;
    }

    match pos.side_to_move() {
        Color::White => balance,
        Color::Black => -balance,
    }
}

// ---------------------------------------------------------------- the timing loop

/// How long one timed point should last. Below a few tens of milliseconds a point measures the
/// scheduler; far above it, the pass stops fitting between two interruptions on a busy machine.
const POINT: Duration = Duration::from_millis(20);

/// How many alternating rounds each copy is timed for. Odd, so the median is an observation and not
/// the mean of two. Small because the dominant noise here is **not** the clock -- it is which
/// machine code the compiler produced for a given copy, and no number of rounds averages that out.
/// [`REPLICAS`] is the axis that does.
const ROUNDS: usize = 5;

/// Nanoseconds per call of `f` over `positions`, repeated `reps` times.
///
/// `black_box` on both the argument and the result is what stops the optimiser from hoisting the
/// call out of the loop or deleting it as dead: without them, a function whose result is unused and
/// whose argument does not change is free, and the harness would print the cost of an empty loop.
fn time_one(f: fn(&Position) -> i32, positions: &[Position], reps: u32) -> f64 {
    let start = Instant::now();
    let mut acc = 0i64;
    for _ in 0..reps {
        for p in positions {
            acc += black_box(f(black_box(p))) as i64;
        }
    }
    let elapsed = start.elapsed();
    black_box(acc);
    elapsed.as_secs_f64() * 1e9 / (reps as f64 * positions.len() as f64)
}

/// How many repetitions of `positions` it takes for one point to last [`POINT`], measured rather
/// than guessed so the harness behaves the same on a fast machine and a slow one.
fn calibrate(f: fn(&Position) -> i32, positions: &[Position]) -> u32 {
    let mut reps = 4u32;
    loop {
        let start = Instant::now();
        let ns = time_one(f, positions, reps);
        if start.elapsed() >= POINT || reps > 1 << 22 {
            // One pass costs `ns * positions.len()`; ask for as many as fill POINT.
            let want = POINT.as_secs_f64() * 1e9 / (ns * positions.len() as f64);
            return (want as u32).max(1);
        }
        reps *= 4;
    }
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.partial_cmp(b).expect("a duration is never NaN"));
    xs[xs.len() / 2]
}

/// What one variant costs: the median over its copies of the median over the rounds.
struct Reading {
    /// The variant's cost in nanoseconds per call.
    ns: f64,
    /// The spread between its copies, as a percentage of `ns` -- how much recompiling the same
    /// source moves its cost.
    spread: f64,
    /// The medians of the first and second half of the copies.
    ///
    /// **This is what the blank is made of, and the spread above is not.** A spread is the range of
    /// eight samples; what a term's share has to be compared against is the error on a *median of
    /// eight*, which is several times smaller. Splitting `full`'s copies in two and comparing the
    /// two halves reproduces exactly the arithmetic the table does to a term -- one median against
    /// another -- on two groups that are known to be identical.
    halves: (f64, f64),
}

/// Time every copy of every variant, alternating between them round after round.
///
/// Alternation and median rather than a mean of consecutive runs, for the reason `cpu-cost.sh`
/// gives: a machine drifts over the seconds a measurement takes -- another process starts, the
/// clock boosts and settles -- and a drift that lands entirely on the variant timed last reads as
/// that variant being slower.
fn measure(rows: &Rows, positions: &[Position], reps: u32) -> Vec<Reading> {
    let mut samples: Vec<Vec<Vec<f64>>> =
        rows.iter().map(|(_, fs)| vec![Vec::with_capacity(ROUNDS); fs.len()]).collect();
    // Every (variant, copy) pair that has to be timed, as a flat list. Shuffled anew each round,
    // which is the second half of what alternation is for: a machine that drifts over a round --
    // the clock settling out of turbo, a browser tab waking up -- would otherwise hand a
    // systematic bias to whichever variant always sits at the same place in the order. Fixing the
    // order and only alternating rounds does not remove that; shuffling does.
    let mut order: Vec<(usize, usize)> = rows
        .iter()
        .enumerate()
        .flat_map(|(i, (_, fs))| (0..fs.len()).map(move |j| (i, j)))
        .collect();
    let mut rng = Xorshift(0x51DE_0DDE);
    for _ in 0..ROUNDS {
        // Fisher-Yates, at a fixed seed so two runs of the harness shuffle identically and the
        // measurement stays reproducible.
        for k in (1..order.len()).rev() {
            order.swap(k, (rng.next() % (k as u64 + 1)) as usize);
        }
        for &(i, j) in &order {
            samples[i][j].push(time_one(rows[i].1[j], positions, reps));
        }
    }
    samples
        .into_iter()
        .map(|per_copy| {
            let copies: Vec<f64> = per_copy.into_iter().map(median).collect();
            let ns = median(copies.clone());
            let (lo, hi) = copies.iter().fold((f64::MAX, 0.0f64), |(l, h), &x| (l.min(x), h.max(x)));
            let half = copies.len() / 2;
            let halves = if half > 0 {
                (median(copies[..half].to_vec()), median(copies[half..].to_vec()))
            } else {
                (ns, ns)
            };
            Reading {
                ns,
                spread: if copies.len() > 1 { 100.0 * (hi - lo) / ns } else { 0.0 },
                halves,
            }
        })
        .collect()
}

// ---------------------------------------------------------------- the report

/// What a reduction of `share` in the cost of a node is worth against the anchor.
///
/// `58 * ln(1 / (1 - share)) / ln(2.14)`: 58 +- 7 Elo per ply and an effective branching factor of
/// 2.14, both measured on 11 200 games on 2026-09-03. **This is the one conversion the rule
/// covers** -- it was obtained by giving the engine more time, so it prices throughput. A pruning
/// cut, which changes the shape of the tree rather than its size, is overstated by it by about a
/// factor of four and has to be measured in Elo directly.
fn elo_from_saving(share: f64) -> f64 {
    if share <= 0.0 || share >= 1.0 {
        return 0.0;
    }
    58.0 * (1.0 / (1.0 - share)).ln() / 2.14f64.ln()
}

#[test]
fn the_copy_scores_exactly_what_the_real_function_scores() {
    // The guard that keeps the whole harness honest, and the reason it is a plain `#[test]` rather
    // than part of the ignored measurement: a copy of `evaluate` that drifts prices a function
    // nobody runs, and the drift would otherwise be invisible until someone next took a
    // measurement. Here it goes red on the `cargo test` of the commit that causes it.
    let sets = position_set();
    let copies = full();
    let mut checked = 0;
    for set in &sets {
        for p in set {
            let want = evaluate(p);
            for (i, f) in copies.iter().enumerate() {
                assert_eq!(f(p), want, "copy {i} has drifted from `evaluate` on {}", p.to_fen());
            }
            checked += 1;
        }
    }
    assert_eq!(checked, PER_STRATUM * STRATA.len(), "the position set is not the size it claims");
}

#[test]
fn the_replicas_are_really_distinct_machine_code() {
    // **The blank's own blank.** With `lto = true` and `codegen-units = 1`, LLVM merges functions
    // with identical bodies: the first version of this harness compared `full` against itself at
    // one address and reported a floor of 0.04 %, which is the reading of an instrument that
    // cannot fail. Two variants then came out *slower* with a term removed -- impossible
    // physically, and the sign that the real floor was several per cent and hidden.
    //
    // So the copies must be distinct functions, and that has to be checked rather than trusted: it
    // depends on an optimiser decision, and nothing about `black_box(TAG)` is guaranteed by the
    // language to keep them apart for ever.
    let mut seen: Vec<usize> = Vec::new();
    for (name, fs) in [("full", full()), ("minus mobility", minus_mobility())] {
        for (i, f) in fs.iter().enumerate() {
            let addr = *f as usize;
            assert!(
                !seen.contains(&addr),
                "{name} copy {i} was merged with an earlier one: the spread between copies is \
                 then not a floor but the repeatability of the clock, and every share this \
                 harness prints is unguarded",
            );
            seen.push(addr);
        }
    }
}

#[test]
fn every_variant_actually_changes_the_score_it_claims_to_drop() {
    // Without this, a variant whose term never fires on the position set would read as free, and
    // the breakdown would report a zero that means "the door never opened" rather than "the term
    // is cheap". That exact reading has already been paid for twice on this repository, on the
    // endgame bricks the node bench is blind to.
    let sets = position_set();
    let all: Vec<&Position> = sets.iter().flatten().collect();
    let reference = full()[0];
    for (name, f) in [
        ("material and PST", minus_material_pst()[0]),
        ("passed pawns", minus_passed()[0]),
        ("mobility", minus_mobility()[0]),
    ] {
        let differing = all.iter().filter(|p| f(p) != reference(p)).count();
        assert!(
            differing > 0,
            "dropping {name} changed no score on {} positions: the variant is inert and its \
             timing measures nothing",
            all.len(),
        );
    }
    // The endgame scale is the one term that fires on a minority of positions by construction --
    // it needs a pawnless board -- so it gets its own witness rather than the set above, which is
    // pawnless only by accident.
    let drawn = Position::from_fen("4k3/8/8/8/8/8/8/3NK3 w - - 0 1").unwrap();
    assert_ne!(
        minus_scale()[0](&drawn),
        reference(&drawn),
        "the endgame scale is inert even on a lone minor against a bare king",
    );
}

/// The depth of the search leg. Deep enough that the transposition table and the pruning have all
/// come into play, shallow enough that eight positions finish in seconds.
const SEARCH_DEPTH: u32 = 8;
/// How many positions of each band the search leg uses. Two, because the figure it produces is a
/// ratio over hundreds of thousands of nodes and not an average over positions.
const SEARCH_POSITIONS: usize = 2;

#[test]
#[ignore = "measurement: times `evaluate` for about half a minute, run explicitly with --ignored"]
fn where_the_cost_of_a_node_goes() {
    let sets = position_set();

    println!(
        "\nPOSITION SET  {} positions, {PER_STRATUM} per stratum, pseudo-random play at seed \
         0x5EEDC057",
        PER_STRATUM * STRATA.len(),
    );
    for (i, (name, _, _)) in STRATA.iter().enumerate() {
        let phases: Vec<i32> = sets[i].iter().map(phase).collect();
        let pieces: f64 =
            sets[i].iter().map(|p| occupied_count(p) as f64).sum::<f64>() / sets[i].len() as f64;
        println!(
            "  {name}  n={:3}  phase {}..{}  {pieces:.1} pieces on the board",
            sets[i].len(),
            phases.iter().min().expect("a stratum is never empty"),
            phases.iter().max().expect("a stratum is never empty"),
        );
    }

    // ---- the control that decides how everything below is measured
    //
    // Timing the four strata concatenated would be the obvious simplification, and it is wrong: at
    // 256 positions the set stops fitting in cache and every call pays for fetching its own
    // position. The first run of this harness did exactly that and read 60 % more per call than
    // the strata it is made of -- a cost that belongs to the harness's own working set and not to
    // `evaluate`. In a real search the position being evaluated was built by the move just played
    // and is hot, so the small-set regime is the honest one.
    //
    // The control stays in the harness rather than in a comment because the mistake it catches is
    // a *simplification*: it is what a later reader would do to make the code shorter.
    let all: Vec<Position> = sets.iter().flatten().cloned().collect();
    let one_row: Vec<Row> = vec![("full", full().to_vec())];
    let concatenated = measure(&one_row, &all, calibrate(full()[0], &all))[0].ns;
    let per_stratum: Vec<Reading> = (0..STRATA.len())
        .map(|i| measure(&one_row, &sets[i], calibrate(full()[0], &sets[i])).remove(0))
        .collect();
    let mean_stratum: f64 =
        per_stratum.iter().map(|r| r.ns).sum::<f64>() / per_stratum.len() as f64;
    println!("\nWORKING-SET CONTROL, and it is why nothing below is timed on the whole set at once");
    println!(
        "  {} positions at once: {concatenated:.1} ns/call     mean of the four strata of \
         {PER_STRATUM}: {mean_stratum:.1} ns/call     {:+.0} %",
        all.len(),
        100.0 * (concatenated - mean_stratum) / mean_stratum,
    );

    // ---- the breakdown, one stratum at a time
    let rows: Vec<Row> = vec![
        ("evaluate (production fn)", vec![evaluate as fn(&Position) -> i32]),
        ("full (copy)", full().to_vec()),
        ("minus mobility", minus_mobility().to_vec()),
        ("minus passed pawns", minus_passed().to_vec()),
        ("minus endgame scale", minus_scale().to_vec()),
        ("minus material and PST", minus_material_pst().to_vec()),
        ("bare walk (no term)", bare_walk().to_vec()),
    ];
    let readings: Vec<Vec<Reading>> = (0..STRATA.len())
        .map(|i| measure(&rows, &sets[i], calibrate(full()[0], &sets[i])))
        .collect();

    println!("\nBLANK, read before anything else");
    println!(
        "  {:<20} {:>12} {:>12} {:>10}",
        "stratum", "half A", "half B", "blank",
    );
    let mut worst_blank = 0.0f64;
    for (i, (name, _, _)) in STRATA.iter().enumerate() {
        let r = &readings[i][1];
        let blank = 100.0 * (r.halves.0 - r.halves.1).abs() / r.ns;
        worst_blank = worst_blank.max(blank);
        println!(
            "  {name:<20} {:>11.1} {:>12.1} {:>9.1} %",
            r.halves.0, r.halves.1, blank,
        );
    }
    println!(
        "  {REPLICAS} compiled copies of the SAME source, split in two halves of {}. Worst: \
         {worst_blank:.1} %.",
        REPLICAS / 2,
    );
    println!(
        "  This bounds ONE source compiled twice. It is only a LOWER bound on the error between \
         two DIFFERENT sources,\n  which is what every row below actually compares -- a layout or \
         inlining penalty attached to a body is\n  systematic across all {REPLICAS} of its copies, \
         so no number of replicas can average it out. The row after\n  the table measures that \
         second floor.",
    );
    println!(
        "  For reference, the full spread between the {REPLICAS} copies of `full`: {}",
        STRATA
            .iter()
            .enumerate()
            .map(|(i, _)| format!("{:.1} %", readings[i][1].spread))
            .collect::<Vec<_>>()
            .join("  "),
    );

    println!("\nCOST OF ONE CALL, per phase band, ns");
    print!("  {:<26}", "variant");
    for (name, _, _) in STRATA.iter() {
        print!(" {:>12}", name.split_whitespace().next().expect("a name is never empty"));
    }
    println!();
    for (j, (name, _)) in rows.iter().enumerate() {
        print!("  {name:<26}");
        for r in readings.iter() {
            print!(" {:>12.1}", r[j].ns);
        }
        println!();
    }

    println!("\nSHARE OF `evaluate` EACH TERM COSTS, per phase band, %");
    print!("  {:<26}", "term");
    for (name, _, _) in STRATA.iter() {
        print!(" {:>12}", name.split_whitespace().next().expect("a name is never empty"));
    }
    println!();
    // **The floor between two different bodies, read off the table's own impossibilities.**
    // A negative share means removing code made the function slower, which cannot be true of the
    // work done and is therefore a pure artefact of how the two bodies were compiled. So the
    // largest negative share in the table is a *measured* lower bound on the error between two
    // different sources -- the quantity the blank above cannot see, produced by the same
    // arithmetic the table applies to a real term. It is a lower bound rather than the floor
    // itself: an artefact that happens to be positive is indistinguishable from a term.
    let mut body_floor = 0.0f64;
    for (j, (name, _)) in rows.iter().enumerate().skip(2) {
        print!("  {:<26}", name.replace("minus ", ""));
        for r in readings.iter() {
            let share = 100.0 * (r[1].ns - r[j].ns) / r[1].ns;
            // `bare walk` is excluded: it is not a term and its share is not a subtraction of one.
            if j < rows.len() - 1 && share < 0.0 {
                body_floor = body_floor.max(-share);
            }
            print!(" {share:>11.1} %");
        }
        println!();
    }
    println!(
        "  BODY FLOOR {body_floor:.1} % -- the largest negative share above. Removing code cannot \
         make the work smaller\n  and the function slower, so that reading is an artefact of \
         compilation, measured by the same\n  subtraction the real terms go through. Any share \
         under it is not readable, whatever the blank says.",
    );
    // A term is *priced* only where its share is positive AND above the floor. Taking the absolute
    // value instead would let a term qualify on the strength of an impossible reading, which is the
    // opposite of what the floor is for.
    for (j, (name, _)) in rows.iter().enumerate().skip(2).take(rows.len() - 3) {
        let best = readings
            .iter()
            .map(|r| 100.0 * (r[1].ns - r[j].ns) / r[1].ns)
            .fold(f64::MIN, f64::max);
        if best < body_floor {
            println!(
                "    -> `{}` never clears it in any band: this measurement cannot price that \
                 term, only bound it under {body_floor:.1} % of the function.",
                name.replace("minus ", ""),
            );
        }
    }
    println!(
        "  The last row is NOT the sum of the others and must not be read as one. With every term          off,\n  the optimiser also drops the plumbing they shared -- the colour lookup, the          square flip, the two\n  running scores -- so it is a lower bound on a walk that computes          nothing, not the walk's own cost.",
    );
    println!(
        "\n  What that row does say, and it is the one figure here that points somewhere: the          bare walk\n  costs {:.0}-{:.0} ns whatever the position holds, against {:.0} pieces on          the board in the opening\n  and {:.0} in the endgame band. It is a loop over 64 squares,          and most of those squares are empty.",
        readings.iter().map(|r| r[6].ns).fold(f64::MAX, f64::min),
        readings.iter().map(|r| r[6].ns).fold(0.0, f64::max),
        sets[0].iter().map(|p| occupied_count(p) as f64).sum::<f64>() / sets[0].len() as f64,
        sets[3].iter().map(|p| occupied_count(p) as f64).sum::<f64>() / sets[3].len() as f64,
    );

    // ---- what a node costs, and what fraction of it this function is
    //
    // Measured in the same process, on the same positions, at the same moment as everything above.
    // Quoting a cost per node taken weeks ago on another build would compare two machines as much
    // as two functions.
    println!(
        "\nWHAT A NODE COSTS, same binary, same positions, real search at depth {SEARCH_DEPTH}",
    );
    println!(
        "  {:<20} {:>11} {:>11} {:>10} {:>9} {:>9} {:>9}",
        "stratum", "nodes", "eval calls", "calls/node", "ns/node", "ns/call", "share",
    );
    let (mut nodes_total, mut calls_total) = (0u64, 0u64);
    let (mut time_total, mut eval_time_total) = (0.0f64, 0.0f64);
    let mut calls_by_band = [0u64; STRATA.len()];
    for (i, (name, _, _)) in STRATA.iter().enumerate() {
        let (mut n, mut c, mut t) = (0u64, 0u64, Duration::ZERO);
        for p in sets[i].iter().take(SEARCH_POSITIONS) {
            EVAL_CALLS.with(|x| x.set(0));
            let start = Instant::now();
            let stats = search_timed(p, Limits::depth(SEARCH_DEPTH));
            t += start.elapsed();
            n += stats.nodes;
            c += EVAL_CALLS.with(|x| x.get());
        }
        let ns_call = readings[i][1].ns;
        let ns_node = t.as_secs_f64() * 1e9 / n as f64;
        println!(
            "  {name:<20} {n:>11} {c:>11} {:>10.3} {ns_node:>9.1} {ns_call:>9.1} {:>8.1} %",
            c as f64 / n as f64,
            100.0 * (c as f64 * ns_call) / (t.as_secs_f64() * 1e9),
        );
        nodes_total += n;
        calls_total += c;
        calls_by_band[i] = c;
        time_total += t.as_secs_f64() * 1e9;
        eval_time_total += c as f64 * ns_call;
    }

    let calls_per_node = calls_total as f64 / nodes_total as f64;
    let ns_per_node = time_total / nodes_total as f64;
    let share = eval_time_total / time_total;
    // What an incremental evaluation would still have to compute at every node: the terms that
    // depend on the whole board's occupation, which no `make_move` can maintain from the piece that
    // moved. Read from the variant with material and the tables removed -- the part that *is*
    // incrementalisable. Weighted by each band's evaluation calls, so the figure describes the
    // positions the search actually evaluates and not the four bands equally.
    let non_incremental: f64 = readings
        .iter()
        .zip(calls_by_band.iter())
        .map(|(r, &calls)| (r[5].ns / r[1].ns) * calls as f64)
        .sum::<f64>()
        / calls_total as f64;

    // The cost per call the share is actually built on, and it has to be this one rather than the
    // mean of the four bands: the share weights each band by the evaluation calls the search made
    // there, so an unweighted mean prints a chain that does not close. Measured difference on this
    // set: 163.7 ns weighted against 157.5 unweighted, 3.8 % apart -- more than the blank this
    // report opens with, and enough to make the reader who checks the arithmetic find 22.7 % where
    // the line below says 23.5 %.
    let ns_per_call = eval_time_total / calls_total as f64;

    println!("\nTHE ANSWER");
    println!("  cost of one node          {ns_per_node:.1} ns");
    println!("  calls to evaluate / node  {calls_per_node:.3}");
    println!(
        "  cost of one call          {ns_per_call:.1} ns (weighted by each band's calls; the \
         per-band figures are above)",
    );
    println!("  -> evaluate is            {:.1} % of a node", 100.0 * share);
    // What `cfg(test)` adds to `evaluate` and to nothing else: the call counter and the
    // thread-local read of `SCALING`. The two rows are the same source, so their difference is that
    // overhead -- **and it does not cancel**, contrary to what this line claimed until a reviewer
    // checked it. The numerator uses `full (copy)`, which deliberately mirrors the *production*
    // function and carries neither; the overhead sits in the denominator alone, through the search
    // leg, which runs the test build. It therefore biases the share **down** by about
    // `share^2 x overhead`, computed below rather than described.
    //
    // Two warnings on how to read it. `evaluate` is the one row with a single compiled copy, so
    // this is a median of one against a median of eight -- the comparison the rest of this module
    // argues does not survive. And it comes out negative on some runs, the production function
    // timing *faster* than a copy doing strictly less work, which is a reading below the floor
    // rather than a result. Both are why the correction is stated and not applied.
    let test_overhead: f64 = readings
        .iter()
        .map(|r| (r[0].ns - r[1].ns) / r[1].ns)
        .sum::<f64>()
        / readings.len() as f64;
    let bias = 100.0 * share * share * test_overhead;
    println!(
        "     (the test build's own counter and SCALING read cost {:+.1} % of a call, in the \
         denominator only:\n      the share above is biased low by {bias:+.2} points. {})",
        100.0 * test_overhead,
        if test_overhead < 0.0 {
            "Negative here, which is below the floor and not a result"
        } else {
            "One compiled copy against eight, so it is itself a reading near the floor"
        },
    );
    println!(
        "  of which not incremental  {:.1} % (mobility, passed pawns and the scale read the \
         whole board)",
        100.0 * non_incremental,
    );
    let recoverable = share * (1.0 - non_incremental).max(0.0);
    println!(
        "  -> ceiling of an incremental evaluation: {:.1} % of a node, worth {:.1} Elo anchored",
        100.0 * recoverable,
        elo_from_saving(recoverable),
    );
    println!(
        "  for scale: -10 % of a node is {:.1} Elo, -20 % is {:.1}, -40 % is {:.1}",
        elo_from_saving(0.10),
        elo_from_saving(0.20),
        elo_from_saving(0.40),
    );

    // Assertions, and they are deliberately about the instrument rather than about the result. A
    // threshold on a share would pin a number this measurement exists to discover; a threshold on
    // the blank catches the one failure that makes every number here meaningless. It is wide on
    // purpose: `check.sh` runs the ignored sweeps in parallel, and a developer running the whole
    // suite at once will see a noisier blank. A gate that cries wolf gets commented out.
    assert!(
        worst_blank < 20.0,
        "the blank reads {worst_blank:.1} %: the machine is too busy for this measurement to mean \
         anything, re-run it with --test-threads=1 on a quiet machine",
    );
    assert!(nodes_total > 0 && calls_total > 0, "the search leg produced nothing to divide by");
    // Dimensional coherence, and it is the only property of the search leg that can be asserted
    // without pinning a number this measurement exists to discover: the time attributed to
    // `evaluate` is a part of the search's time, so it cannot exceed it. What that catches is a
    // cost per call measured in the wrong regime or in the wrong unit -- the two mistakes that
    // would silently multiply the published share.
    //
    // **What it deliberately does not claim, having been checked by mutation.** An earlier version
    // asserted `calls_per_node < 1.0`, on the reasoning that a ratio at one would mean the gate in
    // `static_eval_for` had stopped gating. Running that mutation -- `usable = true`, every
    // interior node paying for an evaluation -- moved the ratio from 0.656 to **0.727** and left
    // the assertion green. It was inert: `negamax_inner` returns through the transposition table
    // and through several cuts before it ever reaches the gate, so most interior nodes never call
    // `evaluate` whatever the gate says. The property is real and it is guarded, but **not** by
    // `one_evaluation_per_node_and_never_one_per_move` -- that test stays green under the same
    // mutation, since it asserts `evals <= nodes` and `considered > evals` and a gate that stops
    // gating raises `evals` without breaking either. The three tests that actually go red were
    // read off the mutation rather than reasoned about:
    // `search::tests::a_node_in_check_is_never_evaluated`,
    // `search::tests::the_gate_stops_at_the_deeper_of_the_two_ceilings`, and
    // `search::tests::pruning_losing_captures_keeps_every_forced_mate` on its own precondition.
    // Retiring the inert assertion is therefore safe; naming the wrong survivor would have been
    // the same defect one layer up, in the comment that exists to record the defect.
    assert!(
        eval_time_total > 0.0 && eval_time_total < time_total,
        "{:.0} ns attributed to evaluate out of {:.0} ns of search: the two legs were measured in \
         different units or in different regimes, and every share above is unreadable",
        eval_time_total,
        time_total,
    );
    // **And the size of the hole it leaves, because a guard that does not state its own reach is
    // half of one.** The assertion above only fires when the share reaches 1.0, so at a measured
    // share it tolerates an error of `1 / share` -- about 4.3x here. Checked from both sides rather
    // than assumed: a cost per call multiplied by three passes it and publishes "evaluate is
    // 70.5 % of a node"; multiplied by five it fails. The mistake it is built for is a unit slip,
    // which is a factor of a thousand; the mistake it cannot see is a regime or weighting slip,
    // which is a factor of two or three, and that is the plausible one.
    //
    // Which is what the next assertion is for. It is not another dimensional check but an
    // **algebraic identity**: the three numbers `THE ANSWER` prints above the share must reduce to
    // the share exactly, since `calls/nodes x (eval_time/calls) / (time/nodes)` is `eval_time/time`
    // by construction. It costs nothing and it closes precisely the hole above -- a cost per call
    // taken from the wrong pass, or averaged without the weights it needs, breaks the identity
    // while leaving the dimensional check green. That is the defect this very block shipped with
    // and a reviewer found, and the reason it is now checked rather than described.
    let chain = calls_per_node * ns_per_call / ns_per_node;
    assert!(
        (chain - share).abs() < 1e-9,
        "the printed chain does not close: {calls_per_node:.6} x {ns_per_call:.4} / \
         {ns_per_node:.4} = {chain:.9}, against a share of {share:.9}. The cost per call printed \
         in THE ANSWER is not the one the share is built on",
    );
}
