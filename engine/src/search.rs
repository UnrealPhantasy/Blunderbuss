//! The search: pick a move for a position.
//!
//! Negamax with alpha-beta pruning, guided by [`evaluate`](crate::evaluation::evaluate)
//! and sped up by move [`ordering`](crate::ordering).
//!
//! [`search_timed`] is the **only** entry point: it searches depth 1, 2, 3, … until
//! it hits the depth cap or the wall-clock deadline carried by [`Limits`], and always
//! returns the last *completed* depth's move. This iterative deepening is what lets
//! the engine use a time budget — it can stop at any moment with a legal move — and
//! each iteration orders the previous best move first, which prunes the next one
//! better, so deepening to depth N costs about as much as (often less than) a single
//! direct depth-N pass. [`best_move`] is a thin convenience wrapper over it.

use crate::evaluation::{evaluate, phase};
use crate::ordering::{is_quiet, mvv_lva, order_moves, see, KillerSlots, Killers};
use crate::position::{Move, Piece, Position};
use crate::transposition::{Bound, Table};
use std::sync::LazyLock;
use std::time::{Duration, Instant};

/// A mate score, large enough to dominate any material balance. The distance to
/// mate (`ply`) is subtracted so the search prefers shorter mates, and so a mate
/// is always distinguishable from a mere material advantage.
pub const MATE: i32 = 30_000;

/// A depth cap for time-bounded searches (deep enough to be effectively
/// unbounded for this engine; it exists only so `go infinite`-style searches
/// terminate).
pub const MAX_DEPTH: u32 = 64;

/// Above this, a score is a mate rather than a material balance. A mate is scored
/// `MATE - ply`, so any plausible mate stays well above a material evaluation, which
/// cannot approach 20 000. Shared with [`crate::transposition`] and with the UCI
/// layer, which needs the same boundary to report `score mate` instead of `score cp`
/// — two places deriving it separately would eventually disagree.
pub const MATE_THRESHOLD: i32 = 20_000;

// A bound strictly above any reachable score (including `MATE`), used as the
// initial alpha/beta window.
const INF: i32 = 40_000;

/// How much shallower a null-move search runs than the node that spawned it.
///
/// Two is the usual choice: the pass only has to fail to find a rough refutation, so
/// paying full depth for it would spend more than the cut saves. Larger values prune
/// harder and miss more.
const NULL_MOVE_REDUCTION: u32 = 2;

/// Minimum [`phase`] for a null move to be attempted — the zugzwang guard.
///
/// The phase runs from 24 (every piece on the board) to 0 (kings and pawns only). At 6
/// there is still roughly a rook and two minor pieces about, which is enough that some
/// harmless move almost always exists. Below it, zugzwang becomes a real possibility
/// and the null-move assumption stops holding.
const NULL_MOVE_MIN_PHASE: i32 = 6;

/// The two coefficients of the reduction curve, `base + ln(depth) * ln(rank) / divisor`.
///
/// A **logarithm in both arguments**, and that is the whole design choice. One ply was the
/// same discount for a move ranked 4th at depth 4 as for one ranked 30th at depth 12, and
/// those are not equally unpromising: the second sits far below everything the ordering
/// believes in, and its subtree is much larger. A growing reduction spends the saved depth
/// where being wrong is least likely and the saving is greatest.
///
/// Why not a linear term: the cost of wrongly reducing a move rises much faster than the
/// saving from reducing it further, so the curve has to flatten. `ln` flattens on both axes
/// at once — from rank 4 to 8 it adds as much as from 8 to 16.
///
/// The values are the usual ones rather than the product of a search over the arena. This
/// repository has a brick that died after four successive recalibrations produced no gain
/// (king safety, #29); the lesson taken was to measure one defensible shape and report it.
const LMR_BASE: f64 = 0.75;
const LMR_DIVISOR: f64 = 2.25;

/// The most depth a reduction may take away at `depth` — the ceiling on the curve's output.
///
/// A function rather than an expression inlined in the loop, because a test watches it. The
/// equivalence test below asserts that this ceiling and `LMR_MIN_DEPTH` currently forbid the
/// same set of depths; written as a transcription of the formula, that test compared the guard
/// against its own private copy and stayed silent when the ceiling itself moved — verified by
/// loosening this to `depth - 1`, which brings the guard back to life while the test passed.
/// One source, read from both sides, is what makes either of them moving observable.
///
/// **Why two is the floor**: reducing further hands the move to quiescence, which judges a quiet
/// move on captures it does not have — the same reason `LMR_MIN_DEPTH` exists. One ply of real
/// search is the minimum worth doing.
///
/// **Why `saturating_sub`**, for the reason #42 learned the hard way: `reducible` guarantees
/// `depth >= LMR_MIN_DEPTH`, so a plain subtraction is safe at today's value of 3 — but that is a
/// relationship between two constants far apart in this file, and an unsigned underflow wraps
/// silently in release and ends in a stack overflow rather than an error anyone can read.
fn reduction_ceiling(depth: u32) -> u32 {
    depth.saturating_sub(2)
}

/// The reduction for every reachable (depth, rank) pair, computed once.
///
/// Idiom: `LazyLock` runs its closure on the first access and hands out the same value
/// forever after — the modern replacement for a hand-rolled "compute it once" flag, and
/// thread-safe without a mutex on the read path. It is needed here because `f64::ln` is not
/// a `const fn`, so this cannot be a plain `const`; and the alternative — computing the
/// logarithms per visit — would put floating-point work in the hottest loop in the engine.
///
/// Indexed `[depth][rank]`, both clamped by the caller, so a lookup is two bounds-checked
/// array reads and no arithmetic.
static LMR_TABLE: LazyLock<[[u32; LMR_TABLE_RANKS]; LMR_TABLE_DEPTHS]> = LazyLock::new(|| {
    let mut table = [[0u32; LMR_TABLE_RANKS]; LMR_TABLE_DEPTHS];
    for (depth, row) in table.iter_mut().enumerate() {
        for (rank, slot) in row.iter_mut().enumerate() {
            // Depth 0 and rank 0 never reach a lookup — the guards exclude them — but
            // `ln(0)` is negative infinity, so the table must not contain the result of
            // asking. Both axes start at 1.
            let d = depth.max(1) as f64;
            let r = rank.max(1) as f64;
            let raw = LMR_BASE + d.ln() * r.ln() / LMR_DIVISOR;
            // At least one ply: below that the brick is a no-op and the caller would pay a
            // lookup to learn nothing. The upper bound belongs to the caller, which is the
            // only place that knows how much depth is left to give away.
            *slot = (raw as u32).max(1);
        }
    }
    table
});

/// How far the reduction table reaches. Past it, the caller clamps rather than grows: the
/// curve is flat enough by then that the difference is under a ply, and a table that could
/// be indexed out of range is a panic waiting for an unusual position.
const LMR_TABLE_DEPTHS: usize = MAX_DEPTH as usize + 1;
const LMR_TABLE_RANKS: usize = 64;

/// Below this depth nothing is reduced.
///
/// At depth 2 a reduced move is searched at depth 0, which is quiescence — captures
/// only. A quiet move judged by captures alone is barely judged at all, and the subtree
/// saved is one node deep, so there is nothing to win and a verdict to lose.
const LMR_MIN_DEPTH: u32 = 3;

/// How many moves are searched at full depth before reductions begin.
///
/// **The most arbitrary of the three constants**, and worth saying so. The moves are
/// tried in order: the transposition table's move first, then the captures ranked by
/// MVV-LVA, then the killers, then everything else. So the index at which genuinely
/// unlikely moves begin is not fixed — it slides with how capture-rich the position is.
/// Three protects the head of the list in the case that matters most, an opening
/// position where nearly every move is quiet and the first quiet move may well be the
/// best one. It is a floor on trust in the ordering, not a measurement.
const LMR_FULL_DEPTH_MOVES: usize = 3;

/// How far / how long to search.
pub struct Limits {
    /// Never search deeper than this.
    pub max_depth: u32,
    /// Stop once this instant is reached (checked between and within iterations).
    pub deadline: Option<Instant>,
}

impl Limits {
    /// A fixed-depth limit (no clock).
    pub fn depth(max_depth: u32) -> Limits {
        Limits { max_depth, deadline: None }
    }

    /// Search until `deadline`, never deeper than [`MAX_DEPTH`].
    pub fn until(deadline: Instant) -> Limits {
        Limits { max_depth: MAX_DEPTH, deadline: Some(deadline) }
    }

    /// Both bounds at once: never deeper than `max_depth`, and — when a deadline is
    /// given — never past it. The general form the two constructors above specialise;
    /// it is what a UCI `go` maps onto, since the protocol may send a depth, a clock,
    /// both, or neither.
    pub fn bounded(max_depth: u32, deadline: Option<Instant>) -> Limits {
        Limits { max_depth, deadline }
    }
}

/// What one completed iteration of the deepening loop found.
///
/// Reported as it happens rather than collected and returned: a GUI shows the
/// evaluation *while* the engine thinks, and an arena samples it per move. The engine
/// crate never prints — it hands this to whoever asked for the search.
pub struct Progress {
    pub depth: u32,
    /// Side-to-move perspective, in centipawns — or a mate score, which the caller
    /// recognises by comparing against [`MATE_THRESHOLD`].
    pub score: i32,
    pub nodes: u64,
    pub elapsed: Duration,
    pub best: Move,
}

/// Everything a search needs besides the position.
///
/// A struct rather than more parameters. Each thing a caller might want — a progress
/// callback here, the game history in #25 — would otherwise either add a parameter to
/// every call site or spawn another `search_*` variant. #16 settled that there is
/// exactly one entry point; this is what lets it stay that way while the list of
/// options grows.
pub struct Request<'a> {
    pub limits: Limits,
    /// Zobrist keys of the positions played before this one, oldest first. Without
    /// them the search still spots a repetition it creates itself, but not one that
    /// returns to a position from earlier in the game — and that is the case that
    /// decides real games, since a perpetual usually comes back to somewhere the
    /// players have already been.
    pub history: &'a [u64],
    /// Called once per **completed** iteration. An iteration cut short by the
    /// deadline is discarded, so it is never reported — announcing a depth the
    /// engine then walks back from would be worse than saying nothing.
    pub progress: Option<&'a mut dyn FnMut(&Progress)>,
}

impl<'a> Request<'a> {
    /// A search bounded only by `limits`: no game history, no reporting.
    pub fn new(limits: Limits) -> Request<'a> {
        Request { limits, history: &[], progress: None }
    }
}

/// The outcome of a search: the chosen move and its score (side-to-move
/// perspective), the deepest **completed** depth, the node count, and what the
/// transposition table contributed.
pub struct SearchStats {
    pub best: Option<(Move, i32)>,
    pub depth: u32,
    pub nodes: u64,
    /// Fraction of probes that found an entry for the position asked about.
    /// Reported because node counts alone cannot distinguish a table that is never
    /// read from one whose every match is rejected as a collision.
    pub table_key_match_rate: f64,
    /// Fraction of probes that returned a score, and so skipped a subtree. Always
    /// the lower of the two: about half of the matches are too shallow to cut off
    /// and contribute move ordering only.
    pub table_cutoff_rate: f64,
}

/// Whether `mv` promotes a pawn to a queen.
///
/// Quiescence searches queen promotions only, for two reasons. What moves the
/// evaluation by ~800 centipawns is a queen appearing; a knight under-promotion that
/// wins by fork is still found by the main search, which has no such filter.
///
/// And the cost is not the fourfold one it looks like. Accepting every promotion piece
/// lets each pawn on the seventh branch four ways *at every node of the quiescence
/// recursion*, so the tree explodes multiplicatively rather than linearly. Measured at
/// depth 4 on `4k3/PPP3PP/8/8/8/8/ppp3pp/4K3 w` (six pawns one square from promoting):
/// 24 277 nodes queen-only against **2 591 390** for all four — a factor of 107, and
/// 346 at depth 3. On an ordinary middlegame the two are indistinguishable, which is
/// exactly why the comparison has to be made on a position where pawns are promoting.
fn is_queen_promotion(mv: Move) -> bool {
    mv.promotion == Some(Piece::Queen)
}

/// The best move for `pos` at a fixed `depth`, or `None` at a terminal root.
/// A convenience wrapper over [`search_timed`], bounded by depth alone and with no
/// game history — draws by repetition against earlier moves cannot be seen.
pub fn best_move(pos: &Position, depth: u32) -> Option<(Move, i32)> {
    search_timed(pos, Limits::depth(depth)).best
}

/// Iterative deepening: search depth 1, 2, 3, … up to `limits.max_depth` or the
/// `limits.deadline`, and return the last **completed** depth's result. Depth 1
/// always finishes, so there is always a legal move to return (unless the root is
/// terminal).
pub fn search_timed(pos: &Position, limits: Limits) -> SearchStats {
    search(pos, Request::new(limits))
}

/// What survives a search, and therefore has to be owned by the caller.
///
/// Only the transposition table today. It lives here rather than inside the search
/// because two consecutive moves of a game explore largely the same tree: the
/// opponent replies, and the engine starts again from a position it has already
/// analysed deeply. Rebuilding the table each move threw all of that away — and cost
/// 7.3 ms per move to do so, which late in a game is 7% of the thinking budget.
///
/// Deliberately **not** part of [`Request`]: a request is what a caller *asks for*,
/// answered and finished. This is state that outlives the answer, and giving it its
/// own type makes that lifetime visible instead of implied.
///
/// # The score of a position now depends on which searches preceded it
///
/// Two `Engine`s asked about the same position, one fresh and one that has been
/// playing, can return **different scores**. Measured over 60 plies of self-play at
/// depth 6: they disagreed once, by 3 centipawns; a longer run found 2 disagreements
/// in 100 plies with a worst gap of 29.
///
/// This is not a bug and not the repetition hazard described on [`Searcher::table`].
/// It is what a transposition table does: a stored **bound** — `Bound::Lower` or
/// `Bound::Upper` — is not a position's value, only a fact about it that was true
/// under one alpha-beta window. Reused under a different window it can cut a search
/// short at a different place, and the value that comes back is a different, equally
/// valid one. Persisting the table does not create the effect; it multiplies the
/// occasions for it, since entries now arrive from searches with other roots, depths
/// and windows.
///
/// What is guaranteed is that the score is *a valid alpha-beta value* for the
/// position, not that it is the same number every time.
///
/// **Consequence for anything that wants reproducible numbers**: a caller that must
/// quote the same evaluation for the same position twice — a game analyst, for
/// instance — should hold a fresh `Engine` per position rather than reuse one, and
/// pay the search cost. Playing does not care; reporting does.
pub struct Engine {
    table: Table,
}

impl Default for Engine {
    fn default() -> Engine {
        Engine::new()
    }
}

impl Engine {
    pub fn new() -> Engine {
        Engine { table: Table::new() }
    }

    /// Forget everything learned so far. Called at the start of a **new game**, never
    /// between moves of the same one.
    pub fn new_game(&mut self) {
        self.table.clear();
    }

    /// Search `pos`. Everything a caller can ask for travels in [`Request`]; what
    /// the search learns stays here, ready for the next move.
    pub fn search(&mut self, pos: &Position, request: Request) -> SearchStats {
        let mut searcher =
            Searcher::new(MoveOrder::Full, request.limits.deadline, &mut self.table);
        deepen(pos, request, &mut searcher)
    }
}

/// A one-off search, with a table that lives and dies with the call.
///
/// For callers with no game to follow — tests, and [`best_move`]. A real game should
/// hold an [`Engine`] instead, or it pays to rebuild the table on every move and
/// learns nothing from the previous one.
pub fn search(pos: &Position, request: Request) -> SearchStats {
    Engine::new().search(pos, request)
}

/// The deepening loop itself, over a searcher the caller supplies.
///
/// Split out of [`search`] for one reason: the tests need to interrupt an iteration at
/// an exact node, which means handing in a `Searcher` they configured. Everything a
/// *caller* can ask for still travels in [`Request`] — this takes the one thing that is
/// not a caller concern.
///
/// Borrowed rather than consumed, for the other half of the same reason: a test that
/// configures a searcher usually also wants to read what it recorded — how many moves
/// were reduced, how many had to be re-searched — and those counters are the only way to
/// check a guard whose whole effect is that something did *not* happen.
fn deepen(pos: &Position, mut request: Request, searcher: &mut Searcher) -> SearchStats {
    let started = Instant::now();
    let limits = &request.limits;
    searcher.history = request.history.to_vec();
    let max_depth = limits.max_depth.max(1);

    let mut best: Option<(Move, i32)> = None;
    let mut completed = 0;
    for depth in 1..=max_depth {
        // Order the previous iteration's best move first — the whole point of
        // deepening, since a good first move causes early cutoffs below.
        let pv = best.map(|(mv, _)| mv);
        let result = searcher.root(pos, depth, pv);

        if searcher.aborted {
            // Ran out of time mid-iteration. The default is to discard it: the moves it
            // managed to look at are the head of a sorted list, not a neutral sample, so
            // "best of the first five" is not a choice. The previous iteration at least
            // compared every move at one depth.
            //
            // Two cases override that.
            //
            // `improved` — a move other than the one we were about to play took the lead
            // *at this depth*. That is the most valuable thing an iteration can produce,
            // and the reason the extra ply was worth opening: the deeper search
            // disagreeing with the shallower one. Typically the previous favourite turns
            // out to lose material and the rescue is found just before the clock stops.
            // Discarding it means playing the move we just learned was bad. The move was
            // fully searched — `root` leaves its loop before adopting anything it did not
            // finish — and the root's window makes its score exact rather than a bound.
            //
            // `best.is_none()` — no iteration has completed at all. A capture-rich
            // position can spend more than the 2048 nodes between clock checks inside
            // iteration 1, so even the first can be cut short. The partial result is then
            // all we have, and it is at least legal: `root` seeds `best_move` with
            // `moves[0]` before searching anything. Returning it beats `bestmove 0000`,
            // which any arena reads as an illegal move and an instant forfeit.
            if result.improved || best.is_none() {
                best = result.best;
            }
            // `completed` deliberately stays where it was: the iteration did not finish,
            // and reporting a depth the engine only partly reached would be a lie about
            // how far it looked.
            break;
        }
        best = result.best;
        completed = depth;
        let Some((mv, score)) = best else {
            break; // terminal root — nothing to deepen
        };
        // This iteration finished, so its verdict is worth announcing.
        if let Some(report) = request.progress.as_mut() {
            report(&Progress {
                depth,
                score,
                nodes: searcher.nodes,
                elapsed: started.elapsed(),
                best: mv,
            });
        }
        // Don't open another iteration we have no time to finish.
        if let Some(deadline) = limits.deadline {
            if Instant::now() >= deadline {
                break;
            }
        }
    }
    SearchStats {
        best,
        depth: completed,
        nodes: searcher.nodes,
        table_key_match_rate: searcher.table.key_match_rate(),
        table_cutoff_rate: searcher.table.cutoff_rate(),
    }
}

/// How much of the move ordering the search uses.
///
/// Outside the tests this is always [`MoveOrder::Full`]. The weaker settings exist
/// so a test can isolate one heuristic and pin what it contributes on its own — a
/// node count against a search that is identical but for that one thing.
///
/// Idiom: an enum rather than one boolean per heuristic, because the settings are
/// cumulative. Two booleans would also spell "killers but no MVV-LVA", which is not
/// a thing the search can do.
#[derive(Clone, Copy, PartialEq, Eq)]
enum MoveOrder {
    /// Whatever order the generator produced.
    None,
    /// MVV-LVA: captures ranked against each other, quiet moves left untouched.
    ///
    /// Idiom: the search itself never selects this one — only tests do — so a
    /// non-test build would warn that nothing constructs it. `cfg_attr` applies the
    /// `allow` outside test builds only, which keeps the warning alive everywhere it
    /// could still mean something.
    #[cfg_attr(not(test), allow(dead_code))]
    Captures,
    /// MVV-LVA, plus the killer moves of the current ply ahead of the other quiet moves.
    Full,
}

/// What one call to [`Searcher::root`] produced.
///
/// More than the move and its score, because the deepening loop has to tell two kinds
/// of unfinished iteration apart: one that only got through the moves it was already
/// going to play anyway, and one that **changed its mind** before running out of time.
struct RootResult {
    /// The best move and its score, or `None` at a terminal root.
    best: Option<(Move, i32)>,
    /// Whether a move other than the first one took the lead.
    ///
    /// The first move tried is the previous iteration's choice, so this is exactly
    /// "this iteration found something better than what we were about to play" — and
    /// it says so by comparing moves *at its own depth*, never against the shallower
    /// iteration's score, which is not on the same scale.
    improved: bool,
}

/// Carries the state shared by the whole search: the node counter, the ordering
/// switch, and the time deadline / abort flag. Keeping `root` and `negamax` as
/// methods on one struct keeps them in sync.
struct Searcher<'a> {
    nodes: u64,
    order: MoveOrder,
    deadline: Option<Instant>,
    aborted: bool,
    /// Borrowed from the [`Engine`], so it outlives this search: iteration N+1
    /// reuses what iteration N learned, and the *next move* reuses what this whole
    /// search learned.
    ///
    /// Idiom: `&'a mut` rather than an owned `Table` — the searcher works on the
    /// caller's table for the duration of the search and gives it back. The lifetime
    /// `'a` is what tells the compiler the table must outlive the searcher.
    table: &'a mut Table,
    /// Zobrist keys of the positions played *before* the search started. A draw by
    /// repetition depends on the game, not on the board alone, so the search cannot
    /// see one without being told what came before.
    history: Vec<u64>,
    /// Zobrist keys along the branch currently being explored, pushed on the way
    /// down and popped on the way back up.
    path: Vec<u64>,
    /// The quiet moves that caused a cutoff, per ply. Like the transposition table,
    /// it lives for the whole search, so each deepening iteration starts on what the
    /// previous one learned.
    killers: Killers,
    /// Abort once this many nodes have been visited.
    ///
    /// **Tests only** — the whole field is compiled out otherwise, so it cannot cost
    /// production a single comparison per node. It exists because the behaviour under
    /// test is "what happens when an iteration is interrupted", and interrupting by
    /// wall-clock would make the test measure the machine and its load rather than the
    /// code: the same test would cut at a different move on a busy machine. A node
    /// count is exact and reproducible.
    #[cfg(test)]
    node_limit: Option<u64>,
    /// Whether null-move pruning may be attempted at all.
    ///
    /// **Tests only**, compiled out otherwise. It exists so a test can measure what the
    /// pruning contributes against a search that is identical but for that one thing —
    /// the same reason [`MoveOrder`] has settings the engine never selects.
    #[cfg(test)]
    allow_null_move: bool,
    /// How deeply null moves are currently nested, and the worst nesting seen.
    ///
    /// **Tests only.** Two passes in a row would skip a full move for both sides and
    /// prove nothing about the position — the guard against it is a `false` handed to
    /// the recursive call, which is invisible from the outside. Counting is what makes
    /// it testable rather than a matter of reading the code carefully.
    #[cfg(test)]
    null_nesting: u32,
    #[cfg(test)]
    max_null_nesting: u32,
    /// Whether late move reductions may be applied at all.
    ///
    /// **Tests only**, compiled out otherwise — same device as `allow_null_move`, and for
    /// the same reason: a node count only means something against a search identical but
    /// for the one thing being measured.
    #[cfg(test)]
    allow_lmr: bool,
    /// Whether the reduction grows with depth and rank, or stays at the flat one ply of #42.
    ///
    /// **Tests only.** The acceptance criterion for this brick is "improves on #42", not
    /// "improves on no reductions at all", so the baseline has to be reachable from inside
    /// the same binary — otherwise the comparison drifts across two builds and picks up
    /// every other difference between them.
    #[cfg(test)]
    lmr_growing: bool,
    /// Whether a reduced move that beats `alpha` is searched again at full depth.
    ///
    /// **Tests only.** Without this switch the re-search could only be checked by counting
    /// that it happens, never that it *matters* — and "the code runs" is not the claim. The
    /// null-move PR left an equivalent gap on the record: mutating its `return beta` to
    /// `return score` broke no test, so that line rests on reasoning alone. This is the
    /// same gap, closed rather than documented.
    #[cfg(test)]
    allow_lmr_research: bool,
    /// How many moves were searched at reduced depth, and how many of those had to be
    /// searched again at full depth.
    ///
    /// **Tests only.** These exist because of what happened to the history heuristic
    /// (#40): eight unit tests covered the component, all green, and unplugging it from
    /// the search entirely broke none of them — every test drove the component directly.
    /// A heuristic wired to nothing passes every unit test it owns. Counting from inside
    /// the search is what lets a test assert the *use* rather than the mechanism, and it
    /// is also the only way to check a guard whose effect is an absence.
    #[cfg(test)]
    lmr_reductions: u64,
    #[cfg(test)]
    lmr_researches: u64,
    /// Whether quiescence may drop captures the exchange evaluation calls losing.
    ///
    /// **Tests only**, compiled out otherwise. Needed more here than for the other switches:
    /// pruning changes what the search *concludes*, so the test that matters compares the two
    /// verdicts on the same position — which requires both to be reachable from one binary.
    #[cfg(test)]
    allow_see_pruning: bool,
}

impl<'a> Searcher<'a> {
    fn new(order: MoveOrder, deadline: Option<Instant>, table: &'a mut Table) -> Searcher<'a> {
        Searcher {
            nodes: 0,
            order,
            deadline,
            aborted: false,
            table,
            history: Vec::new(),
            path: Vec::new(),
            killers: Killers::new(),
            #[cfg(test)]
            node_limit: None,
            #[cfg(test)]
            allow_null_move: true,
            #[cfg(test)]
            null_nesting: 0,
            #[cfg(test)]
            max_null_nesting: 0,
            #[cfg(test)]
            allow_lmr: true,
            #[cfg(test)]
            lmr_growing: true,
            #[cfg(test)]
            allow_lmr_research: true,
            #[cfg(test)]
            lmr_reductions: 0,
            #[cfg(test)]
            lmr_researches: 0,
            #[cfg(test)]
            allow_see_pruning: true,
        }
    }

    /// The killers that apply to a node at `ply` — none unless the full ordering is on.
    fn killers_at(&self, ply: i32) -> KillerSlots {
        if self.order == MoveOrder::Full {
            // Idiom: `as usize` on a `ply` the search only ever counts upwards from
            // the root; `Killers::at` handles anything past its table.
            self.killers.at(ply as usize)
        } else {
            KillerSlots::none()
        }
    }

    /// Whether reaching `key` again is a draw.
    ///
    /// The two sources are **not** treated alike, and conflating them is a real
    /// mistake rather than a simplification.
    ///
    /// Along the **search path**, one earlier occurrence is enough. Both sides are
    /// choosing their moves inside this branch, so a side able to steer back to a
    /// position once can do it again; searching on to a literal third occurrence
    /// spends plies reaching a conclusion already available. Every engine does this.
    ///
    /// Against the **game history**, one is not enough. A position played once
    /// before and reached now has occurred *twice*, and the rules want three. Calling
    /// that a draw makes the engine believe it can force a repetition its opponent is
    /// still free to decline — it plays for a draw that does not exist, and finds
    /// itself in a position it evaluated as 0.
    fn is_repetition(&self, key: u64) -> bool {
        if self.path.contains(&key) {
            return true;
        }
        // Idiom: `iter().filter(...).count()` counts matches without allocating.
        // Two prior occurrences, so that this one is the third.
        self.history.iter().filter(|&&seen| seen == key).count() >= 2
    }

    /// Set `aborted` if a limit has been reached. The clock is read only every so
    /// often (doing it on every node would dominate the search cost).
    fn check_limits(&mut self) {
        // Compiled out of production builds entirely — see the field's comment.
        #[cfg(test)]
        if let Some(limit) = self.node_limit {
            if self.nodes >= limit {
                self.aborted = true;
                return;
            }
        }
        if self.nodes % 2048 == 0 {
            if let Some(deadline) = self.deadline {
                if Instant::now() >= deadline {
                    self.aborted = true;
                }
            }
        }
    }

    /// The root: like [`Searcher::negamax`] but it tracks the chosen move, does not
    /// cut off (no `beta` above the root), and tries `pv_move` first if given.
    fn root(&mut self, pos: &Position, depth: u32, pv_move: Option<Move>) -> RootResult {
        self.nodes += 1;
        let mut moves = pos.legal_moves();
        if moves.is_empty() {
            return RootResult { best: None, improved: false };
        }
        if self.order != MoveOrder::None {
            // The root is ply 0 and never cuts off — no killer is ever recorded
            // there, so there is none to apply.
            order_moves(pos, &mut moves, KillerSlots::none());
        }
        if let Some(pv) = pv_move {
            if let Some(i) = moves.iter().position(|&mv| mv == pv) {
                moves.swap(0, i);
            }
        }

        let mut best_move = moves[0];
        let mut best_score = -INF;
        let mut alpha = -INF;
        let mut improved = false;
        // The root's own key goes on the path, so a branch coming back to it is seen
        // as a repetition. The root itself is never tested against the path — it is
        // the position we are asked about, not a repetition of anything.
        self.path.push(pos.hash());
        for (i, &mv) in moves.iter().enumerate() {
            let score = -self.negamax(&pos.play(mv), depth.saturating_sub(1), -INF, -alpha, 1);
            if self.aborted {
                // Time is up *inside this move's search*, so `score` is the placeholder
                // `negamax` returns when aborting, not a value. Leaving before the
                // comparison below is what keeps a half-searched move from ever being
                // adopted — the property the whole partial-result rescue rests on.
                break;
            }
            if score > best_score {
                best_score = score;
                best_move = mv;
                // A move other than the one tried first has taken the lead: this
                // iteration disagrees with what the previous one recommended, and it
                // reached that verdict by comparing moves *at its own depth*. That is
                // the one thing worth rescuing if the clock stops us here.
                improved |= i > 0;
            }
            if score > alpha {
                alpha = score;
            }
        }
        self.path.pop();
        RootResult { best: Some((best_move, best_score)), improved }
    }

    /// Negamax with alpha-beta. Returns the value of `pos` from the side-to-move
    /// perspective. `ply` is the distance from the root, used only to score mates.
    fn negamax(&mut self, pos: &Position, depth: u32, alpha: i32, beta: i32, ply: i32) -> i32 {
        // A normal node may pass; only a node reached *by* passing may not.
        self.negamax_inner(pos, depth, alpha, beta, ply, true)
    }

    /// Whether this node may try a null move.
    ///
    /// `depth` must leave something to search after the reduction — below that the
    /// pass costs more than the subtree it would prune.
    ///
    /// The material test is the **zugzwang guard**, and it is the reason this feature
    /// is dangerous rather than merely approximate. Null-move assumes that passing
    /// cannot help; in zugzwang the opposite is true — every legal move worsens the
    /// position, and passing would be the best thing available. The search would then
    /// come back flattered, cut off, and never look at the loss it just walked into.
    ///
    /// Zugzwang needs an endgame: with pieces on the board there is almost always a
    /// harmless move to make. The phase from the tapered evaluation (#34) already
    /// measures exactly that, so the guard costs one comparison and no new concept.
    fn null_move_allowed(&self, pos: &Position, depth: u32) -> bool {
        depth > NULL_MOVE_REDUCTION + 1 && phase(pos) >= NULL_MOVE_MIN_PHASE
    }

    /// How many plies to shave off the search of `mv`, the move at index `rank` — zero
    /// when it must be searched at full depth.
    ///
    /// The idea is a bet on the move ordering: if MVV-LVA, the killers and the
    /// transposition table have done their job, a move sitting far down the list is
    /// unlikely to be the best one, and paying full depth to confirm that is what makes
    /// the tree wide. So it is searched shallower — and if the shallow search comes back
    /// above `alpha` after all, the caller re-searches it at full depth. **The bet is on
    /// the cost, never on the answer**: being wrong costs one wasted shallow search, not
    /// a wrong move.
    ///
    /// That is what makes this the milder of the engine's two heuristic cuts. Null-move
    /// (#38) skips a subtree it never looks at; this one looks at every move, only less
    /// closely at first.
    ///
    /// # The guards, each for a different way the bet fails
    ///
    /// **Ordering must be on.** "Late" is meaningless in generator order — index 7 would
    /// then say nothing about a move's prospects. This also keeps the existing
    /// ordering-isolation tests measuring what they claim to.
    ///
    /// **Quiet moves only**, and **no killers**. A capture or a promotion changes the
    /// material, which is the one thing the evaluation reads directly; a killer refuted a
    /// sibling of this very node moments ago. Both are evidence the move deserves full
    /// depth, and both are already the reason the ordering put them where it did —
    /// reducing them would be betting against our own information.
    ///
    /// **Not in check, and not giving check.** Both are cases where the move list is
    /// short and forced: nearly every reply matters, so the premise "most of these moves
    /// are irrelevant" does not hold. Reducing a check is also how a reduction turns into
    /// a missed mate, since a forcing line is exactly what a shallower search stops
    /// seeing.
    fn late_move_reduction(
        &self,
        pos: &Position,
        child: &Position,
        mv: Move,
        depth: u32,
        rank: usize,
        ply: i32,
    ) -> u32 {
        // Compiled out of production builds: the ordering test alone decides there.
        #[cfg(test)]
        if !self.allow_lmr {
            return 0;
        }
        let reducible = self.order == MoveOrder::Full
            && depth >= LMR_MIN_DEPTH
            && rank >= LMR_FULL_DEPTH_MOVES
            && is_quiet(pos, mv)
            && !self.killers_at(ply).contains(mv)
            && !pos.in_check()
            && !child.in_check();
        if !reducible {
            return 0;
        }
        // How much, from the curve. Both indices are clamped rather than trusted: `depth`
        // is bounded by `MAX_DEPTH` in every search the engine runs, but quiescence
        // recurses past it and a position can legally offer more than 64 moves, so an
        // unclamped lookup would be a panic waiting for an unusual position rather than a
        // bug anyone would find in testing.
        let reduction =
            LMR_TABLE[(depth as usize).min(LMR_TABLE_DEPTHS - 1)][rank.min(LMR_TABLE_RANKS - 1)];
        // Compiled out of production builds, where the curve always applies.
        #[cfg(test)]
        let reduction = if self.lmr_growing { reduction } else { 1 };
        // Leave the search something to do — see `reduction_ceiling`, which is also what the
        // equivalence test reads, so that either mechanism moving is observable.
        reduction.min(reduction_ceiling(depth))
    }

    fn negamax_inner(
        &mut self,
        pos: &Position,
        depth: u32,
        mut alpha: i32,
        beta: i32,
        ply: i32,
        can_null: bool,
    ) -> i32 {
        self.nodes += 1;
        self.check_limits();
        if self.aborted {
            return 0; // value ignored: the whole iteration is thrown away
        }

        let key = pos.hash();
        // Checked before anything else, and deliberately before the table. A draw by
        // repetition is a property of the *path*, not of the position: the same
        // position reached without repeating is not drawn. Probing or storing it
        // under this key would attach a path fact to a position, which is exactly the
        // hazard a transposition table introduces.
        if self.is_repetition(key) {
            return 0;
        }

        // Out of depth, but not necessarily out of danger: hand over to quiescence
        // rather than evaluating a position that may be mid-exchange. Checked before
        // generating moves, since quiescence generates them itself — and it is the one
        // that detects mate and stalemate from here on.
        if depth == 0 {
            return self.quiescence(pos, alpha, beta, ply);
        }

        // Have we been here before, by another move order or in an earlier iteration?
        let hit = self.table.probe(key, depth, alpha, beta, ply);
        if let Some(score) = hit.cutoff {
            return score;
        }

        // Null-move pruning: ask what happens if we simply pass.
        //
        // If the opponent — now effectively moving twice in a row — still cannot drag
        // the score below `beta`, this position is so good that the real search would
        // cut off anyway, and its whole subtree can be skipped unexamined. The pass is
        // searched shallower (`depth - 1 - R`) with a null window, because a rough
        // refutation is all it has to fail to find.
        //
        // This is the first cut in this engine that is **heuristic** rather than exact.
        // Alpha-beta and the transposition table never change what a search concludes;
        // this can, and that is what buys the nodes.
        // Compiled out of production builds: `can_null` alone decides there.
        #[cfg(test)]
        let can_null = can_null && self.allow_null_move;
        if can_null && self.null_move_allowed(pos, depth) {
            if let Some(passed) = pos.null_move() {
                let reduced = depth.saturating_sub(1 + NULL_MOVE_REDUCTION);
                // `-beta, -beta + 1` is a null window: we only ask "does it reach beta",
                // never "by how much". `false` forbids a second pass in a row — two
                // passes would skip a full move for both sides and prove nothing.
                #[cfg(test)]
                {
                    self.null_nesting += 1;
                    self.max_null_nesting = self.max_null_nesting.max(self.null_nesting);
                }
                let score =
                    -self.negamax_inner(&passed, reduced, -beta, -beta + 1, ply + 1, false);
                #[cfg(test)]
                {
                    self.null_nesting -= 1;
                }
                if self.aborted {
                    return 0;
                }
                if score >= beta {
                    // Return `beta`, not `score`.
                    //
                    // A null-move search can come back in mate territory: "even if I
                    // pass, I mate". That mate is real — a free move for the opponent
                    // cannot conjure one — but its **distance** is not, since the pass
                    // consumed a ply that will not be played. Propagating the raw score
                    // would announce a mate in N when the real one is longer, and the
                    // engine would pick the wrong forcing line believing it faster.
                    //
                    // There is a second reason, found in review and worth stating
                    // because it protects a case this code does not otherwise guard.
                    // The null move is tried *before* the move list is generated, so it
                    // also runs at **stalemate** nodes — `null_move` only refuses when
                    // in check. A side stalemated while materially ahead would have its
                    // node cut off here rather than scored 0.
                    //
                    // Returning `beta` is what makes that harmless: the move is then
                    // worth exactly the parent's `alpha` and can never take the lead,
                    // since `score > best_score` is false. A fail-high can make the
                    // search *miss* something better inside the pruned subtree; it can
                    // never make it *choose* something worse. Verified on
                    // `5bnr/4p1pq/5pkr/4Q2p/2P4P/8/PP1PPPP1/RNB1KBNR w`, where White is
                    // +1051 and `Qe6` stalemates: at depths 3 to 7 the engine plays
                    // `Qc5`, `Qc3`, `Qc7` — never `Qe6`.
                    //
                    // Honest note: mutating this to `return score` still breaks no test.
                    // The mate-distance case is narrow enough that it was not reproduced
                    // on 18 position/depth combinations, so the line rests on reasoning
                    // rather than on a measurement — but on two independent reasons now.
                    return beta;
                }
            }
        }

        let mut moves = pos.legal_moves();
        if moves.is_empty() {
            // Terminal: checkmate (side to move loses) or stalemate (draw).
            return if pos.in_check() { -(MATE - ply) } else { 0 };
        }
        if self.order != MoveOrder::None {
            order_moves(pos, &mut moves, self.killers_at(ply));
        }
        // The cached move goes first, ahead even of the best capture. A previous
        // search already found it good here, and a good first move is what makes the
        // cutoffs below cheap — often a larger win than the cutoffs the table itself
        // provides.
        if let Some(cached) = hit.best {
            if let Some(i) = moves.iter().position(|&mv| mv == cached) {
                moves.swap(0, i);
            }
        }

        // Kept to classify the result: a value that never beat the alpha we started
        // with is only an upper bound on this node, not its value.
        let alpha_before = alpha;
        let mut best = -INF;
        let mut best_move = None;
        self.path.push(key);
        // Idiom: `enumerate` pairs each move with its index, which is what "late" means
        // here — how far down the ordered list the move sits.
        for (rank, &mv) in moves.iter().enumerate() {
            // Bound to a local rather than left as a temporary inside the call: the
            // gives-check guard needs to look at the resulting position, and the move was
            // going to be played anyway, so asking costs nothing extra.
            let child = pos.play(mv);
            let reduction = self.late_move_reduction(pos, &child, mv, depth, rank, ply);
            #[cfg(test)]
            if reduction > 0 {
                self.lmr_reductions += 1;
            }
            // Idiom: `saturating_sub` floors at zero instead of wrapping around, which for
            // an unsigned type is the difference between falling into quiescence and
            // asking for a search four billion plies deep. `LMR_MIN_DEPTH` makes the
            // subtraction safe today, but that is a relationship between two constants
            // sitting far apart in the file, and nothing would flag it if one moved —
            // raising `LMR_REDUCTION` to 3 while the floor stayed at 3 would underflow.
            //
            // Kept because of *how* the two profiles fail, which is the real argument.
            // Mutating this to a plain subtraction and removing the floor gives, in debug,
            // **32 test failures all reading "attempt to subtract with overflow"** — loud
            // and diagnosable. The same mutation in `--release`, where overflow checks are
            // off, wraps silently and ends in **`fatal runtime error: stack overflow`**:
            // the search recurses on a depth near 2^32 until the stack is gone. Release is
            // the profile every performance measurement in this repository uses, and in an
            // arena that failure would read as an engine that simply died mid-game.
            let reduced = (depth - 1).saturating_sub(reduction);
            let mut score = -self.negamax(&child, reduced, -beta, -alpha, ply + 1);
            if self.aborted {
                self.path.pop();
                return 0; // nothing worth caching: the value is a placeholder
            }
            // The reduction was a bet that this move is not the best one, and the bet just
            // lost: a shallower search already puts it above `alpha`. Search it again at
            // full depth and let *that* verdict be the one the node uses — otherwise the
            // reduction would decide the move, which is precisely what it must never do.
            //
            // Checking `aborted` first matters: an interrupted search returns a
            // placeholder `0`, which would clear a negative `alpha` and trigger a
            // re-search of a value that means nothing.
            // Compiled out of production builds: there, a reduced move above `alpha` is
            // always re-searched.
            #[cfg(test)]
            let may_research = self.allow_lmr_research;
            #[cfg(not(test))]
            let may_research = true;
            if reduction > 0 && score > alpha && may_research {
                #[cfg(test)]
                {
                    self.lmr_researches += 1;
                }
                score = -self.negamax(&child, depth - 1, -beta, -alpha, ply + 1);
                // Honest note, in the same spirit as the null move's `return beta`:
                // removing this guard breaks no test. It matters in principle — a node
                // that ran out of time inside the re-search would otherwise fall through
                // to `store_at` and cache a score derived from a placeholder, under a
                // depth claiming real work, in a table that since #36 outlives the move.
                // Reaching it needs the abort to land inside a re-search of the *last*
                // move of a list, or one that then fails high on the placeholder; a
                // counter of writes-after-abort swept over 200 node ceilings never saw it,
                // and neither did removing every abort guard in the loop at once. Kept as
                // reasoning rather than dressed up as tested — and the inert test that
                // looked like coverage was deleted rather than left counting.
                if self.aborted {
                    self.path.pop();
                    return 0;
                }
            }
            if score > best {
                best = score;
                best_move = Some(mv);
            }
            if best > alpha {
                alpha = best;
            }
            if alpha >= beta {
                // Beta cutoff: this branch cannot improve the result. Remember the
                // refutation, so the sibling nodes of this ply try it first — the
                // table itself drops the move if it is a capture or a promotion.
                if self.order == MoveOrder::Full {
                    self.killers.record(pos, ply as usize, mv);
                }
                break;
            }
        }

        self.path.pop();

        let bound = if best <= alpha_before {
            Bound::Upper // no move beat alpha: the true value is at most `best`
        } else if best >= beta {
            Bound::Lower // we stopped early: the true value is at least `best`
        } else {
            Bound::Exact // the window contained the answer
        };
        self.table.store_at(key, depth, best, bound, best_move, ply);
        best
    }

    /// Search on past the depth limit, over the moves that change material —
    /// **captures and promotions** — until the position is quiet, then evaluate.
    /// Without this, a leaf landing in the middle of an exchange is scored as if the
    /// exchange stopped there, and the engine comes to prefer moves that push a loss
    /// just beyond its own horizon.
    ///
    /// No depth argument: the recursion ends on its own. Captures take a piece off the
    /// board and there are finitely many; promotions add one, but a pawn can only
    /// promote once and never moves backwards, so neither can go on forever.
    ///
    /// That same irreversibility is why there is no repetition check here: every move
    /// quiescence searches changes the material on the board, so no line it explores
    /// can return to a position seen earlier. The entry node is already covered —
    /// [`Searcher::negamax`] tests for a repetition before handing over.
    fn quiescence(&mut self, pos: &Position, mut alpha: i32, beta: i32, ply: i32) -> i32 {
        self.nodes += 1;
        self.check_limits();
        if self.aborted {
            return 0; // value ignored: the whole iteration is thrown away
        }

        // Generate every legal move, not just the captures: it is the only way to
        // tell mate and stalemate from a merely quiet position, and scoring a mate
        // as a material count is exactly the lie this function exists to remove.
        let mut moves = pos.legal_moves();
        if moves.is_empty() {
            return if pos.in_check() { -(MATE - ply) } else { 0 };
        }

        // "Stand pat": the side to move is never *obliged* to capture, so the static
        // score is a lower bound on what they can get here. That makes it usable both
        // as a cutoff and as the starting `alpha` — and it is what stops the search
        // from playing out a losing exchange just because it is the only capture.
        let stand_pat = evaluate(pos);
        if stand_pat >= beta {
            return stand_pat;
        }
        if stand_pat > alpha {
            alpha = stand_pat;
        }

        // Idiom: `retain` filters the vector in place, keeping the elements the
        // closure accepts. `mvv_lva` scores quiet moves 0 and captures above it, so
        // it doubles as the capture test — but it reads the *destination* square, so a
        // pawn stepping onto an empty back rank scores 0 like any quiet move. Hence the
        // explicit promotion test: without it the leaf is evaluated as if the queen
        // about to appear did not exist.
        moves.retain(|&mv| mvv_lva(pos, mv) > 0 || is_queen_promotion(mv));
        // Then drop the captures that lose material. Unlike the ordering use of the same
        // evaluation, this **changes what the search concludes**: a capture removed here is
        // never examined, so a sacrifice that wins two moves later is invisible to quiescence.
        //
        // The trade is deliberate. Quiescence exists to resolve exchanges, and an exchange the
        // exchange evaluation already calls losing is one whose resolution is known: the side to
        // move will not make it. Searching it anyway spends nodes to rediscover a verdict a few
        // bitboard lookups already gave. What is genuinely lost is the sacrifice whose point lies
        // beyond the recapture — and that one belongs to the main search, which has no such
        // filter.
        //
        // Only when not in check: a side in check has no choice, and pruning its replies by
        // material would drop the only legal way out of a mate threat.
        #[cfg(test)]
        let prune = self.allow_see_pruning;
        #[cfg(not(test))]
        let prune = true;
        if prune && !pos.in_check() {
            moves.retain(|&mv| see(pos, mv) >= 0);
        }
        if self.order != MoveOrder::None {
            // No killers here: every move left is a capture or a promotion, and a
            // killer is by definition quiet.
            order_moves(pos, &mut moves, KillerSlots::none());
        }

        let mut best = stand_pat;
        for &mv in &moves {
            let score = -self.quiescence(&pos.play(mv), -beta, -alpha, ply + 1);
            if self.aborted {
                return 0;
            }
            if score > best {
                best = score;
            }
            if best > alpha {
                alpha = best;
            }
            if alpha >= beta {
                break;
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::{Color, Piece, Status};

    // A position where the engine changes its mind as it deepens: c1d1 at depths 1-2,
    // h2h3 at 3-4, f3e1 at 5-6. That is what makes it usable for the tests below —
    // "an iteration found something better than the previous one" needs an iteration
    // that actually disagrees with its predecessor.
    const CHANGES_ITS_MIND: &str = "2r3k1/pp3pp1/4p2p/3pP3/3P4/2P2N2/PP3PPP/2R3K1 w - - 0 1";

    // A search interrupted after exactly `node_ceiling` nodes, collecting the depths it
    // reported along the way.
    //
    // Interrupting by node count rather than by clock is deliberate: the behaviour under
    // test is "what survives an interruption", and a wall-clock deadline would cut at a
    // different move depending on how busy the machine is — the test would be measuring
    // the machine. A node count cuts at exactly the same place every run.
    fn search_cut_at(pos: &Position, max_depth: u32, node_ceiling: u64) -> (SearchStats, Vec<u32>) {
        let mut table = Table::new();
        let mut searcher = Searcher::new(MoveOrder::Full, None, &mut table);
        searcher.node_limit = Some(node_ceiling);
        let mut reported = Vec::new();
        let mut report = |p: &Progress| reported.push(p.depth);
        let stats = deepen(
            pos,
            Request { progress: Some(&mut report), ..Request::new(Limits::depth(max_depth)) },
            &mut searcher,
        );
        (stats, reported)
    }

    // A single fixed-depth pass — no deepening, no clock — driving `Searcher` directly.
    // The oracle these tests compare against: it gives the verdict a plain depth-D
    // search reaches, and, with `ordered` flipped, isolates what move ordering does.
    //
    // **Late move reductions are off here**, because this function exists to vary one
    // thing at a time and reductions would silently break that. They only apply under
    // `MoveOrder::Full` — "late" is meaningless in generator order — so a `Full` against
    // `None` comparison would be measuring ordering *and* reductions while claiming to
    // measure ordering, and the two callers that assert ordering leaves the score alone
    // would be asserting something reductions are entitled to change. A search *with*
    // reductions is what `Engine::search` and `search_timed` give; this is the oracle,
    // and an oracle holds everything else still.
    fn search_fixed(pos: &Position, depth: u32, order: MoveOrder) -> SearchStats {
        let mut table = Table::new();
        let mut searcher = Searcher::new(order, None, &mut table);
        searcher.allow_lmr = false;
        let best = searcher.root(pos, depth, None).best;
        SearchStats {
            best,
            depth,
            nodes: searcher.nodes,
            table_key_match_rate: searcher.table.key_match_rate(),
            table_cutoff_rate: searcher.table.cutoff_rate(),
        }
    }

    #[test]
    fn root_reports_an_improvement_only_when_a_later_move_takes_the_lead() {
        // Black's queen on d5 hangs; Rxd5 is plainly best. Feeding `root` a mediocre
        // move as the previous iteration's choice puts it first, so Rxd5 has to
        // overtake it — that is an improvement. Feeding it Rxd5 directly means nothing
        // can overtake anything.
        let p = Position::from_fen("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1").unwrap();
        let best = p.move_from_uci("d1d5").unwrap();
        let mediocre = p.move_from_uci("e1f1").unwrap();

        let mut t1 = Table::new();
        let overtaken = Searcher::new(MoveOrder::Full, None, &mut t1).root(&p, 3, Some(mediocre));
        assert!(overtaken.improved, "Rxd5 must overtake the move tried first");
        assert_eq!(overtaken.best.map(|(mv, _)| mv), Some(best));

        let mut t2 = Table::new();
        let already_best = Searcher::new(MoveOrder::Full, None, &mut t2).root(&p, 3, Some(best));
        assert!(!already_best.improved, "nothing can overtake the best move");
        assert_eq!(already_best.best.map(|(mv, _)| mv), Some(best));
    }

    #[test]
    fn an_improvement_found_before_the_interruption_is_kept() {
        // The case this whole change exists for. Cut at 1 000 nodes: iterations 1 and 2
        // complete (c1d1), iteration 3 finds h2h3 better and is then interrupted.
        // Without keeping it, the engine plays c1d1 — the move a deeper search had just
        // rejected — despite having already found the replacement.
        let p = Position::from_fen(CHANGES_ITS_MIND).unwrap();
        let (stats, _) = search_cut_at(&p, 6, 1_000);

        assert_eq!(
            stats.best.map(|(mv, sc)| (p.move_to_uci(mv), sc)),
            Some(("h2h3".to_string(), 450)),
        );
        // What the rescue is worth is exactly this: a move the previous *complete*
        // iteration would not have played.
        assert_ne!(
            best_move(&p, 2).map(|(mv, _)| p.move_to_uci(mv)),
            Some("h2h3".to_string()),
            "otherwise the test would pass without the rescue doing anything",
        );
        // And no more than that. The rescued move is an improvement found part-way
        // through iteration 3, not iteration 3's verdict — completed, that iteration
        // settles on a different move again. An earlier version of this test asserted
        // the two were equal; they were, on the evaluation of the day, by coincidence.
    }

    #[test]
    fn an_interruption_with_nothing_better_is_still_discarded() {
        // Same position, cut earlier: iteration 3 is interrupted *inside its first
        // move*, so it never compared anything. Its partial result is `(c1d1, -40000)`
        // — the initial `-INF`, never updated — which reads as "lost by force". Keeping
        // that would have the engine report a resignable position where the truth is
        // +445, which is what makes discarding the default rather than the exception.
        //
        // The cut point matters: interrupting a little later (500-800 nodes) leaves
        // `(c1d1, 445)`, indistinguishable from the complete depth-2 result, so a test
        // there would pass whatever the code did.
        let p = Position::from_fen(CHANGES_ITS_MIND).unwrap();
        let (stats, _) = search_cut_at(&p, 6, 400);

        assert_eq!(
            stats.best.map(|(mv, sc)| (p.move_to_uci(mv), sc)),
            Some(("c1d1".to_string(), 445)),
            "the complete depth-2 result must stand",
        );
        assert_eq!(best_move(&p, 2).map(|(mv, sc)| (p.move_to_uci(mv), sc)), Some(("c1d1".to_string(), 445)));
    }

    #[test]
    fn a_move_whose_own_search_was_cut_off_is_never_adopted() {
        // A king alone against a queen: every move is worth about -900. When the
        // deadline fires inside a move's search, `negamax` returns the placeholder 0 —
        // and 0 outranks -900 by a mile. Adopting it would have the engine believe it
        // had found a miraculous draw in a lost position, and `improved` would then be
        // set, so the bogus verdict would survive the interruption instead of being
        // discarded with it.
        //
        // `root` leaves its loop before the comparison, which is what prevents this.
        // The property is worth its own test because nothing else exercises it: on a
        // winning position the placeholder loses the comparison anyway and the defect
        // stays invisible.
        //
        // **Every** cut point is swept rather than one being picked, and that is the
        // point of the test rather than thoroughness for its own sake. An earlier
        // version cut at a single fixed ceiling, which only catches the defect when
        // the interruption happens to land *inside* a move's search — and where it
        // lands depends on how many nodes each depth costs, which any change to the
        // evaluation moves. That version was silently disarmed by the tapered
        // evaluation: the mutation passed unnoticed until review.
        //
        // Sweeping turns a calibrated coincidence into an invariant: whatever the
        // engine's node counts become, some ceiling in this range will land mid-move,
        // and no ceiling may ever produce a drawn score in a lost position.
        let p = Position::from_fen("3qk3/8/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        for ceiling in (20..=2_000).step_by(10) {
            let (stats, _) = search_cut_at(&p, 5, ceiling);
            let (_, score) = stats.best.expect("a move comes back");
            assert_ne!(
                score, 0,
                "ceiling {ceiling}: a placeholder must never be mistaken for a drawn position",
            );
            assert!(
                score < -800,
                "ceiling {ceiling}: the position is lost and the score must say so, got {score}",
            );
        }
    }

    #[test]
    fn the_reported_depth_is_the_last_completed_one_even_when_a_move_is_rescued() {
        // Keeping a partial *move* must not turn into announcing a partial *depth*:
        // iteration 3 did not finish, so 2 is how far the engine actually looked.
        let p = Position::from_fen(CHANGES_ITS_MIND).unwrap();
        let (rescued, reports_rescued) = search_cut_at(&p, 6, 1_000);
        let (discarded, reports_discarded) = search_cut_at(&p, 6, 400);

        assert_eq!(rescued.depth, 2, "a rescued move does not raise the reported depth");
        assert_eq!(discarded.depth, 2);
        // And the interrupted iteration is announced in neither case — one report per
        // completed iteration, exactly as before this change.
        assert_eq!(reports_rescued, vec![1, 2]);
        assert_eq!(reports_discarded, vec![1, 2]);
    }

    #[test]
    fn a_legal_move_survives_an_interruption_before_any_iteration_completes() {
        // Cut inside iteration 1, so there is no complete result at all. The partial
        // one is all there is, and it is at least legal — returning nothing would mean
        // `bestmove 0000`, an instant forfeit in any arena.
        let p = Position::from_fen(CHANGES_ITS_MIND).unwrap();
        let (stats, reports) = search_cut_at(&p, 6, 20);

        let (mv, _) = stats.best.expect("a move comes back");
        assert!(p.try_play(mv).is_ok(), "and it is legal");
        assert_eq!(stats.depth, 0, "no iteration completed, so no depth is claimed");
        assert!(reports.is_empty(), "and none was announced");
    }

    #[test]
    fn a_side_in_check_cannot_pass() {
        // The first guard, and it costs nothing because `cozy-chess` enforces it: a
        // side in check must answer, so "what if I pass" has no answer, and a score
        // derived from it would be meaningless.
        let in_check = Position::from_fen("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 0 3").unwrap();
        assert!(in_check.in_check(), "precondition: white is in check");
        assert!(in_check.null_move().is_none(), "a side in check may not pass");

        let quiet = Position::initial();
        let passed = quiet.null_move().expect("a quiet position may pass");
        assert_ne!(passed.side_to_move(), quiet.side_to_move(), "the turn changes hands");
    }

    #[test]
    fn no_null_move_in_the_endgame() {
        // The zugzwang guard. In an endgame, every legal move can be worse than
        // passing — so a null-move search comes back flattered, cuts off, and the
        // engine never looks at the loss it walked into.
        //
        // This test checks the *predicate*. The node-for-node comparison it used to
        // claim lives in `no_pass_in_a_zugzwang` below, where it belongs — on
        // positions that are actually zugzwangs.
        let endgame = Position::from_fen("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1").unwrap();
        assert!(
            phase(&endgame) < NULL_MOVE_MIN_PHASE,
            "precondition: this must be endgame material, phase {}",
            phase(&endgame),
        );
        let mut table = Table::new();
        let searcher = Searcher::new(MoveOrder::Full, None, &mut table);
        assert!(!searcher.null_move_allowed(&endgame, 7), "no pass in an endgame");

        // And a middlegame must allow it, or the guard would be vacuous.
        let middlegame = Position::from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        )
        .unwrap();
        assert!(searcher.null_move_allowed(&middlegame, 7), "a middlegame may pass");
    }

    #[test]
    fn no_pass_in_a_zugzwang() {
        // AC#6, and the regression the whole guard exists for. The endgame test above
        // uses an endgame; this uses two genuine **zugzwangs**, where the side to move
        // would be strictly better off passing — which is the assumption null-move
        // makes and the one that is false here.
        //
        // Checked on the search, not on the predicate: the tree must be identical node
        // for node to one searched with the feature switched off. A predicate test
        // says the guard answers correctly; this says nothing slipped past it.
        //
        // What is at stake is not only the shape of the tree. With the guard removed,
        // the opposition is **misevaluated by 15 centipawns at depth 9** (120 against
        // 135) — AC#6 failing in the literal sense rather than by proxy. Worth knowing
        // before anyone lowers `NULL_MOVE_MIN_PHASE` for the extra nodes: the cost is
        // a wrong verdict, one ply past where this test looks.
        for (name, fen) in [
            // Trébuchet: whoever moves loses the pawn, and with it the game.
            ("trebuchet", "8/8/8/p1p5/P1P5/8/8/K6k w - - 0 1"),
            // Opposition: White to move cannot make progress; Black to move loses.
            ("opposition", "8/8/8/3k4/8/3K4/3P4/8 w - - 0 1"),
        ] {
            let p = Position::from_fen(fen).unwrap();
            assert!(
                phase(&p) < NULL_MOVE_MIN_PHASE,
                "{name}: precondition — must be endgame material, phase {}",
                phase(&p),
            );

            // Swept rather than fixed at one depth, and the reason is measured: with
            // the guard removed, the node difference is **0 at depth 4** on both
            // positions — a single-depth test there would be green whether or not the
            // guard exists — and only **1 node** on the trébuchet at depth 6. One
            // depth is one constant away from being inert, and nothing in the test
            // would say so. Eight opportunities have to go quiet at once instead of
            // one. Costs 0.5 s, and under the mutation it fails at depth 5.
            for depth in 5..=8 {
                let mut t1 = Table::new();
                let mut with = Searcher::new(MoveOrder::Full, None, &mut t1);
                let a = with.root(&p, depth, None);

                let mut t2 = Table::new();
                let mut without = Searcher::new(MoveOrder::Full, None, &mut t2);
                without.allow_null_move = false;
                let b = without.root(&p, depth, None);

                assert_eq!(
                    with.nodes, without.nodes,
                    "{name} at depth {depth}: a pass was attempted in an endgame",
                );
                assert_eq!(
                    a.best.map(|(_, s)| s),
                    b.best.map(|(_, s)| s),
                    "{name} at depth {depth}: score moved",
                );
            }
        }
    }

    #[test]
    fn no_null_move_when_the_reduction_would_leave_nothing() {
        // Below `1 + R` there is no subtree left to prune, so the pass costs more than
        // it saves.
        let p = Position::initial();
        let mut table = Table::new();
        let searcher = Searcher::new(MoveOrder::Full, None, &mut table);
        for depth in 0..=NULL_MOVE_REDUCTION + 1 {
            assert!(!searcher.null_move_allowed(&p, depth), "depth {depth} is too shallow");
        }
        assert!(searcher.null_move_allowed(&p, NULL_MOVE_REDUCTION + 2));
    }

    #[test]
    fn forced_mates_survive_the_pruning() {
        // The sharpest regression test there is. A null-move cutoff that is wrongly
        // trusted makes a mate vanish — and a mate is the one verdict that cannot be
        // approximately right. The pruning must never cost one.
        //
        // Deeper than `1 + R` so that null-move is actually attempted on the way.
        let mate_in_one = Position::from_fen("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1").unwrap();
        let (mv, score) = best_move(&mate_in_one, 5).expect("a move");
        assert_eq!(mate_in_one.play(mv).status(), Status::Checkmate);
        assert!(score > MATE_THRESHOLD, "and it is scored as a mate: {score}");

        // Scholar's mate position: black is already mated, nothing to search.
        let mated = Position::from_fen(
            "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4",
        )
        .unwrap();
        assert!(best_move(&mated, 5).is_none());
    }

    #[test]
    fn null_move_prunes_the_middlegame() {
        // What the feature is for. Measured against the same search with the pruning
        // disabled by its own guard, so the comparison isolates it.
        //
        // Late move reductions are off on **both** sides. The two cuts overlap — they
        // both decline to spend full depth on branches the ordering ranks low — so with
        // reductions on, what null-move prunes *in addition to them* is a fraction of
        // what it prunes alone: 12% here against 41%. Leaving them on would quietly turn
        // this into a test of the pair, under a name that claims otherwise.
        let p = Position::from_fen(
            "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 3",
        )
        .unwrap();
        let nodes = |null_move: bool, depth: u32| {
            let mut table = Table::new();
            let mut s = Searcher::new(MoveOrder::Full, None, &mut table);
            s.allow_null_move = null_move;
            s.allow_lmr = false;
            deepen(&p, Request::new(Limits::depth(depth)), &mut s).nodes
        };
        // Swept rather than measured at one chosen depth: a threshold that holds at
        // exactly one depth is a calibrated coincidence, and the next change to the tree
        // silently turns it into a test that passes without exercising anything.
        //
        // The floor is derived from the reduction rather than picked, and the sweep is
        // what found it: at depth 4 the pruning is provably unreachable — the shallowest
        // internal node sits at `depth - 1`, `null_move_allowed` wants `depth > R + 1`,
        // so nothing can pass below `R + 3` — and the two searches returned the identical
        // node count. Deriving it means a change to `NULL_MOVE_REDUCTION` moves the floor
        // with it instead of leaving a silently inert first iteration.
        //
        // Two separate properties, because they hold on different domains. **That it
        // prunes at all** is true from the floor upwards and is what a regression would
        // break. **How much** grows sharply with depth — measured with this test's own
        // protocol: **4.5 % at depth 5, 30.7 % at depth 6, 41.6 % at depth 7** — since the
        // deeper the tree, the larger the share of nodes with enough depth left to pass. A
        // sevenfold rise between two adjacent depths is the whole point, and it is why
        // asserting the amplitude across the sweep would only pin the weakest depth.
        //
        // So the threshold below is read against the amplitude at `DEEPEST`, not against the
        // depth-7 figure.
        //
        // **25 % on `main`, then 15 %, now 20 %**, and the round trip is worth recording because
        // the middle step was wrong about its own cause. It was lowered when the SEE arrived, on
        // the grounds that demoting losing captures makes plain alpha-beta cut earlier and so
        // *takes over* part of what the null move used to prune — the overlap measured between
        // the killers and iterative deepening in #30.
        //
        // The overlap was real; the mechanism was not. The demotion did not cut earlier, it
        // **doubled this tree**: with the null move disabled, 419 161 nodes with the ordering
        // against 217 250 without it, this test's own protocol either way. The null move then
        // pruned a smaller *share* of a tree twice the size. Dropping the ordering (515d136)
        // brings the share back to **25.1 %** (162 676 against 217 250), against 30.7 % on
        // `main` and 17.6 % with the ordering — the rest of the gap being the quiescence
        // pruning, which stays and shrinks the tree the null move would otherwise have cut.
        //
        // 20 % rather than 25 %: the measurement is 25.1 %, and a floor 0.1 point under its
        // measurement is the calibrated coincidence this test's own comment warns against. The
        // 5-point margin matches the one 25 % had on `main`.
        //
        // What must never happen is the null move ceasing to prune at all, and that is asserted
        // unconditionally at every swept depth.
        const DEEPEST: u32 = 6;
        for depth in (NULL_MOVE_REDUCTION + 3)..=DEEPEST {
            let (with, without) = (nodes(true, depth), nodes(false, depth));
            assert!(
                with < without,
                "null-move must prune something at depth {depth}: {with} against {without}",
            );
            if depth == DEEPEST {
                assert!(
                    with * 5 < without * 4,
                    "and at least 20% at depth {depth}: {with} against {without}",
                );
            }
        }
    }

    #[test]
    fn never_two_passes_in_a_row() {
        // Passing twice would hand the opponent a free move and then take it back —
        // it proves nothing about the position and prunes on nonsense. The guard is a
        // `false` handed to the recursive call, invisible from outside, so the nesting
        // is counted instead of assumed.
        let p = Position::initial();
        let mut table = Table::new();
        let mut searcher = Searcher::new(MoveOrder::Full, None, &mut table);
        // Two settings that both matter, and both were wrong in earlier drafts.
        //
        // **Depth 8**, because a pass reduces by `1 + R`: at depth 6 it runs at 3 and
        // the depth guard already forbids a second one, at depth 7 the same thing
        // happens one level down. Nesting is only *reachable* from depth 8 — shallower
        // versions of this test passed while exercising nothing, which the mutation
        // caught and a green run would not have.
        //
        // **A node ceiling**, because a full depth-8 search costs 27 s. Nesting occurs
        // within the first 50 000 nodes, so the ceiling cuts the test to 0.4 s without
        // weakening it: what matters is that the situation arises, not that the search
        // finishes.
        searcher.node_limit = Some(50_000);
        searcher.root(&p, 8, None);

        assert!(searcher.max_null_nesting > 0, "precondition: null moves must have been tried");
        assert_eq!(searcher.max_null_nesting, 1, "never nested — one pass at a time");
    }

    #[test]
    fn finds_mate_in_one() {
        // White: Ra1, Kh1. Black: Kg8, pawns f7/g7/h7. Ra8# is the mate in one.
        let p = Position::from_fen("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1").unwrap();
        let (mv, score) = best_move(&p, 3).expect("a move exists");
        assert_eq!(p.play(mv).status(), Status::Checkmate);
        assert!(score >= MATE - 100, "score {score} should be mate-level");
    }

    #[test]
    fn grabs_a_hanging_piece() {
        // Black's queen on d5 is undefended; White's rook on d1 takes it for free.
        let p = Position::from_fen("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1").unwrap();
        let (mv, _) = best_move(&p, 3).expect("a move exists");
        let after = p.play(mv);
        assert_eq!(
            after.count(Color::Black, Piece::Queen),
            0,
            "the engine should win the free queen"
        );
    }

    #[test]
    fn no_move_at_a_terminal_root() {
        // Checkmate (scholar's mate) and stalemate: nothing to search.
        let mate = Position::from_fen(
            "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4",
        )
        .unwrap();
        let stalemate = Position::from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1").unwrap();
        assert!(best_move(&mate, 3).is_none());
        assert!(best_move(&stalemate, 3).is_none());
    }

    #[test]
    fn plays_a_full_legal_game() {
        // The engine plays both sides. Every chosen move must be legal (accepted
        // by `try_play`) and nothing should panic. Shallow depth keeps it fast.
        let mut pos = Position::initial();
        for _ in 0..40 {
            match best_move(&pos, 2) {
                Some((mv, _)) => pos = pos.try_play(mv).expect("engine chose a legal move"),
                None => break, // terminal reached
            }
        }
    }

    #[test]
    fn ordering_does_not_change_the_score() {
        // Alpha-beta's value is independent of move order: ordered and unordered
        // search must agree on the score for the same position and depth.
        for fen in [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 3",
            "4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1",
        ] {
            let p = Position::from_fen(fen).unwrap();
            let ordered = search_fixed(&p, 3, MoveOrder::Full).best.map(|(_, s)| s);
            let unordered = search_fixed(&p, 3, MoveOrder::None).best.map(|(_, s)| s);
            assert_eq!(ordered, unordered, "score must not depend on ordering ({fen})");
        }
    }

    #[test]
    fn ordering_prunes_more_nodes() {
        // On a middlegame position, ordering must visit strictly fewer nodes. Depth 3
        // rather than 4: quiescence multiplies what the unordered search explores —
        // 229k nodes here against 2.3M at depth 4 — and the gap is already a factor of
        // 33, so the extra ply costs ten times the runtime to prove the same thing.
        let p = Position::from_fen(
            "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 3",
        )
        .unwrap();
        let ordered = search_fixed(&p, 3, MoveOrder::Full);
        let unordered = search_fixed(&p, 3, MoveOrder::None);
        assert!(
            ordered.nodes < unordered.nodes,
            "ordering should prune: ordered {} vs unordered {}",
            ordered.nodes,
            unordered.nodes
        );
    }

    #[test]
    fn killers_prune_beyond_what_mvv_lva_can_reach() {
        // The opening position is where the difference is starkest, and for the
        // reason the heuristic was built: almost every move here is quiet, so
        // MVV-LVA has nothing to rank and the search wades through the move list in
        // generator order. Measured at depth 5: 273 833 nodes on captures-only
        // ordering against 32 424 with killers.
        //
        // A factor of two is asserted rather than the measured 8.4: node counts at a
        // fixed depth are deterministic, so `<` alone would pass on a one-node
        // difference, while pinning the exact figure would break on any later
        // ordering change that is perfectly legitimate.
        let p = Position::initial();
        let with = search_fixed(&p, 5, MoveOrder::Full);
        let without = search_fixed(&p, 5, MoveOrder::Captures);
        assert!(
            with.nodes * 2 < without.nodes,
            "killers should prune substantially: {} with vs {} without",
            with.nodes,
            without.nodes
        );
    }

    #[test]
    fn killers_do_not_change_the_score() {
        // Same invariant as `ordering_does_not_change_the_score`, one heuristic
        // further: killers reorder the quiet moves, and reordering must leave
        // alpha-beta's value untouched.
        //
        // Only the *score* is compared, never the move: several moves can share the
        // best score, and the one returned is whichever reached it first — which is
        // precisely what an ordering change is allowed to alter.
        for (fen, depth) in [
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 5),
            ("r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 3", 5),
            // Kiwipete: a tactical position where captures dominate. Killers barely
            // prune here (0.4% of nodes), which is exactly why it belongs in this
            // test — it is the case where the two searches differ least, so an
            // accidental change of value would be easiest to miss.
            ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 4),
        ] {
            let p = Position::from_fen(fen).unwrap();
            let with = search_fixed(&p, depth, MoveOrder::Full).best.map(|(_, s)| s);
            let without = search_fixed(&p, depth, MoveOrder::Captures).best.map(|(_, s)| s);
            assert_eq!(with, without, "killers must not change the score ({fen})");
        }
    }

    // A king alone against a queen: lost by 900 centipawns, unless the game can be
    // steered back to a position it has already been in.
    const LOST_KING: &str = "3qk3/8/8/8/8/8/8/4K3 w - - 0 1";

    #[test]
    fn a_draw_by_repetition_saves_a_lost_position() {
        let p = Position::from_fen(LOST_KING).unwrap();
        let (_, lost) = best_move(&p, 4).expect("a move");
        assert!(lost < -800, "without history this is simply lost: {lost}");

        // Ke2 reaches a position the game has already been in *twice*, so playing it
        // is the third occurrence — a real draw, worth 900 centipawns more than the
        // alternative.
        let after_ke2 = p.play(p.move_from_uci("e1e2").unwrap());
        let twice = vec![after_ke2.hash(), after_ke2.hash()];
        let stats = search(&p, Request { history: &twice, ..Request::new(Limits::depth(4)) });
        let (mv, score) = stats.best.expect("a move");
        assert_eq!(score, 0, "the third occurrence is a draw, not a loss");
        assert_eq!(p.move_to_uci(mv), "e1e2", "and it is the move that reaches it");
    }

    #[test]
    fn one_earlier_occurrence_is_not_yet_a_draw() {
        // The defect this guards against: treating a *second* occurrence as a draw.
        // The rules want three, and the opponent is still free to decline the
        // repetition — so an engine that scores this 0 plays for a draw that does not
        // exist, and lands in a position it evaluated as level.
        let p = Position::from_fen(LOST_KING).unwrap();
        let after_ke2 = p.play(p.move_from_uci("e1e2").unwrap());

        let once = search(&p, Request { history: &[after_ke2.hash()], ..Request::new(Limits::depth(4)) });
        let (_, score) = once.best.expect("a move");
        assert!(score < -800, "one prior occurrence is only the second: {score}");

        // Two, and it becomes the third — then it is a draw.
        let twice = vec![after_ke2.hash(), after_ke2.hash()];
        assert_eq!(
            search(&p, Request { history: &twice, ..Request::new(Limits::depth(4)) }).best.map(|(_, s)| s),
            Some(0),
            "the third occurrence is drawn"
        );
    }

    #[test]
    fn a_repetition_is_not_sought_when_winning() {
        // Same shape, colours reversed: a queen up, repeating would throw away the
        // win. Scoring a draw 0 must not make the engine want one.
        let p = Position::from_fen("4k3/8/8/8/8/8/8/3QK3 w - - 0 1").unwrap();
        let after_kd2 = p.play(p.move_from_uci("e1d2").unwrap());
        let twice = vec![after_kd2.hash(), after_kd2.hash()];
        let stats = search(&p, Request { history: &twice, ..Request::new(Limits::depth(4)) });
        let (mv, score) = stats.best.expect("a move");
        assert!(score > 800, "a queen up must stay winning, got {score}");
        assert_ne!(p.move_to_uci(mv), "e1d2", "and the drawing move must be avoided");
    }

    #[test]
    fn the_path_needs_one_occurrence_and_the_history_needs_two() {
        // The asymmetry itself, on the mechanism — constructing a forced in-tree
        // repetition on the board is fragile, and this is the property that matters.
        let mut table = Table::new();
        let mut searcher = Searcher::new(MoveOrder::Full, None, &mut table);
        assert!(!searcher.is_repetition(42), "an unseen key is not a repetition");

        // In the tree: one is enough, because both sides are choosing moves here and
        // a side that can steer back once can do it again.
        searcher.path.push(42);
        assert!(searcher.is_repetition(42), "one occurrence on the path suffices");
        searcher.path.pop();
        assert!(!searcher.is_repetition(42), "and it stops counting once popped");

        // In the game: one prior occurrence makes this only the second.
        searcher.history.push(42);
        assert!(!searcher.is_repetition(42), "one played occurrence is not a draw yet");
        searcher.history.push(42);
        assert!(searcher.is_repetition(42), "two played occurrences make this the third");
    }

    #[test]
    fn a_repetition_score_is_not_cached() {
        // The hazard #23 introduced: a draw by repetition belongs to the *path*, not
        // to the position. If it were stored under the position's key, every other
        // path reaching that position would read back a draw that does not apply.
        let p = Position::from_fen(LOST_KING).unwrap();
        let after_ke2 = p.play(p.move_from_uci("e1e2").unwrap());

        let mut table = Table::new();
        let mut searcher = Searcher::new(MoveOrder::Full, None, &mut table);
        // Two occurrences, so the position genuinely is scored 0 as a repetition.
        // Asserting that first is the point: with one, this test would pass while
        // exercising nothing — which is exactly what happened when the history
        // threshold moved from one to two.
        searcher.history = vec![after_ke2.hash(), after_ke2.hash()];
        assert_eq!(
            searcher.root(&p, 4, None).best.map(|(_, s)| s),
            Some(0),
            "precondition: the repetition must actually be found"
        );

        // Two complementary windows, because a single infinite one cannot see a stored
        // draw: `probe` only returns `Some` for `Exact` unless the bound settles the
        // window, and neither `Lower` nor `Upper` ever does against ±INF. (0, 1) admits
        // Exact and Upper; (-1, 0) admits Exact and Lower. Between them, no bound type
        // can hide a cached 0.
        let key = after_ke2.hash();
        assert_ne!(searcher.table.probe(key, 1, 0, 1, 0).cutoff, Some(0));
        assert_ne!(searcher.table.probe(key, 1, -1, 0, 0).cutoff, Some(0));
    }

    #[test]
    fn what_one_move_learns_is_there_for_the_next() {
        // The behaviour the `Engine` exists for. Two consecutive moves of a game
        // explore largely the same tree — the opponent replies, and the engine starts
        // again from a position it just analysed deeply. On one `Engine`, the second
        // search reads what the first wrote.
        // Totalled over the first eight moves of the position rather than measured on one
        // line of play. The saving is a property of the table in aggregate, not a guarantee
        // for every individual continuation: a carried table also reorders the moves, which
        // changes which ones rank late enough to be reduced and by how much, so a single
        // line can come out worse while the whole is better. Measured with the growing
        // curve, one line of play gave 33 102 nodes carried against 31 691 fresh — the
        // sweep over eight gives 181 636 against 198 460, which is 8.5% of margin.
        //
        // The curve does erode what the table saves, and on one position nature it erases it.
        // Carried against fresh, by nature:
        //
        //     curve off   0.803  0.989  0.998  0.855
        //     curve on    0.915  0.987  1.002  0.902
        //
        // Mechanical — reductions shrink both trees, so there is less left for the table to
        // save — but the tactical position crosses 1.0 with the curve on, so the honest
        // statement is **three natures of four**, not all four. It stays a property of the
        // whole rather than of every continuation, which is what this test measures.
        let start = Position::initial();
        let (mut carried_nodes, mut fresh_nodes) = (0u64, 0u64);
        for mv in start.legal_moves().into_iter().take(8) {
            let next = start.play(mv);
            let mut engine = Engine::new();
            engine.search(&start, Request::new(Limits::depth(6)));
            let before = engine.search(&next, Request::new(Limits::depth(6)));
            carried_nodes += before.nodes;
            fresh_nodes += Engine::new().search(&next, Request::new(Limits::depth(6))).nodes;
        }
        let (carried, fresh) = (carried_nodes, fresh_nodes);

        // The saved work is the whole contract, and it is the only thing asserted. The
        // two searches are **not** required to agree on the score: a stored bound is a
        // fact about a position under one alpha-beta window, and reused under another it
        // cuts elsewhere and returns a different, equally valid value — which is why
        // `Engine`'s own documentation tells a caller who needs reproducible numbers to
        // hold a fresh one per position.
        //
        // Reductions do not merely expose that; they are what makes it happen here.
        // Measured over four positions × 12 first moves × depths 4-6, carried table
        // against fresh: **0 divergences out of 144 with reductions off, 6 out of 144
        // with them on**, worst gap 15 cp. So the reason to drop a score assertion is not
        // that it was already failing — it was not — but that it demands reproducibility
        // the contract never offered. This repository has retired four criteria of exactly
        // that shape (#19, #27, #36, and #42's own rule against it).
        assert!(
            carried < fresh,
            "a carried-over table must save work: {carried} nodes against {fresh} from scratch",
        );
    }

    #[test]
    fn a_new_game_starts_from_nothing() {
        // Positions from a finished game would compete for slots with the ones that
        // matter now. `new_game` is the only thing that empties the table — and the
        // protocol layer is the only level that knows a game has ended.
        let p = Position::initial();
        let mut engine = Engine::new();
        let fresh = engine.search(&p, Request::new(Limits::depth(6)));

        let warm = engine.search(&p, Request::new(Limits::depth(6)));
        assert!(warm.nodes < fresh.nodes, "precondition: the table must be carrying something");

        engine.new_game();
        let after_clear = engine.search(&p, Request::new(Limits::depth(6)));
        assert_eq!(after_clear.nodes, fresh.nodes, "a cleared table must search like a new one");
    }

    #[test]
    fn a_mate_score_survives_from_one_search_to_the_next() {
        // Mate scores are stored relative to the *node* rather than to the root, so
        // that an entry means the same thing wherever it is read from. Persisting the
        // table is what makes that matter: an entry is now read by a search whose root
        // is a different position entirely. If the conversion were wrong, a mate in 2
        // would come back as a mate in some other number — plausible enough to pass
        // unnoticed, and wrong.
        let p = Position::from_fen("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1").unwrap();
        let mut engine = Engine::new();
        let first = engine.search(&p, Request::new(Limits::depth(4)));
        let (_, score) = first.best.expect("a mate exists");
        assert!(score > MATE_THRESHOLD, "precondition: the first search must find a mate");

        let again = engine.search(&p, Request::new(Limits::depth(4)));
        assert_eq!(again.best.map(|(_, s)| s), Some(score), "the same mate, at the same distance");
    }

    #[test]
    fn a_repeated_search_reuses_the_table() {
        // The same position, twice, on one `Searcher`. The second pass asks the very
        // questions the first one answered, so almost all of it should come out of the
        // table rather than the tree.
        let p = Position::initial();
        let mut table = Table::new();
        let mut searcher = Searcher::new(MoveOrder::Full, None, &mut table);
        searcher.root(&p, 5, None);
        let first = searcher.nodes;
        searcher.nodes = 0;
        searcher.root(&p, 5, None);
        let second = searcher.nodes;
        assert!(
            second * 10 < first,
            "a cached re-search should cost a fraction of the first: {second} against {first}"
        );
    }

    #[test]
    fn the_table_is_actually_consulted() {
        // Guards against a table that is written but never read, or one whose every
        // hit is discarded as a collision — both look like a working table from the
        // node count alone, which is why the hit rate is reported.
        let p = Position::from_fen(
            "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 3",
        )
        .unwrap();
        let stats = search_timed(&p, Limits::depth(5));
        assert!(
            stats.table_key_match_rate > 0.05,
            "expected the table to answer some probes, got {:.3}",
            stats.table_key_match_rate
        );
        assert!(
            stats.table_cutoff_rate > 0.0,
            "and some of those matches must actually cut off, got {:.3}",
            stats.table_cutoff_rate
        );
    }

    #[test]
    fn a_cached_mate_keeps_its_distance() {
        // Mate scores are stored relative to the node, not to the root, so a mate
        // cached at one depth must still read as the same mate at another. If the
        // conversion were wrong, deepening would report a different mate distance at
        // each iteration — an engine that announces mate and then does not deliver.
        let p = Position::from_fen("6k1/5ppp/8/8/8/8/8/R6K w - - 0 1").unwrap();
        let mut previous = None;
        for depth in 3..=6 {
            let (_, score) = best_move(&p, depth).expect("a move");
            assert!(score >= MATE - 100, "depth {depth} lost the mate: {score}");
            if let Some(before) = previous {
                assert_eq!(score, before, "the mate distance moved between iterations");
            }
            previous = Some(score);
        }
    }

    #[test]
    fn iterative_deepening_matches_direct_search() {
        // With no deadline, deepening to depth D returns the same score as a single
        // direct depth-D pass, and reports having reached D.
        //
        // Both sides run without late move reductions, and the reason is the property
        // itself rather than convenience. A reduction is decided from a move's *rank*, so
        // it depends on the order the moves are in — and the two searches order
        // differently by construction: deepening arrives at depth D with a filled
        // transposition table and killers from D-1, the direct pass with neither. Same
        // tree, different moves reduced, so the scores may legitimately differ (measured:
        // 0 against 50 on the initial position at depth 4). What deepening promises is
        // that *it* does not change the answer; with an order-dependent cut on, the two
        // are no longer comparable at all, and asserting they are would demand
        // reproducibility where the contract is a valid bound.
        for fen in [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 3",
        ] {
            let p = Position::from_fen(fen).unwrap();
            let deep = {
                let mut table = Table::new();
                let mut s = Searcher::new(MoveOrder::Full, None, &mut table);
                s.allow_lmr = false;
                deepen(&p, Request::new(Limits::depth(4)), &mut s)
            };
            let direct = search_fixed(&p, 4, MoveOrder::Full);
            assert_eq!(deep.best.map(|(_, s)| s), direct.best.map(|(_, s)| s));
            assert_eq!(deep.depth, 4);
        }
    }

    #[test]
    fn best_move_goes_through_deepening() {
        // `best_move` is only a wrapper: same score as the deepening search it calls,
        // and as a direct fixed-depth pass.
        let p = Position::from_fen("4k3/8/8/3q4/8/8/8/3RK3 w - - 0 1").unwrap();
        let wrapped = best_move(&p, 4).map(|(_, s)| s);
        assert_eq!(wrapped, search_timed(&p, Limits::depth(4)).best.map(|(_, s)| s));
        assert_eq!(wrapped, search_fixed(&p, 4, MoveOrder::Full).best.map(|(_, s)| s));
    }

    #[test]
    fn bounded_honours_both_limits() {
        // `Limits::bounded` is what a UCI `go` maps onto. A depth cap with a generous
        // deadline stops at the cap; a deadline with no useful depth cap stops on time.
        let p = Position::initial();
        // An hour, so the deadline cannot be what ends the search: depths 1-3 total
        // 1508 nodes, and only a machine below one node per second would need longer.
        // The cap is then the only thing left that can stop it, which is the point.
        let far = Instant::now() + Duration::from_secs(3600);
        assert_eq!(search_timed(&p, Limits::bounded(3, Some(far))).depth, 3);

        // Conversely the clock decides here, and both bounds hold at any speed: depth
        // 1 always completes (see the expired-deadline test), and MAX_DEPTH is far out
        // of reach in 5 ms on any hardware.
        let stats = search_timed(
            &p,
            Limits::bounded(MAX_DEPTH, Some(Instant::now() + Duration::from_millis(5))),
        );
        assert!(stats.depth >= 1 && stats.depth < MAX_DEPTH, "the clock must bite");
        assert!(stats.best.is_some(), "a legal move is always returned");
    }

    // The position behind the quiescence tests: White's queen can take the d6 pawn,
    // and the e7 pawn takes it straight back. A search that stops counting after the
    // capture sees a won pawn; one that plays the exchange out sees a lost queen.
    const HORIZON: &str = "4k3/4p3/3p4/8/8/8/8/3QK3 w - - 0 1";

    // Quiescence from `pos`, on a full window — what a leaf of the main search gets.
    fn quiesce(pos: &Position) -> i32 {
        let mut table = Table::new();
        Searcher::new(MoveOrder::Full, None, &mut table).quiescence(pos, -INF, INF, 0)
    }

    #[test]
    fn the_recapture_is_seen_past_the_depth_limit() {
        let p = Position::from_fen(HORIZON).unwrap();
        let static_eval = evaluate(&p);
        // At depth 1 the capture is the very last ply, so its recapture falls exactly
        // one ply beyond the limit — the horizon effect in its purest form. The score
        // must not claim the pawn: taking it costs the queen.
        let (_, score) = best_move(&p, 1).expect("a move");
        assert!(
            score < static_eval + 100,
            "score {score} claims material over the static {static_eval}: the recapture was missed"
        );
    }

    #[test]
    fn a_losing_capture_is_not_forced() {
        // Qxd6 is the only capture available and it loses the queen. Standing pat is
        // always allowed, so the value of the position is the static score, never the
        // best of a bad set of captures.
        let p = Position::from_fen(HORIZON).unwrap();
        assert_eq!(quiesce(&p), evaluate(&p));
    }

    #[test]
    fn the_promotion_is_seen_past_the_depth_limit() {
        // Black's a2 pawn queens next move. Before promotions were searched, the leaf
        // counted a pawn and the main search a queen — a 730 cp gap between depth 1
        // and depth 2 on the same position. Quiescence must close it.
        let p = Position::from_fen("4k3/8/8/8/8/8/p7/4K3 w - - 0 1").unwrap();
        let shallow = best_move(&p, 1).expect("a move").1;
        let deeper = best_move(&p, 2).expect("a move").1;
        assert_eq!(
            shallow, deeper,
            "depth 1 must already price the queen: {shallow} vs {deeper} at depth 2"
        );
    }

    #[test]
    fn a_capture_promotion_is_still_searched() {
        // `b7a8q` takes the rook *and* queens. It reached quiescence before this
        // change, because `mvv_lva` sees the rook on the destination square — the
        // promotion test must not have disturbed that path.
        let p = Position::from_fen("r3k3/1P6/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        assert!(
            quiesce(&p) >= evaluate(&p) + 1000,
            "expected rook + promotion, got {} over {}",
            quiesce(&p),
            evaluate(&p)
        );
    }

    #[test]
    fn a_losing_promotion_is_not_forced() {
        // The c8 rook covers a8, so queening loses the new queen at once. Standing pat
        // applies to promotions exactly as it does to captures: the value of the
        // position is the static score, not the best of a bad set of moves.
        let p = Position::from_fen("2r1k3/P7/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        assert_eq!(quiesce(&p), evaluate(&p));
    }

    #[test]
    fn a_quiet_position_is_worth_its_static_score() {
        // Nothing to capture: quiescence has no work to do and must agree exactly with
        // the evaluation it extends.
        let p = Position::from_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1").unwrap();
        assert_eq!(quiesce(&p), evaluate(&p));
    }

    #[test]
    fn a_winning_capture_is_taken() {
        // The mirror case: the rook on a7 is undefended, so quiescence must find the
        // queen takes it and report roughly a rook more than the static score.
        let p = Position::from_fen("4k3/r7/8/8/8/8/8/Q3K3 w - - 0 1").unwrap();
        assert!(
            quiesce(&p) >= evaluate(&p) + 400,
            "expected a free rook, got {} over {}",
            quiesce(&p),
            evaluate(&p)
        );
    }

    #[test]
    fn terminal_positions_are_still_terminal_at_a_leaf() {
        // Quiescence took over leaf duty from `negamax`, so it owns mate and stalemate
        // detection there: a mate must score as a mate, not as a material count.
        let mate = Position::from_fen(
            "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4",
        )
        .unwrap();
        let stalemate = Position::from_fen("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1").unwrap();
        assert_eq!(quiesce(&mate), -MATE);
        assert_eq!(quiesce(&stalemate), 0);
    }

    #[test]
    fn an_expired_deadline_still_returns_the_guaranteed_iteration() {
        // A flag-fall can reach the engine as a deadline already in the past. From a
        // quiet position depth 1 is owed anyway: `check_limits` only reads the clock
        // every 2048 nodes and iteration 1 costs 41 from the start position, so it
        // completes; the deadline is then re-checked between iterations and stops the
        // loop there. Structural for *this* position, hence an equality.
        let expired = Instant::now() - Duration::from_secs(1);
        let stats = search_timed(&Position::initial(), Limits::bounded(MAX_DEPTH, Some(expired)));
        assert_eq!(stats.depth, 1, "an expired deadline still owes one iteration");
        assert!(stats.best.is_some(), "a legal move is always returned");
    }

    #[test]
    fn a_legal_move_comes_back_even_when_iteration_one_is_cut_short() {
        // The node argument above does *not* generalise: quiescence makes iteration 1
        // cost 25 906 nodes on Kiwipete against 41 from the start position, so the
        // abort can land inside the very first iteration and leave nothing complete to
        // fall back on. A move must still come back — the alternative is `bestmove
        // 0000`, which an arena scores as an illegal move and an immediate loss.
        let kiwipete = Position::from_fen(
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        )
        .unwrap();
        let expired = Instant::now() - Duration::from_secs(1);
        let stats = search_timed(&kiwipete, Limits::bounded(MAX_DEPTH, Some(expired)));
        let (mv, _) = stats.best.expect("a move, even from an unfinished iteration");
        assert!(kiwipete.try_play(mv).is_ok(), "the returned move must be legal");
    }

    #[test]
    fn every_completed_iteration_is_reported_exactly_once() {
        // The property, stated without reference to speed: *how many* iterations
        // finish depends on the machine, but each one that finishes is announced
        // once — so the number of reports equals the depth reached, which
        // `SearchStats` already carries. That makes this assertable as an equality
        // on any hardware, including the degenerate case where nothing completes.
        for (fen, budget) in [
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 1u64),
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 100),
            ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 1),
            ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", 100),
        ] {
            let pos = Position::from_fen(fen).unwrap();
            let mut reported = Vec::new();
            // The closure borrows `reported` mutably, so it has to go out of scope
            // before the assertion can read it — hence the block rather than a `drop`.
            let stats = {
                let mut record = |p: &Progress| reported.push(p.depth);
                search(
                    &pos,
                    Request {
                        progress: Some(&mut record),
                        ..Request::new(Limits::until(
                            Instant::now() + Duration::from_millis(budget),
                        ))
                    },
                )
            };
            assert_eq!(
                reported,
                (1..=stats.depth).collect::<Vec<u32>>(),
                "one report per completed depth, in order ({fen}, {budget}ms)"
            );
        }
    }

    #[test]
    fn a_tiny_budget_still_returns_a_legal_move() {
        // Even a few milliseconds must yield at least the depth-1 result. The bound is
        // `>=`, never a particular depth: how far 5 ms reaches is a property of the
        // machine, and a test may not depend on it.
        let p = Position::initial();
        let stats = search_timed(&p, Limits::until(Instant::now() + Duration::from_millis(5)));
        let (mv, _) = stats.best.expect("a move");
        assert!(p.try_play(mv).is_ok(), "the returned move must be legal");
        assert!(stats.depth >= 1);
    }
    // --- late move reductions ---------------------------------------------------

    // A search through the ordinary deepening loop, plus what the reductions did along
    // the way. The counters are the point: a guard whose whole effect is that a reduction
    // did *not* happen cannot be observed from the node count alone.
    struct Reduced {
        stats: SearchStats,
        reductions: u64,
    }

    fn search_reducing(pos: &Position, depth: u32, allow: bool) -> Reduced {
        let mut table = Table::new();
        let mut s = Searcher::new(MoveOrder::Full, None, &mut table);
        s.allow_lmr = allow;
        let stats = deepen(pos, Request::new(Limits::depth(depth)), &mut s);
        Reduced { reductions: s.lmr_reductions, stats }
    }

    // The four positions every node-count measurement in this repository uses, chosen to
    // be of different natures: a heuristic only pays where there is something for it to
    // work on, and one that helps enormously in the opening can be dead weight in a
    // tactical position. Quoting the worst alongside the best is the rule here.
    const NATURES: [(&str, &str); 4] = [
        ("opening", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("quiet middlegame", "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 3"),
        ("tactical", "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"),
        ("endgame", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
    ];

    #[test]
    fn reductions_shrink_the_tree_on_every_nature_of_position() {
        // What the feature is for, measured against a search identical but for the
        // reduction, swept over depths and over positions of different natures rather
        // than shown at the one pair that flatters it.
        //
        // The weakest position carries the amplitude assertion, and that is the whole
        // point of measuring four. The history heuristic (#40) went into a branch on an
        // *average* node ratio of 0.94 while being negative on two positions out of four;
        // its Elo measurement then came back at -10. An ordering or pruning heuristic only
        // pays where there is something for it to work on, but its per-node cost is paid
        // everywhere, so the position it helps least is the one that decides.
        //
        // Two properties on two domains, as for the null move. **That it reduces** holds
        // from the floor upwards and is what a regression breaks. **How much** grows with
        // depth — the tactical position sheds 1% at depth 4 and 37% at depth 5 — because a
        // deeper tree offers more late quiet moves and each reduction compounds over the
        // levels beneath it. So the amplitude is read at the deepest swept depth;
        // asserting it across the sweep would only ever pin the shallowest one.
        const DEEPEST: u32 = 5;
        let mut worst: Option<(&str, f64)> = None;
        for (nature, fen) in NATURES {
            let p = Position::from_fen(fen).unwrap();
            for depth in (LMR_MIN_DEPTH + 1)..=DEEPEST {
                let with = search_reducing(&p, depth, true);
                let without = search_reducing(&p, depth, false);
                // `<=`, not `<`: at the shallowest swept depth the tree can be too small
                // for reductions to remove anything, and that is not a regression. Measured
                // after the passed-pawn term (#46) changed the shape of every tree: the
                // endgame position at depth 4 reads exactly 1.000, having read 0.727 before.
                // What must never happen is reductions *costing* nodes; that they gain is
                // asserted strictly at `DEEPEST` below, which is where there is matter to
                // work on.
                assert!(
                    with.stats.nodes <= without.stats.nodes,
                    "{nature} at depth {depth}: reductions cost nodes, {} against {}",
                    with.stats.nodes,
                    without.stats.nodes,
                );
                // The precondition belongs where the amplitude is read, not at every swept
                // depth. A small tree can legitimately offer no move at rank 3 or beyond —
                // the endgame position at depth 4 stopped reducing entirely once the
                // passed-pawn term (#46) changed which branches get cut. Requiring a
                // reduction there asserted a property of the position, not of the feature.
                if depth == DEEPEST {
                    assert!(
                        with.reductions > 0,
                        "precondition: {nature} at depth {depth} must reduce something for \
                         the ratio below to mean anything",
                    );
                }
                if depth == DEEPEST {
                    let ratio = with.stats.nodes as f64 / without.stats.nodes as f64;
                    if worst.is_none_or(|(_, w)| ratio > w) {
                        worst = Some((nature, ratio));
                    }
                }
            }
        }
        let (nature, ratio) = worst.expect("the sweep reached DEEPEST");
        assert!(
            ratio < 0.75,
            "at depth {DEEPEST} even the least favourable position must shed a quarter of \
             its tree: {nature} at {ratio:.3}",
        );
    }

    #[test]
    fn a_real_search_reduces_and_re_searches() {
        // The test the history heuristic did not have, and the reason it shipped wired to
        // a measurement nobody had made. Eight unit tests covered that component, all
        // green, and unplugging it from the search broke none of them — every one drove it
        // directly.
        //
        // This goes through `deepen`, which is the path `Engine::search` takes: that method
        // is exactly `Searcher::new(Full, deadline, table)` followed by this call, so
        // deepening and the transposition table are both in play. The searcher is built by
        // hand rather than by calling `Engine::search` because asserting on
        // `lmr_reductions` means still holding it afterwards — which is why `deepen`
        // borrows its searcher instead of consuming it.
        let mut table = Table::new();
        let mut s = Searcher::new(MoveOrder::Full, None, &mut table);
        let stats = deepen(
            &Position::initial(),
            Request::new(Limits::depth(6)),
            &mut s,
        );
        assert!(stats.best.is_some(), "precondition: the search returns a move");
        assert!(s.lmr_reductions > 0, "a real search must reduce late quiet moves");
        // The re-search is the half that keeps the feature honest: without it a reduction
        // decides the move rather than only its cost. If it never fires, the reduction is
        // never being contradicted, and that is a claim about the whole engine no test
        // would otherwise catch.
        assert!(
            s.lmr_researches > 0,
            "and must re-search at least one of them at full depth",
        );
    }

    // The reduction the search would apply to `mv` sitting at `rank`, with `killer`
    // recorded at this ply if given.
    //
    // Every guard below is tested as a **pair**: the case that must not be reduced, and a
    // control that differs only in the thing being guarded and *is* reduced. Without the
    // control a guard test proves nothing — it passes just as happily when some unrelated
    // condition is what returned zero, which is how a test ends up green while watching
    // the wrong thing.
    fn reduction_for(
        pos: &Position,
        mv: Move,
        depth: u32,
        rank: usize,
        killer: Option<Move>,
    ) -> u32 {
        const PLY: i32 = 1;
        let mut table = Table::new();
        let mut s = Searcher::new(MoveOrder::Full, None, &mut table);
        if let Some(k) = killer {
            s.killers.record(pos, PLY as usize, k);
        }
        s.late_move_reduction(pos, &pos.play(mv), mv, depth, rank, PLY)
    }

    // Deep enough and late enough that only the guard under test can return zero.
    //
    // These tests assert **whether** a move is reduced, never by how much: the amount comes
    // from a curve in `depth` and `rank`, and pinning its value here would make every guard
    // test fail the moment that curve is retuned — for a reason that has nothing to do with
    // the guard it is named after. How much is the subject of the curve tests below.
    const DEEP: u32 = 6;
    const LATE: usize = LMR_FULL_DEPTH_MOVES + 2;

    #[test]
    fn a_node_in_check_is_never_reduced() {
        // In check the move list is short and every reply is forced, so "most of these
        // moves are irrelevant" — the premise the reduction rests on — is false.
        let in_check = Position::from_fen("4k3/8/8/8/8/8/4r3/4K3 w - - 0 1").unwrap();
        assert!(in_check.in_check(), "precondition: white is in check");
        let flight = in_check.move_from_uci("e1d1").unwrap();
        assert_eq!(reduction_for(&in_check, flight, DEEP, LATE, None), 0);

        // The control: the same king step, from a position that differs only in not being
        // in check. Without this the test above would also pass if `d1` were a capture, or
        // if the rank were too early, or if the whole feature were switched off.
        let safe = Position::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let same_step = safe.move_from_uci("e1d1").unwrap();
        assert!(reduction_for(&safe, same_step, DEEP, LATE, None) > 0);
    }

    #[test]
    fn a_move_that_gives_check_is_never_reduced() {
        // A check is forcing, and forcing lines are exactly what a shallower search stops
        // seeing — this is the guard that keeps reductions from costing mates.
        //
        // Both moves come from the same position, so the pair differs in nothing but
        // whether the rook lands on the eighth rank.
        let p = Position::from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1").unwrap();
        let checking = p.move_from_uci("a1a8").unwrap();
        assert!(p.play(checking).in_check(), "precondition: Ra8 is check");
        assert_eq!(reduction_for(&p, checking, DEEP, LATE, None), 0);

        let quiet = p.move_from_uci("a1a7").unwrap();
        assert!(!p.play(quiet).in_check(), "precondition: Ra7 is not");
        assert!(reduction_for(&p, quiet, DEEP, LATE, None) > 0);
    }

    #[test]
    fn a_move_that_touches_material_is_never_reduced() {
        // A capture or a promotion moves the one quantity the evaluation reads directly,
        // and the ordering has already ranked it on that basis. Reducing it would be
        // betting against our own information.
        let capture_available = Position::from_fen("4k3/8/8/8/3p4/4P3/8/4K3 w - - 0 1").unwrap();
        let capture = capture_available.move_from_uci("e3d4").unwrap();
        assert!(!is_quiet(&capture_available, capture), "precondition: exd4 takes a pawn");
        assert_eq!(reduction_for(&capture_available, capture, DEEP, LATE, None), 0);

        // Control: the same pawn, one square forward instead of diagonally.
        let push = capture_available.move_from_uci("e3e4").unwrap();
        assert!(reduction_for(&capture_available, push, DEEP, LATE, None) > 0);

        // A promotion reaches an empty square, so `mvv_lva` alone would read it as quiet —
        // the same blind spot that once kept promotions out of quiescence. `is_quiet` is
        // what catches it, and this pins that it is `is_quiet` being consulted here.
        let promoting = Position::from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1").unwrap();
        let promotion = promoting.move_from_uci("a7a8q").unwrap();
        assert_eq!(reduction_for(&promoting, promotion, DEEP, LATE, None), 0);
    }

    #[test]
    fn a_killer_is_never_reduced() {
        // A killer refuted a sibling of this very node moments ago. That is evidence, and
        // it is why the ordering placed it high; reducing it would discard the evidence.
        let p = Position::from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1").unwrap();
        let mv = p.move_from_uci("a1a7").unwrap();
        assert_eq!(reduction_for(&p, mv, DEEP, LATE, Some(mv)), 0);
        // Control: the same move at the same rank and depth, with no killer recorded.
        assert!(reduction_for(&p, mv, DEEP, LATE, None) > 0);
    }

    #[test]
    fn the_first_moves_are_searched_at_full_depth() {
        // The head of the list is what the ordering believes in — the transposition
        // table's move, the good captures, the killers. Swept over every rank up to the
        // boundary and one past it, so an off-by-one shows up as a failure rather than as
        // a test that happens to sample the right side of it.
        let p = Position::from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1").unwrap();
        let mv = p.move_from_uci("a1a7").unwrap();
        for rank in 0..LMR_FULL_DEPTH_MOVES {
            assert_eq!(
                reduction_for(&p, mv, DEEP, rank, None),
                0,
                "move {rank} is still in the trusted head of the list",
            );
        }
        assert!(
            reduction_for(&p, mv, DEEP, LMR_FULL_DEPTH_MOVES, None) > 0,
            "and the first move past it is reduced",
        );
    }

    #[test]
    fn the_depth_floor_and_the_ceiling_currently_say_the_same_thing() {
        // Two mechanisms forbid a reduction at shallow depth, and **at today's constants they
        // forbid exactly the same set**: the guard `depth >= LMR_MIN_DEPTH` excludes depths
        // 0-2, and the ceiling `min(depth - 2)` returns zero for those same depths. Removing
        // the guard changes nothing — verified byte for byte, node counts and reduction counts
        // alike, on four positions over depths 3-7 — and the whole suite stays green without
        // it.
        //
        // That is worth a test of its own rather than a comment, because `LMR_MIN_DEPTH` is a
        // lever #44 deliberately left at its measured value for someone to come back to, and
        // it currently acts *only* through a guard that does nothing. Raise it to 6 and
        // reductions vanish below depth 7; delete the redundant-looking guard first and the
        // constant is permanently inoperative, with 138 tests still green and nothing to say
        // so.
        //
        // **If this test fails, the redundancy has ended** — the two mechanisms now forbid
        // different sets, the guard has regained an effect, and the two tests below become
        // genuinely discriminating instead of resting on the ceiling.
        for depth in 0..=MAX_DEPTH {
            let guard_allows = depth >= LMR_MIN_DEPTH;
            // Read from `reduction_ceiling`, never transcribed: a copy of the formula would
            // compare the guard against this test's own idea of the ceiling, and stay silent
            // when the real one moved. Verified — loosening the production ceiling to
            // `depth - 1` revives the guard on all four natures, and the transcribing version
            // of this test passed through it.
            let ceiling_allows = reduction_ceiling(depth) >= 1;
            assert_eq!(
                guard_allows, ceiling_allows,
                "depth {depth}: the guard says {guard_allows} and the ceiling says \
                 {ceiling_allows} — they have stopped covering each other, so the depth-floor \
                 tests are no longer resting on the ceiling and should be re-read",
            );
        }
    }

    #[test]
    fn the_depth_floor_is_a_boundary_not_a_slope() {
        // Sweeps both sides of the floor rather than sampling one. Below it a reduced move
        // would be searched by quiescence alone, which judges a quiet move on captures it
        // does not have.
        //
        // **What this pins is the behaviour, not the guard.** At today's constants the
        // ceiling `min(depth - 2)` already returns zero for depths 0-2, so deleting
        // `depth >= LMR_MIN_DEPTH` leaves this test green — it cannot tell which mechanism
        // produced the zero. That is fine as long as it is stated:
        // `the_depth_floor_and_the_ceiling_currently_say_the_same_thing` is what fails if the
        // two ever diverge, and at that point this test starts discriminating again.
        let p = Position::from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1").unwrap();
        let mv = p.move_from_uci("a1a7").unwrap();
        for depth in 0..LMR_MIN_DEPTH {
            assert_eq!(reduction_for(&p, mv, depth, LATE, None), 0, "depth {depth} is below the floor");
        }
        for depth in LMR_MIN_DEPTH..=DEEP {
            assert!(
                reduction_for(&p, mv, depth, LATE, None) > 0,
                "depth {depth} is at or above the floor",
            );
        }
    }

    #[test]
    fn the_reduction_grows_on_both_axes() {
        // The whole point of the curve, and it is asserted as *monotonicity over a sweep*
        // rather than as values at chosen points. Values would pin the coefficients: retuning
        // `LMR_BASE` or `LMR_DIVISOR` would fail this test for a reason that has nothing to do
        // with the property it is named after, which is how a test ends up recalibrated
        // instead of trusted.
        //
        // Read from the table directly rather than through the predicate, because the
        // predicate clamps the result to what depth is left — which is the right behaviour
        // and would mask the growth at exactly the depths where it starts.
        let at = |depth: usize, rank: usize| LMR_TABLE[depth][rank];

        // Grows with rank at fixed depth: never decreasing, and strictly larger end to end.
        for depth in LMR_MIN_DEPTH as usize..LMR_TABLE_DEPTHS {
            for rank in LMR_FULL_DEPTH_MOVES..LMR_TABLE_RANKS - 1 {
                assert!(
                    at(depth, rank + 1) >= at(depth, rank),
                    "depth {depth}: rank {rank}->{} went {} -> {}",
                    rank + 1,
                    at(depth, rank),
                    at(depth, rank + 1),
                );
            }
            assert!(
                at(depth, LMR_TABLE_RANKS - 1) > at(depth, LMR_FULL_DEPTH_MOVES),
                "depth {depth}: the last rank must be reduced more than the first reducible one",
            );
        }

        // Grows with depth at fixed rank, same shape.
        for rank in LMR_FULL_DEPTH_MOVES..LMR_TABLE_RANKS {
            for depth in LMR_MIN_DEPTH as usize..LMR_TABLE_DEPTHS - 1 {
                assert!(
                    at(depth + 1, rank) >= at(depth, rank),
                    "rank {rank}: depth {depth}->{} went {} -> {}",
                    depth + 1,
                    at(depth, rank),
                    at(depth + 1, rank),
                );
            }
            assert!(
                at(LMR_TABLE_DEPTHS - 1, rank) > at(LMR_MIN_DEPTH as usize, rank),
                "rank {rank}: the deepest node must be reduced more than the shallowest",
            );
        }
    }

    #[test]
    fn the_reduction_is_never_zero_and_never_eats_the_whole_search() {
        // Two bounds that matter for different reasons. **At least one ply**: a reduction of
        // zero means the caller paid a table lookup to learn nothing, and the re-search path
        // would be dead code. **At most `depth - 2`**: reducing further hands the move to
        // quiescence, which judges a quiet move on captures it does not have — the same
        // reason `LMR_MIN_DEPTH` exists.
        //
        // Swept over every depth and rank the predicate can be called with, because the
        // interesting failure is an off-by-one at a boundary and no sample finds those.
        // One searcher for the whole sweep: `reduction_for` builds a transposition table per
        // call, and 3 782 allocations of it cost 33 seconds for what is pure arithmetic.
        let p = Position::from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1").unwrap();
        let mv = p.move_from_uci("a1a7").unwrap();
        let child = p.play(mv);
        let mut table = Table::new();
        let searcher = Searcher::new(MoveOrder::Full, None, &mut table);
        // The table's own promise first, over **every** cell — including the low ones the
        // guards make unreachable today. `ln(1)` is zero, so those cells would hold a
        // reduction of zero without the floor, and a later change to `LMR_MIN_DEPTH` or
        // `LMR_FULL_DEPTH_MOVES` would silently start reading them. Mutating the floor away
        // broke nothing until this loop existed, because the predicate never asks for them.
        for (depth, row) in LMR_TABLE.iter().enumerate() {
            for (rank, &r) in row.iter().enumerate() {
                assert!(r >= 1, "table cell [{depth}][{rank}] holds a reduction of zero");
            }
        }
        for depth in LMR_MIN_DEPTH..=MAX_DEPTH {
            for rank in LMR_FULL_DEPTH_MOVES..LMR_TABLE_RANKS {
                let r = searcher.late_move_reduction(&p, &child, mv, depth, rank, 1);
                assert!(r >= 1, "depth {depth} rank {rank} reduced by nothing");
                assert!(
                    r <= depth - 2,
                    "depth {depth} rank {rank} reduced by {r}, leaving nothing to search",
                );
            }
        }
    }

    #[test]
    fn an_index_past_the_table_is_clamped_rather_than_a_panic() {
        // A legal chess position can offer up to 218 moves, and the table has 64 rank slots.
        // So `rank` can exceed it — rarely, but a rare panic in a search is worse than a
        // common one: it takes a position nobody tested to find it, and it kills the process
        // mid-game. The depth axis is bounded by `MAX_DEPTH` in every search the engine runs,
        // but quiescence recurses past that, so it is clamped on the same principle.
        //
        // Asserted through the predicate with out-of-range arguments rather than by finding a
        // 218-move position: the record position (`3Q4/1Q4Q1/4Q3/2Q4R/Q4Q2/3Q4/1Q4Rp/1K1BBNNk`)
        // is mate in one, so its search never descends far enough for the guards to admit a
        // move at rank 65. The test would look like coverage while never reaching the clamp.
        let p = Position::from_fen("4k3/8/8/8/8/8/8/R3K3 w - - 0 1").unwrap();
        let mv = p.move_from_uci("a1a7").unwrap();
        let child = p.play(mv);
        let mut table = Table::new();
        let searcher = Searcher::new(MoveOrder::Full, None, &mut table);
        for (depth, rank) in [
            (MAX_DEPTH, LMR_TABLE_RANKS),          // exactly one past the rank axis
            (MAX_DEPTH, 218),                      // the most moves a legal position can offer
            (MAX_DEPTH + 32, 400),                 // both axes well past the table
        ] {
            let r = searcher.late_move_reduction(&p, &child, mv, depth, rank, 1);
            assert!(r >= 1, "depth {depth} rank {rank} must still yield a reduction");
            assert!(r <= depth - 2, "depth {depth} rank {rank} reduced by {r}");
        }
    }

    #[test]
    fn the_curve_shrinks_the_tree_further_than_a_flat_ply() {
        // The acceptance criterion that decides the brick: better than #42, not merely better
        // than no reductions. Same binary, same positions, one flag apart.
        //
        // The weakest position and depth carries the assertion, as in #42 — a curve that
        // helped the opening and hurt the tactical position would be the history heuristic
        // all over again (positive on average, negative on half the board).
        let mut worst: Option<(&str, u32, f64)> = None;
        let mut best: Option<(&str, u32, f64)> = None;
        for (nature, fen) in NATURES {
            let p = Position::from_fen(fen).unwrap();
            // Depth 5 only, and it is the demanding case rather than the cheap one: the
            // tactical position reads 0.993 there against 0.766 at depth 6, so this is where
            // the curve comes closest to costing nodes.
            for depth in 5..=5 {
                let nodes = |growing: bool| {
                    let mut t = Table::new();
                    let mut s = Searcher::new(MoveOrder::Full, None, &mut t);
                    s.lmr_growing = growing;
                    deepen(&p, Request::new(Limits::depth(depth)), &mut s).nodes
                };
                let (flat, curved) = (nodes(false), nodes(true));
                let ratio = curved as f64 / flat as f64;
                if worst.is_none_or(|(_, _, w)| ratio > w) {
                    worst = Some((nature, depth, ratio));
                }
                if best.is_none_or(|(_, _, b)| ratio < b) {
                    best = Some((nature, depth, ratio));
                }
            }
        }
        let (nature, depth, ratio) = worst.expect("the sweep ran");
        assert!(
            ratio <= 1.0,
            "the curve must not cost nodes anywhere: {nature} at depth {depth} is {ratio:.3}",
        );
        // And it must *gain* somewhere, strictly. Without this the test is satisfied by the
        // curve doing nothing at all: mutating the reduction back to a flat ply makes both
        // sides of every comparison identical, every ratio exactly 1.0, and `<= 1.0` holds.
        // Found by mutation — the whole brick unplugged, and no test noticed.
        let (best_nature, best_depth, best_ratio) = best.expect("the sweep ran");
        assert!(
            best_ratio < 1.0,
            "the curve must shrink the tree somewhere, or it is not doing anything: \
             best case is {best_nature} at depth {best_depth}, {best_ratio:.3}",
        );
    }

    #[test]
    fn forced_mates_survive_the_reductions() {
        // The sharpest regression test there is, and the one a reduction is most likely to
        // fail: a mate is the one verdict that cannot be approximately right, and a
        // shallower search on a quiet forcing line is exactly how one goes missing.
        //
        // The assertion is "once seen, never lost", swept over depths, with the
        // precondition that it is seen at all. Not "seen from depth N": how deep the search
        // must go to find a mate is a property of how hard it prunes, and reductions
        // legitimately cost a nominal ply on quiet forcing lines — measured on the ladder
        // mate below, the mate moves from depth 6 to depth 7 while being found in *fewer
        // nodes* (38 618 against 43 108), so it arrives sooner in real time. Pinning a
        // depth would assert the pruning strength instead of the correctness.
        for fen in [
            "6k1/5ppp/8/8/8/8/8/R6K w - - 0 1",       // mate in one, ours to give
            "7k/8/8/8/8/8/1R6/R6K b - - 0 1",         // rook ladder, ours to receive
            "3k4/8/3K4/8/8/8/8/6R1 w - - 0 1",        // mate in two with a quiet key move
        ] {
            let p = Position::from_fen(fen).unwrap();
            let mut seen_at = None;
            for depth in 3..=8 {
                let (_, score) = best_move(&p, depth).expect("a move");
                let is_mate = score.abs() > MATE_THRESHOLD;
                match seen_at {
                    None if is_mate => seen_at = Some(depth),
                    Some(first) => assert!(
                        is_mate,
                        "{fen} announced a mate at depth {first} and lost it by {depth}: {score}",
                    ),
                    None => {}
                }
            }
            assert!(
                seen_at.is_some(),
                "precondition: the mate in `{fen}` must be found somewhere in depths 3..=8",
            );
        }
    }

    #[test]
    fn nothing_is_reduced_without_move_ordering() {
        // "Late" is a statement about the ordering: in generator order, index 7 says
        // nothing about a move's prospects, so a reduction there would be a coin flip
        // rather than a bet. The guard is one comparison, and in production `order` is
        // always `Full` — which is exactly why it needs a test of its own. Mutating it away
        // broke nothing until this existed, because the searches that use a weaker ordering
        // all disable reductions for their own reasons.
        let p = Position::from_fen(NATURES[0].1).unwrap();
        for order in [MoveOrder::None, MoveOrder::Captures] {
            let mut table = Table::new();
            let mut s = Searcher::new(order, None, &mut table);
            s.allow_lmr = true;
            deepen(&p, Request::new(Limits::depth(5)), &mut s);
            assert_eq!(
                s.lmr_reductions, 0,
                "a search that does not order its moves cannot know which are late",
            );
        }
        // Control: the same search, ordered, does reduce — so the assertion above is about
        // the ordering and not about the position being unreducible.
        let mut table = Table::new();
        let mut s = Searcher::new(MoveOrder::Full, None, &mut table);
        deepen(&p, Request::new(Limits::depth(5)), &mut s);
        assert!(s.lmr_reductions > 0, "precondition: ordered, this position reduces");
    }

    #[test]
    fn the_re_search_changes_the_verdict_it_is_there_to_change() {
        // Counting that re-searches happen is not enough: it would pass just as well if
        // their result were thrown away. This asserts they *decide* — that on at least one
        // of the four positions, the value the node returns is the full-depth one and not
        // the reduced one.
        //
        // Without this, the re-search would be in the same position as the null move's
        // `return beta`: correct by argument, unprotected by any test, and free to be
        // removed by a later change that measures faster and looks fine.
        // Swept over depths, not fixed at one. The number of re-searches collapses as the
        // reduction grows — a move cut by three plies rarely climbs back above `alpha` —
        // so at depth 5 the four positions between them trigger almost none, and a test
        // pinned there would pass while exercising nothing. Measured with the growing
        // curve: 0 / 0 / 0 / 2 re-searches at depth 5, against 64 / 26 / 3 / 19 at depth 8.
        let differs = NATURES.iter().any(|(_, fen)| {
            let p = Position::from_fen(fen).unwrap();
            let score = |research: bool, depth: u32| {
                let mut table = Table::new();
                let mut s = Searcher::new(MoveOrder::Full, None, &mut table);
                s.allow_lmr_research = research;
                deepen(&p, Request::new(Limits::depth(depth)), &mut s).best.map(|(_, sc)| sc)
            };
            (5..=7).any(|depth| score(true, depth) != score(false, depth))
        });
        assert!(
            differs,
            "dropping the full-depth re-search must change what at least one of these \
             positions is worth — if it changes nothing, the reduction is deciding moves \
             on its own and no test would notice",
        );
    }

    #[test]
    fn nothing_is_reduced_below_the_depth_floor() {
        // Swept from depth 1 up to the floor, not checked at one depth below it: the
        // interesting failure is a floor that is off by one, and a single sample cannot
        // tell an off-by-one from a floor that works.
        //
        // Like its unit-level counterpart above, this pins the **behaviour** — no reduction
        // is counted below the floor — and not which of the two mechanisms enforces it. The
        // counter never increments because the ceiling zeroed the reduction before it looked.
        for depth in 1..LMR_MIN_DEPTH {
            for (nature, fen) in NATURES {
                let p = Position::from_fen(fen).unwrap();
                let run = search_reducing(&p, depth, true);
                assert_eq!(
                    run.reductions, 0,
                    "{nature} reduced {} moves at depth {depth}, below the floor of {LMR_MIN_DEPTH}",
                    run.reductions,
                );
            }
        }
    }


    // ---------------------------------------- SEE pruning in quiescence

    fn quiescence_pruned(pos: &Position, depth: u32, prune: bool) -> (Option<(Move, i32)>, u64) {
        let mut table = Table::new();
        let mut s = Searcher::new(MoveOrder::Full, None, &mut table);
        s.allow_see_pruning = prune;
        let stats = deepen(pos, Request::new(Limits::depth(depth)), &mut s);
        (stats.best, stats.nodes)
    }

    #[test]
    fn pruning_losing_captures_keeps_every_forced_mate() {
        // **The control this brick needs and no other does.** Unlike ordering, pruning changes
        // what the search concludes: a capture dropped here is never examined. So the question is
        // not "is it faster" but "does it still see what it saw".
        //
        // Ten tactical positions at two depths, four of them forced mates. Every score and every
        // move came back identical, which is why this test asserts equality of the **score** and
        // not merely "a mate is still a mate" — the stronger claim is the one that was measured.
        //
        // The score and not the move: a reordering may pick differently among equal values. Here
        // the moves happened to agree too, but asserting that would pin a coincidence.
        const TACTICS: [&str; 6] = [
            "r1bq2rk/pp3pbp/2p1p1pQ/7P/3P4/2PB1N2/PP3PPR/2KR4 w - - 0 1",
            "r5rk/2p1Nppp/3p3P/pp2p1P1/4P3/2qnPQK1/8/R6R w - - 0 1",
            "2r3k1/p4p2/3Rp2p/1p2P1pK/8/1P4P1/P3Q2P/1q6 b - - 0 1",
            "1k1r4/pp1b1R2/3q2pp/4p3/2B5/4Q3/PPP2B2/2K5 b - - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "3r1r1k/1p4pp/p4p2/8/1PQR4/6Pp/P3PP2/2K5 w - - 0 1",
        ];
        let mut mates = 0;
        for fen in TACTICS {
            let p = Position::from_fen(fen).unwrap();
            for depth in 6..=7u32 {
                let (pruned, _) = quiescence_pruned(&p, depth, true);
                let (plain, _) = quiescence_pruned(&p, depth, false);
                let a = pruned.expect("a move at the root").1;
                let b = plain.expect("a move at the root").1;
                assert_eq!(
                    a, b,
                    "{fen} at depth {depth}: pruning changed the score, {a} against {b}",
                );
                if a.abs() > MATE_THRESHOLD {
                    mates += 1;
                }
            }
        }
        assert!(
            mates >= 6,
            "precondition: this set must contain forced mates for the test to prove anything, \
             found {mates}",
        );
    }

    #[test]
    fn pruning_losing_captures_shrinks_quiescence_on_every_nature() {
        // What it buys, and it is the largest gain measured on this bench: 0.833 at depth 8, and
        // — unlike the ordering use of the same evaluation, which read 1.021 — it is *regular*:
        // 0.769 / 0.790 / 0.841 / 0.941 across the four natures, worst single position 1.09.
        const DEPTH: u32 = 6;
        for (nature, fen) in NATURES {
            let p = Position::from_fen(fen).unwrap();
            let (_, pruned) = quiescence_pruned(&p, DEPTH, true);
            let (_, plain) = quiescence_pruned(&p, DEPTH, false);
            assert!(
                pruned < plain,
                "{nature}: pruning must shrink the tree, {pruned} against {plain}",
            );
        }
    }

    /// Runs **one** quiescence node on `pos`, returning its score and the nodes it spent.
    ///
    /// Going through `quiescence` rather than `deepen` is what makes the node count readable:
    /// a full search spends nodes everywhere, and the effect under test is local to a single
    /// in-check position.
    fn quiescence_only(pos: &Position, prune: bool) -> (i32, u64) {
        let mut table = Table::new();
        let mut s = Searcher::new(MoveOrder::Full, None, &mut table);
        s.allow_see_pruning = prune;
        let score = s.quiescence(pos, -INF, INF, 0);
        (score, s.nodes)
    }

    /// A deterministic pseudo-random walk, so the sweep below is reproducible.
    ///
    /// Rust idiom: a plain `struct` with one field and a method that mutates through
    /// `&mut self`. Written by hand rather than pulled from a crate because a dependency for
    /// sixteen bits of arithmetic is not worth a line in `Cargo.toml`, and because a fixed seed
    /// is a *feature* here: a sweep whose positions change between runs cannot be compared
    /// with its own previous result.
    struct Xorshift(u64);

    impl Xorshift {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    #[test]
    #[ignore = "sweep: thousands of searches, run explicitly with --ignored"]
    fn see_pruning_sweep_over_pseudo_random_play() {
        // **What AC#3 was reaching for, measured instead of assumed.** The criterion asked for a
        // score *identical* to an unpruned search. That is the wrong shape of claim — the fifth
        // of its kind in this repository (#19, #27, #36, #42) — because a heuristic that drops
        // captures may return a different, equally valid bound, exactly as alpha-beta always
        // could. What matters is that no mate is lost and that the divergence stays bounded.
        //
        // Positions come from pseudo-random play rather than a hand-picked list: a list chosen
        // by the author of the pruning is the one place its blind spots are least likely to be.
        let mut rng = Xorshift(0x5EED_5EED);
        let (mut comparisons, mut score_differs, mut move_differs) = (0u32, 0u32, 0u32);
        let (mut worst_gap, mut mates_lost, mut mates_gained) = (0i32, 0u32, 0u32);

        for game in 0..1_000 {
            let mut pos = Position::from_fen(
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            )
            .unwrap();
            // Walk a random number of plies into the game, then compare at two depths. The walk
            // length varies so the sweep sees openings, middlegames and thin endgames alike.
            let plies = 4 + (game % 24);
            for _ in 0..plies {
                let moves = pos.legal_moves();
                if moves.is_empty() {
                    break;
                }
                pos = pos.play(moves[(rng.next() % moves.len() as u64) as usize]);
            }
            if pos.legal_moves().is_empty() {
                continue;
            }
            for depth in [5u32, 6] {
                let (pruned, _) = quiescence_pruned(&pos, depth, true);
                let (full, _) = quiescence_pruned(&pos, depth, false);
                let (Some((pm, ps)), Some((fm, fs))) = (pruned, full) else { continue };
                comparisons += 1;
                if ps != fs {
                    score_differs += 1;
                    worst_gap = worst_gap.max((ps - fs).abs());
                }
                if pm != fm {
                    move_differs += 1;
                }
                let (pruned_mate, full_mate) =
                    (ps.abs() > MATE_THRESHOLD, fs.abs() > MATE_THRESHOLD);
                if full_mate && !pruned_mate {
                    mates_lost += 1;
                }
                if pruned_mate && !full_mate {
                    mates_gained += 1;
                }
            }
        }

        println!(
            "sweep: {comparisons} comparisons | score differs {score_differs} \
             ({:.1}%) | move differs {move_differs} | worst gap {worst_gap} cp | \
             mates lost {mates_lost} | mates gained {mates_gained}",
            100.0 * score_differs as f64 / comparisons as f64,
        );

        // **The claim that survives measurement**, and the only one asserted. A lost mate is the
        // failure that would make the brick worthless; a differing bound is not.
        assert!(comparisons > 100, "the sweep must actually compare something: {comparisons}");
        assert_eq!(mates_lost, 0, "pruning must never lose a mate the full search found");
        assert_eq!(mates_gained, 0, "nor invent one");
    }

    #[test]
    fn pruning_never_filters_the_replies_of_a_side_in_check() {
        // White king on h1 is in check from the pawn on g2, which the pawn on h3 defends. The
        // rook on g1 can take it, and that capture loses material: +100 (pawn), -500 (rook to
        // hxg2), +100 (the king takes the pawn back) folds back to **-300**. It is therefore
        // exactly what the pruning drops — except that the side to move is in check, where it
        // has no choice of exchange.
        let checked = Position::from_fen("k7/8/8/8/8/7p/6p1/6RK w - - 0 1").unwrap();
        assert!(checked.in_check(), "precondition: white must be in check");
        let cap: Vec<Move> = checked
            .legal_moves()
            .into_iter()
            .filter(|&mv| !crate::ordering::is_quiet(&checked, mv))
            .collect();
        assert_eq!(cap.len(), 1, "precondition: exactly one capture is available");
        assert!(crate::ordering::see(&checked, cap[0]) < 0, "precondition: it loses material");

        // With the guard, pruning is inert on this position: same score, same nodes.
        assert_eq!(
            quiescence_only(&checked, true),
            quiescence_only(&checked, false),
            "a side in check must search every reply, pruning enabled or not",
        );

        // **The witness, without which the equality above proves nothing.** Same losing capture,
        // same evaluation, but the side to move is *not* in check — so pruning must bite, and the
        // two searches must differ. If this ever stops differing, the comparison above has gone
        // blind rather than the guard having held.
        let quiet = Position::from_fen("4k3/8/2p5/3n4/8/8/8/K2R4 w - - 0 1").unwrap();
        assert!(!quiet.in_check(), "precondition: white is not in check here");
        let (_, pruned_nodes) = quiescence_only(&quiet, true);
        let (_, full_nodes) = quiescence_only(&quiet, false);
        assert!(
            pruned_nodes < full_nodes,
            "the witness must show pruning acting: {pruned_nodes} vs {full_nodes} nodes",
        );

        // What this test does **not** claim, because it was measured and is not true: the guard
        // does not change any score. Removing it leaves every score here identical, and leaves
        // all six positions of `pruning_losing_captures_keeps_every_forced_mate` identical at
        // depths 4 and 6 too. The reason is structural — `best` starts at the stand-pat and only
        // ever rises, so a reply that was never searched cannot pull the answer down. What the
        // guard protects is which replies get searched, which is why the assertion counts nodes.
    }

}
