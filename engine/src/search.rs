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

use crate::evaluation::evaluate;
use crate::ordering::{mvv_lva, order_moves, KillerSlots, Killers};
use crate::position::{Move, Piece, Position};
use crate::transposition::{Bound, Table};
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
        let searcher = Searcher::new(MoveOrder::Full, request.limits.deadline, &mut self.table);
        deepen(pos, request, searcher)
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
fn deepen(pos: &Position, mut request: Request, mut searcher: Searcher) -> SearchStats {
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
    fn negamax(&mut self, pos: &Position, depth: u32, mut alpha: i32, beta: i32, ply: i32) -> i32 {
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
        for &mv in &moves {
            let score = -self.negamax(&pos.play(mv), depth - 1, -beta, -alpha, ply + 1);
            if self.aborted {
                self.path.pop();
                return 0; // nothing worth caching: the value is a placeholder
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
            searcher,
        );
        (stats, reported)
    }

    // A single fixed-depth pass — no deepening, no clock — driving `Searcher` directly.
    // The oracle these tests compare against: it gives the verdict a plain depth-D
    // search reaches, and, with `ordered` flipped, isolates what move ordering does.
    fn search_fixed(pos: &Position, depth: u32, order: MoveOrder) -> SearchStats {
        let mut table = Table::new();
        let mut searcher = Searcher::new(order, None, &mut table);
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
        let start = Position::initial();
        let mut engine = Engine::new();
        let first = engine.search(&start, Request::new(Limits::depth(6)));

        // Play a move each side, as a game would.
        let (mv, _) = first.best.expect("a move");
        let after = start.play(mv);
        let (reply, _) = best_move(&after, 3).expect("a reply");
        let next = after.play(reply);

        let carried = engine.search(&next, Request::new(Limits::depth(6)));
        let fresh = Engine::new().search(&next, Request::new(Limits::depth(6)));

        assert_eq!(carried.best.map(|(_, s)| s), fresh.best.map(|(_, s)| s), "same verdict");
        assert!(
            carried.nodes < fresh.nodes,
            "a carried-over table must save work: {} nodes against {} from scratch",
            carried.nodes,
            fresh.nodes,
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
        for fen in [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 3",
        ] {
            let p = Position::from_fen(fen).unwrap();
            let deep = search_timed(&p, Limits::depth(4));
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
}
