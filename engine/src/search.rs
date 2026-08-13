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
use crate::ordering::{mvv_lva, order_moves};
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
    /// Called once per **completed** iteration. An iteration cut short by the
    /// deadline is discarded, so it is never reported — announcing a depth the
    /// engine then walks back from would be worse than saying nothing.
    pub progress: Option<&'a mut dyn FnMut(&Progress)>,
}

impl<'a> Request<'a> {
    /// A search bounded only by `limits`, reporting nothing.
    pub fn new(limits: Limits) -> Request<'a> {
        Request { limits, progress: None }
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
/// A convenience wrapper over [`search_timed`], bounded by depth alone.
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

/// The search. Everything a caller can ask for travels in [`Request`].
pub fn search(pos: &Position, mut request: Request) -> SearchStats {
    let started = Instant::now();
    let limits = &request.limits;
    let mut searcher = Searcher::new(true, limits.deadline);
    let max_depth = limits.max_depth.max(1);

    let mut best: Option<(Move, i32)> = None;
    let mut completed = 0;
    for depth in 1..=max_depth {
        // Order the previous iteration's best move first — the whole point of
        // deepening, since a good first move causes early cutoffs below.
        let pv = best.map(|(mv, _)| mv);
        let result = searcher.root(pos, depth, pv);

        if searcher.aborted {
            // Ran out of time mid-iteration: discard it and keep the last complete one.
            // Unless there is none — a capture-rich position can spend more than the
            // 2048 nodes between clock checks inside iteration 1 alone, so even the
            // first iteration can be cut short. The partial root result is then all we
            // have, and it is still a legal move: `root` seeds `best_move` with
            // `moves[0]` before searching anything. Returning it beats returning
            // nothing, which the UCI layer would report as `bestmove 0000` — an
            // illegal move, and an instant forfeit in any arena.
            if best.is_none() {
                best = result;
            }
            break;
        }
        best = result;
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

/// Carries the state shared by the whole search: the node counter, the ordering
/// switch, and the time deadline / abort flag. Keeping `root` and `negamax` as
/// methods on one struct keeps them in sync.
struct Searcher {
    nodes: u64,
    ordered: bool,
    deadline: Option<Instant>,
    aborted: bool,
    /// Lives for the whole `search_timed` call, so iteration N+1 reuses what
    /// iteration N learned. That reuse is a large part of why the table pays.
    table: Table,
}

impl Searcher {
    fn new(ordered: bool, deadline: Option<Instant>) -> Searcher {
        Searcher { nodes: 0, ordered, deadline, aborted: false, table: Table::new() }
    }

    /// Set `aborted` if the deadline has passed. Checked only every so often
    /// (reading the clock on every node would dominate the search cost).
    fn check_time(&mut self) {
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
    fn root(&mut self, pos: &Position, depth: u32, pv_move: Option<Move>) -> Option<(Move, i32)> {
        self.nodes += 1;
        let mut moves = pos.legal_moves();
        if moves.is_empty() {
            return None;
        }
        if self.ordered {
            order_moves(pos, &mut moves);
        }
        if let Some(pv) = pv_move {
            if let Some(i) = moves.iter().position(|&mv| mv == pv) {
                moves.swap(0, i);
            }
        }

        let mut best_move = moves[0];
        let mut best_score = -INF;
        let mut alpha = -INF;
        for &mv in &moves {
            let score = -self.negamax(&pos.play(mv), depth.saturating_sub(1), -INF, -alpha, 1);
            if self.aborted {
                break; // time is up; this iteration will be discarded by the caller
            }
            if score > best_score {
                best_score = score;
                best_move = mv;
            }
            if score > alpha {
                alpha = score;
            }
        }
        Some((best_move, best_score))
    }

    /// Negamax with alpha-beta. Returns the value of `pos` from the side-to-move
    /// perspective. `ply` is the distance from the root, used only to score mates.
    fn negamax(&mut self, pos: &Position, depth: u32, mut alpha: i32, beta: i32, ply: i32) -> i32 {
        self.nodes += 1;
        self.check_time();
        if self.aborted {
            return 0; // value ignored: the whole iteration is thrown away
        }

        // Out of depth, but not necessarily out of danger: hand over to quiescence
        // rather than evaluating a position that may be mid-exchange. Checked before
        // generating moves, since quiescence generates them itself — and it is the one
        // that detects mate and stalemate from here on.
        if depth == 0 {
            return self.quiescence(pos, alpha, beta, ply);
        }

        // Have we been here before, by another move order or in an earlier iteration?
        let key = pos.hash();
        let hit = self.table.probe(key, depth, alpha, beta, ply);
        if let Some(score) = hit.cutoff {
            return score;
        }

        let mut moves = pos.legal_moves();
        if moves.is_empty() {
            // Terminal: checkmate (side to move loses) or stalemate (draw).
            return if pos.in_check() { -(MATE - ply) } else { 0 };
        }
        if self.ordered {
            order_moves(pos, &mut moves);
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
        for &mv in &moves {
            let score = -self.negamax(&pos.play(mv), depth - 1, -beta, -alpha, ply + 1);
            if self.aborted {
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
                break; // Beta cutoff: this branch cannot improve the result.
            }
        }

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
    fn quiescence(&mut self, pos: &Position, mut alpha: i32, beta: i32, ply: i32) -> i32 {
        self.nodes += 1;
        self.check_time();
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
        if self.ordered {
            order_moves(pos, &mut moves);
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

    // A single fixed-depth pass — no deepening, no clock — driving `Searcher` directly.
    // The oracle these tests compare against: it gives the verdict a plain depth-D
    // search reaches, and, with `ordered` flipped, isolates what move ordering does.
    fn search_fixed(pos: &Position, depth: u32, ordered: bool) -> SearchStats {
        let mut searcher = Searcher::new(ordered, None);
        let best = searcher.root(pos, depth, None);
        SearchStats {
            best,
            depth,
            nodes: searcher.nodes,
            table_key_match_rate: searcher.table.key_match_rate(),
            table_cutoff_rate: searcher.table.cutoff_rate(),
        }
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
            let ordered = search_fixed(&p, 3, true).best.map(|(_, s)| s);
            let unordered = search_fixed(&p, 3, false).best.map(|(_, s)| s);
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
        let ordered = search_fixed(&p, 3, true);
        let unordered = search_fixed(&p, 3, false);
        assert!(
            ordered.nodes < unordered.nodes,
            "ordering should prune: ordered {} vs unordered {}",
            ordered.nodes,
            unordered.nodes
        );
    }

    #[test]
    fn a_repeated_search_reuses_the_table() {
        // The same position, twice, on one `Searcher`. The second pass asks the very
        // questions the first one answered, so almost all of it should come out of the
        // table rather than the tree.
        let p = Position::initial();
        let mut searcher = Searcher::new(true, None);
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
            let direct = search_fixed(&p, 4, true);
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
        assert_eq!(wrapped, search_fixed(&p, 4, true).best.map(|(_, s)| s));
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
        Searcher::new(true, None).quiescence(pos, -INF, INF, 0)
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
        // quiet position depth 1 is owed anyway: `check_time` only reads the clock
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
                        limits: Limits::until(Instant::now() + Duration::from_millis(budget)),
                        progress: Some(&mut record),
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
