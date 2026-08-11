//! The search: pick a move for a position.
//!
//! Negamax with alpha-beta pruning, to a fixed depth, guided by the material
//! [`evaluate`](crate::evaluation::evaluate) and sped up by move
//! [`ordering`](crate::ordering). Move ordering does not change the result — only
//! the number of nodes visited.

use crate::evaluation::evaluate;
use crate::ordering::order_moves;
use crate::position::{Move, Position};

/// A mate score, large enough to dominate any material balance. The distance to
/// mate (`ply`) is subtracted so the search prefers shorter mates, and so a mate
/// is always distinguishable from a mere material advantage.
pub const MATE: i32 = 30_000;

// A bound strictly above any reachable score (including `MATE`), used as the
// initial alpha/beta window.
const INF: i32 = 40_000;

/// The outcome of a search: the chosen move and its score (from the side-to-move
/// perspective), plus how many nodes were visited — the metric that move ordering
/// is meant to shrink.
pub struct SearchStats {
    pub best: Option<(Move, i32)>,
    pub nodes: u64,
}

/// Search `pos` to `depth` plies (with move ordering) and report the best move,
/// its score, and the node count.
pub fn search(pos: &Position, depth: u32) -> SearchStats {
    let mut searcher = Searcher { nodes: 0, ordered: true };
    let best = searcher.root(pos, depth);
    SearchStats { best, nodes: searcher.nodes }
}

/// The best move for `pos` at `depth`, or `None` at a terminal root. Thin wrapper
/// over [`search`].
pub fn best_move(pos: &Position, depth: u32) -> Option<(Move, i32)> {
    search(pos, depth).best
}

/// Carries the state shared by the whole search: the node counter and whether to
/// order moves. Making `root` and `negamax` methods on one struct keeps them in
/// sync — they share the counter and the ordering switch instead of passing them
/// around by hand.
struct Searcher {
    nodes: u64,
    ordered: bool,
}

impl Searcher {
    /// The root: like [`Searcher::negamax`] but it tracks the chosen move and does
    /// not cut off (there is no `beta` above the root).
    fn root(&mut self, pos: &Position, depth: u32) -> Option<(Move, i32)> {
        self.nodes += 1;
        let mut moves = pos.legal_moves();
        if moves.is_empty() {
            return None;
        }
        if self.ordered {
            order_moves(pos, &mut moves);
        }

        let mut best_move = moves[0];
        let mut best_score = -INF;
        let mut alpha = -INF;
        for &mv in &moves {
            // Copy-make: `play` returns the child; `pos` stays untouched. The child
            // is searched from the opponent's side, hence the negation and window.
            let score = -self.negamax(&pos.play(mv), depth.saturating_sub(1), -INF, -alpha, 1);
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
        let mut moves = pos.legal_moves();
        if moves.is_empty() {
            // Terminal: checkmate (side to move loses) or stalemate (draw).
            return if pos.in_check() { -(MATE - ply) } else { 0 };
        }
        if depth == 0 {
            return evaluate(pos);
        }
        if self.ordered {
            order_moves(pos, &mut moves);
        }

        let mut best = -INF;
        for &mv in &moves {
            let score = -self.negamax(&pos.play(mv), depth - 1, -beta, -alpha, ply + 1);
            if score > best {
                best = score;
            }
            if best > alpha {
                alpha = best;
            }
            if alpha >= beta {
                break; // Beta cutoff: this branch cannot improve the result.
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::position::{Color, Piece, Status};

    // A search with ordering disabled — used only to prove that ordering keeps the
    // score identical (correctness) while cutting the node count (the payoff).
    fn search_unordered(pos: &Position, depth: u32) -> SearchStats {
        let mut searcher = Searcher { nodes: 0, ordered: false };
        let best = searcher.root(pos, depth);
        SearchStats { best, nodes: searcher.nodes }
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
            let ordered = search(&p, 3).best.map(|(_, s)| s);
            let unordered = search_unordered(&p, 3).best.map(|(_, s)| s);
            assert_eq!(ordered, unordered, "score must not depend on ordering ({fen})");
        }
    }

    #[test]
    fn ordering_prunes_more_nodes() {
        // On a middlegame position, ordering must visit strictly fewer nodes.
        let p = Position::from_fen(
            "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 3",
        )
        .unwrap();
        let ordered = search(&p, 4);
        let unordered = search_unordered(&p, 4);
        println!(
            "MVV-LVA @ depth 4: {} nodes vs {} unordered ({:.1}x fewer)",
            ordered.nodes,
            unordered.nodes,
            unordered.nodes as f64 / ordered.nodes as f64
        );
        assert!(
            ordered.nodes < unordered.nodes,
            "ordering should prune: ordered {} vs unordered {}",
            ordered.nodes,
            unordered.nodes
        );
    }
}
