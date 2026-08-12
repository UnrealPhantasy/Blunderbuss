//! The `Position` interface — the **boundary** between our own engine code and
//! the borrowed rules layer (`cozy-chess`).
//!
//! Every other part of the engine (search, evaluation, move ordering) talks to
//! this module only, never directly to `cozy-chess`. That is what keeps the
//! dependency on the rules library **confined to a single place**.
//!
//! # Copy-make (not make-unmake)
//!
//! `cozy-chess` offers no move undo: its `Board` can *play* a move but not
//! *unplay* it. So we apply a move by **cloning** the position and playing on
//! the clone; the original stays untouched. This is the "copy-make" model, as
//! opposed to "make-unmake".

// `Board` is cozy-chess's board. We also re-export the types below under our own
// names.
use cozy_chess::{Board, GameStatus};

// --- Types re-exposed at the boundary ------------------------------------
//
// Rust idiom: a re-export (`pub use`) exposes a type from another crate under
// our own path without wrapping it. Wrapping `Move`/`Color` in home-grown
// `newtype`s would be the "clean" seam of the boundary, but with no immediate
// benefit — we'll do it if the need arises.

/// A move (from-square, to-square, optional promotion).
pub use cozy_chess::Move;

/// The side to move. (cozy-chess's variants are `White` / `Black`.)
pub use cozy_chess::Color;

/// A piece type (pawn … king), used to read material.
pub use cozy_chess::Piece;

/// A board square (a1 … h8), used to read what sits on a move's endpoints.
pub use cozy_chess::Square;

/// The status of a position, purely from the rules' point of view.
///
/// Deliberately coarse for this first brick: it mirrors what `cozy-chess` can
/// decide on its own. Threefold repetition and insufficient material — which
/// need history or home-grown analysis — will come later, driven by the engine.
///
/// `#[derive(...)]` asks the compiler to generate trivial implementations:
/// `Debug` (debug printing), `Clone` + `Copy` (the enum fits in one byte, we
/// copy it freely), and equality.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    /// The game goes on.
    Ongoing,
    /// The side **to move** is checkmated (it has lost).
    Checkmate,
    /// Draw: stalemate, or the fifty-move rule (whatever `cozy-chess` decides).
    Draw,
}

// --- The position ---------------------------------------------------------

/// A full chess position.
///
/// Rust idiom: this is a *newtype* — a single-field `struct` wrapping `Board`.
/// The field is **private** (no `pub`), so the outside cannot reach the `Board`
/// directly: it must go through the methods below. That is how the boundary is
/// kept airtight.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Position(Board);

impl Position {
    /// The starting position of a chess game.
    pub fn initial() -> Position {
        // `Board::default()` is the standard starting position.
        Position(Board::default())
    }

    /// Builds a position from a FEN string.
    ///
    /// Rust idiom: the `Result<_, _>` return type makes failure explicit — the
    /// caller *must* decide what to do with an invalid FEN, it cannot ignore it
    /// by accident. `?` propagates the error if parsing fails. The `false` in
    /// `from_fen` disables Chess960 mode (out of scope).
    pub fn from_fen(fen: &str) -> Result<Position, cozy_chess::FenParseError> {
        Ok(Position(Board::from_fen(fen, false)?))
    }

    /// The side to move.
    pub fn side_to_move(&self) -> Color {
        self.0.side_to_move()
    }

    /// Is the side to move in check?
    pub fn in_check(&self) -> bool {
        // `checkers()` returns the set (bitboard) of pieces giving check; it is
        // empty when there is no check.
        !self.0.checkers().is_empty()
    }

    /// The status of the position (see [`Status`]).
    pub fn status(&self) -> Status {
        // Rust idiom: `match` is exhaustive — the compiler refuses to build if a
        // `GameStatus` variant is left out. No hidden default arm.
        match self.0.status() {
            GameStatus::Ongoing => Status::Ongoing,
            GameStatus::Won => Status::Checkmate,
            GameStatus::Drawn => Status::Draw,
        }
    }

    /// The Zobrist hash key of the position (for a future transposition table).
    pub fn hash(&self) -> u64 {
        self.0.hash()
    }

    /// How many pieces of the given `color` and `piece` type are on the board.
    ///
    /// Kept here so the rest of the engine (e.g. evaluation) can read material
    /// without touching `cozy-chess` directly — the boundary stays confined to
    /// this module.
    pub fn count(&self, color: Color, piece: Piece) -> u32 {
        (self.0.colors(color) & self.0.pieces(piece)).len()
    }

    /// The piece sitting on `square`, if any (regardless of color).
    ///
    /// Used by move ordering to read a capture's victim and attacker while
    /// keeping `cozy-chess` confined to this module.
    pub fn piece_on(&self, square: Square) -> Option<Piece> {
        self.0.piece_on(square)
    }

    /// This move rendered as a UCI string (e.g. `"e2e4"`, `"e7e8q"`, `"e1g1"`).
    ///
    /// Goes through cozy-chess's UCI helper, which converts the crate's internal
    /// king-captures-rook castling notation (`e1h1`) to the standard UCI form
    /// (`e1g1`). Keeping this here confines that quirk to the boundary.
    pub fn move_to_uci(&self, mv: Move) -> String {
        cozy_chess::util::display_uci_move(&self.0, mv).to_string()
    }

    /// Parses a UCI move string for this position, or `None` if it does not
    /// parse. Also converts UCI castling (`e1g1`) to the internal notation.
    pub fn move_from_uci(&self, s: &str) -> Option<Move> {
        cozy_chess::util::parse_uci_move(&self.0, s).ok()
    }

    /// All legal moves in the position.
    ///
    /// First version: we fill a `Vec`. Move ordering and staged generation (not
    /// producing everything at once) belong to the engine and will come later —
    /// they are out of scope here.
    pub fn legal_moves(&self) -> Vec<Move> {
        let mut moves = Vec::new();
        // cozy-chess idiom: generation is a *visitor*. We pass it an `FnMut`
        // closure that receives moves grouped by piece (`PieceMoves`, iterable
        // as `Move`). It returns a `bool`: `true` would stop generation, `false`
        // keeps it going to the end.
        self.0.generate_moves(|group| {
            moves.extend(group);
            false
        });
        moves
    }

    /// Plays a move and returns the **new** position; `self` is left untouched
    /// (copy-make).
    ///
    /// `play_unchecked` does not validate legality: that is both correct **and**
    /// fast here, because the moves come from `legal_moves()`, hence already
    /// legal. For a move of uncertain origin, see [`Position::try_play`].
    pub fn play(&self, mv: Move) -> Position {
        let mut board = self.0.clone();
        board.play_unchecked(mv);
        Position(board)
    }

    /// Like [`Position::play`], but validates the move's legality and returns an
    /// error if it is illegal (useful for a move coming from outside, e.g. UCI).
    pub fn try_play(&self, mv: Move) -> Result<Position, cozy_chess::IllegalMoveError> {
        let mut board = self.0.clone();
        board.try_play(mv)?;
        Ok(Position(board))
    }
}

/// *perft* ("performance test"): counts the leaves of the legal-move tree at a
/// fixed depth. Here it is not a speed measure but a **correctness test** of the
/// generation + application wiring: the counts from the starting position are
/// known and invariant (20, 400, 8 902, …).
pub fn perft(position: &Position, depth: u32) -> u64 {
    if depth == 0 {
        return 1;
    }
    let moves = position.legal_moves();
    if depth == 1 {
        // Classic optimization ("bulk counting"): one ply from the end, the
        // number of leaves is exactly the number of legal moves.
        return moves.len() as u64;
    }
    // Rust idiom: iterator chain. For each move, descend one ply into the child
    // position (copy-make), and sum the counts.
    moves
        .iter()
        .map(|&mv| perft(&position.play(mv), depth - 1))
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn perft_starting_position() {
        let p = Position::initial();
        assert_eq!(perft(&p, 1), 20);
        assert_eq!(perft(&p, 2), 400);
        assert_eq!(perft(&p, 3), 8_902);
        assert_eq!(perft(&p, 4), 197_281);
    }

    #[test]
    fn perft_depth_5() {
        // Optional but cheap (bulk counting): a stronger validation.
        let p = Position::initial();
        assert_eq!(perft(&p, 5), 4_865_609);
    }

    #[test]
    fn copy_make_leaves_original_untouched() {
        let p = Position::initial();
        let fingerprint = p.hash();
        let moves = p.legal_moves();
        assert_eq!(moves.len(), 20);

        // Playing a move must produce a different position…
        let child = p.play(moves[0]);
        assert_ne!(child.hash(), fingerprint);
        // …without touching the original.
        assert_eq!(p.hash(), fingerprint, "copy-make: the original must stay intact");
        assert_eq!(p.legal_moves().len(), 20);
    }

    #[test]
    fn scholars_mate() {
        // Final position of scholar's mate (1.e4 e5 2.Bc4 Nc6 3.Qh5 Nf6?? 4.Qxf7#).
        // Black to move, and checkmated.
        let fen = "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4";
        let p = Position::from_fen(fen).unwrap();
        assert_eq!(p.side_to_move(), Color::Black);
        assert!(p.in_check());
        assert_eq!(p.status(), Status::Checkmate);
        assert!(p.legal_moves().is_empty());
    }

    #[test]
    fn known_stalemate() {
        // Classic stalemate: black king on h8, white queen f7, white king g6.
        // Black to move has no legal move and is not in check.
        let fen = "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1";
        let p = Position::from_fen(fen).unwrap();
        assert!(!p.in_check());
        assert!(p.legal_moves().is_empty());
        assert_eq!(p.status(), Status::Draw);
    }

    #[test]
    fn invalid_fen_is_an_error() {
        // Must not panic: just return an error.
        assert!(Position::from_fen("this is not a FEN").is_err());
    }

    #[test]
    fn board_size_is_documented() {
        // Cost of copy-make: we clone a Board on every move. We measure its real
        // size rather than estimate it.
        // Measured 2026-08-11 with cozy-chess 0.3.4: 104 bytes — negligible.
        let size = std::mem::size_of::<Board>();
        println!("size_of::<cozy_chess::Board>() = {size} bytes");
        assert!(size > 0);
    }

    #[test]
    fn uci_move_roundtrips_including_castling() {
        // A normal move round-trips through UCI.
        let start = Position::initial();
        let e2e4 = start.move_from_uci("e2e4").expect("e2e4 parses");
        assert_eq!(start.move_to_uci(e2e4), "e2e4");

        // Castling: cozy-chess stores kingside castling as `e1h1` internally, but
        // UCI is `e1g1`. Both directions must speak the standard UCI form.
        let p = Position::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1").unwrap();
        let castle = p.move_from_uci("e1g1").expect("e1g1 parses as a castle");
        assert_eq!(p.move_to_uci(castle), "e1g1");
        let after = p.try_play(castle).expect("kingside castling is legal here");
        assert_eq!(after.piece_on(Square::G1), Some(Piece::King));
    }
}
