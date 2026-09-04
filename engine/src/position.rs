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

/// A set of squares — one bit per square, `a1` in bit 0.
///
/// The static exchange evaluation in [`crate::ordering`] needs to reason about *hypothetical*
/// occupations: "who would attack this square once these two pieces have left it". That is a
/// question about the rules, so the primitive belongs here; what to do with the answer is move
/// ordering, and stays in maison.
///
/// **Why a type of our own rather than re-exporting the dependency's.** Every other borrowed type
/// this module re-exports — `Square`, `Piece`, `Color` — is a *chess concept*, and its shape is
/// forced by the rules. A bitboard is a *representation*: sixty-four bits in some order, chosen
/// by the library. Re-exporting it would put a dependency's representation in our public
/// signatures, and [`Position::pawns`] had already declined to do exactly that, in a comment
/// calling it "the rule the rest of this type follows". Following it here costs one wrapper.
///
/// Rust idiom: a **newtype** is a struct wrapping a single value in order to give it a distinct
/// type. It costs nothing at run time — the wrapper has the same layout as what it holds, and the
/// compiler erases it — so this is a boundary drawn in the type system and paid for at compile
/// time only. The field is private, which is what actually closes the boundary: callers reach the
/// contents only through the methods below.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct SquareSet(cozy_chess::BitBoard);

impl SquareSet {
    /// The set containing `square` alone.
    pub fn of(square: Square) -> SquareSet {
        SquareSet(square.bitboard())
    }

    /// Whether the set holds no square at all.
    pub fn is_empty(self) -> bool {
        self.0.is_empty()
    }

    /// One square of the set, or `None` if it is empty.
    ///
    /// *Which* square is unspecified on purpose: every caller here either loops until the set is
    /// empty or has already narrowed it to a single piece type, so depending on the order would
    /// be depending on the representation this type exists to hide.
    pub fn next_square(self) -> Option<Square> {
        self.0.next_square()
    }
}

// Rust idiom: implementing `std::ops::BitAnd` is what makes `a & b` work on our own type. The
// operators are traits like any other, so a newtype can offer exactly the ones that make sense
// for it — intersection and toggling here, and deliberately not, say, arithmetic.
impl std::ops::BitAnd for SquareSet {
    type Output = SquareSet;
    fn bitand(self, other: SquareSet) -> SquareSet {
        SquareSet(self.0 & other.0)
    }
}

impl std::ops::BitXor for SquareSet {
    type Output = SquareSet;
    fn bitxor(self, other: SquareSet) -> SquareSet {
        SquareSet(self.0 ^ other.0)
    }
}

impl std::ops::BitXorAssign for SquareSet {
    fn bitxor_assign(&mut self, other: SquareSet) {
        self.0 ^= other.0;
    }
}

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

/// The largest number of legal moves a chess position is known to admit.
///
/// Held by a constructed position rather than by anything reachable from the opening, which is why
/// it is a bound and not an average — real positions offer about 39. Used to size the move buffer
/// once instead of growing it, and as the ceiling of the stack array `ordering::order_moves` sorts
/// in. `legal_moves_never_outgrows_the_buffer_it_reserved` pins it to that position, so lowering it
/// by one goes red.
pub const MAX_LEGAL_MOVES: usize = 218;

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
    /// The position as FEN. Written so a sweep can **name** the position it disagreed on, and a
    /// defect be reproduced from a string rather than from a seed and a walk.
    ///
    /// It went uncalled from the day it was written until #69, where the sweep it exists for had
    /// been red for three days and could not say on which positions.
    #[cfg(test)]
    pub fn to_fen(&self) -> String {
        format!("{}", self.0)
    }

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

    /// Every pawn of `color`, as a bitboard — one bit per square, `a1` in bit 0.
    ///
    /// A bitboard rather than a square-by-square scan because of what reads it: deciding
    /// whether a pawn is passed asks "is there any enemy pawn on my file or the two beside
    /// it, on any rank ahead of me". As a scan that is up to 21 lookups per pawn; against a
    /// precomputed mask it is one `&` and a comparison. `evaluate` runs at every leaf, and
    /// this engine has already abandoned one term (king safety, #29) for costing 0.23 µs a
    /// node.
    ///
    /// Returned as a plain `u64` rather than `cozy_chess::BitBoard` so the dependency stays
    /// confined to this module, which is the rule the rest of this type follows. A bitboard
    /// is a *fact about the position*, so borrowing it sits on the "borrow" side of the
    /// project's guiding principle; what has a design space — which pawns count as passed,
    /// and what each is worth — is written by hand in `evaluation`.
    pub fn pawns(&self, color: Color) -> u64 {
        (self.0.colors(color) & self.0.pieces(Piece::Pawn)).0
    }

    /// The piece sitting on `square`, if any (regardless of color).
    ///
    /// Used by move ordering to read a capture's victim and attacker while
    /// keeping `cozy-chess` confined to this module.
    ///
    /// **The occupancy test in front is the whole point of this wrapper.** The borrowed
    /// implementation is `Piece::ALL.iter().find(|p| self.pieces(p).has(square))`, which walks up
    /// to six bitboards — and walks *all six* precisely when the answer is `None`. That is the
    /// common case here: `evaluate` visits all 64 squares of the board on every call, and about
    /// half of them are empty in an opening and four fifths in an endgame. One test against the
    /// occupancy answers those in a single read instead of six.
    ///
    /// It cannot change an answer. `occupied` is the union of the six piece boards by
    /// construction, which `the_occupancy_is_exactly_the_union_of_the_piece_boards` asserts rather
    /// than assumes — if the two ever parted company this would return `None` for occupied squares
    /// and every evaluation would quietly lose material, with no panic anywhere.
    pub fn piece_on(&self, square: Square) -> Option<Piece> {
        if !self.0.occupied().has(square) {
            return None;
        }
        self.0.piece_on(square)
    }

    /// The colour of the piece on `square`, if any.
    ///
    /// Companion to [`Position::piece_on`], used by evaluation to read whose
    /// piece sits where — again keeping `cozy-chess` confined to this module.
    pub fn color_on(&self, square: Square) -> Option<Color> {
        self.0.color_on(square)
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

    /// Every square currently holding a piece.
    pub fn occupied(&self) -> SquareSet {
        SquareSet(self.0.occupied())
    }

    /// The squares holding `piece` of `color`.
    pub fn pieces_of(&self, color: Color, piece: Piece) -> SquareSet {
        SquareSet(self.0.pieces(piece) & self.0.colors(color))
    }

    /// Which pieces of `color` attack `square`, under a **hypothetical** occupation.
    ///
    /// `occupied` is passed rather than read from the board because that is the whole point: a
    /// static exchange evaluation removes pieces one by one as they capture, and each removal can
    /// **uncover** a sliding attacker behind the one that left. Asking the real board would miss
    /// every battery — a rook behind a rook, a queen behind a bishop — and those are exactly the
    /// exchanges an exchange evaluation exists to get right.
    ///
    /// Pawn attacks are looked up *from the target square with the opposite colour*: the squares
    /// a white pawn attacks from are the squares a black pawn attacks, mirrored. That inversion
    /// is the one place this function is easy to get backwards, so it is stated rather than
    /// implied.
    pub fn attackers(&self, square: Square, color: Color, occupied: SquareSet) -> SquareSet {
        use cozy_chess::{get_bishop_moves, get_king_moves, get_knight_moves, get_pawn_attacks,
                         get_rook_moves};
        let mine = self.0.colors(color);
        let bishops = self.0.pieces(Piece::Bishop) | self.0.pieces(Piece::Queen);
        let rooks = self.0.pieces(Piece::Rook) | self.0.pieces(Piece::Queen);
        // `occupied.0` is the one place the wrapper is opened, and it is inside the module that
        // owns it — which is the point of a private field rather than a public one.
        let hypothetical = occupied.0;
        SquareSet(
            ((get_pawn_attacks(square, !color) & self.0.pieces(Piece::Pawn))
                | (get_knight_moves(square) & self.0.pieces(Piece::Knight))
                | (get_king_moves(square) & self.0.pieces(Piece::King))
                | (get_bishop_moves(square, hypothetical) & bishops)
                | (get_rook_moves(square, hypothetical) & rooks))
                & mine
                & hypothetical,
        )
    }

    /// How many squares the `piece` standing on `square` commands, excluding squares occupied by
    /// its own side.
    ///
    /// **One attack lookup and one popcount, and it takes the square rather than the piece type
    /// on purpose.** The evaluation already walks the 64 squares once; asking per square lets the
    /// term ride that walk instead of adding a second pass. A king-safety term was abandoned on
    /// this engine for costing 0.23 µs per node, so the shape of the call matters as much as what
    /// it computes.
    ///
    /// What is counted is *pseudo-legal* reach: a pinned bishop still counts the squares it looks
    /// at. That is the standard formulation and it is deliberate — a pin is a fact about one move,
    /// while mobility says how much a position constrains a side, and paying for legality here
    /// would cost more than the term is worth.
    pub fn mobility_from(&self, square: Square, piece: Piece, color: Color) -> u32 {
        use cozy_chess::{get_bishop_moves, get_king_moves, get_knight_moves, get_pawn_attacks,
                         get_rook_moves};
        let occupied = self.0.occupied();
        let reach = match piece {
            Piece::Knight => get_knight_moves(square),
            Piece::Bishop => get_bishop_moves(square, occupied),
            Piece::Rook => get_rook_moves(square, occupied),
            // A queen is a rook and a bishop on one square, which is how the lookups are built —
            // two probes rather than a third table.
            Piece::Queen => get_rook_moves(square, occupied) | get_bishop_moves(square, occupied),
            Piece::King => get_king_moves(square),
            Piece::Pawn => get_pawn_attacks(square, color),
        };
        (reach & !self.0.colors(color)).len()
    }

    pub fn legal_moves(&self) -> Vec<Move> {
        // **Reserved rather than grown**, and the size is not a guess: [`MAX_LEGAL_MOVES`] is the
        // largest number of legal moves any chess position admits, so this vector never
        // reallocates. `Vec::new()` followed by `extend` re-allocates and copies as it crosses 4,
        // 8, 16, 32 and 64 moves — in the hottest loop in the engine, at about 39 moves a position.
        //
        // Rust idiom: `with_capacity` allocates the buffer once at the requested size and leaves
        // the length at zero, so the vector still knows how many moves it actually holds. Capacity
        // is the size of the allocation; length is what is in it.
        let mut moves = Vec::with_capacity(MAX_LEGAL_MOVES);
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

    /// The moves quiescence can be interested in: everything landing on an enemy piece, plus
    /// everything landing on a promotion rank.
    ///
    /// **A superset, on purpose, and that is what makes it safe.** The caller still applies its own
    /// filter — today `mvv_lva(..) > 0 || is_queen_promotion(..)` — so this only has to be a set no
    /// wanted move can fall outside of. A capture lands on an enemy piece; a promotion lands on the
    /// last rank. Nothing else is claimed: a rook stepping onto the eighth rank comes through here
    /// and is dropped by the caller, which costs a filtered move and buys the guarantee that the
    /// result is exactly what generating everything and filtering would have produced.
    ///
    /// **Why it exists.** `legal_moves` materialises every legal move — about 39 a position — and
    /// quiescence then throws most of them away. It did so for one reason: an empty list is how
    /// mate and stalemate were told apart from a quiet position. That verdict is needed at a
    /// vanishing fraction of nodes, and [`Position::has_any_legal_move`] answers it far more
    /// cheaply on the few where this returns nothing.
    ///
    /// The *order* is the same as the corresponding subsequence of [`Position::legal_moves`], since
    /// the mask restricts which moves the generator emits and not the order it walks its pieces in.
    /// That is not a detail: move ordering is stable, so a different order here would change which
    /// nodes the search visits. `tactical_moves_are_the_filtered_legal_moves_in_the_same_order`
    /// pins it.
    pub fn tactical_moves(&self) -> Vec<Move> {
        let us = self.0.side_to_move();
        let them = self.0.colors(!us);
        let promotion_rank = match us {
            Color::White => cozy_chess::Rank::Eighth,
            Color::Black => cozy_chess::Rank::First,
        };
        // **Our own rooks belong in the target set, and that is not a mistake.** cozy-chess spells
        // castling as the king *capturing its own rook* (`e1h1`), so a castle lands on a friendly
        // square — and `ordering::mvv_lva` therefore reads a rook on the destination and scores it
        // as a capture. Quiescence consequently searches castling today.
        //
        // That is very probably a defect: a castle is not an exchange and has nothing to resolve
        // in a quiescence search. But **fixing it here would change which nodes are visited**,
        // which is exactly what this brick promises not to do, and it would take the node-equality
        // control down with it. So the behaviour is preserved to the move, deliberately, and the
        // defect is left to an issue that can measure it in Elo.
        //
        // No other move can land on a friendly square, so this widens the set by castles alone.
        // Found by `tactical_moves_are_the_filtered_legal_moves_in_the_same_order`, not by
        // reasoning — which is the whole reason that test compares against the old path.
        let our_rooks = self.0.colors(us) & self.0.pieces(Piece::Rook);
        let targets = them | promotion_rank.bitboard() | our_rooks;
        let mut moves = Vec::with_capacity(MAX_LEGAL_MOVES);
        // cozy-chess idiom: the mask `generate_moves_for` takes selects the **pieces** to move, not
        // the squares to move to — its own example masks the knights. Destinations are narrowed by
        // intersecting the `to` board of each group instead, which is why this reads as a full
        // generation with a masked visitor rather than as a masked generation.
        //
        // What that saves, and it is the larger half: the attack sets are still computed, but the
        // moves are no longer **materialised** — roughly 39 `Move` values built, pushed and later
        // dropped at every quiescence node.
        self.0.generate_moves(|mut group| {
            group.to &= targets;
            moves.extend(group);
            false
        });
        moves
    }

    /// Whether the side to move has **any** legal move, without building the list.
    ///
    /// The distinction between a stalemate and a quiet position, and between a mate and a position
    /// with no capture available. Quiescence needs that verdict, and needed a full move list to get
    /// it; this stops at the first move found.
    ///
    /// cozy-chess idiom: the generation visitor returns a `bool`, and `true` means *stop*. The
    /// generator then returns whether it was stopped — so a position with a legal move returns
    /// `true` on its first group, and one without walks the (short) remainder and returns `false`.
    pub fn has_any_legal_move(&self) -> bool {
        self.0.generate_moves(|_| true)
    }

    /// Hands the turn to the opponent **without playing a move** — a position that
    /// cannot occur in a real game, used by the search to ask "how bad is it if I do
    /// nothing here?".
    ///
    /// Returns `None` when the side to move is **in check**, and that refusal is the
    /// point rather than an edge case: a side in check must answer, so "what if I
    /// pass" is not a question with an answer, and a score computed from it would be
    /// meaningless. Delegating the test to `cozy-chess` means the search cannot forget
    /// it.
    ///
    /// The en-passant square is cleared and the halfmove clock advanced, both by the
    /// crate: after passing, the opponent's pawn is no longer capturable en passant.
    pub fn null_move(&self) -> Option<Position> {
        self.0.null_move().map(Position)
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

    /// Boards of different densities, so a property is checked over a range rather than over one
    /// square: two openings, a middlegame, two endgames and a bare-king position — 20 % empty to
    /// 95 % empty. The failure mode of a short-circuit is a *class* of square, and one example
    /// cannot see a class.
    const BOARDS: [&str; 6] = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 3",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "4k3/8/8/8/8/8/4P3/4K3 w - - 0 1",
        "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
    ];

    #[test]
    fn piece_on_answers_exactly_what_a_full_scan_would() {
        // The occupancy short-circuit is an optimisation that must be invisible, so it is checked
        // against the thing it replaces: a scan of all six piece boards with no early exit, over
        // every square of every board.
        for fen in BOARDS {
            let p = Position::from_fen(fen).expect("test fixture must parse");
            for sq in Square::ALL {
                let scanned = Piece::ALL.into_iter().find(|&piece| {
                    (p.pieces_of(Color::White, piece).0 | p.pieces_of(Color::Black, piece).0)
                        .has(sq)
                });
                assert_eq!(p.piece_on(sq), scanned, "{fen} at {sq}");
            }
        }
    }

    #[test]
    fn the_occupancy_is_exactly_the_union_of_the_piece_boards() {
        // The premise the short-circuit rests on, asserted rather than assumed. If these two ever
        // part company, `piece_on` returns `None` for occupied squares and every evaluation
        // silently loses material — a failure with no panic and no other test anywhere.
        for fen in BOARDS {
            let p = Position::from_fen(fen).expect("test fixture must parse");
            let union = Piece::ALL.into_iter().fold(cozy_chess::BitBoard::EMPTY, |acc, piece| {
                acc | p.pieces_of(Color::White, piece).0 | p.pieces_of(Color::Black, piece).0
            });
            assert_eq!(union, p.occupied().0, "{fen}: occupancy and piece boards disagree");
        }
    }

    #[test]
    fn legal_moves_never_outgrows_the_buffer_it_reserved() {
        // What `with_capacity` buys is a single allocation, and that only holds while the
        // reservation is an upper bound. The second position is the constructed record holder at
        // exactly `MAX_LEGAL_MOVES` moves, so lowering the constant by one fails this test.
        for fen in [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "R6R/3Q4/1Q4Q1/4Q3/2Q4Q/Q4Q2/pp1Q4/kBNN1KB1 w - - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        ] {
            let p = Position::from_fen(fen).expect("test fixture must parse");
            let moves = p.legal_moves();
            assert!(
                moves.len() <= MAX_LEGAL_MOVES,
                "{fen}: {} moves against a reservation of {MAX_LEGAL_MOVES}",
                moves.len(),
            );
            assert_eq!(
                moves.capacity(),
                MAX_LEGAL_MOVES,
                "{fen}: the vector reallocated, so the reservation bought nothing",
            );
        }
        let record = Position::from_fen("R6R/3Q4/1Q4Q1/4Q3/2Q4Q/Q4Q2/pp1Q4/kBNN1KB1 w - - 0 1")
            .expect("test fixture must parse");
        assert_eq!(
            record.legal_moves().len(),
            MAX_LEGAL_MOVES,
            "the position that pins the constant must produce exactly that many moves",
        );
    }


    #[test]
    fn tactical_moves_are_the_filtered_legal_moves_in_the_same_order() {
        // **The property quiescence rests on.** `tactical_moves` is allowed to be a superset of
        // what the caller keeps, but it must be exactly the subsequence of `legal_moves` that
        // lands on the target squares -- same moves, same order. Move ordering is stable, so a
        // different order would change which nodes the search visits while leaving every score
        // correct, and no other test in the tree would see it.
        //
        // Swept over pseudo-random play rather than over chosen positions, and counted: a sweep
        // that never reached a promotion or a position in check would be asserting on captures
        // alone, which is the easy third of the problem.
        let mut rng = Xorshift(0x7AC7_1CA1);
        let (mut swept, mut in_check, mut with_promotion) = (0, 0, 0);
        for game in 0..300u64 {
            let mut pos = Position::initial();
            for _ in 0..(2 + game % 60) {
                let moves = pos.legal_moves();
                if moves.is_empty() {
                    break;
                }
                pos = pos.play(moves[(rng.next() % moves.len() as u64) as usize]);
            }
            let them = pos.0.colors(!pos.0.side_to_move());
            let promotion_rank = match pos.0.side_to_move() {
                Color::White => cozy_chess::Rank::Eighth,
                Color::Black => cozy_chess::Rank::First,
            };
            let our_rooks = pos.0.colors(pos.0.side_to_move()) & pos.0.pieces(Piece::Rook);
            let targets = them | promotion_rank.bitboard() | our_rooks;
            let expected: Vec<Move> =
                pos.legal_moves().into_iter().filter(|mv| targets.has(mv.to)).collect();
            assert_eq!(pos.tactical_moves(), expected, "on {}", pos.to_fen());
            swept += 1;
            if pos.in_check() {
                in_check += 1;
            }
            if expected.iter().any(|mv| mv.promotion.is_some()) {
                with_promotion += 1;
            }
        }
        assert!(swept > 250, "precondition: the sweep must reach positions, got {swept}");
        assert!(in_check > 0, "precondition: the sweep never reached a position in check");
        assert!(
            with_promotion > 0,
            "precondition: the sweep never reached a promotion, so the rank half of the target \
             mask is untested",
        );
    }

    #[test]
    fn has_any_legal_move_agrees_with_building_the_list() {
        // The cheap verdict against the expensive one, on the three shapes that matter: a mate
        // (no move, in check), a stalemate (no move, not in check), and ordinary positions. The
        // first two are what quiescence calls it for, and they are the only ones where it can
        // return something the search then reports as a mate score.
        for (fen, expected, name) in [
            ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", true, "opening"),
            ("7k/5KQ1/8/8/8/8/8/8 b - - 0 1", false, "mate"),
            ("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", false, "stalemate"),
            ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", true, "bare kings"),
        ] {
            let p = Position::from_fen(fen).expect("test fixture must parse");
            assert_eq!(p.has_any_legal_move(), expected, "{name}");
            assert_eq!(
                p.has_any_legal_move(),
                !p.legal_moves().is_empty(),
                "{name}: the cheap verdict and the list disagree",
            );
        }
    }

    /// A xorshift64, so the sweeps draw their positions instead of having them chosen by the
    /// author of the code under test.
    struct Xorshift(u64);

    impl Xorshift {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }
}
