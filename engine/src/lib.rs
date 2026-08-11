//! Chess engine core.
//!
//! This crate holds **all the logic**: the `Position` interface, search, move
//! ordering and evaluation. The future UCI binary and Python bindings will be
//! only thin wrappers around it.
//!
//! # Module plan
//!
//! ```text
//! position    the Position interface + its cozy-chess-backed implementation
//! evaluation  material, piece-square tables (PST), then the rest
//! ordering    move ordering — MVV-LVA, killer moves, history
//! search      alpha-beta, quiescence, iterative deepening, transposition
//! ```
//!
//! Modules are declared here as they are written. In Rust, a module is compiled
//! only if it is declared with `pub mod <name>;` — an orphan `src/foo.rs` file
//! would be silently ignored, a classic source of confusion.

pub mod position;
// pub mod evaluation;
// pub mod ordering;
// pub mod search;
