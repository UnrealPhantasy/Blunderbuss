//! UCI front-end for the Blunderbuss engine.
//!
//! UCI (Universal Chess Interface) is the text protocol chess GUIs (Cute Chess,
//! Arena, …) speak to engines: line-based commands on stdin, replies on stdout.
//! This binary is a thin adapter — it keeps the current [`Position`] and answers
//! `go` with a move from [`engine::search::best_move`]. All the chess logic lives
//! in the `engine` crate; nothing here knows about `cozy-chess`.
//!
//! Supported: `uci`, `isready`, `ucinewgame`, `position`, `go` (fixed depth),
//! `quit`. Time-control arguments to `go` are accepted but ignored for now.

use engine::position::Position;
use engine::search::best_move;
use std::io::{self, BufRead, Write};

/// Search depth used when `go` does not specify one. Fixed for now — real time
/// management comes later, with iterative deepening.
const DEFAULT_DEPTH: u32 = 4;

fn main() {
    let mut uci = Uci::new();
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    // Read commands line by line until stdin closes or `quit` arrives.
    for line in stdin.lock().lines() {
        let Ok(line) = line else { break };
        let response = uci.handle(&line);
        for out in &response.lines {
            let _ = writeln!(stdout, "{out}");
        }
        let _ = stdout.flush();
        if response.quit {
            break;
        }
    }
}

/// The protocol state: the current position and the default search depth. Kept
/// separate from `main` so it can be unit-tested without spawning a process.
struct Uci {
    position: Position,
    depth: u32,
}

/// What [`Uci::handle`] produced: the lines to print, and whether to stop.
struct Response {
    lines: Vec<String>,
    quit: bool,
}

impl Response {
    fn none() -> Response {
        Response { lines: Vec::new(), quit: false }
    }
    fn lines(lines: Vec<String>) -> Response {
        Response { lines, quit: false }
    }
}

impl Uci {
    fn new() -> Uci {
        Uci { position: Position::initial(), depth: DEFAULT_DEPTH }
    }

    /// Handle one UCI command line and return what to print (and whether to quit).
    /// Unknown or empty commands are ignored, as the protocol requires.
    fn handle(&mut self, line: &str) -> Response {
        // Idiom: `split_whitespace` also trims, so blank lines yield no tokens.
        let tokens: Vec<&str> = line.split_whitespace().collect();
        match tokens.first().copied() {
            Some("uci") => Response::lines(vec![
                "id name Blunderbuss".to_string(),
                "id author UnrealPhantasy".to_string(),
                "uciok".to_string(),
            ]),
            Some("isready") => Response::lines(vec!["readyok".to_string()]),
            Some("ucinewgame") => {
                self.position = Position::initial();
                Response::none()
            }
            Some("position") => {
                self.set_position(&tokens[1..]);
                Response::none()
            }
            Some("go") => Response::lines(vec![self.go(&tokens[1..])]),
            Some("quit") => Response { lines: Vec::new(), quit: true },
            _ => Response::none(),
        }
    }

    /// `position [startpos | fen <6 fields>] [moves <m1> <m2> …]`.
    fn set_position(&mut self, args: &[&str]) {
        // `moves` (if present) separates the setup from the move list.
        let moves_at = args.iter().position(|&t| t == "moves");
        let setup_end = moves_at.unwrap_or(args.len());

        let mut pos = match args.first().copied() {
            Some("startpos") => Position::initial(),
            Some("fen") => {
                let fen = args[1..setup_end].join(" ");
                match Position::from_fen(&fen) {
                    Ok(p) => p,
                    Err(_) => return, // malformed FEN: leave the position unchanged
                }
            }
            _ => return,
        };

        if let Some(i) = moves_at {
            for &tok in &args[i + 1..] {
                match pos.move_from_uci(tok).and_then(|mv| pos.try_play(mv).ok()) {
                    Some(next) => pos = next,
                    None => break, // stop at the first move that does not apply
                }
            }
        }
        self.position = pos;
    }

    /// `go [depth N] …` → the `bestmove` line. Any other argument (`wtime`,
    /// `movetime`, …) is ignored for now.
    fn go(&self, args: &[&str]) -> String {
        let depth = args
            .iter()
            .position(|&a| a == "depth")
            .and_then(|i| args.get(i + 1))
            .and_then(|n| n.parse::<u32>().ok())
            .unwrap_or(self.depth);

        match best_move(&self.position, depth) {
            Some((mv, _score)) => format!("bestmove {}", self.position.move_to_uci(mv)),
            None => "bestmove 0000".to_string(), // no legal move (terminal position)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uci_command_announces_uciok() {
        let out = Uci::new().handle("uci");
        assert!(out.lines.iter().any(|l| l == "uciok"));
        assert!(out.lines.iter().any(|l| l.starts_with("id name")));
        assert!(out.lines.iter().any(|l| l.starts_with("id author")));
        assert!(!out.quit);
    }

    #[test]
    fn isready_answers_readyok() {
        assert_eq!(Uci::new().handle("isready").lines, vec!["readyok".to_string()]);
    }

    #[test]
    fn position_startpos_with_moves_updates_state() {
        let mut uci = Uci::new();
        uci.handle("position startpos moves e2e4 e7e5");

        let mut expected = Position::initial();
        for mv in ["e2e4", "e7e5"] {
            let m = expected.move_from_uci(mv).unwrap();
            expected = expected.play(m);
        }
        assert_eq!(uci.position.hash(), expected.hash());
    }

    #[test]
    fn position_fen_is_parsed() {
        let mut uci = Uci::new();
        let fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1";
        uci.handle(&format!("position fen {fen}"));
        assert_eq!(uci.position.hash(), Position::from_fen(fen).unwrap().hash());
    }

    #[test]
    fn go_returns_a_legal_bestmove() {
        let mut uci = Uci::new();
        uci.handle("position startpos");
        let out = uci.handle("go depth 2");
        let line = out.lines.first().expect("a bestmove line");
        let mv_str = line.strip_prefix("bestmove ").expect("bestmove prefix");
        let mv = uci.position.move_from_uci(mv_str).expect("a parseable move");
        assert!(uci.position.try_play(mv).is_ok(), "the reported move must be legal");
    }

    #[test]
    fn bare_go_uses_the_default_depth_and_returns_a_legal_move() {
        // `go` with no depth must fall back to DEFAULT_DEPTH and still answer with
        // a legal move (the fallback branch not covered by `go depth N`).
        let mut uci = Uci::new();
        uci.handle("position startpos");
        let out = uci.handle("go");
        let line = out.lines.first().expect("a bestmove line");
        let mv_str = line.strip_prefix("bestmove ").expect("bestmove prefix");
        let mv = uci.position.move_from_uci(mv_str).expect("a parseable move");
        assert!(uci.position.try_play(mv).is_ok(), "the reported move must be legal");
    }

    #[test]
    fn quit_stops_the_loop() {
        assert!(Uci::new().handle("quit").quit);
    }
}
