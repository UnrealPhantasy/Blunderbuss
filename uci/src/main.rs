//! UCI front-end for the Blunderbuss engine.
//!
//! UCI (Universal Chess Interface) is the text protocol chess GUIs (Cute Chess,
//! Arena, …) speak to engines: line-based commands on stdin, replies on stdout.
//! This binary is a thin adapter — it keeps the current [`Position`] and answers
//! `go` with a move from the engine's search. All the chess logic lives in the
//! `engine` crate; nothing here knows about `cozy-chess`.
//!
//! Supported: `uci`, `isready`, `ucinewgame`, `position`, `go` (fixed depth,
//! `movetime`, or a real clock via `wtime`/`btime`/`winc`/`binc`), `quit`.

use engine::position::{Color, Position};
use engine::search::{search, search_timed, Limits};
use std::io::{self, BufRead, Write};
use std::time::{Duration, Instant};

/// Depth used for a bare `go` (no limit given) and for `go infinite` (we do not
/// implement `stop`, so "infinite" is a fixed-depth search rather than a hang).
const DEFAULT_DEPTH: u32 = 4;

/// Time kept in reserve so we answer before the flag falls, in milliseconds.
const SAFETY_MS: u64 = 50;

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

/// How long to spend on this move, derived from the clock. A simple, tunable
/// rule: a thirtieth of the remaining time plus half the increment — never more
/// than the remaining time minus a safety margin, and at least 1 ms.
fn time_budget(remaining: Duration, increment: Duration) -> Duration {
    let remaining = remaining.as_millis() as u64;
    let increment = increment.as_millis() as u64;
    let alloc = remaining / 30 + increment / 2;
    // Keep a safety margin, and never propose more time than we actually have:
    // once the margin already eats the whole clock, spend nothing (play instantly).
    let cap = remaining.saturating_sub(SAFETY_MS);
    let budget = if cap > 0 { alloc.clamp(1, cap) } else { 0 };
    Duration::from_millis(budget)
}

/// The protocol state: the current position and the default search depth. Kept
/// separate from `main` so it can be unit-tested without spawning a process.
struct Uci {
    position: Position,
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
        Uci { position: Position::initial() }
    }

    /// Handle one UCI command line and return what to print (and whether to quit).
    /// Unknown or empty commands are ignored, as the protocol requires.
    fn handle(&mut self, line: &str) -> Response {
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

    /// `go [depth N | movetime N | wtime N btime N winc N binc N | infinite]`
    /// → the `bestmove` line. Unknown arguments are ignored.
    fn go(&self, args: &[&str]) -> String {
        // Parse the recognized `key value` tokens (and the flag `infinite`).
        let value = |key: &str| -> Option<u64> {
            args.iter()
                .position(|&a| a == key)
                .and_then(|i| args.get(i + 1))
                .and_then(|v| v.parse::<u64>().ok())
        };
        let infinite = args.contains(&"infinite");

        let stats = if let Some(d) = value("depth") {
            search(&self.position, d as u32)
        } else if let Some(mt) = value("movetime") {
            let budget = mt.saturating_sub(SAFETY_MS).max(1);
            search_timed(&self.position, Limits::until(Instant::now() + Duration::from_millis(budget)))
        } else if value("wtime").is_some() || value("btime").is_some() {
            // Use the side-to-move's own clock.
            let white = self.position.side_to_move() == Color::White;
            let remaining = if white { value("wtime") } else { value("btime") }.unwrap_or(0);
            let inc = if white { value("winc") } else { value("binc") }.unwrap_or(0);
            let budget = time_budget(Duration::from_millis(remaining), Duration::from_millis(inc));
            search_timed(&self.position, Limits::until(Instant::now() + budget))
        } else if infinite {
            // No `stop` support: bound it to a fixed depth so it always ends.
            search(&self.position, DEFAULT_DEPTH)
        } else {
            search(&self.position, DEFAULT_DEPTH)
        };

        match stats.best {
            Some((mv, _score)) => format!("bestmove {}", self.position.move_to_uci(mv)),
            None => "bestmove 0000".to_string(), // no legal move (terminal position)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: the move string a `go` reply advertises must be legal here.
    fn bestmove_is_legal(uci: &Uci, line: &str) {
        let mv_str = line.strip_prefix("bestmove ").expect("bestmove prefix");
        let mv = uci.position.move_from_uci(mv_str).expect("a parseable move");
        assert!(uci.position.try_play(mv).is_ok(), "the reported move must be legal");
    }

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
    fn go_depth_returns_a_legal_bestmove() {
        let mut uci = Uci::new();
        uci.handle("position startpos");
        let out = uci.handle("go depth 2");
        bestmove_is_legal(&uci, out.lines.first().expect("a bestmove line"));
    }

    #[test]
    fn bare_go_returns_a_legal_move() {
        let mut uci = Uci::new();
        uci.handle("position startpos");
        let out = uci.handle("go");
        bestmove_is_legal(&uci, out.lines.first().expect("a bestmove line"));
    }

    #[test]
    fn go_movetime_returns_a_legal_move() {
        let mut uci = Uci::new();
        uci.handle("position startpos");
        let out = uci.handle("go movetime 50");
        bestmove_is_legal(&uci, out.lines.first().expect("a bestmove line"));
    }

    #[test]
    fn go_with_a_clock_returns_a_legal_move() {
        let mut uci = Uci::new();
        uci.handle("position startpos");
        let out = uci.handle("go wtime 1000 btime 1000 winc 0 binc 0");
        bestmove_is_legal(&uci, out.lines.first().expect("a bestmove line"));
    }

    #[test]
    fn go_infinite_returns_a_legal_move() {
        let mut uci = Uci::new();
        uci.handle("position startpos");
        let out = uci.handle("go infinite");
        bestmove_is_legal(&uci, out.lines.first().expect("a bestmove line"));
    }

    #[test]
    fn quit_stops_the_loop() {
        assert!(Uci::new().handle("quit").quit);
    }

    #[test]
    fn time_budget_grows_with_time_and_increment() {
        let more = time_budget(Duration::from_secs(60), Duration::from_secs(0));
        let less = time_budget(Duration::from_secs(6), Duration::from_secs(0));
        assert!(more > less, "more time should mean a larger budget");
        let with_inc = time_budget(Duration::from_secs(60), Duration::from_secs(2));
        assert!(with_inc > more, "an increment should raise the budget");
    }

    #[test]
    fn time_budget_never_exceeds_remaining() {
        // A realistic clock: a positive budget, safely under the time left.
        let remaining = Duration::from_millis(300);
        let b = time_budget(remaining, Duration::from_secs(10));
        assert!(b >= Duration::from_millis(1) && b < remaining);
        // Edge cases: the budget is never more than the remaining time, and it is
        // exactly zero once the safety margin already eats the whole clock.
        assert_eq!(time_budget(Duration::ZERO, Duration::ZERO), Duration::ZERO);
        assert!(time_budget(Duration::from_millis(10), Duration::ZERO) <= Duration::from_millis(10));
    }
}
