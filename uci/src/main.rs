//! UCI front-end for the Blunderbuss engine.
//!
//! UCI (Universal Chess Interface) is the text protocol chess GUIs (Cute Chess,
//! Arena, …) speak to engines: line-based commands on stdin, replies on stdout.
//! This binary is a thin adapter — it keeps the current [`Position`] and answers
//! `go` with a move from the engine's search. All the chess logic lives in the
//! `engine` crate; nothing here knows about `cozy-chess`.
//!
//! Supported: `uci`, `isready`, `ucinewgame`, `position`, `go` (a depth cap,
//! `movetime`, a real clock via `wtime`/`btime`/`winc`/`binc`, or nothing at all),
//! `quit`. Whatever the form, `go` is first turned into a [`GoPlan`] and then
//! answered by the engine's single search function — see [`parse_go`].

use engine::position::{Color, Position};
use engine::search::{search_timed, Limits, MAX_DEPTH};
use std::io::{self, BufRead, Write};
use std::time::{Duration, Instant};

/// How long to think when `go` names no limit at all — a bare `go` (an untimed
/// game) or `go infinite` (we do not implement `stop`, so it cannot be endless).
/// A compromise: long enough to search well past the shallow depths, short enough
/// that a GUI never looks frozen.
const DEFAULT_BUDGET_MS: u64 = 2_000;

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

/// What a `go` command asks for, stated without reading the clock: a depth cap and,
/// when the command implies one, a thinking budget. Splitting the *decision* from the
/// *deadline* (`Instant::now() + budget`, computed by the caller) is what makes the
/// parsing a pure function, hence testable without a real clock.
#[derive(Debug, PartialEq, Eq)]
struct GoPlan {
    max_depth: u32,
    budget: Option<Duration>,
}

/// Map the arguments of `go` to a [`GoPlan`]. `side_to_move` selects which side's
/// clock to read; `default_budget` is used when the command names no limit at all.
///
/// The rules, in order: an explicit `depth` sets the cap; `movetime` sets the budget,
/// otherwise a clock (`wtime`/`btime`) does. A command with a depth but no time runs
/// unbounded in time; one with neither — a bare `go`, or `go infinite`, which we
/// cannot interrupt without `stop` — falls back to `default_budget`.
fn parse_go(args: &[&str], side_to_move: Color, default_budget: Duration) -> GoPlan {
    // The protocol sends `key value` pairs; read the number following `key`.
    let value = |key: &str| -> Option<u64> {
        args.iter()
            .position(|&a| a == key)
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.parse::<u64>().ok())
    };

    let depth = value("depth");
    let budget = if let Some(mt) = value("movetime") {
        // The GUI granted exactly this much: keep the safety margin, but never less
        // than 1 ms, so a tiny `movetime` still yields the depth-1 move.
        Some(Duration::from_millis(mt.saturating_sub(SAFETY_MS).max(1)))
    } else if value("wtime").is_some() || value("btime").is_some() {
        // Use the side-to-move's own clock and increment.
        let white = side_to_move == Color::White;
        let remaining = if white { value("wtime") } else { value("btime") }.unwrap_or(0);
        let inc = if white { value("winc") } else { value("binc") }.unwrap_or(0);
        Some(time_budget(
            Duration::from_millis(remaining),
            Duration::from_millis(inc),
        ))
    } else if depth.is_some() {
        None // an explicit depth and no clock: run it to completion
    } else {
        Some(default_budget)
    };

    // A requested depth is clamped to the engine's own ceiling — both to avoid a
    // lossy `u64 as u32` cast on absurd input, and because nothing above it is
    // reachable anyway.
    let max_depth = depth.map_or(MAX_DEPTH, |d| d.min(MAX_DEPTH as u64) as u32);
    GoPlan { max_depth, budget }
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

/// The protocol state: the current position, and how long to think when the GUI
/// names no limit. Kept separate from `main` so it can be unit-tested without
/// spawning a process — and the budget is a field, not a constant, so tests can
/// shorten it.
struct Uci {
    position: Position,
    default_budget: Duration,
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
        Uci {
            position: Position::initial(),
            default_budget: Duration::from_millis(DEFAULT_BUDGET_MS),
        }
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
    ///
    /// Every form goes through the same two steps — plan, then search — so the engine
    /// has a single search path whatever the GUI sends.
    fn go(&self, args: &[&str]) -> String {
        let plan = parse_go(args, self.position.side_to_move(), self.default_budget);
        let deadline = plan.budget.map(|b| Instant::now() + b);
        let stats = search_timed(&self.position, Limits::bounded(plan.max_depth, deadline));

        match stats.best {
            Some((mv, _score)) => format!("bestmove {}", self.position.move_to_uci(mv)),
            None => "bestmove 0000".to_string(), // no legal move (terminal position)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A protocol instance whose default thinking time is short, so the tests that
    // exercise a limitless `go` stay fast. Everything else matches `Uci::new`.
    fn quick_uci() -> Uci {
        Uci { default_budget: Duration::from_millis(20), ..Uci::new() }
    }

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
        let mut uci = quick_uci();
        uci.handle("position startpos");
        let out = uci.handle("go depth 2");
        bestmove_is_legal(&uci, out.lines.first().expect("a bestmove line"));
    }

    #[test]
    fn bare_go_returns_a_legal_move() {
        let mut uci = quick_uci();
        uci.handle("position startpos");
        let out = uci.handle("go");
        bestmove_is_legal(&uci, out.lines.first().expect("a bestmove line"));
    }

    #[test]
    fn go_movetime_returns_a_legal_move() {
        let mut uci = quick_uci();
        uci.handle("position startpos");
        let out = uci.handle("go movetime 50");
        bestmove_is_legal(&uci, out.lines.first().expect("a bestmove line"));
    }

    #[test]
    fn go_with_a_clock_returns_a_legal_move() {
        let mut uci = quick_uci();
        uci.handle("position startpos");
        let out = uci.handle("go wtime 1000 btime 1000 winc 0 binc 0");
        bestmove_is_legal(&uci, out.lines.first().expect("a bestmove line"));
    }

    #[test]
    fn go_infinite_returns_a_legal_move() {
        let mut uci = quick_uci();
        uci.handle("position startpos");
        let out = uci.handle("go infinite");
        bestmove_is_legal(&uci, out.lines.first().expect("a bestmove line"));
    }

    #[test]
    fn quit_stops_the_loop() {
        assert!(Uci::new().handle("quit").quit);
    }

    // --- `go` parsing: one plan per command form -------------------------------

    // The default budget used by the parsing tests; a distinctive value, so an
    // assertion that sees it knows it came from the fallback and not from the args.
    const FALLBACK: Duration = Duration::from_millis(777);

    #[test]
    fn go_depth_plans_a_depth_cap_and_no_clock() {
        assert_eq!(
            parse_go(&["depth", "5"], Color::White, FALLBACK),
            GoPlan { max_depth: 5, budget: None }
        );
    }

    #[test]
    fn go_movetime_plans_a_budget_minus_the_safety_margin() {
        let plan = parse_go(&["movetime", "1000"], Color::White, FALLBACK);
        assert_eq!(plan.max_depth, MAX_DEPTH, "no depth cap was asked for");
        assert_eq!(plan.budget, Some(Duration::from_millis(1000 - SAFETY_MS)));
        // A movetime smaller than the margin must still leave something to search
        // with, rather than underflowing to a huge budget or to zero.
        let tiny = parse_go(&["movetime", "10"], Color::White, FALLBACK).budget.unwrap();
        assert_eq!(tiny, Duration::from_millis(1));
    }

    #[test]
    fn go_with_a_clock_reads_the_side_to_moves_own_clock() {
        let args = ["wtime", "60000", "btime", "6000", "winc", "0", "binc", "0"];
        let white = parse_go(&args, Color::White, FALLBACK).budget.expect("a budget");
        let black = parse_go(&args, Color::Black, FALLBACK).budget.expect("a budget");
        // White has ten times Black's time, so White must plan to think longer —
        // proof that the side to move selects which clock is read.
        assert!(white > black, "white {white:?} should exceed black {black:?}");
        assert_eq!(white, time_budget(Duration::from_secs(60), Duration::ZERO));
        assert_eq!(black, time_budget(Duration::from_secs(6), Duration::ZERO));

        // The increment is read from the same side.
        let with_inc = ["wtime", "60000", "btime", "60000", "winc", "0", "binc", "4000"];
        let black_inc = parse_go(&with_inc, Color::Black, FALLBACK).budget.expect("a budget");
        assert!(black_inc > parse_go(&with_inc, Color::White, FALLBACK).budget.unwrap());
    }

    #[test]
    fn go_depth_and_movetime_honours_both_bounds() {
        // The protocol allows both; `Limits` can carry both, so neither is dropped.
        let plan = parse_go(&["depth", "6", "movetime", "2000"], Color::White, FALLBACK);
        assert_eq!(plan.max_depth, 6);
        assert_eq!(plan.budget, Some(Duration::from_millis(2000 - SAFETY_MS)));
    }

    #[test]
    fn a_limitless_go_falls_back_to_the_default_budget() {
        // A bare `go` and `go infinite` are the same case: no limit was given, and we
        // cannot be interrupted (`stop` is unimplemented), so we time-box ourselves.
        for args in [vec![], vec!["infinite"]] {
            assert_eq!(
                parse_go(&args, Color::White, FALLBACK),
                GoPlan { max_depth: MAX_DEPTH, budget: Some(FALLBACK) },
                "args {args:?} should fall back to the default budget"
            );
        }
    }

    #[test]
    fn unknown_go_arguments_are_ignored() {
        // Arguments we do not implement must not disturb the ones we do.
        let plan = parse_go(&["ponder", "searchmoves", "e2e4", "depth", "3"], Color::White, FALLBACK);
        assert_eq!(plan, GoPlan { max_depth: 3, budget: None });
    }

    // --- time allocation --------------------------------------------------------

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
