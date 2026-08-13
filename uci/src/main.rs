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
use engine::search::{search, Limits, Progress, Request, MATE, MATE_THRESHOLD, MAX_DEPTH};
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

/// One UCI `info` line describing what an iteration found.
///
/// The format a GUI and an arena both parse: `info depth 6 score cp 25 nodes 12345
/// time 100 nps 123450 pv e2e4`. c-chess-cli reads `depth` and `score` from it and
/// writes them into the PGN, which is how a finished game can be asked what the
/// engine believed at each move.
fn info_line(pos: &Position, p: &Progress) -> String {
    let ms = p.elapsed.as_millis() as u64;
    // Nodes per second is undefined for an iteration that took no measurable time;
    // report the raw count rather than dividing by zero.
    let nps = if ms == 0 { p.nodes } else { p.nodes * 1000 / ms };
    format!(
        "info depth {} score {} nodes {} time {} nps {} pv {}",
        p.depth,
        score_field(p.score),
        p.nodes,
        ms,
        nps,
        pos.move_to_uci(p.best)
    )
}

/// `score cp X` for a material judgement, `score mate N` for a forced mate.
///
/// The search scores a mate `MATE - ply`, which is an implementation detail: a GUI
/// showing "29994" instead of "mate in 3" is reporting our internals. `N` counts
/// **moves**, not plies, and is negative when it is us being mated.
///
/// The plies-to-moves conversion rounds up — a mate delivered on the opponent's ply
/// still costs a whole move — and the mate/material boundary is `MATE_THRESHOLD`,
/// shared with the engine rather than re-derived here.
fn score_field(score: i32) -> String {
    if score.abs() < MATE_THRESHOLD {
        return format!("cp {score}");
    }
    let plies = MATE - score.abs();
    let moves = (plies + 1) / 2;
    format!("mate {}", if score > 0 { moves } else { -moves })
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
    /// Zobrist keys of every position the game passed through before the current
    /// one, oldest first. A draw by repetition is a fact about the game, not about
    /// the board, so the search cannot see one unless it is handed this.
    history: Vec<u64>,
    /// Where `info` lines go, **as they are produced**. A GUI shows the evaluation
    /// while the engine is still thinking, so these cannot be collected and flushed
    /// with the `bestmove` — by then the search is over and there is nothing to watch.
    ///
    /// A boxed closure rather than a `Write`: `main` sends them to stdout and flushes
    /// each one, while a test collects them into a vector and can assert on what was
    /// said and in which order.
    info: Box<dyn FnMut(String)>,
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
    /// Ready to talk to a GUI: `info` lines go straight to stdout, flushed one by one
    /// so nothing waits in a buffer while the engine thinks.
    fn new() -> Uci {
        Uci::with_info(Box::new(|line| {
            let mut out = io::stdout();
            let _ = writeln!(out, "{line}");
            let _ = out.flush();
        }))
    }

    fn with_info(info: Box<dyn FnMut(String)>) -> Uci {
        Uci {
            position: Position::initial(),
            default_budget: Duration::from_millis(DEFAULT_BUDGET_MS),
            history: Vec::new(),
            info,
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
                self.history.clear();
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

        // The history is rebuilt from scratch on every `position` command rather than
        // appended to: a GUI may send an entirely different game, or the same one
        // truncated, and there is no reliable way to tell from the command alone.
        let mut history = Vec::new();
        if let Some(i) = moves_at {
            for &tok in &args[i + 1..] {
                match pos.move_from_uci(tok).and_then(|mv| pos.try_play(mv).ok()) {
                    Some(next) => {
                        // The key of the position we are *leaving* — the one that
                        // would be repeated if the game came back to it.
                        history.push(pos.hash());
                        pos = next;
                    }
                    None => break, // stop at the first move that does not apply
                }
            }
        }
        self.position = pos;
        self.history = history;
    }

    /// `go [depth N | movetime N | wtime N btime N winc N binc N | infinite]`
    /// → the `bestmove` line. Unknown arguments are ignored.
    ///
    /// Every form goes through the same two steps — plan, then search — so the engine
    /// has a single search path whatever the GUI sends.
    fn go(&mut self, args: &[&str]) -> String {
        let plan = parse_go(args, self.position.side_to_move(), self.default_budget);
        let deadline = plan.budget.map(|b| Instant::now() + b);

        // The position is cloned so the reporting closure can borrow it while
        // `self.info` is borrowed mutably — `Position` is copy-make, so this is cheap.
        // `history` is borrowed from a different field, which the borrow checker
        // allows alongside the mutable borrow of `info`.
        let pos = self.position.clone();
        let history = &self.history;
        let sink = &mut self.info;
        let mut report = |p: &Progress| sink(info_line(&pos, p));
        let stats = search(
            &pos,
            Request {
                limits: Limits::bounded(plan.max_depth, deadline),
                history,
                progress: Some(&mut report),
            },
        );

        match stats.best {
            Some((mv, _score)) => format!("bestmove {}", pos.move_to_uci(mv)),
            None => "bestmove 0000".to_string(), // no legal move (terminal position)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;
    use std::rc::Rc;

    // A protocol instance whose default thinking time is short, so the tests that
    // exercise a limitless `go` stay fast, and whose `info` lines are collected instead
    // of printed. Returns the shared buffer alongside, so a test can assert on what was
    // said and in which order.
    //
    // Idiom: `Rc<RefCell<…>>` gives two owners of one vector — the closure inside `Uci`
    // and the test — with the borrow checked at runtime rather than compile time. It is
    // the usual way to observe a callback from a test.
    fn uci_with_log() -> (Uci, Rc<RefCell<Vec<String>>>) {
        let log = Rc::new(RefCell::new(Vec::new()));
        let sink = Rc::clone(&log);
        let mut uci = Uci::with_info(Box::new(move |line| sink.borrow_mut().push(line)));
        uci.default_budget = Duration::from_millis(20);
        (uci, log)
    }

    // A protocol instance whose default thinking time is short, so the tests that
    // exercise a limitless `go` stay fast. Everything else matches `Uci::new`.
    fn quick_uci() -> Uci {
        uci_with_log().0
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
    fn the_move_list_becomes_the_game_history() {
        // One key per move played, and each is the position *left* by that move —
        // the one the game would repeat if it came back to it.
        let mut uci = quick_uci();
        uci.handle("position startpos moves e2e4 e7e5 g1f3");
        assert_eq!(uci.history.len(), 3, "one key per move played");
        assert_eq!(uci.history[0], Position::initial().hash(), "starting with the start");

        // The knight round trip 1.Nf3 Nf6 2.Ng1 Ng8 returns to the initial position,
        // so that key must appear both at the start of the history and as the
        // position now on the board — which is what makes it a repetition.
        let mut round_trip = quick_uci();
        round_trip.handle("position startpos moves g1f3 g8f6 f3g1 f6g8");
        assert_eq!(round_trip.position.hash(), Position::initial().hash());
        assert!(
            round_trip.history.contains(&Position::initial().hash()),
            "the repeated position must be in the history the search is given"
        );
    }

    #[test]
    fn a_new_game_forgets_the_history() {
        // Otherwise the next game inherits repetitions that never happened in it.
        let mut uci = quick_uci();
        uci.handle("position startpos moves e2e4 e7e5");
        assert!(!uci.history.is_empty());
        uci.handle("ucinewgame");
        assert!(uci.history.is_empty(), "`ucinewgame` starts from nothing");
    }

    #[test]
    fn a_new_position_replaces_the_history_rather_than_extending_it() {
        // A GUI may send a different game, or the same one truncated. Appending would
        // leave keys from a game that is no longer being played.
        let mut uci = quick_uci();
        uci.handle("position startpos moves e2e4 e7e5 g1f3 b8c6");
        assert_eq!(uci.history.len(), 4);
        uci.handle("position startpos moves d2d4");
        assert_eq!(uci.history.len(), 1, "the previous game's keys must be gone");
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

    // --- UCI `info` reporting ---------------------------------------------------

    #[test]
    fn every_completed_iteration_is_reported_in_order() {
        let (mut uci, log) = uci_with_log();
        uci.handle("position startpos");
        let out = uci.handle("go depth 4");

        let lines = log.borrow();
        let depths: Vec<u32> = lines
            .iter()
            .map(|l| {
                l.split_whitespace()
                    .nth(2)
                    .and_then(|d| d.parse().ok())
                    .expect("an `info depth N` prefix")
            })
            .collect();
        assert_eq!(depths, vec![1, 2, 3, 4], "one line per depth, in order");
        for line in lines.iter() {
            for field in ["score", "nodes", "time", "nps", "pv"] {
                assert!(line.contains(field), "`{field}` missing from `{line}`");
            }
        }
        assert_eq!(out.lines.len(), 1, "and exactly one bestmove line");
        assert!(out.lines[0].starts_with("bestmove "));
    }

    #[test]
    fn the_last_report_agrees_with_the_move_played() {
        // An engine that announces one move and plays another is worse than a silent
        // one: the analysis would describe a game that never happened.
        let (mut uci, log) = uci_with_log();
        uci.handle("position startpos");
        let out = uci.handle("go depth 4");

        let last = log.borrow().last().cloned().expect("at least one info line");
        let announced = last.split_whitespace().last().expect("a pv move");
        let played = out.lines[0].strip_prefix("bestmove ").expect("a bestmove");
        assert_eq!(announced, played, "the pv of the final info must be the move played");
    }

    #[test]
    fn a_mate_is_reported_in_moves_not_centipawns() {
        // `MATE - ply` is an internal detail; a GUI showing "29994" is reading our
        // implementation rather than the position.
        let (mut uci, log) = uci_with_log();
        uci.handle("position fen 6k1/5ppp/8/8/8/8/8/R6K w - - 0 1");
        uci.handle("go depth 3");
        let last = log.borrow().last().cloned().expect("an info line");
        assert!(last.contains("score mate 1"), "expected `score mate 1` in `{last}`");
    }

    #[test]
    fn being_mated_is_reported_as_a_negative_mate() {
        let (mut uci, log) = uci_with_log();
        uci.handle("position fen 7k/8/8/8/8/8/1R6/R6K b - - 0 1");
        uci.handle("go depth 6");
        let last = log.borrow().last().cloned().expect("an info line");
        assert!(
            last.contains("score mate -"),
            "expected a negative mate score in `{last}`"
        );
    }

    #[test]
    fn an_ordinary_position_is_reported_in_centipawns() {
        let (mut uci, log) = uci_with_log();
        uci.handle("position startpos");
        uci.handle("go depth 3");
        for line in log.borrow().iter() {
            assert!(line.contains("score cp "), "`{line}` should be a cp score");
            assert!(!line.contains("score mate"), "and never a mate");
        }
    }

    #[test]
    fn an_unfinished_iteration_is_not_reported() {
        // The search discards an aborted iteration, so announcing its depth would mean
        // walking the claim back — worse than saying nothing.
        //
        // Measured, not assumed: Kiwipete's first iteration costs ~25 900 nodes against
        // a 2048-node clock check, so a 1 ms budget always cuts it short and **nothing**
        // completes. The log is therefore empty, and asserting emptiness is what makes
        // this discriminating — an upper bound of one would also accept the single line
        // a defective implementation emits.
        let (mut uci, log) = uci_with_log();
        uci.handle("position fen r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1");
        uci.handle("go movetime 51");
        assert!(
            log.borrow().is_empty(),
            "an aborted iteration must not be announced, got {:?}",
            log.borrow()
        );
    }

    #[test]
    fn score_field_converts_plies_to_moves() {
        // The boundary cases of the conversion, without going through a search.
        assert_eq!(score_field(0), "cp 0");
        assert_eq!(score_field(-250), "cp -250");
        assert_eq!(score_field(MATE_THRESHOLD - 1), format!("cp {}", MATE_THRESHOLD - 1));
        // Mate on the ply after ours is still a whole move away: rounding up.
        assert_eq!(score_field(MATE - 1), "mate 1");
        assert_eq!(score_field(MATE - 2), "mate 1");
        assert_eq!(score_field(MATE - 3), "mate 2");
        assert_eq!(score_field(-(MATE - 3)), "mate -2");
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
