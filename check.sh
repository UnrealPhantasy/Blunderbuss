#!/usr/bin/env bash
# Every gate the repository claims, in one command — including the one that does not run by
# default and stayed red on `main` for three days because of it.
#
# `#[ignore]` without something that runs it is a deletion with a comment attached. The SEE
# pruning sweep is a good test, written for the right reason, and it went red on 21 August when
# check extensions were merged; nobody saw it until 24 August. This script is the cheap fix, and
# it is deliberately not CI: it costs nothing to run and needs no service.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
fail=0
run() {
  printf '\n\033[1m▶ %s\033[0m\n' "$1"; shift
  if "$@"; then printf '\033[32m  ✓\033[0m\n'; else printf '\033[31m  ✗ FAILED\033[0m\n'; fail=1; fi
}

run "tests (debug)"              cargo test
run "tests (release)"            cargo test --release
run "clippy (debug)"             cargo clippy --all-targets -- -D warnings
run "clippy (release)"           cargo clippy --all-targets --release -- -D warnings
# Deliberately last and announced: about 100 seconds of pseudo-random play, so a wait is not
# mistaken for a hang.
printf '\n\033[1m▶ ignored sweeps (~100 s, be patient)\033[0m\n'
if cargo test --release -p engine -- --ignored; then printf '\033[32m  ✓\033[0m\n'; else printf '\033[31m  ✗ FAILED\033[0m\n'; fail=1; fi

printf '\n%s\n' "────────────────────────────────"
[ "$fail" -eq 0 ] && printf '\033[32mEverything is green.\033[0m\n' || printf '\033[31mAt least one gate is red.\033[0m\n'
exit "$fail"
