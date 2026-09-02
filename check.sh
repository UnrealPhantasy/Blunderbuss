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

# ---------------------------------------------------------------- language and identity gates
#
# This repository is English-only for everything tracked, and its author identity is a pseudonym.
# Four leaks in three days, every one caught by a reader rather than by a gate: an accented word in
# a table (#67), decimal commas (#65), four French lines in *this script* (#70), and a test fixture
# named in French (#84). A fifth was a real first name in a comment and in a commit message (#82),
# which is the one that cannot be repaired in place — editing a message rewrites history, and a pull
# request keeps its commits reachable for ever. So that one has to be caught before the push.
#
# THREE GATES, AND THEY DO NOT SHARE AN INPUT.
#
#   accents   raw bytes, a class of French accented letters written as \x{} escapes
#   words     the transliterated stream, so a word written with its accents is caught by its
#             accent-free spelling -- which is why the list holds no accented form at all
#   identity  the transliterated stream, with a pattern derived at RUN TIME (see below)
#
# Transliteration (`iconv //TRANSLIT`) is what lets the word and identity patterns stay plain ASCII:
# it maps every accented letter to its base letter, line for line, so line numbers still hold. The
# accent gate is the one that must NOT see a transliterated stream — there would be nothing left to
# find.
#
# WHAT IS DELIBERATELY ABSENT FROM EACH LIST, because a gate that cries wolf on its first run gets
# commented out and then the class is worse off than with no gate:
#
#   from the word list   `a`, `on` and `plus`, which are English and occur 1826, 385 and 9 times in
#                        the tracked tree; anything under three letters; and `car`, `est`, `sur`,
#                        `son`, `ses`, `ces`, `fait`, which read as English words or abbreviations.
#                        Every other candidate was grepped word-wise over the tree and occurs zero
#                        times -- including the very word that leaked in #84, which is what
#                        makes this list shippable.
#   from the accent list the German umlauts `a`, `o`, `u` with diaereses, because `Grunfeld` written
#                        properly is a legitimate opening name in `book.txt`; and U+00D7, the
#                        multiplication sign, which sits inside the Latin-1 *letter* block and
#                        matches the obvious `[\x{00C0}-\x{017F}]` range. Both were real false
#                        positives on this tree.
#
# THE IDENTITY PATTERN IS NEVER WRITTEN DOWN HERE. Writing it would be the leak. It is derived from
# the machine's global git identity, which carries the real name — this is a work machine, and the
# repository-local identity is the pseudonym — split on camel case and on non-letters. An untracked
# `.check-identity`, one string per line, adds to it. If neither yields anything the gate FAILS
# LOUDLY rather than passing, because a gate that silently does nothing is the failure mode this
# whole thing exists to close.
#
# And the identity gate prints `file:line` WITHOUT the matched text, so a terminal log that gets
# pasted somewhere does not carry the name it just found. Same for a commit: it prints the short
# SHA and the date and NOT the subject, because the subject is where the name was. `git log -1
# <sha>` reads the rest locally, which is where it belongs. Its own test caught this: the first
# version printed `%s` and echoed the name straight back.

# Lines carrying this marker are skipped by the gates: a pattern definition necessarily contains
# what it looks for. Everything else in this file is still scanned, which is the part that matters —
# #70 leaked four French lines into this very script.
#
# THE HOLE THIS OPENS, stated rather than left to be discovered: a line carrying the marker is not
# scanned, so French could be hidden behind it. That is deliberate and it is auditable -- exactly
# **two** lines carry it, and they are the two immediately below: the marker's own definition and
# the word list. The alternative was excluding this whole file, which is the file French actually
# leaked into.
#
# AND THIS COMMENT DELIBERATELY DOES NOT SPELL THE MARKER OUT. An earlier version did, in prose,
# to give the reader a `grep` to run -- which exempted this very line from all three gates. The
# comment documenting the hole was inside it, and it announced the wrong count as well: three
# lines carried the marker while the text claimed one. A sentence about a pattern must not match
# it. To audit, grep for the value assigned just below.
GATE_MARKER='LANG-GATE-PATTERN'

# French words with no accent, matched word-wise and case-insensitively. See the note above for what
# is absent and why.
FRENCH_WORDS='une|des|pour|avec|dans|sont|qui|que|pas|mais|donc|sans|sous|entre|comme|tous|moins|bien|deja|encore|toujours|jamais|rien|meme|autre|chaque|leur|notre|votre|celui|celle|ceux|alors|aussi|ainsi|cela|etre|avoir|faire|parce|lorsque|puisque|afin|plutot|ici|cette|nous|vous|elles|ils'   # LANG-GATE-PATTERN

# French accented letters, as codepoints so that no accented character appears in this file:
# a-grave a-circ c-cedilla e-grave e-acute e-circ e-diaeresis i-circ i-diaeresis o-circ u-grave
# u-circ y-diaeresis oe ae, and their capitals. No umlauts, no U+00D7.
ACCENTS='[\x{00E0}\x{00E2}\x{00E7}\x{00E8}\x{00E9}\x{00EA}\x{00EB}\x{00EE}\x{00EF}\x{00F4}\x{00F9}\x{00FB}\x{00FF}\x{0153}\x{00E6}\x{00C0}\x{00C2}\x{00C7}\x{00C8}\x{00C9}\x{00CA}\x{00CB}\x{00CE}\x{00CF}\x{00D4}\x{00D9}\x{00DB}\x{0178}\x{0152}\x{00C6}]'

# The strings the identity gate looks for, derived rather than stored. Empty output means the gate
# cannot work, and the caller treats that as a failure.
#
# THE FOUR-CHARACTER FLOOR, why it exists and why it does NOT apply to both sources.
#
# The derived half needs a floor: `git config --global user.name` is split on camel case and on
# non-letters, so it yields fragments, and fragments of one to three letters fire on ordinary
# English -- an initial, or a three-letter particle out of a surname. Without the floor this gate
# is noise, and a noisy gate gets commented out.
#
# But the floor applied to BOTH sources would contradict the reason this gate is separate from the
# language ones. Their tolerances are opposite: a false positive on language costs a moment, a
# false **negative** on identity is not repairable in place -- editing a message rewrites history,
# and a pull request keeps its commits reachable for ever. So this gate is tuned against MISSES,
# and a floor that silently drops a three-letter given name is exactly a miss.
#
# Hence: the floor governs what is DERIVED, and an untracked `.check-identity` bypasses it. A
# string written there by hand is deliberate -- whoever wrote it knows what it will match, which is
# precisely what cannot be assumed of an automatically split fragment. Its own floor is two
# characters, which drops blank lines and stray single letters and nothing else.
# The two floors, named so that a test can read them instead of transcribing them. The property
# that matters is not either value but the fact that the DERIVED floor is the higher of the two —
# that is what says the two sources are trusted differently, and it is the thing a later edit could
# quietly undo by making them equal.
#
# HONEST NOTE ON WHAT IS TESTED HERE, in the same spirit as the null move's `return beta`.
# Three mutations are caught by the tests: equalising the two constants, lowering the derived one,
# and making the function ignore the explicit one. The fourth — making the function ignore the
# DERIVED constant, i.e. dropping its floor to 1 — **cannot be caught on this machine and probably
# not on any**: the check is only observable if the global git identity yields a fragment shorter
# than four letters, and a name that splits into such a fragment is unusual. Verified: with the
# floor at 1 and at 4, this machine derives the same two strings. So that line rests on reasoning,
# and saying so is better than an assertion that cannot fail.
IDENTITY_FLOOR_DERIVED=4
IDENTITY_FLOOR_EXPLICIT=2

identity_strings() {
  {
    # Derived: split on camel case, then on anything that is not a letter, then floored.
    git config --global user.name 2>/dev/null \
      | sed -E 's/([a-z])([A-Z])/\1\n\2/g' | tr -c 'A-Za-z\n' '\n' \
      | awk -v n="$IDENTITY_FLOOR_DERIVED" 'length($0) >= n'
    # Explicit: one string per line, with the lower floor. See the note above for why it differs.
    [ -f .check-identity ] \
      && tr -c 'A-Za-z\n' '\n' < .check-identity \
       | awk -v n="$IDENTITY_FLOOR_EXPLICIT" 'length($0) >= n'
  } | sort -u
}

# Every tracked file, transliterated, one `grep -n` per file so the name of the file is known.
# `git ls-files -z` and `-0` because a filename with a space would otherwise split.
scan_transliterated() {   # $1 = extended pattern
  git ls-files -z | while IFS= read -r -d '' f; do
    [ -f "$f" ] || continue
    iconv -f UTF-8 -t ASCII//TRANSLIT "$f" 2>/dev/null \
      | grep -nwiE -e "$1" 2>/dev/null | sed "s|^|$f:|"
  done | grep -v "$GATE_MARKER" || true
}

gate_accents() {
  local hits
  hits=$(git ls-files -z | xargs -0 -r grep -nP -e "$ACCENTS" -- 2>/dev/null | grep -v "$GATE_MARKER" || true)
  [ -z "$hits" ] && return 0
  printf '%s\n' "$hits" | sed 's/^/    /'
  return 1
}

gate_french_words() {
  local hits
  hits=$(scan_transliterated "$FRENCH_WORDS")
  [ -z "$hits" ] && return 0
  printf '%s\n' "$hits" | sed 's/^/    /'
  return 1
}

gate_identity() {
  local pattern hits commits
  pattern=$(identity_strings | paste -sd'|' -)
  if [ -z "$pattern" ]; then
    echo "    cannot derive any identity string: set a global git user.name, or add an untracked"
    echo "    .check-identity with one string per line. Refusing to pass a gate that cannot work."
    return 1
  fi
  # Tracked files. Only `file:line` is printed — never the matched text, so a pasted log does not
  # carry the name.
  hits=$(scan_transliterated "$pattern" | cut -d: -f1,2)
  # And the messages of commits not yet on `main`, which no `git ls-files` grep can see and which
  # are the ones that cannot be repaired in place.
  commits=$(git log main..HEAD --format='%H' 2>/dev/null | while read -r h; do
    git log -1 --format='%B' "$h" | iconv -f UTF-8 -t ASCII//TRANSLIT 2>/dev/null \
      | grep -qwiE -e "$pattern" && git log -1 --format='commit %h  (%ad)' --date=short "$h"
  done)
  [ -z "$hits" ] && [ -z "$commits" ] && return 0
  [ -n "$hits" ] && printf '%s\n' "$hits" | sed 's/^/    /'
  [ -n "$commits" ] && printf '%s\n' "$commits" | sed 's/^/    /'
  return 1
}

# Before the sweep on purpose: these cost milliseconds, and a fast gate placed last is a gate that
# gets interrupted.
run_gates() {
  echo "test" | grep -qP 'te.t' 2>/dev/null || {
    printf '\n\033[31m  grep -P is unavailable: the accent gate cannot run\033[0m\n'; fail=1; }
  run "no accented French"          gate_accents
  run "no accent-free French"       gate_french_words
  run "no real name"                gate_identity
}

# `--gates-only` runs the three language gates and nothing else. They cost milliseconds against the
# ~100 s of the sweep, so this is the form you run before a commit rather than after one — which is
# the whole point for the identity gate, since a name in a commit message cannot be repaired in
# place once pushed.
run_gates
if [ "${1:-}" = "--gates-only" ]; then
  printf '\n%s\n' "────────────────────────────────"
  [ "$fail" -eq 0 ] && printf '\033[32mGates are green.\033[0m\n' || printf '\033[31mAt least one gate is red.\033[0m\n'
  exit "$fail"
fi

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
