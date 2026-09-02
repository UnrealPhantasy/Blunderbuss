#!/usr/bin/env bash
# Tests for the three language and identity gates in `check.sh`.
#
#   ./check-gates-test.sh
#
# WHY THIS SCRIPT HOLDS NO FIXTURE LITERAL. A test for a French-word gate that contained a French
# word would trip the gate it tests; so would one carrying an accented letter, and one carrying the
# real name would be the very leak the gate exists to prevent. So every fixture is DERIVED at run
# time from the same source the gate reads: one word taken out of `FRENCH_WORDS`, accented letters
# built from `printf` escapes, and the identity strings from `identity_strings`.
#
# That has a second benefit worth more than the first: a test built from the gate's own source cannot
# drift away from it. Change the word list and the fixture changes with it.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")"
pass=0; fail=0
ok()   { printf '\033[32m  ✓\033[0m %s\n' "$1"; pass=$((pass+1)); }
bad()  { printf '\033[31m  ✗\033[0m %s\n' "$1"; fail=$((fail+1)); }

# The gate definitions, read from check.sh rather than duplicated.
FRENCH_WORDS=$(grep -E "^FRENCH_WORDS=" check.sh | sed -E "s/^FRENCH_WORDS='([^']*)'.*/\1/")
[ -n "$FRENCH_WORDS" ] || { echo "cannot read FRENCH_WORDS from check.sh"; exit 1; }
A_FRENCH_WORD=${FRENCH_WORDS%%|*}          # the first entry, whatever it is

FIXTURE=gate-fixture-tmp.txt
cleanup() { git rm --cached -q --force "$FIXTURE" 2>/dev/null; rm -f "$FIXTURE"; }
trap cleanup EXIT

# A fixture is a TRACKED file, because that is what the gates scan. `git add -N` registers the path
# without staging content, which `git ls-files` reports and which leaves nothing to commit.
with_fixture() {   # $1 = the bytes to write; runs the gates, echoes their output
  printf '%b\n' "$1" > "$FIXTURE"
  git add -N "$FIXTURE" 2>/dev/null
  ./check.sh --gates-only 2>&1 | sed 's/\x1b\[[0-9;]*m//g'
  cleanup
}

echo "── the gates must fire"

out=$(with_fixture "let x = 1; // r\\xc3\\xa9p\\xc3\\xa8te")
if grep -q 'no accented French' <<< "$out" && grep -qE "FAILED" <<< "$out" \
   && grep -q "$FIXTURE:1" <<< "$out"; then
  ok "an accented French word fails, and the output names file:line"
else bad "accented French not caught, or file:line missing"; fi

out=$(with_fixture "(\"$A_FRENCH_WORD-un-fen\", \"not a FEN at all\"),")
if grep -q "$FIXTURE:1" <<< "$out"; then
  ok "an accent-free French word fails — the #84 leak, which no accent grep can see"
else bad "accent-free French not caught"; fi

# THE LITERAL CASE, and it exists because deriving the fixture from the list makes the list's own
# CONTENT untestable: drop a word and the derived fixture silently follows it. So one fixture is the
# real string that leaked in #84, written out. The line carries the gate marker, which is precisely
# what that exemption is for — and `grep LANG-GATE-PATTERN` shows every line that uses it.
LEAK='pas-un-fen'   # LANG-GATE-PATTERN
out=$(with_fixture "(\"$LEAK\", \"not a FEN at all\"),")
if grep -q "$FIXTURE:1" <<< "$out"; then
  ok "the exact string that leaked in #84 is caught, asserted against a literal"
else bad "the #84 string is not caught — the word list has lost the word it needed"; fi

# The identity fixture is derived, so this file never carries the name.
NAME=$(bash -c 'source /dev/stdin <<< "$(sed -n "/^identity_strings()/,/^}/p" check.sh)"; identity_strings' | head -1)
if [ -z "$NAME" ]; then
  bad "cannot derive an identity string — the gate itself would refuse to pass, see below"
else
  out=$(with_fixture "// Spotted by $NAME looking at the board")
  if grep -q 'no real name' <<< "$out" && grep -q "$FIXTURE:1" <<< "$out"; then
    ok "a real name in a tracked file fails, and only file:line is printed"
  else bad "real name in a tracked file not caught"; fi
  # TRANSLITERATION, which nothing else here exercises: every other fixture is ASCII. The accented
  # form of the name must be caught by the IDENTITY gate, whose pattern is plain ASCII — that only
  # works because the stream is transliterated before matching. Built by substitution rather than
  # written out, so this file still carries no accented byte.
  # Any vowel will do, and which one depends on the name — the first derived string here has no `e`,
  # which is how the first version of this test came back "untested" instead of silently passing.
  ACCENTED="$NAME"
  for pair in "a:$(printf '\xc3\xa0')" "e:$(printf '\xc3\xa9')" "i:$(printf '\xc3\xae')" \
              "o:$(printf '\xc3\xb4')" "u:$(printf '\xc3\xbb')"; do
    v=${pair%%:*}; acc=${pair#*:}
    t=$(printf '%s' "$NAME" | sed "s/$v/$acc/")
    [ "$t" != "$NAME" ] && { ACCENTED="$t"; break; }
  done
  if [ "$ACCENTED" = "$NAME" ]; then
    printf '\033[33m  ~\033[0m no vowel in the derived name, transliteration untested\n'
  else
    out=$(with_fixture "// Spotted by $ACCENTED looking at the board")
    # The identity section is the one that must report it, not merely the accent gate.
    ident=$(sed -n '/no real name/,$p' <<< "$out")
    if grep -q "$FIXTURE:1" <<< "$ident"; then
      ok "the accented form of the name is caught by the identity gate — transliteration works"
    else bad "the accented name reached only the accent gate: transliteration is not applied"; fi
  fi

  # And the output must NOT echo the name back, so a pasted log does not carry it.
  if grep -q "$NAME" <<< "$(grep "$FIXTURE" <<< "$out")"; then
    bad "the gate echoed the name it found — a pasted log would carry it"
  else ok "the gate did not echo the name, only its location"; fi
fi

echo
echo "── the accent class, asserted on its CONTENT and not only on its behaviour"

# WHY A CONTENT ASSERTION AND NOT JUST A FIXTURE. A behavioural fixture carries several accented
# letters, so removing one codepoint from the class leaves the others to fire and the test stays
# green — measured: dropping e-acute changed nothing. And deriving the fixture from the class is
# self-neutralising, the same trap the word list has. So the expected codepoints are written out
# here, in escaped form, and each is checked for presence. A removal now fails by name.
ACC=$(grep -E "^ACCENTS=" check.sh | sed -E "s/^ACCENTS='([^']*)'.*/\1/")
[ -n "$ACC" ] || { echo "cannot read ACCENTS from check.sh"; exit 1; }
missing=""
for cp in 00E0 00E2 00E7 00E8 00E9 00EA 00EB 00EE 00EF 00F4 00F9 00FB 00FF 0153 00E6 \
          00C0 00C2 00C7 00C8 00C9 00CA 00CB 00CE 00CF 00D4 00D9 00DB 0178 0152 00C6; do
  grep -q "x{$cp}" <<< "$ACC" || missing="$missing $cp"
done
if [ -z "$missing" ]; then ok "all 30 expected French codepoints are in the class"
else bad "codepoints missing from the accent class:$missing"; fi

# And the ones that must stay OUT, each for a reason that was a real false positive here.
present=""
for cp in 00E4 00F6 00FC 00C4 00D6 00DC 00D7 00F7; do
  grep -q "x{$cp}" <<< "$ACC" && present="$present $cp"
done
if [ -z "$present" ]; then
  ok "the German umlauts and the multiplication and division signs are excluded"
else bad "codepoints that must stay out are in the class:$present"; fi

echo
echo "── the gates must NOT fire (the false positives that shaped the lists)"

out=$(with_fixture "four positions \\xc3\\x97 12 first moves")
if grep -q "$FIXTURE" <<< "$out"; then bad "U+00D7 tripped a gate"; else
  ok "the multiplication sign does not trip the accent gate"; fi

out=$(with_fixture "# Gr\\xc3\\xbcnfeld")
if grep -q "$FIXTURE" <<< "$out"; then bad "a German umlaut tripped the accent gate"; else
  ok "a German umlaut does not trip it — Grunfeld is in book.txt"; fi

out=$(with_fixture "a plus on the board, on the file, a rook")
if grep -q "$FIXTURE" <<< "$out"; then bad "an English word tripped the word gate"; else
  ok "the three English words excluded from the list do not trip it"; fi

out=$(with_fixture "the theorem is a theory, theoretically")
if grep -q "$FIXTURE" <<< "$out"; then bad "theorem/theory tripped the identity gate"; else
  ok "theorem, theory and theoretical do not trip the identity gate"; fi

echo
echo "── the commit-message path, which no git-ls-files grep can see"

# This is the criterion the whole identity gate exists for: a name in a tracked file is repairable by
# a commit, a name in a MESSAGE is not — editing it rewrites history, and a pull request keeps its
# commits reachable for ever. So it has to be caught before the push, and that means a gate that
# reads `git log main..HEAD`.
if [ -n "${NAME:-}" ]; then
  BR="gate-test-tmp-$$"
  # WHERE TO COME BACK TO, and it must survive a detached HEAD. `git rev-parse --abbrev-ref HEAD`
  # returns the literal string `HEAD` when nothing is checked out by name — which is the state of a
  # worktree created with `git worktree add <dir> origin/<branch>`, i.e. exactly how a pull request
  # on this repository gets reviewed. `git switch HEAD` then fails, `git branch -D` fails too
  # because the branch is still checked out, and the caller is left standing on the test branch
  # **with a commit whose message contains a real name**. That is the leak this gate exists to
  # prevent, produced by the gate's own test suite.
  #
  # `git symbolic-ref` succeeds only when HEAD names a branch, so it is the test: a branch to switch
  # back to, or a SHA to detach onto.
  if ORIG=$(git symbolic-ref -q --short HEAD); then
    RETOUR=(switch -q "$ORIG")
  else
    ORIG=$(git rev-parse HEAD)
    RETOUR=(switch -q --detach "$ORIG")
  fi
  git switch -q -c "$BR" || { bad "cannot create the temporary branch"; exit 1; }
  git -c user.name=UnrealPhantasy -c user.email=x@example.invalid -c commit.gpgsign=false \
      commit -q --allow-empty -m "test: spotted by $NAME on the board" 2>/dev/null
  out=$(./check.sh --gates-only 2>&1 | sed 's/\x1b\[[0-9;]*m//g')
  # NOT silenced, and that is the point of this whole block. Everything else in this script may
  # fail quietly; a failed restore leaves a name behind, so it has to be loud and fatal.
  git "${RETOUR[@]}" || {
    echo "FATAL: cannot return to $ORIG — branch $BR still holds a commit naming a real person."
    echo "       Repair by hand:  git switch --detach $ORIG && git branch -D $BR"
    exit 1
  }
  git branch -q -D "$BR" || {
    echo "FATAL: branch $BR could not be deleted and holds a commit naming a real person."
    echo "       Repair by hand:  git branch -D $BR"
    exit 1
  }
  if grep -q 'no real name' <<< "$out" && grep -qE '^ +commit [0-9a-f]+' <<< "$out"; then
    ok "a real name in a commit message fails, and the output names the commit"
  else bad "a real name in a commit message was not caught"; fi
  if grep -qE "^ +commit .*$NAME" <<< "$out"; then
    bad "the commit line echoed the name"
  else ok "the commit line names the commit without echoing the name"; fi
else
  bad "no identity string derived, commit-message path untested"
fi

echo
echo "── the ordering inside check.sh"

# A gate costing milliseconds placed after a 100-second sweep is a gate that gets interrupted.
g=$(grep -n '^run_gates$' check.sh | head -1 | cut -d: -f1)
w=$(grep -n 'ignored sweeps' check.sh | head -1 | cut -d: -f1)
if [ -n "$g" ] && [ -n "$w" ] && [ "$g" -lt "$w" ]; then
  ok "the gates run before the sweep (line $g against $w)"
else bad "the gates do not run before the sweep"; fi

# And a red gate must still make the whole script exit non-zero, not just print in red.
if grep -q 'exit "$fail"' check.sh; then
  ok "check.sh exits on its accumulated failure flag"
else bad "check.sh does not exit on the failure flag"; fi

echo
echo "── the real tree"
if ./check.sh --gates-only >/dev/null 2>&1; then
  ok "the tracked tree is clean: all three gates green, and they printed nothing"
else bad "the tracked tree does not pass its own gates"; fi

echo
printf '%s\n' "────────────────────────────────"
printf '%d passed, %d failed\n' "$pass" "$fail"
exit $(( fail > 0 ))
