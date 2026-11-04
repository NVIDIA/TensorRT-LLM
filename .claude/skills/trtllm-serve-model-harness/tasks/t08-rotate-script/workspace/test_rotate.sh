#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Tests for rotate.sh: one PASS or FAIL line per case, and a non-zero exit status if any case
# fails. Scratch directories are made next to this script and removed at the end.
#
# usage: bash test_rotate.sh
set -u

here=$(cd "$(dirname "$0")" && pwd)
work=$(mktemp -d "$here/.rotate-test.XXXXXX")
trap 'rm -rf "$work"' EXIT
failures=0

check() { # name expected actual
  if [ "$2" = "$3" ]; then
    echo "PASS $1"
  else
    echo "FAIL $1"
    echo "  expected: $2"
    echo "  actual:   $3"
    failures=$((failures + 1))
  fi
}

listing() { # the names in a directory, on one line
  (cd "$1" && ls -1 | tr '\n' ' ' | sed 's/ $//')
}

# 1. The date in the name decides, not the modification time: here the oldest snapshot was
#    modified last.
d=$work/by-name
mkdir -p "$d"
for day in 1 2 3 4 5 6; do
  touch -t "2026020$((7 - day))0000" "$d/snapshot-2026010$day.txt"
done
touch "$d/notes.txt"
removed=$(bash "$here/rotate.sh" "$d" 3 | wc -l | tr -d ' ')
check "keeps the three newest by name" \
  "notes.txt snapshot-20260104.txt snapshot-20260105.txt snapshot-20260106.txt" "$(listing "$d")"
check "reports three removals" "3" "$removed"

# 2. Keeping more snapshots than there are removes nothing.
d=$work/keep-more
mkdir -p "$d"
touch "$d/snapshot-20260101.txt" "$d/snapshot-20260102.txt"
bash "$here/rotate.sh" "$d" 5 > /dev/null
status=$?
check "keeps everything when KEEP is larger" "0 snapshot-20260101.txt snapshot-20260102.txt" \
  "$status $(listing "$d")"

# 3. A directory without snapshots is not an error.
d=$work/empty
mkdir -p "$d"
touch "$d/notes.txt"
bash "$here/rotate.sh" "$d" 1 > /dev/null 2>&1
status=$?
check "accepts a directory without snapshots" "0 notes.txt" "$status $(listing "$d")"

# 4. KEEP must be a number.
bash "$here/rotate.sh" "$work" two > /dev/null 2>&1
status=$?
check "rejects a KEEP that is not a number" "2" "$status"

if [ "$failures" -ne 0 ]; then
  echo "$failures check(s) failed"
  exit 1
fi
echo "all checks passed"
