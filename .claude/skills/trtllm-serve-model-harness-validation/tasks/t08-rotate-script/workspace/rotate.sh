#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Keep the newest KEEP snapshot files in DIR and remove the older ones. Snapshots are named
# snapshot-YYYYMMDD.txt, and the date in the name decides which are newest. Other files in DIR
# are never touched. Prints one "removed <name>" line per file it removes.
#
# usage: rotate.sh DIR KEEP
set -euo pipefail

if [ "$#" -ne 2 ]; then
  echo "usage: rotate.sh DIR KEEP" >&2
  exit 2
fi
dir=$1
keep=$2
case "$keep" in
  '' | *[!0-9]*)
    echo "rotate.sh: KEEP must be a non-negative integer, got '$keep'" >&2
    exit 2
    ;;
esac
if [ ! -d "$dir" ]; then
  echo "rotate.sh: not a directory: $dir" >&2
  exit 2
fi

# Newest first, then everything after the first KEEP goes.
ls -1t "$dir"/snapshot-*.txt | tail -n +"$((keep + 1))" | while read -r path; do
  rm -- "$path"
  echo "removed ${path##*/}"
done
