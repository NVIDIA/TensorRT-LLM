#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Collect what this branch changes in the release image into one payload directory: every file under tensorrt_llm/
# that the branch adds or modifies relative to the image commit, the list of files it deletes, and the compiled files
# rebuilt from this branch (the nanobind bindings, DeepEP and the two core libraries; nothing else in the wheel
# depends on the branch's C++ changes). apply_payload.sh then lays the payload over the installed package.
#
# Usage (from a checkout of this branch whose C++ has been built in place, e.g. scripts/build_wheel.py
#        --skip_building_wheel, which leaves the compiled files under tensorrt_llm/):
#   docker/qwen35_pareto_repro/make_payload.sh <out dir> [compiled-files root, default: <checkout>/tensorrt_llm]
set -euo pipefail
IMAGE_COMMIT=${IMAGE_COMMIT:-1b480b5af97dc063c7674900cdf15e65bbe521e2}  # nvidia/tensorrt-llm/release:1.3.0a0.dev202610151219
COMPILED=(bindings.cpython-312-aarch64-linux-gnu.so deep_ep_cpp_tllm.cpython-312-aarch64-linux-gnu.so
          libs/libtensorrt_llm.so libs/libth_common.so)
OUT=${1:?usage: make_payload.sh <out dir> [compiled-files root]}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(git -C "$HERE" rev-parse --show-toplevel)
LIBROOT=${2:-$REPO/tensorrt_llm}
cd "$REPO"
git merge-base --is-ancestor "$IMAGE_COMMIT" HEAD || { echo "HEAD does not contain the image commit $IMAGE_COMMIT" >&2; exit 1; }
if [ -n "$(git status --porcelain --untracked-files=no -- tensorrt_llm)" ]; then
  echo "uncommitted changes under tensorrt_llm/: commit them first, the payload is taken from HEAD" >&2
  exit 1
fi
[ -e "$OUT" ] && { echo "$OUT exists; use a new directory" >&2; exit 1; }
mkdir -p "$OUT/site"
# Python (and any other tracked file) under tensorrt_llm/, taken from HEAD, not from the working tree.
git diff --name-status --no-renames "$IMAGE_COMMIT" HEAD -- tensorrt_llm/ > "$OUT/changes.txt"
awk '$1 == "D" {print $2}' "$OUT/changes.txt" > "$OUT/deleted.txt"
awk '$1 != "D" {print $2}' "$OUT/changes.txt" > "$OUT/updated.txt"
if [ -s "$OUT/updated.txt" ]; then
  git archive --format=tar HEAD -- $(cat "$OUT/updated.txt") | tar -x -C "$OUT/site"
fi
# Compiled files from the build of this commit.
for f in "${COMPILED[@]}"; do
  [ -f "$LIBROOT/$f" ] || { echo "missing compiled file $LIBROOT/$f (build the C++ first)" >&2; exit 1; }
  install -D -m 0755 "$LIBROOT/$f" "$OUT/site/tensorrt_llm/$f"
done
printf '%s\n' "${COMPILED[@]}" > "$OUT/compiled.txt"
cp "$HERE/apply_payload.sh" "$OUT/"
(cd "$OUT/site" && find . -type f | LC_ALL=C sort | sed 's|^\./||' | xargs sha256sum) > "$OUT/MANIFEST.sha256"
{
  echo "branch $(git rev-parse --abbrev-ref HEAD) @ $(git rev-parse HEAD)"
  echo "image commit $IMAGE_COMMIT"
  echo "python files: $(wc -l < "$OUT/updated.txt") added/modified, $(wc -l < "$OUT/deleted.txt") deleted (changes.txt)"
  echo "compiled files from: $LIBROOT"
  echo "made $(date -Is) on $(hostname)"
  git log --oneline "$IMAGE_COMMIT..HEAD" | head -400
} > "$OUT/SOURCE.txt"
echo "payload $OUT: $(wc -l < "$OUT/updated.txt") files + ${#COMPILED[@]} compiled files, $(du -sh "$OUT" | cut -f1)"
