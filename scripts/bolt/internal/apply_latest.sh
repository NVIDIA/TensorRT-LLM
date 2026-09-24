#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# apply_latest.sh - premerge "consume the latest bundle" step.
#
# Pulls the branch-keyed `latest` BOLT profile bundle promoted by postmerge and
# applies it to a BOLT-compatible artifact, producing a bolted one (no
# recompile; apply_bolt.py swaps bolted ELFs into the wheel + tree).
#
# The artifact is either a release tarball or a standalone .whl -- the released
# SBSA manylinux wheel is built and uploaded outside any tarball, so it has to
# be BOLTed on its own. The mode is picked from the input extension.
#
# STRICT/FATAL by design. This runs only when the caller has opted into BOLT
# consumption (premerge BOLT_CONSUME), i.e. it expects a BOLTed build. Silently
# proceeding un-BOLTed would let premerge "pass" while actually testing the wrong
# binary, so every failure here is fatal (distinct non-zero codes for triage):
#   1  usage / input artifact missing
#   2  apply_bolt failed (bundle present but did not apply cleanly)
#   3  no bundle promoted for the branch (cold start / new branch): nothing to
#      consume -- await/trigger a postmerge BoltProfileGen (PROMOTE=true), or turn
#      BOLT_CONSUME off for this build.
#
# Usage: apply_latest.sh <branch> <triple> <in_artifact> <out_bolted_artifact>
#        <in_artifact> is a release tarball or a .whl.
# Requires: llvm-bolt on PATH; urm-artifactory-creds for artifactory.sh pull-latest.

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # .../scripts/bolt/internal
TOOLKIT="$(dirname "$HERE")"                            # .../scripts/bolt

BRANCH="${1:?apply_latest: <branch> <triple> <in_artifact> <out_bolted_artifact>}"
TRIPLE="${2:?triple required}"
IN_ARTIFACT="${3:?in_artifact required}"
OUT_ARTIFACT="${4:?out_bolted_artifact required}"

if [ ! -f "$IN_ARTIFACT" ]; then
    echo "[apply_latest] FATAL: input artifact not found: $IN_ARTIFACT" >&2
    exit 1
fi

DEST="$(mktemp -d)"
trap 'rm -rf "$DEST"' EXIT

# 1) Pull the branch `latest` bundle. A missing bundle is fatal here (see header):
#    consumption was requested but there is nothing promoted to consume.
if ! bash "$HERE/artifactory.sh" pull-latest "$BRANCH" "$TRIPLE" "$DEST"; then
    echo "[apply_latest] FATAL: no promoted bundle for ${BRANCH}/${TRIPLE}." >&2
    echo "[apply_latest]   run/await a postmerge BoltProfileGen (PROMOTE=true) or unset BOLT_CONSUME." >&2
    exit 3
fi

# 2) Apply the pulled profiles. Fatal on failure -- do NOT fall back to the
#    un-BOLTed artifact (that would silently test/ship the wrong binary).
#    --manifest verifies ELF hashes against the release layout, which only the
#    tarball has; a standalone wheel has no tree to walk, so it is omitted there.
case "$IN_ARTIFACT" in
    *.whl) APPLY_ARGS=(--wheel "$IN_ARTIFACT") ;;
    *)     APPLY_ARGS=(--tarball "$IN_ARTIFACT" --manifest "$DEST/manifest.json") ;;
esac

if ! python3 "$TOOLKIT/apply_bolt.py" \
        "${APPLY_ARGS[@]}" \
        --profiles "$DEST" \
        --output "$OUT_ARTIFACT"; then
    echo "[apply_latest] FATAL: apply_bolt failed for ${BRANCH}/${TRIPLE}" >&2
    exit 2
fi

echo "[apply_latest] bolted artifact -> $OUT_ARTIFACT"
