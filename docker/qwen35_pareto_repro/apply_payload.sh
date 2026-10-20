#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Runs INSIDE the base release image as root (Dockerfile RUN, or a pyxis step with --container-remap-root and
# --container-save): lays a make_payload.sh payload over the installed tensorrt_llm package, removes the files the
# branch deletes and their stale bytecode, checks every installed file against the payload's sha256 manifest and
# records the provenance under /opt/qwen35_pareto_repro.
# Usage: apply_payload.sh <payload dir>
set -euo pipefail
P=${1:?usage: apply_payload.sh <payload dir>}
SITE=${SITE:-/usr/local/lib/python3.12/dist-packages}
PKG=$SITE/tensorrt_llm
[ -d "$PKG/_torch" ] && [ -f "$PKG/libs/libtensorrt_llm.so" ] || { echo "no installed tensorrt_llm under $SITE" >&2; exit 1; }
# Bytecode of every replaced or deleted module goes too: the image's .pyc must not shadow the new sources.
while read -r f; do
  [ -n "$f" ] || continue
  rel=${f#tensorrt_llm/}
  rm -f "$PKG/$rel"
  base=$(basename "$rel" .py)
  rm -f "$PKG/$(dirname "$rel")/__pycache__/$base".*.pyc
done < <(cat "$P/updated.txt" "$P/deleted.txt")
cp -a --no-preserve=ownership "$P/site/tensorrt_llm/." "$PKG/"
(cd "$SITE" && sha256sum --quiet -c "$P/MANIFEST.sha256") || { echo "installed files do not match MANIFEST.sha256" >&2; exit 1; }
while read -r f; do
  [ -n "$f" ] && [ -e "$SITE/$f" ] && { echo "deleted file still present: $f" >&2; exit 1; }
done < "$P/deleted.txt"
mkdir -p /opt/qwen35_pareto_repro
cp "$P/SOURCE.txt" "$P/MANIFEST.sha256" "$P/changes.txt" "$P/compiled.txt" /opt/qwen35_pareto_repro/
echo "applied $(date -Is): $(head -1 "$P/SOURCE.txt")" >> /opt/qwen35_pareto_repro/IMAGE_INFO.txt
echo "payload applied: $(wc -l < "$P/MANIFEST.sha256") files verified under $SITE"
