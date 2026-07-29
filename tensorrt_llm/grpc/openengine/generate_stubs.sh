#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Regenerate the committed openengine gRPC Python stubs from the vendored
# .proto sources under proto/openengine/v1/.
#
# The .proto files are vendored from ai-dynamo/openengine at the commit
# recorded in PIN.md. To change the pin, re-vendor the .proto files first,
# then run this script.
#
# Usage: ./generate_stubs.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROTO_ROOT="$HERE/proto"
GRPCIO_TOOLS_VERSION="1.64.1"  # gencode targets Protobuf Python 5.26.x

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

python3 -m venv "$TMP/venv"
"$TMP/venv/bin/pip" install -q "grpcio-tools==${GRPCIO_TOOLS_VERSION}"

mkdir -p "$TMP/gen"
"$TMP/venv/bin/python" -m grpc_tools.protoc \
  -I "$PROTO_ROOT" \
  --python_out="$TMP/gen" \
  --grpc_python_out="$TMP/gen" \
  --pyi_out="$TMP/gen" \
  "$PROTO_ROOT"/openengine/v1/*.proto

# Clean-slate: drop previously generated stubs so a removed or renamed proto
# does not leave a stale module behind.
rm -f "$HERE"/*_pb2*.py "$HERE"/*_pb2*.pyi
cp "$TMP"/gen/openengine/v1/*_pb2.py \
   "$TMP"/gen/openengine/v1/*_pb2_grpc.py \
   "$TMP"/gen/openengine/v1/*_pb2.pyi \
   "$HERE/"

# protoc emits absolute imports ('from openengine.v1 import X'); make them
# package-relative so the flat, self-contained stubs resolve in-tree.
# Uses GNU sed in-place syntax (Linux dev containers).
sed -i -E 's/^from openengine\.v1 import /from . import /' \
  "$HERE"/*_pb2.py "$HERE"/*_pb2_grpc.py "$HERE"/*_pb2.pyi

echo "Regenerated openengine stubs in $HERE"
