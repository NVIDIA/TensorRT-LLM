#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# setup_env.sh - Overlay a BOLT-compatible TRT-LLM wheel into a runtime-capable
# container WITHOUT clobbering the container's working `tensorrt` install.
#
#   scripts/bolt/setup_env.sh <dir-containing-TensorRT-LLM/>
#
# Background (the hard-won lesson): a plain `pip install <trtllm-wheel>` (with
# deps) makes pip re-satisfy the wheel's `tensorrt~=10.16.x` requirement by
# rebuilding the NVIDIA `tensorrt` meta source package, which produces a stub
# that provides NO importable `tensorrt` module -- it uninstalls the container's
# good TensorRT and breaks `import tensorrt` (and thus `import tensorrt_llm`).
# So we install the wheel with --no-deps (libs + python only, tensorrt
# untouched). The remaining runtime deps are expected to already be present in
# the pinned container image (kept in sync with requirements.txt), so we don't
# reinstall them here.
#
# Works for both a clean devel image (no trtllm preinstalled) and a release-like
# image (trtllm preinstalled): --force-reinstall overlays the BOLT libs either way.

set -euo pipefail

EXTRACT="${1:?usage: setup_env.sh <dir containing TensorRT-LLM/ (wheel)>}"
PYTHON="${PYTHON:-python3}"

WHEEL=$(ls "$EXTRACT"/TensorRT-LLM/tensorrt_llm-*.whl 2>/dev/null | head -1 || true)
[[ -f "$WHEEL" ]] || { echo "[ERROR] No tensorrt_llm wheel under $EXTRACT/TensorRT-LLM/"; exit 1; }

# 0. The container must already have a working tensorrt (runtime-capable image).
if ! "$PYTHON" -c "import tensorrt" 2>/dev/null; then
    echo "[ERROR] 'import tensorrt' fails in this container BEFORE install." >&2
    echo "        Use a runtime-capable image (devel/release) with TensorRT intact;" >&2
    echo "        a bare CUDA/pytorch base or an image with a stripped tensorrt will not work." >&2
    exit 1
fi
echo "[INFO] tensorrt OK: $("$PYTHON" -c 'import tensorrt; print(tensorrt.__version__)')"

# 1. Wheel WITHOUT deps -> never touches tensorrt. --force-reinstall so the BOLT
#    libs overlay any preinstalled trtllm (release image).
echo "[INFO] Installing BOLT wheel (--no-deps): $(basename "$WHEEL")"
pip install --no-deps --force-reinstall "$WHEEL"

# 2. Verify from a neutral cwd so the extracted source tree doesn't shadow the
#    installed package (a source `tensorrt_llm/` has no compiled bindings).
( cd /tmp && "$PYTHON" -c "import tensorrt, tensorrt_llm; print('[INFO] runtime + trtllm ok:', tensorrt_llm.__file__)" )
echo "[SUCCESS] Environment ready for BOLT flow."
