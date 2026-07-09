#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# setup_env.sh - Install a BOLT-compatible TRT-LLM wheel + its runtime deps
# into a runtime-capable container WITHOUT clobbering the container's working
# `tensorrt` install.
#
#   scripts/bolt/setup_env.sh <dir-containing-TensorRT-LLM/>
#
# Background (the hard-won lesson): a plain `pip install <trtllm-wheel>` (with
# deps) makes pip re-satisfy the wheel's `tensorrt~=10.16.x` requirement by
# rebuilding the NVIDIA `tensorrt` meta source package, which produces a stub
# that provides NO importable `tensorrt` module -- it uninstalls the container's
# good TensorRT and breaks `import tensorrt` (and thus `import tensorrt_llm`).
# This script avoids that by:
#   1. installing the wheel with --no-deps (libs + python only, tensorrt untouched)
#   2. installing the remaining deps with `tensorrt` excluded AND pinned to the
#      already-installed version so nothing reinstalls it.
#
# Works for both a clean devel image (no trtllm preinstalled) and a release-like
# image (trtllm preinstalled): --force-reinstall overlays the BOLT libs either way.

set -euo pipefail

EXTRACT="${1:?usage: setup_env.sh <dir containing TensorRT-LLM/ (wheel + src)>}"
PYTHON="${PYTHON:-python3}"

WHEEL=$(ls "$EXTRACT"/TensorRT-LLM/tensorrt_llm-*.whl 2>/dev/null | head -1 || true)
SRC="$EXTRACT/TensorRT-LLM/src"
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

# 2. Remaining runtime deps, tensorrt excluded + pinned to the installed version.
if [[ -f "$SRC/requirements.txt" ]]; then
    echo "[INFO] Installing runtime deps (tensorrt excluded + pinned)"
    ( cd "$SRC"
      grep -viE '^[[:space:]]*tensorrt([[:space:]=~<>!]|$)' requirements.txt > req.notrt.txt
      echo "tensorrt==$("$PYTHON" -c 'import tensorrt; print(tensorrt.__version__)')" > /tmp/keep-trt.txt
      pip install -c /tmp/keep-trt.txt -r req.notrt.txt )
else
    echo "[WARN] $SRC/requirements.txt not found; assuming deps already present"
fi

# The devel/release container already ships the full CUDA stack (e.g. 13.2).
# Installing requirements.txt pulls older pip cu13 runtime wheels (e.g.
# nvidia-nvjitlink/nvidia-cuda-nvrtc 13.0) into site-packages that SHADOW and
# downgrade the container's libs -- which breaks e.g. DS R1's MLA JIT kernel
# (needs nvJitLink >= 13.1). Remove them so the container's matched stack wins.
# (Pair this with sourcing cuda_env.sh to put the container CUDA libs on the path.)
echo "[INFO] Removing pip CUDA runtime libs that shadow the container's stack"
pip uninstall -y \
    nvidia-nvjitlink nvidia-cuda-nvrtc nvidia-cuda-runtime nvidia-cuda-cupti \
    2>/dev/null || true

# 3. Verify from a neutral cwd so the extracted source tree doesn't shadow the
#    installed package (a source `tensorrt_llm/` has no compiled bindings).
( cd /tmp && "$PYTHON" -c "import tensorrt, tensorrt_llm; print('[INFO] runtime + trtllm ok:', tensorrt_llm.__file__)" )
echo "[SUCCESS] Environment ready for BOLT flow."
