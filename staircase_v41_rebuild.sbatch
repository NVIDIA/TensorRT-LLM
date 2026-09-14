#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Incremental rebuild of this checkout's native artifacts after a C++/CUDA
# change, plus the check that actually matters afterwards: the new
# torch.ops.trtllm symbol has to be exported by a library IN THE CHECKOUT,
# because PYTHONPATH is what makes the checkout the thing that imports and a
# Python tree referencing an op no built library carries raises at import.
#
# Modelled on staircase_rebuild.sbatch; same build command, same ccache and
# keepalive requirements, only the symbol it verifies differs.
#
#SBATCH --job-name=staircase-v41-rebuild
#SBATCH --partition=batch
#SBATCH --account=coreai_comparch_trtllm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/fredw/staircase-v41/logs/rebuild-%j.log

set -euo pipefail

SCRATCH=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/fredw
REPO=$SCRATCH/TensorRT-LLM
IMAGE=/lustre/fsw/portfolios/coreai/users/fredw/containers/trtllm-pytorch-26.05-py3-sbsa-ubuntu24.04-skip-tritondevel-202608271702-17084.sqsh

# The symbol the rebuild exists for.
SYMBOL="${1:-mhc_split_sinkhorn}"
shift || true
# The C++ files to clang-format-check before building. Defaults to the mHC trio
# this script was first written for; pass the files a Goal actually touched, or
# the gate silently checks somebody else's already-clean files.
FMT_FILES="${*:-cpp/tensorrt_llm/kernels/mhcKernels/mhcKernels.cu cpp/tensorrt_llm/kernels/mhcKernels/mhcKernels.h cpp/tensorrt_llm/thop/mhcOp.cpp}"

srun --container-image="$IMAGE" \
     --container-mount-home \
     --container-mounts="$SCRATCH:$SCRATCH:rw" \
     bash -c '
set -euo pipefail
export HOME='"$SCRATCH"'/compile_home
export CCACHE_DIR='"$SCRATCH"'/.ccache_trtllm
export CCACHE_MAXSIZE=50G
export SYMBOL='"$SYMBOL"'
export FMT_FILES="'"$FMT_FILES"'"
cd '"$REPO"'

# The idle-GPU watchdog kills jobs whose GPUs sit at 0% util (~10 min). This
# build is CPU-bound, so keep the allocated GPUs nominally busy.
python3 - <<PY &
import time
try:
    import torch
    xs = [torch.randn(512, 512, device=f"cuda:{i}")
          for i in range(torch.cuda.device_count())]
    while True:
        for x in xs:
            _ = (x @ x).sum().item()
        time.sleep(20)
except Exception as e:
    print(f"[keepalive] stopped: {e}", flush=True)
PY
KA=$!
trap "kill $KA 2>/dev/null || true" EXIT

echo "[build] node=$(hostname) date=$(date) symbol=$SYMBOL"

# The repo formats C++ with clang-format v16 via pre-commit
# (.pre-commit-config.yaml, mirrors-clang-format rev v16.0.0) and this image
# carries no clang-format at all, so install the pinned version into the same
# shared prefix the Python linters use. Checked BEFORE the build, so a reformat
# cannot invalidate the artifacts it produced.
CFDIR='"$SCRATCH"'/staircase-v41/lintdeps
# pip on this image uses the Debian posix_local scheme, so --prefix lands the
# console script in $CFDIR/local/bin rather than $CFDIR/bin.
export PATH="$CFDIR/bin:$CFDIR/local/bin:$PATH"
if ! command -v clang-format >/dev/null 2>&1; then
  # NOT --quiet: the first attempt hid its own failure behind `|| true` and the
  # job then reported "unavailable" with no reason in the log.
  pip install --prefix "$CFDIR" --no-cache-dir clang-format==16.0.0 2>&1 | tail -6 || true
fi
# The wheel ships the real ELF under clang_format/data/bin and a python
# console-script shim in local/bin that only works with its own site-packages
# importable. Prefer the ELF: it needs no interpreter and no PYTHONPATH.
CF=$(find "$CFDIR" -path "*clang_format/data/bin/clang-format" -type f 2>/dev/null | head -1)
if [ -z "$CF" ]; then
  CF=$(find /usr/lib/llvm-* /usr/bin /opt -maxdepth 3 -name clang-format -type f 2>/dev/null | head -1)
fi
if [ -z "$CF" ]; then
  CF=$(command -v clang-format 2>/dev/null || true)
fi
if [ -n "$CF" ] && "$CF" --version >/dev/null 2>&1; then
  "$CF" --version
  echo "[fmt] checking: $FMT_FILES"
  if "$CF" --dry-run -Werror $FMT_FILES; then
    echo "[fmt] clang-format clean"
  else
    echo "[fmt] clang-format REPORTED DIFFS (see above)"; exit 4
  fi
else
  echo "[fmt] clang-format unavailable (CF=$CF)"; exit 4
fi

ccache -s 2>/dev/null | head -5 || true
python3 ./scripts/build_wheel.py --use_ccache -G Ninja -a "103-real" --nvtx
echo "[build] DONE date=$(date)"

# grep -a rather than `strings`: this image carries no binutils, and
# `strings | grep -q` then evaluates false for every library instead of
# failing loudly -- which is exactly how a good build reported a missing
# symbol on job 733878.
found=0
for so in tensorrt_llm/libs/*.so tensorrt_llm/*.so; do
  [ -e "$so" ] || continue
  if grep -aq "$SYMBOL" "$so"; then
    echo "[build] $SYMBOL present in $so"
    found=1
  fi
done
if [ $found -eq 0 ]; then echo "[build] FAILED: $SYMBOL in no checkout library"; exit 3; fi

# And the check the symbol scan cannot make: the op is registered, callable,
# and reached from the checkout.
export VENV='"$REPO"'/.venv-3.12
export PATH="$VENV/bin:$PATH"
export PYTHONPATH='"$REPO"':${PYTHONPATH:-}
export PYTHONNOUSERSITE=1
export TRTLLM_STAIRCASE=require
python3 - <<PY
import os, torch, tensorrt_llm
print("trtllm", tensorrt_llm.__version__, "from", tensorrt_llm.__file__)
assert tensorrt_llm.__file__.startswith(os.environ["PYTHONPATH"].split(":")[0])
import tensorrt_llm._torch.custom_ops  # noqa: F401
op = getattr(torch.ops.trtllm, os.environ["SYMBOL"], None)
assert op is not None, "op not registered"
print("registered:", op, "schema:", op.default._schema)
PY
'
