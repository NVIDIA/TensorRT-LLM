#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The in-container half of a Goal 1.1 reference-ladder job. Invoked by both
# paths, exactly like staircase_v41_body.sh is:
#
#   staircase_v41_refladder.sbatch  -> sbatch, a fresh allocation per command
#   staircase_v41_refladder.sh      -> srun into the persistent allocation when
#                                      one is up, falling back to the sbatch path
#
#   bash staircase_v41_refladder_body.sh small [prompt_len] [tag]
#   bash staircase_v41_refladder_body.sh full  [num_prompts] [tag]
#   bash staircase_v41_refladder_body.sh assets
#   bash staircase_v41_refladder_body.sh lint
#
# Split into a real file for the reason the existing body script records: the
# body used to be a single-quoted string passed to `bash -c`, so a literal
# apostrophe or a quoted heredoc delimiter ended it early and the job died with
# "unexpected end of file" before running anything.
#
# THIS IS THE REFERENCE LEG. It runs the checkpoint's own inference/ package
# under the pinned reference venv and never imports tensorrt_llm, so the trtllm
# build bootstrap deliberately does not apply here. `assets` and `lint` touch no
# GPU and no model.

set -uo pipefail

MODE="${1:-small}"
ARG1="${2:-}"
TAG="${3:-${SLURM_JOB_ID:-local}}"

: "${REPO:=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/fredw/TensorRT-LLM}"
: "${MODELS:=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/fredw/models}"
: "${WORK:=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/fredw/staircase-v41}"

VENV=$WORK/refenv/venv
REF=$MODELS/DeepSeek-V4.1-Flash/inference
HF=$MODELS/DeepSeek-V4.1-Flash
CKPT=$WORK/ckpt-mp4
DIR=$REPO/tensorrt_llm/_torch/staircase/models/deepseek_v41/targets/v41_flash/sm_103/dep4
DRIVER=$DIR/refcapture.py

cd "$REPO" || exit 2
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$VENV/lib/python3.12/site-packages:${PYTHONPATH:-}"
export PYTHONPYCACHEPREFIX=$WORK/refenv/pycache
export TILELANG_CACHE_DIR=$WORK/refenv/tilelang-cache
export TOKENIZERS_PARALLELISM=false
# Read by the allocator before the first allocation; the in-process API is
# deprecated in this torch, so the environment variable is the supported route.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# --container-mount-home carries a user-site editable install whose .pth hook
# roots under the repo and leaks into sys.path. Disabling user site removes it.
export PYTHONNOUSERSITE=1

echo "node=$(hostname) mode=$MODE arg=$ARG1 tag=$TAG"
mkdir -p "$WORK/logs"

COMMON="--ref-dir $REF --hf-ckpt $HF --out $WORK/anchor --tag $TAG"

case "$MODE" in
  assets)
    # No GPU, no model, no reference venv needed: it reads files and hashes.
    python3 "$REPO/staircase_v41_assets_probe.py"
    rc=$?
    ;;
  hash)
    # The 487 GiB full-payload verification of the four converted shards,
    # against anchor/ckpt-mp4-sha256.txt. Separate from `assets` because it
    # reads every byte; `assets` then re-checks the receipt this writes, and
    # fails when it is absent or when a shard moved under it.
    python3 "$REPO/staircase_v41_assets_probe.py" --hash-shards
    rc=$?
    ;;
  lint)
    # The four gates in their NON-mutating form. Formatting runs before this, so
    # a clean result here means "was clean" rather than "was made clean".
    export PATH="$REPO/.venv-3.12/bin:$WORK/lintdeps/bin:$PATH"
    F="$DIR/refmods.py $DIR/refcapture.py $REPO/staircase_v41_assets_probe.py"
    rc=0
    ruff format --check $F       || rc=$?
    ruff check --select I $F     || rc=$?
    ruff check $F                || rc=$?
    ty check --python "$VENV" $F || rc=$?
    echo "lint: ruff format --check / ruff check --select I / ruff check / ty check -> $rc"
    # The reference leg must not reach through the package it is a reference
    # for, and refmods must not share a helper with the implementation it
    # checks. Both asserted rather than assumed.
    python3 "$REPO/staircase_v41_refladder_imports.py" "$DIR" || rc=$?
    ;;
  fmt)
    export PATH="$REPO/.venv-3.12/bin:$WORK/lintdeps/bin:$PATH"
    F="$DIR/refmods.py $DIR/refcapture.py $REPO/staircase_v41_assets_probe.py"
    rc=0
    ruff format $F                || rc=$?
    ruff check --select I --fix $F || rc=$?
    echo "fmt -> $rc"
    ;;
  small)
    nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader | head -1
    python3 -c "import tilelang, tvm_ffi; print('tilelang', tilelang.__version__, 'tvm_ffi', tvm_ffi.__version__)" || exit 3
    CUDA_VISIBLE_DEVICES=0 python3 "$DRIVER" $COMMON --mode small \
        --decode-steps 8 --small-prompt-len "${ARG1:-21}" --max-calls 6 --save-activations
    rc=$?
    ;;
  full)
    nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader | head -1
    python3 -c "import tilelang, tvm_ffi; print('tilelang', tilelang.__version__, 'tvm_ffi', tvm_ffi.__version__)" || exit 3
    # A deadline, because the comparisons issue collectives of their own from
    # inside forward hooks: every rank runs the same gated path, but a
    # divergence there does not raise, it WEDGES, and a wedged job records no
    # result while holding the allocation. 75 minutes covers a 4x121 GiB load
    # plus the five fixtures with room to spare.
    # No --decode-steps here, deliberately: full mode decodes each frozen
    # fixture to its own recorded length (7/24/48/2/48 = 129 tokens, 124 decode
    # steps). Passing a step count is how the transparency check came to be
    # made on a 9-token prefix of a 48-token continuation while reporting that
    # the whole thing matched.
    timeout --signal=INT 4500 \
    torchrun --nproc-per-node 4 "$DRIVER" $COMMON --mode full \
        --ckpt-path "$CKPT" --fixtures "$WORK/anchor/fixtures-frozen-frozen.json" \
        --num-prompts "${ARG1:-5}" --max-calls 3 \
        --layers 0,1,2,5,8,14,20,22,24,36,39 --save-activations
    rc=$?
    [ "$rc" -eq 124 ] && echo "full mode hit the 4500s deadline -- treat as a wedged collective, not a result"
    ;;
  *) echo "unknown mode $MODE"; rc=2 ;;
esac

echo "---- refladder $MODE exit $rc ----"
exit $rc
