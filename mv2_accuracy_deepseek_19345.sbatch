#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The deepseek-r1-0528-nvfp4 / dep4 whole-model gates, the way
# l0_gb300_multi_gpus runs them.
#
# Two job steps rather than one pytest invocation: TRTLLM_MODELING_V2 is read by
# the ranks as it stood when they started, so a mode is a property of the step.
# The `stock` leg deliberately asks for `off` -- it is the reference the
# acceptance anchor was populated from.
#
#SBATCH --job-name=mv2-acc-deepseek
#SBATCH --partition=batch
#SBATCH --account=coreai_comparch_trtllm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=04:00:00
#SBATCH --output=mv2-acc-deepseek-%j.log

set -uo pipefail

REPO=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/fredw/TensorRT-LLM
MODELS=/lustre/fsw/portfolios/coreai/users/fredw/llm-models
IMAGE=/lustre/fsw/portfolios/coreai/users/fredw/containers/trtllm-pytorch-26.08-py3-sbsa-ubuntu24.04-skip-tritondevel-202609210610-19345.sqsh

ENV_PREAMBLE='
set -uo pipefail
export VENV='"$REPO"'/.venv-3.12
export PATH="$VENV/bin:$PATH"
export PYTHONPATH='"$REPO"':${PYTHONPATH:-}
export LLM_MODELS_ROOT='"$MODELS"'
export PYTHONNOUSERSITE=1
export TLLM_LOG_LEVEL=INFO
# Stock cannot boot this checkpoint at dep4 with MTP under DeepEP: the low
# latency dispatch takes only NVFP4 uint8 hidden states and the MTP layer is
# bf16. Disabling it lands on AllGatherReduceScatter, which is the strategy the
# target implements by hand -- so it makes the two comparable, not less so.
export TRTLLM_CAN_USE_DEEP_EP=0
cd '"$REPO"'/tests
F=integration/defs/accuracy/test_modeling_v2_deepseek_v3.py
K=TestModelingV2DeepseekR10528Nvfp4Sm103Dep4
'

# One task per rank, session built from the allocation. MPI_Comm_spawn (the LLM
# API default for tp>1) hangs under srun/pyxis on this cluster, so the ranks are
# started by the launcher instead.
run4 () {
  srun --mpi=pmix --overlap --ntasks=4 \
       --container-image="$IMAGE" --container-mount-home \
       --container-mounts="$REPO:$REPO:rw,$MODELS:$MODELS:ro" \
       bash -c "$ENV_PREAMBLE
$1"
}

echo "=== step 1/2: TRTLLM_MODELING_V2=require ==="
run4 '
export TRTLLM_MODELING_V2=require
[ "${SLURM_PROCID:-0}" = "0" ] && { echo "node=$(hostname) models=$LLM_MODELS_ROOT"; git -C '"$REPO"' log --oneline -2; }
trtllm-llmapi-launch python3 -m pytest -q -s --timeout=10800 \
    "$F::$K::test_gsm8k_identity_vs_mtp3" \
    "$F::$K::test_mtp3_acceptance[modeling_v2]" 2>&1 | tee '"$REPO"'/mv2_require.out
rc=${PIPESTATUS[0]}
# A skipped case exits 0. The mode is set per job step, so a skip means the
# export did not reach the ranks -- which must fail, not look like a pass.
#
# pytest itself can exit 4 (USAGE_ERROR) on an all-pass run: a pre-existing
# nvidia_cutlass_dsl sys.path guard trips during collection teardown and has
# nothing to do with test outcome. Trusting that raw code is also what lets
# it leak into the job step exit status -- under srun --mpi=pmix --ntasks=4,
# one rank exiting nonzero makes srun force-terminate the other three
# (observed as the step reported exit flipping to 143/SIGTERM even though
# every test had already printed passed). So the summary line, not the exit
# code, decides pass/fail here, and rc is pinned to exactly 0 or 1
# accordingly -- never left as whatever pytest happened to return.
if grep -qE "[0-9]+ skipped" '"$REPO"'/mv2_require.out; then
    echo "FAIL: a case SKIPPED -- TRTLLM_MODELING_V2 did not reach the ranks"; rc=1
elif grep -qE "2 passed" '"$REPO"'/mv2_require.out; then
    rc=0
else
    echo "FAIL: expected exactly 2 passed"; rc=1
fi
[ "${SLURM_PROCID:-0}" = "0" ] && echo "---- require exit $rc ----"
exit $rc'
rc_require=$?

echo ""
echo "=== step 2/2: TRTLLM_MODELING_V2=off  (the stock reference leg) ==="
run4 '
export TRTLLM_MODELING_V2=off
trtllm-llmapi-launch python3 -m pytest -q -s --timeout=10800 \
    "$F::$K::test_mtp3_acceptance[stock]" 2>&1 | tee '"$REPO"'/mv2_off.out
rc=${PIPESTATUS[0]}
if grep -qE "[0-9]+ skipped" '"$REPO"'/mv2_off.out; then
    echo "FAIL: the stock leg SKIPPED"; rc=1
elif grep -qE "1 passed" '"$REPO"'/mv2_off.out; then
    rc=0
else
    echo "FAIL: expected exactly 1 passed"; rc=1
fi
[ "${SLURM_PROCID:-0}" = "0" ] && echo "---- off exit $rc ----"
exit $rc'
rc_off=$?

echo ""
echo "==================== RESULT ===================="
echo "require step exit: $rc_require"
echo "off step exit:     $rc_off"
echo ""
echo "A skipped case exits 0, so read the counts rather than the exit code:"
echo "  require step: $(grep -hoE '[0-9]+ (passed|failed|skipped)' "$REPO"/mv2_require.out 2>/dev/null | sort -u | tr '\n' ' ')"
echo "  off step:     $(grep -hoE '[0-9]+ (passed|failed|skipped)' "$REPO"/mv2_off.out 2>/dev/null | sort -u | tr '\n' ' ')"
echo ""
grep -hE "^\[staircase\]|^\[modeling_v2\]|^\[AL\]|gsm8k|acceptance_length" "$REPO"/mv2_require.out "$REPO"/mv2_off.out 2>/dev/null | tail -20
fail=0
[ $rc_require -ne 0 ] && fail=1
[ $rc_off -ne 0 ] && fail=1
echo "==================== deepseek: $([ $fail -eq 0 ] && echo PASS || echo FAIL) ===================="
exit $fail
