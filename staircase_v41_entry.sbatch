#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# One staircase-v41 entry command in its own fresh allocation.
#
#   sbatch staircase_v41_entry.sbatch test   unittest/_torch/staircase/gemm/test_staircase_mxfp8_mxfp8_gemm.py
#   sbatch staircase_v41_entry.sbatch probe  staircase_v41_mxfp8_guard_probe.py
#   sbatch staircase_v41_entry.sbatch lint   "<space separated files>"
#   sbatch staircase_v41_entry.sbatch fmt    "<space separated files>"
#   sbatch staircase_v41_entry.sbatch verify "<files>" <probe.py|-> <test-target|->
#
# PREFER `staircase_v41_run.sh`, which takes the same arguments, uses a
# persistent allocation when one is up, falls back to this script when it is
# not, and blocks until the result is in. This file remains the fallback path
# and the CLI is unchanged, so every command already cached against it still
# works.
#
# The work itself lives in staircase_v41_body.sh. It used to be inline here as
# a single-quoted string passed to `bash -c`, which meant a literal apostrophe
# or a quoted heredoc delimiter ended the string early and the job died with
# "unexpected end of file" before running anything. Two callers now need that
# body, and a real file has neither the escaping hazard nor a second copy to
# drift.
#
#SBATCH --job-name=staircase-v41-entry
#SBATCH --partition=batch
#SBATCH --account=coreai_comparch_trtllm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/fredw/staircase-v41/logs/entry-%j.log

set -uo pipefail

REPO=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/fredw/TensorRT-LLM
WORK=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/fredw/staircase-v41
# task.yaml's extra_container_mounts. Read-only, and only a probe reads it --
# a catalog GPU test must stay self-contained enough for CI to collect it, so
# the checkpoint's own tensors are measured here and quoted in the contract
# rather than opened from a test.
MODELS=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/fredw/models
IMAGE=/lustre/fsw/portfolios/coreai/users/fredw/containers/trtllm-pytorch-26.05-py3-sbsa-ubuntu24.04-skip-tritondevel-202608271702-17084.sqsh

srun --container-image="$IMAGE" \
     --container-mount-home \
     --container-mounts="$REPO:$REPO:rw,$MODELS:$MODELS:ro,$WORK:$WORK:rw" \
     bash "$REPO/staircase_v41_body.sh" "$@"
