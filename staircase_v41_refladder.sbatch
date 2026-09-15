#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Goal 1.1 rung 2: align the pure-PyTorch module references (refmods.py)
# against the checkpoint's own inference/ forward (refcapture.py).
#
#   sbatch staircase_v41_refladder.sbatch small [prompt_len] [tag]
#   sbatch staircase_v41_refladder.sbatch full  [num_prompts] [tag]
#   sbatch staircase_v41_refladder.sbatch assets
#   sbatch staircase_v41_refladder.sbatch hash
#   sbatch staircase_v41_refladder.sbatch fmt
#   sbatch staircase_v41_refladder.sbatch lint
#
# The fallback path. Prefer `bash staircase_v41_refladder.sh <mode> ...`, which
# runs the same body through a held allocation when one is up.
#
#SBATCH --job-name=staircase-v41-refladder
#SBATCH --partition=batch
#SBATCH --account=coreai_comparch_trtllm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/fredw/staircase-v41/logs/refladder-%j.log

set -uo pipefail

REPO=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/fredw/TensorRT-LLM
MODELS=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/fredw/models
WORK=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/fredw/staircase-v41
IMAGE=/lustre/fsw/portfolios/coreai/users/fredw/containers/trtllm-pytorch-26.05-py3-sbsa-ubuntu24.04-skip-tritondevel-202608271702-17084.sqsh

srun --container-image="$IMAGE" \
     --container-mount-home \
     --container-mounts="$REPO:$REPO:rw,$MODELS:$MODELS:ro,$WORK:$WORK:rw" \
     bash "$REPO/staircase_v41_refladder_body.sh" "$@"
rc=$?
echo "---- refladder job exit $rc ----"
exit $rc
