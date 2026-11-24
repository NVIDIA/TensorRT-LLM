#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Launcher for one rank of the DWDP context-server MPI step, used in place of
# trtllm-llmapi-launch. Every rank runs the perf-sanity pytest, which starts
# `trtllm-serve disaggregated_mpi_worker` as this rank's MPI process.
#
# pytest imports tensorrt_llm, which initializes MPI. If it saw this task's
# PMIx identity it would claim the rank its trtllm-serve child needs, so the
# MPI/Slurm variables are hidden from pytest (the same set trtllm-llmapi-launch
# hides from its rank-0 pytest) and kept as TLLM_DWDP_SAVED_<name>. pytest
# restores them for the child only.
#
# The ranks of a node share one container, and the step starts their pytest
# together. Only the node's first rank lets pytest pip-install the media deps
# (TRTLLM_AUTO_INSTALL_MEDIA_DEPS in tests/integration/defs/conftest.py):
# concurrent installs into one site-packages fail with OSError, and the failed
# rank takes the whole step down.
#
# Usage: bash dwdp_ctx_rank.sh pytest <args>...

set -euo pipefail

if [[ "${SLURM_LOCALID:-0}" != "0" ]]; then
    export TRTLLM_AUTO_INSTALL_MEDIA_DEPS=0
fi

mpi_prefixes=(OMPI_ PMIX_ PMI_ SLURM_ MPI_ I_MPI_ HYDRA_ MPICH_ MV2_ CRAY_)
for var in $(compgen -e); do
    for prefix in "${mpi_prefixes[@]}"; do
        if [[ "$var" == "$prefix"* ]]; then
            export "TLLM_DWDP_SAVED_${var}=${!var}"
            unset "$var"
            break
        fi
    done
done

exec "$@"
