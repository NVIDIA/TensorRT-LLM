#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# slurm_mpi_env.sh - SOURCE this to make TRT-LLM's in-container MPI spawn work
# under srun/pyxis on a SINGLE node.
#
#   source scripts/bolt/slurm_mpi_env.sh
#
# Why: TRT-LLM's LLM() spawns its worker via mpi4py MPI.COMM_SELF.Spawn (even at
# tensor_parallel_size=1). Inside an srun-launched container the inherited
# SLURM_*/PMI*/PMIX_* variables make MPI believe it is part of SLURM's PMI
# world, so the in-container Spawn fails with MPI_ERR_SPAWN. Clearing those
# launcher variables lets the container's own MPI spawn standalone.
#
# NOTE: this blanket-clears SLURM_* and is therefore intended for SINGLE-NODE
# profiling collection only. Multi-node collection needs the allocation info
# preserved and a different launch strategy (the perf-sanity harness uses
# trtllm-llmapi-launch) -- do NOT source this for multi-node runs.

for _p in OMPI_ PMIX_ PMI_ SLURM_ MPI_ UCX_ I_MPI_ HYDRA_ KMP_ MPICH_ MV2_ CRAY_; do
    for _i in $(env | grep "^${_p}" | cut -d= -f1); do unset -v "$_i"; done
done
unset _p _i

export NCCL_IB_DISABLE=1
export UCX_TLS=tcp,cuda_copy,cuda_ipc
export OMPI_MCA_btl=vader,self
export UCX_CUDA_IPC_ENABLE_MNNVL=n
export UCX_RNDV_SCHEME=put_zcopy
unset UCX_NET_DEVICES

echo "[INFO] SLURM/MPI launcher env scrubbed for single-node in-container MPI spawn" >&2
