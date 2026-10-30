#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run a command on N GPUs.
#
#   spoolctl.py submit free-<name> 'bash <scripts>/gpu_slot.sh N <command...>'
#   spoolctl.py submit --gpus N <name> '<command...>'          # the same thing, preferred
#
# Inside a spool job the scheduler has already allocated the devices: it reads
# `gpu_slot.sh N` in the job body as `gpus=N`, exports CUDA_VISIBLE_DEVICES to
# exactly those devices and removes the allocation's MPI context. This script
# then only checks that the job holds enough devices and runs the command.
#
# Outside a spool job it is a convenience wrapper: it removes the MPI context,
# pins CUDA_VISIBLE_DEVICES to the first N devices if unset, and applies a
# deadline (GPU_SLOT_TIMEOUT seconds, default 7200, 0 = none).
ulimit -c 0
set -u
N=${1:?usage: gpu_slot.sh <ngpus> <command...>}
shift

if [ -n "${MV2FLOW_SPOOL_JOB:-}" ]; then
  held=0
  [ -n "${MV2FLOW_SPOOL_DEVICES:-}" ] && held=$(awk -F, '{print NF}' <<< "$MV2FLOW_SPOOL_DEVICES")
  if [ "$N" != all ] && [ "$held" -lt "$N" ]; then
    echo "gpu_slot: job $MV2FLOW_SPOOL_JOB holds $held device(s) but asks for $N; submit it with --gpus $N" >&2
    exit 64
  fi
  echo "gpu_slot: CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-} (allocated by the spool scheduler)"
  if [ -n "${GPU_SLOT_TIMEOUT:-}" ] && [ "${GPU_SLOT_TIMEOUT}" != 0 ]; then
    exec timeout -k 60 "$GPU_SLOT_TIMEOUT" "$@"
  fi
  exec "$@"
fi

for v in $(env | sed -n 's/^\(\(PMI\|PMIX\|OMPI\|SLURM\|PRTE\|PALS\|MPI_LOCALRANKID\|MPI_LOCALNRANKS\)[A-Za-z0-9_]*\)=.*/\1/p'); do
  unset "$v"
done
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ "$N" != all ]; then
  CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((N - 1)))
  export CUDA_VISIBLE_DEVICES
fi
echo "gpu_slot: not inside a spool job; no slot accounting. CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<all>}" >&2
if [ "${GPU_SLOT_TIMEOUT:-7200}" != 0 ]; then
  timeout -k 60 "${GPU_SLOT_TIMEOUT:-7200}" "$@"
  rc=$?
  [ "$rc" = 124 ] && echo "gpu_slot: TIMED OUT after ${GPU_SLOT_TIMEOUT:-7200}s -- the job was stopped, not failed" >&2
  exit $rc
fi
exec "$@"
