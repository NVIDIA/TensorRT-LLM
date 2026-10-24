#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Single-node MPI launch inside any spool job, with the same rules as mn_mpirun.sh:
#
#   spoolctl.py submit --spool <spool> --gpus all --tree <checkout> <name> \
#       'bash <scripts>/mn/mn_local_mpirun.sh 4 <command...>'
#
# The allocation's Slurm/PMIx context is removed, ranks start in the caller's
# working directory, and MN_TIMEOUT (default 0 inside a spool job, 7200
# outside) bounds the run.
set -u
NP=${1:?usage: mn_local_mpirun.sh <ranks> <command...>}
shift
for v in $(env | sed -n 's/^\(\(PMI\|PMIX\|OMPI\|SLURM\|PRTE\|PALS\|MPI_LOCALRANKID\|MPI_LOCALNRANKS\)[A-Za-z0-9_]*\)=.*/\1/p'); do
  unset "$v"
done
ulimit -c 0
# shellcheck disable=SC1090
[ -n "${MV2FLOW_MN_ENV:-}" ] && source "$MV2FLOW_MN_ENV"
echo "mn_local_mpirun: $(date -u +%H:%M:%SZ) np=$NP host=$(hostname) cwd=$PWD"
if [ -n "${MV2FLOW_SPOOL_JOB:-}" ]; then DEFAULT_TIMEOUT=0; else DEFAULT_TIMEOUT=7200; fi
TO=()
[ "${MN_TIMEOUT:-$DEFAULT_TIMEOUT}" != 0 ] && TO=(timeout -k 60 "${MN_TIMEOUT:-$DEFAULT_TIMEOUT}")
# shellcheck disable=SC2086
exec "${TO[@]}" mpirun -np "$NP" --oversubscribe --allow-run-as-root --bind-to none --wdir "$PWD" \
  --prtemca prte_tmpdir_base /tmp ${MN_MPIRUN_ARGS:-} "$@"
