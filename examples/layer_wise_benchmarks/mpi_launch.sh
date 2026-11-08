#!/bin/bash

set -euo pipefail

# Clear slurm envs
unset $(env | awk -F'=' '{print $1}' | (grep -E "SLURM_|SLURMD_|slurm_|MPI_|PMIX_" || true))

extra_args=()
if [ -v TLLM_AUTOTUNER_CACHE_PATH ]; then
    extra_args+=(-x TLLM_AUTOTUNER_CACHE_PATH)
fi

set -x
mpirun --allow-run-as-root --np $NP ${extra_args[@]+"${extra_args[@]}"} "$@"
