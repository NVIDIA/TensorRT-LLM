#!/bin/bash

set -euo pipefail

# Clear slurm envs
unset $(env | grep -i slurm | awk -F'=' '{print $1}')
unset $(env | grep MPI | awk -F'=' '{print $1}')

extra_args=
if [ -v TLLM_AUTOTUNER_CACHE_PATH ]; then
    extra_args+="-x TLLM_AUTOTUNER_CACHE_PATH"
fi

set -x
mpirun --allow-run-as-root --np ${NP} $extra_args "$@"
