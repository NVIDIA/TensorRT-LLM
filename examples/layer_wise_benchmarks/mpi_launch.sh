#!/bin/bash

set -euo pipefail

# Clear slurm envs
unset $(env | grep -i slurm | awk -F'=' '{print $1}')
unset $(env | grep MPI | awk -F'=' '{print $1}')

set -x
mpirun --allow-run-as-root --np ${NP} "$@"
