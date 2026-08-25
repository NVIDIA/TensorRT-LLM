#!/bin/bash

# Set up error handling
set -xEeuo pipefail
trap 'rc=$?; echo "Error in file ${BASH_SOURCE[0]} on line $LINENO: $BASH_COMMAND (exit $rc)"; exit $rc' ERR

# TRT-LLM is installed in a separate srun step before this script runs.

# Export user-defined env vars
if [ -n "${CUSTOM_ENV:-}" ]; then
    eval "export $CUSTOM_ENV"
fi

# Single-node runs: strip MPI/SLURM env vars to prevent phantom MPI init
if [ "${SLURM_JOB_NUM_NODES:-1}" -eq 1 ]; then
    for v in ${!PMI@} ${!PMIX@} ${!MPI@} ${!OMPI@} ${!SLURM@}; do
        if [ "$v" != "SLURM_PROCID" ]; then
            unset "$v"
        fi
    done
fi

# Change to project root (not tests/integration/defs)
cd "${CUSTOM_WORKDIR:-$llmSrcNode}"

# Turn off "exit on error" so we capture the exit code
set +e

eval "$CUSTOM_COMMAND"
exit_code=$?
echo "Rank${SLURM_PROCID:-0} Custom command finished with exit code $exit_code"

exit $exit_code
