#!/bin/bash

# Set up error handling
set -xEeuo pipefail
trap 'rc=$?; echo "Error in file ${BASH_SOURCE[0]} on line $LINENO: $BASH_COMMAND (exit $rc)"; exit $rc' ERR

# Aggregated mode will run install together with pytest in slurm_run.sh
# Disaggregated mode will run install separately in slurm_install.sh
if [[ -z "${DISAGG_SERVING_TYPE:-}" ]]; then
    installScriptPath="$(dirname "${BASH_SOURCE[0]}")/slurm_install.sh"
    source "$installScriptPath"
    slurm_build_wheel
    slurm_install_setup
fi

cd $llmSrcNode/tests/integration/defs

# Force PMIx to use the in-memory hash GDS instead of ds12/ds21 shared-memory.
# Under `srun --mpi=pmix` with the DLFW 26.04 OpenMPI build, the shared-memory
# GDS modes can fail to publish UCX worker addresses across nodes, producing:
#   pml_ucx.c:178  Error: Failed to receive UCX worker address: Not found (-13)
#   pml_ucx.c:482  Error: Failed to resolve UCX endpoint for rank N
# See https://github.com/open-mpi/ompi/issues/6981. Setting this is a no-op
# when PMIx isn't used.
export PMIX_MCA_gds=hash

# Turn off "exit on error" so the following lines always run
set +e

# For disaggregated benchmark/server runs, clear all environment variables
# related to Slurm and MPI. This prevents test processes (e.g., pytest) from
# incorrectly initializing MPI when running under srun --overlap without
# --mpi=pmix. (Same fix as jenkins/scripts/slurm_run.sh)
if [ "${DISAGG_SERVING_TYPE:-}" == "BENCHMARK" ] || \
   [ "${DISAGG_SERVING_TYPE:-}" == "DISAGG_SERVER" ]; then
    for v in ${!PMI@} ${!PMIX@} ${!MPI@} ${!OMPI@} ${!SLURM@}; do
        if [ "$v" != "SLURM_PROCID" ]; then
            unset "$v"
        fi
    done
fi

pytest_exit_code=0

eval $pytestCommand
pytest_exit_code=$?
echo "Rank${SLURM_PROCID} Pytest finished execution with exit code $pytest_exit_code"

exit $pytest_exit_code
