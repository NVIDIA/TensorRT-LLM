#!/bin/bash

# Shared runtime-environment setup for SLURM test steps. Sourced by
# slurm_run.sh (real test steps) AND by
# jenkins/scripts/perf/disaggregated/slurm_precheck_run.sh (the cache
# transceiver precheck), so both run with IDENTICAL library paths and
# UCX/PMIx fixups -- the precheck must observe exactly the network
# environment the real disaggregated workers will use. Keep any change here
# valid for both callers.

slurm_setup_runtime_env() {
    # Prepend the installed tensorrt_llm wheel's libs to LD_LIBRARY_PATH.
    local containerPipLLMLibPath
    containerPipLLMLibPath=$(pip3 show tensorrt_llm | grep "Location" | awk -F ":" '{ gsub(/ /, "", $2); print $2"/tensorrt_llm/libs"}')
    containerPipLLMLibPath=$(echo "$containerPipLLMLibPath" | sed 's/[[:space:]]+/_/g')
    local containerLDLibPath=$LD_LIBRARY_PATH
    containerLDLibPath=$(echo "$containerLDLibPath" | sed 's/[[:space:]]+/_/g')
    if [[ "$containerLDLibPath" != *"$containerPipLLMLibPath"* ]]; then
        containerLDLibPath="$containerPipLLMLibPath:$containerLDLibPath"
        containerLDLibPath="${containerLDLibPath%:}"
    fi
    export LD_LIBRARY_PATH=$containerLDLibPath

    # Slurm ENROOT/pyxis may inject UCX_TLS=tcp from the host MPI stack
    # (intended for host-only MPI jobs). That disables CUDA transports and
    # breaks NIXL GPU memory registration. Unset it so UCX can auto-select.
    if [ "${UCX_TLS:-}" = "tcp" ]; then
        unset UCX_TLS
        echo "Unset UCX_TLS (cluster injected UCX_TLS=tcp)"
    fi

    # Force PMIx to use the in-memory hash GDS instead of ds12/ds21
    # shared-memory. Under `srun --mpi=pmix` with the DLFW 26.04 OpenMPI
    # build, the shared-memory GDS modes can fail to publish UCX worker
    # addresses across nodes, producing:
    #   pml_ucx.c:178  Error: Failed to receive UCX worker address: Not found (-13)
    #   pml_ucx.c:482  Error: Failed to resolve UCX endpoint for rank N
    # See https://github.com/open-mpi/ompi/issues/6981. Setting this is a
    # no-op when PMIx isn't used.
    export PMIX_MCA_gds=hash
}
