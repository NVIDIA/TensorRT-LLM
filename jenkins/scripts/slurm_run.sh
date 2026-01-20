#!/bin/bash

# Set up error handling
set -xEeuo pipefail
trap 'rc=$?; echo "Error in file ${BASH_SOURCE[0]} on line $LINENO: $BASH_COMMAND (exit $rc)"; exit $rc' ERR

cd $resourcePathNode
llmSrcNode=$resourcePathNode/TensorRT-LLM/src

set_value_in_command() {
    # Parameters
    local key="$1"
    local value="$2"
    local command="$3"

    # Transform the key
    local placeholder="__PLACEHOLDER_${key}__"

    # Check if placeholder exists
    if [[ "$command" != *"$placeholder"* ]]; then
        echo "Error: placeholder '$placeholder' not found in the command" >&2
        return 1
    fi

    # Replace all occurrences
    local result="${command//${placeholder}/${value}}"

    # Return the result
    echo "$result"
}

# Only the first process will set the git config
if [ $SLURM_PROCID -eq 0 ]; then
    # Update HOME/.gitconfig
    if ! git config --global --get-all safe.directory | grep -Fxq "*"; then
        git config --global --add safe.directory "*"
    fi
fi

# Aggregated mode will run install together with pytest in slurm_run.sh
# Disaggregated mode will run install separately in slurm_install.sh
if [[ "$stageName" != *Disagg* ]]; then
    installScriptPath="$(dirname "${BASH_SOURCE[0]}")/$(basename "${BASH_SOURCE[0]}" | sed 's/slurm_run\.sh/slurm_install.sh/')"
    source "$installScriptPath"
    slurm_install_setup
fi

if [[ "$stageName" == *GB200* ]]; then
    echo "Checking Coherent GPU mapping (for GB200)..."
    grep Coherent /proc/driver/nvidia/params || echo "Unable to grep Coherent from /proc/driver/nvidia/params"
fi

llmapiLaunchScript="$llmSrcNode/tensorrt_llm/llmapi/trtllm-llmapi-launch"
chmod +x $llmapiLaunchScript
cd $llmSrcNode/tests/integration/defs

# get trtllm wheel path and add to pytest command
trtllmWhlPath=$(pip3 show tensorrt_llm | grep Location | cut -d ' ' -f 2)
trtllmWhlPath=$(echo "$trtllmWhlPath" | sed 's/[[:space:]]+/_/g')
echo "TRTLLM WHEEL PATH: $trtllmWhlPath"
pytestCommand=$(set_value_in_command "TRTLLM_WHL_PATH" "$trtllmWhlPath" "$pytestCommand")

# Only the first process will save the coverage config file
if [ $SLURM_PROCID -eq 0 ]; then
    sed -i "s|---wheel_path---|$trtllmWhlPath|g" "$coverageConfigFile"
else
    # Sleep 30 seconds to wait for the coverage config file to be saved
    sleep 30
fi

containerPipLLMLibPath=$(pip3 show tensorrt_llm | grep "Location" | awk -F ":" '{ gsub(/ /, "", $2); print $2"/tensorrt_llm/libs"}')
containerPipLLMLibPath=$(echo "$containerPipLLMLibPath" | sed 's/[[:space:]]+/_/g')
containerLDLibPath=$LD_LIBRARY_PATH
containerLDLibPath=$(echo "$containerLDLibPath" | sed 's/[[:space:]]+/_/g')
if [[ "$containerLDLibPath" != *"$containerPipLLMLibPath"* ]]; then
    containerLDLibPath="$containerPipLLMLibPath:$containerLDLibPath"
    containerLDLibPath="${containerLDLibPath%:}"
fi
export LD_LIBRARY_PATH=$containerLDLibPath
echo "Library Path:"
echo "$LD_LIBRARY_PATH"

echo "Environment Variables (Before):"
env | sort

echo "Full Command: $pytestCommand"

# For single-node test runs, clear all environment variables related to Slurm and MPI.
# This prevents test processes (e.g., pytest) from incorrectly initializing MPI
# when running under a single-node srun environment.
# TODO: check if we can take advantage of --export=None arg when execute srun instead
# of unset them in the script
if [ "${SLURM_JOB_NUM_NODES:-1}" -eq 1 ]; then
    # Unset MPI/SLURM variables except essential SLURM vars needed for testing and debugging
    # Pad with spaces for exact matching
    keep_vars=" SLURM_PROCID SLURM_LOCALID SLURM_NODEID SLURMD_NODENAME "
    keep_vars+="SLURM_JOB_ID SLURM_JOB_NAME SLURM_JOB_NUM_NODES SLURM_JOB_NODELIST SLURM_RESTART_COUNT "
    keep_vars+="SLURM_CPUS_ON_NODE SLURM_GPUS_ON_NODE SLURM_GPUS_PER_NODE "
    keep_vars+="SLURM_CLUSTER_NAME SLURM_JOB_PARTITION SLURM_JOB_ACCOUNT SLURM_JOB_USER "
    keep_vars+="SLURM_SUBMIT_DIR SLURM_SUBMIT_HOST "

    for v in $(env | grep -E "^(OMPI_|PMIX_|PMI_|SLURM_|MPI_|UCX_|I_MPI_|HYDRA_|KMP_|MPICH_|MV2_|CRAY_)" | cut -d= -f1); do
        # Use bash native string matching (avoid forking echo/grep per variable)
        if [[ "$keep_vars" != *" $v "* ]]; then
            unset "$v"
        fi
    done

    # Force UCX to use TCP and CUDA IPC, disable InfiniBand
    export UCX_TLS=tcp,cuda_copy,cuda_ipc
    export NCCL_IB_DISABLE=1

    # Prevent NCCL from loading external network plugins (avoids hangs)
    export NCCL_NET_PLUGIN=none

    # Configure OpenMPI for shared memory/loopback
    export OMPI_MCA_btl=vader,self

    # Disable UCX Multi-Node NVLink and use stable rendezvous protocol
    export UCX_CUDA_IPC_ENABLE_MNNVL=n
    export UCX_RNDV_SCHEME=put_zcopy
fi

echo "Environment Variables (After):"
env | sort

# Attempt to avoid "unrecognized arguments" failure (Exit Code 4) from relaunching
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.pyc" -delete 2>/dev/null || true
rm -rf .pytest_cache 2>/dev/null || true
export PYTHONPATH=$llmSrcNode/tests/integration:$llmSrcNode/examples/auto_deploy:${PYTHONPATH:-}

# Turn off "exit on error" so the following lines always run
set +e
trap - ERR

pytest_exit_code=0
perf_check_exit_code=0
perf_report_exit_code=0

# "-c pytest.ini" is attempt to avoid "unrecognized arguments" failure (Exit Code 4) from relaunching
eval $pytestCommand -c pytest.ini
pytest_exit_code=$?
echo "Rank${SLURM_PROCID} Pytest finished execution with exit code $pytest_exit_code"

# DEBUG: Diagnose intermittent "unrecognized arguments" failure (Exit Code 4)
# Remove this after the issue is resolved
if [ $pytest_exit_code -eq 4 ]; then
    echo "DEBUG: Pytest failed with usage error (exit code 4)"
    echo "DEBUG: Directory state at $(pwd):"
    ls -l
    echo "DEBUG: Directory state at $llmSrcNode/tests/integration/defs:"
    ls -l $llmSrcNode/tests/integration/defs

    echo "DEBUG: conftest.py content:"
    md5sum $llmSrcNode/tests/integration/defs/conftest.py

    echo "DEBUG: pytest.ini content:"
    md5sum $llmSrcNode/tests/integration/defs/pytest.ini

    echo "DEBUG: Check importability of conftest.py"
    python3 -c "import sys; sys.path.insert(0, '.'); import conftest; print('DEBUG: conftest imported successfully')"
fi

if [ $SLURM_PROCID -eq 0 ] && [ "$perfMode" = "true" ]; then
    if [[ "$stageName" == *PyTorch* ]]; then
        basePerfFilename="base_perf_pytorch.csv"
    else
        basePerfFilename="base_perf.csv"
    fi
    basePerfPath="$llmSrcNode/tests/integration/defs/perf/$basePerfFilename"
    echo "Check Perf Result"
    python3 $llmSrcNode/tests/integration/defs/perf/sanity_perf_check.py \
        $stageName/perf_script_test_results.csv \
        $basePerfPath
    perf_check_exit_code=$?
    echo "Rank${SLURM_PROCID} Check perf result finished execution with exit code $perf_check_exit_code"

    echo "Create Perf Report"
    python3 $llmSrcNode/tests/integration/defs/perf/create_perf_comparison_report.py \
        --output_path $stageName/report.pdf \
        --files $stageName/perf_script_test_results.csv \
        $basePerfPath
    perf_report_exit_code=$?
    echo "Rank${SLURM_PROCID} Create perf report finished execution with exit code $perf_report_exit_code"

    if [ "$perf_check_exit_code" -eq 0 ] && [ "$perf_report_exit_code" -ne 0 ]; then
        perf_check_exit_code=$perf_report_exit_code
    fi
    echo "Rank${SLURM_PROCID} Perf check finished execution with exit code $perf_check_exit_code"
fi

if [ "$pytest_exit_code" -ne 0 ]; then
    final_exit_code=$pytest_exit_code
elif [ "$perf_check_exit_code" -ne 0 ]; then
    final_exit_code=$perf_check_exit_code
else
    final_exit_code=0
fi
echo "Rank${SLURM_PROCID} Final Slurm run finished execution with exit code $final_exit_code"
exit $final_exit_code
