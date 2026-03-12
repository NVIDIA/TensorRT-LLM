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

# Process test list to separate regular and isolated tests
echo "Processing test list to extract regular and isolated tests..."
processScriptPath="$llmSrcNode/jenkins/scripts/process_test_list.py"

# Get the test list path from pytestCommand
# Extract --test-list parameter value
testListPath=$(echo "$pytestCommand" | grep -oP '(?<=--test-list=)[^ ]+' | head -1)
if [ -z "$testListPath" ]; then
    echo "WARNING: Could not extract test list path from command, skipping test list processing"
    testListPath=""
fi

regularTestList=""
isolateTestList=""

if [ -n "$testListPath" ] && [ -f "$testListPath" ]; then
    # Run process_test_list.py to separate tests
    processOutput=$(python3 "$processScriptPath" \
        --llm-src "$llmSrcNode" \
        --test-list "$testListPath" \
        --perf-mode)

    echo "$processOutput"

    # Extract paths from output
    while IFS='=' read -r key value; do
        if [ "$key" = "REGULAR_TEST_LIST" ]; then
            regularTestList="$value"
        elif [ "$key" = "ISOLATE_TEST_LIST" ]; then
            isolateTestList="$value"
        fi
    done <<< "$processOutput"

    echo "Regular tests: $regularTestList"
    echo "Isolated tests: $isolateTestList"
else
    echo "WARNING: Test list not found, will run all tests"
fi

# get trtllm wheel path and add to pytest command
trtllmWhlPath=$(pip3 show tensorrt_llm | grep Location | cut -d ' ' -f 2)
trtllmWhlPath=$(echo "$trtllmWhlPath" | sed 's/[[:space:]]+/_/g')
echo "TRTLLM WHEEL PATH: $trtllmWhlPath"
# In disaggregated mode, we only set coverage config file in benchmark pytest.
if [[ -z "${DISAGG_SERVING_TYPE:-}" || "${DISAGG_SERVING_TYPE}" == "BENCHMARK" ]]; then
    pytestCommand=$(set_value_in_command "TRTLLM_WHL_PATH" "$trtllmWhlPath" "$pytestCommand")
fi

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
env | sort

echo "Full Command: $pytestCommand"

# Separate handling for regular and isolated tests
regular_tests_exit_code=0
isolated_tests_exit_code=0

# Run regular tests (if there are any)
if [ -z "$regularTestList" ] || [ ! -f "$regularTestList" ] || [ ! -s "$regularTestList" ]; then
    echo "No regular tests to run or test list processing failed, running all tests with original command"

    # For single-node test runs, clear all environment variables related to Slurm and MPI.
    # This prevents test processes (e.g., pytest) from incorrectly initializing MPI
    # when running under a single-node srun environment.
    # TODO: check if we can take advantage of --export=None arg when execute srun instead
    # of unset them in the script
    if [ "${SLURM_JOB_NUM_NODES:-1}" -eq 1 ]; then
        for v in ${!PMI@} ${!PMIX@} ${!MPI@} ${!OMPI@} ${!SLURM@}; do
            if [ "$v" != "SLURM_PROCID" ]; then
                unset "$v"
            fi
        done
    fi

    set +e
    eval $pytestCommand
    regular_tests_exit_code=$?
    set -e
    echo "Rank${SLURM_PROCID} Regular tests (all) finished with exit code $regular_tests_exit_code"
else
    # Run regular tests only
    echo "Running regular tests from: $regularTestList"
    regularPytestCommand=$(echo "$pytestCommand" | sed "s|--test-list=[^ ]*|--test-list=$regularTestList|")
    echo "Regular tests command: $regularPytestCommand"

    # For single-node test runs, clear all environment variables related to Slurm and MPI.
    if [ "${SLURM_JOB_NUM_NODES:-1}" -eq 1 ]; then
        for v in ${!PMI@} ${!PMIX@} ${!MPI@} ${!OMPI@} ${!SLURM@}; do
            if [ "$v" != "SLURM_PROCID" ]; then
                unset "$v"
            fi
        done
    fi

    set +e
    eval $regularPytestCommand
    regular_tests_exit_code=$?
    set -e
    echo "Rank${SLURM_PROCID} Regular tests finished with exit code $regular_tests_exit_code"

    # Run isolated tests (if there are any)
    if [ -f "$isolateTestList" ] && [ -s "$isolateTestList" ]; then
        echo "Running isolated tests from: $isolateTestList"

        # Extract base command from pytestCommand (without --test-list and related args)
        basePytestCommand=$(echo "$pytestCommand" | sed 's|--test-list=[^ ]*||g' | sed 's|--csv=[^ ]*||g' | sed 's|--periodic-junit-xmlpath [^ ]*||g')

        # Run isolated tests
        set +e
        python3 "$llmSrcNode/jenkins/scripts/run_isolated_tests.py" \
            --llm-src "$llmSrcNode" \
            --isolate-list "$isolateTestList" \
            --stage-name "$stageName" \
            --output-dir "$stageName" \
            --base-cmd "$basePytestCommand"
        isolated_tests_exit_code=$?
        set -e
        echo "Rank${SLURM_PROCID} Isolated tests finished with exit code $isolated_tests_exit_code"
    fi
fi

# For single-node test runs, clear all environment variables related to Slurm and MPI.
# This prevents test processes (e.g., pytest) from incorrectly initializing MPI
# when running under a single-node srun environment.
# TODO: check if we can take advantage of --export=None arg when execute srun instead
# of unset them in the script
 if [ "${SLURM_JOB_NUM_NODES:-1}" -eq 1 ]; then
    for v in ${!PMI@} ${!PMIX@} ${!MPI@} ${!OMPI@} ${!SLURM@}; do
        if [ "$v" != "SLURM_PROCID" ]; then
            unset "$v"
        fi
    done
 fi

# Turn off "exit on error" so the following lines always run
set +e

pytest_exit_code=0
perf_check_exit_code=0
perf_report_exit_code=0

# Combine exit codes from regular and isolated tests
if [ "$regular_tests_exit_code" -ne 0 ]; then
    pytest_exit_code=$regular_tests_exit_code
elif [ "$isolated_tests_exit_code" -ne 0 ]; then
    pytest_exit_code=$isolated_tests_exit_code
fi

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

    echo "Create Perf Report"
    python3 $llmSrcNode/tests/integration/defs/perf/create_perf_comparison_report.py \
        --output_path $stageName/report.pdf \
        --files $stageName/perf_script_test_results.csv \
        $basePerfPath
    perf_report_exit_code=$?
    echo "Rank${SLURM_PROCID} Perf report finished execution with exit code $perf_report_exit_code"

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
