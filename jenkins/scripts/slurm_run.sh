#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Set up error handling
set -xEeuo pipefail
trap 'rc=$?; echo "Error in file ${BASH_SOURCE[0]} on line $LINENO: $BASH_COMMAND (exit $rc)"; exit $rc' ERR

cd $resourcePathNode
llmSrcNode=$resourcePathNode/TensorRT-LLM/src

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

# Wheel path for the CBTS .coveragerc @TRTLLM_WHEEL_PATH@ substitution below.
trtllmWhlPath=$(pip3 show tensorrt_llm | grep Location | cut -d ' ' -f 2)
trtllmWhlPath=$(echo "$trtllmWhlPath" | sed 's/[[:space:]]+/_/g')
echo "TRTLLM WHEEL PATH: $trtllmWhlPath"

# Only the first process will save the coverage config file
if [ $SLURM_PROCID -eq 0 ]; then
    sed -i "s|@TRTLLM_WHEEL_PATH@|$trtllmWhlPath|g" "$coverageConfigFile"
else
    # Sleep 30 seconds to wait for the coverage config file to be saved
    sleep 30
fi

# Library path + UCX/PMIx fixups shared with the disagg cache-transceiver
# precheck (slurm_precheck_run.sh) -- keeping them in one place guarantees the
# precheck observes the same network environment as the real test steps.
source "$llmSrcNode/jenkins/scripts/slurm_env_setup.sh"
slurm_setup_runtime_env
echo "Library Path:"
echo "$LD_LIBRARY_PATH"
env | sort

echo "Full Command: $pytestCommand"

# Multiple srun steps may run concurrently in disaggregated performance
# stages. Preserve this step identifier before the test environment removes
# Slurm variables, so rank-local artifacts cannot collide between steps.
slurm_step_id="${SLURM_STEP_ID:-0}"

# For single-node test runs or disaggregated benchmark/server runs, clear all
# environment variables related to Slurm and MPI. This prevents test processes
# (e.g., pytest) from incorrectly initializing MPI when running under a
# single-node srun environment.
# TODO: check if we can take advantage of --export=None arg when execute srun instead
# of unset them in the script
if [ "${SLURM_JOB_NUM_NODES:-1}" -eq 1 ] || \
   [ "${DISAGG_SERVING_TYPE:-}" == "BENCHMARK" ] || \
   [ "${DISAGG_SERVING_TYPE:-}" == "DISAGG_SERVER" ]; then
    for v in ${!PMI@} ${!PMIX@} ${!MPI@} ${!OMPI@} ${!SLURM@}; do
        if [ "$v" != "SLURM_PROCID" ]; then
            unset "$v"
        fi
    done
fi

# Paths used for timeout classification.  All ranks share $jobWorkspace via
# NFS, so each rank can write there and uploadResults() can collect the
# completed records from the login node without any extra copy step.
#
# A pytest-timeout kill can happen on ANY rank in a multi-node distributed
# run, not just rank 0 -- so every rank captures its own log and classifies
# it independently. Each rank writes to its OWN per-rank JSONL
# (timeout_data_step${slurm_step_id}_rank${SLURM_PROCID}.jsonl) rather than
# the shared timeout_data.jsonl: multiple ranks appending concurrently to one
# file over NFS is not safe (a >PIPE_BUF snippet write is not guaranteed
# atomic), so concurrent writes are avoided entirely.
# After the Slurm job finishes, uploadResults() downloads every rank file and
# generate_timeout_xml.py merges them while it creates results-timeout.xml.
# The full pytest log is transient, so keep it on the worker node rather than
# streaming it to the shared workspace.
PYTEST_LOG="/tmp/pytest_output_job${SLURM_JOB_ID:-unknown}_step${slurm_step_id}_rank${SLURM_PROCID}.log"
TIMEOUT_DATA_RANK="${jobWorkspace}/timeout_data_step${slurm_step_id}_rank${SLURM_PROCID}.jsonl"
UNFINISHED_FILE="${jobWorkspace}/unfinished_test.txt"
CLASSIFY_SCRIPT="${llmSrcNode}/jenkins/scripts/classify_timeout.py"

# The normal post-processing path removes this node-local log explicitly.
# Keep an EXIT trap as well so normal early exits do not leave it behind.
cleanup_pytest_log() {
    rm -f "${PYTEST_LOG}" || true
}

cleanup_and_exit() {
    cleanup_pytest_log
    trap - EXIT
    exit "$1"
}

trap cleanup_pytest_log EXIT
trap 'cleanup_and_exit 129' HUP
trap 'cleanup_and_exit 130' INT
trap 'cleanup_and_exit 143' TERM

# Turn off "exit on error" so the following lines always run
set +e

pytest_exit_code=0
perf_check_exit_code=0
perf_report_exit_code=0

# Every rank captures its own stdout+stderr via tee for post-run timeout
# classification.
export TRTLLM_TIMEOUT_DATA_FILE="${TIMEOUT_DATA_RANK}"
if eval "$pytestCommand" 2>&1 | tee "${PYTEST_LOG}"; then
    pytest_exit_code=0
else
    pytest_exit_code=${PIPESTATUS[0]}
fi
echo "Rank${SLURM_PROCID} Pytest finished execution with exit code $pytest_exit_code"
python3 "$llmSrcNode/tests/test_common/s3_output.py" \
    --drain-spool "$jobWorkspace" || true

# Every rank scans its own captured log for pytest-timeout banners and
# writes records to its own per-rank JSONL. All steps are best-effort: a
# classify failure must not change pytest_exit_code.
python3 "${CLASSIFY_SCRIPT}" \
    --log        "${PYTEST_LOG}" \
    --out        "${TIMEOUT_DATA_RANK}" \
    --unfinished "${UNFINISHED_FILE}" || \
    echo "WARNING: slurm_run.sh: classify_timeout.py failed for rank" \
         "${SLURM_PROCID}; timed-out tests in this invocation may be" \
         "reported as 'unknown' instead of 'pytest_timeout'." >&2
cleanup_pytest_log

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
    # Only PyTorch perf stages remain; the TensorRT perf baseline was removed.
    basePerfFilename="base_perf_pytorch.csv"
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
