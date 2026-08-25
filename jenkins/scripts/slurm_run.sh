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

# Aggregated mode and infrastructure dry runs install in slurm_run.sh.
# Disaggregated mode will run install separately in slurm_install.sh
if [[ "${infraDryRun:-false}" == "true" || "$stageName" != *Disagg* ]]; then
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

# The install lock in slurm_install.sh lives under $resourcePathNode (/tmp), so it
# is node-local: its wait loop only fences $SLURM_LOCALID peers on the same node,
# and a node can never observe another node's lock. Nothing else stops one node
# from reaching `eval $pytestCommand` below while another is still installing, and
# the per-rank work above skews the ranks further (non-zero ranks cover the
# coverage-config write with a blind `sleep 30`, and slurm_setup_runtime_env shells
# out to pip3). Pytest's first action is `import tensorrt_llm`, whose module-scope
# MPI collective must be entered by every rank; under --mpi=pmix -- added exactly
# when nodeCount > 1 -- that collective has a 300s fence timeout, so the skew
# aborts every rank instead of merely running late. Fence every rank on the shared
# $jobWorkspace so they enter pytest together.
slurm_wait_all_ranks() {
    local numRanks="${SLURM_NTASKS:-1}"
    if [ "$numRanks" -le 1 ] || [ -z "${jobWorkspace:-}" ]; then
        return 0
    fi

    # Keyed per job *and* per step: $jobWorkspace outlives a single step, so
    # markers from another job, or from an earlier step of this job, must not
    # satisfy the count. Slurm assigns one step id per step across all of its
    # nodes, so every rank of a step agrees on this path.
    local readyDir="$jobWorkspace/run_ready_job_${SLURM_JOB_ID:-local}_step_${SLURM_STEP_ID:-0}"
    mkdir -p "$readyDir"
    touch "$readyDir/rank_${SLURM_PROCID}.ready"

    # Bounded so a dead rank fails the stage loudly instead of hanging until the
    # partition walltime kills it; the ceiling exceeds the 2700s pip3 retry budget
    # in slurm_install.sh so a merely slow rank still releases the barrier.
    local timeoutSecs=3600
    local deadline=$((SECONDS + timeoutSecs))
    local markers ready
    while true; do
        # Counted with a glob rather than `ls | wc -l`: under `set -Eeuo pipefail` a
        # failing `ls` propagates into the assignment and fires the ERR trap. The
        # touch above guarantees at least one match, so no nullglob is needed.
        markers=("$readyDir"/*.ready)
        ready=${#markers[@]}
        if [ "$ready" -ge "$numRanks" ]; then
            return 0
        fi
        if [ "$SECONDS" -ge "$deadline" ]; then
            echo "ERROR: rank ${SLURM_PROCID} timed out after ${timeoutSecs}s waiting for" \
                 "all $numRanks ranks to be ready; ready: $ready/$numRanks"
            return 1
        fi
        # One rank reports progress; all of them would spam the log every 10s.
        if [ "$SLURM_PROCID" -eq 0 ]; then
            echo "(Waiting for all $numRanks ranks to be ready) ready: $ready/$numRanks"
        fi
        sleep 10
    done
}

slurm_wait_all_ranks

# Turn off "exit on error" so the following lines always run
set +e

pytest_exit_code=0
perf_check_exit_code=0
perf_report_exit_code=0

eval $pytestCommand
pytest_exit_code=$?
echo "Rank${SLURM_PROCID} Pytest finished execution with exit code $pytest_exit_code"
python3 "$llmSrcNode/tests/test_common/s3_output.py" \
    --drain-spool "$jobWorkspace" || true

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
