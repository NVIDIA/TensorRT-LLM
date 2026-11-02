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

# Source utilities
bashUtilsPath="$(dirname "${BASH_SOURCE[0]}")/$(basename "${BASH_SOURCE[0]}" | sed 's/slurm_install\.sh/bash_utils.sh/')"
source "$bashUtilsPath"

# The read helper (jenkins/scripts/fetch_build_artifact.sh) sits beside this script, under the same
# per-job prefix. LLM_ARTIFACTS_PRIMARY, ARTIFACTORY_NETRC and ARTIFACTORY_CREDENTIAL_ID reach it
# from the job script. fetchHelperPath is used after slurm_install_setup changes into
# $resourcePathNode, so run or source this script by an absolute path.
fetchHelperPath="$(dirname "${BASH_SOURCE[0]}")/$(basename "${BASH_SOURCE[0]}" | sed 's/slurm_install\.sh/fetch_build_artifact.sh/')"
# The read helper's exit status for a transfer that still failed after its own retries, the one
# failure of the helper that the download tries again. Its other failures, such as an absent package
# or a rejected credential, are final.
FETCH_TRANSFER_FAILED=4

# ARTIFACTORY_NETRC holds the token of the credential that publishes build artifacts to
# artifactory.nvidia.com and pushes CI's container images, and the package download is its only
# reader, so it is deleted once every node of this step has finished its download, before any rank
# starts the tests; on one node, before the package is extracted. The nodes of a multi-node step
# share the one file in the job workspace: each marks its download finished beside it and goes on to
# install the package, and the node that finds every node of this run of the step finished deletes
# it. No later step of the job downloads the package again. A run that Slurm requeues after the
# deletion has no netrc, so its download fails whenever it needs artifactory.nvidia.com.
remove_artifactory_netrc() {
    local netrc="${ARTIFACTORY_NETRC:-}"
    if [ -z "$netrc" ]; then
        return 0
    fi
    local numNodes="${SLURM_STEP_NUM_NODES:-1}"
    # With an unexpected count, the job cleanup removes the netrc instead.
    if ! [[ "$numNodes" =~ ^[1-9][0-9]*$ ]]; then
        return 0
    fi
    if [ "$numNodes" -gt 1 ]; then
        local doneDir
        doneDir="$(dirname "$netrc")/netrc_done_job_${SLURM_JOB_ID:-local}"
        doneDir+="_restart_${SLURM_RESTART_COUNT:-0}_step_${SLURM_STEP_ID:-0}"
        mkdir -p "$doneDir"
        touch "$doneDir/node_${SLURM_NODEID:-0}"
        local markers=("$doneDir"/node_*)
        if [ "${#markers[@]}" -lt "$numNodes" ]; then
            return 0
        fi
    fi
    rm -f "$netrc"
}

slurm_install_setup() {
    : "${tarName:?tarName is required}"
    : "${llmTarArtifactPath:?llmTarArtifactPath is required}"
    : "${resourcePathNode:?resourcePathNode is required}"
    : "${stageName:?stageName is required}"
    : "${pytestCommand:?pytestCommand is required}"
    cd $resourcePathNode
    llmSrcNode=$resourcePathNode/TensorRT-LLM/src

    # Use unique lock file for this job ID
    lock_file="install_lock_job_${SLURM_JOB_ID:-local}_node_${SLURM_NODEID:-0}.lock"

    if [ $SLURM_LOCALID -eq 0 ]; then
        # Authenticate github.com traffic. GITHUB_CLONE_TOKEN is exported by the sbatch launch script
        set +x
        if [ -n "${GITHUB_CLONE_TOKEN:-}" ]; then
            git config --global --replace-all \
                url."https://x-access-token:${GITHUB_CLONE_TOKEN}@github.com/".insteadOf \
                "https://github.com/"
            echo "Configured authenticated github.com access via git insteadOf."
        fi
        set -x

        if [ -f "$lock_file" ]; then
            rm -f "$lock_file"
        fi

        archive_path="$resourcePathNode/$tarName"
        # Job/node-specific tmp path to avoid collisions on concurrent jobs
        archive_tmp="${archive_path}.tmp.${SLURM_JOB_ID:-local}.${SLURM_NODEID:-0}"
        rm -f "$archive_path" "$archive_tmp"
        # Download the artifact idempotently with retry: each attempt writes to the tmp path, and
        # only a successful one promotes it. A failed transfer, or an attempt that timed out (124),
        # is tried again; any other failure is final.
        if ! retry_command --timeout 1800 --retry-on "$FETCH_TRANSFER_FAILED,124" bash -c 'bash "$1" "$2" "$3" && mv -f "$3" "$4"' _ "$fetchHelperPath" "$llmTarArtifactPath" "$archive_tmp" "$archive_path"; then
            rm -f "$archive_tmp"
            remove_artifactory_netrc
            echo "Artifact download failed: $llmTarArtifactPath"
            return 1
        fi
        remove_artifactory_netrc
        if [ ! -f "$archive_path" ]; then
            rm -f "$archive_tmp"
            echo "Artifact download did not produce $archive_path"
            return 1
        fi
        tar -zxf "$archive_path"

        which python3
        python3 --version
        retry_command apt-get install -y libffi-dev
        nvidia-smi && nvidia-smi -q && nvidia-smi topo -m
        if [[ $pytestCommand == *--run-ray* ]]; then
            retry_command --timeout 2700 pip3 install --retries 10 "ray[default]==2.55.1"
        fi
        retry_command --timeout 2700 bash -c "cd $llmSrcNode && pip3 install --retries 10 -r requirements-dev.txt"
        retry_command --timeout 2700 bash -c "cd $llmSrcNode && pip3 install --retries 10 -r requirements-grpc-smg.txt"
        if [[ $pytestCommand == *--run-ray* ]]; then
            retry_command --timeout 2700 bash "$llmSrcNode/jenkins/scripts/install_mamba.sh"
        fi
        retry_command --timeout 2700 bash -c "cd $resourcePathNode && pip3 install --retries 10 --force-reinstall --no-deps TensorRT-LLM/tensorrt_llm-*.whl"
        gpuUuids=$(nvidia-smi -q | grep "GPU UUID" | awk '{print $4}' | tr '\n' ',' || true)
        hostNodeName="${HOST_NODE_NAME:-$(hostname -f || hostname)}"
        echo "HOST_NODE_NAME = $hostNodeName ; GPU_UUIDS = $gpuUuids ; STAGE_NAME = $stageName"
        echo "(Writing install lock) Current directory: $(pwd)"
        touch "$lock_file"
    else
        echo "(Waiting for install lock) Current directory: $(pwd)"
        while [ ! -f "$lock_file" ]; do
            sleep 10
        done
    fi
}

# Only run slurm_install_setup when script is executed directly (not sourced)
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    slurm_install_setup
fi
