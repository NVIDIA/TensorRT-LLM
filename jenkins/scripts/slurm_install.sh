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

slurm_install_setup() {
    : "${tarName:?tarName is required}"
    : "${llmTarfile:?llmTarfile is required}"
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
        # Download the artifact idempotently with retry. A bare "retry_command wget <url>" will
        # save artifact as $tarName.1 when the first attempt fails in the middle of downloading.
        # Here we download to the tmp path and only promote it on success
        if ! retry_command --timeout 1800 bash -c 'wget -nv "$1" -O "$2" && mv -f "$2" "$3"' _ "$llmTarfile" "$archive_tmp" "$archive_path"; then
            rm -f "$archive_tmp"
            echo "Artifact download failed after retries: $llmTarfile"
            return 1
        fi
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
            # TODO(dlfw-26.08): reinstate causal-conv1d and mamba-ssm once upstream
            # publishes wheels built against this base image's torch. They used to be
            # installed here, from
            #   causal-conv1d v1.6.2  causal_conv1d-1.6.1+cu13torch26.04cxx11abiTRUE
            #   mamba v2.3.0          mamba_ssm-2.3.0+cu13torch26.01cxx11abiTRUE
            # but the newest builds upstream offers target torch 26.07 and 26.04, so
            # on DLFW 26.08 the extension loads with an undefined c10 symbol,
            # materialize_cow_storage(StorageImpl&). A broken install is worse than
            # none: transformers gates its causal_conv1d import on a package-metadata
            # probe, which a broken install still passes, so modeling_qwen3_5_moe
            # raises at import and every test collected from a module that imports it
            # dies as a collection error. Absent, the gate says no and the model falls
            # back to its Python path -- slower, and it OOMs on Nemotron-H, which is
            # why test_llm_update_weights_nemotron_h is waived under nvbugs/6729495.
        fi
        retry_command --timeout 2700 bash -c "cd $llmSrcNode && pip3 install --retries 10 -r requirements-dev.txt"
        retry_command --timeout 2700 bash -c "cd $llmSrcNode && pip3 install --retries 10 -r requirements-grpc-smg.txt"
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
