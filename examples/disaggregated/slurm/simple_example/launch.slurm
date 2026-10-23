#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#SBATCH --partition=${partition}
#SBATCH --account=${account}
#SBATCH --job-name=${job_name}
#SBATCH --time=02:00:00

container_image=""
mount_paths=""
work_path=""
ctx_port=8001
gen_port=8002

# Generated configs must be on a shared path mounted at the same location on each node.
: "${work_path:?Set work_path to the shared configuration directory}"
auth_config_dir=$(mktemp -d "${work_path}/.trtllm-disagg.XXXXXX") || exit 1
trap 'rm -rf "${auth_config_dir}"' EXIT
python3 - "${work_path}" "${auth_config_dir}" <<'PYCONFIG' || exit 1
import os
import secrets
import sys
from pathlib import Path
import yaml

source, destination = map(Path, sys.argv[1:])
configs = {}
for name in ("ctx", "gen", "disagg"):
    with (source / f"{name}_config.yaml").open() as stream:
        configs[name] = yaml.safe_load(stream) or {}
key = secrets.token_hex(32)
for name, config in configs.items():
    config["internal_request_auth_key"] = key
    fd = os.open(destination / f"{name}_config.yaml", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        yaml.safe_dump(config, stream)
PYCONFIG

# The `container_image` must have the TensorRT-LLM wheel package pre-installed.
# Once the task is successfully launched, an API service will be available externally at http://host_ip:PORT.
# Launch a context with `tp_size=8` using two 4-GPU nodes.
srun --container-image=${container_image} \
     --container-mounts=${mount_paths} \
     -N 2 --ntasks-per-node=4 \
     --mpi=pmix \
     bash -c "trtllm-llmapi-launch trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --tp_size 8 --host 0.0.0.0 --port ${ctx_port} --config ${auth_config_dir}/ctx_config.yaml" &

# Launch a generation with `tp_size=4` using one 4-GPU node.
srun --container-image=${container_image} \
     --container-mounts=${mount_paths} \
     -N 1 --ntasks-per-node=4 \
     --mpi=pmix \
     bash -c "trtllm-llmapi-launch trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --tp_size 4 --host 0.0.0.0 --port ${gen_port} --config ${auth_config_dir}/gen_config.yaml" &

# Launch a proxy.
# The above-mentioned value needs to be replaced with the IP address of the host machine accessible to external
# clients, and filled in the `disagg_config.yaml` file.
srun --container-image=${container_image} \
     --container-mounts=${mount_paths} \
     -N 1 --ntasks-per-node=1 \
     --mpi=pmix \
     bash -c "trtllm-llmapi-launch trtllm-serve disaggregated -c ${auth_config_dir}/disagg_config.yaml"
