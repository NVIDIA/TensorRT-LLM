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

# Copy the shared serving key into generated worker configs.
launch_config_dir=$(mktemp -d "${work_path}/.disagg-config.XXXXXX") || exit 1
trap 'rm -rf "$launch_config_dir"' EXIT
python3 - "$work_path" "$launch_config_dir" <<'PYTHON' || exit 1
import pathlib, sys, yaml
source, output = map(pathlib.Path, sys.argv[1:])
key = (yaml.safe_load((source / "disagg_config.yaml").read_text()) or {}).get("internal_request_auth_key")
for role in ("ctx", "gen"):
    config = yaml.safe_load((source / f"{role}_config.yaml").read_text()) or {}
    if key is not None:
        config["internal_request_auth_key"] = key
    (output / f"{role}_config.yaml").write_text(yaml.safe_dump(config))
PYTHON

# The `container_image` must have the TensorRT-LLM wheel package pre-installed.
# Once the task is successfully launched, an API service will be available externally at http://host_ip:PORT.
# Launch a context with `tp_size=8` using two 4-GPU nodes.
srun --container-image=${container_image} \
     --container-mounts=${mount_paths} \
     -N 2 --ntasks-per-node=4 \
     --mpi=pmix \
     bash -c "trtllm-llmapi-launch trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --tp_size 8 --host 0.0.0.0 --port ${ctx_port} --config ${launch_config_dir}/ctx_config.yaml" &

# Launch a generation with `tp_size=4` using one 4-GPU node.
srun --container-image=${container_image} \
     --container-mounts=${mount_paths} \
     -N 1 --ntasks-per-node=4 \
     --mpi=pmix \
     bash -c "trtllm-llmapi-launch trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --tp_size 4 --host 0.0.0.0 --port ${gen_port} --config ${launch_config_dir}/gen_config.yaml" &

# Launch a proxy.
# The above-mentioned value needs to be replaced with the IP address of the host machine accessible to external
# clients, and filled in the `disagg_config.yaml` file.
srun --container-image=${container_image} \
     --container-mounts=${mount_paths} \
     -N 1 --ntasks-per-node=1 \
     --mpi=pmix \
     bash -c "trtllm-llmapi-launch trtllm-serve disaggregated -c ${work_path}/disagg_config.yaml"
