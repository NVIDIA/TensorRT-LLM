#!/bin/bash
#SBATCH --partition=${partition}
#SBATCH --account=${account}
#SBATCH --job-name=${job_name}
#SBATCH --time=02:00:00

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

container_image=""
mount_paths=""
work_path=""
ctx_port=8001
gen_port=8002

# Keep the generated configurations on the shared mount used by all three services.
set -eu
: "${work_path:?Set work_path to the shared example directory}"
config_dir=$(mktemp -d "${work_path%/}/.disagg-config.XXXXXXXX")
trap 'rm -f -- "$config_dir/ctx_config.yaml" "$config_dir/gen_config.yaml" "$config_dir/disagg_config.yaml"; rmdir -- "$config_dir"' EXIT
python3 - "$work_path" "$config_dir" <<'CONFIG'
from pathlib import Path
import secrets
import sys

import yaml

source, destination = map(Path, sys.argv[1:])
configs = {}
for output, template in (("ctx_config.yaml", "ctx_extra-llm-api-config.yaml"),
                         ("gen_config.yaml", "gen_extra-llm-api-config.yaml"),
                         ("disagg_config.yaml", "disagg_config.yaml")):
    configured = source / output
    path = configured if configured.is_file() else source / template
    configs[output] = yaml.safe_load(path.read_text()) or {}
keys = {cfg["internal_request_auth_key"] for cfg in configs.values() if cfg.get("internal_request_auth_key")}
if len(keys) > 1:
    raise ValueError("The disaggregated services must use the same internal_request_auth_key")
key = next(iter(keys)) if keys else secrets.token_hex(32)
for output, cfg in configs.items():
    cfg["internal_request_auth_key"] = key
    (destination / output).write_text(yaml.safe_dump(cfg, sort_keys=False))
CONFIG

# The `container_image` must have the TensorRT-LLM wheel package pre-installed.
# Once the task is successfully launched, an API service will be available externally at http://host_ip:PORT.
# Launch a context with `tp_size=8` using two 4-GPU nodes.
srun --container-image=${container_image} \
     --container-mounts=${mount_paths} \
     -N 2 --ntasks-per-node=4 \
     --mpi=pmix \
     trtllm-llmapi-launch trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --tp_size 8 --host 0.0.0.0 --port ${ctx_port} --config "${config_dir}/ctx_config.yaml" &

# Launch a generation with `tp_size=4` using one 4-GPU node.
srun --container-image=${container_image} \
     --container-mounts=${mount_paths} \
     -N 1 --ntasks-per-node=4 \
     --mpi=pmix \
     trtllm-llmapi-launch trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --tp_size 4 --host 0.0.0.0 --port ${gen_port} --config "${config_dir}/gen_config.yaml" &

# Launch a proxy.
# The above-mentioned value needs to be replaced with the IP address of the host machine accessible to external
# clients, and filled in the `disagg_config.yaml` file.
srun --container-image=${container_image} \
     --container-mounts=${mount_paths} \
     -N 1 --ntasks-per-node=1 \
     --mpi=pmix \
     trtllm-llmapi-launch trtllm-serve disaggregated -c "${config_dir}/disagg_config.yaml"
