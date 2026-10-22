#!/bin/bash
#SBATCH --partition=${partition}
#SBATCH --account=${account}
#SBATCH --job-name=${job_name}
#SBATCH --time=02:00:00

container_image=""
mount_paths=""
work_path=""
ctx_port=8001
gen_port=8002

auth_config_dir=$(mktemp -d "${work_path}/.trtllm-disagg.XXXXXX")
original_umask=$(umask)
umask 077
python3 - "${work_path}" "${auth_config_dir}" <<'PY'
import secrets
import sys
from pathlib import Path

import yaml

source_dir, output_dir = map(Path, sys.argv[1:])
names = ("ctx_config.yaml", "gen_config.yaml", "disagg_config.yaml")
configs = {}
configured_keys = set()
for name in names:
    config = yaml.safe_load((source_dir / name).read_text()) or {}
    if not isinstance(config, dict):
        raise ValueError(f"{name} must contain a YAML mapping")
    configs[name] = config
    key = config.get("internal_request_auth_key")
    if key is not None:
        if not isinstance(key, str) or not key:
            raise ValueError("internal_request_auth_key must be a non-empty string")
        configured_keys.add(key)
if len(configured_keys) > 1:
    raise ValueError("internal_request_auth_key must match across configurations")
key = next(iter(configured_keys), None) or secrets.token_hex(32)
for name, config in configs.items():
    config["internal_request_auth_key"] = key
    (output_dir / name).write_text(yaml.safe_dump(config, sort_keys=False))
PY
umask "${original_umask}"
trap 'rm -rf "${auth_config_dir}"' EXIT

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
