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

internal_request_auth_key=$(python3 -c 'import secrets; print(secrets.token_hex(32))')
auth_config_dir=$(mktemp -d "${work_path}/.trtllm-disagg.XXXXXX")
original_umask=$(umask)
umask 077
for config_name in ctx gen disagg; do
    {
        printf 'internal_request_auth_key: "%s"\n' "${internal_request_auth_key}"
        cat "${work_path}/${config_name}_config.yaml"
    } >"${auth_config_dir}/${config_name}_config.yaml"
done
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
