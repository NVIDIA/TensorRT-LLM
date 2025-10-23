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

# The `container_image` must have the TensorRT-LLM wheel package pre-installed.
# Once the task is successfully launched, an API service will be available externally at http://host_ip:PORT.
# Launch a context with `tp_size=8` using two 4-GPU nodes.
srun --container-image=${container_image} \
     --container-mounts=${mount_paths} \
     -N 2 --ntasks-per-node=4 \
     --mpi=pmix \
     bash -c "trtllm-llmapi-launch trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --tp_size 8 --host 0.0.0.0 --port ${ctx_port} --extra_llm_api_options ${work_path}/ctx_extra-llm-api-config.yaml" &

# Launch a generation with `tp_size=4` using one 4-GPU node.
srun --container-image=${container_image} \
     --container-mounts=${mount_paths} \
     -N 1 --ntasks-per-node=4 \
     --mpi=pmix \
     bash -c "trtllm-llmapi-launch trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --tp_size 4 --host 0.0.0.0 --port ${gen_port} --extra_llm_api_options ${work_path}/gen_extra-llm-api-config.yaml" &

# Launch a proxy.
# The above-mentioned value needs to be replaced with the IP address of the host machine accessible to external
# clients, and filled in the `disagg_config.yaml` file.
srun --container-image=${container_image} \
     --container-mounts=${mount_paths} \
     -N 1 --ntasks-per-node=1 \
     --mpi=pmix \
     bash -c "trtllm-llmapi-launch trtllm-serve disaggregated -c ${work_path}/disagg_config.yaml"
