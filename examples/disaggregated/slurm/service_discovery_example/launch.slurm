bin/bash
#SBATCH --partition=${partition}
#SBATCH --account=${account}
#SBATCH --job-name=${job_name}
#SBATCH --time=02:00:00

container_image="${container_image:-}"
mount_paths="${mount_paths:-}"
work_path="${work_path:-}"
enable_etcd="${enable_etcd:-0}"
disagg_port="8000"
ctx_port="8001"
gen_port="8002"

# use the first node as the disaggregated server node
disagg_server_node=$(head -n 1 <(scontrol show hostnames $SLURM_JOB_NODELIST))

if [[ "$enable_etcd" == "1" ]]; then
     # you can optionally launch a etcd server, the container image must have etcd installed
     disagg_cluster_uri="etcd://${disagg_server_node}:2379"
     srun --container-image=${container_image} \
          --container-mounts=${mount_paths} \
          -w $disagg_server_node -N 1 --ntasks-per-node=1 \
          --mpi=pmix \
          bash -c "etcd" &
     sleep 5 # wait for etcd to start
else
     # or use the disaggregated server's http address as built-in service discovery server
     disagg_cluster_uri="http://${disagg_server_node}:${disagg_port}"
fi

cat >${work_path}/disagg_config.yaml << EOL
hostname: localhost
port: ${disagg_port}
backend: pytorch
disagg_cluster:
  cluster_uri: ${disagg_cluster_uri}
  cluster_name: example_cluster
EOL

cat >${work_path}/ctx_config.yaml << EOL
disable_overlap_scheduler: True
cache_transceiver_config:
  backend: UCX
  max_tokens_in_buffer: 2048
EOL

cat >${work_path}/gen_config.yaml << EOL
cache_transceiver_config:
  backend: UCX
  max_tokens_in_buffer: 2048
EOL

# Launch a proxy without any context/generation servers.
srun --container-image=${container_image} \
     --container-mounts=${mount_paths} \
     -w $disagg_server_node -N 1 --ntasks-per-node=1 \
     --mpi=pmix \
     bash -c "trtllm-llmapi-launch trtllm-serve disaggregated -c ${work_path}/disagg_config.yaml" &

# Launch a context with `tp_size=8` using two 4-GPU nodes, and register itself through disagg_cluster_uri
srun --container-image=${container_image} \
     --container-mounts=${mount_paths} \
     -N 2 --ntasks-per-node=4 \
     --mpi=pmix \
     bash -c "trtllm-llmapi-launch trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --tp_size 8 --host 0.0.0.0 --port ${ctx_port} --config ${work_path}/ctx_config.yaml --disagg_cluster_uri ${disagg_cluster_uri} --server-role context" &

# Launch a generation with `tp_size=4` using one 4-GPU node.
srun --container-image=${container_image} \
     --container-mounts=${mount_paths} \
     -N 1 --ntasks-per-node=4 \
     --mpi=pmix \
     bash -c "trtllm-llmapi-launch trtllm-serve TinyLlama/TinyLlama-1.1B-Chat-v1.0 --tp_size 4 --host 0.0.0.0 --port ${gen_port} --config ${work_path}/gen_config.yaml --disagg_cluster_uri ${disagg_cluster_uri} --server-role generation" &
