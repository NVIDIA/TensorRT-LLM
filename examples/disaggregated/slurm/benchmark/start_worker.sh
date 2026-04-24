#! /bin/bash
set -u
set -e
set -x

role=${1}
instance_id=${2}
model_path=${3}
port=${4}
numa_bind=${5}
log_dir=${6}
enable_nsys=${7}
config_file=${8}
server_role=${9}
disagg_cluster_uri=${10}

# CUDA_VISIBLE_DEVICES selection:
#   - Default packing (no gpu_map file): each node is dedicated to one
#     worker, so SLURM_LOCALID maps directly to the physical GPU id.
#   - Compact packing (gpu_map file emitted by submit.py): two workers may
#     share a node and would both see LOCALID=0, so look up the per-worker
#     gpu_map "<rank> <host> <local_gpu_id>" by SLURM_PROCID. srun
#     --distribution=arbitrary assigns PROCID in hostfile order, so it
#     indexes directly into the map.
gpu_map_file="${log_dir}/gpu_map_${role}_${instance_id}.txt"
if [ -f "${gpu_map_file}" ]; then
    gpu_id=$(awk -v p="${SLURM_PROCID}" '$1==p {print $3; exit}' "${gpu_map_file}")
    if [ -z "${gpu_id}" ]; then
        echo "ERROR: no GPU mapping for SLURM_PROCID=${SLURM_PROCID} in ${gpu_map_file}" >&2
        exit 1
    fi
    export CUDA_VISIBLE_DEVICES=${gpu_id}
else
    export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID}
fi

# Clear UCX_TLS for specific clusters
unset UCX_TLS

# Resolve KVBM leader ZMQ hostname to IPv4 so ZMQ can bind (leader) and
# connect (workers) using the same address across nodes.
if [ -n "${DYN_KVBM_LEADER_ZMQ_HOST:-}" ]; then
    resolved_ip=$(getent ahostsv4 "${DYN_KVBM_LEADER_ZMQ_HOST}" | awk '{print $1}' | head -1)
    if [ -n "${resolved_ip}" ]; then
        export DYN_KVBM_LEADER_ZMQ_HOST="${resolved_ip}"
        echo "Resolved DYN_KVBM_LEADER_ZMQ_HOST to ${resolved_ip}"
    fi
fi

echo "SLURM_PROCID: ${SLURM_PROCID}, hostname: $(hostname), instance_id: ${instance_id}"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES}"

if [ "${numa_bind}" = "true" ]; then
    numa_bind_cmd="numactl -m 0,1"
    echo "numactl -m 0,1 - Only allocate memory from nodes on GB200/GB300 NVL72"
else
    numa_bind_cmd=""
    echo "Not binding memory. If on GB200/GB300 NVL72, use \"numactl -m 0,1\" to only allocate memory from nodes."
fi

echo "config_file: ${config_file}"

nsys_prefix=""
if [ "${enable_nsys}" != "true" ]; then
    echo "nsys is not enabled, start normal flow"
else
    nsys_file=${log_dir}/nsys_worker_proc_${role}_${instance_id}_${SLURM_PROCID}
    echo "nsys is enabled on ${role} GPUs, TLLM_PROFILE_START_STOP=${TLLM_PROFILE_START_STOP}"
    nsys_prefix="nsys profile -o ${nsys_file} -f true -t cuda,nvtx,python-gil -c cudaProfilerApi --cuda-graph-trace node --capture-range-end=stop --gpu-metrics-devices=none"
fi

${nsys_prefix} trtllm-llmapi-launch ${numa_bind_cmd} \
    trtllm-serve ${model_path} \
        --host $(hostname) --port ${port} \
        --config ${config_file} \
        --server_role ${server_role} \
        --disagg_cluster_uri ${disagg_cluster_uri}
