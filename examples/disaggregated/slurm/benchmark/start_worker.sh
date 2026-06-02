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

# Do not set CUDA_VISIBLE_DEVICES here. Let Slurm/enroot expose the devices
# assigned to this step; TRT-LLM relies on that visibility for peer checks.

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
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"

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
