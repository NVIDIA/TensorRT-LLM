#!/bin/bash
set -u
set -e
set -x

server_cmd=${1}
log_dir=${2}
numa_bind=${3}
enable_nsys=${4:-false}

unset UCX_TLS

echo "SLURM_PROCID: ${SLURM_PROCID}, hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"

if [ "${numa_bind}" = "true" ]; then
    numa_bind_cmd="numactl -m 0,1"
    echo "numactl -m 0,1 - Only allocate memory from nodes on GB200/GB300 NVL72"
else
    numa_bind_cmd=""
fi

nsys_prefix=""
if [ "${enable_nsys}" != "true" ]; then
    echo "nsys is not enabled, start normal flow"
else
    nsys_file=${log_dir}/nsys_worker_proc_${SLURM_PROCID}
    echo "nsys is enabled, TLLM_PROFILE_START_STOP=${TLLM_PROFILE_START_STOP:-not set}"
    nsys_prefix="nsys profile -o ${nsys_file} -f true -t cuda,nvtx,python-gil -c cudaProfilerApi --cuda-graph-trace node --capture-range-end=stop --gpu-metrics-devices=none"
fi

echo "Rank${SLURM_PROCID} run ${server_cmd} in background"

${nsys_prefix} ${numa_bind_cmd} trtllm-llmapi-launch ${server_cmd}
