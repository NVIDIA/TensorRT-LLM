#!/bin/bash
set -u
set -e
set -x

server_cmd=${1}
log_dir=${2}
numa_bind=${3}

unset UCX_TLS

echo "SLURM_PROCID: ${SLURM_PROCID}, hostname: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"

if [ "${numa_bind}" = "true" ]; then
    numa_bind_cmd="numactl -m 0,1"
    echo "numactl -m 0,1 - Only allocate memory from nodes on GB200/GB300 NVL72"
else
    numa_bind_cmd=""
fi

echo "Rank${SLURM_PROCID} run ${server_cmd} in background"

${numa_bind_cmd} trtllm-llmapi-launch ${server_cmd}
