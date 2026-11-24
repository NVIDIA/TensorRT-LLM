#! /bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
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
cuda_devices=${9}
# Optional, defaults to false so that older command lines keep working.
numa_local=${10:-false}
# CUDA_VISIBLE_DEVICES selection:
#   - Default packing (no gpu_map file): each node is dedicated to one
#     worker, so every rank on the node is given the node's full GPU list
#     (passed as ${9} by submit.py) and binds to its own device via
#     mapping.local_rank (= rank % gpus_per_node). Exposing the whole node
#     is required for intra-node TP custom all-reduce (attention_dp=false /
#     TEP): its cudaDeviceCanAccessPeer() topology check must be able to see
#     the peer GPUs. Pinning a single GPU per rank only works for DEP
#     (attention_dp=true), which has no intra-node TP all-reduce.
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
    export CUDA_VISIBLE_DEVICES=${cuda_devices}
fi

# Container runtimes (pyxis/enroot) reset image-defined variables like PATH
# at container start, so values passed via srun --export are lost for them.
# Allow the launcher config to prepend entries from inside the container.
# srun --export keeps any quotes in the exported values literal; strip them.
TRTLLM_PATH_PREPEND="${TRTLLM_PATH_PREPEND:-}"
TRTLLM_PATH_PREPEND="${TRTLLM_PATH_PREPEND#\'}"; TRTLLM_PATH_PREPEND="${TRTLLM_PATH_PREPEND%\'}"
TRTLLM_PYTHONPATH_PREPEND="${TRTLLM_PYTHONPATH_PREPEND:-}"
TRTLLM_PYTHONPATH_PREPEND="${TRTLLM_PYTHONPATH_PREPEND#\'}"; TRTLLM_PYTHONPATH_PREPEND="${TRTLLM_PYTHONPATH_PREPEND%\'}"
if [ -n "${TRTLLM_PATH_PREPEND:-}" ]; then
    export PATH="${TRTLLM_PATH_PREPEND}:${PATH}"
fi
if [ -n "${TRTLLM_PYTHONPATH_PREPEND:-}" ]; then
    export PYTHONPATH="${TRTLLM_PYTHONPATH_PREPEND}${PYTHONPATH:+:${PYTHONPATH}}"
fi

# Clear UCX_TLS for specific clusters. Some clusters instead need an
# explicit transport list (e.g. NVL72 nodes whose verbs transports cannot
# initialize): set TRTLLM_WORKER_UCX_TLS in worker_env_var to re-pin
# UCX_TLS here, after the container-provided value is cleared.
if [ -n "${TRTLLM_WORKER_UCX_TLS:-}" ]; then
    export UCX_TLS="${TRTLLM_WORKER_UCX_TLS}"
else
    unset UCX_TLS
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

# Per-rank NUMA locality (slurm.numa_local): start the worker on the CPUs and
# the memory of its GPU's NUMA node, before any Python runs, so that imports,
# the CUDA context and the malloc heap are first-touched on the GPU's socket.
# TensorRT-LLM binds CPUs only after it selects the device and sets no memory
# policy, so pages touched before that stay wherever the process started.
#   - GPU: the device TensorRT-LLM selects, global rank % visible GPUs, mapped
#     through CUDA_VISIBLE_DEVICES.
#   - Node: the numa_node of the GPU's PCI device in sysfs.
#   - Wrap: numactl --cpunodebind=<node> --membind=<node>, outermost, so the
#     launcher's helper processes (on rank 0 also the MPI leader and the
#     trtllm-serve frontend) inherit it.
#   - TLLM_NUMA_AWARE_WORKER_AFFINITY defaults to 1 here; when unset,
#     TensorRT-LLM clears a CPU mask it did not set.
#   - With numa_bind also on, --membind=<node> subsumes "numactl -m 0,1", so
#     only the numa_local wrap is applied.
#   - Fail-open: if the node cannot be resolved, log NUMA_LOCAL_SKIP and start
#     unbound (numa_bind still applies).
# One line per rank:
#   NUMA_LOCAL rank= gpu= bus= numa= cpus= mems= membind=
# cpus / mems are Cpus_allowed_list / Mems_allowed_list of a probe started
# under the same wrap; membind is its memory policy (numactl --show).
numa_local_skip() {
    echo "NUMA_LOCAL_SKIP rank=${SLURM_PROCID:-?} reason=$1"
}
numa_local_resolve() {
    local visible_gpus gpu bus pci node probe cpus mems membind
    if ! command -v numactl >/dev/null 2>&1; then
        numa_local_skip "no numactl"
        return 1
    fi
    if [ -z "${SLURM_PROCID:-}" ]; then
        numa_local_skip "no SLURM_PROCID"
        return 1
    fi
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        IFS=',' read -ra visible_gpus <<< "${CUDA_VISIBLE_DEVICES}"
    else
        mapfile -t visible_gpus < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ')
    fi
    if [ "${#visible_gpus[@]}" -eq 0 ]; then
        numa_local_skip "no visible GPUs"
        return 1
    fi
    gpu=${visible_gpus[$((SLURM_PROCID % ${#visible_gpus[@]}))]}
    bus=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "${gpu}" 2>/dev/null | tr -d ' ')
    if [[ ! ${bus} =~ ^([0-9A-Fa-f]+):([0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.[0-9A-Fa-f])$ ]]; then
        numa_local_skip "bad PCI bus id '${bus}' for GPU ${gpu}"
        return 1
    fi
    pci=$(printf '%04x' "0x${BASH_REMATCH[1]}"):${BASH_REMATCH[2],,}
    node=$(cat "/sys/bus/pci/devices/${pci}/numa_node" 2>/dev/null)
    if [[ ! ${node} =~ ^[0-9]+$ ]]; then
        numa_local_skip "no numa_node for ${pci} ('${node}')"
        return 1
    fi
    if ! probe=$(numactl --cpunodebind="${node}" --membind="${node}" \
            sh -c 'grep -E "^(Cpus|Mems)_allowed_list:" /proc/self/status; numactl --show' 2>&1); then
        numa_local_skip "numactl failed on node ${node}: $(echo "${probe}" | tr '\n' ' ')"
        return 1
    fi
    cpus=$(awk '/^Cpus_allowed_list:/ {print $2}' <<< "${probe}")
    mems=$(awk '/^Mems_allowed_list:/ {print $2}' <<< "${probe}")
    membind=$(awk '/^membind:/ {$1 = ""; sub(/^ /, ""); print}' <<< "${probe}")
    echo "NUMA_LOCAL rank=${SLURM_PROCID} gpu=${gpu} bus=${bus} numa=${node} cpus=${cpus} mems=${mems} membind=${membind}"
    numa_local_cmd="numactl --cpunodebind=${node} --membind=${node}"
}

numa_local_cmd=""
if [ "${numa_local}" = "true" ]; then
    if numa_local_resolve; then
        export TLLM_NUMA_AWARE_WORKER_AFFINITY="${TLLM_NUMA_AWARE_WORKER_AFFINITY:-1}"
        echo "TLLM_NUMA_AWARE_WORKER_AFFINITY: ${TLLM_NUMA_AWARE_WORKER_AFFINITY}"
        if [ -n "${numa_bind_cmd}" ]; then
            echo "numa_local: ${numa_local_cmd} replaces ${numa_bind_cmd}"
            numa_bind_cmd=""
        fi
    fi
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

# In-place (.pth-style) TRT-LLM installs may lack the trtllm-serve console
# script; fall back to the module entry point in that case.
trtllm_serve_cmd="trtllm-serve"
if ! command -v trtllm-serve >/dev/null 2>&1; then
    trtllm_serve_cmd="python3 -m tensorrt_llm.commands.serve"
fi

${numa_local_cmd} ${nsys_prefix} trtllm-llmapi-launch ${numa_bind_cmd} \
    ${trtllm_serve_cmd} ${model_path} \
        --host $(hostname) --port ${port} \
        --config ${config_file}
