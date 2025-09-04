#!/bin/bash

if [[ -z "$SLURM_PARTITION" || \
     -z "$SLURM_ACCOUNT" || \
     -z "$SLURM_JOB_NAME" || \
     -z "$CONTAINER_IMAGE" || \
     -z "$CONTAINER_MOUNTS" || \
     -z "$WORK_DIR" || \
     -z "$MODEL_DIR" || \
     -z "$REPO_DIR" || \
     -z "$DATA_DIR" ]]; then
  echo "Environment not set, please source submit_mjoux_env.sh first"
  exit 1
fi

ctx_tp_size=8
ctx_pp_size=1
ctx_cp_size=1
ctx_chunked_prefill=false
gen_tp_size=1
gen_pp_size=1
gen_cp_size=4
gen_ep_size=2
batch=1

partition=$SLURM_PARTITION
account=$SLURM_ACCOUNT
job_name=${SLURM_JOB_NAME}_ctxtp${ctx_tp_size}cp${ctx_cp_size}$(if [ "${ctx_chunked_prefill}" = "true" ]; then echo "chunked"; fi)_gentp${gen_tp_size}cp${gen_cp_size}ep${gen_ep_size}bs${batch}
container_image=$CONTAINER_IMAGE
mounts=$CONTAINER_MOUNTS
workdir=$WORK_DIR
model_dir=$MODEL_DIR
repo_dir=$REPO_DIR
data_dir=$DATA_DIR

ntasks_per_node=4 # 4 GPUs per GB200 node

isl=1048576
osl=1024
concurrency=$((batch * 16))
multi_round=1
streaming=true
benchmark_mode=e2e
build_wheel=true
cuda_architectures="100a-real"
ctx_max_tokens=$((batch * (isl + 10)))
gen_max_tokens=$((batch * (1 + 10)))
tokens_per_block=32
transceiver_blocks=$(((ctx_max_tokens + tokens_per_block - 1) / tokens_per_block))
cache_transceiver_max_num_tokens=$((tokens_per_block * transceiver_blocks + 512))

ctx_nodes_num=$(((ctx_tp_size * ctx_pp_size * ctx_cp_size + ntasks_per_node - 1) / ntasks_per_node))
gen_nodes_num=$(((gen_tp_size * gen_pp_size * gen_cp_size + ntasks_per_node - 1) / ntasks_per_node))
total_node_num=$((gen_nodes_num))       # gen-only mode.
ntasks=$((total_node_num * ntasks_per_node))

export TRTLLM_DISAGG_BENCHMARK_GEN_ONLY=1

echo "Calling sbatch with TP $gen_tp_size, CP $gen_cp_size, EP $gen_ep_size, ISL $isl"

args=(
    # Context - [num_instances, tp_size, pp_size, cp_size, batch_size, max_num_tokens, enable_attention_dp, gpu_memory_fraction]
    1 $ctx_tp_size $ctx_pp_size $ctx_cp_size $batch $ctx_max_tokens false "0.85"
    # Generation - [num_instances, tp_size, pp_size, cp_size, batch_size, max_num_tokens, enable_attention_dp, gpu_memory_fraction]
    1 $gen_tp_size $gen_pp_size $gen_cp_size $batch $gen_max_tokens false "0.85"
    # Other arguments - [eplb_num_slots, mtp_size]
    0 0
    # Benchmarking arguments
    $concurrency
    $isl
    $osl
    $multi_round
    $streaming
    # User specific arguments
    $container_image
    $mounts
    $workdir
    $model_dir
    $benchmark_mode
    $repo_dir
    $build_wheel
    $cuda_architectures
    $data_dir
    $cache_transceiver_max_num_tokens
    $gen_ep_size
)

# This command starts a job with 8 nodes, 32 GPUs in total.
# The server will include 4 context workers with DEP4, and 1 generation worker with DEP8.
# `--segment` makes sure that all nodes are in the same NVLink domain
sbatch --nodes=${total_node_num} \
    --ntasks=${ntasks} \
    --ntasks-per-node=${ntasks_per_node} \
    --partition=${partition} \
    --account=${account} \
    --job-name=${job_name} \
    --gres=gpu:${ntasks_per_node} \
    --segment=${total_node_num} \
    ${workdir}/disaggr_torch.slurm "${args[@]}"
