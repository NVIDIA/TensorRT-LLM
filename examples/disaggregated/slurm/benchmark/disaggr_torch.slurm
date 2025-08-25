#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --partition=${partition} # add your partition here or specify in the sbatch command
#SBATCH --account=${account} # add your account here or specify in the sbatch command
#SBATCH --job-name=${job_name} # add your job name here or specify in the sbatch command
#SBATCH --time=02:00:00

set -u
set -e
set -x

# Context servers arguments
num_ctx_servers=${1}
ctx_tp_size=${2}
ctx_batch_size=${3}
ctx_max_num_tokens=${4}
ctx_enable_attention_dp=${5}
ctx_gpu_memory_fraction=${6}

# Generation servers arguments
num_gen_servers=${7}
gen_tp_size=${8}
gen_batch_size=${9}
gen_max_num_tokens=${10}
gen_enable_attention_dp=${11}
gen_gpu_memory_fraction=${12}

# Other arguments
eplb_num_slots=${13}
mtp_size=${14}

# Benchmarking arguments
concurrency=${15}
isl=${16}
osl=${17}
multi_round=${18}
streaming=${19}

# User specific arguments
container_image=${20}
mounts=${21}
workdir=${22}
model_dir=${23}
benchmark_mode=${24}
trtllm_repo=${25}

# Get GPUs per node dynamically from SLURM
ntasks_per_node=${SLURM_NTASKS_PER_NODE:-4}  # Default to 4 for GB200

echo "================= parameters ================="
echo "num_ctx_servers: ${num_ctx_servers}"
echo "ctx_tp_size: ${ctx_tp_size}"
echo "ctx_batch_size: ${ctx_batch_size}"
echo "ctx_max_num_tokens: ${ctx_max_num_tokens}"
echo "ctx_enable_attention_dp: ${ctx_enable_attention_dp}"
echo "ctx_gpu_memory_fraction: ${ctx_gpu_memory_fraction}"
echo "num_gen_servers: ${num_gen_servers}"
echo "gen_tp_size: ${gen_tp_size}"
echo "gen_batch_size: ${gen_batch_size}"
echo "gen_max_num_tokens: ${gen_max_num_tokens}"
echo "gen_enable_attention_dp: ${gen_enable_attention_dp}"
echo "gen_gpu_memory_fraction: ${gen_gpu_memory_fraction}"
echo "eplb_num_slots: ${eplb_num_slots}"
echo "mtp_size: ${mtp_size}"
echo "concurrency: ${concurrency}"
echo "isl: ${isl}"
echo "osl: ${osl}"
echo "multi_round: ${multi_round}"
echo "streaming: ${streaming}"
echo "container_image: ${container_image}"
echo "mounts: ${mounts}"
echo "workdir: ${workdir}"
echo "model_dir: ${model_dir}"
echo "benchmark_mode: ${benchmark_mode}"
echo "trtllm_repo: ${trtllm_repo}"
echo "ntasks_per_node: ${ntasks_per_node}"
echo "==========================================="


ctx_max_seq_len=$((isl + 1))
gen_max_seq_len=$((isl + osl))
ctx_gpu_frac=${ctx_gpu_memory_fraction}
cache_transceiver_max_num_tokens=8448

container_name=disaggregated_serving
logdir=${workdir}/slurm-${SLURM_JOB_ID}/benchmark-${isl}-${osl}
mkdir -p ${logdir}
full_logdir=${logdir}/ctx${num_ctx_servers}_gen${num_gen_servers}_dep${gen_tp_size}_batch${gen_batch_size}_eplb${eplb_num_slots}_mtp${mtp_size}

echo "concurrency: ${concurrency}"

ctx_gpus=$((num_ctx_servers * ctx_tp_size))
gen_gpus=$((num_gen_servers * gen_tp_size))

echo "enable_attention_dp: ${ctx_enable_attention_dp}, ${gen_enable_attention_dp}, gpu_memory_fraction: ${gen_gpu_memory_fraction}"

enable_pdl=false
if [ "${gen_enable_attention_dp}" = "false" ]; then
    enable_pdl=true
    echo "enable_pdl: ${enable_pdl}"
    full_logdir=${logdir}/ctx${num_ctx_servers}_gen${num_gen_servers}_tep${gen_tp_size}_batch${gen_batch_size}_eplb${eplb_num_slots}_mtp${mtp_size}
fi
mkdir -p ${full_logdir}
echo "Log will be saved to: ${full_logdir}"

# check benchmark_mode
if [ "${benchmark_mode}" != "gen_only" ] && [ "${benchmark_mode}" != "e2e" ]; then
    echo "benchmark_mode: ${benchmark_mode} is not supported, change to default value: e2e"
    benchmark_mode="e2e"
fi

if [ -z "${TRT_LLM_GIT_COMMIT:-}" ]; then
    export TRT_LLM_GIT_COMMIT=$(git -C ${trtllm_repo} rev-parse --short HEAD 2>/dev/null || echo "unknown")
    echo "TRT_LLM_GIT_COMMIT: ${TRT_LLM_GIT_COMMIT}"
fi

nsys_on=""
# nsys_on=${full_logdir} # Uncomment this line to enable Nsys profiling

# start the container
srun -l --container-image=${container_image} \
        --container-name=${container_name} \
        --container-mounts=${mounts} \
        --mpi=pmix \
        echo "Container up."

if [ -n "${trtllm_repo}" ]; then
    srun --container-name=${container_name} \
        --container-mounts=${mounts} \
        --mpi=pmix --overlap -N $SLURM_NNODES --ntasks-per-node=1 \
        bash -c "cd ${trtllm_repo} && echo 'Running install operation...' && pip install -e .  " 2>&1 | tee ${full_logdir}/install.log
fi

echo "Generating YAML file for workers."
srun -l -N 1 -n 1 \
        --container-name=${container_name} \
        --container-mounts=${mounts} \
        --mpi=pmix --overlap \
        python3 ${workdir}/gen_worker_config.py \
                --work_dir ${full_logdir} \
                --ctx_tp_size ${ctx_tp_size} \
                --ctx_batch_size ${ctx_batch_size} \
                --ctx_max_num_tokens ${ctx_max_num_tokens} \
                --ctx_max_seq_len ${ctx_max_seq_len} \
                --ctx_free_gpu_memory_fraction ${ctx_gpu_frac} \
                --gen_tp_size ${gen_tp_size} \
                --gen_batch_size ${gen_batch_size} \
                --gen_max_num_tokens ${gen_max_num_tokens} \
                --gen_max_seq_len ${gen_max_seq_len} \
                --gen_gpu_memory_fraction ${gen_gpu_memory_fraction} \
                --eplb_num_slots ${eplb_num_slots} \
                --mtp_size ${mtp_size} \
                --cache_transceiver_max_num_tokens ${cache_transceiver_max_num_tokens} \
                $(if [ "${ctx_enable_attention_dp}" = "true" ]; then echo "--ctx_enable_attention_dp"; fi) \
                $(if [ "${gen_enable_attention_dp}" = "true" ]; then echo "--gen_enable_attention_dp"; fi) \
                2>&1 | tee ${full_logdir}/gen_worker_config.log

echo "YAML file generated."

ctx_nodes_num=$(((ctx_tp_size + ntasks_per_node - 1) / ntasks_per_node))
gen_nodes_num=$(((gen_tp_size + ntasks_per_node - 1) / ntasks_per_node))

all_nodes=($(scontrol show hostname $SLURM_NODELIST | sort))
total_nodes_num=${#all_nodes[@]}
echo "all_nodes: ${all_nodes[@]}, total_nodes_num: ${total_nodes_num}"

# get the node list for the gen workers
total_gen_nodes_num=$((gen_nodes_num * num_gen_servers))
gen_nodes=(${all_nodes[@]:0:${total_gen_nodes_num}})
echo "gen_nodes: ${gen_nodes[@]}, total_gen_nodes_num: ${total_gen_nodes_num}"

# get the node list for the ctx workers
total_ctx_nodes_num=$((ctx_nodes_num * num_ctx_servers))
ctx_nodes=(${all_nodes[@]:${total_gen_nodes_num}:${total_nodes_num}})
echo "ctx_nodes: ${ctx_nodes[@]}, total_ctx_nodes_num: ${total_ctx_nodes_num}"

rm -rf ${full_logdir}/hostnames

# start the gen workers
for i in $(seq 0 $((num_gen_servers - 1))); do
    srun -l -N ${gen_nodes_num} \
        --ntasks=${gen_tp_size} \
        --ntasks-per-node=${ntasks_per_node} \
        --container-image=${container_image} \
        --container-name=${container_name} \
        --container-mounts=${mounts} \
        --mpi=pmix \
        bash ${workdir}/start_worker.sh "GEN" ${i} ${model_dir} "8336" ${benchmark_mode} ${concurrency} ${enable_pdl} ${full_logdir} ${nsys_on} \
        &> ${full_logdir}/output_gen_${i}.log &
done

# start the ctx workers
for i in $(seq 0 $((num_ctx_servers - 1))); do
    srun -l -N ${ctx_nodes_num} \
        --ntasks=${ctx_tp_size} \
        --ntasks-per-node=${ntasks_per_node} \
        --container-image=${container_image} \
        --container-name=${container_name} \
        --container-mounts=${mounts} \
        --mpi=pmix \
        bash ${workdir}/start_worker.sh "CTX" ${i} ${model_dir} "8336" ${benchmark_mode} ${concurrency} ${enable_pdl} ${full_logdir} ${nsys_on} \
            &> ${full_logdir}/output_ctx_${i}.log &
done

# start the server
srun -l --container-name=${container_name} \
    --container-image=${container_image} \
    --container-mounts=${mounts} \
    --mpi=pmix --overlap -N 1 -n 1 \
    bash ${workdir}/start_server.sh ${num_ctx_servers} ${num_gen_servers} ${full_logdir} ${workdir} \
        &> ${full_logdir}/output_server.log &

# start benchmarking
srun -l --container-name=${container_name} \
    --container-mounts=${mounts} \
    --mpi=pmix --overlap -N 1 -n 1 \
    bash ${workdir}/run_benchmark.sh ${isl} ${osl} ${multi_round} ${model_dir} "${concurrency}" ${streaming} ${full_logdir} \
        &> ${full_logdir}/benchmark.log 2>&1

scancel ${SLURM_JOB_ID}
