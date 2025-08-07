#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --partition=${partition} # add your partition here or specify in the sbatch command
#SBATCH --account=${account} # add your account here or specify in the sbatch command
#SBATCH --job-name=${job_name} # add your job name here or specify in the sbatch command
#SBATCH --time=02:00:00

# Context servers arguments
num_ctx_servers=${1}
ctx_tp_size=${2}
ctx_batch_size=${3}
ctx_max_num_tokens=${4}
ctx_enable_attention_dp=${5}

# Generation servers arguments
num_gen_servers=${6}
gen_tp_size=${7}
gen_batch_size=${8}
gen_max_num_tokens=${9}
gen_enable_attention_dp=${10}
gen_gpu_memory_fraction=${11}

# Other arguments
eplb_num_slots=${12}
mtp_size=${13}

# Benchmarking arguments
concurrency=${14}
isl=${15}
osl=${16}
multi_round=${17}
streaming=${18}

# User specific arguments
container_image=${19}
mounts=${20}
workdir=${21}
model_dir=${22}
repo_dir=${23}

ctx_max_seq_len=$((isl + 1))
gen_max_seq_len=$((isl + osl))
ctx_gpu_frac=0.75
cache_transceiver_max_num_tokens=8448

container_name=disaggr
logdir=${workdir}/benchmark-${isl}-${osl}/
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

nsys_on=""
# nsys_on=${full_logdir} # Uncomment this line to enable Nsys profiling

# start the container
srun -l --container-image=${container_image} \
        --container-name=${container_name} \
        --container-mounts=${mounts} \
        --mpi=pmix \
        echo "Container up."

# generate the yaml file
srun -l --container-name=${container_name} \
        --container-mounts=${mounts} \
        --mpi=pmix --overlap \
        python3 ${workdir}/gen_yaml.py --config ${full_logdir}/config.yaml \
            --model ${model_dir} \
            --num_ctx_servers ${num_ctx_servers} \
            --ctx_tp_size ${ctx_tp_size} \
            --ctx_batch_size ${ctx_batch_size} \
            --ctx_max_num_tokens ${ctx_max_num_tokens} \
            --ctx_max_seq_len ${ctx_max_seq_len} \
            --ctx_free_gpu_memory_fraction ${ctx_gpu_frac} \
            --cache_transceiver_max_num_tokens ${cache_transceiver_max_num_tokens} \
            --num_gen_servers ${num_gen_servers} \
            --gen_tp_size ${gen_tp_size} \
            --gen_batch_size ${gen_batch_size} \
            --gen_max_num_tokens ${gen_max_num_tokens} \
            --gen_max_seq_len ${gen_max_seq_len} \
            --gen_gpu_memory_fraction ${gen_gpu_memory_fraction} \
            --eplb_num_slots ${eplb_num_slots} \
            $(if [ "${gen_enable_attention_dp}" = "true" ]; then echo "--gen_enable_attention_dp"; fi) \
            $(if [ "${ctx_enable_attention_dp}" = "true" ]; then echo "--ctx_enable_attention_dp"; fi) \
            $(if [ "${mtp_size}" -gt 0 ]; then echo "--mtp_size ${mtp_size}"; fi)

echo "YAML file generated."

hostname_value=$(grep '^hostname:' ${full_logdir}/config.yaml | awk -F': ' '{print $2}')
echo "server host name: $hostname_value"

# start the workers
srun -l --container-name=${container_name} \
        --container-mounts=${mounts} \
    --mpi=pmix --overlap \
    bash ${workdir}/start_worker.sh ${full_logdir}/config.yaml "${enable_pdl}" ${ctx_gpus} ${nsys_on} ${repo_dir} &> ${full_logdir}/output_workers.log &

# start the server
srun -l --container-name=${container_name} \
        --container-mounts=${mounts} \
        --mpi=pmix --overlap -N 1 -n 1 \
        -w ${hostname_value} \
        bash ${workdir}/start_server.sh ${full_logdir}/config.yaml ${repo_dir} &> ${full_logdir}/output_server.log &

# start benchmarking
srun -l --container-name=${container_name} \
        --container-mounts=${mounts} \
        --mpi=pmix --overlap -N 1 -n 1 \
        bash ${workdir}/run_benchmark.sh ${isl} ${osl} ${multi_round} ${model_dir} "${concurrency}" ${streaming} ${full_logdir} > ${full_logdir}/benchmark.log 2>&1

# try to kill the server and workers
srun -l --container-name=${container_name} \
        --container-mounts=${mounts} \
        --mpi=pmix --overlap \
        kill -9 $(ps aux | grep '[t]rtllm-serve' | awk '{print $2}') >/dev/null 2>&1 || true
wait
