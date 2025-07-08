#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --partition=${partition} # add your partition here
#SBATCH --account=${account} # add your account here
#SBATCH --time=01:00:00
#SBATCH --job-name=${job_name} # add your job name here

isl=1024
osl=1024
multi_round=1
gen_yaml_file=gen_yaml.py
container_image=${container_image} # add your container image here
mount_dir=${mount_dir} # add your mount directory here
workdir=${mount_dir}/bench-large-ep/slurm_scripts/
model_dir=${model_dir} # add your model directory here
logdir=${workdir}/bm_20250703_deepseek-r1-${isl}-${osl}/
streaming=false
mkdir -p ${logdir}

container_name=disaggr-test

num_ctx_servers=$1
ctx_tp_size=$2
ctx_batch_size=$3
ctx_max_num_tokens=$4
ctx_enable_attention_dp=$5
num_gen_servers=$6
gen_tp_size=$7
gen_batch_size=$8
gen_max_num_tokens=$9
gen_enable_attention_dp=${10}
gen_gpu_memory_fraction=${11}
eplb_num_slots=${12}
mtp_size=${13}
concurrency=${14}

sub_dir=${logdir}/dep${gen_tp_size}_concurrency${concurrency}_eplb${eplb_num_slots}_mtp${mtp_size}

ctx_gpus=$((num_ctx_servers * ctx_tp_size))
gen_gpus=$((num_gen_servers * gen_tp_size))

echo "enable_attention_dp: ${ctx_enable_attention_dp}, ${gen_enable_attention_dp}, gpu_memory_fraction: ${gen_gpu_memory_fraction}"

enable_pdl=false
if [ "${gen_enable_attention_dp}" = "false" ]; then
    enable_pdl=true
    echo "enable_pdl: ${enable_pdl}"
    sub_dir=${logdir}/tep${gen_tp_size}_concurrency${concurrency}_eplb${eplb_num_slots}_mtp${mtp_size}
fi

full_logdir=${sub_dir}
mkdir -p ${full_logdir}

# start the container
srun -l --container-image=${container_image} \
        --container-name=${container_name} \
        --container-mounts=${mount_dir}:${mount_dir} \
        --mpi=pmix \
        echo "Container up."

# generate the yaml file
srun -l --container-name=${container_name} \
        --container-mounts=${mount_dir}:${mount_dir} \
        --mpi=pmix --overlap \
        python3 ${workdir}/${gen_yaml_file} --config ${full_logdir}/config.yaml \
            --model ${model_dir} \
            --num_ctx_servers ${num_ctx_servers} \
            --ctx_tp_size ${ctx_tp_size} \
            --ctx_batch_size ${ctx_batch_size} \
            --ctx_max_num_tokens ${ctx_max_num_tokens} \
            --num_gen_servers ${num_gen_servers} \
            --gen_tp_size ${gen_tp_size} \
            --gen_batch_size ${gen_batch_size} \
            --gen_max_num_tokens ${gen_max_num_tokens} \
            --gen_gpu_memory_fraction ${gen_gpu_memory_fraction} \
            --eplb_num_slots ${eplb_num_slots} \
            $(if [ "${gen_enable_attention_dp}" = "true" ]; then echo "--gen_enable_attention_dp"; fi) \
            $(if [ "${ctx_enable_attention_dp}" = "true" ]; then echo "--ctx_enable_attention_dp"; fi) \
            $(if [ "${mtp_size}" -gt 0 ]; then echo "--mtp_size ${mtp_size}"; fi)

echo "YAML file generated."

hostname_value=$(grep '^hostname:' ${full_logdir}/config.yaml | awk -F': ' '{print $2}')
echo "server host name: $hostname_value"

# try to kill the server and workers
srun -l --container-name=${container_name} \
        --container-mounts=${mount_dir}:${mount_dir} \
        --mpi=pmix --overlap \
        pkill -f "trtllm-serve" || true

nsys_on=""
# nsys_on=${full_logdir}

# start the workers
srun -l --container-name=${container_name} \
        --container-mounts=${mount_dir}:${mount_dir} \
    --mpi=pmix --overlap \
    bash ${workdir}/start_worker.sh ${full_logdir}/config.yaml "${concurrency}" "${enable_pdl}" ${ctx_gpus} ${nsys_on} &> ${full_logdir}/output_workers.log &
# start the server
srun -l --container-name=${container_name} \
        --container-mounts=${mount_dir}:${mount_dir} \
        --mpi=pmix --overlap -N 1 -n 1 \
        -w ${hostname_value} \
        bash ${workdir}/start_server.sh ${full_logdir}/config.yaml &> ${full_logdir}/output_server.log &
# start benchmarking
srun -l --container-name=${container_name} \
        --container-mounts=${mount_dir}:${mount_dir} \
        --mpi=pmix --overlap -N 1 -n 1 \
        bash ${workdir}/run_benchmark.sh ${isl} ${osl} ${multi_round} ${model_dir} "${concurrency}" ${streaming} ${full_logdir}/ > ${full_logdir}/benchmark.log 2>&1

# try to kill the server and workers
srun -l --container-name=${container_name} \
        --container-mounts=${mount_dir}:${mount_dir} \
        --mpi=pmix --overlap \
        kill -9 $(ps aux | grep '[t]rtllm-serve' | awk '{print $2}') >/dev/null 2>&1 || true
wait
