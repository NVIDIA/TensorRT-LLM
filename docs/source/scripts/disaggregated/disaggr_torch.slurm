#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --partition=batch
#SBATCH --account=${account}
#SBATCH --time=02:00:00
#SBATCH --job-name="${account}:disaggr-test"

isl=8192
osl=256
multi_round=10
gen_yaml_file=gen_yaml.py
container_image=${docker_image}
mount_dir=/${account}/${user}/
workdir=/${account}/${user}/8k-${osl}/disaggr-e2e/
model_dir=/${account}/${user}/DeepSeek-R1-nvfp4_allmoe/
logdir=$workdir/bm_deepseek-r1-8k-${osl}-disaggr-e2e-nostream
streaming=false
mkdir -p ${logdir}

dep_dir=${workdir}
run_benchmark_cmd="bash ${dep_dir}/run_benchmark.sh"

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
concurrency_list=${12}
sub_file=${13}

# concurrency=$((concurrency * gen_tp_size))
echo "concurrency_list: ${concurrency_list}"

ctx_gpus=$((num_ctx_servers * ctx_tp_size))
gen_gpus=$((num_gen_servers * gen_tp_size))

echo "enable_attention_dp: ${ctx_enable_attention_dp}, ${gen_enable_attention_dp}, gpu_memory_fraction: ${gen_gpu_memory_fraction}"

enable_pdl=false
if [ "${gen_enable_attention_dp}" = "false" ]; then
    enable_pdl=true
fi

full_logdir=${logdir}/${sub_file}
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
        python3 ${dep_dir}/${gen_yaml_file} --config ${full_logdir}/config.yaml \
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
            $(if [ "${gen_enable_attention_dp}" = "true" ]; then echo "--gen_enable_attention_dp"; fi) \
            $(if [ "${ctx_enable_attention_dp}" = "true" ]; then echo "--ctx_enable_attention_dp"; fi)

echo "YAML file generated."

hostname_value=$(grep '^hostname:' ${full_logdir}/config.yaml | awk -F': ' '{print $2}')
echo "server host name: $hostname_value"

nsys_on=""
# nsys_on=${full_logdir}

# start the workers
srun -l --container-name=${container_name} \
        --container-mounts=${mount_dir}:${mount_dir} \
        --mpi=pmix --overlap \
        bash ${dep_dir}/start_worker.sh ${full_logdir}/config.yaml "${enable_pdl}" ${ctx_gpus} ${nsys_on} &> ${full_logdir}/output_workers.log &
# start the server
srun -l --container-name=${container_name} \
        --container-mounts=${mount_dir}:${mount_dir} \
        --mpi=pmix --overlap -N 1 -n 1 \
        bash trtllm-serve disaggregated -c ${full_logdir}/config.yaml -t 1800 -r 1800 &> ${full_logdir}/output_server.log &
# start benchmark
srun -l --container-name=${container_name} \
        --container-mounts=${mount_dir}:${mount_dir} \
        --mpi=pmix --overlap -N 1 -n 1 \
        --nodelist=${hostname_value} \
        ${run_benchmark_cmd} ${isl} ${osl} ${multi_round} ${model_dir} "${concurrency_list}" ${streaming} ${full_logdir}/ > ${full_logdir}/benchmark.log 2>&1
wait

# try to kill the server and workers
srun -l --container-name=${container_name} \
        --container-mounts=${mount_dir}:${mount_dir} \
        --mpi=pmix --overlap \
        pkill -f "trtllm-serve" || true
