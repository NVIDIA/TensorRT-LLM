#! /usr/bin/bash

#SBATCH -N1
#SBATCH -n1
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:4

set -ex
# The job must be executed under a TRT-LLM clone root folder

env && hostname && nvidia-smi
start_time=$(date '+%Y-%m-%d-%H:%M:%S')
output_folder=benchmark.run.${SLURM_JOB_ID}.$(date '+%Y-%m-%d-%H:%M:%S')
mkdir -p ${output_folder} && chmod 777 ${output_folder} # the docker user is root, otherwise it can not write logs to this folder

run_benchmark() {
    IMAGE="urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm-staging/release:main-x86_64"
    docker run --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
        -v /home/scratch.trt_llm_data:/home/scratch.trt_llm_data:ro \
        -v `pwd`:`pwd` \
        -w `pwd`  \
        --pull always \
        -it  \
        ${IMAGE} \
        bash ./run_benchmark_bench.sh ${output_folder}
}


report() {
    results=$1
    echo "Performance report ${SLURM_JOB_ID} - trtllm-bench"
    echo "==========================================="
    echo "Report path" $(realpath ${results})
    echo "START" $start_time "-" "END" ${end_time} $(hostname)
    grep -Hn "Total Output Throughput" ${results}/bench.*log \
         | awk -F "/bench." '{print $2}' | awk -F ":" '{print $1" "$4}' | awk -F ".log" '{print $1" "$2}' \
         | awk -F "." '{print $1" "$2}'    | awk -F " " '{print $1" "$2" "$3}' | sort -k1,1 -k2,2nr \
         || true
    echo "==========================================="
}

echo "trtllm-bench Job ${SLURM_JOB_ID} started at:${start_time} on:$(hostname) under:$(pwd)
git commit: $(git log --oneline origin/main | head -1)
output: ${output_folder} " | ~/bin/slack.sh

run_benchmark
report ${output_folder} | ~/bin/slack.sh

end_time=$(date '+%Y-%m-%d-%H:%M:%S')