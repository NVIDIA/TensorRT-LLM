#! /usr/bin/bash

#SBATCH -N1
#SBATCH -n1
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:4

set -ex

env && hostname && nvidia-smi
start_time=$(date '+%Y-%m-%d-%H:%M:%S')
output_folder=benchmark.run.${SLURM_JOB_ID}.${start_time}
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# the docker user is root, otherwise it can not write logs to this folder when running in scratch
mkdir -p ${output_folder} && chmod 777 ${output_folder}

DEFAULT_IMAGE="urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm-staging/release:main-x86_64"
IMAGE=${1:-$DEFAULT_IMAGE}

run_benchmark() {
    docker run --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
        -v /home/scratch.trt_llm_data:/home/scratch.trt_llm_data:ro \
        -v `pwd`:`pwd` \
        -w `pwd`  \
        --pull always \
        -it  \
        ${IMAGE} \
        bash ./run_benchmark_bench.sh ${output_folder}
}


parse_report() {
    results=$1
    end_time=$(date '+%Y-%m-%d-%H:%M:%S')
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

report_head() {
    echo "trtllm-bench Job ${SLURM_JOB_ID} started at:${start_time} on:$(hostname) under:$(pwd)
    git commit: $(git log --oneline origin/main | head -1)
    output: ${output_folder} "
}

[[ -e ~/bin/slack.sh ]] && report_head | ~/bin/slack.sh

run_benchmark

[[ -e ~/bin/slack.sh ]] && parse_report ${output_folder} | ~/bin/slack.sh