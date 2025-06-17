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

run_benchmark() {
    bash ~/bin/dev-tekit-d.sh ./run_benchmark_serve.sh ${output_folder}
}

report() {
    results=$1
    echo "Performance report ${SLURM_JOB_ID} - trtllm-serve"
    echo "==========================================="
    echo "Report path" $(realpath ${results})
    echo "START" $start_time "-" "END" ${end_time} $(hostname)
    grep -Hn "Output token throughput (tok/s):" ${results}/serve.*log \
         | awk -F "serve." '{print $2}'  | awk -F ":" '{print $1" "$4}' | awk -F ".log" '{print $1" "$2}' \
         | awk -F "." '{print $1" "$2" "}' | awk -F " " '{print $1" "$2" "$3}' | sort  -k1,1 -k2,2rn \
        ||true

    echo "==========================================="
}

mkdir -p ${output_folder}

echo "trtllm-serve Job ${SLURM_JOB_ID} started at:${start_time} on:$(hostname) under:$(pwd)
git commit: $(git log --oneline origin/main | head -1)
output: ${output_folder} " | ~/bin/slack.sh

run_benchmark
end_time=$(date '+%Y-%m-%d-%H:%M:%S')

report ${output_folder} | ~/bin/slack.sh
