#! /usr/bin/bash

# SLURM configuration for 8-GPU benchmark node
#SBATCH -N1
#SBATCH -n1
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:8

set -ex

env && hostname && nvidia-smi
echo "TRT-LLM GIT COMMIT": $TRT_LLM_GIT_COMMIT

DEFAULT_IMAGE="urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm-staging/release:main-x86_64"
IMAGE=${1:-$DEFAULT_IMAGE}
config_file=${2:-benchmark_config.yaml}
skip_pattern=${3:-}
select_pattern=${4:-}

start_time=$(date '+%Y-%m-%d-%H:%M:%S')
output_folder=benchmark.run.${SLURM_JOB_ID}.${start_time}.${TRT_LLM_GIT_COMMIT}.${skip_pattern}.${select_pattern}
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# the docker user is root, otherwise it can not write logs to this folder when running in scratch
mkdir -p ${output_folder} && chmod 777 ${output_folder}

run_benchmark() {
    # Run benchmark on 8-GPU node using Docker
    docker run --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
        --gpus all \
        -v /home/scratch.trt_llm_data:/home/scratch.trt_llm_data:ro \
        -v `pwd`:`pwd` \
        -w `pwd`  \
        --pull always \
        -it  \
        ${IMAGE} \
        python ${script_dir}/run_benchmark_serve.py --output_folder ${output_folder} --commit ${TRT_LLM_GIT_COMMIT} --config_file ${config_file} --skip ${skip_pattern} --select ${select_pattern}
}

parse_report() {
    results=$1
    end_time=$(date '+%Y-%m-%d-%H:%M:%S')
    echo "Performance report ${SLURM_JOB_ID} - trtllm-serve (8-GPU)"
    echo "==========================================="
    echo "Report path" $(realpath ${results})
    echo "START" $start_time "-" "END" ${end_time} $(hostname)
    
    # Use the Python script to parse and generate Excel report
    if [[ -f "${script_dir}/parse_benchmark_results.py" ]]; then
        echo "Generating Excel report..."
        python ${script_dir}/parse_benchmark_results.py ${results}
        echo "Excel report generated successfully"
    else
        echo "Warning: parse_benchmark_results.py not found, falling back to basic parsing"
        
        # Process each log file in the results directory
        for log_file in ${results}/serve.*log; do
            if [[ -f "$log_file" ]]; then
                # Extract the log filename without path
                log_name=$(basename "$log_file")
                echo "Log: $log_name"
                
                # Extract Total Token throughput
                total_throughput=$(grep "Total Token throughput (tok/s):" "$log_file" | awk '{print $5}')
                if [[ -n "$total_throughput" ]]; then
                    echo "  Total Token throughput (tok/s): $total_throughput"
                fi
                
                # Extract User throughput
                user_throughput=$(grep "User throughput (tok/s):" "$log_file" | awk '{print $4}')
                if [[ -n "$user_throughput" ]]; then
                    echo "  User throughput (tok/s): $user_throughput"
                fi
                
                echo ""
            fi
        done
    fi
    
    echo "==========================================="
}

report_head() {
    echo "trtllm-serve Job ${SLURM_JOB_ID} started at:${start_time} on:$(hostname) under:$(pwd)
    output: ${output_folder} "
}

[[ -e ~/bin/slack.sh ]] && report_head | ~/bin/slack.sh

run_benchmark

[[ -e ~/bin/slack.sh ]] && parse_report ${output_folder} | ~/bin/slack.sh