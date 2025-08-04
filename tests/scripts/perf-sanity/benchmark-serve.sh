#! /usr/bin/bash
#SBATCH -N1
#SBATCH -n1
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:8

set -ex

env && hostname && nvidia-smi

DEFAULT_IMAGE="urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm-staging/release:main-x86_64"
IMAGE=${1:-$DEFAULT_IMAGE}
bench_dir=${2:-$(pwd)}
output_dir=${3:-$(pwd)}
select_pattern=${4:-default}
skip_pattern=${5:-default}

start_time=$(date '+%Y-%m-%d-%H:%M:%S')
output_folder=${output_dir}/benchmark.run.${SLURM_JOB_ID}.${start_time}.${select_pattern}.${skip_pattern}

# Validate bench_dir exists
if [[ ! -d "$bench_dir" ]]; then
    echo "Error: bench_dir '$bench_dir' does not exist"
    exit 1
fi

if [[ ! -d "$output_dir" ]]; then
    echo "Error: output_dir '$output_dir' does not exist"
    exit 1
fi

# the docker user is root, otherwise it can not write logs to this folder when running in scratch
chmod 777 ${output_dir}
mkdir -p ${output_folder} && chmod 777 ${output_folder}

cd ${output_dir}

report_head() {
    echo "trtllm-serve Job ${SLURM_JOB_ID} started at:${start_time} on:$(hostname) under:$(pwd)
    output: ${output_folder} "
}

run_benchmark_and_parse() {
    # Run benchmark and parse results in a single Docker container
    docker run --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
        --gpus all \
        -v /home/scratch.trt_llm_data:/home/scratch.trt_llm_data:ro \
        -v /home/scratch.svc_compute_arch:/home/scratch.svc_compute_arch:ro \
        -v /home/scratch.omniml_data_2:/home/scratch.omniml_data_2:ro \
        -v $output_dir:$output_dir:rw \
        -v $bench_dir:$bench_dir:ro \
        -w `pwd`  \
        --pull always \
        ${IMAGE} \
        bash -c "
            echo 'Running benchmarks...'
            python3 ${bench_dir}/run_benchmark_serve.py --output_folder ${output_folder} --config_file ${bench_dir}/benchmark_config.yaml --select ${select_pattern} --skip ${skip_pattern}

            echo 'Benchmarks completed. Generating CSV report...'
            if [[ -f '${bench_dir}/parse_benchmark_results.py' ]]; then
                python3 ${bench_dir}/parse_benchmark_results.py --config_file ${bench_dir}/benchmark_config.yaml --input_folder ${output_folder} --output_csv ${output_folder}.csv
                echo 'CSV report generated successfully'
            else
                echo 'Warning: parse_benchmark_results.py not found'
            fi
        "
}

report_head
run_benchmark_and_parse
