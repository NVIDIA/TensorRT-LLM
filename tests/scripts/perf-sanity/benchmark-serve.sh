#! /usr/bin/bash
#SBATCH -N1
#SBATCH -n1
#SBATCH --time=08:00:00
#SBATCH --gres=gpu:8

set -ex

env && hostname && nvidia-smi

DEFAULT_IMAGE="urm.nvidia.com/sw-tensorrt-docker/tensorrt-llm-staging/release:main-x86_64"
IMAGE=${1:-$DEFAULT_IMAGE}
config_file=${2:-$(pwd)}
select_pattern=${3:-default}
bench_dir=${4:-$(pwd)}
output_dir=${5:-$(pwd)}
trtllm_dir=${6:-""}
extra_options=${7:-""}

start_time=$(date '+%Y-%m-%d-%H:%M:%S')
output_folder=${output_dir}/benchmark.run.${SLURM_JOB_ID}.${start_time}.${select_pattern}

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
    mount=" -v /home/scratch.trt_llm_data:/home/scratch.trt_llm_data:ro -v $output_dir:$output_dir:rw -v $bench_dir:$bench_dir:ro"
    if [[ -n "$trtllm_dir" && -d "$trtllm_dir" ]]; then
        mount="$mount -v $trtllm_dir:$trtllm_dir:ro"
    fi
    docker run --rm --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 $extra_options \
        --gpus all \
        $mount \
        -w `pwd`  \
        --pull always \
        ${IMAGE} \
        bash -c "
            echo 'Running benchmarks...'
            export LLM_MODELS_ROOT=/home/scratch.trt_llm_data/llm-models

            # Handle trtllm_dir parameter
            if [[ -n \"$trtllm_dir\" && -d \"$trtllm_dir\" ]]; then
                echo 'Installing TensorRT-LLM packages from $trtllm_dir...'
                pip uninstall tensorrt_llm -y
                pip install $trtllm_dir/build/tensorrt_llm*.whl --pre
                export PATH=\$HOME/.local/bin:\$PATH
                export PYTHONPATH=$trtllm_dir
                echo 'TensorRT-LLM packages installed successfully'
            else
                echo 'No trtllm_dir specified or directory does not exist, running with default packages'
            fi

            python3 ${bench_dir}/run_benchmark_serve.py --output_folder ${output_folder} --config_file ${bench_dir}/${config_file} --select ${select_pattern}

            echo 'Benchmarks completed. Parsing results...'
            if [[ -f '${bench_dir}/parse_benchmark_results.py' ]]; then
                python3 ${bench_dir}/parse_benchmark_results.py --log_folder ${output_folder}
                echo 'Results parsed successfully'
            else
                echo 'Warning: parse_benchmark_results.py not found'
            fi
        "
}

report_head
run_benchmark_and_parse
