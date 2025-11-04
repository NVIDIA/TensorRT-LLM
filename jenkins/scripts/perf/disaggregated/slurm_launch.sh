#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=4
#SBATCH --time=04:00:00
#SBATCH --partition=gb200nvl72_preprod
#SBATCH --account=blackwell
#SBATCH --job-name=coreai_comparch_trtllm
#SBATCH --output=/home/scratch.chenfeiz_gpu/another_repo/tekit-2/pytest_disagg_workspace-1/job-output.log

echo "Starting job $SLURM_JOB_ID on $SLURM_NODELIST"

set -Eeuo pipefail
export jobWorkspace='/home/scratch.chenfeiz_gpu/another_repo/tekit-2/pytest_disagg_workspace-1'
export llmSrcNode='/home/scratch.chenfeiz_gpu/another_repo/tekit-2'
export stageName='GB200-8_GPUs-2_Nodes-PyTorch-Perf-Sanity-Disagg'
export perfMode='true'
export pytestCommand='LLM_ROOT=/home/scratch.chenfeiz_gpu/another_repo/tekit-2 LLM_BACKEND_ROOT=/home/scratch.chenfeiz_gpu/another_repo/tekit-2/triton_backend __LUNOWUD="-thread_pool_size=12" LLM_MODELS_ROOT=/home/scratch.trt_llm_data/llm-models /home/scratch.chenfeiz_gpu/another_repo/tekit-2/tensorrt_llm/llmapi/trtllm-llmapi-launch pytest -v --perf --perf-log-formats=csv --perf-log-formats yaml --timeout-method=thread --timeout=7200 --rootdir /home/scratch.chenfeiz_gpu/another_repo/tekit-2/tests/integration/defs --test-prefix=GB200-8_GPUs-2_Nodes-PyTorch-Perf-Sanity-Disagg --output-dir=/home/scratch.chenfeiz_gpu/another_repo/tekit-2/pytest_disagg_workspace-1 --csv=/home/scratch.chenfeiz_gpu/another_repo/tekit-2/pytest_disagg_workspace-1/report.csv --junit-xml /home/scratch.chenfeiz_gpu/another_repo/tekit-2/pytest_disagg_workspace-1/results.xml -o junit_logging=out-err --test-list=/home/scratch.chenfeiz_gpu/another_repo/tekit-2/pytest_disagg_workspace-1/test_list.txt --splitting-algorithm least_duration --splits 1 --group 1'
export pytestCommandNoLLMAPILaunch='LLM_ROOT=/home/scratch.chenfeiz_gpu/another_repo/tekit-2 LLM_BACKEND_ROOT=/home/scratch.chenfeiz_gpu/another_repo/tekit-2/triton_backend __LUNOWUD="-thread_pool_size=12" LLM_MODELS_ROOT=/home/scratch.trt_llm_data/llm-models pytest -v --perf --perf-log-formats=csv --perf-log-formats yaml --timeout-method=thread --timeout=7200 --rootdir /home/scratch.chenfeiz_gpu/another_repo/tekit-2/tests/integration/defs --test-prefix=GB200-8_GPUs-2_Nodes-PyTorch-Perf-Sanity-Disagg --output-dir=/home/scratch.chenfeiz_gpu/another_repo/tekit-2/pytest_disagg_workspace-1 --csv=/home/scratch.chenfeiz_gpu/another_repo/tekit-2/pytest_disagg_workspace-1/report.csv --junit-xml /home/scratch.chenfeiz_gpu/another_repo/tekit-2/pytest_disagg_workspace-1/results.xml -o junit_logging=out-err --test-list=/home/scratch.chenfeiz_gpu/another_repo/tekit-2/pytest_disagg_workspace-1/test_list.txt --splitting-algorithm least_duration --splits 1 --group 1'
export NVIDIA_IMEX_CHANNELS=0
export NVIDIA_VISIBLE_DEVICES=$(seq -s, 0 $(($(nvidia-smi --query-gpu=count -i 0 --format=noheader)-1)))
export TRTLLM_CONFIG_FOLDER='/home/scratch.chenfeiz_gpu/another_repo/tekit-2/jenkins/scripts/perf/disaggregated'
export OPEN_SEARCH_DB_BASE_URL='http://gpuwa.nvidia.com'
export runScript=/home/scratch.chenfeiz_gpu/another_repo/tekit-2/jenkins/scripts/perf/disaggregated/slurm_run.sh
export numCtxServers=1
export numGenServers=1
export gpusPerNode=4
export gpusPerCtxServer=4
export gpusPerGenServer=4
export nodesPerCtxServer=1
export nodesPerGenServer=1
export totalNodes=2
export totalGpus=8
srunArgs=(
  "--container-image=urm.nvidia.com#sw-tensorrt-docker/tensorrt-llm:pytorch-25.10-py3-aarch64-ubuntu24.04-trt10.13.3.9-skip-tritondevel-202511200955-9055"
  "--container-workdir=/home/scratch.chenfeiz_gpu/another_repo/tekit-2"
  "--container-mounts=/home/chenfeiz/:/home/chenfeiz/,/home/chenfeiz/.cache:/root/.cache,/home/scratch.chenfeiz_gpu:/home/scratch.chenfeiz_gpu,/home/scratch.trt_llm_data:/home/scratch.trt_llm_data"
  "--container-env=NVIDIA_IMEX_CHANNELS"
  "--mpi=pmix"
  "--container-env=DISAGG_SERVER_IDX"
  "--container-env=OPEN_SEARCH_DB_BASE_URL"
)

cleanup_on_failure() {
    echo "Error: $1"
    scancel ${SLURM_JOB_ID}
    exit 1
}

mkdir -p $jobWorkspace

# Start container
echo "Starting container..."
if ! srun "${srunArgs[@]}" echo "Container up." &> $jobWorkspace/container_launch.log; then
    cleanup_on_failure "Failed to start container. Check $jobWorkspace/container_launch.log"
fi

chmod +x $runScript
# Start gen servers
echo "Starting gen servers..."
for i in $(seq 0 $((numGenServers - 1))); do
    gen_world_size=$((nodesPerGenServer * gpusPerNode))
    export DISAGG_SERVER_IDX="GEN_$i"
    srun "${srunArgs[@]}" --kill-on-bad-exit=1 \
        -N $nodesPerGenServer \
        --ntasks=$gen_world_size \
        --ntasks-per-node=$gpusPerNode \
        $runScript &> $jobWorkspace/gen_server_$i.log &
    echo "Started gen server $i"
done

# Start ctx servers
echo "Starting ctx servers..."
for i in $(seq 0 $((numCtxServers - 1))); do
    ctx_world_size=$((nodesPerCtxServer * gpusPerNode))
    export DISAGG_SERVER_IDX="CTX_$i"
    srun "${srunArgs[@]}" --kill-on-bad-exit=1 \
        -N $nodesPerCtxServer \
        --ntasks=$ctx_world_size \
        --ntasks-per-node=$gpusPerNode \
        $runScript &> $jobWorkspace/ctx_server_$i.log &
    echo "Started ctx server $i"
done

# Wait until all nodes's installation is completed
sleep 300

# Start disagg server
echo "Starting disagg server..."
export DISAGG_SERVER_IDX="DISAGG_SERVER"
srun "${srunArgs[@]}" --kill-on-bad-exit=1 --overlap -l \
    -N 1 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    $runScript &> $jobWorkspace/disagg_server.log &
echo "Started disagg server"

# Start benchmark
echo "Starting benchmark..."
export DISAGG_SERVER_IDX="BENCHMARK"
if ! srun "${srunArgs[@]}" --kill-on-bad-exit=1 --overlap -l \
    -N 1 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    $runScript &> $jobWorkspace/benchmark.log; then
    cleanup_on_failure "Benchmark failed. Check logs in ${jobWorkspace} for details"
fi

echo "Disagg server and benchmark completed successfully"
echo "Total runtime: $SECONDS seconds"
