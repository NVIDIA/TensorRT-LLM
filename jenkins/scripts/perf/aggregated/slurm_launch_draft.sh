
cleanup_on_failure() {
    echo "Error: $1"
    scancel ${SLURM_JOB_ID}
    exit 1
}

mkdir -p $jobWorkspace

if [ "${buildWheel}" = "true" ]; then
    echo "Building TensorRT-LLM wheel on one node..."

    build_command="python3 ./scripts/build_wheel.py --benchmarks --use_ccache --clean --cuda_architectures '100-real'"
    if ! srun "${srunArgs[@]}" --kill-on-bad-exit=1 --overlap -N 1 --ntasks-per-node=1 --ntasks=1 \
        bash -c "cd ${llmSrcNode} && ${build_command}" \
        &> ${jobWorkspace}/build.log; then
        cleanup_on_failure "TensorRT-LLM build failed. Check ${jobWorkspace}/build.log for details"
    fi
    echo "TensorRT-LLM build completed successfully"
fi

echo "Running aggregated perf test..."
srun "${srunArgs[@]}" \
        --container-remap-root \
        -n ${totalGpus} \
        --nodes ${totalNodes} \
        bash ${runScript} &> ${jobWorkspace}/run.log

echo "Run completed. Log: ${jobWorkspace}/run.log"
echo "Job completed successfully!"
