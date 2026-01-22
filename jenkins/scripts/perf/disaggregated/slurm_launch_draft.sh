
cleanup_on_failure() {
    echo "Error: $1"
    scancel ${SLURM_JOB_ID}
    exit 1
}

mkdir -p $jobWorkspace
chmod +x $runScript
chmod +x $installScript

# Run installation on all nodes
echo "Running installation on all nodes..."
if ! srun "${srunArgs[@]}" $installScript &> $jobWorkspace/install.log; then
    cleanup_on_failure "Failed to run installation. Check $jobWorkspace/install.log"
fi
echo "Installation completed on all nodes"

# Start gen servers
echo "Starting gen servers..."
for i in $(seq 0 $((numGenServers - 1))); do
    gen_world_size=$((nodesPerGenServer * gpusPerfNodePerfGenServer))
    export DISAGG_SERVING_TYPE="GEN_$i"
    export pytestCommand="$pytestCommandWorker"
    srun "${srunArgs[@]}" --kill-on-bad-exit=1 \
        -N $nodesPerGenServer \
        --ntasks=$gen_world_size \
        --ntasks-per-node=$gpusPerfNodePerfGenServer \
        $runScript &> $jobWorkspace/gen_server_$i.log &
    echo "Started gen server $i"
done

# Start ctx servers (skip if gen_only mode)
if [ "${TRTLLM_DISAGG_BENCHMARK_GEN_ONLY:-0}" != "1" ]; then
    echo "Starting ctx servers..."
    for i in $(seq 0 $((numCtxServers - 1))); do
        ctx_world_size=$((nodesPerCtxServer * gpusPerfNodePerfCtxServer))
        export DISAGG_SERVING_TYPE="CTX_$i"
        export pytestCommand="$pytestCommandWorker"
        srun "${srunArgs[@]}" --kill-on-bad-exit=1 \
            -N $nodesPerCtxServer \
        --ntasks=$ctx_world_size \
        --ntasks-per-node=$gpusPerfNodePerfCtxServer \
            $runScript &> $jobWorkspace/ctx_server_$i.log &
        echo "Started ctx server $i"
    done
else
    echo "Skipping ctx servers (gen_only mode)"
fi


# Start disagg server
echo "Starting disagg server..."
export DISAGG_SERVING_TYPE="DISAGG_SERVER"
export pytestCommand="$pytestCommandDisaggServer"
srun "${srunArgs[@]}" --kill-on-bad-exit=1 --overlap \
    -N 1 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    $runScript &> $jobWorkspace/disagg_server.log &
echo "Started disagg server"

# Start benchmark
echo "Starting benchmark..."
export DISAGG_SERVING_TYPE="BENCHMARK"
export pytestCommand="$pytestCommandBenchmark"
if ! srun "${srunArgs[@]}" --kill-on-bad-exit=1 --overlap \
    -N 1 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    $runScript; then
    cleanup_on_failure "Benchmark failed. Check logs in ${jobWorkspace} for details"
fi

echo "Disagg server and benchmark completed successfully"
echo "Total runtime: $SECONDS seconds"
