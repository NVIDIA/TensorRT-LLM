
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
    export DISAGG_SERVING_TYPE="GEN_$i"
    export pytestCommand="$pytestCommandWorker"
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
    export DISAGG_SERVING_TYPE="CTX_$i"
    export pytestCommand="$pytestCommandWorker"
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
export DISAGG_SERVING_TYPE="DISAGG_SERVER"
export pytestCommand="$pytestCommandDisaggServer"
srun "${srunArgs[@]}" --kill-on-bad-exit=1 --overlap -l \
    -N 1 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    $runScript &> $jobWorkspace/disagg_server.log &
echo "Started disagg server"

# Start benchmark
echo "Starting benchmark..."
export DISAGG_SERVING_TYPE="BENCHMARK"
export pytestCommand="$pytestCommandBenchmark"
if ! srun "${srunArgs[@]}" --kill-on-bad-exit=1 --overlap -l \
    -N 1 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    $runScript &> $jobWorkspace/benchmark.log; then
    cleanup_on_failure "Benchmark failed. Check logs in ${jobWorkspace} for details"
fi

echo "Disagg server and benchmark completed successfully"
echo "Total runtime: $SECONDS seconds"
