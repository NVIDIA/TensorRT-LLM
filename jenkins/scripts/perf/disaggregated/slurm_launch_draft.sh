
cleanup_on_failure() {
    echo "Error: $1"
    scancel ${SLURM_JOB_ID}
    exit 1
}

mkdir -p $jobWorkspace
mkdir -p "$testOutputDir"
chmod +x $runScript
chmod +x $installScript

# Run installation on all nodes
echo "Running installation on all nodes..."
if ! srun "${srunArgs[@]}" $installScript &> $jobWorkspace/install.log; then
    cleanup_on_failure "Failed to run installation. Check $jobWorkspace/install.log"
fi
echo "Installation completed on all nodes"

# Deterministic node slices per server: gen servers take the first nodes,
# then ctx servers (same order the steps are started in). Both the cache
# transceiver precheck and the real server steps pin to these slices with
# `srun -w`, so the precheck exercises exactly the node pairs / NICs the
# real disaggregated test will use.
mapfile -t allNodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodeCursor=0
genNodeLists=()
for i in $(seq 0 $((numGenServers - 1))); do
    slice=("${allNodes[@]:$nodeCursor:$nodesPerGenServer}")
    genNodeLists+=("$(IFS=,; echo "${slice[*]}")")
    nodeCursor=$((nodeCursor + nodesPerGenServer))
done
ctxNodeLists=()
if [ "${TRTLLM_DISAGG_BENCHMARK_GEN_ONLY:-0}" != "1" ]; then
    for i in $(seq 0 $((numCtxServers - 1))); do
        slice=("${allNodes[@]:$nodeCursor:$nodesPerCtxServer}")
        ctxNodeLists+=("$(IFS=,; echo "${slice[*]}")")
        nodeCursor=$((nodeCursor + nodesPerCtxServer))
    done
fi
if [ "$nodeCursor" -gt "${#allNodes[@]}" ]; then
    cleanup_on_failure "Node slicing needs $nodeCursor nodes but the job only has ${#allNodes[@]} ($SLURM_JOB_NODELIST)"
fi

# Cache transceiver network precheck: same instance count / node slices /
# MPI topology / UCX env as the real ctx+gen server steps. On failure the
# stage aborts HERE, with per-instance verdicts + a synthetic junit entry,
# before any model bring-up. Functions come from slurm_ct_precheck_gate.sh,
# spliced in above this draft by submit.py. No-op unless ctPrecheckEnabled=1.
run_cache_transceiver_precheck

# Start gen servers
echo "Starting gen servers..."
for i in $(seq 0 $((numGenServers - 1))); do
    gen_world_size=$((nodesPerGenServer * gpusPerNodePerGenServer))
    export DISAGG_SERVING_TYPE="GEN_$i"
    export pytestCommand="$pytestCommandGENWorker"
    srun "${srunArgs[@]}" --mpi=pmix --kill-on-bad-exit=1 \
        -N $nodesPerGenServer \
        -w "${genNodeLists[$i]}" \
        --ntasks=$gen_world_size \
        --ntasks-per-node=$gpusPerNodePerGenServer \
        $runScript &> $testOutputDir/gen_server_$i.log &
    echo "Started gen server $i on ${genNodeLists[$i]}"
    sleep 5  # Wait for pyxis container namespace initialization to avoid race condition
done

# Start ctx servers (skip if gen_only_no_context mode)
if [ "${TRTLLM_DISAGG_BENCHMARK_GEN_ONLY:-0}" != "1" ]; then
    echo "Starting ctx servers..."
    for i in $(seq 0 $((numCtxServers - 1))); do
        ctx_world_size=$((nodesPerCtxServer * gpusPerNodePerCtxServer))
        export DISAGG_SERVING_TYPE="CTX_$i"
        export pytestCommand="$pytestCommandCTXWorker"
        srun "${srunArgs[@]}" --mpi=pmix --kill-on-bad-exit=1 \
            -N $nodesPerCtxServer \
            -w "${ctxNodeLists[$i]}" \
        --ntasks=$ctx_world_size \
        --ntasks-per-node=$gpusPerNodePerCtxServer \
            $runScript &> $testOutputDir/ctx_server_$i.log &
        echo "Started ctx server $i on ${ctxNodeLists[$i]}"
        sleep 5  # Wait for pyxis container namespace initialization to avoid race condition
    done
else
    echo "Skipping ctx servers (gen_only_no_context mode)"
fi

sleep 5  # Wait for pyxis container namespace initialization to avoid race condition

# Start disagg server
echo "Starting disagg server..."
export DISAGG_SERVING_TYPE="DISAGG_SERVER"
export pytestCommand="$pytestCommandDisaggServer"
srun "${srunArgs[@]}" --kill-on-bad-exit=1 --overlap \
    -N 1 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    $runScript &> $testOutputDir/disagg_server.log &
echo "Started disagg server"
sleep 5  # Wait for pyxis container namespace initialization to avoid race condition

# Start benchmark
echo "Starting benchmark..."
export DISAGG_SERVING_TYPE="BENCHMARK"
export pytestCommand="$pytestCommandBenchmark"
if ! srun "${srunArgs[@]}" --kill-on-bad-exit=1 --overlap \
    -N 1 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    $runScript; then
    cleanup_on_failure "Benchmark failed. See slurm-${SLURM_JOB_ID}.out"
fi

echo "Disagg server and benchmark completed successfully"
echo "Total runtime: $SECONDS seconds"
