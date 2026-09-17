
cleanup_on_failure() {
    echo "Error: $1"
    scancel ${SLURM_JOB_ID}
    exit 1
}

mkdir -p $jobWorkspace
mkdir -p "$testOutputDir"
chmod +x $runScript

# Run aggregated test
echo "Starting aggregated test..."
world_size=${world_size:-$((totalNodes * gpusPerNodePerServer))}

# Add --mpi=pmix only for multi-rank launches with no --mpi option already in srunArgs.
mpiArg=""
if [ "$world_size" -gt 1 ]; then
    mpiArg="--mpi=pmix"
    for srunArg in "${srunArgs[@]}"; do
        case "$srunArg" in
            --mpi=*) mpiArg="" ;;
        esac
    done
fi

run_aggregated_srun() {
    srun "${srunArgs[@]}" $mpiArg --kill-on-bad-exit=1 \
        -N $totalNodes \
        --ntasks=$world_size \
        --ntasks-per-node=$gpusPerNodePerServer \
        $runScript
}

if [ "${TRTLLM_DISAGG_BENCHMARK_GEN_ONLY:-0}" = "1" ]; then
    genWorkerLog="$testOutputDir/gen_server_0.log"
    rm -f "$genWorkerLog"
    echo "gen_only_no_context: srun output -> $genWorkerLog"
    if ! run_aggregated_srun >> "$genWorkerLog" 2>&1; then
        echo "--- tail -n 200 $genWorkerLog ---"
        tail -n 200 "$genWorkerLog" || true
        cleanup_on_failure "Aggregated test failed. See $genWorkerLog"
    fi
elif ! run_aggregated_srun; then
    cleanup_on_failure "Aggregated test failed. See slurm-${SLURM_JOB_ID}.out"
fi

echo "Aggregated test completed successfully"
echo "Total runtime: $SECONDS seconds"
