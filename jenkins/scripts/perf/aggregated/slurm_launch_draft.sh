
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
if ! srun "${srunArgs[@]}" --mpi=pmi2 --kill-on-bad-exit=1 \
    -N $totalNodes \
    --ntasks=$world_size \
    --ntasks-per-node=$gpusPerNodePerServer \
    $runScript &> $testOutputDir/benchmark.log; then
    cleanup_on_failure "Aggregated test failed. See ${testOutputDir}/benchmark.log"
fi

echo "Aggregated test completed successfully"
echo "Total runtime: $SECONDS seconds"
