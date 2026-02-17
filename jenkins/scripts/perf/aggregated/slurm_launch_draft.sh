
cleanup_on_failure() {
    echo "Error: $1"
    scancel ${SLURM_JOB_ID}
    exit 1
}

mkdir -p $jobWorkspace
chmod +x $runScript

# Run aggregated test
echo "Starting aggregated test..."
world_size=$((totalNodes * gpusPerNodePerServer))
if ! srun "${srunArgs[@]}" --kill-on-bad-exit=1 \
    -N $totalNodes \
    --ntasks=$world_size \
    --ntasks-per-node=$gpusPerNodePerServer \
    $runScript; then
    cleanup_on_failure "Aggregated test failed. Check logs in ${jobWorkspace} for details"
fi

echo "Aggregated test completed successfully"
echo "Total runtime: $SECONDS seconds"
