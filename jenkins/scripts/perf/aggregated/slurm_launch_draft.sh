
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

run_aggregated_srun() {
    srun "${srunArgs[@]}" --mpi=pmix --kill-on-bad-exit=1 \
        -N $totalNodes \
        --ntasks=$world_size \
        --ntasks-per-node=$gpusPerNodePerServer \
        $runScript
}

if [ "${TRTLLM_DISAGG_BENCHMARK_GEN_ONLY:-0}" = "1" ]; then
    # gen_only_no_context runs a disaggregated topology on this aggregated launch
    # path, and its only regression signal is the gen worker's per-iteration
    # prev_device_step_time lines. Those are emitted by the worker's rank 0,
    # which under trtllm-llmapi-launch lives in the mgmn_leader_node process --
    # a *sibling* of pytest, not the trtllm-serve child whose stdout pytest
    # redirects -- so its output goes to this srun's stdout and nowhere else.
    # Land that aggregate in gen_server_0.log: the same filename, and the same
    # role, as the redirect on the disaggregated path, which is what lets
    # parse_gen_worker_device_step_time and the whole upload path stay
    # byte-identical between the two gen modes.
    #
    # Appended, not truncated: the pytest inside this srun also opens this path
    # to add the trtllm-serve child's own output (the single-node case, where no
    # launcher is used and the executor runs in-process, has its iteration lines
    # there instead). Both writers use O_APPEND, so neither can overwrite the
    # other and the byte offsets stay monotonic for the per-client windowing.
    genWorkerLog="$testOutputDir/gen_server_0.log"
    rm -f "$genWorkerLog"
    echo "gen_only_no_context: srun output -> $genWorkerLog"
    if ! run_aggregated_srun >> "$genWorkerLog" 2>&1; then
        # The console has nothing without this: everything the test printed went
        # to the log above, and cleanup_on_failure scancels the job immediately.
        echo "--- tail -n 200 $genWorkerLog ---"
        tail -n 200 "$genWorkerLog" || true
        cleanup_on_failure "Aggregated test failed. See $genWorkerLog"
    fi
elif ! run_aggregated_srun; then
    cleanup_on_failure "Aggregated test failed. See slurm-${SLURM_JOB_ID}.out"
fi

echo "Aggregated test completed successfully"
echo "Total runtime: $SECONDS seconds"
