
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

srunLayoutArgs=(-N "$totalNodes" --ntasks-per-node="$gpusPerNodePerServer")

run_aggregated_srun() {
    srun "${srunArgs[@]}" $mpiArg --kill-on-bad-exit=1 \
        "${srunLayoutArgs[@]}" \
        --ntasks=$world_size \
        $runScript
}

# Write an srun hostfile (one line per task) that starts with this batch node and keeps
# $gpusPerNodePerServer tasks on each of $totalNodes job nodes.
write_batch_node_first_hostfile() {
    local hostfile=$1
    local batchNode=${SLURMD_NODENAME:-}
    local jobNodes node i
    jobNodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST") || return 1
    if [ -z "$batchNode" ] || ! grep -qx "$batchNode" <<< "$jobNodes" ||
        [ "$(grep -c . <<< "$jobNodes")" -lt "$totalNodes" ]; then
        return 1
    fi
    { echo "$batchNode"; grep -vx "$batchNode" <<< "$jobNodes"; } | head -n "$totalNodes" |
        while read -r node; do
            for ((i = 0; i < gpusPerNodePerServer; i++)); do
                echo "$node"
            done
        done > "$hostfile"
}

if [ "${TRTLLM_DISAGG_BENCHMARK_GEN_ONLY:-0}" = "1" ]; then
    genWorkerLog="$testOutputDir/gen_server_0.log"
    rm -f "$genWorkerLog"
    echo "gen_only_no_context: srun output -> $genWorkerLog"
    # pytest in srun task 0 parses $genWorkerLog, which this batch node appends to, and on a
    # shared filesystem another node can see those appends tens of seconds late. srun puts
    # task 0 on the lowest-named node, so when that is another node, pin task 0 here. Tasks
    # stay unbound to CPUs, as with the default layout's all-CPU mask, so TRT-LLM still
    # applies its NUMA-aware pinning.
    if [ "$totalNodes" -gt 1 ] &&
        [ "${SLURMD_NODENAME:-}" != "$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sort | head -n 1)" ]; then
        srunHostfile="$testOutputDir/srun_hostfile.txt"
        if write_batch_node_first_hostfile "$srunHostfile"; then
            export SLURM_HOSTFILE="$srunHostfile"
            srunLayoutArgs=(--distribution=arbitrary --cpu-bind=none)
            # The container's Slurm hook sets MASTER_ADDR to the first node of the step node
            # list, which need not be task 0's node in this layout.
            pytestCommand="export MASTER_ADDR=$SLURMD_NODENAME && $pytestCommand"
            echo "gen_only_no_context: requesting srun task 0 on batch node $SLURMD_NODENAME"
        else
            echo "Warning: could not pin srun task 0 to the batch node (SLURMD_NODENAME=${SLURMD_NODENAME:-unset}); pytest may parse $genWorkerLog from another node"
        fi
    fi
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
