
# Bounded dump of the per-role worker logs into this batch script's stdout,
# i.e. into the file `#SBATCH --output` points at (the "slurm-<jobid>.out" the
# failure messages below refer to).
#
# Every ctx/gen/disagg-server srun redirects its output to a file under
# $testOutputDir, so the rank-level [TRT-LLM] output never reaches the batch
# stdout that the Jenkins tracker tails live. When the benchmark then fails or
# the job is killed, that evidence stays on the cluster and is deleted with the
# job workspace, leaving the Jenkins stage log with no record of what the ranks
# were actually doing. Emitting the tails here is what makes such a run
# diagnosable.
#
# Deliberately bounded: at most DUMP_MAX_FILES logs, DUMP_TAIL_LINES lines each,
# and at most DUMP_MAX_INVOCATIONS times per job (<= 3200 lines total). A hang or
# crash signature is at the end of a rank log, so the tail is the part worth
# shipping. The invocation cap is 2 rather than 1 on purpose: an early SIGTERM
# can dump near-empty role logs, and collapsing to a single dump would then
# permanently suppress the informative one that cleanup_on_failure would emit.
#
# The tails are scrubbed: slurm_run.sh runs `env | sort`, so every role log opens
# with a full environment dump. Without this, dumping them into the batch stdout
# would push those values into a file that is streamed to the Jenkins console and
# archived. Kept in sync with SLURM_LOG_REDACT_CMD in jenkins/L0_Test.groovy.
#
# Best-effort by construction: this only ever runs on a path that is already
# failing, so every step is guarded and the function always returns 0. It must
# not change the exit code that brought us here.
DUMP_TAIL_LINES=${DUMP_TAIL_LINES:-100}
DUMP_MAX_FILES=${DUMP_MAX_FILES:-16}
DUMP_MAX_INVOCATIONS=${DUMP_MAX_INVOCATIONS:-2}
# Anchored at ^ (with optional xtrace depth and export/declare prefix) so only an
# assignment at the start of a line is rewritten -- matching mid-line would
# truncate genuine diagnostics such as
#   [TRT-LLM] [E] Assertion failed: KEY=missing in map (kvCacheManager.cpp:812)
# Must stay byte-identical to SLURM_LOG_REDACT_CMD's regex in jenkins/L0_Test.groovy.
DUMP_REDACT_RE='s/^(\++ )?(export |declare -x )?(HF_TOKEN|S3_SECRET_KEY|[A-Za-z0-9_]*(TOKEN|SECRET|PASSWORD|PASSWD|PSW|KEY))=.*/\1\2\3=***REDACTED***/'
# Intentionally global: the invocation counter has to survive across calls.
dumped_role_logs=0

dump_role_logs() {
    # The launch script runs under `set -x`; the dump is unreadable with the
    # trace interleaved, so suppress it here and restore the caller's setting.
    local xtraceWasOn=0
    local count
    local roleLog
    case "$-" in
        *x*) xtraceWasOn=1 ;;
    esac
    { set +x; } 2>/dev/null

    if [ "$dumped_role_logs" -ge "$DUMP_MAX_INVOCATIONS" ]; then
        if [ "$xtraceWasOn" -eq 1 ]; then set -x; fi
        return 0
    fi
    dumped_role_logs=$((dumped_role_logs + 1))

    echo "===== Per-role log tails (reason: ${1:-unspecified}) ====="
    if [ -z "${testOutputDir:-}" ] || [ ! -d "${testOutputDir:-}" ]; then
        echo "No per-role logs to dump (testOutputDir='${testOutputDir:-}')"
    else
        count=0
        for roleLog in "${testOutputDir}"/*.log; do
            [ -f "$roleLog" ] || continue
            count=$((count + 1))
            if [ "$count" -gt "$DUMP_MAX_FILES" ]; then
                echo "===== More logs in ${testOutputDir} not dumped (cap ${DUMP_MAX_FILES}) ====="
                break
            fi
            echo "===== Last ${DUMP_TAIL_LINES} lines of ${roleLog} (redacted) ====="
            tail -n "$DUMP_TAIL_LINES" -- "$roleLog" 2>/dev/null | sed -E "$DUMP_REDACT_RE" || true
        done
        if [ "$count" -eq 0 ]; then
            echo "No *.log files found under ${testOutputDir}"
        fi
    fi
    echo "===== End of per-role log tails ====="

    if [ "$xtraceWasOn" -eq 1 ]; then set -x; fi
    return 0
}

# A walltime kill (SLURM job state TIMEOUT) sends SIGTERM to this batch script
# before SIGKILL. Two things happen because of this trap, both measured:
#
#  1. Without any TERM handler, bash's default disposition kills the batch shell
#     immediately, so cleanup_on_failure never runs and nothing is dumped at all.
#     Installing a handler is what keeps the shell alive long enough to react.
#  2. bash defers a trap until the running foreground command returns, so the
#     benchmark srun below is started in the background and waited on: `wait` is
#     interruptible, so the dump happens at once rather than whenever srun
#     finally reaps.
#
# The dump has to finish inside SLURM's KillWait grace (default 30s, not
# overridden anywhere under jenkins/) before SIGKILL. Measured at 47ms against
# 16 x 20MB role logs (313MB total) -- `tail -n` seeks from the end, so log size
# is nearly irrelevant. That is ~600x headroom; the dump is not the constraint,
# and the background/`wait` form above removed the thing that was.
#
# No `#SBATCH --signal` is set to widen the window further, because the directive
# lives in the shared script prefix, which also feeds the aggregated and plain
# single-srun branches -- neither installs a TERM handler, so an early SIGTERM
# would kill those jobs outright.
trap 'dump_role_logs "SIGTERM (walltime kill or scancel)" || true' TERM

cleanup_on_failure() {
    echo "Error: $1"
    # Capture the worker evidence *before* scancel tears the allocation down.
    dump_role_logs "$1" || true
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
    # End-of-write sentinel: gen_server_$i.log is the srun's &> aggregate of
    # every gen-worker rank (the per-iter prev_device_step_time lines the
    # benchmark parses live only here, not in trtllm-serve.GEN_*.log). The
    # file descriptor is owned by this srun, so the log is only guaranteed
    # fully flushed once the srun is reaped. Run srun in the foreground of a
    # backgrounded subshell and touch gen_server_$i.done immediately after it
    # returns: the benchmark srun blocks on that sentinel before parsing, so
    # it never reads a truncated / not-yet-flushed log (nvbugs 6487036 /
    # 6487040). A stale sentinel from a re-run output dir is removed first.
    # Note: srun is foreground inside the subshell (not `srun ... &` + a
    # `kill -0` poll) so `touch` runs strictly after reap, with no
    # late-zombie race that could either skip or prematurely fire the signal.
    rm -f "$testOutputDir/gen_server_$i.done"
    (
        srun "${srunArgs[@]}" --mpi=pmix --kill-on-bad-exit=1 \
            -N $nodesPerGenServer \
            -w "${genNodeLists[$i]}" \
            --ntasks=$gen_world_size \
            --ntasks-per-node=$gpusPerNodePerGenServer \
            $runScript &> $testOutputDir/gen_server_$i.log
        touch "$testOutputDir/gen_server_$i.done"
    ) &
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
# Backgrounded and waited on rather than run in the foreground so the TERM trap
# above can fire while the benchmark is still running (bash defers traps until a
# foreground command returns). `wait` returns >128 when a trapped signal
# interrupts it and the child is still alive, so resume waiting in that case --
# otherwise a SIGTERM would make this look like a benchmark failure and scancel a
# job that had not actually failed.
srun "${srunArgs[@]}" --kill-on-bad-exit=1 --overlap \
    -N 1 \
    --ntasks=1 \
    --ntasks-per-node=1 \
    $runScript &
benchmarkPid=$!
benchmarkRc=0
wait "$benchmarkPid" || benchmarkRc=$?
while [ "$benchmarkRc" -gt 128 ] && kill -0 "$benchmarkPid" 2>/dev/null; do
    benchmarkRc=0
    wait "$benchmarkPid" || benchmarkRc=$?
done
if [ "$benchmarkRc" -ne 0 ]; then
    cleanup_on_failure "Benchmark failed (exit ${benchmarkRc}). See slurm-${SLURM_JOB_ID}.out"
fi

echo "Disagg server and benchmark completed successfully"
echo "Total runtime: $SECONDS seconds"
