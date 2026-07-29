# Cache-transceiver precheck gate for the disaggregated perf-sanity launch
# script. submit.py splices this file into the generated launch script ahead
# of slurm_launch_draft.sh, which calls run_cache_transceiver_precheck after
# computing the per-server node slices. Kept as functions in a separate file
# so the gate logic can be sourced and exercised standalone.
#
# Expects the launch-script globals: srunArgs, numGenServers, numCtxServers,
# nodesPerGenServer/nodesPerCtxServer, gpusPerNodePerGenServer/
# gpusPerNodePerCtxServer, genNodeLists/ctxNodeLists, testOutputDir,
# jobWorkspace, pytestCommandCTXPrecheck/pytestCommandGENPrecheck,
# precheckRunScript, ctPrecheckEnabled, ctPrecheckTimeout, stageName,
# and the cleanup_on_failure function.

# Escape text for embedding in XML; junit parsers also choke on raw control
# bytes that MPI/UCX logs may contain, so strip those too.
ct_xml_escape() {
    sed -e 's/&/\&amp;/g' -e 's/</\&lt;/g' -e 's/>/\&gt;/g' -e 's/"/\&quot;/g' \
        | tr -d '\000-\010\013\014\016-\037'
}

# First few root-cause-shaped lines of a step log — the tail alone can miss
# the real error when it happened early and retry spam follows.
ct_first_errors() {
    grep -m 5 -nE "Traceback \(most recent call last\)|MPI_ABORT|MPIR_Err|srun: error|Segmentation fault|CUDA error|RuntimeError|AssertionError|INIT_ERROR|TRANSFER_ERROR" \
        "$1" 2>/dev/null || true
}

# Shared verdict predicate + failing-log excerpt, so the console summary and
# the junit xml cannot drift apart.
ct_step_passed() {
    local statusFile="$precheckDir/status/$1.status"
    [ -f "$statusFile" ] && grep -q "^PASS" "$statusFile"
}

ct_step_excerpt() {
    local stepLog="$precheckDir/logs/$1.log"
    echo "First error lines (line-numbered):"
    ct_first_errors "$stepLog"
    echo "Log tail ($stepLog):"
    tail -n 60 "$stepLog" 2>/dev/null || true
}

# Console summary for a failed precheck: per-instance verdicts, first error
# lines + tail of each failing step log, and UCX red-flag lines.
# Uses: precheckDir, precheckNames.
ct_print_failure_summary() {
    echo "===================================================================="
    echo "CACHE TRANSCEIVER PRECHECK FAILED - the disaggregated test will NOT run"
    echo "Instance verdicts:"
    cat "$precheckDir"/status/*.status 2>/dev/null \
        || echo "(no status files - steps died before transceiver setup; see logs below)"
    echo ""
    echo "Failing step logs:"
    for k in "${!precheckNames[@]}"; do
        ct_step_passed "${precheckNames[$k]}" && continue
        echo "----- ${precheckNames[$k]} -----"
        ct_step_excerpt "${precheckNames[$k]}"
    done
    echo ""
    echo "UCX red flags (host-staged tcp fallback / UCX errors), if any:"
    grep -hE "sw-emul|UCX +(ERROR|WARN)" "$precheckDir"/logs/*.log 2>/dev/null | sort -u | head -20 || true
    echo "Full artifacts: $precheckDir (status/*.json, logs/, csv/)"
    echo "===================================================================="
}

# Synthetic junit result so the failure shows up as a test entry in the
# Jenkins test report: uploadResults scps $jobWorkspace/results*.xml back and
# junit() ingests them. Best-effort — callers must not let this mask the real
# failure path. Uses: precheckDir, precheckNames, jobWorkspace, stageName.
ct_write_junit_xml() {
    local junitXml="$jobWorkspace/results-ct-precheck.xml"
    local suiteName
    suiteName="$(printf '%s' "${stageName:-${SLURM_JOB_NAME:-disagg_perf_sanity}}" | ct_xml_escape)"
    local junitFailures=0
    local junitCases=""
    local name verdict detail
    for k in "${!precheckNames[@]}"; do
        name="${precheckNames[$k]}"
        if ct_step_passed "$name"; then
            junitCases+="<testcase name=\"cache_transceiver_precheck[$name]\" classname=\"$suiteName\" time=\"0\"/>"$'\n'
            continue
        fi
        junitFailures=$((junitFailures + 1))
        verdict="$( (head -n 1 "$precheckDir/status/$name.status" 2>/dev/null \
            || echo "NO_STATUS: step died before writing a verdict (see log)") | ct_xml_escape)"
        detail="$(ct_step_excerpt "$name" | ct_xml_escape)"
        junitCases+="<testcase name=\"cache_transceiver_precheck[$name]\" classname=\"$suiteName\" time=\"0\">"
        junitCases+="<failure message=\"$verdict\">$detail</failure></testcase>"$'\n'
    done
    if {
        echo '<?xml version="1.0" encoding="UTF-8"?>'
        echo "<testsuites><testsuite name=\"$suiteName\" tests=\"${#precheckNames[@]}\" failures=\"$junitFailures\" errors=\"0\" skipped=\"0\" time=\"0\">"
        printf '%s' "$junitCases"
        echo "</testsuite></testsuites>"
    } > "$junitXml" 2>/dev/null; then
        echo "Synthetic junit result written to $junitXml (will appear in the Jenkins test report)"
    else
        echo "WARNING: could not write synthetic junit xml to $junitXml (non-fatal)"
    fi
}

# Run one precheck srun per ctx/gen server instance with the same node
# slices, MPI topology, and UCX env as the real server steps
# (pytestCommand{CTX,GEN}Precheck embed the same ucx_tls_cmd + worker env var
# strings). On failure: console summary + synthetic junit, then the stage
# aborts via cleanup_on_failure — before any model bring-up.
run_cache_transceiver_precheck() {
    if [ "${ctPrecheckEnabled:-0}" != "1" ] || [ "${TRTLLM_DISAGG_BENCHMARK_GEN_ONLY:-0}" = "1" ]; then
        return 0
    fi
    echo "Starting cache transceiver precheck..."
    precheckDir="$testOutputDir/cache_transceiver_precheck"
    mkdir -p "$precheckDir/logs"
    # A reused work dir (Slurm requeue reruns this batch script with the same
    # directories) may hold a previous run's rendezvous/status/csv/abort files:
    # stale addr files would point gen leaders at dead ports, stale status files
    # would pollute the verdict aggregation below, a stale precheck.abort would
    # fail-fast-skip the whole rerun (a requeued job keeps its SLURM_JOB_ID, so
    # the driver's job-id stamp cannot tell it apart), and stale bandwidth CSVs
    # (the Python transceiver's perf_<uuid>_<rank>.csv are per-run and appended)
    # would make parse_python_bandwidth_gbps median over two runs' samples. The
    # driver also job-id-stamps addr files as a second line of defense.
    rm -f "$precheckDir"/rendezvous/*.addr "$precheckDir"/status/*.status \
        "$precheckDir"/status/*.json "$precheckDir"/precheck.abort 2>/dev/null || true
    rm -rf "$precheckDir"/csv 2>/dev/null || true
    precheckPids=()
    precheckNames=()
    # ct_launch_step <role> <idx> <nodes> <gpusPerNode> <nodeList> <pytestCmd>
    ct_launch_step() {
        local role=$1 i=$2 nodes=$3 gpusPerNode=$4 nodeList=$5 pytestCmd=$6
        export DISAGG_SERVING_TYPE="${role^^}_PRECHECK_$i"
        export pytestCommand="$pytestCmd --server-idx $i"
        timeout -k 60 "${ctPrecheckTimeout:-900}" \
            srun "${srunArgs[@]}" --mpi=pmix --kill-on-bad-exit=1 \
            -N "$nodes" \
            -w "$nodeList" \
            --ntasks=$((nodes * gpusPerNode)) \
            --ntasks-per-node="$gpusPerNode" \
            bash $precheckRunScript &> "$precheckDir/logs/${role}_$i.log" &
        precheckPids+=($!)
        precheckNames+=("${role}_$i")
        sleep 5  # Wait for pyxis container namespace initialization to avoid race condition
    }
    local i
    for i in $(seq 0 $((numGenServers - 1))); do
        ct_launch_step gen "$i" "$nodesPerGenServer" "$gpusPerNodePerGenServer" \
            "${genNodeLists[$i]}" "$pytestCommandGENPrecheck"
    done
    for i in $(seq 0 $((numCtxServers - 1))); do
        ct_launch_step ctx "$i" "$nodesPerCtxServer" "$gpusPerNodePerCtxServer" \
            "${ctxNodeLists[$i]}" "$pytestCommandCTXPrecheck"
    done

    local precheckFailed=0 k rc
    for k in "${!precheckPids[@]}"; do
        if wait "${precheckPids[$k]}"; then
            echo "Precheck step ${precheckNames[$k]} passed"
        else
            rc=$?
            echo "Precheck step ${precheckNames[$k]} FAILED (exit $rc; 124 = external timeout)"
            precheckFailed=1
        fi
    done

    if [ "$precheckFailed" -eq 1 ]; then
        ct_print_failure_summary
        ct_write_junit_xml
        cleanup_on_failure "Cache transceiver precheck failed. See summary above and $precheckDir"
    fi
    # No status files means every step took the skip path (e.g. the yaml has no
    # cache_transceiver_config.backend): the run wrote no verdicts, so report it
    # as SKIPPED rather than an empty "PASSED" that reads like real validation.
    if ls "$precheckDir"/status/*.status >/dev/null 2>&1; then
        echo "Cache transceiver precheck PASSED:"
        cat "$precheckDir"/status/*.status 2>/dev/null || true
    else
        echo "Cache transceiver precheck SKIPPED: not applicable for this config" \
            "(no verdicts written; see $precheckDir/logs)"
    fi
}
