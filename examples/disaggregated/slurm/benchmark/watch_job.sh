#!/bin/bash
# Poll a disaggregated benchmark log directory and print one EVENT line per
# state change. Intended for tracking a running job without tailing megabytes
# of worker log.
#
#   bash watch_job.sh <log_dir> [poll_seconds] [max_minutes]
set -uo pipefail

log_dir="${1:?usage: watch_job.sh <log_dir> [poll_seconds] [max_minutes]}"
poll="${2:-20}"
max_minutes="${3:-60}"
deadline=$(( SECONDS + max_minutes * 60 ))

declare -A seen

announce() {
    key="$1"; shift
    if [ -z "${seen[$key]:-}" ]; then
        seen[$key]=1
        echo "EVENT: $*"
    fi
}

count_matches() {
    grep -h "$1" "${log_dir}"/3_output_CTX_*.log 2>/dev/null | wc -l || true
}

# How much of the pool's contents lives on each node. A segment is one client
# process's donated memory, so a single host here means the pool is
# prefill-only: the donors are absent or not being allocated into.
placement() {
    grep -o "allocation_succeeded size=[0-9]* segment=[0-9.]*:[0-9]*" \
        "${log_dir}/2_mooncake_master.log" 2>/dev/null \
        | awk '{sub(/size=/,"",$2); sub(/segment=/,"",$3); split($3,p,":");
                n[p[1]]++; b[p[1]]+=$2; t+=$2}
               END {if (t == 0) {print "(none yet)"; exit}
                    for (h in n) printf "%s:%d pages/%.2fGiB/%.0f%% ", h, n[h], b[h]/1073741824, 100*b[h]/t}'
}

while [ ${SECONDS} -lt ${deadline} ]; do
    for ready in "${log_dir}"/mooncake_donor_*.ready; do
        [ -s "${ready}" ] || continue
        node="$(basename "${ready}" .ready)"; node="${node#mooncake_donor_}"
        announce "donor_${node}" "memory donor on ${node} mounted its segment:" \
            "$(cat "${ready}")"
    done

    for role in CTX GEN; do
        f="${log_dir}/3_output_${role}_0.log"
        [ -f "$f" ] || continue
        grep -qs "Server started at\|Application startup complete\|Uvicorn running" "$f" \
            && announce "${role}_up" "${role} worker serving"
    done

    if grep -qs "registered layout" "${log_dir}"/3_output_CTX_*.log; then
        announce "registered" "connector registered KV layout:" \
            "$(grep -h "registered layout" "${log_dir}"/3_output_CTX_*.log 2>/dev/null | head -n 1 | cut -c1-400)"
    fi

    matched=$(count_matches "mooncake-store matched")
    loaded=$(count_matches "mooncake-store rank")
    [ "${matched:-0}" -gt 0 ] && announce "first_match" "first store hit; matched lines=${matched}"
    [ "${loaded:-0}" -gt 0 ] && announce "first_load" "first store load; loaded lines=${loaded}"

    # A second host is the first moment prefill-written KV is provably on
    # decode DRAM, which is what the donors exist for.
    hosts=$(grep -o "segment=[0-9.]*:" "${log_dir}/2_mooncake_master.log" 2>/dev/null \
        | sort -u | wc -l || true)
    [ "${hosts:-0}" -ge 2 ] && announce "multi_host" \
        "pool spans ${hosts} hosts; placement: $(placement)"

    for pattern in "failed to load" "failed to save" "lookup failed" "background save failed"; do
        if grep -qs "mooncake-store.*${pattern}" "${log_dir}"/3_output_CTX_*.log; then
            announce "fail_${pattern// /_}" "PROBLEM: mooncake-store ${pattern}"
        fi
    done

    [ -f "${log_dir}/6_bench.log" ] && announce "bench_started" "benchmark client started"
    ls "${log_dir}"/concurrency_*/result.json >/dev/null 2>&1 \
        && announce "result" "result.json written"

    if ls "${log_dir}"/8_done_*.txt >/dev/null 2>&1; then
        echo "EVENT: job finished (8_done marker present)"
        echo "FINAL: matched=${matched:-0} loaded=${loaded:-0}"
        echo "FINAL placement: $(placement)"
        exit 0
    fi

    # A dead batch script leaves the tree untouched, so report it rather than
    # polling until the deadline.
    if grep -qs "Job completed successfully" "${log_dir}"/slurm-*.out; then
        echo "EVENT: batch script reported completion"
        exit 0
    fi
    if grep -qs "^Error: " "${log_dir}"/slurm-*.out; then
        echo "EVENT: PROBLEM: batch script hit cleanup_on_failure"
        grep -h "^Error: " "${log_dir}"/slurm-*.out | tail -n 3
        exit 1
    fi

    sleep "${poll}"
done

echo "EVENT: watcher deadline reached after ${max_minutes} minutes"
echo "FINAL: matched=$(count_matches 'mooncake-store matched') loaded=$(count_matches 'mooncake-store rank')"
