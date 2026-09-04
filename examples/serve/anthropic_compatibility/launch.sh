#!/usr/bin/env bash
# launch.sh - bring up trtllm-serve plus the gateway in one command.
#
#   ./launch.sh <model>                       aggregated
#   ./launch.sh --disagg <model>              1 context + 1 generation worker
#   ./launch.sh --no-gateway <model>          server only
#
# On success it prints the URL to point Claude Code at, and keeps running until
# interrupted. Everything it starts is torn down on exit.
#
# The gateway is included by default because it is what makes the endpoint
# stable: a client bakes in the URL it was started with, so replacing a server
# behind a fixed address is the difference between a restart being invisible
# and a restart ending the session. Use --no-gateway when a single short-lived
# server is all you need.
set -uo pipefail

usage() {
    sed -n '2,15p' "$0" | sed 's/^# \{0,1\}//'
    exit "${1:-0}"
}

MODEL=""
MODE="agg"
WITH_GATEWAY=1
PORT="${PORT:-8000}"
GATEWAY_PORT="${GATEWAY_PORT:-8333}"
CTX_PORT="${CTX_PORT:-8101}"
GEN_PORT="${GEN_PORT:-8102}"
# The username IS the key (see deployments/gateway_users.txt), so the caller is
# allowlisted by default and nobody else is.
GATEWAY_USER="${GATEWAY_USER:-${USER:-local}}"
RUN_DIR="${RUN_DIR:-$(mktemp -d "${TMPDIR:-/tmp}/trtllm-anthropic.XXXXXX")}"
EXTRA_SERVE_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --disagg)       MODE="disagg"; shift ;;
        --agg)          MODE="agg"; shift ;;
        --no-gateway)   WITH_GATEWAY=0; shift ;;
        --port)         PORT="$2"; shift 2 ;;
        --gateway-port) GATEWAY_PORT="$2"; shift 2 ;;
        -h|--help)      usage 0 ;;
        --)             shift; EXTRA_SERVE_ARGS=("$@"); break ;;
        -*)             echo "unknown option: $1" >&2; usage 2 ;;
        *)              if [[ -z "${MODEL}" ]]; then MODEL="$1"; shift
                        else EXTRA_SERVE_ARGS+=("$1"); shift; fi ;;
    esac
done

[[ -n "${MODEL}" ]] || { echo "error: no model given" >&2; usage 2; }
command -v trtllm-serve >/dev/null || {
    echo "error: trtllm-serve is not on PATH; install TensorRT-LLM first" >&2; exit 1; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOSTNAME_FQDN="$(hostname -f 2>/dev/null || hostname)"
FLEET_DIR="${RUN_DIR}/fleet"
mkdir -p "${FLEET_DIR}"

PIDS=()
cleanup() {
    trap - EXIT INT TERM   # a second Ctrl-C must not re-enter this
    # Reverse order: drop the registration first so the gateway stops routing
    # to a server that is already going away, rather than discovering it by
    # timeout after the next heartbeat is missed.
    rm -f "${FLEET_DIR}"/*.json 2>/dev/null
    for pid in "${PIDS[@]:-}"; do
        [[ -n "${pid}" ]] && kill "${pid}" 2>/dev/null
    done
    # Bounded, then forced. A worker that declines to exit would otherwise hold
    # its GPU memory indefinitely, and an unbounded `wait` here would hang the
    # Ctrl-C that was meant to release it.
    for _ in $(seq 1 20); do
        local alive=0
        for pid in "${PIDS[@]:-}"; do
            [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null && alive=1
        done
        (( alive )) || break
        sleep 1
    done
    for pid in "${PIDS[@]:-}"; do
        [[ -n "${pid}" ]] && kill -9 "${pid}" 2>/dev/null
    done
    echo "[launch] stopped; logs kept in ${RUN_DIR}"
}
trap cleanup EXIT INT TERM

log() { printf '[launch] %s\n' "$*"; }

wait_for_health() {  # wait_for_health <url> <label> <timeout_s>
    local url="$1" label="$2" timeout="$3" waited=0
    while (( waited < timeout )); do
        if curl -sf -o /dev/null --connect-timeout 2 "${url}/health"; then
            log "${label} ready after ${waited}s"
            return 0
        fi
        # A dead process will never answer, so stop waiting on it rather than
        # burning the whole timeout: weights take minutes, a crash takes none.
        if [[ -n "${WAIT_PID:-}" ]] && ! kill -0 "${WAIT_PID}" 2>/dev/null; then
            log "FATAL: ${label} exited during startup; see ${RUN_DIR}/${label}.log"
            tail -30 "${RUN_DIR}/${label}.log" >&2 2>/dev/null
            return 1
        fi
        sleep 5
        waited=$(( waited + 5 ))
    done
    log "FATAL: ${label} did not become healthy within ${timeout}s"
    return 1
}

start_server() {  # start_server <label> <port> [extra args...]
    local label="$1" port="$2"; shift 2
    log "starting ${label} on port ${port}"
    trtllm-serve "${MODEL}" --host 0.0.0.0 --port "${port}" \
        "$@" "${EXTRA_SERVE_ARGS[@]+"${EXTRA_SERVE_ARGS[@]}"}" \
        > "${RUN_DIR}/${label}.log" 2>&1 &
    PIDS+=($!)
    WAIT_PID=$! wait_for_health "http://127.0.0.1:${port}" "${label}" "${STARTUP_TIMEOUT:-1800}"
}

# -- serving -----------------------------------------------------------------
if [[ "${MODE}" == "agg" ]]; then
    start_server server "${PORT}" || exit 1
    SERVER_URL="http://${HOSTNAME_FQDN}:${PORT}"
else
    # Both workers hold a full copy of the weights, so a single-GPU box cannot
    # run this mode; CUDA_VISIBLE_DEVICES splits them across two devices.
    CUDA_VISIBLE_DEVICES="${CTX_GPU:-0}" start_server ctx "${CTX_PORT}" || exit 1
    CUDA_VISIBLE_DEVICES="${GEN_GPU:-1}" start_server gen "${GEN_PORT}" || exit 1

    cat > "${RUN_DIR}/disagg.yaml" <<EOF
hostname: 0.0.0.0
port: ${PORT}
model: ${MODEL}
backend: "pytorch"
context_servers:
  num_instances: 1
  urls: ["127.0.0.1:${CTX_PORT}"]
generation_servers:
  num_instances: 1
  urls: ["127.0.0.1:${GEN_PORT}"]
EOF
    log "starting disaggregated server on port ${PORT}"
    trtllm-serve disaggregated -c "${RUN_DIR}/disagg.yaml" \
        > "${RUN_DIR}/disagg.log" 2>&1 &
    PIDS+=($!)
    WAIT_PID=$! wait_for_health "http://127.0.0.1:${PORT}" disagg 300 || exit 1
    SERVER_URL="http://${HOSTNAME_FQDN}:${PORT}"
fi

# -- gateway -----------------------------------------------------------------
if (( WITH_GATEWAY )); then
    echo "${GATEWAY_USER}" > "${RUN_DIR}/users.txt"

    # The gateway routes on these files rather than on anything it is told at
    # startup, which is what lets a server be replaced without restarting it.
    # A heartbeat that stops is how it learns a server is gone.
    (
        while true; do
            printf '{"job_id":"local","url":"%s","state":"serving","end_time":0,"heartbeat":%s}\n' \
                "${SERVER_URL}" "$(date +%s)" > "${FLEET_DIR}/local.json.tmp"
            mv "${FLEET_DIR}/local.json.tmp" "${FLEET_DIR}/local.json"
            sleep 10
        done
    ) &
    PIDS+=($!)

    log "starting gateway on port ${GATEWAY_PORT}"
    # --no-relay: relay submits successor jobs through a scheduler, which a
    # local run has no notion of.
    python3 "${HERE}/gateway.py" --fleet-dir "${FLEET_DIR}" \
        --users "${RUN_DIR}/users.txt" --port "${GATEWAY_PORT}" --no-relay \
        > "${RUN_DIR}/gateway.log" 2>&1 &
    PIDS+=($!)

    CLIENT_URL="http://${HOSTNAME_FQDN}:${GATEWAY_PORT}"
    for _ in $(seq 1 30); do
        # /_gateway/health is unauthenticated and reports whether a backend has
        # been elected; /health would be proxied to the server and rejected.
        if curl -sf "${CLIENT_URL}/_gateway/health" 2>/dev/null | grep -q '"status": *"ok"'; then
            break
        fi
        sleep 2
    done
    curl -sf "${CLIENT_URL}/_gateway/health" 2>/dev/null | grep -q '"status": *"ok"' || {
        log "FATAL: gateway did not elect a backend; see ${RUN_DIR}/gateway.log"
        exit 1
    }
    AUTH_NOTE="ANTHROPIC_AUTH_TOKEN=${GATEWAY_USER}"
else
    CLIENT_URL="${SERVER_URL}"
    AUTH_NOTE="ANTHROPIC_AUTH_TOKEN=dummy   # the server does not authenticate"
fi

MODEL_ID="$(basename "${MODEL}")"

cat <<EOF

  ready - ${MODE}$( ((WITH_GATEWAY)) && echo " + gateway")

  Point Claude Code at it:

    ANTHROPIC_BASE_URL=${CLIENT_URL} \\
    ${AUTH_NOTE} \\
    ANTHROPIC_MODEL=${MODEL_ID} \\
      claude

  Or send a request directly:

    curl ${CLIENT_URL}/v1/messages \\
      -H 'content-type: application/json' -H 'anthropic-version: 2023-06-01' \\
      -H 'x-api-key: ${GATEWAY_USER}' \\
      -d '{"model":"${MODEL_ID}","max_tokens":64,
           "messages":[{"role":"user","content":"hello"}]}'

  Logs: ${RUN_DIR}
  Stop: Ctrl-C

EOF

wait
