#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# One pi 1.4.0 session (default settings) against a fake server, record_requests.py in front
# (NOREC=1: none). Output: runs/<name>/{trace,events.jsonl,stderr.log,server.log,recorder.log}.
# usage: probe.sh <name> <api> <server: agent|responses> <mode> [extra fake-server args...]
set -u
NAME=$1; API=$2; SERVER=$3; MODE=$4; shift 4
HERE=$(cd "$(dirname "$0")" && pwd)
SKILL=$(dirname "$(dirname "$(dirname "$HERE")")")
VPY=${PIKIT_PYTHON:-python3}
NODE=${PIKIT_NODE_DIR:?set PIKIT_NODE_DIR}/bin/node
PI=${PIKIT_PI_DIR:?set PIKIT_PI_DIR}/node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js
OUT=${PIKIT_MEASURE_DIR:-$PWD/pikit-measure}/runs/$NAME
rm -rf "$OUT"; mkdir -p "$OUT"/{home,agent,sessions,ws,trace}
free_port() { $VPY -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1])'; }
SP=$(free_port); RP=$(free_port)
if [ "$SERVER" = responses ]; then
  $VPY $SKILL/scripts/validate/fake_responses_server.py --port $SP --mode $MODE "$@" > "$OUT/server.log" 2>&1 &
else
  $VPY $SKILL/scripts/validate/fake_agent_server.py --port $SP --mode $MODE "$@" > "$OUT/server.log" 2>&1 &
fi
SPID=$!
$VPY $SKILL/scripts/record_requests.py --out-dir "$OUT/trace" --port $RP --forward http://127.0.0.1:$SP --session probe-$NAME > "$OUT/recorder.log" 2>&1 &
RPID=$!
sleep 1.5
FP=$RP; [ -n "${NOREC:-}" ] && FP=$SP   # NOREC=1: pi talks to the fake server directly
BASE=http://127.0.0.1:$FP/v1; [ "$API" = anthropic-messages ] && BASE=http://127.0.0.1:$FP
cat > "$OUT/agent/models.json" <<JSON
{"providers": {"trtllm": {"baseUrl": "$BASE", "api": "$API", "apiKey": "dummy-local-key",
  "models": [{"id": "fake-model", "name": "fake", "reasoning": ${REASONING:-true}, "input": ["text"],
              "contextWindow": ${CTX:-32768}, "maxTokens": 4096,
              "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0},
              "compat": ${COMPAT:-{\}}}]}}}
JSON
echo "hello" > "$OUT/ws/notes.txt"
if [ -n "${BIG:-}" ]; then python3 -c "
import sys
with open(sys.argv[1], 'w') as f:
    for i in range(3000): f.write(f'line {i:05d} the quick brown fox jumps over\\n')" "$OUT/ws/big.txt"; fi
start=$(date +%s%N)
(cd "$OUT/ws" && timeout ${TIMEOUT_S:-120} env -i PATH=/usr/local/bin:/usr/bin:/bin HOME="$OUT/home" \
  PI_CODING_AGENT_DIR="$OUT/agent" PI_OFFLINE=1 PI_SKIP_VERSION_CHECK=1 PI_TELEMETRY=0 \
  $NODE $PI -p --mode json --provider trtllm --model fake-model --tools ${TOOLS:-read,bash,edit,write} \
  --no-extensions --no-mcp --no-skills --no-prompt-templates --no-context-files --no-themes --offline \
  --session-dir "$OUT/sessions" -- "${PROMPT:-Read notes.txt and tell me what it says.}" > "$OUT/events.jsonl" 2> "$OUT/stderr.log" < /dev/null)
rc=$?
end=$(date +%s%N)
sleep 0.5
kill $RPID $SPID 2>/dev/null; wait $RPID $SPID 2>/dev/null
echo "rc=$rc elapsed_ms=$(( (end - start) / 1000000 ))"
