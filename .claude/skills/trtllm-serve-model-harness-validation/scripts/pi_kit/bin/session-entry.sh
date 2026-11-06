#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Entry point of one pi kit session container (docker --network none, no home directory, no
# credentials). Mounts: /workspace (the task), /pi (pi's agent dir and session log), /io (the
# prompt, the endpoint socket and this session's output), /opt/node, /opt/pi and /opt/pikit
# read-only. Everything else comes from PIKIT_* variables set by pikit.py.
#
# Output in /io: events.jsonl (pi --mode json), stderr.log, entry.log, pi_exit (pi's exit code,
# 124 when the in-container time limit cut it).
set -u
exec 3>>/io/entry.log
log() { printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" >&3; }

mkdir -p "$HOME"
node /opt/pikit/inbridge.mjs 127.0.0.1 "$PIKIT_PORT" /io/ep.sock "$HOME/inbridge.ready" 2>&3 &
bridge=$!
for _ in $(seq 100); do
  [ -s "$HOME/inbridge.ready" ] && break
  sleep 0.1
done
if [ ! -s "$HOME/inbridge.ready" ]; then
  log "endpoint bridge did not start"
  echo 125 > /io/pi_exit
  exit 125
fi
log "bridge 127.0.0.1:$PIKIT_PORT -> /io/ep.sock; pi starts (limit ${PIKIT_TIMEOUT_S}s)"

args=(-p --mode json --provider "$PIKIT_PROVIDER" --model "$PIKIT_MODEL" --tools "$PIKIT_TOOLS"
      --no-extensions --no-mcp --no-skills --no-prompt-templates --no-context-files --no-themes
      --offline --no-approve --session-dir /pi/sessions)
[ -n "${PIKIT_THINKING:-}" ] && args+=(--thinking "$PIKIT_THINKING")

cd /workspace || exit 125
timeout --signal=TERM --kill-after=20 "$PIKIT_TIMEOUT_S" \
  node /opt/pi/node_modules/@earendil-works/pi-coding-agent/dist/bundle/cli.js "${args[@]}" \
  -- "$(cat /io/prompt.txt)" > /io/events.jsonl 2> /io/stderr.log < /dev/null &
pi=$!
# docker stop sends TERM: pass it on to pi (through timeout), then collect its real exit code.
trap 'log "TERM received: stopping pi"; kill -TERM "$pi" 2>/dev/null' TERM INT
wait "$pi"
rc=$?
while kill -0 "$pi" 2>/dev/null; do
  wait "$pi"
  rc=$?
done
echo "$rc" > /io/pi_exit
log "pi exited with $rc"
kill "$bridge" 2>/dev/null
exit "$rc"
