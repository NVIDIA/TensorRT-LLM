#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Start or stop a skill fake server on this host for kit tests; PIDs are recorded, never pattern-killed.
# usage: fake_server.sh start <name> <agent|responses> <port> <mode> [fake-server args...]
#        fake_server.sh stop <name>
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
KIT=$(dirname "$HERE")
SKILL=$(dirname "$(dirname "$KIT")")
PY=${PIKIT_PYTHON:-python3}
SERVERS=${PIKIT_WORK:-$PWD/pikit-selftest}/servers
mkdir -p "$SERVERS"
cmd=$1; name=$2
pidfile="$SERVERS/$name.pid"
case "$cmd" in
  start)
    kind=$3; port=$4; mode=$5; shift 5
    script=$SKILL/scripts/validate/fake_agent_server.py
    [ "$kind" = responses ] && script=$SKILL/scripts/validate/fake_responses_server.py
    setsid nohup "$PY" "$script" --host 127.0.0.1 --port "$port" --mode "$mode" "$@" \
      > "$SERVERS/$name.log" 2>&1 < /dev/null &
    echo "$! $port $script" > "$pidfile"
    for _ in $(seq 50); do
      (exec 3<>"/dev/tcp/127.0.0.1/$port") 2>/dev/null && { echo "started $name pid $! port $port"; exit 0; }
      sleep 0.2
    done
    echo "fake server $name did not listen"; exit 1 ;;
  stop)
    [ -f "$pidfile" ] || { echo "no $pidfile"; exit 0; }
    read -r pid port script < "$pidfile"
    if tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -qF -- "--port $port"; then
      kill "$pid" && echo "stopped $name pid $pid"
    else
      echo "$name pid $pid is not running (or is not ours)"
    fi
    rm -f "$pidfile" ;;
esac
