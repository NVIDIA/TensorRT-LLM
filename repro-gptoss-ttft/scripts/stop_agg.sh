#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Stop the backgrounded aggregated trtllm-serve started by launch_agg.sh.
set -uo pipefail

cd "$(dirname "$0")/.."
PID_DIR="logs/pids"
pidfile="${PID_DIR}/agg.pid"

if [[ ! -f "${pidfile}" ]]; then
  echo "no pidfile at ${pidfile} (already stopped?)"
  exit 0
fi

pid="$(cat "${pidfile}")"
if ! kill -0 "${pid}" 2>/dev/null; then
  echo "agg: PID ${pid} not running"
  rm -f "${pidfile}"
  exit 0
fi

echo "Killing agg (PID ${pid})"
kill "${pid}" || true
for _ in 1 2 3 4 5 6 7 8 9 10; do
  if kill -0 "${pid}" 2>/dev/null; then sleep 1; else break; fi
done
if kill -0 "${pid}" 2>/dev/null; then
  echo "Force-killing agg (PID ${pid})"
  kill -9 "${pid}" || true
fi
rm -f "${pidfile}"
