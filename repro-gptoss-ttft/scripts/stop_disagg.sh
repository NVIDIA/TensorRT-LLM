#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Stop the disagg trtllm-serve processes started by launch_disagg.sh.
set -uo pipefail

cd "$(dirname "$0")/.."
PID_DIR="logs/pids"

for name in proxy gen ctx; do
  pidfile="${PID_DIR}/${name}.pid"
  if [[ -f "${pidfile}" ]]; then
    pid="$(cat "${pidfile}")"
    if kill -0 "${pid}" 2>/dev/null; then
      echo "Killing ${name} (PID ${pid})"
      kill "${pid}" || true
      for _ in 1 2 3 4 5 6 7 8 9 10; do
        if kill -0 "${pid}" 2>/dev/null; then sleep 1; else break; fi
      done
      if kill -0 "${pid}" 2>/dev/null; then
        echo "Force-killing ${name} (PID ${pid})"
        kill -9 "${pid}" || true
      fi
    else
      echo "${name}: PID ${pid} not running"
    fi
    rm -f "${pidfile}"
  else
    echo "${name}: no pidfile (already stopped?)"
  fi
done
