#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Run the measurement cases whose name matches $1 (a grep -E pattern; default all), in parallel.
export PIKIT_MEASURE_DIR=$(mkdir -p "${PIKIT_MEASURE_DIR:-$PWD/pikit-measure}/runs" && cd "${PIKIT_MEASURE_DIR:-$PWD/pikit-measure}" && pwd)
cd "$(dirname "$0")"
pat=${1:-.}
while IFS=$'\t' read -r name api server mode envs args; do
  case "$name" in ''|'#'*) continue;; esac
  echo "$name" | grep -Eq "$pat" || continue
  ( env_args=(); IFS=',' read -ra kv <<< "$envs"; for x in "${kv[@]}"; do [ "$x" = "-" ] || env_args+=("$x"); done
    eval "set -- $args"   # the args column is shell-quoted (JSON in single quotes)
    env "${env_args[@]}" ./probe.sh "$name" "$api" "$server" "$mode" "$@" > "$PIKIT_MEASURE_DIR/runs/$name.out" 2>&1 ) &
done < cases.tsv
wait
