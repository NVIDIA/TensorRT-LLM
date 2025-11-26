#!/bin/bash

set -euo pipefail

prefix="pyxis_${SLURM_JOB_ID}_"
matches=$(printf "%s\n" "$(srun -N 1 enroot list)" | grep "^${prefix}" || true)
count=$(printf "%s\n" "$matches" | wc -l)

if [ "$count" -eq 0 ]; then
    echo "Error: No container found" >&2
    exit 1
fi

if [ "$count" -gt 1 ]; then
    echo "Error: Multiple containers found" >&2
    while IFS= read -r match; do
        echo "- ${match#$prefix}" >&2
    done <<< "$matches"
    exit 1
fi

suffix=${matches#$prefix}
echo "Container name: $suffix" >&2
echo "$suffix"
