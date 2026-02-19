#!/bin/bash

set -euo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "Please set SLURM_JOB_ID"
    exit 1
fi

prefix="pyxis_${SLURM_JOB_ID}_"
matches=$(printf '%s\n' "$(srun -N 1 enroot list)" | grep "^${prefix}" || true)

if [ -z "$matches" ]; then
    echo "Error: No container found" >&2
    exit 1
else
    count=$(printf '%s\n' "$matches" | wc -l)
fi

if [ "$count" -gt 1 ]; then
    echo "Error: Multiple containers found" >&2
    while IFS= read -r match; do
        echo "- ${match#"$prefix"}" >&2
    done <<< "$matches"
    exit 1
fi

suffix=${matches#"$prefix"}
echo "Container name: $suffix" >&2
printf '%s\n' "$suffix"
