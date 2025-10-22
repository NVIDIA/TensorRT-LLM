#!/bin/bash

set -euo pipefail

# ACCOUNT=
# PARTITION=
# EXTRA_ARGS="--gres gpu:4"
TIME=${TIME:-01:00:00}

set -x
salloc -A "$ACCOUNT" \
    -p "$PARTITION" \
    -N "$NODES" \
    --segment "$NODES" \
    $EXTRA_ARGS \
    -t "$TIME" \
    --no-shell \
    2>&1 \
    | tee >(cat >&2) \
    | awk '/Granted job allocation/ {print $NF}'
