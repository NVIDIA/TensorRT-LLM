#!/bin/bash

set -euo pipefail

# ACCOUNT=
# PARTITION=
# GPUS_PER_NODE=4
TIME=${TIME:-01:00:00}

set -x
salloc -A "$ACCOUNT" \
    -p "$PARTITION" \
    -N "$NODES" \
    --segment "$NODES" \
    --gres "gpu:$GPUS_PER_NODE" \
    -t "$TIME" \
    --no-shell \
    2>&1 \
    | tee >(cat >&2) \
    | awk '/Granted job allocation/ {print $NF}'
