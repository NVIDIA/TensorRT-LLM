#!/bin/bash

set -euo pipefail

# ACCOUNT=
# PARTITION=
# GPUS_PER_NODE=4

set -x
salloc -A "$ACCOUNT" \
    -p "$PARTITION" \
    --gres "gpu:$GPUS_PER_NODE" \
    -t 01:00:00 \
    -N "$NODES" \
    --segment "$NODES" \
    --no-shell \
    2>&1 \
    | tee >(cat >&2) \
    | awk '/Granted job allocation/ {print $NF}'
