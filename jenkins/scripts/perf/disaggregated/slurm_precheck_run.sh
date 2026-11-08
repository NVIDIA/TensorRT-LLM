#!/bin/bash

# Per-rank container entrypoint for the disagg cache-transceiver PRECHECK
# step (launched by slurm_launch_draft.sh BEFORE the real ctx/gen servers).
#
# It intentionally mirrors slurm_run.sh's runtime environment: the same
# slurm_env_setup.sh is sourced (LD_LIBRARY_PATH, the `unset UCX_TLS=tcp`
# fixup, PMIX_MCA_gds), and $pytestCommand carries the same
# `unset/export UCX_TLS ...` prefix and worker env vars as the real ctx/gen
# worker steps (built from the same strings by jenkins/scripts/perf/submit.py),
# so a precheck PASS/FAIL is representative of the network environment the
# real test will run in. Unlike slurm_run.sh it skips coverage/perf-report
# handling -- the precheck is a plain MPI program, not a pytest run.

set -xEeuo pipefail
trap 'rc=$?; echo "Error in file ${BASH_SOURCE[0]} on line $LINENO: $BASH_COMMAND (exit $rc)"; exit $rc' ERR

# CI (jenkins/L0_Test.groovy) exports resourcePathNode with the /tmp
# extraction layout; the local flow (jenkins/scripts/perf/local/submit.py)
# exports llmSrcNode pointing at the repo directly and has no
# resourcePathNode. Honor whichever layout is present.
if [ -n "${resourcePathNode:-}" ]; then
    cd "$resourcePathNode"
    llmSrcNode=$resourcePathNode/TensorRT-LLM/src
fi
: "${llmSrcNode:?either resourcePathNode or llmSrcNode must be exported}"

source "$llmSrcNode/jenkins/scripts/slurm_env_setup.sh"
slurm_setup_runtime_env

echo "Precheck rank ${SLURM_PROCID:-?} (${DISAGG_SERVING_TYPE:-unknown}) command: $pytestCommand"

set +e
eval $pytestCommand
precheck_exit_code=$?
set -e

echo "Rank${SLURM_PROCID:-?} cache-transceiver precheck finished with exit code $precheck_exit_code"
exit $precheck_exit_code
