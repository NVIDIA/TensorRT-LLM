#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# The in-container half of a staircase-v41 entry job. Invoked by both paths:
#
#   staircase_v41_entry.sbatch   -> sbatch, one fresh allocation per command
#   staircase_v41_run.sh         -> srun into a persistent allocation when one
#                                   is up, falling back to the sbatch path
#
# Split out of staircase_v41_entry.sbatch so the two callers share one body
# rather than two copies that drift. It also removes that file's documented
# hazard: the body used to be a single-quoted string passed to `bash -c`, so a
# literal apostrophe or a quoted heredoc delimiter ended it early and the job
# died with "unexpected end of file" before running anything. As a real file
# there is nothing to escape.
#
#   bash staircase_v41_body.sh test   unittest/_torch/staircase/gemm/test_staircase_mxfp8_mxfp8_gemm.py
#   bash staircase_v41_body.sh probe  staircase_v41_mxfp8_guard_probe.py
#   bash staircase_v41_body.sh lint   "<space separated files>"
#   bash staircase_v41_body.sh fmt    "<space separated files>"
#   bash staircase_v41_body.sh verify "<files>" "<probes|->" "<test-targets|->"
#
# The probe and test arguments each take a space-separated LIST, so one job can
# re-earn two entries' receipts when a change touches a file they share.
#
# `verify` runs fmt -> lint -> probe -> test in that order in ONE job. Two
# reasons, and the second is the load-bearing one:
#
#   1. Four separate jobs cost four container starts and four
#      `import tensorrt_llm`s. Measured on this cluster: a `lint` job whose
#      actual work is seconds has a 97s median, so ~90s of every job is fixed
#      overhead.
#   2. Receipt freshness stops being a thing to check. A receipt is valid only
#      if it post-dates the last write to every file in the entry, and
#      `ruff format` rewrites the test file -- so the ordering fmt-before-test
#      is exactly what makes the receipt honest. Inside one job the order is
#      structural instead of something the Reviewer has to verify against
#      mtimes afterwards.
#
# Pass `-` for a step to skip it. Every requested step runs even if an earlier
# one failed, so one round trip reports everything; the exit code is the first
# non-zero.

set -uo pipefail

MODE="${1:-test}"
ARG1="${2:-}"
ARG2="${3:-}"
ARG3="${4:-}"

: "${REPO:=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/fredw/TensorRT-LLM}"
: "${WORK:=/scratch/fsw/portfolios/coreai/projects/coreai_comparch_trtllm/users/fredw/staircase-v41}"

cd "$REPO" || exit 2
export REPO WORK
export VENV=$REPO/.venv-3.12
export PATH="$VENV/bin:$WORK/lintdeps/bin:$PATH"
export PYTHONPATH=$REPO:${PYTHONPATH:-}
export TLLM_LOG_LEVEL=WARNING

# --container-mount-home puts the login home inside the container, and this
# home carries a user-site editable install (agent-flow) whose .pth file
# appends a path hook rooted under the repo. That leaks into sys.path of every
# python3 here, and tests/unittest/pytest.ini loads
# test_common.magic_import_hooks which correctly fails the session over it --
# reported as
#   Unexpected sys.path entries under the project root:
#   tests/__editable__.agent_flow-0.1.0.finder.__path_hook__
# with pytest exiting 4 while every test passes. The plugin is right; the
# environment was dirty. Disabling user site removes the leak, so the verdict
# below can be pytest's own exit code instead of a parsed summary line, and no
# repo plugin is suppressed to get it.
export PYTHONNOUSERSITE=1

# Exported BEFORE the first TensorRT-LLM import, in the same shell that starts
# the work, so the value the process reads is the value this job claims.
export TRTLLM_STAIRCASE=require

# The work runs on GPU 0. A persistent allocation keeps its idle-watchdog
# keepalive on GPUs 1-3 precisely so nothing shares a device with the
# measurement: the autotuner picks tactics by timing, and a concurrent kernel
# on the same device could change which tactic wins -- which is the one thing
# a receipt must not depend on.
export CUDA_VISIBLE_DEVICES=0

echo "node=$(hostname) mode=$MODE target=$ARG1 ${ARG2:+probe=$ARG2} ${ARG3:+test=$ARG3}"
nvidia-smi --query-gpu=name,compute_cap,memory.total --format=csv,noheader | head -1

python3 - <<'PYBOOT'
import os, sys, torch, tensorrt_llm
from tensorrt_llm._torch.staircase._router_index import StaircaseMode
print("trtllm", tensorrt_llm.__version__, "from", tensorrt_llm.__file__)
print("capability", torch.cuda.get_device_capability())
print("TRTLLM_STAIRCASE", os.environ.get("TRTLLM_STAIRCASE"), "->", StaircaseMode.from_env())
assert StaircaseMode.from_env() is StaircaseMode.REQUIRE, "require mode must be set before import"
assert tensorrt_llm.__file__.startswith(os.environ["REPO"]), "the checkout must be what imports"
leaked = [p for p in sys.path if "agent_flow" in p or "__editable__" in p]
assert not leaked, f"user-site leak still on sys.path: {leaked}"
assert sys.flags.no_user_site, "PYTHONNOUSERSITE must be in effect"
print("sys.path clean, no_user_site", sys.flags.no_user_site)
PYBOOT
boot=$?
if [ $boot -ne 0 ]; then echo "---- entry bootstrap failed $boot ----"; exit $boot; fi

step_fmt() {
    local files="$1" rc=0
    if [ -z "$files" ] || [ "$files" = "-" ]; then
        echo "== fmt: skipped"
        return 0
    fi
    echo "== fmt =="
    ruff format $files                || rc=$?
    ruff check --select I --fix $files || rc=$?
    echo "fmt: ruff format / ruff check --select I --fix -> $rc"
    return $rc
}

step_lint() {
    local files="$1" rc=0
    if [ -z "$files" ] || [ "$files" = "-" ]; then
        echo "== lint: skipped"
        return 0
    fi
    echo "== lint =="
    # Non-mutating on purpose: rerunning `lint` is a verification, not a
    # repair. `fmt` is what makes it pass in the first place.
    ruff format --check $files   || rc=$?
    ruff check --select I $files || rc=$?
    ruff check $files            || rc=$?
    ty check $files              || rc=$?
    echo "lint: ruff format --check / ruff check --select I / ruff check / ty check -> $rc"
    return $rc
}

step_probe() {
    local targets="$1" rc=0 one
    if [ -z "$targets" ] || [ "$targets" = "-" ]; then
        echo "== probe: skipped"
        return 0
    fi
    echo "== probe =="
    # Space-separated, for the same reason `test` takes a list: when a change
    # touches a file two entries share, BOTH entries' receipts have to be
    # re-earned after it, and paying the ~90s container+import overhead twice
    # for that is the thing this script exists to avoid. A single target is
    # unchanged by this -- every cached one-probe command still means the same.
    for one in $targets; do
        echo "-- probe $one"
        python3 "$REPO/$one" || rc=$?
    done
    echo "probe: $(echo "$targets" | wc -w) target(s) -> $rc"
    return $rc
}

step_test() {
    local target="$1"
    if [ -z "$target" ] || [ "$target" = "-" ]; then
        echo "== test: skipped"
        return 0
    fi
    echo "== test =="
    # CI runs pytest with cwd=tests/ and the l0 entry as a relative path.
    local out="$WORK/logs/pytest-${SLURM_JOB_ID:-local}-$$.txt"
    ( cd "$REPO/tests" && python3 -m pytest -q --timeout=2400 $target 2>&1 | tee "$out" )
    local rc=${PIPESTATUS[0]}
    local summary
    summary=$(grep -E "[0-9]+ (passed|failed|error)" "$out" | tail -1)
    echo "pytest exit $rc; summary: ${summary:-<none>}"
    return $rc
}

rc=0
case "$MODE" in
  test)   step_test  "$ARG1"; rc=$? ;;
  probe)  step_probe "$ARG1"; rc=$? ;;
  lint)   step_lint  "$ARG1"; rc=$? ;;
  fmt)    step_fmt   "$ARG1"; rc=$? ;;
  verify)
    # Order is the point: mutate, then verify, then measure. Every step runs
    # so one round trip reports everything; rc is the first non-zero.
    step_fmt   "$ARG1"; r1=$?
    step_lint  "$ARG1"; r2=$?
    step_probe "$ARG2"; r3=$?
    step_test  "$ARG3"; r4=$?
    for r in $r1 $r2 $r3 $r4; do [ "$r" -ne 0 ] && [ "$rc" -eq 0 ] && rc=$r; done
    echo "verify: fmt=$r1 lint=$r2 probe=$r3 test=$r4 -> $rc"
    ;;
  *) echo "unknown mode $MODE"; rc=2 ;;
esac

echo "---- entry $MODE exit $rc ----"
exit $rc
