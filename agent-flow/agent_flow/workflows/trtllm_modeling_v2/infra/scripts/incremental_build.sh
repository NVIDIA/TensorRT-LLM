#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Incremental native rebuild of a TensorRT-LLM checkout after a C++/CUDA change,
# without crashing processes that have the old libraries mapped.
#
#   incremental_build.sh [--repo DIR] [--spool DIR]... [--cuda-archs '100-real;103-real']
#                        [--log-dir DIR] [--no-import-check] [-- <extra build_wheel.py args>]
#
# Run it as a whole-node spool job, so nothing else on that session runs during
# the build:
#
#   spoolctl.py submit --gpus all --wall 3h build-<what> \
#       'bash <scripts>/incremental_build.sh --repo <checkout> --spool <spool A> --spool <spool B>'
#
# What it does, in order:
#   1. takes a per-checkout lock, so two builds never run over one cpp/build;
#   2. announces the rebuild on every listed spool: each session's scheduler
#      holds new GPU jobs against this checkout until the build ends;
#   3. gives every installed shared library a fresh inode (copy + rename), so a
#      running process keeps the old file and build_wheel.py's in-place rewrite
#      lands on a file nobody has mapped;
#   4. runs build_wheel.py --skip_building_wheel with ccache, from a venv of
#      the build's own (<repo>/.dev-build/build-venv) that sees the run's
#      packages through a .pth: build_wheel.py pip-installs its development
#      requirements into the venv it runs in, and the run's venv is the one
#      every gate imports from, so a build must never change it (it fails if
#      the run's venv changed under it);
#   5. checks the checkout still imports, and retracts the announcement.
#
# Defaults, all overridable by flag:
#   --repo      $MV2FLOW_REPO, else the git checkout containing the current directory
#   --spool     $MV2FLOW_SESSION_SPOOLS (':'-separated), plus $MV2FLOW_SPOOL (this job's own)
#   --cuda-archs $MV2FLOW_CUDA_ARCHS, else build_wheel.py's own default
#   --log-dir   $MV2FLOW_BUILD_LOG_DIR, else <repo>/.dev-build/logs
#   build python  $MV2FLOW_BUILD_PYTHON, else python3 (the venv on PATH): the
#                 venv the build's own is made beside; MV2FLOW_BUILD_ISOLATED=0
#                 builds in it directly
#
# The build needs the user site (the CUTLASS configure step imports
# cutlass_library from it), so PYTHONNOUSERSITE and PYTHONPATH are unset for
# the build and only the import check runs with PYTHONNOUSERSITE=1.
set -u

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REBUILD="$HERE/../substrate/rebuild.py"
PY=${MV2FLOW_PYTHON:-python3}
# rebuild.py needs Python 3.7; a login node's python3 can be older.
if ! "$PY" -c 'import sys; sys.exit(sys.version_info < (3, 7))' 2>/dev/null; then
  echo "incremental_build: $PY is not Python >= 3.7; set MV2FLOW_PYTHON to the run's interpreter" >&2
  exit 64
fi

REPO=${MV2FLOW_REPO:-}
ARCHS=${MV2FLOW_CUDA_ARCHS:-}
LOG_DIR=${MV2FLOW_BUILD_LOG_DIR:-}
IMPORT_CHECK=1
SPOOLS=()
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --repo) REPO=$2; shift 2 ;;
    --spool) SPOOLS+=("$2"); shift 2 ;;
    --cuda-archs) ARCHS=$2; shift 2 ;;
    --log-dir) LOG_DIR=$2; shift 2 ;;
    --no-import-check) IMPORT_CHECK=0; shift ;;
    --) shift; EXTRA=("$@"); break ;;
    *) echo "incremental_build: unknown argument $1" >&2; exit 64 ;;
  esac
done

if [ -z "$REPO" ]; then
  REPO=$(git rev-parse --show-toplevel 2>/dev/null) || { echo "incremental_build: no --repo and not inside a git checkout" >&2; exit 64; }
fi
REPO=$(cd "$REPO" && pwd)
[ -f "$REPO/scripts/build_wheel.py" ] || { echo "incremental_build: $REPO has no scripts/build_wheel.py" >&2; exit 64; }

if [ ${#SPOOLS[@]} -eq 0 ] && [ -n "${MV2FLOW_SESSION_SPOOLS:-}" ]; then
  IFS=: read -r -a SPOOLS <<< "$MV2FLOW_SESSION_SPOOLS"
fi
if [ -n "${MV2FLOW_SPOOL:-}" ]; then
  SPOOLS+=("$MV2FLOW_SPOOL")
fi
SPOOL_ARGS=()
for s in "${SPOOLS[@]}"; do
  [ -n "$s" ] && SPOOL_ARGS+=(--spool "$s")
done

LOG_DIR=${LOG_DIR:-$REPO/.dev-build/logs}
LOCK=${MV2FLOW_BUILD_LOCK:-$REPO/.dev-build/native-build.lock}
mkdir -p "$LOG_DIR" "$(dirname "$LOCK")"
LOG=$LOG_DIR/incremental-$(date +%Y%m%d-%H%M%S).log

(
  flock 8
  echo "=== incremental build start $(date -Is) on $(hostname) repo=$REPO ==="
  TOKEN=$("$PY" "$REBUILD" begin --tree "$REPO" "${SPOOL_ARGS[@]}" --note "incremental build on $(hostname)")
  trap '"$PY" "$REBUILD" end --token "$TOKEN" "${SPOOL_ARGS[@]}"; echo "=== rebuild announcement $TOKEN retracted ==="' EXIT
  echo "=== rebuild announced as $TOKEN on ${#SPOOLS[@]} spool(s): ${SPOOLS[*]:-none} ==="
  git -C "$REPO" log --oneline -1
  git -C "$REPO" status --porcelain -uno -- cpp | head -40
  "$PY" "$REBUILD" refresh-inodes --root "$REPO/tensorrt_llm" || exit 1
  (
    unset PYTHONNOUSERSITE PYTHONPATH
    export CCACHE_DIR=${CCACHE_DIR:-$REPO/.dev-build/ccache}
    export PIP_CACHE_DIR=${PIP_CACHE_DIR:-$REPO/.dev-build/pip-cache}
    cd "$REPO" || exit 1
    start=$(date +%s)
    ARCH_ARGS=()
    [ -n "$ARCHS" ] && ARCH_ARGS=(--cuda_architectures "$ARCHS")
    RUN_PY=${MV2FLOW_BUILD_PYTHON:-python3}
    BUILD_PY=$RUN_PY
    run_lib=$("$RUN_PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') || exit 1
    installed() { ls -1 "$run_lib" 2>/dev/null | grep -E '\.(dist|egg)-info$' | sort; }
    before=$(installed)
    if [ "${MV2FLOW_BUILD_ISOLATED:-1}" = 1 ]; then
      BUILD_VENV=$REPO/.dev-build/build-venv
      if [ ! -x "$BUILD_VENV/bin/python" ]; then
        "$RUN_PY" -m venv --system-site-packages "$BUILD_VENV" || exit 1
      fi
      own_lib=$("$BUILD_VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])') || exit 1
      echo "$run_lib" > "$own_lib/mv2flow-run-venv.pth"
      BUILD_PY=$BUILD_VENV/bin/python
      echo "=== building from $BUILD_VENV (the run's packages through a .pth) ==="
    fi
    "$BUILD_PY" ./scripts/build_wheel.py --use_ccache --skip_building_wheel \
      "${ARCH_ARGS[@]}" --yes "${EXTRA[@]}"
    rc=$?
    echo "INCREMENTAL_BUILD_WALL_SECONDS=$(( $(date +%s) - start )) rc=$rc"
    if [ "$(installed)" != "$before" ]; then
      echo "incremental_build: the run's venv ($run_lib) changed during the build:" >&2
      diff <(echo "$before") <(installed) >&2
      [ "$rc" = 0 ] && rc=65
    fi
    exit $rc
  )
  rc=$?
  if [ "$rc" = 0 ] && [ "$IMPORT_CHECK" = 1 ]; then
    (cd "$REPO" && PYTHONNOUSERSITE=1 "${MV2FLOW_BUILD_PYTHON:-python3}" -c \
      "import tensorrt_llm; print('import ok', tensorrt_llm.__version__, tensorrt_llm.__file__)") || rc=$?
  fi
  exit $rc
) 8>"$LOCK" 2>&1 | tee "$LOG"
exit "${PIPESTATUS[0]}"
