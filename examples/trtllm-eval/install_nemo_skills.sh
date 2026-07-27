#!/bin/bash
# Install the Python lib + deps for the in-process NeMo-Skills accuracy benches
# (gpqa_ns, ifbench, scicode_ns, hle_aa) and their integration guards
# (TestNemotronV3Super::test_nvfp4_nemo_skills_*).
#
# Installs ONLY the Python lib -- datasets and grader assets come from the shared,
# read-only ns_acc_bench_infra folder, so there is no download / prepare step here.
#
# By default it installs into the CURRENT Python environment (run this inside a
# TensorRT-LLM container; nemo_skills + the small grader deps sit alongside the
# existing tensorrt_llm/torch). Installing into the same interpreter that runs the
# tests keeps it simple and means the ifbench grader's `python -m run_eval`
# subprocess just works -- no venv to activate. Set VENV_DIR only if you need
# isolation (e.g. a read-only base image, or to persist on a mount across jobs).
#
# Usage:
#   bash examples/trtllm-eval/install_nemo_skills.sh
#   VENV_DIR=/path/to/venv bash examples/trtllm-eval/install_nemo_skills.sh   # optional venv
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
# Pin a known-good NeMo-Skills commit (v0.7.0) for reproducible grading; override
# with NS_REF=git+https://github.com/NVIDIA/NeMo-Skills.git@<ref> if needed.
NS_COMMIT=${NS_COMMIT:-da85a881d972e6fec847b90cf553a0bf9bf10638}
NS_REF=${NS_REF:-git+https://github.com/NVIDIA/NeMo-Skills.git@${NS_COMMIT}}

if [ -n "${VENV_DIR:-}" ]; then
  [ -x "$VENV_DIR/bin/python" ] || python -m venv --system-site-packages "$VENV_DIR"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  echo "[install] using venv $VENV_DIR (activate it before running)"
else
  echo "[install] using the current Python environment ($(command -v python))"
fi

python -c "import nemo_skills, math_verify" 2>/dev/null || {
  python -m pip install -q --upgrade pip
  pip install -q --no-deps --no-warn-conflicts "nemo-skills @ $NS_REF"
}
pip install -q --no-warn-conflicts -r "$HERE/requirements_nemo_skills.txt"

echo
echo "[done] nemo_skills installed. Now just run -- the benches autowire the shared,"
echo "read-only ns_acc_bench_infra (datasets + IFBench/SciCode grader paths + sandbox)"
echo "from NS_ACC_BENCH_INFRA."
echo "  default infra: <scratch_folder>/datasets/ns_acc_bench_infra  (override: export NS_ACC_BENCH_INFRA=<dir>)"
echo "  trtllm-eval --model <hf_or_path> gpqa_ns   # or ifbench / scicode_ns / hle_aa"
