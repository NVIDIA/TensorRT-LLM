#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#SBATCH --job-name=disagg-stress-ab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=224
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --time=16:00:00
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120

# Account, partition and QoS must be explicitly selected at submission according
# to the site's current policy. This script never submits or retries a job.
set -Eeuo pipefail

: "${AB_PROJECT_ROOT:?Persistent project root containing the clone, worktree, wheel and results}"
: "${AB_HARNESS:?Pinned PR checkout under AB_PROJECT_ROOT}"
: "${AB_WHEEL:?Control wheel from build_runtime.py}"
: "${AB_PROVENANCE:?Matching provenance.json from build_runtime.py}"
: "${AB_PYTHON:?Absolute runtime_python from provenance.json; retain its build venv}"
: "${AB_MODELS_ROOT:?Readable GPT-OSS and Eagle3 model root}"
: "${AB_RUN_ROOT:?New persistent run directory under AB_PROJECT_ROOT}"
: "${AB_IMAGE:?Historical container registry/repository:tag}"
: "${AB_IMAGE_DIGEST:?Resolved immutable sha256 digest}"
: "${SLURM_JOB_ID:?Launch through sbatch in an approved compute allocation}"
[[ ${SLURM_JOB_NUM_NODES:-0} == 1 ]] || { echo 'Exactly one node is required' >&2; exit 2; }
[[ $AB_IMAGE_DIGEST =~ ^sha256:[0-9a-f]{64}$ ]] || { echo 'Invalid image digest' >&2; exit 2; }
[[ $AB_IMAGE =~ ^[a-zA-Z0-9._/-]+:[a-zA-Z0-9._-]+$ ]] || { echo 'Use a registry/repository:tag image' >&2; exit 2; }
[[ ! -e $AB_RUN_ROOT ]] || { echo 'Run directory already exists; refusing to overwrite' >&2; exit 2; }
for path in "$AB_PROJECT_ROOT" "$AB_HARNESS" "$AB_WHEEL" "$AB_PROVENANCE" "$AB_PYTHON" "$AB_MODELS_ROOT" "$AB_RUN_ROOT"; do
    [[ $path == /* && $path != *','* && $path != *':'* && $path != *$'\n'* ]] || {
        echo 'Use absolute mount paths without comma, colon or newline' >&2; exit 2;
    }
done
for path in "$AB_HARNESS" "$AB_WHEEL" "$AB_PROVENANCE" "$AB_PYTHON" "$AB_RUN_ROOT"; do
    [[ $path == "$AB_PROJECT_ROOT/"* ]] || { echo 'Project artifacts must be under AB_PROJECT_ROOT' >&2; exit 2; }
done
[[ -f $AB_WHEEL && -f $AB_PROVENANCE && -d $AB_HARNESS && -d $AB_MODELS_ROOT ]] || {
    echo 'Missing prepared source, wheel, provenance or models' >&2; exit 2;
}

trial_timeout=${AB_TRIAL_TIMEOUT:-12600}
[[ $trial_timeout =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid trial timeout' >&2; exit 2; }
# Four full test timeouts plus 10 minutes of preflight per step and 20 minutes
# for container/setup/flush. A shorter allocation is rejected before testing.
required_seconds=$((4 * (trial_timeout + 600) + 1200))
job_time_limit=$(scontrol show job "$SLURM_JOB_ID" -o | python3 -c '
import re, sys
value = re.search(r"\bTimeLimit=(\S+)", sys.stdin.read()).group(1)
days, clock = value.split("-", 1) if "-" in value else ("0", value)
parts = [int(v) for v in clock.split(":")]
if len(parts) == 2:
    parts.insert(0, 0)
print(int(days)*86400 + parts[0]*3600 + parts[1]*60 + parts[2])')
(( job_time_limit >= required_seconds )) || { echo 'Allocation cannot fit the bounded ABBA plan' >&2; exit 2; }

mkdir -p "$AB_RUN_ROOT"
runner_dir="$AB_HARNESS/examples/disaggregated/slurm/stress_ab"
export TLLM_AB_IMAGE="$AB_IMAGE" TLLM_AB_IMAGE_DIGEST="$AB_IMAGE_DIGEST"
image_repo=${AB_IMAGE%:*}
container_image="${image_repo/\//#}@${AB_IMAGE_DIGEST}"
# Preserve absolute paths so detached worktree .git pointers remain valid.
mounts="$AB_PROJECT_ROOT:$AB_PROJECT_ROOT,$AB_MODELS_ROOT:$AB_MODELS_ROOT:ro"
printf 'index\tarm\texit_code\n' > "$AB_RUN_ROOT/slurm-steps.tsv"
python3 - "$AB_RUN_ROOT/plan.json" "$AB_HARNESS" "$AB_IMAGE" "$AB_IMAGE_DIGEST" "$trial_timeout" <<'PY'
import json, os, socket, subprocess, sys
from datetime import datetime, timezone
from pathlib import Path
path, harness, image, digest, timeout = sys.argv[1:]
plan = {
    'schema_version': 1, 'order': ['control', 'treatment', 'treatment', 'control'],
    'slurm_job_id': os.environ['SLURM_JOB_ID'], 'node': socket.gethostname(),
    'started_at': datetime.now(timezone.utc).isoformat(), 'trial_timeout': int(timeout),
    'harness_sha': subprocess.check_output(['git', '-C', harness, 'rev-parse', 'HEAD'], text=True).strip(),
    'image': image, 'image_digest': digest, 'same_node': True,
    'cleanup_boundary': 'fresh Slurm step and container for each trial',
}
Path(path).write_text(json.dumps(plan, indent=2) + '\n')
PY

finish() {
    status=$?
    trap - EXIT
    set +e
    python3 "$runner_dir/summarize.py" "$AB_RUN_ROOT"
    summary_status=$?
    if (( status != 0 )); then exit "$status"; fi
    exit "$summary_status"
}
trap finish EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

index=0
for arm in control treatment treatment control; do
    index=$((index + 1))
    trial_dir=$(printf '%s/%02d-%s' "$AB_RUN_ROOT" "$index" "$arm")
    if srun --nodes=1 --ntasks=1 --gpus-per-node=8 --kill-on-bad-exit=1 \
        --container-image="$container_image" --container-mounts="$mounts" \
        --container-workdir="$AB_PROJECT_ROOT" \
        --output="$AB_RUN_ROOT/step-$index.out" --error="$AB_RUN_ROOT/step-$index.err" \
        "$AB_PYTHON" "$runner_dir/run_ab.py" --harness "$AB_HARNESS" \
        --wheel "$AB_WHEEL" --provenance "$AB_PROVENANCE" \
        --models-root "$AB_MODELS_ROOT" --output "$trial_dir" \
        --order "$arm" --trial-timeout "$trial_timeout"; then
        step_status=0
    else
        step_status=$?
    fi
    printf '%s\t%s\t%s\n' "$index" "$arm" "$step_status" >> "$AB_RUN_ROOT/slurm-steps.tsv"
    # An expected pytest failure is an observation. Infrastructure, timeout or
    # invalid setup stops this bounded run; no automatic retries are made.
    if (( step_status != 0 && step_status != 1 )); then exit "$step_status"; fi
    # srun itself can also return 1; only a complete classified pytest failure
    # may continue to the next arm.
    python3 - "$trial_dir/summary.json" "$arm" "$step_status" <<'PY'
import json, sys
from pathlib import Path
path, arm, code = sys.argv[1:]
data = json.loads(Path(path).read_text())
trials = data.get('trials', [])
expected = 'pass' if code == '0' else 'fail'
if (data.get('status') != expected or len(trials) != 1
        or trials[0].get('arm') != arm or trials[0].get('status') != expected):
    sys.exit(2)
PY
done
