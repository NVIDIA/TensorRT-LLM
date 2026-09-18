<!-- Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved. -->

# NVBUG 6649384: full disaggregated stress A/B

This investigation harness compares the executor before and after the revert
of PR #16687 in PR #18327. It supports the historical comparison by default
and explicitly pinned current-head profiles. It does not establish that #16687 caused the bug,
and the revert PR must not be merged as a production fix on this evidence.

| Historical default identity | Pinned revision |
| --- | --- |
| Control A: historical source with #16687 | `0f2c3a95f9415045bdf06a7230759475692483b6` |
| Treatment B: exact executor revert | `3245fc3ecd76e2fb610f42f2422102e2430c28fe` |
| Shared test harness | The new PR head, recorded separately in each manifest |

Build the control once. Both arms use that wheel's identical compiled artifacts
and dependencies; only `tensorrt_llm/_torch/pyexecutor/py_executor.py` changes.
The driver extracts the two executor files directly from the immutable Git
revisions, verifies the original wheel contains the control file, and checks
compiled artifact hashes. Do not rebase the PR, change these runtime revisions,
or replace the historical wheel with an rc25 or current release overlay.

## Current-head comparison

To investigate whether the behavior still matters today, pin a current control
commit, prepare a reviewed semantic revert on that same commit, and build the
control once. Later refactors can move the original behavior across files: for
example, the September 18 comparison changes both `py_executor.py` and
`disagg_adapter.py`. A historical single-file overlay on a newer wheel is not
the same experiment.

Pass an explicit JSON profile to both the builder and runner with `--profile`;
the Slurm launcher forwards `AB_PROFILE`. Its schema is:

```json
{
  "schema_version": 1,
  "name": "current-head-comparison",
  "control": "<full 40-character control commit>",
  "treatment": "<full 40-character treatment commit>",
  "harness_sha": "<full 40-character common test-harness commit>",
  "runtime_files": [
    "tensorrt_llm/_torch/pyexecutor/disagg_adapter.py",
    "tensorrt_llm/_torch/pyexecutor/py_executor.py"
  ],
  "non_runtime_files": [],
  "expected_requests": 60000,
  "dependency_versions": {
    "aiperf": "0.8.0",
    "lm_eval": "0.4.10",
    "nixl-cu13": "1.4.0"
  }
}
```

Use actual full commit IDs and source-verified dependency pins. The profile
must match the wheel provenance, the entire control/treatment diff and the
clean common test checkout. Runtime changes are limited to explicitly listed
Python files; compiled changes require a different comparison strategy.
All overlaid modules are checked for exact import paths and file hashes.

Keep the common test checkout at `AB_HARNESS` and the runner checkout at
`AB_RUNNER_ROOT` when they differ. Both must be inside the mounted project
root. Fetch the exact Git objects for both variants before running, including
any local treatment and harness commits transferred with a Git bundle.
Record the current source's test/configuration changes from the historical
case. Both arms use the same current test and its original gates.

Resolve the current source's build environment to an immutable digest. A
verified public base used by its Dockerfile can support a source build when
the CI development image is inaccessible; record that environment difference
and validate its prerequisites. Do not present such a run as historical or
complete CI reproduction. If both current arms pass, the finding is simply
no reproduction under the current conditions.

## Experiment

The launcher runs **A, B, B, A**, sequentially in one exclusive allocation on
one physical node with eight B200 GPUs and approximately 2 TB host RAM. Each
trial gets a fresh Slurm step and container so the scheduler can clean up MPI
descendants before the next trial. It invokes the original test directly:

```text
disaggregated/test_disaggregated.py::test_disaggregated_stress_test[input8k-output1k-conc512-gpt_oss_120b_eagle_trtllm_stress]
```

The historical test retains its GPT-OSS-120B plus Eagle3 setup: context TP4/EP4
on GPUs 0–3, generation TP4/EP4 on GPUs 4–7, attention DP, PP1, disabled overlap,
Python NIXL, maximum batch 128, maximum draft length 3, concurrency 512,
60,000 requests, input length 8192, output length 1024 ± 102, seed 100 and 10%
intentional cancellations after 0.5 seconds. The test's non-cancellation error
gate remains 5%, and its subsequent GSM8K accuracy gate remains 0.42. No waive
file is supplied. A skipped test is invalid, never a pass.

The test helper's opt-in `TLLM_DISAGG_STRESS_KEEP_LOGS=1` preserves the worker
logs that were previously deleted even with `--keep-workspace`. Both arms use
the same modified test helper. Default cleanup for other callers is unchanged.

## Prepare on an approved compute node

Check the target cluster's current policy before source preparation or builds.
Use a current allocation owned by the submitting user. Keep account, QoS,
partition, registry credentials and private paths in a local run record.

Use a persistent project root containing the complete Git clone, both source
checkouts, build directory, wheel and results. Mount that root at the same
absolute path in every container; detached worktree Git pointers and the build
virtual environment must remain valid. Initialize the control's pinned
submodules and keep that checkout completely clean. Use a new directory for
each build and experiment. Do not modify shared model files during the matrix.

The historical CI build environment is:

```text
artifactory.nvidia.com/sw-tensorrt-llm-docker-local/tensorrt-llm:pytorch-26.05-py3-x86_64-ubuntu24.04-skip-tritondevel-202607311529-16970
```

Resolve an accessible registry manifest and record its immutable `sha256:...`
digest. A tag alone is insufficient. An installed wheel inside that image is
not proof of control-source identity. If the historical image is unavailable,
stop and resolve an exact mirror or explicitly agree on a common replacement
and its comparability limitation before testing. No replacement is automatic.

In a bounded build allocation, launch that image **by digest**, mount the
persistent project root, set `TLLM_AB_IMAGE` and `TLLM_AB_IMAGE_DIGEST` to the
recorded reference and digest, and run with the image's system Python:

```bash
python3 "$AB_HARNESS/examples/disaggregated/slurm/stress_ab/build_runtime.py" \
  --source "$AB_PROJECT_ROOT/control" \
  --output "$AB_PROJECT_ROOT/control-build" \
  --build-root "$AB_PROJECT_ROOT/control-build-state" \
  --image "$AB_IMAGE" --image-digest "$AB_IMAGE_DIGEST" \
  --jobs 16 --timeout 21600
```

The builder uses the historical `scripts/build_wheel.py`, an out-of-tree clean
build and B200 `100-real` kernels. It pre-stages the unchanged
`requirements-grpc-smg.txt` omitted by that revision's packaging helper and
records the file hash. It records the command, source and submodules,
image digest, build log, wheel hash and prepared build virtual environment.
The build's six-hour limit is separate from the test allocation. Select a
permitted build allocation with time for preparation and artifact flush too.

With an explicit profile, use that control revision's build script and pass
the same `--profile /persistent/path/profile.json` to the builder. Optional
`--cpp-build-dir /node-local/path/cpp` puts large CMake intermediates on local
scratch while preserving the build venv, wheel and logs persistently.
`--skip-stubs` skips Python type-stub generation for a CPU-only build; it does
not skip native compilation. These choices are recorded in the build command.

**Retain the build virtual environment.** `build_wheel.py` installs the pinned
source's development requirements there, not into the base image's Python.
The experiment runs with the recorded interpreter and verifies its dependency
snapshot. Installing the wheel with `--no-deps` into an unprepared base image
is not sufficient. Do not install or upgrade dependencies between trials.

Before allocating eight GPUs, verify available storage for the image, build,
wheel, installed runtime copies and four complete log sets. Do not use a small
home quota or assume the build fits because the model weights are shared.
The driver retains runtime trees and refuses a trial if free space is less
than four wheel sizes plus 20 GiB; that check is not a build-size estimate.

Required readable shared assets below `AB_MODELS_ROOT`:

- `gpt_oss/gpt-oss-120b`: configuration, local tokenizer and all indexed shards.
- `gpt_oss/gpt-oss-120b-Eagle3`: configuration and all weight shards.
- `datasets/openai/gsm8k/main/test-00000-of-00001.parquet` with `question` and
  `answer` columns.

The runner preflights local-only tokenizer loading, dataset readability,
`aiperf==0.8.0`, `lm_eval==0.4.10`, CLI entry points, NIXL, runtime import paths
and GPU identity. Model checkpoints being present alone is not a runtime
validation. The driver records input hashes and versions; keep the shared
mount unchanged throughout the matrix.

## Submit one bounded matrix

Set these variables in a private run configuration. `AB_PYTHON` must be the
literal `runtime_python` recorded in `provenance.json`; do not resolve its
symlink to the system Python. All project paths must be beneath the mounted
`AB_PROJECT_ROOT`.

```bash
export AB_PROJECT_ROOT=/persistent/path/to/experiment
export AB_HARNESS="$AB_PROJECT_ROOT/harness"
export AB_WHEEL="$AB_PROJECT_ROOT/control-build/wheels/<exact-wheel>.whl"
export AB_PROVENANCE="$AB_PROJECT_ROOT/control-build/provenance.json"
export AB_PYTHON="$AB_PROJECT_ROOT/control-build-state/venv-3.12/bin/python3"
export AB_MODELS_ROOT=/shared/path/to/llm-models
export AB_RUN_ROOT="$AB_PROJECT_ROOT/results/abba-<unique-run-id>"
export AB_IMAGE='<verified-registry/repository:historical-tag>'
export AB_IMAGE_DIGEST='sha256:<verified-64-hex-digest>'
export AB_TRIAL_TIMEOUT=12600

# For an explicit current-head comparison, also set:
# export AB_PROFILE="$AB_PROJECT_ROOT/profile.json"
# export AB_RUNNER_ROOT="$AB_PROJECT_ROOT/runner"

sbatch --parsable --account='<approved-account>' \
  --partition='<eight-B200-partition>' --qos='<approved-long-qos>' \
  --output="$AB_PROJECT_ROOT/slurm-%j.out" \
  "$AB_HARNESS/examples/disaggregated/slurm/stress_ab/launch.slurm"
```

The script requests one node, eight GPUs, 224 CPUs, all host memory and 16 hours.
Four 3.5-hour test limits plus setup and flush require at least 15 hours. It
checks the actual granted wall time before starting. A four-hour allocation
cannot fit this matrix. Queue wait and the separate source build are not part
of these 16 hours. Use a permitted longer QoS or a real reservation; a longer
partition maximum alone does not establish QoS eligibility.

On a preemptible route, `--no-requeue` prevents automatic repetitions. An
interruption leaves the matrix invalid/incomplete. Record the job ID once,
verify scheduler resources and arrange status monitoring. Do not resubmit
after an ambiguous submission response without reconciling the unique run ID.

## Results and interpretation

`plan.json` records the intended matrix. `slurm-steps.tsv` preserves every
step exit code. Each step writes its own `summary.json`, `manifest.json`, pip
and preflight logs, JUnit XML, pytest output and retained worker/request logs.
Progress is saved to persistent storage before and after each trial.

The final `summary.json` reports per-arm pass/fail/invalid counts. Exit codes
are 0 for all four passing trials, 1 for a complete matrix containing a test
failure, and 2 for an invalid matrix. Scheduler interruption can return a
different nonzero status. Missing or extra tests, skipped/setup failures,
missing worker logs, incomplete 60,000-request accounting, runtime mismatch,
timeout and residual processes cannot become a pass. A valid test failure can
continue to the next arm; an invalid setup stops the matrix without retries.

Compare the **first initiating context/generation worker exception**, request
accounting and accuracy result across both repetitions. The proxy's later
`LLM is shutting down` error alone does not identify the cause. A failing
control and passing revert supports an effect in this environment; two runs
per arm do not establish flake causality. Both passing means no reproduction
under these conditions. Both failing with different initiating exceptions
does not establish the same bug or exonerate the candidate change.

The earlier TP2 rc25 eight-request smoke test and PR CI build/release checks
are not this experiment. The earlier CI success ran zero mapped tests and
must not be reported as B200 stress coverage.

## Local validation without GPUs

```bash
python3 -m unittest discover -s examples/disaggregated/slurm/stress_ab -v
bash -n examples/disaggregated/slurm/stress_ab/launch.slurm
python3 examples/disaggregated/slurm/stress_ab/run_ab.py \
  --harness . --wheel /unused/control.whl --provenance /unused/provenance.json \
  --models-root /unused/models --output /unused/results --dry-plan
```

These checks validate the harness and plan only. They do not build the control,
validate model/runtime availability or execute the GPU test.
