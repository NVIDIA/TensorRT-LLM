# Perf Sanity Scripts

This directory contains scripts for running perf sanity tests and managing perf sanity data.

## Directory Structure

```
jenkins/scripts/perf/
  aggregated/
    slurm_launch_draft.sh    # Draft template for aggregated SLURM launch scripts
  disaggregated/
    submit.py                # CI pipeline submit script (disaggregated only)
    slurm_launch_draft.sh    # Draft template for disaggregated SLURM launch scripts
  local/
    submit.py                # Local submit script (aggregated and disaggregated)
    slurm_install.sh         # Build wheel + pip install inside container
    slurm_run.sh             # Run pytest inside container
    run_disagg.sh            # Config-driven wrapper: read .conf, generate slurm_launch.sh, sbatch
    configs/                 # Per-cluster / per-case .conf files consumed by run_disagg.sh
      example.conf           # Reference template — copy and tweak
  perf_utils.py              # Shared utilities (regression detection, baseline, charts, OpenSearch queries)
  get_pre_merge_html.py      # Pre-merge HTML report with history, baseline, and threshold
  perf_sanity_triage.py      # Query/update OpenSearch data and send Slack notifications
```

## Submit Scripts

Both `local/submit.py` and `disaggregated/submit.py` share a similar workflow. They read
a test config YAML and use the appropriate draft template
(`aggregated/slurm_launch_draft.sh` or `disaggregated/slurm_launch_draft.sh`) to generate
a complete `slurm_launch.sh`. Then the user or CI pipeline can run `sbatch slurm_launch.sh`
to submit the job. Inside the SLURM job, `slurm_install.sh` builds the wheel and runs
installation, then `slurm_run.sh` runs pytest.

```
submit.py
  |
  v
slurm_launch.sh  (generated)
  |
  |-- srun --> slurm_install.sh   (build wheel + pip install)
  |-- srun --> slurm_run.sh       (run pytest)
```

Both submit scripts read `AGG_CONFIG_FOLDER` and `DISAGG_CONFIG_FOLDER` environment
variables (with defaults of `tests/scripts/perf-sanity/aggregated` and
`tests/scripts/perf-sanity/disaggregated`) and propagate them via `PYTEST_COMMON_VARS`
into the pytest execution environment where `test_perf_sanity.py` uses them to locate
config files.

### `local/submit.py`

Used for **local runs**. Supports both **aggregated** and **disaggregated** modes. It
detects the mode from the test config YAML (aggregated configs have `server_configs`,
disaggregated configs have `worker_config`) and selects the correct draft template
automatically.

See [`local/README.md`](local/README.md) for full argument reference and examples.

### `disaggregated/submit.py`

Used by the **CI pipeline** (called from `jenkins/L0_Test.groovy`'s
`runLLMTestlistWithSbatch`). Only supports **disaggregated** mode. It receives a
script prefix and srun args from the CI pipeline and combines them with disagg-specific
environment variables and hardware configuration to generate `slurm_launch.sh`.

## Running Local Tests with `run_disagg.sh`

`local/run_disagg.sh` is a config-driven wrapper around `local/submit.py`. It reads a
single `.conf` file, applies defaults, generates one `slurm_launch.sh` per test, and
submits each via `sbatch`. Despite the historical name, it supports **both** aggregated
and disaggregated runs — the runtime mode (and the right draft template) is derived
automatically from each `test_id`.

### Quick Start

Run this on a SLURM login node:

```bash
cd ${YOUR_TRTLLM_PATH}/jenkins/scripts/perf/local

# 1. Copy the example and edit it for your cluster + test case
cp configs/example.conf configs/mycluster.conf
$EDITOR configs/mycluster.conf

# 2. Submit (one sbatch per entry in test_ids)
bash run_disagg.sh -c configs/mycluster.conf

# 3. Watch — logs and outputs are under $work_dir as printed by the script
squeue -u $USER
```

`bash run_disagg.sh -h` prints the inline header plus a list of available `.conf`
files under `configs/`.

### Config File Reference

The `.conf` file is `source`d as bash. Variables not set in the file fall back to
defaults inside `run_disagg.sh`. Anything `export`ed in the shell before invoking the
script takes precedence over both.

All examples below use placeholders like `${YOUR_TRTLLM_PATH}` — replace with your
own absolute paths.

#### Paths

| Variable | Required | Description |
|----------|----------|-------------|
| `trtllm` | recommended | Login-node path to your TensorRT-LLM checkout. The container must be able to see it via `mounts`. Example: `${YOUR_TRTLLM_PATH}`. |
| `work_dir` | no | One run goes in one work dir; created if missing. Holds `slurm_launch.sh`, `test_list.txt`, `slurm-<jobid>.out`, `report.xml`, and per-role logs. Default: `$HOME/perf_runs/disagg_<timestamp>`. Keeping it under `$trtllm` lets a single mount of the source tree cover it. |
| `llm_models_path` | yes | Host path to the model weights tree. Must be reachable inside the container via `mounts`. Example: `${YOUR_MODELS_PATH}/llm-models`. |
| `mounts` | recommended | Comma-separated `host:container` bind-mount pairs. Every path the container touches at runtime — source tree, wheel, work_dir, models — must be reachable through one of these. Recommended: `"$trtllm:$trtllm,$llm_models_path:$llm_models_path"`. |

#### SLURM

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `partition` | **yes** | — | SLURM partition. Script errors if left as the `CHANGE_ME` placeholder. |
| `account` | no | `coreai_comparch_trtllm` | SLURM billing account. |
| `job_name` | no | `disagg_test` | Base job name. For multi-test runs, each sub-job is suffixed with `_<idx>`. |
| `time_limit` | no | `02:00:00` | `HH:MM:SS` SLURM wall-time. |

#### Docker image (pick ONE of two modes)

| Mode | `image` | `image_var` | Behavior |
|------|---------|-------------|----------|
| (a) — pinned | non-empty | ignored | `image` used verbatim. Reproducible, recommended for benchmarking. |
| (b) — auto-resolve | empty / unset | used as key name | Resolved from `${YOUR_TRTLLM_PATH}/jenkins/current_image_tags.properties` by `image_var=` key. |

- `image` — Full image URI (mode a). URIs beginning with `urm.nvidia.com/` are
  auto-rewritten to enroot form (`urm.nvidia.com#`); the script is a no-op if you
  already passed the enroot form.
- `image_var` — Key in `current_image_tags.properties` (mode b). Common keys:
  `LLM_DOCKER_IMAGE` (x86 — H100 / B200), `LLM_SBSA_DOCKER_IMAGE` (SBSA — GB200).
  Default: `LLM_DOCKER_IMAGE`.

Do **not** expect setting both `image` and `image_var` to "combine" — `image` always
wins when set.

#### Test selection

- `test_ids` (preferred) — Bash array of full pytest IDs. Each entry is submitted as
  its own `sbatch`. With >1 entry, each lands in its own subdir under `$work_dir`
  named `<idx>_<test-slug>`.
- `test_id` (legacy) — Single test string. Equivalent to a 1-element `test_ids`.

Test-ID format:

```
perf/test_perf_sanity.py::test_e2e[<runtime>-<mode>-<yaml-stem>[-<server-cfg>]]
```

- `<runtime>` = `disagg` | `aggr`
- `<mode>` (disagg) = `e2e` | `gen_only` | `ctx_only`
- `<yaml-stem>` matches a YAML file in `tests/scripts/perf-sanity/disaggregated/`
  (or `aggregated/`)
- `<server-cfg>` — only for normal aggregated tests — the `name:` field of one of
  the YAML's `server_configs` entries

`run_disagg.sh` errors out if any entry still contains the literal placeholder
`CHANGE_ME`.

#### Install mode

- `install_mode` — `wheel` (default) or `source`.
  - `wheel`: container does `pip install <wheel_path>`.
  - `source`: container does `pip install -e .` against the mounted `$trtllm`.
    `wheel_path` is ignored in this mode.
- `wheel_path` — Local `.whl` file. **Required when `install_mode=wheel`** (must
  exist; script errors otherwise). Must follow PEP 427 naming
  (`tensorrt_llm-<ver>-<py>-<abi>-<plat>.whl`); pip rejects malformed names. Find
  the exact name via:
  ```bash
  ls ${YOUR_TRTLLM_PATH}/build/tensorrt_llm-*.whl
  ```
  Keeping it under `$trtllm/build` lets the single `$trtllm` mount cover it.

#### Optional flags (leave unset = disabled)

- `build_wheel_flag` — Set to `"--build-wheel"` to build the wheel inside the
  container before installing. Works with either `install_mode`.
- `capture_nsys_flag` — Set to `"--capture-nsys"` to wrap the worker pytest in
  `nsys profile`. Profiles land in `$work_dir/nsys.*.<rank>.qdrep` (per-role for
  disagg, per-rank for aggregated).

#### Cluster compatibility

- `strip_sbatch_opts` — Comma-separated `#SBATCH` directives to comment out in the
  generated `slurm_launch.sh`, for clusters whose SLURM version / configuration
  doesn't accept them. Default in `run_disagg.sh`: `--segment`. Why these may need
  stripping:
  - `--segment` — newer SLURM topology option, missing on older clusters (e.g.,
    EOS).
  - `--gres` — EOS doesn't register GPUs as a `gres` at all.
  - `--gpus-per-node` — SLURM translates this into a `gres` request internally,
    which also fails on EOS for the same reason.

  Recommended:
  | Cluster | `strip_sbatch_opts` |
  |---------|---------------------|
  | EOS | `"--segment,--gres,--gpus-per-node"` |
  | B200 / GB200 modern clusters | `"--segment"` (or `""` if even `--segment` works) |

### Multi-Test Work Directory Layout

For a `test_ids` array with more than one entry:

```
$work_dir/
├── 00_<slug-of-test-0>/
│   ├── slurm_launch.sh
│   ├── slurm-<jobid>.out
│   ├── install.log
│   ├── (disagg only) gen_server_*.log, ctx_server_*.log, disagg_server.log
│   ├── test_list.txt
│   └── report.xml
├── 01_<slug-of-test-1>/
│   └── ...
└── ...
```

Each sub-job is submitted independently — one bad `test_id` does not stop the rest.
The script exits non-zero if any `submit.py` or `sbatch` invocation failed.

## Shared Utilities

### `perf_utils.py`

Shared module imported by `get_post_merge_html.py`, `get_pre_merge_html.py`, and
`perf_sanity_triage.py`. Contains:

- **Constants**: `CHART_METRICS` (4 key throughput metrics), `METRIC_LABELS`,
  algorithm parameters, curve type colors/labels.
- **Baseline computation**: Rolling smooth (window=3) + P95 percentile algorithm.
  Replaces the previous `max(daily_values)` approach which was vulnerable to
  occasional spikes inflating the baseline.
- **Regression detection**: Two-step classification (regression check + subtype
  pattern matching). Supports per-metric thresholds from baseline data
  (`d_threshold_pre_merge_*` fields, defaulting to 5%).
- **OpenSearch query + grouping**: `get_history_data()` queries both baseline and
  non-baseline data, groups by `(s_test_case_name, s_gpu_type)`.
- **SVG chart generation**: Unified chart function supporting history lines,
  new data points, baseline line, threshold line, curve type badges, and jump
  interval shading.
- **HTML dashboard**: `generate_post_merge_html()` produces a full interactive
  report with three-way cascading filters and click-to-inspect data-point popups.

## MPI/PMI Handling in Disaggregated Tests

### Background

Disaggregated tests run four srun steps within a single SLURM job. Only CTX/GEN workers
need MPI (they use `trtllm-llmapi-launch`). The disagg server (`trtllm-serve
disaggregated`) and benchmark client are single-process, non-MPI tasks.

When srun launches a process with `--mpi=pmix`, it sets PMI/PMIx environment variables.
If the launched process imports libraries with MPI support (e.g., PyTorch links Open MPI),
`MPI_Init` may be triggered automatically. If the container's MPI build lacks SLURM PMI
support, this causes:
```
PMI2_Init failed to initialize.  Return code: 14
```

### Solution

The `--mpi=pmix` flag is added **only** to the CTX/GEN worker srun commands in
`slurm_launch_draft.sh`, not to the shared `srunArgs` array. This way, the disagg server
and benchmark srun steps never see MPI flags.

**Where MPI is configured:**
- `jenkins/scripts/perf/disaggregated/slurm_launch_draft.sh` — `--mpi=pmix` on
  ctx/gen srun commands only
- `jenkins/scripts/perf/local/submit.py` — `--mpi=pmi2` for aggregated mode only,
  no MPI flag for disaggregated mode (handled by the draft template)
- `jenkins/L0_Test.groovy` — `--mpi=pmi2` for non-disagg multi-node only

### Key Rules

When modifying disaggregated SLURM scripts, keep these invariants:

1. **srunArgs are shared**: All srun steps in `slurm_launch_draft.sh` use the same
   `"${srunArgs[@]}"`. Never add MPI flags to srunArgs for disaggregated mode.
2. **Only CTX/GEN workers need MPI**: Add `--mpi=pmix` directly on their srun command
   lines in `slurm_launch_draft.sh`, not in the shared srunArgs.
3. **Non-MPI roles must stay MPI-free**: The disagg server and benchmark steps must
   not receive `--mpi` flags. If adding a new srun step, consider whether it needs MPI.

## Adding or Re-enabling Perf Sanity Tests in CI

When adding or re-enabling perf sanity tests, two files must be updated:

1. **Test-db YAML** in `tests/integration/test_lists/test-db/` — add or uncomment the test case line
2. **`jenkins/L0_Test.groovy`** — update or add the CI stage in `launchTestJobs()`

### Where to Find CI Stage Definitions

In `jenkins/L0_Test.groovy`, search for `launchTestJobs`. Perf sanity stages are grouped by test type:

| Config Variable | Test Type | Platform |
|-----------------|-----------|----------|
| `x86SlurmTestConfigs` | Single-node aggregated perf sanity (x86) | `"auto:h100-cr-x8"` etc. |
| `SBSASlurmTestConfigs` | Single-node aggregated perf sanity (SBSA/Grace) | `"auto:gb200-x4"` etc. |
| `multiNodesSBSAConfigs` | Multi-node aggregated **and** disaggregated perf sanity | `"auto:gb200-flex"` etc. |

### `buildStageConfigs` Function

Disaggregated and multi-node perf sanity stages use `buildStageConfigs()`:

```groovy
def buildStageConfigs(stageName, platform, testlist, testCount, gpuCount, nodeCount, runWithSbatch=false)
```

- `testlist`: test-db YAML filename without `.yml` extension
- `testCount`: must equal the number of **active (uncommented)** tests in the test-db file (each disagg test gets its own CI stage)
- `gpuCount`: total GPUs allocated per stage = `total_nodes * gpus_per_node`
- `nodeCount`: total SLURM nodes per stage

When adding a test, either increment `testCount` on an existing entry or add a new `buildStageConfigs` block. Stages are grouped by node count (2 Nodes, 3 Nodes, 4 Nodes, etc.).

For the full step-by-step guide including how to derive test-db filenames and GPU/node counts from disaggregated config YAMLs, see [`tests/scripts/perf-sanity/README.md`](../../tests/scripts/perf-sanity/README.md) ("Step-by-Step: Adding or Re-enabling Disaggregated Perf Sanity Tests").

## Post-Processing and Triage

### `get_pre_merge_html.py`

Triggered at the end of the CI pipeline in `jenkins/L0_MergeRequest.groovy`. It has
3 main functions:

1. **`load_perf_data`**: Reads perf_data.yaml files produced by test stages and
   gathers all new perf data together.
2. **`get_pre_merge_history_data`**: Queries OpenSearch for post-merge history data
   (both baseline and non-baseline), grouped by `(s_test_case_name, s_gpu_type)`.
3. **`generate_pre_merge_html`**: Generates an HTML report visualizing each test
   case's key metrics (`d_seq_throughput`, `d_token_throughput`,
   `d_total_token_throughput`, `d_user_throughput`) with history curve, new data
   points, baseline line, and threshold line for regression comparison.

### `perf_sanity_triage.py`

Triggered by `jenkins/runPerfSanityTriage.groovy`. It supports two operations:

1. **`SLACK BOT SENDS MESSAGE`**: Runs the perf-regression-detector pipeline
   (`get_history_data` -> `get_baseline` -> `classify_test_case` ->
   `generate_post_merge_html`), then sends the generated HTML dashboard to a
   Slack channel.

2. **`UPDATE SET ... (WHERE ...)`**: Updates fields on existing perf records that match
   a query scope and posts the updated documents back to OpenSearch.

**Examples**

```
SLACK BOT SENDS MESSAGE
```

```
UPDATE SET b_is_valid=false WHERE s_test_case_name='test1'
UPDATE SET b_is_valid=false WHERE ts_created <= 'Feb 18, 2026 @ 22:32:02.960' AND s_test_case_name='test1'
```

See the `UPDATE` operation section below for supported operators and date formats.

#### UPDATE Operators

- SET clause: Only `=` is supported.
- WHERE clause: Supports `=`, `!=`, `>`, `<`, `>=`, `<=` operators.
- `=` and `!=` operators are allowed for all fields.
- `>`, `<`, `>=`, `<=` operators are only allowed for `ts_created` field (timestamp) or fields starting with `d_` (double type) or `l_` (integer type).

#### `ts_created` Date Formats

The `ts_created` field accepts date strings in the following formats:
- `'Feb 18, 2026 @ 22:32:02.960'` (with milliseconds)
- `'Feb 18, 2026 @ 22:32:02'` (without milliseconds)
- `'2026/02/18'` (date only)

All date strings are interpreted as UTC for consistent timestamp conversion.
