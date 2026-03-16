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
