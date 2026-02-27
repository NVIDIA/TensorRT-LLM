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
  perf_sanity_postprocess.py # Post-process perf data and generate HTML report
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

## Post-Processing and Triage

### `perf_sanity_postprocess.py`

Triggered at the end of the CI pipeline in `jenkins/L0_MergeRequest.groovy`. It reads
perf data YAML files produced by test stages and generates an HTML report with inline SVG
performance charts. The report visualizes key throughput metrics (total token throughput,
output token throughput, request throughput, user throughput) along with historical data
for regression comparison.

### `perf_sanity_triage.py`

Triggered by `jenkins/runPerfSanityTriage.groovy`. It supports two operations:

1. **`SLACK BOT SENDS MESSAGE`**: Queries regression data (post-merge only) from the
   OpenSearch database and sends a formatted summary to a Slack channel.

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
