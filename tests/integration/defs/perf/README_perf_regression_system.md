# Perf Regression System

Generic performance regression detection pipeline used by `test_perf_sanity.py` and any future perf test scripts. This document describes the three-layer architecture and how to integrate a new test script.

For `test_perf_sanity.py`-specific details (config files, test case formats, CI rules, SLURM execution), see [README_test_perf_sanity.md](README_test_perf_sanity.md).

## Three-Layer Architecture

```
test_perf_sanity.py              (test-specific data assembly)
  └── perf_regression_utils.py       (generic regression pipeline)
        └── open_search_db_utils.py      (pure DB operations)
              └── OpenSearchDB           (low-level OpenSearch client)
```

### Layer 1: `open_search_db_utils.py` — Pure DB Operations

The lowest layer. It knows only how to talk to OpenSearch and nothing about regression logic or metrics.

| Function | Purpose |
|----------|---------|
| `add_id(data)` | Generates a unique `_id` for a data dict |
| `get_history_data(new_data_dict, match_keys, common_values_dict)` | Queries OpenSearch for the last 90 days of valid post-merge data, matches each history record to the correct `cmd_idx` based on `match_keys`. Returns `latest_history_data_dict` (most recent per cmd_idx) and `history_data_dict` (all entries per cmd_idx) |
| `post_new_perf_data(new_data_dict)` | Posts all entries to OpenSearch |

### Layer 2: `perf_regression_utils.py` — Generic Regression Pipeline

The reusable middle layer. Contains all business logic but is completely agnostic to what test produced the data. It takes metric lists as parameters rather than using hardcoded constants.

The main entry point is `process_and_upload_test_results()`, which orchestrates the full pipeline:

| Step | Function | Description |
|------|----------|-------------|
| 1 | `get_job_info()` | Reads CI environment variables (BUILD_URL, JOB_NAME, globalVars, etc.) to build a `job_config` dict with branch, commit, job URL, PR info, and `b_is_post_merge` flag |
| 2 | Enrich data | Merges `job_config` and `extra_fields` into each entry, then calls `add_id()` |
| 3 | `get_common_values()` | Scans all entries to find match_keys where every entry has the same value (e.g., all share `s_gpu_type=H100`). These become additional query filters |
| 4 | `get_history_data()` | Queries OpenSearch with the narrowed filters, matches results back to `cmd_idx` |
| 5 | `prepare_regressive_test_cases()` | Compares each metric's new value against baseline. Baseline comes from the latest history entry's embedded `d_baseline_*` field; if missing, falls back to `calculate_baseline_metrics()`. Sets `b_is_regression=True` if any regression metric exceeds the threshold |
| 6 | `add_baseline_fields_to_post_merge_data()` | Post-merge only: embeds `d_baseline_*` and `d_threshold_*` fields into new data so future runs can use them directly |
| 7 | `post_new_perf_data()` | Uploads to OpenSearch |
| 8 | `check_perf_regression()` | Prints regression details. For pre-merge, raises `RuntimeError` if `fail_on_regression=True` (default for pre-merge, auto-detected) |

**Baseline calculation** (`calculate_baseline_metrics`):
1. Group all data points by day, average within each day
2. Apply trailing rolling mean (window=3) to smooth noise
3. Take P95 for maximize-metrics (throughput) and P5 for minimize-metrics (latency)

**Regression detection** (`prepare_regressive_test_cases`):
- A metric is regressive if the new value breaches `baseline * (1 +/- threshold)`
- Default thresholds: **5%** for post-merge, **10%** for pre-merge (can be overridden per-metric via embedded `d_threshold_*` fields from history)

### Layer 3: Test-Specific Data Assembly

Each test script (e.g., `test_perf_sanity.py`) is responsible for:

1. Running benchmarks and collecting metric outputs
2. Assembling `new_data_dict` — a `Dict[int, dict]` mapping `cmd_idx` to data dicts
3. Building `match_keys` — the list of fields that uniquely identify a test case
4. Calling `process_and_upload_test_results()` with its own metric definitions

## How to Add a New Test Script That Reuses the Pipeline

A new test script (e.g., `test_module_perf.py`) needs to do three things: define metrics, assemble data, and call the pipeline.

### Step 1: Define Your Metric Lists

```python
# test_module_perf.py

from .perf_regression_utils import process_and_upload_test_results

# Which metrics does your test produce?
MAXIMIZE_METRICS = ["d_flops", "d_bandwidth"]           # larger = better
MINIMIZE_METRICS = ["d_latency_ms", "d_p99_latency_ms"] # smaller = better
REGRESSION_METRICS = ["d_flops", "d_latency_ms"]        # which ones gate pass/fail
```

### Step 2: Assemble `new_data_dict`

Build a `Dict[int, dict]` where each key is a sequential command index and each value is a flat dict with:

- Test config fields that identify this test case (e.g., `s_gpu_type`, `s_module_name`, `l_batch_size`). Use the `s_`, `l_`, `b_`, `d_` type prefixes to match OpenSearch field naming.
- Metric fields with `d_` prefix (e.g., `d_flops`, `d_latency_ms`).
- `s_test_case_name` — a human-readable name shown in regression reports.

```python
new_data_dict = {}
for idx, result in enumerate(test_results):
    new_data_dict[idx] = {
        # Test config (these become match keys)
        "s_gpu_type": gpu_type,
        "s_module_name": result.module_name,
        "l_batch_size": result.batch_size,
        "s_dtype": result.dtype,
        # Human-readable name
        "s_test_case_name": f"{result.module_name}-bs{result.batch_size}-{result.dtype}",
        # Metrics
        "d_flops": result.flops,
        "d_bandwidth": result.bandwidth,
        "d_latency_ms": result.latency_ms,
        "d_p99_latency_ms": result.p99_latency_ms,
    }
```

Do NOT include job config fields (`s_branch`, `s_job_url`, etc.) or call `add_id()` — the pipeline handles those automatically.

### Step 3: Define `match_keys`

List the fields that uniquely identify a test case for history lookup. Two data entries with the same `match_keys` values are considered the "same test" across runs.

```python
match_keys = ["s_gpu_type", "s_module_name", "l_batch_size", "s_dtype"]
```

### Step 4: Call the Pipeline

```python
process_and_upload_test_results(
    new_data_dict=new_data_dict,
    match_keys=match_keys,
    maximize_metrics=MAXIMIZE_METRICS,
    minimize_metrics=MINIMIZE_METRICS,
    regression_metrics=REGRESSION_METRICS,
    extra_fields={
        "s_stage_name": os.environ.get("stageName", ""),
        "s_test_list": my_test_labels,
    },
    upload_to_db=True,
    # fail_on_regression=None means auto:
    #   fail for pre-merge, warn for post-merge
)
```

### Parameters of `process_and_upload_test_results`

| Parameter | Type | Description |
|-----------|------|-------------|
| `new_data_dict` | `Dict[int, dict]` | Test config + metric values per test case |
| `match_keys` | `List[str]` | Fields that uniquely identify a test case |
| `maximize_metrics` | `List[str]` | Metrics where larger is better (baseline = P95) |
| `minimize_metrics` | `List[str]` | Metrics where smaller is better (baseline = P5) |
| `regression_metrics` | `List[str]` | Subset checked for pass/fail regression |
| `extra_fields` | `dict` or `None` | Additional fields merged into every entry |
| `upload_to_db` | `bool` | Whether to actually post to OpenSearch |
| `fail_on_regression` | `bool` or `None` | `None` = auto (fail pre-merge, warn post-merge) |

### What the Pipeline Handles for You

You don't need to worry about:
- Reading CI environment variables (job URL, branch, commit, PR info)
- Generating document IDs
- Querying history from OpenSearch
- Calculating baselines from historical data
- Detecting regressions against thresholds
- Embedding baseline fields into post-merge data for future runs
- Uploading to the database
- Failing the build on regression (pre-merge) vs. warning (post-merge)
