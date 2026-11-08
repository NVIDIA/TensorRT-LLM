<!--
SPDX-FileCopyrightText: Copyright (c) 2011-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# nsys Post-Collection Commands

Commands for analyzing, exporting, and summarizing collected `.nsys-rep` reports.

## nsys stats — Statistical Summaries

Generates tabular statistical summaries from a collected report.

```bash
nsys stats [options] <input-file>
```

### Key Options

| Flag | Default | Description |
|------|---------|-------------|
| `--report` / `-r` | all default reports | Specific report(s) to generate |
| `--format` / `-f` | `column` | Output format: `column`, `table`, `csv`, `tsv`, `json`, `hdoc`, `htable` |
| `--output` / `-o` | `-` (stdout) | Output: `-` (console), `@<cmd>` (pipe), `<basename>` (file) |
| `--timeunit` | `nsec` | `nsec`, `usec`, `msec`, `seconds` |
| `--force-export` | `false` | Force SQLite re-export |
| `--force-overwrite` | `false` | Overwrite existing output files |
| `--quiet` / `-q` | off | Suppress verbose messages |

### Report Selection

```bash
# All default reports
nsys stats report.nsys-rep

# Specific CUDA reports
nsys stats -r cuda_api_sum,cuda_gpu_kern_sum report.nsys-rep

# CSV output for scripting
nsys stats -r cuda_gpu_kern_sum --format csv --output kern_summary report.nsys-rep

# JSON output with microsecond time units
nsys stats -r cuda_gpu_kern_sum --format json --timeunit usec report.nsys-rep
```

## nsys analyze — Expert System Rules

Runs rule-based analysis to detect known performance anti-patterns.

```bash
nsys analyze [options] <input-file>
```

### Key Options

| Flag | Default | Description |
|------|---------|-------------|
| `--rule` / `-r` | none | Rule(s) to run (comma-separated, or `all`) |
| `--format` / `-f` | `column` | Output format (same as stats) |
| `--output` / `-o` | `-` (stdout) | Output destination |
| `--timeunit` | `nsec` | Time unit |
| `--help-rules` | — | Show rule descriptions (`ALL` for all) |

### Available Rules

| Rule | Detects |
|------|---------|
| `cuda_memcpy_async` | Async memcpy using pageable memory (becomes synchronous) |
| `cuda_memcpy_sync` | Synchronous memory transfers blocking host |
| `cuda_memset_sync` | Synchronous memset operations |
| `cuda_api_sync` | Synchronization APIs blocking host |
| `gpu_gaps` | GPU idle periods |
| `gpu_time_util` | Low GPU utilization regions |

```bash
# Run all CUDA rules
nsys analyze -r cuda_memcpy_async,cuda_memcpy_sync,cuda_memset_sync,cuda_api_sync \
    report.nsys-rep

# Run all rules
nsys analyze -r all report.nsys-rep

# CSV output for further processing
nsys analyze -r gpu_gaps,gpu_time_util --format csv report.nsys-rep
```

## nsys export — Format Conversion

Exports `.nsys-rep` to other formats for external analysis.

```bash
nsys export [options] <nsys-rep-file>
```

### Key Options

| Flag | Default | Description |
|------|---------|-------------|
| `--type` / `-t` | `sqlite` | Export format |
| `--output` / `-o` | auto | Output filename |
| `--force-overwrite` / `-f` | `false` | Overwrite existing files |
| `--tables` | all | Export specific tables (pattern match) |
| `--times` | all | Export events in time range(s) |
| `--lazy` / `-l` | `false` | Only create tables with data |

### Export Formats

| Format | Description | Use case |
|--------|-------------|----------|
| `sqlite` | SQLite database (default) | Custom SQL queries, stats/analyze input |
| `arrow` | Apache Arrow columnar | Large-scale analysis, Pandas integration |
| `arrowdir` | Arrow directory | Multi-file Arrow output |
| `parquetdir` | Parquet directory | Spark/Dask analysis |
| `hdf` | HDF5 | Scientific computing (x86_64 Linux/Windows) |
| `jsonlines` | JSON Lines | Streaming processing |
| `text` | Plain text | Human-readable dump |

```bash
# Export to SQLite for custom queries
nsys export -t sqlite report.nsys-rep

# Export to Parquet for Dask/Pandas
nsys export -t parquetdir -o my_data report.nsys-rep

# Export only CUDA kernel tables
nsys export -t sqlite --tables "CUPTI_ACTIVITY_KIND_KERNEL*" report.nsys-rep

# Export specific time range (ns)
nsys export -t sqlite --times "1000000000,2000000000" report.nsys-rep
```

## nsys recipe — Advanced Analysis

Runs Python-based analysis recipes on one or more reports.

```bash
nsys recipe [args] <recipe-name> [recipe-args]
```

### Key Options

```bash
# List available recipes
nsys recipe --help

# Get help for a specific recipe
nsys recipe <recipe-name> --help

# Run a recipe
nsys recipe cuda_gpu_kern_sum -- report.nsys-rep

# Run with multiple input files (multi-node)
nsys recipe nccl_gpu_time_util_map -- report_rank0.nsys-rep report_rank1.nsys-rep
```

Recipes produce output directories containing CSV/Parquet data and Plotly visualizations. See `references/recipes-dl.md` for DL-specific recipes.

## nsys launch / start / shutdown — Interactive Sessions

For interactive profiling with separate launch and collection control.

### Workflow

```bash
# Step 1: Launch app in profiling-ready state
nsys launch --session-new my_session -t cuda,nvtx -- python train.py

# Step 2: Start collection (from another terminal)
nsys start --session my_session

# Step 3: Stop and generate report
nsys shutdown --session my_session
```

### nsys sessions — List Active Sessions

```bash
nsys sessions list                    # Plain text
nsys sessions list -f json            # JSON format
```

## Common Post-Collection Workflows

### Quick DL Profile Summary

```bash
# Profile with auto-stats
nsys profile --stats=true -t cuda,nvtx,cudnn,cublas -o dl_profile -- python train.py

# Or generate stats from existing report
nsys stats -r cuda_gpu_kern_sum,cuda_api_sum,cuda_gpu_mem_time_sum dl_profile.nsys-rep
```

### Detect Performance Issues

```bash
# Run expert system on existing report
nsys analyze -r all dl_profile.nsys-rep
```

### Export for Custom Analysis

```bash
# SQLite for SQL queries
nsys export -t sqlite dl_profile.nsys-rep

# Then query directly
sqlite3 dl_profile.sqlite "SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY end - start DESC LIMIT 10;"
```
