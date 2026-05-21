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

# Python Report Interface (ncu_report)

Programmatic access to Nsight Compute profiling reports via the `ncu_report` Python module.

## Setup

Located in `extras/python/` of the Nsight Compute installation. Requires Python 3.7+.

```python
import ncu_report
```

## Loading Reports

```python
context = ncu_report.load_report("report.ncu-rep")
# Also supports: .ncu-repz (compressed)
```

## Core Classes

### IContext — Report Container

```python
context = ncu_report.load_report("report.ncu-rep")
num_ranges = len(context)            # Number of ranges
range_obj = context[0]               # First range
for range_obj in context:            # Iterate ranges
    ...
```

### IRange — Execution Stream

Represents a CUDA stream or execution context containing ordered profiling actions.

```python
range_obj = context[0]
num_actions = len(range_obj)
action = range_obj[0]               # First action
for action in range_obj:            # Iterate actions
    ...

# Filter by NVTX
actions = range_obj.actions_by_nvtx(
    includes=["training/"],
    excludes=["warmup/"]
)
```

### IAction — Profiling Result

Represents one profiled kernel, range, or graph.

```python
action = range_obj[0]

# Basic info
name = action.name()                              # Kernel function name
wtype = action.workload_type()                     # kernel, graph, etc.

# Metrics
metric_names = action.metric_names()               # Tuple of all metric names
metric = action.metric_by_name("sm__throughput.avg.pct_of_peak_sustained_elapsed")
metric = action["sm__throughput.avg.pct_of_peak_sustained_elapsed"]  # Shorthand

# NVTX context
nvtx = action.nvtx_state()

# Analysis rules
rules = action.rule_results()
rules_dicts = action.rule_results_as_dicts()       # As Python dicts

# Source correlation
source = action.source_info(address)
ptx = action.ptx_by_pc(address)
sass = action.sass_by_pc(address)
```

### IMetric — Performance Measurement

```python
metric = action["gpu__time_duration.sum"]

# Value access
value = metric.value()               # Smart accessor (returns appropriate type)
string_val = metric.as_string()
uint_val = metric.as_uint64()
float_val = metric.as_double()

# Metadata
metric.name()                        # Metric identifier
metric.metric_type()                 # COUNTER, RATIO, THROUGHPUT, OTHER
metric.metric_subtype()              # Specialized classification
metric.unit()                        # e.g., "nanosecond", "percent"
metric.description()                 # Human-readable explanation
metric.rollup_operation()            # AVG, MAX, MIN, SUM

# Instanced metrics
metric.num_instances()               # Number of instances
metric.has_correlation_ids()         # Whether instances have IDs
metric.value(idx=0)                  # Value at specific instance
```

### INvtxState — NVTX Context

```python
nvtx = action.nvtx_state()
for domain_id in nvtx.domains():
    domain = nvtx.domain_by_id(domain_id)
    print(domain.name())
    for range_name in domain.push_pop_ranges():
        print(f"  Range: {range_name}")
```

### IRuleResult — Analysis Output

```python
for rule in action.rule_results():
    print(rule.name())
    print(rule.rule_identifier())
    if rule.has_rule_message():
        print(rule.rule_message())
    print(rule.focus_metrics())
    print(rule.speedup_estimation())
    for table in rule.result_tables():
        ...
```

## Enumerations

### MetricType

```python
ncu_report.MetricType_COUNTER
ncu_report.MetricType_RATIO
ncu_report.MetricType_THROUGHPUT
ncu_report.MetricType_OTHER
```

### RollupOperation

```python
ncu_report.RollupOperation_AVG
ncu_report.RollupOperation_MAX
ncu_report.RollupOperation_MIN
ncu_report.RollupOperation_SUM
```

### MsgType (Rule Messages)

```python
ncu_report.MsgType.OK
ncu_report.MsgType.OPTIMIZATION
ncu_report.MsgType.WARNING
ncu_report.MsgType.ERROR
```

### SpeedupType

```python
ncu_report.SpeedupType.LOCAL
ncu_report.SpeedupType.GLOBAL
```

## Usage Examples

### Extract SOL% for All Kernels

```python
import ncu_report

ctx = ncu_report.load_report("report.ncu-rep")
for rng in ctx:
    for action in rng:
        name = action.name()
        compute = action["sm__throughput.avg.pct_of_peak_sustained_elapsed"].as_double()
        memory = action["dram__throughput.avg.pct_of_peak_sustained_elapsed"].as_double()
        duration = action["gpu__time_duration.sum"].as_uint64()
        print(f"{name}: compute={compute:.1f}%, memory={memory:.1f}%, duration={duration}ns")
```

### Compare Kernels Across Runs

```python
import ncu_report

def get_kernel_metrics(report_path, kernel_regex):
    ctx = ncu_report.load_report(report_path)
    results = []
    for rng in ctx:
        for action in rng:
            if kernel_regex in action.name():
                results.append({
                    "name": action.name(),
                    "duration": action["gpu__time_duration.sum"].as_uint64(),
                    "compute_sol": action["sm__throughput.avg.pct_of_peak_sustained_elapsed"].as_double(),
                    "memory_sol": action["dram__throughput.avg.pct_of_peak_sustained_elapsed"].as_double(),
                })
    return results

baseline = get_kernel_metrics("baseline.ncu-rep", "my_kernel")
optimized = get_kernel_metrics("optimized.ncu-rep", "my_kernel")
```

### Extract Rule Recommendations

```python
import ncu_report

ctx = ncu_report.load_report("report.ncu-rep")
for rng in ctx:
    for action in rng:
        for rule in action.rule_results():
            if rule.has_rule_message():
                print(f"[{action.name()}] {rule.name()}: {rule.rule_message()}")
```

### Filter by NVTX Range

```python
import ncu_report

ctx = ncu_report.load_report("report.ncu-rep")
for rng in ctx:
    training_actions = rng.actions_by_nvtx(
        includes=["training/forward/"],
        excludes=[]
    )
    for action in training_actions:
        print(action.name())
```

## Report File Format

- `.ncu-rep` — Standard report (binary + Protocol Buffer)
- `.ncu-repz` — Compressed with zstd
- Proto definitions: `extras/FileFormat/` directory
