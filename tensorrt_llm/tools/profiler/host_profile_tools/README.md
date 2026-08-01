<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Host Profiler

Line-by-line CPU profiler for diagnosing host-side overhead in the
**PyExecutor** event loop. Built on top of
[line_profiler](https://github.com/pyutils/line_profiler), it measures
wall-clock time per source line of selected Python functions, making it easy to
pinpoint where the executor spends time between GPU kernel launches.

---

## Quick Start

### 1. Install `line_profiler`

```bash
pip install line_profiler
```

### 2. Run with the environment variable

Set `TLLM_LINE_PROFILER_PATH` to an output file path. The profiler activates
automatically inside the PyExecutor worker thread:

```bash
TLLM_LINE_PROFILER_PATH=./lp_results.txt \
    python my_inference_script.py
```

When the executor shuts down, results are written to the specified file.

### 2b. Profile specific iterations only (recommended)

Use `TLLM_PROFILE_START_STOP` (the same env var used by nsys/cudaProfiler) to
restrict line profiling to specific iteration ranges. This avoids warmup noise
and reduces overhead:

```bash
TLLM_PROFILE_START_STOP=10-20 \
TLLM_LINE_PROFILER_PATH=./lp_results.txt \
    python my_inference_script.py
```

Multiple ranges are supported: `"10-20,50-60"`. When this env var is **not**
set, profiling covers the entire event loop.

### 3. Read the results

The output follows `line_profiler`'s standard format — one section per
function, with per-line hit counts, total time, and percentage:

```
Timer unit: 1e-06 s

Total time: 0.012345 s
File: tensorrt_llm/_torch/pyexecutor/py_executor.py
Function: _forward_step at line 123

Line #   Hits   Time  Per Hit  % Time  Line Contents
=======  =====  =====  =======  ======  =============
   123       5   1234    246.8    10.0      def _forward_step(self):
   124       5    456     91.2     3.7          inputs = self._prepare(...)
   ...
```

---

## Registering Functions to Profile

There are **two** ways to tell the profiler which functions to trace.

### Method 1: Static Config Dict (Built-in Defaults)

The module ships with a built-in `_DEFAULT_PROFILE_CONFIG` dictionary that
registers critical PyExecutor methods by default:

```python
# In host_profiler.py
_DEFAULT_PROFILE_CONFIG = {
    "tensorrt_llm._torch.pyexecutor.py_executor": {
        "PyExecutor": [
            "_prepare_and_schedule_batch",
            "_schedule",
            "_forward_step",
            ...
        ],
    },
    "tensorrt_llm._torch.pyexecutor.sampler": {
        "TorchSampler": ["sample_async", "update_requests", ...],
        None: ["_group_requests_by_strategy_key"],  # standalone functions
    },
}
```

**Wildcard support:**

| Pattern | Meaning |
|---------|---------|
| `{"ClassName": ["*"]}` | All methods of `ClassName` |
| `{None: ["*"]}` | All standalone functions in the module |
| `{"*": ["*"]}` | All classes + all their methods in the module |

### Method 2: Environment Variable (Ad-hoc, No Code Changes)

Set `TLLM_LINE_PROFILER_FUNCTIONS` to a comma-separated list of paths.
When set, this **replaces** the default targets — only the specified functions
are profiled:

```bash
# Class method format: module.Class.method
# Standalone function format: module::function

TLLM_LINE_PROFILER_FUNCTIONS="tensorrt_llm._torch.pyexecutor.py_executor.PyExecutor.event_loop,tensorrt_llm._torch.pyexecutor.sampler::_group_requests_by_strategy_key" \
TLLM_LINE_PROFILER_PATH=./results.txt \
    python my_script.py
```

This is ideal for one-off investigations — no code changes required.

> **Note:** If `TLLM_LINE_PROFILER_FUNCTIONS` is *not* set, the built-in
> default targets (PyExecutor, sampler, scheduler, etc.) are used automatically.

---

## Programmatic API

For scripts or tests, you can use the `HostProfiler` class directly:

```python
from tensorrt_llm.tools.profiler.host_profile_tools import HostProfiler

# Create a profiler (defaults include the standard PyExecutor targets)
profiler = HostProfiler(output_path="./results.txt")

# Optionally add more targets
profiler.add_function(
    module_path="my.module",
    class_name="MyClass",
    method_name="my_method",
)
profiler.add_standalone_function("my.module", "helper_func")

# Profile using a context manager
with profiler.profile():
    run_inference()

# Or start/stop manually
profiler.start()
run_inference()
profiler.stop()
```

### Useful methods

| Method | Description |
|--------|-------------|
| `profiler.add_function(module, cls, method)` | Add a class method target |
| `profiler.add_standalone_function(module, func)` | Add a module-level function target |
| `profiler.clear_targets()` | Remove all targets (including defaults) |
| `profiler.list_targets()` | List all registered target paths |
| `profiler.get_stats_string()` | Get stats as a string (while profiling is active) |

### Global profiler instance

The executor sets up a global profiler instance accessible anywhere:

```python
from tensorrt_llm.tools.profiler.host_profile_tools import get_global_profiler

profiler = get_global_profiler()
if profiler is not None and profiler.enabled:
    print(profiler.list_targets())
```

---

## How It Works

1. **Registration** — Target functions are collected from the static config
   dict or the environment variable (which replaces the defaults when set).

2. **Unwrapping** — Decorated functions (e.g., `@torch.inference_mode`) are
   unwrapped via `__wrapped__` to obtain the original function's `__code__`
   object. `line_profiler` traces by `__code__` identity, so this step is
   critical.

3. **Profiling** — `line_profiler.LineProfiler` is enabled on the executor
   worker thread. It hooks into Python's tracing mechanism to record per-line
   timing for the registered `__code__` objects.

4. **Output** — When `stop()` is called (or the context manager exits),
   results are written to the output file.

### Integration with PyExecutor

The profiler is wired into the executor's event loop via `host_profiler_context`:

```python
# In py_executor.py → _event_loop_wrapper()
with host_profiler_context(enable=enable_profiler):
    self.event_loop()
```

This ensures profiling covers the entire event loop lifetime and skips the
warmup phase (which would produce incomplete/misleading results).

---

## Tips

- **Overhead**: `line_profiler` adds measurable overhead (~2–5x slowdown on
  profiled functions). Only profile the functions you care about.
- **Thread affinity**: The profiler only traces the thread it is started on
  (the executor worker thread). Other threads are not affected.
- **Multiple ranks**: Debug output (`dump_profiler_functions`) only prints on
  rank 0 to avoid interleaved logs in multi-GPU runs.
- **Override vs defaults**: Use `TLLM_LINE_PROFILER_FUNCTIONS` to replace
  the defaults with your own set. Omit it to use the built-in defaults.
