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

# Phase Classification

TRT-LLM batches interleave **context (prefill)** and **generation (decode)** iterations. These have fundamentally different host overhead characteristics, and aggregate metrics can mask phase-specific bottlenecks. For example, a workload might show 25% GPU idle overall (below the 30% threshold → NO verdict), but context iterations alone might have 50% GPU idle (clearly host-bound) while generation iterations have 5% (healthy).

## Iteration NVTX Markers

Each `_forward_step` iteration is wrapped with an NVTX range at `py_executor.py` encoding the request composition:

```
[Executor] _forward_step {iter}: {N} ctx reqs, {M} gen reqs
```

Parse the NVTX marker text to classify each iteration:

| Phase | Condition | Characteristics |
|-------|-----------|-----------------|
| Context | `N > 0` (any ctx reqs present) | Eager execution, no CUDA graphs, heavier `_prepare_tp_inputs` |
| Generation-only | `N == 0, M > 0` | CUDA graph replay (if enabled), minimal host prep |

**Regex for extraction**: `_forward_step\s+\d+:\s+(\d+)\s+ctx reqs,\s+(\d+)\s+gen reqs`

- If captured group 1 (`N`) > 0 → **context** iteration (may also contain generation requests)
- If captured group 1 (`N`) == 0 and group 2 (`M`) > 0 → **generation-only** iteration

## Per-Iteration Extraction

Extract `_forward_step` NVTX ranges from the nsys SQLite trace:

```sql
SELECT s.value, n.start, n.end, (n.end - n.start)/1000.0 AS dur_us
FROM NVTX_EVENTS n
JOIN StringIds s ON n.textId = s.id
WHERE n.end > 0 AND s.value LIKE '%[Executor] _forward_step%'
ORDER BY n.start;
```

Each row is one iteration. Parse the NVTX text to extract the step number and request counts.

## Classify Iterations into Phases

Parse each iteration's NVTX marker to assign it to a phase:

```python
import re

ITER_PATTERN = re.compile(
    r'_forward_step\s+\d+:\s+(\d+)\s+ctx reqs,\s+(\d+)\s+gen reqs'
)

context_iterations = []
generation_iterations = []

for marker_text, stats in per_iteration_stats.items():
    m = ITER_PATTERN.search(marker_text)
    if not m:
        continue
    n_ctx, n_gen = int(m.group(1)), int(m.group(2))
    if n_ctx > 0:
        context_iterations.append(stats)
    elif n_gen > 0:
        generation_iterations.append(stats)
```

## Aggregate Per-Phase Metrics

Sum the timing components across iterations in each phase:

```
For each phase (context_iterations, generation_iterations):
    phase_total_time_us    = SUM(iter.total_time_us)
    phase_gpu_active_us    = SUM(iter.gpu_active_time_us)
    phase_gpu_idle_us      = SUM(iter.gpu_idle_time_us)

    phase_gpu_idle_ratio   = phase_gpu_idle_us / phase_total_time_us
    phase_gpu_utilization  = phase_gpu_active_us / phase_total_time_us
```

## Per-Phase Verdict Logic

For each phase (context, generation) that has iterations present, compute metrics using only that phase's aggregated times and apply **phase-specific thresholds** (see [thresholds.md](thresholds.md)):

```
For each phase in [context, generation]:
    if phase has no iterations → skip

    compute M1 (gpu_idle_ratio) using phase-aggregated times
    compute M4 (gpu_utilization) using phase-aggregated times
    # M2 (launch overhead) and M3 (host prep exposed) use aggregate values
    # unless per-iteration NVTX breakdown is available

    apply phase-specific thresholds
    phase_crossed_count = count of metrics that cross phase thresholds
    phase_applicable_count = count of applicable metrics

    if phase_crossed_count >= 2:
        phase_verdict = YES
    else:
        phase_verdict = NO
```
