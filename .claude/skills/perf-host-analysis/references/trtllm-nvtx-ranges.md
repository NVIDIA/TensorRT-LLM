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

# TRT-LLM NVTX Range Reference

## Overview

TRT-LLM PyTorch backend instruments the inference loop with NVTX ranges for
timeline analysis. These ranges are visible in nsys traces when profiled with
`-t nvtx` (or `-t cuda,nvtx,osrt`).

**Important**: NVTX event text is stored as `textId` (integer) in the
`NVTX_EVENTS` table, referencing the `StringIds` table. Always join:

```sql
SELECT s.value, n.start, n.end, (n.end - n.start)/1000.0 AS dur_us
FROM NVTX_EVENTS n
JOIN StringIds s ON n.textId = s.id
WHERE n.end > 0
ORDER BY n.start
```

## Executor Loop Ranges

### [Executor] _forward_step N: X ctx reqs, Y gen reqs

The top-level forward step marker. Contains:
- `N`: Step number (monotonically increasing)
- `X ctx reqs`: Number of context/prefill requests in this step
- `Y gen reqs`: Number of generation/decode requests in this step

In TP configurations, one range per TP rank (de-duplicate by grouping
entries within 100us of each other).

**Duration**: 500us (small batch, gen only) to 18ms (large batch, mixed ctx+gen)

### Lifecycle within each step

```
[Executor] _forward_step N
  ├── _schedule              (~100-200 us)
  ├── _fetch_new_requests    (~30-300 us, version-dependent)
  ├── broadcast_requests     (~250 us, TP only, newer versions)
  ├── _prepare_inputs        (~500-900 us)
  ├── [GPU execution]        (model forward pass)
  ├── _sample_async          (~700-1200 us)
  ├── _process_requests      (~400-1100 us)
  ├── _update_requests       (~400-700 us)
  ├── _handle_responses      (~300-400 us)
  ├── _write_finish_reasons  (~120-200 us, newer versions)
  └── prepare_resources      (~170-350 us)
```

## Per-Operation Reference

### _schedule
Determines which requests are ready for the next forward step.
Typical: 100-200 us. Usually stable across versions.

### _fetch_new_requests
Fetches newly arrived requests from the request queue.
- v1.1: ~30-60 us (lightweight queue poll)
- main (newer): ~250-300 us (includes validation, state setup)
**Known regression point**: Refactored in recent versions with significantly
higher overhead in steady state.

### broadcast_requests (NEW in recent versions)
Broadcasts request state to all TP ranks. Only present in TP >= 2.
Not present in v1.1 or earlier versions.
Typical: 200-300 us.
**This is a common source of regression** when upgrading from older versions.

### _prepare_inputs
Prepares model input tensors (token IDs, position IDs, attention mask).
Typical: 500-900 us. Scales with batch size.

### _sample_async
Runs the sampling operation (greedy, top-k, top-p, beam search).
- v1.1: ~1,100-1,200 us
- main (newer): ~700-900 us (improved with fast_greedy_sample_kernel)
**Known improvement point**: Recent versions optimized greedy sampling.

### _process_requests
Post-processing of request state after sampling.
- v1.1: ~1,000-1,100 us
- main (newer): ~400-500 us (improved)
**Known improvement point**: Significant optimization in recent versions.

### _update_requests
Updates request states (token counts, finished status, KV cache pointers).
- v1.1: ~330-430 us
- main (newer): ~620-730 us (increased bookkeeping)
**Known regression point**: More bookkeeping in newer versions.

### _handle_responses
Processes completed requests and queues responses.
Typical: 300-420 us. Moderate increase in newer versions.

### _write_finish_reasons (NEW in recent versions)
Writes finish reason metadata for completed requests.
Not present in v1.1. Typical: 120-200 us.

### prepare_resources
Allocates or reclaims KV cache blocks and other GPU resources.
- v1.1: ~170-200 us
- main (newer): ~330-350 us (increased, more resource tracking)

### fast_greedy_sample_kernel (NEW in recent versions)
Optimized GPU kernel for greedy sampling.
Not present in v1.1. Typical: 50-100 us.
Replaces part of the Python-based sampling in _sample_async.

### _fetch_and_activate_new_requests (REMOVED in recent versions)
Combined fetch-and-activate in v1.1. Replaced by separate
_fetch_new_requests + broadcast_requests in newer versions.
Typical in v1.1: ~30-40 us.

## Version Differences Summary

| Operation | v1.1 | Recent main | Change |
|-----------|------|-------------|--------|
| _fetch_new_requests | ~36 us | ~270 us | +643% (REGRESSION) |
| broadcast_requests | — | ~250 us | NEW |
| _update_requests | ~413 us | ~723 us | +75% (REGRESSION) |
| prepare_resources | ~192 us | ~349 us | +81% (REGRESSION) |
| _write_finish_reasons | — | ~121 us | NEW |
| fast_greedy_sample_kernel | — | ~52 us | NEW |
| _sample_async | ~1,163 us | ~720 us | -38% (IMPROVED) |
| _process_requests | ~1,056 us | ~390 us | -63% (IMPROVED) |
| _schedule | ~121 us | ~120 us | ~0% (UNCHANGED) |

*Values from Llama 3.2 1B, TP=2, 500 gen reqs steady state on H200.*

## Batch Size Scaling

Forward step duration scales with batch size (ctx + gen requests):

| Batch Size | Typical Step Duration | Notes |
|------------|----------------------|-------|
| 1-2 reqs | 500-600 us | Minimal, mostly host overhead |
| 50-100 reqs | 1,000-1,500 us | |
| 200-300 reqs | 8,000-12,000 us | |
| 400-500 reqs | 12,000-16,000 us | Near max batch |

Context requests (prefill) are more expensive than generation (decode) because
they process full input sequences rather than single tokens.

## Tips for Analysis

1. **Filter by step number** to isolate specific phases:
   - Low step numbers (< 10): warmup
   - Steps with ctx > 0: ramp-up phase
   - Steps with ctx = 0, gen = max: steady state

2. **Compare same batch sizes**: A step with 500 gen reqs is not comparable
   to one with 50 gen reqs. Always filter to matching workload.

3. **Watch for TP duplication**: In TP=2, each NVTX range appears twice
   (one per rank). Divide counts by TP degree or de-duplicate by proximity.

4. **Inter-step gap is between _forward_step end and next start**:
   ```
   gap = next_step.start - prev_step.end
   ```
   This captures all host work between steps.
