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

# Example Scenarios

## Host-Bound Workload (Aggregate)

```
GPU idle ratio: 42.1%           → >30% threshold  → CROSSED
Launch overhead: 12.0%          → >10% threshold  → CROSSED
Host prep exposed ratio (3a): 60%  → >50% threshold  → CROSSED
Host prep perf impact (3b): 15.0%  → >5% threshold   → CROSSED
Host prep idle attribution (3c): 85%→ >50% threshold  → CROSSED
GPU utilization: 57.9%          → <60% threshold  → CROSSED

Crossed: 6/6 → Verdict: YES (host overhead IS the bottleneck)
Host prep confirmed: YES (3b=15% AND 3c=85% both crossed)
```

## GPU-Bound Workload (Aggregate)

```
GPU idle ratio: 8.0%            → >30% threshold  → not crossed
Launch overhead: 2.0%           → >10% threshold  → not crossed
Host prep exposed ratio (3a): 10%  → >50% threshold  → not crossed
Host prep perf impact (3b): 0.5%   → >5% threshold   → not crossed
Host prep idle attribution (3c): 6% → >50% threshold  → not crossed
GPU utilization: 92.0%          → <60% threshold  → not crossed

Crossed: 0/6 → Verdict: NO (host overhead is NOT the bottleneck)
```

## Communication-Dominated Workload (Aggregate)

Shows why 3c (attribution) matters — GPU idle time is high but caused by NCCL, not host prep:

```
GPU idle ratio: 35.0%           → >30% threshold  → CROSSED
Launch overhead: 1.5%           → >10% threshold  → not crossed
Host prep exposed ratio (3a): 40%  → >50% threshold  → not crossed
Host prep perf impact (3b): 8.0%   → >5% threshold   → CROSSED
Host prep idle attribution (3c): 23%→ >50% threshold  → not crossed
GPU utilization: 65.0%          → <60% threshold  → not crossed
NCCL ratio: 45.0%               → >20% threshold  → CAVEAT

Crossed: 2/6 → Verdict: YES (but read caveat)
Host prep confirmed: NO (3c=23% — NCCL is the dominant idle cause, not host prep)
Caveat: NCCL communication accounts for 45% of GPU active time.
→ GPU idle gaps are primarily caused by communication stalls, not host overhead.
→ Even though 3b crossed, 3c shows host prep is only 23% of idle time.
→ Prioritize communication optimization over host prep optimization.
```

## Context-Only Bottleneck (Phase-Specific)

Aggregate metrics are below threshold, but per-phase analysis reveals context iterations are host-bound. This is the key scenario that per-phase analysis was designed to catch:

```
Aggregate: GPU idle 25%, utilization 75% → aggregate verdict: NO
  (generation iterations dilute the context-phase bottleneck)

Context phase (5 iterations):
  GPU idle ratio: 48.2%           → >30% → CROSSED
  GPU utilization: 51.8%          → <60% → CROSSED
  Launch overhead: 8.5%           → >10% → not crossed
  Host prep exposed ratio (3a): 62%  → >50% → CROSSED
  Host prep perf impact (3b): 22%    → >5%  → CROSSED
  Host prep idle attribution (3c): 88%→ >50% → CROSSED
  Phase verdict: YES (5/6 crossed)
  Host prep confirmed: YES

Generation phase (95 iterations):
  GPU idle ratio: 5.1%   → >15% → not crossed
  GPU utilization: 94.9% → <80% → not crossed
  Launch overhead: 0.8%  → >10% → not crossed
  Phase verdict: NO (0/3 crossed)

Overall verdict: YES (context phase elevated)
→ Optimize context-phase host preparation (_prepare_tp_inputs eager path)
→ Generation phase healthy — CUDA graphs working effectively
```

## Both Phases Bottlenecked

Both phases show host overhead — suggests CUDA graphs may be disabled or ineffective:

```
Context phase (10 iterations):
  GPU idle ratio: 45.0%  → >30% → CROSSED
  GPU utilization: 55.0% → <60% → CROSSED
  Phase verdict: YES (2/4 crossed)

Generation phase (90 iterations):
  GPU idle ratio: 32.0%  → >15% → CROSSED
  GPU utilization: 68.0% → <80% → CROSSED
  Launch overhead: 15.0% → >10% → CROSSED
  Phase verdict: YES (3/3 crossed)

Overall verdict: YES
→ Check if CUDA graphs are enabled (generation should not have high launch overhead)
→ If CUDA graphs are disabled, enable them first before optimizing host code
```

## Generation-Only Bottleneck (Unusual)

Context phase is healthy but generation shows host overhead — suggests CUDA graph issues:

```
Context phase (8 iterations):
  GPU idle ratio: 15.0%  → >30% → not crossed
  GPU utilization: 85.0% → <60% → not crossed
  Phase verdict: NO (0/4 crossed)

Generation phase (92 iterations):
  GPU idle ratio: 22.0%  → >15% → CROSSED
  GPU utilization: 78.0% → <80% → CROSSED
  Launch overhead: 12.0% → >10% → CROSSED
  Phase verdict: YES (3/3 crossed)

Overall verdict: YES (generation phase elevated)
→ CUDA graph replay may be failing (falling back to eager execution)
→ Investigate CUDA graph capture errors or dynamic shape changes
```
