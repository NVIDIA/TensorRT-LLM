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

# Model test policy

Scope: end-to-end model tests (accuracy, disaggregated serving, model smoke and
model perf-sanity cases) registered in `test-db/` and the QA lists. This policy
decides which models, configurations and platforms run in pre-merge,
post-merge or QA, and how to keep each pre-merge entry fast. The general
placement questions in [tests/AGENTS.md](../../AGENTS.md) still apply.

## Measure before deciding

- The weekly cost of an entry is its wall time × the GPUs of its stage × the
  number of runs per week. The CI test records hold all three: test name,
  stage name, end-to-end time and GPU count per run.
- The value of an entry is the regression it catches that no cheaper entry
  catches. An entry that never fails, costs a lot and duplicates coverage
  elsewhere is the first candidate to leave pre-merge.

## Which models run in pre-merge

1. Models on the product priority list. Within a model family, keep the newest
   version; variants with different architectures (for example a dense and a
   flash variant) count separately.
2. Older versions move to post-merge while deployments still use them, and are
   deleted otherwise. Before an older model leaves pre-merge, move the features
   that only its entries cover onto the newest model.
3. A small vehicle model (such as DeepSeek-V3-Lite) may carry feature coverage
   for its architecture, under the one-representative rule below.

## Which configurations

Each priority model covers these in pre-merge; one entry may cover several:

| Configuration | Pre-merge | Why |
|---|---|---|
| Typical deployment (precision, parallelism, speculative decoding, platform) | Required | A regression here reaches users directly |
| Attention DP + expert parallelism, with speculative decoding when the model ships it | Required | Rank imbalance, padding, dummy requests and load balancing break most often here |
| Disaggregated serving (context TEP, generation DEP, NIXL) | Required | KV transfer and cache-manager paths are separate from aggregated serving |
| Tensor parallelism alone | Optional | Simple path; move it to post-merge when other entries cover it |

Full feature matrices (tactic × parallelism × KV dtype × scheduler options) run
in post-merge. Pre-merge keeps one representative case per feature.

## Which platforms

- Prefer the newest platforms (GB300, GB200, B300) and NVFP4. Keep Hopper
  entries for Hopper-only kernels.
- Run a case on one GPU type when its kernels are shared across SMs (sm_100f).
  Keep both only when separate kernels ship per SM (sm_100a and sm_103a).
- Match the stage to the test: a single-GPU case does not belong in a
  multi-GPU stage, where the idle GPUs are paid for as well.

## Keep each entry near the hardware limit

The floor for an accuracy entry is roughly weight loading plus evaluation.
Startup usually dominates instead:

- On Blackwell, cold JIT builds (DeepGEMM, FlashInfer, CuTe DSL, Triton) and
  autotuning can take several times longer than the evaluation itself. Group
  a model's entries in one stage so they share the caches, and do not disable
  autotuning in entries that guard the deployment configuration.
- Bound CUDA graph capture to the evaluation's concurrency, for example
  `CudaGraphConfig(max_batch_size=..., enable_padding=True)`; the default
  captures a graph for every size up to the default maximum batch size.
- Cover pipeline logic (DEP, KV transfer, speculative-decoding acceptance) with
  layer-truncated models and logits parity, which finish in minutes. Keep
  full-weight accuracy for the typical configuration.
- Size evaluation samples from the hypothesis test in
  `defs/accuracy/accuracy_core.py`: pre-merge needs to catch breakage, while
  post-merge with the full sample catches drift.
- Budget per run: up to 1 GPU-hour for a 4-GPU entry and 2.5 GPU-hours for an
  8-GPU disaggregated entry. Explain anything above that in the PR.

## Process

- A PR that adds a pre-merge model entry states the model or feature, the
  configuration it covers, its measured time × GPUs, and why post-merge is not
  enough.
- Before moving an entry to post-merge, confirm that a running stage selects
  the destination block: `scripts/test_to_stage_mapping.py` does not check GPU
  counts, so also check that the block's existing entries ran recently.
- Review the lists periodically for: entries of models off the priority list,
  the same case on several GPU types, single-GPU cases in multi-GPU stages,
  costly entries without real failures, and entries that fail often for
  infrastructure reasons.
- When post-merge catches a regression that a pre-merge entry would have
  caught, move that entry, or a cheaper equivalent, back to pre-merge.
