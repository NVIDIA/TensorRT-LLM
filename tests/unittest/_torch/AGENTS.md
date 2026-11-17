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

# AGENTS.md — PyTorch unit tests

Scope: this directory and its descendants. Supplements `../../AGENTS.md`.

## GPU test cost and coverage

- A deprecated configuration alias does not imply that its execution backend is
  deprecated. Trace argument normalization and dispatch before removing coverage.
- Check effective CI coverage after platform gates and waivers. A registered but
  skipped case cannot replace a removed check.
- Distinct parameters are not by themselves a reason to keep every cross-product
  combination. State the extra regression a costly case protects, check existing
  lower-level and integration coverage, and retain representatives for independent
  mechanisms and known interactions. Record the combinations deliberately removed.
- Merge related assertions into an existing model load when they share the same
  setup. Keep real checkpoint and numerical checks; avoid loading a complete model
  again to recheck a derived dtype/parameter-size property. Preserve useful scalar
  summaries when merging checks, and keep a small reference tensor or layer
  rather than retaining an otherwise unused full model.
- Separate setup, compilation, reference computation and device execution when
  profiling a test. Use the stage's effective worker count and allocation; serial
  timings alone do not establish the cost of a parallel CI batch.
- Measure fresh-container and warm-cache runs separately. A before/after sequence
  in one container can attribute shared JIT or model-cache warmup to a test change.
  Include cache preparation cost when moving compilation into a shared image.
- Distinguish a Triton runtime dimension from a constexpr specialization. When
  thinning a shape matrix, retain representative tile sizes, mask boundaries,
  dtypes and state transitions; name the specializations deliberately removed.
- Batch tensor comparisons and host/device transfers when they check the same
  elements. Preserve valid-row masks, padding, packed values, scale factors,
  ordering, dtypes and tolerances. Keep the reference independent of the operation
  under test, and retain device, aliasing or stream checks when those are part of
  its contract.
- Exclude only positions that the test contract masks. Do not drop non-finite
  values from both actual and reference tensors: that can turn invalid outputs
  into a passing comparison. Check the required finite domain explicitly.
- Put validation readbacks after the required completion point and outside the
  timed or no-synchronization region. Optimizing the checker must not weaken an
  asynchronous-execution assertion or add synchronization to the operation being
  tested.
- Check that a changed reference produces the same values on representative
  layouts, and that existing assertions still reject relevant corruptions, such
  as a bad last row, an altered scale or a write to an unused slot. Keep temporary
  investigation probes out of CI unless they add distinct regression protection.
- Inspect fixture consumers before calling a parameter unused. For a redundant
  Cartesian-product axis, compare complete parameter bindings and markers rather
  than counts or test names alone. Keep each meaningful input combination; remove
  only repeated work that adds no separate check.
- Include peak memory and retry behavior when evaluating GPU parallelism or a
  batched reference. More workers can increase allocation pressure and import/JIT
  work; a larger batch can need more temporary device memory. Inspect failed
  parallel-phase reports even when serial retries make the final result pass.
