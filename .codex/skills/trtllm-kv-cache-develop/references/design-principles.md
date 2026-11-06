<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Design Principles for KV-Cache Features

These principles are the review bar for KV-cache work. Each one comes from a
change that was reworked in review. The incidents are summarized in
`lessons-learned.md`.

## 1. Start From the Behavior Contract

Write down what the feature changes and for whom before choosing a mechanism.

- What a user configures, and what they observe: outputs, memory, speed,
  errors.
- What stays exactly the same when the feature is off. The answer is
  "everything": outputs, memory footprint, KV budget, speed, error messages,
  and public API.
- Which combinations are supported, rejected, or not applicable. Use the
  `trtllm-kv-cache-compat-testing` table.
- How success will be shown: accuracy contract, performance hypothesis,
  activation evidence.

## 2. Survey Existing Mechanisms Before Building

Most KV-cache features are a new policy on top of machinery that already
exists. Before designing anything new, search for these and read them:

| Need | Existing mechanism to check first | Where on main |
|---|---|---|
| Keep some tokens and drop others by position | Sliding-window and sink handling in KVCM V2 layer groups and life cycles | Window and sink fields of the layer configuration built in `KVCacheManagerV2`; `block_reuse_config.swa_endpoint_rewind_tokens` |
| Keep a second copy of some tokens in another dtype | An extra layer group in KVCM V2 | `KVCacheManagerV2._add_mixed_precision_layers()` |
| Prefer keeping some pages under pressure | Retention priorities | Per-request retention configuration; `kv_cache_config.cold_tier_offload_min_priority`; `swa_endpoint_rewind_tokens` |
| Change what a later request may reuse | Commit timing and per-life-cycle reuse policies | `block_reuse_config.policy`; `ReusePolicy` on the layer specification |
| Attach per-layer side state, such as an indexer cache | KVCM V2 extra buffers | `_extra_buffers_per_layer()` in a manager subclass, as the sparse backends do; `get_disagg_role_mapper_kinds()` to transfer it |
| Run work at request or iteration boundaries | Lifecycle hooks | `KVCacheCompressionManager` in `resource_manager.py` |
| Transform pages that move to Host or Disk | The cold-page codec interface | `ColdPageQuantizationCompression` (`kv_cache_compression/quantization_for_cold_page/`) |
| Select or skip tokens inside attention | The sparse attention framework | `SparseAttentionConfig`; `attention/backends/sparse/registry.py` |
| Gather a different storage format before attention | A staging buffer filled in metadata preparation, with kernels unchanged | `prepare_mixed_precision_kv()` in `attention/backends/interface.py`; `kv_cache/mixed_precision_staging.py` |
| Move KV between context and generation workers | The Python cache transceiver and its layout resolution | `tensorrt_llm/_torch/disaggregation/`; the optional manager hooks `get_disagg_global_layer_ids` and `get_disagg_role_mapper_kinds`, read by the KV extractor |
| Observe cache behavior | KV-cache statistics and events | `enable_iter_perf_stats` with `LLM.get_stats()`; `LLM.get_kv_cache_events()` |
| Prove pages are never read before being written | KVCM debug fills | `TRTLLM_KV_FRESH_PAGE_FILL` (for example `nan`) and `TRTLLM_KV_GUARD_PAGE` |

Record what you found in the design gate. If a reviewer can name an existing
mechanism the design duplicates, the design is not ready.

## 3. Climb Down the Extension Ladder Only With a Reason

Pick the highest level that can express the feature. Each step down needs a
written reason why the level above cannot.

1. An existing configuration value or combination.
2. A new field on an existing configuration class.
3. A subclass or registration at an existing extension point: a compression
   manager, a codec provider, a sparse-attention method, a model-specific
   default.
4. A new code path in the feature's own Python package.
5. A change to shared Python core: KVCM V2, the executor, attention metadata.
6. A change to native code: C++ manager, bindings, kernels.

For level 5 or 6, list a layer-by-function table before editing: every shared
function you will change, why, and how the change stays neutral for other
users. A shared-core change must extend an algorithm-neutral contract, never
add a branch for one method or one model.

## 4. Keep Ownership Boundaries

- KVCM V2 owns pages, slots, pools, layer groups, mappings, migration,
  publication, reuse, and release. Features make decisions; KVCM performs the
  page operations.
- Attention backends consume the published active representation. A feature
  that changes storage adapts the representation before attention, typically
  through metadata and staging, rather than adding feature branches to
  kernels.
- Method-specific logic lives in the method's package and configuration. The
  runtime, KVCM, and C++ never learn a method's name.
- Do not route around an owner's component with a fallback because it lacks
  something you need. Extend its contract properly, with its owners
  reviewing.

## 5. Prefer the Executor-Side Layer Over Native Core Changes

The KVCM V2 core is C++ (`cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/`,
exposed through nanobind). Its executor-side wrapper
(`tensorrt_llm/_torch/pyexecutor/kv_cache/kv_cache_manager_v2.py`) decides
how the model's layers map onto layer groups, pools, life cycles, retention
priorities, and reuse policies. It also owns the extra buffers and staging
around them.

Configuration, that wrapper, admission checks, and existing callbacks are
cheaper to change, test, and review than the native core. Change the core
only when:
- the contract is algorithm-neutral, and
- the executor-side route is shown to be insufficient, for example by a
  profile of a hot path.

When you do, read the core's own `AGENTS.md` first. It lists the high-risk
areas: eviction ownership, block hashing, partial coverage, sliding-window and
sink handling, beam forks, and stream lifetime. Keep the change small, and
add a native test with it.

## 6. Fail Fast; Never Fall Back Silently

- Reject unsupported combinations early, with an actionable message: during
  configuration validation, or at executor construction before KVCM
  allocation when the check needs the device or the engine.
- Call external libraries and kernels directly. Do not probe versions or
  catch import errors to switch to a different path. A missing capability
  should fail loudly where it is first needed.
- After a feature has started mutating state, it must complete or fail
  according to the framework's completion contract. It must never fall back
  partway and leave visible state half-updated.
- Load calibration or other offline artifacts at initialization, and fail
  there if they are missing or malformed. Inference must never calibrate.

## 7. Preserve Algorithm Semantics

- Do not change an algorithm's selection rule, granularity, or tie-breaking
  to make an implementation easier. If a change is needed, propose it with
  measurements.
- Parallel execution must reproduce single-GPU semantics. A decision that
  depends on a tensor-parallel shard must be made consistently across ranks.
- Cleanup must not remove an algorithm mode, a supported model family, or a
  performance optimization. Simplifying means removing redundancy, not
  features. When asked to simplify, say what was removed and what was kept.

## 8. Names, Comments, and Text Describe Behavior

- Name things by what they do for the user or the caller, not by how they are
  implemented. Use `mixed_precision_high` and `mixed_precision_low` pools, not
  the name of the trick behind them.
- Comments explain the code as it stands to a future reader. Rationale,
  history, and references belong in the PR description.
- No internal jargon in code, comments, commit messages, docs, or PR text: no
  version labels for kernels, no milestone or branch nicknames, no section
  numbers from private notes. Use plain functional names a first-time reader
  understands.
- Keep text short. Do not restate the code; explain what is not obvious.

## 9. Treat Code Size as a Consideration, Not a Target

Concise code is a goal, but lines of code are not a hard limit. The size of a
change follows the requirement. What matters is how the code is added:

- **Before adding a feature, look for a home in what exists.** Can an existing
  mechanism carry it? Can it merge into the existing framework as a
  configuration, a parameter, a subclass, or a hook implementation? Write a
  new standalone implementation only when the answer is no, and say why.
- **Never remove features to make a change smaller.** Supported modes,
  configurations, model families, performance optimizations, and conversion
  paths stay. Requirements decide what is added. A smaller diff never
  justifies deleting something users rely on.
- **Read size as a signal.** A large diff for a small behavior usually means
  the design duplicates something that exists, or adds a layer the framework
  does not need. Investigate the design; do not cut functionality.
- **Test the decision.** When you reuse or merge into an existing mechanism,
  show with tests that it covers the requirement, including the combinations
  the feature claims. When the right home is unclear, prototype the options
  and compare them on behavior, performance, and change size.

## 10. Keep the Diff On Topic

- One concern per PR. Split preparatory refactors from behavior changes when
  either is large.
- Measure the diff after every fix round. If fixes keep growing it, stop and
  look for a design that fits the existing code better.
- Delete dead code, duplicate helpers, and unused parameters that the change
  makes obsolete. Do not leave a second implementation "for safety".
- Do not touch another component's files without saying why in the PR, and
  check with its owners when the change is not mechanical.
- Configuration changes follow the repository rules: regenerate the LLM-args
  golden manifest and the serve schemas, keep API-stability references
  current, and add field descriptions.

## 11. Innovate Deliberately

The extension ladder is a default, not a ban on new ideas. When a method
truly needs something the framework cannot express, extend the framework,
but on purpose and with evidence.

**When an extension is justified**
- The idea is valuable enough to measure: a prototype shows the accuracy,
  capacity, or speed gain on the target workload.
- Every rung of the ladder was tried and the reason it fails is written
  down. "It would be cleaner" is not a reason.
- The need is general: another method, on main or concretely planned, would
  use the same seam.

**How to extend**
1. **Prototype inside your own package**, behind your own configuration, and
   with no change to shared code where possible. Measure it first.
2. **Write the design down** in the PR or a design doc. Give the options
   (including "change nothing"), what each costs other owners, and what is
   being asked of them. Get the owners' decision before building.
3. **Design the seam to be algorithm-neutral.** It is a hook, a capability
   flag, a buffer role, a life-cycle property, or a codec interface that
   knows nothing about your method. It defaults to today's behavior.
4. **Land the seam on its own** when it is not trivial, with tests that prove
   existing behavior is unchanged. Then land the method on top of it.
5. **Remove scaffolding** that the seam makes unnecessary. Do not leave two
   ways to do the same thing.

**Examples on main**
- *Composition, no new seam.* The mixed-precision KV cache keeps its first-N
  and last-N copy as one extra sliding-window-with-sinks life cycle. It needed
  no new C++.
- *Extending a kernel.* Physical compaction reused the existing sparse
  compaction op, with a layer dimension added, instead of a new mover.
- *A new generic seam.* Cold-page compression added one codec interface to
  the KVCM V2 core. The core knows nothing about NVFP4; format logic lives in
  the provider.
- *A new generic core rule.* The `SINK_REQUIRED_ONLY` reuse policy was added
  to the C++ core as a general policy, because the existing policies could
  not say "sinks required, window optional". No method name entered the core.

**Rigor that does not change**
Feature-off invariance, fail-fast admission, the compatibility table, and
the accuracy and performance gates apply to innovative work exactly as to
routine work. Being new is a reason to measure more, not less.
