<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Design Principles for KV Cache Manager V2 and KV Cache Compression Work

These principles are the review bar for work on KV cache manager V2 and on
the KV cache compression framework. Each one comes from a change that was
reworked in review. The incidents are summarized in
`lessons-learned.md`.

## The Goal: A Working Feature on Every Model Family

Everything below serves one goal: a new feature that works.

- **Get the whole path working first.** Wire the configuration, the
  managers, attention metadata, executor callbacks, and outputs, and run the
  feature end to end on a real model in the production configuration. A
  feature that is not wired in, or that runs only in a unit test, is not a
  feature yet.
- **Then make it work on every model family.** Dense models with multi-head
  or grouped-query attention, multi-head latent attention (MLA) models,
  sparse-attention models such as DeepSeek-V4, sliding-window and hybrid
  models, hybrid recurrent models, multimodal models, and the draft layers
  of speculative decoding. Each family is either supported, with an
  end-to-end run, or rejected at configuration time with the reason and a
  follow-up. Support for one family is not support for all, and the same
  holds for GPU architectures: a run on B200 does not show that B300,
  GB300, or Hopper GPUs work.
- **The other principles are how to get there well.** Reusing mechanisms,
  keeping the diff small, and clean naming make the feature easier to
  review and maintain. They never justify a feature that does less, covers
  fewer models, or is not finished.

## 1. Start From the Behavior Contract

Write down what the feature changes and for whom before choosing a mechanism.

- What a user configures, and what they observe: outputs, memory, speed,
  errors.
- What stays exactly the same when the feature is off. The answer is
  "everything": outputs, memory footprint, KV cache memory budget, speed,
  error messages, and public API.
- Which combinations are supported, rejected, or not applicable. Use the
  table in `../../trtllm-kv-cache-compat-testing/SKILL.md`.
- How success will be shown: accuracy contract, performance hypothesis,
  activation evidence.

## 2. Know What Already Exists

Many features are a new policy on top of machinery that already exists.
Spend a short time finding what you can build on, and use what fits. Where
nothing fits, write the new code: the survey shapes the design, it does not
stop the feature. Mechanisms to check:

| Need | Existing mechanism to check first | Where on main |
|---|---|---|
| Keep some tokens and drop others by position | Sliding-window and sink-token handling in KV cache manager V2 layer groups and life cycles | Sliding-window and sink-token fields of the layer configuration built in `KVCacheManagerV2`; `block_reuse_config.swa_endpoint_rewind_tokens` |
| Keep a second copy of some tokens in another dtype | An extra layer group in KV cache manager V2 | `KVCacheManagerV2._add_mixed_precision_layers()` |
| Prefer keeping some pages under pressure | Retention priorities | Per-request retention configuration; `kv_cache_config.cold_tier_offload_min_priority`; `swa_endpoint_rewind_tokens` |
| Change what a later request may reuse | Commit timing and per-life-cycle reuse policies | `block_reuse_config.policy`; `ReusePolicy` on the layer specification |
| Attach per-layer side state, such as an indexer cache | KV cache manager V2 extra buffers | `_extra_buffers_per_layer()` in a KV cache manager V2 subclass, as the sparse backends do; `get_disagg_role_mapper_kinds()` to transfer it |
| Run work at request or iteration boundaries | The KV cache compression manager's lifecycle hooks | `KVCacheCompressionManager` in `resource_manager.py` |
| Transform pages that move to Host or Disk | The cold-page codec interface | `ColdPageQuantizationCompression` (`kv_cache_compression/quantization_for_cold_page/`) |
| Select or skip tokens inside attention | The sparse attention framework | `SparseAttentionConfig`; `attention/backends/sparse/registry.py` |
| Gather a different storage format before attention | A staging copy filled during attention metadata preparation, with kernels unchanged, as the mixed-precision staging copy does | `prepare_mixed_precision_kv()` in `attention/backends/interface.py`; `kv_cache/mixed_precision_staging.py` |
| Move the KV cache between context and generation workers | The Python cache transceiver and its layout resolution | `tensorrt_llm/_torch/disaggregation/`; the optional KV cache manager V2 hooks `get_disagg_global_layer_ids` and `get_disagg_role_mapper_kinds`, read in `disaggregation/resource/kv_extractor.py` |
| Observe cache behavior | KV-cache statistics and events | `enable_iter_perf_stats` with `LLM.get_stats()`; `LLM.get_kv_cache_events()` |
| Prove pages are never read before being written | KV cache manager V2 debug fills | `TRTLLM_KV_FRESH_PAGE_FILL` (for example `nan`) and `TRTLLM_KV_GUARD_PAGE` |

Record what you found in the design gate. If a reviewer can name an existing
mechanism the design duplicates, the design is not ready.

## 3. Put the Change Where the Behavior Belongs

The levels below form the extension ladder. Higher levels are cheaper to
review, so prefer them when they can express the feature. When the behavior
belongs lower, go there and write down why. Do not bend the feature to stay
high, and do not add a flag, a wrapper, or a parallel path to avoid changing
the code that owns the behavior.

1. An existing configuration value or combination.
2. A new field on an existing configuration class.
3. A subclass or registration at an existing extension point: a KV cache
   compression manager, a cold-page codec provider, a sparse-attention
   method, a model-specific default.
4. A new code path in the feature's own Python package.
5. A change to shared Python runtime code: the KV cache manager V2
   executor-side wrapper, the executor, attention metadata.
6. A change to native code: the KV cache manager V2 native core, bindings,
   kernels.

For level 5 or 6, list a layer-by-function table before editing: every shared
function you will change, why, and how the change stays neutral for other
users. A change to shared runtime code or native code must extend an
algorithm-neutral contract, never add a branch for one method or one model.

## 4. Keep Ownership Boundaries

- KV cache manager V2 owns pages, slots, pools, layer groups, mappings,
  migration, publication, reuse, and release. Features make decisions;
  KV cache manager V2 performs the page operations.
- Attention backends consume the published active representation. A feature
  that changes storage adapts the representation before attention, typically
  through attention metadata and a staging copy (as the mixed-precision KV
  cache does), rather than adding feature branches to kernels.
- Method-specific logic lives in the method's package and configuration. The
  runtime, KV cache manager V2, and C++ never learn a method's name.
- Do not route around an owner's component with a fallback because it lacks
  something you need. Extend its contract properly, with its owners
  reviewing.

## 5. Prefer the Executor-Side Layer Over Native Core Changes

The KV cache manager V2 native core is C++
(`cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/`, exposed through
nanobind). Its executor-side wrapper
(`tensorrt_llm/_torch/pyexecutor/kv_cache/kv_cache_manager_v2.py`) decides
how the model's layers map onto layer groups, pools, life cycles, retention
priorities, and reuse policies. It also owns the extra buffers and the page
tables that the mixed-precision staging copy reads.

Configuration, that wrapper, admission checks, and existing callbacks are
cheaper to change, test, and review than the KV cache manager V2 native core.
Change the native core only when:
- the contract is algorithm-neutral, and
- the executor-side route is shown to be insufficient, for example by a
  profile of a hot path.

When you do, read the native core's own
`cpp/tensorrt_llm/batch_manager/kv_cache_manager_v2/AGENTS.md` first. It
lists the high-risk areas: eviction ownership, block hashing, partial
coverage, sliding-window and sink-token handling, beam forks, and stream
lifetime. Keep the change small, and add a native test with it.

## 6. Fail Fast; Never Fall Back Silently

- Reject unsupported combinations early, with an actionable message: during
  configuration validation, or at executor construction before KV cache
  manager V2 allocation when the check needs the device or the engine.
- Call external libraries and kernels directly. Do not probe versions or
  catch import errors to switch to a different path. A missing capability
  should fail loudly where it is first needed.
- Do not write defensive code: no try/except around code that should not
  fail, no `getattr` or `hasattr` defaults for attributes that always exist,
  no checks for states that validation already rules out, and no options
  nobody asked for. Each guard names the real failure it handles; everything
  else fails loudly.
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
- **One concept, one name, written in full everywhere.** Choose the name
  once, preferably the one the code or the public docs already use, and use
  it unchanged in identifiers, test names and parameter IDs, comments,
  docstrings, log and error messages, docs, commit messages, PR text, review
  replies, and conversation. Never shorten it to a fragment:
  - the mixed-precision high window is `mixed_precision_high_window` in code
    and "mixed-precision high window" in text, never `window` or "the
    window";
  - a test of the FP8 KV cache says "FP8 KV cache" in its name and its
    description (`fp8_kv_cache` in test IDs), never just "FP8" or "KV".

  A fragment that is clear to the author is ambiguous to everyone else: the
  same model may also have sliding windows, and FP8 may mean weights or
  activations.
- **Abbreviations.** Use one only if it is common knowledge to every reader
  (LLM, GPU, CPU, CUDA, API, `attn` in code), or if the text spells it out at
  first use as "full name (ABBR)" and uses that same abbreviation from then
  on. Define it again in each separate document, because readers open
  documents separately. Do not invent an abbreviation for a name that the
  code spells out, such as "KVCM" for `KVCacheManagerV2`.
- No internal jargon in code, comments, commit messages, docs, or PR text: no
  version labels for kernels, no milestone or branch nicknames, no codenames
  such as "path A", no section numbers from private notes. Use plain
  functional names a first-time reader understands.
- Comments explain the code as it stands to a future reader. Rationale,
  history, and references belong in the PR description.
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
  use the same extension point.

**How to extend**
1. **Prototype inside your own package**, behind your own configuration, and
   with no change to shared code where possible. Measure it first.
2. **Write the design down** in the PR or a design doc. Give the options
   (including "change nothing"), what each costs other owners, and what is
   being asked of them. Get the owners' decision before building.
3. **Design the extension point to be algorithm-neutral.** It is a hook, a
   capability flag, a buffer role, a life-cycle property, or a codec
   interface that knows nothing about your method. It defaults to today's
   behavior.
4. **Land the extension point on its own** when it is not trivial, with
   tests that prove existing behavior is unchanged. Then land the method on
   top of it.
5. **Remove scaffolding** that the extension point makes unnecessary. Do not
   leave two ways to do the same thing.

**Examples on main**
- *Composition, no new extension point.* The mixed-precision KV cache keeps
  its high-precision copy of the first and last tokens as one extra
  sliding-window-with-sinks life cycle: the mixed-precision high layer group.
  It needed no new C++.
- *Extending a kernel.* Physical compaction reused the existing sparse
  compaction op, with a layer dimension added, instead of a new mover.
- *A new generic extension point.* Cold-page compression added one cold-page
  codec interface to the KV cache manager V2 native core. The native core
  knows nothing about NVFP4; format logic lives in the compression provider.
- *A new generic native-core rule.* The `SINK_REQUIRED_ONLY` reuse policy was
  added to the KV cache manager V2 native core as a general policy, because
  the existing policies could not say "sink tokens required, sliding window
  optional". No method name entered the native core.

**Rigor that does not change**
Feature-off invariance, fail-fast admission, the compatibility table, and
the accuracy and performance gates apply to innovative work exactly as to
routine work. Being new is a reason to measure more, not less.

## 12. Keep Tests Few and Strong

Tests protect a working feature; they do not replace running it.
- Each test exercises the real path and fails when the feature breaks.
  Check this by breaking the feature on purpose.
- Use real components. Mock only what cannot run in the test environment,
  and never mock the behavior under test.
- Compute expected values independently of the code under test: exact
  bookkeeping, invariants, or a simple reference. A test that compares the
  code with itself proves nothing.
- Skip tests of trivial getters, and parametrize only over axes that change
  the behavior.
- Unit tests check function: the path runs, bookkeeping matches exactly, and
  unsupported combinations are rejected. Keep pytest small and cheap, with
  few GPU tests. Lossy kernel numerics, accuracy, and end-to-end behavior and
  performance are checked locally on real runs, and the commands and results
  go in the PR description.
- Never edit, skip, delete, or loosen a test, or special-case its inputs, to
  make it pass. If the test contradicts the requirement, stop and report it.
  When an approved behavior change makes an expectation obsolete, update the
  test and say why in the PR.
