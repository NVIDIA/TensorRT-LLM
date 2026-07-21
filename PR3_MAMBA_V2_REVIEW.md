<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# PR3 V2 Mamba snapshot reuse review

This document records the design decisions, code changes, experiments, and
verification for PR3. It is intended to make the final PR diff reviewable
against upstream `main`, which already contains PR1, rather than against an
older stacked-PR base.

## Baseline

- PR1 was squash-merged into upstream `main` as `bb90835f8a`.
- PR3 was rebuilt on the latest upstream `main` from its four PR3-specific
  commits.
- The original PR1 commit chain and the two stacked-branch synchronization
  merges were removed from PR3 history.

The final PR3 diff must be reviewed relative to upstream `main`.

## Review requirements

1. Select the V1 Mamba manager by default. Select V2 only when
   `use_kv_cache_manager_v2: true` is explicitly configured.
2. Move Mamba snapshot policy under `mamba_state_config`:
   - `periodic_snapshot_interval` controls regular snapshot boundaries.
   - `additional_snapshot_offsets_from_start` expresses fixed boundaries
     measured from the prompt start.
   - `additional_snapshot_offsets_from_end` expresses fixed boundaries
     measured backward from the prompt end; zero selects the prompt end.
3. Reject unsupported combinations before manager construction:
   - V1 supports periodic snapshots only.
   - V2 does not support disaggregated serving.
4. Keep generic KV cache manager changes minimal and express Mamba behavior
   through a specialized extension wherever possible.
5. Determine whether `commit_min_snapshot` removes the need for
   `allow_prefix_sibling`, using multi-turn and forked-prefix reuse tests.
6. Explain and justify the extra attention-slot bound.

## Configuration and manager-selection decisions

`forward` and `backward` were rejected as field names because they can be
misread as execution or traversal directions. The selected names state both
the unit and reference anchor explicitly.

| Serving mode | `use_kv_cache_manager_v2` | Additional offsets | Result |
| --- | --- | --- | --- |
| Aggregated | false or auto | Empty | V1 C++ manager |
| Aggregated | true | Any valid policy | V2 Mamba manager |
| Aggregated | false or auto | Non-empty | Configuration error |
| Disaggregated | false or auto | Empty | V1 C++ or mixed manager, according to the transceiver |
| Disaggregated | true | Any | Configuration error |
| Disaggregated | false or auto | Non-empty | Configuration error |

The `TLLM_MAMBA_MANAGER_PREFERENCE=V2` override conflicts with the requirement
that V2 be selected only by the explicit public setting. The V2 override has
therefore been removed. An explicit V2 setting that conflicts with the CPP,
MIXED, or `TRTLLM_USE_PY_MAMBA` compatibility overrides is rejected rather
than silently routed to another implementation.

The snapshot resolver uses these exact semantics:

- `periodic_snapshot_interval=0` disables periodic snapshots;
- start offsets are strictly positive absolute token boundaries;
- end offsets are non-negative and resolve to `prompt_len - offset`, so zero
  selects the final prompt boundary;
- resolved positions outside `(0, prompt_len]` are ignored;
- periodic and fixed boundaries are deduplicated and sorted.

## Public configuration API

This PR deliberately makes the snapshot-policy rename a breaking Python API
change. The former top-level `mamba_state_cache_interval` constructor argument
and Python property are removed without a model validator or deprecated alias,
and strict model construction rejects the removed field. YAML/JSON file loaders
retain a narrow compatibility migration to
`mamba_state_config.periodic_snapshot_interval`; setting both paths is an
error. All in-tree callers, examples, telemetry metadata, and canonical
documentation use the nested field.

## KV cache extension audit

The completed read-only audit found that the generic manager should retain only
two neutral PR3 changes:

- a neutral constructor input for reserved index slots, used only to size the
  stable `IndexMapper` storage;
- reporting every physical pool group in iteration statistics, which is a
  generic correctness fix and needs a non-Mamba test.

The remaining changes are Mamba-specific and have moved into
`V2MambaHybridCacheManager`, following `DeepseekV4CacheManager`:

- CUDA-graph and attention-DP dummy-slot calculation;
- SSM/conv roles and page-table layout;
- SWA scratch layout for recurrent pools;
- snapshot commit, history, final-free, and context-update lifecycle;
- recurrent-state invalid-value checking.

`KVCacheDesc.capacity` remains token capacity. The only Mamba-specific sizing
field is `num_ssm_slots`, which counts the live, snapshot, and dummy recurrent
state slots associated with that logical request. The storage planner sums it
for SSM lifecycles. If the batch has any non-live SSM capacity, the planner also
reserves one retained partial attention page per request lineage. There is no
separate `num_extra_attention_slots` field or layer-specific `BatchDesc`
descriptor.

The subclass computes the exact CUDA-graph and attention-DP dummy count before
calling the base constructor, then passes that count through the neutral
`num_reserved_index_slots` extension. Its CUDA and pinned-host state-index
tensors are allocated once for `max_batch_size + reserved slots`; later dummy
insertion therefore cannot replace an aliased tensor or invalidate a captured
CUDA graph. The subclass also owns the mixed-pool page table, event-window
mapping for `SsmLayerConfig`, snapshot commit/history lifecycle, and state
diagnostics. The generic manager no longer knows any Mamba roles or policies.

Metadata preparation still runs on attention-only pipeline-parallel ranks even
though no local Mamba kernel consumes state indices there. Both V1 and V2 now
return harmless zero placeholders on those ranks instead of consulting Mamba
state maps or buffers that are intentionally absent. The V2 invalid-value
diagnostic also scans every attention layer: a runtime layer group represents
a lifecycle, not necessarily one physical pool, so layers with different
buffer sizes cannot safely be deduplicated by group ID.

The audit classified each generic KVCacheManagerV2 hunk as one of:

- generic prerequisite already supplied by `commit_min_snapshot`;
- generic extension point required by Mamba;
- Mamba-only policy that belongs in `mamba_cache_manager.py`;
- obsolete code removable from PR3.

## `allow_prefix_sibling` experiment

The PR3-specific constructor flag, sibling-retention rule, alternate SSM
matcher, and both call sites were removed. `commit_min_snapshot` remains the
only runtime prerequisite.

The first experiment exposed a generic replacement-order bug: when a longer
partial block replaced the only child of a root, `detach_next()` could prune
the now-empty root before the replacement was attached. The replacement is now
registered before covered children are detached.

The resulting semantics are deliberately simpler:

- monotonically extending snapshots at 10, 20, and 25 tokens reuse 10, 20,
  and 25 tokens respectively;
- a fork sharing only 15 tokens cannot consume the retained 20-token state and
  safely re-prefills from zero;
- a fork sharing 25 tokens reuses the 20-token state;
- a block-aligned 32-token snapshot remains reusable by a later fork.

Only the latest partial snapshot within one token block and request lineage is
retained. An earlier same-block fork may therefore miss a reusable state, but
it cannot read a state computed from tokens beyond the fork boundary. This is
an accepted efficiency tradeoff for removing the custom sibling structure.

## Extra attention slots

These are physical attention-cache slots used
when a non-block-aligned Mamba snapshot preserves a copied partial attention
page in the radix tree. They are not request slots or CUDA-graph/attention-DP
dummy slots. Their capacity is additional to the normal attention-page budget
because the active request still owns a writable partial page while the radix
tree retains the snapshot copy.

Each logical request descriptor budgets at least one live SSM state. The
planner uses `total_num_ssm_slots > num_requests` only to detect that the batch
has non-live SSM capacity. It then reserves one retained partial attention page
for every request lineage and adds that bound to every attention lifecycle;
nothing is supplied by an external caller. Covered partial siblings are removed
as a lineage advances, so more than one retained page per lineage is
unnecessary.

This intentionally trades some capacity precision for a smaller interface.
The bound does not require `total_num_ssm_slots >= 2 * num_requests`. When the
number of non-live slots is smaller than the number of lineages, or when those
slots are aligned snapshots or reserved dummy states, it simply over-reserves
attention capacity. It never depends on snapshot alignment or slot
distribution. The normal request token capacity already accounts for a resumed
request's writable page; this bound covers the retained radix-tree snapshot
page. The V1 estimator retains its existing policy because this single-field
bound is specific to V2.

## PP/spec layout and affine sizing

One-model MTP with pipeline parallelism is expected to provide an explicit
`pp_partition` covering the base-model layers. The existing PP helper uses that
partition as the base/spec boundary and assigns appended speculative layers to
the last rank. Supporting an automatically derived boundary when
`pp_partition` is absent is intentionally left to a separate change.

The hybrid affine estimator uses the same normalized masks and explicit PP
layout to count local attention and Mamba layers. This avoids two failures:

- custom `pp_partition` being applied to an attention-only layer count and
  raising during startup;
- a separate, attention-only MTP draft cache being charged for phantom target
  Mamba state.

Target-only and draft-only masks are shared between estimation and runtime
construction, so the budget split uses the physical layout each manager will
actually allocate.

## Verification log

- `python3 -m py_compile` passed after the configuration, routing, subclass,
  and test migrations.
- Focused configuration/routing/snapshot-point tests: 34 passed.
- Full `test_mamba_cache_manager.py`: 85 passed.
- Three focused final-audit regressions passed: V1 and V2 metadata preparation
  on ranks with no local Mamba layers, and invalid-value detection in a second
  differently sized attention pool that shares a lifecycle.
- Six focused PP/MTP sizing cases passed across default/custom PP partitions,
  Cpp/V2 estimators, target-only masks, and attention-only draft masks.
- The combined Mamba, cache-budget split, dual-pool, and executor-creator suite
  passed: 151 tests.
- Generic KVCMv2 and executor-creator tests: 24 passed.
- Full `test_llm_args.py`: 255 passed, 3 skipped. A preceding attempt failed
  because the environment's MPI session-reuse worker did not report an identity
  within 60 seconds on either spawn attempt (`0/1` workers); it did not reach
  Mamba configuration or manager code.
- The C++ estimator regression for disabled periodic snapshots on a
  recurrent-only rank passed, as did the V2 partial-attention sizing test.
- Runtime radix-tree focused tests: 2 passed.
- Runtime `TestSSMSupport`: 13 passed.
- Runtime `TestNoBatching`: 24 passed, 12 skipped.
- `scripts/generate_llm_args_golden_manifest.py --check` passed. The manifest
  exposes only `kv_cache_config.mamba_state_config.periodic_snapshot_interval`.
- The telemetry generator cannot load legacy `TrtLlmArgs` in this local
  environment and would incorrectly erase that entire table. The generated
  noise was discarded; the two existing Mamba telemetry rows were updated to
  the manifest's canonical path without changing unrelated rows.
- Isort/YAPF formatting and the completed runtime experiment's lint/diff
  checks passed. The final read-only audit found no remaining correctness
  blockers after the PP/spec sizing fixes.
- Final manifest, syntax, diff, and full pre-commit checks passed before
  publication.
