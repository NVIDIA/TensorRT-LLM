<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Confidence scheduling: validation and developer reference

Start with the [usage guide](CONFIDENCE_REPRODUCTION.md) for artifact reuse,
self-generation and ordinary inference/benchmarking. This reference separates
validation history and advanced contracts from that user workflow.

## Recorded validation scope

These are the distinct integration scopes recorded before this documentation
reorganization. Original source/native/artifact provenance remains unchanged.

| Scope | Recorded result | Does not establish |
| --- | --- | --- |
| Earlier physical-K3, global-BS512/G64 across eight attention-DP ranks, separately recorded Python/native environment | Full-model whole-step profiling/export, genuinely paired calibration collection/fitting, and independent static/enabled-compact generation completed with output and natural-shutdown checks. | Fresh measurements for another source/runtime/node or every other K/BS. |
| Newer source composition, same-source/same-node K3, global-BS512/G64, 512 requests × 32 output tokens | Native/API compatibility and independent static/enabled-compact lifetimes completed source/native/artifact/input/output/routing, public comparison and natural-shutdown checks, reusing original costs/calibration. Exact generated token sequences differed. | Exact-source native rebuild, global clean install, task accuracy or sustained-serving performance. |
| Ordinary `trtllm-bench --config` confidence recipe in the usage guide | Public CLI/YAML/sampler interfaces are source-supported. | A new GPU execution or performance result for that recipe. |
| Small-K drafter-fusion extension | Source-matched extension passed 98 GPU unit cases on B300/SM103. | Full-model/performance qualification of every composed integration, SM100 qualification or arbitrary-K kernel support. |

Other prepared batch sizes and configurable physical K are interface coverage,
not additional measured results. Existing source tests and historical
measurements must not be relabeled as current-head GPU validation.

The controlled helper reports fixed-cohort enqueue-to-final-response TPS,
including prefill and drain. Historical plateau measurements and ordinary
benchmark load protocols use different timing boundaries; percentages from them
must not be pooled. Exact output-token equality is a diagnostic field, not a
replacement for original task-accuracy tests or a universal bitwise requirement.
No confidence interval or best-static/best-dynamic claim is produced by one pair.

## Supported routes and configuration

The confidence route currently supports embedded DeepSeek-V4 DSpark with trained
confidence-head weights, PP=1 and padded CUDA graphs. Standalone drafters retain
native verification; disable confidence scheduling for that route. The ordinary
target verification, sampling/acceptance, KV/graph, kernel and fatal-error checks
remain in force.

Guided decoding, LoRA, per-request logits post-processors and incompatible sparse
top-k paths are unsupported. In particular, `enable_heuristic_topk` and
`use_cute_dsl_topk` assume uniform speculative strides. On ragged steps,
`use_cute_dsl_paged_mqa_logits` falls back to the expanded DeepGEMM path.
Advanced rejection draft sampling is unsupported because its sampled prefix
differs from the greedy prefix that produced the confidence scores. Greedy
requests, including `top_k=1`, and requests with rejection draft sampling
disabled retain their supported behavior. Confidence-off execution retains its
ordinary sampling/post-processing paths.

The offline collector and fixed-cohort helper additionally require aggregate
PP=1/CP=1 generation, embedded DSpark, MPI world matching TP, sufficient
rank-local request/token/KV capacity and padded decode graphs. They reject
guided decoding, LoRA, custom logits processing, disaggregated KV transfer,
in-graph sampling, encode-only and resource-governor control. Paired collection
uses the original overlap-scheduler seam. These helper restrictions are not a
claim that all ordinary TensorRT LLM routes share the same limitations.

### Cost/configuration contract

`V=0` denotes native full-width verification; positive V denotes a measured
compact layout at rank-local graph size G. Physical K is not reduced by
scheduling. K1 has no compact tier and resolves to feature-off static execution.
Equal row counts with different graph/attention layouts are not interchangeable.

The schema-v2 table must contain finite positive measurements and compatible
physical-K, graph/layout and capacity coverage. Default reuse can select a
configured subset of the original measured ladder while validating the entire
original table/fingerprint; a drain-guard tail must still be measured and
captured. Strict freshness requires the exact original ladder. Missing cells
are not interpolated, copied from another K or fabricated.

The current whole-step exporter supports attention-DP fanout or one rank;
multi-rank attention-DP-off export is refused. This is an export-schema limit,
not a blanket DSpark runtime restriction. Replay-entry timing remains a separate
diagnostic; it is not renamed to a complete-loop cost or summed with overlapping
host/GPU spans.

Optional temperature vectors must have finite positive values for each physical
draft position. Omitting calibration uses raw sigmoid probabilities. Reuse
retains original artifacts and records the actual destination runtime separately;
it does not rewrite old fingerprints or guarantee positive performance.

### Requested configuration versus execution

| Case | Resolved behavior |
| --- | --- |
| Explicit confidence enablement, default artifact reuse | Enabled, subject to structural/runtime checks; warnings state freshness/economics are unchecked. |
| Optional strict freshness with missing, stale, unreadable or non-positive evidence | Feature-off static; confidence flags and artifact paths/digests are cleared with a warning. |
| Physical K1 | Feature-off static K1, confidence inputs cleared. |
| Strict freshness with matching positive evidence | Remains requested; normal runtime/checkpoint/graph checks still apply. |

Startup feature-off fallback is not an enabled-native policy decision. An enabled
policy can choose native while retaining its confidence/selection work. Optional
fused-scheduler failure can instead fall back to the tensor scheduler for an
affected graph size. Record resolved and runtime enablement plus actual routing;
a YAML flag or successful server start is insufficient.

The selector retains the table's immediate predicted-goodput margin, normally
1%, and any supplied iteration/drain guard. Default reuse without a drain guard
uses immediate goodput; strict validation without one retains native
verification. Fixed confidence-path overhead is a separate workload-level
charge, not redistributed to individual candidate choices. None certifies an
end-to-end speedup.

### Drafter fusion

The BF16 CuTe attention and RMSNorm–RoPE path is eligible for physical K=1..8
under the existing SM100/SM103 geometry/layout guards; unsupported inputs retain
the reference path. The eight-row storage bound is unchanged. This dispatch is
independent of the confidence flag and therefore also affects static DSpark.
See the validation table above for the actual exercised scope.

## Optional strict freshness and admission

`validate_confidence_artifact_freshness: true` opts into the stricter path.
It consumes four externally supplied inputs:

- `confidence_sps_table_path`: schema-v2 measured costs.
- `confidence_sps_live_fingerprint_path`: independently observed destination
  fingerprint, not a relabeled historical fingerprint.
- `confidence_admission_receipt_path`: schema-v1 workload receipt.
- `confidence_admission_receipt_sha256`: externally trusted SHA256 of its bytes.

Ordinary artifact reuse does not require these additional artifacts. The public
cost/calibration commands do not produce a positive strict-admission receipt.
A separately reviewed producer must retain measured costs, policy overhead,
calibration, workload, selector-replay and runtime evidence. Source, workload or
artifact changes require reassessing that evidence's applicability.

The receipt envelope has exactly `schema_version: 1`, `admission` and
`admission_sha256`. The last field hashes the canonical admission body:
`json.dumps(body, sort_keys=True, separators=(",", ":")).encode()`.
Its fields are:

```text
physical_k, source_head, source_diff_sha256, runtime_snapshot
sps_cost_table_sha256, live_engine_fingerprint_sha256, selector_identity_sha256
workload_identity_sha256, calibration_result_sha256, selector_replay_sha256
policy_steps, compact_choices, gross_compact_value_ms_lower_bound
fixed_confidence_path_overhead_ms_upper_bound, safety_margin_ms, admitted
```

The selector identity must match `EXACT_SPS_SELECTOR_IDENTITY_SHA256` in
`tensorrt_llm/_torch/speculative/dspark_planner.py`. Admission recomputes:

```text
net_value_lower_bound_ms =
    gross_compact_value_ms_lower_bound
    - policy_steps * fixed_confidence_path_overhead_ms_upper_bound
    - safety_margin_ms
admitted = compact_choices > 0 and net_value_lower_bound_ms > 0
```

All policy steps, including native choices, pay enabled-path overhead. Do not
allocate that fixed charge back to candidates using a
`policy_steps / compact_choices` ratio. Admission does not replace per-decision
selection or its original drain/price guards.

Hashes check integrity/cross-artifact consistency, not signatures or origin.
The consumer does not independently derive running checkpoint/workload identity,
rerun calibration/replay or verify every referenced result file. The external
producer remains the trust boundary; hashing arbitrary JSON or fixtures is not
admission evidence.

## Optional mixed context/generation scheduling

`confidence_mixed_evidence` defaults to `None`: mixed steps use native
verification while supported decode-only scheduling remains enabled. Decode
T(G,V) costs do not price mixed work; context chunks/positions, history,
bootstrap rows, ordered ranks and shared body capacity enter the mixed key.

The evidence object has `costs_path`, `costs_sha256`, `admission_path`,
`admission_sha256`, `protocol_sha256`, `live_fingerprint_path` and
`live_fingerprint_sha256`. Cost and receipt schemas are
`dspark-mixed-breakable-costs-v1` and `dspark-mixed-breakable-admission-v1`.
Pins come from a trusted producer; the live source/runtime/model/GPU/topology/
selector fingerprint must be observed independently. Decode artifact reuse does
not relax this separate contract.

The supported mixed route is embedded physical-K3 DSpark, TP8/PP1/CP1,
attention DP, greedy sampling and captured BREAKABLE buckets
`[96, 128, 192, 256, 384, 512]`. Unsupported/uncovered routes stay native for
mixed steps without disabling valid decode scheduling. First context chunks
stay native because prefix-cache resolution can change geometry. Later chunks
need their full-width native rows to fit; quoted context work is not trimmed.

When a peer prefills, ranks use shared BREAKABLE padding, not a rank-local full
decode graph. Only authenticated generation owners can receive shorter
prefixes; context/bootstrap/padding rows retain required rows. Host-authoritative
lengths are published before attention metadata preparation, without changing
backend, sampler or acceptance. Stale/unready owners remain full-width.

Selection requires an exact measured geometry/action cell and at least 1%
predicted-goodput margin. Current `rank0_forward_pricing_proxy` prices forward
work, not the whole distributed step. Workload admission must charge CPU curve
preparation, producer-stamp transfer and the existing allgather's fixed trailer
on all policy steps, including native/all-decode decisions. Policy equivalence
does not establish zero overhead.

There is no public mixed receipt-generation command here. Valid-shaped fixture
JSON is not deployment evidence. A separately reviewed producer must supply
measured costs, accuracy/runtime results, selector replay and positive workload
economics; exact integration GPU/full-model and production evidence remain
separate from CPU contracts.

## Paired collection and developer checks

Calibration pairs are not matched by arrival order. Step n confidence describes
a proposal verified later, usually at n+1. The adapter binds that forward to its
original `SampleState.device`, snapshots proposals before buffer reuse, and
records confidence/draft stamps before the normal sampler event. Host copies
are read only after the sampler's existing synchronization/update.

Request incarnation, proposal content, local confidence row, sampler slot and
full-K extent must agree. Prefill placeholders, retired/unscored bootstrap rows
are counted rather than fitted. Wrong-step/stale/censored observations fail;
current-step logits cannot substitute for the producing logits. Compact
verification cannot label an unverified suffix.

Collection emits `rank-N-pairs.pt` and `rank-N-run.json`. Tensor-only shards
contain logits [rows,K], cumulative accepted-prefix labels, pairing metadata and
source/runtime identity. The fitter retains shard hashes, sample count,
temperatures and before/after ECE; degenerate positions are reported rather than
promoted as informative fits. This is not task-accuracy qualification.

The optional constructor test below uses the real argument validator and
internal factory seam without constructing a model. Run it only in a compatible
installed native runtime; imports may require a driver. It is a developer
regression, not a colleague pre-run gate or GPU qualification:

```bash
python -m pytest -q tests/unittest/executor/test_dspark_reproduction_factory.py
```

`executor_cls` is internal and remains rejected as a public LLM argument.
Additional synthetic contract regressions are:

- `tests/unittest/_torch/speculative/hw_agnostic/test_dspark_workload_admission.py`
- `tests/unittest/_torch/speculative/hw_agnostic/test_dspark_admission_resolution.py`
- `tests/unittest/_torch/speculative/hw_agnostic/test_dspark_artifact_reuse.py`
- `tests/unittest/llmapi/test_dspark_workload_admission_config.py`
- `tests/unittest/_torch/speculative/hw_agnostic/test_dspark_mixed_evidence.py`
- `tests/unittest/_torch/speculative/hw_agnostic/test_dspark_mixed_runtime.py`
