<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Handing measured costs to DSpark

The offline graph collector and runtime cost consumer have different contracts.
The collector's `verify-cost-measured-summary-v1` output measures the original
graph replay entry, including static input copies. The consumer's schema-v2
`step_time_ms` represents direct whole-step costs. Do not rename the replay
median to `step_time_ms`, sum potentially overlapping graph/host components, or
construct a new engine fingerprint from the partial loaded-runtime observation.

The explicit `whole-step` profile mode reuses the executor's existing
`hostStepTimeMS` boundary. It joins the just-finished complete loop to its actual
executor iteration; it does not sum graph and host spans, introduce new CUDA
events, alter streams or add synchronization. The raw exporter retains all-rank
joins, warm-up exclusions and shutdown closure. Graph timing is retained only as
a separately labeled diagnostic in this mode.

`dspark_cost_export.py` maps those direct whole-step medians into the existing
runtime schema without filling missing cells. It also accepts an existing
schema-v2 artifact and can copy its original bytes after structural validation.
Both paths use the actual runtime schema validator. Plain graph summaries remain
ineligible; that does not prevent reuse of structurally valid supplied costs.

These are source/CPU-prepared development interfaces. The complete new GPU
collection/export workflow has not been exercised or qualified yet.

For a previously measured cost table and a runtime capturing G16 and G64:

```bash
python tensorrt_llm/tools/dspark_cost_export.py \
  --input /path/to/recorded-costs.json \
  --physical-k 3 --graph-batch-sizes 16 64
```

`K` is configurable and has no fixed pilot restriction. `G` is the rank-local
request count of the actual captured graph, not always the global batch size.
With eight attention-DP ranks, BS128 and BS512 correspond to G16 and G64; with
attention DP off they do not. Pass the exact graph ladder the engine resolves.
Native V0, positive finite timing values, physical K, cell provenance and graph
coverage must satisfy the existing consumer. Missing cells are not interpolated,
copied from another K or filled with zero. No cluster name or site path is used.

Add `--output /path/to/new-costs.json` only to deliver a new byte-identical copy.
The destination must not exist. No original values, recorded source head or
engine fingerprint are rewritten. In this reuse path there is no exact-head
freshness check and no economic admission requirement; the result warns that
destination freshness and performance are unchecked. Confidence scheduling is
still default-off and must be explicitly enabled through its normal configuration.

The structural-check report is not a calibration result, GPU health check,
task-accuracy certificate, positive admission receipt or benchmark. Genuine
enabled routing, output checks and a matched static/dynamic comparison remain
separate measurements. The source-only CPU tests do not qualify this complete
clean-checkout procedure on a GPU.

## Preparing new direct whole-step costs

Use the configurable K/G/V grid and authorized inputs described in
[COST_PROFILING.md](COST_PROFILING.md). Set `enable_iter_perf_stats: true` in the
ordinary runtime YAML so the existing complete-loop timer is active. Use a
read-only checkout of the original source as `--source-root`; its actual Git
head/diff and loaded module bytes are recorded as provenance, not used as a
destination freshness gate. Runtime/library/GPU observations are preserved in
the raw identity. No new native build is implied by this Python-only tooling.
The Git head is only the original base: the raw `loaded_source_sha256` roster
and recorded runtime snapshot retain executed local Python overlays, including
untracked tooling. A dirty execution is not described as pristine just because
it has the same base head; original tracked diff bytes are hashed separately.

The additional arguments to the normal profile interface are:

```bash
python -m tensorrt_llm.tools.verify_costs profile \
  --model /path/to/authorized/checkpoint --config /path/to/runtime.yaml \
  --grid /path/to/profile-grid.json --physical-k 3 \
  --cost-scope whole-step --source-root /path/to/original-public-checkout \
  --inputs /path/to/tokenized-prompts.jsonl \
  --output /path/to/new-private-profile-directory
```

Use this command under the site's normal MPI/container launcher with the actual
GPU count and full startup/generation/export/shutdown budget. No cluster or
account is embedded in the tooling. This source-prepared interface is **not** a
GPU-tested reproduction instruction. No extra sweep is needed before reusing
valid supplied cost/calibration predictions.

Reduce the closed raw observations with the existing `verify_costs.py export`
command. Its defaults retain at least 12 exact joins in each retained generation
and 24 pooled per cell. Then use the DSpark adapter:

```bash
python tensorrt_llm/tools/dspark_cost_export.py \
  --input /path/to/whole-step-measured-summary.json \
  --physical-k 3 --graph-batch-sizes 16 64 \
  --output /path/to/new-costs.json
```

The adapter exports the retained per-iteration rank maximum's median unchanged;
every schema-v2 measurement references the original summary hash. The recorded
fingerprint is preserved, not refreshed by the CPU exporter. It neither creates
drain constants nor measures confidence-policy overhead, calibration quality or
economic admission. The default table policy margin remains 1%; it is not a
timing measurement. The original summary and raw observations remain available
to inspect scope, source/configuration and raw joins.

The existing schema-v2 consumer checks global graph counts as rank-local G
times `gpu_count`. This new export currently supports attention-DP fanout or a
single rank; multi-rank attention-DP-off needs a separate schema decision and
is explicitly refused rather than recording incorrect global batch sizes.

## Graph-only exports stay separate

The default `graph-replay` collector records
`original-graph-replay-entry-including-static-copies` and
`DIAGNOSTIC_PARTIAL_LOADED_RUNTIME`, with
`production_sps_table_eligible: false`. This command deliberately refuses that
summary even if an eligibility flag is manually changed. Only choosing and
actually collecting the explicit whole-step boundary produces whole-step raw
observations. Changing a label does not extend an existing measured scope.

GPU-forward values and overlap `iterLatencyMS` are not replacements for clean
complete-loop measurements: the former omits host work, while the latter can
span more than one loop. Optional policy-overhead/economic validation stays
separate. Authentic clean-checkout GPU collection, enabled-routing/output checks
and matched benchmarking are still required before publishing this as a verified
end-to-end self-generation recipe.
