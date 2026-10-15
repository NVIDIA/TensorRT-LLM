<!-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Offline DSpark graph-cost profiling

This is a development interface for collecting and exporting measured graph
replay costs. It is not yet a GPU-qualified colleague reproduction recipe.
The default graph-replay collector does not itself generate a serving SPS table,
paired calibration observations, an admission receipt, or an enabled throughput
result. The explicit complete-loop mode and CPU schema-v2 adapter are described
in [COST_EXPORT.md](COST_EXPORT.md); that new producer path remains GPU-unqualified.
The separate CPU calibration fitter below can fit already collected, correctly
paired shards. The complete public clean-checkout workflow is not yet qualified.

The tools contain no private model, dataset, scratch path, historical receipt or
source-head pin. Use your own authorized local checkpoint and tokenized inputs.
Raw observations include prompt and output tokens; store them in an appropriately
access-controlled output directory.
The Linux collector creates only a new directory with mode0700 and new raw files
with mode0600, using exclusive/non-overwriting creation. It does not change
existing paths, ACLs, process umask or host settings. Existing/symlink targets are
rejected. POSIX permission behavior still needs actual Linux validation; Windows
CPU controls do not qualify Linux permissions or Windows ACLs.

## Scope and terminology

- `K` is the physical draft block produced by the model. Set it explicitly with
  `--physical-k`; it must equal the runtime's `speculative_config.max_draft_len`.
  Profiling never relabels a K5 execution as K3.
- `G` is the rank-local graph request count.
- `V` is the total verifier token budget in that rank-local graph. `V=0` denotes
  the native full-block control, not an empty execution. Positive compact cells
  require `K>1` and `G <= V < G*(K+1)`; K1 currently supports only native cells.
- The measured scope is **the original graph replay entry, including static
  input copies**. It includes work already inside that graph, including the
  embedded drafter and optional confidence head. It is not a target-only timer,
  a complete confidence-policy price, or an end-to-end generation timer.

The offline engine uses the original model loader, attention/backend selection,
graph capture/replay, request scheduler, KV allocator/estimator, sampler, token
updates and shutdown. It prescribes a declared verification cell; it does not
enable or bypass production confidence admission. Ordinary requests outside a
prescribed offline generation are rejected, and no serving RPC endpoint is exposed.
The profiler duplicates its control communicator collectively before worker
construction, retains it through shutdown/evidence closure, then frees it.
Original request broadcasts and runtime communicator/thread-local bindings are
unchanged. Multi-rank profiling requires existing MPI `THREAD_MULTIPLE` support;
the collector checks it and does not modify MPI initialization or thread level.

Each cell runs a fixed cohort of real requests through prefill, generation and
drain, then repeats that generation. It does not time a single synthetic request
replayed in isolation. Only decode steps with exactly `G` requests on every rank
and the declared captured graph key become samples. Initial prefill, first-draft,
peer-not-ready and shorter drain batches retain the native path and are not timed
as that cell. No forced acceptance is allowed. The default reducer excludes
generation 0 and the first two eligible samples of every retained generation.

## Prepare a matching runtime and workload

Use a complete Linux checkout containing this tooling. Build/install TensorRT-LLM
using the repository's [build instructions](../../../../docs/source/installation/build-from-source.md),
or reuse genuinely compatible existing native artifacts through the documented
precompiled/editable-install route. A Python-only tooling change does not by
itself require a native rebuild or new cost/calibration measurements.
Do not combine these Python modules with incompatible native interfaces. Retain
the original native provenance and record changed Python source separately.
The collector records
the actual loaded Python/native-library hashes, package version, GPU class,
topology, graph configuration and observed attention-module classes. A module
roster is not proof of every dispatched kernel or of GPU health.
These raw identity observations are also not a complete weight/JIT/source/build
qualification or a production serving live fingerprint.

Prepare a runtime YAML accepted by `TorchLlmArgs`, including:

- `tensor_parallel_size` matching the MPI world size, `pipeline_parallel_size: 1`
  and `context_parallel_size: 1`;
- the intended expert parallelism and `enable_attention_dp` setting;
- enough `max_batch_size`, `max_num_tokens`, `max_seq_len` and KV capacity for the
  full cohort and requested output length;
- padded decode CUDA graphs whose **resolved rank-local ladder** exactly matches
  all `G` values in the profile grid;
- an embedded DSpark configuration with the selected physical K, confidence
  scheduling and the optional fused scheduler disabled, no draft-length schedule,
  and no confidence SPS/fingerprint/admission/calibration fields.

This initial driver is aggregate, fixed greedy generation. It rejects LoRA,
guided decoding, custom logits processors, disaggregated KV transfer/connectors,
encode-only execution and in-graph sampling. Prefill graph capture is not part of
this interface; resource-governor control is also unsupported. Keep the intended
numerical/quantization/backend configuration;
do not change it merely to make profiling launch.

For example, an attention-DP eight-rank profile of global BS512 uses G64. One
possible **declared grid**, not a measured cost table, is:

```json
{
  "physical_k": 3,
  "record_confidence_head": true,
  "max_samples": 4096,
  "cells": [
    {"graph_batch_size": 64, "verifier_budget": 0},
    {"graph_batch_size": 64, "verifier_budget": 96},
    {"graph_batch_size": 64, "verifier_budget": 128},
    {"graph_batch_size": 64, "verifier_budget": 192}
  ]
}
```

Use `cuda_graph_config.batch_sizes: [64]` and `enable_padding: true` for that
single-G example; the original runtime's token/request limits must retain G64.
If the resolved ladder has more G values, include their native V0 controls and
desired compact cells. Missing or silently clamped coverage is an error.

The interface can prepare all previously used global batch sizes
128/256/512/1024/1536/2048. With eight attention-DP ranks these map to
G16/G32/G64/G128/G192/G256. With attention DP off, G is the full global batch,
not BS/8. Smaller points are explicit G inputs too; no pilot batch is hardcoded.
Each additional G requires its own captured graph and measured compatible cells.
Preparation does not imply validation of that batch size. Native/GPU validation
is still unperformed for this new tooling, including BS512/1024.

The JSONL input file must contain at least the largest global cohort size, with
one object per line and a nonempty list of integer `prompt_token_ids`. Use the
actual licensed tokenizer and workload. The driver assigns stable input IDs and
uses the same prompt prefix for repeated cells; prompts and requested output
length must fit `max_seq_len`.

## Development collection command

After the complete launch procedure, matching native build and GPU health are
qualified, invoke the CLI under the site's normal MPI/container launcher. For
an already configured, healthy eight-GPU Linux environment the interface is:

```bash
mpirun -np 8 --bind-to none python -m tensorrt_llm.tools.verify_costs profile \
  --model /path/to/authorized/checkpoint \
  --config /path/to/runtime.yaml \
  --grid /path/to/profile-grid.json \
  --physical-k 3 \
  --inputs /path/to/tokenized-prompts.jsonl \
  --output /path/to/new-private-output-directory \
  --generations 3 --output-tokens 768 --timeout 12180
```

These paths are explicit inputs, not supplied artifacts. `--timeout` is the total
generation-phase request budget after construction/identity, not a guarantee that
startup fits a scheduler lease. Reserve the full startup, model, export and
shutdown budget in the outer launcher. Do not restart on quiet logs or an
observation timeout. The output directory must be new and its parent must exist;
no existing output is overwritten.

The result contains one `rank-R-record-N.json` per cell/generation/rank. It records
actual replay keys, lease-completed GPU events, request/sampler/KV lifecycle joins,
fixed-length original responses and closure. Completed evidence is retained on
failure with `run_complete: false`; a failure record does not qualify costs.
Successful shutdown is required before raw records can be exported as complete.

## Export and inspect raw measured costs

Reduction is stdlib-only and can run without importing native TensorRT-LLM:

```bash
python tensorrt_llm/tools/verify_costs.py export \
  --raw /path/to/new-private-output-directory/rank-*-record-*.json \
  --output /path/to/new-measured-summary.json \
  --exclude-generations 1 --warmup-samples 2 \
  --minimum-generation-samples 12 --minimum-samples 24
```

The reducer joins exact replay ordinals and executor iterations across **all**
ranks, takes each joined rank maximum, then reports the retained median. It does
not average ranks or sum overlapping spans. It preserves raw-file hashes,
per-rank samples and every exclusion. It rejects duplicate JSON keys, nonfinite
or nonpositive durations, unequal scopes/identities, missing declared cells or
generations, missing/duplicate ranks, reused/reset ordinals, changed observed
layouts and incomplete generation/run/shutdown closure.
Each declared cell must observe one consistent graph key over its generations.
If the workload crosses different sequence-length/attention graph regimes, the
collector refuses to pool them; prepare separate appropriately declared runs.
Each retained generation needs at least 12 exact samples after exclusions, and
each cell needs at least 24 pooled retained samples. A large count in one
generation does not waive another generation's floor.

Cost keys contain physical K, graph G/V, attention layout, captured-head mode and
the observed graph key. Rows with equal counts but different layouts are not
pooled. Inspect the source/runtime/model/topology identity before comparing or
reusing a summary. Raw summary structure and timing scope remain different from
the runtime's schema-v2 prediction table; renaming a file does not convert it.

## Fit existing paired confidence observations on CPU

The standalone fitter accepts tensor-only `.pt` shards containing
`logits: [N, K]`, binary cumulative `prefix_mask: [N, K]` and
`meta.pairing: "draft_seq_ring"`. The labels must be observed against the same
stamped draft sequence and request incarnation as the raw logits, with full
physical-K verification. A shortened compact verification cannot provide labels
for its unverified suffix. This producer contract is not inferred from tensor
dimensions or from a metadata string alone.

Fit the established 41-point, left-to-right cumulative-survival ECE objective:

```bash
python tensorrt_llm/tools/dspark_calibration.py \
  --shards /path/to/private/paired-observations-*.pt \
  --physical-k 3 --output /path/to/new-sts.json
```

The shell expands the shard glob. The default floor is 1,000 paired rows; use
representative data rather than interpreting a small fixture as a calibrated
head. Physical K is explicit, not fixed to the profiling pilot. The output
retains both `sts_temperatures` and `temperatures`, the original shard hashes,
sample count and before/after ECE. Degenerate prefix positions are reported,
not promoted as informative fits. Inputs use PyTorch's tensor-only safe loader;
duplicate shards, wrong pairing/width, nonfinite data and invalid prefix order
are rejected. Existing output files are never overwritten.

This command has CPU math/CLI coverage, including exact agreement with the
established fitter. The new public paired-observation collector is still a
separate integration task: the graph-cost collector does not emit these shards.
No private collector command or environment variable is a public prerequisite
advertised here. CPU fitting does not certify GPU freshness, task accuracy or
positive scheduling economics. Ordinary runtime reuse of a valid supplied STS
vector remains available without fitting it again.

## Next stages and validation boundary

A measured summary deliberately sets `production_sps_table_eligible`,
`production_admission_qualified`, `policy_overhead_measured` and
`task_accuracy_qualified` to false. Do not rename it to
`confidence_sps_table_path` or create a positive receipt from graph timings alone.

Confidence scheduling remains default-off. With explicit enablement, the
default reuse path accepts structurally valid supplied costs and calibration,
with clear warnings when freshness and performance economics are unchecked.
It retains physical-K/graph/layout coverage, finite positive values and ordinary
runtime/acceptance safety. Fresh costs/calibration and positive economic
admission are not universal prerequisites to a fair measured comparison.

The candidate [self-generation workflow](CONFIDENCE_REPRODUCTION.md) includes
whole-step cost export, correctly paired calibration collection/fitting, and
ordinary static/dynamic execution plus comparison. The graph-replay mode keeps
its narrower recorded scope; it does not price online policy, layout preparation
or end-to-end request execution.
Measure those exposed costs separately when evaluating prediction quality or
scheduling headroom. Do not double-count overlapping spans.

Users choosing `validate_confidence_artifact_freshness: true` additionally need
an actual destination fingerprint and the optional workload-economic checks,
including measured enabled-path overhead. Native decisions pay that overhead
too. A positive receipt must come from genuine measured/replay evidence, never
from graph timings alone. Neither this collector nor the CPU fitter fabricates
one. The public strict-validation producer remains an explicit pending piece.

A clean-checkout colleague run must ultimately verify requested versus resolved
enablement, genuine compact routing, completed outputs/task checks and a fair
matched static-versus-enabled benchmark. CPU fake-event tests and smoke imports
are not CUDA replay, full-model accuracy, production admission or performance proof.

The reusable measurement/export module is
`tensorrt_llm/tools/verify_costs.py`; DSpark shape/runtime adapters are under
`_torch/speculative/dspark_cost_profile.py`,
`_torch/speculative/dspark_cost_profiler.py` and the explicit profile-only engine
and request owner under `_torch/pyexecutor`. Future workload adapters can reuse
timing leases, exact-rank reduction and export while supplying their own execution
shape and layout identity. This does not assume unpublished model-specific details.

Focused developer controls are:

```bash
python -B tests/unittest/tools/test_dspark_cost_export.py
python -B tests/unittest/tools/test_dspark_calibration.py
python -B tests/unittest/tools/test_dspark_observations.py
python -m pytest -q tests/unittest/executor/test_dspark_reproduction_factory.py
```

The artifact and observation suites use CPU data and finite lifecycle controls;
they do not qualify CUDA events or real distributed execution. The factory suite
needs a matching native import environment but does not construct a model. It
covers the ordinary public argument boundary and the explicit observation/profile
constructor seams, including resource-governor rejection for the profile owner.
It also checks the imported parent replay keyword/default contract and preserves
the same position-offset object in native and measured profile replay branches.
It also covers native-collection and enabled-observation graph parents, including
the same position-offset forwarding and routing bookkeeping.
The file contains fourteen parameter cases; its complete matching-runtime pytest run
is distinct from selected-source CPU boundary controls.
These controls are not full-model GPU, real-MPI, accuracy or performance results.
