# Confidence calibration and a fair static/enabled comparison

These commands have completed a bounded full-model GPU integration for physical
K3 and global BS512 (G64 across eight attention-DP ranks) in an earlier, separately
recorded Python/native environment: whole-step cost profiling/export, genuine
paired calibration collection/fitting, and independent static and enabled-compact
generation with output and natural-shutdown checks. A newer source composition
has separately passed native/API compatibility and a same-source, same-node
static/enabled-compact comparison at physical K3, global BS512 and G64 across
eight attention-DP ranks, with 512 requests of 32 output tokens. Both independent
model lifetimes completed source/native/artifact/input/output/routing, public
comparison and natural-shutdown checks. The original calibration and costs were
reused; exact generated token sequences were not equal. These are distinct
evidence scopes, not fresh current-source cost/calibration measurements, an
exact-source native rebuild, task accuracy, sustained-serving performance or
global clean-install qualification. Original source/native provenance is
preserved. The tools take explicit model, input, K, batch-size and output
arguments, with no private harness, cluster, account or node requirement.

The two routes are distinct:

- Reuse a structurally valid supplied calibration/cost table, preserving their
  original measurements and provenance. Once confidence is explicitly enabled,
  reuse is the default and warnings explain that predictions may be stale.
- Generate new calibration and measured whole-step costs using the offline
  commands below. Optional freshness/economic validation is a separate choice;
  it is not required before reusing valid artifacts.

Confidence scheduling remains off by default. Target verification, sampling,
KV/graph contracts, kernel guards and fatal errors are not relaxed by reuse.
Poor cost/calibration estimates can choose a slower verifier shape; reuse is
not a performance guarantee.

## 1. Build or reuse an ordinary compatible runtime

Start from a complete public checkout containing these tools, the configurable
cost-profile adapter, and the default-reuse consumer. Use the repository's
[source-build instructions](../../../../docs/source/installation/build-from-source.md)
or its supported precompiled/editable installation route for compatible native
artifacts. Unrelated Python changes do not force native recompilation. Preserve
the original native provenance; the tools separately record loaded Python bytes.

Use your cluster's ordinary allocation/container/MPI launch procedure and model
mounts. Keep the intended precision, attention backend, fused/prenorm paths,
parallelism and original startup/model/shutdown budget. Do not change numerical
configuration or disable failed hardware/operator checks to make a command run.
The code neither repairs machines nor changes cluster/security policy.

Optional developer constructor regression checks do not construct a model.
Run them in a matching installed native runtime; some imports still require an
available driver. They are not a required colleague pre-run gate or a
driverless/native GPU qualification:

```bash
python -m pytest -q tests/unittest/executor/test_dspark_reproduction_factory.py
```

This exercises the actual public argument validator and the existing internal
factory seam without constructing a model. Internal `executor_cls` is not a
public LLM argument and remains rejected. Passing this check is not a native
GPU/operator/model result.

Prepare tokenized JSONL inputs: each line has a nonempty integer
`prompt_token_ids` list. Model weights, inputs and raw observations stay in
access-controlled storage; they are not published with the tooling. The selected
first `--batch-size` rows form one fixed request cohort. Each request generates
the same requested output length with greedy sampling and EOS ignored. The
original sequence-length limit must contain prompt plus output.

Use a YAML accepted by `TorchLlmArgs`: matching physical K, intended topology,
enough rank-local request/token/KV capacity and padded decode graphs. Aggregate
PP=1/CP=1 generation is supported by this initial driver. Guided decoding,
LoRA, custom logits processing, disaggregated transfer, in-graph sampling and
resource-governor control are not part of this bounded fixed-cohort workflow.
No alternative attention backend is selected by the tool.

## 2. Reuse supplied artifacts, if that is your choice

Set `--cost-table` and optionally `--calibration` for the dynamic command in
step 5. No new cost profiling, forged live fingerprint or admission receipt is
needed first. The normal loader still validates table structure, physical K,
legal captured graph cells, finite costs and calibration geometry. Missing or
incompatible cost coverage cannot be reported as dynamic throughput. Default
reuse permits a configured subset of the originally measured graph ladder while
validating the complete original table and fingerprint. Any drain-guard tail
must remain both measured and captured.

`validate_confidence_artifact_freshness` stays at its public default `False`.
Users who choose `True` must supply the corresponding genuine destination
fingerprint and workload-admission artifacts; this driver does not fabricate
them. Original cost/calibration hashes are recorded separately from the runtime
actually used for a new comparison.

## 3. Generate calibration from real native full-K proposals

```bash
mpirun -n "$RANKS" python tensorrt_llm/tools/dspark_reproduction.py run \
  --mode collect --model "$MODEL_DIR" --config "$RUNTIME_YAML" \
  --inputs "$INPUTS_JSONL" --source-root "$SOURCE_ROOT" \
  --physical-k "$K" --batch-size "$BS" --output-tokens "$OUTPUT_TOKENS" \
  --timeout "$FULL_BUDGET_SECONDS" --output "$RUN_ROOT/calibration-observations"

python tensorrt_llm/tools/dspark_calibration.py \
  --shards "$RUN_ROOT"/calibration-observations/rank-*-pairs.pt \
  --physical-k "$K" --output "$RUN_ROOT/calibration.json"
```

The current collector supports the original overlap scheduler input seam;
`disable_overlap_scheduler: False` must already be your intended configuration.
It runs native full-K verification and evaluates the existing confidence head.
There is no confidence-selected shortening and no loaded cost table. Real
warm-up cohorts may contribute calibration observations, but never benchmark
timing. The fitter's default sample floor is 1000 genuine paired rows; collect
enough requests/output tokens rather than weakening that floor for real use.

Pairing is not by arrival order. Step n's confidence and newly proposed tokens
describe the proposal verified later, usually at n+1. The adapter binds that
later forward to the exact original input `SampleState.device` object. It
snapshots verification-input proposals before persistent buffers can be reused,
and captures the producing confidence row and draft stamp before the normal
sampler records its existing event. It reads those host copies only after the
original sampler's normal synchronization/update. Request incarnation, source
proposal content, worker-local confidence row, sampler slot and full-K extent
must agree. Prefill placeholders, retired requests and unscored bootstrap rows
are counted, not fitted as genuine observations. Wrong-step/stale/censored
pairs fail; the current live confidence is never substituted.

Expected files are `rank-N-pairs.pt`, `rank-N-run.json` and the fitted JSON with
`sts_temperatures`/`temperatures`. Each shard has logits `[rows,K]`, cumulative
accepted-prefix labels, exact join metadata and actual source/runtime identity.
These observations do not establish downstream task accuracy by themselves.

## 4. Generate directly measured costs

Follow [Offline cost profiling](COST_PROFILING.md) and
[Whole-step export](COST_EXPORT.md). Declare native V0 plus legal compact cells
for your configurable physical K and resolved rank-local graph tiers. Use the
`--cost-scope whole-step` producer and export its actual completed measurements
to the schema-v2 table consumed by `confidence_sps_table_path`.

Do not substitute graph-only timing, sum overlapping components or invent
serving prices from calibration samples. The whole-step producer reuses the
original executor-loop timing boundary; profile-only forced cells cannot serve
ordinary traffic or prove automatic-policy gains. Keep measured confidence-head
and policy cost boundaries explicit. This candidate benchmark measures actual
online policy in its E2E time; it does not manufacture a separate overhead or
economic-admission receipt from that result.

For attention DP on, global BS determines rank-local G through the attention-DP
fanout. The current schema-v2 cost exporter supports this fanout or one rank;
multi-rank attention-DP-off costs need their separate schema decision. This
restriction belongs to this cost export, not to generic DSpark support.

## 5. Benchmark ordinary static and genuinely enabled execution

Each command is an independent model lifetime. Use the same original inputs,
physical K, runtime YAML, output work, warm-up count and full feasible budget:

```bash
mpirun -n "$RANKS" python tensorrt_llm/tools/dspark_reproduction.py run \
  --mode static --model "$MODEL_DIR" --config "$RUNTIME_YAML" \
  --inputs "$INPUTS_JSONL" --source-root "$SOURCE_ROOT" \
  --physical-k "$K" --batch-size "$BS" --output-tokens "$OUTPUT_TOKENS" \
  --timeout "$FULL_BUDGET_SECONDS" --output "$RUN_ROOT/static"

mpirun -n "$RANKS" python tensorrt_llm/tools/dspark_reproduction.py run \
  --mode dynamic --model "$MODEL_DIR" --config "$RUNTIME_YAML" \
  --inputs "$INPUTS_JSONL" --source-root "$SOURCE_ROOT" \
  --physical-k "$K" --batch-size "$BS" --output-tokens "$OUTPUT_TOKENS" \
  --cost-table "$COST_TABLE_JSON" --calibration "$CALIBRATION_JSON" \
  --timeout "$FULL_BUDGET_SECONDS" --output "$RUN_ROOT/dynamic"

python tensorrt_llm/tools/dspark_reproduction.py compare \
  --runs "$RUN_ROOT/static/rank-0-run.json" "$RUN_ROOT/dynamic/rank-0-run.json" \
  --output "$RUN_ROOT/comparison.json"
```

This uses the normal worker/request/executor/automatic confidence policy, not
the prescribed profile-only engine. Benchmark arms make no calibration copies.
Actual completed decode graph keys are recorded. Warm-up routing is reset before
the retained cohort; requested flags alone do not prove enabled execution.
The comparison requires both resolved config and runtime enabled status, actual
compact replay in the retained dynamic cohort, unchanged numerical runtime/input
work, completed outputs and natural shutdown. An enabled policy that only chose
native is a valid execution, but this consumer refuses to label it a reproduced
compact/dynamic gain.

Reported TPS is global fixed-cohort enqueue-to-final-response throughput,
including prefill and final drain. It is not measured TPOT, sustained-refill
throughput, isolated verifier time or the historical plateau measurement.
No confidence interval is fabricated. Repeat in balanced order with fresh
lifetimes if you need a stable estimate; feed equal numbers of static and dynamic
records to `compare`. Exact output-token equality is reported separately and is
not silently substituted for the model's original task-accuracy tests.

All previously tried batch sizes 128/256/512/1024/1536/2048 and source-backed
smaller points use the same CLI, with compatible configured graph/K/cost coverage.
K is an input, not fixed to K3. K1 native calibration/static execution is legal
but cannot demonstrate compact dynamic scheduling. There is no new broad BS
validation sweep implied: existing and freshly tested points must be labeled
truthfully, with unperformed points listed as prepared, not validated.
