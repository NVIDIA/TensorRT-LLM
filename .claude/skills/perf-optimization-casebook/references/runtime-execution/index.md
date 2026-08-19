# Casebook — Runtime / Execution

Classic optimizations *above the kernel layer* that change how work is
scheduled, launched, and overlapped — not what is computed. Almost all are
**lossless** (output-equivalent). Add entries using the schema in
[case-template.md](../case-template.md); match on **Applies when** signals, adapt
knob values to the current model/hardware, and measure before claiming a win.

> Routing note: classify host-bound vs GPU-bound **first** (`perf-host-analysis`
> — GPU-idle ratio, host-prep-exposed ratio, per-iteration host breakdown).
> These levers only help when the PyExecutor loop is the bottleneck; on a
> GPU-bound loop they move nothing. Scheduling/graph/overlap knobs are config;
> host-overhead refactors are code changes delegated to `perf-host-optimization`.

## Recurring patterns in this family

Match on these transferable patterns, not on a case title — most situations are
a *variation* of a case, not its exact instance.

- **Push language-boundary crossings off the host hot path.** In the PyTorch
  executor loop, per-iteration objects that wrap a C++ binding type pay a fixed
  cost on *every* attribute read (each `__getattr__` is a Python↔C++ crossing),
  and an object that must cross a *process* boundary gets marshalled twice.
  Three moves remove that cost without touching GPU work: (1) represent hot
  per-iteration objects as pure-Python (`@dataclass`) instead of binding
  proxies; (2) mirror the few C++-derived values the loop needs onto plain
  Python attributes once at construction (the `py_*` field convention) so the
  loop reads Python, not the binding; (3) when an object must cross a process
  boundary anyway, bulk-serialize it in one native call and deserialize lazily
  on the consumer, hoisting just the flag the dispatcher branches on. Host-bound
  win — confirm the loop is host-bound first. _(Instance: [the LlmResponse case](pybind-wrapper-pure-python.md).)_
- **Hoist step-invariant work out of the per-layer loop (loop-invariant code
  motion on the host).** A function called once per decoder layer often
  recomputes values that are constant across *all* layers within a forward step
  — KV-pool reshapes/views, stride factors, block-table and request-index
  slices. Compute them once per step, cache them on the per-step metadata object,
  and reuse across layers; invalidate once at step start with a cheap boolean
  flag plus an identity guard. Keep the cache fields as plain instance attributes
  (init in `__init__`, not class-level annotations) so they stay invisible to
  dataclass/`torch.compile` introspection. Host-bound win — confirm the loop is
  host-bound first. _(Instance: [the DSA pool-view cache case](cache-step-invariant-per-layer.md).)_
- **Keep a `torch.compile` target stable; never (re)define it on the hot path,
  and don't compile trivial work.** A `@torch.compile` decorator on a function
  defined *inside* a hot method rebinds a fresh compiled callable every call, so
  the compiled artifact never sticks and Dynamo re-traces/re-guards each time.
  Hoist the decorated function to a stable method/module-level def (passing only
  the tensors it needs, not a whole metadata object so guards stay small); drop
  the decorator entirely when the body is a single cheap op (the compile
  machinery costs more than it saves). _(Instance: [the MTP/DSA torch.compile-closure case](hoist-torch-compile-closures.md).)_
- **Overlap two independent pieces of work.** Host-prep for step N+1 vs device
  compute for step N (overlap scheduler); two data-independent kernels on a side
  CUDA stream (shared-vs-routed expert, MLA RoPE-vs-uk-BGEMM); periodic background
  bookkeeping vs steady-state compute (online-EPLB rebalance); draft forward vs
  target work (two-model MTP). The hard part is proving data-independence (disjoint
  output slices / disjoint step state); stream-switching only pays off under
  CUDA-graph capture. _(Instances: [overlap scheduler](overlap-scheduler.md), [multi-stream](multi-stream-shared-routed-expert.md), [EPLB](overlap-online-eplb.md), and [two-model MTP](two-model-mtp-eagle.md) cases.)_
- **Process only the real work, not padded-to-max.** Variable-length DP
  collectives, a single dummy request instead of pad-every-rank, capping
  CUDA-graph capture / KV to the reachable set. Carries to any DP/ragged-batch
  collective and any resource sized off `max_*`. _(Instance: [the attention-DP padding case](attention-dp-padding.md); cf. the CUDA-graph capping example in [case-template](../case-template.md).)_
- **Pad the dynamic dimension to a captured bucket so the graph always matches.**
  When CUDA graphs need a fixed shape but per-step token/batch counts vary
  (spec-decode draft width, uneven DP ranks), inject shape-correct dummies up to
  the nearest captured size instead of falling back to eager. _(Instance: [the CUDA-graph padding case](cuda-graph-padding.md).)_
- **Reuse computed state instead of recomputing it.** Cache and reuse the
  compressed/intermediate KV representation across requests on a prefix hit (MLA
  KV-cache reuse); free large intermediates at their last use to trade peak memory
  for batch/KV capacity. Carries to MHA/GQA block reuse, chunked-context reuse, any
  forward with chained large temporaries. _(Instances: [the MLA KV-reuse](mla-kv-cache-reuse.md) and [free-MLA-intermediates](free-mla-intermediates.md) cases.)_
- **Cut launch/dependency latency on SM≥90 with PDL — but verify per kernel.**
  Programmatic Dependent Launch overlaps a consumer's preamble with the producer's
  tail; broad and enabled by default, but it changes inter-kernel ordering and has
  produced real NaN/accuracy regressions, so some kernels must keep it off.
  _(Instance: [the PDL case](pdl.md).)_
- **Trade exact-match draft verification for a bounded relaxed criterion where a
  little divergence is tolerable** (spec-decode acceptance inside a delimited
  low-stakes phase). Lossy — gate to where it is safe and keep an accuracy record.
  _(Instance: [the relaxed-acceptance case](relaxed-mtp-acceptance.md).)_
- **Fold a model-specific auxiliary cache into the unified paged KV-cache manager.**
  Instead of a bespoke shadow allocator, register the side cache (DSA indexer K
  cache, scale-factor pool, landmark cache) as a typed pool sharing block IDs with
  the main KV — it then inherits paging, prefix reuse, host offload, eviction, and
  correct per-pool size accounting. Corollaries: sum mixed-dtype pools **additively
  by physical dtype**; index offloaded blocks by **decoded pool index** (not logical
  block ID); release Python views before the native free. _(Instance: [the auxiliary-cache-in-manager case](auxiliary-cache-in-kv-manager.md).)_
- **Chunk long prefill to bound its memory — and align any auxiliary structure's
  chunking to the primary's.** Fixed-size KV chunks cap prefill activation/logit
  memory (turning a quadratic full-sequence intermediate into a per-chunk one); an
  auxiliary structure (sparse indexer) must gather only the current chunk, recompute
  cache-aware position offsets per chunk, and **inherit the primary chunker's
  boundaries** rather than running a second scheme. _(Instance: [the chunked-prefill case](chunked-prefill-aligned-auxiliary.md).)_
- **Skip the sparse/approximate path when it degenerates to the exact one.** Below a
  size threshold a sparse selection may pick *everything* (selectivity → 1) — detect
  it, skip the selection machinery entirely, and substitute a trivially-constructed
  exact result; when the fast path changes control flow, add a CUDA-graph-key
  dimension so it stays captured. A cousin of "process only the real work." The
  opposite-regime move: shard replicated per-token work across TP ranks at the long
  end. _(Instance: [the degenerate-to-dense case](skip-sparse-path-when-degenerate.md).)_
- **Author a custom op so CUDA-graph capture can include its graphable half.** Split a
  monolithic op at the graphable/non-graphable seam (shape-static token-wise compute
  vs batch/data-dependent work), route it through its registered `torch.ops` op so
  `torch.compile` can trace it, and force straight-line control flow under compile
  (constant-arity `register_fake`, disable length-adaptive branches). The producer-side
  complement to turning piecewise capture on. _(Instance: [the split-custom-op case](split-custom-op-for-piecewise-capture.md).)_

## Cases

Match on **Applies when** / **Generalizes to**, then open that case file. Almost
all runtime/execution cases are **lossless** (output-equivalent).

> Risk key: **lossless** = safe to judge on perf alone · **lossy** = needs an
> accuracy record + rollback before promotion · **lossless\*** = lossless unless a
> quant variant (e.g. FP8 KV) is enabled.

### Overlap, launch & scheduling knobs

| Case | Applies when (signal) | Pattern (generalizes to) | Risk |
|------|----------------------|--------------------------|------|
| [Overlap scheduler](overlap-scheduler.md) | GPU idle between steps; host prep (sched/bookkeeping/input build) on the critical path | pipeline host-prep(N+1) against device compute(N) | lossless |
| [PDL (dependent launch)](pdl.md) | launch-bound SM≥90; back-to-back dependent kernels with launch/grid-tail exposed | producer-tail / consumer-preamble overlap per dependent pair | lossy |
| [Remove attention-DP padding](attention-dp-padding.md) | attention-DP with ragged per-rank tokens padded to max; low MFU | process only real tokens, not padded-to-max (length-aware collectives) | lossless |
| [Skip sparse path when degenerate](skip-sparse-path-when-degenerate.md) | a sparse/approximate path equals dense below a length (DSA short seq) yet still pays selection overhead | skip the selection when it degenerates to exact; keep both paths graph-captured | lossless |

### Host overhead — executor & per-step code

| Case | Applies when (signal) | Pattern (generalizes to) | Risk |
|------|----------------------|--------------------------|------|
| [Pure-Python hot wrapper](pybind-wrapper-pure-python.md) | PyExecutor loop CPU-bound; per-iter handling on a pybind wrapper (per-field `__getattr__`) / pickled for IPC | push language-boundary crossings off the host hot path | lossless |
| [Cache step-invariant per-layer values](cache-step-invariant-per-layer.md) | host-bound; a per-decoder-layer fn recomputes step-invariant views/strides/slices | loop-invariant code motion on the host — hoist & cache once per step | lossless |
| [Hoist torch.compile closures](hoist-torch-compile-closures.md) | host-bound; a hot method defines a `@torch.compile` fn inside its body (recompiles/call), or compiles a trivial op | keep a `torch.compile` target stable & module-level; don't compile trivial work | lossless |
| [Move bookkeeping into a C++ op](move-bookkeeping-into-cpp-op.md) | host-bound short GPU step; Python per-step reshape/stride + `.item()` count syncs | push Python tensor-wrangling + count syncs into the op that already runs | lossless |

### System scheduling & parallelism

| Case | Applies when (signal) | Pattern (generalizes to) | Risk |
|------|----------------------|--------------------------|------|
| [Multi-stream shared/routed expert](multi-stream-shared-routed-expert.md) | launch/latency-bound small batch; two independent per-layer sub-computations; CUDA graphs on | issue two data-independent kernels on a side stream to overlap | lossless |
| [Overlap MLA RoPE with uk-BGEMM](overlap-mla-rope-uk-bgemm.md) | MLA decode, low-latency + CUDA graphs; up-proj BGEMM and RoPE+cache-write run serially | overlap two independent intra-op steps on an aux stream (disjoint output slices) | lossless |
| [Overlap online-EPLB rebalance](overlap-online-eplb.md) | MoE EP>1 + ADP; expert-load imbalance; online EPLB (`layer_updates_per_iter>0`) rebalance bubble | move periodic background bookkeeping/weight-movement off the critical stream | lossless |
| [MLA KV-cache reuse](mla-kv-cache-reuse.md) | compute-bound MLA prefill with repeated prefixes (shared prompts/multi-turn), SM90/100 | cache & reuse the compressed/intermediate representation across requests | lossless\* |
| [Free MLA intermediates](free-mla-intermediates.md) | memory-bound MLA; peak activation memory caps batch/KV (OOM at higher concurrency) | drop refs to large intermediates at last use; trade peak mem for batch/KV | lossless |
| [Auxiliary cache in the KV manager](auxiliary-cache-in-kv-manager.md) | a model-specific side cache (DSA indexer K) on a bespoke shadow allocator; can't reuse/offload; wrong size accounting | fold a side cache into the unified paged KV manager as a typed pool | lossless |
| [Chunked prefill + aligned auxiliary](chunked-prefill-aligned-auxiliary.md) | long-context prefill OOM / capped ISL; a quadratic auxiliary intermediate (indexer MQA logits) | chunk prefill to bound memory; align the auxiliary's chunking to the primary's | lossless |

### CUDA graphs

| Case | Applies when (signal) | Pattern (generalizes to) | Risk |
|------|----------------------|--------------------------|------|
| [Piecewise CUDA graph](piecewise-cuda-graph.md) | launch-bound gen; full-graph capture blocked by a non-graphable op (attention); already on torch.compile | graph the graphable, run the rest eagerly (split at the un-capturable op) | lossless |
| [CUDA-graph padding](cuda-graph-padding.md) | launch-bound gen; graphs miss because per-step token/batch count varies (MTP draft width, uneven DP) | pad the dynamic dim to a captured bucket so the graph always matches | lossless |
| [Split a custom op for piecewise capture](split-custom-op-for-piecewise-capture.md) | one custom op mixes graphable (projections) + non-graphable (cache scatter, top-k) work → the whole op stays eager | split at the seam; route through the registered op; straight-line control flow | lossless |

### Speculative decoding / MTP

| Case | Applies when (signal) | Pattern (generalizes to) | Risk |
|------|----------------------|--------------------------|------|
| [Relaxed MTP acceptance](relaxed-mtp-acceptance.md) | MTP spec-decode, low accept rate, reasoning model with a thinking phase (R1); not with attention_dp | trade exact-match draft verify for a bounded relaxed criterion where divergence is tolerable | lossy |
| [Two-model MTP-Eagle overlap](two-model-mtp-eagle.md) | MTP-Eagle where the one-model path constrains and the draft forward is exposed | split a fused speculator into target+draft so the draft forward overlaps target work | lossless |

## Suggested slots (optional — replace or delete)

Areas still open in this family: scheduler ceilings (`max_batch_size` /
`max_num_tokens` / `max_seq_len`) and further host/sync-free refactors (delegate to
**perf-host-optimization** / **perf-torch-sync-free**). (Chunked prefill is now
covered by [chunked-prefill + aligned auxiliary](chunked-prefill-aligned-auxiliary.md).)
