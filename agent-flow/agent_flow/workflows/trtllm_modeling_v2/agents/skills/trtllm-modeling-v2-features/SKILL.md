---
name: trtllm-modeling-v2-features
description: Load when working on, testing or debugging a TensorRT-LLM engine feature around a ModelingV2 target -- KV-cache block (prefix) reuse, disaggregated serving, chunked prefill, KV-cache dtype (fp8/nvfp4 and their scales), CUDA graphs and the overlap scheduler, multimodal inputs, speculative decoding, or attention-DP/TP/EP where it changes one of those -- including a check that fails on a feature field, or a number that moves only with a feature on. Says which engine stage owns the feature and what the target must honour, where its code and tests are in the tree, and how to prove it from a CPU check up to the flow's accuracy, disagg or acceptance gate. Not needed to write or certify one catalog entry (trtllm-modeling-v2-entry) or to debug core accuracy with every feature at its default.
---

# ModelingV2 runtime features

## When to load

Load when a task names one of these features, when a verdict names a feature
field (`block_reuse`, `chunked_prefill`, `kv_cache_dtype`, `cuda_graph`,
`speculative`) or the disagg test, or when a result moves only with a feature on.
Open the feature's section in `references/features.md`, read the code it points
to in the current tree, and imitate the tests it names.

Paths here are where things were when this was written; the code moves, and
what it defines moves with it. Every path sits next to a symbol (a class,
function, flag or test name): when the path is missing, `git grep -n` the
symbol and use where it is now. `tree_map.md` in the run's workspace says where
the backends, accuracy and perf tests and CI test lists are in the checkout
under test. Where the tree disagrees with this skill, the tree wins.

## Where a target sits

A request runs LLM API -> executor -> resource managers -> model engine ->
**the forward** -> sampler. A target is the forward and nothing else.

| stage | owns | read first (`_torch/` = `tensorrt_llm/_torch/`; paths as written, symbols find them) |
|---|---|---|
| LLM API, loader | arguments, KV-cache quant resolution, `modeling_v2` routing | `tensorrt_llm/llmapi/llm_args.py`, `_torch/pyexecutor/model_loader.py` |
| executor loop | admission, batching, context chunking, overlap loop, attention-DP router and dummy requests, cache-transceiver calls | `_torch/pyexecutor/py_executor.py`, `_torch/pyexecutor/scheduler/` |
| resource managers | blocks: allocate, free, prefix reuse, snapshots, side buffers, and what disagg transfers | `_torch/pyexecutor/kv_cache/`, `resource_manager.py`, `get_kv_cache_manager_cls` in `_util.py` |
| model engine | per-step attention/spec/multimodal metadata; CUDA-graph plan, capture, replay | `_torch/pyexecutor/model_engine.py` (`ModelEngine`), `_torch/pyexecutor/engine/` |
| **the target** | one forward: reads metadata, writes into the pools it is handed | `_torch/_experimental/modeling_v2/` |
| sampler, speculative | tokens, draft verification, acceptance | `_torch/pyexecutor/sampler/`, `_torch/speculative/` |

The target never schedules, allocates per-request storage, samples or
transfers. One rule explains most feature bugs: **per-sequence state the
target keeps outside manager-owned pools is invisible to block reuse, eviction,
partial-reuse copies, CUDA-graph replay and the disaggregated transfer.** Such
a target passes every single-engine, single-request measurement and fails the
first time a feature crosses that state.

## Testing ladder

Climb only as far as the claim needs. Each rung says what it cannot see.

1. **Source and CPU (seconds).** Grep the forward for what the feature forbids:
   host syncs or value-dependent Python branches under graphs, a context
   assumed to start at position 0, state in target-owned tensors. Run
   `tests/unittest/_torch/modeling_v2/test_modeling_v2_claims.py` and
   `test_modeling_v2_routing.py` (no GPU) and `trtllm-modeling-v2 intake lint`
   on the spec. *Blind to* numerics and to whether the engine turned the
   feature on.
2. **In-tree unit tests to imitate (minutes, one GPU).** Each section names the
   op-, cache- or transfer-level test that already exercises the feature. Copy
   its shape for the target's own case: an op cell with a cached prefix, a
   manager test carrying the target's side buffer, a transfer test with its
   role. *Blind to* the target's forward under the real scheduler.
3. **One-member boot with the feature on.** `trtllm-modeling-v2 gate boot
   --task task.yaml --unit <unit or variant> --member <m>` runs it as the
   orchestrator does; `--mode off` reruns the same command on the stock path as
   a control. Read the verdict's `[engine ...]` segment: every declared field is
   read back from every rank's worker (`MV2FLOW_RANK_ENGINE`), with the reuse
   probe and pool sizes. A field read as UNKNOWN, or only from the request, is
   not evidence. *Blind to* accuracy: boot checks a few short continuations
   for keywords.
4. **Accuracy, disagg, acceptance.** `gate accuracy` runs the benchmark
   with CUDA graphs (lint blocks an eager accuracy check) and re-attests each
   rank's KV dtype and chunking; `gate disagg --unit disagg` runs the target's
   disaggregated-serving test as CI does; `gate acceptance` judges
   speculators. *Blind spots* are per feature in the reference; they are why
   each feature carries a proof of its own.

| feature | on in `task.yaml` by | proof that it ran | judged by |
|---|---|---|---|
| block reuse | a `reuse` variant (`kv_cache_config.enable_block_reuse: true`); off in core | reuse probe: a repeated prompt reports `cached_tokens` > 0 | variant accuracy |
| chunked prefill | a `chunked` variant with a small `max_num_tokens`; off in core | a prompt longer than `max_num_tokens` past the cache, served | variant accuracy |
| KV-cache dtype | a variant setting `llm_api_options.kv_cache_config.dtype` | each rank's resolved `kv_cache_quant_algo` | variant accuracy |
| CUDA graphs | `runtime.cuda_graph` | each rank's graph settings and capture plan (not replays) | accuracy (graphs always on) |
| overlap scheduler | `runtime.overlap_scheduler` (default on, core) | the argument only | nothing of its own |
| disaggregation | feature `disagg` + `accuracy.disagg.test` | a benchmark scored through `trtllm-serve disaggregated` | the tree's disaggregated-serving test |
| multimodal | feature `multimodal` | an image prompt answered | boot image, image benchmark |
| speculative | features `mtp`, `eagle3`, `dflash`, `dspark` | each rank's algorithm and depth; accepted tokens per iteration | acceptance gate |

## Rules that save hours

- Start from the in-tree test that already runs the feature (each section of
  the reference names them) rather than writing a launcher, server, transport
  or harness: the engine moves state and serves, the flow deploys and measures,
  and the target's work is what they hand it. How another engine (vLLM,
  SGLang) serves and tests the same feature for this model family is evidence
  too.
- Prove a feature **on versus off in one session**, compared by token ids --
  not text prefixes, and not scores from different sessions.
- A result that moves only with reuse, chunking or drafts on is the
  paged-context path, not the model: all three turn on `use_paged_context_fmha`
  and make a context call attend to KV it did not compute in this step.
- The core configuration is CUDA graphs and the overlap scheduler at TP, TEP
  and ADP, with the KV and SSM caches in the checkpoint's own dtype; block reuse
  and chunked prefill are off there. Prove every other feature once, as one
  variant on core at the member the request validates fully (`spec/INTAKE.md`),
  never two at once.
- **Write the disaggregated test the way the tree's own are written,** in
  `tests/integration/defs/accuracy/test_disaggregated_serving.py`: a
  `LlmapiAccuracyTestHarness` class with context and generation server configs,
  `launch_disaggregated_llm`, the benchmark task of `accuracy_core.py` (GSM8K
  and its dataset under `$LLM_MODELS_ROOT/datasets/`), the CI list's timeout.
  Its layout is `accuracy.disagg` (derived from the checkpoint's size): on
  `devices` of one node, a context and a generation engine of half each, as
  `TestDeepSeekV4Flash` puts TP2/EP2 and TP2/EP2+ADP on four GPUs; with `nodes`,
  one engine per node, its role from `MV2FLOW_JOB_NODE_RANK`. Never hand-roll a
  launcher, a server or a transport. What the test keeps for after it (HTTP
  records, continuations, transfer logs) goes under `$MV2FLOW_GATE_ARTIFACTS`:
  the snapshot the gate runs in is pruned once it ends.
- A change outside the target is allowed where the target truly needs it (a
  runtime, executor or serving file of the tree): follow the ModelingV2 design,
  keep the model's own logic in the target, and make the shared change the
  smallest general one, never a branch on this model. Name each such file and
  why in your report; the regression check follows it.
- A stock-path control (`--mode off`, `modeling_v2: off`) splits a target bug
  from an engine bug. A new architecture has none; say so instead of guessing.
- Report each feature as *verified* (its proof passed), *configured* (on, not
  proven) or *unsupported*. Never report configured as verified.

## Index

`references/features.md`, one section each, every section in the same order
(what; engine owns; target must honour; code and grep; tests to imitate; fast
check; how the flow proves it; blind spots):

0. Common machinery: enabling features in a spec, running one gate, reading
   engine evidence
1. Block reuse (prefix KV-cache reuse), including side state and recurrent snapshots
2. Chunked prefill
3. Disaggregated serving: what crosses the hop, transceiver, one- and multi-node launch
4. KV-cache dtype: fp8, nvfp4, mixed, and KV scales
5. CUDA graphs and the overlap scheduler
6. Multimodal inputs
7. Speculative decoding
8. Attention DP, TP and EP where they meet the features above
