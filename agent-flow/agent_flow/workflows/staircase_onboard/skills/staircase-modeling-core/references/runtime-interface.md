# Target integration with the TensorRT-LLM runtime

How a target plugs into the TensorRT-LLM engine: registration,
construction, weight loading, the per-step forward duties, KV-cache
boundaries, CUDA-graph discipline, and the fail-fast ladder. Together
with the catalog contracts and `step-args.md`, this is the
complete integration knowledge for writing a target — no TensorRT-LLM
source reading is needed.

Everything below was established by probing and verified by
the qwen3-8b/sm_100/tp1 gates (that target is the readable exemplar of
every pattern named here).
<!-- PORT: verify — qwen3-8b/sm_100/tp1 is not among the targets migrated into
     the tree (README: "the five other targets" were not migrated), so the
     exemplar it points at cannot be read here. Either re-point this at an
     in-tree target or drop the pointer. -->

> **This file is one part of a split.** It was `docs/references/trtllm-runtime-integration.md`;
> its section numbers are kept unchanged so that existing citations — in
> target comments, in the onboarding records — still resolve. The four
> sections that moved:
>
> | former section | now at |
> |---|---|
> | §7 CUDA-graph discipline | `staircase-modeling-core/references/graph-and-sync.md` |
> | §10 What varies per model | `staircase-kernel-feasibility/references/derivation-checklist.md` |
> | §12 Multi-rank launch | `staircase-modeling-parallel/references/multi-rank.md` |
> | §13 One-engine speculative decoding | `staircase-modeling-speculator/references/runtime-boundary.md` |

## 1. Directory shape

```
tensorrt_llm/_torch/_experimental/modeling_v2/
└── models/<family>/
    ├── routing.py                       # which config lands on which target
    └── <ckpt>__<arch>__<parallel>/       # identity = the directory's name
        ├── modeling.py   # registration shell + core + step-args + fail-fast
        └── weights.py    # MANIFEST + load loop + post-load derivations
```

That is the whole target. There is no stub model directory, no
`llm_args.yaml`, no `smoke.py` and no `TARGET.md`: **the checkpoint is read
exactly as published**, the topology comes from the routing decision and the
boot gate's CLI flags, and gate records go in the run journal. `LLM(model=...)`
is pointed at the real checkpoint, and everything the runtime learns about the
model it learns from that checkpoint's own `config.json`.

## 2. Registration and resolution

- The target's shell class carries `@register_auto_model(<synthetic name>)`,
  which writes the class into trtllm's process-global registry at import time.
  **The name is routing's, not the target's**: `models/<family>/routing.py`
  decides it and lists the module in `TARGET_MODULES`, and
  `modeling_v2_resolve` imports that module once its tree has claimed the
  config — nothing else would put a class behind the synthetic key. See
  `routing-identity.md`.
- The import must precede engine construction in **every process that
  builds the model** — which above world size 1 is not this one. An ordinary
  dotted module satisfies that by itself: each rank resolves the config, the
  routing module imports the target, and pickle resolves the class by module
  name. §12 has the rest of what changes there.
- `LLM(model=<checkpoint>)` then resolves the configuration through
  `AutoModelForCausalLM._resolve_class`, which consults
  `modeling_v2_resolve` and looks the returned name up in the ordinary
  registry, and construction begins.

### The runtime still keys behaviour on the architecture string — and it reads the checkpoint's

Parts of the runtime branch on `architectures[0]` (and on other published
config fields) to decide **what to allocate**, not merely which class to
build. The known case is hybrid linear attention: a recurrent-state pool is
allocated only when the predicate matches, and otherwise the model builds, the
boot gate passes, and the linear layers have no state.

In tree that predicate reads the **checkpoint's own** config. The synthetic
name exists only as a local rewrite inside `_resolve_class`;
`pretrained_config.architectures` is not mutated, so nothing downstream sees
it. Two consequences:

- a target **cannot** change what the runtime allocates by choosing its
  registered name, and must not try. If the runtime's predicate does not match
  the published config, that is a fact about the checkpoint and the runtime —
  report it as a symptom rather than routing around it;
- conversely the published name is always the honest one, so the failure mode
  the standalone project had to defend against (a stub `config.json` declaring
  a name the runtime did not recognize) cannot occur.

The registry facts behind this are unchanged and still worth knowing, because
they are what makes a target's registration win its slot:

- `MODEL_CLASS_MAPPING` is consulted **before** the lazy static index
  `_torch/models/_arch_index.py`, so a name already in the mapping never
  triggers the built-in module's import. Verified by resolving
  `Qwen3_5MoeForCausalLM` — a name that index maps to `modeling_qwen3_5` —
  with a locally registered class in place: `get_model_architecture` returned
  the local class and the built-in module was never imported.
- `register_auto_model` **overwrites silently** on a duplicate name; there is
  no exception and last registration wins. A synthetic name no checkpoint
  declares is what keeps that from mattering.
- `executor/worker.py` does `MODEL_CLASS_MAPPING.update(**_torch_model_class_mapping)`,
  which is how the registration crosses into the executor worker process.

*Re-check the precedence by registering a throwaway class under a name present
in `_arch_index`'s `MODEL_ARCH_TO_MODULE` and confirming
`get_model_architecture` still returns it; the precedence, not the name list,
is what matters.*

## 3. The shell and the core

```python
class ModelingV2Core(DecoderModel):                 # the computation
    def __init__(self, model_config): ...          # geometry asserts + weights
    def forward(self, attn_metadata, input_ids=None, position_ids=None,
                inputs_embeds=None, lora_params=None, **kwargs): ...

@register_auto_model("ModelingV2<Ckpt><Sm><Par>")
class ModelingV2<Ckpt><Sm><Par>(DecoderModelForCausalLM[ModelingV2Core, ...]):
    def __init__(self, model_config):
        super().__init__(ModelingV2Core(model_config), config=model_config,
                         hidden_size=..., vocab_size=...)
    def load_weights(self, weights, *args, **kwargs): ...
    def post_load_weights(self): ...
```

- The shell (`DecoderModelForCausalLM`) is composition: it stores the
  core as `self.model`, creates `self.lm_head`, and owns the logits path
  — its forward calls the core, then gathers exactly the rows that need
  logits (last token per context sequence + every generation token) and
  applies lm_head. **The core returns final-normed hidden states
  `[num_tokens, hidden]` and never touches logits.**
- Inputs are **packed**: first dimension = total tokens of the batch,
  context sequences first. `position_ids` arrives as an int32 `[1, T]`
  view of an engine buffer — flatten with `reshape(..., [-1])`.
- Construction asserts the target's identity — geometry, dtype, and every
  config property the assembly depends on (tie_word_embeddings, qk-norm
  presence, rope scaling, sliding window, attention bias). Assert, never
  adapt: a mismatched checkpoint is a different target. These asserts and the
  routing tree's criteria are two readings of the same fact; routing declines
  the configuration, the asserts refuse it if it arrives anyway.
- Inputs the target does not implement (`lora_params`, `spec_metadata`
  in kwargs) must **assert loudly** — silently ignoring them produces
  wrong output with no signal. This is distinct from runtime-owned
  feature flags, which pass through (see §6). A target that *does*
  implement speculative decoding stops asserting on `spec_metadata` and
  grows a second forward it owns; §13 is that axis.

## 4. The weight lifecycle

| stage | who | what happens |
|---|---|---|
| t0 construct | engine (`MetaInitMode`) + your `__init__` | `torch.empty` is intercepted to the meta device: parameters are shape/dtype shadows, zero memory. Declare with `torch.empty` only (`zeros`/`full` are NOT intercepted) and do no tensor math in `__init__` (meta tensors raise on compute). |
| t1 materialize | engine | walks **registered** parameters (`nn.Parameter` / `ParameterDict` / module tree — what `named_parameters()` can see) and reallocates each on CUDA, contents garbage. Registration is what requests the allocation; anything held in a plain dict/list stays a dead shadow. |
| t2 read | engine | reads every safetensors in the checkpoint directory into a dict `{ckpt_key: tensor}`. The values arrive **already materialized** as `torch.Tensor`, not as lazy slices — measured at world size 4, where every rank is handed the whole dict (25,687 keys on all four). So the engine pre-shards nothing: a multi-rank split is entirely the manifest's, and a `src` transform's benefit is that only this rank's bytes cross to the device, not that fewer bytes are read off disk. |
| t3 load | your `load_weights(weights)` | full delegation, never validated by the engine — the manifest loop copies bytes into the materialized storage (`param.data[a:b].copy_(src)`). |
| t4 derive | your `post_load_weights()` | the designated home for derived state: `.t()` GEMM views, per-layer tuples, and checkpoint-calibrated call tensors such as fp8 KV scales. Real tensors may be created here — meta is over. |
| t5 serve | your `forward` | reads weights by reference; zero copies or transforms on the hot path. |

Storage layout: declare parameters in **HF `[out, in]` row-major** so t3
copies are layout-preserving, and consume GEMMs through `.t()` views
(row-major `[N, K]` transposed is exactly the dense column-major `[K, N]`
that `cublas_mm`'s contract requires; the view is zero-copy and shares
storage, so reloading weights in place keeps views valid).

## 5. The weight manifest convention

`weights.py` owns a pure data table:

```python
MANIFEST[param_key] = [(ckpt_key, dst_slice_or_None), ...]
# tp1: no src transform. Sharded targets add one:
#   (ckpt_key, src=col_shard(rank, tp), dst_slice)
# and declare per-rank shapes in modeling.
```

Rules: fused parameters (qkv, gate_up) fill by destination row slices —
no intermediate concat buffers; assert shape and dtype per copy; assert
**bidirectional coverage** (manifest keys == declared parameters before
the loop; consumed ckpt keys == the whole checkpoint after it, minus the
explicitly predicted non-load set). The
shell-registered exceptions (`lm_head.weight`, and `embed_tokens` when
the base class owns it for tied checkpoints) are fed explicitly.

## 6. Forward duties, KV cache, and feature flags

The engine drives everything around the forward: scheduling, KV block
bookkeeping (allocation at admission, per-token growth, prefix reuse),
metadata filling and `prepare()` — all before your forward runs. Your
duties per step:

1. Build the attention step-args once from the prepared metadata and
   share them across layers (`step-args.md` is the
   normative mapping).
2. Keep the loop flat: catalog calls, tensor-metadata reads, Python
   control flow — nothing else (the closed-vocabulary rule; audit =
   collect the calls in `forward` **and the private methods it reaches**
   and match them against
   `tensorrt_llm/_torch/_experimental/modeling_v2/catalog/index.yaml`. Scope
   to that closure: a
   whole-file scan flags the rope table's `arange`/`cos`/`sin` and
   load-time `.t()`/`.to()`, which run at init and are outside the rule).
3. Let one attention call serve any batch composition
   (`attention_input_type=0` for the standard configuration — no phase
   branches; MLA targets dispatch per phase by metadata reads).

The KV-ownership red line does not move: the pool is sized by the engine from
the **checkpoint's own** `config.json` declarations (`num_hidden_layers`,
`num_key_value_heads`, `head_dim`, `torch_dtype`, `vocab_size`,
`max_position_embeddings`), pages are assigned by the C++ manager, and the
new tokens' K/V are written **inside** the attention op. The target only
forwards the address book — it restates none of those fields and must not
contradict them; its construction asserts read the same config the engine
sized the pool from.

Runtime-owned feature flags (block reuse, CUDA graphs, beam, spec-dec
plumbing) **pass through** from metadata exactly as prepared: the target
runs trtllm defaults, feature behavior is upstream's responsibility, and
the gates validate the result. Kernel-surface certification accounting
lives in the catalog contracts, not in target config.

## 8. The fail-fast ladder

| fuse | when | checks | catches |
|---|---|---|---|
| static op declaration | CI, `tests/unittest/_torch/modeling_v2/test_modeling_v2_target_contract.py` | every `torch.ops.trtllm.*` symbol named in the target's `REQUIRED_TRTLLM_OPS` is registered in this build; the thop binding exists | missing/renamed ops in a drifting engine. torch mirrors (`embedding`, `empty`, `reshape`, ...) are exempt — core PyTorch API, upstream-owned |
| step contract | first forward | metadata is `TrtllmAttentionMetadata`; every `_STEP_FIELDS` name exists; `position_ids` is int32 | private-surface layout drift the op list cannot see. Everything checked is fixed at engine construction — once per model instance is sound |
| gates | explicit runs | boot-gate keywords; accuracy score vs `tests/integration/defs/accuracy/references/<benchmark>.yaml` | the computation itself |

There is no version pin to assert against — a target moves with the trunk, so
the assert would be meaningless — which is exactly why the first two rows
carry the weight they do. Keep `_STEP_FIELDS` exactly equal to the set of
metadata attributes the code reads, and `REQUIRED_TRTLLM_OPS` exactly equal to
the trtllm ops the forward calls: the lists are dependency declarations, and a
stale entry is as wrong as a missing one.

**One metadata surface is conditional, and a flat existence check over it
turns a legal config into a hard failure.** MLA's cached-KV fields —
`enable_context_mla_with_cached_kv`, `ctx_cached_token_indptr`,
`ctx_kv_indptr`, `ctx_uncached_token_indptr`, `max_ctx_seq_len`,
`max_ctx_kv_len`, `num_ctx_cached_tokens` — exist only when block reuse
is on. With `enable_block_reuse: false` they are **absent, not False**, so
a `_STEP_FIELDS` tuple containing them fails the first forward on a
configuration the target is supposed to support. Split the tuple: the
unconditional fields keep the existence check, and the conditional ones
select the context flavor by their *presence* rather than being asserted.
See `.claude/skills/staircase-modeling-core/references/latent-kv-cache.md`.

## 9. Gates and launch

- **Every GPU command in a run is launched one way: the run's launch prefix.**
  It carries the environment the ranks need — `TRTLLM_MODELING_V2` above all,
  which must be exported before the ranks start and is not yours to set — and
  the declared interpreter. Do not construct a launcher, and do not probe for
  one. The devices claimed for the run are given to you: one for a `tp1`
  target, `N` for a `tepN`/`depN` one.
- Boot gate: `examples/llm-api/quickstart_advanced.py` with the target's
  topology flags and `TRTLLM_MODELING_V2=require`, run by the orchestrator.
  Engine cold start, weight-manifest coverage, greedy keyword asserts; the
  exit verdict is binary. The cases are target-authored: pick prompts whose
  greedy continuation is high-confidence for this model, and **verify every
  keyword on the real model before freezing it** — an unverified keyword bakes
  a false failure into the gate, which is why the keyword list starts empty and
  the first engine run doubles as its verification.
- Release: `trtllm-eval` under the run's launch prefix, with the protocol from
  `tests/integration/defs/accuracy/references/<benchmark>.yaml` and
  `TRTLLM_MODELING_V2=require` — one-sided gate,
  measured >= reference − tol. On the MMLU
  path — which is where an entry with no `eval` block lands — add
  `--dataset_path /home/scratch.trt_llm_data/llm-models/datasets/mmlu`,
  because trtllm-eval's own download 404s and fails as a corrupt tar rather
  than as a missing file.

## 11. Serving-time costs a target inherits

The engine owns these, not the forward, but they land on the target's
Pareto curve and the first two are worth checking on every new target
before any modeling work.

**Decode CUDA-graph coverage defaults to batch 128.** Above it the whole
decode phase runs eager. The deeper the model the worse this is: measured
on a 48-layer target, concurrency 256 lost 19.7% throughput and 151% TPOT
against its own graph-covered configuration, and raising
`cuda_graph_config.max_batch_size` to 256 with `enable_padding: true` was
worth +56.5% at that point with every other point unchanged. This should
be the first config experiment on any new target — but raise the ceiling
and A/B the padding **separately**: on a later target `enable_padding:
true` measured 11% *worse* at concurrency 256 against the same raised
ceiling, because it swaps the fine `[1..32]` batch grid for a coarse one.

**The stock decode capture grid is the fine one.** `enable_padding` defaults to
**False**, so a stock engine captures `[1..32, 64, 128]` — 34 batch sizes, every
integer up to 32 — not the sparse padded grid `[1, 2, 4] + [8, 16, …, 128]` that
padding produces. This is the list a certified-column audit must be read against,
and assuming the padded one leaves the odd small counts (6, 9–15, 17–23, 25–31)
uncovered while looking complete. Enabling padding also makes `192` reachable
only when `max_batch_size >= 192`.

**`max_seq_len` is a per-step cost, not only an admission cap.** It sizes
`max_blocks_per_seq = max_seq_len / tokens_per_block`, which sizes a
pinned `[1, num_seqs, 2, max_blocks_per_seq]` int32 block-offset staging
buffer that the resource manager allocates, memcpys and pushes H2D **every
step**. Left at a model cap far above the served workload (40960 vs a
1024/1024 benchmark) that is 2.62 MB per step at 256 in flight. Capping it
is a real knob — but size the expectation: a measured 10× shrink moved
`_prepare_inputs` only 5.17 → 4.44 ms/step, so the staging is a minority
of that cost and the residual is `O(num_seqs)` per-request executor
Python. Capping also **restricts capability** (longer requests are
rejected), and the floor is set by the accuracy gate's own prompts, not by
the benchmark — 5-shot MMLU prompts reach 2687 tokens, so a cap at the
benchmark's exact ISL+OSL makes the gate unrunnable.
