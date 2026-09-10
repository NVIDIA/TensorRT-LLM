# Target integration with the TensorRT-LLM runtime

How a staircase target plugs into the TensorRT-LLM engine: registration,
construction, weight loading, the per-step forward duties, KV-cache
boundaries, CUDA-graph discipline, and the fail-fast ladder. Together
with the catalog contracts and `thop-attention-step-args.md`, this is the
complete integration knowledge for writing a target — no TensorRT-LLM
source reading is needed.

Everything below was written against `tensorrt-llm == 1.3.0rc21` and
verified by the qwen3-8b/sm_100/tp1 gates (that target was the readable
exemplar of every pattern named here; it is not one of the two migrated in
this batch).

> **Written before staircase moved in-tree.** The *runtime* content — the
> engine's construction order, the per-step forward duties, KV-cache
> boundaries, CUDA-graph discipline, the fail-fast ladder, and §13's MTP
> shell specification — is what this file is for and still holds. The
> *packaging* content does not: `uv run`, `bench/`, `model_dir/`,
> `llm_args.yaml`, `scripts/env.sh`, `STAIRCASE_TARGET` and the multi-rank
> registration ritual are all gone. Sections 1, 9 and 12 are marked where
> they describe the old shape; `../../README.md` is the current one.

## 1. Directory shape

**Superseded — see `../../README.md`.** In-tree a target is a package under
`models/<family>/targets/<ckpt>/<arch>/<parallel>/`; the identity triple is
unchanged, but `llm_args.yaml`, `model_dir/` and `perf/data` are gone (the
topology is the caller's input, the checkpoint is read unpatched). The
pre-move shape, kept because the rest of this file refers to it:

```
targets/<ckpt>/<arch>/<parallel>/     # identity = the path triple
├── modeling.py      # registration shell + core + step-args + fail-fast
├── weights.py       # MANIFEST + load loop + post-load derivations
├── smoke.py         # self-locating keyword-assert gate (uv run <path>)
├── llm_args.yaml    # LLM-API overrides; empty {} = trtllm defaults;
│                    # parallel topology lives here for multi-rank targets
├── TARGET.md        # identity, checkpoint hashes, pins, vocabulary,
│                    # verification records, and — added by the tuner —
│                    # the Performance section
├── configs/         # tuner knob variants (tracked); added by the perf
│                    # campaign, absent on a freshly assembled target
├── perf/            # data/ machine-local and ignored; figures/ tracked
└── model_dir/       # what trtllm loads
    ├── config.json  # committed; architectures[0] = registered class name;
    │                # all other fields = the upstream checkpoint config
    └── *.safetensors, tokenizer*, generation_config.json
                     # machine-local symlinks into the real checkpoint —
                     # untracked; relink against the TARGET.md hashes
```

`model_dir/` is a stub HF checkpoint directory: trtllm resolves the model
class from its `config.json`, reads weights and tokenizer from the
symlinks, and never sees the rest of the target. Tools pass
`model=<target>/model_dir` and read `llm_args.yaml` next to it.

## 2. Registration and resolution

- `@register_auto_model("StaircaseForCausalLM")` on the shell class
  writes the class into trtllm's process-global registry at import time.
  Every target registers the **same** name (identity lives in the
  directory path), so one process hosts one target; comparative runs use
  separate processes.
- Tools import `modeling.py` **by file path** (the target dir name
  contains a hyphen and is not a package):

  ```python
  spec = importlib.util.spec_from_file_location(
      "staircase_target_modeling", target / "modeling.py")
  module = importlib.util.module_from_spec(spec)
  sys.modules["staircase_target_modeling"] = module   # §12: pickle needs it
  spec.loader.exec_module(module)          # registration side effect
  ```

  The import must precede engine construction in **every process that
  builds the model** — which above world size 1 is not this one. The
  `sys.modules` line binds the loaded module to the name the class will
  carry in `__module__`; without it a multi-rank run dies in the launcher
  before any rank starts. §12 has the mechanism and the rest of what
  changes there.
- `modeling.py` bootstraps `sys.path` from its own `__file__` (the repo
  root is the target directory's `parents[3]` — the path depth is frozen
  by the `targets/<ckpt>/<arch>/<parallel>/` shape) before its
  `catalog.*` imports — this is why E402 is disabled for `targets/**` in
  pyproject.
- `LLM(model=<target>/model_dir)` then resolves `architectures[0]`
  against the registry and construction begins.

## 3. The shell and the core

```python
class StaircaseCore(DecoderModel):                 # the computation
    def __init__(self, model_config): ...          # geometry asserts + weights
    def forward(self, attn_metadata, input_ids=None, position_ids=None,
                inputs_embeds=None, lora_params=None, **kwargs): ...

@register_auto_model("StaircaseForCausalLM")
class StaircaseForCausalLM(DecoderModelForCausalLM[StaircaseCore, ...]):
    def __init__(self, model_config):
        super().__init__(StaircaseCore(model_config), config=model_config,
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
  adapt: a mismatched checkpoint is a different target.
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
| t2 read | engine | reads every safetensors in `model_dir/` into a dict `{ckpt_key: tensor}`. At this pin the values arrive **already materialized** as `torch.Tensor`, not as lazy slices — measured at world size 4, where every rank is handed the whole dict (25,687 keys on all four). So the engine pre-shards nothing: a multi-rank split is entirely the manifest's, and a `src` transform's benefit is that only this rank's bytes cross to the device, not that fewer bytes are read off disk. |
| t3 load | your `load_weights(weights)` | full delegation, never validated by the engine — the manifest loop copies bytes into the materialized storage (`param.data[a:b].copy_(src)`). |
| t4 derive | your `post_load_weights()` | the designated home for derived state: `.t()` GEMM views, per-layer tuples, and (future) checkpoint-calibrated call tensors such as fp8 KV scales. Real tensors may be created here — meta is over. |
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
the loop; consumed ckpt keys == the whole checkpoint after it). The
shell-registered exceptions (`lm_head.weight`, and `embed_tokens` when
the base class owns it for tied checkpoints) are fed explicitly.

## 6. Forward duties, KV cache, and feature flags

The engine drives everything around the forward: scheduling, KV block
bookkeeping (allocation at admission, per-token growth, prefix reuse),
metadata filling and `prepare()` — all before your forward runs. Your
duties per step:

1. Build the attention step-args once from the prepared metadata and
   share them across layers (`thop-attention-step-args.md` is the
   normative mapping).
2. Keep the loop flat: catalog calls, tensor-metadata reads, Python
   control flow — nothing else (the closed-vocabulary rule; audit =
   collect the calls in `forward` **and the private methods it reaches**
   and match them against `catalog/index.yaml`. Scope to that closure: a
   whole-file scan flags the rope table's `arange`/`cos`/`sin` and
   load-time `.t()`/`.to()`, which run at init and are outside the rule).
3. Let one attention call serve any batch composition
   (`attention_input_type=0` for the standard configuration — no phase
   branches; MLA targets dispatch per phase by metadata reads).

The KV-ownership red line does not move: the pool is sized by the engine from the
`config.json` declarations (`num_hidden_layers`, `num_key_value_heads`,
`head_dim`, `torch_dtype`, `vocab_size`, `max_position_embeddings` —
declare them honestly), pages are assigned by the C++ manager, and the
new tokens' K/V are written **inside** the attention op. The target only
forwards the address book.

Runtime-owned feature flags (block reuse, CUDA graphs, beam, spec-dec
plumbing) **pass through** from metadata exactly as prepared: the target
runs trtllm defaults, feature behavior is upstream's responsibility, and
the gates validate the result. Kernel-surface certification accounting
lives in the catalog contracts, not in target config.

## 7. CUDA-graph discipline

Decode-only steps are captured and replayed with **no Python executing**.
Every value the attention call consumes must therefore fall into one of
three classes:

- **reference**: engine-owned persistent buffers, refreshed in place each
  step — replays read fresh contents automatically (all step-args
  tensors);
- **GPU-derived**: recomputed by captured kernels on replay;
- **host-derived**: Python scalars frozen at capture — legal only when
  they are per-capture constants (`num_contexts == 0` in a decode graph).

The step-args builder is the single audit point: classify every entry.
Per-forward `empty(...)` output buffers are fine (the caching allocator
serves stable blocks; upstream captures the same pattern).

## 8. The fail-fast ladder

| fuse | when | checks | catches |
|---|---|---|---|
| static contract | import | trtllm version == pin; every `torch.ops.trtllm.*` symbol the forward calls; the thop binding | version drift (voids receipts and gate records), missing/renamed ops. torch mirrors (`embedding`, `empty`, `reshape`, ...) are exempt — core PyTorch API, upstream-owned |
| step contract | first forward | metadata is `TrtllmAttentionMetadata`; every `_STEP_FIELDS` name exists; `position_ids` is int32 | private-surface layout drift within a same-version build. Everything checked is fixed at engine construction — once per model instance is sound |
| gates | explicit runs | smoke keywords; accuracy score vs `bench/references/accuracy.yaml` | the computation itself |

Keep `_STEP_FIELDS` exactly equal to the set of metadata attributes the
code reads, and the static-contract op list exactly equal to the trtllm
ops the forward calls — the lists are dependency declarations.

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
See `docs/models/latent-kv-cache.md`.

## 9. Gates and launch

**Superseded — see `../../README.md` for the current commands.** The
substance below (what each gate is for, and how to author smoke cases)
carries forward; the invocations do not.

- `source scripts/env.sh` before any GPU run (exports the
  single-process-worker flag; consumed upstream only at world_size == 1,
  ignored by multi-rank runs). Claim the devices with
  `CUDA_VISIBLE_DEVICES` — one for a `tp1` target, `N` for a `tepN`/`depN`
  one.
- Smoke: `uv run targets/<t>/smoke.py` — self-locating, boots from its
  own directory, greedy keyword asserts, exit code is the verdict. The
  cases are target-authored: pick prompts whose greedy continuation is
  high-confidence for this model, and **verify every keyword on the real
  model before freezing it** — an unverified keyword bakes a false
  failure into the gate.
- Release: `uv run bench/accuracy.py --target targets/<t>` — one-sided gate,
  measured >= reference − tol; after a first pass on an external anchor,
  write the measured score back (see the references file header).

## 10. What varies per model — the re-derive list

The exemplar target shows the pattern; every model-specific value must be
re-derived from the new checkpoint's `config.json` and the catalog
contracts, never copied. The known variation axes:

**attention structure** — GQA/MHA vs **MLA**, the axis that reshapes the
most: MLA replaces per-head K/V with one compressed latent, needs two
attention calls per forward (context and generation are separate call
shapes, mixed batches rejected upstream), and brings its own load-time
obligations. It is not a variant of the geometry row below; see
`docs/references/mla-custom-op-decomposition.md` for the vocabulary and
`docs/models/latent-kv-cache.md` for what a target had to derive.
Then: geometry (layers, heads, kv-heads, head_dim, intermediate, vocab);
qk-norm presence and its eps; rope kind, theta, scaling, `is_neox`,
partial-rotary factor; `tie_word_embeddings` (changes the lm_head/embed
feeding and the shell's tying path); attention bias (needs cublas_mm's
fused-bias argument); sliding window; activation function; checkpoint
key names and fusion grouping; dtype and quantization scheme — including
*which modules it excludes*, since a checkpoint may quantize only its MLP
path and leave attention bf16; parallel topology (llm_args.yaml +
per-rank shapes + manifest src transforms + collective-communication
catalog entries — §12 has the launch substrate, the rest is the first
multi-rank target's to derive); KV-cache dtype (turns the kv-scale
constants into post-load tensors — certification extension first).

A construction-time assert exists for each axis the assembly depends on:
if the new config violates one, the right response is to re-derive that
part of the assembly, not to delete the assert.

### Read the axes off the engine's config object, not off AutoConfig

`model_config.pretrained_config` — the object the engine hands
`__init__` — is the **un-migrated** config: fields sit where the
checkpoint's own `config.json` put them. A standalone
`AutoConfig.from_pretrained(model_dir)` probe can return a *different*
shape of the same information, because transformers migrates fields
across versions.

Observed at transformers 5.5.4 on a checkpoint written by 4.51: through
AutoConfig, rope had been migrated into
`rope_parameters = {'rope_theta': ..., 'rope_type': 'default'}` with
`rope_scaling` an alias of that same dict, and `cfg.rope_theta` raising
`AttributeError`. Through the engine, `rope_parameters is None`,
`rope_scaling is None`, and `rope_theta` is a plain instance attribute.
Deriving the axis from the AutoConfig surface produced a target that
failed at construction with `TypeError: 'NoneType' object is not
subscriptable`.

So: an AutoConfig probe is **not** a valid oracle for what
`pretrained_config` will look like. Read the checkpoint's `config.json`
directly to learn what the model *is*, and use the flat
`cfg.<field>` idiom inside the target. If a probe is needed, instrument
the target's own `__init__`.

### A missing `dtype` is not inherited — declare it in the stub config

A checkpoint may declare **no** `dtype` and no `torch_dtype` at all
(observed on gpt-oss-120b, whose weights are all BF16 outside the
quantized expert blocks). The two config surfaces then disagree:
`model_config.pretrained_config.torch_dtype` is `None` while the engine's
own `ModelConfig.torch_dtype` resolves `torch.bfloat16`. The shell reads
the **pretrained** one, so `DecoderModelForCausalLM` materializes its
shell-owned parameters — `lm_head.weight` — as **fp32**, and a target with
a dtype-checking weight manifest fails there with a cause two layers
upstream of the symptom.

The fix belongs in `model_dir/config.json`, which is a stub the target
owns: declare `"dtype": "bfloat16"` alongside the `architectures` patch.
That makes the stub carry two deliberate divergences from the checkpoint's
config rather than one, which is correct — the field states what the
checkpoint *is*, and it is the field the engine sizes the KV pool from.
Assert both surfaces in `__init__` so a future drift is loud.

### `model_dir/` linking is not a `tokenizer*` glob

An accuracy protocol that applies a chat template needs the template
files, and a recent checkpoint may keep **no** `chat_template` inside
`tokenizer_config.json` — it lives in `chat_template.jinja` (plus
`chat_template.json`). Those names, and `special_tokens_map.json`, match
no `tokenizer*` pattern. Link by inspecting what the checkpoint actually
ships, not by a fixed glob, or the gate fails at template application
after a full engine build.

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

## 12. Multi-rank launch — what changes above world size 1

Everything here was measured at the pinned version on B200 hosts at world
size 4. Two layers, and they answer different questions: the **launch
substrate** (process model, registration, the failure modes that exist only
above world size 1) applies to every multi-rank segment, while **what a
rank does differently** and **what attention DP changes** are per-topology
and were each established by the first target on that topology. See "Still
not established here" at the end for what no target has reached yet.

### The process picture

`LLM(..., tensor_parallel_size=N)` with `N > 1` does not build the model
in the calling process. It spawns one MPI worker per rank
(`MpiPoolSession` → `mpi4py.futures.MPIPoolExecutor` → `MPI_Comm_spawn`)
and the engine is constructed there; the launcher becomes a proxy. The
single-process-worker flag is read only after that branch, so at world
size > 1 it neither helps nor hurts.

### Registration reaches a rank through exactly one hook

mpi4py re-imports the launcher's **main module** in every spawned worker,
under `__name__ == "__worker__"`. Measured consequences:

| | launcher | worker rank |
|---|---|---|
| module-scope code | runs | **runs** |
| `if __name__ == "__main__":` body | runs | does not run |
| `sys.argv` | full | **script path only** |
| `os.environ` | — | **inherited** |

**This whole problem is gone in-tree, and the table above is now only an
explanation of why the old code looked the way it did.** A target class is
an ordinary member of an installed package, so a spawned rank resolves it
through the normal registry — no module bound into `sys.modules` before
pickling, no `__worker__` re-import to time correctly, and no
`STAIRCASE_TARGET`, which existed solely because argv does not reach the
workers and `LlmArgs` does. `bench/register.py` and its module-scope
`from_env()` were deleted rather than migrated.

What the table still explains: any *tool* that has to influence worker
ranks must do it through `LlmArgs` or the environment, never argv.

### Two failure modes that exist only above world size 1

**The loaded module must be bound in `sys.modules`.** The engine resolves
`architectures[0]` to a class object in the launcher and ships it to the
ranks by pickle, which stores a class *by reference* — `__module__` plus
`__qualname__` — and both sides resolve those names through
`sys.modules`. Loading by path without binding the name fails in the
launcher, before any rank starts:

```
PicklingError: Can't pickle <class 'staircase_target_modeling.StaircaseForCausalLM'>:
               import of module 'staircase_target_modeling' failed
```

Binding it *twice* is equally fatal — the second load leaves a different
class object under the same name and pickle's identity check rejects it
(`it's not the same object as ...`) — so the load has to be idempotent.

**An environment variable set after MPI initializes never reaches a
rank.** Importing `tensorrt_llm` initializes MPI, and OpenMPI hands a
spawned process the environment as it stood at that moment. A tool that
exports into its own environment must do so *before* that import; one
that hands a freshly spawned server a prepared environment satisfies this
by construction.

### The parallel segment and its knobs

| segment | `llm_args.yaml` |
|---|---|
| `tp<N>` | `tensor_parallel_size: N` |
| `tep<N>` | the above plus `moe_expert_parallel_size: N` |
| `dep<N>` | the above plus `enable_attention_dp: true` |

`tep4` and `dep4` were both observed building a serving engine and
generating correct greedy text from the stock DeepSeek-V3-Lite NVFP4
checkpoint (254 s and 142 s to a ready engine). That is a statement about
the LLM API on this host — not about any staircase target.

### What the engine hands the model

`model_config.mapping` carries the rank's place in the topology. Fields
observed on `Mapping(world_size=4, tp_size=4, moe_ep_size=4, rank=1)`:

| field | value |
|---|---|
| `rank` / `world_size` | 1 / 4 |
| `tp_size` / `tp_rank` | 4 / 1 |
| `pp_size` | 1 |
| `moe_ep_size` / `moe_ep_rank` | 4 / 1 |
| `moe_tp_size` / `moe_tp_rank` | 1 / 0 |
| `enable_attention_dp` | False |
| `tp_group` | `[0, 1, 2, 3]` |

Read the topology off this object, never off the segment string: the
segment names the intent, `mapping` is what the engine actually built.

### What a rank actually has to do differently

Established by the first multi-rank target (`tep4`, world size 4). These
are the four things this section used to list as unobserved.

**`lm_head` is topology-aware, and that is a manifest obligation §4/§5 do
not mention.** At `tp_size = 4` `DecoderModelForCausalLM` builds an
`LMHead` of `[vocab/tp, hidden]` — vocab-parallel — so the manifest must
feed *this rank's contiguous block of vocabulary rows*, and the logits
gather stays upstream's. Consequence: `vocab % tp_size == 0` belongs in
the construction asserts, because nothing else checks it.

**Where the collectives belong: after every row/column-sharded producer,
and nowhere else.** For a TP attention + EP MoE layer that is two per
layer — after `o_proj`, and after `routed_window + shared_partial`
(summed locally first, so one collective serves the whole MLP). The
entry is `comm/allreduce`; the workspace-free certified path is
`strategy=0` (NCCL) or `8`, `workspace=None`. Keeping `op=0` (plain sum)
and leaving the existing fused-add-rmsnorm in place preserves the
single-rank residual structure, which is also the CUDA-graph-friendly
shape.

**Which transport, though, is worth about 2x at decode sizes, and the
workspace-free default is the slow one.** A decode message is three orders
of magnitude below the point where NCCL's ring algorithm starts to pay for
itself, and a blocking collective serializes far more than its own time.
`docs/references/collective-allreduce-transport.md` has the algorithms,
the measured crossover, and the two profiling traps that make this easy to
get wrong.

**The manifest's `src` transforms: a fourth column, applied before the
relayout.** `(ckpt_key | tuple, src, dst_index, transform)`, with `src`
selecting this rank's slice per key of a multi-key row. Order is
load-bearing: relayout transforms are functions of the *per-rank* row
count, so a transform that ran on the whole tensor and sharded afterwards
produces a different byte order. Column shards are strided views, so
densify before any transform reinterprets bytes.

**Per-rank parameter shapes** follow the topology mechanically — heads,
dense/shared intermediates and the expert-axis window all divide — with
one trap worth stating: re-derive every kernel alignment rule at the
*per-rank* width rather than inheriting the single-rank conclusion.

**Coverage asserts change shape.** `leftover == {}` no longer holds: under
EP each rank legitimately leaves the off-window experts' keys unconsumed.
Replace it with an explicitly predicted leftover set computed from the
rank's window, and add a parameter-side assert that the shell-registered
parameters are exactly `{lm_head.weight}`.

How MLA's latent cache behaves under attention TP is a mechanism fact
rather than a runtime one, and lives in `docs/models/latent-kv-cache.md`:
it is **replicated, not sharded** — attention TP buys zero KV memory on
MLA.

### What attention data parallelism changes on top of that

Established by the first `depN` target (`dep4`, world size 4, an MLA + EP
MoE checkpoint). **Three of the four TP bullets above come out
differently**, so read this section as replacing them rather than adding to
them whenever `enable_attention_dp` is on.

**The split is over requests, not over heads.** Attention is *replicated*:
each rank builds the full head count (32, not `32/tp`), holds the full
attention weights, and serves its own subset of the batch. Consequently
**the post-`o_proj` all-reduce disappears** — a rank's attention output is
already complete for its own tokens.

**`lm_head` is replicated, not vocab-parallel.** At `tp_size = 4` *without*
attention DP the shell builds an `LMHead` of `[vocab/tp, hidden]`; with it,
the shell builds the full `[vocab, hidden]` on every rank. The manifest
obligation reverses, and `vocab % tp_size == 0` stops being a construction
requirement.

**The manifest's `src` transform column is a TP artifact.** Nothing outside
the routed expert stacks is sharded, so a `depN` manifest is the tp1
three-column form with the expert loop windowed.

**The collectives move to the MoE and change identity.** Not two
all-reduces per layer, but one `comm/allgather` + one `comm/reducescatter`
per **MoE** layer (a dense layer has none). The gather **belongs before the
router GEMM, not merely before the expert call** — see
`docs/models/expert-weight-packing.md` for why the EP tiling invariant
depends on that placement and fails silently otherwise.

**How a rank learns the other ranks' token counts:**
`attn_metadata.all_rank_num_tokens`, a host int list, identical on every
rank. Every rank pads to `max(...)` so both collectives run in their
uniform (`sizes=None`) form — which is also the only form a CUDA graph can
replay, since `sizes` is a host argument frozen at capture.

**That padding creates an obligation.** Collectives pair **by position** on
the communicator, and at equal byte counts an ordering divergence is
**silent** — every rank wrong in 98-99% of elements, bitwise reproducibly,
no hang. Padding guarantees equal byte counts, so a `depN` forward must
issue an identical call sequence on every rank and must not wait for a hang
to detect that it did not. Certified in `catalog/comm/allgather.md` and
`catalog/comm/reducescatter.md`.

**The runtime is not a second party on that communicator.** Traced live:
one NCCL communicator per rank, 14,036 collectives over 242 forwards,
**zero** issued by the runtime. The engine's own attention-DP
synchronization is host-side MPI on a *different* communicator
(`MPIDist.tp_comm`, built by `MPI_Comm_create_group`). So a target's
collectives cannot be mispaired against the engine's.

**The CUDA-graph gate that looks like a landmine and is not.**
`cuda_graph_runner.py` replays a graph under `enable_attention_dp` only if
*all* ranks are generation-only **and** their batch sizes are exactly
equal; the padding that would force equality is off by default. Measured at
steady-state serving load, it **never binds**: `cudaGraphLaunch` = 1.00 per
rank per decode step, i.e. **100% replay coverage** without
`enable_padding`. It does bind on a draining workload — an accuracy-gate
run observed captures but no replays across 334 forwards, where the batch
shrinks monotonically. Both observations are correct about different loads.
Consequences measured on `dep4`: `cuda_graph_config.enable_padding: true`
bought nothing and cost **-2.55%** at concurrency 256 (it can only coarsen
the grid), and `attention_dp_config.enable_balance: true` cost **-13.10%**
at concurrency 128 with mean TTFT doubling, because its `batching_wait_iters`
hold costs more than the alignment buys once random arrival already
balances the ranks.

**KV cache.** No rank factor: see `docs/models/latent-kv-cache.md` — on
MLA the per-rank pool is identical across tp1/tepN/depN, and what `dep`
buys is `world_size x` *aggregate* capacity from holding disjoint requests,
not a narrower pool.

### Still not established here

Pipeline parallelism (`pp_size > 1`); multi-node, where the loopback
pinning `scripts/env.sh` applies for single-host spawn must be overridden;
and `moe_tp_size > 1`, i.e. splitting experts a second way along the
intermediate dimension.

## 13. One-engine speculative decoding — what changes when the target drafts

Everything above describes a target that emits **one** token per generation
sequence per step. A speculative target emits `1 + draft_len`, and the draft
tokens come from a second forward that the target itself owns. This section
is the axis that brings: what the engine looks for on the model object, what
the drafting loop calls, where the ownership line falls, and what a rank has
to do differently inside its forward.

Read like §12, in two layers. The **binding surface** applies to every
one-engine speculative mode. The **per-step consequences** below it were
established for **MTP-Eagle one-model** (`MTP_EAGLE_ONE_MODEL`) on an MLA +
EP-MoE checkpoint at `dep4`, and are marked where another mode differs.
`docs/models/multi-token-prediction.md` carries the other half — what an MTP
layer computes, which is a checkpoint fact rather than a runtime one.

### The checkpoint picks the mode; the target does not

`update_spec_config_from_model_config` runs **before the model is built**
and reads the MTP layer count out of the pretrained config
(`num_nextn_predict_layers`, or `mtp_num_hidden_layers` on Qwen3Next-style
configs; 1 if neither is present). `MTPDecodingConfig`'s defaults are
`use_mtp_vanilla=False` and `mtp_eagle_one_model=True`, so:

| checkpoint layer count | resulting mode |
|---|---|
| `n == 1` | **`MTP_EAGLE_ONE_MODEL`** — one layer of MTP weights, replayed |
| `n > 1` | `MTP` (vanilla) — one distinct layer per draft position |

**Under MTP-Eagle, `max_draft_len` is not bounded by the checkpoint's layer
count.** That bound belongs to vanilla MTP. The single MTP layer is replayed
autoregressively `max_draft_len` times, so "how many MTP layers do we turn
on" is the wrong question and "what is `max_draft_len`" is the right one.

**Spell `max_draft_len` out in every config variant.** Left unset on the
MTP-Eagle path it resolves to **1**, not to anything derived from the
workload. `max_total_draft_tokens` is then mirrored from it (linear tree),
and `tokens_per_gen_step = 1 + max_total_draft_tokens`.

### The engine finds the drafter through exactly one getattr

```python
# _torch/pyexecutor/model_engine.py
def _get_spec_worker(self):
    return getattr(self.model, 'spec_worker', None)
```

That is the whole registration. Everything else the runtime touches on the
model side, it reaches through the worker's own arguments:

| attribute / callable | type | what reads it |
|---|---|---|
| `model.spec_worker` | `SpecWorkerBase` | the engine's one getattr |
| `model.config` | pretrained config | `update_spec_config_from_loaded_model` (the base shell already provides it) |
| `model.draft_config` | — | read with `getattr(..., None)`; **absent is correct** for a single-checkpoint MTP target |
| `draft_model.mtp_layers` | `nn.ModuleList` | only `[0]` is ever indexed — MTP-Eagle replays one layer |
| `draft_model.embed_tokens` | module | passed to the layer as a kwarg |
| `draft_model.lm_head` | module | passed to `shared_head` |
| `draft_model.model.d2t` | — | read with nested `getattr(..., None)`; **absent is correct** (draft and target share a vocabulary) |
| `mtp_layers[0](...)` | callable | the draft loop, once per draft step |
| `mtp_layers[0].shared_head(h, lm_head, attn_metadata, True)` | method | returns draft logits |

`draft_model` is a container the target defines and hands to the worker; the
runtime never constructs it and never inspects it beyond the four names
above.

The layer is called by keyword, with `inputs` splatted in:

```python
hidden_states = draft_model.mtp_layers[0](
    embed_tokens=draft_model.embed_tokens,
    all_rank_num_tokens=<per-step, see below>,
    input_ids=..., position_ids=..., hidden_states=...,
    attn_metadata=..., spec_metadata=...,
)
```

It returns **one tensor**, `[rows_this_step, hidden]`, unpadded — the same
row count its `input_ids` carried. The loop slices it with its own
`gather_ids` afterwards. (Eagle3 one-model returns a second tensor here;
MTP-Eagle does not.)

### Where the ownership line falls

| responsibility | owner |
|---|---|
| accept/reject, rejection sampling, the golden token | runtime |
| KV rewind, `attn_metadata` rewrite between draft steps and its restore | runtime |
| the draft loop, `gather_ids`, position shifting, sampling draft tokens | runtime |
| `runtime_draft_len` scheduling and padding, `(bs, draft_len)` graph capture | runtime |
| the KV pool's extra layer and extra tokens | runtime |
| **the MTP layer's forward** | target |
| **`shared_head`** | target |
| **the `draft_model` container** | target |
| **the shell's speculative branch** | target |
| **loading the MTP layer's weights** | target |
| **attention-DP padding inside the MTP layer** | target |

### Inheriting the in-tree one-engine shell is not the shortcut it looks like

`SpecDecOneEngineForCausalLM.__init__` builds its drafter by calling
`get_draft_model(...)`, which is a **module-level function, not a method** —
a subclass has no override point. It dispatches on the config's
`model_type`, and a staircase stub config patches `architectures` only, so
`model_type` still names the upstream family and the call returns
**trtllm's own MTP layer**. Inheriting therefore hands the one computation
this project exists to write to the engine instead.

This is a statement about what gets constructed, not a rule against
inheriting: a shell already inherits `DecoderModelForCausalLM` from the same
package, and the isolation hook gates *reading* whole-model definitions, not
importing them. Writing the branch out (about 40 lines) keeps the forward
readable end to end, which is what the Vocabulary table and the
closed-vocabulary audit both rest on.

### The shell's shape, and the four things it has to get right

```python
class StaircaseForCausalLM(DecoderModelForCausalLM[StaircaseCore, ...]):
    def __init__(self, model_config):
        ...                                    # unchanged
        self.spec_config = getattr(model_config, "spec_config", None)
        self.draft_model = None
        self.spec_worker = None
        if self.spec_config is not None:
            assert self.spec_config.spec_dec_mode.is_mtp_eagle_one_model()
            self.draft_model = <the target's own container>(...)
            self.spec_worker = get_spec_worker(self.spec_config, model_config,
                                               model_config.mapping)

    def forward(self, attn_metadata, **kw):
        if self.spec_worker is None:
            assert kw.get("spec_metadata") is None
            return super().forward(attn_metadata, **kw)      # bit-identical
        spec_metadata = kw["spec_metadata"]
        hidden = self.model(attn_metadata=attn_metadata, **kw)
        logits = self.logits_processor.forward(
            hidden[spec_metadata.gather_ids], self.lm_head, attn_metadata, True)
        return self.spec_worker(
            input_ids=kw["input_ids"], position_ids=kw["position_ids"],
            hidden_states=hidden, logits=logits,
            attn_metadata=attn_metadata, spec_metadata=spec_metadata,
            draft_model=self.draft_model,
            resource_manager=kw.get("resource_manager"))
```

`get_spec_worker` is imported from `tensorrt_llm._torch.speculative` — the
**runtime**, the part of the stack this project reuses, not modeling.

Four things this shape is load-bearing about:

**The non-speculative path must be a delegation, not a reimplementation.**
A target's release criterion was measured on the inherited base forward; the
only way to keep it bit-identical is to call it. Declaring
`resource_manager` as a named parameter would silently drop it from `**kw`,
so leave it in the dict and pull it with `.get` in the speculative branch
only — then the base receives exactly what it receives today.

**The shell gathers the logits; the engine does not.** For every one-model
mode `without_logits` is True, so `_forward_step` returns the model's dict
verbatim and applies no second gather. `spec_metadata.gather_ids` holds one
row per context request (its last token) and `runtime_draft_len + 1` rows
per generation request — pass `hidden` **ungathered** to the worker and the
gathered logits alongside it.

**`position_ids` reaches the worker in the engine's `[1, T]` shape.** The
worker does `position_ids.squeeze(0)` itself. A shell that flattens before
handing it over produces a silently wrong draft position sequence.

**Nothing needs adding to `epilogue`, and nothing needs a `layer_idx`.**
`epilogue` is only consulted by `__pp_init__`'s `skip_forward`, so at
`pp_size == 1` there is nothing to register. And on this mode
`Eagle3OneModelSpecMetadata` sets `layers_to_capture = ()`, which makes
`is_layer_capture()` False at every layer and leaves `hidden_states`
unallocated — the trunk owes the runtime **no hidden-state capture hook**
(that is Eagle3's requirement, not MTP-Eagle's). Measured: nothing in
`_torch/pyexecutor/` or `_torch/speculative/` reads `model.layer_idx`.

### Draft length is per iteration, not per request

`_handle_dynamic_draft_len` runs **before** `prepare_resources`, so KV
allocation already knows the answer:

1. `draft_len_schedule` maps a batch-size threshold to a draft length.
2. The current `scheduled_batch.batch_size` selects `runtime_draft_len`.
3. Every generation request's `py_draft_tokens` is padded or truncated to
   **exactly** that length — the source comment names CUDA-graph replay and
   the attention kernel as the reasons.
4. It lands on `spec_metadata.runtime_draft_len`;
   `runtime_tokens_per_gen_step = 1 + runtime_draft_len`.
5. `runtime_draft_len == 0` takes `skip_drafting`, i.e. speculation is off
   for that iteration only.

**For modeling this means there is no ragged draft tree to handle.** The
draft length is a host int, constant across the batch, constant within a
capture. `cuda_graph_runner.get_graph_key` asserts it directly: *"All draft
lengths must be the same"*.

Without a schedule, `runtime_draft_len` is simply `max_draft_len` every
step.

**`draft_len_schedule` deadlocks under attention data parallelism, and
nothing rejects the combination.** Step 2 above reads
`scheduled_batch.batch_size` — **each rank's own local batch** — with no
cross-rank reduction, and attention DP does not equalize batch sizes:
`_pad_attention_dp_dummy_request` only tops a rank up from zero to one, and
`attention_dp_config.enable_balance` is off unless asked for. Two ranks
either side of a schedule threshold therefore run **different numbers of
draft replays**, hence different numbers of MoE collectives — and the
collectives pair by position, so the job hangs.

Measured at world size 4: boot, CUDA-graph capture and warmup all pass, and
the hang needs real traffic to make the rank batch sizes diverge. trtllm's
own `HangDetector` fired at 300 s and hard-killed all four ranks, whose
stacks sat at three different points of one forward. **On an attention-DP
target, skip this knob.** With `tp`/`ep` alone the ranks share one batch and
the mechanism is sound.

Setting it also **silently turns on `cuda_graph_config.enable_padding`** —
logged at INFO only — so it is never a single-variable experiment.

### What changes in the trunk's own forward

**One argument, and it is the whole of it on Blackwell.** A generation
request arrives with `runtime_draft_len + 1` query tokens instead of 1, and
that is expressed to the attention op through **`predicted_tokens_per_seq`**
alone — the value the total-token arithmetic uses
(`num_ctx_tokens + (num_seqs - num_contexts) * predicted_tokens_per_seq`).

**Those extra query tokens stay on the generation call.** A one-engine mode
returns False from the runtime's `extend_ctx` predicate — "1-model has
separate logic for handling draft tokens" — so a generation request carrying
drafts is *not* re-shaped into a chunked context request the way two-model
speculation does it. The batch keeps its `[context | generation]` split and
the generation call simply gets a taller query block. What that block is
allowed to attend to — each draft position seeing the cache plus the earlier
positions of its own block, and no later one — is a **kernel** fact, so the
catalog contract for the attention entry is its authority, not this file.
Do not assume it from `mask_type` alone.

**The spec-dec mask machinery is forced off at sm_100 and stays inert.**

```python
# _torch/attention/backends/trtllm.py  (was _torch/attention_backend/)
# Blackwell trtllm-gen spec-dec is enabled only for dynamic-tree masks.
self.is_spec_decoding_enabled = is_spec_decoding_enabled and (
    not self.is_sm_version_trtllm_gen_kernel(sm=get_sm_version())
    or is_spec_dec_dynamic_tree)
```

`is_sm_version_trtllm_gen_kernel(sm)` is `not (sm < 100 or sm in [120, 121])`,
so it is True on sm_100; a linear-tree MTP has `is_spec_dec_dynamic_tree`
False; the conjunction is **False**. `is_spec_decoding_enabled`,
`use_spec_decoding` and `is_spec_dec_tree` are all False and every
`spec_decoding_*` tensor is None — the same inert values the MLA columns of
`catalog/attention/thop_attention.md` already certify. **On a pre-Blackwell
arch this is not true** and the mask surface would need certifying first.

The context path is unchanged: a context request still contributes its
prompt, one row of logits, and one attention call of the same shape.

### The draft loop rewrites the batch between step 0 and step 1+

The single most important fact for writing the layer. After the first draft
step the worker mutates `attn_metadata` in place:

| | step 0 | step 1+ |
|---|---|---|
| `_seq_lens` / `_seq_lens_cuda` | real | **filled with 1** |
| `num_contexts` | real | **0** (when a KV cache manager is present) |
| `num_ctx_tokens` | real | **0** (recomputed) |
| `host_request_types` | mixed | context entries overwritten to generation |
| tokens per generation request | `runtime_draft_len + 1` | **1** |
| a context request | the whole prompt | **1 token** |
| `use_spec_decoding` | as the engine set it | False |
| `kv_lens_cuda` | as the engine set it | rewound, then `+1` per step |

**So the layer must read its phase from `attn_metadata` on every call.** It
is invoked N times inside one forward, and a value computed on the first
call is wrong on the rest. This is not the trunk's situation — the trunk
builds step-args once per forward because there is only one step in it.

The read is cheap and sync-free. `on_update()` recomputes `_num_ctx_tokens`,
`_num_generations` and `_num_tokens` from `_seq_lens`, which is a **pinned
host** tensor, and the loop calls it (and triggers it again through the
`num_contexts` setter) at exactly the step-0/step-1+ boundary. Therefore

```
tokens_per_gen_seq = (rows - md.num_ctx_tokens) // (md.num_seqs - md.num_contexts)
```

evaluates to `runtime_draft_len + 1` on step 0 and to exactly `1` on step
1+, with no step counter threaded through and no device read. That is the
`predicted_tokens_per_seq` the layer's own generation call needs. Guard the
all-context case (`num_seqs == num_contexts`) rather than dividing by zero.

**What a context request is fed on step 0** is `prompt[1:]` with the
request's first accepted (golden) token written at its last position —
`_prepare_context_input_ids`, shared by both MTP flavours. Its hidden states
are the trunk's, in full, at full prompt length. So step 0 runs a real
context-phase attention for those rows and the layer needs both phases,
exactly like the trunk.

### Attention DP: the padding basis changes source, and reading the wrong one is silent

Under `enable_attention_dp`, §12 established that every rank pads its token
block to `max(attn_metadata.all_rank_num_tokens)` so both collectives run in
their uniform form. **Inside the draft loop that list is the wrong one from
step 1 onwards.**

The worker passes the right basis in **as a keyword argument** and leaves
`attn_metadata.all_rank_num_tokens` holding the trunk's value for the whole
loop (it saves and restores it around the loop, but does not maintain it
during it):

| draft step | `all_rank_num_tokens` kwarg |
|---|---|
| 0 | `spec_metadata.all_rank_num_tokens` — the trunk's token counts |
| 1+ | `spec_metadata.subseq_all_rank_num_tokens` — the per-rank **sequence** counts |

`subseq_all_rank_num_tokens` is set to `all_rank_num_seqs` by the engine for
the one-model modes, which is semantically right: from step 1 every sequence
contributes exactly one token.

**Pad from `md.all_rank_num_tokens` inside the MTP layer and the collectives
mispair silently.** `catalog/comm/allgather.md` and
`catalog/comm/reducescatter.md` both certify that calls pair by *position*
on the communicator and that at equal byte counts a divergence produces no
hang — every rank wrong in 98–99% of elements, bitwise reproducibly. The
defence is structural: give the layer's padding helper a signature that
**only accepts a passed-in list**, so it has no way to reach the metadata.

(A `_dp_rows` that asserts `all_rank_num_tokens[rank] == rows` would in fact
fire here rather than corrupt, since the two counts differ from step 1. Do
not rely on it: the assert is a property of one target's helper, not of the
rule, and it is exactly the kind of guard a later edit removes.)

### KV cache: the engine adds the layer, and the target declares nothing

Under a one-model MTP mode `ModelConfig` raises the pool's layer count
itself:

```python
num_layers            += spec_config.num_nextn_predict_layers
num_attention_layers  += spec_config.num_nextn_predict_layers
```

so the pool is sized for `num_hidden_layers + n` layers and the MTP layer's
own attention addresses layer index `num_hidden_layers`. **Nothing in the
target's config stub or manifest declares any of this** — but the layer's
attention calls must pass the right `layer_idx`, and the pool-addressing
surface they use has to be certified at the new layer count.

**A `speculative_config` also raises `max_seq_len`, and not by the amount the
extra-KV-token helper suggests.** Three separate terms are added to the model
engine's `max_seq_len`, in `py_executor_creator`:

```python
if not disable_overlap_scheduler and spec_config is not None:
    max_seq_len += spec_config.tokens_per_gen_step - 1
if spec_config is not None:
    max_seq_len += get_num_extra_kv_tokens(spec_config)   # max_draft_len - 1
    max_seq_len += spec_config.tokens_per_gen_step - 1
```

With a linear tree (`tokens_per_gen_step = 1 + max_draft_len`) and the overlap
scheduler at its default (**on**), that is `3 * max_draft_len - 1` — 2, 5 and
8 at `max_draft_len` 1, 2 and 3. Measured: a `163840` model cap becomes
**`163848`** at `max_draft_len: 3`.

**So read `max_seq_len` off the metadata, never compute it.** A target that
derives anything from the config's own `max_position_embeddings` — a rope
table sized to it, a bound asserted against it — is off by that amount the
moment speculation is switched on, and by a different amount per
`max_draft_len`. The failure is a silent out-of-bounds read on a rope table,
or an assert that fires on a legal config.

### One more construction-time trap, from §4's weight lifecycle

**A reference to another module's parameter, captured in `__init__`, stays
bound to the meta-device shadow.** The engine materializes at t1 by
*replacing* tensor objects, not by filling them in place, so a drafter
container that does `self.embed_tokens = core.w["embed"]` at construction
holds a meta tensor forever and fails at the first draft step. §4 already says
an unregistered parameter "stays a dead shadow"; this is the adjacent case —
the parameter *is* registered, on the trunk, and the copy of the reference is
what goes stale. Resolve it lazily (a `property` that reads through to the
trunk on each access) rather than caching it.

### CUDA graphs: the whole draft loop is inside the capture

Capture wraps `_forward_step`, which calls the shell's forward, which calls
the worker — so every MTP-layer invocation is captured. The key is
`(batch_size, draft_len, is_first_draft, short_seq_len_mode,
is_all_greedy_sample)`, and the capture set becomes `(bs, draft_len)` pairs
rather than bare batch sizes; with a `draft_len_schedule` the runner also
captures one extra `(max_bs, original_max_draft_len)` graph, whose stated
purpose is to keep a later graph from resizing the shared attention
workspace and invalidating pointers baked into earlier ones.

Consequences for the layer, all of them §7's discipline applied one level
down:

* The step-0 / step-1+ divergence is **re-traced per draft step at capture
  time** and frozen at each step's position in the loop. Reading the phase
  from `attn_metadata` every call is what makes that correct — a cached
  first-call decision would be baked in at all N positions.
* Host ints read from `attn_metadata` (`num_contexts`, `num_ctx_tokens`) and
  the `all_rank_num_tokens` kwarg are per-capture constants, which is the
  host-derived class §7 permits.
* `attn_metadata.padded_num_tokens` is **`None`** unless torch-compile
  piecewise CUDA graphs are configured; without a `torch_compile_config` the
  padding path is not reachable and the base shell's slice never applies.
* The attention-DP replay gate of §12 is unchanged: all ranks
  generation-only and equal batch sizes, or the step runs eager — and eager
  is where the layer meets a real mixed batch.

### The acceptance rate is the only correctness signal

`stats.specdec_stats.acceptance_length` is computed per iteration over the
generation requests that carried draft tokens:

```
acceptance_length = (accepted_draft_tokens + requests_with_draft) / requests_with_draft
```

i.e. the mean number of tokens a drafting request produces per step, `1.0`
meaning every draft was rejected.

**This matters more than it looks.** Rejection sampling guarantees the
output distribution is unchanged, so a *miscomputed* MTP layer does not
produce wrong text — it produces correct text more slowly, with every draft
rejected. An accuracy gate cannot see it. A subtly wrong layer (a
transposed concatenation, a norm applied to the wrong operand) lands
somewhere above 1.0 and well below the reference, which no other measurement
distinguishes from "this model is just hard to draft for". The reference is
the same checkpoint under stock in-tree modeling at the same load and the
same `speculative_config`; acceptance is a property of the model, so a
reference forced onto a different `max_num_tokens` / `max_seq_len` to boot
is still a valid comparison.

### Still not established here

Vanilla MTP (`n > 1`, one layer per draft position) and its per-layer
sequential loop; Eagle3 in either form, including the hidden-state capture
hook and `apply_eagle3_fc`; tree drafting of any kind (static or dynamic
`eagle_choices`), which is also the only way the Blackwell spec-dec mask
surface becomes reachable; two-model speculation; and speculative decoding
combined with pipeline parallelism, where `skip_forward` and the `epilogue`
path start to matter.
