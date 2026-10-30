# Target integration with the TensorRT-LLM runtime

How a target plugs into the TensorRT-LLM engine: resolution,
construction, weight loading, the per-step forward duties, KV-cache
boundaries, CUDA-graph discipline, and the fail-fast ladder. Together
with the entries' executable contracts and `step-args.md`, this is the
complete integration knowledge for writing a target — no TensorRT-LLM
source reading is needed.

Everything below was established by probing and verified by gates. The
readable exemplars in tree are
`tensorrt_llm/_torch/_experimental/modeling_v2/models/gpt_oss/gpt_oss_120b__sm_103__tp1/`
(single rank, no phase branch) and
`tensorrt_llm/_torch/_experimental/modeling_v2/models/deepseek_v3/r1_0528_nvfp4__sm_103__dep4/`
(MLA, attention DP, MTP): read them for patterns, never for values.

> **This file is one part of a split.** It was `docs/references/trtllm-runtime-integration.md`;
> its section numbers are kept unchanged so that existing citations — in
> target comments, in the onboarding records — still resolve. The four
> sections that moved:
>
> | former section | now at |
> |---|---|
> | §7 CUDA-graph discipline | `trtllm-modeling-v2-core/references/graph-and-sync.md` |
> | §10 What varies per model | `trtllm-modeling-v2-kernel-feasibility/references/derivation-checklist.md` |
> | §12 Multi-rank launch | `trtllm-modeling-v2-parallel/references/multi-rank.md` |
> | §13 One-engine speculative decoding | `trtllm-modeling-v2-speculator/references/runtime-boundary.md` |

## 1. Directory shape

```
tensorrt_llm/_torch/_experimental/modeling_v2/
├── _router_index.py                     # architectures[0] -> routing module, NoTarget
├── catalog/
│   ├── _op.py                           # OpWrapper, Cell, Arch: what an entry is
│   └── <category>/<name>.py             # shared entries, one OpWrapper subclass each
└── models/<family>/
    ├── routing.py                       # route(): which config lands on which target
    └── <ckpt>__<arch>__<parallel>/       # identity = the directory's name
        ├── modeling.py   # core + PrefillTarget + DecodeTarget + causal-LM shell
        ├── weights.py    # the weight table: W, the table class, MODEL_WEIGHTS
        └── ops/<category>/<name>.py      # optional: ops only this target calls
                                          # (and, until promoted, local variants of
                                          # shared entries, <name>__<site>.py)
```

That is the whole target. There is no stub model directory, no
`llm_args.yaml`, no `smoke.py` and no `TARGET.md`: **the checkpoint is read
exactly as published**, the topology comes from the routing decision and the
boot gate's CLI flags, and gate records go in the run journal. `LLM(model=...)`
is pointed at the real checkpoint, and everything the runtime learns about the
model it learns from that checkpoint's own `config.json`.

A target-owned op is an `OpWrapper` subclass like any catalog entry, imported
by the target as its module-level instance,
`from .ops.<category>.<name> import <name>`; its GPU test
mirrors the target's path,
`tests/unittest/_torch/modeling_v2/models/<family>/<ckpt>__<arch>__<parallel>/<category>/test_modeling_v2_<name>.py`,
so CI collects it by directory. An op a second target needs moves into
`catalog/`, file and test and nothing else: the catalog is its entry files,
with no index beside them.

While the run develops a shared entry, the target calls a **local variant**
of it: `ops/<category>/<name>__<site>.py`, which a kernel task wrote with the
cells this target needs (its receipt names it), imported like any own op,
`from .ops.<category>.<name>__<site> import <name>`. Two variants of one entry
imported in one file are aliased. Before delivery a `promote` task merges the
variants into `catalog/<category>/<name>.py` and points these imports at the
catalog; delivery refuses a target that still imports a variant, unless
`task.yaml` keeps that entry local (`target.keep_local`).

## 2. Resolution

- **Nothing registers.** `route(pretrained_config, mapping,
  compute_capability, spec_config)` in `models/<family>/routing.py` returns
  the target's causal-LM class itself, importing it on the branch that
  returns it (`from .<target>.modeling import ModelingV2<...>`), or raises
  `NoTarget` stating the facts. There is no `register_auto_model` and no
  registry key: the class is the answer. See `routing-identity.md`.
- The model loader calls `modeling_v2_resolve(config, llm_args)` after the
  built-in model's defaults have been applied, and carries the answer on
  `ModelConfig.modeling_v2_target`. Above world size 1 each rank's loader
  does this itself, so an ordinary dotted module is all a rank needs. §12 has
  the rest of what changes there.
- `LLM(model=<checkpoint>, modeling_v2=...)` then builds through
  `AutoModelForCausalLM._resolve_class`, which returns
  `modeling_v2_target` when it is set without consulting the registry, and
  construction begins.

### The runtime still keys behaviour on the architecture string — and it reads the checkpoint's

Parts of the runtime branch on `architectures[0]` (and on other published
config fields) to decide **what to allocate**, not merely which class to
build. The known case is hybrid linear attention: a recurrent-state pool is
allocated only when the predicate matches, and otherwise the model builds, the
boot gate passes, and the linear layers have no state.

In tree that predicate reads the **checkpoint's own** config. Nothing
rewrites the architecture string: the target class rides on
`ModelConfig.modeling_v2_target`, and `pretrained_config.architectures` is
not mutated. Two consequences:

- a target **cannot** change what the runtime allocates by any name of its
  own, and must not try. If the runtime's predicate does not match
  the published config, that is a fact about the checkpoint and the runtime —
  report it as a symptom rather than routing around it;
- conversely the published name is always the honest one, so the failure mode
  the standalone project had to defend against (a stub `config.json` declaring
  a name the runtime did not recognize) cannot occur.

The registry is not on this path at all: `_resolve_class` returns
`modeling_v2_target` before `get_registered_model_class` is reached, so a
target never competes with a built-in class for a name, and
`register_auto_model`'s silent last-wins overwrite cannot reach it.

## 3. The shell, the core and the two targets

```python
class <Family><Par>ModelingV2(DecoderModel):        # the core: what the model statically is
    def __init__(self, model_config):
        super().__init__(model_config)
        d = _weights.MODEL_WEIGHTS.dims(self)
        self.w = _weights.MODEL_WEIGHTS.declare(d)   # nn.ParameterDict, meta here
        self._prefill = self._decode = None
    def post_load_weights(self):                    # weights are real now
        self._prefill = PrefillTarget(self)
        self._decode = DecodeTarget(self)
    def _select_target(self, attn_metadata):
        return self._decode if attn_metadata.num_contexts == 0 else self._prefill
    def forward(self, attn_metadata, *args, **kwargs):
        return self._select_target(attn_metadata).forward(attn_metadata, *args, **kwargs)

class PrefillTarget:                                # context rows, maybe generation rows too
    def __init__(self, core): ...                   # take the weight views + constants it reads
    def _check_step_contract(self, md): ...         # calls step_args once, if enabled
    def step_args(self, md): ...                    # reads num_contexts / num_ctx_tokens
    def forward(self, attn_metadata, input_ids=None, position_ids=None,
                inputs_embeds=None, lora_params=None, **kwargs): ...

class DecodeTarget:                                 # routed to only when num_contexts == 0
    def __init__(self, core): ...
    def _check_step_contract(self, md): ...
    def step_args(self, md): ...                    # states num_contexts=0, num_ctx_tokens=0
    def forward(self, attn_metadata, ...): ...

class ModelingV2<Ckpt><Sm><Par>(DecoderModelForCausalLM[<Family><Par>ModelingV2, PretrainedConfig]):
    def __init__(self, model_config):
        super().__init__(<Family><Par>ModelingV2(model_config), config=model_config,
                         hidden_size=..., vocab_size=...)
    def load_weights(self, weights, *args, **kwargs):
        _weights.MODEL_WEIGHTS.load(self, weights)
    def post_load_weights(self):
        super().post_load_weights()
        self.model.post_load_weights()
```

- **Names.** The core class carries the family and the parallelism strategy
  (`GptOssTp1ModelingV2`, `DeepseekV3Dep4ModelingV2`): a second strategy over
  the same checkpoint is a different core, not a flag on this one. The shell
  carries the routing triple (`ModelingV2GptOss120bSm103Tp1`) and is the class
  `route` returns; `test_modeling_v2_claims.py` asserts the name agrees with
  the directory. The two target classes are named exactly `PrefillTarget` and
  `DecodeTarget` — the claims suite finds them by name.
- The shell (`DecoderModelForCausalLM`) is composition: it stores the
  core as `self.model`, creates `self.lm_head`, and owns the logits path
  — its forward calls the core, then gathers exactly the rows that need
  logits (last token per context sequence + every generation token) and
  applies lm_head. **The core returns final-normed hidden states
  `[num_tokens, hidden]` and never touches logits.**
- **The core dispatches and does nothing else per step.** It runs any
  unconditional first-step derivation (deepseek grows its rope table to the
  engine's resolved `max_seq_len` here — not behind the opt-in contract
  check, which a served engine never runs), then hands the step to one
  target. `_select_target` reads `num_contexts` and nothing else; a mixed
  batch goes to `PrefillTarget`.
- **Targets are built after the weights are real**, in the core's
  `post_load_weights`, never in `__init__` (parameters are meta tensors
  there). A target holds the core by reference (`self.core`) and takes its
  weight views through it in `__init__`; it never snapshots a core attribute
  that the core may later replace.
- **The two targets share no body.** Each is a plain class with its own
  `__init__`, `step_args`, `_check_step_contract` and `forward` — no base
  class, no mixin. When the phases happen to compute the same thing (gpt_oss
  is not MLA), the bodies are duplicated: each must stay readable and
  changeable without reference to the other.
- Inputs are **packed**: first dimension = total tokens of the batch,
  context sequences first. `position_ids` arrives as an int32 `[1, T]`
  view of an engine buffer — flatten with `torch.reshape(..., [-1])`.
- **No configuration asserts.** A target is not a second line of defence
  against routing: the routing tree picks the target from the architecture,
  the config-shape fingerprint, SM and parallel topology, and `require` mode
  raises naming the criterion that missed. Every config property the assembly
  depends on is either one of those criteria or read from `pretrained_config`
  where it is bound. If a property would change the structure of the forward
  and the fingerprint does not pin it, the fix is a routing criterion
  (`routing-identity.md`), not an assert inside the target.
- A target that implements speculative decoding grows a draft step routed
  through the same two targets (`mtp_forward` on each); §13 is that axis.

## 3a. Calling the entries

Every entry — catalog or target-owned — is a stateless `OpWrapper` with one
module-level instance, which the target imports. The target takes what does
not change per call in its `__init__`, and every call site passes everything
the kernel receives:

```python
from tensorrt_llm._torch._experimental.modeling_v2.catalog.gemm.cublas_mm import cublas_mm

# PrefillTarget.__init__ / DecodeTarget.__init__ -- once, against real weights
self._layers = _layer_views(core.w, cfg.num_hidden_layers)  # per-layer views, .t() taken here

def forward(self, attn_metadata, ...):
    self._check_step_contract(attn_metadata)
    step = self.step_args(attn_metadata)                     # once per step
    attn_out = torch.empty(...)                              # scratch: once per step
    for i, L in enumerate(self._layers):
        qkv = cublas_mm(x, L.qkv, L.qkv_bias)
        attn_out = thop_attention(q=qkv, output=attn_out, local_layer_idx=i,
                                  num_heads=heads_q, ..., **step, **_CALL_CONSTANTS)
```

- `__call__` takes every argument explicitly and reads nothing off `self`
  (`test_modeling_v2_claims.py` enforces it), so what a cell drives is the
  whole input and the call site is the whole story. There is no instance
  state to set up and none to subclass around.
- **Take in the target's `__init__`** whatever is fixed once weights exist:
  per-layer weight views and their `.t()`, derived scalars, call constants (a
  module-level dict of the op's inert arguments). **Compute at the top of
  `forward`** what is known only per step — the `step_args` projection, a
  window derived from `md.max_seq_len` — and **never inside the layer loop**.
  Per-step scratch buffers are allocated once per forward, outside the loop,
  and reused by every layer.
- **Every call states its outputs**: `attn_out = thop_attention(...,
  output=attn_out, ...)`, even for an op that writes in place, so a reader
  can see what each statement overwrites.
- Instances carry no state, so both targets — or two models — calling the
  same module-level instance is safe; a target may also construct its own.

## 4. The weight lifecycle

| stage | who | what happens |
|---|---|---|
| t0 construct | engine (`MetaInitMode`) + your `__init__` | `torch.empty` is intercepted to the meta device: parameters are shape/dtype shadows, zero memory. Declare with `torch.empty` only (`zeros`/`full` are NOT intercepted) and do no tensor math in `__init__` (meta tensors raise on compute). |
| t1 materialize | engine | walks **registered** parameters (`nn.Parameter` / `ParameterDict` / module tree — what `named_parameters()` can see) and reallocates each on CUDA, contents garbage. Registration is what requests the allocation; anything held in a plain dict/list stays a dead shadow. |
| t2 read | engine | reads every safetensors in the checkpoint directory into a dict `{ckpt_key: tensor}`. The values arrive **already materialized** as `torch.Tensor`, not as lazy slices — measured at world size 4, where every rank is handed the whole dict (25,687 keys on all four). So the engine pre-shards nothing: a multi-rank split is entirely the manifest's, and a `src` transform's benefit is that only this rank's bytes cross to the device, not that fewer bytes are read off disk. |
| t3 load | `MODEL_WEIGHTS.load(model, weights)` from the shell's `load_weights` | full delegation, never validated by the engine — the table's `load` unrolls its own `manifest()` and copies bytes into the materialized storage (`param.data[index].copy_(transform(src))`). |
| t4 derive | the core's `post_load_weights()` (driven by the shell's) | the designated home for derived state: checkpoint-calibrated call tensors such as fp8 KV scales, rope tables, and **constructing the two targets**, which take `.t()` GEMM views and per-layer operands as their own. Real tensors may be created here — meta is over. |
| t5 serve | the targets' `forward` | passes the views it took to each entry call; zero copies or transforms on the hot path. |

Storage layout: declare parameters in **HF `[out, in]` row-major** so t3
copies are layout-preserving, and consume GEMMs through `.t()` views
(row-major `[N, K]` transposed is exactly the dense column-major `[K, N]`
that `cublas_mm`'s contract requires; the view is zero-copy and shares
storage, so reloading weights in place keeps views valid).

## 5. The weight table

`weights.py` states every weight **once**, as a `W` row of the target's own
table class. `W`, `ALL_LAYERS` and the table are all defined in that file —
there is no shared base:

```python
@dataclass(frozen=True)
class W:                                # one weight role
    name: str
    shape: Callable[[Any], tuple[int, ...]]
    src: str | tuple[tuple[str, Callable], ...] | Callable
    dtype: torch.dtype | None = None
    transform: Callable | None = None
    layers: object | None = ALL_LAYERS

class <Family><Par>Weights:
    RELEASE_AFTER = "_fc2_s"            # optional: free transform scratch per layer
    WEIGHTS: tuple[W, ...] = (
        W("norm1", shape=lambda d: (d.hidden,), src="{p}.input_layernorm.weight"),
        W("qkv", shape=lambda d: (d.q_width + 2 * d.kv_width, d.hidden),
          src=(("{p}.self_attn.q_proj.weight", lambda d: slice(0, d.q_width)), ...)),
        W("fc1_w", shape=..., src=..., dtype=torch.uint8, transform=_prep_fc1_weight),
        W("embed", shape=..., src="model.embed_tokens.weight", layers=None),
    )
    def dims(self, core) -> SimpleNamespace: ...     # the only place a width is derived
    def declare(self, d) -> nn.ParameterDict: ...    # torch.empty per parameter
    def manifest(self, core): ...                    # param key -> [(ckpt key, index, transform)]
    def load(self, model, weights): ...              # fill, then assert coverage
    def expected_unconsumed(self, core, weights): ...  # when needed: the known-not-loaded key set

MODEL_WEIGHTS = <Family><Par>Weights()
```

- `src` is a key template (`{p}` = the layer prefix), a tuple of
  `(template, slice_fn)` when one parameter is assembled from several
  checkpoint tensors (fused qkv, gate_up — no intermediate concat buffer), or
  a callable `(d, i) -> [(key, index), ...]` for sources that are not a fixed
  template (this rank's expert window, a layer-family prefix, several keys
  fed to one transform together).
- `layers` is `ALL_LAYERS` (default), `None` for a non-per-layer weight, or a
  callable `d -> layer indices` — dense and MoE layers carry different sets,
  and two rows may share a name over disjoint layer ranges.
- `transform(src, core)` runs on the destination device before the copy:
  the place for a load-time layout change (padding, interleave, swizzle).
- Parameter keys are `l{i}_<name>` per layer and `<name>` otherwise.
  `declare` and `manifest` share one enumeration, so they cannot disagree
  about whether a weight exists; **never write a manifest by hand** beside
  the table.
- `load` asserts each key present, shape and dtype per copy, and
  **bidirectional coverage**: every checkpoint key consumed, minus
  `expected_unconsumed(core, weights)` — the explicitly predicted non-load
  set, each member named with its reason. `lm_head.weight` is the shell's
  parameter, and `load` fills it explicitly. Each table writes its own
  `load`; a model whose load differs structurally writes a different one.

## 6. Forward duties, KV cache, and feature flags

The engine drives everything around the forward: scheduling, KV block
bookkeeping (allocation at admission, per-token growth, prefix reuse),
metadata filling and `prepare()` — all before your forward runs. Your
duties per step:

1. Project the prepared metadata through the target's `step_args(md)` once
   at the top of `forward` and pass the projection to every attention call,
   shared across layers (`step-args.md` is the normative mapping).
2. Keep the loop flat: entry calls, native torch calls, tensor-metadata
   reads, Python control flow — nothing else (the closed-vocabulary rule;
   audit = collect the calls in each target's `forward` **and the private
   methods it reaches**, and map every trtllm op to its entry —
   `tensorrt_llm/_torch/_experimental/modeling_v2/catalog/<category>/<name>.py`
   or the target's own `ops/`. Scope to that closure: a whole-file scan flags the
   rope table's `arange`/`cos`/`sin` and load-time `.t()`/`.to()`, which run
   at init and are outside the rule).
3. Serve every batch composition the phase admits. `PrefillTarget` takes
   mixed batches (context rows first, then generation rows); `DecodeTarget`
   has no context rows by routing. MLA, whose attention refuses a mixed
   batch, splits the rows inside `PrefillTarget` only.

The KV-ownership red line does not move: the pool is sized by the engine from
the **checkpoint's own** `config.json` declarations (`num_hidden_layers`,
`num_key_value_heads`, `head_dim`, `torch_dtype`, `vocab_size`,
`max_position_embeddings`), pages are assigned by the C++ manager, and the
new tokens' K/V are written **inside** the attention op. The target only
forwards the address book — it restates none of those fields and must not
contradict them; the values it passes are read from the same config the
engine sized the pool from.

Runtime-owned feature flags (block reuse, CUDA graphs, beam, spec-dec
plumbing) **pass through** from metadata exactly as prepared: the target
runs trtllm defaults, feature behavior is upstream's responsibility, and
the gates validate the result. Kernel-surface certification accounting
lives in each entry's `CELLS` and `ARCHS`, not in target config.

## 8. The fail-fast ladder

| fuse | when | checks | catches |
|---|---|---|---|
| routing | `_resolve_class`, every rank | the family's decision tree: SM, config-shape fingerprint, parallel topology. `require` raises naming the criterion that missed | a configuration no target implements. The target itself carries no configuration asserts |
| pybind surface | CI, `tests/unittest/_torch/modeling_v2/test_modeling_v2_target_contract.py` | the thop attention binding exists in this build | a build that dropped the binding. Custom ops are deliberately not listed: a renamed op fails at its call, which says where |
| step contract | first forward of each target, only under `TRTLLM_MODELING_V2_VALIDATE=1` (`_router_index.step_contract_enabled()`) | the target's `_check_step_contract` calls its own `step_args(md)`, which reads every metadata field the target consumes | private-surface layout drift. Both targets run it: their projections differ, so probing one proves nothing about the other |
| gates | explicit runs | boot-gate keywords; accuracy score vs `tests/integration/defs/accuracy/references/<benchmark>.yaml` | the computation itself |

There is no version pin to assert against — a target moves with the trunk —
and no hand-kept list of the surface either. `REQUIRED_TRTLLM_OPS` and the
`_STEP_FIELDS` tuple are gone: the projection *is* the declaration, so a
field added to `step_args` is checked the moment it is read, and there is no
second list to forget. Do not reintroduce one.

The step-contract check is opt-in and off in a served engine. Anything that
decides **what the model computes** — growing the rope table to the resolved
`max_seq_len`, picking the cached-context flavor — must run unconditionally
on the first step (on the core), never behind that flag.

**One metadata surface is conditional, and a flat existence check over it
turns a legal config into a hard failure.** MLA's cached-KV fields —
`enable_context_mla_with_cached_kv`, `ctx_cached_token_indptr`,
`ctx_kv_indptr`, `ctx_uncached_token_indptr`, `max_ctx_seq_len`,
`max_ctx_kv_len`, `num_ctx_cached_tokens` — exist only when block reuse
is on. With `enable_block_reuse: false` they are **absent, not False**, so
a `step_args` that reads them unconditionally fails the first forward on a
configuration the target is supposed to support. Keep them out of
`step_args`: select the context flavor once, unconditionally, by their
*presence* (deepseek's `_CACHED_CTX_FIELDS`, checked once on the core's first
forward), and pass them only on the flavor that has them.
See `.claude/skills/trtllm-modeling-v2-core/references/latent-kv-cache.md`.

## 9. Gates and launch

- **Every GPU command in a run is launched one way: the run's launch prefix.**
  It carries the environment the ranks need and the declared interpreter.
  Target selection is the LLM-API argument `modeling_v2`, which reaches every
  rank the way the rest of the arguments do; the gates pass it, and it is not
  yours to set. Do not construct a launcher, and do not probe for
  one. The devices claimed for the run are given to you: one for a `tp1`
  target, `N` for a `tepN`/`depN` one.
- Boot gate: `examples/llm-api/quickstart_advanced.py` with the target's
  topology flags and `modeling_v2=require`, run by the orchestrator.
  Engine cold start, weight-manifest coverage, greedy keyword asserts; the
  exit verdict is binary. The cases are target-authored: pick prompts whose
  greedy continuation is high-confidence for this model, and **verify every
  keyword on the real model before freezing it** — an unverified keyword bakes
  a false failure into the gate, which is why the keyword list starts empty and
  the first engine run doubles as its verification.
- Release: `trtllm-eval` under the run's launch prefix, with the protocol from
  `tests/integration/defs/accuracy/references/<benchmark>.yaml` and
  `modeling_v2=require` — one-sided gate,
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
