# Multi-rank launch — what changes above world size 1

> Extracted from the former `trtllm-runtime-integration.md` §12; the section
> number is kept so existing citations still resolve.

## 12. Multi-rank launch — what changes above world size 1

Everything here was measured on B200 hosts at world size 4; a record is keyed
by GPU architecture and nothing else. Two layers, and they answer different
questions: the **launch substrate** (process model, how a rank reaches the
target, the failure modes that exist only above world size 1) applies to every
multi-rank segment, while **what a rank does differently** and **what attention
DP changes** are per-topology and were each established by the first target on
that topology. See "Still not established here" at the end for what no target
has reached yet.

### The process picture

`LLM(..., tensor_parallel_size=N)` with `N > 1` does not build the model
in the calling process. It spawns one MPI worker per rank
(`MpiPoolSession` → `mpi4py.futures.MPIPoolExecutor` → `MPI_Comm_spawn`)
and the engine is constructed there; the launcher becomes a proxy. The
single-process-worker flag is read only after that branch, so at world
size > 1 it neither helps nor hurts.

### What reaches a rank, and how

mpi4py re-imports the launcher's **main module** in every spawned worker,
under `__name__ == "__worker__"`. Measured consequences:

| | launcher | worker rank |
|---|---|---|
| module-scope code | runs | **runs** |
| `if __name__ == "__main__":` body | runs | does not run |
| `sys.argv` | full | **script path only** |
| `os.environ` | — | **inherited** |

A target is **not** registered by a script the ranks re-import. `_router_index.py`
maps `architectures[0]` to `models/<family>/routing.py`, whose `route` returns
the target class, and the model loader carries it on
`ModelConfig.modeling_v2_target` — in whichever process builds the model.
There is nothing for a launcher to register and nothing for a worker to
re-import, so nothing above depends on where a target's code sits in a file.

What does have to reach a rank is the **mode**. `modeling_v2` is an LLM-API
argument, `off` / `auto` / `require`; anything attributed to modeling_v2 runs
under `require`. Being an argument, it reaches every rank the way the rest of
the arguments do — the second failure mode below is why a variable would not.
A driver resolving a target while its workers resolve the built-in is the
silent split `require` exists to catch.

### Two failure modes that exist only above world size 1

**The loaded module must be bound in `sys.modules`.** The engine resolves
`architectures[0]` to a class object in the launcher and ships it to the
ranks by pickle, which stores a class *by reference* — `__module__` plus
`__qualname__` — and both sides resolve those names through
`sys.modules`. Loading by path without binding the name fails in the
launcher, before any rank starts:

```
PicklingError: Can't pickle <class 'newmodel_target_modeling.NewModelForCausalLM'>:
               import of module 'newmodel_target_modeling' failed
```

Binding it *twice* is equally fatal — the second load leaves a different
class object under the same name and pickle's identity comparison rejects it
(`it's not the same object as ...`) — so the load has to be idempotent.

In tree both halves are satisfied by construction: a target is an ordinary
module of the modeling_v2 package, imported by name by its routing module, so
the name resolves in every process and the import is idempotent. That is what
an ordinary package module buys, and it is why the rule is stated rather than
dropped — any mechanism that reaches a model class by file path re-opens both
failures.

**An environment variable set after MPI initializes never reaches a
rank.** Importing `tensorrt_llm` initializes MPI, and OpenMPI hands a
spawned process the environment as it stood at that moment. A tool that
exports into its own environment must do so *before* that import; one
that hands a freshly spawned server a prepared environment satisfies this
by construction. Long-lived ranks under `trtllm-llmapi-launch` receive the
environment once, at launch, so a value set later reaches the driver and not
them. It is why target selection is the LLM-API argument `modeling_v2`
rather than a variable; `TRTLLM_MODELING_V2_VALIDATE` is still a variable and
is still subject to it.

### The parallel segment and its knobs

| segment | topology arguments (LLM API, and the boot gate's flag) |
|---|---|
| `tp<N>` | `tensor_parallel_size: N` (`--tp_size N`) |
| `tep<N>` | the above plus `moe_expert_parallel_size: N` (`--moe_ep_size N`) |
| `dep<N>` | the above plus `enable_attention_dp: true` (`--enable_attention_dp`) |

Routing derives the target *from* these: the family's `routing.py` matches the
segment against the mapping the engine actually built (`world_size`,
`moe_ep_size`, `moe_tp_size`, `enable_attention_dp`) and raises `NoTarget`
naming them when it does not fit. The target itself carries no configuration
asserts.

`tep4` and `dep4` were both observed building a serving engine and
generating correct greedy text from the stock DeepSeek-V3-Lite NVFP4
checkpoint (254 s and 142 s to a ready engine). That is a statement about
the LLM API on this host — not about any modeling_v2 target.

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

Four obligations a multi-rank target (for example `tep4`, world size 4) has
that a single-rank one does not.

**`lm_head` is topology-aware, and that is a load obligation §4/§5 do
not mention.** At `tp_size = 4` `DecoderModelForCausalLM` builds an
`LMHead` of `[vocab/tp, hidden]` — vocab-parallel — so the load must
feed *this rank's contiguous block of vocabulary rows* (the table's own
`load` fills `lm_head.weight`, and the exemplars feed it whole), and the
logits gather stays upstream's. Consequence: `vocab % tp_size == 0` is an
identity fact for routing, which carries no such criterion today — put it
in the family's `routing.py`, because the target carries no configuration
asserts and nothing else checks it.

**Where the collectives belong: after every row/column-sharded producer,
and nowhere else.** For a TP attention + EP MoE layer that is two per
layer — after `o_proj`, and after `routed_window + shared_partial`
(summed locally first, so one collective serves the whole MLP). The catalog
has no all-reduce entry (`catalog/comm/` holds `allgather` and
`reducescatter`), so a target that needs one is a vocabulary gap for smith:
a new `OpWrapper` entry with cells driven on this arch. What an earlier
certification established, and what the new entry's cells should start from:
the workspace-free path is `strategy=0` (NCCL) or `8`, `workspace=None`.
Keeping `op=0` (plain sum) and leaving the existing
fused-add-rmsnorm in place preserves the single-rank residual structure, which
is also the CUDA-graph-friendly shape.

**Which transport, though, is worth about 2x at decode sizes, and the
workspace-free default is the slow one.** A decode message is three orders
of magnitude below the point where NCCL's ring algorithm starts to pay for
itself, and a blocking collective serializes far more than its own time. So a
workspace-free all-reduce is the right thing to *gate* on — it is the
certified path, and it needs no allocation — and the wrong thing to quote a
decode latency from. Choosing a transport, the measured crossover, and the two
profiling traps around it are tuning material, and they live with the tuning
workflow, which is separate and later work; onboarding neither measures nor
gates them.

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
rather than a runtime one, and lives in
`.claude/skills/trtllm-modeling-v2-core/references/latent-kv-cache.md`:
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
all-reduces per layer, but one
`tensorrt_llm/_torch/_experimental/modeling_v2/catalog/comm/allgather` + one
`.../catalog/comm/reducescatter` per **MoE** layer (a dense layer has none).
The gather **belongs before the router GEMM, not merely before the expert
call** — see
`.claude/skills/trtllm-modeling-v2-core/references/expert-weight-packing.md`
for why the EP tiling invariant depends on that placement and fails silently
otherwise.

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
to detect that it did not. Certified by the entries
`tensorrt_llm/_torch/_experimental/modeling_v2/catalog/comm/allgather.py` and
`.../catalog/comm/reducescatter.py`, whose 4-rank rank bodies and launcher
live in `tests/unittest/_torch/modeling_v2/comm/`.

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

**KV cache.** No rank factor: see
`.claude/skills/trtllm-modeling-v2-core/references/latent-kv-cache.md` — on
MLA the per-rank pool is identical across tp1/tepN/depN, and what `dep`
buys is `world_size x` *aggregate* capacity from holding disjoint requests,
not a narrower pool.

### Still not established here

Pipeline parallelism (`pp_size > 1`); multi-node, where the loopback
pinning that a single-host spawn relies on must be overridden; and
`moe_tp_size > 1`, i.e. splitting experts a second way along the
intermediate dimension.
<!-- PORT: verify — out of tree the loopback pinning lived in the standalone
repo's `scripts/env.sh`; the in-tree equivalent for a single-host spawn was
not located. -->
