# Token-Sharded Tensor Parallelism Developer Guide

## Scope

This guide covers `parallel_config.tp_layout='token_sharded'` in VisualGen:

- `tensorrt_llm/_torch/visual_gen/parallel/token_sharded_tp.py`: the plan of which tokens
  each TP rank holds and the collectives (`TokenShardPlan`, `TokenShardedTP`).
- `tensorrt_llm/_torch/visual_gen/parallel/token_sharded_modules.py`: the adapters that
  convert a model's existing TP modules, and the rules that pick them.
- `SequenceSharder` (`tensorrt_llm/_torch/visual_gen/utils.py`): the call sites models
  already have for Ulysses also carry this layout.
- `tensorrt_llm/_torch/visual_gen/models/wan/transformer_wan.py`: the first adopter.

Use it when adopting token-sharded TP in another DiT, or when changing the layout code. User-facing configuration is in
[docs/source/models/visual-generation.md](../../../../docs/source/models/visual-generation.md)
(Multi-GPU Parallelism).

Status: prototype. `tp_layout` unset (the default) or `replicated` means off.

## What it does

With plain tensor parallelism (TP) every rank holds the full residual stream
`[B, S, D]` and each row-parallel projection (attention `to_out`, cross-attention
`to_out`, MLP `down_proj`) ends with an all-reduce. Everything between two projections
(residual add, LayerNorm, AdaLN modulation, NVFP4 quantization) runs redundantly on all
`B * S` tokens on every rank.

Token-sharded TP (Megatron-style sequence parallelism) keeps the residual stream **token-sharded**
across the TP group between projections. Each all-reduce becomes a reduce-scatter
(to the rank's rows) plus an all-gather (before the next column-parallel projection):

```
            x_loc [m, D]  (this rank's rows of the residual stream)
               │ norm1 + AdaLN (+ NVFP4 quantize)              row-local, m rows
               ▼
   all-gather ──► attn1.qkv_proj (COLUMN) ─► QK-norm, RoPE, attention (all B*S tokens)
               ▼
A: attn1.to_out (ROW, K-partials) ─► reduce-scatter ─► gated residual, norm2 (+quant)
               ▼
   all-gather ──► attn2.to_q (COLUMN) ─► text (+image) cross-attention
               ▼
B: attn2.to_out (ROW) ─► reduce-scatter ─► residual, norm3 + AdaLN (+quant)
               ▼
   all-gather ──► ffn.up_proj + GELU ─► ffn.down_proj (ROW, K-partials)
               ▼
C: reduce-scatter ─► gated residual ─► next block's norm1
```

- Row-local work runs on `m ≈ B * S / tp` rows instead of `B * S`.
- With BF16 gathers the same bytes cross NVLink per block as with a ring all-reduce (a
  reduce-scatter plus an all-gather is how a ring all-reduce is built).
- When the next projection has a **static** NVFP4 input scale, the all-gather moves the
  NVFP4 payload plus its scaling factors (0.5625 bytes/element) instead of BF16
  (2 bytes/element), so fewer bytes cross NVLink than with the all-reduce.

It is **not** Ulysses or Ring attention: those shard the sequence *through* attention
(`ulysses_size`, `ring_size`, `attn2d_size`); token-sharded TP only shards it *between* projections
inside a TP group. Combining the two is currently rejected.

## Invariants

1. Only row-local ops run on the shard: residual adds, norms, modulation, quantization.
2. Token-mixing ops (self/cross attention, QK-norm, RoPE) always see the full `[B, S]`
   token set, heads sharded exactly as in plain TP.
3. Every GEMM runs on all tokens: `B * S` rows, except that the row-parallel output
   projections (`to_out`) see `B * S_pad` rows (zero pad rows) when a shape is padded, so
   the autotuner may pick a different tactic for them on such shapes. Dynamic activation
   quantization is unchanged (zero pad rows do not change an amax; the MLP runs on the
   real rows only). A static NVFP4 quantize for `D != 5120` runs on the shard's `m`
   rows, which is row-local and gives the same bytes per row.
4. All ranks of a TP group run the transformer on identically shaped inputs
   (`begin()` checks this on the first use of each shape).
5. Per-row math is the model's own code, run on the shard: the block's norms, modulation
   and residual adds are unchanged. The unit tests assert that the transformer blocks'
   output is **bitwise equal** to the all-reduce model's when that model's all-reduces are
   done as the same reduce-scatter + all-gather. What remains is listed under Numerics.

## Numerics versus all-reduce TP

- **Reduction order and algorithm of the collective.** The reduce-scatter sums the
  K-partials in a different order than the all-reduce, and NCCL may pick a different
  algorithm for it. At `tp = 2` both add the same two partials (bitwise equal). At
  `tp = 4` both use the same algorithm class and reach the same precision. At
  **`tp >= 8` on NVSwitch systems** NCCL's default tuning runs the all-reduce as NVLS but
  the reduce-scatter as a ring, which rounds to bf16 after each of its `tp - 1` hops, so
  Token-sharded TP is measurably less precise than all-reduce TP there.
  Measured on 8x B200 (NGC 1.3.0rc29), bf16 partials of shape `[151200, 5120]` (Wan
  14B, 720p 81 frames, batched CFG) against the fp32 sum rounded once:

  | TP8 collective | Time | Elements not correctly rounded | Rel-L2 |
  |---|---|---|---|
  | All-reduce (NVLS, AR-TP) | 3.74 ms | 24.4 % | 2.9e-3 |
  | Reduce-scatter, default (ring) | 2.14 ms | 52.0 % | 4.0e-3 |
  | Reduce-scatter, `NCCL_ALGO="ReduceScatter:NVLS"` | 3.00 ms | 24.4 % | 2.9e-3 |

  On one static-NVFP4 Wan block (`D = 5120`, TP8) the default ring reduce-scatter
  raises the block's rel-L2 against an fp32-exact all-reduce from 5.3e-3 (AR-TP) to
  6.4e-3; with `NCCL_ALGO="ReduceScatter:NVLS"` token-sharded TP is bitwise equal to AR-TP, at
  about 40 % more reduce-scatter time (`[75600, 5120]`: 1.09 ms ring, 1.52 ms NVLS).
  The block test D8 bounds token-sharded TP's error against an fp32-exact all-reduce at 1.5x
  AR-TP's.
- **Per-token AdaLN (2-D timesteps, e.g. TI2V).** The all-reduce path uses the fused
  per-token AdaLN kernel (SM100, hidden size % 256 == 0); token-sharded TP applies per-token
  modulation on the shard with the model's LayerNorm module. The results differ in the
  last bits.

These are the "REDUCTION-REORDERING" LPIPS class of plain TP.

## Layout and padding (`TokenShardPlan`)

The `B` samples of `S` tokens are flattened to `[B * S_pad]` rows; rank `r` owns rows
`[r * m, (r + 1) * m)` with `m = B * S_pad / tp`.

Padding rule (per sample):

- `d = gcd(tp, B)`, `t' = tp / d`, `S_pad = round_up(S, t')`.
- Each sample is zero-padded at its end to `S_pad` tokens.
- Padding happens iff `B * S % tp != 0`; at most `B * (t' - 1)` extra rows.
- Standard Wan shapes need none (720p/81f `S = 75600`, 480p/81f `S = 32760`, the default
  warmup shapes); odd latent grids do (e.g. 720x720 gives `S = 42525`).

Why per sample and not a global tail pad: with `g = S_pad / t'` rows per group, every
aligned group of `g` local rows lies inside one sample (sample boundaries are multiples
of `S_pad = t' * g`, and `row_start = r * m` is a multiple of `g`). So a per-sample
modulation table on any rank needs only `n = B / d <= B` entries
(`TokenShardPlan.entry_batch`), for every shape. With `B = 2` CFG and `tp` in
`{2, 4, 8}`, `n = 1`: a single slice.

Padding rules at the boundaries:

- `shard` zero-fills pad rows; `unshard` / `all_gather` drop them (for `B = 1` this is a
  free prefix view).
- `TokenShardedRow` pads the GEMM **input** (`tp` times smaller than the output partial);
  zero pad rows leave a dynamic amax unchanged and only carry tp_rank 0's bias, which
  stays confined to pad rows.
- `TokenShardedMLP` runs the MLP on the real `B * S` rows only (pad rows are non-zero
  after a norm and would perturb dynamic-amax quantization inside the MLP) and pads the
  output partial before the reduce-scatter.

The plan is a frozen dataclass of Python ints (compile and CUDA-graph safe), cached per
`(B, S)`.

## How a model runs it

One forward, unchanged, with two pieces installed by `BaseDiffusionModel._apply_tp_layout()`
(called at the end of the model's `__init__`).

**1. The sharder carries the layout.** `SequenceSharder.use_token_sharded_tp(helper)` makes
the call sites a model already has for Ulysses carry token-sharded TP:

| Call | Under token-sharded TP |
|---|---|
| `shard(x, dim=1)`: the residual stream | `begin(B, S)` (selects the forward's plan), then this rank's rows as `[n, g, ...]` sample groups |
| `shard(t, dim=1, expected_seq_len=S)`: per-token tables | the same rows; `t` must match the stream's `[B, S]` |
| `shard_per_sample(t)`: per-sample `[B, ...]` tables | the shard's `[n, ...]` entries |
| `gather(x, dim=1)` | all real tokens, `[B, S, ...]` |
| `shard_rope(...)`, `shard_attention_input(...)`: RoPE, positions, masks, attention-side K/V | unchanged: attention sees all tokens |
| `is_active`, `size`, `rank`, `group` | their sequence-parallel meaning (inactive), so Ulysses-only code stays off |
| `token_sharded_tp` | True |

`[n, g, D]` is `n = len(plan.entry_batch)` groups of `g = plan.rows_per_entry` rows, each
inside one sample. A per-sample `[n, 1, D]` table broadcasts over it as a `[B, 1, D]` table
does over `[B, S, D]`, so block code written for `[B, S, D]` runs unchanged on the shard,
and a fused AdaLN kernel gets `seq_len_per_batch = g` from the shapes.

**2. The converter swaps the boundary modules.** Each converted module becomes a cached
subclass of its own class (a `module.__class__` swap, as PyTorch's FSDP2 `fully_shard`
does): parameters, state-dict names, quant methods and `isinstance` checks are unchanged,
and the GEMM stays the module's (`super().forward`).

| Adapter | Applied to (rule) | Its forward |
|---|---|---|
| `TokenShardedColumn` | an `Attention`'s projections that read the hidden states: `qkv_proj`, `to_q`, and `to_k` / `to_v` when `separate_qkv_is_self_attention` (R3) | quantize first if the consumer has a static NVFP4 input scale; all-gather the real rows; the GEMM; returns `[B, S, N_local]` |
| `TokenShardedRow` | a row-parallel `Linear` that all-reduces (R1) | pad the GEMM input per sample; the GEMM with its all-reduce off; reduce-scatter; returns `[n, g, N]` |
| `TokenShardedMLP` | an `MLP` / `GatedMLP` whose `down_proj` all-reduces (R2) | all-gather the real rows (quantized first for a static NVFP4 input projection); the MLP; pad the output; reduce-scatter |

Everything else runs as in plain TP: QK-norm (`RMSNormTPAware` reduces over heads within
each token), encoder-side K/V projections, the embedders, the output head. The model is built
exactly as for plain TP: the converter turns off the converted projections' all-reduce
(`reduce_output`, `all_reduce`, `use_fused_gemm_allreduce`). Attention takes batch and
sequence lengths from the projected q, so a column adapter may return more tokens than it
was given.

**Wan.** The rules give exactly five conversions per block: `attn1.qkv_proj` and
`attn2.to_q` (column), `attn1.to_out.0` and `attn2.to_out.0` (row), `ffn` (MLP). Wan
declares no exceptions. Its forward adds one `shard_per_sample` call and keys four sites on
`sharder.token_sharded_tp`: the `expand_timesteps` broadcast, the per-token AdaLN runtime
(off on shards), the static-scale rule for its norms' NVFP4 output, and the VSA
rejection. `WanBlock.forward` raises when the modulation table does not match its hidden
states, so a global table cannot silently mix samples in the fused AdaLN kernel. The check
is necessary, not sufficient: when `n == B` (`gcd(tp, B) == 1` with `B > 1`, e.g. TP3 with
`B = 2`) the global table has the shard's shape, and only the sharder's table is right.

## The converter

- `classify(model, containers=..., exceptions=...)` returns `{module name: kind}` without
  converting; `convert_to_token_sharded_tp(...)` converts and logs one line.
- **Exceptions** (`_token_sharded_tp_exceptions`, relative to a block):
  `{pattern: "column" | "row" | "mlp" | adapter class | "keep"}`, for what the rules
  cannot decide.
- **`register_token_sharded_adapter(module_cls, adapter)`**: an adapter for a custom module
  class (e.g. a joint projection with its own all-reduce), used wherever the class appears.
- **`TokenShardedAdapter.prepare(module, tp, name)`** runs on every converted module before
  its class is swapped: it checks that the module was built for the adapter and stops its
  all-reduce. The built-in adapters check the column / row `Linear`, the MLP's projections
  and the TP size, and refuse a `Linear` built for the fused GEMM + all-reduce. A custom
  adapter overrides it to stop its module's own all-reduce.
- **Validation** raises when:
  - a block has nothing to convert (is the model built with `tp_size > 1`?);
  - an all-reduce would act on a token shard. Allowed are a converted projection's own, a
    TP-aware RMSNorm's per-token head reduction, and a column `Linear`'s unused one.
    Anything else needs a registered adapter or an exception.
  - the blocks of one container would convert differently (they share one compiled graph).

  Converting twice also raises.
- **Adapter classes are cached** per (adapter, base class), so one compiled block graph serves
  all blocks. Deepcopy and pickle of a converted model are not supported.

## Adopting it in another model

1. Plain TP must work: the rules read its TP metadata.
2. Set `_supports_token_sharded_tp = True`, call `self._apply_tp_layout()` at the end of
   `__init__` (after `self.sharder` exists), and set `_token_sharded_tp_blocks` if the block
   containers are not `blocks`. The pipeline loader calls `check_tp_layout_applied()`, so a
   model that declares support but never applies the layout raises.
3. Route the residual stream through `sharder.shard` / `gather` (as for Ulysses), then
   per-token tables through `sharder.shard(..., expected_seq_len=S)`, per-sample tables
   through `sharder.shard_per_sample`, and attention-side inputs (RoPE, positions, masks)
   through `shard_rope` / `shard_attention_input`.
4. Code between projections must be row-local, and must take batch and sequence lengths
   from the projected q (not from the block input) where it needs them.
5. Add exceptions or register adapters for custom modules, and add a bitwise test at
   `B = 2`, `TP = 2` against the model's plain-TP twin with its all-reduces emulated
   (`classify` finds them).

## Current limits

- **One token stream per model.** MMDiT text + image or audio + video need a plan per
  stream, and Attention recording which stream its q and k/v read.
- **Separate q/k/v self-attention projections gather three times**, once per projection.
  Such an `Attention` must set `separate_qkv_is_self_attention=True`; without it only
  `to_q` converts, and `get_qkv` raises because q and k cover different tokens.
- **The rank-agreement check runs only on the first use of each shape.** A rank that reuses a
  cached shape while a peer starts a new one is not detected: the collectives then hang or
  fail instead of raising.
- **NVFP4 gathers need static scales;** dynamic NVFP4 gathers BF16 (see below).
- **Per-token AdaLN fused kernels stay off under this layout.**

## Quantization: what is all-gathered

The rule: gather NVFP4 iff the consumer has a **static** NVFP4 input scale
(`static_nvfp4_input_scale(linear)` is not `None`); quantizing each shard with that
scale gives exactly the bytes the consumer would produce on all rows. The column and MLP
adapters quantize a bf16 input by this rule (decided per call); Wan's fused norms already
emit NVFP4 at `D == 5120`, with their scales wired by the same rule. That quantize is pinned
to `trtllm::fp4_quantize`, even when VisualGen tunes the Linears' quantize: the tunable op
may pick FlashInfer's kernel, which shuffles rows and scaling factors in 128-row tiles,
while the all-gather and the scale regroup need plain row order.

| Checkpoint / mode | Gathered | Numerics vs all-reduce TP |
|---|---|---|
| ModelOpt static NVFP4, `D = 5120` (fused LN op) | FP4 payload + 128x4-swizzled SF | Identical bytes per row; only the collective differs |
| Static NVFP4, other `D` | FP4 (`quantize_nvfp4` on the shard) | Identical bytes per row |
| NVFP4 `dynamic: true` (dynamic activations) | BF16 | Unchanged: the consumer takes amax over the gathered rows |
| NVFP4-AWQ, `force_dynamic_quantization`, FP8, BF16, excluded layers | BF16 | Unchanged |

The FP4-gather bandwidth win therefore needs a calibrated (static) NVFP4 checkpoint;
dynamic NVFP4 gathers BF16. The Wan model logs once after loading how many
block-boundary activations are gathered as NVFP4.

Scaling-factor layout: each rank's SF buffer is laid out for `pad128(m)` rows;
`regroup_swizzled_sf` re-tiles the gathered buffers for `B * S` rows (zero-copy when
`m % 128 == 0` and the shape is unpadded or `B == 1`, otherwise one gather copy under
Inductor).

## torch.compile and CUDA graphs

- The sharder's `shard` / `gather` run in the eager model forward (with `begin`); blocks
  are compiled one by one as today. Inside a block token-sharded TP adds only the adapters'
  functional collectives (`all_gather_single` / `reduce_scatter_single` on the group
  *name*, each followed by an explicit wait), views/pads, existing custom ops and reads of
  the plan's Python ints. The adapter classes are cached per base class, so blocks share
  one graph.
- Each new `(B, S)` specializes the block graphs: the plan's ints are guarded as
  constants, so unlike the all-reduce path the blocks do not become shape-dynamic after
  a few shapes. Warm up every served shape, and keep the number of distinct shapes well
  under `torch._dynamo.config.cache_size_limit` (the pipeline sets 128), beyond which
  Dynamo falls back to eager.
- No logging and no object collectives run inside blocks.
- Token-sharded TP adds no graph break of its own (tested against the all-reduce path's break
  reasons); blocks keep the pre-existing QK-norm all-reduce break.
- CUDA graphs: the runner's eager warmups create the plan (and its shape check) before
  capture; nothing in the captured region allocates on the host or copies host to device.
  Call `begin()` once eagerly per new shape before any capture.
- Captured graphs contain collectives on the TP process group's NCCL communicator:
  release them (`CUDAGraphRunner.clear()`, as `BasePipeline.cleanup()` does) before
  `destroy_process_group()`, otherwise the teardown can hang.

## Supported combinations

| Setting | Behavior | Tested |
|---|---|---|
| `cfg_size = 2` | Composes (TP groups sit inside each CFG half). | E2E LPIPS (`cfg2_tp2_ts`) |
| Batched CFG (`B = 2`) | Composes; a shard may straddle the cond/uncond boundary (per-shard tables handle it). | Unit (TP 2/3/4/8) |
| Uneven heads (e.g. `tp_size = 3`) | Composes (token sharding is independent of head sharding). | Unit |
| torch.compile, CUDA graphs | Compose. | Unit |
| ulysses / ring / attn2d / async ulysses | Rejected. | Unit |
| Cache-DiT | Rejected (per-block skip decisions would see token-sharded hidden states). | Unit |
| VSA | Rejected (the gates are projected from the token-sharded block input). | Unit |
| TeaCache | Expected to compose (it wraps the whole forward); accepted by the config. | Config only |
| CPU offload, parallel VAE, runtime LoRA | Expected to compose (they do not touch the block boundaries). | Not tested |

## Where later optimizations land

| Optimization | Lands in | Model change |
|---|---|---|
| FP8 block-scale all-gather (1x128 per-token scales) | `TokenShardedColumn` / `TokenShardedMLP`, plus row alignment in the plan | none |
| Fused GEMM + reduce-scatter; per-destination GEMM + copy-engine push | `TokenShardedRow` (an inner adapter on `down_proj` for MLPs) | none |
| Per-source GEMMs as rows arrive (copy-engine all-gather) | the column / MLP adapters; needs `out=` on the GEMM runners | none |
| Attention overlapped with `to_out`'s reduce-scatter | an Attention method plus a per-chunk API on `TokenShardedRow` | one branch in the block |
| Reduce-scatter fused with the residual + norm | a layout op with one return type in both layouts | the block passes its epilogue |
| Dynamic-scale NVFP4 gather; coalesced payload + scaling-factor gathers | the column / MLP adapters | none |
| Composition with Ulysses / ring | nest the token shard inside the sequence shard | none expected |
| Symbolic plan sizes (blocks stay shape-dynamic across `(B, S)`) | `TokenShardPlan` | none |

Each optimization keeps the adapter's plain path (`super().forward`) as its fallback, chosen
by a shape/dtype rule, and is tested against it.

## Tests

| Level | File |
|---|---|
| Plan, per-shard tables, SF layout, shape checks, capability gate, Wan's table check (CPU) | `tests/unittest/_torch/visual_gen/test_token_sharded_tp.py` |
| Sharder mode, converter rules and validation (CPU) | `tests/unittest/_torch/visual_gen/test_token_sharded_tp_layout.py` |
| Collectives, sharder round trip and adapters (gloo, exact); real Linear / MLP / GatedMLP, NVFP4, fullgraph compile, CUDA graphs (NCCL) | `tests/unittest/_torch/visual_gen/multi_gpu/test_token_sharded_tp_collectives.py` |
| Fused LN + NVFP4 on shards (Blackwell) | `tests/unittest/_torch/visual_gen/kernels/parallel/test_token_sharded_tp_norm.py` |
| Wan model / block level (bitwise vs emulated all-reduce, vs single GPU, vs fp32-exact all-reduce) | `tests/unittest/_torch/visual_gen/multi_gpu/test_wan_token_sharded_tp.py` |
| E2E LPIPS (`cfg2_tp2_ts`) | `tests/integration/defs/examples/visual_gen/test_visual_gen_multi_gpu.py` |
| Shared SF-layout references for the tests above | `tests/unittest/_torch/visual_gen/token_sharded_tp_test_utils.py` |
