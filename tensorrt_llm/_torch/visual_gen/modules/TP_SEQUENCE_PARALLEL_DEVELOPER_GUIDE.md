# Sequence-Parallel Tensor Parallelism (SP-TP) Developer Guide

## Scope

This guide covers `parallel_config.tp_sequence_parallel` in VisualGen:

- `tensorrt_llm/_torch/visual_gen/modules/tp_sequence_parallel.py` — the model-neutral
  helper (`TPSequenceParallel`, `TokenShardPlan`, `RowNorm`, row-local functions).
- `tensorrt_llm/_torch/visual_gen/modules/fused_norm_quant.py` — the fused
  LayerNorm(+AdaLN / affine)(+NVFP4) wrappers the helper uses at `D == 5120`.
- `tensorrt_llm/_torch/visual_gen/models/wan/transformer_wan.py` — the first adopter
  (`WanBlock._forward_sp_tp`).

Use it when adding SP-TP to another DiT (in-tree or your own model code), or when
changing the helper. User-facing configuration is in
[docs/source/models/visual-generation.md](../../../../docs/source/models/visual-generation.md)
(Multi-GPU Parallelism).

Status: prototype. `tp_sequence_parallel: None` (the default) means off.

## What it does

With plain tensor parallelism (TP) every rank holds the full residual stream
`[B, S, D]` and each row-parallel projection (attention `to_out`, cross-attention
`to_out`, MLP `down_proj`) ends with an all-reduce. Everything between two projections
(residual add, LayerNorm, AdaLN modulation, NVFP4 quantization) runs redundantly on all
`B * S` tokens on every rank.

SP-TP (Megatron-style sequence parallelism) keeps the residual stream **token-sharded**
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
(`ulysses_size`, `ring_size`, `attn2d_size`); SP-TP only shards it *between* projections
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
5. Per-row math is the model's own: the fused op at `D == 5120`, the model's
   `LayerNorm` module (`RowNorm.module`) otherwise. The unit tests assert that the
   transformer blocks' output is **bitwise equal** to the all-reduce model's when that
   model's all-reduces are done as the same reduce-scatter + all-gather. What remains is
   listed under Numerics.

## Numerics versus all-reduce TP

- **Reduction order and algorithm of the collective.** The reduce-scatter sums the
  K-partials in a different order than the all-reduce, and NCCL may pick a different
  algorithm for it. At `tp = 2` both add the same two partials (bitwise equal). At
  `tp = 4` both use the same algorithm class and reach the same precision. At
  **`tp >= 8` on NVSwitch systems** NCCL's default tuning runs the all-reduce as NVLS but
  the reduce-scatter as a ring, which rounds to bf16 after each of its `tp - 1` hops, so
  SP-TP is measurably less precise than all-reduce TP there.
  Measured on 8x B200 (NGC 1.3.0rc29), bf16 partials of shape `[151200, 5120]` (Wan
  14B, 720p 81 frames, batched CFG) against the fp32 sum rounded once:

  | TP8 collective | Time | Elements not correctly rounded | Rel-L2 |
  |---|---|---|---|
  | All-reduce (NVLS, AR-TP) | 3.74 ms | 24.4 % | 2.9e-3 |
  | Reduce-scatter, default (ring) | 2.14 ms | 52.0 % | 4.0e-3 |
  | Reduce-scatter, `NCCL_ALGO="ReduceScatter:NVLS"` | 3.00 ms | 24.4 % | 2.9e-3 |

  On one static-NVFP4 Wan block (`D = 5120`, TP8) the default ring reduce-scatter
  raises the block's rel-L2 against an fp32-exact all-reduce from 5.3e-3 (AR-TP) to
  6.4e-3; with `NCCL_ALGO="ReduceScatter:NVLS"` SP-TP is bitwise equal to AR-TP, at
  about 40 % more reduce-scatter time (`[75600, 5120]`: 1.09 ms ring, 1.52 ms NVLS).
  The block test D8 bounds SP-TP's error against an fp32-exact all-reduce at 1.5x
  AR-TP's.
- **Per-token AdaLN (2-D timesteps, e.g. TI2V).** The all-reduce path uses the fused
  per-token AdaLN kernel (SM100, hidden size % 256 == 0); SP-TP applies per-token
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
- `row_linear` pads the GEMM **input** (`tp` times smaller than the output partial);
  zero pad rows leave a dynamic amax unchanged and only carry tp_rank 0's bias, which
  stays confined to pad rows.
- `mlp_residual` runs the MLP on the real `B * S` rows only (pad rows are non-zero after
  a norm and would perturb dynamic-amax quantization inside the MLP) and pads the output
  partial before the reduce-scatter.

The plan is a frozen dataclass of Python ints (compile and CUDA-graph safe), cached per
`(B, S)`.

## Building blocks

`TPSequenceParallel` is not an `nn.Module`: it holds no parameters or tensors, one
instance per transformer (per token stream), and blocks keep a plain reference.

| Call | Shapes / contract |
|---|---|
| `begin(B, S)` | Eager, once per forward before the blocks. Selects the cached plan; the first use of a shape does one `all_gather_object` shape check (error on disagreement). |
| `shard(x)` | `[B, S, D]` (any strides) → contiguous `[m, D]`. |
| `unshard(x_loc)` | `[m, D]` → `[B, S, D]` (all-gather, padding dropped). |
| `shard_rows(t)` | Per-token metadata `[B, S, *rest]` → `[m, *rest]` (zeros on pad rows). |
| `per_sample_table(t)` | Per-sample table `[B, *rest]` → this shard's `[n, *rest]`; entry `j` applies to local rows `[j * g, (j + 1) * g)`. Built from slices only (graph-capture safe). |
| `reduce_scatter(partial)` | `[B * S_pad, N]` (or `[B * S, N]` / `[B, S, N]`, padded here) → `[m, N]`. |
| `all_gather(act_loc)` | `[m, K]` → `[B * S, K]`; an `Fp4QuantizedTensor` → payload `[B * S, K/2]` + SF regrouped for `B * S` rows. |
| `residual(x, y, gate=None)` | `x + y`, or `x + y * gate` in fp32 (`gate`: this shard's `[n, D]` table). |
| `norm(x, RowNorm)` | Row-local LayerNorm (+AdaLN and/or affine) (+static NVFP4 quantize). |

Every call checks its input's leading dimensions against the current plan (so a
forgotten `shard()` or a `begin()` for another shape in between raises instead of
silently reinterpreting rows), and `residual` / `norm` check that each table has this
shard's entry count (`per_sample_table`) or row count (`shard_rows`).

GEMM-owning boundary ops — the seam that later overlap / fused GEMM+collective kernels
replace, so model code does not change again:

| Call | Does |
|---|---|
| `column_linear(linear, act_loc)` | all-gather → column-parallel `linear` → `[B, S, N_local]` |
| `row_linear(linear, act)` | pad → row-parallel `linear` (K-partials) → reduce-scatter → `[m, N]` |
| `row_linear_residual_norm(linear, act, residual, *, gate, norm)` | `row_linear` → residual → norm; returns `(x, h)` |
| `mlp_residual(mlp, act_loc, residual, *, gate)` | all-gather → MLP → reduce-scatter → residual |

TRT-LLM modules passed to them are checked: a `Linear` must have the expected TP mode
(`ROW` with `reduce_output=False`, or `COLUMN` without `gather_output`) and the helper's
TP size; for `mlp_residual` any module with a TRT-LLM `down_proj` (`MLP`, `GatedMLP`,
...) has its `down_proj` checked the same way; an NVFP4-gathered input must reach a
`Linear` that has a static NVFP4 input scale. Any other callable is accepted as is and
must follow the contracts below.

`RowNorm(eps, weight, bias, scale, shift, quant_scale, identity, module)`: affine
(`weight`/`bias`) and/or AdaLN (`scale`/`shift`, `[n, D]` tables); `identity=True` skips
the norm but still quantizes. At `D == 5120` with bf16 on CUDA and exactly one of affine
or AdaLN the fused `fused_adaptive_layernorm(_quant)` op runs; otherwise fp32 LayerNorm
(`module(x.float())` when `module` is given — pass the model's own LayerNorm so the
kernel matches the all-reduce path — else `F.layer_norm`), then the modulation, then
`quantize_nvfp4` if `quant_scale` is set. Custom norms (e.g. RMSNorm) can be passed to
`row_linear_residual_norm(norm=callable)` and call `quantize_nvfp4`.

Never pass a global `[B, D]` modulation table to a row-local op on a shard: the fused
AdaLN kernel indexes `batch = local_row // seq_len_per_batch`, so on a shard it would
silently mix CFG halves. Build the shard's table with `per_sample_table` /
`shard_rows`. The helper rejects a table whose length is neither the shard's entry count
nor its row count; when `B` and `tp` are coprime the per-sample table has `B` entries,
the same as the global one, so that case cannot be caught.

## Contracts for custom callables

| Argument | Contract |
|---|---|
| `row_linear` / `row_linear_residual_norm` `linear` | Takes `[B * S_pad, K_local]` (or the input's leading shape); returns this rank's K-partial sums `[rows, N]`. Add the bias on exactly one rank, otherwise the reduce-scatter sums it `tp` times. |
| `column_linear` `linear` | Takes `[B * S, K]` bf16 or an `Fp4QuantizedTensor`; returns a contiguous `[B * S, N_local]` (the result is viewed as `[B, S, N_local]`). |
| `mlp_residual` `mlp` | Takes `[B * S, K]` (or FP4); returns K-partial sums `[B * S, N]`, bias on exactly one rank. |
| `norm` callable | Row-local `[m, N] → [m, N]` or an `Fp4QuantizedTensor` for `m` rows. |
| NVFP4 activations | Quantize with the **consumer's** static input scale (`static_nvfp4_input_scale(consumer)`) and give the result only to that consumer: a TRT-LLM `Linear` fed a pre-quantized input without `reciprocal_scale` uses its own `input_scale`-derived alpha, and the helper cannot check that the two scales are the same tensor. |

## Using SP-TP in your own DiT

1. **Group.** Inside VisualGen: `TPSequenceParallel.from_model_config(model_config)`
   (returns `None` unless `tp_sequence_parallel` is set, and validates the mapping).
   Elsewhere the helper itself only needs a `torch.distributed` group:
   `TPSequenceParallel(tp_process_group)`; the group rank order is the token-shard order.
2. **Layers.** TRT-LLM `Linear` / `MLP` / `GatedMLP` need a TRT-LLM `Mapping` whose TP
   communicators exist: inside VisualGen that is `model_config.mapping`; outside it build
   the mapping with `VisualGenMapping(...).to_llm_mapping()` and give `MLP` / `GatedMLP`
   a `ModelConfig(mapping=..., allreduce_strategy=AllReduceStrategy.NCCL)` (their column
   projections are built with the default all-reduce, whose `AUTO` strategy needs an IPC
   workspace). Or pass your own GEMM callables.
   - Column projections: `Linear(..., tensor_parallel_mode=COLUMN, reduce_output=False)`.
   - Row projections: `Linear(..., tensor_parallel_mode=ROW, reduce_output=False)`,
     `MLP(reduce_output=False)` / `GatedMLP(reduce_output=False)`, VisualGen
     `Attention(reduce_output=False)` (its `to_out`).
   - For a `BaseDiffusionModel` subclass set `_supports_tp_sequence_parallel = True`
     (otherwise the base class rejects the option).
3. **Per forward (eager).** `sp.begin(B, S)`, `x = sp.shard(x)`, the blocks, then
   `x = sp.unshard(x)` before the (replicated) output head.
4. **Per boundary.**
   `x, h = sp.row_linear_residual_norm(out_proj, o, x, gate=..., norm=RowNorm(...))`,
   then `y = sp.column_linear(in_proj, h)`; MLPs: `sp.mlp_residual(mlp, h, x, gate=...)`.
5. **Modulation.** `sp.per_sample_table(table + temb)` for `[B, ...]` AdaLN;
   `sp.shard_rows(...)` for per-token metadata.
6. **FP4 gather.** `RowNorm(quant_scale=static_nvfp4_input_scale(next_linear))`.
7. **Attention.** VisualGen `Attention` exposes `split_qkv()` and `attend()` (QK-norm,
   RoPE and attention on already-projected q/k/v, returning the output before `to_out`),
   so the column projection can run through `column_linear`.

```python
class MyBlock(nn.Module):
    def __init__(self, model_config, sp):
        super().__init__()
        self.sp = sp  # TPSequenceParallel or None
        mapping = model_config.mapping
        self.norm1 = LayerNorm(hidden_size=D, eps=1e-6, has_weights=False, has_bias=False)
        self.qkv = Linear(D, 3 * D, mapping=mapping, reduce_output=False,
                          tensor_parallel_mode=TensorParallelMode.COLUMN, ...)
        self.out = Linear(D, D, mapping=mapping, reduce_output=sp is None,
                          tensor_parallel_mode=TensorParallelMode.ROW, ...)
        self.mlp = MLP(hidden_size=D, intermediate_size=F, bias=True, config=model_config,
                       reduce_output=sp is None)

    def forward_sp(self, x_loc, temb):
        sp = self.sp
        shift, scale, gate, c_shift, c_scale, c_gate = sp.per_sample_table(
            self.table.float() + temb.float()).unbind(1)
        h = sp.norm(x_loc, RowNorm(scale=scale, shift=shift, module=self.norm1,
                                   quant_scale=static_nvfp4_input_scale(self.qkv)))
        o = my_attention(sp.column_linear(self.qkv, h))  # all [B, S] tokens
        x_loc, h = sp.row_linear_residual_norm(
            self.out, o, x_loc, gate=gate,
            norm=RowNorm(scale=c_scale, shift=c_shift,
                         quant_scale=static_nvfp4_input_scale(self.mlp.up_proj)))
        return sp.mlp_residual(self.mlp, h, x_loc, gate=c_gate)
```

## Current limits

- One helper holds one plan: `begin()` replaces it. Dual-stream (MMDiT) models need one
  helper per token stream.
- `column_linear` all-gathers on every call, so separate q/k/v projections done that
  way gather three times. The alternative, one `sp.all_gather` followed by direct GEMM
  calls, works but sits outside the boundary ops that later overlap work replaces.
- Model-owned attention classes must split projection, attention and output projection
  themselves (VisualGen `Attention` provides `split_qkv()` / `attend()`).
- `RowNorm` is LayerNorm only; RMSNorm goes through a `norm=` callable.
- The rank-agreement check runs on the first use of each shape only: a rank that reuses
  a cached shape while a peer starts a new one is not detected (the collectives then
  hang or fail instead of raising).
- NVFP4 gathers need static scales; dynamic NVFP4 gathers BF16 (see below).

## Quantization: what is all-gathered

The rule: gather NVFP4 iff the consumer has a **static** NVFP4 input scale
(`static_nvfp4_input_scale(linear)` is not `None`); quantizing each shard with that
scale gives exactly the bytes the consumer would produce on all rows. Wan wires its
`RowNorm.quant_scale` values with this function when SP-TP is on.

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

- `begin`, `shard` and `unshard` run in the eager model forward; blocks are compiled one
  by one as today. Inside a block SP-TP adds only functional collectives
  (`all_gather_single` / `reduce_scatter_single` on the group *name*, each followed by
  an explicit wait), views/pads, existing custom ops and reads of the plan's Python ints.
- Each new `(B, S)` specializes the block graphs: the plan's ints are guarded as
  constants, so unlike the all-reduce path the blocks do not become shape-dynamic after
  a few shapes. Warm up every served shape, and keep the number of distinct shapes well
  under `torch._dynamo.config.cache_size_limit` (the pipeline sets 128), beyond which
  Dynamo falls back to eager.
- No logging and no object collectives run inside blocks.
- SP-TP adds no graph break of its own (tested against the all-reduce path's break
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
| `cfg_size = 2` | Composes (TP groups sit inside each CFG half). | E2E LPIPS (`cfg2_tp2_sp`) |
| Batched CFG (`B = 2`) | Composes; a shard may straddle the cond/uncond boundary (per-shard tables handle it). | Unit (TP 2/3/4/8) |
| Uneven heads (e.g. `tp_size = 3`) | Composes (token sharding is independent of head sharding). | Unit |
| torch.compile, CUDA graphs | Compose. | Unit |
| ulysses / ring / attn2d / async ulysses | Rejected. | Unit |
| Cache-DiT | Rejected (per-block skip decisions would see token-sharded hidden states). | Unit |
| VSA | Rejected (the gates are projected from the token-sharded block input). | Unit |
| TeaCache | Expected to compose (it wraps the whole forward); accepted by the config. | Config only |
| CPU offload, parallel VAE, runtime LoRA | Expected to compose (they do not touch the block boundaries). | Not tested |

## Extension points (later work)

- Communication/compute overlap, fused GEMM + reduce-scatter (with a residual + norm +
  quantize epilogue) and fused all-gather + GEMM replace `column_linear`,
  `row_linear_residual_norm` and `mlp_residual`; call sites do not change.
- Dynamic-scale NVFP4 gather (all-reduce-max of the local amax, then quantize the shard)
  and static FP8 gather.
- Composition with Ulysses / ring (nest the token shard inside the sequence shard).
- Other models adopt through the helper and `_supports_tp_sequence_parallel`.
- Coalescing the payload and scaling-factor all-gathers.
- Symbolic plan sizes, so blocks stay shape-dynamic across `(B, S)`.

## Tests

| Level | File |
|---|---|
| Plan, row-local ops, SF layout, shape/table checks, validation (CPU) | `tests/unittest/_torch/visual_gen/test_tp_sequence_parallel.py` |
| Collectives, gloo and NCCL (bitwise), Linear/MLP/GatedMLP checks, compile, CUDA graphs | `tests/unittest/_torch/visual_gen/multi_gpu/test_tp_sequence_parallel_collectives.py` |
| Fused LN + NVFP4 on shards (SM100) | `tests/unittest/_torch/visual_gen/kernels/parallel/test_tp_sequence_parallel_norm.py` |
| Wan model / block level (bitwise vs emulated all-reduce, vs single GPU, vs fp32-exact all-reduce) | `tests/unittest/_torch/visual_gen/multi_gpu/test_wan_tp_sequence_parallel.py` |
| E2E LPIPS (`cfg2_tp2_sp`) | `tests/integration/defs/examples/visual_gen/test_visual_gen_multi_gpu.py` |
| Shared SF-layout references for the tests above | `tests/unittest/_torch/visual_gen/tp_sequence_parallel_test_utils.py` |
