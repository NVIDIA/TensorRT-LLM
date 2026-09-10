# Expert weight packing (sparse MoE)

What a target assembled on a sparse-MoE checkpoint had to establish by
observation, because no contract or reference stated it. Written from the
qwen3-30b-a3b/sm_100/tp1 onboard (Qwen3-30B-A3B: 48 layers, all MoE, 128
experts, top-8, `moe_intermediate_size` 768, bf16). The mechanism — a
router selecting k of E stacked expert MLPs — recurs across model
families; this file is about the mechanism, not that checkpoint.

**Every shape below is a whole-stack shape, measured at world size 1.**
Under expert parallelism a rank holds `E / ep_size` of the stack and the
first dimension of every weight and scale shrinks accordingly.

**What survives the split — established by the first expert-parallel
target** (`tep4`, 72 experts over 4 ranks, 18 per rank at offset
`18 * ep_rank`):

* **Every per-expert preparation step survives unchanged.** The
  `[up; gate]` concat, the interleave, the 32-row block shuffle and the
  128x4 scale swizzle all happen *within* one expert, and EP splits the
  stack on the expert axis only. So every transform below applies
  verbatim, with the first dimension `E → E/ep_size` and the destination
  index `e - local_expert_offset`. Nothing about the shuffle or the
  swizzle is a function of the stack height.
* **The three per-expert scale scalars must be `[local_num_experts]`, but
  the checkpoint's scalars must still be read whole.** The runner rejects
  `num_experts`-long scalars, so the window slice is real — but the
  *validity* argument for the shared-expert activation scale is global
  (`shared.input_scale == max` over all `E` routed `input_scale`), and a
  target that windows at load time can no longer assert it. Load all `E`
  (six fp32 per expert — negligible), assert globally, slice in the
  post-load derivation.
* **`leftover == {}` stops being the coverage assert.** Under EP each rank
  legitimately leaves the off-window experts' weight and scale keys
  unconsumed. Replace it with an explicitly predicted leftover set
  computed from the rank's window — that keeps the assert bidirectional
  rather than relaxing it to a warning, which is the whole value of having
  it.

One cross-rank invariant a single rank cannot check, recorded because the
assembly rests on it: **the four windows must tile the routing space
exactly once.** They do because the router GEMM and the routing op are
replicated and deterministic and every rank sees identical all-reduced
inputs — so each token's top-k ids are the same on every rank, and each
id falls in exactly one window.

**That justification is topology-specific, and it does not survive
attention data parallelism.** Under a `depN` segment the ranks hold
*different* tokens, so "every rank sees identical inputs" is false by
construction and the invariant loses its support — while remaining just as
load-bearing, because EP still requires each expert id to belong to exactly
one rank's window. Established by the first attention-DP MoE target
(`dep4`): what restores it is **placing the token all-gather before the
router GEMM, not merely before the expert call.** The router then runs on a
byte-identical full token set on every rank, the original argument applies
again verbatim, and the four windows tile as before.

The failure this rules out is worth naming, because it is silent. Gathering
*after* the router — the arrangement that looks equivalent, and is cheaper
by the width of the routing tensors — leaves every rank routing only its
own tokens, so a token's expert ids exist on one rank alone and the other
three windows never see it. There is no error and no hang; the output is
simply missing most of its expert contributions. Both gates would be the
only thing that catches it.

Stated as the rule: **on any topology where the ranks' token sets differ,
the collective that reconstitutes the token set belongs upstream of the
router, and the tiling invariant is what decides that placement** — not the
expert call's input requirements, which the later placement also satisfies.

## The shape of the vocabulary

A bf16 MoE block is three catalog calls, not one:

```
router_logits          = cublas_mm(x, router_weight_view)     # [T, E]
expert_ids, scales     = renorm_moe_routing_op(router_logits, topk)
moe_out                = fused_moe(x, expert_ids, scales, fc1, None, fc2, None, ...)[0]
```

`fused_moe` contains the permutation, both grouped GEMMs, the gated
activation and the routing-weighted combine. It does **not** contain the
router GEMM or the selection — those stay the caller's, which is why the
routing op exists as a separate entry.

Consequence worth stating because it looks like an omission: a
fully-MoE checkpoint's forward references **no `activation/*` entry**. The
SwiGLU lives inside `fused_moe` (`activation_type=5`). An audit that
expects `silu_and_mul` in every model's forward is wrong for this class.

## The `[up | gate]` order — the one silent-wrong-answer trap

`fused_moe`'s `fc1_expert_weights` is `[E, 2I, H]` with **up rows first**:

- `fc1[e, :I]` is HF `up_proj` (trtllm `w3`)
- `fc1[e, I:]` is HF `gate_proj` (trtllm `w1`) — the sigmoid is applied to
  this half

`fc2_expert_weights` is `[E, H, I]`, plain `down_proj`.

Getting the halves backwards produces a plausible, entirely different
result: finite, correctly scaled, no error, no NaN. Nothing in the stack
detects it — not the op, not a smoke test. Only an accuracy gate catches
it, and only because it lands ~50 points low.

**This is the opposite of the dense convention.** A dense target packs
`gate_up` with gate rows first, because that is what
`flashinfer_silu_and_mul` expects over a packed last dim. The two orders
must never be copied across; each is correct for its own consumer.

## Checkpoint layout, and where the key names actually live

A 4.51-era checkpoint stores experts **unstacked**: 128 × 3 separate 2D
tensors per layer,
`model.layers.{i}.mlp.experts.{e}.{gate,up,down}_proj.weight`. Recent
transformers models the same block with stacked 3D `gate_up_proj` /
`down_proj` parameters — so **the installed modeling file is not the
source of truth for checkpoint key names**. The safetensors index is.
Read `model.safetensors.index.json`, not `modeling_*.py`, when writing
the manifest.

The unstacked on-disk layout reaches `fused_moe`'s stacked layout with
**zero transforms** — every copy is layout-preserving:

```python
(f"{p}.mlp.experts.{e}.up_proj.weight",   (e, slice(0, inter))),
(f"{p}.mlp.experts.{e}.gate_proj.weight", (e, slice(inter, 2 * inter))),
(f"{p}.mlp.experts.{e}.down_proj.weight", (e,)),
```

This needs one generalization of the manifest convention: the destination
widens from a 2D `(row_start, row_end)` range to an index tuple into
`param.data`, so one table serves both fused-2D parameters (qkv) and
stacked-3D ones (experts). It is a one-line change in `fill`.

3D parameters need no special handling anywhere else: they pass through
`MetaInitMode` and engine materialization unchanged. Measured at this
scale — 434 `ParameterDict` entries, 58 GB of expert stacks, 18432
per-expert copies of ~3.1 MB — the whole load sits inside a 14.9 s model
init.

## What the structure dictates, and what it makes irrelevant

Established by measurement during the perf campaign; these are properties
of the mechanism, so the next MoE target can skip the sweeps.

**The MoE is HBM-roofline-bound at serving batch sizes.** Above a decode
batch of roughly `E / topk × small factor` (~40 here) essentially every
expert is active, so a decode step reads the **entire** expert stack once
— 58 GB for this geometry. Measured 7.350 ms/step for the two grouped
GEMMs at concurrency 256 ⇒ **7.89 TB/s**, i.e. the B200 HBM roofline. No
config knob and no scheduling change moves this. Only fewer weight bytes
(quantization — a modeling change) would.

**The kernel-count floor is dominated by the MoE.** `fused_moe` plus
`renorm_moe_routing_op` contribute 7 of the 16 kernels per layer
(`customMoeRouting`, `fusedBuildExpertMapsSortFirstToken`,
`expandInputRows`, grouped GEMM 1, `doActivation`, grouped GEMM 2,
`computeStridesTmaWarpSpecialized`). Across 48 layers that is 336 of 768
kernels per decode step. At concurrency 1 the measured 3.87 ms TPOT over
794 kernels is 4.9 µs each — the same order as the smallest kernels
measured at batch 256, i.e. a fixed per-kernel floor rather than work. The
single-stream latency point is structural until the vocabulary gains a
coarser op.

**KV capacity does not bind.** A 3B-active/30B-total model at tp1 leaves
the KV pool enormous relative to any realistic concurrency (measured:
1,168,096 pool tokens = 570 requests at ISL+OSL 2048, against a 256-request
ceiling). `kv_cache_config.free_gpu_memory_fraction` and the capacity
scheduler policy are inert — do not spend sweeps on them.

**The autotuner is not a cold-cache risk inside a served engine.**
`fused_moe`'s contract warns that a cold tuning cache silently falls back
to a default tactic. In serving that does not apply: the runtime's
`_run_autotuner_warmup` wraps the forward in `autotune()` before capture
(observed `Cache size after warmup is 28` = 2 tunable GEMMs × 14
power-of-2 token buckets). Worth knowing before anyone spends an iteration
on it — and worth knowing in the other direction too: the tuned tactic
changes the *bits*, not just the speed (measured 2.96 ulp against the cold
fallback on the same input), so a catalog receipt taken cold certifies a
tactic the served engine never runs. Both states sit inside the entry's
accuracy gate; the point is that "same inputs, same outputs" holds only
within one tuner state.

## Routing traps

Both from `renorm_moe_routing_op`'s certification:

- **Ties break toward the lower expert index**, the opposite of
  `torch.topk`. On bf16 logits exact ties are common, so a
  `torch.topk`-based reference disagrees on indices while the weights
  still match. Do not validate routing against `torch.topk`.
- **The kernel ignores strides**, reading `router_logits` as a dense
  row-major buffer from `data_ptr()`. A strided view routes silently
  wrong. The wrapper guards it; a caller building logits as a slice of a
  wider buffer must materialize them contiguous.

And from `fused_moe`: expert ids outside the rank's slot range are
**silently dropped**, not clamped and not rejected, so a routing bug
surfaces as a quietly weaker token rather than an error.

## Deriving "is every layer MoE?"

Do not assume uniformity, and do not assume `intermediate_size` is live.
The dense-vs-sparse branch per layer is
`layer_idx not in mlp_only_layers and num_experts > 0 and (layer_idx + 1)
% decoder_sparse_step == 0`. With `decoder_sparse_step: 1` and
`mlp_only_layers: []` every layer is sparse, which makes
`intermediate_size` **dead config** that no layer reads — while
`moe_intermediate_size` is the live one. A checkpoint with a nonzero
`decoder_sparse_step` or a non-empty `mlp_only_layers` needs both branches
and both sets of weights. Assert the derivation at construction rather
than trusting it.

## MXFP4 expert stacks — the same trap, in interleaved form

From the gpt-oss-120b/sm_100/tp1 onboard (36 layers, all MoE, 128
experts, top-4, hidden = intermediate = 2880, experts MXFP4 while router,
attention, embedding and lm_head stay bf16). Everything above about the
`[up | gate]` order still holds; a block-scale-quantized stack adds three
things.

**The parity is inverted relative to the half-split convention.** The
checkpoint stores `gate_up_proj_blocks` as `[E, 2I, K/32, 16]` uint8 —
already in `nn.Linear` `[out, in]` orientation, transposed relative to
HF's `[E, hidden, 2*inter]` parameter — and HF reads `gate =
gate_up[..., ::2]`, `up = gate_up[..., 1::2]`. So the stored row order
along the `2I` axis is (gate, up, gate, up, …), while the trtllm-gen
kernel's interleave wants destination row `2i` = **up** `i`, `2i+1` =
**gate** `i`. The split is therefore `up = t[:, 1::2]`, `gate =
t[:, 0::2]`, re-concatenated `[up ; gate]` before the row permutation.
Measured discrimination on real layer-0 tensors: 1.12 bf16 ulp against a
correct pure-torch reference, 146 ulp against the swapped one — a 130x
separation. Worth running that check *before* the first engine boot: it
covers nibble order, the parity split, the concat, the interleave, the
32-row block shuffle, the 128x4 scale swizzle, both padded axes, both
fp32 biases and the alpha/beta/limit triple in one shot, and the accuracy
gate that would otherwise catch it costs a full evaluation.

**Where the relayout lives decides whether the model fits.** Expressing
pad → concat → interleave → shuffle → swizzle as *manifest source
transforms* (the sharding convention's `src` slot, generalized to any
callable) keeps peak memory at one layer of scratch. Declaring
checkpoint-shaped parameters and transforming them in `post_load_weights`
instead needs both forms resident — 63 GB checkpoint plus 66 GB
kernel-ready, on a 183 GB card that also wants a KV pool. The streaming
form loaded 63 GB including the on-device relayout in 14.1-14.5 s. The
kernel-ready operands at this geometry are `[128, 5888, 1536]` +
`[128, 5888, 96]` + fp32 `[128, 5888]` (FC1) and `[128, 2944, 1472]` +
`[128, 2944, 92]` + fp32 `[128, 2944]` (FC2) per layer, ~1.85 GB/layer,
+4% over the checkpoint.

**Dtypes on disk are not the dtypes the kernels want.** Expert biases and
per-head attention sinks are bf16 in the checkpoint; the trtllm-gen MoE
requires fp32 biases and the attention op an fp32 sink tensor. Promote at
load. The same row permutation applied to the weight bytes must also be
applied to the scale bytes **and** the fp32 bias — a non-permuted bias is
a silent wrong answer.

## W4A16 vs W4A8: the same weights, a different activation path

Both members of the trtllm-gen block-scale MoE family consume the
**byte-identical** prepared expert stack — confirmed from the quant-method
source (both inherit one base, overriding only `create_weights` /
`load_quant_scales`, which call `super()`) and by measurement across every
certified geometry. Moving between them is a forward change only;
`weights.py` does not move.

| | W4A16 | W4A8 |
|---|---|---|
| activations | bf16 straight in | `mxfp8_quantize(x, swizzled_layout=False, alignment=512)` first |
| hidden padding | caller pads 2880 -> 3072 | the quantizer does it — **the pad call disappears** |
| `valid_hidden_size` | 2880 | 2880 (output width; unrelated to the widening, and `None` is rejected) |

Two consequences that are not obvious from the signatures:

- **The W4A8 kernel requantizes the FC1 activation to MXFP8 between the
  GEMMs**, on the OCP scale (`e = floor(log2 amax) - 8`), *not* the
  round-up scale the standalone quantizer uses. Skipping that step in a
  reference deviates 13.5 ulp element-wise / 8.5 ulp RMS (~3% relative)
  from the kernel — so it is a real perturbation of the layer output, and
  it is the mechanism behind the accuracy difference between the two
  members. On gpt-oss-120b the measured cost was 1.44 GSM8K points
  (90.5989 W4A16 -> 89.1585 W4A8), reproducible bit-identically on both
  sides, i.e. the recipe's price rather than sampling.
- **`swizzled_layout=True` is silently accepted** by the W4A8 MoE
  whenever the byte counts coincide (`T % 128 == 0` — exactly the CUDA
  graph batch sizes), and is 260 ulp wrong. No metadata distinguishes the
  two layouts, so no wrapper can guard it. Put the `False` in a named
  constant.

The perf reason to pay that accuracy: on gpt-oss-120b the W4A16 MoE ran
5.6x slower than the W4A8 one on byte-identical weights (FC1 711.8 vs
126.7 µs per layer per step at concurrency 128), which was the target's
*entire* gap to stock trtllm. The swap moved peak throughput +88% and put
the step within 0.5% of the reference.

## The dtype chain, end to end

Worth writing out because the names invite a wrong guess: **W4A8
quantizes activations to fp8, never to fp4.** The op name spells both
operands — `mxe4m3_mxe2m1_...` is e4m3 activations against e2m1 weights.
The `mx` prefix is OCP micro-scaling: 32 elements share one E8M0
(power-of-two) scale, so MXFP8 is 32 e4m3 values plus one scale byte and
MXFP4 is 32 e2m1 values plus one.

| step | format |
|---|---|
| hidden states entering the block | bf16 `[T, H]` |
| `mxfp8_quantize(x, False, 512)` | e4m3 `[T, pad_up(H, 512)]` + E8M0, one byte per 32 |
| FC1 GEMM | e4m3 x e2m1, **fp32 accumulate** |
| clamped GLU (FC1 epilogue) | fp32 |
| requantization of the intermediate | e4m3 + E8M0 per 32 columns |
| FC2 GEMM | e4m3 x e2m1, **fp32 accumulate** |
| routing-weighted combine | fp32 |
| store | **bf16** `[T, valid_hidden_size]` |

Two quantization points, both to fp8, with fp32 everywhere between them.
Only the operand *storage* formats are narrow — the arithmetic never drops
to fp4. The output is bf16 at the model's true hidden width, not the
padded one, and `valid_hidden_size` has to be passed explicitly to get it.

## Why the W4A16 member is slow, and the W4A8 one is not

The two paths read the **same weight bytes**, so the 5.6x is not
bandwidth. It is what each kernel does with them, and the kernel names say
it outright:

| | operand fields in the kernel name | conversion |
|---|---|---|
| W4A16 | `bmm_Bfloat16_MxE2m1Bfloat16_castBfloat16_...` | `castBfloat16` — MXFP4 expanded to bf16, then a bf16 MMA |
| W4A8 | `bmm_MxE4m3_MxE2m1MxE4m3_...` | none — MXFP4 goes straight into a block-scaled MMA |

The expansion's cost is not mainly the conversion arithmetic. A
dequantized weight block occupies **4x the on-chip bytes** (4-bit e2m1 ->
16-bit bf16), so fewer of them fit in registers and shared memory, which
shrinks the tile and shortens the software pipeline — visible in the same
kernel names:

- W4A16: `m128x8x16`, 3 pipeline stages, 1-CTA clusters
- W4A8: `m256x16x32`, 6 stages, 2-CTA clusters

Fewer weight bytes in flight means HBM latency stops being hidden, and the
GEMM lands far below the memory roofline instead of at it. At concurrency
128 on gpt-oss-120b, where the batch touches essentially every expert so a
decode step reads the whole stack once (~61 GB at valid sizes):

| | MoE per step | implied rate |
|---|---|---|
| W4A16 | 38.89 ms | ~1.6 TB/s |
| W4A8 | 8.34 ms | ~7.3 TB/s |
| stock trtllm | 7.51 ms | ~8.1 TB/s |

against a roofline this repo measured at 7.89 TB/s on the same machine. So
the W4A16 member was not paying more bandwidth — it was failing to use the
bandwidth it had, at roughly a fifth of the achievable rate. The residual
W4A8-vs-reference gap is autotuner tactic choice (~11%), not the recipe.

*Measured:* the kernel names with their tile / stage / cluster fields, the
per-GEMM times (FC1 711.8 vs 126.7 µs, FC2 352.9 vs 65.6 µs per layer per
step), the MoE step times, and that the weights are byte-identical.
*Derived:* the byte figure behind the implied rates, and the chain from
on-chip footprint to tile size to unhidden latency — consistent with every
number above, but not isolated experimentally.

## Blackwell only, and it fails quietly elsewhere

This kernel family is sm_100 / sm_103. Three independent signals:

- TensorRT-LLM gates its own tests on it — `get_sm_version() not in (100,
  103)`, reason "TRTLLM Gen MoE supports SM100 and SM103 only".
- The quantization kernel's body compiles only under `__CUDA_ARCH__ >=
  1000`; below that the empty branch **returns without writing the
  outputs**, so a pre-Blackwell run produces garbage rather than failing.
  Both catalog entries carry sm_100-only receipts for that reason — a
  statement about the hardware, not caution.
- The performance signature itself. A software emulation of the mixed
  fp8 x fp4 MMA — unpack, widen, then a wide MMA — would look exactly like
  the W4A16 path measured above, and nothing can be 5.6x faster than the
  thing it emulates.

The hardware feature is Blackwell's block-scaled MMA: the instruction
takes narrow operands plus their per-block scale factors and applies the
scaling inside the tensor-core datapath, so no widened operand is ever
materialized. The E8M0 scale being a power of two is what makes that
nearly free — it is an exponent adjustment. (The arch gating and the
performance signature are measured here; the datapath description is the
standard account of the feature and was not verified at the ISA level.)

## NVFP4 expert stacks — the same traps, plus per-expert scales

From the deepseek-v3-lite-nvfp4/sm_100/tp1 onboard (DeepSeek-V3-Lite: 30
layers, layer 0 dense, layers 1-29 MoE with 72 routed experts at top-6,
`moe_intermediate_size` 1536, plus 2 shared experts fused into one
`[3072, ...]` dense pair; experts NVFP4 while attention, router, embed
and lm_head stay bf16). Everything above about the `[up | gate]` order
still holds. NVFP4 differs from the MXFP4 case in three ways that matter.

**The consumer is a different op with a different operand shape.** The
trtllm-gen NVFP4 MoE takes `[E, 2I, H/16]` **`float8_e4m3fn`** scales,
where the MXFP4 members take `[E, 2I, H/32]` uint8 UE8M0. The prepared
stacks are **not** interchangeable between the two families, even though
both are "block-scale MoE runners".

| operand | shape | dtype |
|---|---|---|
| `gemm1_weights` | `[E, 2I, H/2]` | uint8 (two e2m1 codes per byte) |
| `gemm1_weights_scale` | `[E, 2I, H/16]` | float8_e4m3fn |
| `gemm2_weights` | `[E, H, I/2]` | uint8 |
| `gemm2_weights_scale` | `[E, H, I/16]` | float8_e4m3fn |

Per-expert preparation is: concat `[up ; gate]` (**up rows first**) ->
interleave (dst row `2i` = up `i`, `2i+1` = gate `i`) -> 32-row block
shuffle (src `4u+v` -> dst `8v+u`) applied to weight bytes **and** scale
bytes -> 128x4 swizzle of the scales only.
`torch.ops.trtllm.shuffle_matrix` and
`torch.ops.trtllm.block_scale_interleave` perform the last two; both are
load-time transforms, outside the closed-vocabulary rule.

Measured discrimination against a correct pure-torch reference, on real
checkpoint tensors — the reason to run this check before the first engine
boot rather than after an accuracy gate:

| variant | bf16 ulp |
|---|---|
| **correct** | **4.75** |
| `[gate\|up]` instead of `[up\|gate]` | 263 |
| weight scales not 128x4 swizzled | 408 |
| expert rows not 32-row shuffled | 508 |
| `output1_scale_scalar` / `_gate_scalar` swapped | 177 |
| no intermediate requantization | 42.7 |

**The half-order trap appears twice in one layer on a shared-expert
model.** The dense and shared-expert paths pack `[gate ; up]` for
`flashinfer_silu_and_mul`; the MoE FC1 packs `[up ; gate]`. Both live in
the same decoder layer, three lines apart. The two orders must never be
copied across — each is correct for its own consumer.

### modelopt stores reciprocals

A modelopt NVFP4 checkpoint stores `input_scale` and `weight_scale_2` as
`amax / (448 * 6)` — the **reciprocal** of the global scale a quantizer
divides by. So:

```
global_scale (for fp4_quantize) = 1 / input_scale
dense GEMM alpha                = input_scale * weight_scale_2   # both off disk
```

Reading them the other way is wrong by `global_scale^2`, finite, and
invisible short of an accuracy gate. Settle it by arithmetic rather than
by convention: under the reciprocal reading the implied amax values are
0.2-3.0, physically sensible for normed activations and NVFP4 weights;
under the other reading they are ~1e7. `gate_proj` and `up_proj` share
both scalars exactly (verified across all experts of every MoE layer) —
assert it at load rather than assuming it.

The trtllm-gen runner's three `[E]` fp32 arrays, in checkpoint terms:

```
output1_scale_gate_scalar[e] = input_scale_gate_up * weight_scale_2_gate_up[e]
output1_scale_scalar[e]      = output1_scale_gate_scalar[e] / input_scale_down
output2_scale_scalar[e]      = input_scale_down * weight_scale_2_down[e]
```

The gate scalar dequantizes the half feeding the sigmoid; the other
carries the extra FC2-input global scale because the FC1 epilogue
re-quantizes its output to NVFP4.

### One activation scale, but per-expert weight scales

Two facts that look like they should match and do not.

**The activation global scale `g1` is one number, and the shared expert
carries it.** The runner quantizes the hidden states once, so it takes a
single `g1`. Which one? On this checkpoint the shared expert's
`gate_proj.input_scale` is **exactly** the max over all routed experts'
— exact equality in every MoE layer. That is not a coincidence worth
guessing at: the shared expert sees every token, so its amax *is* the
global amax. `g1 = 1 / shared.gate_proj.input_scale` therefore serves
both the dense and MoE quantization calls. Routed experts' own values sit
up to 1.4-2.3x below it.

**The FC2 input scale `g2` is genuinely per-expert** and does not
collapse. It spans **54.5x** across experts in a single layer. The
runner's three `[E]` arrays exist precisely to carry it; feeding a scalar
would be wrong by that factor on the extreme experts.

### Two quantization calls, not one

`nvfp4_gemm` (dense and shared-expert linears) consumes the **swizzled**
scale-factor buffer; the trtllm-gen MoE runner consumes the **linear**
one, viewed as `float8_e4m3fn`. The same hidden states must therefore be
quantized **twice**, once in each layout — they cannot share a result.

The failure mode is nasty: at `T % 128 == 0` the swizzled buffer happens
to be the *right size* for the runner, so it is silently accepted and is
276 ulp wrong. At every other token count it is rejected on size. A smoke
test at a round batch size will not catch it, and a smoke test at an odd
one will look like a size bug rather than a layout bug.

### Two smaller obligations

- **`hidden` must be a multiple of 256** for the trtllm-gen NVFP4 runner.
  At a hidden that is only a multiple of 128 the kernel reads FC1 weight
  scales for the wrong blocks: no error, 301-3370 bf16 ulp wrong. This is
  an upstream hazard too — trtllm's own weight creation only raises
  alignment to 256 when `hidden_size > 1024`.
- **Per-tensor quantization scalars are stored 0-dim.** `entry[:]` on
  them raises `IndexError: slice() cannot be applied to a 0-dim tensor`,
  so a manifest's materialize step needs a rank check that the bf16 path
  never needed.

## Group-limited routing, and where the grouping lives

From `deepseek-r1-0528-nvfp4/sm_100/dep4` (256 routed experts, top-8,
`n_group 8` / `topk_group 4`), the first target on a grouped checkpoint —
its siblings all run `n_group: 1`.

**The grouping is entirely the routing op's.** `noaux_tc_op` takes
`n_group`/`topk_group` and does the group-limited top-k; the block-scale
MoE runner's own `n_group`/`topk_group` arguments stay **`None`** at
`routing_method_type = 1`, because the runner is on its pre-routed path
and consumes finished `topk_ids`/`topk_weights`. Passing the grouping
twice is not how it composes.

Two constraints the grouped path adds that the ungrouped one does not, and
that the routing op reports as one opaque "unsupported configuration": at
`n_group > 1` it requires `1 <= topk_group <= n_group`, `topk <= 8`,
`num_experts <= 256`, and `num_experts / n_group <= 32`. Assert them
separately or a violation is unattributable.

**Router bias dtype.** The combine weights follow the **logits'** dtype,
not the bias's. A bf16 router GEMM with the fp32 `e_score_correction_bias`
this checkpoint ships returns bf16 weights — which is what the MoE runner
demands (it rejects fp32 `topk_weights`). No cast is needed and adding one
is a mistake.

## A four-way expert-parallel split at 256 experts

The invariant is the one this file already records — the windows must tile
the routing space exactly once — now measured at 64 wide:

* the four 64-wide windows **sum to the whole-layer result** (0.85 ulp RMS
  against an fp32 reference), but are **not bitwise equal** to a single
  256-expert call: up to 2.00 ulp apart, because each window rounds its own
  partial to bf16 before the add. Dropping any one window lands ≥ 125 ulp
  RMS away, so the sum is a real check rather than a formality;
* the **autotuner's key holds `local_num_experts` but not
  `local_expert_offset`** — warming one window of a split warms all four,
  and a 256 → 64 change forces a fresh sweep. Cold and warm agree bitwise
  across 85 configurations here, but do not generalize that: the bf16
  `fused_moe` entry records the same check *failing*.
