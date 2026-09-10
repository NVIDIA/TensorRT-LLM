---
receipts:
  sm_100: {status: passed, trtllm: 1.3.0rc21}
  sm_103: {status: passed, trtllm: 1.3.0rc26, tests: 19}
---

# mxe4m3_mxe2m1_block_scale_moe_runner

**Wraps** `torch.ops.trtllm.mxe4m3_mxe2m1_block_scale_moe_runner` (one call).

## Semantics

One complete mixture-of-experts layer for **MXFP4 (E2M1 + per-32 E8M0 block
scale) weights over MXFP8 (E4M3 + per-32 E8M0 block scale) activations** — the
trtllm-gen "block scale MoE" family's W4A8 member. In a single call: optional
top-k routing over router logits, expert permutation, the grouped FC1 GEMM, the
clamped gated activation, **requantization of that activation to MXFP8**, the
grouped FC2 GEMM, and the routing-weighted combine back to one row per token.

Let `T = hidden_states.shape[0]`, `H = valid_hidden_size` (the model's true
hidden size), `I = valid_intermediate_size` (the true per-rank intermediate
size), `K = top_k`, `E = local_num_experts`, `off = local_expert_offset`, and
let `W_up[e]`, `W_gate[e]` (`[I, H]`) and `W_down[e]` (`[H, I]`) be the fp32
values of this rank's dequantized MXFP4 weights (§ *Preconditions* fixes how
they are packed).

The activations arrive **already quantized**: `hidden_states` holds e4m3
elements and `hidden_states_scale` one UE8M0 byte per 32 consecutive columns,
so the value the kernel computes with is

```
hidden[t, k] = hidden_states[t, k].float() * 2^(sf[t, k // 32] - 127)
```

where `sf` is `hidden_states_scale` read as `[T, hidden_width / 32]` row-major
(§ *Preconditions*).

**Routing** — two mutually exclusive entry points, both certified:

- *Routed here*: pass `routing_logits` `[T, num_experts]`, leave
  `topk_weights`/`topk_ids` at `None`. `routing_method_type` selects
  - `0` (Default): `p = softmax_fp32(logits)` over **all** experts, then
    `(w, id) = top_k(p)` — the combine weights are those softmax
    probabilities and do **not** sum to 1;
  - `1` (Renormalize): `(v, id) = top_k(logits)` first, then
    `w = softmax_fp32(v)` over the `K` selected logits — the weights **do**
    sum to 1. This is the gpt-oss / Qwen3 style.
- *Pre-routed*: pass `topk_ids` `[T, K]` int32 and `topk_weights` `[T, K]`
  **bf16**, leave `routing_logits` at `None`. The kernel's routing stage is
  bypassed entirely; `routing_method_type` is still validated but never
  used (values `0/1/2/4/5/6` all give bitwise identical output).

Passing both is not an error: the op announces it uses the pre-routed pair.

**Per token `t` and slot `j < K`:**

```
g = topk_ids[t, j]                          # a GLOBAL expert id
skip this slot unless off <= g < off + E
e = g - off                                 # index into this rank's weights

up   = hidden[t] @ W_up[e].T   (+ gemm1_bias_up[e])
gate = hidden[t] @ W_gate[e].T (+ gemm1_bias_gate[e])

# gemm1_clamp_limit (per expert), applied only when the tensor is given:
gate = min(gate, limit[e])
up   = clamp(up, -limit[e], limit[e])

# act_type 0 (SwiGlu) — the only kernel that exists on this path:
act  = (up + beta[e]) * gate * sigmoid(alpha[e] * gate)
       # alpha defaults to 1.0 when gemm1_alpha is None,
       # beta  defaults to 0.0 when gemm1_beta  is None,
       # so the default is plain SwiGLU: up * silu(gate)

act  = mx_requantize(act)                   # see below — the W4A8 step
y    = act @ W_down[e].T (+ gemm2_bias[e])
out[t] += topk_weights[t, j] * y            # fp32 accumulation
```

and `out` is stored as bf16. The gpt-oss clamped GLU is exactly
`alpha = 1.702`, `beta = 1.0`, `limit = 7.0`.

**The intermediate is MXFP8, on the OCP scale.** FC2 is an MXFP8 x MXFP4 GEMM,
so the FC1 epilogue quantizes its post-activation output before FC2 reads it.
Per token row and per **32 consecutive intermediate columns** (natural column
order, blocks starting at column 0 of the padded intermediate):

```
amax  = max |act| over the 32-element block
e     = floor(log2(amax)) - 8               # E8M0 byte = e + 127; e = -127 if amax == 0
act'  = round_to_nearest_even(clamp(act / 2^e, -448, +448)) * 2^e
```

This is the OCP MX scale — `floor` of the block max's exponent, with the block
max itself **saturating** to `448 * 2^e` whenever its mantissa exceeds 1.75
(18–21% of blocks measured, clipping that one element by up to 12.5%). It is
**not** the round-up scale `torch.ops.trtllm.mxfp8_quantize` applies to
activations, and modelling it as such is wrong for exactly those blocks. An
intermediate element below roughly `2^-18` of its block's largest magnitude
rounds to zero. This entry's test reads the requantized values straight out of
the kernel (a down projection set to the identity) and matches the formula
above bit-exactly on all 16384 + 46080 elements of two geometries
(`H = I = 512` and `H = I = 2880`); on the same data the round-up scale
matches 99.2% of elements and the *un*quantized activation 6%.

**FC1 half order is `[up | gate]`.** Before the kernel's row interleave (§
*Preconditions*) the first `I_pad` rows of the FC1 operand are the up
projection (trtllm's `w3`, HF's `up_proj`) and the last `I_pad` rows the gate
projection (trtllm's `w1`, HF's `gate_proj`). The sigmoid is applied to the
**gate** half. Swapping them produces a plausible-looking, entirely different
result; nothing detects it — this entry's test checks that a gate/up-swapped
reference lands far outside the numerical gate (measured 246 ulp per element).

**Fusion boundary.** Inside the call: routing (when driven by logits),
permutation, both GEMMs with on-the-fly MXFP4/MXFP8 dequantization, the clamped
gated activation, the MXFP8 requantization between the GEMMs, and the weighted
combine. Outside, and the caller's job: the router GEMM that produces
`routing_logits`, **adding the router bias to those logits** (see *Notes* —
`routing_bias` is a no-op here), **quantizing the bf16 hidden states to MXFP8**
(`torch.ops.trtllm.mxfp8_quantize`, § *Preconditions* — the hidden-width
padding happens inside that call, not here), all weight preprocessing, any
shared/dense expert branch, the TP all-reduce or EP gather of this call's
output, and the residual add.

**Nothing is renormalized inside.** `topk_weights` is used exactly as given
(need not sum to 1, may be negative). `routed_scaling_factor` is accepted but
had no effect on any certified path.

**Output.** With `output=None` the call returns a fresh contiguous
`[T, valid_hidden_size]` **bf16** tensor (the output is bf16 regardless of the
fp8 input). With `output` given, the result is written into that buffer (every
element overwritten, nothing outside its rows touched) and the call returns an
**empty `[0]` bf16 tensor** — the caller must read its own buffer. No input is
mutated; two identical calls are bitwise equal.

## Signature

```python
def mxe4m3_mxe2m1_block_scale_moe_runner(
    routing_logits: Optional[torch.Tensor],
    routing_bias: Optional[torch.Tensor],
    hidden_states: torch.Tensor,
    hidden_states_scale: torch.Tensor,
    gemm1_weights: torch.Tensor,
    gemm1_weights_scale: torch.Tensor,
    gemm1_bias: Optional[torch.Tensor],
    gemm1_alpha: Optional[torch.Tensor],
    gemm1_beta: Optional[torch.Tensor],
    gemm1_clamp_limit: Optional[torch.Tensor],
    gemm2_weights: torch.Tensor,
    gemm2_weights_scale: torch.Tensor,
    gemm2_bias: Optional[torch.Tensor],
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    valid_hidden_size: Optional[int],
    valid_intermediate_size: Optional[int],
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    routing_method_type: int,
    act_type: int,
    topk_weights: Optional[torch.Tensor] = None,
    topk_ids: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    tune_max_num_tokens: int = 8192,
    use_dp: bool = False,
) -> torch.Tensor
```

The signature is the bf16-activation sibling op's plus `hidden_states_scale`
in position 4; every other argument means the same thing.

Derived sizes used below (`pad_up(x, a) = ceil(x / a) * a`):

```
I_pad  = pad_up(I, 128)     # FC1 rows per half, FC2 K axis
H1_pad = pad_up(H, 512)     # FC1 K axis == hidden_states width
H2_pad = pad_up(H, 128)     # FC2 rows
```

For gpt-oss-120b (`H = I = 2880`): `I_pad = 2944`, `H1_pad = 3072`,
`H2_pad = 2944`.

### Certified arguments

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `routing_logits` | `[T, num_experts]` or `None` | bf16 or fp32 (bitwise identical results) | contiguous | CUDA |
| `routing_bias` | must be `None` — see *Notes* | — | — | — |
| `hidden_states` | `[T, H1_pad]` | **float8_e4m3fn** | contiguous | CUDA |
| `hidden_states_scale` | **1-D**, exactly `T * H1_pad/32` elements | **uint8** (E8M0) | contiguous, **linear** (row-major `[T, H1_pad/32]` flattened) — *not* swizzled | CUDA |
| `gemm1_weights` | `[E, 2*I_pad, H1_pad/2]` | **uint8** (2 E2M1 codes per byte) | contiguous, pre-shuffled | CUDA |
| `gemm1_weights_scale` | `[E, 2*I_pad, H1_pad/32]` | **uint8** (E8M0) | contiguous, pre-shuffled + swizzled | CUDA |
| `gemm1_bias` | `[E, 2*I_pad]` or `None` | **fp32** | contiguous, pre-shuffled | CUDA |
| `gemm1_alpha` / `gemm1_beta` / `gemm1_clamp_limit` | `[E]` or `None` | **fp32** | contiguous | CUDA |
| `gemm2_weights` | `[E, H2_pad, I_pad/2]` | **uint8** | contiguous, pre-shuffled | CUDA |
| `gemm2_weights_scale` | `[E, H2_pad, I_pad/32]` | **uint8** (E8M0) | contiguous, pre-shuffled + swizzled | CUDA |
| `gemm2_bias` | `[E, H2_pad]` or `None` | **fp32** | contiguous, pre-shuffled | CUDA |
| `num_experts` | scalar | int, the global routing space; `> top_k` | — | — |
| `top_k` | scalar | int, `0 < top_k < num_experts` | — | — |
| `n_group` / `topk_group` | `None` | — | — | — |
| `intermediate_size` | scalar | int, **must equal `I_pad`** | — | — |
| `valid_hidden_size` | scalar (not `None`) | int, `H`; multiple of 32 with `pad_up(H, 128) == H2_pad` | — | — |
| `valid_intermediate_size` | scalar or `None` | int, multiple of 32, `<= I_pad` | — | — |
| `local_expert_offset` / `local_num_experts` | scalars | int, `E = local_num_experts >= 1` | — | — |
| `routed_scaling_factor` | `None` | — | — | — |
| `routing_method_type` | scalar | int: **0** (Default) or **1** (Renormalize) | — | — |
| `act_type` | scalar | int: **0** (SwiGlu) only | — | — |
| `topk_weights` | `[T, top_k]` or `None` | **bf16** | contiguous | CUDA |
| `topk_ids` | `[T, top_k]` or `None` | **int32** | contiguous | CUDA |
| `output` | `[T, valid_hidden_size]` or `None` | bf16 | contiguous | CUDA |
| return | `[T, valid_hidden_size]`, or `[0]` when `output` is given | bf16 | contiguous, newly allocated | same as `hidden_states` |

Biases are independently optional: `gemm1_bias` and `gemm2_bias` may each be
`None`, in any combination.

### Arguments held inert (not certified)

- `routing_bias` — see *Notes*; the wrapper rejects it on every path where it
  was observed to be a no-op.
- `n_group` / `topk_group` — the grouped (DeepSeek-V3 style) routing path,
  reachable only with `routing_method_type = 2`.
- `routing_method_type` values other than `0` and `1`. `4` (RenormalizeNaive)
  and `6` (SigmoidRenorm) do run on the logits entry point here, but their
  weight formulas are not pinned; `2` (DeepSeekV3), `3` (Llama4) and
  `5` (MiniMax2) were not exercised on that entry point at all. On the
  pre-routed entry point every one of `0/1/2/4/5/6` is inert.
- `routed_scaling_factor` — changed nothing on either certified entry point
  (`None`, `1.0` and `2.5` give bitwise identical results); it belongs to the
  grouped-routing path.
- `tune_max_num_tokens` — an autotuner bucket cap; `8192` and `128` give
  bitwise identical results. Left at its default.
- `use_dp` — an autotuner token-bucket deflation hint for data-parallel
  deployments; `True`/`False` give bitwise identical results at `ep_size = 1`.
  Left at `False`.
- Multi-rank tensor parallelism. `local_expert_offset` / `local_num_experts`
  (expert parallelism) **are** certified; TP, which instead splits `I` across
  ranks and reduces this call's output afterwards, is not exercised here.

## Metadata consumed

None. The op reads no attention metadata, no KV cache, no registered layer
and no module state — every tensor and every scalar it uses is an argument.

Two process-global caches sit behind it, neither of which changes the result:

- an **autotuner profiling cache** that picks the two GEMM tactics. Cold —
  the state every call certified here ran in — it takes a fallback tactic;
  tactic choice is a performance knob only.
- a cache of C++ `MxE4m3MxE2m1BlockScaleMoERunner` objects keyed by
  `act_type`, each owning a GPU workspace allocated on first use.

## Preconditions

### Activation preparation — MXFP8, linear scales, padded hidden

`hidden_states` / `hidden_states_scale` are exactly the two outputs of

```python
#                                              swizzled_layout  alignment
data, sf = torch.ops.trtllm.mxfp8_quantize(x,  False,           512)
```

on the bf16 `[T, H]` hidden states, and that pairing is certified: for
`H = 2880` it yields `data [T, 3072] float8_e4m3fn` and a 1-D
`sf [T * 96] uint8`, which is what this op wants. Concretely, whatever
produces them:

- `hidden_states` is 2-D, **float8_e4m3fn** (a uint8 view of the same bytes is
  rejected: `hidden_states must be Float8_e4m3fn`), contiguous, CUDA, with
  width exactly `gemm1_weights.shape[-1] * 2 == H1_pad`. `alignment` must
  therefore be the FC1 K alignment (512 for this weight family); quantizing
  with `alignment=32` leaves the tensor at width `H` and is rejected (`the
  third dimension of weights must be equal to hidden_size`). The op never
  pads the hidden itself — the widening `2880 -> 3072` happens inside
  `mxfp8_quantize`, and the caller does no `F.pad` at all.
- `hidden_states_scale` is **1-D** (`hidden_states_scale must be 1D`),
  **uint8** (`must be UInt8`), contiguous, CUDA, and holds exactly
  `T * H1_pad/32` bytes (`hidden_states_scale has incorrect size` otherwise —
  one byte too few or too many both raise). Byte `t * (H1_pad/32) + b` is the
  E8M0 scale (`value = 2^(byte - 127)`) of columns `[32b, 32b+32)` of token
  `t`. The **128x4 swizzled** scale order is a silent wrong answer, not an
  error, whenever it happens to have the same byte count (`T % 128 == 0`) —
  see *Notes*.
- Columns `[H, H1_pad)` multiply zero-valued padded weights; `mxfp8_quantize`
  writes zero data bytes and zero scale bytes there, but any **finite**
  content is inert (verified with e4m3 `400.0` under a live scale byte). A
  NaN there propagates into the output.
- `T >= 1`; `T = 0` is rejected. Certified `T`: 1, 2, 4, 6, 8, 10, 12, 16, 17,
  24, 32, 128, 256, 1024, 8192.

### Weight preparation — the caller owns all of it

The activation dtype changes none of it: this is the same prepared weight
layout `torch.ops.trtllm.bf16_mxe2m1_block_scale_moe_runner` consumes, and in
this build both ops are fed by the same trtllm weight-loading class
(`MXFP4WeightTRTLLMGenFusedMoEMethod`, which the W4A16 and W4A8-MXFP8 methods
inherit without overriding a single padding, shuffle or swizzle step). One
prepared expert stack therefore serves both.

Start from the checkpoint's per-expert MXFP4 tensors, laid out **row-major
over the output dim**, K packed two E2M1 codes per byte with the **low nibble
holding the even K index**, and one E8M0 byte (`value = 2^(byte - 127)`) per
32 consecutive K elements:

```
up   (w3 / up_proj)   packed [E, I, H/2]  uint8   scale [E, I, H/32]  uint8
gate (w1 / gate_proj) packed [E, I, H/2]  uint8   scale [E, I, H/32]  uint8
down (w2 / down_proj) packed [E, H, I/2]  uint8   scale [E, H, I/32]  uint8
up/gate bias [E, I] , down bias [E, H]  -> both must reach the kernel as fp32
```

(Checkpoints that store gate and up interleaved on one `2I` axis must
de-interleave first.) Then, **per expert**:

1. **Zero-pad.** `up`/`gate`: rows `I -> I_pad`, byte columns
   `H/2 -> H1_pad/2`, scale columns `H/32 -> H1_pad/32`. `down`: rows
   `H -> H2_pad`, byte columns `I/2 -> I_pad/2`, scale columns
   `I/32 -> I_pad/32`. Biases: `I -> I_pad` and `H -> H2_pad`.
2. **FC1 concat.** Stack the padded halves on the row axis as
   `[up ; gate]`, giving `[2*I_pad, ...]`. Same for the bias.
3. **FC1 row permute** = interleave, then block shuffle:
   - interleave: destination row `2i` takes the up half's row `i`,
     destination row `2i+1` takes the gate half's row `i`;
   - block shuffle (also applied to FC2, which skips step 2/3's interleave):
     within each aligned block of 32 rows, source row `4u + v`
     (`0 <= u < 8`, `0 <= v < 4`) moves to destination row `8v + u`.

   Apply the **same** permutation to the weight bytes, the scale bytes and
   the fp32 bias, so row `i` of all three still describe the same output
   channel. A bias not permuted with its weights is silently wrong.
4. **Scale swizzle.** After the row permute, each expert's scale matrix
   `[M, C]` (`M % 128 == 0`, `C % 4 == 0`) is rewritten into the trtllm-gen
   128x4 layout: the byte at `(m, c)` moves to flat offset

   ```
   (m // 128) * 512 * (C // 4) + (c // 4) * 512
     + (m % 32) * 16 + ((m % 128) // 32) * 4 + (c % 4)
   ```

   and the flat result is handed to the kernel with its nominal `[E, M, C]`
   shape. Weight bytes and biases are **not** swizzled — only scales.

The two trtllm helpers `torch.ops.trtllm.shuffle_matrix(x, perm)` (a plain
row gather, `out[i] = x[perm[i]]`) and
`torch.ops.trtllm.block_scale_interleave(x)` (the 128x4 swizzle over a
`[E, M, C]` uint8 tensor, returning a flat buffer of
`E * pad_up(M, 128) * pad_up(C, 4)` bytes) produce byte-identical results to
steps 3 and 4; this entry's test asserts that equivalence.

### Shapes and layout

- `intermediate_size` must equal `gemm1_weights.shape[1] // 2`; anything else
  is rejected (`No valid config found for the given problem shape`).
- `valid_hidden_size` is the **output width**, unrelated to the widened
  activation: it stays at the model's true hidden `H` (2880 for gpt-oss) even
  though `hidden_states` is `H1_pad` (3072) wide. It must be a multiple of 32
  satisfying `pad_up(valid_hidden_size, 128) == gemm2_weights.shape[1]`.
  `None` means "use the `hidden_states` width", which now fails whenever
  `H1_pad != H2_pad` (`gemm2_weights_scale has incorrect dim 1`) — so on this
  op it must always be passed explicitly. Setting it to `H2_pad` yields the
  wider output whose tail columns are all zero; other values are rejected
  (`gemm2_weights_scale has incorrect dim 1` or `No valid config found`).
- `valid_intermediate_size` must be a multiple of 32 and at most
  `intermediate_size`; `None` means `intermediate_size`. It is a bandwidth
  hint: the kernel only reads the first `valid_intermediate_size` intermediate
  columns. Since padded columns carry zero weight, every value `>= I` gives
  the same result — but a **smaller** value silently truncates the layer.
- `topk_ids` is `[T, top_k]` **int32** contiguous CUDA and `topk_weights` is
  `[T, top_k]` **bf16** contiguous CUDA; the two must be given together
  (`routing_logits or (topk_ids and topk_weights) must be provided`), their
  row count must match `hidden_states` and their column count must match
  `top_k`. int64 ids and fp32 weights are both rejected.
- `routing_logits` is `[T, num_experts]`, bf16 or fp32, contiguous CUDA. A
  column count other than `num_experts` is rejected.
- `gemm1_alpha` / `gemm1_beta` / `gemm1_clamp_limit`, when given, are fp32
  CUDA tensors with exactly **`local_num_experts`** elements (not
  `num_experts`), indexed by local expert.
- Both bias tensors, when given, must be **fp32** (bf16 is rejected); both
  weight-scale tensors must be **uint8** (an int8 view is rejected).
- `output`, when given, must be a contiguous CUDA bf16 tensor of shape
  exactly `[T, valid_hidden_size]`; wrong dtype and wrong shape are both
  rejected. A leading row-slice of a taller contiguous buffer is valid — rows
  past `T` are left bitwise untouched.
- **Every tensor argument must be contiguous.** A strided view is accepted
  silently and read as if dense — see *Notes*. The wrapper asserts this.

### Sizes and counts

- `0 < top_k < num_experts` (`num_experts must be greater than top_k`;
  `top_k = 0` is rejected). Certified `top_k`: 1, 2, 3, 4.
- `num_experts` is only the routing space; it need not equal
  `local_num_experts` (certified at `num_experts = 8`,
  `local_num_experts = 4`, `local_expert_offset = 4`). Certified
  `num_experts`: 2, 3, 4, 5, 8, 16, 128; certified `local_num_experts`:
  2, 3, 4, 5, 8, 16, 128.
- Certified geometries `(H, I)`: (512, 128), (512, 256), (512, 512),
  (640, 128), (1024, 512), (2048, 512), (2880, 1024), (2880, 2880). `H` and
  `I` need not be multiples of the kernel's alignments — that is what the
  padding is for — but both must be multiples of 32, since they are passed as
  `valid_hidden_size` / `valid_intermediate_size` (a `valid_hidden_size` of
  500 was rejected with `No valid config found`).
- `local_expert_offset + local_num_experts > num_experts` is **not** checked.
- Expert ids outside `[local_expert_offset, local_expert_offset +
  local_num_experts)` — including negative ids and ids `>= num_experts` — are
  silently dropped; the token's other slots still combine normally.
- A repeated expert id inside one token's row contributes **once**, with the
  weight of its **first** occurrence; the duplicate slot's weight is
  discarded.

### Dtypes and modes

- `hidden_states` must be float8_e4m3fn; bf16 is rejected
  (`hidden_states must be Float8_e4m3fn`). bf16 hidden states belong to the
  sibling op `torch.ops.trtllm.bf16_mxe2m1_block_scale_moe_runner`, whose
  signature has no `hidden_states_scale` parameter at all.
- `act_type` must be `0`. `1` (Relu2) and `2` (Silu) fail with
  `No kernel found for the given options: mDtypeA: MxE4m3, mDtypeB: MxE2m1
  ...` — the non-gated activations have no cubin in this MXFP8 x MXFP4 family.
  `SwigluBias` is not a separate value: the per-expert `alpha`/`beta`/
  `clamp_limit` tensors turn `act_type = 0` into it.
- `hidden_states` must be 2-D; a 3-D `[1, T, H1_pad]` view is rejected.
- sm_100 only. The receipt covers sm_100 (B200), the only arch available
  here; the installed build carries the assertion `Only SM100f is supported
  by MXFP4 block scale MOE` in this op's C++ entry point, so other Blackwell
  variants are expected to raise rather than compute something wrong.

A caller violating none of the above gets the result described under
*Semantics*, to within the bound in *Notes*.

## Notes

- **Numerics.** Against a native-torch reference that consumes bit-identical
  operands (exact MXFP4 weight and MXFP8 activation dequantization, fp32 GEMM
  accumulation, the MXFP8 requantization of the FC1 output modelled exactly,
  fp32 combine), the kernel's worst element-wise deviation over every
  configuration in this entry's test was **2.0 ulp of the token row's largest
  magnitude** (bf16 ulp = 2^-8, worst case: 128 experts, 8192 tokens,
  `H = I = 2880`) and its worst relative RMS deviation **0.87 ulp**. Dropping
  the intermediate requantization from that reference — i.e. modelling this
  layer the way a bf16-intermediate MoE would be modelled — costs a factor of
  seven: 13.5 ulp element-wise and 8.5 ulp RMS (~3% relative) at the gpt-oss
  geometry. That gap is a property of the kernel, not of the test: it is what
  the extra e4m3 rounding between the two GEMMs does to the layer's output.
- **`routing_bias` is silently ignored** for `routing_method_type` 0, 1, 4
  and 6, and on the pre-routed entry point for every routing method: a bias
  of `+1e3 / -1e3` on two experts leaves the output bitwise unchanged.
  A checkpoint whose router carries a bias (gpt-oss does) must therefore have
  it **added into `routing_logits`** before the call — that is, use the
  router `nn.Linear`'s own bias and pass `routing_bias=None`. The wrapper
  asserts `routing_bias is None` on exactly those paths, because the failure
  is otherwise invisible: expert selection quietly ignores the bias and the
  model degrades without any error. The argument is presumably live for the
  grouped (`routing_method_type = 2`) path, which is not certified here.
- **A swizzled activation-scale buffer is a silent wrong answer.**
  `mxfp8_quantize(x, True, 512)` returns the same data bytes and a scale
  buffer of `pad_up(T,128) * pad_up(cols,4)` bytes in 128x4 order. Whenever
  that count coincides with the linear one (`T` a multiple of 128, `cols` a
  multiple of 4 — true for gpt-oss at any graph-friendly batch size) the size
  check passes and the kernel reads scales for the wrong blocks: measured
  260 ulp element-wise off at `T = 128`, `H = I = 2880`. Nothing in the
  metadata distinguishes the two layouts, so no guard can catch it — pass
  `swizzled_layout=False`.
- **Non-contiguous tensors are a silent wrong answer.** The kernel takes raw
  data pointers and assumes a dense row-major layout. A strided view of
  `hidden_states` (measured 479 ulp off), of `hidden_states_scale` (311 ulp,
  using a same-length stride-2 1-D view), of any of the six
  weight/scale/bias tensors, of `routing_logits`, `topk_weights`, `topk_ids`
  or `output` is accepted without complaint and reads (or writes) the wrong
  elements. The wrapper asserts contiguity on every tensor argument.
- **The FC1 accumulation precision is not pinned by this entry**, and cannot
  be: the epilogue's e4m3 requantization (3 mantissa bits) erases any
  difference between an fp32 and a bf16 intermediate accumulator. What *is*
  pinned is the value FC2 consumes — bit-exactly, per the formula in
  *Semantics*.
- Sibling ops in this build address the same job with other operand dtypes:
  `torch.ops.trtllm.bf16_mxe2m1_block_scale_moe_runner` (bf16 activations,
  same prepared weight layout, no `hidden_states_scale` parameter),
  `torch.ops.trtllm.fp4_block_scale_moe_runner` (NVFP4) and
  `torch.ops.trtllm.fp8_block_scale_moe_runner`. Catalog membership is
  `index.yaml`'s fact alone.


## The FC1 epilogue's block-scale recipe is architecture-specific

trtllm-gen ships one cubin per architecture, and the two differ **bit-exactly**
in how the FC1 epilogue picks the e8m0 scale when it requantizes its activation
output to MXFP8 for FC2. Measured with an identity down-projection, which reads
the intermediate out element by element rather than inferring it from output
noise (`test_intermediate_is_mxfp8_quantized`):

| Arch | e8m0 exponent | Name |
|---|---|---|
| sm_100 | `floor(log2(amax)) - 8` | OCP scale |
| sm_103 | `ceil(log2(amax / 448))` | round-up scale |

Both were verified bit-exact on their own architecture (0 mismatched elements)
and each *refutes* the other's, so the cases genuinely separate the recipes.
The round-up form is what `torch.ops.trtllm.mxfp8_quantize` has always used, so
sm_103 brings the MoE epilogue into agreement with the standalone quantizer.

This is the whole of the sm_100 -> sm_103 numerical difference for this entry.
Before the reference was made architecture-aware, 10 of 19 cells failed --
relative RMS ~5.5 ulp against a 4 ulp gate, max abs ~0.07 against 0.031. With
the correct recipe in force all 19 pass **at the original tolerances**, which
is what identifies the recipe as the sole cause rather than one contributor.

A future cubin that changes recipe again will fail the bit-exact test rather
than drift quietly; record the new recipe in `_SCALE_RECIPE_BY_SM`, and do not
widen a tolerance instead.
