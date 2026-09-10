---
receipts:
  sm_100: {status: passed, trtllm: 1.3.0rc21}
  sm_103: {status: passed, trtllm: 1.3.0rc26, tests: 16}
---

# fp4_block_scale_moe_runner

**Wraps** `torch.ops.trtllm.fp4_block_scale_moe_runner` (one call).

## Semantics

One complete mixture-of-experts layer for **NVFP4 weights over NVFP4
activations** (E2M1 elements with one E4M3 scale per 16 contiguous K elements
on both operands) — the trtllm-gen "block scale MoE" family's W4A4 member. In
a single call: optional top-k routing over router logits, expert permutation,
the grouped FC1 GEMM, the gated activation, **requantization of that
activation to NVFP4**, the grouped FC2 GEMM, and the routing-weighted combine
back to one row per token.

Let `T = hidden_states.shape[0]`, `H` = the model hidden size,
`I = intermediate_size`, `K = top_k`, `E = local_num_experts`,
`off = local_expert_offset`.

### The three scale scalars, and the two global scales behind them

The kernel's two GEMMs consume raw E2M1 codes and E4M3 block scales; the
**per-tensor global scales** that a modelopt NVFP4 checkpoint carries are not
inside those operands, so they arrive as the three `[E]` fp32 tensors. Write

- `g1` = the FC1 **activation** global scale (`hidden_states` was quantized
  with it),
- `g2` = the FC2 **activation** global scale (the FC1 epilogue quantizes with
  it),
- `gw1[e]`, `gw2[e]` = the per-expert weight global scales of the FC1 and FC2
  weights.

Then, exactly:

```
output1_scale_gate_scalar[e] = 1 / (g1 * gw1[e])          # "alpha" of FC1
output1_scale_scalar[e]      = g2 / (g1 * gw1[e])         # that, times g2
output2_scale_scalar[e]      = 1 / (g2 * gw2[e])          # "alpha" of FC2
```

A modelopt checkpoint stores **reciprocals**: its `input_scale` and
`weight_scale_2` scalars are `amax / (448*6)`, so `g1 = 1 / input_scale_fc1`,
`g2 = 1 / input_scale_fc2` (the `down_proj`'s `input_scale`), and

```
output1_scale_gate_scalar[e] = input_scale_fc1 * weight_scale_2_fc1[e]
output1_scale_scalar[e]      = output1_scale_gate_scalar[e] / input_scale_fc2
output2_scale_scalar[e]      = input_scale_fc2 * weight_scale_2_fc2[e]
```

Getting the two `output1_*` scalars the wrong way round is finite and
plausible — measured 177 bf16 ulp off at `H = 256`, `I = 128` and only **48**
(40 ulp RMS) at the R1 shape `H = 7168`, `I = 2048`, in both cases with no
error. It is the *least* loud of the operand mistakes catalogued here, and it
gets quieter at the larger geometry, so a caller cannot expect it to announce
itself.

### Per token `t` and slot `j < K`

With `A[t]` = the fp32 value of `hidden_states` **after** dividing out `g1`
(i.e. `code * blockscale / g1`), and `W_up[e]`, `W_gate[e]` (`[I, H]`),
`W_down[e]` (`[H, I]`) the fp32 values of this rank's dequantized NVFP4
weights (`code * blockscale`, with the weight global scale `gw` divided out —
it lives in the scalars above):

```
gid = topk_ids[t, j]                        # a GLOBAL expert id
skip this slot unless off <= gid < off + E
e   = gid - off                             # index into this rank's weights

up   = A[t] @ W_up[e].T
gate = A[t] @ W_gate[e].T

# act_type 0 (SwiGlu), the only value certified here:
act  = up * gate * sigmoid(gate)            # fp32

act  = nvfp4_requantize(act)                # see below — the W4A4 step
y    = act @ W_down[e].T
out[t] += topk_weights[t, j] * y            # fp32 accumulation
```

and `out` is stored as bf16. Note the sigmoid is applied to the **gate** half
and `up` is the plain linear one.

**The intermediate is NVFP4.** FC2 is an NVFP4 x NVFP4 GEMM, so the FC1
epilogue quantizes its post-activation output before FC2 reads it. Per token
row and per **16 consecutive intermediate columns** (natural column order,
blocks starting at column 0):

```
sf   = e4m3_round_to_nearest_even(clamp(g2 * max|act| over the block / 6, max 448))
act' = e2m1_round_to_nearest_even(act * g2 / sf) * sf / g2
```

— i.e. exactly what `torch.ops.trtllm.fp4_quantize(act, g2, 16, ...)` would
emit, dequantized. `6` is e2m1's largest magnitude, `448` e4m3's largest
finite value; e2m1 rounding is ties-to-even-code (`0.25 -> 0`, `0.75 -> 1`,
`1.25 -> 1`, `1.75 -> 2`, `2.5 -> 2`, `3.5 -> 4`, `5.0 -> 4`), saturating at
`±6`. This entry's test reads the requantized values straight out of the
kernel (a down projection set to the identity, `output2_scale_scalar = 1`) and
matches the formula above **bit-exactly on all 16384 elements**; on the same
data an unrounded (exact) block scale reproduces 23.5% of the elements and the
unquantized activation 0.0%. Modelling this layer without the
intermediate requantization deviates 43 bf16 ulp element-wise / 23 ulp RMS
from the kernel at `H = 256`, `I = 128`, and 28 / 23 at the R1 shape
`H = 7168`, `I = 2048` — it is a real perturbation of the layer output, not a
rounding detail, and the RMS figure barely moves with geometry. An
intermediate element below `1/24` of its block's largest
magnitude rounds to zero (e2m1's smallest nonzero code is `0.5` against a
block top of `6`).

**FC1 half order is `[up | gate]`.** Before the kernel's row interleave (§
*Preconditions*) the first `I` rows of the FC1 operand are the up projection
(trtllm's `w3`, HF's `up_proj`) and the last `I` rows the gate projection
(trtllm's `w1`, HF's `gate_proj`). Swapping them produces a plausible-looking,
entirely different result; nothing detects it — this entry's test checks that
a gate/up-swapped reference lands 263 ulp away at `H = 256`, `I = 128` and
278 ulp away at the R1 shape `H = 7168`, `I = 2048`.

**Fusion boundary.** Inside the call: routing (when driven by logits),
permutation, both GEMMs with on-the-fly NVFP4 dequantization, the gated
activation, the NVFP4 requantization between the GEMMs, and the weighted
combine. Outside, and the caller's job: the router GEMM that produces the
logits and the routing itself when pre-routing (e.g.
`torch.ops.trtllm.noaux_tc_op`), **quantizing the bf16 hidden states to
NVFP4** (`torch.ops.trtllm.fp4_quantize`, § *Preconditions* — there is no
padding step inside this op), all weight preprocessing, any shared/dense
expert branch, the TP all-reduce or EP gather of this call's output, and the
residual add.

**Nothing is renormalized inside.** `topk_weights` is used exactly as given
(it need not sum to 1). `routed_scaling_factor` is accepted but had no effect
on the certified path.

**Output.** `do_finalize=True` with `output=None` returns a **one-element
list** whose tensor is a fresh contiguous `[T, H]` **bf16** result. With
`output` given the result is written there (every element overwritten, nothing
outside its rows touched) and the list holds an **empty `[0]` bf16 tensor** —
the caller must read its own buffer. `do_finalize=False` returns three
tensors; see § *Signature*. No input is mutated; two identical calls are
bitwise equal.

## Signature

```python
def fp4_block_scale_moe_runner(
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
    output1_scale_scalar: torch.Tensor,
    output1_scale_gate_scalar: torch.Tensor,
    output2_scale_scalar: torch.Tensor,
    num_experts: int,
    top_k: int,
    n_group: Optional[int],
    topk_group: Optional[int],
    intermediate_size: int,
    local_expert_offset: int,
    local_num_experts: int,
    routed_scaling_factor: Optional[float],
    routing_method_type: int,
    do_finalize: bool,
    act_type: int = 0,
    topk_weights: Optional[torch.Tensor] = None,
    topk_ids: Optional[torch.Tensor] = None,
    output: Optional[torch.Tensor] = None,
    tune_max_num_tokens: int = 8192,
    use_dp: bool = False,
) -> list[torch.Tensor]
```

Unlike the MXFP4 members of this family the signature carries **no
`valid_hidden_size` / `valid_intermediate_size`**: this op has no padded-vs-
true size distinction at all (§ *Preconditions*), and it gains the three scale
scalars and `do_finalize`.

**`gemm1_bias`, `gemm1_alpha`, `gemm1_beta`, `gemm1_clamp_limit` and
`gemm2_bias` accept `None`** even though the registered schema spells them
`Tensor`, not `Tensor?`. The schema is generated from the Python custom-op's
annotations; the torch dispatcher turns a `None` argument into an undefined
tensor, the C++ entry point takes `std::optional<at::Tensor>` for exactly
these five (visible in the exported `FP4BlockScaleMoeRunner::run_moe` symbol),
and it arrives there as an empty optional. Verified at runtime: all five at
`None` are bitwise identical to their neutral values (zero bias, `alpha = 1`,
`beta = 0`, an effectively infinite clamp limit), and each slot is live — a
non-neutral value changes the result. A DeepSeek-V3 style MoE, which has no
expert bias and no clamp, passes `None` for all five.

### Certified arguments

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `routing_logits` | `None` — the only certified value; see *Notes* | — | — | — |
| `routing_bias` | `None`, or `[num_experts]` and inert | bf16 | contiguous | CUDA |
| `hidden_states` | `[T, H/2]` | **uint8** (2 E2M1 codes per byte, element `2i` in the low nibble) | contiguous | CUDA |
| `hidden_states_scale` | **1-D**, exactly `T * H/16` elements | **float8_e4m3fn** | contiguous, **linear** (row-major `[T, H/16]` flattened) — *not* swizzled | CUDA |
| `gemm1_weights` | `[E, 2*I, H/2]` | **uint8** | contiguous, pre-shuffled | CUDA |
| `gemm1_weights_scale` | `[E, 2*I, H/16]` | **float8_e4m3fn** | contiguous, pre-shuffled + 128x4 swizzled | CUDA |
| `gemm1_bias` / `gemm1_alpha` / `gemm1_beta` / `gemm1_clamp_limit` | `None` | — | — | — |
| `gemm2_weights` | `[E, H, I/2]` | **uint8** | contiguous, pre-shuffled | CUDA |
| `gemm2_weights_scale` | `[E, H, I/16]` | **float8_e4m3fn** | contiguous, pre-shuffled + 128x4 swizzled | CUDA |
| `gemm2_bias` | `None` | — | — | — |
| `output1_scale_scalar` | `[E]` (**local** experts) | **fp32** | contiguous | CUDA |
| `output1_scale_gate_scalar` | `[E]` | **fp32** | contiguous | CUDA |
| `output2_scale_scalar` | `[E]` | **fp32** | contiguous | CUDA |
| `num_experts` | scalar | int, the global routing space; `> top_k` | — | — |
| `top_k` | scalar | int, `0 < top_k < num_experts` | — | — |
| `n_group` / `topk_group` | `None`, `(1,1)`, or (with `routing_method_type = 2`) `(4,2)` / `(8,4)` — all inert | int | — | — |
| `intermediate_size` | scalar | int, **must equal `gemm1_weights.shape[1] // 2`** | — | — |
| `local_expert_offset` / `local_num_experts` | scalars | int, `E = local_num_experts >= 1` | — | — |
| `routed_scaling_factor` | `None`, `1.0`, `2.5` — all inert | float | — | — |
| `routing_method_type` | scalar | int: `0/1/2/4/5/6` all inert on this path | — | — |
| `do_finalize` | scalar | bool | — | — |
| `act_type` | scalar | int: **0** (SwiGlu) only | — | — |
| `topk_weights` | `[T, top_k]` | **bf16** | contiguous | CUDA |
| `topk_ids` | `[T, top_k]` | **int32** | contiguous | CUDA |
| `output` | `[T, H]` or `None` | bf16 | contiguous | CUDA |
| return, `do_finalize=True` | `[[T, H]]`, or `[[0]]` when `output` is given | bf16 | contiguous, newly allocated | same as `hidden_states` |
| return, `do_finalize=False` | see below | | | |

`torch.ops.trtllm.noaux_tc_op` fed **bf16** router logits produces a directly
consumable `(topk_weights, topk_ids)` pair — bf16 weights and int32 ids, in
that order. Catalog membership of that op is `index.yaml`'s fact alone.

### `do_finalize=False`

Returns three tensors:

0. `[P, H]` bf16 — one row per (token, slot) expert output **before** the
   routing-weighted combine, in the kernel's internal permuted/padded order.
   `P` depends on `T`, `top_k`, `num_experts` and an internal tile size; it is
   always `>= T * top_k`.
1. `[T, top_k]` bf16 — **not written on the pre-routed path.** Observed to
   hold uninitialized memory (values of order `1e24`, and NaN once combined).
   It is the slot for the routing stage's own weights; do not read it.
2. `[T, top_k]` int32 — the expanded-index -> permuted-row map: token `t`'s
   slot `j` expert output is row `out2[t, j]` of output 0.

Recombining with the caller's own weights,
`out[t] = sum_j topk_weights[t, j] * out0[out2[t, j]]` accumulated in fp32 and
stored as bf16, reproduces the `do_finalize=True` result **bitwise** (verified
here). `output=` is rejected in this mode
(`out_tensor is only supported when do_finalize=true`). **A plain, non
expert-parallel target wants `do_finalize=True`**: the mode exists so an
all-to-all deployment can move the per-slot rows before combining them.

## Metadata consumed

None. The op reads no attention metadata, no KV cache, no registered layer and
no module state — every tensor and every scalar it uses is an argument.

Two process-global caches sit behind it, neither of which changed the result
on any path measured here:

- an **autotuner profiling cache** that picks the two GEMM tactics, keyed by
  `(top_k, intermediate_size, local_num_experts, act_type)` together with the
  call's input shapes and a token bucket (the row count rounded down to a
  power of two, capped at 8192). Two consequences of that key were measured
  here, by counting the profiling sweeps the tuner performed rather than
  reading the key: **`local_num_experts` separates entries** — going from 256
  to 64 with everything else fixed forced a fresh sweep, so a cache warmed at
  one expert-parallel window size says nothing about another — while
  **`local_expert_offset` does not**: after warming the offset-0 64-wide
  window, the offset-64 / 128 / 192 windows profiled *nothing*, reusing its
  entries. Warming one window of a split therefore warms all four.

  **Cold is the state every call compared against a reference here ran in** —
  the tuner then returns the fallback tactic. A serving engine is *not* cold:
  the PyTorch runtime profiles this op during warm-up whenever
  `enable_autotuner` is set (its default), so a deployed call runs on a tuned
  tactic instead. Two direct cold-vs-warm comparisons were made, both by
  warming with `tensorrt_llm._torch.autotuner.autotune()` and re-running the
  identical operands. At `H = 2560`, `I = 1536`, top-6, `T = 8192`: **bitwise
  identical** for `local_num_experts = 72` and all four 18-wide windows. At
  the **R1 geometry** (`H = 7168`, `I = 2048`, top-8) the comparison is part
  of this entry's test and is broader: warming wrote **28 cache entries** (14
  token buckets x the two `unique_id`s `(8, 2048, 256, 0)` and
  `(8, 2048, 64, 0)`), every one of them a **non-fallback** tactic, and all
  85 (expert layout, token count) results — `local_num_experts = 256` plus
  the four 64-wide windows, over the full certified `T` column — were
  **bitwise unchanged**. *Which* tactics those are is not reproducible: the
  tuner selects by measured time, so it is the one figure this test prints
  that varies run to run — four runs of the comparison on the same device
  recorded between 12 and 15 distinct tactics across the same 28 entries.
  The bitwise result therefore held across four different tactic assignments,
  which is stronger than one — but it is still a finite sample of tuner picks
  on one machine, not a guarantee across every tactic the tuner might select.
- a cache of C++ `FP4BlockScaleMoERunner` objects keyed by `act_type`, each
  owning a GPU workspace allocated on first use.

## Preconditions

### Sizes

- **`H` must be a multiple of 256.** With the prepared block-scale layout
  below, a hidden size that is only a multiple of 128 makes the kernel read
  the FC1 weight scales for the wrong blocks: no error, 300-3400 bf16 ulp
  wrong. Measured wrong at `H` = 384, 640 and 896; correct at 256, 512, 768,
  2560 and 7168. The wrapper asserts it. (`H` enters the FC1 GEMM's K axis;
  the FC2 K axis, `I`, carries no such rule — every multiple of 64 worked.)
- `I` must be a multiple of **64**, so that `2*I` is a multiple of 128 and
  `I/16` a multiple of 4 — the alignment the 128x4 scale swizzle needs on
  both operands. Certified `I`: 64, 128, 192, 256, 512, 1536, 2048.
- There is **no padded-vs-true size distinction**. `gemm1_weights.shape[-1] *
  2`, `gemm2_weights.shape[1]` and the width of the returned output are all
  the same `H`; a mismatch is rejected (`gemm2_weights_scale has incorrect dim
  1`). `intermediate_size` must equal `gemm1_weights.shape[1] // 2`; anything
  else is rejected (`gemm2_weights_scale has incorrect dim 2`, or `No valid
  config found for the given problem shape`).
- `0 < top_k < num_experts`. `top_k = 0` is rejected (`only supports
  top_k<=32 && top_k>0`), `num_experts == top_k` too (`num_experts must be
  greater than top_k`). Certified `top_k`: 1, 2, 4, 6, 8.
- `num_experts` is only the routing space; it need not equal
  `local_num_experts`. Certified `num_experts`: 2, 3, 4, 8, 16, 72, 256;
  certified `local_num_experts`: 2, 3, 4, 8, 16, 18, 64, 72, 256. Certified
  `(local_expert_offset, local_num_experts)` pairs: `(0, E)` for each of
  those `E`; `(4, 4)` of `num_experts = 8`; the four-way expert-parallel
  split of 72 — `(0, 18)`, `(18, 18)`, `(36, 18)`, `(54, 18)`; and the
  four-way split of 256 — `(0, 64)`, `(64, 64)`, `(128, 64)`, `(192, 64)` —
  see § *Routing and expert ids*.
- `T >= 1`. Certified `T`: 1, 2, 3, 5, 7, 8, 16, 24, 32, 33, 40, 64, 128,
  256, 1024, 4096, 8192. `T` is the **row count of this call** —
  `hidden_states.shape[0]`. It is not `tune_max_num_tokens`, which is an
  autotuner bucket hint whose default is also 8192 and which was measured
  inert (§ *Notes*); a call passing 8192 rows and a call passing 8 rows with
  `tune_max_num_tokens = 8192` are different things, and only the first
  certifies this row count. 8192 is trtllm's default `max_num_tokens`
  (`llm_args.py`), so it is the widest prefill a stock engine hands this op
  in one unchunked call, and it is the top of this list: **a caller whose own
  token count exceeds 8192 must chunk into calls this column covers.** The
  whole column is certified at the **DeepSeek-R1 routed geometry**
  (`H = 7168`, `I = 2048`, `num_experts = 256`, `top_k = 8`, `act_type = 0`,
  `do_finalize = True`, pre-routed), for the full 256-expert stack *and*
  independently for each of the four 64-wide expert-parallel windows. The
  DeepSeek-V3-Lite routed geometry (`H = 2560`, `I = 1536`, top-6) is
  certified over a subset of it topping out at the same 8192 rows, for
  `local_num_experts = 72` at offset 0 and for the four 18-wide windows — see
  § *Routing and expert ids*. Other geometries in the *Certified* lists above
  cover only the smaller counts.

### Activation preparation — NVFP4, linear scales

`hidden_states` / `hidden_states_scale` are exactly the two outputs of

```python
#                                          global  sf_vec  ue8m0  swizzled
data, sf = torch.ops.trtllm.fp4_quantize(x,  g1,     16,   False,  False)
```

on the bf16 `[T, H]` hidden states, with `sf` re-viewed as `float8_e4m3fn`.
Concretely, whatever produces them:

- `hidden_states` is 2-D **uint8** (`hidden_states must be byte` — a
  `float8_e4m3fn` view of the same bytes is rejected), contiguous, CUDA, of
  width exactly `gemm1_weights.shape[-1]`.
- `hidden_states_scale` is **1-D** (`hidden_states_scale must be 1D`),
  **float8_e4m3fn** (`must be fp8` — a uint8 view is rejected), contiguous,
  CUDA, and holds exactly `T * H/16` elements (`hidden_states_scale has
  incorrect size` otherwise). Element `t * (H/16) + b` is the e4m3 scale of
  columns `[16b, 16b+16)` of token `t`.
- The **128x4 swizzled** scale order is a silent wrong answer whenever it
  happens to have the same element count (`T % 128 == 0`, `H/16 % 4 == 0` —
  true at any graph-friendly batch size) — see *Notes*. Pass
  `is_sf_swizzled_layout=False`.
- This op never pads: `H` is used as given, so the quantizer is called on the
  full `[T, H]` hidden with no `F.pad` before it.

### Weight preparation — the caller owns all of it

Start from the checkpoint's per-expert NVFP4 tensors, laid out **row-major
over the output dim**, K packed two E2M1 codes per byte with the **low nibble
holding the even K index**, and one E4M3 byte per 16 consecutive K elements:

```
up   (w3 / up_proj)   packed [E, I, H/2]  uint8   scale [E, I, H/16]  e4m3
gate (w1 / gate_proj) packed [E, I, H/2]  uint8   scale [E, I, H/16]  e4m3
down (w2 / down_proj) packed [E, H, I/2]  uint8   scale [E, H, I/16]  e4m3
```

Then, **per expert**:

1. **FC1 concat.** Stack the two halves on the row axis as `[up ; gate]`,
   giving `[2*I, ...]` for both the packed bytes and the scale bytes.
2. **FC1 row permute** = interleave, then block shuffle:
   - interleave: destination row `2i` takes the up half's row `i`,
     destination row `2i+1` takes the gate half's row `i`;
   - block shuffle (also applied to FC2, which skips the interleave):
     within each aligned block of 32 rows, source row `4u + v`
     (`0 <= u < 8`, `0 <= v < 4`) moves to destination row `8v + u`.

   Apply the **same** permutation to the weight bytes and the scale bytes, so
   row `i` of both still describes the same output channel. Skipping it is a
   silent wrong answer (measured 508 ulp at `H = 256`, `I = 128`; 432 at
   `H = 7168`, `I = 2048`).
3. **Scale swizzle.** After the row permute, each expert's scale matrix
   `[M, C]` (`M % 128 == 0`, `C % 4 == 0`) is rewritten into the trtllm-gen
   128x4 layout: the byte at `(m, c)` moves to flat offset

   ```
   (m // 128) * 512 * (C // 4) + (c // 4) * 512
     + (m % 32) * 16 + ((m % 128) // 32) * 4 + (c % 4)
   ```

   and the flat result is handed to the kernel with its nominal `[E, M, C]`
   shape. Weight bytes are **not** swizzled — only scales. Skipping it is a
   silent wrong answer (measured 408 ulp at `H = 256`, `I = 128`; 357 at
   `H = 7168`, `I = 2048`).

**No padding is needed anywhere at either certified routed geometry**, and the
swizzle's alignment must be checked per geometry rather than assumed — it is a
property of `H` and `I`, not of the family:

| geometry | FC1 weights + scales | FC2 weights + scales | `2I % 128`, `H/16 % 4`, `H % 128`, `I/16 % 4` |
|---|---|---|---|
| `H = 2560`, `I = 1536` | `[E, 3072, 1280]` + `[E, 3072, 160]` | `[E, 2560, 768]` + `[E, 2560, 96]` | 0, 0, 0, 0 |
| `H = 7168`, `I = 2048` | `[E, 4096, 3584]` + `[E, 4096, 448]` | `[E, 7168, 1024]` + `[E, 7168, 128]` | 0, 0, 0, 0 |

At the R1 shapes `torch.ops.trtllm.block_scale_interleave` was checked here to
return exactly `E*M*C` bytes for an `[E, M, C]` scale tensor — measured at
`4096 x 448` and at `7168 x 128` — so the swizzled buffer reshapes straight
back to its nominal `[E, M, C]` shape with nothing appended, and the operand
handed to the kernel is the same size as the one the checkpoint carries.

The two trtllm helpers `torch.ops.trtllm.shuffle_matrix(x, perm)` (a plain row
gather, `out[i] = x[perm[i]]`) and `torch.ops.trtllm.block_scale_interleave(x)`
(the 128x4 swizzle over a `[E, M, C]` uint8 tensor, returning a flat buffer of
`E * pad_up(M, 128) * pad_up(C, 4)` bytes) produce byte-identical results to
steps 2 and 3; this entry's test asserts that equivalence.

### Layout and dtypes

- **Every tensor argument must be contiguous.** A strided view is accepted
  silently and read as if dense — see *Notes*. The wrapper asserts this.
- `topk_ids` is `[T, top_k]` **int32** (`topk_ids must be int` — int64 is
  rejected) and `topk_weights` is `[T, top_k]` **bf16** (`topk_weights must be
  bfloat16` — fp32 and fp16 are both rejected). The two must be given together
  (`routing_logits or (topk_ids and topk_weights) must be provided`).
- Both weight-scale tensors must be **float8_e4m3fn** (a uint8 view is
  rejected with `must be fp8`).
- The three scale scalars must be **fp32** (`must be float`) with exactly
  **`local_num_experts`** elements — not `num_experts` (`has incorrect dim
  0`).
- `output`, when given, must be a contiguous CUDA bf16 tensor of shape exactly
  `[T, H]` (`out_tensor must be bfloat16`, `out_tensor dim0 must match
  num_tokens`). A leading row-slice of a taller contiguous buffer is valid —
  rows past `T` are left bitwise untouched.

### Routing and expert ids

- Expert ids outside `[local_expert_offset, local_expert_offset +
  local_num_experts)` — including negative ids and ids `>= num_experts` — are
  silently dropped; the token's other slots still combine normally. Verified
  two ways: a `local_expert_offset = 4`, `local_num_experts = 4` window of
  `num_experts = 8` against a reference that masks the same slots, and ids of
  `-1`, `num_experts` and `num_experts + 5` all giving the same output as a
  dropped slot.
- **The window is a range test plus an index shift, not a validated
  partition.** Local weight slot `i` answers for global id
  `local_expert_offset + i`; nothing checks the window against
  `num_experts`. `local_expert_offset + local_num_experts > num_experts` is
  **accepted** — measured at `(60, 18)` of `num_experts = 72`, where the
  result matched a reference mapping ids 60..71 onto local slots 0..11 and
  the surplus six local slots were simply never addressed — and a window
  entirely past the routing space (`local_expert_offset = 72`) returns an
  all-zero output. Making the windows tile the routing space is the caller's
  arithmetic, not something this op enforces.
- **The four-way expert-parallel split of 72 experts is certified**:
  `num_experts = 72` on every call (the routing space is not sharded),
  `top_k = 6`, `local_num_experts = 18`, `local_expert_offset` 0 / 18 / 36 /
  54, `H = 2560`, `I = 1536`, `act_type = 0`, `do_finalize = True`,
  pre-routed, at T = 1, 8, 256, 1024, 4096 and 8192. Per window — with
  `gemm1_weights`, `gemm1_weights_scale`, `gemm2_weights`,
  `gemm2_weights_scale` and all three scale scalars sliced to
  `[off, off+18)` along the expert axis and made contiguous — the result
  matches a torch reference that masks every slot outside the window (worst
  9.97 ulp element-wise, 0.74 ulp RMS; at T = 8192 alone, 9.97 and 0.72),
  and every token with **no** slot in the window comes back **bitwise
  zero** — at T = 8192 that is ~1300 rows per window (1265-1406 measured
  across the four windows: top-6 of 72 lands nothing in a given 18-wide
  window for about a sixth of the batch). The four bf16 outputs summed
  reproduce the 72-expert result under the same gate (worst 4.86 ulp
  element-wise, 0.88 ulp RMS against the fp32 reference; at T = 8192 alone,
  4.86 and 0.84); they are **not** bitwise equal to a single 72-expert
  call — measured up to 2.0 ulp element-wise apart at every certified token
  count, 8192 included, because each window rounds its own partial to bf16
  before the add. Leaving any one window out of that sum lands >= 124 ulp
  RMS away (>= 128 at T = 8192). When every routed slot of a batch happens
  to fall inside one window, that window's call **is** bitwise equal to the
  72-expert call and the other three return exactly zero.
- **The four-way expert-parallel split of 256 experts is certified**, the
  DeepSeek-R1 routed layout: `num_experts = 256` on every call (the routing
  space is not sharded), `top_k = 8`, `local_num_experts = 64`,
  `local_expert_offset` 0 / 64 / 128 / 192, `H = 7168`, `I = 2048`,
  `act_type = 0`, `do_finalize = True`, pre-routed, over the **whole**
  certified `T` column (1 through 8192, § *Sizes*). Sliced the same way — all
  four weight/scale operands and all three scale scalars restricted to
  `[off, off+64)` along the expert axis and made contiguous — each window
  matches a torch reference masking every slot outside it (worst 9.62 ulp
  element-wise, 0.74 ulp RMS; at T = 8192 alone, 6.35 and 0.73), and every
  token with **no** slot in the window comes back **bitwise zero** — at
  T = 8192 that is 764-804 rows per window, i.e. top-8 of 256 lands nothing
  in a given 64-wide window for ~10% of the batch (`(3/4)^8`). The four bf16
  outputs summed reproduce the 256-expert result under the same gate (worst
  5.02 ulp element-wise, 0.85 ulp RMS against the fp32 reference; identical
  at T = 8192 alone); as with the 18-wide split they are **not** bitwise
  equal to a single 256-expert call — up to 2.0 ulp element-wise apart at
  every certified token count, because each window rounds its own partial to
  bf16 before the add. Leaving any one window out of that sum lands >= 125
  ulp RMS away (>= 128 at T = 8192); a window left at
  `local_expert_offset = 0`, a window fed the neighbouring window's weights,
  and all four windows issued at offset 0 land 310-363 ulp RMS away.
  When every routed slot of a batch falls inside one window, that
  window's call **is** bitwise equal to the 256-expert call and the other
  three return exactly zero.
- **A repeated expert id inside one token's row is not reliably
  deduplicated.** With one geometry it contributed once (the first slot's
  weight) at 16 tokens and *twice* at 24 tokens. Supply distinct ids per row;
  every routing op in this build does.
- `n_group > 1` requires `routing_method_type = 2` (`Routing kernel with
  groups implies DeepSeekV3 routing method`), but on the pre-routed path the
  whole grouped-routing configuration is inert: `(n_group, topk_group)` of
  `(1,1)`, `(4,2)` and `(8,4)` under `routing_method_type = 2` are all bitwise
  identical to `None/None` under `routing_method_type = 1`.
- `routing_method_type = 3` (Llama4) is rejected for `top_k > 1` (`Current
  routing kernel (no groups, Llama4) only supports top_k=1`).

A caller violating none of the above gets the result described under
*Semantics*, to within the bound in *Notes*.

## Notes

- **Numerics.** Against a native-torch reference that consumes bit-identical
  operands (exact NVFP4 weight and activation dequantization, fp32 GEMM
  accumulation, the NVFP4 requantization of the FC1 output modelled exactly,
  fp32 combine), the kernel's worst element-wise deviation over every
  configuration in this entry's test is **9.97 ulp of the token row's largest
  magnitude** (bf16 ulp = 2^-8; worst case: an 18-wide expert-parallel window
  of 72 experts, 8192 tokens, `H = 2560`, `I = 1536`), against the test's
  16-ulp element gate; a full-window call sits at 4.75 (3.20 at T = 8192).
  Its worst relative RMS deviation over the whole test file is
  **0.88 ulp** (those four windows summed), against a 2-ulp aggregate gate.
  The **R1 routed geometry** (`H = 7168`, `I = 2048`, 256 experts, top-8)
  sits just inside both, over the whole certified token column: 9.62 elt /
  0.74 RMS for a 64-wide window, 4.68 / 0.70 for the 256-expert stack, 5.02 /
  0.85 for the four windows summed.
  The element-wise outliers are single intermediate values landing on opposite
  sides of an e2m1 rounding boundary — an e2m1 step is coarse (2 mantissa
  bits), so one flipped element moves an output row by a visible fraction.
  Two things inflate the element figure and not the RMS one: it is an extreme
  order statistic over every element compared, and it is normalized by the
  reference row's own magnitude, so a window carrying a quarter of the routed
  slots divides a similar absolute error by a smaller row. At `H = 2560`,
  re-drawn over 144 independent (seed, window) comparisons at T = 1024 / 4096
  it reached **12.0 ulp** while the RMS figure stayed at 0.86; re-drawn again
  over 16 fresh seeds at **T = 8192** (16 full-stack + 64 window comparisons)
  it reached **16.33 ulp** for a window (p90 9.82, mean 7.15) and 10.41 for
  the full stack, while the RMS figure was unchanged from T = 4096 (window
  0.74, full 0.70, four windows summed 0.85). So the element metric's upper
  tail *crosses* the 16-ulp gate at the top of the certified token range — it
  is a max over `T * H` elements and therefore grows with `T` at constant
  accuracy, which is a property of the statistic, not of the kernel. At
  `H = 7168` / `I = 2048` the same re-draw (12 fresh weight + activation
  seeds, 12 full-stack + 48 window comparisons per token count) has a
  **lower** tail: max **12.74** for a window at T = 8192 (p90 8.40, mean
  6.73) and 12.15 at T = 4096, 7.11 / 6.42 for the 256-expert stack, with the
  RMS figures flat at 0.74 (window) / 0.70 (full) / 0.86 (sum) at both token
  counts — so the element gate keeps ~1.3x headroom on the *distribution*
  there, not ~1.0x. The shipped test is seeded and deterministic — two
  independent processes on two devices printed byte-identical numbers — and
  its T = 8192 draws land at 9.97 / 3.20 (`H = 2560`) and 6.35 / 4.11
  (`H = 7168`). **The RMS figure is the scale-free one and the one a caller
  must size its own tolerance from**: it is identical at 4096 and 8192 at both
  geometries, and every wrong variant this entry's test constructs sits
  22.5-370 ulp RMS away — 40-370 for every layout, scalar-role or
  expert-window mistake (the 40 is an `output1_*` scalar-role swap at the R1
  shape; the same swap is 124 at `H = 256`), and ~23 for the one variant that
  is not a mistake in the operands at all (a reference that skips the
  FC1-output requantization).
- **Only the pre-routed entry point is certified here.** `routing_logits` was
  left at `None` on every certified call and the routing was done outside (see
  `torch.ops.trtllm.noaux_tc_op` for the DeepSeek-V3 formula). The in-kernel
  routing path — `routing_logits` given, `topk_ids`/`topk_weights` `None`,
  `routing_method_type` selecting softmax / renormalize / DeepSeekV3 — is
  **not certified by this entry**; nor is `routing_bias`, which was only
  observed to be inert on the pre-routed path.
- **A swizzled activation-scale buffer is a silent wrong answer.**
  `fp4_quantize(x, g, 16, False, True)` returns the same data bytes and a
  scale buffer of `pad_up(T,128) * pad_up(H/16,4)` bytes in 128x4 order.
  Whenever that count coincides with the linear one (`T` a multiple of 128 —
  exactly the CUDA-graph batch sizes) the size check passes and the kernel
  reads scales for the wrong blocks: measured 276 ulp element-wise off at
  `T = 128`, `H = 256`. Nothing in the metadata distinguishes the two layouts,
  so no guard can catch it — pass `is_sf_swizzled_layout=False`, in a named
  constant.
- **Non-contiguous tensors are a silent wrong answer.** The kernel takes raw
  data pointers and assumes a dense row-major layout. A strided view of
  `hidden_states`, `topk_weights` or any weight/scale operand is accepted
  without complaint and reads the wrong elements. The wrapper asserts
  contiguity on every tensor argument.
- **`act_type` other than 0 is not certified.** `1` (Relu2) and `2` (Silu) —
  the two non-gated activations, which take an `[E, I, H/2]` FC1 operand
  rather than `[E, 2*I, H/2]` — run without error when handed a gated-shaped
  operand and return something different; so does the undefined value `3`.
  Only `0` (SwiGlu) is characterized here.
- **Inert on the certified path** (bitwise identical results):
  `routing_method_type` `0/1/2/4/5/6`; `n_group`/`topk_group` as above;
  `routed_scaling_factor` `None`/`1.0`/`2.5`; `tune_max_num_tokens` `8192`
  and `128` (an autotuner bucket cap); `use_dp` `False` and `True` (an
  autotuner token-bucket deflation hint); a zero `routing_bias`.
- **Multi-rank tensor parallelism is not exercised.** `local_expert_offset` /
  `local_num_experts` (expert parallelism) **are** certified, including the
  four-way splits of 72 and of 256 experts that a 4-rank EP deployment uses —
  but every certified call ran in **one process on one device**, the four
  windows issued in sequence. This op reads no rank state, holds no
  communicator and
  launches no collective, so a window is an argument pair rather than a
  multi-rank behaviour; summing the per-rank outputs is the caller's
  all-reduce and is outside this entry. TP, which instead splits `I` across
  ranks and reduces this call's output afterwards, is not certified.
- sm_100 only. The receipt covers sm_100 (B200), the only arch available here;
  the installed build carries the assertion `Only SM100f is supported by FP4
  block scale MOE` in `libth_common.so`, so other Blackwell variants are
  expected to raise rather than compute something wrong.
- The registered fake (meta) function disagrees with the kernel on **both**
  dimensions of the `do_finalize=False` first output: it returns
  `[1152, 128]` where the kernel returned `[256, 256]` (`T = 16`,
  `top_k = 2`, `num_experts = 8`, `H = 256`). The rows come from a fixed
  internal tile of 128; the columns are the **packed** width, because the
  helper unpacks only under
  `isinstance(hidden_states, Fp4QuantizedTensor)` and the op's schema hands
  it a plain `Tensor`, so that branch can never be taken. The
  `do_finalize=True` fake path doubles unconditionally and is correct. Eager
  callers are unaffected; this is an upstream defect in this build, not
  behaviour to rely on.
- Sibling ops in this build address the same job with other operand dtypes:
  `torch.ops.trtllm.bf16_mxe2m1_block_scale_moe_runner` (bf16 activations,
  MXFP4 weights), `torch.ops.trtllm.mxe4m3_mxe2m1_block_scale_moe_runner`
  (MXFP8 activations, MXFP4 weights),
  `torch.ops.trtllm.e4m3_mxe2m1_block_scale_moe_runner`,
  `torch.ops.trtllm.fp8_block_scale_moe_runner` and
  `torch.ops.trtllm.fp8_fp4_block_scale_moe_runner` (fp8 activations, NVFP4
  weights). The MXFP4 members take an `[E, 2*I, H/32]` **uint8** UE8M0 scale
  and carry `valid_hidden_size` / `valid_intermediate_size` parameters this op
  does not have, so their prepared weights are **not** interchangeable with
  this one's. Catalog membership is `index.yaml`'s fact alone.
