---
receipts:
  sm_100: {status: passed, trtllm: 1.3.0rc21}
  sm_103: {status: passed, trtllm: 1.3.0rc26, tests: 19}
---

# nvfp4_gemm

**Wraps** `torch.ops.trtllm.nvfp4_gemm` (one call).

## Semantics

Dense GEMM in nn.Linear orientation over two **block-scaled NVFP4** operands.
Let `A` be the activation and `B` the weight, both stored as packed e2m1
nibbles plus one e4m3 scale per 16 contiguous elements along `K`:

```
a[m, k] = e2m1(act_fp4  nibble (m, k)) * e4m3(act_sf      scale (m, k // 16))
b[n, k] = e2m1(weight   nibble (n, k)) * e4m3(weight_scale scale (n, k // 16))

out[m, n] = output_dtype( alpha * sum_k a[m, k] * b[n, k] + bias[n] )
```

i.e. `out = alpha * (a @ b.T) + bias`. The sum is accumulated in **fp32**;
`alpha` (a device fp32 scalar) multiplies the whole accumulator; `bias` (an
optional per-column vector) is added after `alpha`; the result is cast to
`output_dtype` once, at the end.

Fusion boundary — inside the call: both operands' **block-scale
dequantization**, the fp32 accumulation, the per-tensor `alpha`, the optional
per-column bias, and the output cast. Outside the call, still the caller's:
producing `act_fp4`/`act_sf` (an activation quantizer such as
`torch.ops.trtllm.fp4_quantize`), putting **both** scale buffers into the
swizzled layout below, folding the two per-tensor global scales into `alpha`,
any activation function, any reduction across ranks, and any narrowing of a
padded `N` back to `out_features`.

### The two global scales live in `alpha`

An NVFP4 tensor quantized with a per-tensor global scale `g` reconstructs as
`data * sf / g` — this op applies `data * sf` only. With `g_act` for the
activation and `g_w` for the weight, the caller owes

```
alpha = 1 / (g_act * g_w)
```

For a modelopt/HF checkpoint the two conventions meet like this: the stored
`input_scale` is `amax_act / (448*6)` = `1 / g_act` (the quantizer's
`global_scale` is its reciprocal) and the stored `weight_scale_2` is
`amax_w / (448*6)` = `1 / g_w`, so

```
alpha = input_scale * weight_scale_2          # both as stored on disk
      = weight_scale_2 / g_act                # g_act = the quantizer's global_scale
```

`alpha` is a load-time scalar for weight-only-static checkpoints; nothing in
this op derives it.

### Scale-factor layout (both operands)

Each scale buffer is **1-D** and holds one e4m3 byte per (row, 16-element
block) in the **128x4 swizzled** order: the order the trtllm NVFP4 activation
quantizer emits with `is_sf_swizzled_layout=True`
(`torch.ops.trtllm.fp4_quantize`), and the order
`torch.ops.trtllm.block_scale_interleave` produces from a row-major
`[rows, cols]` tensor — the latter verified byte-identical to the formula below
on this machine. With `cols = K / 16`, the byte for `(row r, block c)` sits at
flat offset

```
(c % 4)
+ (c // 4)   * 512                       # 4 cols x 128 rows per column group
+ (r % 32)   * 16
+ ((r % 128) // 32) * 4
+ (r // 128) * 128 * pad_up(cols, 4)
```

and the buffer's length is `pad_up(rows, 128) * pad_up(cols, 4)` bytes —
`rows = M` for the activation, `rows = N` for the weight. Bytes at offsets no
real `(r, c)` addresses (row padding up to a multiple of 128, column padding up
to a multiple of 4) are **ignored**: filling them with `0x00` or with `0x7E`
(448) gives bitwise-identical output.

**A checkpoint that stores `weight_scale` row-major `[N, K/16]` therefore owes a
load-time relayout.** Feeding the linear buffer instead is accepted and silently
wrong — the observed error was of the same order as the output itself. Both
scale buffers are passed as `uint8`; an fp8 `float8_e4m3fn` view of the same
bytes is rejected.

## Signature

```python
def nvfp4_gemm(
    act_fp4: torch.Tensor,
    weight: torch.Tensor,
    act_sf: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: torch.Tensor,
    output_dtype: torch.dtype,
    output_buffer_kind: int = 0,
    allowed_backends: str = "cutlass,cublaslt,cuda_core",
    group: list[int] | None = None,
    bias: torch.Tensor | None = None,
) -> torch.Tensor
```

| Argument | Shape | Dtype | Layout / value | Device |
|---|---|---|---|---|
| `act_fp4` | `[M, K/2]` (2-D only) | uint8 | contiguous; element `2i` in the low nibble, `2i+1` in the high nibble | CUDA |
| `weight` | `[N, K/2]` (2-D only) | uint8 | contiguous **row-major nn.Linear weight**, not its `.t()` view | CUDA |
| `act_sf` | 1-D, `pad_up(M,128) * pad_up(K/16,4)` | uint8 | contiguous, 128x4 swizzled (above) | CUDA |
| `weight_scale` | 1-D, `pad_up(N,128) * pad_up(K/16,4)` | uint8 | contiguous, 128x4 swizzled (above) | CUDA |
| `alpha` | exactly 1 element (`[1]` or 0-D) | float32 | `1/(g_act * g_w)` | CUDA |
| `output_dtype` | scalar | `torch.dtype` | bfloat16, float16 or float32; the `cutedsl` backend supports bfloat16 only | — |
| `output_buffer_kind` | scalar | Python int | `0` = plain allocation (certified). `1` = userbuffers, `2` = NCCL window — for multi-rank output buffers, not certified here | — |
| `allowed_backends` | scalar | Python str | comma-separated subset of `cutlass`, `cublaslt`, `cuda_core`, `cutedsl`, `marlin`; all compute the same result | — |
| `group` | list of ranks or `None` | Python list | only meaningful with a window output buffer; `None` otherwise | — |
| `bias` | `[N]`, 1-D | **equal to `output_dtype`** | contiguous; added per output column | CUDA |
| returns | `[M, N]` | `output_dtype` | freshly allocated, contiguous | same as `act_fp4` |

`K` is `2 * act_fp4.shape[1]`. Inputs are never written (verified bitwise), and
the returned buffer aliases nothing.

`allowed_backends` is a *selection* set, not a composition: exactly one backend
runs per call and every one of them computes the formula above (all verified
equal to the reference on this machine). The op's own schema spells the
parameters `act_fp4, weight, act_sf, weight_scale, alpha, output_dtype,
output_buffer_kind, allowed_backends, group, bias`.

## Metadata consumed

The process-global AutoTuner profiling cache
(`tensorrt_llm._torch.autotuner.AutoTuner`). Nothing must be prepared:

- Outside a tuning context, a cache miss logs a warning and runs the **fallback
  tactic**: the `cutlass` backend with its default config if `cutlass` is in
  `allowed_backends`, otherwise the first backend listed.
- Inside `tensorrt_llm._torch.autotuner.autotune()` the call profiles every
  valid (backend, tactic) pair and caches the winner per M-bucket; one tuning
  call at rows `M` fills every power-of-2 bucket `1..last_pow2(M)` for that
  `(K, N)` (14 buckets at `M = 8192`, verified at four shapes). Later calls map
  their `M` to its bucket. Observed winners on this machine span `cublaslt` and
  `cutlass` depending on shape; across the 56 cache entries the four R1 shapes
  produce, every winner carried a profiled tactic id rather than the fallback
  marker.

Correctness is certified for both cache states. At the four DeepSeek-R1-0528
shapes below the tuned tactic's output was **bitwise identical** to the cold
fallback's at `M` in {1, 3, 8, 4097, 8192} — a measured property of this op at
those shapes, not a general guarantee that tactic choice is bit-neutral. No
workspace, runtime object or attention metadata is involved; the op is
otherwise stateless.

## Preconditions

Violations marked *silent* were observed on this machine to produce wrong
results without raising; the wrapper rejects each with a metadata assert.
Everything else raises.

- `K % 32 == 0` and `N % 32 == 0` (16-byte operand lines). Both raise otherwise
  — `Expected k to be divisible by 32` / `Expected n to be divisible by 32`.
  `M` is unconstrained (1, 3, 7, 9, 127, 128, 129, 1000, 4097, 8192, ... all
  verified); `M == 0` raises. This rule is the whole shape domain, and it was
  re-measured as such: `K` up to 18432, `N` up to 36864 and `M` up to 8192
  behave exactly like the smaller shapes (see the shape-dependence note below).
- `act_fp4` and `weight` are 2-D, `uint8`, on CUDA, and **contiguous**. A 3-D
  activation raises; a non-uint8 view raises; a CPU tensor raises. Contiguity is
  *silent*: the `cutlass` and `cuda_core` backends raise, but `cublaslt` and
  `cutedsl` ignore the stride and return wrong results, and `cublaslt` is both
  in the default backend list and a frequent tuner pick.
- `weight.shape[1] == act_fp4.shape[1]` (same `K`); mismatch raises.
- `act_sf` and `weight_scale` are `uint8`, **contiguous**, and hold the
  swizzled layout at the full padded size above. Non-contiguity is *silent*
  under `cublaslt`/`cutedsl` (and under `cuda_core` for `weight_scale`); an
  unswizzled (row-major linear) buffer of the right size is *silent* on the
  default path, and no backend has any way to detect it; a buffer shorter than
  the padded size reads out of bounds. Padding bytes may hold anything.
- `alpha` is a float32 CUDA tensor with **exactly one element**. A half or CPU
  tensor raises. More than one element is *silent*: element 0 is used and the
  rest ignored — this build has no per-token alpha.
- `bias`, when given, is 1-D with exactly `N` elements and dtype equal to
  `output_dtype`. Every violation raises.
- `output_dtype` is bfloat16, float16 or float32. The `cutedsl` backend accepts
  bfloat16 only and raises `ValueError` otherwise — forcing
  `allowed_backends="cutedsl"` with fp16/fp32 raises, and so does an
  `autotune()` pass over any list that merely contains `cutedsl` (its runner is
  constructed while enumerating tactics); outside tuning such a list is fine,
  because the fallback never constructs it.
- `allowed_backends` must be a non-empty comma-separated subset of the five
  names; an empty or misspelled string raises `ValueError`. `marlin` is
  Hopper-only and raises on sm_100. `cuda_core` requires SM >= 100 and is only
  ever *selected* by the tuner for `M <= 8`; forcing it computes correctly up to
  `M = 16` and raises `Failed to dispatch cudaCoreGemmLauncher` above that.
- `output_buffer_kind=1` (userbuffers) raises unless a userbuffers workspace of
  sufficient size was allocated by the runtime.
- The op is deterministic: repeating a call with the same inputs and the same
  tuner state gives bitwise-identical output.

## Notes

- Certified on sm_100 (B200) only. The kernels behind every backend are
  Blackwell block-scaled MMA paths (the CUTLASS one is instantiated for
  `cutlass::arch::Sm100`/`Sm103` and traps on other archs); no receipt is
  claimed elsewhere.
- Receipt coverage, shapes: the four DeepSeek-V3-Lite NVFP4 dense linears in
  `[out, in]` orientation — `(K, N)` = (2560, 24576), (12288, 2560),
  (2560, 6144), (3072, 2560) — at `M` in {1, 2, 8, 64, 1024, 4096} (first
  shape) and {1, 8, 64, 1024} (rest); the four DeepSeek-R1-0528 dense linears —
  `(K, N)` = (7168, 36864), (18432, 7168), (7168, 4096), (2048, 7168) — each at
  `M` in {1, 2, 8, 64, 1024, 4096, 8192}; `M` in {1, 3, 7, 9, 127, 128, 129,
  1000} at (2560, 512); `N = 160` (not a multiple of 128, so the weight scale
  buffer pads rows); and a wide block-scale *range* case (scale bytes spanning
  2^-5..2^5, which is what makes the accumulation round) at `K = 12288` and at
  `K = 18432`.
- Receipt coverage, everything else: bf16 / fp16 / fp32 outputs — bf16 at every
  shape, fp16 at (2560, 512), fp32 at (2560, 512) and at both wide-range shapes
  (`K` = 12288 and `K` = 18432); `alpha` in {1, 0.5, 0.03125, 1e-4}; bf16 and
  fp32 bias at (2560, 512) only, the R1 shapes having been driven without bias
  as the dense path uses none; the default `allowed_backends`
  string at every shape above, plus `cutlass`, `cublaslt`, `cuda_core` and
  `cutedsl` each forced alone at (2560, 512) and, at every R1 shape, `cutlass` /
  `cublaslt` / `cutedsl` forced alone at `M` = 8 and `M` = 8192 with
  `cuda_core` alongside them at `M` = 8; the autotuned path at (2560, 6144) and
  at each R1 shape (tuned at `M` = 8192, then replayed warm at `M` in
  {1, 3, 8, 4097, 8192} and compared bitwise against the same calls on a cold
  cache); scale-padding insensitivity; input non-mutation and call-to-call
  determinism; the swizzled-vs-linear weight-scale contrast at (2560, 512) and
  at both extreme R1 shapes; 20 rejected-domain calls; and the five silent
  domains the wrapper guards.
- **Nothing shape-dependent changes past the smaller shapes' maxima.** What is
  shape-dependent is the tactic space, not the arithmetic: the number of
  cuBLASLt heuristic algorithms varies with `(M, K, N)` (1..8 observed), the
  CuTe DSL tile/cluster candidate count varies (120..240 observed), and
  `cuda_core` is offered only for `M <= 8`. Measured at the four R1 shapes
  against the smaller certified ones, every one of those counts landed inside
  the range the smaller shapes already produce, no shape produced an empty
  tactic list, the CUTLASS tactic list is 32 configs everywhere (its
  enumeration takes no shape at all), and the scale-buffer size formula holds
  unchanged at its widest here (`K/16 = 1152` columns, `K = 18432`). At those
  shapes `cutlass`, `cublaslt`, `cutedsl` and — at `M <= 8` — `cuda_core`
  returned **bitwise identical** outputs, each matching the native-torch
  reference, and the warm tuner cache reproduced the cold fallback's bits
  exactly. So the domain rule in Preconditions was sufficient on its own at
  this scale-up: the shapes' only effect was on which tactic wins.
- Numerics. Against a native-torch reference built from the operand bytes, the
  bf16 output matches `torch.testing.assert_close` at **default bf16
  tolerances** (rtol 1.6e-2, atol 1e-5) at every shape above; the observed max
  relative deviation was 3.9e-3, i.e. one bf16 ulp. **That figure is against
  the unrounded fp32 reference** (measured 3.891e-3 vs a 2^-8 = 3.906e-3 ulp).
  The test itself compares against `ref.to(torch.bfloat16)`, so its own
  assertions come out **bit-exact** — an instrumented run that records
  `assert_close` deviations will therefore report zero here, and that is not a
  contradiction of this number. When the block scales sit
  in a narrow band the products and partial sums are exactly representable and
  the **fp32 output is bit-exact** against an fp64 reference. With block scales
  spanning 2^-5..2^5 the fp32 sum does round, and the deviation stays inside
  the recursive-summation bound `K * 2^-24 * sum|a_i b_i|` with ~600x margin —
  which is what an fp32 accumulator looks like and a bf16/fp16 one would not.
  Both regimes were re-measured at `K = 18432`, the largest `K` certified:
  narrow band still bit-exact in fp32, wide band inside the same bound with
  715x margin.
- Discrimination at the largest shapes was measured, not assumed: moving a
  single e4m3 scale byte, or a single e2m1 nibble, by one code — the smallest
  perturbation either operand admits — makes the default-tolerance comparison
  fail at `(K, N, M)` = (18432, 7168, 8192) and (7168, 36864, 8192). Feeding
  the weight scale row-major (unswizzled) instead of swizzled lands ~0.5x the
  output's own magnitude off at those shapes.
- One call is one kernel launch on the certified default path: profiling a
  single call showed exactly one device kernel for `cutlass` (bias fused into
  its `LinCombPerColBias` epilogue) and for `cublaslt` (bias fused, the
  `..._bias_TNT` nvjet variant). Two internal implementation details do not
  change the result but are worth knowing: the `cuda_core` backend first
  un-swizzles `act_sf` with its own kernel (2 launches), and `cutedsl` adds
  `bias` as a post-GEMM elementwise add (2 launches with bias, 1 without).
- `cutedsl` JIT-compiles on first use (seconds) and additionally validates
  `alpha.numel() == 1` itself.
- The op's Python body is registered straight through
  `torch.library.Library.define/impl` (trtllm's `fast_custom_op`), so
  `torch.ops.trtllm.nvfp4_gemm` is an ordinary dispatcher op with a registered
  fake kernel returning `[M, N]` in `output_dtype`. There is no autograd
  kernel — inference only.
- Upstream (`_torch/modules/linear.py`, `NVFP4LinearMethod`) calls this op with
  exactly these operands: `fp4_quantize`'d activations in the **swizzled**
  layout, the raw `[N, K/2]` weight, `module.weight_scale` — which its loader
  fills with `block_scale_interleave(weight_scale)` — and
  `alpha = input_scale * weight_scale_2`. It flattens 3-D activations to
  `[M, K]` before the call and reshapes after, and slices `N` back to
  `out_features` when the weight was padded.
- Sibling ops with the same operand vocabulary exist:
  `torch.ops.trtllm.nvfp4_gemm_cutlass` and
  `torch.ops.trtllm.nvfp4_gemm_cublaslt` (single-backend entry points with
  their own tuning caches), `torch.ops.trtllm.cuda_core_nvfp4_gemm` and
  `torch.ops.trtllm.marlin_nvfp4_gemm` and
  `torch.ops.trtllm.cute_dsl_nvfp4_gemm_blackwell` (the raw per-backend
  kernels; the first two take the *unswizzled* activation scales that this op's
  `cuda_core`/`marlin` paths derive internally, while the CuTe DSL one takes the
  swizzled buffer like this op), `torch.ops.trtllm.fp4_gemm` (the older typed
  entry point, also covering
  W4A8 MXFP4xMXFP8), `torch.ops.trtllm.fp4_gemm_trtllmgen` and
  `torch.ops.trtllm.fp4_bmm`, and `torch.ops.trtllm.nvfp4_gemm_allreduce`
  (this GEMM fused with a tensor-parallel allreduce). This entry covers the
  unified `nvfp4_gemm` launch only.
- Behaviour under CUDA-graph capture was not exercised (unknown); note that
  autotuning and any first-call JIT must complete before capture.
