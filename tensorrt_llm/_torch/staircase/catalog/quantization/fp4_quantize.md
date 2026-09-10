---
receipts:
  sm_100: {status: passed, trtllm: 1.3.0rc21}
  sm_103: {status: passed, trtllm: 1.3.0rc26, tests: 21}
---

# fp4_quantize

**Wraps** `torch.ops.trtllm.fp4_quantize` (one call).

## Semantics

Block-scaled FP4 quantization of an activation tensor: one call converts
values to 4-bit **e2m1** elements (two packed per output byte) plus one
scale byte per `sf_vec_size` **contiguous elements along the last dim**.
Nothing else is fused in — no normalization, no activation, no transpose,
and (unlike `mxfp8_quantize`) **no width padding**: the last dim is used
exactly as given, so any padding a consumer needs is the caller's own
`pad` before the call. The caller owns producing the input and owns both
outputs afterwards.

The op has two modes, selected by `sf_use_ue8m0`:

| Mode | `sf_vec_size` | `sf_use_ue8m0` | `global_scale` | Scale byte |
|---|---|---|---|---|
| **NVFP4** | 16 | `False` | required | e4m3 (UE4M3 in practice) |
| MXFP4 | 32 | `True` | optional | UE8M0 power of two |

Any other combination raises. **Only the NVFP4 mode is certified here**;
what follows describes it.

Let `M` = product of all leading dims, `K` = last dim, `cols = K / 16`.
For row `m` and block `b` of 16 elements, with `g` = the `global_scale`
value:

```
vecmax     = max |x[m, 16b : 16b+16]|          # exact in the input dtype
sf[m, b]   = e4m3_round_to_nearest_even( g * vecmax / 6 )   # saturates at 448
out_scale  = g / float(sf[m, b])               # 0 when vecmax == 0
data[m, i] = e2m1_round_to_nearest_even( x[m, i] * out_scale )   # saturates at +-6
```

`6` is e2m1's largest magnitude and `448` is e4m3's largest finite value,
so the canonical `g = 448 * 6 / amax(x)` keeps every block's `sf` inside
e4m3's finite range.

**Mind the direction.** A modelopt NVFP4 checkpoint stores its per-tensor
`input_scale` as `amax / (448*6)` — the **reciprocal** of this `g` — so a
caller passes `1 / input_scale` here, never `input_scale`. Passing it
straight is accepted and silently destructive: measured on `64 x 7168`
bf16 with `amax = 5.1875`, it drove **96.4 % of the scale bytes to zero**
(those blocks read back as all-zero, the `g`-underflow path in
*Preconditions*) while the surviving blocks saturated — max reconstructed
magnitude 6.07 against a true 5.1875. The result is neither an error nor
an obviously dead tensor.

**Dequantization is exactly `data * sf / g`.** A consumer GEMM folds
`1/g` into its own `alpha` (typically `alpha = (amax_input / (448*6)) *
weight_scale_2`); this op emits `data` and `sf` only, and never `g` or
`1/g`.

e2m1 encodes 8 magnitudes — `0, 0.5, 1, 1.5, 2, 3, 4, 6` — as
`code = (exponent << 1) | mantissa`, with bit 3 the sign; `-0.0` keeps
its sign bit (code 8). Rounding is **round-to-nearest, ties to even
code**, saturating: `0.25 -> 0`, `0.75 -> 1.0`, `1.25 -> 1.0`,
`1.75 -> 2.0`, `2.5 -> 2.0`, `3.5 -> 4.0`, `5.0 -> 4.0`, and anything
above `6` clamps to `6`.

Because `sf` has only a 3-bit mantissa, `sf` may round *below* `g*vecmax/6`,
which makes `out_scale` slightly larger than `6/vecmax` and lets the
block's largest element clip to `6`. That is the format, not an error: it
happened in about half the blocks of random bf16 input, and the
reconstruction still obeys the bound below.

**Tie non-determinism (the one place this op is not bit-reproducible from
a plain fp32 model).** The kernel forms `out_scale` through two
`rcp.approx.ftz.f32` reciprocals (~2^-23 relative error each) rather than
an exact division. An element whose exact `x * out_scale` sits within a
few fp32 ulps of an e2m1 midpoint (`0.25, 0.75, 1.25, 1.75, 2.5, 3.5,
5.0`) can therefore land on either neighbouring code. Every other element
matches the formula above bit for bit, and the scale bytes are always
exact. The op itself is deterministic: repeating the same call gives the
same bytes.

**What sets how many such elements there are: the global scale, not the
shape.** A near-tie needs the block's `sf` to come out *exactly*
`g * vecmax / 6` — then `out_scale` is exactly `6 / vecmax` and
`6 * x / vecmax` is an exact small rational, which lands on a midpoint
for a sizeable minority of the block's lanes. Measured on this machine
(bf16, `g = 448*6/amax`, the canonical convention):

- **the width is inert.** One 16,515,072-element sample reshaped to
  `K` = 7168, 18432 and 2048 gave the *identical* counts — 34,405
  near-tie elements and 12,091 kernel/reference disagreements in all
  three. Scale blocks are 16 *contiguous* elements, so a reshape never
  moves a block boundary; `M` and `K` reach the arithmetic only through
  `amax`, hence through `g`.
- **`g` is the whole story.** With that `g`, 0.68 % of blocks had an
  exactly representable `sf`. Multiplying `g` by 1.001 left **no** block
  with one, and the same call then matched the reference **bit for bit**,
  zero near-ties — while multiplying by 2 (a power of two, so `sf` scales
  exactly and `out_scale` is unchanged) kept the population at 33,113.
- across the 24 bf16 shapes of the R1 sweep, near-ties were **0.03 % to
  0.60 %** of all elements and the kernel disagreed with an exact-fp32
  reference on **0.00 % to 0.14 %** — always by one adjacent e2m1 code
  with the sign unchanged.

So the population is a property of the *exactly calibrated* case. A
serving call passes a static per-tensor `input_scale` rather than
`448*6/amax` of the tensor in hand, which is the regime measured above as
bit-exact — but that was one `g` offset (0.1 %), not a proof for every
static scale, and no run here used a real checkpoint's scale.

### Output layouts

`data` is **identical byte for byte under both `is_sf_swizzled_layout`
values** — only the scale buffer's size and element order change.

- `is_sf_swizzled_layout=False` (**linear**): `M * cols` bytes, row-major.
  Byte `b` of row `m` is at flat offset `m * cols + b`; `sf.view(M, cols)`
  is exactly the per-block scale matrix.
- `is_sf_swizzled_layout=True` (**128x4 swizzled**):
  `pad_up(M, 128) * pad_up(cols, 4)` bytes. The scale of `(m, b)` sits at
  flat offset

  ```
  (b % 4)
  + (b // 4)   * 512                      # 4 cols x 128 rows per column group
  + (m % 32)   * 16
  + ((m % 128) // 32) * 4
  + (m // 128) * 128 * pad_up(cols, 4)
  ```

  Every offset not addressed by a real `(m, b)` pair — row padding up to
  `pad_up(M,128)`, column padding up to `pad_up(cols,4)` — is `0x00`.

## Signature

```python
def fp4_quantize(
    input: torch.Tensor,
    global_scale: Optional[torch.Tensor],
    sf_vec_size: int,
    sf_use_ue8m0: bool = False,
    is_sf_swizzled_layout: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]
```

| Argument | Shape | Dtype | Layout / value | Device |
|---|---|---|---|---|
| `input` | `[..., K]`, rank >= 2 | bfloat16 or float16 (e4m3 accepted, uncertified) | contiguous | CUDA |
| `global_scale` | exactly 1 element (`[1]` or 0-D) | float32 | positive, finite; `448*6/amax` by convention | CUDA |
| `sf_vec_size` | scalar | Python int | `16` (NVFP4). `32` only with `sf_use_ue8m0=True` | — |
| `sf_use_ue8m0` | scalar | Python bool | `False` for NVFP4 e4m3 scales | — |
| `is_sf_swizzled_layout` | scalar | Python bool | `True` = 128x4 swizzled scale order, `False` = linear | — |
| returns `data` | `[..., K/2]` (leading dims preserved) | uint8 | contiguous, freshly allocated; element `2i` in the low nibble, `2i+1` in the high nibble | same as `input` |
| returns `sf` | 1-D: `[M*cols]` (linear) or `[pad_up(M,128)*pad_up(cols,4)]` (swizzled) | uint8 | contiguous, freshly allocated; reinterpret as `float8_e4m3fn` to read the values | same as `input` |

`M` = product of `input`'s leading dims, `cols = K / sf_vec_size`.
`input` is not modified. Leading dims are collapsed for the scale buffer
in both layouts (a `[B, S, K]` input yields a scale buffer laid out for
`B*S` rows), while `data` keeps `input`'s leading dims.

The op's own schema spells the parameters `globalScale`, `sfVecSize`,
`sfUseUE8M0`, `isSfSwizzledLayout`; all five are positional here.

## Metadata consumed

None. Stateless — no runtime, attention metadata, or workspace.

## Preconditions

- `input` is on CUDA, contiguous, rank >= 2, dtype bfloat16 / float16 /
  float8_e4m3fn. Violations raise: a CPU tensor, a non-contiguous view
  (column slice or transpose), a 1-D tensor and an fp32 tensor were each
  observed to raise on this machine rather than compute something wrong.
- `K % sf_vec_size == 0`. Raises otherwise. This is the whole width
  domain, and it was re-measured as such: `K` = 2048, 7168 and 18432 —
  the last 1.5x wider than any previously certified width — behave
  exactly like 2560 / 3072 / 12288, at every row count from 1 to 8192.
  Nothing about the op is a function of `K` beyond `cols = K / 16` (see
  *Notes*: one kernel, no width-keyed dispatch).
- `sf_vec_size == 16` with `sf_use_ue8m0=False`, or `sf_vec_size == 32`
  with `sf_use_ue8m0=True`. Every other pairing (including
  `sf_vec_size=8` or `0`) raises.
- `global_scale` is required when `sf_use_ue8m0=False` (passing `None`
  raises), must be a **float32 CUDA** tensor (a half tensor and a CPU
  tensor each raise), and must hold **exactly one element**. A tensor
  with more elements is *not* rejected by the op: it silently reads
  element 0 and applies it to every row — this build has no per-token
  global scale. The wrapper asserts the element count for that reason.
- `global_scale` must be **positive and finite**:
  - `g = 0` yields an all-zero scale buffer and every data lane at `+6`;
    a consumer dividing by `g` gets NaN.
  - `g < 0` yields **negative** e4m3 scale bytes (e.g. `0xB0`), which
    every consumer reads as unsigned UE4M3 — silently wrong.
- Every block must satisfy `2^-10 < g * vecmax / 6 <= 448` (or
  `vecmax == 0`). Both ends fail silently:
  - **Above 448** (a stale/mis-calibrated static `input_scale`, or a
    runtime `amax` above the calibration one): `sf` clamps to `0x7E`
    (448) and `out_scale` collapses to `g/448`. Only lanes above
    `5 * 448 / g` then clip to `+-6`; the rest quantize normally against
    the collapsed scale and still dequantize close to their true value —
    at 1.17x overshoot a hand-built block came back
    `[6, 4, 2, 1, 0.5, 0, 0, 0]`, not all-`+-6` (measured 2026-07-28).
    **Do not look for an all-`+-6`, signs-only block as the signature**:
    a mis-calibrated block looks ordinary, and only ~100x overshoot
    produces the saturated form.
  - **At or below `2^-10`** (`g` far too small): `sf` rounds to `0x00`,
    the reciprocal of that zero scale is `+inf`, so every lane saturates
    to `+6` (exact zeros included, via `0 * inf = NaN` which the e2m1
    cast saturates). Because `sf` is zero, a consumer reads the whole
    block back as zeros. One binade higher (`g*vecmax/6 = 2^-9`, the
    e4m3 minimum subnormal) is already the normal, exact path.
  - An all-zero block is fine: scale byte `0x00`, all-zero data bytes.
- `input` must be **finite**. Non-finite lanes are accepted silently and
  are destructive:
  - a `+inf`/`-inf` lane drives the block max to infinity, so the scale
    saturates at 448 and every other lane in the block is divided by 448
    — small lanes round to zero; the inf lane itself becomes `+6`;
  - a NaN lane is excluded from the block max, so the scale stays correct
    and the NaN lane saturates to `+6` without disturbing its neighbours.
- `M == 0` (zero rows) is accepted and returns empty `data` and `sf`.
- The caller owns both returned buffers; nothing else writes them.

## Notes

- Certified on sm_100 (B200) only, and only for the **NVFP4** mode
  (`sf_vec_size=16`, `sf_use_ue8m0=False`) with bfloat16 and float16
  inputs. The kernel behind this op is Blackwell-targeted, and the
  installed binary shows it directly: the shipped library carries cubins
  of this kernel for sm_80, sm_86, sm_89, sm_90a, sm_100f and sm_120f,
  but only the sm_100f and sm_120f ones have a body (32 and 36 registers,
  1024 B shared). The sm_80/86/89 cubins are **4 registers and no shared
  memory** and the sm_90a one 4 registers — i.e. the conversion is
  compiled away below Blackwell, leaving a kernel that launches, writes
  nothing and returns success. A pre-Blackwell arch is therefore a
  silent-garbage risk rather than a raise. No receipt is claimed there,
  and sm_120 was not run.
- **Not certified** (accepted by the op, no observed-behaviour claims
  here): the MXFP4 mode (`sf_vec_size=32`, `sf_use_ue8m0=True`, UE8M0
  scale bytes) beyond its argument validation, and float8_e4m3fn input
  (which routes to a different in-kernel conversion). Also uncertified,
  though the shape rule covers them: `M > 8192`; float16 and rank-3
  inputs at `K` in {2048, 7168, 18432} (only bf16, rank-2 ran there);
  and CUDA-graph capture, which nothing here exercises.
- Against a native-torch reference built from the formula above, the
  kernel was **bit-exact on every scale byte** and on every data nibble
  outside the near-tie window described in Semantics; every element
  inside that window differed by at most one adjacent e2m1 code with the
  sign unchanged. Shapes run:
  - **linear** layout, bf16: `T x K` for every `T` in
    {1, 2, 7, 64, 1023, 1024, 2048} crossed with `K` in
    {2560, 3072, 12288} except `2048 x 12288`; plus `T` in
    {1, 2, 7, 64, 1023, 1024, 2048, **8192**} crossed with `K` in
    {**2048**, **7168**, **18432**}; plus one 16,515,072-element sample
    reshaped to `8064 x 2048`, `2304 x 7168` and `896 x 18432`.
  - **swizzled** layout, bf16: `1 x 2560`, `7 x 2560`, `129 x 2560`,
    `1024 x 2560`, `200 x 3072`, `3 x 112` (`cols = 7`, the only case
    exercising column padding), and the full `T` x `K` grid above at
    `K` in {2048, 7168, 18432}. Every call listed here was checked
    offset-by-offset against the same input's linear call, and its data
    bytes asserted equal to the linear call's.
  - fp16: `T` in {1, 512, 1024} at `K = 2560`, linear only. A 3-D
    `2 x 5 x 2560` bf16 input: linear offset-checked, swizzled checked for
    buffer size and data equality only.
  The bf16 `K` in {2048, 7168, 18432} grid is the DeepSeek-R1 activation
  set — model hidden, dense-MLP intermediate, shared-expert intermediate
  — with `T` reaching `max_num_tokens = 8192`. At those widths
  `cols = K/16` is 128 / 448 / 1152, all already multiples of 4, so the
  swizzled buffer there carries row padding but no column padding.
- Reconstruction bound, verified elementwise on `1024 x 2560` bf16 input
  in fp32 and on `1024 x 18432` bf16 input in **float64**:
  `|data*sf/g - x| <= max(0.25 * sf/g, |x|/4)`. The two terms are the
  e2m1 subnormal half-step carried back through the block scale, and the
  half-step of the coarsest e2m1 binade (which also covers the clipping
  of a block max when `sf` rounds down). Every operand is exact in
  float64, so that form of the check is the mathematical bound; a
  consumer reconstructing in **fp32** can land up to half an fp32 ulp
  outside it (measured 3.0e-8 at `1024 x 18432`) purely from its own
  rounding.
- **One kernel — nothing about rows or width selects a different one.**
  Every call of this op in this build launches exactly one CUDA kernel,
  `tensorrt_llm::_v1::kernels::quantize_with_block_size<FP16_TO_FP4, T,
  16, false>`, and it is the same instantiation at 7 rows and at 8192, at
  widths 2048 / 7168 / 18432 (multiples of 512) and at 112 (not), in both
  scale layouts and for bf16 and fp16 alike — measured with the CUDA
  profiler in this entry's test. Consistent with the library's symbol
  table, inspected separately, which carries 14 instantiations of that one
  kernel template and no TMA variant at all.
  A TMA high-throughput variant selected on `M >= 1024 && width %
  512 == 0` **does** exist in FlashInfer's copy of this kernel family,
  which ships in this install as readable source — but it is reachable
  only through `torch.ops.trtllm.tunable_fp4_quantize`, never through
  this op. Treat that vendored source as a *newer* upstream revision, not
  as this build: its `invokeFP4Quantization` takes `enable_pdl`,
  `use_row_wise_scale` and `inverse_scale` parameters that the installed
  symbol does not have, and it accepts a per-token `globalScale` that
  this build silently ignores (see *Preconditions*).
- Consumer pairing, read from the TensorRT-LLM call paths in this install
  (each consumer's own contract is authoritative; this entry certifies
  only what each layout *contains*):
  - the NVFP4 dense GEMM family (`torch.ops.trtllm.nvfp4_gemm` and the
    backends it dispatches to) is fed the **swizzled** buffer —
    `is_sf_swizzled_layout=True`, the op's default; its CUDA-core backend
    explicitly un-swizzles before use;
  - the trtllm-gen NVFP4 block-scale MoE runner
    (`torch.ops.trtllm.fp4_block_scale_moe_runner`) is fed the **linear**
    buffer — `is_sf_swizzled_layout=False`, then viewed as
    `[num_tokens, cols]` — matching how `mxfp8_quantize` feeds the
    trtllm-gen W4A8 runner.
  So a target running an NVFP4 dense GEMM and an NVFP4 trtllm-gen MoE
  needs two different calls to this op, not one shared result. Both
  layouts are certified at `K = 7168`, which is where a DeepSeek-R1-shaped
  target needs exactly that pair of calls.
- **How discriminating the certification above is**, measured rather than
  assumed — the check tolerates one adjacent e2m1 code inside the
  near-tie window and nothing else, so the window is its only blind spot,
  and it is 0.03–0.60 % of elements. Driven through the same comparison at
  `1024 x 7168`: a single data nibble moved one code *outside* the window
  fails it; two codes *inside* the window fails it; a sign flip inside the
  window fails it; a single scale byte moved one code fails it (the scale
  bytes have no carve-out at all); one code inside the window passes, and
  is visible only as one extra disagreement in the count. A swizzled scale
  buffer handed over as if it were linear is a different byte string even
  when the two have the same length (rows a multiple of 128), so the
  layout check is not vacuous either.
- Sibling quantizers with the same scale-layout vocabulary exist
  (`torch.ops.trtllm.mxfp8_quantize` for fixed-32-block MXFP8,
  `torch.ops.trtllm.fp8_quantize_1x128` for 1x128 block-scaled fp8), as
  do fp4 variants that fuse extra work into the same launch
  (`torch.ops.trtllm.fp4_quantize_with_residual`,
  `torch.ops.trtllm.fp4_batched_quantize`) and a helper that converts an
  existing linear scale buffer to the swizzled order
  (`torch.ops.trtllm.block_scale_interleave`).
  `torch.ops.trtllm.tunable_fp4_quantize` is a different thing again: an
  autotuned chooser between this op's kernel and FlashInfer's, so it can
  reach the TMA kernel described above. This entry covers the plain
  `fp4_quantize` launch only.
- The op allocates both outputs itself: it is functional, with no `out=`
  parameter and no in-place mode.
