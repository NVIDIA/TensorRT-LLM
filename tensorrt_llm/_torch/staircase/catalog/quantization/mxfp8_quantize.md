---
receipts:
  sm_100: {status: passed, trtllm: 1.3.0rc21}
  sm_103: {status: passed, trtllm: 1.3.0rc26, tests: 13}
---

# mxfp8_quantize

**Wraps** `torch.ops.trtllm.mxfp8_quantize` (one call).

## Semantics

Dynamic MXFP8 (OCP microscaling FP8) quantization of an activation tensor:
one call converts bf16/fp16 values to e4m3 elements plus one UE8M0
(power-of-two) scale per **32 contiguous elements along the last dim**.
Nothing else is fused in — no normalization, no activation, no transpose.
The caller owns producing the input (e.g. bf16 hidden states) and owns
both outputs afterwards.

Let `M` = product of all leading dims, `K` = last dim, and
`padded_k = ceil(K / alignment) * alignment`. For row `m` and block
`b` of 32 elements:

```
amax      = max |x[m, 32b : 32b+32]|          # fp32; exact for bf16/fp16 inputs
scale     = smallest power of two >= amax / 448    # 448 = e4m3 max finite
sf[m, b]  = log2(scale) + 127                 # UE8M0 byte, so scale == 2^(sf-127)
data[m,i] = e4m3_round_to_nearest_even( x[m, i] / scale )   for i in the block
```

The scale is rounded **up** (toward +inf) to the next power of two, so
the block max always lands at or below e4m3's 448 and no element
saturates. Dequantization is exactly `data.float() * 2^(sf - 127)`;
per element the reconstruction error is at most `2^-4` relative (e4m3's
3-bit mantissa) plus `2^-10 * scale` absolute (e4m3 subnormals).

Padding is part of the op, not the caller's job:

- **Column padding** (`padded_k > K`, i.e. `alignment` wider than `K`):
  columns `K .. padded_k` of `data` are written as the zero byte `0x00`
  and their scale bytes are `0x00`. The valid columns are bit-identical
  to what an unpadded call produces.
- **Row padding** (swizzled layout only): the scale buffer is sized for
  `pad_up(M, 128)` rows; the bytes belonging to rows `M .. pad_up(M,128)`
  are `0x00`. The `data` tensor is never row-padded.

`data` is identical byte-for-byte under both `swizzled_layout` values —
only the scale buffer's size and element order change:

- `swizzled_layout=False` (**linear**): `M * cols` bytes, row-major, where
  `cols = padded_k / 32`. Byte `b` of row `m` is at flat offset
  `m * cols + b`; `sf.view(M, cols)` is exactly the per-block scale matrix.
- `swizzled_layout=True` (**128x4 swizzled**): `pad_up(M,128) * pad_up(cols,4)`
  bytes. The scale of `(m, b)` sits at flat offset

  ```
  (b % 4)
  + (b // 4)   * 512                      # 4 cols x 128 rows per column group
  + (m % 32)   * 16
  + ((m % 128) // 32) * 4
  + (m // 128) * 128 * pad_up(cols, 4)
  ```

  Every offset not addressed by a real `(m, b)` pair is `0x00` padding.

## Signature

```python
def mxfp8_quantize(
    input: torch.Tensor,
    swizzled_layout: bool = True,
    alignment: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]
```

| Argument | Shape | Dtype | Layout / value | Device |
|---|---|---|---|---|
| `input` | `[..., K]`, rank >= 2 | bfloat16 or float16 | contiguous | CUDA |
| `swizzled_layout` | scalar | Python bool | `False` = linear scale order, `True` = 128x4 swizzled | — |
| `alignment` | scalar | Python int | positive multiple of 32; pads `K` up to `ceil(K/alignment)*alignment` | — |
| returns `data` | `[..., padded_k]` (leading dims preserved) | float8_e4m3fn | contiguous, freshly allocated | same as `input` |
| returns `sf` | 1-D: `[M*cols]` (linear) or `[pad_up(M,128)*pad_up(cols,4)]` (swizzled) | uint8 | contiguous, freshly allocated | same as `input` |

`M` = product of `input`'s leading dims, `cols = padded_k / 32`. `input`
is not modified. Leading dims are collapsed for the scale buffer in both
layouts (a `[B, S, K]` input yields a scale buffer laid out for `B*S`
rows), while `data` keeps `input`'s leading dims.

The op's own schema names the second parameter `swizzedLayout` (upstream
spelling); it is positional here.

## Metadata consumed

None. Stateless — no runtime, attention metadata, or workspace.

## Preconditions

- `input` is on CUDA, contiguous, rank >= 2, dtype bfloat16 or float16.
  Violations raise: a CPU tensor, a non-contiguous view (column slice or
  transpose), a 1-D tensor and an fp32 tensor were each observed to raise
  on this machine rather than compute something wrong.
- `K % 32 == 0` (the block size). Raises otherwise.
- `alignment % 32 == 0` and `alignment > 0`. A non-multiple raises;
  `alignment = 0` is **not** an exception — it kills the process with
  SIGFPE (integer division by zero in the padding computation). There is
  no upper bound: `alignment` may exceed `K` (`K=2880, alignment=512`
  gives `padded_k=3072`) or equal the block size (`alignment=32`, no pad).
  A **negative** multiple of 32 is the quiet one: it passes the
  `% 32 == 0` guard, raises nothing, and **silently truncates** — at
  `K = 2880` the values `-32 / -64 / -512` return `padded_k`
  `2816 / 2752 / 2048`, matching `((K + a - 1) / a) * a` exactly, i.e. the
  round-up above turning into a round-*down* under truncating division.
  The columns
  that survive are bit-exact against the reference and the scale tensor is
  self-consistent, so the result looks entirely well-formed; the tail of
  every row is simply gone. Check the sign before the modulus.
- `input` must be **finite**. Non-finite lanes are accepted silently and
  are destructive:
  - a `+inf`/`-inf` lane drives the block scale to the E8M0 finite max
    (byte 254 = 2^127); the inf lane becomes NaN and **every other lane in
    that 32-element block is zeroed**;
  - a NaN lane is excluded from the block max, so the scale stays correct
    and the NaN stays confined to its own lane.
- Every 32-element block must have `amax == 0` or `amax > 448 * 2^-127`
  (~2.63e-36). An all-zero block is fine (scale byte `0x00`, all-zero
  data). A block whose max is nonzero but at or below that threshold
  silently produces garbage: scale byte `0x00`, every nonzero lane
  saturated to +-448, and **every exact-zero lane turned into NaN**. Real
  bf16 hidden states never reach this range; a block of scaled-down
  denormals does.
- `M == 0` (zero rows) is accepted and returns empty `data` and `sf`.
- The caller owns both returned buffers; nothing else writes them.

## Notes

- Certified on sm_100 (B200) only. The kernel behind this op is
  Blackwell-targeted: the same TensorRT-LLM quantization kernel is
  vendored as readable source elsewhere in this install (flashinfer's
  bundled `nv_internal` copy) and compiles its body only under
  `__CUDA_ARCH__ >= 1000`, the empty branch returning without writing the
  output buffers. A pre-Blackwell arch is therefore a silent-garbage risk
  rather than a raise. No receipt is claimed there.
- Against a native-torch reference built from the formula above, the
  kernel was **bit-exact** on every element and every scale byte across
  `T x 2880` and `T x 3072` bf16 inputs for `T` in {1, 2, 7, 64, 1024,
  8192} and `alignment` in {32, 128, 512}, plus fp16 inputs and a 3-D
  input. The scale math runs through approximate reciprocals (the
  flush-to-zero surface above is their footprint), yet the result did not
  shift even when `amax` is exactly `448 * 2^k` — the only value where a
  1-ulp difference could flip the round-up (verified for `k = -2, 0, 1, 3`).
- Scale-byte dynamic range from bf16/fp16 inputs: the largest finite bf16
  magnitude (3.39e38) yields byte 247, so the E8M0 finite max (254) is
  reachable only from a non-finite input.
- Consumer pairing. The trtllm-gen W4A8 MXFP4xMXFP8 MoE runner
  `torch.ops.trtllm.mxe4m3_mxe2m1_block_scale_moe_runner` takes this op's
  two outputs as its `hidden_states` / `hidden_states_scale` and expects
  the **linear** scale buffer (`swizzled_layout=False`), sized exactly
  `M * padded_k/32` bytes, with `alignment` equal to that kernel's
  input-hidden alignment — 512 for the trtllm-gen MXFP4 weight path, which
  is what pads gpt-oss's 2880 hidden to 3072 inside this call (so the
  caller does not pre-pad). The swizzled layout is what the CUTLASS-side
  mxfp8 GEMM/MoE consumers take instead. Those pairings are fixed by the
  consuming ops' own contracts; this entry certifies only what each layout
  contains.
- Sibling quantizers with the same scale-layout vocabulary exist
  (`torch.ops.trtllm.fp4_quantize` for NVFP4/MXFP4 with a 16- or
  32-element vector, `torch.ops.trtllm.fp8_quantize_1x128` for 1x128
  block-scaled fp8); this entry covers the fixed-32-block mxfp8 path only.
- The op allocates both outputs itself: it is functional, with no `out=`
  parameter and no in-place mode.
