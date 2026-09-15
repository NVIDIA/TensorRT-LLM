---
receipts:
  sm_103: {status: passed, trtllm: 1.3.0rc26, tests: 111}
---

<!--
The `sm_100` key was REMOVED, not lost. It recorded a 1.3.0rc21 run against a
GPU test file that has since been extended with the V4.1 column, so it predated
a file in the entry and a receipt that predates its own test file is what the
freshness rule exists to catch. No sm_100 device was reachable to rerun the
current files on; "key absent = unknown" is the honest state, not a claim that
the op fails there.
-->


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

## Certified coverage

What the receipt covers. Every row is one or more cases in
`tests/unittest/_torch/staircase/quantization/test_staircase_mxfp8_quantize.py`;
the receipt is that file passing on sm_103.

| axis | certified values | cases |
|---|---|---|
| **V4.1 target widths x rows** | **the full cross-product**: `K` in `{8192, 6144, 5120, 2304, 1280}` x 17 row buckets, bf16, `alignment=32`, **both layouts in each case** | 85 |
| V4.1 target widths | `8192` attn wo_b input, `6144` engram.wkv input, `5120` attn wq_a / wkv / shared w1,w3 input, `2304` shared-expert w2 input, `1280` attn wq_b / indexer wq_b input | — |
| row buckets | 1, 2, 3, 7, 8, 16, 31, 32, 64, 127, 128, 129, 255, 256, 512, 1024, 4096 | — |
| reference-only width | `K = 2048`, `M` in {1, 129, 4096} — the native implementation's row-parallel shard of `wo_b`'s input, **not a target call** | 3 |
| V4.1 leading dims | `[3, 43, 5120]` collapsing to 129 scale rows, bf16, linear layout | 1 |
| loud rejections | 8 domains, each with its exact message asserted | 8 |
| silent alignment domains | negative alignment truncating to 2816 / 2752 / 2048 with bit-exact survivors; `alignment=0` returning `[M, 0]` in a child process; the wrapper guard firing on both | 2 |
| pre-existing column | gpt-oss geometry: `K` 2880/3072, `alignment` 32/128/512, bf16 and fp16, 3-D input, dequant error bound, power-of-two amax boundary, extreme magnitudes, non-finite inputs, zero and denormal blocks, zero rows, input not mutated | 12 |
| total | | **111**, all passing; `tests: 111` in the receipt |

At every V4.1 width the op is **bit-exact** against a native-torch reference —
both the e4m3 data bytes and the UE8M0 scale bytes, at every row bucket. All
widths are multiples of 32, so `alignment=32` gives `padded_k == K` and there
is no padding on this path.

### Where the width set comes from, and why 8192 is in it

The widths are the activation widths of this checkpoint's dense FP8 surfaces,
read from the **raw checkpoint's safetensors headers** (`[out, in]`) — which is
what the target's `weights.py` reads — not from the reference implementation's
rank shard.

**`K = 8192` is the one an earlier revision of this entry missed, and `K = 2048`
was in its place.** `layers.<L>.attn.wo_b.weight` is `[5120, 8192]` in the raw
checkpoint. The reference implementation declares `wo_b` as a
`RowParallelLinear`, which splits the **reduction** dim, so each of its four
ranks quantizes a 2048-wide activation and 8192 never appears anywhere in that
implementation. The staircase target does not shard it: `plan.md` line 30 fixes
attention DP at dep4 and replicates every dense projection, and line 79 states
`wo_b [5120, 8192]` as the target's own geometry. So 8192 is the width the
target quantizes and 2048 is one it never calls; 2048 stays above as explicitly
**reference-only** coverage because the module-parity leg runs the reference at
it.

**The cross-entry length agreement is asserted, not assumed.** The swizzled
scale buffer this op returns is exactly what `gemm/mxfp8_mxfp8_gemm` consumes as
`act_scale`, and that contract states its required length as
`pad_up(M,128) * pad_up(K/32,4)`. The two entries are certified separately, so
nothing else in the catalog would notice if they disagreed; the V4.1 cases
assert the returned length against that formula and check the bytes land at the
128x4 offsets. Measured lengths at `M=128`: 32,768 / 24,576 / 20,480 / 9,216 /
5,120 bytes for `K` = 8192 / 6144 / 5120 / 2304 / 1280 (and 8,192 for the
reference-only 2048).

Uncertified and therefore not claimed: `M > 4096`, widths outside the two sets
above, fp16 at the V4.1 widths, `alignment != 32` at the V4.1 widths, and any
arch other than sm_103.

## Metadata consumed

None. Stateless — no runtime, attention metadata, or workspace.

## Preconditions

Every rejection below was driven on **sm_103 / trtllm 1.3.0rc26** — the
configuration this receipt certifies — and each is asserted by a case in the GPU
test with its exact message.

### Rejected loudly by the op — the wrapper repeats none of them

| violation | observed message |
|---|---|
| `K % 32 != 0` | `k must be divisible by SF_VEC_SIZE = 32` |
| `alignment % 32 != 0`, including any value below 32 | `alignment must be divisible by SF_VEC_SIZE = 32` |
| fp32 input | `NotImplementedError: mxfp8_quantize only supports input tensor with dtypes fp16/bf16.` |
| 1-D input | `Input should be >=2D tensor.` |
| non-contiguous input (column slice or transpose) | `self must be contiguous` |
| CPU input | `NotImplementedError: Could not run 'trtllm::mxfp8_quantize' with arguments from the 'CPU' backend.` |

`alignment = 16` and `alignment = 48` produce the **same** message: below-block
and non-multiple are one divisibility check in the op, not two domains.

### The `alignment` sign — not validated by the op, and the wrapper guards it

`alignment` has no upper bound: it may exceed `K` (`K=2880, alignment=512` gives
`padded_k=3072`) or equal the block size (`alignment=32`, no pad). The **sign**
is the problem, and both wrong-sign cases are silent:

* **`alignment = 0` returns an empty result on this host; it does not trap.**
  `0 % 32 == 0` passes the op's own check, and
  `cpp/tensorrt_llm/thop/mxFp8Quantize.cpp:60` then computes
  `padded_k = ((k + alignment - 1) / alignment) * alignment` with no zero guard.
  Measured in a child process on this GB300: the call **returns**, with `data`
  of shape `[M, 0]` and an empty scale buffer, exit status 0.

  A previous revision of this bullet said it "kills the process with SIGFPE",
  which is the expensive direction to be wrong in — it promises a crash the
  caller then does not get. That claim was **host-ISA dependent, not merely
  stale**: x86-64's `DIV` traps on a zero divisor, aarch64's `SDIV` returns 0.
  This checkout's receipts are taken on an aarch64 (Grace) host, where the
  outcome is the silent one.

* **A negative multiple of 32 truncates instead of padding.** It passes the
  `% 32 == 0` guard, raises nothing, and returns fewer columns than `K`: at
  `K = 2880` the values `-32 / -64 / -512` give `padded_k`
  `2816 / 2752 / 2048`, matching `((K + a - 1) / a) * a` under C's
  truncate-toward-zero division, i.e. the round-up turning into a round-*down*.
  The surviving columns are **bit-exact** against the reference and the scale
  tensor is self-consistent, so the result looks entirely well-formed; the tail
  of every row is simply gone.

Both are `alignment <= 0`, so the wrapper asserts `alignment > 0` — a
scalar-metadata check on a domain the op leaves silent. No existing caller of
this entry passes a non-positive alignment (gpt-oss passes 512, V4.1 passes 32),
so the guard rejects nothing anyone calls.

### The rest

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

- **Blackwell-only, and a pre-Blackwell arch is a silent-garbage risk rather
  than a raise.** (This bullet previously read "Certified on sm_100 (B200)
  only", which was stale on both counts: the sm_100 receipt has been removed as
  predating the current test file, and the live receipt is sm_103.) The kernel
  behind this op is Blackwell-targeted: the same TensorRT-LLM quantization
  kernel is
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
