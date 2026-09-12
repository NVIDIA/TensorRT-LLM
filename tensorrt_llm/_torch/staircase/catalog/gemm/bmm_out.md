---
receipts:
  sm_100: {status: passed, trtllm: 1.3.0rc21}
  sm_103: {status: passed, trtllm: 1.3.0rc26, tests: 5}
---

# bmm_out

**Wraps** `torch.ops.trtllm.bmm_out` (one call).

## Semantics

Batched matrix multiply written into a caller-provided output buffer:

```
out[i, m, n] = sum_k a[i, m, k] * b[i, k, n]    for every batch index i
```

Accumulation is fp32 for bf16/fp16 inputs: kernel and reference sum the same
exactly-representable products in a different order, so bf16 results agree to
within one rounding step but are **not** bit-identical. The per-element
mismatch rate against `torch.bmm(a.float(), b.float())` tracks K (~7e-6 at
K=16, ~5e-5 at K=64, ~3e-4 at K=576), so whether any one call happens to come
back bit-exact is a matter of how many elements it produces — and some shapes
sit in cuBLAS kernel-selection windows that are bit-identical over millions of
elements. Compare with this test's `atol=1e-3`, which is load-bearing: at
torch's default bf16 `atol` of 1e-5 a prefill-shaped
`(B=8, M=2048, K=512, N=128)` output fails 4 of 10 seeds. There is no
broadcasting: all three tensors are strictly 3D with equal batch sizes.

Fusion boundary: the single call computes the batched gemm and nothing else —
no bias, no scaling, no activation, no quantization. The caller owns the
allocation of `out`, any transposition of `b` (pass a transpose *view*), and
any packing of head/group dims into the batch dim. Sibling ops exist for
quantized or arch-specialized batched gemms (`fp8_block_scaling_bmm_out`,
`fp4_bmm`, `cute_dsl_bf16_bmm_blackwell`).

The op's reason to exist over plain `torch.bmm(..., out=)`: it is registered
as an opaque custom op so a torch.compile graph does not break when `out` is
a non-contiguous view. Strided views are first-class for all three arguments
(verified: transposed `out`, transposed `b`, row-strided `a`).

## Signature

```python
def bmm_out(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None
```

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `a` | `[B, M, K]` (3D only) | bf16 / fp16 / fp32 | any strides (strided views verified) | CUDA |
| `b` | `[B, K, N]` (3D only) | same as `a` | any strides (e.g. `weight.transpose(1, 2)` view) | CUDA |
| `out` | `[B, M, N]` (3D only) | same as `a` | any strides; written in place, never reallocated when the shape is right | CUDA |
| returns | — | — | `None`; the result is the mutation of `out` | — |

## Metadata consumed

None. Stateless.

## Preconditions

- All three tensors are 3D, on the same CUDA device. 2D inputs raise
  (`batch1 must be a 3D tensor`); there is no implicit batch broadcasting.
- Batch sizes and the contraction dim `K` must match between `a` and `b`
  (mismatch raises).
- `out.shape == (B, M, N)` exactly. **The op does not validate this**: a
  wrong-shaped `out` is silently resized (with only a deprecation
  warning). The hazard is real but is **shape/stride corruption, not a
  lost write**: measured 2026-07-28, an aliased view with room to grow
  keeps its `data_ptr` and the result *does* land in the caller's buffer,
  now under a silently rewritten shape; a genuine reallocation moves the
  shared storage, so sibling views follow it rather than being detached.
  The wrapper asserts the shape.
- One dtype across `a`, `b`, `out`. **Mixed input dtypes always raise** —
  there is no promotion path and no silent hazard here. The meta check
  demands `out` in `b.dtype` while the kernel demands `a.dtype`, so when
  `a.dtype != b.dtype` the two can never both be satisfied: all six
  combinations over {bf16, fp16, fp32} raise, including the bf16-`a` /
  fp32-`b` / fp32-`out` case an earlier revision of this bullet described
  as working (measured 2026-07-28). The wrapper's single-dtype assert is
  therefore redundant rather than load-bearing.
- `out.dtype` must equal the input dtype; a mismatch raises
  (`Expected out tensor to have dtype ...`).
- Dtypes verified: bf16, fp16, fp32. float8_e4m3fn raises
  (`"baddbmm_cuda" not implemented for 'Float8_e4m3fn'`). fp64 and integer
  dtypes are untested (unknown).
- Arbitrary strides are supported for all three tensors, including
  zero-copy transpose views of `b` and `out` (verified correct).

## Notes

- The op body is exactly one `torch.bmm(a, b, out=out)`; the launched kernel
  is torch's cuBLAS strided-batched gemm, so numerical behavior follows the
  process-wide torch matmul settings (tf32 flags for fp32, bf16
  reduced-precision-reduction flag). Receipts here were taken under torch
  defaults.
- Not arch-gated; TRT-LLM uses it as the bf16 batched-gemm path in MLA
  weight-absorption and output projections, with the batch dim carrying
  head groups.
- Behavior above was established empirically under trtllm 1.3.0rc21 /
  torch 2.11.0 on sm_100.
