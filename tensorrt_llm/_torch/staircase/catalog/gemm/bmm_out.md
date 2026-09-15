---
receipts:
  sm_103: {status: passed, trtllm: 1.3.0rc26, tests: 53}
---

<!--
The `sm_100` key was REMOVED, not lost: it recorded a 1.3.0rc21 run against a
GPU test file since rewritten, so it predated a file in the entry. No sm_100
device was reachable to rerun the current files on, and "key absent = unknown"
is the honest state rather than a claim of failure there.

`tests:` is filled from the observed pytest count of the certifying run, never
from arithmetic.
-->

# bmm_out

**Wraps** `torch.ops.trtllm.bmm_out` (one call).

## Semantics

Batched matrix multiply written into a caller-provided output buffer:

```
out[i, m, n] = sum_k a[i, m, k] * b[i, k, n]    for every batch index i
```

Accumulation is fp32 for bf16/fp16 inputs: kernel and reference sum the same
exactly-representable products in a different order, so bf16 results agree to
within one rounding step but are **not** bit-identical. Compare with this
test's `atol=1e-3`, which is load-bearing — see "The tolerance" below. There is
no broadcasting: all three tensors are strictly 3D with equal batch sizes.

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

## Certified coverage

What the receipt covers. Every row is one or more cases in
`tests/unittest/_torch/staircase/gemm/test_staircase_bmm_out.py`; the receipt is
that file passing on sm_103.

| axis | certified values | cases |
|---|---|---|
| **V4.1 grouped output LoRA, TARGET** | batch **8** x `[M, 4096] @ [4096, 1024]`, bf16, `b` a transpose view, x 17 row buckets | 17 |
| row buckets | 1, 2, 3, 7, 8, 16, 31, 32, 64, 127, 128, 129, 255, 256, 512, 1024, 4096 | — |
| reference-only rank shard | batch **2**, same surface, `M` in {1, 129, 4096} — the native implementation's per-rank shard, **not a target call** | 3 |
| discrimination | the eight local groups' weights rolled by one, at `M=129` | 1 |
| reference independence | the per-batch `mm` reference against a GEMM-free mul+sum construction | 1 |
| loud rejections | 2-D operands, batch mismatch, K mismatch, `out` dtype mismatch, fp8 operands, mixed device — each with its message asserted | 6 |
| the same six **through the wrapper** | the wrapper must surface the op's message, not shadow it with its own error | 6 |
| mixed input dtypes | all 12 combinations over {bf16, fp16, fp32} x both `out` dtypes, each asserted to raise | 12 |
| accepted, not rejected | all-CPU operands compute correctly on the CPU | 1 |
| silent resize | standalone reallocation, in-place arena corruption, 2-D reshape, and the wrapper guard firing on all three | 1 |
| pre-existing column | generic batched shapes, non-contiguous views, unaligned shapes, fp16, fp32 | 5 |
| total | | **53**, all passing; `tests: 53` in the receipt |

### Where the output-LoRA geometry comes from, and why the batch is 8

Derived from the **raw checkpoint's safetensors header**, which is what the
target's `weights.py` reads — not from the reference implementation's rank
shard, and not from a summary:

```
layers.<L>.attn.wo_a.weight   [8192, 4096]   F8_E4M3
  = [o_groups * o_lora_rank, n_heads * head_dim // o_groups]
  = [8 * 1024,               64 * 512 / 8]
```

Viewed as `[o_groups, o_lora_rank, 4096] = [8, 1024, 4096]` and applied with
`einsum("bsgd,grd->bsgr")`, that is a batched matmul of **batch 8**,
`[M, 4096] @ [4096, 1024]`. The `b` operand is a **transpose view** of the
stored weight, never a contiguous copy — this entry already documents strided
`b` as first-class, and the V4.1 cases pin it at the shape that matters.

**The batch is 8, not 2, and an earlier revision of this entry had it wrong.**
The reference implementation declares `wo_a` as a `ColumnParallelLinear`, so at
`world_size=4` each rank holds `[2048, 4096]` and
`n_local_groups = o_groups // world_size = 2`. That is a property of the
*reference's* tensor parallelism, not of this checkpoint. The staircase target
does not shard it: `plan.md` line 30 fixes attention DP at dep4 and replicates
attention and every dense projection across ranks, and line 79 states the
target's own geometry as "eight group-local A projections followed by the full
B projection". Certifying batch 2 certified a shape the target never calls.
Batch 2 remains above as explicitly **reference-only** coverage, because the
module-parity leg compares against the reference implementation and knowing the
op is correct at the shard the reference runs removes one variable from that
comparison.

### The tolerance, and why nothing moved

This entry's existing gate — dtype-default `rtol`, `atol=1e-3` — **holds
unchanged** at the target shape: **0 elements outside** at every one of the 17
row buckets, with the worst correct error `9.998e-01` on a tensor of scale
`359` (batch 8, `M=4096`). `K = 4096` here is four times the `K <= 1024` the
existing `atol` rationale was written for, so this is the measurement that
extends it rather than an assumption: at the unmodified `atol=1e-5` the same
correct results put 1 (`M=1`) to 2,341 (`M=4096`) elements outside, all
near-zero outputs carrying fp32 summation-order noise — the regime `atol=1e-3`
exists for.

**Discrimination.** Rolling the eight local groups' weights by one —
block-diagonality means each group must project only its own heads, and a roll
keeps every shape and value identical — puts **33,383,466** elements outside the
same bound at `M=4096`, 1,051,284 at `M=129`, and 8,154 at `M=1`. The entry's
own case asserts over 90%.

### The reference is not `torch.bmm`

The op's body is exactly `torch.bmm(a, b, out=out)`. An expected value built
with `torch.bmm` is therefore built from the op under test: it cannot separate
"my reference is wrong" from "the op is wrong", and a defect inside `torch.bmm`
would be invisible because both sides carry it. The GPU test's `_ref` loops the
batch calling `torch.mm` per slice in fp32 — a different op and a different
kernel — and `test_the_reference_itself_is_independent` pins that loop against a
broadcast multiply plus a sum reduction, which launches no GEMM at all. Measured
agreement between the two: `max_abs = 1.907e-06`.

## Signature

```python
def bmm_out(a: torch.Tensor, b: torch.Tensor, out: torch.Tensor) -> None
```

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `a` | `[B, M, K]` (3D only) | bf16 / fp16 / fp32 | any strides (strided views verified) | CUDA (CPU accepted, see below) |
| `b` | `[B, K, N]` (3D only) | same as `a` | any strides (e.g. `weight.transpose(1, 2)` view) | same device as `a` |
| `out` | `[B, M, N]` (3D only) | same as `a` | any strides; written in place, never reallocated when the shape is right | same device as `a` |
| returns | — | — | `None`; the result is the mutation of `out` | — |

## Metadata consumed

None. Stateless.

## Preconditions

Every statement below was driven on **sm_103 / trtllm 1.3.0rc26 / torch
2.12.0a0** — the configuration this receipt certifies — and each is asserted by
a case in the GPU test with its message quoted. (A previous revision attributed
all of it to trtllm 1.3.0rc21 / torch 2.11.0 on sm_100, a path no receipt here
covers.)

### Rejected loudly by the op — the wrapper repeats none of them

| violation | observed message |
|---|---|
| a 2-D operand | `batch1 must be a 3D tensor` |
| batch size differs between `a` and `b` | `Expected size for first two dimensions of batch2 tensor to be: [4, 64] but got: [2, 64].` |
| `K` differs between `a` and `b` | `Expected size for first two dimensions of batch2 tensor to be: [4, 64] but got: [4, 128].` |
| `out.dtype` differs from the inputs | `Expected out tensor to have dtype c10::BFloat16, but got float instead` |
| fp8 operands | `NotImplementedError: "baddbmm_cuda" not implemented for 'Float8_e4m3fn'` |
| operands on different devices | `Expected all tensors to be on the same device, but got self is on cpu, different from other tensors on cuda:0` |

**Mixed input dtypes always raise** — all twelve combinations over
{bf16, fp16, fp32} x both choices of `out` dtype. There is no promotion path and
no silent hazard here: the meta check demands `out` in `b.dtype`
(`Expected out tensor to have dtype ...`) while the kernel demands `a.dtype`
(`expected scalar type BFloat16 but found Half`), so when `a.dtype != b.dtype`
the two can never both be satisfied, whichever the caller picks. The wrapper
therefore carries no dtype assert.

### Accepted, not rejected

* **All-CPU operands.** An all-CPU call is **not** rejected: it computes the
  correct result on the CPU (measured: 0 elements outside the gate). The
  precondition is "all three on the same device", and the CUDA part of it is a
  caller's obligation rather than a guard the op provides — a caller who passes
  CPU tensors silently takes that matmul off the GPU. Only a *mixed* device
  raises.

### Not validated by the op — the one the wrapper guards

* **`out.shape == (B, M, N)` exactly.** The op does not validate this. It
  resizes `out` and emits only a deprecation warning
  (`An output with one or more elements was resized since it had shape ...`).
  The damage depends on where the buffer came from, and no form of it is
  catchable:

  | `out` passed | measured outcome |
  |---|---|
  | standalone `[B, M, N/2]` | **reallocated** — `data_ptr` moves, the correct product lands in the *new* storage, and anything that aliased the old storage is silently stale |
  | `[B, M, N/2]` **view into a larger arena** | resized **in place** — `data_ptr` unchanged, the view itself holds the correct product, and the arena's surrounding layout is overwritten: 991 of 1024 elements of the arena's original `[:, :, :N]` window then disagree with it |
  | a sibling view of that same storage | keeps its own stale shape over the rewritten bytes — 495 of 512 elements outside the gate |
  | 2-D or 4-D `out` | silently **reshaped to 3-D**, same path |

  So this is shape/stride corruption plus collateral damage to aliased storage,
  never a lost write, and the wrapper asserts the shape. It is a wrapper guard
  rather than a documented precondition precisely because the op gives the
  caller nothing to catch.

  **The guard is rank-gated, and that is load-bearing rather than tidiness.**
  The assert reads `b.shape[2]`, which on a 2-D `b` raises
  `IndexError: tuple index out of range` from inside the wrapper before the op
  is ever reached — replacing the documented `batch1 must be a 3D tensor` with
  an error that names no operand, and making the first row of the table above
  unreachable through the wrapper. It is therefore evaluated only once `a` and
  `b` are both 3-D; every other rank falls through to the op.
  `test_wrapper_preserves_the_op_rejection` drives all six loud cases **through
  the wrapper** for exactly this reason.

A caller inside the coverage table above, violating none of the above, gets a
result within `rtol` (dtype default) and `atol=1e-3` of the independent fp32
per-batch `mm` reference. Outside that coverage this entry makes no claim.

## Notes

* The op body is exactly one `torch.bmm(a, b, out=out)`; the launched kernel
  is torch's cuBLAS strided-batched gemm, so numerical behavior follows the
  process-wide torch matmul settings (tf32 flags for fp32, bf16
  reduced-precision-reduction flag). Receipts here were taken under torch
  defaults.
* Not arch-gated; TRT-LLM uses it as the bf16 batched-gemm path in MLA
  weight-absorption and output projections, with the batch dim carrying
  head groups.
* Dtypes verified: bf16, fp16, fp32. fp64 and integer dtypes are untested
  (unknown).
