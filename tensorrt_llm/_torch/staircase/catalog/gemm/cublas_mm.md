---
receipts:
  sm_100: {status: passed, trtllm: 1.3.0rc21}
  sm_103: {status: passed, trtllm: 1.3.0rc26, tests: 7}
---

# cublas_mm

**Wraps** `torch.ops.trtllm.cublas_mm` (one call).

## Semantics

General matrix multiply with an optional fused bias epilogue, executed as a
single cuBLASLt matmul:

```
out[m, n] = sum_k mat_a[m, k] * mat_b[k, n]  (+ bias[n])
```

Accumulation is fp32 (verified: bf16/fp16 results match an fp32-accumulated
reference exactly at K=256). The result is cast to the output dtype.

Fusion boundary: the single call computes the gemm and, when `bias` is given,
adds it per output column inside the epilogue. Nothing else happens inside:
no activation, no quantization, no input/output scaling (fp8 inputs are
multiplied with alpha=1; a sibling op `cublas_scaled_mm` exists for the
scaled variant). The caller owns weight transposition metadata (pass a
transpose *view*, see Preconditions) and any flattening of leading dims.

## Signature

```python
def cublas_mm(
    mat_a: torch.Tensor,
    mat_b: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    output_buffer_kind: int = 0,
    group: list[int] | None = None,
) -> torch.Tensor
```

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `mat_a` | `[M, K]` (2D only) | bf16 / fp16 / fp32 / float8_e4m3fn | dense row-major (contiguous) | CUDA |
| `mat_b` | `[K, N]` (2D only) | same as `mat_a` | dense column-major: strides `(1, K)`, e.g. `weight.t()` of a contiguous `[N, K]` weight | CUDA |
| `bias` | `[N]` or `None` | must equal the output dtype | contiguous | CUDA |
| `out_dtype` | — | `None` → same as `mat_a`; conversions verified: bf16→fp32, fp8e4m3→bf16 | — | — |
| `output_buffer_kind` | scalar | Python int: 0=DEFAULT, 1=USERBUFFERS, 2=NCCL_WINDOW | — | — |
| `group` | list of ranks or `None` | Python ints | only meaningful with NCCL_WINDOW | — |
| returns | `[M, N]` | `out_dtype` or `mat_a.dtype` | newly allocated, contiguous | CUDA |

## Metadata consumed

None. Stateless for the default path. `output_buffer_kind` selects where the
output tensor is allocated: 0 is a plain CUDA allocation; 1 (USERBUFFERS) and
2 (NCCL_WINDOW, together with `group` = the TP rank list) draw from
communication buffer pools so a downstream allreduce can consume the output
in place — those pools must have been initialized by the runtime beforehand.
Single-GPU callers pass `0, None` (2 with `group=None` silently falls back to
a plain allocation). The enum is importable as
`tensorrt_llm.bindings.internal.thop.BufferKind`.

## Preconditions

- `mat_a` and `mat_b` are 2D, on the same CUDA device, same dtype. 3D+
  inputs raise; flatten leading dims to `[M, K]` and unflatten after.
- `mat_a` is dense row-major. **The kernel ignores `mat_a` row strides**: a
  row-strided view (e.g. a column slice of a wider buffer) is accepted and
  produces silently wrong results (verified). The wrapper asserts
  contiguity.
- `mat_b` has strides exactly `(1, K)` — a dense column-major buffer, i.e.
  the `.t()` view of a contiguous `[N, K]` weight. The op checks
  `stride(0) == 1` and raises otherwise, but does **not** check
  `stride(1) == K`: the transpose view of a row-strided weight is accepted
  and produces silently wrong results (verified). The wrapper asserts both.
- Dtypes: bf16, fp16, fp32 work with `out_dtype=None`; float8_e4m3fn
  requires an explicit `out_dtype` (`None` raises a cuBLAS error).
  `mat_a.dtype != mat_b.dtype` raises `CUBLAS_STATUS_NOT_SUPPORTED`.
- `bias`, when given:
  - shape exactly `[N]`, contiguous, on the same device. **The op does not
    validate the bias shape** (a wrong-length bias was accepted); the
    wrapper asserts it.
  - dtype must equal the *output* dtype (bf16 bias for bf16 out, fp32 bias
    for bf16→fp32, bf16 bias for fp8→bf16). A mismatched dtype (e.g. fp32
    bias with bf16 output) is accepted and produces silently wrong results
    (verified).
  - not supported with fp32 inputs: the bias is accepted and **silently
    ignored** (verified: output equals the unbiased product). The wrapper
    asserts against this combination.
- `out_dtype` conversions other than the verified ones are not guaranteed:
  bf16→fp16 raises a cuBLASLt runtime error on this machine; combinations
  not listed above are untested (unknown).

## Notes

- The fake (meta) registration of this op accepts N-D `mat_a`, but the real
  kernel requires 2D — shape inference under compile can diverge from
  eager behavior for N-D inputs.
- Error traces name `cpp/tensorrt_llm/thop/cublasScaledMM.cpp` as the
  implementation (shared with the scaled variant); the C++ source is not
  shipped in the wheel, so all behavior above was established empirically
  on sm_100.
- TRT-LLM routes bf16 linears to this op on sm>=100 because cuBLASLt picks
  single-pass cluster-mode kernels for small-M (decode) gemms there; the
  op itself is not gated on arch.
- All silently-wrong-result behaviors listed under Preconditions were
  observed under trtllm 1.3.0rc21 on sm_100.


## The fp32 cell needs its reference pinned, not its tolerance widened

torch 2.12 defaults `torch.backends.cuda.matmul.fp32_precision` to `tf32` (and
`allow_tf32` to True) on this hardware, so the natural reference
`mat_a.float() @ mat_b.float()` is a **TF32** product. Against a correct
kernel that reads as a ~0.035 absolute error at K=1024 -- about 30x the op's
own -- and it is the *reference* that is wrong.

Measured on sm_103 against a float64 product on the same inputs:

| | max abs error vs float64 |
|---|---|
| `cublas_mm` | 1.9e-5 |
| torch, `allow_tf32=False` | 1.9e-5 (**bit-identical to the op**) |
| torch, `allow_tf32=True` | 3.5e-2 |

So the op is exact to fp32 and this entry's `_ref_mm` now forces TF32 off for
the duration of the reference product. The same trap applies to any entry
whose reference multiplies in fp32; the bf16 and fp16 cells were merely
tolerant enough to hide it.
