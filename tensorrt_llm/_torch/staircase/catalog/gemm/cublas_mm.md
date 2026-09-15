---
receipts:
  sm_103: {status: passed, trtllm: 1.3.0rc26, tests: 61}
---

<!--
The `sm_100` key was REMOVED, not lost: it recorded a 1.3.0rc21 run against a
GPU test file since rewritten, so it predated a file in the entry. No sm_100
device was reachable to rerun the current files on, and "key absent = unknown"
is the honest state rather than a claim of failure there.

`tests:` is filled from the observed pytest count of the certifying run, never
from arithmetic.
-->

# cublas_mm

**Wraps** `torch.ops.trtllm.cublas_mm` (one call).

## Semantics

General matrix multiply with an optional fused bias epilogue, executed as a
single cuBLASLt matmul:

```
out[m, n] = sum_k mat_a[m, k] * mat_b[k, n]  (+ bias[n])
```

Accumulation is fp32 (verified: bf16/fp16 results match an fp32-accumulated
reference exactly at K=256, and with fp32 operands the op is **bit-identical**
to the fp32 reference — `max_abs` exactly `0.000e+00` at every head row bucket).
The result is cast to the output dtype.

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

## Certified coverage

What the receipt covers. Every row is one or more cases in
`tests/unittest/_torch/staircase/gemm/test_staircase_cublas_mm.py`; the receipt
is that file passing on sm_103.

| axis | certified values | cases |
|---|---|---|
| **V4.1 language head, TARGET** | `[M, 5120] @ [5120, 129280]` x 17 row buckets, **bf16** (the checkpoint's own dtype) | 17 |
| same surface, fp32 | `[M, 5120] @ [5120, 129280]` x 17 row buckets, fp32 (the reference's dtype) | 17 |
| row buckets | 1, 2, 3, 7, 8, 16, 31, 32, 64, 127, 128, 129, 255, 256, 512, 1024, 4096 | — |
| reference-only rank shard | `[M, 5120] @ [5120, 32320]`, `M` in {1, 129, 4096}, fp32 and bf16 — the native implementation's per-rank shard, **not a target call** | 6 |
| discrimination | reversed vocabulary at `(M=129, N=129280)`, bf16 | 1 |
| loud rejections | row-major `mat_b`, 3-D `mat_a`, mixed input dtypes, fp8 without `out_dtype`, bf16→fp16 out, CPU operands, K mismatch — each with its message asserted | 7 |
| accepted-but-wrong domains | row-strided `mat_a`, `mat_b` with `stride(1) != K`, wrong-length bias, wrong-dtype bias, bias dropped with fp32 inputs — each measured wrong **and** shown to fire the wrapper guard | 5 |
| meta registration | the fake kernel accepts N-D `mat_a` where the real one raises | 1 |
| pre-existing column | generic `[M,K]@[K,N]` up to 2048 wide: bf16 with and without bias, bf16->fp32 out, unaligned shapes, fp16, fp32, fp8-e4m3->bf16 | 7 |
| total | | **61**, all passing; `tests: 61` in the receipt |

### Where the head geometry comes from, and why N is 129280

Derived from the **raw checkpoint's safetensors header**, which is what the
target's `weights.py` reads:

```
head.weight    [129280, 5120]   BF16      (untied: tie_word_embeddings = false)
embed.weight   [129280, 5120]   BF16
```

**The target width is 129280, not 32320, and an earlier revision of this entry
had it the other way round.** The reference implementation's `ParallelHead`
holds `[vocab_size // world_size, dim]` = `[32320, 5120]` per rank and
all-gathers the four shards — that is a property of the *reference's* tensor
parallelism. The staircase target does not shard it: `plan.md` line 88 requires
it to "Replicate and load independently" and "Produce complete local logits
without a vocab collective", and line 30 places the head among the weights
replicated under attention DP. Certifying 32320 certified a width the target
never calls. It remains above as explicitly **reference-only** coverage, because
`N = 32320` is 64-aligned but **not** 128-aligned — the kind of `N` a gemm
kernel can have an opinion about, driven rather than assumed — and because the
module-parity leg runs the reference at that width.

**Both dtypes are certified at the target width**, for a stated reason rather
than for coverage: the checkpoint stores the head in bf16 (`plan.md` line 35,
"Replicate the BF16 embedding and untied BF16 language head") while the
reference promotes the same weight to fp32 so its logits come out fp32
directly. The parity leg compares the two, so both are on the path.

### The tolerance, and why nothing moved

This entry's existing gate — dtype-default `rtol`, `atol=1e-3`, justified by
fp32 summation-order noise of about `K * 2**-24` — **holds unchanged at the
target head surface**. Measured across all 17 row buckets at `N=129280`:
**0 elements outside that bound** in both fp32 and bf16 (and 0 at `N=32320`
too). In fp32 the op is **bit-identical** to the reference (`max_abs` exactly
`0.000e+00`) at every row bucket; in bf16 the worst correct error is
`1.002e+00` on a tensor of scale `418`, which is the bf16 output cast and sits
far inside `rtol`.

For the record of what the looser-than-default `atol` costs in discrimination:
at the unmodified `atol=1e-5` the same correct bf16 results put 5 (`M=1`) to
48,961 (`M=4096`) elements outside at `N=129280`, all of them near-zero logits
carrying fp32 accumulation noise — precisely the regime `atol=1e-3` exists for.

**Discrimination.** Reversing the vocabulary rows of the weight — same shape,
same values, only the row order changed, i.e. the shape of a real
weight-loading bug — puts **526,830,345 of 529,530,880** elements outside the
same bound at `(M=4096, N=129280)`, and 128,637 of 129,280 at `M=1`. The
entry's own case asserts over 90% at `M=129`.

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

Every statement below was driven on **sm_103 / trtllm 1.3.0rc26** — the
configuration this receipt certifies — and each is asserted by a case in the
GPU test. (A previous revision attributed all of it to trtllm 1.3.0rc21 on
sm_100, a path no receipt here covers.)

### Rejected loudly by the op — the wrapper repeats none of them

| violation | observed message |
|---|---|
| `mat_b` dense row-major (`stride(0) != 1`) | `Expected mat_b.strides()[0] == 1 to be true, but got false.` |
| `mat_a` or `mat_b` not 2-D | `Expected mat_a.dim() == 2 && mat_b.dim() == 2 to be true, but got false.` |
| `mat_a.dtype != mat_b.dtype` | `CUDA runtime error in cublasLtMatmulAlgoInit(...): CUBLAS_STATUS_NOT_SUPPORTED (cpp/tensorrt_llm/thop/cublasScaledMM.cpp:100)` |
| fp8 operands with `out_dtype=None` | the same `cublasLtMatmulAlgoInit` / `CUBLAS_STATUS_NOT_SUPPORTED` |
| bf16 operands with `out_dtype=fp16` | `CUDA runtime error in cublasLtMatmul(...): CUBLAS_STATUS_NOT_SUPPORTED (cpp/tensorrt_llm/common/cublasMMWrapper.cpp:215)` |
| operands on the CPU | `NotImplementedError: Could not run 'trtllm::cublas_mm' with arguments from the 'CPU' backend.` |
| `K` differs between operands | `Expected out.sizes()[0] == mat_a.sizes()[0] && mat_a.sizes()[1] == mat_b.sizes()[0] && mat_b.sizes()[1] == out.sizes()[1] to be true, but got false.` |

### Not validated by the op — the four the wrapper guards

Each of these is **accepted** and produces a wrong answer with no diagnostic,
which is what earns the guard. The numbers are from the certified run at
`M=64, K=256, N=128`, bf16:

* **`mat_a` must be dense row-major.** The kernel ignores `mat_a`'s row
  strides, so a row-strided view (a column slice of a wider buffer) is accepted
  and reads the wrong elements: `max_abs 8.547e+01` from its own truth, on a
  tensor of scale `6.343e+01`. The wrapper asserts contiguity.
* **`mat_b` must have strides exactly `(1, K)`.** The op checks
  `stride(0) == 1` and raises otherwise, but does **not** check `stride(1)`:
  the `.t()` view of a row-strided weight has `stride = (1, 2K)`, passes that
  check, and is accepted — `max_abs 9.004e+01` on a scale of `6.649e+01`. The
  wrapper asserts both strides.
* **`bias` shape and dtype.** A bias of length `N//2` is accepted and a
  full-width `[M, N]` result comes back. A bias whose dtype differs from the
  *output* dtype (fp32 bias with bf16 output) is accepted and wrong by
  `3.000e+00`. The wrapper asserts `bias.shape == (N,)`, contiguity, and
  `bias.dtype == out_dtype`.
* **`bias` with fp32 inputs is silently dropped**, exactly: the result is
  `torch.equal` to the unbiased product, and `2.500e+00` from the biased one.
  The wrapper asserts against that combination rather than letting a caller
  believe the epilogue ran.

A caller inside the coverage table above, violating none of the above, gets a
result within `rtol` (dtype default) and `atol=1e-3` of an fp32 reference.
Outside that coverage this entry makes no claim.

* `out_dtype` conversions other than the verified ones are not guaranteed:
  bf16→fp16 raises (above); combinations not listed are untested (unknown).

## Notes

- The fake (meta) registration of this op accepts N-D `mat_a` (driven:
  a `[2, 64, 256]` meta tensor returns `[2, 64, 128]`), while the real kernel
  requires 2-D — shape inference under compile can diverge from eager behavior
  for N-D inputs.
- The implementation is `cpp/tensorrt_llm/thop/cublasScaledMM.cpp` (shared with
  the scaled variant), which **is** present in this checkout: the rejection
  messages above quote its path and line directly. An earlier revision of this
  note claimed the C++ source is not shipped and that everything here was
  inferred empirically; both halves of that are now superseded by the driven
  messages.
- TRT-LLM routes bf16 linears to this op on sm>=100 because cuBLASLt picks
  single-pass cluster-mode kernels for small-M (decode) gemms there; the
  op itself is not gated on arch.

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
