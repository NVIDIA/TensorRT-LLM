---
receipts:
  sm_100: {status: passed, tests: 6}
---

# attn_res_rmsnorm_fwd

**Wraps** `torch.ops.trtllm.attn_res_rmsnorm_fwd` (one call).

## Semantics

Attention-residual selection followed by the RMSNorm that consumes it, in one kernel. The selection is
`attn_res_fwd`'s: for every token, a softmax over `N = K + 1` candidate rows -- the `K` snapshots of
`block_residual` in their stored order, then `layer_residual` -- of fp32 logits, mixing the raw candidates. The
fused norm then normalizes that mixture.

For token `t` (`B == 1`), every step in fp32 unless rounded explicitly (`H = 7168`):

```
V[n]      = block_residual[n, t, 0, :]      for n < K
V[K]      = layer_residual[t, 0, :]
q         = fp32(rms_weight) * fp32(res_weight)       # elementwise; never rounded to bf16
rsigma[n] = rsqrt(sum_h V[n, h]^2 / H + rms_eps)
logits[n] = rsigma[n] * sum_h V[n, h] * q[h]
probs     = softmax(logits)                           # over the token's N candidates
mixed     = bf16_rn(sum_n probs[n] * V[n])            # attn_res_fwd's output
r         = rsqrt(sum_h mixed[h]^2 / H + output_rms_eps)
normed    = bf16_rn(mixed * r)
output    = bf16_rn(normed * fp32(output_rms_weight))
```

The op keeps two bf16 rounding boundaries:

- `mixed` is rounded to bf16 before the norm reads it, as when `attn_res_fwd` and a separate RMSNorm run in turn.
- The norm rounds twice, in `KimiK3RMSNorm`'s order: the normalized value is rounded to bf16, then multiplied by
  the bf16 weight and rounded again. An RMSNorm that rounds once (normalize and scale in fp32, then cast, as the
  `flashinfer_rmsnorm` entry describes) differs from this op by one unit in the last bf16 place in a large share of
  the elements: about a fifth of them when both formulas are evaluated in torch on this entry's test inputs.

As in `attn_res_fwd`, the kernels are compiled with `--use_fast_math`: `rsqrt`, the softmax exponentials and the
reciprocal of the softmax denominator are hardware approximations, denormals flush to zero, and every sum runs in
the kernel's own order. The output is close to an fp32 torch evaluation of the formulas, not bit-identical to it.

Fusion boundary: the selection and the RMSNorm that follows it. The op returns only the normalized output -- not
the pre-norm mixture and not `attn_res_fwd`'s `rsigma`, `probs` or `logits`. No residual add (the separate op
`attn_res_add_rmsnorm_fwd` adds one first).

## Signature

```python
def attn_res_rmsnorm_fwd(
    layer_residual: torch.Tensor,
    block_residual: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    output_rms_weight: torch.Tensor,
    rms_eps: float,
    output_rms_eps: float,
) -> torch.Tensor
```

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `layer_residual` | `[T, 1, 7168]` | bf16 | contiguous | CUDA |
| `block_residual` | `[K, T, 1, 7168]`, `K = N - 1 >= 0` | bf16 | contiguous | `layer_residual`'s |
| `res_weight` | 7168 elements (`[H]`, `[1, H]` or `[H, 1]`) | bf16 | contiguous | `layer_residual`'s |
| `rms_weight` | 7168 elements | bf16 | contiguous | `layer_residual`'s |
| `output_rms_weight` | 7168 elements | bf16 | contiguous | `layer_residual`'s |
| `rms_eps` | scalar | Python float, passed as fp32 | — | — |
| `output_rms_eps` | scalar | Python float, passed as fp32 | — | — |
| returns | `[T, 1, 7168]` | bf16 | contiguous, newly allocated | `layer_residual`'s |

`res_weight` and `rms_weight` score the candidates, with `rms_eps`; `output_rms_weight` and `output_rms_eps`
belong to the trailing norm. `K = 0` is passed as a zero-size `[0, T, 1, 7168]` tensor. No input is mutated and the
output aliases no input.

### Certified arguments

- `H = 7168`, `B = 1`, bf16 inputs, `res_weight`, `rms_weight` and `output_rms_weight` of shape `[H]`.
- Every `N = 2 ... 9` -- the candidate counts Kimi K3 produces: its 93 layers append a snapshot every 12 layers, so
  the bank holds at most `ceil(93 / 12) = 8` snapshots -- at `T = 1 ... 8` (decode, including multi-token
  speculative steps) and at `T = 32` and `T = 300`.
- `N = 1, 10, 11, 12` at `T = 1` and `T = 8`.
- `rms_eps = output_rms_eps = 1e-6` in every cell; the pairs `(1e-5, 1e-5)`, `(1e-2, 1e-6)` and `(1e-6, 1e-2)` in
  three cells that cover both kernels. `1e-2` exceeds the mean square of both the test inputs and their mixture,
  so the two unequal pairs pin which norm each eps feeds.

## Metadata consumed

None. Stateless: no attention metadata, no workspace, no caller-visible state. The op caches per device the
architecture check and the one-time raise of the split-K kernel's dynamic shared-memory limit, and reads
`TRTLLM_ENABLE_PDL` once per process. These select launch parameters, never results.

## Preconditions

Each item is a `TORCH_CHECK` in `cpp/tensorrt_llm/thop/attnResOp.cpp`, evaluated in this order; a failure raises
`RuntimeError` with the quoted message. This entry's test does not exercise the rejections.

1. `layer_residual.dim() == 3` and `block_residual.dim() == 4` (`attn_res_rmsnorm_fwd: layer_residual must be
   [T, B, H]`, `attn_res_rmsnorm_fwd: block_residual must be [K, T, B, H]`).
2. All five tensors are CUDA tensors on one device (`attn_res_rmsnorm_fwd: all input tensors must be CUDA
   tensors`, `... must be on the same CUDA device`).
3. The device has compute capability `10.x`; `B == 1`; `1 <= N <= 12` with `N = block_residual.shape[0] + 1`;
   `1 <= T <= 16384`; `H` a multiple of 1024 in `[4096, 8192]`. This is the contract check `attn_res_fwd` shares,
   and its messages name that op: `attn_res_fwd requires an sm_100-family (datacenter Blackwell) GPU`,
   `attn_res_fwd: unsupported B=...`, `N=...`, `T=...`, `H=...`.
4. `H == 7168` (`attn_res_rmsnorm_fwd: requires B=1 and H=7168, got T=... B=... H=...`): the kernels derive their
   per-thread tiling from `H` at compile time.
5. All five tensors are bf16 (`attn_res_rmsnorm_fwd: <name> must be bf16`) and contiguous
   (`attn_res_rmsnorm_fwd: inputs must be contiguous`). A strided view raises; it is never read as if it were dense.
6. `block_residual.shape == (N - 1, T, B, H)` (`attn_res_rmsnorm_fwd: block_residual shape must match
   layer_residual`).
7. `res_weight`, `rms_weight` and `output_rms_weight` each have `H` elements (`attn_res_rmsnorm_fwd: <name> must
   have H elements`).

The kernels take any token count as a grid dimension, but the shared contract check still caps `T` at 16384.

## Notes

- **Code paths.** The kernel depends on `N` only, never on `T`:

  | `N` | Kernel | Launch |
  |---|---|---|
  | 1 ... 4 | single-CTA kernel | `T` CTAs of 256 threads, one per token |
  | 5 ... 12 | split-K kernel | `T` clusters of 8 CTAs of 256 threads, one per token |

  They are the templates of `attn_res_fwd`'s two decode kernels, instantiated with the trailing norm for every `N`
  (`attn_res_fwd` itself runs them only at `T == 1`, for five `N` values). The single-CTA kernel gives each of its
  256 threads 28 of the 7168 elements and keeps every candidate's values for them in registers as bf16. In the
  split-K kernel each CTA of the cluster owns 896 of the 7168 elements, keeps them in shared memory as fp32, and
  exchanges the per-candidate sums, then the mixture's sum of squares, through distributed shared memory. This op
  has no persistent path; the persistent fused kernel is the separate op `attn_res_add_rmsnorm_persistent_fwd`.
- **Architecture.** The op requires compute capability `10.x`, and the kernels are built only for the sm_100
  family target (`100f`). Both use paired fp32 arithmetic (`.f32x2`); the split-K kernel also uses an 8-CTA
  thread-block cluster with distributed shared memory and takes `3,652 * N` bytes of dynamic shared memory per CTA.
- **PDL: the output is written after the dependents are released.** `TRTLLM_ENABLE_PDL` (read once per process;
  unset or `1` enables it, `0` disables it) controls programmatic dependent launch. When it is enabled both
  kernels launch with programmatic stream serialization. Every CTA first waits on the grid dependency
  (`cudaGridDependencySynchronize`), so the op reads its inputs only after the preceding kernel has completed, and
  then immediately triggers its dependents (`cudaTriggerProgrammaticLaunchCompletion`), before it computes
  anything. A dependent kernel launched with PDL can therefore start while this op is still writing `output`: it
  must wait on the grid dependency before it reads `output` or overwrites any of this op's inputs. A kernel
  launched without the PDL attribute is ordered by the stream as usual. With PDL disabled the kernels launch
  normally and contain no grid-dependency instructions. PDL changes scheduling, not arithmetic. The test chains two
  calls, the second reading the first's output as its `layer_residual`, and checks the chained result bit for bit
  against the same call on a settled copy.
- **Determinism.** Every token is computed by its own CTA or cluster, with the same code at every `T`, reducing in
  a fixed order without atomics. Identical inputs give identical bits run to run, which the test asserts in every
  cell, and a token's result does not depend on `T` or on its position in the call.
- **Tolerance.** The test compares the output with an fp32 torch evaluation of *Semantics*, both rounding
  boundaries included, using the metric of main's op test
  (`tests/unittest/_torch/modules/kimi_k3_attn_res/test_attn_res_rmsnorm_op.py`): cosine similarity above 0.9999
  and relative L2 error below 5e-3.
- **CUDA graphs.** The call does not synchronize with the host, and its launch configuration depends on shapes
  only. The test captures one call per kernel after an eager warm-up and checks that replay reproduces the eager
  bits.
- **Kimi K3 call sites.** `modeling_kimi_linear.py` uses this op for the attention-residual + RMSNorm pairs before
  attention, before the MLP on the layers that append a snapshot, and at the model output, for calls of at most
  `KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS` tokens (1 by default, 32 when `KIMI_K3_ATTN_RES_TOPOLOGY` is on) that the
  persistent topology does not take. Layers whose pre-norm mixture the speculative drafter captures run
  `attn_res_fwd` and a separate norm instead. The `register_fake` in `custom_ops/cpp_custom_ops.py` returns
  `empty_like(layer_residual)`.
