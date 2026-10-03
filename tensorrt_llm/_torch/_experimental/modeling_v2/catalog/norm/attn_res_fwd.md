---
receipts: {}
---

# attn_res_fwd

**Wraps** `torch.ops.trtllm.attn_res_fwd` (one call).

## Semantics

Attention-residual selection: Kimi K3's per-token mixing of residual-stream snapshots. For every token the op
scores `N = K + 1` candidate rows -- the `K` snapshots of `block_residual` in their stored order, then
`layer_residual` -- and returns the softmax-weighted sum of the raw candidates.

For token `t` (`B == 1`) and candidate `n`, every step in fp32 (`H` is the hidden size):

```
V[n]      = block_residual[n, t, 0, :]      for n < K
V[K]      = layer_residual[t, 0, :]
q         = fp32(rms_weight) * fp32(res_weight)       # elementwise; never rounded to bf16
rsigma[n] = rsqrt(sum_h V[n, h]^2 / H + rms_eps)
logits[n] = rsigma[n] * sum_h V[n, h] * q[h]
probs     = softmax(logits)                           # over the token's N candidates
output    = bf16_rn(sum_n probs[n] * V[n])            # the only rounding
```

`logits[n]` is the dot product of `res_weight` with the RMSNorm of `V[n]` under weight `rms_weight`, the
normalized row kept in fp32, as in HF Kimi's `_apply_attn_res`. The RMSNorm only scores the candidates: `output`
mixes the raw rows, not their normalized form. `rsigma`, `probs` and `logits` are returned as the fp32 values the
formulas name. With `N == 1` the single probability is 1, and the formulas return `layer_residual` as `output`.

The kernels are compiled with `--use_fast_math`: `rsqrt`, the softmax exponentials and the reciprocal of the
softmax denominator are hardware approximations, denormals flush to zero, and every sum runs in the kernel's own
order. The results are close to an fp32 torch evaluation of the formulas, not bit-identical to it.

Fusion boundary: selection only. No residual add, no trailing RMSNorm (`attn_res_rmsnorm_fwd` fuses the norm that
follows), and no snapshot write: the caller owns the snapshot bank, including appending the running residual to
it.

## Signature

```python
def attn_res_fwd(
    layer_residual: torch.Tensor,
    block_residual: torch.Tensor,
    res_weight: torch.Tensor,
    rms_weight: torch.Tensor,
    rms_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
```

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `layer_residual` | `[T, 1, H]` | bf16 | contiguous | CUDA |
| `block_residual` | `[K, T, 1, H]`, `K = N - 1 >= 0` | bf16 | contiguous | `layer_residual`'s (not checked) |
| `res_weight` | `H` elements (`[H]`, `[1, H]` or `[H, 1]`) | bf16 | contiguous | `layer_residual`'s (not checked) |
| `rms_weight` | `H` elements | bf16 | contiguous | `layer_residual`'s (not checked) |
| `rms_eps` | scalar | Python float, passed as fp32 | — | — |
| returns `[0]` (`output`) | `[T, 1, H]` | bf16 | contiguous, newly allocated | `layer_residual`'s |
| returns `[1]` (`rsigma`) | `[N, T, 1]` | fp32 | contiguous, newly allocated | `layer_residual`'s |
| returns `[2]` (`probs`) | `[N, T, 1]` | fp32 | contiguous, newly allocated | `layer_residual`'s |
| returns `[3]` (`logits`) | `[N, T, 1]` | fp32 | contiguous, newly allocated | `layer_residual`'s |

On the candidate axis of `rsigma`, `probs` and `logits`, index `n < K` is snapshot `n` and index `K` is
`layer_residual`. `K = 0` is passed as a zero-size `[0, T, 1, H]` tensor. No input is mutated and no output
aliases an input. `res_weight` is the flattened `[1, H]` weight of the scoring `nn.Linear(H, 1)` projection;
`rms_weight` is the scoring RMSNorm's weight.

### Certified arguments

- `H = 7168` (Kimi K3), `B = 1`, bf16 inputs, `res_weight` and `rms_weight` of shape `[H]`.
- Every `N = 2 ... 9` -- the candidate counts Kimi K3 produces: its 93 layers append a snapshot every 12 layers, so
  the bank holds at most `ceil(93 / 12) = 8` snapshots -- at `T = 1 ... 8` (decode, including multi-token
  speculative steps) and at `T = 300` and `T = 2048` (prefill).
- The remaining dispatch branches and `N` extremes: `N = 1, 10, 11, 12` at `T = 1`; `N = 1, 12` at `T = 8`;
  `N = 12` at `T = 1024`.
- `rms_eps = 1e-6` in every cell; `1e-5` and `1e-2` in four cells that cover the three kernels. `1e-2` exceeds the
  test inputs' mean square of `2.5e-3`, so it pins where `rms_eps` enters.

## Metadata consumed

None. Stateless: no attention metadata, no workspace, no caller-visible state. The op caches per device the SM
count, the architecture check and the one-time raise of a kernel's dynamic shared-memory limit, and reads
`TRTLLM_ENABLE_PDL` once per process. These select launch parameters, never results.

## Preconditions

Each item is a `TORCH_CHECK` in `cpp/tensorrt_llm/thop/attnResOp.cpp`, evaluated in this order; a failure raises
`RuntimeError` with the quoted message. This entry's test does not exercise the rejections.

1. `layer_residual.dim() == 3` and `block_residual.dim() == 4` (`attn_res_fwd: layer_residual must be [T, B, H]`,
   `attn_res_fwd: block_residual must be [K, T, B, H]`).
2. All four tensors are CUDA tensors (`attn_res_fwd: all input tensors must be CUDA tensors`).
3. `layer_residual`'s device has compute capability `10.x` (`attn_res_fwd requires an sm_100-family (datacenter
   Blackwell) GPU`).
4. `B == 1`; `1 <= N <= 12` with `N = block_residual.shape[0] + 1`; `1 <= T <= 16384`; `H` a multiple of 1024 in
   `[4096, 8192]` (`attn_res_fwd: unsupported B=...`, `N=...`, `T=...`, `H=...`).
5. All four tensors are bf16 (`attn_res_fwd: <name> must be bf16`) and contiguous (`attn_res_fwd: inputs must be
   contiguous`). A strided view raises; it is never read as if it were dense.
6. `block_residual.shape == (N - 1, T, B, H)` (`attn_res_fwd: block_residual shape must match layer_residual`).
7. `res_weight.numel() == H` and `rms_weight.numel() == H` (`attn_res_fwd: <name> must have H elements`).

Not checked: that `block_residual`, `res_weight` and `rms_weight` are on `layer_residual`'s device. The kernels
dereference all four on that device, so the caller must keep them together (`attn_res_rmsnorm_fwd` checks it).

## Notes

- **Code paths.** The op picks a kernel from `(T, N)`. At `H = 7168`:

  | `T`, `N` | Kernel | Launch |
  |---|---|---|
  | `T == 1`, `N` in {1, 2, 4} | single-CTA decode kernel | 1 CTA of 256 threads |
  | `T == 1`, `N` in {8, 12} | split-K decode kernel | 1 cluster of 8 CTAs of 256 threads |
  | `T == 1024`, `N == 12` | online kernel, fixed-`N = 12` variant | persistent: (SM count - 1) CTAs of 288 threads |
  | every other `(T, N)` | online kernel | persistent: one CTA of 288 threads per SM |

  The single-CTA kernel gives each of its 256 threads 28 of the 7168 elements and keeps every candidate's values
  for them in registers. The split-K kernel gives each CTA of the cluster 896 of the 7168 elements, keeps them in
  shared memory as fp32, and exchanges per-candidate sums through distributed shared memory. The online kernel is
  warp-specialized: one warp streams candidate rows into shared memory with bulk asynchronous copies, eight warps
  score them and keep the fp32 rows in Tensor Memory until the mixing pass, the softmax runs online over chunks of
  four candidates, and each CTA loops over tokens. Other hidden sizes run the online kernel with other tilings, or
  a row-tiled kernel at `N == 1` for `H` 4096 and 8192; none of them is certified here.
- **Architecture.** The op requires compute capability `10.x`, and the kernels are built only for the sm_100
  family target (`100f`). They use sm_100 instructions: paired fp32 arithmetic (`.f32x2`) in all three kernels,
  Tensor Memory (`tcgen05`) and bulk asynchronous copies in the online kernel, which also takes 115,040 bytes of
  dynamic shared memory per CTA at `H = 7168` (one CTA per SM), and an 8-CTA thread-block cluster with distributed
  shared memory in the split-K kernel.
- **PDL.** `TRTLLM_ENABLE_PDL` (read once per process; unset or `1` enables it, `0` disables it) controls
  programmatic dependent launch for the two decode kernels. When it is enabled they launch with programmatic
  stream serialization, wait on the grid dependency (`cudaGridDependencySynchronize`) before reading any input,
  and trigger their dependents (`cudaTriggerProgrammaticLaunchCompletion`) after their last store. The online
  kernel launches without the attribute: it starts after its predecessor completes and releases its dependents
  when it completes. A dependent kernel launched with PDL must wait on the grid dependency before it reads any
  output: the trigger only lets it start, and only the wait makes this op's stores visible. PDL changes scheduling,
  not arithmetic.
- **Determinism.** Each kernel reduces in a fixed order without atomics, so identical inputs give identical bits
  run to run; the test asserts it in every cell. The kernels' summation orders differ from each other, so a token
  that a decode kernel computes in a `T == 1` call can come out different in the last bits when the online kernel
  computes it inside a larger call. The online kernel's per-token arithmetic depends on neither `T` nor the grid
  size, except that its fixed-`N = 12` variant sums the softmax denominator in a different order.
- **Tolerance.** The test compares `output` and each fp32 output with an fp32 torch evaluation of *Semantics*,
  using the metric of main's op test (`tests/unittest/_torch/modules/kimi_k3_attn_res/test_attn_res_op.py`):
  cosine similarity above 0.999 and relative L2 error below 3e-2.
- **CUDA graphs.** The call does not synchronize with the host, and its launch configuration depends on shapes
  only. The test captures one call per kernel after an eager warm-up and checks that replay reproduces the eager
  bits.
- **Kimi K3 call sites.** `modeling_kimi_linear.py` calls this op, followed by a separate RMSNorm, where the fused
  `attn_res_rmsnorm_fwd` is not taken: above `KIMI_K3_FUSED_ATTN_RES_MAX_TOKENS` tokens (1 by default) unless
  `KIMI_K3_ATTN_RES_TOPOLOGY` routes the call to the persistent fused op, and on the layers whose pre-norm mixture
  the speculative drafter captures. The `register_fake` in `custom_ops/cpp_custom_ops.py` reports the shapes and
  dtypes above.
