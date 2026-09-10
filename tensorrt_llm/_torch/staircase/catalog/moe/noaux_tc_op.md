---
receipts:
  sm_100: {status: passed, trtllm: 1.3.0rc21}
  sm_103: {status: passed, trtllm: 1.3.0rc26, tests: 16}
---

# noaux_tc_op

**Wraps** `torch.ops.trtllm.noaux_tc_op` (one call).

## Semantics

DeepSeek-V3 style MoE routing (`noaux_tc` is DeepSeek's `topk_method` name for
auxiliary-loss-free routing with a bias correction term): per-token expert
selection by a *bias-corrected* sigmoid score, with combine weights taken from
the *uncorrected* sigmoid.

Let `L` be one row of `router_logits` (`num_experts` values) and `b` the
per-expert correction bias. For every token, with the precision of each step
annotated:

```
s      = sigmoid(L)                       # fp32; see "The kernel's sigmoid"
c      = s + b                            # fp32; SELECTION SCORE ONLY

# group stage — active only when n_group > 1 (identity at n_group == 1):
#   split c into n_group contiguous groups of num_experts/n_group experts
#   gscore[g] = sum of the two largest c inside group g
#   keep the topk_group groups with the largest gscore, set c to -inf elsewhere

ids    = indices of the topk largest c, ordered by decreasing c   # int32
w      = s[ids]                           # from BARE s — the bias is NOT in here
tot    = sum(w)                           # fp32 accumulation
w      = fp64(w) / (fp64(tot) + eps) * routed_scaling_factor    # fp64
```

The call returns `(w, ids)` — **weights first, ids second**.

The renormalization and the scaling are evaluated in **fp64** — pinned by
observation: an all-fp32 reference disagrees with the kernel on a third of the
elements at `routed_scaling_factor = 2.5`, an fp64 one is bit-exact. Only the
sum `tot` is accumulated in fp32. `eps` is a guard term that keeps a row whose
selected scores are all exactly zero at exactly zero instead of `NaN`; HF uses
`1e-20` and that value reproduces the kernel everywhere, but the constant
itself is not observable from outside (this kernel's sigmoid returns either
exactly `0` or at least `2^-25`, so any `eps <= 1e-20` behaves identically).

Three consequences that a reader of the argument name `scores` would get
wrong, each verified here:

- The first argument is **raw router logits**, not scores. The sigmoid is
  applied *inside* the call. Passing pre-sigmoided values silently routes on
  `sigmoid(sigmoid(x))`.
- The bias enters **selection only**. The returned weights are gathered from
  the bias-free sigmoid, so they are not the values that were ranked.
- The renormalization (`norm_topk_prob: true` in HF configs) and the
  `routed_scaling_factor` multiply are both **done inside**. There is no flag
  to skip either; a model with `norm_topk_prob: false` cannot use this op.
  Each returned row sums to `routed_scaling_factor` — except a row whose
  selected scores all saturated to zero, which sums to 0 (see below).

**The kernel's sigmoid.** The kernel evaluates `0.5 * tanh(0.5 * x) + 0.5` in
fp32, not `1 / (1 + exp(-x))`. The two are algebraically identical, and over
the logit range a router actually produces (`|x| <= 8`) they agree well inside
dtype tolerance — a `torch.sigmoid` reference reproduces this op there, which
is what makes it a drop-in for the HF DeepSeek-V3 gate. But the tanh form
saturates early: it returns **exactly 1.0** for `x >= ~17` and **exactly 0.0**
for `x <= ~-18.5`, and its relative error against the exponential form grows
through the negative tail (measured 4.3e-5 at `x = -8`, 2.6e-2 at `x = -15`).
A row whose *selected* experts all sit below `~-18.5` therefore comes back as
all-zero weights (the `eps` turns `0/0` into `0`, not `NaN`) where the
exponential form would return `routed_scaling_factor / topk` each. A torch
reference built on `0.5 * torch.tanh(0.5 * x) + 0.5` reproduced this op at
every shape, dtype and logit range in the test — out to `|logits| ~ 40`.

**Ordering and ties.** Within a row the slots are sorted by decreasing
selection score `c`, and equal `c` values are ordered by *increasing* expert
index — i.e. the result equals a stable descending sort of `c` truncated to
`topk`. This differs from `torch.topk`, which on this machine emits the
*larger* index first among equals; the difference is visible whenever the
logits are bf16/fp16 (exact ties are common) or the bias is coarse. Note the
returned *weights* are therefore **not** monotone: they are `s` read out in
`c` order.

**Fusion boundary.** The call is routing only. The router GEMM
(`hidden_states @ gate.weight.T`) that produces `router_logits` is the
caller's, and so is everything downstream: no expert permutation, no expert
GEMMs, no combine, no shared-expert branch. The two returned tensors are
shaped and typed as the `topk_weights` / `topk_ids` pair that trtllm's
pre-routed MoE runners take (see *Notes* for the dtype those runners demand).

## Signature

```python
def noaux_tc_op(
    router_logits: torch.Tensor,
    bias: torch.Tensor,
    n_group: int,
    topk_group: int,
    topk: int,
    routed_scaling_factor: float,
) -> tuple[torch.Tensor, torch.Tensor]
```

The wrapper renames the op's first two schema arguments (`scores`, `bias`) to
`router_logits`, `bias`; they are passed positionally and unchanged.

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `router_logits` | `[num_tokens, num_experts]` | bf16 / fp16 / fp32 | **contiguous** (see Preconditions) | CUDA |
| `bias` | `[num_experts]` | bf16 / fp16 / fp32, independent of `router_logits` except one combination (see Preconditions) | **contiguous** | CUDA, same device as `router_logits` |
| `n_group` | scalar | Python int, `>= 1` | — | — |
| `topk_group` | scalar | Python int | — | — |
| `topk` | scalar | Python int, `0 <= topk <= min(32, num_experts)` | — | — |
| `routed_scaling_factor` | scalar | Python float (any finite value, including 0 and negative) | — | — |
| returns `[0]` (`topk_weights`) | `[num_tokens, topk]` | **`router_logits.dtype`** | contiguous, newly allocated | CUDA (same device as `router_logits`) |
| returns `[1]` (`topk_ids`) | `[num_tokens, topk]` | **int32** | contiguous, newly allocated | CUDA (same device as `router_logits`) |

The weight dtype follows `router_logits`, **not** `bias` — a bf16 logits /
fp32 bias call returns bf16 weights. Neither input is mutated (verified
bitwise) and neither is retained; both outputs are fresh allocations.

## Metadata consumed

None. Stateless — no attention metadata, no cache, no workspace, no
process-global tuner state. Two identical calls return bitwise identical
tensors, and the call runs on the ambient CUDA stream (verified: an
alternate-stream call returns the same result).

## Preconditions

- `router_logits` is a **2D** CUDA tensor. 1D and 3D inputs raise
  `RuntimeError: scores must be a 2D Tensor`; a CPU tensor raises
  `NotImplementedError` from the dispatcher.
- `bias` is a **1D** CUDA tensor with exactly `num_experts` elements, on the
  same device as `router_logits`. Anything else raises `RuntimeError: bias
  must be 1D with length == number of experts`; a CPU bias with CUDA logits
  raises `RuntimeError: scores and bias must be CUDA tensors`. The bias is
  mandatory — there is no "no bias" mode; pass a zero tensor.
- **Both tensors must be contiguous. The kernel ignores strides**: it
  addresses `router_logits` as a dense `[num_tokens, num_experts]` row-major
  buffer and `bias` as a dense `[num_experts]` buffer, each from `data_ptr()`.
  A column slice of a wider buffer, a transposed view, or a strided bias is
  silently routed on the wrong elements and returns a plausible-looking wrong
  answer — it never raises. The wrapper asserts both; call `.contiguous()`
  first (e.g. when the router logits are a slice of a fused projection).
- Dtypes: `router_logits` and `bias` are each bf16, fp16 or fp32. Anything
  else raises `ValueError: Invalid dtype, only supports float16, float32, and
  bfloat16` (or the `Invalid bias dtype` variant). The two are independent
  **except** that `router_logits` fp16 + `bias` bf16 is rejected with `Invalid
  bias dtype`, even though the corresponding kernel is compiled into this
  build — an op-level dispatch gap, not a kernel limit. The other eight
  combinations all work.
- `0 <= topk <= 32`. `topk > 32` raises `RuntimeError: topk should be smaller
  than or equal to 32 for now`; a negative `topk` raises from the output
  allocation. `topk == 0` is accepted and returns two `[num_tokens, 0]`
  tensors.
- **`topk <= num_experts` is not enforced by the op.** With `topk >
  num_experts` the kernel reads past the end of each row and emits expert ids
  `>= num_experts` at non-zero weight (an out-of-bounds read; the values are
  meaningless but **finite and deterministic** — at `num_experts = 3,
  topk = 16` eight independent processes, three of them churning the caching
  allocator first, each returned 104 of 128 ids out of range, identical
  weights, and no NaN, so the read stays inside the logits tensor's own
  buffer rather than picking up allocator residue). It never raises. The
  wrapper asserts this.
- `num_experts % n_group == 0`, else `RuntimeError: num_experts should be
  divisible by n_group`. `n_group <= 32`, else `RuntimeError: n_group should
  be smaller than or equal to 32 for now`.
- **`n_group >= 1` is not checked.** `n_group == 0` reaches a host-side
  `num_experts % n_group` and raises **SIGFPE, killing the process** — not a
  Python exception, so it cannot be caught. The wrapper does not guard it
  (guarding would require a non-metadata policy call); callers must never
  pass 0.
- Configuration support, enforced inside the kernel with `RuntimeError:
  [TensorRT-LLM][ERROR] Assertion failed: invokeNoAuxTc: unsupported
  configuration (n_group=..., num_experts=..., topk_group=..., topk=...)`:
  - `n_group == 1` (the ungrouped case): requires only `num_experts <= 1024`.
    `topk_group` is then **completely ignored** — values `0, 1, 2, 5, 32, 100`
    all give bitwise identical results. `num_experts = 1025` is rejected.
  - `n_group > 1`: requires `1 <= topk_group <= n_group`, `topk <= 8`,
    `num_experts <= 256`, and `num_experts / n_group <= 32` (experts per
    group). `topk_group == 0` is *accepted* without error at `n_group > 1` but
    is not characterized by this entry — do not use it.
  - `topk_group == n_group` keeps every group and is bitwise identical to
    `n_group = 1`.
- `num_tokens == 0` is accepted and returns empty `[0, topk]` tensors.

A caller violating none of the above gets the result described under
*Semantics*.

## Notes

- **Certified surface.** Passing on sm_100 / trtllm 1.3.0rc21: `num_tokens`
  in `{0, 1, 2, 4, 7, 8, 16, 64, 128, 256, 512, 1024, 2048, 4096, 8192}`;
  `num_experts` in `{1, 2, 7, 8, 16, 32, 64, 72, 100, 128, 256, 257, 512,
  1024}`; `topk` in `{0, 1, 2, 3, 4, 6, 8, 16, 31, 32}`; all eight accepted
  (logits, bias) dtype pairs; `routed_scaling_factor` in `{-1, 0, 0.5, 1, 2,
  2.5, 3, 1000}`; ungrouped and the grouped configurations `(num_experts,
  n_group, topk_group, topk)` = `(256,8,4,8)`, `(128,4,2,6)`, `(72,8,2,6)`,
  `(72,4,2,6)`, `(64,8,4,8)`.
- **Numerics: the reference of *Semantics* is not an approximation, it is the
  kernel.** Expert ids matched exactly everywhere. Weights were bit-identical
  except for a last-bit rounding step that appears when the kernel's fp32
  accumulation order for `tot` differs from `torch.sum`'s — measured at 1
  element in 49152 (8192 tokens x 72 experts, bf16) and 4 in ~10^7 over a
  soak. Every element stayed within **one ulp** of the reference, so this
  entry's test gates at `rtol = finfo(dtype).eps, atol = 0` (an order of
  magnitude tighter than `assert_close`'s defaults for fp32) plus a
  bit-exactness budget. Getting the fp64 detail of *Semantics* wrong is not a
  rounding difference: an all-fp32 normalization misses a third of the
  elements at `routed_scaling_factor = 2.5`.
- **The `register_fake` meta function disagrees with the kernel.** Under
  `FakeTensorMode` (torch.compile / export / AD tracing) the first output is
  given `bias.dtype`; eager execution gives `router_logits.dtype`. They differ
  whenever the two input dtypes differ — e.g. fp32 logits with a bf16 bias
  traces as bf16 and runs as fp32. Eager callers are unaffected. This is an
  upstream defect in this build, not a behaviour to rely on.
- **Downstream dtype match.** `torch.ops.trtllm.fp4_block_scale_moe_runner`,
  the NVFP4 pre-routed MoE runner, was observed here to reject a fp32
  `topk_weights` (`RuntimeError: topk_weights must be bfloat16.`) and an int64
  `topk_ids` (`RuntimeError: topk_ids must be int`), and to accept the
  bf16/int32 pair. Since this op's weight dtype follows `router_logits`,
  feeding it **bf16** router logits produces a directly consumable pair with
  no cast. (Catalog membership of that runner is `index.yaml`'s fact alone.)
- Sigmoid, bias correction, group scoring and selection are computed in fp32
  regardless of the input dtype (the normalization then steps up to fp64);
  only the final weights are cast back to `router_logits.dtype`.
- `-inf` logits map to score `0` exactly, like any logit below `~-18.5`, so a
  `-inf`-masked expert loses to every expert with a positive score (verified:
  none was selected while `topk` fit in the unmasked experts). It is *not* a
  hard mask — with more slots than positively-scored experts the masked ones
  come back at weight 0. `+inf` saturates to 1 like any logit above `~17`.
  `NaN` logits *are* selected and yield `NaN` weights.
- A hand-written PyTorch path with the same semantics exists in this build
  (`tensorrt_llm._torch.modules.fused_moe.routing.Deepseekv3RoutingImpl`) and
  is what trtllm falls back to outside the supported configurations above; it
  uses `torch.sigmoid`, so it differs from this op in the saturating tails and
  in tie-breaking.
- Related routing ops in this build: `torch.ops.trtllm.renorm_moe_routing_op`
  and `torch.ops.trtllm.default_moe_routing_op` (softmax routing, no bias),
  and the block-scale MoE runners' built-in `routing_method_type = 2`
  (DeepSeekV3) path, which is a different code path from this op. Catalog
  membership is `index.yaml`'s fact alone.
