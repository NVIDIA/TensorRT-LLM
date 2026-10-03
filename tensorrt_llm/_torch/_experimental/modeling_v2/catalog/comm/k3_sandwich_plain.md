---
receipts:
  sm_100: {status: pending, world_size: 4}
---

# k3_sandwich_plain

**Wraps** `torch.ops.trtllm.k3_sandwich_plain` (one call).

## Semantics

A row-parallel projection, its TP all-reduce, the residual add and an RMSNorm in one kernel, for a decode batch of at
most 8 tokens: Kimi K3's drafter (DSpark) layers' attention output projection, or with `swiglu` their MLP's
SiLU-and-mul and down projection. Every rank of the workspace's TP group calls with its own `x` and weight slice and
the same `residual`; every rank gets back the same two tensors. Per token, with `W` ranks and `K` the slice's width:

```
a_r       = x_r, or with swiglu silu_and_mul(x_r) = bf16(silu(x_r[:, :K]) * x_r[:, K:])   # gate columns first
partial_r = bf16(a_r @ weight_r^T)                                   # this rank's [M, 7168] share, fp32 accumulator
updated   = bf16(residual + bf16(sum over the W ranks of partial_r))
normed    = bf16(updated * rsqrt(mean over the 7168 columns of updated^2 + eps) * norm_weight)
```

and returns `(normed, updated)`. The arithmetic and summation order are those of the all-reduce the call replaces,
the MNNVL one-shot with `AllReduceFusionOp.RESIDUAL_RMS_NORM` (`kARResidualRMSNorm`): ranks summed in chunks of 8 in
rank order, the residual added to the rounded sum, the mean square taken over bf16-rounded squares in that kernel's
reduction tree, `normed` rounded once — so the outputs are bit-identical to the projection followed by that
all-reduce (the kernel's statement). With `swiglu`, `silu_and_mul` follows `k3_ctm_gemv_swiglu` (fp32, the sigmoid
as the correctly rounded reciprocal of `1 + exp(-gate)`, one bf16 rounding), and the projection accumulates the even
and the odd k-tiles in two fp32 accumulators added as `(0 + even) + odd` before its rounding, split 2's order (the
kernel's statement). The op's kernel test checks the plain form against `k3_ctm_gemv` (split 1) and the SwiGLU form
against `k3_ctm_gemv_swiglu` (split 2), each followed by the MNNVL one-shot RESIDUAL_RMS_NORM all-reduce, bit for
bit. The result is bitwise identical on every rank (certified, every call of the test).

Fusion boundary. Inside: (with `swiglu`) the SiLU-and-mul, the projection, the all-reduce, the residual add, the
RMSNorm. Outside: what produced `x` (the drafter's attention, or its gate_up projection), chaining `residual` from
call to call, and what consumes `normed`. The drafter's calls run on the workspace of the target's sandwiches
(`comm/k3_sandwich_oproj`, `comm/k3_sandwich_tail`).

## Signature

```python
def k3_sandwich_plain(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    workspace: K3SandwichWorkspace,
    swiglu: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]
```

### Certified arguments

| Argument | Shape | Dtype | Layout | Device |
|---|---|---|---|---|
| `x` | `[M, 384]`, `M` 1-8; with `swiglu` `[M, 1792]` (gate first) | bf16 | contiguous | CUDA, this rank's device |
| `weight` | `[7168, 384]`; with `swiglu` `[7168, 896]` (this rank's slice) | bf16 | contiguous | CUDA |
| `residual` | `[M, 7168]` | bf16 | contiguous | CUDA |
| `norm_weight` | `[7168]` | bf16 | contiguous | CUDA |
| `eps` | scalar (certified at 1e-6) | Python float | — | — |
| `workspace` | a `K3SandwichWorkspace` of this rank's TP group (see *State*) | — | — | — |
| `swiglu` | `False` (o_proj, `K` 384) or `True` (down, `K` 896) | bool | — | — |
| returns | `(normed, updated)`, each `[M, 7168]` | bf16 | contiguous, newly allocated | = `x.device` |

Single calls are certified at every `M` in both forms. Without `swiglu` the op also takes `K` any multiple of 128 up
to 896 (its `supports_plain`); only Kimi K3 TP16's 384 is certified. The inputs are read only.

Inert (not exposed by the wrapper): `ipc_order=False`. With it the op follows the IPC one-shot's order instead
(`allreduce_fusion_kernel_oneshot_lamport`: fp32 squares, its own reduction tree; TP <= 8, `ValueError` above), for a
drafter whose all-reduce is the IPC one — another compiled kernel, not certified here.

## State

**Object.** `K3SandwichWorkspace`, the object `comm/k3_sandwich_oproj` and `comm/k3_sandwich_tail` take, defined in
`tensorrt_llm/_torch/cute_dsl_kernels/k3_sandwich/op.py` and re-exported by this entry's wrapper too; one per TP
group, owned by the caller.

**Contents and size.** As in `k3_sandwich_oproj.md`: two alternating halves of `[8 tokens][W ranks][7168]` bf16 per
rank, as int32 words (`0x80000000` = empty), behind one multicast mapping (`uc`, `mc`) — `2 x 8 x W x 7168 x 2`
bytes, 0.92 MB at `W` = 4 and 3.67 MB at `W` = 16 — and `flags`, int32 `[64]`, the call counts of the kernel's 56
CTAs, with the handle that owns the memory and the communicator. This op's partial rows go into the same slots as the
target's sandwiches'. Certified after `create`: sized for the group, every word empty, every counter zero.

**Who creates it, and when.** The target, in `post_load_weights`, with
`K3SandwichWorkspace.create(mapping, fabric_handle=None)`: collective over the TP group (it returns on every rank or
raises on every rank), eager, every word emptied and every counter zeroed before any rank returns. Under CUDA-graph
capture it raises `RuntimeError` before it enters any collective: certified on every rank at once, and on one rank
alone while its peers do not call it. Details in `k3_sandwich_oproj.md`. The drafter does not create its own: it
takes the target's.

**Which ops may share one object.** The three sandwich entries — `comm/k3_sandwich_oproj`, `comm/k3_sandwich_tail`
and this one — take the same object, and in the model the drafter's calls run on the target's workspace of their TP
group: one call sequence. That sharing is certified by `comm/k3_sandwich_oproj`'s matrix: decode steps of target
layers with two drafter layers (this op, then its SwiGLU form) interleaved at the drafter's own token count, eager
with a random rank late and captured with eager calls between replays. The two forms of this op share one sequence
too (certified here: every sequence of this entry's test alternates them). Two workspaces keep independent
counters: calls of both forms alternating between two in an irregular pattern are all correct (certified, 20
calls).

**Call-order invariant.** As in `k3_sandwich_oproj.md`: every rank makes the same sequence of sandwich calls on one
workspace (the same number of calls, the `k`-th with the same op, form and `M`) across layers, decode steps, eager
calls and graph replays; the same order of calls across objects on one stream; and no kernel between two calls that
skips its grid-dependency wait.

**What a later launch reads.** `flags`: every call adds one to every CTA's counter (certified for both forms: all 56
by exactly one per call, the spare words untouched), and the counter's parity before the call selects the half this
call pushes into and polls; and that half's words, which must be empty except for this call's pushes.

**How it is re-armed.** By its readers, as for `k3_sandwich_oproj`: a thread empties every word it read right after
reading it, so no separate clear and no record of the previous call's size is needed (the kernel's statement).

**Why the test drives call sequences.** See `k3_sandwich_oproj.md`. This entry's test runs 11 decode steps of 6
drafter layers (12 calls chained through the residual, the two forms alternating) at `M` = 8, 8, 8, 2, 7, 8, 1, 1, 8,
3, 8, a random rank 5 ms late at every call, each call against the reference.

**What a wrong order does.** Certified by the test's negative control: rank 0 issues two same-shaped calls on one
workspace in swapped order. Nothing raises and nothing hangs, but every rank's two results are wrong (more than half
the `updated` elements differ on every rank). A plain call right after is correct again.

## Metadata consumed

Besides `workspace`, the op module's process-wide compile cache, keyed by (`W`, `K`, order, `swiglu`, PDL): the two
forms are two kernels, each compiled (seconds) on its first call, which must be eager — under capture the op raises
"must run once per configuration outside CUDA-graph capture first". The cache is result-neutral. PDL
(`TRTLLM_ENABLE_PDL`, read at every call, default on) changes scheduling, not results.

## Preconditions

- bf16, contiguous; `M` 1-8; `weight` `[7168, K]` with `K` a multiple of 128 up to 896 and `x` `[M, K]`, or with
  `swiglu` `K` = 896 exactly and `x` `[M, 1792]`; `residual` `[M, 7168]`, `norm_weight` `[7168]`: the op's
  `supports_plain`. Anything else raises `ValueError` on every rank before any launch — certified for `M` = 9, the
  SwiGLU form on a `K` 384 slice and an `x` whose `K` is not the weight's (every counter unchanged) — and the next
  call is correct.
- Every rank calls with the same `M` and form, and the same `residual` and `norm_weight` (the replicated residual
  stream) for the same result on every rank; the call order is the *State* invariant.
- `workspace` was created, and each kernel compiled (one eager call per key), before any capture. Calls may be
  captured: certified with a captured drafter step of 6 layers (12 chained calls, both forms) at `M` = 8 replayed 8
  times with rewritten inputs, an eager call of another `M` on the same workspace between replays, every replayed
  and eager call against the reference.
- Under PDL the kernel launches its dependents once every CTA has pushed, before its outputs are written (the
  kernel's statement): a kernel launched after it reads the outputs only after its own grid-dependency wait.

## Notes

- Certified path: 4 ranks of one GB200 tray (sm_100), one rank per GPU, POSIX-fd handles, Kimi K3 TP16 per-rank
  shapes. Test: `tests/unittest/_torch/modeling_v2/comm/_k3_sandwich_plain_op_matrix.py`, helpers in
  `_k3_sandwich_common.py`. The reference is native torch: `x` and the weights are small multiples of 1/8, 1/16 and
  1/64, and the SwiGLU gates are 0, 32 or 64, on which silu is exact in fp32 (silu(0) = 0; from 32 up,
  `1 + exp(-gate)` rounds to 1), so `silu_and_mul(x)` is `gate * up` exactly and every partial sum is exact: `updated`
  is compared bit for bit in both forms. `normed` is compared with torch's fp32 RMSNorm within 2e-2 of its largest
  magnitude (the one-shot rounds the squares to bf16 and sums them in its own tree). SiLU-and-mul's rounding on
  general inputs is the kernel test's to certify (bit for bit against `k3_ctm_gemv_swiglu`), not this matrix's.
- World sizes: as for `k3_sandwich_oproj`; this entry's 16-rank receipt is pending. The op's kernel test
  (`tests/unittest/_torch/cute_dsl_kernels/kimi_k3/test_k3_sandwich.py`) passed every case at 16 ranks on four
  trays in a recorded run: the kernel's record, not this entry's receipt.
- State: `mutates_args` names `ws_uc`, `ws_mc` and `ws_flags`, every buffer the op writes. The compile cache is the
  documented process-wide cache above.
