# 1a.2 + 1a.3 (kernel mask + Python binding) — DEFERRED

The other 5 stubs in this prototype validate Python-side seams without
needing the production AlltoAll kernel to actually skip dead peers.

But the test recipe in
[`mvp-prototype-plan.md` §4.3](../../docs/design/wide-ep-fault-tolerance/mvp-prototype-plan.md#43-sequence)
explicitly stresses "kill during dispatch (kernel actively spinning on dead
peer's `completion_flags`)". Without an in-kernel mask check, the
`NVLinkOneSided` dispatch / combine kernels will spin forever on the dead
peer's flag entry, regardless of how cleanly the Python-side stubs handle
the failure. The kernel needs *some* way to skip a dead peer.

This directory documents the two integration paths for the kernel-side
work; pick one before the prototype runs end-to-end on real hardware.

## Path A — Cherry-pick PR #13404 (the real 1a.2)

PR #13404 is the production-grade kernel mask: `kMaxRanks` 64 → 128, full
`active_rank_mask[2]` ABI through dispatch + combine + torch op + Python
custom op. It is more than the prototype spec asked for ("Add a single
rank-test branch ... no `kMaxRanks` 64→128 bump"), but it's already
implemented, already in review, and already mergeable.

Cherry-pick command (run from this branch):

```bash
cd /path/to/TensorRT-LLM-mvp-prototype
git cherry-pick fork/WideEP-FT/1a.2-nvlink-kernel-mask  # or the explicit SHA
```

**Pros:** zero new code; matches what the production data-plane will look
like; if PR #13404 changes during review, this branch picks up the changes
naturally on rebase.

**Cons:** brings in extra surface area (the `kMaxRanks` bump, `[2]`-word
mask, full ABI version handling) that the prototype could go without.

## Path B — Inline minimal stub modifications

If reviewers prefer to keep the prototype's kernel change strictly minimal,
here is the smallest set of inline changes that exercises the seam:

### `cpp/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.h`

Add to **both** `MoeA2ADispatchParams` and `MoeA2ACombineParams`:

```cpp
// PROTOTYPE (WideEP FT MVP): single uint64 mask; bit i = rank i is active.
// Default ~0ULL keeps the production code path intact; set from Python under
// the FT feature flag. Production replacement: PR #13404's active_rank_mask[2].
uint64_t active_rank_mask{~uint64_t{0}};
```

### `cpp/tensorrt_llm/kernels/communicationKernels/moeAlltoAllKernels.cu`

Inside the dispatch + combine polling loops, before each peer's
`completion_flags` is consulted, gate the read on the mask bit:

```cpp
// PROTOTYPE (WideEP FT MVP): skip dead peers. Removed when PR #13404 lands.
if (!((params.active_rank_mask >> peer) & 1ULL))
{
    continue;
}
```

### `cpp/tensorrt_llm/thop/moeAlltoAllOp.cpp`

Add a `int64_t active_rank_mask` argument to both `moeA2ADispatchOp` and
`moeA2ACombineOp`, defaulting to `-1` (all-ones in two's complement),
plumbed straight into the params struct.

### `tensorrt_llm/_torch/custom_ops/cpp_custom_ops.py`

Add the corresponding `active_rank_mask: int = -1` to both
`register_fake` definitions.

### `tensorrt_llm/_torch/modules/fused_moe/communication/nvlink_one_sided.py`

At the call sites of the dispatch + combine ops, read the mask from
`EPGroupHealth` and pass it through:

```python
mask_word, = ep_group_health.get_mask_words(num_words=1)
torch.ops.trtllm.moe_a2a_dispatch(..., active_rank_mask=mask_word)
torch.ops.trtllm.moe_a2a_combine(..., active_rank_mask=mask_word)
```

**Pros:** strictly minimal; matches the prototype-plan spec to the letter;
no unrelated surface area.

**Cons:** writes a second implementation of the same kernel mask, which has
to be reconciled when PR #13404 lands. Practically: the prototype branch is
discarded anyway, so the divergence only matters for the duration of the
prototype.

## Recommended choice

**Path A** if you have a working build and PR #13404's tip is reasonably
stable (no in-flight reviewer-requested kernel changes today). **Path B** if
PR #13404 is moving rapidly and you want a self-contained stub that doesn't
chase its tip.

Either choice unblocks the kill-during-dispatch / kill-during-combine
variants of the test recipe in §4.3 of the plan. Without either, the
prototype is limited to inter-iteration kill points (still useful for
exercising the watchdog → broadcast → reconfigure seam, just not the
in-kernel polling-loop case).
