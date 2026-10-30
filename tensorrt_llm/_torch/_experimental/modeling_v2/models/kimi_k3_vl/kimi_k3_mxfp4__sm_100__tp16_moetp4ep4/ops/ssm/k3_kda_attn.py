# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3's KDA projection and speculative verify of one request's 8 tokens in one launch, over a caller-owned
:class:`K3KdaBuffers`; and ``k3_kda_qkvg``, the projection stream alone."""

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3_kda_attn import (  # noqa: F401 — registers the ops
    op as _k3_kda_attn_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3_kda_verify import (  # noqa: F401 — k3_kda_attn's helpers
    op as _k3_kda_verify_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.ssm.k3_kda_buffers import (
    K3KdaBuffers,
)

from .k3_kda_verify import k3_kda_verify


class K3KdaAttn(OpWrapper):
    """Return the gated-norm core output bf16 [8, 6, 128] of one request's golden token and 7 drafts ``x`` bf16
    [8, 7168]: the projection ``x w^T`` and the verify of ``ssm/k3_kda_verify`` on it, in one launch. Updates the
    slot's conv caches, state and per-draft states in place; advances ``buffers`` by one launch."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS = (
        Cell(
            why="TP16 golden token plus seven drafts: two successive requests, two layers, pending through 0..7",
            spec=dict(requests=2, rounds=9),
        ),
    )
    note = """
    x is contiguous bf16 [8, 7168]; w is contiguous bf16 [3208, 7168],
    ordered [q | k | v | og | f_a | b | pad] (768 rows each for q/k/v/og,
    128 for f_a, 6 for b, 2 padding). The 26 projection clusters of four CTAs
    sum four split-K partials in rank order for q/k/f_a and two fp32 K halves
    for v/og/b, rounding the projection to bf16. w_fb is bf16 [768, 128];
    conv taps are fp32 [768, 4], oldest first, a_log [6], dt_bias [768],
    onorm_w [128]. Kimi K3 uses lower_bound=-5, scale=128**-0.5, eps=1e-5.
    The verify commits the golden state, draft records (vn, beta k, decay),
    and raw conv inputs at positions -2..7. It is bit-identical to k3_kda_qkvg
    followed by k3_kda_verify on copies of the pools, including every pool word.

    K3KdaBuffers is caller-owned, one per device, created by the target in
    post_load_weights with K3KdaBuffers.create(device), outside graph capture.
    Creation allocates, fills sentinel words and zero indices, rejects capture
    and CTA counts other than FUSED_CTAS=128 or CTAS=104, and reads no environment.
    p1 is int16 [3 * 8 * 1664] (79,872 bytes): three buffers of eight rows of
    q/k/f_a bf16 bits. part is int32 [3 * 3 * 2 * 8 * 768] (442,368 bytes):
    three buffers of the fp32 bits of two K halves of v/og/b. epoch is int32
    [128], each CTA's buffer index, or [104] for the separate qkvg stream.
    All-ones words are sentinel NaNs; computed all-ones values are canonicalized
    to another NaN. Consumers poll the data words themselves, with no fence or
    counter, until producers replace every needed sentinel. Eight token rows
    are always published, whatever the batch.

    All KDA layers may share one 128-CTA set between k3_kda_attn and
    k3_kda_decode_attn: either may follow the other, one launch at a time in
    stream order, never concurrently on two streams. Both publish/re-arm the
    same words and advance every CTA once. qkvg needs its own 104-CTA set;
    passing that set here raises ValueError before any write. After the
    grid-dependency wait, each CTA reads e=epoch[cta], writes buffer e, re-arms
    (e+1)%3 for the next launch and leaves epoch[cta]=(e+1)%3. All indices
    agree after a launch. A later launch reads those indices and the words the
    previous launch re-armed; create arms buffer 0 for the first launch.
    Keeping indices in 0..2 avoids signed-count overflow indexing before the
    buffers. Four launches from 2**31-2 agree with a fresh set and end in 0..2;
    the kernel's own guard-band test checks writes stay inside the buffers.
    Shared, per-layer, alternating and decode/verify-interleaved sets give
    identical bits; captured rounds replay identically to eager rounds.

    Head CTAs read slots, pending, state, records and conv windows before
    their grid-dependency wait. Metadata must be written before graph replay,
    not by a kernel of the same step ahead of this launch. No concurrent
    request may use the same slot, and launches on the same pools need an
    intervening kernel that waits, or a non-PDL kernel. An attn/decode_attn
    launch on other pools suffices: dependents launch only after its own wait.
    k3_kda_verify releases dependents before its wait and does not suffice.
    Tests drive two requests through nine rounds and all acceptance counts,
    every layer against the unfused path and layer zero against committed
    history. Swapping rounds silently changes that history (sm_100 output
    and state relative differences 1.00 and 1.01); one call cannot detect order.


    qkvg alone accepts T<=8: the pre-call epoch selects the published p1/part
    rows, and its consumer rounds the two K-half sums to bf16. Against a
    float64 projection, measured maximum-scale errors are 2.5e-3 at T=1 and
    2.4e-3 at T=3/8. Repeated runs of both paths are bit-identical.
    """

    def __call__(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        w_fb: torch.Tensor,
        w_q: torch.Tensor,
        w_k: torch.Tensor,
        w_v: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        onorm_w: torch.Tensor,
        cs_q: torch.Tensor,
        cs_k: torch.Tensor,
        cs_v: torch.Tensor,
        ssm: torch.Tensor,
        state_tok: torch.Tensor,
        slots: torch.Tensor,
        pending: torch.Tensor,
        buffers: K3KdaBuffers,
        num_spec: int,
        lower_bound: float,
        scale: float,
        eps: float,
    ) -> torch.Tensor:
        return torch.ops.trtllm.k3_kda_attn(
            x,
            w,
            w_fb,
            w_q,
            w_k,
            w_v,
            a_log,
            dt_bias,
            onorm_w,
            cs_q,
            cs_k,
            cs_v,
            ssm,
            state_tok,
            slots,
            pending,
            buffers.p1,
            buffers.part,
            buffers.epoch,
            num_spec,
            lower_bound,
            scale,
            eps,
        )

    def reference(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        w_fb: torch.Tensor,
        w_q: torch.Tensor,
        w_k: torch.Tensor,
        w_v: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        onorm_w: torch.Tensor,
        cs_q: torch.Tensor,
        cs_k: torch.Tensor,
        cs_v: torch.Tensor,
        ssm: torch.Tensor,
        state_tok: torch.Tensor,
        slots: torch.Tensor,
        pending: torch.Tensor,
        buffers: K3KdaBuffers,
        num_spec: int,
        lower_bound: float,
        scale: float,
        eps: float,
    ) -> torch.Tensor:
        with true_fp32_matmul():
            proj = (x.float() @ w.float().t()).to(x.dtype)
        return k3_kda_verify.reference(
            proj,
            w_fb,
            w_q,
            w_k,
            w_v,
            a_log,
            dt_bias,
            onorm_w,
            cs_q,
            cs_k,
            cs_v,
            ssm,
            state_tok,
            slots,
            pending,
            num_spec,
            lower_bound,
            scale,
            eps,
        )

    def is_valid(
        self,
        x: torch.Tensor,
        w: torch.Tensor,
        w_fb: torch.Tensor,
        w_q: torch.Tensor,
        w_k: torch.Tensor,
        w_v: torch.Tensor,
        a_log: torch.Tensor,
        dt_bias: torch.Tensor,
        onorm_w: torch.Tensor,
        cs_q: torch.Tensor,
        cs_k: torch.Tensor,
        cs_v: torch.Tensor,
        ssm: torch.Tensor,
        state_tok: torch.Tensor,
        slots: torch.Tensor,
        pending: torch.Tensor,
        buffers: K3KdaBuffers,
        num_spec: int,
        lower_bound: float,
        scale: float,
        eps: float,
    ) -> None:
        for weight in (w_q, w_k, w_v):
            assert weight.shape == (768, 4), "conv taps must be [768, 4], oldest first"
            if weight.t().is_contiguous():
                assert weight.is_contiguous(), (
                    "conv taps must store each channel's four taps together"
                )
        assert a_log.shape == (6,), "a_log must have one value per head"
        assert dt_bias.shape == (768,), "dt_bias must have one value per key channel"
        assert onorm_w.shape == (128,), "onorm_w must have one value per output channel"
        for cache in (cs_q, cs_k, cs_v):
            assert cache.shape[1] == 768, "conv caches must cover every channel"
            if cache.is_contiguous():
                assert cache.stride(-2) == 1, "conv caches must be channel-contiguous"
        for cache in (cs_k, cs_v):
            assert cache.shape[-1] == 3 + num_spec, "conv caches must cover every replay column"
        assert pending.ndim == 1 and pending.stride(0) > 0, "pending must be a strided vector"
        slot = int(slots.item())
        pool_size = min(
            ssm.shape[0],
            state_tok.shape[0],
            cs_q.shape[0],
            cs_k.shape[0],
            cs_v.shape[0],
            pending.numel(),
        )
        assert 0 <= slot < pool_size, "slot must be inside every pool and pending"
        assert 0 <= int(pending[slot].item()) <= num_spec, "pending must be in 0..num_spec"


k3_kda_attn = K3KdaAttn()


def k3_kda_qkvg(x: torch.Tensor, w: torch.Tensor, buffers: K3KdaBuffers) -> None:
    """The projection ``x w^T`` of T <= 8 tokens (``x`` bf16 [T, 7168], ``w`` bf16 [3208, 7168]) published into
    ``buffers`` (made with ``ctas=CTAS``): buffer ``e = buffers.epoch[0]`` (before the call) holds the rows; advances
    ``buffers`` by one launch. Returns None."""
    torch.ops.trtllm.k3_kda_qkvg(x, w, buffers.p1, buffers.part, buffers.epoch)
