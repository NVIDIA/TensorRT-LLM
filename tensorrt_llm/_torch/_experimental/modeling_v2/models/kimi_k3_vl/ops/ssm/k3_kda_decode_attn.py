# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Kimi K3's KDA projection and plain one-token decode of up to 8 requests in one launch, over a caller-owned
:class:`K3KdaBuffers`."""

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3_kda_attn import (  # noqa: F401 — registers the op
    op as _k3_kda_attn_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3_kda_verify import (  # noqa: F401 — the op's helpers
    op as _k3_kda_verify_op,
)

from .k3_kda_buffers import K3KdaBuffers


class K3KdaDecodeAttn(OpWrapper):
    """Return the gated-norm core output bf16 [R, 6, 128] of one token of each of R <= 8 requests ``x`` bf16
    [R, 7168]: the projection ``x w^T`` and the plain decode of ``ssm/kda_decode`` on it, in one launch. Updates the
    slots' conv and state pools in place; advances ``buffers`` by one launch."""

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS = tuple(
        Cell(
            why=f"{requests} active TP16 decode rows and {8 - requests} zero-padded rows on three manager layers",
            spec=dict(requests=requests),
        )
        for requests in range(1, 9)
    )
    note = """
    One token per request, 1<=R<=8: x bf16 [R,7168], w bf16 [3208,7168],
    w_fb bf16 [768,128], all contiguous. Projection order is q/k/v/og (768
    columns each), f_a (128), b (6), padding (2); q/k/f_a use the four split-K
    partials, v/og/b the two fp32 K halves, rounded to bf16. Rows >=R read as
    zero. f_b is rounded to bf16; conv taps are fp32 [768,4], oldest first,
    a_log [6], dt_bias [768], onorm_w [128]. Kimi K3 uses lower_bound=-5,
    scale=128**-0.5, eps=1e-5. Six head clusters of four CTAs own V quarters.
    The recurrence uses kda_decode's row/key layout; output/state agree within
    fp32 tolerances, not bitwise, because their sums have different orders.
    Raw bf16 conv windows agree bitwise with the native projection/decode path.

    The caller creates one K3KdaBuffers per device before graph capture,
    shared by every KDA layer's decode and verify launches. It holds three
    sentinel-armed p1/part Lamport buffers and 128 per-CTA indices. Launches
    sharing it run one at a time in stream order: read epoch, publish all
    eight rows, re-arm the next buffer, advance every index modulo three.
    k3_kda_attn.note records the buffer contents and ownership lifecycle.
    Decode/verify interleaving, one shared set versus per-layer/alternating
    sets, and four launches from epoch=2**31-2 give identical bits; indices
    stay in 0..2. A qkvg 104-CTA set raises ValueError before writes; creating
    buffers during capture raises RuntimeError.

    Pools are manager-owned, not buffer contents: a V2 hybrid manager with
    conv_state_layout='q_k_v' supplies get_conv_states(l), bf16 [pool,2304,3],
    q/k/v channels with oldest raw input first, and get_ssm_states(l), fp32
    [pool,6,128,128], V rows with contiguous K. Each slot is dense, with
    manager-coalesced slot strides (conv multiples of 8, ssm multiples of 4)
    and 16-byte bases. slots is contiguous int32 [R] from get_state_indices.
    A launch touches only its named slots of its own layer; the cell tests
    preserve other slots and layers. Each step's state/window feeds the next;
    graph-replayed steps with rewritten inputs match eager steps bitwise.

    Head CTAs read slots, conv windows and state before their grid-dependency
    wait, so slots must be written before graph replay. Successive launches
    on the same pools need an intervening waiting/non-PDL kernel. An attn or
    decode_attn launch on other pools suffices, releasing dependents only
    after its own wait. Swapped steps silently produce a different history
    (sm_100 relative output/state differences 0.81/0.89); one call cannot
    detect order. Calls must not concurrently reuse the same pools' slots.

    The kernel rejects unsupported x/w/w_fb, pool shapes/strides/alignment,
    metadata counts/dtypes and buffer sizes before launch. First use of each
    lower_bound/scale/eps/PDL configuration compiles outside graph capture;
    capture raises RuntimeError. PDL reads TRTLLM_ENABLE_PDL, default on,
    each call. slots may start at any element offset (0..3 certified); pool
    offsets are 64-bit, including pools beyond 2 GiB covered by
    test_k3_kda_pools_past_2g.py. Repeated runs are bit-identical.
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
        conv: torch.Tensor,
        ssm: torch.Tensor,
        slots: torch.Tensor,
        buffers: K3KdaBuffers,
        lower_bound: float,
        scale: float,
        eps: float,
    ) -> torch.Tensor:
        return torch.ops.trtllm.k3_kda_decode_attn(
            x,
            w,
            w_fb,
            w_q,
            w_k,
            w_v,
            a_log,
            dt_bias,
            onorm_w,
            conv,
            ssm,
            slots,
            buffers.p1,
            buffers.part,
            buffers.epoch,
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
        conv: torch.Tensor,
        ssm: torch.Tensor,
        slots: torch.Tensor,
        buffers: K3KdaBuffers,
        lower_bound: float,
        scale: float,
        eps: float,
    ) -> torch.Tensor:
        heads, width = ssm.shape[1], ssm.shape[-1]
        hk = heads * width
        with true_fp32_matmul():
            rows = (x.float() @ w.float().t()).to(x.dtype).float()
            gate = (rows[:, 4 * hk : 4 * hk + width] @ w_fb.float().t()).to(x.dtype).float()
        conv_w = torch.stack((w_q, w_k, w_v)).float()
        out = torch.empty(x.shape[0], heads, width, dtype=torch.float32, device=x.device)
        for row, slot in enumerate(slots.tolist()):
            raw = rows[row, : 3 * hk].reshape(3, hk)
            window = conv[slot].float().reshape(3, hk, 3)
            u = torch.cat((window, raw[:, :, None]), dim=-1)
            act = (u * conv_w).sum(-1)
            act = act * torch.sigmoid(act)
            conv[slot].copy_(u[:, :, 1:].reshape(3 * hk, 3))
            q, k, v = act.reshape(3, heads, width).unbind(0)
            q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * scale
            k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
            beta = torch.sigmoid(rows[row, 4 * hk + width : 4 * hk + width + heads])
            xg = torch.exp(a_log.float())[:, None] * (
                gate[row].reshape(heads, width) + dt_bias.float().reshape(heads, width)
            )
            decay = torch.exp(lower_bound * torch.sigmoid(xg))
            state = ssm[slot].float() * decay[:, None, :]
            residual = (v - (state * k[:, None, :]).sum(-1)) * beta[:, None]
            state = state + residual[:, :, None] * k[:, None, :]
            ssm[slot].copy_(state)
            o = (state * q[:, None, :]).sum(-1)
            rms = torch.rsqrt((o * o).mean(-1, keepdim=True) + eps)
            output_gate = torch.sigmoid(rows[row, 3 * hk : 4 * hk].reshape(heads, width))
            out[row] = o * rms * onorm_w.float() * output_gate
        return out.to(x.dtype)

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
        conv: torch.Tensor,
        ssm: torch.Tensor,
        slots: torch.Tensor,
        buffers: K3KdaBuffers,
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
        indices = slots.long()
        assert indices.unique().numel() == indices.numel(), "a call must name each slot once"
        assert bool(((indices >= 0) & (indices < min(conv.shape[0], ssm.shape[0]))).all()), (
            "slots must be inside every pool"
        )


k3_kda_decode_attn = K3KdaDecodeAttn()
