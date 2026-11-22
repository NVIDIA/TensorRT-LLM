# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-token (one token per sequence) Gated-DeltaNet decode kernels over the quantized
state pool, and the same step with the gated RMS norm folded in.

Both are specializations of the generic ``fused_sigmoid_gating_delta_rule_update_kernel``
for pure decode: the same math in the same order (sigmoid / softplus gating, in-kernel
q/k L2 norm, fp32 delta-rule update), the state dequantized from the pool once and
requantized once (see ``_gdn_state_q8``), without the per-token time loop and with a
flat grid. The q/k L2 norm is ``x / sqrt(sum(x^2) + 1e-6)``, the generic decode
kernel's and the prefill front end's form.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from ._gdn_state_q8 import q8_load_rows, q8_state_rows, q8_store_rows

#: Gate constants of the checkpoint's GDN (`softplus(x) = log(1 + exp(x))` below 20).
SOFTPLUS_BETA, SOFTPLUS_THRESHOLD = 1.0, 20.0


@triton.jit
def _gdn_decode_t1_kernel(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    o,
    h0_source,
    h0_indices,
    scale,
    stride_q,
    stride_k,
    stride_v,
    stride_a,
    stride_b,
    s_h0_0,
    h0_dim0,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BV: tl.constexpr,
):
    NV: tl.constexpr = V // BV
    pid = tl.program_id(0)
    i_nh = pid // NV
    i_v = pid % NV
    # Pure decode: sequence i_n owns exactly token i_n.
    i_n = (i_nh // HV).to(tl.int64)
    i_hv = i_nh % HV
    i_h = i_hv // (HV // H)

    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)

    # The state load depends on the slot index: issue it first.
    idx = tl.load(h0_indices + i_n).to(tl.int64)
    has_state = idx >= 0
    tl.device_assert(idx < h0_dim0, "idx out of bounds in h0_source access")
    # pool [slots, HV, 2V + 4, K] int8 bytes: V int16 rows + V fp32 row scales.
    p_head = h0_source + idx * s_h0_0 + i_hv * (2 * V + (V * 4) // K) * K
    b_h = q8_load_rows(p_head, o_v, o_k, has_state, K, V)

    b_q = tl.load(q + i_n * stride_q + i_h * K + o_k).to(tl.float32)
    b_k = tl.load(k + i_n * stride_k + i_h * K + o_k).to(tl.float32)
    b_v = tl.load(v + i_n * stride_v + i_hv * V + o_v).to(tl.float32)
    b_b = tl.load(b + i_n * stride_b + i_hv).to(tl.float32)
    b_a = tl.load(a + i_n * stride_a + i_hv).to(tl.float32)
    b_A_log = tl.load(A_log + i_hv).to(tl.float32)
    b_dt_bias = tl.load(dt_bias + i_hv).to(tl.float32)

    # g = -exp(A_log) * softplus(a + dt_bias)
    x = b_a + b_dt_bias
    beta_x = softplus_beta * x
    softplus_x = tl.where(
        beta_x <= softplus_threshold,
        (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
        x,
    )
    b_g = -tl.exp(b_A_log) * softplus_x

    # beta = sigmoid(b)
    b_beta = 1.0 / (1.0 + tl.exp(-b_b))

    b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
    b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
    b_q = b_q * scale

    # h *= exp(g)
    b_h *= tl.exp(b_g)
    # v -= sum_k(h * k)
    b_v -= tl.sum(b_h * b_k[None, :], 1)
    # v *= beta
    b_v *= b_beta
    # h += k (x) v
    b_h += b_k[None, :] * b_v[:, None]
    # o = sum_k(h * q)
    b_o = tl.sum(b_h * b_q[None, :], 1)

    # o is [N, HV, V] with V contiguous; its token stride is HV * V.
    tl.store(o + (i_n * HV + i_hv) * V + o_v, b_o.to(o.dtype.element_ty))
    q8_store_rows(p_head, o_v, o_k, b_h, has_state, K, V)


@triton.jit
def _warp32x4_sum_sq(x):
    """sum(x * x) over a 1-D [128] fp32 vector in exactly the order Triton uses for
    tl.sum over a single-warp blocked layout with 4 contiguous elements per lane
    (sizePerThread [4], threadsPerWarp [32]): per lane the packed tree
    (x0^2 + x2^2) + (x1^2 + x3^2) (mul.f32 x4, add.f32x2, add.f32), then an xor
    butterfly over the 32 lanes (offsets 16, 8, 4, 2, 1). This is the order of the
    q/k L2 norms in the single-warp kernel above and of the sum of squares in the
    multi-row gated RMS norm kernel ([1, 4] x [1, 32]). Element extraction is a
    masked sum with zeros (exact), every butterfly level is one add per pair, and
    squares of bf16-valued x are exact in fp32 (so FMA contraction cannot change
    them): the result is bitwise independent of the layout Triton picks for this
    multi-warp program."""
    xr = tl.reshape(x, (32, 4))
    cols = tl.arange(0, 4)[None, :]
    e0 = tl.sum(tl.where(cols == 0, xr, 0.0), axis=1)
    e1 = tl.sum(tl.where(cols == 1, xr, 0.0), axis=1)
    e2 = tl.sum(tl.where(cols == 2, xr, 0.0), axis=1)
    e3 = tl.sum(tl.where(cols == 3, xr, 0.0), axis=1)
    s = (e0 * e0 + e2 * e2) + (e1 * e1 + e3 * e3)
    s = tl.sum(tl.reshape(s, (2, 16)), axis=0)
    s = tl.sum(tl.reshape(s, (2, 8)), axis=0)
    s = tl.sum(tl.reshape(s, (2, 4)), axis=0)
    s = tl.sum(tl.reshape(s, (2, 2)), axis=0)
    return tl.sum(s, axis=0)


@triton.jit
def _gdn_decode_t1_gated_norm_kernel(
    A_log,
    a,
    dt_bias,
    softplus_beta,
    softplus_threshold,
    q,
    k,
    v,
    b,
    z,
    norm_w,
    y,
    h0_source,
    h0_indices,
    scale,
    eps,
    stride_q,
    stride_k,
    stride_v,
    stride_a,
    stride_b,
    stride_z,
    s_h0_0,
    h0_dim0,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
):
    """The single-token step with the GDN gated RMS norm (silu gate, norm before gate)
    folded in: one program (4 or 8 warps) per (sequence, value head) owns the whole
    [V, K] state, so the norm over V needs no cross-program step. Bitwise equal to
    ``_gdn_decode_t1_kernel`` followed by the multi-row gated RMS norm kernel: the
    state tile keeps the single-warp kernel's per-warp [1, 8] x [2, 16] grouping over
    K (warps split V only), so the K contractions are unchanged; the q/k L2 norms and
    the norm's sum of squares reproduce those kernels' single-warp reduce order with
    ``_warp32x4_sum_sq``; o is rounded to the output dtype before the norm, as the
    stored o would be, and the norm's elementwise ops are the multi-row kernel's in
    the same order."""
    tl.static_assert(K == 128 and V == 128)
    pid = tl.program_id(0)
    i_n = (pid // HV).to(tl.int64)
    i_hv = pid % HV
    i_h = i_hv // (HV // H)

    o_k = tl.arange(0, K)
    o_v = tl.arange(0, V)

    idx = tl.load(h0_indices + i_n).to(tl.int64)
    has_state = idx >= 0
    tl.device_assert(idx < h0_dim0, "idx out of bounds in h0_source access")
    p_head = h0_source + idx * s_h0_0 + i_hv * (2 * V + (V * 4) // K) * K
    b_h = q8_load_rows(p_head, o_v, o_k, has_state, K, V)

    b_q = tl.load(q + i_n * stride_q + i_h * K + o_k).to(tl.float32)
    b_k = tl.load(k + i_n * stride_k + i_h * K + o_k).to(tl.float32)
    b_v = tl.load(v + i_n * stride_v + i_hv * V + o_v).to(tl.float32)
    b_b = tl.load(b + i_n * stride_b + i_hv).to(tl.float32)
    b_a = tl.load(a + i_n * stride_a + i_hv).to(tl.float32)
    b_A_log = tl.load(A_log + i_hv).to(tl.float32)
    b_dt_bias = tl.load(dt_bias + i_hv).to(tl.float32)
    b_z = tl.load(z + i_n * stride_z + i_hv * V + o_v).to(tl.float32)
    b_w = tl.load(norm_w + o_v).to(tl.float32)

    x = b_a + b_dt_bias
    beta_x = softplus_beta * x
    softplus_x = tl.where(
        beta_x <= softplus_threshold,
        (1.0 / softplus_beta) * tl.log(1.0 + tl.exp(beta_x)),
        x,
    )
    b_g = -tl.exp(b_A_log) * softplus_x
    b_beta = 1.0 / (1.0 + tl.exp(-b_b))

    b_q = b_q / tl.sqrt(_warp32x4_sum_sq(b_q) + 1e-6)
    b_k = b_k / tl.sqrt(_warp32x4_sum_sq(b_k) + 1e-6)
    b_q = b_q * scale

    b_h *= tl.exp(b_g)
    b_v -= tl.sum(b_h * b_k[None, :], 1)
    b_v *= b_beta
    b_h += b_k[None, :] * b_v[:, None]
    b_o = tl.sum(b_h * b_q[None, :], 1)

    q8_store_rows(p_head, o_v, o_k, b_h, has_state, K, V)
    b_o = b_o.to(y.dtype.element_ty)
    # Gated RMSNorm (norm before gate, silu gate), as the multi-row kernel.
    xo = b_o.to(tl.float32)
    var = _warp32x4_sum_sq(xo) / V
    rstd = 1.0 / tl.sqrt(var + eps)
    b_y = xo * rstd * b_w
    gate = tl.sigmoid(b_z)
    gate *= b_z
    b_y *= gate
    tl.store(y + (i_n * HV + i_hv) * V + o_v, b_y.to(y.dtype.element_ty))


def t1_config(num_heads_total: int, v: int) -> tuple[int, int]:
    """(BV, num_warps) of the single-token kernel for ``N * HV`` (sequence, value-head)
    pairs. BV 8 keeps enough programs in flight up to N * HV = 96 (HV 48: two rows);
    above that the kernel is state-bandwidth bound and BV 16 is marginally best. One
    warp per program keeps every reduction (q/k L2 norm, k- and q-contractions) in the
    same per-thread grouping and order as the generic single-warp kernel."""
    bv = 8 if num_heads_total <= 96 else 16
    return min(bv, v), 1


def t1_norm_warps(num_seqs: int) -> int:
    """Warps of the folded-norm kernel: 8 below 8 rows, 4 from 8 rows."""
    return 4 if num_seqs >= 8 else 8


def launch_t1(q, k, v, a, b, A_log, dt_bias, pool, slots, out) -> torch.Tensor:
    """One step for N sequences; writes and returns ``out`` ``[N, HV, V]``."""
    _, n, h, kd = k.shape
    hv, vd = v.shape[2], v.shape[3]
    bv, warps = t1_config(n * hv, vd)
    grid = (n * hv * (vd // bv),)
    with torch.cuda.device(q.device.index):
        _gdn_decode_t1_kernel[grid](
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=SOFTPLUS_BETA,
            softplus_threshold=SOFTPLUS_THRESHOLD,
            q=q,
            k=k,
            v=v,
            b=b,
            o=out,
            h0_source=pool,
            h0_indices=slots,
            scale=kd**-0.5,
            stride_q=q.stride(1),
            stride_k=k.stride(1),
            stride_v=v.stride(1),
            stride_a=a.stride(-2),
            stride_b=b.stride(-2),
            s_h0_0=pool.stride(0),
            h0_dim0=pool.shape[0],
            H=h,
            HV=hv,
            K=kd,
            V=vd,
            BV=bv,
            num_warps=warps,
            num_stages=1,
        )
    return out


def launch_t1_gated_norm(q, k, v, a, b, A_log, dt_bias, pool, slots, z, norm_weight, eps):
    """One step plus the gated norm for N sequences; returns y ``[N * HV, V]``."""
    _, n, h, kd = k.shape
    hv, vd = v.shape[2], v.shape[3]
    norm_weight = norm_weight.contiguous()
    y = q.new_empty(n * hv, vd)
    with torch.cuda.device(q.device.index):
        _gdn_decode_t1_gated_norm_kernel[(n * hv,)](
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            softplus_beta=SOFTPLUS_BETA,
            softplus_threshold=SOFTPLUS_THRESHOLD,
            q=q,
            k=k,
            v=v,
            b=b,
            z=z,
            norm_w=norm_weight,
            y=y,
            h0_source=pool,
            h0_indices=slots,
            scale=kd**-0.5,
            eps=eps,
            stride_q=q.stride(1),
            stride_k=k.stride(1),
            stride_v=v.stride(1),
            stride_a=a.stride(-2),
            stride_b=b.stride(-2),
            stride_z=z.stride(0),
            s_h0_0=pool.stride(0),
            h0_dim0=pool.shape[0],
            H=h,
            HV=hv,
            K=kd,
            V=vd,
            num_warps=t1_norm_warps(n),
            num_stages=1,
        )
    return y


def check_operands(q, k, v, a, b, A_log, dt_bias, pool, slots) -> None:
    """What both single-token entries read the same way, and would misread silently."""
    _, n, hk, kd = k.shape
    hv, vd = v.shape[2], v.shape[3]
    assert q.shape == k.shape and v.shape[:2] == (1, n), "q, k [1, N, H, K]; v [1, N, HV, V]"
    assert kd == vd == 128, "K == V == 128 (the pool's scale rows fill exactly 4 K-rows)"
    assert hv % hk == 0, "HV a multiple of H"
    for name, t, d in (("q", q, kd), ("k", k, kd), ("v", v, vd)):
        assert t.dtype == torch.bfloat16, f"{name} must be bf16"
        assert t.stride(3) == 1 and t.stride(2) == d, (
            f"{name}'s heads must be packed (head stride {d}, unit inner stride): the "
            "kernel addresses head h at h * D from the token's base"
        )
    for name, t in (("a", a), ("b", b)):
        assert t.shape[-2:] == (n, hv) and t.stride(-1) == 1, f"{name} [N, HV], unit head stride"
    for name, t in (("A_log", A_log), ("dt_bias", dt_bias)):
        assert t.shape == (hv,) and t.is_contiguous(), f"{name} contiguous [HV]"
    rows = q8_state_rows(vd, kd)
    assert pool.dtype == torch.int8 and pool.dim() == 4, "int8 pool [S, HV, V + 4, K]"
    assert tuple(pool.shape[1:]) == (hv, rows, kd), (
        f"pool [S, {hv}, {rows}, {kd}]: V int8 rows then 4 rows of fp32 row scales"
    )
    assert pool.stride()[1:] == (rows * kd, kd, 1), "each slot's [HV, V + 4, K] block contiguous"
    assert pool.stride(0) % 16 == 0 and pool.data_ptr() % 16 == 0, (
        "16-byte aligned slots: the fp32 scales are read through the int8 base"
    )
    assert slots.dtype == torch.int32 and slots.shape == (n,) and slots.is_contiguous(), (
        "slots int32 contiguous [N]"
    )
    live = [s for s in slots.tolist() if s >= 0]
    assert len(set(live)) == len(live), "two rows naming one slot race on its state"
    assert all(s < pool.shape[0] for s in live), "a slot past the pool is an illegal access"


# ---------------------------------------------------------------------------------------
# Plain-torch twin of one step, for the entries' `reference`s.
# ---------------------------------------------------------------------------------------


def reference_step(q, k, v, a, b, A_log, dt_bias, h):
    """``(o fp64 [N, HV, V], h_after fp64 [N, HV, V, K])`` of one decode step from the
    fp32 states ``h`` ``[N, HV, V, K]``, in float64: the gates, the q/k L2 norm in the
    ``x / sqrt(|x|^2 + 1e-6)`` form, then ``h *= exp(g); u = (v - h k) beta; h += u k^T;
    o = h q``. Value head ``i`` reads key head ``i // (HV // H)``."""
    dt = torch.float64
    hk, kd = q.shape[2], q.shape[3]
    hv = v.shape[2]
    rep = hv // hk
    qf = q[0].to(dt)
    kf = k[0].to(dt)
    qf = qf / (qf.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()
    kf = kf / (kf.pow(2).sum(-1, keepdim=True) + 1e-6).sqrt()
    qf = (qf * kd**-0.5).repeat_interleave(rep, dim=1)
    kf = kf.repeat_interleave(rep, dim=1)
    x = a.to(dt) + dt_bias.to(dt)
    softplus = torch.where(x <= SOFTPLUS_THRESHOLD, torch.log1p(torch.exp(x)), x)
    g = -torch.exp(A_log.to(dt)) * softplus
    beta = torch.sigmoid(b.to(dt))
    h = h.to(dt) * torch.exp(g)[..., None, None]
    u = (v[0].to(dt) - torch.einsum("nhvk,nhk->nhv", h, kf)) * beta[..., None]
    h = h + u[..., :, None] * kf[..., None, :]
    o = torch.einsum("nhvk,nhk->nhv", h, qf)
    return o, h
