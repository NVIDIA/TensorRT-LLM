# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_combined.py

import torch
from einops import rearrange

from .causal_conv1d import causal_conv1d_fwd
from .ssd_bmm import _bmm_chunk_fwd
from .ssd_chunk_scan import _chunk_scan_fwd
from .ssd_chunk_state import _chunk_cumsum_fwd, _chunk_state_fwd
from .ssd_state_passing import _state_passing_fwd


def rearrange_and_update_stride(tensor, pattern=None, dim=2):
    # ensure tensor.stride(dim) is a multiple of eight after rearranging according to pattern,
    # if not call contiguous(), rearrange only if pattern is not None
    tensor_rearranged = rearrange(tensor,
                                  pattern) if pattern is not None else tensor
    return (tensor_rearranged.contiguous() if tensor_rearranged.stride(dim) %
            8 != 0 else tensor_rearranged)


def _mamba_chunk_scan_combined_fwd(
        x,
        dt,
        A,
        B,
        C,
        chunk_size,
        D=None,
        z=None,
        dt_bias=None,
        initial_states=None,
        seq_idx=None,
        cu_seqlens=None,
        dt_softplus=False,
        dt_limit=(0.0, float("inf")),
):
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert x.shape == (batch, seqlen, nheads, headdim)
    assert dt.shape == (batch, seqlen, nheads)
    assert A.shape == (nheads, )
    assert C.shape == B.shape
    if z is not None:
        assert z.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads, )
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if B.stride(-1) != 1:
        B = B.contiguous()
    if C.stride(-1) != 1:
        C = C.contiguous()
    if (x.stride(-1) != 1 and x.stride(1)
            != 1):  # Either M or K dimension should be contiguous
        x = x.contiguous()
    if (z is not None and z.stride(-1) != 1 and z.stride(1)
            != 1):  # Either M or K dimension should be contiguous
        z = z.contiguous()
    if D is not None and D.stride(-1) != 1:
        D = D.contiguous()
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, headdim, dstate)
    # # (batch, nchunks, chunk_size, chunk_size) or (batch, nchunks, nheads, chunk_size, chunk_size)
    # dA_cumsum_tmp0, dt_tmp0 = _chunk_cumsum_fwd(dt[:, :147], A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus)
    # dA_cumsum_tmp1, dt_tmp1 = _chunk_cumsum_fwd(dt[:, 147:], A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus)
    # dA_cumsum_tmp2, dt_tmp2 = _chunk_cumsum_fwd(dt[:, 147:256], A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus)
    dA_cumsum, dt = _chunk_cumsum_fwd(dt,
                                      A,
                                      chunk_size,
                                      dt_bias=dt_bias,
                                      dt_softplus=dt_softplus,
                                      dt_limit=dt_limit)
    states = _chunk_state_fwd(B,
                              x,
                              dt,
                              dA_cumsum,
                              seq_idx=seq_idx,
                              states_in_fp32=True)
    # states_tmp0 = _chunk_state_fwd(B[:, :147], x[:, :147], dt_tmp0, dA_cumsum_tmp0, states_in_fp32=True)
    # states_tmp1 = _chunk_state_fwd(B[:, 147:], x[:, 147:], dt_tmp1, dA_cumsum_tmp1, states_in_fp32=True)
    # states_tmp2 = _chunk_state_fwd(B[:, 147:256], x[:, 147:256], dt_tmp2, dA_cumsum_tmp2, states_in_fp32=True)
    states, final_states = _state_passing_fwd(
        rearrange(states, "... p n -> ... (p n)"),
        dA_cumsum[:, :, :, -1],
        initial_states=(rearrange(initial_states, "... p n -> ... (p n)")
                        if initial_states is not None else None),
        seq_idx=seq_idx,
        chunk_size=chunk_size,
        out_dtype=C.dtype,
    )
    states, final_states = [
        rearrange(t, "... (p n) -> ... p n", n=dstate)
        for t in [states, final_states]
    ]
    # states_tmp0 = rearrange(_state_passing_fwd(rearrange(states_tmp0, "... p n -> ... (p n)"), dA_cumsum_tmp0[:, :, :, -1], chunk_size=chunk_size), "... (p n) -> ... p n", n=dstate)
    # states_tmp1 = rearrange(_state_passing_fwd(rearrange(states_tmp1, "... p n -> ... (p n)"), dA_cumsum_tmp1[:, :, :, -1], chunk_size=chunk_size), "... (p n) -> ... p n", n=dstate)
    CB = _bmm_chunk_fwd(C,
                        B,
                        chunk_size,
                        seq_idx=seq_idx,
                        output_dtype=torch.float32)
    out, out_x = _chunk_scan_fwd(CB,
                                 x,
                                 dt,
                                 dA_cumsum,
                                 C,
                                 states,
                                 D=D,
                                 z=z,
                                 seq_idx=seq_idx)
    if cu_seqlens is None:
        return out, out_x, dt, dA_cumsum, states, final_states
    else:
        assert (
            batch == 1
        ), "passing cu_seqlens to get the varlen states is only supported if batch dimension is 1"
        varlen_states = chunk_state_varlen(
            B.squeeze(0),
            x.squeeze(0),
            dt.squeeze(0),
            dA_cumsum.squeeze(0),
            cu_seqlens,
            states.squeeze(0),
        )
        return out, out_x, dt, dA_cumsum, states, final_states, varlen_states


def mamba_split_conv1d_scan_combined(
    zxbcdt,
    conv1d_weight,
    conv1d_bias,
    dt_bias,
    A,
    D,
    chunk_size,
    initial_states=None,
    return_final_states=False,
    activation="silu",
    headdim=None,
    ngroups=1,
    norm_before_gate=True,
):
    assert activation in [None, "silu", "swish"]
    if D.dim() == 1:
        assert headdim is not None
        (nheads, ) = D.shape
    else:
        nheads, headdim = D.shape
    batch, seqlen, _ = zxbcdt.shape
    dim = nheads * headdim
    assert nheads % ngroups == 0
    dstate = (conv1d_weight.shape[0] - dim) // ngroups // 2
    d_nonssm = (zxbcdt.shape[-1] - 2 * dim - 2 * ngroups * dstate - nheads) // 2
    assert d_nonssm >= 0
    assert zxbcdt.shape == (
        batch,
        seqlen,
        2 * d_nonssm + 2 * dim + 2 * ngroups * dstate + nheads,
    )
    assert dt_bias.shape == (nheads, )
    assert A.shape == (nheads, )
    zx0, z, xBC, dt = torch.split(
        zxbcdt, [2 * d_nonssm, dim, dim + ngroups * dstate * 2, nheads], dim=-1)

    xBC_conv = rearrange(
        causal_conv1d_fwd(
            rearrange_and_update_stride(xBC, "b s d -> b d s"),
            conv1d_weight,
            conv1d_bias,
        ),
        "b d s -> b s d",
    )
    x, B, C = torch.split(xBC_conv, [dim, ngroups * dstate, ngroups * dstate],
                          dim=-1)
    x = rearrange(x, "b l (h p) -> b l h p", h=nheads)
    B = rearrange(B, "b l (g n) -> b l g n", g=ngroups)
    C = rearrange(C, "b l (g n) -> b l g n", g=ngroups)
    z = rearrange(z, "b l (h p) -> b l h p",
                  h=nheads) if z is not None else None

    out, out_x, dt_out, dA_cumsum, states, final_states = (
        _mamba_chunk_scan_combined_fwd(
            x,
            dt,
            A,
            B,
            C,
            chunk_size=chunk_size,
            D=D,
            z=z,
            dt_bias=dt_bias,
            initial_states=initial_states,
            dt_softplus=True,
        ))
    out = rearrange(out, "b s h p -> b s (h p)")

    if d_nonssm > 0:
        out = torch.cat([_swiglu_fwd(zx0), out], dim=-1)

    return out if not return_final_states else (out, final_states)
