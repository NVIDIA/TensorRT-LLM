# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_combined.py
# Copyright (c) 2024, Tri Dao, Albert Gu.
#
# SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools

import torch
import torch.nn.functional as F
from einops import rearrange
from flashinfer.mamba import SSDCombined

from tensorrt_llm._utils import is_sm_100f
from tensorrt_llm.logger import logger
from tensorrt_llm.math_utils import pad_up

from .fuse_elementwise_ops import ssd_output_transpose
from .mamba2_metadata import cu_seqlens_to_chunk_indices_offsets_triton
from .ssd_bmm import _bmm_chunk_fwd
from .ssd_chunk_scan import _chunk_scan_fwd
from .ssd_chunk_state import (_chunk_cumsum_fwd, _chunk_state_fwd,
                              chunk_state_varlen)
from .ssd_state_passing import _state_passing_fwd

# FlashInfer's fused CUTLASS SSD kernel lowers to tcgen05 MMA whose M-mode must
# be 64 or 128. `dstate` is the M-mode of the inter-chunk MMA and `chunk_size`
# is the M-mode of the intra-chunk MMAs, so both dimensions must satisfy this
# constraint. `headdim` controls the intra-chunk MMA tile along the head axis;
# only 64 and 128 keep the planned TMEM footprint within SM100's 512-column
# capacity (e.g. headdim=80 trips an internal TMEM-capacity assertion in
# `_plan_tmem_offsets`). Otherwise we fall back to the Triton reference kernels.
_FLASHINFER_SSD_VALID_M_MODES = (64, 128)
_FLASHINFER_SSD_VALID_HEAD_DIMS = (64, 128)


def _flashinfer_ssd_supported(chunk_size, dstate, headdim):
    return (chunk_size in _FLASHINFER_SSD_VALID_M_MODES
            and dstate in _FLASHINFER_SSD_VALID_M_MODES
            and headdim in _FLASHINFER_SSD_VALID_HEAD_DIMS)


@functools.cache
def _get_flashinfer_ssd_kernel(chunk_size, nheads, headdim, dstate, ngroups,
                               has_varlen):
    """Get or compile a cached FlashInfer SSDCombined kernel (SM100+ only).

    has_varlen is in the cache key because flashinfer compiles distinct
    kernels per mode.
    """
    logger.info_once(
        f"Using FlashInfer fused SSD kernel for Mamba2 prefill "
        f"(has_varlen={has_varlen})",
        key=f"flashinfer_ssd_prefill_{has_varlen}")
    return SSDCombined(
        chunk_size=chunk_size,
        nheads=nheads,
        headdim=headdim,
        dstate=dstate,
        ngroups=ngroups,
        io_dtype=torch.bfloat16,
        state_dtype=torch.bfloat16,
        has_d=True,
        d_has_hdim=False,
        has_initial_states=True,
        has_varlen=has_varlen,
        has_z=False,
        seq_idx_dtype=torch.int32,
    )


def _mamba_chunk_scan_flashinfer_fwd(
    x,
    dt,
    A,
    B,
    C,
    chunk_size,
    D=None,
    dt_bias=None,
    initial_states=None,
    seq_idx=None,
    chunk_indices=None,
    chunk_offsets=None,
    cu_seqlens=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    out=None,
    return_final_states=False,
    state_dtype=None,
):
    """FlashInfer fused SSD forward.

    cu_seqlens != None means varlen (packed batch=1; fstate per-sequence).
    cu_seqlens == None means non-varlen (fstate per-batch).
    """
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    io_dtype = torch.bfloat16
    has_varlen = cu_seqlens is not None

    ssd = _get_flashinfer_ssd_kernel(chunk_size, nheads, headdim, dstate,
                                     ngroups, has_varlen)
    num_seqs = cu_seqlens.shape[0] - 1 if has_varlen else batch

    # Pad seqlen to chunk_size boundary — padded tokens use dt=-100
    # so softplus ≈ 0, contributing nothing to state or output.
    pad_len = pad_up(seqlen, chunk_size) - seqlen
    if pad_len > 0:
        x = F.pad(x, (0, 0, 0, 0, 0, pad_len))
        B = F.pad(B, (0, 0, 0, 0, 0, pad_len))
        C = F.pad(C, (0, 0, 0, 0, 0, pad_len))
        dt = F.pad(dt, (0, 0, 0, pad_len), value=-100.0)
        if seq_idx is not None:
            seq_idx = F.pad(seq_idx, (0, pad_len), value=int(num_seqs - 1))

    # The SSDCombined kernel was compiled with B/C compact stride
    # `b.strides[0] = N*G` (N=dstate=const), so the runtime check requires
    # the seqlen stride divisible by N. Upstream split-then-view producers
    # (e.g. AutoDeploy's projected_states split) leave seqlen-stride equal
    # to the full projection width, which need not be a multiple of N.
    # Materialize contiguous copies when necessary.
    if not B.is_contiguous():
        B = B.contiguous()
    if not C.is_contiguous():
        C = C.contiguous()
    if not x.is_contiguous():
        x = x.contiguous()
    if not dt.is_contiguous():
        dt = dt.contiguous()

    if x.dtype != io_dtype:
        x = x.to(io_dtype)
        B = B.to(io_dtype)
        C = C.to(io_dtype)
        dt = dt.to(io_dtype)

    D_bf16 = D.to(io_dtype) if D is not None and D.dtype != io_dtype else D

    if initial_states is not None:
        fi_initial_states = (initial_states if initial_states.dtype == io_dtype
                             else initial_states.to(io_dtype))
    else:
        fi_initial_states = x.new_zeros(num_seqs,
                                        nheads,
                                        headdim,
                                        dstate,
                                        dtype=io_dtype)

    if has_varlen and (chunk_indices is None or chunk_offsets is None):
        chunk_indices, chunk_offsets = (
            cu_seqlens_to_chunk_indices_offsets_triton(cu_seqlens,
                                                       chunk_size,
                                                       total_seqlens=seqlen))
    elif not has_varlen:
        # Non-varlen kernel doesn't read these.
        seq_idx = None
        chunk_indices = None
        chunk_offsets = None

    # raw_out is only consumed by the fused transpose path.
    # TODO: enable for non-varlen — batch=1 is just a gate change; batch>1
    # needs a batch axis in the transpose kernel grid.
    use_fast_transpose = out is not None and has_varlen and batch == 1
    if use_fast_transpose:
        padded_seqlen = x.shape[1]
        nchunks = padded_seqlen // chunk_size
        raw_out = torch.empty(batch,
                              nheads,
                              headdim,
                              nchunks,
                              chunk_size,
                              dtype=io_dtype,
                              device=x.device)
    else:
        raw_out = None

    out_view, fstate = ssd.run(
        x,
        dt,
        A,
        B,
        C,
        D=D_bf16,
        dt_bias=dt_bias,
        dt_softplus=dt_softplus,
        dt_limit=dt_limit,
        initial_states=fi_initial_states,
        seq_idx=seq_idx,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        out=raw_out,
        return_final_states=True,
    )

    if out is not None:
        if use_fast_transpose:
            num_prefill_tokens = out.shape[1]
            dst = out.view(num_prefill_tokens, nheads * headdim)
            ssd_output_transpose(raw_out, dst, num_prefill_tokens)
        else:
            out.copy_(out_view[:, :seqlen])

    if state_dtype is not None and fstate.dtype != state_dtype:
        fstate = fstate.to(state_dtype)

    # Match Triton return: varlen yields (final, varlen) — same tensor for
    # batch=1 packed; non-varlen has no varlen_states.
    if has_varlen:
        return (fstate, fstate) if return_final_states else fstate
    else:
        return fstate if return_final_states else None


def is_int_pow_2(n):
    return isinstance(n, int) and n > 0 and (n & (n - 1)) == 0


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
        chunk_indices=None,
        chunk_offsets=None,
        cu_seqlens=None,
        dt_softplus=False,
        dt_limit=(0.0, float("inf")),
        state_dtype=None,
        out=None,
):
    assert is_int_pow_2(chunk_size), "chunk_size must be integer power of 2"
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
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
        if cu_seqlens is None:
            assert initial_states.shape == (batch, nheads, headdim, dstate)
        else:
            assert initial_states.shape == (
                len(cu_seqlens) - 1,
                nheads,
                headdim,
                dstate,
            )

    # This function executes 5 sub-functions for computing mamba
    # - a good resource is the blog https://goombalab.github.io/blog/2024/mamba2-part3-algorithm/
    #   which has a minimal implementation to understand the below operations
    # - as explained by the blog, mamba is a special case of causal attention
    # - the idea is to chunk the attention matrix and compute each
    #   submatrix separately using different optimizations.
    # - see the blog and paper for a visualization of the submatrices
    #   which we refer to in the comments below

    # 1. Compute chunked cumsum of A * dt
    # - here dt may go through a softplus activation
    dA_cumsum, dt = _chunk_cumsum_fwd(dt,
                                      A,
                                      chunk_size,
                                      dt_bias=dt_bias,
                                      dt_softplus=dt_softplus,
                                      dt_limit=dt_limit)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    states = _chunk_state_fwd(B,
                              x,
                              dt,
                              dA_cumsum,
                              seq_idx=seq_idx,
                              states_in_fp32=True)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    # - for handling chunked prefill, this requires i) initial_states
    #   ii) seq_idx iii) is_cont_batched and (iv) chunk_offsets to be all specified.
    # - When a new seq_idx is detected, we will stop passing the prev_state
    #   and switch accordingly to the init_state corresponding to the new seq_idx.
    # - We will also make sure that the dA_cumsum is taken only from the start of the
    #   sequence (hence we need the full dA_cumsum tensor and not just the values at chunk boundaries)
    # - this will ensure that states will be updated with the rightmost flushed seq_idx
    #   of the previous chunk. This implies that the first chunk of states is either 0
    #   or equal to init_states of the first example.
    states, final_states = _state_passing_fwd(
        rearrange(states, "... p n -> ... (p n)"),
        dA_cumsum,
        initial_states=(rearrange(initial_states, "... p n -> ... (p n)")
                        if initial_states is not None else None),
        seq_idx=seq_idx,
        chunk_size=chunk_size,
        out_dtype=state_dtype if state_dtype is not None else C.dtype,
        is_cont_batched=cu_seqlens is not None,
        chunk_offsets=chunk_offsets,
    )
    states, final_states = (rearrange(t, "... (p n) -> ... p n", n=dstate)
                            for t in [states, final_states])

    # 4. Compute batched matrix multiply for C_j^T B_i terms
    CB = _bmm_chunk_fwd(C,
                        B,
                        chunk_size,
                        seq_idx=seq_idx,
                        output_dtype=torch.float32)

    # 5. Scan and compute the diagonal blocks, taking into
    #    account past causal states.
    # - if initial states are provided, then states information will be
    #   augmented with initial_states.
    # - to do this properly, we need to account for example changes in
    #   the continuous batch, therefore we introduce pseudo chunks, which is
    #   a chunk that is split up each time an example changes.
    # - in each (pseudo) chunk, we detect if the previous (pseudo) chunk had
    #   a seq_idx change, in which case we take states information from
    #   init_states.
    out_x = _chunk_scan_fwd(
        CB,
        x,
        dt,
        dA_cumsum,
        C,
        states,
        D=D,
        z=z,
        seq_idx=seq_idx,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        initial_states=initial_states,
        out=out,
    )
    if cu_seqlens is None:
        return out_x, dt, dA_cumsum, states, final_states
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
            initial_states=initial_states,
        )
        return out_x, dt, dA_cumsum, states, final_states, varlen_states


def mamba_chunk_scan_combined(
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
    chunk_indices=None,
    chunk_offsets=None,
    cu_seqlens=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
    out=None,
    return_final_states=False,
    return_varlen_states=False,
    state_dtype=None,
):
    """
    Argument:
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, seqlen, nheads)
        A: (nheads)
        B: (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, ngroups, dstate)
        chunk_size: int
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
        dt_bias: (nheads,)
        initial_states: (batch, nheads, headdim, dstate)
        seq_idx: (batch, seqlen)
        cu_seqlens: (num_sequences + 1) or None, only used if return_varlen_states is True
        dt_softplus: Whether to apply softplus to dt
        out: Preallocated output tensor
        state_dtype: The data type of the ssm state
    """

    if not return_varlen_states:
        cu_seqlens = None
    else:
        assert (cu_seqlens is not None
                ), "cu_seqlens must be provided if return_varlen_states is True"

    # FlashInfer fused CUTLASS kernel on Blackwell (SM100+); both varlen and
    # non-varlen route here based on cu_seqlens. Falls back to Triton when the
    # MMA tile constraints on (chunk_size, dstate, headdim) aren't met.
    dstate = B.shape[-1]
    headdim = x.shape[-1]
    flashinfer_eligible = (z is None and is_sm_100f())
    if flashinfer_eligible and _flashinfer_ssd_supported(
            chunk_size, dstate, headdim):
        return _mamba_chunk_scan_flashinfer_fwd(
            x,
            dt,
            A,
            B,
            C,
            chunk_size,
            D=D,
            dt_bias=dt_bias,
            initial_states=initial_states,
            seq_idx=seq_idx,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            cu_seqlens=cu_seqlens,
            dt_softplus=dt_softplus,
            dt_limit=dt_limit,
            out=out,
            return_final_states=return_final_states,
            state_dtype=state_dtype,
        )
    if flashinfer_eligible:
        logger.info_once(
            f"FlashInfer SSD unavailable for chunk_size={chunk_size}, "
            f"dstate={dstate}, headdim={headdim} "
            f"(requires chunk_size/dstate in {_FLASHINFER_SSD_VALID_M_MODES} "
            f"and headdim in {_FLASHINFER_SSD_VALID_HEAD_DIMS}); "
            f"falling back to Triton SSD prefill",
            key=f"triton_ssd_fallback_{chunk_size}_{dstate}_{headdim}",
        )

    out_x, dt_out, dA_cumsum, states, final_states, *rest = (
        _mamba_chunk_scan_combined_fwd(
            x,
            dt,
            A,
            B,
            C,
            chunk_size,
            D=D,
            z=z,
            dt_bias=dt_bias,
            initial_states=initial_states,
            seq_idx=seq_idx,
            chunk_indices=chunk_indices,
            chunk_offsets=chunk_offsets,
            cu_seqlens=cu_seqlens,
            dt_softplus=dt_softplus,
            dt_limit=dt_limit,
            out=out,
            state_dtype=state_dtype,
        ))
    if not return_varlen_states:
        if not return_final_states:
            return
        else:
            return final_states
    else:
        varlen_states = rest[0]
        return ((varlen_states) if not return_final_states else
                (final_states, varlen_states))
