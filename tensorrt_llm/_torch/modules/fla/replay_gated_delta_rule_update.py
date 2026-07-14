# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#
# Gated-delta-rule (GDN) replay state update for MTP speculative decoding.
#
# This is the GDN analogue of the Mamba SSM
# ``replay_selective_state_update`` kernel.  Instead of caching a full
# intermediate recurrent state for every draft position (the "legacy" MTP
# path), it keeps a compact double-buffered *history* of the raw per-token
# inputs needed to advance the gated-delta-rule recurrence (k, v, g, beta)
# and "replays" (re-runs) that history from a checkpoint to reconstruct the
# committed state at the start of each verification step.
#
# Contract shared with the Mamba replay path (see
# ``pyexecutor/mamba_cache_manager.py`` ``update_mamba_states`` and
# ``modules/mamba/replay_selective_state_update.py``):
#   * ``temporal`` (``ssm_states``) holds a *checkpoint* recurrent state.
#   * ``prev_num_accepted_tokens[slot]`` (PNAT) counts committed-but-unfolded
#     tokens whose inputs live in the active history buffer at ``[0, PNAT)``.
#   * ``cache_buf_idx[slot]`` selects the active history buffer (double
#     buffered along a size-2 dimension).
#   * ``wrote_checkpoint = PNAT + T > HISTORY`` decides whether this step
#     folds the PNAT tokens into ``temporal`` and stages the current tokens in
#     the other buffer (write path) or appends the current tokens to the active
#     buffer (nowrite path).  ``update_mamba_states`` bumps PNAT / flips
#     ``cache_buf_idx`` after acceptance using the *same* predicate.

from typing import Optional

import torch
import triton
import triton.language as tl

from tensorrt_llm._torch.modules.fla.op import exp


@triton.jit
def _gdc_wait():
    tl.inline_asm_elementwise(
        "griddepcontrol.wait; // dummy $0",
        "=r,~{memory}",
        [],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@triton.jit(do_not_specialize=["T"])
def _replay_gated_delta_rule_update_kernel(
    q,
    k,
    v,
    g,
    beta,
    o,
    ssm_states,
    old_k,
    old_v,
    old_g,
    old_beta,
    cache_buf_idx,
    prev_num_accepted_tokens,
    state_batch_indices,
    scale,
    T,
    s_state_slot,
    s_state_hv,
    s_state_v,
    s_state_k,
    HISTORY: tl.constexpr,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    USE_QK_L2NORM_IN_KERNEL: tl.constexpr,
    LAUNCH_WITH_PDL: tl.constexpr,
):
    if LAUNCH_WITH_PDL:
        _gdc_wait()

    i_v = tl.program_id(0)
    i_ndhv = tl.program_id(1)
    i_nd = i_ndhv // HV
    i_hv = i_ndhv % HV
    # GVA: map value head to its key/query head group.
    i_h = i_hv // (HV // H)

    slot = tl.load(state_batch_indices + i_nd).to(tl.int64)
    pnat = tl.load(prev_num_accepted_tokens + slot).to(tl.int32)
    buf = tl.load(cache_buf_idx + slot).to(tl.int32)

    wrote_checkpoint = (pnat + T) > HISTORY
    write_buf = tl.where(wrote_checkpoint, 1 - buf, buf).to(tl.int64)
    write_off = tl.where(wrote_checkpoint, 0, pnat)

    o_k = tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_k[:, None] & mask_v[None, :]

    # Checkpoint state from the pool: logical layout [slots, HV, V, K].
    # Strides are explicit so this works for both the dense Mixed-manager pool
    # and the strided unified C++ pool view.
    p_h = (ssm_states + slot * s_state_slot + i_hv * s_state_hv +
           o_v[None, :] * s_state_v + o_k[:, None] * s_state_k)
    b_h = tl.load(p_h, mask=mask_h, other=0.0).to(tl.float32)

    # History buffer base pointers for the active (read) buffer.
    #   old_k:    [cache, 2, HISTORY, H,  K]
    #   old_v:    [cache, 2, HISTORY, HV, V]
    #   old_g:    [cache, 2, HISTORY, HV]
    #   old_beta: [cache, 2, HISTORY, HV]
    buf64 = buf.to(tl.int64)
    rbase_k = old_k + (slot * 2 + buf64) * HISTORY * H * K
    rbase_v = old_v + (slot * 2 + buf64) * HISTORY * HV * V
    rbase_g = old_g + (slot * 2 + buf64) * HISTORY * HV
    rbase_beta = old_beta + (slot * 2 + buf64) * HISTORY * HV

    # --- Replay committed-but-unfolded tokens [0, PNAT) to reconstruct the
    #     state at the start of this verification step (state only, no output).
    for t in range(0, pnat):
        b_k = tl.load(rbase_k + t * H * K + i_h * K + o_k, mask=mask_k,
                      other=0.0).to(tl.float32)
        b_v = tl.load(rbase_v + t * HV * V + i_hv * V + o_v, mask=mask_v,
                      other=0.0).to(tl.float32)
        b_g = tl.load(rbase_g + t * HV + i_hv).to(tl.float32)
        b_beta = tl.load(rbase_beta + t * HV + i_hv).to(tl.float32)
        if USE_QK_L2NORM_IN_KERNEL:
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k)) + 1e-6)
        b_h *= exp(b_g)
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        b_v *= b_beta
        b_h += b_k[:, None] * b_v[None, :]

    # State immediately before the current tokens = new checkpoint (write path).
    b_h_checkpoint = b_h

    # History buffer base pointers for the write buffer.
    wbase_k = old_k + (slot * 2 + write_buf) * HISTORY * H * K
    wbase_v = old_v + (slot * 2 + write_buf) * HISTORY * HV * V
    wbase_g = old_g + (slot * 2 + write_buf) * HISTORY * HV
    wbase_beta = old_beta + (slot * 2 + write_buf) * HISTORY * HV

    # k / g / beta are shared across V blocks (and k across the GVA group), so
    # only one program writes each to avoid redundant traffic.
    is_first_hv_in_group = (i_hv % (HV // H)) == 0

    for t in range(0, T):
        b_q = tl.load(q + ((i_nd * T + t) * H + i_h) * K + o_k, mask=mask_k,
                      other=0.0).to(tl.float32)
        b_k = tl.load(k + ((i_nd * T + t) * H + i_h) * K + o_k, mask=mask_k,
                      other=0.0).to(tl.float32)
        b_v = tl.load(v + ((i_nd * T + t) * HV + i_hv) * V + o_v, mask=mask_v,
                      other=0.0).to(tl.float32)
        b_g = tl.load(g + (i_nd * T + t) * HV + i_hv).to(tl.float32)
        b_beta = tl.load(beta + (i_nd * T + t) * HV + i_hv).to(tl.float32)

        # Append raw inputs to the write buffer at write_off + t.
        wpos = write_off + t
        if i_v == 0:
            if is_first_hv_in_group:
                tl.store(wbase_k + wpos * H * K + i_h * K + o_k,
                         b_k.to(old_k.dtype.element_ty),
                         mask=mask_k)
            tl.store(wbase_g + wpos * HV + i_hv, b_g.to(old_g.dtype.element_ty))
            tl.store(wbase_beta + wpos * HV + i_hv,
                     b_beta.to(old_beta.dtype.element_ty))
        tl.store(wbase_v + wpos * HV * V + i_hv * V + o_v,
                 b_v.to(old_v.dtype.element_ty),
                 mask=mask_v)

        if USE_QK_L2NORM_IN_KERNEL:
            b_q = b_q / (tl.sqrt(tl.sum(b_q * b_q)) + 1e-6)
            b_k = b_k / (tl.sqrt(tl.sum(b_k * b_k)) + 1e-6)
        b_q = b_q * scale
        b_h *= exp(b_g)
        b_v -= tl.sum(b_h * b_k[:, None], 0)
        b_v *= b_beta
        b_h += b_k[:, None] * b_v[None, :]
        b_o = tl.sum(b_h * b_q[:, None], 0)
        tl.store(o + ((i_nd * T + t) * HV + i_hv) * V + o_v,
                 b_o.to(o.dtype.element_ty),
                 mask=mask_v)

    # On the write path, fold the PNAT tokens into the pool checkpoint.  On the
    # nowrite path the checkpoint is left untouched (temporal keeps lagging).
    if wrote_checkpoint:
        tl.store(p_h,
                 b_h_checkpoint.to(ssm_states.dtype.element_ty),
                 mask=mask_h)


def replay_gated_delta_rule_update(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    ssm_states: torch.Tensor,
    old_k: torch.Tensor,
    old_v: torch.Tensor,
    old_g: torch.Tensor,
    old_beta: torch.Tensor,
    cache_buf_idx: torch.Tensor,
    prev_num_accepted_tokens: torch.Tensor,
    state_batch_indices: torch.Tensor,
    replay_step_width: int,
    replay_history_size: int,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = True,
    output: Optional[torch.Tensor] = None,
    launch_with_pdl: bool = False,
) -> torch.Tensor:
    """Fused GDN replay state update for the MTP target-verify decode path.

    Args:
        q, k: current-step queries/keys, shape ``[num_decodes, T, H, K]``.
        v: current-step values, shape ``[num_decodes, T, HV, V]``.
        g, beta: per-head decay / beta, shape ``[num_decodes, T, HV]``.
        ssm_states: recurrent state pool ``[slots, HV, V, K]`` (checkpoint,
            updated in place on the write path).
        old_k, old_v, old_g, old_beta: this layer's double-buffered history
            caches with a leading ``[cache, 2, HISTORY, ...]`` layout.
        cache_buf_idx: ``[cache]`` int active-buffer selector (read only here;
            flipped by ``update_mamba_states`` after acceptance).
        prev_num_accepted_tokens: ``[cache]`` int PNAT (read only here).
        state_batch_indices: ``[num_decodes]`` int cache slot per decode.
        replay_step_width: fixed ``T`` (``runtime_draft_len + 1``).
        replay_history_size: history buffer capacity along the window dim.
        output: optional preallocated ``[num_decodes, T, HV, V]`` output.

    Returns:
        Output tensor ``[num_decodes, T, HV, V]``.
    """
    num_decodes, T, H, K = q.shape
    HV = v.shape[2]
    V = v.shape[3]
    assert T == replay_step_width, (
        f"runtime token width {T} must match fixed replay step width "
        f"{replay_step_width}")
    assert k.shape == (num_decodes, T, H, K)
    assert v.shape == (num_decodes, T, HV, V)
    assert g.shape[-1] == HV and beta.shape[-1] == HV

    if scale is None:
        scale = K**-0.5

    BK = triton.next_power_of_2(K)
    BV = min(triton.next_power_of_2(V), 8)
    NV = triton.cdiv(V, BV)
    assert triton.cdiv(K, BK) == 1, "K larger than one tensor tile is unsupported"

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()

    if output is None:
        o = q.new_empty(num_decodes, T, HV, V)
    else:
        o = output

    grid = (NV, num_decodes * HV)
    _replay_gated_delta_rule_update_kernel[grid](
        q,
        k,
        v,
        g,
        beta,
        o,
        ssm_states,
        old_k,
        old_v,
        old_g,
        old_beta,
        cache_buf_idx,
        prev_num_accepted_tokens,
        state_batch_indices,
        scale,
        T,
        ssm_states.stride(0),
        ssm_states.stride(1),
        ssm_states.stride(2),
        ssm_states.stride(3),
        HISTORY=replay_history_size,
        H=H,
        HV=HV,
        K=K,
        V=V,
        BK=BK,
        BV=BV,
        USE_QK_L2NORM_IN_KERNEL=use_qk_l2norm_in_kernel,
        LAUNCH_WITH_PDL=launch_with_pdl,
        num_warps=1,
        num_stages=3,
    )
    return o
