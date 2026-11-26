# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GDP with packed prefill substeps and indexed per-request decode state."""

import torch
import triton
import triton.language as tl
from torch.nn import functional as F

from tensorrt_llm._torch.modules.fla.fused_recurrent import fused_recurrent_gated_delta_rule_update


@triton.jit
def _decode_update(
    Q,
    K,
    V,
    DECAY,
    BETA,
    POOL,
    SLOTS,
    OUT,
    SLOT_STRIDE: tl.constexpr,
    HEAD_STRIDE: tl.constexpr,
    M: tl.constexpr,
    H: tl.constexpr,
    DK: tl.constexpr,
    DV: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    seq, head, tile = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    # Coalesced cache planes can place a live slot more than 2**31 elements away.
    slot = tl.load(SLOTS + seq).to(tl.int64)
    k_col = tl.arange(0, DK)
    v_row = tile * BLOCK_V + tl.arange(0, BLOCK_V)
    address = POOL + slot * SLOT_STRIDE + head * HEAD_STRIDE + v_row[:, None] * DK + k_col[None, :]
    live = (slot >= 0) & (v_row < DV)
    state = tl.load(address, mask=live[:, None], other=0.0).to(tl.float32)
    state *= tl.load(DECAY + seq * H + head)
    for j in tl.static_range(M):
        offset = (seq * M + j) * H + head
        key = tl.load(K + offset * DK + k_col).to(tl.float32)
        key *= tl.rsqrt(tl.sum(key * key, axis=0) + 1e-6)
        value = tl.load(V + offset * DV + v_row, v_row < DV, 0.0).to(tl.float32)
        beta = tl.load(BETA + offset).to(tl.float32)
        delta = (value - tl.sum(state * key[None, :], axis=1)) * beta
        state += delta[:, None] * key[None, :]
    tl.store(address, state, mask=live[:, None])
    query = tl.load(Q + (seq * H + head) * DK + k_col).to(tl.float32)
    query *= tl.rsqrt(tl.sum(query * query, axis=0) + 1e-6) * (DK**-0.5)
    value = tl.sum(state * query[None, :], axis=1)
    tl.store(OUT + (seq * H + head) * DV + v_row, value, mask=v_row < DV)


def _prefill(q, k, v, log_decay, beta, pool, slots, resumed, bounds):
    tokens, heads, key_dim = q.shape
    steps = v.shape[1]
    queries = q.new_zeros(tokens, steps, heads, key_dim)
    queries[:, -1] = q
    # The first Householder update applies the token's log decay. The other
    # updates retain all prior state. Pass log gates directly to avoid an
    # epsilon offset when converting very small linear gates back to logs.
    gates = log_decay.new_zeros(tokens, steps, heads)
    gates[:, 0] = log_decay
    initial = torch.where(resumed[:, None, None, None].bool(), pool[slots], 0.0)
    # The recurrent op reads and writes a contiguous [request, head, V, K] pool.
    # The cache manager may expose a strided plane, so scatter the final states back.
    local_slots = torch.arange(slots.numel(), device=slots.device, dtype=torch.int64)
    output = fused_recurrent_gated_delta_rule_update(
        q=queries.flatten(0, 1).unsqueeze(0),
        k=k.flatten(0, 1).unsqueeze(0),
        v=v.flatten(0, 1).unsqueeze(0),
        g=gates.flatten(0, 1).unsqueeze(0),
        beta=beta.flatten(0, 1).float().unsqueeze(0),
        scale=key_dim**-0.5,
        initial_state_source=initial,
        initial_state_indices=local_slots,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=bounds * steps,
    )
    pool[slots] = initial
    return output.squeeze(0).unflatten(0, (tokens, steps))[:, -1]


def gdp_mixer(core, layer, x, md, layer_idx):
    if x.shape[0] == 0:
        return x.new_empty((0, core.gdp_v))
    projection = core.mm(x, layer["gdp_in"])
    gate, raw, beta, time = projection.split(
        (core.gdp_v, core.gdp_conv, core.m * core.gdp_heads, core.gdp_heads), dim=-1
    )
    contexts = md.num_contexts
    ctx_tokens = md.num_ctx_tokens if contexts else 0
    cache = md.kv_cache_manager.mamba_layer_cache(layer_idx)
    meta = md.mamba_metadata
    slots = meta.state_indices[: md.seq_lens.shape[0]]
    decodes = slots.shape[0] - contexts
    if ctx_tokens + decodes != x.shape[0]:
        raise ValueError("GDP expects exactly one token per decode request.")
    rows = raw
    if contexts:
        offsets = meta.query_start_loc[: contexts + 1]
        resumed = meta.has_initial_states[:contexts]
        convolved = torch.empty_like(raw[:ctx_tokens], memory_format=torch.contiguous_format)
        torch.ops.trtllm.causal_conv1d_fwd(
            raw[:ctx_tokens].T,
            layer["gdp_conv"],
            None,
            cache.conv,
            offsets,
            slots[:contexts],
            resumed,
            True,
            -1,
            convolved.T,
        )
        rows = convolved
    if decodes:
        tail = raw[ctx_tokens:]
        torch.ops.trtllm.causal_conv1d_update(
            tail.unsqueeze(-1),
            cache.conv,
            layer["gdp_conv"],
            None,
            True,
            None,
            slots[contexts:],
            -1,
        )
        if contexts:
            rows = torch.cat((convolved, tail))
    h, m, dv, dk = core.gdp_heads, core.m, core.gdp_dv, core.gdp_dk
    # The family runs one key group per value head.
    v, k, q = rows.split((m * h * dv, m * h * dk, h * dk), dim=-1)
    v = v.reshape(-1, m, h, dv)
    k = k.reshape(-1, m, h, dk)
    q = q.reshape(-1, h, dk)
    log_decay = -layer["gdp_a_log"].float().exp() * F.softplus(time.float() + layer["gdp_dt_bias"])
    log_decay = log_decay.clamp_min(-core.gdp_g_clamp_value)
    beta = beta.sigmoid().view(-1, m, h)
    out = x.new_empty((x.shape[0], h, dv))
    if contexts:
        out[:ctx_tokens] = _prefill(
            q[:ctx_tokens],
            k[:ctx_tokens],
            v[:ctx_tokens],
            log_decay[:ctx_tokens],
            beta[:ctx_tokens],
            cache.temporal,
            slots[:contexts].long(),
            resumed,
            meta.query_start_loc_long[: contexts + 1],
        )
    if decodes:
        state = cache.temporal
        _decode_update[(decodes, h, triton.cdiv(dv, 8))](
            q[ctx_tokens:].contiguous(),
            k[ctx_tokens:].contiguous(),
            v[ctx_tokens:].contiguous(),
            log_decay[ctx_tokens:].exp().contiguous(),
            beta[ctx_tokens:].contiguous(),
            state,
            slots[contexts:],
            out[ctx_tokens:],
            state.stride(0),
            state.stride(1),
            m,
            h,
            dk,
            dv,
            8,
        )
    gated = out.float() * F.silu(gate.float().reshape_as(out))
    normalized = gated * torch.rsqrt(gated.square().mean(-1, keepdim=True) + core.gdp_eps)
    return (normalized.flatten(1) * layer["gdp_norm"].float()).to(x.dtype)
