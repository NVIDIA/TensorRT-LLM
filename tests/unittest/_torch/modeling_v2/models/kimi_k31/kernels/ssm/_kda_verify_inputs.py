# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inputs of one Kimi K3.1 KDA fused verify call in the target's runtime layouts.

The q / k / v / output-gate projections are column sections of one fused per-token projection
(row stride 4 * H * 128), beta is the H-wide tail of the padded [f_a | beta] projection, the conv
caches are dim-contiguous bf16 ``[pool, H * 128, 3 + num_spec]``, the replay caches hold plausible
pending drafts (L2-normalized k, gates in (lower_bound, 0), SiLU'd v, sigmoid'd beta), the beta
cache rows are padded to 16 bytes, the state is fp32 V-first and the requests sit on a random
subset of the pool's slots with accepted counts cycling over 0..num_spec.
"""

from __future__ import annotations

import torch

K = V = 128
W = 4
LOWER_BOUND = -5.0
ONORM_EPS = 1e-5
_BETA_ALIGN_HEADS = 8  # 16 bytes of bf16


def build(
    N: int,
    H: int,
    num_spec: int,
    pool: int | None = None,
    seed: int = 0,
    state_layer_view: bool = False,
) -> dict:
    """Keyword arguments of ``kda_mtp_decode_v2.k31_kda_mtp_verify`` (and of the OpWrapper).

    ``state_layer_view``: the state is layer 1 of a 3-layer pool ``[pool, 3, H, V, K]`` (slot
    stride 3 * H * V * K), like the runtime's per-layer view of the SSM cache.
    """
    dev = "cuda"
    f32, bf16 = torch.float32, torch.bfloat16
    M = num_spec
    T = M + 1
    R = N * T
    dim = H * K
    pool = N if pool is None else pool
    gen = torch.Generator(device=dev).manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=gen, device=dev, dtype=f32)

    qkvg = randn(1, R, 4 * dim).to(bf16)
    bfa_width = (K + H + 7) // 8 * 8
    bfa = randn(1, R, bfa_width).to(bf16)

    def conv_cache():
        return randn(pool, W - 1 + M, dim).to(bf16).transpose(1, 2)

    if state_layer_view:
        state = (randn(pool, 3, H, V, K) * 0.5)[:, 1]
    else:
        state = randn(pool, H, V, K) * 0.5
    k_cache = (
        torch.nn.functional.normalize(randn(pool, M, H, K), p=2, dim=-1)
        .reshape(pool, M, dim)
        .to(bf16)
    )
    g_cache = LOWER_BOUND * torch.sigmoid(randn(pool, M, dim))
    v_raw = 2.0 * randn(pool, M, dim)
    v_cache = (v_raw * torch.sigmoid(v_raw)).to(bf16)
    padded_heads = (H + _BETA_ALIGN_HEADS - 1) // _BETA_ALIGN_HEADS * _BETA_ALIGN_HEADS
    beta_cache = torch.zeros(pool, M, padded_heads, dtype=bf16, device=dev)[..., :H]
    beta_cache.copy_(torch.sigmoid(randn(pool, M, H)))
    slots = torch.randperm(pool, generator=gen, device=dev)[:N].to(torch.int32)
    accepted = torch.zeros(pool, dtype=torch.int32, device=dev)
    accepted[slots.long()] = (torch.arange(N, device=dev) % (M + 1)).to(torch.int32)
    return dict(
        x_q=qkvg[..., :dim].view(1, R, H, K),
        x_k=qkvg[..., dim : 2 * dim].view(1, R, H, K),
        x_v=qkvg[..., 2 * dim : 3 * dim].view(1, R, H, V),
        w_q=(randn(dim, W) * 0.5).contiguous(),
        w_k=(randn(dim, W) * 0.5).contiguous(),
        w_v=(randn(dim, W) * 0.5).contiguous(),
        cs_q=conv_cache(),
        cs_k=conv_cache(),
        cs_v=conv_cache(),
        g=randn(1, R, H, K).to(bf16),
        beta=bfa[..., K : K + H],
        A_log=randn(H) * 0.5,
        dt_bias=randn(dim) * 0.5,
        recurrent_state=state,
        k_cache=k_cache,
        g_cache=g_cache,
        v_cache=v_cache,
        beta_cache=beta_cache,
        ssm_state_indices=slots,
        num_spec=M,
        num_accepted_tokens=accepted,
        lower_bound=LOWER_BOUND,
        scale=K**-0.5,
        onorm_g=qkvg[..., 3 * dim :].view(1, R, H, V),
        onorm_weight=torch.linspace(0.75, 1.25, V, dtype=f32, device=dev),
        onorm_eps=ONORM_EPS,
    )


#: The tensors a verify call writes in place.
MUTATED = ("recurrent_state", "cs_q", "cs_k", "cs_v", "k_cache", "g_cache", "v_cache", "beta_cache")
