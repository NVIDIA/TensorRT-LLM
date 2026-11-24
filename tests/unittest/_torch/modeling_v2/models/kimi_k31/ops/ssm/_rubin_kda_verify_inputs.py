# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Inputs of the Rubin KDA verify and fab entries in the target's runtime layouts.

Not a test file: the entry tests import it by name from their own directory. The builders take a
device, so the CPU reference test uses them too.

The q / k / v / output-gate projections are column sections of one fused per-token projection
(row stride 4 * H * 128), beta (and with the fused gate f_a) columns of the padded [f_a | b]
projection, the conv caches dim-contiguous bf16 `[pool, H * 128, 3 + num_spec]`, the replay caches
plausible pending drafts, the beta cache rows padded to 16 bytes, the state V-first, the requests on
a random subset of the pool's slots with accepted counts cycling over 0..num_spec. With `stacked`
the projections are column views of the specialized targets' one `[q | k | v | g | f_a | b | 0]`
GEMM output.
"""

from __future__ import annotations

import torch
from _rubin_kda_prefill_inputs import stacked_projection

K = V = 128
W = 4
LOWER_BOUND = -5.0
ONORM_EPS = 1e-5
#: The tensors a verify call writes in place.
MUTATED = ("recurrent_state", "cs_q", "cs_k", "cs_v", "k_cache", "g_cache", "v_cache", "beta_cache")


def verify_args(spec: dict, seed: int, device="cuda") -> dict:
    """Keyword arguments of `rubin_kda_mtp_verify` for a cell spec (N, H, num_spec, pool,
    state_dtype, gate "g" | "fa", out "bf16" | "mxfp8" | "mxfp8_exact", use_pdl, the two early
    releases and stacked)."""
    N, H, M = spec["N"], spec["H"], spec["num_spec"]
    f32, bf16 = torch.float32, torch.bfloat16
    R, dim = N * (M + 1), H * K
    pool = spec.get("pool") or N
    gen = torch.Generator(device=device).manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=gen, device=device, dtype=f32)

    if spec.get("stacked"):
        proj = stacked_projection(R, H, gen, device).unsqueeze(0)
        qkvg, fab = proj[..., : 4 * dim], proj[..., 4 * dim :]
    else:
        qkvg = randn(1, R, 4 * dim).to(bf16)
        fab = randn(1, R, (K + H + 7) // 8 * 8).to(bf16)
    padded = (H + 7) // 8 * 8
    beta_cache = torch.zeros(pool, M, padded, dtype=bf16, device=device)[..., :H]
    beta_cache.copy_(torch.sigmoid(randn(pool, M, H)))
    v_raw = 2.0 * randn(pool, M, dim)
    slots = torch.randperm(pool, generator=gen, device=device)[:N].to(torch.int32)
    accepted = torch.zeros(pool, dtype=torch.int32, device=device)
    accepted[slots.long()] = (torch.arange(N, device=device) % (M + 1)).to(torch.int32)
    state = (randn(pool, H, V, K) * 0.5).to(getattr(torch, spec.get("state_dtype", "float32")))
    args = dict(
        x_q=qkvg[..., :dim].view(1, R, H, K),
        x_k=qkvg[..., dim : 2 * dim].view(1, R, H, K),
        x_v=qkvg[..., 2 * dim : 3 * dim].view(1, R, H, V),
        w_q=(randn(dim, W) * 0.5).contiguous(),
        w_k=(randn(dim, W) * 0.5).contiguous(),
        w_v=(randn(dim, W) * 0.5).contiguous(),
        cs_q=randn(pool, W - 1 + M, dim).to(bf16).transpose(1, 2),
        cs_k=randn(pool, W - 1 + M, dim).to(bf16).transpose(1, 2),
        cs_v=randn(pool, W - 1 + M, dim).to(bf16).transpose(1, 2),
        g=randn(1, R, H, K).to(bf16),
        fa=None,
        w_fb=None,
        beta=fab[..., K : K + H],
        A_log=randn(H) * 0.5,
        dt_bias=randn(dim) * 0.5,
        recurrent_state=state,
        k_cache=torch.nn.functional.normalize(randn(pool, M, H, K), dim=-1)
        .reshape(pool, M, dim)
        .to(bf16),
        g_cache=LOWER_BOUND * torch.sigmoid(randn(pool, M, dim)),
        v_cache=(v_raw * torch.sigmoid(v_raw)).to(bf16),
        beta_cache=beta_cache,
        ssm_state_indices=slots,
        num_spec=M,
        num_accepted_tokens=accepted,
        lower_bound=LOWER_BOUND,
        scale=K**-0.5,
        onorm_g=qkvg[..., 3 * dim :].view(1, R, H, V),
        onorm_weight=torch.linspace(0.75, 1.25, V, dtype=f32, device=device),
        onorm_eps=ONORM_EPS,
        quantize_output=spec.get("out", "bf16") in ("mxfp8", "mxfp8_exact"),
        use_pdl=spec.get("use_pdl", False),
        early_trigger=spec.get("early_trigger", False),
        mxfp8_pdl_early_trigger=spec.get("mxfp8_pdl_early_trigger", False),
    )
    if spec.get("gate", "g") == "fa":
        w_fb = (randn(dim, K) * 0.05).to(bf16)
        args.update(g=None, fa=fab[..., :K], w_fb=w_fb)
    if spec.get("out") == "mxfp8_exact":  # the catalog quantizer's 1x32 form of the MXFP8 output
        args["mxfp8_exact"] = True
    return args


def fab_args(spec: dict, seed: int, device="cuda") -> dict:
    """Keyword arguments of `rubin_kda_fab`: decode rows with a few outlier channels, the
    `[f_a; b; 0]` and `f_b` weights."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    rows, heads, hidden = spec["rows"], spec["heads"], spec["hidden"]
    n1 = (K + heads + 7) // 8 * 8
    chan = torch.ones(hidden)
    chan[torch.randperm(hidden, generator=gen)[:16]] = 30.0
    w_bfa = torch.randn(n1, hidden, generator=gen) * 0.02
    w_bfa[K + heads :] = 0
    return dict(
        x=(torch.randn(rows, hidden, generator=gen) * chan).to(torch.bfloat16).to(device),
        w_bfa=w_bfa.to(torch.bfloat16).to(device),
        w_fb=(torch.randn(heads * K, K, generator=gen) * 0.05).to(torch.bfloat16).to(device),
        num_heads=heads,
        fa_only=spec["fa_only"],
        multi_tile=spec["multi_tile"],
    )


def clone_mutated(args: dict) -> dict:
    """A copy of `args` whose in-place outputs are same-stride copies."""
    out = dict(args)
    for name in MUTATED:
        t = args[name]
        out[name] = torch.empty_strided(t.shape, t.stride(), dtype=t.dtype, device=t.device)
        out[name].copy_(t)
    return out


def bits(t: torch.Tensor) -> torch.Tensor:
    """Reinterpret as integers, so that equality means identical bits."""
    return t.view({1: torch.uint8, 2: torch.int16, 4: torch.int32}[t.element_size()])


def scale_offsets(rows: int, heads: int, device) -> torch.Tensor:
    """R128c4 byte offset of the first copy of each (row, head) scale, `[rows, heads]`."""
    r = torch.arange(rows, device=device).view(-1, 1)
    h = torch.arange(heads, device=device).view(1, -1)
    return (r // 128) * heads * 512 + h * 512 + (r % 32) * 16 + ((r % 128) // 32) * 4


def scale_bytes(scales: torch.Tensor, rows: int, heads: int) -> torch.Tensor:
    """The four copies of every written scale byte (the padding rows are never written)."""
    off = scale_offsets(rows, heads, scales.device)
    off = (off[..., None] + torch.arange(4, device=scales.device)).reshape(-1)
    return scales.view(torch.uint8)[off]


def dequantize(codes: torch.Tensor, scales: torch.Tensor, heads: int) -> tuple:
    """`(values [T, H, 128], biased exponents [T, H])` of an MXFP8 verify output."""
    rows = codes.shape[0]
    off = scale_offsets(rows, heads, codes.device).reshape(-1)
    exp = scales.view(torch.uint8)[off].view(rows, heads).to(torch.int32)
    values = codes.float().view(rows, heads, K) * torch.pow(2.0, (exp - 127).float()).unsqueeze(-1)
    return values, exp
