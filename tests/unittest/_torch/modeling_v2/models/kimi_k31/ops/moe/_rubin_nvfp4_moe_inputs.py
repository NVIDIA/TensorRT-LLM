# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stored-form NVFP4 experts, and the layer the engine's NVFP4 CuteDSL MoE method makes of them on
the GPU, for the family's MoE entry tests; the routing and the FC1 -> FC2 finalize chain of the
CuteDSL context entries' tests; and the die-placed window, rows, routing and runner inputs of the
family FC12 runner's tests."""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe import _rubin_moe_common as common

SF_VEC = common.SF_VEC


def experts(num: int, hidden: int, inter: int, seed: int, device="cuda") -> list:
    """Stored-form NVFP4 experts with gate / up / down outputs of order one. ``weight_scale_2``
    differs between experts, so an alpha read for the wrong expert shows."""
    g = torch.Generator(device=device).manual_seed(seed)

    def proj(n, k, scale_2):
        codes = torch.randint(0, 256, (n, k // 2), generator=g, device=device, dtype=torch.uint8)
        steps = torch.randint(120, 132, (n, k // SF_VEC), generator=g, device=device).float()
        scales = (steps / 126.0).to(torch.float8_e4m3fn)
        return codes, scales, scale_2, torch.tensor(1.0, device=device)

    out = []
    for _ in range(num):
        s13, s2 = 0.02 + 0.02 * torch.rand(2, generator=g, device=device)
        gate_up = dict(w1=proj(inter, hidden, s13), w3=proj(inter, hidden, s13))
        out.append(dict(gate_up, w2=proj(hidden, inter, s2)))
    return out


def _sm_version() -> int:
    """The CUDA device's SM version; SM107's on a host without CUDA."""
    if not torch.cuda.is_available():
        return 107
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def skip_tiles_the_gpu_lacks(tile: int) -> None:
    """Skips a CuteDSL context cell whose tile the GPU's kernels do not take."""
    if _sm_version() != 107 and tile > 256:
        pytest.skip("SM103's Blackwell kernels take tiles 128 and 256")


#: The K3.1 SiTU betas the CuteDSL FC1 epilogue takes.
SITU = dict(situ_beta=4.0, situ_linear_beta=25.0)


def route(rows, num_experts, top_k, seed, device="cpu"):
    """``top_k`` distinct experts per row and FP32 weights."""
    g = torch.Generator(device=device).manual_seed(seed)
    logits = torch.randn(rows, num_experts, generator=g, device=device)
    ids = torch.topk(logits, top_k, dim=-1).indices.to(torch.int32)
    weights = torch.rand(rows, top_k, generator=g, device=device)
    return ids, weights / weights.sum(-1, keepdim=True)


def fc1_fc2(x4, sf, layer, out, sort, weights, total, top_k, local, offset, tile):
    """rubin_cutedsl_fc1, then rubin_cutedsl_fc2_finalize into ``out``, over one rank's window of
    ``local`` experts at ``offset`` on the ``rubin_moe_sort`` tables ``sort``."""
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe.rubin_cutedsl_fc1 import (
        rubin_cutedsl_fc1,
    )
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe.rubin_cutedsl_fc2_finalize import (  # noqa: E501
        rubin_cutedsl_fc2_finalize,
    )
    from tensorrt_llm._torch.utils import ActivationType

    t2e, t2l, _, p2e, _, n = sort
    h, h_sf = rubin_cutedsl_fc1(
        x4,
        layer["fc1_w"],
        sf,
        layer["fc1_s"],
        layer["fc1_alpha"],
        t2e,
        t2l,
        p2e,
        n,
        layer["g2"],
        total,
        top_k,
        local,
        offset,
        tile,
        int(ActivationType.SiTu),
        **SITU,
    )
    rubin_cutedsl_fc2_finalize(
        h,
        layer["fc2_w"],
        h_sf,
        layer["fc2_s"],
        layer["fc2_alpha"],
        out,
        t2e,
        t2l,
        p2e,
        n,
        weights,
        total,
        top_k,
        local,
        offset,
        tile,
    )


def fc1_rows(gate: torch.Tensor, up: torch.Tensor, sm_version=None) -> torch.Tensor:
    """FC1 rows ``[..., 2 * I, cols]`` as the GPU's CuteDSL FC1 kernel reads them: on SM107 groups of
    16 rows of the gate (w1), each followed by the up (w3) rows of the same 16 outputs; on SM103
    groups of 64 up rows, each followed by the gate rows of the same 64 outputs."""
    if (sm_version or _sm_version()) == 107:
        first, second, group = gate, up, 16
    else:
        first, second, group = up, gate, 64
    *lead, inter, cols = gate.shape
    groups = (*lead, inter // group, group, cols)
    pairs = torch.stack((first.reshape(groups), second.reshape(groups)), dim=-3)
    return pairs.reshape(*lead, 2 * inter, cols)


def layer(experts: list, hidden: int) -> dict:
    """The layer as the engine's NVFP4CuteDslFusedMoEMethod leaves it after loading on this GPU: FC1
    rows in the GPU's kernel order (``fc1_rows``), block scales 128x4 swizzled, global scales
    ``1 / max(input_scale)`` over the rank's experts and per-expert alphas in that method's
    arithmetic."""
    from tensorrt_llm._torch.utils import swizzle_sf

    local, inter = len(experts), experts[0]["w1"][0].shape[0]

    def stack(w, i):
        return torch.stack([e[w][i] for e in experts])

    g1 = stack("w1", 3).float().max().reciprocal()
    g2 = stack("w2", 3).float().max().reciprocal()
    fc1_s = swizzle_sf(
        fc1_rows(stack("w1", 1).view(torch.uint8), stack("w3", 1).view(torch.uint8)),
        2 * inter,
        hidden,
    )
    fc2_s = swizzle_sf(stack("w2", 1).view(torch.uint8), hidden, inter)
    return dict(
        fc1_w=fc1_rows(stack("w1", 0), stack("w3", 0)).view(torch.float4_e2m1fn_x2),
        fc1_s=fc1_s.view(local, 2 * inter, hidden // SF_VEC),
        fc2_w=stack("w2", 0).view(torch.float4_e2m1fn_x2),
        fc2_s=fc2_s.view(local, hidden, inter // SF_VEC),
        g1=g1,
        g2=g2,
        fc1_alpha=1.0 / (g1 * (1.0 / stack("w1", 2).float())),
        fc2_alpha=1.0 / (g2 * (1.0 / stack("w2", 2).float())),
    )


# ---------------------------------------------------------------------------------------------
# The family FC12 runner (kernels/moe/rubin_fc12_launch.py)
# ---------------------------------------------------------------------------------------------


def two_dies() -> bool:
    """Whether this GPU is an SM107 with at least two dies (memory locality domains)."""
    if not (torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)):
        return False
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.moe import (
        rubin_die_locality as die_locality,
    )

    return die_locality.locality_domain_count(torch.cuda.current_device()) >= 2


def placed_window(local: int, inter: int, hidden: int, seed: int) -> dict:
    """NVFP4 FC12 operands of one rank's window of ``local`` experts, the weights placed in
    die-alternating memory: FC1 codes ``w1``, block scales ``s1`` and alphas ``a1``; FC2 ``w2``,
    ``s2``, ``a2``."""
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.moe.rubin_die_locality import (
        die_alternating_copy,
        is_die_alternating,
    )

    gen = torch.Generator(device="cuda").manual_seed(seed)

    def nibbles(*shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8, device="cuda", generator=gen)

    def block_scales(*shape):
        values = torch.rand(shape, device="cuda", generator=gen) * 1.5 + 0.5
        return values.to(torch.float8_e4m3fn).view(torch.uint8)

    w = dict(
        w1=nibbles(local, 2 * inter, hidden // 2),
        s1=block_scales(local, 2 * inter, hidden // 16),
        a1=torch.full((local,), 2.0e-6, device="cuda"),
        w2=nibbles(local, hidden, inter // 2),
        s2=block_scales(local, hidden, inter // 16),
        a2=torch.full((local,), 2e-5, device="cuda"),
    )
    for key in ("w1", "w2"):
        w[key] = die_alternating_copy(w[key])
        assert w[key] is not None and is_die_alternating(w[key])
    return w


def nvfp4_rows(m: int, hidden: int, seed: int) -> tuple:
    """``m`` random BF16 rows quantized to NVFP4: the codes and their ``[m, hidden / 16]`` block
    scales."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((m, hidden), device="cuda", generator=gen).to(torch.bfloat16)
    x4, sf = torch.ops.trtllm.fp4_quantize(
        x, ((448 * 6) / x.abs().max().float()).reshape(1), 16, False, False
    )
    return x4, sf.view(m, hidden // 16)


def request_routing(
    m: int,
    per_request: int,
    pool: int,
    seed: int,
    offset: int,
    local: int,
    tile: int,
    num_experts: int = 896,
    top_k: int = 16,
) -> tuple:
    """Routing of ``m`` rows in requests of ``per_request`` tokens, a request's experts drawn from a
    pool of ``pool``: the routing scales, the window's ``token_ordered_sort`` tables (window at
    ``offset``, ``local`` experts, ``tile`` rows) and each row's expert count in the window."""
    gen = torch.Generator().manual_seed(seed)
    rows = []
    for _ in range(m // per_request):
        request_pool = torch.randperm(num_experts, generator=gen)[:pool]
        for _ in range(per_request):
            rows.append(request_pool[torch.randperm(pool, generator=gen)[:top_k]])
    ids = torch.stack(rows).to(torch.int32).cuda()
    scales = (torch.rand((m, top_k), generator=gen) + 0.1).cuda()
    tables = common.token_ordered_sort(ids, offset, local, tile)
    in_window = ((ids >= offset) & (ids < offset + local)).sum(dim=1)
    return scales, tables, in_window


def fc12_inputs(x4, sf, w: dict, tables: tuple, scales, out) -> list:
    """The FC12 runner's 16 inputs: ``placed_window`` operands ``w``, ``nvfp4_rows`` ``x4`` /
    ``sf``, ``request_routing``'s ``tables`` and ``scales``, and the output ``out``."""
    t2e, t2l, e2p, p2e, _, num_tiles = tables
    return [
        x4.view(torch.float4_e2m1fn_x2),
        w["w1"].view(torch.float4_e2m1fn_x2),
        sf.view(torch.uint8),
        w["s1"],
        w["a1"],
        t2e,
        t2l,
        p2e,
        num_tiles,
        torch.tensor([4.0], device="cuda"),
        w["w2"].view(torch.float4_e2m1fn_x2),
        w["s2"],
        w["a2"],
        out,
        scales,
        e2p,
    ]


def runner_class():
    """``RubinFc12Runner``; skips the test where it is not defined (no CuTe DSL with Rubin
    support)."""
    from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kernels.moe import rubin_fc12_launch

    cls = getattr(rubin_fc12_launch, "RubinFc12Runner", None)
    if cls is None:
        pytest.skip("no CuTe DSL with Rubin support here: the runner is not defined")
    return cls


def bare_runner(**levers):
    """A ``RubinFc12Runner`` of 224 local experts (of 896, top-k 16, tile 128, SiTU 4 / 25) built
    without its constructor, which checks the GPU's SM version; ``levers`` are ``mma_n``,
    ``die_claim`` and ``l2_prefetch``."""
    runner = object.__new__(runner_class())
    runner.__dict__.update(
        num_experts=896,
        top_k=16,
        num_local_experts=224,
        local_expert_offset=0,
        tile_size=128,
        scaling_vector_size=16,
        situ_beta=4.0,
        situ_linear_beta=25.0,
        **levers,
    )
    return runner
