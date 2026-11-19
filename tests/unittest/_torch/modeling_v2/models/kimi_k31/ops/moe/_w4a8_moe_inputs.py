# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One rank's W4A8 routed-expert operands (MXFP4 weights, MXFP8 rows) for the tests of the CuTe DSL
entry (k31_w4a8_moe_cute) and the TRTLLM-Gen runner entry (k31_w4a8_moe_runner)."""

import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.moe.w4a8_moe_runner import (
    HIDDEN,
    NUM_EXPERTS,
    TOP_K,
)


def w4a8_operands(rows: int, local: int, inter: int, seed: int) -> dict:
    """Random prepared operands of one rank (any bytes are a valid MXFP4 / UE8M0 operand; the scale
    exponents stay in 2^-9..2^-4 so the outputs are finite) and random distinct top-16 ids over the
    896 experts, keyed as ``k31_w4a8_moe_cute``'s arguments."""
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)

    def nib(*shape):
        return torch.randint(0, 256, shape, generator=g, device=dev, dtype=torch.uint8)

    def sf(*shape):
        return torch.randint(118, 124, shape, generator=g, device=dev, dtype=torch.uint8)

    x = (torch.randn(rows, HIDDEN, generator=g, device=dev) * 2).to(torch.bfloat16)
    xq, xsf = torch.ops.trtllm.mxfp8_quantize(x, False, alignment=32)
    ids = torch.argsort(torch.rand(rows, NUM_EXPERTS, generator=g, device=dev), dim=1)[:, :TOP_K]
    w = torch.rand(rows, TOP_K, generator=g, device=dev) + 0.5
    w = (w / w.sum(1, keepdim=True)).to(torch.bfloat16)
    return dict(
        hidden_states=xq,
        hidden_states_scale=xsf.view(rows, -1),
        topk_ids=ids.to(torch.int32).contiguous(),
        topk_weights=w.contiguous(),
        gemm1_weights=nib(local, 2 * inter, HIDDEN // 2),
        gemm1_weights_scale=sf(local, 2 * inter, HIDDEN // 32),
        gemm2_weights=nib(local, HIDDEN, inter // 2),
        gemm2_weights_scale=sf(local, HIDDEN, inter // 32),
    )


def runner_args(a: dict, tactic) -> tuple:
    """``k31_w4a8_moe_runner``'s arguments on the ``w4a8_operands`` ``a``: SiTU alpha 4 and beta 25,
    the rank owning local experts ``[0, local)``, the given ``tactic`` (None: autotuned)."""
    local, inter = a["gemm1_weights"].shape[0], a["gemm1_weights"].shape[1] // 2
    dev = a["hidden_states"].device
    return (
        a["hidden_states"],
        a["hidden_states_scale"],
        a["topk_ids"],
        a["topk_weights"],
        a["gemm1_weights"],
        a["gemm1_weights_scale"],
        torch.full((local,), 4.0, dtype=torch.float32, device=dev),
        torch.full((local,), 25.0, dtype=torch.float32, device=dev),
        a["gemm2_weights"],
        a["gemm2_weights_scale"],
        inter,
        inter,
        0,
        local,
        tactic,
    )
