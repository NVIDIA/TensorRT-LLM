# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from tensorrt_llm._torch.visual_gen.catalog.attention.cute_fmha import cute_fmha


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3), (10, 7)),
    reason="CuTe FMHA needs SM100, SM103 or SM107",
)
@pytest.mark.parametrize("cell", cute_fmha.CELLS, ids=lambda cell: cell.why)
def test_cute_fmha(cell):
    spec = cell.spec
    rng = torch.Generator(device="cuda").manual_seed(371)

    def tensor(seq, heads):
        return (
            torch.randn(1, seq, heads, 128, device="cuda", dtype=torch.bfloat16, generator=rng)
            * 0.5
        )

    q, k, v = tensor(96, 4), tensor(160, 2), tensor(160, 2)
    kwargs = dict(
        sm_scale=0.078125,
        qk_dtype=spec["qk_dtype"],
        is_causal=spec.get("is_causal", False),
        v_block_size=spec.get("v_block_size"),
    )
    if spec.get("prequantized"):
        qs, ks, vs = [torch.tensor(scale, device="cuda") for scale in (0.0125, 0.02, 0.01)]
        q, k, v = [
            (x.float() / scale).to(torch.float8_e4m3fn) for x, scale in ((q, qs), (k, ks), (v, vs))
        ]
        kwargs.update(static_q_scale=qs, static_k_scale=ks, static_v_scale=vs)
    cute_fmha.is_valid(q, k, v, **kwargs)
    out, lse = cute_fmha(q, k, v, **kwargs)
    expected_out, expected_lse = cute_fmha.reference(q, k, v, **kwargs)
    # Blockscaled Q/K and FP8 softmax probabilities round before the output reduction.
    atol = {None: 0.005, "fp8": 0.015, "nvfp4": 0.035}[spec["qk_dtype"]]
    torch.testing.assert_close(out, expected_out, rtol=0.01, atol=atol)
    torch.testing.assert_close(lse, expected_lse, rtol=0.001, atol=atol)


@pytest.mark.cpu_only
def test_prequantized_dequant_scale_is_scalar():
    q = torch.empty(1, 1, 1, 128, dtype=torch.float8_e4m3fn)
    with pytest.raises(AssertionError, match="one FP32 dequant scale"):
        cute_fmha.is_valid(q, q, q, sm_scale=0.125, qk_dtype="fp8", static_q_scale=torch.ones(2))
