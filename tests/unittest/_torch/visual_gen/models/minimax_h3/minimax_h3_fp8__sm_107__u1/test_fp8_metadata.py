# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU layer identifiers retain exact results while compiled blocks share a graph."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


@pytest.fixture(scope="module")
def metadata_ops():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("The FP8 scenario requires an SM107 GPU")
    if torch.cuda.get_device_properties(0).multi_processor_count != 212:
        pytest.skip("The FP8 scenario requires the qualified 212-SM GPU")
    from tensorrt_llm._torch.visual_gen.models.minimax_h3.minimax_h3_fp8__sm_107__u1.config import (
        select,
    )

    select()
    from tensorrt_llm._torch.visual_gen.models.minimax_h3.minimax_h3_fp8__sm_107__u1 import (
        conditioning,
        modeling,
    )
    from tensorrt_llm._torch.visual_gen.models.minimax_h3.minimax_h3_nvfp4__sm_107__u1_u4.kernels import (
        fused_fp8_qk,
    )

    return conditioning, modeling, fused_fp8_qk


def test_table_identifiers_share_a_graph_and_keep_distinct_results(metadata_ops):
    conditioning, _, _ = metadata_ops

    class TimeProjection(torch.nn.Module):
        num_channels = 4

        def forward(self, values):
            return values[:, None].expand(-1, 4).float().clone()

    class Modulation(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(4, 3 * 6 * 5376, device="cuda", dtype=torch.bfloat16)
            self.linear.dtype = torch.bfloat16

        def table(self, temb):
            return self.linear(F.silu(temb).to(self.linear.dtype)).reshape(-1, 6 * 5376)

    blocks = [SimpleNamespace(adaln_proj=Modulation().eval()) for _ in range(2)]
    transformer = SimpleNamespace(time_proj=TimeProjection(), transformer_blocks=blocks)
    cache = conditioning.ConditioningCache(transformer)
    table = torch.compile(lambda module, temb: module.table(temb), fullgraph=True, dynamic=True)
    with torch.inference_mode():
        temb = transformer.time_proj(torch.tensor([0.5], device="cuda"))
        expected = [native(temb) for native in cache.tables]
        first = table(blocks[0].adaln_proj, temb)
        graphs = int(torch._dynamo.utils.counters.get("stats", {}).get("unique_graphs", 0))
        second = table(blocks[1].adaln_proj, temb)
        assert graphs == int(torch._dynamo.utils.counters.get("stats", {}).get("unique_graphs", 0))
        assert torch.equal(first.view(torch.uint8), expected[0].view(torch.uint8))
        assert torch.equal(second.view(torch.uint8), expected[1].view(torch.uint8))
        assert not torch.equal(first.view(torch.uint8), second.view(torch.uint8))
        for block in blocks:
            assert block.adaln_proj._h3_fp8_block_id.device.type == "cpu"
            assert block.adaln_proj._h3_fp8_block_id.dtype == torch.int64


@pytest.mark.parametrize("index", [0, 17, 49])
def test_qk_identifier_matches_shared_producer(index, metadata_ops):
    _, modeling, shared = metadata_ops
    generator = torch.Generator(device="cuda").manual_seed(180 + index)
    qkv = torch.randn((1, 257, 3 * 56 * 128), device="cuda", generator=generator).to(torch.bfloat16)
    weight_q = torch.randn((128,), device="cuda", generator=generator).to(torch.bfloat16)
    weight_k = torch.randn((128,), device="cuda", generator=generator).to(torch.bfloat16)
    half = torch.randn((257, 32), device="cuda", generator=generator)
    cos, sin = (
        torch.cat((half.cos(), half.cos()), dim=-1),
        torch.cat((half.sin(), half.sin()), dim=-1),
    )
    descale = shared.qk_static_descale(128, qkv.device)
    with torch.inference_mode():
        actual = modeling.qk_norm_rope(
            qkv,
            weight_q,
            weight_k,
            cos,
            sin,
            descale,
            1e-6,
            56,
            128,
            torch.tensor(index, device="cpu", dtype=torch.int64),
            None,
        )
        outputs, scales = shared._qk_norm_rope_fp8_impl(
            qkv,
            weight_q,
            weight_k,
            cos,
            sin,
            descale,
            1e-6,
            56,
            128,
            index,
            None,
            0,
            rope_halves_equal=True,
        )
        for value, reference in zip(actual, (*outputs, scales)):
            assert torch.equal(value.view(torch.uint8), reference.view(torch.uint8))
