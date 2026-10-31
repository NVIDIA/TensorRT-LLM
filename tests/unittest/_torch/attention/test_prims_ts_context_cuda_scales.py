# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Device quantization scales remain live across PrimTS context graph replay."""

import math

import pytest
import torch

from tensorrt_llm._torch.attention.backends.prims_ts.context import BatchPrefillTSWrapper


@pytest.mark.parametrize(
    "qk_fp8,override_output_scale", [(False, True), (True, False), (True, True)]
)
def test_context_cuda_scales_graph_replay(qk_fp8: bool, override_output_scale: bool) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("PrimTS context validation requires SM100 or SM103")
    torch.manual_seed(20261030)
    q_dtype = torch.float8_e4m3fn if qk_fp8 else torch.bfloat16
    q = torch.randn(1, 32, 4, 128, device="cuda").to(q_dtype)
    k = torch.randn(1, 64, 4, 128, device="cuda").to(q_dtype)
    v = torch.randn(1, 64, 4, 128, device="cuda").to(torch.float8_e4m3fn)
    out = torch.empty_like(q, dtype=torch.bfloat16)
    softmax_scale = torch.tensor([math.log2(math.e) / math.sqrt(128)], device="cuda")
    output_scale = torch.ones(1, device="cuda")
    wrapper = BatchPrefillTSWrapper()
    wrapper.plan(
        device=q.device,
        batch_size=1,
        max_q_len=32,
        max_kv_len=64,
        num_qo_heads=4,
        num_kv_heads=4,
        head_dim=128,
        q_dtype=q.dtype,
        k_dtype=k.dtype,
        v_dtype=v.dtype,
        out_dtype=out.dtype,
        cuda_scales=True,
    )

    def run() -> None:
        wrapper.run(
            q,
            k,
            v,
            scale_softmax_log2=softmax_scale if qk_fp8 else None,
            output_scale=output_scale if override_output_scale else None,
            out=out,
            validate=False,
        )

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for qk_scale, v_scale in ((0.5, 2.0), (1.5, 0.25)):
        softmax_scale.fill_(qk_scale * math.log2(math.e) / math.sqrt(128))
        output_scale.fill_(v_scale)
        graph.replay()
        expected = torch.nn.functional.scaled_dot_product_attention(
            q.float().transpose(1, 2),
            k.float().transpose(1, 2),
            v.float().transpose(1, 2),
            scale=(qk_scale if qk_fp8 else 1.0) / math.sqrt(128),
        ).transpose(1, 2) * (v_scale if override_output_scale else 1.0)
        torch.testing.assert_close(out.float(), expected, rtol=0.04, atol=0.04)
