# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm as tllm
from tensorrt_llm._torch.moe.fused_moe.communication.deep_ep_low_latency import DeepEPLowLatency
from tensorrt_llm._torch.moe.fused_moe.deep_ep_utils import deep_ep_installed
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.mapping import Mapping

_SM_VERSION = get_sm_version() if torch.cuda.is_available() else 0

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    pytest.mark.skipif(not deep_ep_installed, reason="requires DeepEP"),
    pytest.mark.skipif(_SM_VERSION not in (100, 103), reason="requires SM100/SM103"),
]


def _make_bf16_input(case: str, num_tokens: int, hidden_size: int) -> torch.Tensor:
    """Create deterministic BF16 input for an NVFP4 packing test.

    Args:
        case: Input pattern name.
        num_tokens: Number of input tokens.
        hidden_size: Hidden width of each token.

    Returns:
        A CUDA BF16 tensor with shape ``[num_tokens, hidden_size]``.
    """
    if case == "zeros":
        return torch.zeros(num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda")
    if case == "finite_extremes":
        finfo = torch.finfo(torch.bfloat16)
        pattern = torch.tensor(
            [finfo.min, -448.0, -6.0, -0.5, 0.0, 0.5, 6.0, 448.0, finfo.max],
            dtype=torch.bfloat16,
            device="cuda",
        )
        num_elements = num_tokens * hidden_size
        repeat_count = (num_elements + pattern.numel() - 1) // pattern.numel()
        return pattern.repeat(repeat_count)[:num_elements].view(num_tokens, hidden_size)

    generator = torch.Generator(device="cuda").manual_seed(20260722 + num_tokens)
    return (
        torch.randn(
            num_tokens, hidden_size, dtype=torch.bfloat16, device="cuda", generator=generator
        )
        * 32
    )


@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("case", ["zeros", "finite_extremes", "random"])
def test_deep_ep_nvfp4_pack_matches_trtllm_bitwise(num_tokens: int, case: str) -> None:
    """Verify that DeepEP packing is bit-identical to TensorRT-LLM.

    Args:
        num_tokens: Number of BF16 tokens to quantize.
        case: Input pattern name.
    """
    from tensorrt_llm.deep_ep.buffer import Buffer

    hidden_size = 4096
    bf16_input = _make_bf16_input(case, num_tokens, hidden_size)
    static_scale = torch.tensor(0.75, dtype=torch.float32, device="cuda")

    reference_values, reference_scales = torch.ops.trtllm.fp4_quantize(
        bf16_input, static_scale, 16, False, False
    )
    deep_ep_values, deep_ep_scales = Buffer.quantize_bf16_to_nvfp4(
        bf16_input,
        static_scale.expand(num_tokens, 1).contiguous(),
    )

    assert torch.equal(
        reference_values.view(torch.uint8).reshape(-1),
        deep_ep_values.view(torch.uint8).reshape(-1),
    )
    assert torch.equal(
        reference_scales.view(torch.uint8).reshape(-1),
        deep_ep_scales.view(torch.uint8).reshape(-1),
    )


def _run_deep_ep_fused_transport_paths(_: None) -> int:
    """Exercise fused dispatch, dynamic output scale, and staged combine."""
    from tensorrt_llm.deep_ep.buffer import Buffer

    rank = tllm.mpi_rank()
    world_size = tllm.mpi_world_size()
    assert world_size == 2
    torch.cuda.set_device(rank)
    comm = None
    try:
        num_tokens = 8
        max_num_tokens = 16
        hidden_size = 4096
        num_experts = 32
        experts_per_rank = num_experts // world_size
        mapping = Mapping(
            rank=rank,
            tp_size=world_size,
            moe_ep_size=world_size,
            world_size=world_size,
        )
        quant_config = SimpleNamespace(
            layer_quant_mode=SimpleNamespace(
                has_nvfp4=lambda: True,
                has_fp8_qdq=lambda: False,
            ),
            quant_mode=SimpleNamespace(is_int4_weight_only_per_group=lambda: False),
        )
        comm = DeepEPLowLatency(
            mapping=mapping,
            num_slots=num_experts,
            hidden_size=hidden_size,
            weight_dtype=torch.bfloat16,
            quant_config=quant_config,
            expert_size_per_partition=experts_per_rank,
            max_num_tokens=max_num_tokens,
            use_low_precision_combine=True,
            moe_max_num_tokens=max_num_tokens,
        )

        generator = torch.Generator(device="cuda").manual_seed(20260904)
        bf16_input = torch.randn(
            num_tokens,
            hidden_size,
            dtype=torch.bfloat16,
            device="cuda",
            generator=generator,
        )
        global_scale = torch.tensor([0.75], dtype=torch.float32, device="cuda")
        reference_values, reference_scales = Buffer.quantize_bf16_to_nvfp4(
            bf16_input,
            global_scale.expand(num_tokens, 1).contiguous(),
        )
        destination_rank = (rank + 1) % world_size
        topk_idx = torch.full(
            (num_tokens, 1),
            destination_rank * experts_per_rank,
            dtype=torch.int32,
            device="cuda",
        )

        recv_values, recv_scales, recv_count, handle = (
            comm.deep_ep_buffer.low_latency_dispatch_bf16_to_fp4(
                bf16_input,
                global_scale,
                topk_idx,
                max_num_tokens,
                num_experts,
            )
        )
        torch.cuda.synchronize()

        assert recv_count[0].item() == num_tokens
        assert recv_count.sum().item() == num_tokens
        assert torch.equal(recv_values[0, :num_tokens], reference_values)
        assert torch.equal(recv_scales[0, :num_tokens], reference_scales)

        expert_output = torch.zeros(
            experts_per_rank,
            world_size * max_num_tokens,
            hidden_size,
            dtype=torch.bfloat16,
            device="cuda",
        )
        topk_weights = torch.ones(num_tokens, 1, dtype=torch.float32, device="cuda")
        combined = comm.deep_ep_buffer.low_latency_combine_low_precision(
            "nvfp4",
            expert_output,
            None,
            topk_idx,
            topk_weights,
            handle,
            stage_recv_metadata=True,
        )
        torch.cuda.synchronize()

        assert torch.isfinite(combined).all().item()
        assert torch.count_nonzero(combined).item() == 0
        return rank
    finally:
        if comm is not None:
            comm.destroy()


@pytest.mark.threadleak(enabled=False)
@pytest.mark.parametrize("mpi_pool_executor", [2], indirect=True)
def test_deep_ep_fused_transport_paths_match_numeric_references(mpi_pool_executor) -> None:
    ranks = sorted(mpi_pool_executor.map(_run_deep_ep_fused_transport_paths, [None, None]))
    assert ranks == [0, 1]
