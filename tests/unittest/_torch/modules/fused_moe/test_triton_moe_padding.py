# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for Triton MoE token-dim bucketing (``MoeConfig.triton_pad_token_dim``).

Run as a pytest:
    pytest tests/unittest/_torch/modules/fused_moe/test_triton_moe_padding.py -v
"""

from __future__ import annotations

import pytest
import torch
from utils.util import skip_no_hopper

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe.fused_moe_triton import (
    TritonFusedMoE,
    iter_triton_moe_m_buckets,
    round_up_to_triton_moe_m_bucket,
)
from tensorrt_llm._torch.modules.fused_moe.routing import RenormalizeMoeRoutingMethod
from tensorrt_llm._utils import mpi_rank
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

MAX_TOKENS = 65536


def test_bucket_round_up_properties():
    buckets = set(iter_triton_moe_m_buckets(MAX_TOKENS))
    prev_bucket = 0
    for num_tokens in range(1, MAX_TOKENS + 1):
        bucket = round_up_to_triton_moe_m_bucket(num_tokens)
        # Rounds up, by at most 12.5%.
        assert num_tokens <= bucket <= num_tokens + max(num_tokens // 8, 1)
        # Buckets are fixed points (padding a padded size is a no-op).
        assert round_up_to_triton_moe_m_bucket(bucket) == bucket
        # Monotonic in num_tokens.
        assert bucket >= prev_bucket
        prev_bucket = bucket
        # iter_triton_moe_m_buckets enumerates exactly the reachable buckets.
        assert bucket in buckets


def test_bucket_iter_is_sorted_and_compact():
    buckets = iter_triton_moe_m_buckets(MAX_TOKENS)
    assert buckets == sorted(set(buckets))
    assert buckets[0] == 1
    assert buckets[-1] >= MAX_TOKENS // 8 * 8
    # 8 buckets per power-of-two octave => O(8 * log2(n)) buckets in total.
    assert len(buckets) <= 8 * (MAX_TOKENS.bit_length() + 1)


@skip_no_hopper
@pytest.mark.parametrize("num_tokens", [5, 17, 100, 1000])
@pytest.mark.parametrize("bias", [True, False])
def test_triton_moe_padding_bit_identical(num_tokens, bias):
    """Padded and unpadded forwards must produce bit-identical outputs on the
    W4A16 MXFP4 path."""
    assert round_up_to_triton_moe_m_bucket(num_tokens) != num_tokens or num_tokens == 5, (
        "test sizes should exercise actual padding"
    )

    mapping = Mapping()
    mapping.rank = mpi_rank()

    with torch.device(f"cuda:{mapping.rank}"):
        dtype = torch.bfloat16
        HIDDEN_SIZE = 2880
        INTERMEDIATE_SIZE = 720
        NUM_EXPERTS = 32
        TOP_K = 4
        routing_method = RenormalizeMoeRoutingMethod(top_k=TOP_K)
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        x = torch.randn((num_tokens, HIDDEN_SIZE), dtype=dtype).cuda()
        router_logits = torch.randn((num_tokens, NUM_EXPERTS), dtype=dtype).cuda()

        weights = {}
        for expert_id in range(NUM_EXPERTS):
            weights[f"{expert_id}.w1.weight"] = torch.randn(
                (INTERMEDIATE_SIZE, HIDDEN_SIZE), dtype=dtype
            ).cuda()
            weights[f"{expert_id}.w2.weight"] = torch.randn(
                (HIDDEN_SIZE, INTERMEDIATE_SIZE), dtype=dtype
            ).cuda()
            weights[f"{expert_id}.w3.weight"] = torch.randn(
                (INTERMEDIATE_SIZE, HIDDEN_SIZE), dtype=dtype
            ).cuda()
            if bias:
                weights[f"{expert_id}.w1.bias"] = torch.randn(
                    (INTERMEDIATE_SIZE,), dtype=dtype
                ).cuda()
                weights[f"{expert_id}.w2.bias"] = torch.randn((HIDDEN_SIZE,), dtype=dtype).cuda()
                weights[f"{expert_id}.w3.bias"] = torch.randn(
                    (INTERMEDIATE_SIZE,), dtype=dtype
                ).cuda()

        quant_config = QuantConfig(quant_algo=QuantAlgo.W4A16_MXFP4)
        fused_moe = TritonFusedMoE(
            num_experts=NUM_EXPERTS,
            routing_method=routing_method,
            hidden_size=HIDDEN_SIZE,
            intermediate_size=INTERMEDIATE_SIZE,
            dtype=dtype,
            reduce_results=True,
            bias=bias,
            model_config=ModelConfig(
                quant_config=quant_config, moe_triton_pad_token_dim=True, mapping=mapping
            ),
        )
        fused_moe.load_weights([weights])
        fused_moe.cuda()
        assert fused_moe.pad_token_dim

        with torch.inference_mode():
            output_padded = fused_moe.forward(x, router_logits)
            # The flag is captured at init and only consumed in forward_impl,
            # so flipping it on the module exercises the unpadded path with
            # identical weights.
            fused_moe.pad_token_dim = False
            output_unpadded = fused_moe.forward(x, router_logits)
        torch.cuda.synchronize()

        assert output_padded.shape == output_unpadded.shape == x.shape
        assert torch.equal(output_padded, output_unpadded)


@skip_no_hopper
def test_triton_moe_padding_requires_w4a16_mxfp4():
    """The padding flag must be ignored (with a warning) on quantization paths
    where bit-identity is not guaranteed."""
    mapping = Mapping()
    mapping.rank = mpi_rank()
    with torch.device(f"cuda:{mapping.rank}"):
        fused_moe = TritonFusedMoE(
            num_experts=8,
            routing_method=RenormalizeMoeRoutingMethod(top_k=2),
            hidden_size=2880,
            intermediate_size=720,
            dtype=torch.bfloat16,
            reduce_results=True,
            model_config=ModelConfig(
                quant_config=QuantConfig(quant_algo=QuantAlgo.FP8),
                moe_triton_pad_token_dim=True,
                mapping=mapping,
                skip_create_weights_in_init=True,
            ),
        )
        assert not fused_moe.pad_token_dim
