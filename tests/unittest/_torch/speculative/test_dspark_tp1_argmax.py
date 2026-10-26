# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Full-vocabulary BF16 Markov feedback with no distributed collective."""

from __future__ import annotations

import math
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.models import modeling_dspark
from tensorrt_llm._torch.models.modeling_dspark import CompactTPArgmax
from tensorrt_llm._torch.models.modeling_speculative import VanillaMarkov, markov_prev_embeddings
from tensorrt_llm._torch.modules.embedding import LMHead
from tensorrt_llm._torch.modules.linear import TensorParallelMode

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _capture(call):
    # Initialize GEMM, CuTe kernels and derived weights before graph capture.
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(graph, stream=stream):
        output = call()
    torch.cuda.synchronize()
    return graph, output


@pytest.mark.parametrize(
    "batch,vocab,column",
    [(1, 129280, False), (32, 129280, False), (2, 8, False), (1, 129280, True)],
)
def test_full_vocab_five_step_chain_and_changed_input_graph(monkeypatch, batch, vocab, column):
    # The chain on the BF16 head, without its MXFP8 copy.
    monkeypatch.setattr(CompactTPArgmax, "_fp8_head_logits", lambda *args: None)
    device = torch.device("cuda", torch.cuda.current_device())
    block, hidden, markov_rank = 5, 5120, 256
    with torch.inference_mode():
        head = LMHead(
            vocab,
            hidden,
            dtype=torch.bfloat16,
            tensor_parallel_mode=TensorParallelMode.COLUMN if column else None,
            gather_output=column,
        ).to(device)
        markov = VanillaMarkov(vocab_size=vocab, markov_rank=markov_rank).to(device)
        local_generator = torch.Generator(device=device).manual_seed(
            14236 + torch.cuda.current_device()
        )
        head.weight.copy_(
            torch.randn(
                head.weight.shape, device=device, dtype=torch.bfloat16, generator=local_generator
            )
            / math.sqrt(hidden)
        )
        # Replicate checkpoint BF16 values in the original FP32 parameter storage.
        shared_generator = torch.Generator(device=device).manual_seed(35719)
        for parameter in (markov.markov_w1.weight, markov.markov_w2.weight):
            parameter.copy_(
                torch.randn(
                    parameter.shape, device=device, dtype=torch.bfloat16, generator=shared_generator
                )
                / 16
            )
        normalized = torch.randn(
            (batch, block, hidden), device=device, dtype=torch.bfloat16, generator=shared_generator
        )
        bonus = torch.arange(batch, device=device, dtype=torch.int64) % vocab
        state = CompactTPArgmax()

        def compact():
            result = state.try_propose(
                head,
                markov,
                normalized,
                bonus,
                block_size=block,
                temperature=0.0,
                confidence_threshold=0.0,
                return_logits=False,
            )
            assert result is not None
            return result

        def full_vocab_bf16_reference():
            base = head.apply_linear(normalized, head.bias)
            previous, sampled = bonus, []
            for step in range(block):
                bias = F.linear(markov_prev_embeddings(previous, state._w1), state._w2)
                corrected = base[:, step].float() + bias.float()
                previous = corrected[:, :vocab].argmax(dim=-1)
                sampled.append(previous)
            return torch.stack(sampled, dim=1)

        def forbidden_collective(*args, **kwargs):
            raise AssertionError("The full-vocabulary TP1 path must not call a collective")

        # Guard eager execution and capture, including every dependent Markov
        # step. LMHead.apply_linear does no communication.
        with patch.object(modeling_dspark, "allgather", forbidden_collective):
            actual, counts = compact()
            assert torch.equal(actual, full_vocab_bf16_reference())
            assert torch.equal(counts, torch.full_like(counts, block))
            compact_graph, compact_output = _capture(compact)
        for iteration in range(4):
            normalized.copy_(
                torch.randn(
                    normalized.shape,
                    device=device,
                    dtype=torch.bfloat16,
                    generator=shared_generator,
                )
            )
            # Exercise the original invalid-anchor-to-zero rule without host reads in capture.
            bonus.copy_((torch.arange(batch, device=device) + iteration) % vocab)
            if iteration == 1:
                bonus[0] = -1
            elif iteration == 2:
                bonus[0] = vocab
            compact_graph.replay()
            assert torch.equal(compact_output[0], full_vocab_bf16_reference())


def test_fp8_head_feeds_the_chain_its_mxfp8_logits():
    """The base logits come from an MXFP8 copy of the head (E4M3 values, one power-of-two scale
    per 32 inputs) through FlashInfer's MXFP8 GEMM on MXFP8 activations, and the copy stays within
    E4M3 rounding of the BF16 weights."""
    import flashinfer

    from tensorrt_llm._torch.custom_ops.torch_custom_ops import (
        IS_FLASHINFER_MXFP8_CUTE_DSL_AVAILABLE,
    )

    if not IS_FLASHINFER_MXFP8_CUTE_DSL_AVAILABLE or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("The MXFP8 draft head needs FlashInfer's CuTe DSL MXFP8 GEMM on SM100/SM103")
    device = torch.device("cuda", torch.cuda.current_device())
    vocab, block, hidden, markov_rank, batch = 129280, 5, 5120, 256, 4
    with torch.inference_mode():
        head = LMHead(vocab, hidden, dtype=torch.bfloat16).to(device)
        markov = VanillaMarkov(vocab_size=vocab, markov_rank=markov_rank).to(device)
        generator = torch.Generator(device=device).manual_seed(4711)
        head.weight.copy_(
            torch.randn(head.weight.shape, device=device, dtype=torch.bfloat16, generator=generator)
            / math.sqrt(hidden)
        )
        for parameter in (markov.markov_w1.weight, markov.markov_w2.weight):
            parameter.copy_(
                torch.randn(
                    parameter.shape, device=device, dtype=torch.bfloat16, generator=generator
                )
                / 16
            )
        normalized = torch.randn(
            (batch, block, hidden), device=device, dtype=torch.bfloat16, generator=generator
        )
        bonus = torch.arange(batch, device=device, dtype=torch.int64)
        state = CompactTPArgmax()

        def compact():
            result = state.try_propose(
                head,
                markov,
                normalized,
                bonus,
                block_size=block,
                temperature=0.0,
                confidence_threshold=0.0,
                return_logits=False,
            )
            assert result is not None
            return result[0]

        def fp8_reference():
            _, values, swizzled = state._head_fp8
            act, act_scale = torch.ops.trtllm.mxfp8_quantize(normalized.reshape(-1, hidden), True)
            base = flashinfer.mm_mxfp8(
                act,
                values.t(),
                act_scale,
                swizzled,
                out_dtype=torch.bfloat16,
                use_8x4_sf_layout=False,
                backend="cute-dsl",
            )
            base = base.view(batch, block, -1)
            previous, sampled = bonus, []
            for step in range(block):
                bias = F.linear(markov_prev_embeddings(previous, state._w1), state._w2)
                previous = (base[:, step].float() + bias.float())[:, :vocab].argmax(dim=-1)
                sampled.append(previous)
            return torch.stack(sampled, dim=1)

        assert torch.equal(compact(), fp8_reference())
        _, values, _ = state._head_fp8
        # E4M3 rounding: 2^-4 relative for normal values, half a subnormal step of the block scale.
        for rows in (slice(0, 2048), slice(vocab - 2048, vocab)):
            _, scales = modeling_dspark._quantize_mxfp8_rows(head.weight[rows])
            weight = head.weight[rows].float().view(2048, hidden // 32, 32)
            scale = torch.exp2(scales.float() - 127)[..., None]
            dequantized = values[rows].float().view(2048, hidden // 32, 32) * scale
            bound = torch.maximum(weight.abs() * 2.0**-4, scale * 2.0**-10)
            assert bool(((dequantized - weight).abs() <= bound).all())
