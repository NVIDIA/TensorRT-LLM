# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FlashInfer's linear FP4 activation scales retain the token dimension."""

import inspect
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.moe.fused_moe.moe_op_backend import FlashinferOpBackend


@pytest.mark.parametrize("flag", [None, "0", "1"])
@pytest.mark.parametrize("enable_pdl", [None, False, True])
def test_fp4_quantize_provider_options_are_keyword_arguments(enable_pdl, flag, monkeypatch):
    if flag is None:
        monkeypatch.delenv("TRTLLM_FLASHINFER_FP4_CONTRACT_FIX", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_FLASHINFER_FP4_CONTRACT_FIX", flag)
    backend = FlashinferOpBackend()
    input = torch.zeros(2, 128)
    scale = torch.ones(1)
    requested_pdl = enable_pdl

    def quantize(
        tensor,
        global_scale,
        sf_vec_size,
        sf_use_ue8m0,
        is_sf_swizzled_layout,
        is_sf_8x4_layout,
        per_token_activation=False,
        enable_pdl=None,
    ):
        # New provider options may precede PDL in the positional signature.
        assert tensor is input
        assert global_scale is scale
        assert sf_vec_size == 16
        assert not sf_use_ue8m0
        assert not is_sf_swizzled_layout
        assert not is_sf_8x4_layout
        assert per_token_activation is (False if flag == "1" else requested_pdl)
        return enable_pdl

    backend._fp4_quantize = quantize
    assert backend.fp4_quantize(
        input, scale, is_sf_swizzled_layout=False, enable_pdl=enable_pdl
    ) is (enable_pdl if flag == "1" else None)


@pytest.mark.parametrize("flag", [None, "0", "1"])
@pytest.mark.parametrize("tokens", [0, 1, 2, 16])
@pytest.mark.parametrize("routed", [False, True])
@pytest.mark.parametrize("flattened", [False, True])
def test_fp4_input_scales_are_token_major(tokens, routed, flattened, flag, monkeypatch):
    if flag is None:
        monkeypatch.delenv("TRTLLM_FLASHINFER_FP4_CONTRACT_FIX", raising=False)
    else:
        monkeypatch.setenv("TRTLLM_FLASHINFER_FP4_CONTRACT_FIX", flag)
    backend = FlashinferOpBackend()
    observed = []

    def run(*args, **kwargs):
        scales = args[3]
        observed.append(scales)
        return [torch.zeros(tokens, 4096)]

    backend._fused_moe = SimpleNamespace(
        trtllm_fp4_block_scale_moe=run,
        trtllm_fp4_block_scale_routed_moe=run,
    )
    backend.cvt_routing_method_type = lambda _: 1
    backend.cvt_activation_type = lambda _: 3
    params = {
        name: None
        for name, value in inspect.signature(backend.run_fp4_block_scale_moe).parameters.items()
        if value.default is inspect.Parameter.empty
    }
    scales = torch.zeros(tokens, 256, dtype=torch.uint8)
    params.update(
        hidden_states=torch.zeros(tokens, 2048, dtype=torch.uint8),
        hidden_states_scale=scales.flatten() if flattened else scales,
        router_logits=None if routed else torch.zeros(tokens, 256),
        topk_ids=torch.zeros(tokens, 8, dtype=torch.int32),
        topk_weights=torch.ones(tokens, 8, dtype=torch.float32) / 8,
        gemm1_weights_scale=torch.zeros(1, dtype=torch.uint8),
        gemm2_weights_scale=torch.zeros(1, dtype=torch.uint8),
    )
    backend.run_fp4_block_scale_moe(**params)
    assert len(observed) == 1
    expected_shape = (tokens, 256)
    if flattened and (flag != "1" or tokens == 0):
        expected_shape = (tokens * 256,)
    assert observed[0].shape == expected_shape
    assert observed[0].dtype == torch.float8_e4m3fn
    assert observed[0].data_ptr() == scales.data_ptr()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("enable_pdl", [None, False, True])
def test_opt_in_real_fp4_provider_and_graph(enable_pdl, monkeypatch):
    monkeypatch.setenv("TRTLLM_FLASHINFER_FP4_CONTRACT_FIX", "1")
    backend = FlashinferOpBackend()
    x = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16)
    scale = torch.ones(1, device="cuda")

    def quantize():
        return backend.fp4_quantize(x, scale, is_sf_swizzled_layout=False, enable_pdl=enable_pdl)

    quantize()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = quantize()
    x.normal_()
    expected = quantize()
    graph.replay()
    torch.cuda.synchronize()
    assert actual[0].shape == (2, 64)
    assert actual[1].numel() == 16
    for got, want in zip(actual, expected):
        torch.testing.assert_close(got, want, rtol=0, atol=0)
