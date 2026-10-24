# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit-scale CTX output fusion preserves BF16-to-FP8 rounding."""

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="Rubin MXFP8 GEMM requires SM107",
)


@pytest.mark.parametrize("m", [1, 124, 257])
@pytest.mark.parametrize("use_tvm_ffi", [False, True])
def test_fp8_output_matches_bf16_then_quant(m, use_tvm_ffi):
    import tensorrt_llm  # noqa: F401
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLMXFP8RubinLinear

    torch.manual_seed(107)
    x = torch.randn(m, 512, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(256, 512, dtype=torch.bfloat16, device="cuda")
    a, a_sf = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(x)
    b, b_sf = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(w)
    alpha = torch.ones((), dtype=torch.float32, device="cuda")
    inputs = [a, b, a_sf, b_sf, alpha]
    tactic = ("base", (128, 128, 128), (128, 128, 64), (1, 1), False, False, "static", "m")
    reference = CuteDSLMXFP8RubinLinear(torch.bfloat16, use_tvm_ffi=use_tvm_ffi)
    candidate = CuteDSLMXFP8RubinLinear(torch.float8_e4m3fn, use_tvm_ffi=use_tvm_ffi)
    # Values above the E4M3 range exercise finite saturation, in addition to
    # the ordinary distribution and a non-unit GEMM alpha.
    scales = [1.0, 0.5, 1.3, 32.0]
    if m == 124 and use_tvm_ffi:
        scales += [float("inf"), float("-inf"), float("nan")]
    for scale in scales:
        alpha.fill_(scale)
        expected = reference(inputs, tactic=tactic)
        expected, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
            expected, torch.ones_like(alpha)
        )
        actual = candidate(inputs, tactic=tactic)
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("heads", [12, 96])
def test_weight_read_fp8_output_matches_projection(heads):
    from tensorrt_llm._torch.models.modeling_kimi_linear import _Fp8BlockScaleWeightReadLinear

    torch.manual_seed(10698)
    x = torch.randn(257, 512, dtype=torch.bfloat16, device="cuda")
    w = torch.randn(heads * 256, 512, dtype=torch.bfloat16, device="cuda") / 32
    weight, weight_scale = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(w)
    projection = _Fp8BlockScaleWeightReadLinear(weight, weight_scale, heads * 256)
    expected, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
        projection(x), torch.ones((), dtype=torch.float32, device="cuda")
    )
    activation, scale = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(x)
    actual = torch.ops.trtllm.cute_dsl_mxfp8_gemm_rubin(
        activation,
        projection.weight,
        scale,
        projection.weight_scale,
        output_dtype=torch.float8_e4m3fn,
    )
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize(
    "weight_dtype,block_scales,cute_backend,disable_deep_gemm,expected_calls,projection_kind",
    [
        (torch.bfloat16, True, True, False, 0, "linear"),
        (torch.float8_e4m3fn, True, True, False, 1, "linear"),
        (torch.float8_e4m3fn, False, True, False, 0, "linear"),
        (torch.float8_e4m3fn, True, False, False, 0, "linear"),
        (torch.float8_e4m3fn, True, False, True, 1, "linear"),
        (torch.float8_e4m3fn, True, True, False, 1, "weight_read"),
        (torch.float8_e4m3fn, True, True, False, 1, "missing_bias"),
        (torch.float8_e4m3fn, True, True, False, 0, "missing_weight"),
        (torch.float8_e4m3fn, True, True, False, 0, "with_bias"),
    ],
)
@pytest.mark.parametrize("cached_prefix", [False, True])
@pytest.mark.parametrize("fp4_kv_cache", [True, False], ids=["nvfp4_kv", "fp8_kv"])
def test_context_projection_dispatch(
    monkeypatch,
    weight_dtype,
    block_scales,
    cute_backend,
    disable_deep_gemm,
    expected_calls,
    projection_kind,
    cached_prefix,
    fp4_kv_cache,
):
    from types import SimpleNamespace

    import tensorrt_llm._torch.modules.mla as mla_module

    calls = []
    device = "cuda"
    tokens = 16

    class Projection:
        bias = None
        quant_config = SimpleNamespace(
            quant_mode=SimpleNamespace(has_fp8_block_scales=lambda: block_scales)
        )
        use_cute_dsl_blockscaling_mm = cute_backend
        weight = torch.empty(24576, 512, dtype=weight_dtype, device=device)
        weight_scale = torch.empty(24576 * 16, dtype=torch.uint8, device=device)

        def __call__(self, x):
            calls.append("legacy")
            return torch.zeros(x.shape[0], 24576, dtype=torch.bfloat16, device=device)

    def fused(a, weight, a_scale, w_scale, **kwargs):
        calls.append("fused")
        assert kwargs["output_dtype"] == torch.float8_e4m3fn
        assert a.dtype == torch.float8_e4m3fn and a_scale.dtype == torch.uint8
        return torch.zeros(a.shape[0], 24576, dtype=torch.float8_e4m3fn, device=device)

    mha = SimpleNamespace(
        uses_fp4_mla_attention=fp4_kv_cache,
        supports_k3_fp8_split_k_context=lambda: True,
        append_fp4_mla_context_cache=lambda *args: calls.append("fp4_append"),
        mla_rope_append_paged_kv_assign_q=lambda *args: calls.append("fp8_append"),
        kv_cache_scaling_factor=torch.ones((), device=device),
        forward_k3_fp8_split_k_context_partition=lambda *args, **kwargs: calls.append("fmha"),
    )
    cu = torch.tensor([0, tokens], dtype=torch.int32, device=device)
    meta = SimpleNamespace(
        num_ctx_tokens=tokens,
        chunked_loop_num=int(cached_prefix),
        mla_prepare_ctx_cu_seqlens=lambda: cu,
        prompt_lens_cuda_runtime=torch.tensor([tokens], device=device),
        prompt_lens_cpu_runtime=torch.tensor([tokens]),
        max_ctx_seq_len=tokens,
        merge_op_tensor=[torch.zeros(1, dtype=torch.int64, device=device)] * 2,
    )
    if cached_prefix:
        meta.num_contexts = 1
        meta.cu_chunked_seq_len = [cu]
        meta.host_cu_chunked_seq_len = torch.tensor([[0, tokens]])
        meta.chunked_global_offset = [torch.zeros(1, dtype=torch.int32, device=device)]
        meta.max_chunk_len_per_loop = [tokens]
        meta.chunked_seq_len = [torch.tensor([tokens], device=device)]
        meta.host_chunked_seq_len = [torch.tensor([tokens])]

        def load_cache(**kwargs):
            # Only NVFP4 storage gathers straight to MXFP8; an FP8 cache loads dense.
            mxfp8 = bool(expected_calls) and fp4_kv_cache
            assert kwargs["output_format"] == ("mxfp8" if mxfp8 else "dense")
            x = torch.zeros(tokens, 512, dtype=torch.bfloat16, device=device)
            k_pe = torch.zeros(tokens, 64, dtype=torch.bfloat16, device=device)
            if mxfp8:
                x = torch.ops.trtllm.fp8_quantize_1x128_packed_ue8m0(x)
                k_pe = k_pe.to(torch.float8_e4m3fn)
            return x, k_pe

        mha.load_chunked_kv_cache_for_mla = load_cache

    projection = Projection()
    projection.disable_deep_gemm = disable_deep_gemm
    if projection_kind == "weight_read":
        from tensorrt_llm._torch.models.modeling_kimi_linear import _Fp8BlockScaleWeightReadLinear

        # Use the actual post-load K3 wrapper, whose interface differs from Linear.
        projection = _Fp8BlockScaleWeightReadLinear(
            projection.weight, projection.weight_scale, 24576
        )
        assert projection.supports_prequantized_input
        assert not hasattr(projection, "bias")
    elif projection_kind == "missing_bias":
        del Projection.bias
    elif projection_kind == "missing_weight":
        del Projection.weight
    elif projection_kind == "with_bias":
        projection.bias = torch.zeros(24576, dtype=torch.bfloat16, device=device)
    module = SimpleNamespace(
        mha=mha,
        out_scale=None,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        num_heads_tp=96,
        v_head_dim=128,
        kv_b_proj=projection,
    )
    monkeypatch.setattr(mla_module, "get_sm_version", lambda: 107)
    monkeypatch.setattr(mla_module, "IS_CUTLASS_DSL_RUBIN_AVAILABLE", True)
    monkeypatch.setattr(torch.ops.trtllm, "cute_dsl_mxfp8_gemm_rubin", fused)
    q = torch.zeros(tokens, 96, 192, dtype=torch.bfloat16, device=device)
    x = torch.zeros(tokens, 512, dtype=torch.bfloat16, device=device)
    latent = torch.zeros(tokens, 576, dtype=torch.bfloat16, device=device)
    output = torch.empty(tokens, 96 * 128, dtype=torch.bfloat16, device=device)
    result = mla_module.MLA.forward_context_with_chunked_prefill_fp4(
        module, q, x, latent, meta, output
    )
    assert result is output
    partitions = 1 + int(cached_prefix)
    assert calls.count("fused") == expected_calls * partitions
    assert calls.count("legacy") == (1 - expected_calls) * partitions
    assert calls.count("fmha") == partitions
    assert calls.count("fp4_append") == int(fp4_kv_cache)
    assert calls.count("fp8_append") == int(not fp4_kv_cache)
