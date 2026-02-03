# test_quant_fusion.py
import pytest
import torch
import torch.nn as nn
from _graph_test_helpers import run_test_transformed_gm
from _torch_test_utils import fp4_compatible, fp8_compatible, trtllm_ops_available

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op
from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fp4_global_scale, fp8_scale


def _has_fused_linear_fp8(gm):
    found_fused = any(
        is_op(n, torch.ops.auto_deploy.torch_quant_fp8_linear) for n in gm.graph.nodes
    )
    found_ref = any(
        is_op(n, torch.ops.auto_deploy.torch_fake_quant_fp8_linear.default) for n in gm.graph.nodes
    )
    return found_fused and not found_ref


def _has_fused_hf_fp8_linear(gm):
    """Check if HF FP8 fake quant ops were replaced with TRT-LLM ops."""
    found_fused = any(is_op(n, torch.ops.auto_deploy.trtllm_hf_fp8_linear) for n in gm.graph.nodes)
    found_ref = any(
        is_op(n, torch.ops.auto_deploy.torch_fake_quant_hf_fp8_linear) for n in gm.graph.nodes
    )
    return found_fused and not found_ref


def _has_fused_linear_fp4(gm):
    found_fused = any(
        is_op(n, torch.ops.auto_deploy.torch_quant_nvfp4_linear) for n in gm.graph.nodes
    )
    found_ref = any(
        is_op(n, torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear) for n in gm.graph.nodes
    )
    return found_fused and not found_ref


class TinyFP8Ref(nn.Module):
    """
    A tiny module whose forward uses the reference FP8 op:
      torch_fake_quant_fp8_linear(input, weight_fp8, bias, [in_s], [w_s], [], [])
    """

    def __init__(self, in_features=16, out_features=32, use_bias=True):
        super().__init__()
        self.use_bias = use_bias
        self.weight = nn.Parameter(torch.rand(out_features, in_features, dtype=torch.float16))
        if use_bias:
            self.bias = nn.Parameter(torch.rand(out_features, dtype=torch.float16))
        else:
            self.register_parameter("bias", None)

        # Precompute FP8 packing + scales as buffers
        with torch.no_grad():
            w_s = fp8_scale(self.weight)  # per-tensor scale
            w_fp8 = (self.weight / w_s).to(torch.float8_e4m3fn)

        self.register_buffer("weight_fp8", w_fp8)
        self.register_buffer("weight_scale", w_s)
        self.register_buffer(
            "input_scale", torch.tensor(1.0, dtype=torch.float32)
        )  # simple test scale

    def forward(self, x):
        bias = self.bias if self.use_bias else None
        return torch.ops.auto_deploy.torch_fake_quant_fp8_linear.default(
            x,
            self.weight_fp8,
            bias,
            [self.input_scale],
            [self.weight_scale],
            [],
            [],
        )


class TinyFP4Ref(nn.Module):
    """
    A tiny module whose forward uses the reference NVFP4 op:
      torch_fake_quant_nvfp4_linear(x, w_fp4, bias, [s_in2], [cutlass_vec, alpha], [], [])
    """

    def __init__(self, in_features=64, out_features=32, use_bias=True):
        super().__init__()
        assert in_features % 16 == 0, "NVFP4 requires K % 16 == 0 for CUTLASS scaling."
        device = torch.device("cuda")

        self.use_bias = use_bias
        self.weight = nn.Parameter(
            torch.rand(out_features, in_features, dtype=torch.half, device=device)
        )
        if use_bias:
            self.bias = nn.Parameter(torch.rand(out_features, dtype=torch.half, device=device))
        else:
            self.register_parameter("bias", None)

        with torch.no_grad():
            s_in2 = fp4_global_scale(torch.rand(1, in_features, dtype=torch.half, device=device))
            s_w2 = fp4_global_scale(self.weight)
            w_fp4, cutlass_vec = torch.ops.trtllm.fp4_quantize(self.weight, s_w2, 16, False)
            alpha = (1.0 / (s_in2 * s_w2)).to(torch.float32)

        self.register_buffer("weight_fp4", w_fp4)  # uint8 packed
        self.register_buffer("input_scale_2", s_in2.to(torch.float32))
        self.register_buffer("weight_scale_cutlass", cutlass_vec)  # uint8 vec
        self.register_buffer("alpha", alpha.to(torch.float32))

    def forward(self, x):
        bias = self.bias if self.use_bias else None
        return torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear(
            x,
            self.weight_fp4,
            bias,
            [self.input_scale_2],
            [self.weight_scale_cutlass, self.alpha],
            [],
            [],
        )


class TinyHFFP8Ref(nn.Module):
    """
    A tiny module whose forward uses the HuggingFace FineGrained FP8 op:
      torch_fake_quant_hf_fp8_linear(x, w_fp8, bias, [], [weight_scale_inv], [], [])

    This simulates models like MiniMax M2 and DeepSeek that use HF's block-wise FP8.
    """

    def __init__(self, in_features=256, out_features=256, use_bias=True):
        super().__init__()
        # HF FP8 uses 128x128 block quantization, so dimensions must be multiples of 128
        assert in_features % 128 == 0, "HF FP8 requires in_features % 128 == 0"
        assert out_features % 128 == 0, "HF FP8 requires out_features % 128 == 0"
        device = torch.device("cuda")

        self.use_bias = use_bias
        self.weight = nn.Parameter(
            torch.rand(out_features, in_features, dtype=torch.bfloat16, device=device)
        )
        if use_bias:
            self.bias = nn.Parameter(torch.rand(out_features, dtype=torch.bfloat16, device=device))
        else:
            self.register_parameter("bias", None)

        # Compute block-wise FP8 quantization (128x128 blocks)
        with torch.no_grad():
            block_n, block_k = 128, 128
            N, K = out_features, in_features

            # Reshape to blocks and compute per-block max
            weight_reshaped = self.weight.view(N // block_n, block_n, K // block_k, block_k)
            amax = weight_reshaped.abs().amax(dim=(1, 3)).to(torch.float32)  # [N/128, K/128]

            # Compute per-block scale (amax / 448 for FP8 E4M3)
            FP8_MAX = 448.0
            eps = torch.finfo(torch.float32).tiny
            weight_scale_inv = torch.clamp(amax / FP8_MAX, min=eps)  # [N/128, K/128]

            # Quantize weight to FP8
            scale_expanded = weight_scale_inv.repeat_interleave(block_n, dim=0).repeat_interleave(
                block_k, dim=1
            )
            w_fp8 = (self.weight.float() / scale_expanded).to(torch.float8_e4m3fn)

        self.register_buffer("weight_fp8", w_fp8)
        self.register_buffer("weight_scale_inv", weight_scale_inv)

    def forward(self, x):
        bias = self.bias if self.use_bias else None
        return torch.ops.auto_deploy.torch_fake_quant_hf_fp8_linear(
            x,
            self.weight_fp8,
            bias,
            [],  # input_scale unused
            [self.weight_scale_inv],
            [],  # input_zp unused
            [],  # weight_zp unused
        )


@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.skipif(not fp8_compatible(), reason="Requires fp8 support")
def test_fuse_quant_rewrites_fp8_linear(use_bias):
    torch.manual_seed(0)
    model = TinyFP8Ref(use_bias=use_bias).to("cuda")
    x = torch.rand(3, 16, dtype=torch.float16, device="cuda")

    gm = torch_export_to_gm(model, args=(x,), clone=True)
    gm_transformed = InferenceOptimizer(
        None,
        {
            "fuse_fp8_linear": {"stage": "post_load_fusion", "backend": "torch"},
        },
    )(None, gm)
    gm_transformed.to("cuda")

    run_test_transformed_gm(
        model,
        x,
        gm_transformed,
        _has_fused_linear_fp8,
        lambda n: n,
        0.1,  # atol
        0.05,  # rtol
        False,  # test_load_hook
        False,  # strict_loading
        None,  # dynamic_shapes
        False,  # skip_output_assert
    )


@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.skipif(
    not (fp4_compatible() and trtllm_ops_available()),
    reason="Requires NVFP4 and TRT-LLM ops",
)
def test_fuse_quant_rewrites_fp4_linear(use_bias):
    torch.manual_seed(0)
    model = TinyFP4Ref(use_bias=use_bias).to("cuda")
    x = torch.rand(3, 64, dtype=torch.float16, device="cuda")

    gm = torch_export_to_gm(model, args=(x,), clone=True)
    gm_transformed = InferenceOptimizer(
        None,
        {
            "fuse_nvfp4_linear": {"stage": "post_load_fusion", "backend": "trtllm"},
        },
    )(None, gm)
    gm_transformed.to("cuda")

    run_test_transformed_gm(
        model,
        x,
        gm_transformed,
        _has_fused_linear_fp4,
        lambda n: n,
        0.1,  # atol
        0.05,  # rtol
        False,  # test_load_hook
        False,  # strict_loading
        None,  # dynamic_shapes
        False,  # skip_output_assert
    )


@pytest.mark.parametrize("use_bias", [True, False])
@pytest.mark.skipif(
    not (fp8_compatible() and trtllm_ops_available()),
    reason="Requires FP8 and TRT-LLM ops",
)
def test_fuse_quant_rewrites_hf_fp8_linear(use_bias):
    """Test that torch_fake_quant_hf_fp8_linear is replaced with trtllm_hf_fp8_linear.

    This tests the fusion transform for HuggingFace FineGrained FP8 models like
    MiniMax M2 and DeepSeek, which use 128x128 block-wise FP8 quantization.
    """
    torch.manual_seed(0)
    model = TinyHFFP8Ref(use_bias=use_bias).to("cuda")
    x = torch.rand(3, 256, dtype=torch.bfloat16, device="cuda")

    gm = torch_export_to_gm(model, args=(x,), clone=True)
    gm_transformed = InferenceOptimizer(
        None,
        {
            "fuse_hf_fp8_linear": {"stage": "post_load_fusion", "backend": "trtllm"},
        },
    )(None, gm)
    gm_transformed.to("cuda")

    run_test_transformed_gm(
        model,
        x,
        gm_transformed,
        _has_fused_hf_fp8_linear,
        lambda n: n,
        0.1,  # atol - HF FP8 has some quantization error
        0.05,  # rtol
        False,  # test_load_hook
        False,  # strict_loading
        None,  # dynamic_shapes
        True,  # skip_output_assert - skip numerical comparison for now
    )
