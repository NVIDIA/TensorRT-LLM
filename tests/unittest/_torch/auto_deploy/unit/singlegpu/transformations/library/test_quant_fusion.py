# test_quant_fusion.py
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from _graph_test_helpers import run_test_transformed_gm
from _torch_test_utils import fp4_compatible, fp8_compatible, trtllm_ops_available

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.interface import TransformConfig
from tensorrt_llm._torch.auto_deploy.transform.library.fuse_rmsnorm_quant_fp8 import (
    FuseRMSNormQuantFP8,
)
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


# ---------------------------------------------------------------------------
# FuseRMSNormQuantFP8 tests
# ---------------------------------------------------------------------------


class _RMSNormFP8LinearModel(nn.Module):
    """Model whose forward already uses flashinfer_rms_norm + trtllm_quant_fp8_linear.

    Simulates the graph state AFTER fuse_rmsnorm and fuse_fp8_linear have run,
    which is the input expected by FuseRMSNormQuantFP8.

    Args:
        num_linears: Number of FP8 linear consumers of the norm output (e.g. 3
            for Q/K/V projections).
        extra_norm_consumer: If True, the norm output is also returned directly
            (mimics a residual connection).
    """

    def __init__(self, in_features=128, out_features=256, num_linears=3, extra_norm_consumer=True):
        super().__init__()
        self.num_linears = num_linears
        self.extra_norm_consumer = extra_norm_consumer

        self.norm_weight = nn.Parameter(torch.randn(in_features, dtype=torch.bfloat16))
        self.input_scale = nn.Buffer(torch.tensor(1.0, dtype=torch.float32))

        with torch.no_grad():
            for i in range(num_linears):
                w_scale = torch.tensor(1.0, dtype=torch.float32)
                w_bf16 = torch.randn(out_features, in_features, dtype=torch.bfloat16)
                w_fp8 = (w_bf16 / w_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
                self.register_parameter(f"w_fp8_{i}", nn.Parameter(w_fp8))
                self.register_buffer(f"w_scale_{i}", w_scale)

    def forward(self, x):
        norm_out = torch.ops.auto_deploy.flashinfer_rms_norm(x, self.norm_weight, 1e-5)

        outputs = []
        for i in range(self.num_linears):
            w_fp8 = getattr(self, f"w_fp8_{i}")
            w_scale = getattr(self, f"w_scale_{i}")
            out = torch.ops.auto_deploy.trtllm_quant_fp8_linear(
                norm_out,
                w_fp8,
                None,
                input_scale=self.input_scale,
                weight_scale=w_scale,
            )
            outputs.append(out)

        combined = outputs[0]
        for o in outputs[1:]:
            combined = combined + o

        if self.extra_norm_consumer:
            return combined, norm_out
        return combined


def _has_fused_rmsnorm_quant_fp8(gm):
    """Check that the graph contains fused ops and no unfused ones."""
    has_fused_norm = any(
        is_op(n, torch.ops.auto_deploy.triton_rms_norm_quant_fp8) for n in gm.graph.nodes
    )
    has_fp8_gemm = any(is_op(n, torch.ops.auto_deploy.trtllm_fp8_gemm) for n in gm.graph.nodes)
    no_old_norm = not any(
        is_op(n, torch.ops.auto_deploy.flashinfer_rms_norm) for n in gm.graph.nodes
    )
    no_old_fp8_linear = not any(
        is_op(n, torch.ops.auto_deploy.trtllm_quant_fp8_linear) for n in gm.graph.nodes
    )
    return has_fused_norm and has_fp8_gemm and no_old_norm and no_old_fp8_linear


@pytest.mark.parametrize("num_linears", [1, 3])
@pytest.mark.parametrize("extra_norm_consumer", [True, False])
@pytest.mark.skipif(
    not (fp8_compatible() and trtllm_ops_available()),
    reason="Requires FP8 support and TRT-LLM ops",
)
def test_fuse_rmsnorm_quant_fp8(num_linears, extra_norm_consumer):
    """Verify FuseRMSNormQuantFP8 transform replaces ops and preserves numerics."""
    torch.manual_seed(0)
    K, N = 128, 256
    model = _RMSNormFP8LinearModel(
        in_features=K,
        out_features=N,
        num_linears=num_linears,
        extra_norm_consumer=extra_norm_consumer,
    ).cuda()
    x = torch.randn(4, K, device="cuda", dtype=torch.bfloat16)

    # Get reference output before transform
    ref_out = model(x)

    # Trace to FX graph (symbolic_trace handles custom ops cleanly)
    gm = torch.fx.symbolic_trace(model)

    # Verify precondition: graph has the unfused ops
    assert any(is_op(n, torch.ops.auto_deploy.flashinfer_rms_norm) for n in gm.graph.nodes), (
        "Precondition: graph should contain flashinfer_rms_norm before transform"
    )
    assert any(is_op(n, torch.ops.auto_deploy.trtllm_quant_fp8_linear) for n in gm.graph.nodes), (
        "Precondition: graph should contain trtllm_quant_fp8_linear before transform"
    )

    # Run the transform
    config = TransformConfig(stage="post_load_fusion")
    transform = FuseRMSNormQuantFP8(config)
    gm, info = transform._apply(gm, MagicMock(), MagicMock(), MagicMock())

    # Verify transform fired
    assert info.num_matches == num_linears, (
        f"Expected {num_linears} matches, got {info.num_matches}"
    )
    assert not info.skipped

    # Verify graph structure
    assert _has_fused_rmsnorm_quant_fp8(gm), (
        "Graph should contain triton_rms_norm_quant_fp8 + trtllm_fp8_gemm, "
        "and no flashinfer_rms_norm or trtllm_quant_fp8_linear"
    )

    # Verify numerical correctness
    fused_out = gm(x)
    if extra_norm_consumer:
        assert len(fused_out) == 2
        torch.testing.assert_close(ref_out[0], fused_out[0], atol=0, rtol=0)
        torch.testing.assert_close(ref_out[1], fused_out[1], atol=0, rtol=0)
    else:
        torch.testing.assert_close(ref_out, fused_out, atol=0, rtol=0)


# ---------------------------------------------------------------------------
# FuseRMSNormQuantNVFP4 tests
# ---------------------------------------------------------------------------


class _RMSNormNVFP4LinearModel(nn.Module):
    """Model whose forward already uses flashinfer_rms_norm + torch_quant_nvfp4_linear.

    Simulates the graph state AFTER fuse_rmsnorm and fuse_nvfp4_linear have run,
    which is the input expected by FuseRMSNormQuantNVFP4.

    Args:
        num_linears: Number of NVFP4 linear consumers of the norm output.
        extra_norm_consumer: If True, the norm output is also returned directly.
    """

    def __init__(
        self,
        in_features=2048,
        out_features=2048,
        num_linears=3,
        extra_norm_consumer=True,
    ):
        super().__init__()
        device = torch.device("cuda")
        self.num_linears = num_linears
        self.extra_norm_consumer = extra_norm_consumer

        self.norm_weight = nn.Parameter(
            torch.randn(in_features, dtype=torch.bfloat16, device=device)
        )

        with torch.no_grad():
            dummy_input = torch.randn(1, in_features, dtype=torch.bfloat16, device=device)
            s_in = fp4_global_scale(dummy_input)
            self.register_buffer("input_scale", s_in.to(torch.float32))

            for i in range(num_linears):
                w_bf16 = torch.randn(out_features, in_features, dtype=torch.bfloat16, device=device)
                s_w = fp4_global_scale(w_bf16)
                w_fp4, w_cutlass = torch.ops.trtllm.fp4_quantize(w_bf16, s_w, 16, False)
                alpha = (1.0 / (s_in * s_w)).to(torch.float32)

                self.register_buffer(f"w_fp4_{i}", w_fp4)
                self.register_buffer(f"w_scale_{i}", w_cutlass)
                self.register_buffer(f"alpha_{i}", alpha)

    def forward(self, x):
        norm_out = torch.ops.auto_deploy.flashinfer_rms_norm(x, self.norm_weight, 1e-5)

        outputs = []
        for i in range(self.num_linears):
            w_fp4 = getattr(self, f"w_fp4_{i}")
            w_scale = getattr(self, f"w_scale_{i}")
            alpha = getattr(self, f"alpha_{i}")
            out = torch.ops.auto_deploy.torch_quant_nvfp4_linear(
                norm_out,
                w_fp4,
                bias=None,
                input_scale=self.input_scale,
                weight_scale=w_scale,
                alpha=alpha,
            )
            outputs.append(out)

        combined = outputs[0]
        for o in outputs[1:]:
            combined = combined + o

        if self.extra_norm_consumer:
            return combined, norm_out
        return combined


def _has_fused_rmsnorm_quant_nvfp4(gm):
    """Check that the graph contains fused NVFP4 ops and no unfused ones."""
    has_fused_norm = any(
        is_op(n, torch.ops.auto_deploy.trtllm_rms_norm_quant_nvfp4) for n in gm.graph.nodes
    )
    has_nvfp4_gemm = any(is_op(n, torch.ops.auto_deploy.trtllm_nvfp4_gemm) for n in gm.graph.nodes)
    no_old_norm = not any(
        is_op(n, torch.ops.auto_deploy.flashinfer_rms_norm) for n in gm.graph.nodes
    )
    no_old_nvfp4_linear = not any(
        is_op(n, torch.ops.auto_deploy.torch_quant_nvfp4_linear) for n in gm.graph.nodes
    )
    return has_fused_norm and has_nvfp4_gemm and no_old_norm and no_old_nvfp4_linear
