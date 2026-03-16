import operator

# test_quant_fusion.py
import pytest
import torch
import torch.nn as nn
from _graph_test_helpers import run_test_transformed_gm
from _torch_test_utils import fp4_compatible, fp8_compatible, trtllm_ops_available

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.custom_ops.normalization.flashinfer_fused_add_rms_norm import (  # noqa: F401
    flashinfer_fused_add_rms_norm,
)
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.interface import TransformConfig
from tensorrt_llm._torch.auto_deploy.transform.library.fuse_rmsnorm_quant_fp8 import (
    FuseRMSNormQuantFP8,
    _get_out_dtype_str,
)
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op
from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import (
    cutlass_fp4_scale_to_modelopt_fp4_scale,
    fp4_global_scale,
    fp8_scale,
)


def _has_fused_linear_fp8(gm):
    found_fused = any(
        is_op(n, torch.ops.auto_deploy.torch_quant_fp8_linear) for n in gm.graph.nodes
    )
    found_ref = any(
        is_op(n, torch.ops.auto_deploy.torch_fake_quant_fp8_linear.default) for n in gm.graph.nodes
    )
    return found_fused and not found_ref


def _has_fused_finegrained_fp8_linear(gm):
    """Check if FineGrained FP8 fake quant ops were replaced with TRT-LLM ops."""
    found_fused = any(
        is_op(n, torch.ops.auto_deploy.trtllm_finegrained_fp8_linear) for n in gm.graph.nodes
    )
    found_ref = any(
        is_op(n, torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear)
        for n in gm.graph.nodes
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
    """A tiny module whose forward uses the reference FP8 op.

    Uses: torch_fake_quant_fp8_linear(input, weight_fp8, bias, [in_s], [w_s], [], [])
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
    """A tiny module whose forward uses the reference NVFP4 op.

    Uses: torch_fake_quant_nvfp4_linear(x, w_fp4, bias, [input_scale_2], [weight_scale_fp8, weight_scale_2], [], [])

    Scales are stored in raw (un-processed) format — kernel-specific processing
    (swizzling, alpha computation, input_scale inversion) is deferred to the fusion pass.
      - input_scale_2:   raw amax / FP4_GLOBAL_SCALE_MAX  (= 1 / fp4_global_scale)
      - weight_scale_fp8: per-block FP8 scale (float8_e4m3fn, 2D [N, K//16])
      - weight_scale_2:  raw amax / FP4_GLOBAL_SCALE_MAX  (= 1 / fp4_global_scale)
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
            # Convert CUTLASS uint8 vec → FP8 per-block 2D scale (new IR format)
            weight_scale_fp8 = cutlass_fp4_scale_to_modelopt_fp4_scale(
                cutlass_vec, (out_features, in_features)
            )
            # Store raw "amax / FP4_GLOBAL_SCALE_MAX" = 1/s_in2 and 1/s_w2
            input_s2_raw = (1.0 / s_in2).to(torch.float32)
            weight_s2_raw = (1.0 / s_w2).to(torch.float32)

        self.register_buffer("weight_fp4", w_fp4)  # uint8 packed
        self.register_buffer("weight_scale", weight_scale_fp8)  # float8_e4m3fn 2D
        self.register_buffer("weight_scale_2", weight_s2_raw)  # raw scalar
        self.register_buffer("input_scale", input_s2_raw)  # raw scalar

    def forward(self, x):
        bias = self.bias if self.use_bias else None
        return torch.ops.auto_deploy.torch_fake_quant_nvfp4_linear(
            x,
            self.weight_fp4,
            bias,
            [self.input_scale],
            [self.weight_scale, self.weight_scale_2],
            [],
            [],
        )


class TinyFineGrainedFP8Ref(nn.Module):
    """A tiny module whose forward uses the FineGrained FP8 op.

      torch_fake_quant_finegrained_fp8_linear(x, w_fp8, bias, [], [weight_scale_inv], [], [])

    This simulates models like MiniMax M2 and DeepSeek that use HF's block-wise FP8.
    """

    def __init__(self, in_features=256, out_features=256, use_bias=True):
        super().__init__()
        # FineGrained FP8 uses 128x128 block quantization, so dimensions must be multiples of 128
        assert in_features % 128 == 0, "FineGrained FP8 requires in_features % 128 == 0"
        assert out_features % 128 == 0, "FineGrained FP8 requires out_features % 128 == 0"
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
        return torch.ops.auto_deploy.torch_fake_quant_finegrained_fp8_linear(
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
def test_fuse_quant_rewrites_finegrained_fp8_linear(use_bias):
    """Test that torch_fake_quant_finegrained_fp8_linear is replaced with trtllm_finegrained_fp8_linear.

    This tests the fusion transform for FineGrained FP8 models like
    MiniMax M2 and DeepSeek, which use 128x128 block-wise FP8 quantization.
    """
    torch.manual_seed(0)
    model = TinyFineGrainedFP8Ref(use_bias=use_bias).to("cuda")
    x = torch.rand(3, 256, dtype=torch.bfloat16, device="cuda")

    gm = torch_export_to_gm(model, args=(x,), clone=True)
    gm_transformed = InferenceOptimizer(
        None,
        {
            "fuse_finegrained_fp8_linear": {"stage": "post_load_fusion", "backend": "trtllm"},
        },
    )(None, gm)
    gm_transformed.to("cuda")

    run_test_transformed_gm(
        model,
        x,
        gm_transformed,
        _has_fused_finegrained_fp8_linear,
        lambda n: n,
        0.1,  # atol - FineGrained FP8 has some quantization error
        0.05,  # rtol
        False,  # test_load_hook
        False,  # strict_loading
        None,  # dynamic_shapes
        True,  # skip_output_assert - skip numerical comparison for now
    )


class TinyRMSNormQuantFP8(nn.Module):
    """Minimal graph containing flashinfer_rms_norm -> trtllm_quant_fp8_linear."""

    def __init__(self, hidden_size=128):
        super().__init__()
        self.eps = 1e-5
        self.norm_weight = nn.Parameter(
            torch.ones(hidden_size, dtype=torch.bfloat16, device="cuda")
        )
        self.input_scale = nn.Buffer(torch.tensor(1.0, dtype=torch.float32, device="cuda"))

        with torch.no_grad():
            w_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
            w_bf16 = torch.randn(hidden_size, hidden_size, dtype=torch.bfloat16, device="cuda")
            w_fp8 = (w_bf16 / w_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        self.register_buffer("weight_fp8", w_fp8)
        self.register_buffer("weight_scale", w_scale)

    def forward(self, x):
        norm_out = torch.ops.auto_deploy.flashinfer_rms_norm(x, self.norm_weight, self.eps)
        return torch.ops.auto_deploy.trtllm_quant_fp8_linear(
            norm_out,
            self.weight_fp8,
            None,
            input_scale=self.input_scale,
            weight_scale=self.weight_scale,
        )


class TinyFusedAddRMSNormQuantFP8(nn.Module):
    """Graph containing flashinfer_fused_add_rms_norm -> getitem[0] -> FP8 linear."""

    def __init__(self, hidden_size=128):
        super().__init__()
        self.eps = 1e-5
        self.norm_weight = nn.Parameter(
            torch.ones(hidden_size, dtype=torch.bfloat16, device="cuda")
        )
        self.input_scale = nn.Buffer(torch.tensor(1.0, dtype=torch.float32, device="cuda"))

        with torch.no_grad():
            w_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
            w_bf16 = torch.randn(hidden_size, hidden_size, dtype=torch.bfloat16, device="cuda")
            w_fp8 = (w_bf16 / w_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        self.register_buffer("weight_fp8", w_fp8)
        self.register_buffer("weight_scale", w_scale)

    def forward(self, x, residual):
        norm_out, add_out = flashinfer_fused_add_rms_norm(
            x,
            residual,
            self.norm_weight,
            self.eps,
        )
        gemm_out = torch.ops.auto_deploy.trtllm_quant_fp8_linear(
            norm_out,
            self.weight_fp8,
            None,
            input_scale=self.input_scale,
            weight_scale=self.weight_scale,
        )
        return gemm_out, add_out


class ExportedRMSNormQuantFP8MultiConsumer(nn.Module):
    """Exported graph path with reshape and two FP8 consumers."""

    def __init__(self, hidden_size=128):
        super().__init__()
        self.eps = 1e-5
        self.norm_weight = nn.Parameter(
            torch.ones(hidden_size, dtype=torch.bfloat16, device="cuda")
        )
        self.input_scale = nn.Buffer(torch.tensor(1.0, dtype=torch.float32, device="cuda"))

        with torch.no_grad():
            w_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
            w_bf16 = torch.randn(hidden_size, hidden_size, dtype=torch.bfloat16, device="cuda")
            w_fp8 = (w_bf16 / w_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        self.register_buffer("weight_fp8", w_fp8)
        self.register_buffer("weight_scale", w_scale)

    def forward(self, x):
        norm_out = torch.ops.auto_deploy.flashinfer_rms_norm(x, self.norm_weight, self.eps)
        reshaped = torch.ops.aten.reshape.default(norm_out, list(norm_out.shape))
        out_trtllm = torch.ops.auto_deploy.trtllm_quant_fp8_linear(
            reshaped,
            self.weight_fp8,
            None,
            input_scale=self.input_scale,
            weight_scale=self.weight_scale,
        )
        out_torch = torch.ops.auto_deploy.torch_quant_fp8_linear(
            reshaped,
            self.weight_fp8,
            None,
            self.input_scale,
            self.weight_scale,
        )
        return out_trtllm, out_torch


class ExportedRMSNormQuantFP8MixedConsumers(nn.Module):
    """Exported graph path with a non-quant consumer before the FP8 consumer."""

    def __init__(self, hidden_size=128):
        super().__init__()
        self.eps = 1e-5
        self.norm_weight = nn.Parameter(
            torch.ones(hidden_size, dtype=torch.bfloat16, device="cuda")
        )
        self.input_scale = nn.Buffer(torch.tensor(1.0, dtype=torch.float32, device="cuda"))

        with torch.no_grad():
            w_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
            w_bf16 = torch.randn(hidden_size, hidden_size, dtype=torch.bfloat16, device="cuda")
            w_fp8 = (w_bf16 / w_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
        self.register_buffer("weight_fp8", w_fp8)
        self.register_buffer("weight_scale", w_scale)

    def forward(self, x):
        norm_out = torch.ops.auto_deploy.flashinfer_rms_norm(x, self.norm_weight, self.eps)
        skipped_consumer = norm_out + 1
        gemm_out = torch.ops.auto_deploy.trtllm_quant_fp8_linear(
            norm_out,
            self.weight_fp8,
            None,
            input_scale=self.input_scale,
            weight_scale=self.weight_scale,
        )
        return skipped_consumer, gemm_out


@pytest.mark.skipif(not fp8_compatible(), reason="Requires fp8 support")
def test_fuse_rmsnorm_quant_fp8_rewrites_graph():
    model = TinyRMSNormQuantFP8().cuda()
    gm = torch.fx.symbolic_trace(model)
    for node in gm.graph.nodes:
        if is_op(node, torch.ops.auto_deploy.flashinfer_rms_norm):
            node.meta["val"] = torch.empty((1, 1), dtype=torch.bfloat16)

    assert any(is_op(n, torch.ops.auto_deploy.flashinfer_rms_norm) for n in gm.graph.nodes)
    assert any(is_op(n, torch.ops.auto_deploy.trtllm_quant_fp8_linear) for n in gm.graph.nodes)

    transform = FuseRMSNormQuantFP8(TransformConfig(stage="post_load_fusion"))
    gm, info = transform._apply(gm, None, None, None)

    assert info.num_matches == 1
    assert any(is_op(n, torch.ops.auto_deploy.triton_rms_norm_quant_fp8) for n in gm.graph.nodes)
    assert any(is_op(n, torch.ops.auto_deploy.trtllm_fp8_prequant_linear) for n in gm.graph.nodes)
    assert not any(is_op(n, torch.ops.auto_deploy.trtllm_quant_fp8_linear) for n in gm.graph.nodes)


@pytest.mark.skipif(not fp8_compatible(), reason="Requires fp8 support")
def test_fuse_rmsnorm_quant_fp8_exported_graph_rewrites_multi_consumer_path():
    model = ExportedRMSNormQuantFP8MultiConsumer().cuda()
    x = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16)

    gm = torch_export_to_gm(model, args=(x,), clone=True)
    transform = FuseRMSNormQuantFP8(TransformConfig(stage="post_load_fusion"))
    gm, info = transform._apply(gm, None, None, None)

    assert info.num_matches == 2
    assert (
        sum(is_op(n, torch.ops.auto_deploy.triton_rms_norm_quant_fp8) for n in gm.graph.nodes) == 1
    )
    assert any(is_op(n, torch.ops.aten.reshape.default) for n in gm.graph.nodes)
    assert (
        sum(is_op(n, torch.ops.auto_deploy.trtllm_fp8_prequant_linear) for n in gm.graph.nodes) == 2
    )
    assert not any(is_op(n, torch.ops.auto_deploy.trtllm_quant_fp8_linear) for n in gm.graph.nodes)
    assert not any(is_op(n, torch.ops.auto_deploy.torch_quant_fp8_linear) for n in gm.graph.nodes)


@pytest.mark.skipif(not fp8_compatible(), reason="Requires fp8 support")
def test_fuse_rmsnorm_quant_fp8_exported_graph_skips_when_non_quant_consumer_precedes_fp8():
    model = ExportedRMSNormQuantFP8MixedConsumers().cuda()
    x = torch.randn(2, 128, device="cuda", dtype=torch.bfloat16)

    gm = torch_export_to_gm(model, args=(x,), clone=True)
    transform = FuseRMSNormQuantFP8(TransformConfig(stage="post_load_fusion"))
    gm, info = transform._apply(gm, None, None, None)

    assert info.num_matches == 0
    assert any(is_op(n, torch.ops.auto_deploy.flashinfer_rms_norm) for n in gm.graph.nodes)
    assert any(is_op(n, torch.ops.auto_deploy.trtllm_quant_fp8_linear) for n in gm.graph.nodes)
    assert not any(
        is_op(n, torch.ops.auto_deploy.triton_rms_norm_quant_fp8) for n in gm.graph.nodes
    )


@pytest.mark.skipif(not fp8_compatible(), reason="Requires fp8 support")
def test_fuse_rmsnorm_quant_fp8_rewrites_torch_quant_linear_graph():
    model = TinyRMSNormQuantFP8().cuda()
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    norm_weight = graph.get_attr("norm_weight")
    weight_fp8 = graph.get_attr("weight_fp8")
    input_scale = graph.get_attr("input_scale")
    weight_scale = graph.get_attr("weight_scale")

    norm_out = graph.call_function(
        torch.ops.auto_deploy.flashinfer_rms_norm.default,
        args=(x, norm_weight, model.eps),
    )
    gemm_out = graph.call_function(
        torch.ops.auto_deploy.torch_quant_fp8_linear.default,
        args=(norm_out, weight_fp8, None, input_scale, weight_scale),
    )
    graph.output(gemm_out)

    norm_out.meta["val"] = torch.empty((1, 1), dtype=torch.bfloat16)

    gm = torch.fx.GraphModule(model, graph)
    transform = FuseRMSNormQuantFP8(TransformConfig(stage="post_load_fusion"))
    gm, info = transform._apply(gm, None, None, None)

    assert info.num_matches == 1
    assert any(is_op(n, torch.ops.auto_deploy.triton_rms_norm_quant_fp8) for n in gm.graph.nodes)
    assert any(is_op(n, torch.ops.auto_deploy.trtllm_fp8_prequant_linear) for n in gm.graph.nodes)
    assert not any(is_op(n, torch.ops.auto_deploy.torch_quant_fp8_linear) for n in gm.graph.nodes)


@pytest.mark.skipif(not fp8_compatible(), reason="Requires fp8 support")
def test_fuse_rmsnorm_quant_fp8_rewrites_torch_rmsnorm_graph():
    model = TinyRMSNormQuantFP8().cuda()
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    norm_weight = graph.get_attr("norm_weight")
    weight_fp8 = graph.get_attr("weight_fp8")
    input_scale = graph.get_attr("input_scale")
    weight_scale = graph.get_attr("weight_scale")

    norm_out = graph.call_function(
        torch.ops.auto_deploy.torch_rmsnorm.default,
        args=(x, norm_weight, model.eps),
    )
    gemm_out = graph.call_function(
        torch.ops.auto_deploy.trtllm_quant_fp8_linear.default,
        args=(norm_out, weight_fp8, None, input_scale, weight_scale),
    )
    graph.output(gemm_out)

    norm_out.meta["val"] = torch.empty((1, 1), dtype=torch.bfloat16)

    gm = torch.fx.GraphModule(model, graph)
    transform = FuseRMSNormQuantFP8(TransformConfig(stage="post_load_fusion"))
    gm, info = transform._apply(gm, None, None, None)

    assert info.num_matches == 1
    assert any(is_op(n, torch.ops.auto_deploy.triton_rms_norm_quant_fp8) for n in gm.graph.nodes)
    assert any(is_op(n, torch.ops.auto_deploy.trtllm_fp8_prequant_linear) for n in gm.graph.nodes)
    assert not any(is_op(n, torch.ops.auto_deploy.torch_rmsnorm) for n in gm.graph.nodes)


@pytest.mark.skipif(not fp8_compatible(), reason="Requires fp8 support")
def test_fuse_rmsnorm_quant_fp8_rewrites_fused_add_rmsnorm_graph():
    model = TinyFusedAddRMSNormQuantFP8().cuda()
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    residual = graph.placeholder("residual")
    norm_weight = graph.get_attr("norm_weight")
    weight_fp8 = graph.get_attr("weight_fp8")
    input_scale = graph.get_attr("input_scale")
    weight_scale = graph.get_attr("weight_scale")

    fused = graph.call_function(
        flashinfer_fused_add_rms_norm, args=(x, residual, norm_weight, model.eps)
    )
    norm_out = graph.call_function(operator.getitem, args=(fused, 0))
    add_out = graph.call_function(operator.getitem, args=(fused, 1))
    gemm_out = graph.call_function(
        torch.ops.auto_deploy.trtllm_quant_fp8_linear.default,
        args=(norm_out, weight_fp8, None, input_scale, weight_scale),
    )
    graph.output((gemm_out, add_out))
    norm_out.meta["val"] = torch.empty((1, 1), dtype=torch.bfloat16)

    gm = torch.fx.GraphModule(model, graph)

    assert any(
        n.op == "call_function" and n.target is flashinfer_fused_add_rms_norm
        for n in gm.graph.nodes
    )
    assert any(is_op(n, torch.ops.auto_deploy.trtllm_quant_fp8_linear) for n in gm.graph.nodes)

    transform = FuseRMSNormQuantFP8(TransformConfig(stage="post_load_fusion"))
    gm, info = transform._apply(gm, None, None, None)

    assert info.num_matches == 1
    assert any(
        is_op(n, torch.ops.auto_deploy.triton_fused_add_rms_norm_quant_fp8) for n in gm.graph.nodes
    )
    assert any(is_op(n, torch.ops.auto_deploy.trtllm_fp8_prequant_linear) for n in gm.graph.nodes)
    assert not any(is_op(n, torch.ops.auto_deploy.trtllm_quant_fp8_linear) for n in gm.graph.nodes)


@pytest.mark.skipif(not fp8_compatible(), reason="Requires fp8 support")
def test_fuse_rmsnorm_quant_fp8_rewrites_torch_rmsnorm_add_graph():
    model = TinyFusedAddRMSNormQuantFP8().cuda()
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    residual = graph.placeholder("residual")
    norm_weight = graph.get_attr("norm_weight")
    weight_fp8 = graph.get_attr("weight_fp8")
    input_scale = graph.get_attr("input_scale")
    weight_scale = graph.get_attr("weight_scale")

    added = graph.call_function(torch.ops.aten.add.Tensor, args=(x, residual))
    casted = graph.call_function(
        torch.ops.aten.to.dtype,
        args=(added, torch.bfloat16),
        kwargs={"non_blocking": False, "copy": False, "memory_format": None},
    )
    norm_out = graph.call_function(
        torch.ops.auto_deploy.torch_rmsnorm.default,
        args=(casted, norm_weight, model.eps),
    )
    gemm_out = graph.call_function(
        torch.ops.auto_deploy.trtllm_quant_fp8_linear.default,
        args=(norm_out, weight_fp8, None, input_scale, weight_scale),
    )
    graph.output(gemm_out)

    norm_out.meta["val"] = torch.empty((1, 1), dtype=torch.bfloat16)

    gm = torch.fx.GraphModule(model, graph)
    transform = FuseRMSNormQuantFP8(TransformConfig(stage="post_load_fusion"))
    gm, info = transform._apply(gm, None, None, None)

    assert info.num_matches == 1
    assert any(
        is_op(n, torch.ops.auto_deploy.triton_fused_add_rms_norm_quant_fp8) for n in gm.graph.nodes
    )
    assert any(is_op(n, torch.ops.auto_deploy.trtllm_fp8_prequant_linear) for n in gm.graph.nodes)
    assert not any(is_op(n, torch.ops.auto_deploy.torch_rmsnorm) for n in gm.graph.nodes)


@pytest.mark.skipif(not fp8_compatible(), reason="Requires fp8 support")
def test_fuse_rmsnorm_quant_fp8_shares_fused_quant_across_multiple_linears():
    model = TinyRMSNormQuantFP8().cuda()
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    norm_weight = graph.get_attr("norm_weight")
    weight_fp8 = graph.get_attr("weight_fp8")
    input_scale = graph.get_attr("input_scale")
    weight_scale = graph.get_attr("weight_scale")

    norm_out = graph.call_function(
        torch.ops.auto_deploy.flashinfer_rms_norm.default,
        args=(x, norm_weight, model.eps),
    )
    gemm_out_1 = graph.call_function(
        torch.ops.auto_deploy.trtllm_quant_fp8_linear.default,
        args=(norm_out, weight_fp8, None, input_scale, weight_scale),
    )
    gemm_out_2 = graph.call_function(
        torch.ops.auto_deploy.torch_quant_fp8_linear.default,
        args=(norm_out, weight_fp8, None, input_scale, weight_scale),
    )
    graph.output((gemm_out_1, gemm_out_2))

    norm_out.meta["val"] = torch.empty((1, 1), dtype=torch.bfloat16)

    gm = torch.fx.GraphModule(model, graph)
    transform = FuseRMSNormQuantFP8(TransformConfig(stage="post_load_fusion"))
    gm, info = transform._apply(gm, None, None, None)

    assert info.num_matches == 2
    assert (
        sum(is_op(n, torch.ops.auto_deploy.triton_rms_norm_quant_fp8) for n in gm.graph.nodes) == 1
    )
    assert (
        sum(is_op(n, torch.ops.auto_deploy.trtllm_fp8_prequant_linear) for n in gm.graph.nodes) == 2
    )
    assert not any(is_op(n, torch.ops.auto_deploy.trtllm_quant_fp8_linear) for n in gm.graph.nodes)
    assert not any(is_op(n, torch.ops.auto_deploy.torch_quant_fp8_linear) for n in gm.graph.nodes)


@pytest.mark.skipif(not fp8_compatible(), reason="Requires fp8 support")
def test_fuse_rmsnorm_quant_fp8_rewrites_through_post_norm_reshape():
    model = TinyRMSNormQuantFP8().cuda()
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    norm_weight = graph.get_attr("norm_weight")
    weight_fp8 = graph.get_attr("weight_fp8")
    input_scale = graph.get_attr("input_scale")
    weight_scale = graph.get_attr("weight_scale")

    norm_out = graph.call_function(
        torch.ops.auto_deploy.flashinfer_rms_norm.default,
        args=(x, norm_weight, model.eps),
    )
    reshaped = graph.call_function(torch.ops.aten.reshape.default, args=(norm_out, [2, 128]))
    gemm_out = graph.call_function(
        torch.ops.auto_deploy.trtllm_quant_fp8_linear.default,
        args=(reshaped, weight_fp8, None, input_scale, weight_scale),
    )
    graph.output(gemm_out)

    norm_out.meta["val"] = torch.empty((2, 128), dtype=torch.bfloat16)
    reshaped.meta["val"] = torch.empty((2, 128), dtype=torch.bfloat16)

    gm = torch.fx.GraphModule(model, graph)
    transform = FuseRMSNormQuantFP8(TransformConfig(stage="post_load_fusion"))
    gm, info = transform._apply(gm, None, None, None)

    assert info.num_matches == 1
    assert any(is_op(n, torch.ops.auto_deploy.triton_rms_norm_quant_fp8) for n in gm.graph.nodes)
    assert any(is_op(n, torch.ops.aten.reshape.default) for n in gm.graph.nodes)
    assert any(is_op(n, torch.ops.auto_deploy.trtllm_fp8_prequant_linear) for n in gm.graph.nodes)
    assert not any(is_op(n, torch.ops.auto_deploy.trtllm_quant_fp8_linear) for n in gm.graph.nodes)


def test_get_out_dtype_str_returns_none_when_norm_meta_missing():
    graph = torch.fx.Graph()
    x = graph.placeholder("x")
    w = graph.placeholder("w")
    norm = graph.call_function(torch.ops.auto_deploy.flashinfer_rms_norm, args=(x, w, 1e-5))
    graph.output(norm)

    # Missing output metadata should skip fusion rather than guessing dtype.
    norm.meta.pop("val", None)

    assert _get_out_dtype_str(norm) is None
