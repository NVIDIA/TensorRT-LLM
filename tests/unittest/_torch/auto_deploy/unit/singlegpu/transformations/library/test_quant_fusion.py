# test_quant_fusion.py
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn
from _graph_test_helpers import run_test_transformed_gm
from _torch_test_utils import fp4_compatible, fp8_compatible, trtllm_ops_available

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
from tensorrt_llm._torch.auto_deploy.custom_ops.normalization.flashinfer_fused_add_rms_norm import (
    flashinfer_fused_add_rms_norm,
)
from tensorrt_llm._torch.auto_deploy.custom_ops.normalization.trtllm_fused_add_rms_norm_quant_nvfp4 import (
    trtllm_fused_add_rms_norm_quant_nvfp4,
)
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.interface import TransformConfig
from tensorrt_llm._torch.auto_deploy.transform.library.fuse_rmsnorm_quant_fp8 import (
    FuseRMSNormQuantFP8,
)
from tensorrt_llm._torch.auto_deploy.transform.library.fuse_rmsnorm_quant_nvfp4 import (
    FuseRMSNormQuantNVFP4,
)
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op
from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fp4_global_scale, fp8_scale

# Keep wrapper as single node in FX trace so fused-add norm+quant transform can match it.
torch.fx.wrap("flashinfer_fused_add_rms_norm")


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


@pytest.mark.parametrize(
    "quant_type",
    [
        pytest.param(
            "fp8", marks=pytest.mark.skipif(not fp8_compatible(), reason="Requires fp8 support")
        ),
        pytest.param(
            "fp4",
            marks=pytest.mark.skipif(
                not (fp4_compatible() and trtllm_ops_available()),
                reason="Requires NVFP4 and TRT-LLM ops",
            ),
        ),
    ],
)
@pytest.mark.parametrize("use_bias", [True, False])
def test_fuse_quant_rewrites_linear(quant_type, use_bias):
    """Fuse reference FP8 or FP4 linear to backend fused op; parametrized by quant_type and use_bias."""
    torch.manual_seed(0)
    if quant_type == "fp8":
        model = TinyFP8Ref(use_bias=use_bias).to("cuda")
        x = torch.rand(3, 16, dtype=torch.float16, device="cuda")
        opt_config = {"fuse_fp8_linear": {"stage": "post_load_fusion", "backend": "torch"}}
        has_fused = _has_fused_linear_fp8
    else:
        model = TinyFP4Ref(use_bias=use_bias).to("cuda")
        x = torch.rand(3, 64, dtype=torch.float16, device="cuda")
        opt_config = {"fuse_nvfp4_linear": {"stage": "post_load_fusion", "backend": "trtllm"}}
        has_fused = _has_fused_linear_fp4

    gm = torch_export_to_gm(model, args=(x,), clone=True)
    gm_transformed = InferenceOptimizer(None, opt_config)(None, gm)
    gm_transformed.to("cuda")

    run_test_transformed_gm(
        model,
        x,
        gm_transformed,
        has_fused,
        lambda n: n,
        0.1,  # atol
        0.05,  # rtol
        False,  # test_load_hook
        False,  # strict_loading
        None,  # dynamic_shapes
        False,  # skip_output_assert
    )


# ---------------------------------------------------------------------------
# Common helpers for norm+quant fusion tests (FP8 and NVFP4)
# ---------------------------------------------------------------------------


def _apply_norm_quant_transform(gm, transform_class):
    """Run transform, return (gm, info)."""
    config = TransformConfig(stage="post_load_fusion")
    transform = transform_class(config)
    return transform._apply(gm, MagicMock(), MagicMock(), MagicMock())


def _assert_numerics(ref_out, fused_out, extra_norm_consumer):
    """Compare ref vs fused output; extra_norm_consumer means tuple (out, norm_or_add_out)."""
    if extra_norm_consumer:
        assert len(fused_out) == 2
        torch.testing.assert_close(ref_out[0], fused_out[0], atol=0, rtol=0)
        torch.testing.assert_close(ref_out[1], fused_out[1], atol=0, rtol=0)
    else:
        torch.testing.assert_close(ref_out, fused_out, atol=0, rtol=0)


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


class _RMSNormFP8FusedAddModel(nn.Module):
    """Model: getitem(flashinfer_fused_add_rms_norm(x, residual, weight, eps), 0) -> trtllm_quant_fp8_linear.

    Forward: (x, residual) -> (combined, add_out). Used to test fused add+norm FP8 path.
    """

    def __init__(self, in_features=128, out_features=256, num_linears=1):
        super().__init__()
        self.num_linears = num_linears
        self.norm_weight = nn.Parameter(torch.randn(in_features, dtype=torch.bfloat16))
        self.input_scale = nn.Buffer(torch.tensor(1.0, dtype=torch.float32))
        with torch.no_grad():
            for i in range(num_linears):
                w_scale = torch.tensor(1.0, dtype=torch.float32)
                w_bf16 = torch.randn(out_features, in_features, dtype=torch.bfloat16)
                w_fp8 = (w_bf16 / w_scale).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
                self.register_parameter(f"w_fp8_{i}", nn.Parameter(w_fp8))
                self.register_buffer(f"w_scale_{i}", w_scale)

    def forward(self, x, residual):
        norm_out, add_out = flashinfer_fused_add_rms_norm(x, residual, self.norm_weight, 1e-5)
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
        return combined, add_out


def _is_flashinfer_fused_add_rms_norm_node(n):
    """True if n is a call to flashinfer_fused_add_rms_norm (by identity or name)."""
    if n.op != "call_function":
        return False
    return n.target is flashinfer_fused_add_rms_norm or (
        getattr(n.target, "__name__", None) == "flashinfer_fused_add_rms_norm"
    )


def _has_fused_rmsnorm_quant_fp8(gm, fused_add=False):
    """Check that the graph contains fused ops and no unfused ones."""
    if fused_add:
        has_fused_norm = any(
            is_op(n, torch.ops.auto_deploy.triton_fused_add_rms_norm_quant_fp8)
            for n in gm.graph.nodes
        )
    else:
        has_fused_norm = any(
            is_op(n, torch.ops.auto_deploy.triton_rms_norm_quant_fp8) for n in gm.graph.nodes
        )
    has_fp8_gemm = any(is_op(n, torch.ops.auto_deploy.trtllm_fp8_gemm) for n in gm.graph.nodes)
    no_old_norm = not any(
        is_op(n, torch.ops.auto_deploy.flashinfer_rms_norm) for n in gm.graph.nodes
    )
    no_old_fused = not any(_is_flashinfer_fused_add_rms_norm_node(n) for n in gm.graph.nodes)
    no_old_fp8_linear = not any(
        is_op(n, torch.ops.auto_deploy.trtllm_quant_fp8_linear) for n in gm.graph.nodes
    )
    return has_fused_norm and has_fp8_gemm and no_old_norm and no_old_fp8_linear and no_old_fused


@pytest.mark.parametrize("num_linears", [1, 3])
@pytest.mark.parametrize(
    "fused_add,extra_norm_consumer",
    [(False, True), (False, False), (True, True)],
)
@pytest.mark.skipif(
    not (fp8_compatible() and trtllm_ops_available()),
    reason="Requires FP8 support and TRT-LLM ops",
)
def test_fuse_rmsnorm_quant_fp8(num_linears, fused_add, extra_norm_consumer):
    """FuseRMSNormQuantFP8: direct RMSNorm+FP8 or fused add+RMSNorm+FP8; graph fused, numerics match."""
    torch.manual_seed(0)
    K, N = 128, 256
    if fused_add:
        model = _RMSNormFP8FusedAddModel(
            in_features=K,
            out_features=N,
            num_linears=num_linears,
        ).cuda()
        x = torch.randn(4, K, device="cuda", dtype=torch.bfloat16)
        residual = torch.randn(4, K, device="cuda", dtype=torch.bfloat16)
        x_ref = x.clone()
        residual_ref = residual.clone()
        ref_out = model(x_ref, residual_ref)
    else:
        model = _RMSNormFP8LinearModel(
            in_features=K,
            out_features=N,
            num_linears=num_linears,
            extra_norm_consumer=extra_norm_consumer,
        ).cuda()
        x = torch.randn(4, K, device="cuda", dtype=torch.bfloat16)
        ref_out = model(x)

    gm = torch.fx.symbolic_trace(model)
    if fused_add:
        assert any(_is_flashinfer_fused_add_rms_norm_node(n) for n in gm.graph.nodes), (
            "Precondition: graph should contain flashinfer_fused_add_rms_norm"
        )
    else:
        assert any(is_op(n, torch.ops.auto_deploy.flashinfer_rms_norm) for n in gm.graph.nodes), (
            "Precondition: graph should contain flashinfer_rms_norm before transform"
        )
    assert any(is_op(n, torch.ops.auto_deploy.trtllm_quant_fp8_linear) for n in gm.graph.nodes), (
        "Precondition: graph should contain trtllm_quant_fp8_linear before transform"
    )

    gm, info = _apply_norm_quant_transform(gm, FuseRMSNormQuantFP8)
    assert info.num_matches == num_linears, (
        f"Expected {num_linears} matches, got {info.num_matches}"
    )
    assert not info.skipped
    assert _has_fused_rmsnorm_quant_fp8(gm, fused_add=fused_add), (
        "Graph should contain fused FP8 norm+quant ops and no unfused norm/linear"
    )

    if fused_add:
        fused_out = gm(x, residual)
        _assert_numerics(ref_out, fused_out, extra_norm_consumer=True)
    else:
        fused_out = gm(x)
        _assert_numerics(ref_out, fused_out, extra_norm_consumer)


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


class _RMSNormNVFP4FusedAddModel(nn.Module):
    """Model: getitem(flashinfer_fused_add_rms_norm(x, residual, weight, eps), 0) -> torch_quant_nvfp4_linear.

    Forward: (x, residual) -> (combined, add_out). Used to test fused add+norm NVFP4 path.
    """

    def __init__(self, in_features=2048, out_features=2048, num_linears=1):
        super().__init__()
        device = torch.device("cuda")
        self.num_linears = num_linears
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

    def forward(self, x, residual):
        norm_out, add_out = flashinfer_fused_add_rms_norm(x, residual, self.norm_weight, 1e-5)
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
        return combined, add_out


def _has_fused_rmsnorm_quant_nvfp4(gm, fused_add=False):
    """Check that the graph contains fused NVFP4 ops and no unfused ones."""
    if fused_add:
        has_fused_norm = any(
            is_op(n, torch.ops.auto_deploy.trtllm_fused_add_rms_norm_quant_nvfp4)
            for n in gm.graph.nodes
        )
    else:
        has_fused_norm = any(
            is_op(n, torch.ops.auto_deploy.trtllm_rms_norm_quant_nvfp4) for n in gm.graph.nodes
        )
    has_nvfp4_gemm = any(is_op(n, torch.ops.auto_deploy.trtllm_nvfp4_gemm) for n in gm.graph.nodes)
    no_old_norm = not any(
        is_op(n, torch.ops.auto_deploy.flashinfer_rms_norm) for n in gm.graph.nodes
    )
    no_old_fused = not any(_is_flashinfer_fused_add_rms_norm_node(n) for n in gm.graph.nodes)
    no_old_nvfp4_linear = not any(
        is_op(n, torch.ops.auto_deploy.torch_quant_nvfp4_linear) for n in gm.graph.nodes
    )
    return (
        has_fused_norm and has_nvfp4_gemm and no_old_norm and no_old_nvfp4_linear and no_old_fused
    )


@pytest.mark.parametrize("num_linears", [1, 3])
@pytest.mark.parametrize(
    "fused_add,extra_norm_consumer",
    [(False, True), (False, False), (True, True)],
)
@pytest.mark.skipif(
    not (fp4_compatible() and trtllm_ops_available()),
    reason="Requires FP4 support and TRT-LLM ops",
)
def test_fuse_rmsnorm_quant_nvfp4(num_linears, fused_add, extra_norm_consumer):
    """FuseRMSNormQuantNVFP4: direct RMSNorm+linear or fused add+RMSNorm+linear; graph fused, numerics match."""
    torch.manual_seed(0)
    K, N = 2048, 2048
    if fused_add:
        model = _RMSNormNVFP4FusedAddModel(
            in_features=K,
            out_features=N,
            num_linears=num_linears,
        ).cuda()
        x = torch.randn(4, K, device="cuda", dtype=torch.bfloat16)
        residual = torch.randn(4, K, device="cuda", dtype=torch.bfloat16)
        x_ref = x.clone()
        residual_ref = residual.clone()
        ref_out = model(x_ref, residual_ref)
    else:
        model = _RMSNormNVFP4LinearModel(
            in_features=K,
            out_features=N,
            num_linears=num_linears,
            extra_norm_consumer=extra_norm_consumer,
        ).cuda()
        x = torch.randn(4, K, device="cuda", dtype=torch.bfloat16)
        ref_out = model(x)

    gm = torch.fx.symbolic_trace(model)
    if fused_add:
        assert any(_is_flashinfer_fused_add_rms_norm_node(n) for n in gm.graph.nodes), (
            "Precondition: graph should contain flashinfer_fused_add_rms_norm"
        )
    else:
        assert any(is_op(n, torch.ops.auto_deploy.flashinfer_rms_norm) for n in gm.graph.nodes), (
            "Precondition: graph should contain flashinfer_rms_norm before transform"
        )
    assert any(is_op(n, torch.ops.auto_deploy.torch_quant_nvfp4_linear) for n in gm.graph.nodes), (
        "Precondition: graph should contain torch_quant_nvfp4_linear before transform"
    )

    gm, info = _apply_norm_quant_transform(gm, FuseRMSNormQuantNVFP4)
    assert info.num_matches == num_linears
    assert not info.skipped
    assert _has_fused_rmsnorm_quant_nvfp4(gm, fused_add=fused_add), (
        "Graph should contain fused NVFP4 norm+quant ops and no unfused norm/linear"
    )

    if fused_add:
        fused_out = gm(x, residual)
        _assert_numerics(ref_out, fused_out, extra_norm_consumer=True)
    else:
        fused_out = gm(x)
        _assert_numerics(ref_out, fused_out, extra_norm_consumer)


@pytest.mark.skipif(
    not (fp4_compatible() and trtllm_ops_available()),
    reason="Requires FP4 support and TRT-LLM ops",
)
def test_debug_nvfp4_fused_add_intermediates():
    """Compare add_out and norm_out (bf16) from FlashInfer vs C++ fused kernel to isolate divergence.

    Reference: flashinfer_fused_add_rms_norm (in-place) then norm_out = first return, add_out = second.
    Fused: trtllm_fused_add_rms_norm_quant_nvfp4(x, residual, weight, eps, scale) -> (bf16_norm, _, _, add_out).
    If add_out or bf16_norm differ, the bug is in the C++ kernel. If they match, the bug is in quant/gemm wiring.
    """
    torch.manual_seed(0)
    K, N = 2048, 2048
    model = _RMSNormNVFP4FusedAddModel(
        in_features=K,
        out_features=N,
        num_linears=1,
    ).cuda()
    x = torch.randn(4, K, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn(4, K, device="cuda", dtype=torch.bfloat16)
    weight = model.norm_weight
    eps = 1e-5
    scale = model.input_scale

    # Reference: FlashInfer in-place; returns (norm_out, add_out) = (x_after_norm, residual_after_add)
    x_ref = x.clone()
    res_ref = residual.clone()
    norm_out_ref, add_out_ref = flashinfer_fused_add_rms_norm(x_ref, res_ref, weight, eps)

    # Fused: C++ kernel (no in-place)
    bf16_norm_fused, fp4_u8_fused, sf_out_fused, add_out_fused = (
        trtllm_fused_add_rms_norm_quant_nvfp4(x, residual, weight, eps, scale)
    )

    # Compare add_out (pre-norm sum: x + residual)
    try:
        torch.testing.assert_close(add_out_ref, add_out_fused, atol=0, rtol=0)
        print("GAGAM debug_nvfp4_fused_add: add_out REF vs FUSED match")
    except AssertionError:
        diff = (add_out_ref - add_out_fused).abs()
        print(
            "GAGAM debug_nvfp4_fused_add: add_out MISMATCH max_abs_diff=%s mean_diff=%s"
            % (diff.max().item(), diff.float().mean().item())
        )
        raise

    # Compare norm_out (bf16 normalized)
    try:
        torch.testing.assert_close(norm_out_ref, bf16_norm_fused, atol=0, rtol=0)
        print("GAGAM debug_nvfp4_fused_add: norm_out (bf16) REF vs FUSED match")
    except AssertionError:
        diff = (norm_out_ref - bf16_norm_fused).abs()
        print(
            "GAGAM debug_nvfp4_fused_add: norm_out MISMATCH max_abs_diff=%s mean_diff=%s"
            % (diff.max().item(), diff.float().mean().item())
        )
        raise

    # If we get here, add and norm match; compare full pipeline (ref linear on norm_out_ref vs gemm on fused fp4)
    ref_linear_out = torch.ops.auto_deploy.torch_quant_nvfp4_linear(
        norm_out_ref,
        model.w_fp4_0,
        bias=None,
        input_scale=model.input_scale,
        weight_scale=model.w_scale_0,
        alpha=model.alpha_0,
    )
    fused_gemm_out = torch.ops.auto_deploy.trtllm_nvfp4_gemm.default(
        fp4_u8_fused,
        model.w_fp4_0,
        sf_out_fused,
        model.w_scale_0,
        model.alpha_0,
        bias=None,
        out_dtype="bfloat16",
    )
    try:
        torch.testing.assert_close(ref_linear_out, fused_gemm_out, atol=0, rtol=0)
        print("GAGAM debug_nvfp4_fused_add: linear_out (ref linear vs fused gemm) match")
    except AssertionError:
        diff = (ref_linear_out - fused_gemm_out).abs()
        print(
            "GAGAM debug_nvfp4_fused_add: linear_out MISMATCH max_abs_diff=%s mean_diff=%s"
            % (diff.max().item(), diff.float().mean().item())
        )
        raise
