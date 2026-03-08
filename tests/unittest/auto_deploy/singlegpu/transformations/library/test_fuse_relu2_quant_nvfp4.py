import pytest
import torch
import torch.nn as nn
from _torch_test_utils import fp4_compatible, trtllm_ops_available

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
import tensorrt_llm._torch.auto_deploy.transform.library  # noqa: F401
from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op
from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import fp4_global_scale


def _fused_relu2_quantize_available() -> bool:
    return hasattr(torch.ops, "trtllm") and hasattr(torch.ops.trtllm, "fused_relu2_quantize")


_skip_condition = not (
    fp4_compatible() and trtllm_ops_available() and _fused_relu2_quantize_available()
)
_skip_reason = "Requires NVFP4 support and trtllm.fused_relu2_quantize kernel"


class TinyRelu2NVFP4Linear(nn.Module):
    def __init__(
        self,
        in_features: int = 64,
        out_features: int = 32,
        relu2_impl: str = "mul",
        use_bias: bool = True,
        input_dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        assert in_features % 16 == 0, "NVFP4 requires K % 16 == 0"
        device = torch.device("cuda")

        weight = torch.rand(out_features, in_features, dtype=torch.float16, device=device)
        bias = torch.rand(out_features, dtype=input_dtype, device=device) if use_bias else None

        with torch.no_grad():
            input_scale = fp4_global_scale(
                torch.rand(1, in_features, dtype=input_dtype, device=device)
            )
            weight_scale_2 = fp4_global_scale(weight)
            weight_fp4, weight_scale = torch.ops.trtllm.fp4_quantize(
                weight, weight_scale_2, 16, False
            )
            alpha = (1.0 / (input_scale * weight_scale_2)).to(torch.float32)

        self.register_buffer("weight_fp4", weight_fp4)
        if bias is not None:
            self.register_buffer("bias", bias)
        else:
            self.bias = None
        self.register_buffer("input_scale", input_scale.to(torch.float32))
        self.register_buffer("weight_scale", weight_scale)
        self.register_buffer("alpha", alpha)
        self.relu2_impl = relu2_impl

    def _apply_relu2(self, x: torch.Tensor) -> torch.Tensor:
        relu_out = torch.nn.functional.relu(x)
        if self.relu2_impl == "square":
            return torch.square(relu_out)
        if self.relu2_impl == "pow":
            return torch.pow(relu_out, 2)
        if self.relu2_impl == "mul":
            return relu_out * relu_out
        raise ValueError(f"Unsupported relu2_impl: {self.relu2_impl}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        relu2_out = self._apply_relu2(x)
        return torch.ops.auto_deploy.torch_quant_nvfp4_linear(
            relu2_out,
            self.weight_fp4,
            self.bias,
            self.input_scale,
            self.weight_scale,
            self.alpha,
        )


class SharedRelu2UserModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.impl = TinyRelu2NVFP4Linear(relu2_impl="mul")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        relu_out = torch.nn.functional.relu(x)
        relu2_out = relu_out * relu_out
        linear_out = torch.ops.auto_deploy.torch_quant_nvfp4_linear(
            relu2_out,
            self.impl.weight_fp4,
            self.impl.bias,
            self.impl.input_scale,
            self.impl.weight_scale,
            self.impl.alpha,
        )
        return linear_out + relu2_out[:, : linear_out.shape[-1]]


def _count_op(gm, op) -> int:
    return sum(1 for n in gm.graph.nodes if is_op(n, op))


def _run_fuse_relu2_quant_nvfp4(model: nn.Module, x: torch.Tensor):
    gm = torch_export_to_gm(model, args=(x,), clone=True)
    gm_transformed = InferenceOptimizer(
        None,
        {
            "fuse_relu2_quant_nvfp4": {
                "stage": "post_load_fusion",
            },
        },
    )(None, gm)
    return gm_transformed.to("cuda")


def _assert_fused(gm_transformed) -> None:
    assert _count_op(gm_transformed, torch.ops.auto_deploy.torch_quant_nvfp4_linear) == 0
    assert (
        _count_op(gm_transformed, torch.ops.auto_deploy.trtllm_fused_relu2_quant_nvfp4.default) == 1
    )
    assert (
        _count_op(gm_transformed, torch.ops.auto_deploy.trtllm_nvfp4_prequant_linear.default) == 1
    )


@pytest.mark.skipif(_skip_condition, reason=_skip_reason)
@pytest.mark.parametrize("relu2_impl", ["square", "pow", "mul"])
@pytest.mark.parametrize("use_bias", [False, True])
def test_fuse_relu2_quant_nvfp4_rewrite_and_numerics(relu2_impl: str, use_bias: bool):
    torch.manual_seed(0)
    model = TinyRelu2NVFP4Linear(relu2_impl=relu2_impl, use_bias=use_bias).to("cuda")
    x = torch.rand(3, 64, dtype=torch.float16, device="cuda")

    gm_transformed = _run_fuse_relu2_quant_nvfp4(model, x)
    _assert_fused(gm_transformed)

    y_ref = model(x)
    y_new = gm_transformed(x)
    torch.testing.assert_close(y_new, y_ref, atol=1e-2, rtol=5e-2)


@pytest.mark.skipif(_skip_condition, reason=_skip_reason)
@pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16])
def test_fuse_relu2_quant_nvfp4_preserves_output_dtype(input_dtype: torch.dtype):
    torch.manual_seed(0)
    model = TinyRelu2NVFP4Linear(
        relu2_impl="mul",
        use_bias=True,
        input_dtype=input_dtype,
    ).to("cuda")
    x = torch.rand(3, 64, dtype=input_dtype, device="cuda")

    gm_transformed = _run_fuse_relu2_quant_nvfp4(model, x)
    _assert_fused(gm_transformed)

    y_ref = model(x)
    y_new = gm_transformed(x)
    assert y_new.dtype == input_dtype
    torch.testing.assert_close(y_new, y_ref, atol=2e-2, rtol=8e-2)


@pytest.mark.skipif(_skip_condition, reason=_skip_reason)
def test_fuse_relu2_quant_nvfp4_does_not_match_non_relu2():
    class NonRelu2Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.impl = TinyRelu2NVFP4Linear()

        def forward(self, x):
            # No relu2 chain (relu->square/mul), should not be fused.
            x = torch.nn.functional.gelu(x)
            return torch.ops.auto_deploy.torch_quant_nvfp4_linear(
                x,
                self.impl.weight_fp4,
                self.impl.bias,
                self.impl.input_scale,
                self.impl.weight_scale,
                self.impl.alpha,
            )

    torch.manual_seed(0)
    model = NonRelu2Model().to("cuda")
    x = torch.rand(3, 64, dtype=torch.float16, device="cuda")

    gm_transformed = _run_fuse_relu2_quant_nvfp4(model, x)

    assert _count_op(gm_transformed, torch.ops.auto_deploy.torch_quant_nvfp4_linear) == 1
    assert (
        _count_op(gm_transformed, torch.ops.auto_deploy.trtllm_fused_relu2_quant_nvfp4.default) == 0
    )
    assert (
        _count_op(gm_transformed, torch.ops.auto_deploy.trtllm_nvfp4_prequant_linear.default) == 0
    )


@pytest.mark.skipif(_skip_condition, reason=_skip_reason)
def test_fuse_relu2_quant_nvfp4_does_not_match_shared_relu2_users():
    torch.manual_seed(0)
    model = SharedRelu2UserModel().to("cuda")
    x = torch.rand(3, 64, dtype=torch.float16, device="cuda")

    gm_transformed = _run_fuse_relu2_quant_nvfp4(model, x)

    assert _count_op(gm_transformed, torch.ops.auto_deploy.torch_quant_nvfp4_linear) == 1
    assert (
        _count_op(gm_transformed, torch.ops.auto_deploy.trtllm_fused_relu2_quant_nvfp4.default) == 0
    )
    assert (
        _count_op(gm_transformed, torch.ops.auto_deploy.trtllm_nvfp4_prequant_linear.default) == 0
    )
