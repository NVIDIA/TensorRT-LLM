import pytest
import torch
import torch.nn as nn
from _torch_test_utils import fp4_compatible, trtllm_ops_available

import tensorrt_llm._torch.auto_deploy.custom_ops  # noqa: F401
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
    def __init__(self, in_features: int = 64, out_features: int = 32):
        super().__init__()
        assert in_features % 16 == 0, "NVFP4 requires K % 16 == 0"
        device = torch.device("cuda")

        weight = torch.rand(out_features, in_features, dtype=torch.half, device=device)
        bias = torch.rand(out_features, dtype=torch.half, device=device)

        with torch.no_grad():
            input_scale = fp4_global_scale(
                torch.rand(1, in_features, dtype=torch.half, device=device)
            )
            weight_scale_2 = fp4_global_scale(weight)
            weight_fp4, weight_scale = torch.ops.trtllm.fp4_quantize(
                weight, weight_scale_2, 16, False
            )
            alpha = (1.0 / (input_scale * weight_scale_2)).to(torch.float32)

        self.register_buffer("weight_fp4", weight_fp4)
        self.register_buffer("bias", bias)
        self.register_buffer("input_scale", input_scale.to(torch.float32))
        self.register_buffer("weight_scale", weight_scale)
        self.register_buffer("alpha", alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        relu_out = torch.nn.functional.relu(x)
        relu2_out = relu_out * relu_out
        return torch.ops.auto_deploy.torch_quant_nvfp4_linear(
            relu2_out,
            self.weight_fp4,
            self.bias,
            self.input_scale,
            self.weight_scale,
            self.alpha,
        )


def _count_op(gm, op) -> int:
    return sum(1 for n in gm.graph.nodes if is_op(n, op))


@pytest.mark.skipif(_skip_condition, reason=_skip_reason)
def test_fuse_relu2_quant_nvfp4_rewrite_and_numerics():
    torch.manual_seed(0)
    model = TinyRelu2NVFP4Linear().to("cuda")
    x = torch.rand(3, 64, dtype=torch.float16, device="cuda")

    gm = torch_export_to_gm(model, args=(x,), clone=True)
    gm_transformed = InferenceOptimizer(
        None,
        {
            "fuse_relu2_quant_nvfp4": {
                "stage": "post_load_fusion",
            },
        },
    )(None, gm).to("cuda")

    assert _count_op(gm_transformed, torch.ops.auto_deploy.torch_quant_nvfp4_linear) == 0
    assert (
        _count_op(gm_transformed, torch.ops.auto_deploy.trtllm_fused_relu2_quant_nvfp4.default) == 1
    )
    assert (
        _count_op(gm_transformed, torch.ops.auto_deploy.trtllm_nvfp4_prequant_linear.default) == 1
    )

    y_ref = model(x)
    y_new = gm_transformed(x)
    torch.testing.assert_close(y_new, y_ref, atol=1e-2, rtol=5e-2)


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

    gm = torch_export_to_gm(model, args=(x,), clone=True)
    gm_transformed = InferenceOptimizer(
        None,
        {
            "fuse_relu2_quant_nvfp4": {
                "stage": "post_load_fusion",
            },
        },
    )(None, gm)

    assert _count_op(gm_transformed, torch.ops.auto_deploy.torch_quant_nvfp4_linear) == 1
    assert (
        _count_op(gm_transformed, torch.ops.auto_deploy.trtllm_fused_relu2_quant_nvfp4.default) == 0
    )
    assert (
        _count_op(gm_transformed, torch.ops.auto_deploy.trtllm_nvfp4_prequant_linear.default) == 0
    )
