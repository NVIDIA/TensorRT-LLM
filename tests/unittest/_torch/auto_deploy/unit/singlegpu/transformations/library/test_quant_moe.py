import pytest
import torch
from _graph_test_helpers import run_test
from _model_test_utils import MoEOpModel
from _torch_test_utils import fp4_compatible, fp8_compatible, trtllm_ops_available

from tensorrt_llm._torch.auto_deploy.transformations.library import quantize_moe
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op


@pytest.mark.parametrize(
    "quant_algo, expected_op",
    [
        pytest.param(
            "FP8",
            torch.ops.auto_deploy.torch_quant_fp8_moe,
            marks=pytest.mark.skipif(not fp8_compatible(), reason="Requires FP8"),
        ),
        pytest.param(
            "NVFP4",
            torch.ops.auto_deploy.torch_quant_fp4_moe,
            marks=pytest.mark.skipif(
                not (fp4_compatible() and trtllm_ops_available()), reason="Requires FP4 + TRTLLM"
            ),
        ),
    ],
)
def test_quantize_moe_transformation(quant_algo, expected_op):
    device = "cuda"
    hidden_size = 64
    intermediate_size = 32
    num_experts = 3
    top_k = 2

    model = MoEOpModel(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        top_k=top_k,
    ).to(device=device, dtype=torch.bfloat16)

    x = model.get_input(device=device, dtype=torch.bfloat16)

    def _check_transformed_graph(gm):
        return any(is_op(n, expected_op) for n in gm.graph.nodes)

    def _expected_num_params(n):
        """
        Return expected parameter count after quantization.
        For FP4, weights are quantized to half-size (simulate 4-bit).
        """
        # gate: Linear(hidden_size, num_experts)
        gate_params = (hidden_size + 1) * num_experts  # with bias

        if quant_algo == "NVFP4":
            expert_params = num_experts * 3 * hidden_size * intermediate_size // 2
            # 3 weights per expert, of shape [hidden_size, intermediate_size] or
            # [intermediate_size, hidden_size], shape will be halved to store quantized uint8 weight
            return gate_params + expert_params
        else:
            return n

    quant_config = {"quant_algo": quant_algo}

    def _transform(gm, *args):
        return quantize_moe(gm, quant_config)

    _ = run_test(
        model=model,
        x=x,
        transform=_transform,
        check_transformed_graph=_check_transformed_graph,
        _get_expected_num_params=_expected_num_params,
        atol=0.5,
        rtol=0.5,
        test_load_hook=False,
        strict_loading=False,
    )
