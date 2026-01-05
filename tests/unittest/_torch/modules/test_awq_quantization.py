from unittest.mock import patch

import pytest
import torch
from transformers.configuration_utils import PretrainedConfig
from utils.util import skip_pre_blackwell

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe import DefaultMoeRoutingMethod, create_moe
from tensorrt_llm._torch.modules.fused_moe.configurable_moe import ConfigurableMoE
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


@skip_pre_blackwell  # NVFP4 AWQ features require Blackwell (SM100) or later
@pytest.mark.parametrize("has_scale", [True, False])
def test_linear_nvfp4_awq_pre_quant_scale(has_scale):
    """
    Test that Linear (NVFP4 mode) applies pre_quant_scale to input before quantization.

    This tests the logic in NVFP4LinearMethod.apply (around line 824-827):
        if module.pre_quant_scale is not None:
            assert input.dtype == module.pre_quant_scale.dtype
            input = input * module.pre_quant_scale
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Create a Linear module with NVFP4 quantization using actual initialization
    mapping = Mapping(world_size=1, rank=0, tp_size=1)
    quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)

    in_features = 128
    out_features = 256

    # Create actual Linear module
    linear = Linear(
        in_features=in_features,
        out_features=out_features,
        dtype=torch.bfloat16,
        mapping=mapping,
        quant_config=quant_config,
    ).cuda()

    # Set pre_quant_scale based on test parameter (skip weight init as it's quantized)
    if has_scale:
        scale = torch.full((in_features,), 0.5, dtype=torch.bfloat16, device="cuda")
        linear.pre_quant_scale = torch.nn.Parameter(scale, requires_grad=False)

    # Prepare input
    x = torch.ones(2, in_features, dtype=torch.bfloat16, device="cuda")

    # Mock torch.ops.trtllm.fp4_quantize to capture the input after scaling
    captured_input = None

    def mock_fp4_quantize(input_tensor, *args, **kwargs):
        nonlocal captured_input
        captured_input = input_tensor
        # Return dummy quantized output
        return (
            torch.zeros(
                input_tensor.shape[0], input_tensor.shape[1] // 2, dtype=torch.uint8, device="cuda"
            ),
            torch.ones(
                input_tensor.shape[0],
                input_tensor.shape[1] // 16,
                dtype=torch.float32,
                device="cuda",
            ),
        )

    # Also mock the GEMM to avoid execution errors
    # just return a dummy output since we are capturing the input before input quantization
    def mock_gemm(act_fp4, *args, **kwargs):
        batch_size = act_fp4.shape[0]
        return torch.zeros(batch_size, out_features, dtype=torch.bfloat16, device="cuda")

    with patch("torch.ops.trtllm.fp4_quantize", side_effect=mock_fp4_quantize, create=True):
        with patch("torch.ops.trtllm.nvfp4_gemm", side_effect=mock_gemm, create=True):
            linear(x)

    assert captured_input is not None, "fp4_quantize was not called"

    if has_scale:
        # Should be scaled
        expected = x * scale
        assert torch.allclose(captured_input, expected, rtol=1e-5, atol=1e-5), (
            "Expected scaled input"
        )
    else:
        # Should be original
        assert torch.equal(captured_input, x), "Expected original input"


@skip_pre_blackwell  # TRTLLMGenFusedMoE requires Blackwell (SM100) or later
@pytest.mark.parametrize("has_scale", [True, False])
def test_fused_moe_trtllm_gen_input_scaling(has_scale):
    """
    Test that TRTLLMGenFusedMoE applies fc31_act_scale to input x if present.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Setup
    num_experts = 8
    hidden_size = 128
    intermediate_size = 256
    top_k = 2
    seq_len = 4

    # Create pretrained_config with necessary parameters (following test_fused_moe.py pattern)
    pretrained_config = PretrainedConfig()
    pretrained_config.num_experts = num_experts
    pretrained_config.hidden_size = hidden_size
    pretrained_config.intermediate_size = intermediate_size
    pretrained_config.torch_dtype = torch.bfloat16

    mapping = Mapping(world_size=1, rank=0, tp_size=1)
    quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    model_config = ModelConfig(
        pretrained_config=pretrained_config,
        mapping=mapping,
        quant_config=quant_config,
        moe_backend="TRTLLM",
    )

    routing_method = DefaultMoeRoutingMethod(top_k=top_k)

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)

    # Create actual MoE module (parameters inferred from model_config.pretrained_config)
    moe = create_moe(
        routing_method=routing_method,
        reduce_results=False,
        model_config=model_config,
    ).cuda()

    # Set fc31_act_scale directly (simulating AWQ pre_quant_scale)
    if has_scale:
        scale = torch.full((hidden_size,), 0.5, dtype=torch.bfloat16, device="cuda")

        # For ConfigurableMoE, set fc31_act_scale on backend instead of the wrapper
        if isinstance(moe, ConfigurableMoE):
            moe.backend.fc31_act_scale = torch.nn.Parameter(scale, requires_grad=False)
        else:
            # For direct TRTLLMGenFusedMoE, set on moe itself
            moe.fc31_act_scale = torch.nn.Parameter(scale, requires_grad=False)

    # Prepare input
    x = torch.ones(seq_len, hidden_size, dtype=torch.bfloat16, device="cuda")
    router_logits = torch.randn(seq_len, num_experts, dtype=torch.bfloat16, device="cuda")

    # Mock torch.ops.trtllm.fp4_quantize to capture the input after scaling
    captured_input = None

    def mock_fp4_quantize(input_tensor, *args, **kwargs):
        nonlocal captured_input
        captured_input = input_tensor
        # Return dummy quantized output
        return (
            torch.zeros(
                input_tensor.shape[0], input_tensor.shape[1] // 2, dtype=torch.uint8, device="cuda"
            ),
            torch.ones(
                input_tensor.shape[0],
                input_tensor.shape[1] // 16,
                dtype=torch.float32,
                device="cuda",
            ),
        )

    # Also mock the MoE runner to avoid execution errors
    # just return a dummy output since we are capturing the input before input quantization
    def mock_moe_runner(*args, **kwargs):
        return [torch.zeros(seq_len, hidden_size, dtype=torch.bfloat16, device="cuda")]

    with patch("torch.ops.trtllm.fp4_quantize", side_effect=mock_fp4_quantize, create=True):
        with patch(
            "torch.ops.trtllm.fp4_block_scale_moe_runner", side_effect=mock_moe_runner, create=True
        ):
            with torch.inference_mode():
                moe.forward(x, router_logits)

    assert captured_input is not None, "fp4_quantize was not called"

    if has_scale:
        # Should be scaled by fc31_act_scale (which is loaded from pre_quant_scale)
        # The scale is 0.5, so x_passed should be x * 0.5
        expected = x * 0.5
        assert torch.allclose(captured_input, expected, rtol=1e-5, atol=1e-5), (
            "Expected scaled input"
        )
    else:
        # Should be original
        assert torch.equal(captured_input, x), "Expected original input"
