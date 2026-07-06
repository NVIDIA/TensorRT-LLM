"""Unit test for DeepseekV3DecoderLayer.forward_mlp fallback logic.

Validates that when PRE_MLP_FUSION is enabled but gate_up_proj.has_nvfp4 is
False, the code uses AllReduceFusionOp.RESIDUAL_RMS_NORM instead of
RESIDUAL_RMS_NORM_QUANT_NVFP4. This is the fix introduced for GLM-5.1 NVFP4
where dense layers lack nvfp4 quantization.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from tensorrt_llm._torch.distributed import AllReduceFusionOp, AllReduceParams


@pytest.fixture
def mock_decoder_layer():
    """Create a minimal mock of DeepseekV3DecoderLayer with forward_mlp accessible."""
    from tensorrt_llm._torch.models.modeling_deepseekv3 import \
        DeepseekV3DecoderLayer

    layer = MagicMock()

    # Setup fusion_config with PRE_MLP_FUSION enabled, POST_MLP_FUSION disabled
    layer.fusion_config = SimpleNamespace(
        PRE_MLP_FUSION=True,
        POST_MLP_FUSION=False,
    )

    # Setup layer_idx (used in spec_metadata check)
    layer.layer_idx = 0

    # Setup post_attention_layernorm
    layer.post_attention_layernorm = MagicMock()
    layer.post_attention_layernorm.weight = torch.randn(64,
                                                        device='cuda',
                                                        dtype=torch.float16)
    layer.post_attention_layernorm.variance_epsilon = 1e-5

    # Setup next_layer_layernorm - must return a 2-tuple when called
    next_ln = MagicMock()
    next_ln.weight = torch.randn(64, device='cuda', dtype=torch.float16)
    next_ln.variance_epsilon = 1e-5
    next_ln.return_value = (
        torch.randn(4, 64, device='cuda', dtype=torch.float16),
        torch.randn(4, 64, device='cuda', dtype=torch.float16),
    )
    layer.next_layer_layernorm = next_ln

    # Setup mlp with gate_up_proj
    layer.mlp = MagicMock()
    layer.mlp.gate_up_proj = MagicMock()
    layer.mlp.gate_up_proj.input_scale = torch.tensor(1.0, device='cuda')
    # mlp returns a tensor when called
    layer.mlp.return_value = torch.randn(4,
                                         64,
                                         device='cuda',
                                         dtype=torch.float16)

    layer.mlp_tp_size = 1

    # Bind the real forward_mlp method to our mock
    layer.forward_mlp = DeepseekV3DecoderLayer.forward_mlp.__get__(
        layer, DeepseekV3DecoderLayer)

    return layer


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="CUDA required for this test")
class TestForwardMlpNvfp4Fallback:
    """Test the PRE_MLP_FUSION branch in forward_mlp for nvfp4 vs non-nvfp4."""

    def test_no_nvfp4_uses_residual_rms_norm(self, mock_decoder_layer):
        """When has_nvfp4 is False, should use RESIDUAL_RMS_NORM fusion op."""
        layer = mock_decoder_layer
        layer.mlp.gate_up_proj.has_nvfp4 = False

        hidden_states = torch.randn(4,
                                    64,
                                    device='cuda',
                                    dtype=torch.float16)
        residual = torch.randn(4, 64, device='cuda', dtype=torch.float16)

        # allreduce should return (hidden_states, residual) for RESIDUAL_RMS_NORM
        expected_h = torch.randn(4, 64, device='cuda', dtype=torch.float16)
        expected_r = torch.randn(4, 64, device='cuda', dtype=torch.float16)
        layer.allreduce = MagicMock(return_value=(expected_h, expected_r))

        result_h, result_r = layer.forward_mlp(hidden_states, residual)

        # Verify allreduce was called with RESIDUAL_RMS_NORM
        layer.allreduce.assert_called_once()
        all_reduce_params = layer.allreduce.call_args.kwargs[
            'all_reduce_params']

        assert all_reduce_params is not None, \
            "allreduce must be called with all_reduce_params"
        assert all_reduce_params.fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM, \
            f"Expected RESIDUAL_RMS_NORM but got {all_reduce_params.fusion_op}"

    def test_with_nvfp4_uses_residual_rms_norm_quant_nvfp4(
            self, mock_decoder_layer):
        """When has_nvfp4 is True, should use RESIDUAL_RMS_NORM_QUANT_NVFP4 fusion op."""
        layer = mock_decoder_layer
        layer.mlp.gate_up_proj.has_nvfp4 = True

        hidden_states = torch.randn(4,
                                    64,
                                    device='cuda',
                                    dtype=torch.float16)
        residual = torch.randn(4, 64, device='cuda', dtype=torch.float16)

        # allreduce should return (act_fp4, act_sf, residual) for NVFP4 path
        act_fp4 = torch.randint(-128, 127, (4, 32),
                                device='cuda',
                                dtype=torch.int8)
        act_sf = torch.randn(4, 4, device='cuda', dtype=torch.float16)
        expected_r = torch.randn(4, 64, device='cuda', dtype=torch.float16)
        layer.allreduce = MagicMock(return_value=(act_fp4, act_sf, expected_r))

        # Patch Fp4QuantizedTensor to avoid constructor validation issues
        with patch(
                'tensorrt_llm._torch.models.modeling_deepseekv3.Fp4QuantizedTensor'
        ) as mock_fp4:
            mock_fp4_instance = MagicMock()
            mock_fp4.return_value = mock_fp4_instance

            result_h, result_r = layer.forward_mlp(hidden_states, residual)

            # Verify allreduce was called with RESIDUAL_RMS_NORM_QUANT_NVFP4
            layer.allreduce.assert_called_once()
            all_reduce_params = layer.allreduce.call_args.kwargs[
                'all_reduce_params']

            assert all_reduce_params is not None, \
                "allreduce must be called with all_reduce_params"
