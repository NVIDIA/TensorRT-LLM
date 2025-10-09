import pytest
import torch

from tensorrt_llm._torch.auto_deploy.custom_ops.quant import FP8_MAX
from tensorrt_llm._torch.auto_deploy.transform.interface import TransformConfig
from tensorrt_llm._torch.auto_deploy.transform.library.quantization import (
    FP8LinearQuantizationFromConfig,
)
from tensorrt_llm._torch.auto_deploy.utils.quantization_utils import (
    fp4_global_scale,
    modelopt_fp4_scale_to_cutlass_fp4_scale,
)
from tensorrt_llm._torch.auto_deploy.utils.sharding_utils import _shard_fp4_weight_scale


@pytest.mark.parametrize("dim", [0, 1])
def test_fp4_scale_sharding(dim):
    weight = torch.rand(130, 64, dtype=torch.half, device="cuda")
    weight_scale_2 = fp4_global_scale(weight)

    weight_scale_modelopt = (
        torch.max(weight.reshape(weight.shape[0], -1, 16), dim=-1).values.to(torch.float)
        / (6.0 * weight_scale_2)
    ).to(torch.float8_e4m3fn)

    weight_scale_cutlass = modelopt_fp4_scale_to_cutlass_fp4_scale(weight_scale_modelopt)

    if dim == 0:
        uint8_weight_shape = (65, 32)
        expected_sharded_weight_scale_shape = 128 * 4
    elif dim == 1:
        uint8_weight_shape = (130, 16)
        expected_sharded_weight_scale_shape = 256 * 4

    fp4_scale_rank_0 = _shard_fp4_weight_scale(weight_scale_cutlass, uint8_weight_shape, dim, 0, 2)
    fp4_scale_rank_1 = _shard_fp4_weight_scale(weight_scale_cutlass, uint8_weight_shape, dim, 1, 2)
    assert (
        tuple(fp4_scale_rank_0.shape)
        == tuple(fp4_scale_rank_1.shape)
        == (expected_sharded_weight_scale_shape,)
    )


def test_fp4_global_scale():
    input = torch.rand(3, 64, dtype=torch.half, device="cuda")
    input[-1][-1] = 448 * 6
    input_scale = fp4_global_scale(input)
    assert input_scale.dtype == torch.float
    assert input_scale == torch.tensor(1.0, dtype=torch.float)


@pytest.mark.parametrize("amax, expected_scale", [(FP8_MAX, 1.0), (FP8_MAX / 2.0, 0.5)])
def test_fp8_convert_amax_hook(amax, expected_scale):
    config = TransformConfig(stage="pattern_matcher")
    fp8_imp = FP8LinearQuantizationFromConfig(config)

    mock_state_dict = {"amax": amax}

    fp8_imp.convert_amax_hook(mock_state_dict, None, None, scale_name="scale", amax_name="amax")

    assert "scale" in mock_state_dict
    assert mock_state_dict["scale"] == expected_scale
