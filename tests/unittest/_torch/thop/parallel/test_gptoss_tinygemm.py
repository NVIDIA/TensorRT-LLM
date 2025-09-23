import pytest
import torch
from utils.util import skip_pre_hopper


@skip_pre_hopper
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize("input_dim", [720, 1440, 2880])
@pytest.mark.parametrize("output_dim", [32, 64, 128])
def test_gptoss_tinygemm(batch_size, input_dim, output_dim):
    x = torch.randn(batch_size, input_dim, device='cuda', dtype=torch.bfloat16)
    weight = torch.randn(output_dim,
                         input_dim,
                         device='cuda',
                         dtype=torch.bfloat16)
    bias = torch.randn(output_dim, device='cuda', dtype=torch.bfloat16)

    # Run the tinygemm2 operation
    output = torch.ops.trtllm.gptoss_tinygemm(x, weight, bias)

    # Check the output shape
    assert output.shape == (batch_size, output_dim)

    output_ref = torch.nn.functional.linear(x, weight, bias)

    max_diff = torch.max(torch.abs(output - output_ref))
    mean_diff = torch.mean(torch.abs(output - output_ref))

    assert max_diff < 2e-2
    assert mean_diff < 1e-3
