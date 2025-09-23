import pytest
import torch
from utils.util import skip_pre_hopper


@skip_pre_hopper
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize("input_dim", [720, 1440, 2880])
@pytest.mark.parametrize("output_dim", [32, 64, 128])
def test_tinygemm2(batch_size, input_dim, output_dim):
    torch.manual_seed(42)

    x = torch.randn(batch_size, input_dim, device='cuda', dtype=torch.bfloat16)
    weight = torch.randn(output_dim,
                         input_dim,
                         device='cuda',
                         dtype=torch.bfloat16)
    bias = torch.randn(output_dim, device='cuda', dtype=torch.bfloat16)

    # Run the tinygemm2 operation
    output = torch.ops.trtllm.tinygemm2(x, weight, bias)

    # Check the output shape
    assert output.shape == (batch_size, output_dim)

    output_ref = torch.nn.functional.linear(x, weight, bias)

    assert torch.allclose(output, output_ref, rtol=1e-2, atol=1e-2)
