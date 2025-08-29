import pytest
import torch


def fused_a_gemm_ref(input, weight, bias, dtype):
    logits_ref = torch.matmul(input, weight)
    return logits_ref


@pytest.mark.parametrize("num_tokens", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("hd_out", [2112])
@pytest.mark.parametrize("hd_in", [7168])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_fused_a_gemm_run(num_tokens, hd_out, hd_in, dtype):
    torch.manual_seed(24)
    torch.cuda.manual_seed(24)

    device = torch.device("cuda")
    input = torch.randn(num_tokens, hd_in, dtype=dtype, device=device)
    weight = torch.randn((hd_out, hd_in), dtype=dtype, device=device)
    bias = None
    logits = torch.ops.trtllm.dsv3_fused_a_gemm_op(input, weight.t(), bias,
                                                   dtype)
    logtis_ref = fused_a_gemm_ref(input, weight.t(), bias, dtype)
    assert torch.allclose(logits, logtis_ref, rtol=0.1)
