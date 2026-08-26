import pytest
import torch


def fused_a_gemm_ref(input, weight, bias, dtype):
    logits_ref = torch.matmul(input, weight)
    return logits_ref


# (hd_in, hd_out) of the fused A-projection weight, one pair per supported model shape:
#   DeepSeek-V3/V3.2:   7168 -> 2112 (fused q_a + kv_a down-proj),  dispatched at tile_m=16.
#   GlmMoeDsaForCausalLM: 6144 -> 2624 (kv_a_proj_with_mqa),        dispatched at tile_m=32.
@pytest.mark.parametrize("hd_in, hd_out", [(7168, 2112), (6144, 2624)])
# num_tokens spans the tile_n dispatch boundary (<=8 uses tile_n=8, >8 uses tile_n=16)
# and the full fused range [1, 16].
@pytest.mark.parametrize("num_tokens", [1, 2, 3, 4, 5, 8, 9, 16])
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
