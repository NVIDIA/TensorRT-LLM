import pytest
import torch

from tensorrt_llm._torch.autotuner import autotune
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

scaling_vector_size = 16


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16],
                         "weights_dtype", [torch.uint8])
def test_w4a16_linear(dtype, weights_dtype):
    SEQ_LEN = 10
    HIDDEN_SIZE = 128
    torch.manual_seed(0)

    x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype).cuda()
    w = torch.randn(
        (HIDDEN_SIZE, HIDDEN_SIZE),
        dtype=weights_dtype).cuda()  # todo type needs to be different

    qc = QuantConfig(quant_algo=QuantAlgo.W4A16_AWQ)
    linear_w4a16 = Linear(in_features=HIDDEN_SIZE,
                          out_features=HIDDEN_SIZE,
                          bias=False,
                          dtype=dtype,
                          quant_config=qc)

    # assert l_fp4.weight.dtype == fp4_utils.float4_e2m1x2
    # assert l_fp4.weight_scale.dtype == fp4_utils.float4_sf_dtype

    linear_w4a16.load_weights([{
        'pre_qunat_scale':
        1.0 / x_sf_global.cpu(),  # Simulates amax/(448*6) in modelopt ckpt
        'weight':
        w_fp4.cpu(),
        'weight_scale':
        w_sf_block_unswizzled.view(
            torch.float8_e4m3fn),  # Simulates float8_e4m3fn in modelopt ckpt
    }])
    linear_w4a16 = linear_w4a16.cuda()

    torch.testing.assert_close(linear_w4a16.weight, w)
    torch.testing.assert_close(l_fp4.input_scale[0], x_sf_global)
    torch.testing.assert_close(l_fp4.weight_scale, w_sf_block)
    alpha_ref = 1.0 / (w_sf_global * x_sf_global)
    torch.testing.assert_close(l_fp4.alpha[0], alpha_ref)

    with torch.inference_mode(), autotune():
        output = linear_w4a16.forward(x)

    # ref linear
    with torch.inference_mode():
        output_ref = torch.ops.trtllm.fp4_gemm(x_fp4, w_fp4, x_sf_block,
                                               w_sf_block, alpha_ref, False,
                                               dtype)

    # compare
    torch.cuda.synchronize()
    torch.testing.assert_close(output, output_ref)
