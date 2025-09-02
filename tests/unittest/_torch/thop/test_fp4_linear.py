import pytest
import torch

import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
from tensorrt_llm._torch.autotuner import autotune
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

# from utils.util import skip_pre_blackwell

scaling_vector_size = 16


# @skip_pre_blackwell
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("SEQ_LEN", [128, 256, 512])
@pytest.mark.parametrize("HIDDEN_SIZE", [7680])
@pytest.mark.parametrize("OUTPUT_SIZE", [1536, 3072])
# TODO: Do we need float32 test case? fp4_quantize only supports fp16, bf16, fp8_e4m3
def test_fp4_linear(dtype, SEQ_LEN, HIDDEN_SIZE, OUTPUT_SIZE):
    # SEQ_LEN = 10
    # HIDDEN_SIZE = 128
    # OUTPUT_SIZE = 256
    torch.manual_seed(0)

    x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype).cuda()
    x_sf_global = (448 * 6) / x.abs().max().float()
    # x_sf_global = torch.tensor(1.0).cuda()

    w = torch.randn((OUTPUT_SIZE, HIDDEN_SIZE), dtype=dtype).cuda()
    w_sf_global = (448 * 6) / w.abs().max().float()
    # w_sf_global = torch.tensor(1.0).cuda()
    w_fp4, w_sf_block = torch.ops.trtllm.fp4_quantize(w, w_sf_global,
                                                      scaling_vector_size,
                                                      False)
    print(f"limin: w_fp4.shape = {w_fp4.shape}")
    print(f"limin: w_fp4.dtype = {w_fp4.dtype}")
    print(f"limin: w_sf_block.shape = {w_sf_block.shape}")
    print(f"limin: w_sf_block.dtype = {w_sf_block.dtype}")

    qc = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    l_fp4 = Linear(in_features=HIDDEN_SIZE,
                   out_features=OUTPUT_SIZE,
                   bias=False,
                   dtype=dtype,
                   quant_config=qc)

    assert l_fp4.weight.dtype == fp4_utils.float4_e2m1x2
    assert l_fp4.weight_scale.dtype == fp4_utils.float4_sf_dtype

    w_sf_block_unswizzled = (torch.ops.trtllm.block_scale_interleave_reverse(
        w_sf_block.cpu().view(OUTPUT_SIZE, -1)))
    print(f"limin: w_sf_block_unswizzled.shape = {w_sf_block_unswizzled.shape}")

    l_fp4.load_weights([{
        'input_scale':
        1.0 / x_sf_global.cpu(),  # Simulates amax/(448*6) in modelopt ckpt
        'weight':
        w_fp4.cpu(),
        'weight_scale':
        w_sf_block_unswizzled.view(
            torch.float8_e4m3fn),  # Simulates float8_e4m3fn in modelopt ckpt
        'weight_scale_2':
        1.0 / w_sf_global.cpu()  # Simulates amax/(448*6) in modelopt ckpt
    }])
    l_fp4 = l_fp4.cuda()

    torch.testing.assert_close(l_fp4.weight, w_fp4)
    torch.testing.assert_close(l_fp4.input_scale[0], x_sf_global)
    torch.testing.assert_close(l_fp4.weight_scale, w_sf_block)
    alpha_ref = 1.0 / (w_sf_global * x_sf_global)
    torch.testing.assert_close(l_fp4.alpha[0], alpha_ref)
    print(f"limin: alpha_ref = {alpha_ref}")

    with torch.inference_mode(), autotune():
        output = l_fp4.forward(x)

    output = l_fp4.forward(x)

    # ref linear
    with torch.inference_mode():
        x_fp4, x_sf_block = torch.ops.trtllm.fp4_quantize(
            x, x_sf_global, scaling_vector_size, False)
        output_ref = torch.ops.trtllm.fp4_gemm(
            x_fp4, w_fp4, x_sf_block, w_sf_block, alpha_ref,
            fp4_utils.FP4GemmType.W4A4_NVFP4_NVFP4, dtype)

    # compare
    torch.cuda.synchronize()
    torch.testing.assert_close(output, output_ref)
    print(f"PASSED")


if __name__ == "__main__":
    test_fp4_linear(torch.bfloat16, 128, 7680, 1536)
