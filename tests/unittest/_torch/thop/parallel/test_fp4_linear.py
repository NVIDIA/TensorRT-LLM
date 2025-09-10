import nvtx
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


def pad_up(x, pad_size):
    return (x + pad_size - 1) // pad_size * pad_size


def test_fp4_linear_perf(dtype, SEQ_LEN, OUTPUT_SIZE, HIDDEN_SIZE):
    torch.manual_seed(0)

    x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype).cuda()
    x_sf_global = (448 * 6) / x.abs().max().float()
    # x_sf_global = torch.tensor(1.0).cuda()

    w = torch.randn((OUTPUT_SIZE, HIDDEN_SIZE), dtype=dtype).cuda()
    w_sf_global = (448 * 6) / w.abs().max().float()
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
                   quant_config=qc,
                   use_cute_dsl_fp4_mm=True)

    assert l_fp4.weight.dtype == fp4_utils.float4_e2m1x2
    assert l_fp4.weight_scale.dtype == fp4_utils.float4_sf_dtype

    w_sf_block_unswizzled = (torch.ops.trtllm.block_scale_interleave_reverse(
        w_sf_block.cpu().view(pad_up(OUTPUT_SIZE, 128), -1)))
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

    # with torch.inference_mode(), autotune():
    #     output = l_fp4.forward(x)

    # # ref linear
    # with torch.inference_mode(), autotune():
    #     x_fp4, x_sf_block = torch.ops.trtllm.fp4_quantize(
    #         x, x_sf_global, scaling_vector_size, False)
    #     output_ref = torch.ops.trtllm.nvfp4_gemm(
    #                x_fp4, w_fp4, x_sf_block, w_sf_block, alpha_ref,
    #                dtype)

    # with torch.inference_mode():
    #     with nvtx.annotate(f"ref warmup, m={SEQ_LEN}, k={HIDDEN_SIZE}, n={OUTPUT_SIZE}", color="red"):
    #         for _ in range(5):
    #             output_ref = torch.ops.trtllm.nvfp4_gemm(
    #                 x_fp4, w_fp4, x_sf_block, w_sf_block, alpha_ref,
    #                 dtype)

    #     with nvtx.annotate(f"ref run, m={SEQ_LEN}, k={HIDDEN_SIZE}, n={OUTPUT_SIZE}", color="green"):
    #         for i in range(10):
    #             output_ref = torch.ops.trtllm.nvfp4_gemm(
    #                 x_fp4, w_fp4, x_sf_block, w_sf_block, alpha_ref,
    #                 dtype)

    l_fp4_ref = Linear(in_features=HIDDEN_SIZE,
                       out_features=OUTPUT_SIZE,
                       bias=False,
                       dtype=dtype,
                       quant_config=qc,
                       use_cute_dsl_fp4_mm=False)

    assert l_fp4_ref.weight.dtype == fp4_utils.float4_e2m1x2
    assert l_fp4_ref.weight_scale.dtype == fp4_utils.float4_sf_dtype

    w_sf_block_unswizzled = (torch.ops.trtllm.block_scale_interleave_reverse(
        w_sf_block.cpu().view(pad_up(OUTPUT_SIZE, 128), -1)))
    print(f"limin: w_sf_block_unswizzled.shape = {w_sf_block_unswizzled.shape}")

    l_fp4_ref.load_weights([{
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
    l_fp4_ref = l_fp4_ref.cuda()

    torch.testing.assert_close(l_fp4_ref.weight, w_fp4)
    torch.testing.assert_close(l_fp4_ref.input_scale[0], x_sf_global)
    torch.testing.assert_close(l_fp4_ref.weight_scale, w_sf_block)
    alpha_ref = 1.0 / (w_sf_global * x_sf_global)
    torch.testing.assert_close(l_fp4_ref.alpha[0], alpha_ref)

    # with torch.inference_mode(), autotune():
    #     output_ref = l_fp4_ref.forward(x)

    # with nvtx.annotate(f"cute_dsl warmup, m={SEQ_LEN}, k={HIDDEN_SIZE}, n={OUTPUT_SIZE}", color="red"):
    for _ in range(2):
        l_fp4.forward(x)

    with nvtx.annotate(
            f"cute_dsl run, m={SEQ_LEN}, k={HIDDEN_SIZE}, n={OUTPUT_SIZE}",
            color="green"):
        for i in range(1000):
            l_fp4.forward(x)

    # # with nvtx.annotate(f"ref warmup, m={SEQ_LEN}, k={HIDDEN_SIZE}, n={OUTPUT_SIZE}", color="red"):
    # for _ in range(5):
    #     output_ref = l_fp4_ref.forward(x)

    # print(f"limin: begin run ref test")
    # # with nvtx.annotate(f"ref run, m={SEQ_LEN}, k={HIDDEN_SIZE}, n={OUTPUT_SIZE}", color="green"):
    # with nvtx.annotate(f"ref", color="green"):
    #     for i in range(10):
    #         output_ref = l_fp4_ref.forward(x)

    # # compare
    # torch.cuda.synchronize()
    # torch.testing.assert_close(output, output_ref)
    # print(f"PASSED")


if __name__ == "__main__":
    # m, n, k
    # test_fp4_linear(torch.bfloat16, 128, 7680, 1536)
    # atten_out_linear_m128_n7168_k16384
    # q_linear_m128_n24576_k1536
    # qkv_down_linear_m128_n2112_k7168
    # shared_fc1_linear_m128_n4096_k7168
    # shared_fc2_linear_m128_n7168_k2048
    # test_fp4_linear_perf(torch.bfloat16, 128, 7168, 16384)
    test_fp4_linear_perf(torch.bfloat16, 128, 24576, 1536)
    # test_fp4_linear_perf(torch.bfloat16, 128, 2112, 7168)
    # test_fp4_linear_perf(torch.bfloat16, 128, 4096, 7168)
    # test_fp4_linear_perf(torch.bfloat16, 128, 7168, 2048)
