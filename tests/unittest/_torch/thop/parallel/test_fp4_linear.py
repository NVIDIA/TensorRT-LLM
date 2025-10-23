import sys

import pytest
import torch
from utils.util import skip_pre_blackwell

import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
from tensorrt_llm._torch.autotuner import autotune
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.math_utils import pad_up
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

scaling_vector_size = 16


@skip_pre_blackwell
@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16]
)  # TODO: Do we need float32 test case? fp4_quantize only supports fp16, bf16, fp8_e4m3
def test_fp4_linear(dtype):
    SEQ_LEN = 10
    HIDDEN_SIZE = 128
    torch.manual_seed(0)

    x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype).cuda()
    x_sf_global = (448 * 6) / x.abs().max().float()

    w = torch.randn((HIDDEN_SIZE, HIDDEN_SIZE), dtype=dtype).cuda()
    w_sf_global = (448 * 6) / w.abs().max().float()
    w_fp4, w_sf_block = torch.ops.trtllm.fp4_quantize(w, w_sf_global,
                                                      scaling_vector_size,
                                                      False)

    qc = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    l_fp4 = Linear(in_features=HIDDEN_SIZE,
                   out_features=HIDDEN_SIZE,
                   bias=False,
                   dtype=dtype,
                   quant_config=qc)

    assert l_fp4.weight.dtype == fp4_utils.float4_e2m1x2
    assert l_fp4.weight_scale.dtype == fp4_utils.float4_sf_dtype

    w_sf_block_unswizzled = (torch.ops.trtllm.block_scale_interleave_reverse(
        w_sf_block.cpu().view(HIDDEN_SIZE, -1)))

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

    with torch.inference_mode(), autotune():
        output = l_fp4.forward(x)

    output_ref = l_fp4.forward(x)

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


@pytest.mark.skipif(sys.version_info < (3, 12),
                    reason="cutlass-dsl 4.1.0 requires Python 3.12+")
@pytest.mark.skipif(
    get_sm_version() != 100,
    reason="This test is only supported in Blackwell architecture",
)
@pytest.mark.skipif(not IS_CUTLASS_DSL_AVAILABLE,
                    reason="cutlass-dsl is not available")
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("mnk", [(128, 7168, 16384), (128, 24576, 1536),
                                 (128, 2112, 7168), (128, 4096, 7168),
                                 (128, 7168, 2048), [127, 1024, 3200]])
def test_fp4_linear_cute_dsl(dtype, mnk):

    SEQ_LEN, OUTPUT_SIZE, HIDDEN_SIZE = mnk
    torch.manual_seed(0)

    x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype).cuda()
    x_sf_global = (448 * 6) / x.abs().max().float()

    w = torch.randn((OUTPUT_SIZE, HIDDEN_SIZE), dtype=dtype).cuda()
    w_sf_global = (448 * 6) / w.abs().max().float()
    w_fp4, w_sf_block = torch.ops.trtllm.fp4_quantize(w, w_sf_global,
                                                      scaling_vector_size,
                                                      False)

    qc = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    l_fp4 = Linear(in_features=HIDDEN_SIZE,
                   out_features=OUTPUT_SIZE,
                   bias=False,
                   dtype=dtype,
                   quant_config=qc,
                   use_cute_dsl_nvfp4_blockscaling_mm=True)

    assert l_fp4.weight.dtype == fp4_utils.float4_e2m1x2
    assert l_fp4.weight_scale.dtype == fp4_utils.float4_sf_dtype

    w_sf_block_unswizzled = (torch.ops.trtllm.block_scale_interleave_reverse(
        w_sf_block.cpu().view(pad_up(OUTPUT_SIZE, 128), -1)))

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


def fp4_linear_perf_test(dtype, SEQ_LEN, OUTPUT_SIZE, HIDDEN_SIZE):
    torch.manual_seed(0)

    x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype).cuda()
    x_sf_global = (448 * 6) / x.abs().max().float()

    w = torch.randn((OUTPUT_SIZE, HIDDEN_SIZE), dtype=dtype).cuda()
    w_sf_global = (448 * 6) / w.abs().max().float()
    w_fp4, w_sf_block = torch.ops.trtllm.fp4_quantize(w, w_sf_global,
                                                      scaling_vector_size,
                                                      False)

    qc = QuantConfig(quant_algo=QuantAlgo.NVFP4)
    l_fp4 = Linear(in_features=HIDDEN_SIZE,
                   out_features=OUTPUT_SIZE,
                   bias=False,
                   dtype=dtype,
                   quant_config=qc,
                   use_cute_dsl_nvfp4_blockscaling_mm=True)

    assert l_fp4.weight.dtype == fp4_utils.float4_e2m1x2
    assert l_fp4.weight_scale.dtype == fp4_utils.float4_sf_dtype

    w_sf_block_unswizzled = (torch.ops.trtllm.block_scale_interleave_reverse(
        w_sf_block.cpu().view(pad_up(OUTPUT_SIZE, 128), -1)))

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

    with torch.inference_mode(), autotune():
        output = l_fp4.forward(x)

    l_fp4_ref = Linear(in_features=HIDDEN_SIZE,
                       out_features=OUTPUT_SIZE,
                       bias=False,
                       dtype=dtype,
                       quant_config=qc,
                       use_cute_dsl_nvfp4_blockscaling_mm=False)

    assert l_fp4_ref.weight.dtype == fp4_utils.float4_e2m1x2
    assert l_fp4_ref.weight_scale.dtype == fp4_utils.float4_sf_dtype

    w_sf_block_unswizzled = (torch.ops.trtllm.block_scale_interleave_reverse(
        w_sf_block.cpu().view(pad_up(OUTPUT_SIZE, 128), -1)))

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

    with torch.inference_mode(), autotune():
        output_ref = l_fp4_ref.forward(x)

    for _ in range(5):
        output = l_fp4.forward(x)

    for i in range(10):
        output = l_fp4.forward(x)

    for _ in range(5):
        output_ref = l_fp4_ref.forward(x)

    for i in range(10):
        output_ref = l_fp4_ref.forward(x)

    # compare
    torch.cuda.synchronize()
    torch.testing.assert_close(output, output_ref)


# cold L2 cache for benchmarking (using circular buffer)
def nvfp4_gemm_perf_test(
    dtype,
    SEQ_LEN,
    OUTPUT_SIZE,
    HIDDEN_SIZE,
    test_ref=True,
    use_cold_l2_cache=True,
    warmup_iterations=2,
    iterations=1000,
):
    import cutlass.cute as cute
    import nvtx

    torch.manual_seed(0)
    x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype).cuda()
    x_sf_global = (448 * 6) / x.abs().max().float()
    w = torch.randn((OUTPUT_SIZE, HIDDEN_SIZE), dtype=dtype).cuda()
    w_sf_global = (448 * 6) / w.abs().max().float()
    w_fp4, w_sf_block = torch.ops.trtllm.fp4_quantize(w, w_sf_global,
                                                      scaling_vector_size,
                                                      False)
    x_fp4, x_sf_block = torch.ops.trtllm.fp4_quantize(x, x_sf_global,
                                                      scaling_vector_size,
                                                      False)

    if use_cold_l2_cache:
        one_workspace_bytes = (x_fp4.numel() * x_fp4.element_size() +
                               w_fp4.numel() * w_fp4.element_size() +
                               x_sf_block.numel() * x_sf_block.element_size() +
                               w_sf_block.numel() * w_sf_block.element_size())
        workspace_count = cute.testing.get_workspace_count(
            one_workspace_bytes, warmup_iterations, iterations)
        x_fp4_list = [x_fp4]
        w_fp4_list = [w_fp4]
        x_sf_block_list = [x_sf_block]
        w_sf_block_list = [w_sf_block]
        for _ in range(workspace_count - 1):
            x_fp4_list.append(x_fp4.clone())
            w_fp4_list.append(w_fp4.clone())
            x_sf_block_list.append(x_sf_block.clone())
            w_sf_block_list.append(w_sf_block.clone())
    else:
        workspace_count = 1
        x_fp4_list = [x_fp4]
        w_fp4_list = [w_fp4]
        x_sf_block_list = [x_sf_block]
        w_sf_block_list = [w_sf_block]

    with torch.inference_mode(), autotune():
        with nvtx.annotate(
                f"cute_dsl tune, m={SEQ_LEN}, k={HIDDEN_SIZE}, n={OUTPUT_SIZE}",
                color="orange",
        ):
            output = torch.ops.trtllm.cute_dsl_nvfp4_gemm_blackwell(
                x_fp4, w_fp4, x_sf_block, w_sf_block, 1.0, dtype)

    alpha_tensor = torch.tensor(1.0).cuda()
    if test_ref:
        with nvtx.annotate(
                f"ref tune, m={SEQ_LEN}, k={HIDDEN_SIZE}, n={OUTPUT_SIZE}",
                color="orange"):
            with torch.inference_mode(), autotune():
                output_ref = torch.ops.trtllm.nvfp4_gemm(
                    x_fp4, w_fp4, x_sf_block, w_sf_block, alpha_tensor, dtype)
        torch.testing.assert_close(output, output_ref)
        print(f"PASSED")

    buffer_idx = 0
    with nvtx.annotate(
            f"cute_dsl warmup, m={SEQ_LEN}, k={HIDDEN_SIZE}, n={OUTPUT_SIZE}",
            color="green"):
        for _ in range(warmup_iterations):
            output = torch.ops.trtllm.cute_dsl_nvfp4_gemm_blackwell(
                x_fp4_list[buffer_idx % workspace_count],
                w_fp4_list[buffer_idx % workspace_count],
                x_sf_block_list[buffer_idx % workspace_count],
                w_sf_block_list[buffer_idx % workspace_count],
                1.0,
                dtype,
            )
            buffer_idx = buffer_idx + 1

    with nvtx.annotate(
            f"cute_dsl run, m={SEQ_LEN}, k={HIDDEN_SIZE}, n={OUTPUT_SIZE}",
            color="green"):
        for i in range(iterations):
            output = torch.ops.trtllm.cute_dsl_nvfp4_gemm_blackwell(
                x_fp4_list[buffer_idx % workspace_count],
                w_fp4_list[buffer_idx % workspace_count],
                x_sf_block_list[buffer_idx % workspace_count],
                w_sf_block_list[buffer_idx % workspace_count],
                1.0,
                dtype,
            )
            buffer_idx = buffer_idx + 1

    if test_ref:
        torch.testing.assert_close(output, output_ref)
        print(f"PASSED")

        buffer_idx = 0
        with nvtx.annotate(
                f"ref warmup, m={SEQ_LEN}, k={HIDDEN_SIZE}, n={OUTPUT_SIZE}",
                color="red"):
            for _ in range(warmup_iterations):
                output_ref = torch.ops.trtllm.nvfp4_gemm(
                    x_fp4_list[buffer_idx % workspace_count],
                    w_fp4_list[buffer_idx % workspace_count],
                    x_sf_block_list[buffer_idx % workspace_count],
                    w_sf_block_list[buffer_idx % workspace_count],
                    alpha_tensor,
                    dtype,
                )
                buffer_idx = buffer_idx + 1
        with nvtx.annotate(
                f"ref run, m={SEQ_LEN}, k={HIDDEN_SIZE}, n={OUTPUT_SIZE}",
                color="red"):
            for i in range(iterations):
                output_ref = torch.ops.trtllm.nvfp4_gemm(
                    x_fp4_list[buffer_idx % workspace_count],
                    w_fp4_list[buffer_idx % workspace_count],
                    x_sf_block_list[buffer_idx % workspace_count],
                    w_sf_block_list[buffer_idx % workspace_count],
                    alpha_tensor,
                    dtype,
                )
                buffer_idx = buffer_idx + 1


@pytest.mark.skipif(
    get_sm_version() not in [100, 103],
    reason="This test is only supported in Blackwell architecture",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("mnk", [(128, 7168, 16384), (128, 24576, 1536),
                                 (128, 2112, 7168), (128, 4096, 7168),
                                 (128, 7168, 2048), [127, 1024, 3200]])
def test_fp4_linear_cublaslt(dtype, mnk):
    """Test cuBLASLt FP4 GEMM implementation and compare with nvfp4_gemm"""
    from tensorrt_llm._torch.cublaslt_utils import IS_CUBLASLT_AVAILABLE
    if not IS_CUBLASLT_AVAILABLE:
        pytest.skip("cuBLASLt FP4 GEMM not available in this build")

    SEQ_LEN, OUTPUT_SIZE, HIDDEN_SIZE = mnk
    torch.manual_seed(0)

    x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype).cuda()
    x_sf_global = (448 * 6) / x.abs().max().float()

    w = torch.randn((OUTPUT_SIZE, HIDDEN_SIZE), dtype=dtype).cuda()
    w_sf_global = (448 * 6) / w.abs().max().float()
    w_fp4, w_sf_block = torch.ops.trtllm.fp4_quantize(w, w_sf_global,
                                                      scaling_vector_size,
                                                      False)

    with torch.inference_mode():
        x_fp4, x_sf_block = torch.ops.trtllm.fp4_quantize(
            x, x_sf_global, scaling_vector_size, False)

        alpha_ref = 1.0 / (w_sf_global * x_sf_global)
        alpha_tensor = torch.tensor(alpha_ref, dtype=torch.float32).cuda()

        # Use cuBLASLt FP4 GEMM with autotuning support
        with autotune():
            output_cublaslt = torch.ops.trtllm.nvfp4_gemm_cublaslt(
                act_fp4=x_fp4,
                weight=w_fp4,
                act_sf=x_sf_block,
                weight_scale=w_sf_block,
                alpha=alpha_tensor,
                output_dtype=dtype)

    # Reference implementation: use torch.ops.trtllm.nvfp4_gemm (CUTLASS)
    with torch.inference_mode():
        output_cutlass = torch.ops.trtllm.nvfp4_gemm(x_fp4, w_fp4, x_sf_block,
                                                     w_sf_block, alpha_ref,
                                                     dtype)

    # Compare results
    torch.cuda.synchronize()
    torch.testing.assert_close(output_cublaslt, output_cutlass)


if __name__ == "__main__":
    # m, n, k
    fp4_linear_perf_test(torch.bfloat16, 128, 7168, 16384)
    fp4_linear_perf_test(torch.bfloat16, 128, 24576, 1536)
    fp4_linear_perf_test(torch.bfloat16, 128, 2112, 7168)
    fp4_linear_perf_test(torch.bfloat16, 128, 4096, 7168)
    fp4_linear_perf_test(torch.bfloat16, 128, 7168, 2048)

    # group-1 test cases
    for tokens in [128, 8192]:
        nvfp4_gemm_perf_test(torch.bfloat16, tokens, 7168, 16384)
        nvfp4_gemm_perf_test(torch.bfloat16, tokens, 24576, 1536)
        nvfp4_gemm_perf_test(torch.bfloat16, tokens, 2112, 7168)
        nvfp4_gemm_perf_test(torch.bfloat16, tokens, 4096, 7168)
        nvfp4_gemm_perf_test(torch.bfloat16, tokens, 7168, 2048)

    # group-2 test cases
    for m in [128, 256, 512]:
        nvfp4_gemm_perf_test(torch.bfloat16, m, 131584, 7168)
        nvfp4_gemm_perf_test(torch.bfloat16, m, 7168, 65792)
        nvfp4_gemm_perf_test(torch.bfloat16, m, 227368, 2560, test_ref=False)
        nvfp4_gemm_perf_test(torch.bfloat16, m, 2560, 113664)
