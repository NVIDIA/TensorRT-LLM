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
@pytest.mark.parametrize("mnk", [(1, 192, 128), (4, 192, 128), (8, 7168, 16384),
                                 (128, 7168, 16384)])
def test_fp4_linear(dtype, mnk):
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
    l_fp4 = Linear(
        in_features=HIDDEN_SIZE,
        out_features=OUTPUT_SIZE,
        bias=False,
        dtype=dtype,
        quant_config=qc,
        nvfp4_allowed_backends=['cutlass'])  # Force CUTLASS to match reference

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


@pytest.mark.skipif(sys.version_info < (3, 12),
                    reason="cutlass-dsl 4.1.0 requires Python 3.12+")
@pytest.mark.skipif(
    get_sm_version() not in [100, 103],
    reason="This test is only supported in sm100 and sm103 architecture",
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
                   nvfp4_allowed_backends=['cutedsl'])

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
                   nvfp4_allowed_backends=['cutedsl'])

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
                       nvfp4_allowed_backends=['cutlass'
                                               ])  # Use CUTLASS as reference

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

    alpha_tensor = torch.tensor([1.0]).cuda()
    with torch.inference_mode(), autotune():
        with nvtx.annotate(
                f"cute_dsl tune, m={SEQ_LEN}, k={HIDDEN_SIZE}, n={OUTPUT_SIZE}",
                color="orange",
        ):
            output = torch.ops.trtllm.cute_dsl_nvfp4_gemm_blackwell(
                x_fp4, w_fp4, x_sf_block, w_sf_block, alpha_tensor, dtype)
    from tensorrt_llm._torch.autotuner import AutoTuner
    AutoTuner.get().print_statistics()

    if test_ref:
        with nvtx.annotate(
                f"ref tune, m={SEQ_LEN}, k={HIDDEN_SIZE}, n={OUTPUT_SIZE}",
                color="orange"):
            with torch.inference_mode(), autotune():
                output_ref = torch.ops.trtllm.nvfp4_gemm_cutlass(
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
                alpha_tensor,
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
                alpha_tensor,
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
                output_ref = torch.ops.trtllm.nvfp4_gemm_cutlass(
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
                output_ref = torch.ops.trtllm.nvfp4_gemm_cutlass(
                    x_fp4_list[buffer_idx % workspace_count],
                    w_fp4_list[buffer_idx % workspace_count],
                    x_sf_block_list[buffer_idx % workspace_count],
                    w_sf_block_list[buffer_idx % workspace_count],
                    alpha_tensor,
                    dtype,
                )
                buffer_idx = buffer_idx + 1


@skip_pre_blackwell
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize(
    "mnk",
    [
        # Small batch sizes (M <= 16) - test small M handling
        (1, 4096, 4096, "Batch=1, Square 4K"),
        (4, 4096, 4096, "Batch=4, Square 4K"),
        (16, 4096, 4096, "Batch=16, Square 4K"),

        # Odd M values
        (3, 4096, 4096, "Odd M: M=3"),
        (7, 4096, 4096, "Odd M: M=7"),
        (9, 4096, 4096, "Odd M: M=9"),

        # Medium batch sizes - common inference scenarios
        (128, 4096, 4096, "Batch=128, Square 4K"),
        (128, 7168, 16384, "Batch=128, Large K/N"),
        (128, 4096, 7168, "Batch=128, Asymmetric"),

        # Large batch sizes - training scenarios
        (512, 4096, 4096, "Batch=512, Square 4K"),
        (1024, 4096, 4096, "Batch=1024, Square 4K"),

        # Very large batch - maximum performance
        (2048, 4096, 4096, "Batch=2048, Square 4K"),
        (4096, 4096, 4096, "Batch=4096, Square 4K"),

        # Large K and N - test memory bandwidth
        (128, 8192, 8192, "Batch=128, Square 8K"),
        (256, 16384, 16384, "Batch=256, Square 16K"),

        # Size asymmetry tests
        (1024, 128, 4096, "Wide M: M >> N"),
        (128, 16384, 128, "Wide N: N >> K"),
    ])
def test_nvfp4_gemm_unified_all_tactics(dtype, mnk):
    """Test nvfp4_gemm with auto backend selection, ensuring all tactics are tested."""
    from tensorrt_llm._torch.autotuner import AutoTuner, autotune
    from tensorrt_llm._torch.cublaslt_utils import IS_CUBLASLT_AVAILABLE

    # Unpack mnk with optional description
    if len(mnk) == 4:
        SEQ_LEN, OUTPUT_SIZE, HIDDEN_SIZE, desc = mnk
    else:
        SEQ_LEN, OUTPUT_SIZE, HIDDEN_SIZE = mnk
        desc = f"M={SEQ_LEN}, K={HIDDEN_SIZE}, N={OUTPUT_SIZE}"
    torch.manual_seed(0)

    x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype).cuda()
    x_sf_global = (448 * 6) / x.abs().max().float()

    w = torch.randn((OUTPUT_SIZE, HIDDEN_SIZE), dtype=dtype).cuda()
    w_sf_global = (448 * 6) / w.abs().max().float()
    w_fp4, w_sf_block = torch.ops.trtllm.fp4_quantize(w, w_sf_global,
                                                      scaling_vector_size,
                                                      False)

    # Prepare input
    with torch.inference_mode():
        x_fp4, x_sf_block = torch.ops.trtllm.fp4_quantize(
            x, x_sf_global, scaling_vector_size, False)
        alpha_ref = 1.0 / (w_sf_global * x_sf_global)
        alpha_tensor = torch.tensor([alpha_ref], dtype=torch.float32).cuda()

    # Reference: Use CUTLASS backend explicitly for reference output
    with torch.inference_mode():
        output_ref = torch.ops.trtllm.nvfp4_gemm(act_fp4=x_fp4,
                                                 weight=w_fp4,
                                                 act_sf=x_sf_block,
                                                 weight_scale=w_sf_block,
                                                 alpha=alpha_tensor,
                                                 output_dtype=dtype,
                                                 to_userbuffers=False,
                                                 allowed_backends='cutlass')

    # Test auto backend selection with autotuning
    with torch.inference_mode(), autotune():
        output_auto = torch.ops.trtllm.nvfp4_gemm(
            act_fp4=x_fp4,
            weight=w_fp4,
            act_sf=x_sf_block,
            weight_scale=w_sf_block,
            alpha=alpha_tensor,
            output_dtype=dtype,
            to_userbuffers=False,
            allowed_backends='cutlass,cublaslt,cuda_core,cutedsl')

    AutoTuner.get().print_profiling_cache()

    # Verify auto mode result matches reference
    torch.cuda.synchronize()
    torch.testing.assert_close(output_auto, output_ref, rtol=1e-2, atol=0.15)

    # Test all combinations of outer layer (backend selection) and inner layer (backend tactics)
    # Outer layer: nvfp4_gemm selects backend
    # Inner layer: each backend has its own tactics
    from collections import defaultdict

    print(f"\n{'='*80}")
    print(f"Testing nvfp4_gemm (2-layer tactics): {desc}")
    print(f"Shape: M={SEQ_LEN}, K={HIDDEN_SIZE}, N={OUTPUT_SIZE}")
    print(f"{'='*80}")

    print(f"\n[Outer Layer] Capturing backend selection tactics...")
    with AutoTuner.get().capture() as outer_capture, torch.inference_mode():
        output = torch.ops.trtllm.nvfp4_gemm(
            act_fp4=x_fp4,
            weight=w_fp4,
            act_sf=x_sf_block,
            weight_scale=w_sf_block,
            alpha=alpha_tensor,
            output_dtype=dtype,
            to_userbuffers=False,
            allowed_backends='cutlass,cublaslt,cuda_core,cutedsl')

    outer_tactics_list = list(outer_capture)
    print(f"  Found {len(outer_tactics_list)} outer layer tactics (backends)")

    # Parse outer tactics to get backend names
    backend_map = {}
    for outer_tactic in outer_tactics_list:
        outer_runner, backend_name = outer_tactic[0]
        backend_map[backend_name] = outer_tactic
        print(f"    - Backend: {backend_name}")

    print(f"\n[Inner Layer] Testing tactics for each backend...")

    # All backends have independent APIs, but cuda_core needs special handling, because it requires unswizzled scale factors
    backend_apis = {}
    if IS_CUTLASS_DSL_AVAILABLE:
        if 'cutlass' in backend_map:
            backend_apis['cutlass'] = torch.ops.trtllm.nvfp4_gemm_cutlass
    if IS_CUBLASLT_AVAILABLE:
        if 'cublaslt' in backend_map:
            backend_apis['cublaslt'] = torch.ops.trtllm.nvfp4_gemm_cublaslt
    if IS_CUTLASS_DSL_AVAILABLE:
        if 'cutedsl' in backend_map:
            backend_apis[
                'cutedsl'] = torch.ops.trtllm.cute_dsl_nvfp4_gemm_blackwell

    # cuda_core needs special handling (different parameters, single tactic)
    test_cuda_core = 'cuda_core' in backend_map

    # Step 3: For each backend, capture and immediately test all tactics
    # Must test immediately after capture to avoid _last_capture being overwritten
    tactics_by_backend = defaultdict(list)
    total_tactics_tested = 0

    for backend_name, backend_api in backend_apis.items():
        print(f"\n  Backend: {backend_name}")

        # Capture inner tactics for this backend
        with AutoTuner.get().capture() as inner_capture, torch.inference_mode():
            output = backend_api(
                x_fp4,  # input/act_fp4
                w_fp4,  # weight
                x_sf_block,  # input_scale/act_sf
                w_sf_block,  # weight_scale
                alpha_tensor,  # alpha
                dtype  # output_dtype
            )

        inner_tactics_list = list(inner_capture)
        print(f"    Found {len(inner_tactics_list)} inner tactics")

        # Verify tactics uniqueness (ensure we're testing different tactics, not repeating the same one)
        tactic_values = [t[0][1] for t in inner_tactics_list]
        unique_tactics = len(set(tactic_values))
        assert len(tactic_values) == unique_tactics, \
            f"Duplicate tactics detected! Total: {len(tactic_values)}, Unique: {unique_tactics}"

        # Test each tactic immediately (while _last_capture is still valid)
        for tactic_idx, inner_tactic in enumerate(inner_tactics_list):
            inner_runner, inner_tactic_value = inner_tactic[0]
            runner_name = inner_runner.__class__.__name__

            # Replay this tactic
            with AutoTuner.get().replay(inner_tactic), torch.inference_mode():
                # Call backend API directly (using positional args)
                output = backend_api(
                    x_fp4,  # input/act_fp4
                    w_fp4,  # weight
                    x_sf_block,  # input_scale/act_sf
                    w_sf_block,  # weight_scale
                    alpha_tensor,  # alpha
                    dtype  # output_dtype
                )

                # Verify correctness
                torch.testing.assert_close(output,
                                           output_ref,
                                           rtol=1e-2,
                                           atol=0.15)

            total_tactics_tested += 1
            tactics_by_backend[runner_name].append(total_tactics_tested)
            print(f"    ✓ Tactic {tactic_idx+1}/{len(inner_tactics_list)}: "
                  f"{runner_name} tactic={inner_tactic_value} - PASSED")

    # Step 4: Test cuda_core if it's available (single tactic, no capture needed)
    if test_cuda_core:
        print(f"\n  Backend: cuda_core")
        print(f"    Found 1 tactic (single implementation, no autotuning)")

        with torch.inference_mode():
            output_cuda_core = torch.ops.trtllm.nvfp4_gemm(
                act_fp4=x_fp4,
                weight=w_fp4,
                act_sf=x_sf_block,
                weight_scale=w_sf_block,
                alpha=alpha_tensor,
                output_dtype=dtype,
                to_userbuffers=False,
                allowed_backends='cuda_core')

            torch.testing.assert_close(output_cuda_core,
                                       output_ref,
                                       rtol=1e-2,
                                       atol=0.15)

        total_tactics_tested += 1
        tactics_by_backend['CudaCoreNVFP4Runner'].append(total_tactics_tested)
        print(f"    ✓ Tactic 1/1: CudaCoreNVFP4Runner tactic=0 - PASSED")

    print(f"\n{'='*80}")
    print(f"All {total_tactics_tested} tactics verified successfully!")
    print(f"\nBreakdown by backend:")
    for runner_name, indices in tactics_by_backend.items():
        print(f"  - {runner_name}: {len(indices)} tactics")
    if test_cuda_core:
        print(f"\n  Note: cuda_core has no autotuning (single tactic)")
    print(f"  Note: Tested all inner layer tactics for each backend")
    print(
        f"  Outer layer (backend selection) was tested separately with all backends allowed"
    )
    print(f"{'='*80}\n")


@pytest.mark.skipif(
    get_sm_version() not in [100, 103],
    reason="This test is only supported in Blackwell architecture",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("mnk", [(128, 7168, 16384), (128, 24576, 1536),
                                 (128, 2112, 7168), (128, 4096, 7168),
                                 (128, 7168, 2048), [127, 1024, 3200]])
def test_fp4_linear_cublaslt(dtype, mnk):
    """Test cuBLASLt FP4 GEMM implementation and compare with nvfp4_gemm_cutlass"""
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

    # Reference implementation: use torch.ops.trtllm.nvfp4_gemm_cutlass (CUTLASS)
    with torch.inference_mode():
        output_cutlass = torch.ops.trtllm.nvfp4_gemm_cutlass(
            x_fp4, w_fp4, x_sf_block, w_sf_block, alpha_ref, dtype)

    # Compare results
    torch.cuda.synchronize()
    torch.testing.assert_close(output_cublaslt, output_cutlass)


@pytest.mark.skipif(
    get_sm_version() < 100,
    reason="CUDA Core backend requires SM >= 100 (Blackwell or newer)",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("mnk", [(1, 4096, 7168), (4, 7168, 16384),
                                 (8, 2112, 7168)])
def test_fp4_linear_cuda_core(dtype, mnk):
    """Test CUDA Core NVFP4 GEMM implementation on SM >= 100 (M <= 8)"""

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

        # Reference: Use CUTLASS backend
        output_ref = torch.ops.trtllm.nvfp4_gemm(act_fp4=x_fp4,
                                                 weight=w_fp4,
                                                 act_sf=x_sf_block,
                                                 weight_scale=w_sf_block,
                                                 alpha=alpha_tensor,
                                                 output_dtype=dtype,
                                                 to_userbuffers=False,
                                                 allowed_backends='cutlass')

        # Test CUDA Core backend
        output_cuda_core = torch.ops.trtllm.nvfp4_gemm(
            act_fp4=x_fp4,
            weight=w_fp4,
            act_sf=x_sf_block,
            weight_scale=w_sf_block,
            alpha=alpha_tensor,
            output_dtype=dtype,
            to_userbuffers=False,
            allowed_backends='cuda_core')

    # Compare results
    torch.cuda.synchronize()
    torch.testing.assert_close(output_cuda_core,
                               output_ref,
                               rtol=1e-2,
                               atol=0.15)
    print(
        f"✓ CUDA Core test passed for M={SEQ_LEN}, N={OUTPUT_SIZE}, K={HIDDEN_SIZE}"
    )


if __name__ == "__main__":
    # m, n, k
    nvfp4_gemm_perf_test(torch.bfloat16, 128, 7168, 16384)

    # # group-1 test cases
    # for tokens in [128, 8192]:
    #     nvfp4_gemm_perf_test(torch.bfloat16, tokens, 7168, 16384)
    #     nvfp4_gemm_perf_test(torch.bfloat16, tokens, 24576, 1536)
    #     nvfp4_gemm_perf_test(torch.bfloat16, tokens, 2112, 7168)
    #     nvfp4_gemm_perf_test(torch.bfloat16, tokens, 4096, 7168)
    #     nvfp4_gemm_perf_test(torch.bfloat16, tokens, 7168, 2048)

    # # group-2 test cases
    # for m in [128, 256, 512]:
    #     nvfp4_gemm_perf_test(torch.bfloat16, m, 131584, 7168)
    #     nvfp4_gemm_perf_test(torch.bfloat16, m, 7168, 65792)
    #     nvfp4_gemm_perf_test(torch.bfloat16, m, 227368, 2560, test_ref=False)
    #     nvfp4_gemm_perf_test(torch.bfloat16, m, 2560, 113664)
