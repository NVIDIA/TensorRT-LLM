# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

import pytest
import torch
from utils.util import skip_pre_blackwell

import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
from tensorrt_llm._torch.autotuner import autotune
from tensorrt_llm._torch.cute_dsl_utils import (IS_CUTLASS_DSL_AVAILABLE,
                                                IS_CUTLASS_DSL_RUBIN_AVAILABLE)
from tensorrt_llm._torch.locality_domain.policy import LocalityDomainPolicy
from tensorrt_llm._torch.locality_domain_utils import is_locality_domain_enabled
from tensorrt_llm._torch.modules.linear import (Linear, WeightMode,
                                                WeightsLoadingConfig)
from tensorrt_llm._torch.modules.swiglu import swiglu
from tensorrt_llm._torch.utils import Fp4QuantizedTensor, model_extra_attrs
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
                                                 output_buffer_kind=0,
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
            output_buffer_kind=0,
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
            output_buffer_kind=0,
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
                output_buffer_kind=0,
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
                                                 output_buffer_kind=0,
                                                 allowed_backends='cutlass')

        # Test CUDA Core backend
        output_cuda_core = torch.ops.trtllm.nvfp4_gemm(
            act_fp4=x_fp4,
            weight=w_fp4,
            act_sf=x_sf_block,
            weight_scale=w_sf_block,
            alpha=alpha_tensor,
            output_dtype=dtype,
            output_buffer_kind=0,
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


@pytest.mark.skipif(
    not (89 <= get_sm_version() < 100 or get_sm_version() in (120, 121)),
    reason="Dense Marlin NVFP4 runs on SM89-99 and SM120/121",
)
@pytest.mark.parametrize("quant_algo", [QuantAlgo.NVFP4, QuantAlgo.W4A16_NVFP4])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize(
    "mnk",
    [
        (1, 1024, 1024),
        (8, 1024, 2048),
        (128, 2048, 1024),
        (1, 18560, 4096),
        (128, 18560, 4096),
        (1, 4096, 8192),
        (128, 4096, 8192),
        # Non-64-aligned N and/or K (e.g. from TP sharding)
        (8, 2576, 672),
        (128, 2576, 672),
        (8, 2576, 4096),
        (8, 4096, 2576),
        (8, 2576, 544),
        (1, 96, 80),
        (3, 176, 144),
        (128, 928, 1360),
    ])
def test_fp4_linear_marlin(quant_algo, dtype, mnk):
    if quant_algo == QuantAlgo.NVFP4 and get_sm_version() in (120, 121):
        pytest.skip(
            "Marlin backend shouldn't be used for NVFP4 quant on SM120/121")
    SEQ_LEN, OUTPUT_SIZE, HIDDEN_SIZE = mnk
    torch.manual_seed(0)

    w_float = torch.randn((OUTPUT_SIZE, HIDDEN_SIZE), dtype=torch.float32)
    w_fp4, w_sf_swizzled, w_dequant = torch.ops.tensorrt_llm.float_to_e2m1_and_ufp8sf_scale(
        w_float,
        scaling_vector_size,
        1,  # ufp8_type=1 (e4m3)
        True,  # is_sf_swizzled_layout=True (modelopt checkpoint native layout)
    )
    assert torch.iinfo(w_sf_swizzled.dtype).bits == 8  # torch.uint8
    w_sf_2d = torch.ops.trtllm.block_scale_interleave_reverse(
        w_sf_swizzled.view(pad_up(OUTPUT_SIZE, 128),
                           -1)).view(torch.float8_e4m3fn)

    with model_extra_attrs({'nvfp4_gemm_allowed_backends': ['marlin']}):
        l_marlin = Linear(
            in_features=HIDDEN_SIZE,
            out_features=OUTPUT_SIZE,
            bias=False,
            dtype=dtype,
            quant_config=QuantConfig(quant_algo=quant_algo),
            nvfp4_allowed_backends=['marlin'],  # key
        )

        # ``float_to_e2m1_and_ufp8sf_scale`` returns ``w_dequant`` that already
        # encodes the per-block FP8 scale. The Marlin BF16-activation path
        # multiplies the kernel output by ``weight_global_scale`` (derived from
        # ``weight_scale_2``); we want that scalar to be 1 so the kernel result
        # matches the reference ``torch.mm(x, w_dequant.T)``. Mirrors the
        # passing GEMM test (test_fp4_gemm.py:453-454, is_bf16_act=True branch).
        l_marlin.load_weights([{
            'weight':
            w_fp4,
            'weight_scale':
            w_sf_2d,
            'weight_scale_2':
            torch.tensor(1.0, dtype=torch.float32),
        }])
        l_marlin = l_marlin.cuda()

        l_marlin.post_load_weights()

        x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype).cuda()
        x_sf_global = (448 * 6) / x.abs().max().float()
        x_fp4, x_sf_block = torch.ops.trtllm.fp4_quantize(
            x, x_sf_global, scaling_vector_size, False)

        with torch.inference_mode():
            output = l_marlin(x)

    w_dequant_bf16 = w_dequant.to(dtype).cuda()
    with torch.inference_mode():
        ref_output = torch.mm(x, w_dequant_bf16.T)

    torch.cuda.synchronize()
    torch.testing.assert_close(output, ref_output, atol=0.5, rtol=2e-2)


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


def _make_nvfp4_inputs_for_bias_test(m, n, k, dtype=torch.bfloat16):
    """Quantize random bf16 act+weight; return (act_fp4, weight_fp4, act_sf, weight_sf, alpha)."""
    torch.manual_seed(0)
    act = torch.randn(m, k, dtype=dtype, device="cuda")
    weight = torch.randn(n, k, dtype=dtype, device="cuda")
    act_gs = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    w_gs = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    act_fp4, act_sf = torch.ops.trtllm.fp4_quantize(act, act_gs, 16, False,
                                                    True)
    w_fp4, w_sf = torch.ops.trtllm.fp4_quantize(weight, w_gs, 16, False, True)
    alpha = (1.0 / act_gs * 1.0 / w_gs).to(torch.float32).reshape(())
    return act_fp4, w_fp4, act_sf, w_sf, alpha


@skip_pre_blackwell
@pytest.mark.parametrize("backend", ["cutlass", "cublaslt", "cuda_core"])
@pytest.mark.parametrize("mnk", [(8, 4096, 4096), (252, 2048, 2048),
                                 (1024, 4096, 4096), (4096, 6144, 4096)])
def test_fp4_gemm_bias_per_backend(backend, mnk):
    """Per-backend numerical parity: nvfp4_gemm(bias=B) ≈ nvfp4_gemm(bias=None) + B."""
    m, n, k = mnk
    if backend == "cuda_core" and m > 8:
        pytest.skip("cuda_core backend only supports M <= 8")

    act_fp4, w_fp4, act_sf, w_sf, alpha = _make_nvfp4_inputs_for_bias_test(
        m, n, k)
    bias = torch.randn(n, dtype=torch.bfloat16, device="cuda") * 0.5

    # Request the unbiased GEMM in fp32 and add the bias in fp32 so the
    # reference rounds to bf16 exactly once, like the fused epilogue does.
    # A bf16 out_no_bias would round twice, and since 1 bf16 ULP is 2.0 once
    # |out| reaches ~256, cancellation against the bias promotes that discarded
    # remainder into a multi-ULP error that the fused (single-rounded) result
    # does not have.
    out_no_bias = torch.ops.trtllm.nvfp4_gemm(act_fp4,
                                              w_fp4,
                                              act_sf,
                                              w_sf,
                                              alpha,
                                              torch.float32,
                                              allowed_backends=backend)
    ref = (out_no_bias + bias.float()).to(torch.bfloat16)

    out_fused = torch.ops.trtllm.nvfp4_gemm(act_fp4,
                                            w_fp4,
                                            act_sf,
                                            w_sf,
                                            alpha,
                                            torch.bfloat16,
                                            allowed_backends=backend,
                                            bias=bias)

    # Tolerance is kept because the biased and unbiased calls may pick
    # different autotuner tactics, i.e. a different accumulation order.
    torch.testing.assert_close(out_fused, ref, rtol=1e-2, atol=5e-3)


def _skip_if_no_locality_domain():
    if not IS_CUTLASS_DSL_RUBIN_AVAILABLE:
        pytest.skip("CuTe DSL package with Rubin support is not available")
    is_locality_domain_enabled.cache_clear()
    if not is_locality_domain_enabled():
        pytest.skip(
            "locality domain localization is not enabled/supported on this system"
        )


def _create_fp4_weights(output_size, hidden_size, dtype):
    weight = torch.randn((output_size, hidden_size), dtype=dtype).cuda()
    weight_sf_global = (448 * 6) / weight.abs().max().float()
    weight_fp4, weight_sf_block = torch.ops.trtllm.fp4_quantize(
        weight, weight_sf_global, scaling_vector_size, False)
    weight_sf_block_unswizzled = (
        torch.ops.trtllm.block_scale_interleave_reverse(
            weight_sf_block.cpu().view(pad_up(output_size, 128), -1)))
    return weight_fp4, weight_sf_block, weight_sf_block_unswizzled, weight_sf_global


def _create_fp4_input(seq_len, hidden_size, dtype):
    input_raw = torch.randn(seq_len, hidden_size, dtype=dtype).cuda()
    input_sf_global = (448 * 6) / input_raw.abs().max().float()
    input_fp4, input_sf_block = torch.ops.trtllm.fp4_quantize(
        input_raw, input_sf_global, scaling_vector_size, False)
    return Fp4QuantizedTensor(input_fp4, input_sf_block), input_sf_global


def _make_locality_domain_weight_dict(weight_fp4,
                                      weight_sf_block_unswizzled,
                                      input_sf_global,
                                      weight_sf_global,
                                      bias=None):
    weight_dict = {
        "input_scale": 1.0 / input_sf_global.cpu(),
        "weight": weight_fp4.cpu(),
        "weight_scale": weight_sf_block_unswizzled.view(torch.float8_e4m3fn),
        "weight_scale_2": 1.0 / weight_sf_global.cpu(),
    }
    if bias is not None:
        weight_dict["bias"] = bias
    return [weight_dict]


@pytest.mark.skipif(
    get_sm_version() != 107,
    reason="This test is only supported on Rubin (SM 107) GPUs",
)
@pytest.mark.parametrize(
    "mnk",
    [(1, 7168, 2112), (256, 7168, 2112)],
)
def test_fp4_linear_locality_domain_correctness(mnk, tmp_path):
    _skip_if_no_locality_domain()
    from tensorrt_llm._torch.autotuner import AutoTuner, OptimizationProfile

    seq_len, output_size, hidden_size = mnk
    dtype = torch.bfloat16
    quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)

    locality_domain_linear = Linear(
        in_features=hidden_size,
        out_features=output_size,
        bias=False,
        dtype=dtype,
        quant_config=quant_config,
        nvfp4_allowed_backends=["cutedsl"],
        locality_domain_policy=LocalityDomainPolicy(enabled=True))
    assert locality_domain_linear.partition_plan.enabled

    weight_fp4, weight_sf, weight_sf_unswizzled, weight_sf_global = (
        _create_fp4_weights(output_size, hidden_size, dtype))
    input_tensor, input_sf_global = _create_fp4_input(seq_len, hidden_size,
                                                      dtype)
    weight_dict = _make_locality_domain_weight_dict(weight_fp4,
                                                    weight_sf_unswizzled,
                                                    input_sf_global,
                                                    weight_sf_global)

    locality_domain_linear.load_weights(weight_dict)
    locality_domain_linear = locality_domain_linear.cuda()
    locality_domain_linear.post_load_weights()

    tuner = AutoTuner.get()
    old_settings = (tuner.warmup, tuner.repeat, tuner.stream_delay_micro_secs)
    tuner.warmup = 0
    tuner.repeat = 1
    tuner.stream_delay_micro_secs = 10
    try:
        tuner.clear_cache()
        with torch.inference_mode():
            output_base = torch.ops.trtllm.nvfp4_gemm_cutlass(
                input_tensor.fp4_tensor,
                weight_fp4,
                input_tensor.scaling_factor,
                weight_sf,
                1.0 / (input_sf_global * weight_sf_global),
                dtype,
            )

        tuner.clear_cache()
        tuner.reset_statistics()
        with torch.inference_mode(), autotune(
                cache_path=str(tmp_path / "locality_domain.json")):
            output_locality_domain = locality_domain_linear.forward(
                input_tensor)

        if seq_len in (1, 256):
            op_name = (
                "trtllm::cute_dsl_nvfp4_gemm_locality_domain_inplace_rubin"
                "::locality_domain_concurrent")
            assert tuner.stats.tuned_op_profiled_configs.get(
                op_name, 0) > 0, str(tuner.stats)
            assert not tuner.stats.failed_profiling_count.get(op_name, set())

            with tuner.capture() as tactics_capture, torch.inference_mode():
                output_locality_domain = locality_domain_linear.forward(
                    input_tensor)

            assert len(tactics_capture._captured_contexts) == 1
            context = tactics_capture._captured_contexts[0]
            assert context["custom_op"] == op_name
            concurrent_runner = context["runners"][0]
            tactics = concurrent_runner.get_valid_tactics(
                context["inputs"], OptimizationProfile())
            base_tactics = [tactic for tactic in tactics if tactic[0] == "base"]
            mixed_tactics = [
                tactic for tactic in tactics if tactic[0] == "mixed_clusters"
            ]

            if seq_len == 1:
                assert {tactic[4] for tactic in base_tactics} == {True}
                assert not mixed_tactics
                replay_tactics = [base_tactics[0]]
            else:
                assert {tactic[4] for tactic in base_tactics} == {False}
                assert {tactic[5] for tactic in mixed_tactics} == {True}
                replay_tactics = [base_tactics[0], mixed_tactics[0]]

            for tactic in replay_tactics:
                with torch.inference_mode():
                    context["inputs"][-1].zero_()
                    concurrent_runner(context["inputs"], tactic=tactic)
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    output_base,
                    context["inputs"][-1][:, :output_size],
                    rtol=1e-2,
                    atol=0.15,
                )
    finally:
        tuner.warmup, tuner.repeat, tuner.stream_delay_micro_secs = old_settings

    torch.cuda.synchronize()
    assert locality_domain_linear.partition_plan.num_partitions == 2
    torch.testing.assert_close(output_base,
                               output_locality_domain,
                               rtol=1e-2,
                               atol=0.15)


@pytest.mark.skipif(
    get_sm_version() != 107,
    reason="This test is only supported on Rubin (SM 107) GPUs",
)
@pytest.mark.parametrize("input_kind", ["quantized-tensor", "tuple"])
def test_fp4_linear_locality_domain_rank3_prequantized_input(input_kind):
    _skip_if_no_locality_domain()
    torch.manual_seed(0)

    batch_size, seq_len = 2, 3
    output_size, hidden_size = 192, 128
    dtype = torch.bfloat16
    common_kwargs = {
        "in_features": hidden_size,
        "out_features": output_size,
        "bias": False,
        "dtype": dtype,
        "quant_config": QuantConfig(quant_algo=QuantAlgo.NVFP4),
        "nvfp4_allowed_backends": ["cutlass"],
    }
    base_linear = Linear(**common_kwargs,
                         locality_domain_policy=LocalityDomainPolicy(
                             enabled=False))
    locality_domain_linear = Linear(**common_kwargs,
                                    locality_domain_policy=LocalityDomainPolicy(
                                        enabled=True))

    input_flat, input_sf_global = _create_fp4_input(batch_size * seq_len,
                                                    hidden_size, dtype)
    reference_input = Fp4QuantizedTensor(
        input_flat.fp4_tensor.reshape(batch_size, seq_len,
                                      input_flat.fp4_tensor.shape[-1]),
        input_flat.scaling_factor,
        input_flat.is_sf_swizzled,
    )
    input_tensor = (reference_input if input_kind == "quantized-tensor" else
                    (reference_input.fp4_tensor,
                     reference_input.scaling_factor))
    weight_fp4, _, weight_sf_unswizzled, weight_sf_global = (
        _create_fp4_weights(output_size, hidden_size, dtype))
    weight_dict = _make_locality_domain_weight_dict(
        weight_fp4,
        weight_sf_unswizzled,
        input_sf_global,
        weight_sf_global,
    )
    base_linear.load_weights(weight_dict)
    base_linear = base_linear.cuda()
    base_linear.post_load_weights()
    locality_domain_linear.load_weights(weight_dict)
    locality_domain_linear = locality_domain_linear.cuda()
    locality_domain_linear.post_load_weights()
    assert locality_domain_linear.partition_plan.enabled

    with torch.inference_mode(), autotune():
        output_base = base_linear(reference_input)
        output_locality_domain = locality_domain_linear(input_tensor)

    assert output_locality_domain.shape == (batch_size, seq_len, output_size)
    torch.testing.assert_close(output_locality_domain,
                               output_base,
                               rtol=1e-2,
                               atol=0.15)


@pytest.mark.skipif(
    get_sm_version() != 107,
    reason="This test is only supported on Rubin (SM 107) GPUs",
)
def test_fp4_fused_gate_up_linear_locality_domain_correctness():
    _skip_if_no_locality_domain()
    torch.manual_seed(0)

    m, intermediate_size, hidden_size = 1, 1024, 2048
    output_size = 2 * intermediate_size
    dtype = torch.bfloat16
    weights_loading_config = WeightsLoadingConfig(
        weight_mode=WeightMode.FUSED_GATE_UP_LINEAR)
    locality_domain_linear = Linear(
        in_features=hidden_size,
        out_features=output_size,
        bias=False,
        dtype=dtype,
        quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4),
        weights_loading_config=weights_loading_config,
        fused_weight_shard_indices_mapping={
            "gate": (0, intermediate_size),
            "up": (intermediate_size, intermediate_size),
        },
        use_cute_dsl_blockscaling_mm=True,
        nvfp4_allowed_backends=["cutlass"],
        locality_domain_policy=LocalityDomainPolicy(enabled=True),
    )
    assert locality_domain_linear.partition_plan.enabled

    input_tensor, input_sf_global = _create_fp4_input(m, hidden_size, dtype)
    full_weight = torch.randn(output_size,
                              hidden_size,
                              dtype=dtype,
                              device="cuda")
    weight_sf_global = (448 * 6) / full_weight.abs().max().float()
    weight_dicts = []
    quantized_weights = []
    quantized_weight_scales = []
    for weight in full_weight.chunk(2, dim=0):
        weight_fp4, weight_sf = torch.ops.trtllm.fp4_quantize(
            weight, weight_sf_global, scaling_vector_size, False)
        weight_sf_unswizzled = (torch.ops.trtllm.block_scale_interleave_reverse(
            weight_sf.cpu().view(pad_up(intermediate_size, 128), -1)))
        quantized_weights.append(weight_fp4)
        quantized_weight_scales.append(weight_sf)
        weight_dicts.append(
            _make_locality_domain_weight_dict(
                weight_fp4,
                weight_sf_unswizzled,
                input_sf_global,
                weight_sf_global,
            )[0])

    locality_domain_linear.load_weights(weight_dicts)
    for attr in (
            "tmp_nvfp4_weight_scales",
            "tmp_nvfp4_input_scales_list",
            "tmp_nvfp4_weight_scale_2_list",
            "tmp_nvfp4_pre_quant_scale",
    ):
        assert not hasattr(locality_domain_linear, attr)
    locality_domain_linear = locality_domain_linear.cuda()
    locality_domain_linear.post_load_weights()

    shards = locality_domain_linear._locality_domain_weight_shards
    assert shards is not None
    for shard, weight, weight_scale in zip(shards, quantized_weights,
                                           quantized_weight_scales):
        assert torch.equal(shard["weight"], weight)
        assert torch.equal(shard["weight_scale"], weight_scale)
    assert locality_domain_linear.weight.numel() == 0
    assert locality_domain_linear.weight_scale.numel() == 0

    with torch.inference_mode(), autotune():
        alpha = 1.0 / (input_sf_global * weight_sf_global)
        base_gate_up = torch.cat([
            torch.ops.trtllm.nvfp4_gemm_cutlass(
                input_tensor.fp4_tensor,
                weight,
                input_tensor.scaling_factor,
                weight_scale,
                alpha,
                dtype,
            ) for weight, weight_scale in zip(quantized_weights,
                                              quantized_weight_scales)
        ],
                                 dim=-1)
        locality_domain_gate_up = locality_domain_linear(input_tensor)
        base_activated = swiglu(base_gate_up)
        locality_domain_activated = swiglu(locality_domain_gate_up)

    torch.cuda.synchronize()
    torch.testing.assert_close(locality_domain_gate_up,
                               base_gate_up,
                               rtol=1e-2,
                               atol=0.15)
    torch.testing.assert_close(locality_domain_activated,
                               base_activated,
                               rtol=3e-2,
                               atol=2.0)


@pytest.mark.skipif(
    get_sm_version() != 107,
    reason="This test is only supported on Rubin (SM 107) GPUs",
)
@pytest.mark.parametrize(
    ("output_size", "hidden_size"),
    [
        pytest.param(7168, 2112, id="aligned-production-shape"),
        pytest.param(192, 128, id="padded-scale-reload"),
    ],
)
def test_fp4_linear_locality_domain_weight_lifecycle_and_global_bias(
        output_size, hidden_size):
    _skip_if_no_locality_domain()
    torch.manual_seed(0)

    m = 8
    dtype = torch.bfloat16
    linear = Linear(
        in_features=hidden_size,
        out_features=output_size,
        bias=True,
        dtype=dtype,
        quant_config=QuantConfig(quant_algo=QuantAlgo.NVFP4),
        nvfp4_allowed_backends=["cutedsl"],
        locality_domain_policy=LocalityDomainPolicy(enabled=True),
    )
    assert linear.partition_plan.enabled

    input_tensor, input_sf_global = _create_fp4_input(m, hidden_size, dtype)

    def make_weight_generation():
        weight_fp4, weight_sf, weight_sf_unswizzled, weight_sf_global = (
            _create_fp4_weights(output_size, hidden_size, dtype))
        bias = torch.randn(output_size, dtype=dtype)
        weight_dict = _make_locality_domain_weight_dict(
            weight_fp4,
            weight_sf_unswizzled,
            input_sf_global,
            weight_sf_global,
            bias,
        )
        return (weight_fp4, weight_sf, weight_sf_unswizzled, weight_sf_global,
                bias, weight_dict)

    def check_output(weight_fp4, weight_sf, weight_sf_global):
        with torch.inference_mode():
            output_no_bias = linear.apply_linear(input_tensor, None)
            output = linear(input_tensor)
            reference = torch.ops.trtllm.nvfp4_gemm_cutlass(
                input_tensor.fp4_tensor,
                weight_fp4,
                input_tensor.scaling_factor,
                weight_sf,
                1.0 / (input_sf_global * weight_sf_global),
                dtype,
            )
        torch.testing.assert_close(output_no_bias,
                                   reference,
                                   rtol=1e-2,
                                   atol=0.15)
        torch.testing.assert_close(output,
                                   output_no_bias + linear.bias,
                                   rtol=1e-2,
                                   atol=0.15)

    first_generation = make_weight_generation()
    first_weight, first_scale, _, first_scale_global, first_bias, weight_dict = (
        first_generation)
    linear.load_weights(weight_dict)
    linear = linear.cuda()
    full_weight = linear.weight.data.clone()
    full_bias = linear.bias.data.clone()
    linear.post_load_weights()
    shards = linear._locality_domain_weight_shards
    assert shards is not None
    assert len(shards) == linear.partition_plan.num_partitions

    layout = linear.partition_plan.layout
    assert layout is not None
    assert layout.padded_axis_extent == full_weight.size(0)
    partition_n = layout.per_partition_axis_extent(padded=True)
    for shard in shards:
        assert shard["weight"].shape == (partition_n, full_weight.shape[1])

    reconstructed = torch.cat([shard["weight"] for shard in shards], dim=0)
    assert torch.equal(reconstructed.cpu(), full_weight.cpu())
    assert linear.weight.numel() == 0
    assert linear.weight_scale.numel() == 0
    assert torch.equal(linear.bias.cpu(), full_bias.cpu())
    torch.testing.assert_close(full_bias.cpu(), first_bias)
    assert all(set(shard) == {"weight", "weight_scale"} for shard in shards)
    assert all("param" not in metadata
               for metadata in linear.rebuild_tensor_metadata.values())
    check_output(first_weight, first_scale, first_scale_global)

    original_weight_shape = (output_size, hidden_size // 2)
    original_scale_shape = tuple(
        linear.rebuild_tensor_metadata["weight_scale"]["meta"].shape)
    linear.pre_reload_weights()
    assert linear._locality_domain_weight_shards is None
    assert tuple(linear.weight.shape) == original_weight_shape
    assert tuple(linear.weight_scale.shape) == original_scale_shape
    assert linear.rebuild_tensor_metadata == {}

    second_generation = make_weight_generation()
    (second_weight, second_scale, second_scale_unswizzled, second_scale_global,
     second_bias, second_weight_dict) = second_generation
    linear.load_weights(second_weight_dict)
    linear.post_load_weights()
    second_shards = linear._locality_domain_weight_shards
    assert second_shards is not None
    assert second_shards is not shards
    assert torch.equal(
        torch.cat([shard["weight"] for shard in second_shards], dim=0),
        second_weight,
    )

    logical_scale_parts = []
    shard_n = output_size // len(second_shards)
    for shard in second_shards:
        padded_rows = pad_up(shard_n, 128)
        scale = shard["weight_scale"].view(padded_rows, -1)
        scale = torch.ops.trtllm.block_scale_interleave_reverse(scale)
        logical_scale_parts.append(scale[:shard_n])
    reloaded_scale = torch.cat(logical_scale_parts, dim=0)
    torch.testing.assert_close(reloaded_scale.cpu(),
                               second_scale_unswizzled[:output_size])
    assert linear.weight.numel() == 0
    assert linear.weight_scale.numel() == 0
    assert all("bias" not in shard for shard in second_shards)
    torch.testing.assert_close(linear.bias.cpu(), second_bias)
    check_output(second_weight, second_scale, second_scale_global)


@pytest.mark.skipif(get_sm_version() != 107
                    or not IS_CUTLASS_DSL_RUBIN_AVAILABLE,
                    reason="Requires Rubin CuTe DSL")
def test_cute_dsl_nvfp4_non_inplace_rubin():
    torch.manual_seed(47)
    x, x_scale = _create_fp4_input(8, 128, torch.bfloat16)
    w, w_sf, _, w_scale = _create_fp4_weights(256, 128, torch.bfloat16)
    alpha = 1.0 / (x_scale * w_scale)
    args = (x.fp4_tensor, w, x.scaling_factor, w_sf, alpha, torch.bfloat16)
    expected = torch.ops.trtllm.nvfp4_gemm_cutlass(*args)
    op = torch.ops.trtllm.cute_dsl_nvfp4_gemm_rubin
    output = op(*args)
    torch.testing.assert_close(output, expected, rtol=1e-2, atol=0.15)
    assert output.shape == (8, 256)
    assert output.dtype == torch.bfloat16

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(*args)
    captured.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(captured, expected, rtol=1e-2, atol=0.15)

    with pytest.raises(ValueError, match="partition_id must be -1"):
        op(*args, partition_id=0)
    with pytest.raises(ValueError, match="must use"):
        op(*args, output_tensor=output)
