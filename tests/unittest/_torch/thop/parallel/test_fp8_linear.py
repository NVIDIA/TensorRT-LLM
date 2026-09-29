import os

import pytest
import torch
from _torch.helpers import calc_diff
from utils.util import skip_pre_hopper

from tensorrt_llm._torch.autotuner import AutoTuner
from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_RUBIN_AVAILABLE
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


@skip_pre_hopper
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_cute_dsl", [True, False])
def test_fp8_linear(dtype, use_cute_dsl, monkeypatch):
    SEQ_LEN = 10
    HIDDEN_SIZE = 128
    OUTPUT_SIZE = 256
    torch.manual_seed(0)
    x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype).cuda()
    x_fp8, x_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(x)
    x_fp8 = x_fp8.view(torch.float8_e4m3fn)
    x_scale = x_scale.float().squeeze()
    w = torch.randn((OUTPUT_SIZE, HIDDEN_SIZE), dtype=dtype).cuda()
    w_fp8, w_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(w)
    w_fp8 = w_fp8.view(torch.float8_e4m3fn)
    w_scale = w_scale.float().squeeze()

    monkeypatch.setenv("USE_CUTE_DSL_FP8_PER_TENSOR_MM",
                       "1" if use_cute_dsl else "0")
    qc = QuantConfig(quant_algo=QuantAlgo.FP8)
    l0 = Linear(in_features=HIDDEN_SIZE,
                out_features=OUTPUT_SIZE,
                bias=False,
                dtype=dtype,
                quant_config=qc)
    assert l0.weight.dtype == torch.float8_e4m3fn
    l0.load_weights([{
        'weight': w_fp8,
        'weight_scale': w_scale,
        'input_scale': x_scale
    }])
    l0.cuda()
    torch.testing.assert_close(l0.weight, w_fp8)
    torch.testing.assert_close(l0.weight_scale, w_scale)
    torch.testing.assert_close(l0.input_scale, x_scale)

    with torch.inference_mode():
        output = l0.forward(x)

    # torch run
    def ref_quant(x_, x_scale_):
        x_ = x_.float()
        finfo = torch.finfo(torch.float8_e4m3fn)
        inv_scale = x_scale_.reciprocal()
        x_fp8_ = (x_ * inv_scale).clamp(min=finfo.min, max=finfo.max)
        return x_fp8_.to(torch.float8_e4m3fn)

    def ref_linear():
        ref_x_fp8 = ref_quant(x, x_scale)
        # Align cublaslt workspace size with trtllm's 32MB.
        # Details see in test_scaled_mm.py
        old_env = os.environ.get("CUBLASLT_WORKSPACE_SIZE", "")
        os.environ["CUBLASLT_WORKSPACE_SIZE"] = f"{32*1024}"
        ref_output = torch._scaled_mm(ref_x_fp8,
                                      w_fp8.t(),
                                      out_dtype=dtype,
                                      scale_a=x_scale,
                                      scale_b=w_scale,
                                      use_fast_accum=True,
                                      bias=l0.bias)
        os.environ["CUBLASLT_WORKSPACE_SIZE"] = old_env
        return ref_output

    with torch.inference_mode():
        ref_output = ref_linear()

    # compare
    torch.cuda.synchronize()
    torch.testing.assert_close(output, ref_output)


@pytest.mark.skipif(
    get_sm_version() != 107 or not IS_CUTLASS_DSL_RUBIN_AVAILABLE,
    reason="The test requires SM107 and SM107 CuTe DSL support.",
)
@pytest.mark.parametrize(
    "k, n",
    [(7168, 2112), (1536, 24576), (512, 32768), (2048, 7168), (1024, 1024)],
)
@pytest.mark.parametrize(
    "m",
    [7, 64, 4096],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("use_tvm_ffi", [True, False])
def test_cute_dsl_per_tensor_fp8(dtype, use_tvm_ffi, m, n, k):
    torch.random.manual_seed(0)
    a = torch.randn((m, k), device='cuda', dtype=dtype) / k
    b = torch.randn((n, k), device='cuda', dtype=dtype) / k

    a_fp8, a_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(a)
    b_fp8, b_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(b)
    # convert to float32
    a_scale = a_scale.to(torch.float32)
    b_scale = b_scale.to(torch.float32)

    output_expected = a @ b.t()

    with AutoTuner.get().capture() as capture, torch.inference_mode():
        output = torch.ops.trtllm.cute_dsl_fp8_per_tensor_gemm_rubin(
            a_fp8,
            b_fp8,
            a_scale,
            b_scale,
            output_dtype=dtype,
            use_tvm_ffi=use_tvm_ffi,
        )

    tactics_list = list(capture)
    print(f"  Found {len(tactics_list)} tactics.")

    for tactic_idx, tactic in enumerate(tactics_list):
        runner, tactic_value = tactic[0]
        runner_name = runner.__class__.__name__

        with AutoTuner.get().replay(tactic), torch.inference_mode():
            output = torch.ops.trtllm.cute_dsl_fp8_per_tensor_gemm_rubin(
                a_fp8,
                b_fp8,
                a_scale,
                b_scale,
                output_dtype=dtype,
                use_tvm_ffi=use_tvm_ffi,
            )

            diff = calc_diff(output, output_expected)
            assert diff < 2e-3, f"Tactic {tactic_idx+1}/{len(tactics_list)}: {runner_name} tactic={tactic_value} - FAILED"
            torch.testing.assert_close(output,
                                       output_expected,
                                       atol=1e-3,
                                       rtol=1e-3)
