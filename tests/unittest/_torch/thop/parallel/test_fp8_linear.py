import os

import pytest
import torch
from utils.util import skip_pre_hopper

from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


@skip_pre_hopper
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fp8_linear(dtype):
    SEQ_LEN = 10
    HIDDEN_SIZE = 128
    torch.manual_seed(0)
    x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype).cuda()
    x_fp8, x_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(x)
    x_fp8 = x_fp8.view(torch.float8_e4m3fn)
    x_scale = x_scale.float().squeeze()
    w = torch.randn((HIDDEN_SIZE, HIDDEN_SIZE), dtype=dtype).cuda()
    w_fp8, w_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(w)
    w_fp8 = w_fp8.view(torch.float8_e4m3fn)
    w_scale = w_scale.float().squeeze()

    qc = QuantConfig(quant_algo=QuantAlgo.FP8)
    l0 = Linear(in_features=HIDDEN_SIZE,
                out_features=HIDDEN_SIZE,
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


if __name__ == '__main__':
    test_fp8_linear(torch.float16)
