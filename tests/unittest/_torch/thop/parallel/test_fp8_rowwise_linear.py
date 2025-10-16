import pytest
import torch
from utils.util import skip_blackwell, skip_pre_hopper

from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


@skip_blackwell
@skip_pre_hopper
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fp8_rowwise_linear(dtype):
    SEQ_LEN = 10
    HIDDEN_SIZE = 128
    torch.manual_seed(0)
    x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype).cuda()
    x_fp8, x_scale = torch.ops.tensorrt_llm.quantize_e4m3_activation(x)
    x_fp8 = x_fp8.view(torch.float8_e4m3fn)
    x_scale = x_scale.float().squeeze()
    w = torch.randn((HIDDEN_SIZE, HIDDEN_SIZE), dtype=dtype).cuda()
    w_fp8, w_scale = torch.ops.tensorrt_llm.quantize_e4m3_activation(w)
    w_fp8 = w_fp8.view(torch.float8_e4m3fn)
    w_scale = w_scale.float().squeeze()

    qc = QuantConfig(quant_algo=QuantAlgo.FP8_PER_CHANNEL_PER_TOKEN)
    l0 = Linear(in_features=HIDDEN_SIZE,
                out_features=HIDDEN_SIZE,
                bias=False,
                dtype=dtype,
                quant_config=qc)
    assert l0.weight.dtype == torch.float8_e4m3fn
    l0.load_weights([{
        'weight': w_fp8,
        'weight_scale': w_scale,
    }])
    l0.cuda()
    torch.testing.assert_close(l0.weight, w_fp8)
    torch.testing.assert_close(l0.weight_scale, w_scale)

    with torch.inference_mode():
        output = l0.forward(x)

    with torch.inference_mode():
        x_dq = x_fp8.to(x_scale.dtype) * x_scale.view(-1, 1)
        w_dq = w_fp8.to(w_scale.dtype).t() * w_scale.view(1, -1)
        ref_output = x_dq.to(dtype) @ w_dq.to(dtype)

    # compare
    torch.cuda.synchronize()
    torch.testing.assert_close(output, ref_output, atol=6e-2, rtol=1e-2)


if __name__ == '__main__':
    test_fp8_rowwise_linear(torch.float16)
