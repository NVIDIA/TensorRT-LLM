import pytest
import torch
from torch.nn.parameter import Parameter

import tensorrt_llm.quantization.functional
from tensorrt_llm._torch.autotuner import autotune
from tensorrt_llm._torch.custom_ops.torch_custom_ops import \
    FinegrainedMixedDtypeGemm
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


@pytest.mark.parametrize("weights_dtype", [torch.uint8])
@pytest.mark.parametrize(
    "dtype",
    [torch.float16],
)
def test_w4a8_linear(dtype, weights_dtype, has_zero=False):

    if get_sm_version() > FinegrainedMixedDtypeGemm.MAX_SUPPORTED_SM_VERSION:
        pytest.skip(
            f"W4A16/W4A8 is not supported in this SM version {get_sm_version()}"
        )

    SEQ_LEN = 10
    HIDDEN_SIZE = 128
    OUTPUT_SIZE = 512
    GROUP_SIZE = 128
    torch.manual_seed(0)

    total_groups = (HIDDEN_SIZE + GROUP_SIZE - 1) // GROUP_SIZE

    x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype).cuda()
    w = torch.randint(0,
                      2**32 - 1, (HIDDEN_SIZE, OUTPUT_SIZE // 8),
                      dtype=torch.uint32,
                      device=x.device)
    w = w.view(weights_dtype)

    pre_quant_scale = torch.rand(HIDDEN_SIZE, dtype=dtype).cuda()
    weight_scale = torch.rand(total_groups, OUTPUT_SIZE,
                              dtype=torch.float16).cuda()
    weight_scale_2 = torch.rand(1, dtype=torch.float32).cuda()
    input_scale = Parameter(torch.tensor(1., dtype=torch.float32),
                            requires_grad=False).cuda()
    bias = torch.randn(OUTPUT_SIZE, dtype=dtype).cuda().contiguous()

    qc = QuantConfig(quant_algo=QuantAlgo.W4A8_AWQ,
                     group_size=GROUP_SIZE,
                     has_zero_point=has_zero)

    linear_w4a8 = Linear(in_features=HIDDEN_SIZE,
                         out_features=OUTPUT_SIZE,
                         bias=True,
                         dtype=dtype,
                         quant_config=qc)

    linear_w4a8.load_weights([{
        'pre_quant_scale': pre_quant_scale,
        'weight': w.T.clone(),
        'weight_scale': weight_scale.T,
        'bias': bias,
        'weight_scale_2': weight_scale_2,
        'input_scale': input_scale
    }])

    linear_w4a8 = linear_w4a8.cuda()

    preprocessor = tensorrt_llm.quantization.functional.preprocess_weights_for_mixed_gemm
    w = preprocessor(
        w.to(torch.int8).contiguous().cpu(), torch.quint4x2,
        torch.float8_e4m3fn).cuda().contiguous()

    torch.testing.assert_close(linear_w4a8.weight, w)

    with torch.inference_mode(), autotune():
        output = linear_w4a8.forward(x)

    # ref linear
    with torch.inference_mode():
        x = x * pre_quant_scale

        quantized_input, _ = torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(
            x, (input_scale))
        alpha = (weight_scale_2.float() * input_scale.float()).item()

        output_ref = torch.ops.trtllm.finegrained_mixed_dtype_gemm(
            input=quantized_input.contiguous(),
            weight=w.contiguous(),
            scales=(weight_scale / weight_scale_2).to(
                torch.float16).contiguous(),
            group_size=GROUP_SIZE,
            has_zero_point=has_zero,
            output_dtype=x.dtype,
            alpha=alpha,
            bias=bias,
            zeros=None)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, output_ref)
