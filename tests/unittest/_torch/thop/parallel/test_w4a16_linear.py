import pytest
import torch

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
def test_w4a16_linear(dtype, weights_dtype, has_zero=False):

    if get_sm_version() > FinegrainedMixedDtypeGemm.MAX_SUPPORTED_SM_VERSION:
        pytest.skip(
            f"W4A16/W4A8 is not supported in this SM version {get_sm_version()}"
        )

    SEQ_LEN = 10
    HIDDEN_SIZE = 128
    GROUP_SIZE = 128
    torch.manual_seed(0)

    total_groups = (HIDDEN_SIZE + GROUP_SIZE - 1) // GROUP_SIZE

    x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype).cuda()
    w = torch.randint(0,
                      2**32 - 1, (HIDDEN_SIZE, HIDDEN_SIZE // 8),
                      dtype=torch.uint32,
                      device=x.device)
    w = w.view(weights_dtype)

    pre_quant_scale = torch.rand(HIDDEN_SIZE, dtype=dtype).cuda()
    weight_scale = torch.rand(total_groups, HIDDEN_SIZE,
                              dtype=torch.float32).cuda()
    bias = torch.randn(HIDDEN_SIZE, dtype=dtype).cuda().contiguous()

    qc = QuantConfig(quant_algo=QuantAlgo.W4A16_AWQ,
                     group_size=GROUP_SIZE,
                     has_zero_point=has_zero)
    linear_w4a16 = Linear(in_features=HIDDEN_SIZE,
                          out_features=HIDDEN_SIZE,
                          bias=True,
                          dtype=dtype,
                          quant_config=qc)

    linear_w4a16.load_weights([{
        'pre_quant_scale': pre_quant_scale,
        'weight': w.T,
        'weight_scale': weight_scale.T,
        'bias': bias
    }])

    linear_w4a16 = linear_w4a16.cuda()

    w = w.to(torch.int8)
    preprocessor = tensorrt_llm.quantization.functional.preprocess_weights_for_mixed_gemm
    w = preprocessor(w.contiguous().cpu(), torch.quint4x2,
                     x.dtype).cuda().contiguous()

    torch.testing.assert_close(linear_w4a16.weight, w)

    with torch.inference_mode(), autotune():
        output = linear_w4a16.forward(x)

    # ref linear
    with torch.inference_mode():
        pre_quant_scale = pre_quant_scale.repeat(SEQ_LEN, 1)
        x = torch.mul(x, pre_quant_scale)

        output_ref = torch.ops.trtllm.finegrained_mixed_dtype_gemm(
            input=x.contiguous(),
            weight=w,
            scales=weight_scale.type(x.dtype),
            group_size=GROUP_SIZE,
            has_zero_point=has_zero,
            bias=bias,
            output_dtype=x.dtype,
            zeros=None)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, output_ref)
