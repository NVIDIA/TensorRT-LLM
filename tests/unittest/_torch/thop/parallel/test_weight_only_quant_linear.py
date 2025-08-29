import pytest
import torch

from tensorrt_llm._torch.autotuner import autotune
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


@pytest.mark.parametrize("weights_dtype", [torch.int8, torch.quint4x2])
@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16],
)
def test_weight_only_quant_linear(dtype, weights_dtype):

    SEQ_LEN = 10
    HIDDEN_SIZE = 128
    OUT_FEATURES = 64
    torch.manual_seed(0)
    x = torch.randn((SEQ_LEN, HIDDEN_SIZE), dtype=dtype, device="cuda")
    w = torch.rand(
        (HIDDEN_SIZE, OUT_FEATURES), dtype=dtype, device="cuda") * 2 - 1.0

    # w: int8 or int4x2 weight, w_processed: preprocessed weight, w_scales: scale of w
    w, w_processed, w_scales = torch.ops.trtllm._symmetric_quantize_last_axis_of_batched_matrix(
        w.cpu(), weights_dtype)
    w = w.cuda()
    w_processed = w_processed.cuda()
    w_scales = w_scales.cuda()

    if weights_dtype == torch.int8:
        qc = QuantConfig(quant_algo=QuantAlgo.W8A16, group_size=1)
    elif weights_dtype == torch.quint4x2:
        qc = QuantConfig(quant_algo=QuantAlgo.W4A16, group_size=1)
    else:
        raise ValueError(f"Unsupported weights_dtype: {weights_dtype}")

    linear_woq = Linear(in_features=HIDDEN_SIZE,
                        out_features=OUT_FEATURES,
                        bias=False,
                        dtype=dtype,
                        quant_config=qc)

    linear_woq.load_weights([{
        'weight': w.T,
        'weight_scale': w_scales,
    }])

    linear_woq = linear_woq.cuda()

    torch.testing.assert_close(linear_woq.weight, w_processed)

    with torch.inference_mode(), autotune():
        output = linear_woq.forward(x)

    # ref linear
    with torch.inference_mode():
        output_ref = torch.ops.trtllm.weight_only_quant_gemm(
            x.contiguous(), w_processed, weights_dtype, w_scales, dtype)
    torch.cuda.synchronize()
    torch.testing.assert_close(output, output_ref)
