import pickle
import sys

import cloudpickle
import pytest
import torch
from mpi4py import MPI
from utils.util import check_accuracy, skip_pre_hopper

from tensorrt_llm._torch.modules.fused_moe.fused_moe_triton import \
    IS_TRITON_KERNELS_AVAILABLE
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.modules.triton_linear import TritonLinear
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(
    cloudpickle.dumps,
    cloudpickle.loads,
    pickle.HIGHEST_PROTOCOL,
)


@pytest.mark.parametrize("linear_cls", [Linear, TritonLinear])
def test_linear_unquantized(linear_cls):
    if not IS_TRITON_KERNELS_AVAILABLE and linear_cls is TritonLinear:
        pytest.skip("Triton kernels are not available")

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    num_tokens = 128
    hidden_size = 64
    out_size = 256
    dtype = torch.bfloat16
    x = torch.randn((num_tokens, hidden_size), dtype=dtype).cuda()
    w = torch.randn((hidden_size, out_size), dtype=dtype).cuda()
    b = torch.randn((out_size, ), dtype=dtype).cuda()

    weights = {
        "weight": w.T,  # Transpose to match TRT-LLM's weight shape
        "bias": b,
    }

    linear = linear_cls(
        in_features=hidden_size,
        out_features=out_size,
        bias=True,
        dtype=dtype,
    )
    linear.load_weights([weights])
    linear.cuda()

    actual_c = linear.forward(x)
    reference_c = torch.matmul(x, w) + b

    check_accuracy(actual_c, reference_c, atol=0.01, rtol=0.01, percent=0.99)


@pytest.mark.parametrize("linear_cls", [Linear, TritonLinear])
def test_linear_fp8qdq(linear_cls):
    if not IS_TRITON_KERNELS_AVAILABLE and linear_cls is TritonLinear:
        pytest.skip("Triton kernels are not available")

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    num_tokens = 128
    hidden_size = 64
    out_size = 256
    dtype = torch.bfloat16
    x = torch.randn((num_tokens, hidden_size), dtype=dtype).cuda()
    qx, sx = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(x)
    w = torch.randn((hidden_size, out_size), dtype=dtype).cuda()
    qw, sw = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(w)
    b = torch.randn((out_size, ), dtype=dtype).cuda()

    weights = {
        "weight": qw.T,  # Transpose to match TRT-LLM's weight shape
        "bias": b,
        "input_scale": sx,
        "weight_scale": sw,
    }

    linear = linear_cls(in_features=hidden_size,
                        out_features=out_size,
                        bias=True,
                        dtype=dtype,
                        quant_config=QuantConfig(quant_algo=QuantAlgo.FP8))
    linear.load_weights([weights])
    linear.cuda()

    actual_c = linear.forward(qx)
    x_qdq = qx.to(torch.float32) * sx
    w_qdq = qw.to(torch.float32) * sw
    reference_c = torch.matmul(x_qdq, w_qdq) + b

    check_accuracy(actual_c,
                   reference_c.to(dtype),
                   atol=0.01,
                   rtol=0.01,
                   percent=0.99)


@skip_pre_hopper
@pytest.mark.parametrize("activation_dtype",
                         [torch.bfloat16, torch.float8_e4m3fn])
def test_linear_mxfp4(activation_dtype):
    if not IS_TRITON_KERNELS_AVAILABLE:
        pytest.skip("Triton kernels are not available")
    if torch.cuda.get_device_capability(
    )[0] < 10 and activation_dtype == torch.float8_e4m3fn:
        pytest.skip("Latest Triton requires BF16 activation on Hopper")
    if torch.cuda.get_device_capability(
    )[0] >= 10 and activation_dtype == torch.bfloat16:
        pytest.skip("Latest Triton requires FP8 activation on Blackwell")

    dtype = torch.bfloat16
    num_tokens = 128
    hidden_size = 256  # Must be even and divisible by 32 for MXFP4
    out_size = 512
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    x = torch.randn((num_tokens, hidden_size), dtype=dtype).cuda()
    w = torch.randn((hidden_size, out_size), dtype=dtype).cuda()
    b = torch.randn((out_size, ), dtype=dtype).cuda()

    from triton_kernels.numerics_details.mxfp import (downcast_to_mxfp_torch,
                                                      upcast_from_mxfp_torch)

    def fp32_to_mxfp4(tensor):
        # tensor (in_features, out_features)
        tensor = tensor.unsqueeze(0)
        tensor_fp4, tensor_scales = downcast_to_mxfp_torch(tensor,
                                                           torch.uint8,
                                                           axis=1)
        return tensor_fp4[0], tensor_scales[0]

    def mxfp4_to_fp32(tensor, scales):
        tensor = tensor.unsqueeze(0)
        scales = scales.unsqueeze(0)
        tensor = upcast_from_mxfp_torch(tensor, scales, torch.float32, axis=1)
        return tensor[0]

    # Convert weight to MXFP4
    w_weight_fp4, w_weight_scale = fp32_to_mxfp4(w)
    w_weight_qdq = mxfp4_to_fp32(w_weight_fp4, w_weight_scale)

    # Create reference linear with dequantized weights
    ref_weights = {
        "weight": w_weight_qdq.T,  # Transpose to match TRT-LLM's weight shape
        "bias": b,
    }

    ref_linear = Linear(  # Always use regular Linear for reference
        in_features=hidden_size,
        out_features=out_size,
        bias=True,
        dtype=dtype,
    )
    ref_linear.load_weights([ref_weights])
    ref_linear.cuda()

    ref_output = ref_linear.forward(x)
    torch.cuda.synchronize()

    # Now test with MXFP4 quantized weights
    weights = {
        "weight": w_weight_fp4.T,  # Transpose to match TRT-LLM's weight shape
        "bias": b,
        "weight_scale":
        w_weight_scale.T,  # Transpose scale to match weight shape
    }

    # Add input scale for FP8 activation
    if activation_dtype == torch.float8_e4m3fn:
        _, input_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(x)
        weights["input_scale"] = input_scale

    quant_algo = QuantAlgo.W4A8_MXFP4_FP8 if activation_dtype == torch.float8_e4m3fn else QuantAlgo.W4A16_MXFP4

    linear = TritonLinear(in_features=hidden_size,
                          out_features=out_size,
                          bias=True,
                          dtype=dtype,
                          quant_config=QuantConfig(quant_algo=quant_algo))
    linear.load_weights([weights])
    linear.cuda()

    output = linear.forward(x)
    torch.cuda.synchronize()

    # Compare outputs with more relaxed tolerance for MXFP4
    check_accuracy(output, ref_output, rtol=0.2, atol=0.2, percent=0.95)
