import os

import pytest
import torch
from _torch.modules.moe.quantize_utils import BaseQuantizeUtil, FP8QuantizeUtil, NVFP4QuantizeUtil
from transformers.configuration_utils import PretrainedConfig
from utils.util import check_accuracy

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.modules.fused_moe import RenormalizeMoeRoutingMethod, create_moe
from tensorrt_llm._utils import mpi_rank
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.models.modeling_utils import QuantAlgo, QuantConfig


@pytest.mark.parametrize(
    "quant_algo",
    [
        "none",
        "nvfp4",
        "fp8",
    ],
    ids=lambda val: f"quant_algo={val}",
)
@pytest.mark.parametrize(
    "moe_backend",
    [
        "CUTLASS",
        "TRTLLM",
    ],
    ids=lambda val: f"moe_backend={val}",
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float16,
        torch.bfloat16,
    ],
    ids=lambda val: f"dtype={val}",
)
def test_moe(dtype, moe_backend, quant_algo, mocker):
    # Enable configurable moe by default
    mocker.patch.dict(os.environ, {"ENABLE_CONFIGURABLE_MOE": "1"})

    if moe_backend == "TRTLLM":
        if dtype == torch.float16:
            pytest.skip("TRTLLM NVFP4 MoE backend does not support float16 yet")

    # Hardcode some parameters for testing
    # activation and weight related
    seq_len = 4
    top_k = 2
    num_experts = 8
    hidden_size = 512
    intermediate_size = 512
    # Other parameters
    finalize_fusion = True

    # Create mapping for current rank
    mapping = Mapping()
    mapping.rank = mpi_rank()

    with torch.device(f"cuda:{mapping.rank}"):
        torch.manual_seed(0)
        torch.cuda.manual_seed(0)
        # Create route method
        routing_method = RenormalizeMoeRoutingMethod(top_k=top_k)

        # Create activation and weight
        x = torch.randn((seq_len, hidden_size), dtype=dtype, device="cuda")
        router_logits = torch.randn((seq_len, num_experts), dtype=dtype, device="cuda")

        # Create quant_config
        quant_config = None
        quant_kwargs = {}
        if quant_algo == "none":
            quantize_util_cls = BaseQuantizeUtil
        elif quant_algo == "fp8":
            quantize_util_cls = FP8QuantizeUtil
            quant_config = QuantConfig(quant_algo=QuantAlgo.FP8)
            _, x_scale = torch.ops.tensorrt_llm.quantize_e4m3_per_tensor(x)
            x_scale = x_scale.float().squeeze()
            quant_kwargs["x_scale"] = x_scale
        elif quant_algo == "nvfp4":
            quantize_util_cls = NVFP4QuantizeUtil
            quant_config = QuantConfig(quant_algo=QuantAlgo.NVFP4)
            x_sf_global = (448 * 6) / x.abs().max().float()
            scaling_vector_size = 16
            quant_kwargs["scaling_vector_size"] = scaling_vector_size
            quant_kwargs["x_sf_global"] = x_sf_global
        else:
            assert False, "unsupported quant_algo"

        quantize_util = quantize_util_cls(
            num_experts=num_experts,
            dtype=dtype,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
            quant_config=quant_config,
        )

        weights = quantize_util.create_weights(**quant_kwargs)

        # Create pretrained config
        pretrained_config = PretrainedConfig()
        pretrained_config.num_experts = num_experts
        pretrained_config.hidden_size = hidden_size
        pretrained_config.intermediate_size = intermediate_size
        pretrained_config.torch_dtype = dtype

        # Create fused MoE module
        fused_moe = create_moe(
            routing_method=routing_method,
            reduce_results=True,
            model_config=ModelConfig(
                pretrained_config=pretrained_config,
                quant_config=quant_config,
                moe_backend=moe_backend,
                moe_disable_finalize_fusion=not finalize_fusion,
            ),
        )
        fused_moe.load_weights([weights])
        fused_moe.post_load_weights()
        fused_moe.cuda()

        ref_fused_moe = quantize_util.create_ref_module(routing_method)
        ref_fused_moe.load_weights([weights])
        ref_fused_moe.cuda()

        # Evaluate the outputs
        with torch.inference_mode():
            ref_output = ref_fused_moe.forward(x, router_logits)
            output = fused_moe.forward(x, router_logits)

        # Here we use same rtol and atol as test_fused_moe
        if quant_algo == "nvfp4":
            torch.testing.assert_close(output, ref_output, rtol=1e-2, atol=0.15)
        else:
            if quant_algo == "fp8":
                rtol, atol, percent = (4e-2, 1e-1, 0.99)
            elif quant_algo == "none":
                rtol, atol, percent = (2e-1, 2e-1, 0.984)
            else:
                assert False, "unsupported quant_algo to check accuracy"
            check_accuracy(output, ref_output, rtol=rtol, atol=atol, percent=percent)
