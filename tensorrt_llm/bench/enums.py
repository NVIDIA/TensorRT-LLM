from __future__ import annotations

from typing import List

from aenum import MultiValueEnum

from tensorrt_llm.bindings.executor import CapacitySchedulerPolicy
from tensorrt_llm.quantization.mode import QuantAlgo

NO_EVICT = "Guaranteed No Evict"
MAX_UTIL = "Max Utilization"


class ModelArchitecture(MultiValueEnum):
    LLAMA = "LlamaForCausalLM"
    GPTJ = "GPTJForCausalLM"
    GEMMA = "GemmaForCausalLM"
    BLOOM = "BloomForCausalLM"
    OPT = "OPTForCausalLM"
    MIXTRAL = "MixtralForCausalLM"
    FALCON = "FalconForCausalLM"


class ResultsSchedulingPolicy(MultiValueEnum):
    MAX_UTILIZTION = MAX_UTIL, CapacitySchedulerPolicy.MAX_UTILIZATION
    NO_EVICT = NO_EVICT, CapacitySchedulerPolicy.GUARANTEED_NO_EVICT
    STATIC = "Static"


class IFBSchedulingPolicy(MultiValueEnum):
    MAX_UTILIZTION = CapacitySchedulerPolicy.MAX_UTILIZATION, MAX_UTIL, "max_utilization"
    NO_EVICT = CapacitySchedulerPolicy.GUARANTEED_NO_EVICT, NO_EVICT, "guaranteed_no_evict"
    STATIC = "Static", "static"


class KVCacheDtypeEnum(MultiValueEnum):
    """Enumeration of KV Cache precisions in TRT-LLM."""
    FP8 = "FP8", "fp8", "float8"
    FP16 = None, "FP16", "fp16", "float16"
    INT8 = "INT8", "int8"

    def get_build_options(self, dtype: str) -> List[str]:
        """Get the build options for TRT-LLM based on KV Cache precision.

        Args:
            dtype (str): The activation dtype for the model. This
            parameter maps the activation dtype for GEMM plugins for certain
            KV cache precisions.

        Returns:
            List[str]: A list of command line arguments to be added to build
            commands.
        """
        if not self.value == self.FP8:
            return ["--gemm_plugin", dtype]


class ComputeDtypeEnum(MultiValueEnum):
    """Enumeration for activation data type."""

    # FLOAT32 = "float32", "fp32", "FP32"
    FLOAT16 = "float16", "FLOAT16", "fp16", "FP16"
    BFLOAT16 = "bfloat16", "BFLOAT16", "bf16", "bfp16", "BF16"


# TODO: use quantization.mode.QuantAlgo eventually
class QuantizationAlgo(MultiValueEnum):
    """Enumerated type for quantization algorithms for string mapping."""

    W8A16 = QuantAlgo.W8A16.value
    W4A16 = QuantAlgo.W4A16.value
    W4A16_AWQ = QuantAlgo.W4A16_AWQ.value
    W4A8_AWQ = QuantAlgo.W4A8_AWQ.value
    W4A16_GPTQ = QuantAlgo.W4A16_GPTQ.value
    FP8 = QuantAlgo.FP8.value
    INT8 = QuantAlgo.INT8.value
    W8A8_SQ_PER_CHANNEL = QuantAlgo.W8A8_SQ_PER_CHANNEL.value
    W8A8_SQ_PER_TENSOR_PLUGIN = QuantAlgo.W8A8_SQ_PER_TENSOR_PLUGIN.value
    W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN = QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TOKEN_PLUGIN.value
    W8A8_SQ_PER_CHANNEL_PER_TENSOR_PLUGIN = QuantAlgo.W8A8_SQ_PER_CHANNEL_PER_TENSOR_PLUGIN.value
    W8A8_SQ_PER_TENSOR_PER_TOKEN_PLUGIN = QuantAlgo.W8A8_SQ_PER_TENSOR_PER_TOKEN_PLUGIN.value
    NONE = None, "None", "FP16", "BF16"
