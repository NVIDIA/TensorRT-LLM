import transformers

from .modeling_auto import AutoModelForCausalLM
from .modeling_bert import BertForSequenceClassification
from .modeling_deepseekv3 import DeepseekV3ForCausalLM
from .modeling_llama import LlamaForCausalLM
from .modeling_llava_next import LlavaNextModel
from .modeling_mamba_hybrid import MambaHybridForCausalLM
from .modeling_mixtral import MixtralForCausalLM
from .modeling_nemotron import NemotronForCausalLM
from .modeling_nvsmall import NVSmallForCausalLM
from .modeling_qwen import (Qwen2ForCausalLM, Qwen2ForProcessRewardModel,
                            Qwen2ForRewardModel)
from .modeling_qwen2vl import Qwen2_5_VLModel, Qwen2VLModel
from .modeling_qwen_moe import Qwen2MoeForCausalLM
from .modeling_utils import get_model_architecture
from .modeling_vila import VilaModel

# Note: for better readiblity, this should have same order as imports above
__all__ = [
    "AutoModelForCausalLM",
    "BertForSequenceClassification",
    "DeepseekV3ForCausalLM",
    "LlamaForCausalLM",
    "LlavaNextModel",
    "MambaHybridForCausalLM",
    "MixtralForCausalLM",
    "NemotronForCausalLM",
    "NVSmallForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen2ForProcessRewardModel",
    "Qwen2ForRewardModel",
    "Qwen2MoeForCausalLM",
    "get_model_architecture",
    "VilaModel",
    "Qwen2VLModel",
    "Qwen2_5_VLModel",
]

if transformers.__version__ >= "4.45.1":
    from .modeling_mllama import MllamaForConditionalGeneration  # noqa
    __all__.append("MllamaForConditionalGeneration")
else:
    print(
        f"Failed to import MllamaForConditionalGeneration as transformers.__version__ {transformers.__version__} < 4.45.1"
    )
