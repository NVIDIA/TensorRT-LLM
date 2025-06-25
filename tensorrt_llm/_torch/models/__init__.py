import transformers

from .modeling_auto import AutoModelForCausalLM
from .modeling_bert import BertForSequenceClassification
from .modeling_clip import CLIPVisionModel
from .modeling_deepseekv3 import DeepseekV3ForCausalLM
from .modeling_gemma3 import Gemma3ForCausalLM
from .modeling_gemma3vl import Gemma3Model
from .modeling_hyperclovax import HCXVisionForCausalLM
from .modeling_llama import LlamaForCausalLM
from .modeling_llava_next import LlavaNextModel
from .modeling_mistral import MistralForCausalLM
from .modeling_mixtral import MixtralForCausalLM
from .modeling_nemotron import NemotronForCausalLM
from .modeling_nemotron_h import NemotronHForCausalLM
from .modeling_nemotron_nas import NemotronNASForCausalLM
from .modeling_qwen import (Qwen2ForCausalLM, Qwen2ForProcessRewardModel,
                            Qwen2ForRewardModel)
from .modeling_qwen2vl import Qwen2_5_VLModel, Qwen2VLModel
from .modeling_qwen3 import Qwen3ForCausalLM
from .modeling_qwen3_moe import Qwen3MoeForCausalLM
from .modeling_qwen_moe import Qwen2MoeForCausalLM
from .modeling_siglip import SiglipVisionModel
from .modeling_utils import get_model_architecture
from .modeling_vila import VilaModel

# Note: for better readiblity, this should have same order as imports above
__all__ = [
    "AutoModelForCausalLM",
    "BertForSequenceClassification",
    "CLIPVisionModel",
    "DeepseekV3ForCausalLM",
    "Gemma3ForCausalLM",
    "HCXVisionForCausalLM",
    "Gemma3Model",
    "LlamaForCausalLM",
    "LlavaNextModel",
    "MistralForCausalLM",
    "MixtralForCausalLM",
    "NemotronForCausalLM",
    "NemotronHForCausalLM",
    "NemotronNASForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen2ForProcessRewardModel",
    "Qwen2ForRewardModel",
    "Qwen2MoeForCausalLM",
    "SiglipVisionModel",
    "get_model_architecture",
    "VilaModel",
    "Qwen2VLModel",
    "Qwen2_5_VLModel",
    "Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM",
]

if transformers.__version__ >= "4.45.1":
    from .modeling_mllama import MllamaForConditionalGeneration  # noqa

    __all__.append("MllamaForConditionalGeneration")
else:
    print(
        f"Failed to import MllamaForConditionalGeneration as transformers.__version__ {transformers.__version__} < 4.45.1"
    )
