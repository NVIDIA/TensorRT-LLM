import transformers

from .modeling_auto import AutoModelForCausalLM
from .modeling_bert import BertForSequenceClassification
from .modeling_clip import CLIPVisionModel
from .modeling_deepseekv3 import DeepseekV3ForCausalLM
from .modeling_exaone4 import Exaone4ForCausalLM
from .modeling_exaone_moe import ExaoneMoeForCausalLM
from .modeling_gemma3 import Gemma3ForCausalLM
from .modeling_gemma3vl import Gemma3VLM
from .modeling_glm import Glm4MoeForCausalLM
from .modeling_gpt_oss import GptOssForCausalLM
from .modeling_hunyuan_dense import HunYuanDenseV1ForCausalLM
from .modeling_hunyuan_moe import HunYuanMoEV1ForCausalLM
from .modeling_hyperclovax import HCXVisionForCausalLM
from .modeling_llama import LlamaForCausalLM
from .modeling_llava_next import LlavaNextModel
from .modeling_minimaxm2 import MiniMaxM2ForCausalLM
from .modeling_mistral import Mistral3VLM, MistralForCausalLM
from .modeling_mixtral import MixtralForCausalLM
from .modeling_nemotron import NemotronForCausalLM
from .modeling_nemotron_h import NemotronHForCausalLM
from .modeling_nemotron_nano import NemotronH_Nano_VL_V2
from .modeling_nemotron_nas import NemotronNASForCausalLM
from .modeling_phi3 import Phi3ForCausalLM
from .modeling_phi4mm import Phi4MMForCausalLM
from .modeling_qwen import (Qwen2ForCausalLM, Qwen2ForProcessRewardModel,
                            Qwen2ForRewardModel)
from .modeling_qwen2vl import Qwen2_5_VLModel, Qwen2VLModel
from .modeling_qwen3 import Qwen3ForCausalLM
from .modeling_qwen3_moe import Qwen3MoeForCausalLM
from .modeling_qwen3_next import Qwen3NextForCausalLM
from .modeling_qwen3vl import Qwen3VLModel
from .modeling_qwen3vl_moe import Qwen3MoeVLModel
from .modeling_qwen_moe import Qwen2MoeForCausalLM
from .modeling_seedoss import SeedOssForCausalLM
from .modeling_siglip import SiglipVisionModel
from .modeling_starcoder2 import Starcoder2ForCausalLM
from .modeling_utils import get_model_architecture
from .modeling_vila import VilaModel

# Note: for better readiblity, this should have same order as imports above
__all__ = [
    "AutoModelForCausalLM",
    "BertForSequenceClassification",
    "CLIPVisionModel",
    "DeepseekV3ForCausalLM",
    "Exaone4ForCausalLM",
    "ExaoneMoeForCausalLM",
    "Gemma3ForCausalLM",
    "Gemma3VLM",
    "HCXVisionForCausalLM",
    "HunYuanDenseV1ForCausalLM",
    "HunYuanMoEV1ForCausalLM",
    "LlamaForCausalLM",
    "LlavaNextModel",
    "Mistral3VLM",
    "MistralForCausalLM",
    "MixtralForCausalLM",
    "NemotronH_Nano_VL_V2",
    "NemotronForCausalLM",
    "NemotronHForCausalLM",
    "NemotronNASForCausalLM",
    "Phi3ForCausalLM",
    "Phi4MMForCausalLM",
    "Qwen2ForCausalLM",
    "Qwen2ForProcessRewardModel",
    "Qwen2ForRewardModel",
    "Qwen2MoeForCausalLM",
    "SiglipVisionModel",
    "Starcoder2ForCausalLM",
    "get_model_architecture",
    "VilaModel",
    "Qwen2VLModel",
    "Qwen2_5_VLModel",
    "Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM",
    "Qwen3NextForCausalLM",
    "Qwen3MoeVLModel",
    "GptOssForCausalLM",
    "SeedOssForCausalLM",
    "Glm4MoeForCausalLM",
    "Qwen3VLModel",
    "MiniMaxM2ForCausalLM",
]

if transformers.__version__ >= "4.45.1":
    from .modeling_mllama import MllamaForConditionalGeneration  # noqa

    __all__.append("MllamaForConditionalGeneration")
else:
    print(
        f"Failed to import MllamaForConditionalGeneration as transformers.__version__ {transformers.__version__} < 4.45.1"
    )
