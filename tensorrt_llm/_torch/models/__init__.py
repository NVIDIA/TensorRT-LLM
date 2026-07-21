import transformers

# Importing _torch.configs triggers AutoConfig registration for TRT-LLM-only
# model_types (deepseek_v32, kimi_k2, gemma4_unified) so AutoConfig /
# AutoTokenizer.from_pretrained work under transformers >= 5.5; see
# _torch/configs/__init__.py.
import tensorrt_llm._torch.configs  # noqa: F401

from .modeling_afmoe import AfmoeForCausalLM
from .modeling_auto import AutoModelForCausalLM
from .modeling_bart import (BartForConditionalGeneration,
                            MBartForConditionalGeneration)
from .modeling_bert import BertForSequenceClassification
from .modeling_clip import CLIPVisionModel
from .modeling_cohere2 import Cohere2ForCausalLM
from .modeling_cosmos3 import Cosmos3Model
from .modeling_deepseekv3 import DeepseekV3ForCausalLM
from .modeling_deepseekv4 import DeepseekV4ForCausalLM
from .modeling_exaone4 import Exaone4ForCausalLM
from .modeling_exaone4_5 import Exaone4_5_ForConditionalGeneration
from .modeling_exaone_moe import ExaoneMoeForCausalLM
from .modeling_gemma3 import Gemma3ForCausalLM
from .modeling_gemma3vl import Gemma3VLM
from .modeling_gemma4 import Gemma4ForCausalLM
from .modeling_gemma4_unified import Gemma4UnifiedForConditionalGeneration
from .modeling_gemma4mm import Gemma4ForConditionalGeneration
from .modeling_glm import Glm4MoeForCausalLM
from .modeling_gpt_oss import GptOssForCausalLM
from .modeling_hunyuan_dense import HunYuanDenseV1ForCausalLM
from .modeling_hunyuan_moe import HunYuanMoEV1ForCausalLM
from .modeling_hyperclovax import HCXVisionForCausalLM
from .modeling_inkling import (InklingForCausalLM,
                               InklingForConditionalGeneration)
from .modeling_kimi_k25 import KimiK25ForConditionalGeneration
from .modeling_laguna import LagunaForCausalLM
from .modeling_llama import LlamaForCausalLM
from .modeling_llava_next import LlavaNextModel
from .modeling_minimaxm2 import MiniMaxM2ForCausalLM
from .modeling_minimaxm3 import (MiniMaxM3ForCausalLM,
                                 MiniMaxM3VLForConditionalGeneration)
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
from .modeling_qwen3_5 import (Qwen3_5ForCausalLM, Qwen3_5MoeForCausalLM,
                               Qwen3_5MoeVLModel, Qwen3_5VLModel)
from .modeling_qwen3_moe import Qwen3MoeForCausalLM
from .modeling_qwen3_next import Qwen3NextForCausalLM
from .modeling_qwen3vl import Qwen3VLModel
from .modeling_qwen3vl_moe import Qwen3MoeVLModel
from .modeling_qwen_image_bench import QwenImageBenchModel
from .modeling_qwen_moe import Qwen2MoeForCausalLM
from .modeling_seedoss import SeedOssForCausalLM
from .modeling_siglip import SiglipVisionModel
from .modeling_starcoder2 import Starcoder2ForCausalLM
from .modeling_step3p7 import Step3p7ForCausalLM
from .modeling_step3p7vl import Step3p7VLForConditionalGeneration
from .modeling_t5 import T5ForConditionalGeneration
from .modeling_utils import get_model_architecture
from .modeling_vila import VilaModel

# Note: for better readiblity, this should have same order as imports above
__all__ = [
    "AfmoeForCausalLM",
    "AutoModelForCausalLM",
    "BartForConditionalGeneration",
    "BertForSequenceClassification",
    "CLIPVisionModel",
    "Cosmos3Model",
    "DeepseekV3ForCausalLM",
    "Exaone4ForCausalLM",
    "Exaone4_5_ForConditionalGeneration",
    "ExaoneMoeForCausalLM",
    "Gemma3ForCausalLM",
    "Gemma3VLM",
    "Gemma4ForCausalLM",
    "Gemma4ForConditionalGeneration",
    "Gemma4UnifiedForConditionalGeneration",
    "HCXVisionForCausalLM",
    "InklingForCausalLM",
    "InklingForConditionalGeneration",
    "LagunaForCausalLM",
    "HunYuanDenseV1ForCausalLM",
    "HunYuanMoEV1ForCausalLM",
    "KimiK25ForConditionalGeneration",
    "LlamaForCausalLM",
    "LlavaNextModel",
    "Mistral3VLM",
    "MistralForCausalLM",
    "MixtralForCausalLM",
    "DeepseekV4ForCausalLM",
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
    "T5ForConditionalGeneration",
    "MBartForConditionalGeneration",
    "get_model_architecture",
    "VilaModel",
    "Qwen2VLModel",
    "Qwen2_5_VLModel",
    "Qwen3ForCausalLM",
    "Qwen3MoeForCausalLM",
    "Qwen3_5ForCausalLM",
    "Qwen3_5MoeForCausalLM",
    "QwenImageBenchModel",
    "Qwen3_5MoeVLModel",
    "Qwen3_5VLModel",
    "Qwen3NextForCausalLM",
    "Qwen3MoeVLModel",
    "GptOssForCausalLM",
    "SeedOssForCausalLM",
    "Glm4MoeForCausalLM",
    "Qwen3VLModel",
    "MiniMaxM2ForCausalLM",
    "MiniMaxM3ForCausalLM",
    "MiniMaxM3VLForConditionalGeneration",
    "Cohere2ForCausalLM",
    "Step3p7ForCausalLM",
    "Step3p7VLForConditionalGeneration",
]

if transformers.__version__ >= "4.45.1":
    from .modeling_mllama import MllamaForConditionalGeneration  # noqa

    __all__.append("MllamaForConditionalGeneration")
else:
    print(
        f"Failed to import MllamaForConditionalGeneration as transformers.__version__ {transformers.__version__} < 4.45.1"
    )
