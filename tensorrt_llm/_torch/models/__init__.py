import transformers

from .modeling_auto import AutoModelForCausalLM
from .modeling_llama import LlamaForCausalLM
from .modeling_mixtral import MixtralForCausalLM
from .modeling_nemotron import NemotronForCausalLM
from .modeling_nvsmall import NVSmallForCausalLM
from .modeling_qwen import Qwen2ForCausalLM
from .modeling_utils import get_model_architecture
from .modeling_vila import VilaModel

__all__ = [
    "AutoModelForCausalLM", "LlamaForCausalLM", "VilaModel",
    "MixtralForCausalLM", "NVSmallForCausalLM", "Qwen2ForCausalLM",
    "NemotronForCausalLM", "get_model_architecture"
]

if transformers.__version__ >= "4.45.1":
    from .modeling_mllama import MllamaForConditionalGeneration  # noqa
    __all__.append("MllamaForConditionalGeneration")
else:
    print(
        f"Failed to import MllamaForConditionalGeneration as transformers.__version__ {transformers.__version__} < 4.45.1"
    )
