import os

from .modeling_deepseek import DeepSeekV3ForCausalLM
from .modeling_gemma3n import Gemma3nForCausalLM, Gemma3nForConditionalGeneration
from .modeling_gemma4 import Gemma4ForCausalLM, Gemma4ForConditionalGeneration
from .modeling_glm4_moe_lite import Glm4MoeLiteForCausalLM
from .modeling_kimi_k2 import KimiK2ForCausalLM, KimiK25ForConditionalGeneration
from .modeling_minimax_m2 import MiniMaxM2ForCausalLM
from .modeling_mistral3 import Mistral3ForConditionalGenerationAD, Mistral4ForCausalLM
from .modeling_nemotron_flash import NemotronFlashForCausalLM, NemotronFlashPreTrainedTokenizerFast
from .modeling_nemotron_h import NemotronHForCausalLM
from .modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM, Qwen3_5MoeForConditionalGeneration

if os.environ.get("AD_USE_IR_MODELS"):
    from .modeling_deepseek_ir import DeepSeekV3ForCausalLM  # noqa: F811
    from .modeling_nemotron_h_ir import NemotronHForCausalLM  # noqa: F811
    from .modeling_qwen3_5_moe_ir import (  # noqa: F811
        Qwen3_5MoeForCausalLM,
        Qwen3_5MoeForConditionalGeneration,
    )

__all__ = (
    "DeepSeekV3ForCausalLM",
    "Gemma3nForCausalLM",
    "Gemma3nForConditionalGeneration",
    "Gemma4ForCausalLM",
    "Gemma4ForConditionalGeneration",
    "Glm4MoeLiteForCausalLM",
    "KimiK2ForCausalLM",
    "KimiK25ForConditionalGeneration",
    "MiniMaxM2ForCausalLM",
    "Mistral3ForConditionalGenerationAD",
    "Mistral4ForCausalLM",
    "NemotronFlashForCausalLM",
    "NemotronFlashPreTrainedTokenizerFast",
    "NemotronHForCausalLM",
    "Qwen3_5MoeForCausalLM",
    "Qwen3_5MoeForConditionalGeneration",
)
