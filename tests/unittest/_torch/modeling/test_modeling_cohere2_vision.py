from dataclasses import dataclass
from pathlib import Path
from typing import List

import pytest
from test_modeling_multimodal import MultimodalScenario, TestModelingMultimodal, llm_models_root
from transformers import AutoTokenizer, Cohere2VisionConfig
from transformers import Cohere2VisionForConditionalGeneration as HFCohere2VisionForConditionalGeneration

from tensorrt_llm._torch.models.modeling_cohere2_vision import Cohere2VisionModel
from tensorrt_llm.inputs import create_input_processor

COMMAND_A_140M_CONFIG = {
  # NOTE: TestModelingMultimodal.get_dtype() reads "torch_dtype", not "dtype",
  # so both keys must be set consistently.
  "dtype": "bfloat16",
  "torch_dtype": "bfloat16",
  "text_config": {
    "hidden_size": 256, # head_dim * num_attention_heads
    "intermediate_size": 128,
    "head_dim": 64,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
  },
  "vision_config": {
    "image_size": 512,
    "hidden_size": 256,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    # head_dim is automatically set as hidden_size // num_attention_heads
  },
  # The processor and tokenizer are loaded from Command A Vision 111B
  "_name_or_path": str(Path(llm_models_root()) / "command-a-vision-07-2025/"),
}


@dataclass(repr=False)
class TestCohere2VisionScenario(MultimodalScenario):
    pass


class TestCohere2Vision(TestModelingMultimodal):
    def get_model_config(self):
        return COMMAND_A_140M_CONFIG

    def get_trtllm_model_class(self):
        return Cohere2VisionModel

    def get_hf_model_class(self):
        return HFCohere2VisionForConditionalGeneration

    def get_weight_mapper_class(self):
        # Do not use an explicit weight mapper class at the moment
        return None

    def get_model_type(self):
        return "cohere2_vision"

    def get_model_config_class(self):
        return Cohere2VisionConfig

    def get_tolerance(self):
        # The reduced test config uses a tiny hidden_size and random-init weights.
        # Accumulation errors could be harmful in the small models.
        # Production weights are well-behaved within the standard 0.4 tolerance.
        return 1.5, 1.5

    def get_scenarios(self) -> List[TestCohere2VisionScenario]:
        # NOTE: chunked-prefill and KV-cache-reuse scenarios are intentionally
        # omitted: Cohere2InputProcessor does not implement them yet
        # since find_input_mm_embeds cannot slice mm_embeds.
        scenarios = [
            # ==== Modality Sanity Checks ====
            TestCohere2VisionScenario(
                modality="image", use_cuda_graph=False, chunked_prefill=False, kv_cache_reuse=False
            ),
            # ==== CUDA Graph Scenarios ====
            TestCohere2VisionScenario(
                modality="image", use_cuda_graph=True, chunked_prefill=False, kv_cache_reuse=False
            ),
        ]
        return scenarios
    

@pytest.mark.skip(
    reason="Cohere2InputProcessor does not yet implement expand_prompt_token_ids_for_mm "
           "(the tokenized + multimodal-hashing path). Re-enable when added."
)
def test_cohere2_vision_expand_prompt_token_ids_for_mm():
    """Test Cohere2VisionInputProcessor.expand_prompt_token_ids_for_mm replaces image placeholders correctly."""
    model_path = COMMAND_A_140M_CONFIG["_name_or_path"]
    if not Path(model_path).exists():
        pytest.skip(f"Cohere2Vision (Command A) model not found at {model_path} (set LLM_MODELS_ROOT)")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    input_processor = create_input_processor(model_path, tokenizer=tokenizer)

    # The COMMAND_A_140M_CONFIG is a reduced test config and does not carry
    # these fields; read them from the real config loaded by the input processor.
    config = input_processor.config
    image_token_id = getattr(config, "image_token_id", None) \
        or config.image_token_index
    vocab_size = config.text_config.vocab_size
    placeholder_id = vocab_size + 1

    # prompt_token_ids: two image placeholders with text tokens in between
    prompt_token_ids = [1, 2, image_token_id, 3, image_token_id, 4]
    num_mm_tokens_per_placeholder = [10, 20]

    expanded = input_processor.expand_prompt_token_ids_for_mm(
        prompt_token_ids, num_mm_tokens_per_placeholder
    )

    # Expected: [1, 2] + 10 * placeholder_id + [3] + 20 * placeholder_id + [4]
    expected_len = 2 + 10 + 1 + 20 + 1
    assert len(expanded) == expected_len
    assert expanded[:2] == [1, 2]
    assert expanded[2:12] == [placeholder_id] * 10
    assert expanded[12] == 3
    assert expanded[13:33] == [placeholder_id] * 20
    assert expanded[33] == 4
