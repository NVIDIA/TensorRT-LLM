# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU unit tests for Qwen3ForTextEmbedding registration and routing.

These do not construct the model (no GPU/weights); they assert that the new
architecture is registered and classified as a non-generation (encoder/embedding)
model so it routes through the llm.encode() path.
"""


def test_qwen3_text_embedding_is_registered():
    # Importing the module triggers @register_auto_model.
    import tensorrt_llm._torch.models.modeling_qwen3  # noqa: F401
    from tensorrt_llm._torch.models.modeling_utils import MODEL_CLASS_MAPPING

    assert "Qwen3ForTextEmbedding" in MODEL_CLASS_MAPPING


def test_qwen3_text_embedding_is_not_generation():
    from tensorrt_llm._torch.model_config import ModelConfig

    assert ModelConfig.is_generation_model(["Qwen3ForTextEmbedding"]) is False
    # Sanity: the plain causal LM is still a generation model.
    assert ModelConfig.is_generation_model(["Qwen3ForCausalLM"]) is True
