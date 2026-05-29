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

from typing import Any, cast

import torch
from transformers.configuration_utils import PretrainedConfig

from tensorrt_llm._torch.attention_backend import TrtllmAttentionMetadata
from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.models.checkpoints.base_config_loader import \
    BaseConfigLoader
from tensorrt_llm._torch.models.checkpoints.base_weight_loader import \
    BaseWeightLoader
from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import \
    BaseWeightMapper
from tensorrt_llm._torch.models.modeling_utils import (
    register_auto_model, register_checkpoint_weight_loader,
    register_config_loader)


class DeterministicBeamConfig(PretrainedConfig):
    model_type = "deterministic_beam"

    def __init__(self) -> None:
        super().__init__()
        self.architectures = ["DeterministicBeamModel"]
        self.torch_dtype = torch.float16
        self.num_key_value_heads = 1
        self.num_attention_heads = 1
        self.hidden_size = 16
        self.vocab_size = 64
        self.num_hidden_layers = 1

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


@register_auto_model("DeterministicBeamModel")
class DeterministicBeamModel(torch.nn.Module):
    """Minimal no-weight model with deterministic beam-search logits."""

    def __init__(self, model_config: ModelConfig[DeterministicBeamConfig]):
        super().__init__()
        assert model_config.pretrained_config is not None
        self.dtype = model_config.pretrained_config.torch_dtype
        self.model_config = model_config

    def infer_max_seq_len(self) -> int:
        return 128

    @property
    def config(self) -> DeterministicBeamConfig:
        return self.model_config.pretrained_config

    def forward(
        self,
        *args,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attn_metadata: AttentionMetadata,
        return_context_logits: bool = False,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        del args, position_ids, kwargs

        num_batch_tokens = input_ids.size(0)
        vocab_size = self.config.vocab_size
        logits = torch.zeros((num_batch_tokens, vocab_size),
                             device=input_ids.device)
        indices = torch.arange(num_batch_tokens, device=input_ids.device)

        # Score the current token and next three token IDs deterministically.
        token_ids = input_ids % vocab_size
        logits[indices, token_ids] += 0.1 * input_ids
        logits[indices, (token_ids + 1) % vocab_size] += 0.2 * input_ids
        logits[indices, (token_ids + 2) % vocab_size] += 0.3 * input_ids
        logits[indices, (token_ids + 3) % vocab_size] += 0.4 * input_ids

        assert attn_metadata.seq_lens_cuda is not None
        last_tokens = torch.cumsum(
            attn_metadata.seq_lens_cuda,
            dim=0,
            dtype=torch.long,
        ) - 1
        if not return_context_logits:
            logits = logits[last_tokens]

        assert attn_metadata.cache_indirection is not None
        num_context_requests = attn_metadata.num_contexts
        beam_width = cast(TrtllmAttentionMetadata, attn_metadata).beam_width
        num_generation_requests = (
            last_tokens.shape[0] - num_context_requests) // beam_width
        num_requests = num_generation_requests + num_context_requests
        context_cache_indirection = attn_metadata.cache_indirection[
            :num_context_requests, 0]
        generation_cache_indirection = attn_metadata.cache_indirection[
            num_context_requests:num_requests].view(
                num_generation_requests * beam_width,
                attn_metadata.cache_indirection.shape[-1],
            )

        return {
            "logits":
            logits,
            "cache_indirection":
            torch.cat([context_cache_indirection, generation_cache_indirection],
                      dim=0),
        }

    def load_weights(
        self,
        weights: dict[str, torch.Tensor],
        weight_mapper: BaseWeightMapper | None = None,
        skip_modules: list[str] = [],
    ) -> None:
        del weights, weight_mapper, skip_modules


@register_checkpoint_weight_loader("DUMMY_FORMAT")
class DummyWeightLoader(BaseWeightLoader):

    def load_weights(self, checkpoint_dir: str, **kwargs) -> dict[str, Any]:  # type: ignore
        del checkpoint_dir, kwargs
        return {}


@register_config_loader("DUMMY_FORMAT")
class DummyConfigLoader(BaseConfigLoader):

    def load(self, checkpoint_dir: str,
             **kwargs) -> ModelConfig[DeterministicBeamConfig]:
        del checkpoint_dir, kwargs
        return ModelConfig(pretrained_config=DeterministicBeamConfig())
