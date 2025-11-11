# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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

from typing import Any

import torch
from transformers.configuration_utils import PretrainedConfig

from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata
from tensorrt_llm._torch.models.checkpoints.base_config_loader import BaseConfigLoader
from tensorrt_llm._torch.models.checkpoints.base_weight_loader import BaseWeightLoader
from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import BaseWeightMapper
from tensorrt_llm._torch.models.modeling_utils import (
    ModelConfig,
    register_auto_model,
    register_checkpoint_weight_loader,
    register_config_loader,
)


# Define a dummy model to create deterministic outputs for the test
class DummyConfig(PretrainedConfig):
    def __init__(self):
        self.architectures: list[str] = ["DummyModel"]
        self.torch_dtype: torch.dtype = torch.float16
        self.num_key_value_heads: int = 16
        self.num_attention_heads: int = 16
        self.hidden_size: int = 256
        self.vocab_size: int = 1000
        self.num_hidden_layers: int = 1

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


@register_auto_model("DummyModel")
class DummyModel(torch.nn.Module):
    def __init__(self, model_config: ModelConfig):
        super().__init__()
        self.dtype = model_config.pretrained_config.torch_dtype
        self.model_config = model_config

    def infer_max_seq_len(self):
        return 2048

    @property
    def config(self):
        return self.model_config.pretrained_config

    def forward(
        self,
        *args,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attn_metadata: AttentionMetadata,
        return_context_logits: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        num_batch_tokens = input_ids.size(0)

        vocab_size = self.config.vocab_size
        last_tokens = (
            torch.cumsum(
                attn_metadata.seq_lens_cuda,
                dim=0,
                dtype=torch.long,
            )
            - 1
        )

        # Logits: fixed values for testing
        logits = torch.zeros((num_batch_tokens, vocab_size), device="cuda")
        indices = torch.arange(num_batch_tokens, device="cuda")

        # multiply the scores with input_ids to prevent paths like AB and BA to have the same score.
        # Score the next 4 tokens starting from the current input token.
        logits[indices, input_ids % vocab_size] += 0.1 * input_ids
        logits[indices, (input_ids + 1) % vocab_size] += 0.2 * input_ids
        logits[indices, (input_ids + 2) % vocab_size] += 0.3 * input_ids
        logits[indices, (input_ids + 3) % vocab_size] += 0.4 * input_ids

        # Logits shape depends on return_context_logits flag
        if not return_context_logits:
            # For context logits, return logits for all positions
            logits = logits[last_tokens]

        num_context_requests = attn_metadata.num_contexts
        # each beam has its own attn_metadata.seq_lens_cuda entry
        num_generation_requests = (
            last_tokens.shape[0] - num_context_requests
        ) // attn_metadata.beam_width
        num_requests = num_generation_requests + num_context_requests

        # return cache indirection, as additional model output.
        # each sequence should only return a 1D cache indirection tensor
        context_cache_indirection = attn_metadata.cache_indirection[:num_context_requests, 0]
        generation_cache_indirection = attn_metadata.cache_indirection[
            num_context_requests:num_requests
        ].view(
            num_generation_requests * attn_metadata.beam_width,
            attn_metadata.cache_indirection.shape[-1],
        )
        return {
            "logits": logits,
            "cache_indirection": torch.cat(
                [context_cache_indirection, generation_cache_indirection], dim=0
            ),
        }

    def load_weights(
        self,
        weights: dict,
        weight_mapper: BaseWeightMapper | None = None,
        skip_modules: list[str] = [],
    ):
        pass


@register_checkpoint_weight_loader("DUMMY_FORMAT")
class DummyWeightLoader(BaseWeightLoader):
    def load_weights(self, checkpoint_dir: str, **kwargs) -> dict[str, Any]:
        """Load weights from your dummy format.
        Args:
            checkpoint_dir: Directory containing checkpoint files
            **kwargs: Additional loading parameters
        Returns:
            Dictionary mapping parameter names to tensors
        """
        weights = {}

        return weights


@register_config_loader("DUMMY_FORMAT")
class DummyConfigLoader(BaseConfigLoader):
    def load(self, checkpoint_dir: str, **kwargs) -> ModelConfig:
        """Load and parse configuration from your dummy format.
        Args:
            checkpoint_dir: Directory containing configuration files
            **kwargs: Additional loading parameters
        Returns:
            ModelConfig object containing parsed configuration
        """
        return ModelConfig(pretrained_config=DummyConfig())


class BeamSearchTestOutput:
    def __init__(self, outputs: torch.Tensor, cache_indirection: torch.Tensor):
        self.outputs = outputs
        self.cache_indirection = cache_indirection


def get_expected_outputs(
    start_token: int, num_iterations: int = 4, vocab_size: int = 1000
) -> BeamSearchTestOutput:
    """Get the expected outputs for the given start token, number of iterations, vocabulary size
    This function only works for a beam width of 2.

    arguments:
    - start_token: the token to start the generation from. This is the last token of the input prompt.
    - num_iterations: the number of iterations to generate
    - vocab_size: the size of the vocabulary
    returns:
    - BeamSearchTestOutput: a named tuple containing the expected outputs and cache indirection
        - expected_outputs: the expected outputs for the given start token, number of iterations, vocabulary size
        - expected_cache_indirection: the expected cache indirection for the
                                      given start token, number of iterations, vocabulary size
    """
    expected_outputs = []
    expected_cache_indirection = []

    # These scores are deterministically set to the 4 tokens following the current input token (see model forward pass)
    base_scores = torch.tensor([0.1, 0.2, 0.3, 0.4])
    # extend the base scores to the full vocabulary size to calculate the correct softmax scores
    full_scores = torch.zeros(vocab_size)
    full_scores[: base_scores.shape[0]] = base_scores

    # calculate the softmax scores for the first token
    softmax_score = torch.log_softmax(full_scores * start_token, dim=-1)
    # get the top 2 offsets => these will become beam 0 and beam 1
    # The base scores are at the start of the full scores. So we can use their index as offset.
    top_offset, second_best_offset = torch.topk(softmax_score, k=2, dim=-1).indices

    # First iteration
    expected_outputs = [[start_token + top_offset], [start_token + second_best_offset]]
    expected_cache_indirection = [[0], [1]]
    beam_scores = [softmax_score[top_offset], softmax_score[second_best_offset]]

    for i in range(1, num_iterations):
        # calculate the softmax scores for the next token
        softmax_score_0 = torch.log_softmax(full_scores * expected_outputs[0][-1], dim=-1)
        softmax_score_1 = torch.log_softmax(full_scores * expected_outputs[1][-1], dim=-1)

        # For the given setup: Beam 0 will always select the highest scoring token, which is at offset 3.
        # naming: score_<beam_id><offset>
        score_03 = softmax_score_0[top_offset] + beam_scores[0]

        # beam 1 will either select the highest scoring token of beam 1, or the second highest scoring token of beam 0.
        score_02 = softmax_score_0[second_best_offset] + beam_scores[0]
        score_13 = softmax_score_1[top_offset] + beam_scores[1]

        # This should always be true
        assert score_03 >= score_13
        # If beam 1 selects the highest scoring token of beam 1, we update the beam scores and cache indirection
        if score_13 >= score_02:
            expected_outputs[1].append(expected_outputs[1][-1] + top_offset)
            beam_scores[1] = score_13
        else:
            # Beam 1 drops its old beam and changes to beam 0 and selects the second highest scoring token of beam 0.
            for j in range(i):
                expected_outputs[1][j] = expected_outputs[0][j]
                if i < num_iterations - 1:
                    # Avoid swapping in the last iteration, as the cache indirection
                    # provided by the model is returned before this change.
                    expected_cache_indirection[1][j] = expected_cache_indirection[0][j]
            expected_outputs[1].append(expected_outputs[0][-1] + second_best_offset)
            beam_scores[1] = score_02

        # Update Beam 0
        beam_scores[0] = score_03
        expected_outputs[0].append(expected_outputs[0][-1] + top_offset)

        # Update cache indirection if necessary
        if i < num_iterations - 1:
            expected_cache_indirection[0].append(0)
            expected_cache_indirection[1].append(1)

    return BeamSearchTestOutput(
        outputs=torch.tensor(expected_outputs),
        cache_indirection=torch.tensor(expected_cache_indirection),
    )
