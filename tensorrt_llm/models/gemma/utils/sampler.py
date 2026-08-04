# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Sampler for Gemma transformer.

An example of a sampling class for a Gemma model.
"""

import chex
import jax
import jax.numpy as jnp
import sentencepiece as spm

from . import modules
from . import params as params_lib
from . import transformer as transformer_lib


def _compute_attention_masks(time_step: jax.Array, seq_len: int,
                             input_mask: jax.Array) -> jax.Array:
    """Computes causal attention mask."""
    bsz = input_mask.shape[0]
    batch_time_step = jnp.full((bsz, 1), time_step, dtype=jnp.uint32)
    causal_padding = jnp.greater(jnp.expand_dims(jnp.arange(seq_len), 0),
                                 batch_time_step)
    causal_padding = causal_padding * jnp.expand_dims(input_mask, axis=-1)
    attention_mask = (
        causal_padding[:, jnp.newaxis, jnp.newaxis, :].astype(jnp.float32) *
        modules.K_MASK)
    attention_mask = jnp.squeeze(attention_mask, axis=1)
    return attention_mask


@chex.dataclass
class _SamplingState:

    # Number of tokens in the prompt.
    num_input_tokens: jnp.int32  # [B]

    # Fixed-size buffer for accumulating the output tokens.
    token_buffer: jnp.ndarray  # [B, L]

    # Model state for conditioning the model on autoregressively.
    cache: dict[str, modules.LayerCache]


class Sampler:
    """Sampler for Gemma transformer."""

    def __init__(
        self,
        transformer_config: transformer_lib.TransformerConfig,
        vocab: spm.SentencePieceProcessor,
        params: params_lib.Params,
        cache_size: int,
        buffer_size: int,
        max_decode_steps: int,
    ):
        self.transformer = transformer_lib.Transformer(
            config=transformer_config)
        self.vocab = vocab
        self.params = params
        self.cache_size = cache_size
        self.buffer_size = buffer_size
        self.max_decode_steps = max_decode_steps
        self._compiled_sample_fn = jax.jit(self._sample_fn)

    def _sample_step(self, params, time_step,
                     sampler_state: _SamplingState) -> _SamplingState:
        """Performs a single sampling step."""
        time_step = jnp.asarray(time_step, dtype=jnp.int32)
        last_token = sampler_state.token_buffer[:, time_step]
        input_mask = last_token != self.vocab.pad_id()
        attention_mask = _compute_attention_masks(
            time_step, self.cache_size, input_mask).astype(jnp.float32)

        logits, cache = self.transformer.apply(
            {'params': params},
            last_token,
            time_step,
            sampler_state.cache,
            attention_mask,
            time_step,
        )

        next_token_candidate = jnp.argmax(logits, axis=-1)  # [B, 1]
        next_token_candidate = next_token_candidate[:, 0]  # [B,]

        next_token_candidate = jnp.where(
            time_step < sampler_state.num_input_tokens - 1,
            sampler_state.token_buffer[:, time_step + 1],
            next_token_candidate,
        )

        token_buffer = sampler_state.token_buffer.at[:, time_step + 1].set(
            next_token_candidate)

        return _SamplingState(
            num_input_tokens=sampler_state.num_input_tokens,
            token_buffer=token_buffer,
            cache=cache,
        )

    def init_cache(self, bsz) -> dict[str, modules.LayerCache]:
        """Initializes the attention cache for each layer."""
        return {
            f'layer_{i}':
            modules.init_layer_cache(
                self.cache_size,
                self.transformer.config.num_heads,
                self.transformer.config.head_dim,
                bsz,
            )
            for i in range(self.transformer.config.num_layers)
        }

    def init_sample_state(self,
                          all_input_ids: list[jax.Array]) -> _SamplingState:
        """Initializes the sampling state given input prompts."""
        bsz = len(all_input_ids)
        num_input_tokens = [len(input_ids) for input_ids in all_input_ids]

        token_buffer = jnp.full(
            (
                bsz,
                self.buffer_size,
            ),
            self.vocab.pad_id(),
            dtype=jnp.int32,
        )
        for i, (input_ids,
                num_tokens) in enumerate(zip(all_input_ids, num_input_tokens)):
            token_buffer = token_buffer.at[i, :num_tokens].set(input_ids)

        return _SamplingState(
            num_input_tokens=jnp.array(num_input_tokens, dtype=jnp.int32),
            token_buffer=token_buffer,
            cache=self.init_cache(bsz),
        )

    def tokenize(self, input_string: str) -> jax.Array:
        """Tokenizes the input string."""
        input_ids = self.vocab.EncodeAsIds(input_string)
        input_ids = jnp.array([self.vocab.bos_id()] +
                              jnp.array(input_ids).tolist(),
                              dtype=jnp.int32)
        return input_ids

    def _sample_fn(
        self,
        params: params_lib.Params,
        initial_sampling_state: _SamplingState,
    ) -> _SamplingState:

        def sample_with_params(time_step: int, sampler_state: _SamplingState):
            return self._sample_step(params, time_step, sampler_state)

        return jax.lax.fori_loop(0, self.max_decode_steps, sample_with_params,
                                 initial_sampling_state)

    def __call__(self, input_strings: list[str] | str) -> list[str]:
        """Samples a completion of the input string."""
        if isinstance(input_strings, str):
            input_strings = [input_strings]
        all_input_ids = [self.tokenize(x) for x in input_strings]
        initial_sampling_state = self.init_sample_state(all_input_ids)

        sampling_state = self._compiled_sample_fn(self.params,
                                                  initial_sampling_state)

        out_tokens = [
            buffer[num_tokens:num_tokens + self.max_decode_steps]
            for buffer, num_tokens in zip(sampling_state.token_buffer,
                                          sampling_state.num_input_tokens)
        ]
        decoded_outputs = [
            self.vocab.DecodeIds(out_tokens.tolist())
            for out_tokens in out_tokens
        ]
        return decoded_outputs
