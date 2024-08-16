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
"""Utils for positional embeddings (including RoPE).
"""

import jax
import jax.numpy as jnp

_MAX_WAVELENGTH = 10_000


def add_positional_embedding(
    input_embedding: jax.Array,
    position: int,
    max_wavelength: int = _MAX_WAVELENGTH,
) -> jax.Array:
    """Adds positional embeddings to input embeddings."""
    embed_dim = input_embedding.shape[-1]
    num_timescales = embed_dim // 2
    log_timescale_increment = jnp.log(float(max_wavelength)) / jnp.maximum(
        jnp.asarray(num_timescales, dtype=jnp.float32) - 1, 1)
    inv_timescales = jnp.exp(
        jnp.arange(num_timescales, dtype=jnp.float32) *
        -log_timescale_increment)
    scaled_time = position * inv_timescales
    signal = jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)])
    signal = jnp.pad(signal, [[0, jnp.mod(embed_dim, 2)]])
    position_embedding = signal.astype(jnp.float32)

    return input_embedding + position_embedding


def _rotary_embed(
    inputs: jax.Array,  # [B, 1, H, D]
    position: jax.Array,  # [B,]
    head_dim: int,
    max_wavelength: int = _MAX_WAVELENGTH,
) -> jax.Array:
    """Helper for RoPE."""
    fraction = 2 * jnp.arange(0, head_dim // 2) / head_dim
    timescale = max_wavelength**fraction
    timescale = timescale[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]

    sinusoid_inp = position[:, jnp.newaxis, jnp.newaxis,
                            jnp.newaxis] / timescale
    sin = jnp.sin(sinusoid_inp)
    cos = jnp.cos(sinusoid_inp)

    first_half, second_half = jnp.split(inputs, 2, axis=-1)
    first_part = first_half * cos - second_half * sin
    second_part = second_half * cos + first_half * sin

    return jnp.concatenate([first_part, second_part], axis=-1)


def apply_rope(
    inputs: jax.Array,
    position: int,
    head_dim: int,
    max_wavelength: int = _MAX_WAVELENGTH,
) -> jax.Array:
    """Applies RoPE."""
    batch_size, seq_length = inputs.shape[0:2]

    position = jnp.broadcast_to(position, [batch_size])[:, jnp.newaxis]
    prefix_position = jnp.arange(seq_length, dtype=jnp.int32)
    prefix_position = (position - jnp.flip(prefix_position)[jnp.newaxis, :]
                       )  # [B, seq_len]
    prefix_position = jnp.where(prefix_position < 0,
                                jnp.zeros_like(prefix_position),
                                prefix_position).reshape((batch_size, ))

    output = _rotary_embed(
        inputs,
        position=prefix_position,
        head_dim=head_dim,
        max_wavelength=max_wavelength,
    )

    return output
