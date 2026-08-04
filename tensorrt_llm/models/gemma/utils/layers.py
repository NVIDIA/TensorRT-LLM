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
"""Base layers."""

import jax
import jax.numpy as jnp
from flax import linen as nn


class Einsum(nn.Module):
    shape: tuple[int, ...]

    @nn.compact
    def __call__(self, eqn: str, x: jax.Array) -> jax.Array:
        w = self.param('w', nn.initializers.zeros_init(), self.shape)
        return jnp.einsum(eqn, x, w)


class RMSNorm(nn.Module):

    @nn.compact
    def __call__(self, x):
        scale = self.param('scale', nn.initializers.zeros_init(), (x.shape[-1]))
        var = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))
        normed_inputs = normed_inputs * (1 + scale)
        return normed_inputs
