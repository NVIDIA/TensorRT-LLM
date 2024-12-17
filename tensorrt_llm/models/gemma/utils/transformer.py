"""Gemma transformer."""

import dataclasses

import jax
import jax.numpy as jnp
from flax import linen as nn

from . import layers, modules
from . import params as params_lib

Cache = dict[str, modules.LayerCache]


@dataclasses.dataclass
class TransformerConfig:
    """Configuration for the Gemma transformer."""

    num_layers: int
    num_embed: int
    embed_dim: int
    hidden_dim: int
    num_heads: int
    head_dim: int
    num_kv_heads: int

    @classmethod
    def from_params(cls, params: params_lib.Params,
                    num_embed: int) -> 'TransformerConfig':
        """Creates a TransformerConfig from loaded parameters."""
        num_layers = (max([
            int(k.split('_')[1])
            for k in params['transformer'].keys() if 'layer_' in k
        ]) + 1)
        hidden_dim, embed_dim = (
            params['transformer']['layer_0']['mlp']['linear'].shape)
        num_heads, head_dim, _ = (params['transformer']['layer_0']['attn']
                                  ['attn_vec_einsum']['w'].shape)
        use_qkv_einsum = 'qkv_einsum' in params['transformer']['layer_0'][
            'attn']
        if use_qkv_einsum:
            num_kv_heads = num_heads
        else:
            num_kv_heads = params['transformer']['layer_0']['attn'][
                'kv_einsum']['w'].shape[1]
        return cls(
            num_layers=num_layers,
            num_embed=num_embed,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            num_kv_heads=num_kv_heads,
        )


def init_cache(config: TransformerConfig, cache_size: int,
               batch_size: int) -> Cache:
    """Initializes a new Transformer cache."""
    return {
        f'layer_{i}':
        modules.init_layer_cache(cache_size, config.num_heads, config.head_dim,
                                 batch_size)
        for i in range(config.num_layers)
    }


class Transformer(nn.Module):
    """Gemma transformer."""

    config: TransformerConfig

    def setup(self):
        self.embedder = modules.Embedder(
            vocab_size=self.config.num_embed,
            embed_dim=self.config.embed_dim,
        )
        self.blocks = [
            modules.Block(
                name=f'layer_{i}',
                num_heads=self.config.num_heads,
                num_kv_heads=self.config.num_kv_heads,
                embed_dim=self.config.embed_dim,
                head_dim=self.config.head_dim,
                hidden_dim=self.config.hidden_dim,
            ) for i in range(self.config.num_layers)
        ]
        self.final_norm = layers.RMSNorm()

    def __call__(
        self,
        last_tokens: jax.Array,  # [B,]
        current_token_position: int,
        cache: Cache,
        attention_mask: jax.Array,  # [B, 1, L]
        time_step: int,
    ) -> tuple[jax.Array, Cache]:
        input_emb = self.embedder.encode(last_tokens)
        x = jnp.expand_dims(input_emb, axis=1)  # adding temporal dimension

        for i, block in enumerate(self.blocks):
            layer_name = f'layer_{i}'
            cache[layer_name], x = block(
                x,
                current_token_position,
                cache[layer_name],
                attention_mask,
                time_step,
            )

        x = self.final_norm(x)
        logits = self.embedder.decode(x)

        return logits, cache
