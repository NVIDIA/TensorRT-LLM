import types

import pytest
import torch

from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm.mapping import Mapping


def make_pretrained_config(
    *,
    num_attention_heads: int = 16,
    num_key_value_heads=8,
    head_dim: int | None = None,
    num_hidden_layers: int = 1,
    vocab_size: int = 3000,
):
    # A minimal config object that provides the attributes used by
    # ModelConfig.get_bindings_model_config().
    hidden_size = head_dim * num_attention_heads
    intermediate_size = hidden_size * 4

    return types.SimpleNamespace(
        architectures=["DummyArchitecture"],
        num_attention_heads=num_attention_heads,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        num_hidden_layers=num_hidden_layers,
        vocab_size=vocab_size,
        torch_dtype=torch.float16,
    )


@pytest.mark.parametrize(
    "num_key_value_heads",
    [
        pytest.param(8, id="kv_heads_scalar"),
        pytest.param([8, 20], id="kv_heads_per_layer_varied"),
    ],
)
@pytest.mark.parametrize("enable_attention_dp", [False, True])
@pytest.mark.parametrize(
    "mapping_kwargs",
    [
        # Same tp/cp sizes, but different ways of setting attention TP:
        # - No explicit attn_tp_size: Mapping infers it.
        # - Explicit attn_tp_size: Mapping uses the provided value.
        dict(world_size=8, tp_size=4, cp_size=2),
        dict(world_size=4, tp_size=2, cp_size=2, attn_tp_size=4),
    ],
)
def test_get_bindings_model_config_attention_dp_attn_tp_override(
    enable_attention_dp, mapping_kwargs, num_key_value_heads
):
    mapping = Mapping(enable_attention_dp=enable_attention_dp, **mapping_kwargs)
    cfg = make_pretrained_config(
        # Keep values consistent:
        # hidden_size = num_attention_heads * head_dim.
        num_attention_heads=16,
        head_dim=4,
        num_key_value_heads=num_key_value_heads,
        num_hidden_layers=2,
    )
    model_config = ModelConfig(pretrained_config=cfg, mapping=mapping)

    tokens_per_block = 32
    bindings_cfg = model_config.get_bindings_model_config(tokens_per_block=tokens_per_block)

    # bindings hidden_size is sharded by attn_tp_size and attn_cp_size.
    attn_tp_size = mapping.attn_tp_size if not mapping.enable_attention_dp else 1
    attn_cp_size = mapping.attn_cp_size
    assert bindings_cfg.num_heads == cfg.num_attention_heads // (attn_tp_size * attn_cp_size)
    # bindings hidden_size is sharded by attn_tp_size.
    assert bindings_cfg.hidden_size == cfg.hidden_size // attn_tp_size
    if isinstance(cfg.num_key_value_heads, (list, tuple)):
        expected_num_kv_heads_per_layer = [
            kv // (attn_tp_size * attn_cp_size) for kv in cfg.num_key_value_heads
        ]
        assert list(bindings_cfg.num_kv_heads_per_layer) == expected_num_kv_heads_per_layer
        assert bindings_cfg.num_kv_heads(0) == expected_num_kv_heads_per_layer[0]
    else:
        assert bindings_cfg.num_kv_heads(0) == cfg.num_key_value_heads // (
            attn_tp_size * attn_cp_size
        )

    # tp_size-dependent value (uses mapping.tp_size, not attn_tp_size).
    assert bindings_cfg.mlp_hidden_size == (cfg.intermediate_size // mapping.tp_size)
    assert bindings_cfg.tokens_per_block == tokens_per_block
