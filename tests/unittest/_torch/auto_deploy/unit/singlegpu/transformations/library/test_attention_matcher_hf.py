"""Test that the attention matcher works with HF's SDPA backends."""

from typing import Any, Callable, Dict

import pytest
import torch
import torch.nn as nn
from _graph_test_helpers import run_test
from torch.export import Dim
from torch.fx import GraphModule
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel

from tensorrt_llm._torch.auto_deploy.transformations.library import (
    match_attention_layout,
    match_causal_attn_mask,
    match_eager_attention,
    match_grouped_attention,
    match_repeat_kv,
)


class MockAttentionDescriptor:
    """A mock class that mimics the AttentionDescriptor interface for testing."""

    layout: str = "bsnd"

    @classmethod
    def get_attention_layout(cls) -> str:
        return cls.layout

    @classmethod
    def get_source_attention_op(cls) -> Callable:
        return torch.ops.attention.bsnd_grouped_sdpa


class HFWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @torch.inference_mode()
    def forward(self, x: torch.Tensor):
        return self.model(x)[0]


def _joint_transform(gm: GraphModule) -> GraphModule:
    gm = match_repeat_kv(gm)
    gm = match_eager_attention(gm)
    gm = match_grouped_attention(gm)
    gm = match_causal_attn_mask(gm)
    gm = match_attention_layout(gm, MockAttentionDescriptor())
    return gm


@pytest.mark.parametrize(
    "config",
    [
        {"num_attention_heads": 8, "num_key_value_heads": 8},
        {"num_attention_heads": 8, "num_key_value_heads": 4},
        {"num_attention_heads": 8, "num_key_value_heads": 1},
    ],
)
@pytest.mark.parametrize(
    "attn_implementation",
    ["eager", "sdpa"],
)
def test_match_llama_attention(config: Dict[str, Any], attn_implementation: str):
    batch_size, seq_len = 4, 12
    full_config = {
        "num_hidden_layers": 1,
        "vocab_size": 256,
        "hidden_size": 128,
        "intermediate_size": 128,
        "attn_implementation": attn_implementation,
        **config,
    }
    dynamic_shapes = {0: Dim("batch_size", max=8), 1: Dim("seq_len", min=4, max=16)}

    model = HFWrapper(LlamaModel(LlamaConfig(**full_config))).to("cuda")
    x = torch.randint(
        0, full_config["vocab_size"], (batch_size, seq_len), dtype=torch.long, device="cuda"
    )

    def verify_matcher(gm: GraphModule):
        """Ensure that there is exactly one torch.ops.attention.bsnd_grouped_sdpa call in the graph."""
        nodes = gm.graph.find_nodes(
            op="call_function", target=torch.ops.attention.bsnd_grouped_sdpa
        )
        assert len(nodes) == 1, "Expected exactly one bsnd_grouped_sdpa call in the graph"

        # TODO: check non-qkv args of node
        attn_node = nodes[0]
        scale = model.model.layers[0].self_attn.scaling
        # TODO (lucaslie, #4783): don't check for causal mask until we have more robust handling
        # assert attn_node.args[3:] == (None, 0.0, True, scale), (
        #     "Expected default args for bsnd_grouped_sdpa"
        # )
        assert attn_node.args[4] == 0.0  # dropout_p
        assert attn_node.args[6] == scale  # scale

        # TODO: check that there is no repeat_kv pattern left...
        nodes = gm.graph.find_nodes(op="call_function", target=torch.ops.attention.repeat_kv)
        assert len(nodes) == 0, "Found repeat_kv pattern in the graph"

        return True

    _ = run_test(
        model,
        x,
        _joint_transform,
        verify_matcher,
        lambda num_p_og: num_p_og,
        atol=1e-3,
        rtol=5e-2,
        test_load_hook=True,
        strict_loading=True,
        dynamic_shapes=dynamic_shapes,
    )
