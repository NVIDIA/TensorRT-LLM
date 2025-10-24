"""Test that the attention matcher works with HF's SDPA backends."""

import copy
from typing import Any, Dict

import pytest
import torch
import torch.nn as nn
from accelerate import init_empty_weights
from torch.export import Dim
from torch.fx import GraphModule
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaModel

from tensorrt_llm._torch.auto_deploy.export import torch_export_to_gm
from tensorrt_llm._torch.auto_deploy.transform.optimizer import InferenceOptimizer
from tensorrt_llm._torch.auto_deploy.utils._graph import move_to_device
from tensorrt_llm._torch.auto_deploy.utils.node_utils import is_op

torch.manual_seed(0)


class HFWrapper(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @torch.inference_mode()
    def forward(self, x: torch.Tensor):
        return self.model(x)[0]


def _joint_transform(gm: GraphModule) -> None:
    gm = InferenceOptimizer(
        None,
        {
            "match_repeat_kv": {
                "stage": "pattern_matcher",
                "run_shape_prop": True,
            },
            "match_eager_attention": {
                "stage": "pattern_matcher",
            },
            "match_grouped_attention_with_repeat_kv": {
                "stage": "pattern_matcher",
            },
            "match_grouped_attention_without_repeat_kv": {
                "stage": "pattern_matcher",
            },
            "match_attention_layout": {
                "stage": "pattern_matcher",
                "attn_layout": "bsnd",
            },
        },
    )(None, gm)


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
    if attn_implementation == "sdpa":
        pytest.skip("https://nvbugspro.nvidia.com/bug/5170222")

    def verify_matcher(gm: GraphModule):
        """Ensure that there is exactly one torch.ops.auto_deploy.torch_attention (layout="bsnd")
        call in the graph. Also check that there is no repeat_kv pattern left.
        """
        nodes = [
            n
            for n in gm.graph.nodes
            if (
                is_op(n, torch.ops.auto_deploy.torch_attention)
                and isinstance(n.args[-1], str)
                and n.args[-1] == "bsnd"
            )
        ]
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
        nodes = gm.graph.find_nodes(
            op="call_function", target=torch.ops.auto_deploy.torch_attention_repeat_kv
        )
        assert len(nodes) == 0, "Found repeat_kv pattern in the graph"
        attn_nodes = gm.graph.find_nodes(
            op="call_function", target=torch.ops.auto_deploy.torch_attention_sdpa
        )
        assert len(attn_nodes) == 0, "Found torch_attention_sdpa node in the graph"

        return True

    batch_size, seq_len = 2, 4
    full_config = {
        "num_hidden_layers": 1,
        "vocab_size": 256,
        "hidden_size": 128,
        "intermediate_size": 128,
        "attn_implementation": attn_implementation,
        **config,
    }
    dynamic_shapes = {0: Dim("batch_size", max=8), 1: Dim("seq_len", min=2, max=8)}

    # Build and export model on meta device
    with init_empty_weights():
        model = HFWrapper(LlamaModel(LlamaConfig(**full_config))).eval()
    x = torch.randint(
        0, full_config["vocab_size"], (batch_size, seq_len), dtype=torch.long, device="cuda"
    )
    gm = torch_export_to_gm(model, args=(x,), dynamic_shapes=(dynamic_shapes,), clone=True)

    print("Exported gm", gm)
    gm_exported = copy.deepcopy(gm)

    # Move model to cuda
    device = "cuda"
    model._apply(
        lambda t: torch.normal(0.0, 1.0, size=t.shape, device=device).to(t.dtype)
        if t.device == torch.device("meta")
        else t.to(device)
    )
    y_model = model(x)

    gm_exported._apply(
        lambda t: torch.normal(0.0, 1.0, size=t.shape, device=device).to(t.dtype)
        if t.device == torch.device("meta")
        else t.to(device)
    )
    gm_exported.load_state_dict(model.state_dict())
    move_to_device(gm_exported, "cuda")
    y_gm_exported = gm_exported(x)
    torch.testing.assert_close(y_gm_exported, y_model, atol=5e-3, rtol=5e-3)

    # Apply transformation
    _joint_transform(gm)
    assert verify_matcher(gm)
    print("Transformed gm", gm)

    # Move gm to cuda
    gm._apply(
        lambda t: torch.normal(0.0, 1.0, size=t.shape, device=device).to(t.dtype)
        if t.device == torch.device("meta")
        else t.to(device)
    )
    gm.load_state_dict(model.state_dict())
    move_to_device(gm, "cuda")

    # Verify output
    y_gm = gm(x)
    torch.testing.assert_close(y_gm_exported, y_gm, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(y_model, y_gm, atol=5e-2, rtol=5e-2)
