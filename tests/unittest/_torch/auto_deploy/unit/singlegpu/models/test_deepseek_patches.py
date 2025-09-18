"""Testing module patches that enable export of deepseek model."""

import types

import pytest
import torch
from _model_test_utils import _hf_model_dir_or_hub_id
from transformers import AutoConfig, AutoModelForCausalLM

from tensorrt_llm._torch.auto_deploy.models.patches.deepseek import (
    deepseek_v3_attention,
    deepseek_v3_moe_exact,
)


def _load_layer_from_model(model_name_or_path, layer_name):
    """
    Loads a specific layer/module from a model without loading the entire model.

    Parameters:
        model_name_or_path (str): Path or name of the pretrained model.
        layer_name (str): Name of the layer to extract.

    Returns:
        module: The specified layer/module if available, otherwise None.
    """
    try:
        # Load only the model configuration
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

        # Load a subset of layers of the model and configure yarn
        config.num_hidden_layers = 1
        config.use_cache = False
        config.first_k_dense_replace = 0
        config.n_routed_experts = 2
        config.num_experts_per_tok = 1
        config.n_group = 1
        config.topk_group = 1
        config.hidden_size = 8
        config.moe_intermediate_size = 8
        config.num_attention_heads = 2
        config.num_key_value_heads = 2
        config.qk_nope_head_dim = 4
        config.qk_rope_head_dim = 2
        config.v_head_dim = 4
        config.intermediate_size = 8
        config.max_position_embeddings = 7

        config.rope_scaling = None

        # Build the model architecture (no weights loaded yet)
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        model.eval()

        # Access the specific layer by its name
        module = dict(model.named_modules()).get(layer_name)
        if module is None:
            print(f"Layer '{layer_name}' not found in the model.")
        else:
            print(f"Successfully extracted layer '{layer_name}'.")
        return module
    except Exception as e:
        print(f"Error extracting layer: {e}")
        return None


def _generate_ds_attention_mask(b, s):
    return torch.where(
        torch.tril(torch.full((s, s), float("-inf"))).unsqueeze(0).unsqueeze(0).expand(b, 1, s, s)
        == float("-inf"),
        torch.tensor(0.0),
        torch.tensor(float(-3.4028e38)),
    )


@pytest.mark.parametrize(
    "model_name, module_name, patch, inputs",
    [
        pytest.param(
            _hf_model_dir_or_hub_id("DeepSeek-R1/DeepSeek-R1", "deepseek-ai/DeepSeek-R1"),
            "model.layers.0.self_attn",
            deepseek_v3_attention,
            [
                torch.randn(2, 6, 8, dtype=torch.bfloat16),
                _generate_ds_attention_mask(2, 6),
                torch.tensor([[0, 1, 2, 3, 4, 5]]),
            ],
        ),  # attention requires  inputs [hidden_states, attention_mask, position_ids]
        pytest.param(
            _hf_model_dir_or_hub_id("DeepSeek-R1/DeepSeek-R1", "deepseek-ai/DeepSeek-R1"),
            "model.layers.0.mlp",
            deepseek_v3_moe_exact,
            [torch.randn(2, 6, 8, dtype=torch.bfloat16)],
        ),  # moe requires  inputs [hidden_states]
    ],
)
def test_module_patches(model_name, module_name, patch, inputs):
    # Get module
    module = _load_layer_from_model(model_name, module_name)

    # Pass test inputs to generate reference
    ref, *_ = module(*inputs)

    # Patch layer
    module.forward = types.MethodType(patch, module)

    # Generate test output
    test, *_ = module(*inputs)

    torch.allclose(ref, test, atol=0, rtol=0)
