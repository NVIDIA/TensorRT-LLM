# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Contract between nvidia/Cosmos3-Edge-FP8 and the Cosmos3 weight remap.

The fixture is the transformer's safetensors key list. Every non-quantizer
tensor must land on the Edge module the runtime constructs, and every ModelOpt
quantizer buffer must be ignored. This does not execute a GEMM; it is the check
that can run without a GPU.
"""

import importlib.util
import json
from pathlib import Path

import pytest

# The CPU lane collects a file only when this marker is present in its source.
pytestmark = pytest.mark.cpu_only


def _load_remap():
    """Import the remap without requiring a CUDA build of TensorRT-LLM.

    A normal installation goes through the package. A workstation without
    torch can still load this module by file path; ``checkpoint_keys`` has no
    further dependencies.
    """
    try:
        from tensorrt_llm._torch.visual_gen.models.cosmos3.checkpoint_keys import (
            remap_cosmos3_checkpoint_key,
        )

        return remap_cosmos3_checkpoint_key
    except ImportError:
        path = (
            Path(__file__).resolve().parents[4]
            / "tensorrt_llm/_torch/visual_gen/models/cosmos3/checkpoint_keys.py"
        )
        spec = importlib.util.spec_from_file_location("cosmos3_checkpoint_keys", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.remap_cosmos3_checkpoint_key


remap_cosmos3_checkpoint_key = _load_remap()

_DATA = Path(__file__).resolve().parent / "testdata"
_NUM_LAYERS = 28
_FP8_SUFFIXES = ("weight", "weight_scale", "input_scale")


def _expected_edge_runtime_keys() -> set[str]:
    """Parameter names the Nemotron-dense static-FP8 transformer constructs."""
    names = {
        "vae2llm.weight",
        "vae2llm.bias",
        "llm2vae.weight",
        "llm2vae.bias",
        "time_embedder.mlp.linear_1.weight",
        "time_embedder.mlp.linear_1.bias",
        "time_embedder.mlp.linear_2.weight",
        "time_embedder.mlp.linear_2.bias",
        "action_proj_in.fc.weight",
        "action_proj_in.bias.weight",
        "action_proj_out.fc.weight",
        "action_proj_out.bias.weight",
        "action_modality_embed",
        "language_model.embed_tokens.weight",
        "norm_moe_gen.weight",
    }
    for index in range(_NUM_LAYERS):
        understanding = f"language_model.layers.{index}"
        generation = f"gen_layers.{index}"
        for projection in (
            "self_attn.to_q",
            "self_attn.to_k",
            "self_attn.to_v",
            "self_attn.to_out.0",
            "mlp.up_proj",
            "mlp.down_proj",
        ):
            for suffix in _FP8_SUFFIXES:
                names.add(f"{understanding}.{projection}.{suffix}")
        for projection in (
            "cross_attention.to_q",
            "cross_attention.to_k",
            "cross_attention.to_v",
            "cross_attention.to_out.0",
            "mlp.up_proj",
            "mlp.down_proj",
        ):
            for suffix in _FP8_SUFFIXES:
                names.add(f"{generation}.{projection}.{suffix}")
        names.update(
            {
                f"{understanding}.input_layernorm.weight",
                f"{understanding}.post_attention_layernorm.weight",
                f"{understanding}.self_attn.k_norm_und_for_gen.weight",
                f"{generation}.input_layernorm.weight",
                f"{generation}.post_attention_layernorm.weight",
                f"{generation}.cross_attention.norm_q.weight",
                f"{generation}.cross_attention.norm_k.weight",
            }
        )
    return names


def test_edge_fp8_manifest_covers_the_runtime_modules():
    keys = json.loads((_DATA / "cosmos3_edge_fp8_keys.json").read_text())
    quant = json.loads((_DATA / "cosmos3_edge_fp8_quant.json").read_text())
    policy = quant["runtime"]["diffusion_step_policy"]

    assert quant["backbone_type"] == "cosmos3_edge_nemotron_dense"
    assert quant["num_hidden_layers"] == _NUM_LAYERS
    assert quant["hidden_act"] == "relu2"
    assert quant["sound_gen"] is False
    assert quant["action_gen"] is True
    assert quant["quant_algo"] == "FP8"
    assert quant["quant_method"] == "modelopt"
    assert quant["quant_type"] == "FP8_FP8"
    assert quant["weight_only"] is False
    assert quant["config_groups"]["group_0"]["weights"] == {
        "dynamic": False,
        "num_bits": 8,
        "type": "float",
    }
    assert quant["config_groups"]["group_0"]["input_activations"]["dynamic"] is False
    assert policy["schema_version"] == 1
    assert policy["type"] == "first_last_n"
    assert policy["index_space"] == "denoising_loop_iteration"
    assert policy["scope"] == ["transformer"]
    assert policy["default_mode"] == "native"
    assert policy["overlap"] == "a16"
    assert policy["first_steps"] == {"count": 3, "mode": "a16"}
    assert policy["last_steps"] == {"count": 3, "mode": "a16"}
    assert policy["reasoner"] == "a16"
    assert "gate_proj" not in "".join(keys)
    assert not any("nvfp4" in key or "fp4" in key for key in keys)

    remapped = set()
    skipped = []
    unmatched = []
    for key in keys:
        action, mapped = remap_cosmos3_checkpoint_key(key)
        if action == "remap":
            remapped.add(mapped)
        elif action == "skip":
            skipped.append(mapped)
        elif action == "quantizer":
            skipped.append(mapped)
        else:
            unmatched.append(key)

    assert unmatched == []
    assert remapped == _expected_edge_runtime_keys()
    assert {key for key in skipped if "quantizer" not in key} == {"lm_head.weight", "norm.weight"}
    scale_keys = [key for key in keys if key.endswith("weight_scale")]
    assert len(scale_keys) == _NUM_LAYERS * 6 * 2


def test_quantizer_buffers_do_not_shadow_real_weights():
    action, mapped = remap_cosmos3_checkpoint_key("layers.0.self_attn.to_q.input_quantizer._amax")
    assert action == "quantizer"
    assert mapped == "layers.0.self_attn.to_q.input_quantizer._amax"
    action, mapped = remap_cosmos3_checkpoint_key("layers.0.self_attn.to_q.weight_scale")
    assert action == "remap"
    assert mapped == "language_model.layers.0.self_attn.to_q.weight_scale"


if __name__ == "__main__":
    test_edge_fp8_manifest_covers_the_runtime_modules()
    test_quantizer_buffers_do_not_shadow_real_weights()
    print("cosmos3 edge fp8 manifest: ok")
