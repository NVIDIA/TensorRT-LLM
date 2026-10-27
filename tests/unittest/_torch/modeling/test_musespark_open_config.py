# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Muse Spark Open config: translating the checkpoint's composite config.

The translation must reproduce the flat config the runtime reads, and must
reject a checkpoint that changes the model rather than its spelling. The
config below is the released checkpoint's ``config.json``, reduced to the
fields the translation reads. The released checkpoints themselves are loaded
too when they are present under ``LLM_MODELS_ROOT``.
"""

import json
import os

import pytest
from utils.llm_data import llm_models_root

from tensorrt_llm._torch.configs import MuseSparkOpenConfig
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.pyexecutor.config_utils import load_pretrained_config
from tensorrt_llm.quantization.mode import QuantAlgo

pytestmark = pytest.mark.cpu_only

NUM_LAYERS = 62
NUM_VISION_LAYERS = 50


def open_format_config_dict():
    layer_types = []
    layer_rope_theta = []
    for i in range(NUM_LAYERS):
        nope = (NUM_LAYERS - i - 1) % 4 == 0
        layer_types.append("full_attention" if nope else "sliding_attention")
        layer_rope_theta.append(0 if nope else 500000.0)
    return {
        "architectures": ["MuseSparkForConditionalGeneration"],
        "model_type": "muse_spark",
        "dtype": "bfloat16",
        "image_token_id": 200092,
        "video_token_id": 200091,
        "audio_token_id": 200038,
        "out_hidden_size": 6144,
        "projector_hidden_size": 4096,
        "projector_hidden_act": "gelu",
        "text_config": {
            "model_type": "muse_spark_text",
            "vocab_size": 202048,
            "hidden_size": 8192,
            "num_hidden_layers": NUM_LAYERS,
            "num_attention_heads": 128,
            "num_key_value_heads": 16,
            "head_dim": 64,
            "max_position_embeddings": 1048576,
            "rms_norm_eps": 1e-5,
            "post_norm_eps": 1e-8,
            "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
            "layer_types": layer_types,
            "layer_rope_theta": layer_rope_theta,
            "sliding_window": 2048,
            "mlp_layer_types": ["sparse"] * NUM_LAYERS,
            "hidden_act": "silu",
            "attention_bias": False,
            "attention_dropout": 0.0,
            "intermediate_size": 16384,
            "moe_hidden_size": 4096,
            "moe_intermediate_size": 4096,
            "num_local_experts": 256,
            "num_experts_per_tok": 8,
            "num_shared_experts": 0,
            "routed_scaling_factor": 1.0,
            "final_logit_softcapping": 20.0,
            "output_multiplier": 0.17677669529663687,
            "qk_scale_factor": 5.4730064863875,
            "residual_gate_temperature": 0.3,
            "tie_word_embeddings": False,
            "bos_token_id": 200000,
            "eos_token_id": 200001,
            "pad_token_id": 200018,
        },
        "vision_config": {
            "model_type": "muse_spark_vision",
            "hidden_size": 1536,
            "num_hidden_layers": NUM_VISION_LAYERS,
            "num_attention_heads": 16,
            "intermediate_size": 8960,
            "layer_norm_eps": 1e-5,
            "patch_size": 14,
            "patch_temporal": 2,
            "merge_size": 2,
            "pos_emb_height": 32,
            "pos_emb_width": 32,
            "rope_parameters": {"rope_theta": 10000, "rope_type": "axial"},
            "hidden_act": "gelu",
            "layer_types": [
                "full_attention"
                if (i + 1) % 4 == 0 or i == NUM_VISION_LAYERS - 1
                else "window_attention"
                for i in range(NUM_VISION_LAYERS)
            ],
        },
        "audio_config": {
            "model_type": "muse_spark_audio",
            "patch_sizes": [16, 6, 5, 4],
            "hidden_sizes": [256, 512, 1024, 2048],
            "depths": [2, 2, 2, 2],
            "num_hidden_layers": 4,
            "conv_kernel_size": 7,
            "hidden_act": "gelu",
            "mlp_ratio": 4.0,
            "out_hidden_size": 8192,
            "rms_norm_eps": 1e-6,
        },
        # The NVFP4 checkpoint's, without its 762-entry modules_to_not_convert.
        "quantization_config": {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "group_size": 16,
            "exclude_modules": ["layers.0.", "layers.61.", "self_attn", "lm_head"],
        },
    }


def test_from_open_format_reproduces_flat_config():
    flat = MuseSparkOpenConfig.from_open_format(open_format_config_dict())
    reference = MuseSparkOpenConfig(
        moe={"topk": 8, "num_experts": 256, "override_routed_hidden_dim": 4096},
        metap={"base_width": 256.0},
        vision_config={
            "navit_config": {"vision_output_dim": 6144},
            "projector_input_dim": 4096,
        },
    )
    for field in (
        "vocab_size",
        "hidden_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "max_position_embeddings",
        "rms_norm_eps",
        "rms_post_norm_eps",
        "rope_theta",
        "every_n_layers_nope",
        "global_attn_cfg",
        "output_soft_cap_temp",
        "num_experts",
        "moe_topk",
        "compression_factor",
        "expert_hidden_size",
        "intermediate_size",
        "sliding_window",
        "layer_types",
        "use_rope_per_layer",
        "scale_query_by",
        "metap_multiplier",
        "patch_token_id",
        "video_token_id",
        "audio_token_id",
    ):
        assert getattr(flat, field) == getattr(reference, field), field
    for field in (
        "hidden_size",
        "num_hidden_layers",
        "intermediate_size",
        "downsample_factor",
        "sparse_attention_factor",
        "image_size",
        "vision_output_dim",
        "projector_input_dim",
        "full_attention_layers",
    ):
        assert getattr(flat.vision_config, field) == getattr(reference.vision_config, field), field
    # The checkpoint publishes the audio tower as audio_config; the runtime's
    # config gate for it is synthesized.
    assert flat.audio_perception_modality_args["input_mode"] == "wav"
    assert flat.model_type == "muse_spark"
    assert flat.torch_dtype == reference.torch_dtype


def _only_the_last_layer_is_nope(config):
    text = config["text_config"]
    text["layer_rope_theta"] = [500000.0] * (NUM_LAYERS - 1) + [0]
    text["layer_types"] = ["sliding_attention"] * (NUM_LAYERS - 1) + ["full_attention"]


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda c: c["text_config"].__setitem__("hidden_act", "gelu"), "hidden_act"),
        (
            lambda c: c["text_config"].__setitem__("residual_gate_temperature", 0.5),
            "residual_gate_temperature",
        ),
        (lambda c: c["text_config"].__setitem__("qk_scale_factor", 1.0), "qk_scale_factor"),
        (lambda c: c["text_config"].__setitem__("output_multiplier", 0.3), "output_multiplier"),
        (lambda c: c["text_config"]["layer_rope_theta"].__setitem__(3, 123.0), "rope"),
        (_only_the_last_layer_is_nope, "period cannot be inferred"),
        (
            lambda c: c["text_config"]["rope_parameters"].__setitem__("rope_type", "yarn"),
            "rope_type",
        ),
        (
            lambda c: c["text_config"]["layer_types"].__setitem__(0, "full_attention"),
            "layer_types|residue",
        ),
        (
            lambda c: c["text_config"].__setitem__("mlp_layer_types", ["dense"] * NUM_LAYERS),
            "sparse",
        ),
        (lambda c: c["audio_config"].__setitem__("patch_sizes", [8, 8, 6, 5]), "patch"),
        (
            lambda c: c["vision_config"]["layer_types"].__setitem__(0, "full_attention"),
            "sparse_attention_factor",
        ),
        (lambda c: c.__setitem__("image_token_id", 12345), "image_token_id"),
        (lambda c: c.__setitem__("dtype", "float16"), "dtype"),
        # A config written before transformers renamed the key.
        (lambda c: (c.pop("dtype"), c.__setitem__("torch_dtype", "float16")), "torch_dtype"),
    ],
)
def test_from_open_format_rejects_semantic_changes(mutate, match):
    config = open_format_config_dict()
    mutate(config)
    with pytest.raises(ValueError, match=match):
        MuseSparkOpenConfig.from_open_format(config)


def test_load_pretrained_config_translates_the_composite_config(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(open_format_config_dict()))
    config = load_pretrained_config(str(tmp_path))
    assert isinstance(config, MuseSparkOpenConfig)
    assert config.model_type == "muse_spark"
    assert config.architectures == ["MuseSparkForConditionalGeneration"]
    assert config.num_hidden_layers == NUM_LAYERS
    assert config.vision_config.num_hidden_layers == NUM_VISION_LAYERS
    assert config.quantization_config == open_format_config_dict()["quantization_config"]


def test_the_inline_quantization_config_is_the_fallback(tmp_path):
    """Without ``hf_quant_config.json`` the NVFP4 quant config comes from config.json."""
    (tmp_path / "config.json").write_text(json.dumps(open_format_config_dict()))
    quant_config = ModelConfig.from_pretrained(str(tmp_path)).quant_config
    assert quant_config.quant_algo == QuantAlgo.NVFP4
    assert quant_config.group_size == 16


@pytest.mark.parametrize(
    "checkpoint", ["Muse-Spark-Open-816B-A42B", "Muse-Spark-Open-816B-A42B-NVFP4"]
)
def test_the_released_checkpoints_load(checkpoint):
    path = os.path.join(llm_models_root(), checkpoint)
    # An absent or empty directory is a checkpoint not synced to this machine
    # (a submodule's directory exists before its contents arrive). A synced
    # checkpoint must load.
    if not os.path.isdir(path) or not os.listdir(path):
        pytest.skip(f"{checkpoint} is not under LLM_MODELS_ROOT")
    config = load_pretrained_config(path)
    assert isinstance(config, MuseSparkOpenConfig)
    assert config.architectures == ["MuseSparkForConditionalGeneration"]
    assert (config.num_hidden_layers, config.hidden_size) == (NUM_LAYERS, 8192)
    assert (config.num_experts, config.moe_topk, config.sliding_window) == (256, 8, 2048)
    assert config.vision_config.num_hidden_layers == NUM_VISION_LAYERS
    quant_algo = (getattr(config, "quantization_config", None) or {}).get("quant_algo")
    assert quant_algo == ("NVFP4" if checkpoint.endswith("-NVFP4") else None)
