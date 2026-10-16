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
"""A DSpark drafter saved in the vLLM speculators format (CPU-only, no weights).

The reference is the config that a hand conversion
(make_redhat_dspark_adapter.py) wrote for RedHatAI/Kimi-K3-speculator.dspark,
which every Kimi K3 DSpark measurement so far used. The checkpoint as downloaded
must give exactly the drafter that the converted one gave.
"""

import copy
import json

import pytest

from tensorrt_llm._torch.pyexecutor.config_utils import load_pretrained_config
from tensorrt_llm.llmapi.llm_args import (
    DSparkDecodingConfig,
    TorchLlmArgs,
    is_speculators_dspark_config,
    translate_speculators_dspark_config,
)

pytestmark = pytest.mark.cpu_only

# RedHatAI/Kimi-K3-speculator.dspark config.json, as published.
SPECULATORS_CONFIG = {
    "architectures": ["DSparkDraftModel"],
    "auto_map": {"": "config.DSparkSpeculatorConfig"},
    "aux_hidden_state_layer_ids": [24, 48, 72, 88, 92],
    "block_size": 8,
    "confidence_head_with_markov": True,
    "draft_vocab_size": 163840,
    "dtype": "bfloat16",
    "enable_confidence_head": True,
    "markov_head_type": "vanilla",
    "markov_rank": 256,
    "mask_token_id": 163837,
    "sample_from_anchor": True,
    "sliding_window_non_causal": False,
    "speculators_config": {
        "algorithm": "dspark",
        "default_proposal_method": "greedy",
        "proposal_methods": [
            {
                "accept_tolerance": 0.0,
                "proposal_type": "greedy",
                "speculative_tokens": 8,
                "verifier_accept_k": 1,
            }
        ],
        "verifier": {
            "architectures": ["KimiK3ForConditionalGeneration"],
            "name_or_path": "moonshotai/Kimi-K3",
        },
    },
    "speculators_model_type": "dspark",
    "speculators_version": "0.7.0.dev141",
    "target_hidden_size": None,
    "tie_word_embeddings": False,
    "transformer_layer_config": {
        "attention_bias": False,
        "attention_dropout": 0.0,
        "bos_token_id": 163584,
        "eos_token_id": 163586,
        "flex_attention_backend": "FLASH",
        "head_dim": 64,
        "hidden_act": "silu",
        "hidden_size": 7168,
        "initializer_range": 0.02,
        "intermediate_size": 14336,
        "layer_types": ["sliding_attention"] * 5,
        "max_position_embeddings": 1048576,
        "max_window_layers": 28,
        "model_type": "qwen3",
        "num_attention_heads": 96,
        "num_hidden_layers": 5,
        "num_key_value_heads": 16,
        "pad_token_id": 163839,
        "rms_norm_eps": 1e-05,
        "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
        "sliding_window": 2048,
        "tie_word_embeddings": False,
        "use_cache": True,
        "use_sliding_window": True,
        "vocab_size": 163840,
    },
    "transformers_version": "5.14.1",
}

# What the hand conversion wrote for it: the backbone flattened, the capture
# layers as 0-indexed layer outputs, the sliding window off.
CONVERTED_CONFIG = {
    "hidden_size": 7168,
    "intermediate_size": 14336,
    "num_hidden_layers": 5,
    "num_attention_heads": 96,
    "num_key_value_heads": 16,
    "head_dim": 64,
    "hidden_act": "silu",
    "rms_norm_eps": 1e-05,
    "max_position_embeddings": 1048576,
    "vocab_size": 163840,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "bos_token_id": 163584,
    "eos_token_id": 163586,
    "pad_token_id": 163839,
    "initializer_range": 0.02,
    "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
    "architectures": ["Qwen3ForCausalLM"],
    "model_type": "qwen3",
    "rope_theta": 10000.0,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "use_sliding_window": False,
    "sliding_window": None,
    "max_window_layers": 5,
    "layer_types": ["full_attention"] * 5,
    "dflash_config": {"mask_token_id": 163837, "target_layer_ids": [23, 47, 71, 87, 91]},
    "markov_rank": 256,
    "markov_head_type": "vanilla",
    "enable_confidence_head": True,
    "confidence_head_with_markov": True,
    "draft_vocab_size": 163840,
}

# Bookkeeping that differs between a config read from a directory and one built
# from a dict, without changing the model.
_LOAD_METADATA = ("_name_or_path", "_commit_hash", "transformers_version")


def _checkpoint(tmp_path, name, config):
    path = tmp_path / name
    path.mkdir()
    (path / "config.json").write_text(json.dumps(config))
    return str(path)


def _spec_config(draft_dir, max_draft_len=7):
    args = TorchLlmArgs(
        model="/tmp/dummy_model",
        skip_tokenizer_init=True,
        speculative_config=DSparkDecodingConfig(
            max_draft_len=max_draft_len, speculative_model=draft_dir
        ),
    )
    return args.speculative_config


def _comparable(config):
    values = config.to_dict()
    for key in _LOAD_METADATA:
        values.pop(key, None)
    return values


def test_translation_is_the_converted_config():
    # Every config.json reader of the drafter goes through this translation.
    assert translate_speculators_dspark_config(SPECULATORS_CONFIG) == CONVERTED_CONFIG


def test_drafter_config_matches_the_converted_checkpoint(tmp_path):
    native = load_pretrained_config(_checkpoint(tmp_path, "native", SPECULATORS_CONFIG))
    converted = load_pretrained_config(_checkpoint(tmp_path, "converted", CONVERTED_CONFIG))

    assert type(native) is type(converted)
    assert native.architectures == ["Qwen3ForCausalLM"]
    assert _comparable(native) == _comparable(converted)


def test_spec_config_matches_the_converted_checkpoint(tmp_path):
    native = _spec_config(_checkpoint(tmp_path, "native", SPECULATORS_CONFIG))
    converted = _spec_config(_checkpoint(tmp_path, "converted", CONVERTED_CONFIG))

    # vLLM-style aux ids name the layer whose input is captured.
    assert native.target_layer_ids == [23, 47, 71, 87, 91]
    assert native.mask_token_id == 163837
    assert native.markov_rank == 256
    # The checkpoint's block_size (8 under sample_from_anchor) does not pin
    # max_draft_len: the converted checkpoint ran at 7.
    assert native.block_size == 7
    for field in (
        "target_layer_ids",
        "mask_token_id",
        "markov_rank",
        "markov_head_type",
        "block_size",
    ):
        assert getattr(native, field) == getattr(converted, field), field


def test_trained_draft_length_is_accepted(tmp_path):
    native = _spec_config(_checkpoint(tmp_path, "native", SPECULATORS_CONFIG), max_draft_len=8)

    assert native.block_size == 8
    assert native.target_layer_ids == [23, 47, 71, 87, 91]


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("model_type", "llama", "supported with a qwen3 backbone"),
        ("aux_hidden_state_layer_ids", [0, 48, 72, 88, 92], "aux_hidden_state_layer_ids"),
    ],
)
def test_untranslatable_drafters_are_rejected(tmp_path, field, value, message):
    config = copy.deepcopy(SPECULATORS_CONFIG)
    if field == "model_type":
        config["transformer_layer_config"]["model_type"] = value
    else:
        config[field] = value

    with pytest.raises(ValueError, match=message):
        load_pretrained_config(_checkpoint(tmp_path, "native", config))


def test_other_drafter_configs_are_left_alone():
    assert is_speculators_dspark_config(SPECULATORS_CONFIG)
    assert not is_speculators_dspark_config(CONVERTED_CONFIG)
    eagle3 = dict(SPECULATORS_CONFIG, speculators_model_type="eagle3")
    assert not is_speculators_dspark_config(eagle3)
    assert translate_speculators_dspark_config(CONVERTED_CONFIG) is CONVERTED_CONFIG
    assert translate_speculators_dspark_config(eagle3) is eagle3
