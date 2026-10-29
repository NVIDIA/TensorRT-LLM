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

import json

import pytest

from tensorrt_llm._torch.models.checkpoints.mistral.config_loader import adapt_config_dict
from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig, TorchLlmArgs

pytestmark = pytest.mark.cpu_only


def test_mistral_large3_vision_config_preserves_max_position_embeddings():
    config = {
        "dim": 7168,
        "hidden_dim": 16384,
        "llama_4_scaling": {
            "beta": 0.1,
            "original_max_position_embeddings": 8192,
        },
        "max_position_embeddings": 294912,
        "max_seq_len": 262144,
        "moe": {
            "expert_hidden_dim": 4096,
            "first_k_dense_replace": 3,
            "num_expert_groups": 1,
            "num_expert_groups_per_tok": 1,
            "num_experts": 128,
            "num_experts_per_tok": 4,
            "num_shared_experts": 1,
            "route_every_n": 1,
            "routed_scale": 1.0,
        },
        "n_heads": 128,
        "n_kv_heads": 128,
        "n_layers": 61,
        "norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "tied_embeddings": False,
        "vision_encoder": {
            "hidden_size": 1664,
            "image_size": 1540,
            "intermediate_size": 8192,
            "num_attention_heads": 16,
            "num_channels": 3,
            "num_hidden_layers": 48,
            "patch_size": 14,
            "rope_theta": 10000.0,
        },
        "vocab_size": 131072,
        "yarn": {
            "alpha": 1,
            "apply_scale": False,
            "beta": 32,
            "factor": 36,
            "original_max_position_embeddings": 8192,
        },
    }

    adapted = adapt_config_dict(config)

    assert adapted.text_config.max_position_embeddings == 294912
    assert adapted.text_config.rope_scaling["rope_type"] == "yarn"


@pytest.mark.parametrize("explicit_path", [False, True])
@pytest.mark.parametrize("transport", ["pickle", "public_fields"])
def test_mistral_dspark_uses_native_checkpoint_once(tmp_path, explicit_path, transport):
    import pickle

    native = {
        "n_layers": 3,
        "depth": 3,
        "target_activation_layers": "1,18,36,54,71",
        "ragged_attention": "2048,2048,2048",
        "norm_strategy": "eagle",
        "dspark": {"markov_rank": 512, "head_type": "rnn", "draft_input": "position_specific"},
    }
    (tmp_path / "params.json").write_text(json.dumps({"vocab_size": 204800, "mtp": native}))
    spec = DSparkDecodingConfig(
        max_draft_len=3,
        speculative_model=str(tmp_path) if explicit_path else None,
    )
    args = TorchLlmArgs(model=str(tmp_path), skip_tokenizer_init=True, speculative_config=spec)
    # The resolved geometry must survive MPI transport, including private attrs.
    if transport == "pickle":
        resolved = pickle.loads(pickle.dumps(args.speculative_config))
    else:
        public_fields = args.speculative_config.model_dump()
        assert not any(name.startswith("_") for name in public_fields)
        resolved = TorchLlmArgs(
            model=str(tmp_path), skip_tokenizer_init=True, speculative_config=public_fields
        ).speculative_config
    assert resolved.target_layer_ids == [1, 18, 36, 54, 71]
    assert resolved.block_size == 3
    assert resolved.mask_token_id == 204800
    assert resolved.markov_head_type == "rnn" and resolved.markov_rank == 512
    assert resolved._draft_model_config == native
    assert not resolved.needs_separate_draft_weights
    assert not resolved.draft_is_embedded_in_target
    from tensorrt_llm._torch.speculative import should_use_separate_draft_kv_cache

    assert should_use_separate_draft_kv_cache(resolved)


@pytest.mark.parametrize(
    "override,field",
    [
        ({"max_draft_len": 2}, "max_draft_len"),
        ({"target_layer_ids": [0, 17, 35, 53, 70]}, "target_layer_ids"),
        ({"markov_head_type": "vanilla"}, "markov_head_type"),
        ({"markov_rank": 0}, "markov_rank"),
        ({"mask_token_id": 42}, "mask_token_id"),
        ({"block_size": 4}, "block_size"),
    ],
)
def test_mistral_dspark_rejects_untrained_geometry(tmp_path, override, field):
    (tmp_path / "params.json").write_text(
        json.dumps(
            {
                "vocab_size": 204800,
                "mtp": {
                    "depth": 3,
                    "target_activation_layers": "1,18,36,54,71",
                    "dspark": {"markov_rank": 512, "head_type": "rnn"},
                },
            }
        )
    )
    spec = DSparkDecodingConfig(**({"max_draft_len": 3} | override))
    with pytest.raises(ValueError, match=field):
        TorchLlmArgs(model=str(tmp_path), skip_tokenizer_init=True, speculative_config=spec)


@pytest.mark.parametrize("layers", ["1,1,2", "2,1,3", "-1,2,3"])
def test_native_dspark_rejects_invalid_capture_layers(tmp_path, layers):
    (tmp_path / "params.json").write_text(
        json.dumps(
            {
                "vocab_size": 204800,
                "mtp": {"target_activation_layers": layers, "dspark": {}},
            }
        )
    )
    with pytest.raises(ValueError, match="unique and ascending"):
        TorchLlmArgs(
            model=str(tmp_path),
            skip_tokenizer_init=True,
            speculative_config=DSparkDecodingConfig(max_draft_len=3),
        )


@pytest.mark.parametrize("same_checkpoint", [False, True])
def test_native_dspark_target_path_identity(tmp_path, same_checkpoint):
    target = tmp_path / "target"
    target.mkdir()
    draft = tmp_path / "draft"
    if same_checkpoint:
        draft.symlink_to(target, target_is_directory=True)
    else:
        draft.mkdir()
    native = {
        "depth": 3,
        "target_activation_layers": "1,18,36,54,71",
        "dspark": {"markov_rank": 512, "head_type": "rnn"},
    }
    (draft / "params.json").write_text(json.dumps({"vocab_size": 204800, "mtp": native}))
    kwargs = dict(
        model=str(target),
        skip_tokenizer_init=True,
        speculative_config=DSparkDecodingConfig(max_draft_len=3, speculative_model=str(draft)),
    )
    if same_checkpoint:
        args = TorchLlmArgs(**kwargs)
        assert not args.speculative_config.needs_separate_draft_weights
    else:
        with pytest.raises(ValueError, match="must use the target checkpoint"):
            TorchLlmArgs(**kwargs)


@pytest.mark.parametrize("embedded", [False, True])
def test_hf_dspark_preserves_weight_and_runtime_ownership(tmp_path, embedded):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "deepseek_v4" if embedded else "llama",
                "dspark_block_size": 3,
                "dspark_target_layer_ids": [1, 2],
            }
        )
    )
    args = TorchLlmArgs(
        model=str(tmp_path),
        skip_tokenizer_init=True,
        speculative_config=DSparkDecodingConfig(
            max_draft_len=3, speculative_model=None if embedded else str(tmp_path)
        ),
    )
    spec = args.speculative_config
    assert spec._draft_model_config is None
    assert spec.needs_separate_draft_weights
    assert spec.draft_is_embedded_in_target == embedded
    from tensorrt_llm._torch.speculative import should_use_separate_draft_kv_cache

    assert should_use_separate_draft_kv_cache(spec) != embedded
