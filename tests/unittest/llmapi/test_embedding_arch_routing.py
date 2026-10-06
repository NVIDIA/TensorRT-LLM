# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""CPU unit tests for embedding-server config handling in serve.py."""

import json

import click
import pytest


def _write_config(tmp_path, architectures, with_pooling=True):
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "architectures": architectures})
    )
    if with_pooling:
        pooling_dir = tmp_path / "1_Pooling"
        pooling_dir.mkdir()
        (pooling_dir / "config.json").write_text(json.dumps({"pooling_mode_lasttoken": True}))
    return str(tmp_path)


def test_qwen3_causal_lm_is_remapped_to_text_embedding(tmp_path):
    from tensorrt_llm.commands.serve import _resolve_embedding_architecture_override

    model = _write_config(tmp_path, ["Qwen3ForCausalLM"], with_pooling=True)
    override = _resolve_embedding_architecture_override(model, trust_remote_code=False)
    assert override == {"architectures": ["Qwen3ForTextEmbedding"]}


def test_unknown_architecture_is_left_alone(tmp_path):
    from tensorrt_llm.commands.serve import _resolve_embedding_architecture_override

    model = _write_config(tmp_path, ["BertForSequenceClassification"])
    override = _resolve_embedding_architecture_override(model, trust_remote_code=False)
    assert override is None


_TOKEN_TYPE_IDS_SPEC = {"name": "token_type_ids", "shape": ["num_tokens"], "dtype": "int32"}


@pytest.mark.parametrize(
    "cuda_graph_config",
    [
        {"num_tokens": [64], "seq_lens": [32], "extra_model_inputs": [_TOKEN_TYPE_IDS_SPEC]},
        "parsed",
    ],
    ids=["yaml_mapping", "parsed_config"],
)
def test_embeddings_server_rejects_extra_model_inputs(cuda_graph_config):
    from tensorrt_llm.commands.serve import _reject_embedding_extra_model_inputs
    from tensorrt_llm.llmapi import EncodeCudaGraphConfig

    if cuda_graph_config == "parsed":
        cuda_graph_config = EncodeCudaGraphConfig(
            num_tokens=[64], seq_lens=[32], extra_model_inputs=[_TOKEN_TYPE_IDS_SPEC]
        )
    with pytest.raises(click.BadParameter, match="extra_model_inputs"):
        _reject_embedding_extra_model_inputs(cuda_graph_config)


@pytest.mark.parametrize(
    "cuda_graph_config",
    [
        None,
        {"num_tokens": [64], "seq_lens": [32]},
        {"num_tokens": [64], "seq_lens": [32], "extra_model_inputs": []},
    ],
    ids=["unset", "graphs_without_extra_inputs", "empty_extra_inputs"],
)
def test_embeddings_server_accepts_config_without_extra_model_inputs(cuda_graph_config):
    from tensorrt_llm.commands.serve import _reject_embedding_extra_model_inputs

    _reject_embedding_extra_model_inputs(cuda_graph_config)
