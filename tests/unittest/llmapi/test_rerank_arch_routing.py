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
"""CPU unit tests for rerank architecture-override and token-id routing in
serve.py.
"""

import json

import click
import pytest


def _write_config(tmp_path, architectures):
    (tmp_path / "config.json").write_text(
        json.dumps({
            "model_type": "qwen3",
            "architectures": architectures
        }))
    return str(tmp_path)


def test_qwen3_causal_lm_is_remapped_to_text_reranking(tmp_path):
    from tensorrt_llm.commands.serve import \
        _resolve_rerank_architecture_override

    model = _write_config(tmp_path, ["Qwen3ForCausalLM"])
    override = _resolve_rerank_architecture_override(model,
                                                      trust_remote_code=False)
    assert override == {"architectures": ["Qwen3ForTextReranking"]}


def test_unknown_architecture_is_left_alone(tmp_path):
    from tensorrt_llm.commands.serve import \
        _resolve_rerank_architecture_override

    model = _write_config(tmp_path, ["BertForSequenceClassification"])
    override = _resolve_rerank_architecture_override(model,
                                                      trust_remote_code=False)
    assert override is None


class _FakeTokenizer:
    """Minimal stand-in for a HF tokenizer's token/id interface."""

    def __init__(self, ids, unk_token_id=0):
        self._ids = ids
        self.unk_token_id = unk_token_id

    def convert_tokens_to_ids(self, token):
        return self._ids.get(token, self.unk_token_id)


def test_resolve_rerank_token_ids_returns_yes_no_ids():
    from tensorrt_llm.commands.serve import _resolve_rerank_token_ids

    tokenizer = _FakeTokenizer({"yes": 9693, "no": 2152})
    override = _resolve_rerank_token_ids(tokenizer, model="fake-model")
    assert override == {
        "reranking_token_true_id": 9693,
        "reranking_token_false_id": 2152,
    }


def test_resolve_rerank_token_ids_rejects_unk_mapping():
    from tensorrt_llm.commands.serve import _resolve_rerank_token_ids

    # "no" is missing from the vocab and falls back to unk_token_id.
    tokenizer = _FakeTokenizer({"yes": 9693}, unk_token_id=0)
    with pytest.raises(click.ClickException, match="yes.*no"):
        _resolve_rerank_token_ids(tokenizer, model="fake-model")
