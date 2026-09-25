# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""Cross-attention KV cache sizing for encoder-decoder models.

The cross pool holds the encoder output, so its length must come from the
encoder's declared position limit. ``max_input_len`` is a decoder-side prompt
limit (default 1024) and must never shrink it: Whisper's encoder emits
``max_source_positions`` = 1500 frames for every clip, and a 1024-long cross
pool fails in ``prepare_cross_metadata`` on the first request.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator

pytestmark = pytest.mark.cpu_only

# openai/whisper-large-v3 as HF reports it: d_model 1280, 20 encoder heads,
# 32 decoder layers, 1500 encoder positions, 448 decoder positions.
_WHISPER_LARGE_V3 = dict(
    decoder_layers=32,
    encoder_layers=32,
    encoder_attention_heads=20,
    decoder_attention_heads=20,
    d_model=1280,
    max_source_positions=1500,
    max_target_positions=448,
)


def _creator(config: dict, *, max_seq_len: int = 4096, max_input_len=1024) -> KvCacheCreator:
    """A KvCacheCreator with only the state _get_cross_kv_cache_layout reads."""
    creator = KvCacheCreator.__new__(KvCacheCreator)
    creator._model_engine = SimpleNamespace(
        model=SimpleNamespace(
            model_config=SimpleNamespace(pretrained_config=SimpleNamespace(**config))
        )
    )
    creator._max_seq_len = max_seq_len
    creator._llm_args = SimpleNamespace(max_input_len=max_input_len)
    return creator


def test_whisper_encoder_length_wins_over_max_input_len():
    """max_source_positions=1500 sizes the pool even though max_input_len=1024."""
    creator = _creator(_WHISPER_LARGE_V3, max_seq_len=4096, max_input_len=1024)

    num_layers, num_kv_heads, head_dim, max_seq_len = creator._get_cross_kv_cache_layout()

    assert (num_layers, num_kv_heads, head_dim) == (32, 20, 64)
    assert max_seq_len == 1500


@pytest.mark.parametrize("max_seq_len", [512, 1024, 4096])
@pytest.mark.parametrize("max_input_len", [None, 0, 512, 1024, 2048])
def test_encoder_limit_is_authoritative(max_seq_len, max_input_len):
    """Neither the engine sequence length nor max_input_len can move the encoder limit."""
    creator = _creator(_WHISPER_LARGE_V3, max_seq_len=max_seq_len, max_input_len=max_input_len)

    assert creator._get_cross_kv_cache_layout()[3] == 1500
    assert creator._get_cross_kv_cache_layout(fallback_max_seq_len=256)[3] == 1500


@pytest.mark.parametrize(
    "limit_attr",
    [
        "max_encoder_input_len",
        "encoder_max_input_length",
        "max_encoder_position_embeddings",
        "encoder_max_position_embeddings",
        "max_source_positions",
        "max_position_embeddings",
        "n_positions",
    ],
)
def test_every_encoder_limit_attribute_is_recognized(limit_attr):
    config = dict(num_hidden_layers=6, num_attention_heads=8, hidden_size=512)
    config[limit_attr] = 1500
    creator = _creator(config, max_seq_len=4096, max_input_len=1024)

    assert creator._get_cross_kv_cache_layout() == (6, 8, 64, 1500)


@pytest.mark.parametrize("max_input_len", [512, 1024, 2048])
def test_max_input_len_is_the_fallback_without_an_encoder_limit(max_input_len):
    """No encoder limit in the config: a positive max_input_len sizes the pool,
    ahead of both the engine's max_seq_len and an explicit fallback."""
    config = dict(num_hidden_layers=6, num_attention_heads=8, hidden_size=512)
    creator = _creator(config, max_seq_len=4096, max_input_len=max_input_len)

    assert creator._get_cross_kv_cache_layout()[3] == max_input_len
    assert creator._get_cross_kv_cache_layout(fallback_max_seq_len=2048)[3] == max_input_len
    assert creator._get_cross_kv_cache_layout(fallback_max_seq_len=256)[3] == max_input_len


@pytest.mark.parametrize("max_input_len", [None, 0, -1, "1024"])
def test_engine_max_seq_len_when_neither_limit_is_usable(max_input_len):
    """Without an encoder limit or a positive int max_input_len, the engine's
    max_seq_len (or the explicit fallback) is used."""
    config = dict(num_hidden_layers=6, num_attention_heads=8, hidden_size=512)
    creator = _creator(config, max_seq_len=4096, max_input_len=max_input_len)

    assert creator._get_cross_kv_cache_layout()[3] == 4096
    assert creator._get_cross_kv_cache_layout(fallback_max_seq_len=2048)[3] == 2048
