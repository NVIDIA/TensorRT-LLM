# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import io
from unittest import mock

import pytest

from tensorrt_llm.bench.utils.data import (
    DatasetFormatError,
    create_dataset_from_stream,
    initialize_tokenizer,
)


class _FakeTokenizer:
    """Minimal tokenizer stub for testing create_dataset_from_stream."""

    def __call__(self, text, **kwargs):
        ids = list(range(len(text.split())))
        return {"input_ids": ids}

    def encode(self, text, **kwargs):
        return list(range(len(text.split())))


def test_empty_stream_raises_dataset_format_error():
    tokenizer = _FakeTokenizer()
    empty_stream = io.StringIO("")

    with pytest.raises(DatasetFormatError, match="No data was read from the dataset stream"):
        create_dataset_from_stream(tokenizer, empty_stream)


def test_initialize_tokenizer_routes_through_transformers_tokenizer():
    """``initialize_tokenizer`` must call ``TransformersTokenizer.from_pretrained``.

    Routing through ``TransformersTokenizer`` is what lets ``trtllm-bench``
    inherit the post-load fixes (e.g. ``maybe_fix_byte_level_tokenizer``,
    which prevents DeepSeek-V3 from loading with a Metaspace pre-tokenizer
    that silently strips spaces -- "hello world" -> "helloworld" -- on
    transformers >= 5.x). Without this routing the bench would see a
    different tokenizer than the rest of TRT-LLM uses.
    """
    inner = mock.MagicMock()
    inner.pad_token_id = 0  # already set => add_special_tokens not invoked
    wrapper = mock.MagicMock(tokenizer=inner)

    with mock.patch(
        "tensorrt_llm.bench.utils.data.TransformersTokenizer.from_pretrained", return_value=wrapper
    ) as routed:
        out = initialize_tokenizer("dummy/model")

    routed.assert_called_once_with("dummy/model", padding_side="left", trust_remote_code=True)
    # Bench code uses the raw HF tokenizer (calls __call__, encode,
    # add_special_tokens on it), so the wrapper must be peeled off.
    assert out is inner
