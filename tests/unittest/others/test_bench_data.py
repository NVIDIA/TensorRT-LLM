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
import io

import pytest

from tensorrt_llm.bench.utils.data import DatasetFormatError, create_dataset_from_stream


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
