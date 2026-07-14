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

from pathlib import Path

import pytest

_SOURCE_MODEL = "mistralai/Mistral-Small-4-119B-2603"


def _require_mistral_small_4_snapshot() -> None:
    snapshot_root = Path(".tmp/hf_home/hub")
    if not any(
        snapshot_root.glob(
            "models--mistralai--Mistral-Small-4-119B-2603/snapshots/*/tokenizer.json"
        )
    ):
        pytest.skip("Mistral Small 4 tokenizer snapshot is not available locally")


def test_wrapper_tokenizer_loads():
    _require_mistral_small_4_snapshot()

    from tensorrt_llm._torch.auto_deploy.models.custom.modeling_mistral3 import (
        ADMistralSmall4Tokenizer,
    )

    tokenizer = ADMistralSmall4Tokenizer.from_pretrained(_SOURCE_MODEL)

    encoded = tokenizer("Hello")
    assert encoded["input_ids"]
    assert tokenizer.pad_token_id is not None
    assert tokenizer.chat_template is not None


def test_wrapper_processor_loads():
    _require_mistral_small_4_snapshot()

    from tensorrt_llm._torch.auto_deploy.models.custom.modeling_mistral3 import (
        ADMistralSmall4Processor,
    )

    processor = ADMistralSmall4Processor.from_pretrained(_SOURCE_MODEL)

    assert processor.tokenizer.pad_token_id is not None
    assert processor.image_processor is not None
