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

import pytest

from tensorrt_llm._torch.models.modeling_gemma3vl import Gemma3VLM
from tensorrt_llm._torch.models.modeling_hyperclovax import HCXVisionForCausalLM
from tensorrt_llm._torch.models.modeling_llava_next import LlavaNextModel
from tensorrt_llm._torch.models.modeling_mistral import Mistral3VLM
from tensorrt_llm._torch.models.modeling_nemotron_nano import NemotronH_Nano_VL_V2
from tensorrt_llm._torch.models.modeling_phi4mm import Phi4MMForCausalLM
from tensorrt_llm._torch.models.modeling_qwen2vl import Qwen2VLModelBase
from tensorrt_llm._torch.models.modeling_qwen3vl import Qwen3VLModelBase
from tensorrt_llm._torch.models.modeling_vila import VilaModel

_VLM_CLASSES = [
    Qwen3VLModelBase,
    Qwen2VLModelBase,
    LlavaNextModel,
    Gemma3VLM,
    Phi4MMForCausalLM,
    Mistral3VLM,
    VilaModel,
    HCXVisionForCausalLM,
    NemotronH_Nano_VL_V2,
]


def _make_instance(cls, vocab_size_padded):
    """Create a bare instance bypassing __init__, injecting a mock llm."""
    instance = cls.__new__(cls)
    instance.llm = type("MockLLM", (), {"vocab_size_padded": vocab_size_padded})()
    return instance


@pytest.mark.parametrize("cls", _VLM_CLASSES, ids=lambda c: c.__name__)
def test_vocab_size_padded_is_property(cls):
    assert isinstance(cls.__dict__.get("vocab_size_padded"), property), (
        f"{cls.__name__} must define vocab_size_padded as a @property in its own __dict__"
    )


@pytest.mark.parametrize("cls", _VLM_CLASSES, ids=lambda c: c.__name__)
def test_vocab_size_padded_delegates_to_llm(cls):
    instance = _make_instance(cls, vocab_size_padded=152064)

    assert instance.vocab_size_padded == 152064


@pytest.mark.parametrize("cls", _VLM_CLASSES, ids=lambda c: c.__name__)
def test_vocab_size_padded_not_cached(cls):
    instance = _make_instance(cls, vocab_size_padded=152064)

    instance.llm.vocab_size_padded = 32000

    assert instance.vocab_size_padded == 32000
