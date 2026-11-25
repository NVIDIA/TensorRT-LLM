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
"""KV budget kept back for a drafter that declares ``context_kv_only``."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.pyexecutor._util import KvCacheCreator

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA tensors")


def _float_bytes(*shape: int) -> int:
    return 4 * int(torch.Size(shape).numel())


def _model(context_kv_only):
    target = nn.Module()
    target.embed = nn.Embedding(64, 32)
    target.proj = nn.Linear(32, 32, bias=False)
    draft = nn.Module()
    # Tied to the target's embedding: the target's cost.
    draft.embed = target.embed
    draft.fc = nn.Linear(32, 16, bias=False)
    target.draft_model = draft
    target.cuda()
    # A table derived from the weights, kept as a plain tensor attribute.
    draft.fc.table = torch.zeros(8, 16, device="cuda")
    if context_kv_only is not None:
        draft.context_kv_only = context_kv_only
    return target


def _creator(model, spec_metadata=None):
    creator = object.__new__(KvCacheCreator)
    creator._model_engine = SimpleNamespace(model=model, spec_metadata=spec_metadata)
    return creator


def test_reserve_is_the_unmeasured_share_of_the_drafter_resident_tensors():
    spec_metadata = SimpleNamespace(captures=torch.zeros(4, 16, device="cuda"))
    creator = _creator(_model(context_kv_only=True), spec_metadata)

    resident = _float_bytes(16, 32) + _float_bytes(8, 16) + _float_bytes(4, 16)
    assert creator._get_context_kv_only_drafter_reserve(0.75) == int(0.25 * resident)


@pytest.mark.parametrize("context_kv_only", [None, False])
def test_no_reserve_without_the_declaration(context_kv_only):
    creator = _creator(_model(context_kv_only))

    assert creator._get_context_kv_only_drafter_reserve(0.75) == 0


def test_no_reserve_without_a_drafter():
    creator = _creator(nn.Linear(4, 4).cuda())

    assert creator._get_context_kv_only_drafter_reserve(0.75) == 0
