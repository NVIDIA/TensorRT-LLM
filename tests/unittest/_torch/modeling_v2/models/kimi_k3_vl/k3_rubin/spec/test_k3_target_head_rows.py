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
"""The Kimi K3 DSpark worker's target-head rows (``DFlashWorker.target_head_rows``).

With ``TLLM_DSPARK_TAIL_V2=1`` (and ``TLLM_SPEC_TAIL_FUSION=1``), a generation-only batch hands
the target head its final hidden states as they are: its gather ids are 0..num_tokens-1. Any
other batch, or the switch off, gathers ``spec_metadata.gather_ids`` as main's worker does.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.k3_rubin.spec.dflash import (
    DFlashWorker,
)


def _rows(num_contexts, gather_ids, num_tokens=6):
    hidden = torch.arange(num_tokens * 4, dtype=torch.float32).view(num_tokens, 4)
    rows = DFlashWorker.target_head_rows(
        object.__new__(DFlashWorker),
        hidden,
        SimpleNamespace(num_contexts=num_contexts),
        SimpleNamespace(gather_ids=gather_ids),
    )
    return hidden, rows


@pytest.fixture
def tail_v2(k3_features):
    k3_features(spec_tail_fusion=True, dspark_tail_v2=True)


def test_generation_only_batch_reads_the_hidden_states_without_a_gather(tail_v2):
    hidden, rows = _rows(0, torch.arange(6, dtype=torch.int32))
    assert rows is hidden


def test_batch_with_context_requests_gathers(tail_v2):
    gather_ids = torch.tensor([2, 5], dtype=torch.int32)
    hidden, rows = _rows(1, gather_ids)
    assert rows is not hidden
    assert torch.equal(rows, hidden[gather_ids])


def test_switch_off_gathers(k3_features):
    k3_features(spec_tail_fusion=True, dspark_tail_v2=False)
    gather_ids = torch.arange(6, dtype=torch.int32)
    hidden, rows = _rows(0, gather_ids)
    assert rows is not hidden
    assert torch.equal(rows, hidden)
