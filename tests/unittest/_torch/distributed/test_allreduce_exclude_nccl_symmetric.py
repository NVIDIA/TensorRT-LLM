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
"""AllReduceRunner(exclude_nccl_symmetric=True) never tunes or runs NCCL_SYMMETRIC (plain NCCL instead);
without it the strategy selection is unchanged."""

import pytest
import torch

from tensorrt_llm._torch.custom_ops import torch_custom_ops as tco
from tensorrt_llm.functional import AllReduceFusionOp, AllReduceStrategy

SYMM = AllReduceStrategy.NCCL_SYMMETRIC.value
NCCL = AllReduceStrategy.NCCL.value


def _runner(exclude):
    return tco.AllReduceRunner(
        4,
        [0, 1, 2, 3],
        torch.bfloat16,
        int(AllReduceFusionOp.NONE),
        1e-6,
        False,
        exclude_nccl_symmetric=exclude,
    )


def _tactics(rows, hidden, exclude):
    inputs = [torch.empty((rows, hidden), dtype=torch.bfloat16)]
    return set(_runner(exclude).get_valid_tactics(inputs, None))


@pytest.mark.parametrize("rows,hidden", [(7, 4096), (65536, 1024)])
def test_nccl_symmetric_is_never_tuned(rows, hidden):
    stock = _tactics(rows, hidden, exclude=False)
    assert SYMM in stock
    tactics = _tactics(rows, hidden, exclude=True)
    assert tactics == stock - {SYMM}
    assert NCCL in tactics


def test_separate_tuning_cache_entries():
    assert _runner(True).unique_id() != _runner(False).unique_id()
