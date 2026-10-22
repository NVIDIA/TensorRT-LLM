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
"""Under the Qwen3.5 fast paths the NCCL_SYMMETRIC all-reduce strategy is never tuned or run (plain NCCL
instead); without them the strategy selection is unchanged."""

import pytest
import torch

from tensorrt_llm._torch import q35_env
from tensorrt_llm._torch.custom_ops import torch_custom_ops as tco
from tensorrt_llm.functional import AllReduceFusionOp, AllReduceStrategy

SYMM = AllReduceStrategy.NCCL_SYMMETRIC.value
NCCL = AllReduceStrategy.NCCL.value


def _tactics(rows, hidden=4096):
    runner = tco.AllReduceRunner(
        4, [0, 1, 2, 3], torch.bfloat16, int(AllReduceFusionOp.NONE), 1e-6, False
    )
    return set(runner.get_valid_tactics([torch.empty((rows, hidden), dtype=torch.bfloat16)], None))


def test_strategy_mapping():
    assert tco.q35_ar_strategy_without_nccl_symm(SYMM) == SYMM
    q35_env.set_fast_paths(True)
    try:
        assert tco.q35_ar_strategy_without_nccl_symm(SYMM) == NCCL
        for other in (AllReduceStrategy.NCCL, AllReduceStrategy.ONESHOT, AllReduceStrategy.AUTO):
            assert tco.q35_ar_strategy_without_nccl_symm(other.value) == other.value
    finally:
        q35_env.set_fast_paths(False)


@pytest.mark.parametrize("rows,hidden", [(7, 4096), (65536, 1024)])
def test_nccl_symmetric_is_never_tuned(rows, hidden):
    stock = _tactics(rows, hidden)
    assert SYMM in stock
    q35_env.set_fast_paths(True)
    try:
        tactics = _tactics(rows, hidden)
    finally:
        q35_env.set_fast_paths(False)
    assert tactics == stock - {SYMM}
    assert NCCL in tactics
