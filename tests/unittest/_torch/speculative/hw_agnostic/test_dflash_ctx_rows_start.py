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
"""``ctx_rows_start``: the DFlash worker's start of the gen requests' page-table rows, and who receives it
(host-side).

* ``DFlashWorker._ctx_rows_start`` is the number of context requests where the drafter reads the manager's block
  table (keyed by batch position, so the gen requests' rows are one run), and None for the private arena (keyed by
  slot) or a step without gen requests.
* ``dflash_ctx_rows_kwargs`` hands it only to a drafter whose ``dflash_forward`` takes it (the Kimi K3 target's
  ``K3DSparkDrafter``), never to the stock drafters, and nothing where there is no start.
"""

import pytest
import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kimi_k3_mxfp4__sm_100__tp16_moetp4ep4.modeling import (  # noqa: E501
    K3DSparkDrafter,
)
from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM
from tensorrt_llm._torch.models.modeling_dspark import GQADSparkForCausalLM, MLADSparkForCausalLM
from tensorrt_llm._torch.speculative.dflash import DFlashWorker, dflash_ctx_rows_kwargs
from tensorrt_llm._torch.speculative.dspark import DSparkWorker

pytestmark = pytest.mark.cpu_only


def _bare(cls):
    """An instance without ``__init__`` (the workers need flashinfer, the drafters a checkpoint)."""
    obj = cls.__new__(cls)
    nn.Module.__init__(obj)
    return obj


@pytest.mark.parametrize("cls", [DFlashWorker, DSparkWorker])
@pytest.mark.parametrize("num_contexts,num_gens", [(0, 1), (0, 8), (3, 2), (5, 1)])
def test_rows_start_where_the_manager_table_is_read(cls, num_contexts, num_gens):
    worker = _bare(cls)
    worker._ctx_block_tables = torch.zeros(num_contexts + num_gens + 1, 4, dtype=torch.int32)
    assert worker._ctx_rows_start(num_contexts, num_gens) == num_contexts


@pytest.mark.parametrize("cls", [DFlashWorker, DSparkWorker])
def test_no_rows_start_for_the_private_arena_or_without_gen_requests(cls):
    worker = _bare(cls)
    worker._ctx_block_tables = None
    assert worker._ctx_rows_start(0, 4) is None
    worker._ctx_block_tables = torch.zeros(4, 4, dtype=torch.int32)
    assert worker._ctx_rows_start(3, 0) is None


def test_the_k3_drafter_takes_the_rows_start():
    drafter = _bare(K3DSparkDrafter)
    assert dflash_ctx_rows_kwargs(drafter, 3) == {"ctx_rows_start": 3}
    assert dflash_ctx_rows_kwargs(drafter, 0) == {"ctx_rows_start": 0}
    assert dflash_ctx_rows_kwargs(drafter, None) == {}


@pytest.mark.parametrize("cls", [DFlashForCausalLM, GQADSparkForCausalLM, MLADSparkForCausalLM])
def test_stock_drafters_do_not_get_the_rows_start(cls):
    assert dflash_ctx_rows_kwargs(_bare(cls), 3) == {}
