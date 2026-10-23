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
"""DFlash shared context block table over a KVCacheManagerV2 draft layout.

A separate draft manager whose layers cannot share one block table falls back to
the private context arena, as it did before unified caching. A unified manager
keeps its invariant, so the same layout there is an error.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
from tensorrt_llm._torch.speculative.dflash import DFlashWorker

pytestmark = pytest.mark.cpu_only


def _draft_manager(*, layer_groups, layer_scales):
    mgr = object.__new__(KVCacheManagerV2)
    mgr.kv_cache_pool_mapping = torch.zeros(len(layer_groups), 2, dtype=torch.int32)
    mgr.layer_offsets = {layer_id: layer_id for layer_id in range(len(layer_groups))}
    mgr.layer_to_pool_mapping_dict = dict(enumerate(layer_groups))
    mgr.enable_swa_scratch_reuse = False
    mgr.impl = SimpleNamespace(
        get_page_index_converter=lambda layer_offset, role: SimpleNamespace(
            scale=layer_scales[layer_offset], layer_offset=0
        ),
        get_page_index_scale=lambda layer_offset, role: layer_scales[layer_offset],
    )
    return mgr


def _worker(*, unified):
    worker = object.__new__(DFlashWorker)
    worker._ctx_layer_ids = (0, 1)
    worker._ctx_unified_cache = unified
    worker._ctx_kv_factor = 2
    return worker


def _init_block_tables(worker, mgr):
    pool = [torch.empty(0)]
    return worker._init_ctx_block_tables(mgr, pool, num_slots=2, L=2, nkv=1, hd=64, page_size=32)


@pytest.mark.parametrize(
    "layer_groups,layer_scales",
    [([0, 1], [2, 2]), ([0, 0], [2, 4])],
    ids=["different_lifecycle", "different_page_index_scale"],
)
def test_separate_v2_draft_manager_with_unsupported_layout_uses_private_arena(
    layer_groups, layer_scales
):
    mgr = _draft_manager(layer_groups=layer_groups, layer_scales=layer_scales)

    assert _init_block_tables(_worker(unified=False), mgr) is False


@pytest.mark.parametrize(
    "layer_groups,layer_scales",
    [([0, 1], [2, 2]), ([0, 0], [2, 4])],
    ids=["different_lifecycle", "different_page_index_scale"],
)
def test_unified_manager_with_unsupported_layout_is_an_error(layer_groups, layer_scales):
    mgr = _draft_manager(layer_groups=layer_groups, layer_scales=layer_scales)

    with pytest.raises(NotImplementedError, match="same lifecycle and page-index scale"):
        _init_block_tables(_worker(unified=True), mgr)


def test_drafter_block_inside_the_allocation_is_accepted():
    num_ctx = torch.tensor([57, 10])
    allocated = torch.tensor([57, 56])

    DFlashWorker._check_block_within_allocation(num_ctx, allocated, block_size=7)


def test_drafter_block_outside_the_allocation_is_an_error():
    num_ctx = torch.tensor([58, 10])
    allocated = torch.tensor([57, 56])

    with pytest.raises(RuntimeError, match="block of 7 positions falls outside"):
        DFlashWorker._check_block_within_allocation(num_ctx, allocated, block_size=7)
