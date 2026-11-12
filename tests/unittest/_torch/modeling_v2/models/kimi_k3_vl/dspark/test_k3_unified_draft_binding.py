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
"""The Kimi K3 DSpark worker binds the drafter layers KV cache manager V2 appends to the target's.

With ``TRTLLM_DISAGG_SHARED_DRAFT_KV`` the engine appends the drafter's context KV layers to
``KimiK3HybridCacheManager`` after the target's (``_appended_draft_layer_specs``) and hands the
worker the target manager itself as the draft cache (``get_draft_kv_cache_view``). The worker
then has to find those layers by their global ids (``draft_layer_ids``), take the per-layer
pool views, and decode the target metadata's ``kv_cache_block_offsets`` with the draft layers'
own pool index and K-buffer offset.

The manager below keeps its real layer bookkeeping, so ``draft_layer_ids``,
``get_layer_kv_factor`` and ``get_draft_kv_cache_view`` are the manager's own; only the native
pool (``get_buffers`` and the page-index converter) is replaced by CPU tensors laid out like V2's
per-layer views (``stride(0)`` is one layer slot).
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.dspark.drafter import (  # noqa: E501
    K3DSparkDrafter,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.dspark.worker import (
    K3DFlashWorker,
)
from tensorrt_llm._torch.modules.kimi_kda.cache_manager import KimiK3HybridCacheManager
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import (
    DataType,
    KVCacheLayerOwner,
    KVCacheLayerSpec,
    Role,
)
from tensorrt_llm._torch.pyexecutor.resource_manager import ResourceManagerType
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

# The engine appends the drafter's layers to the target manager only on a main with the unified
# draft KV; without it the drafter keeps its own manager and these bindings never happen.
requires_appended_draft_layers = pytest.mark.skipif(
    not getattr(KimiK3HybridCacheManager, "accepts_appended_draft_layers", lambda: False)(),
    reason="main does not append the drafter's layers to KimiK3HybridCacheManager",
)

TARGET_LAYER_IDS = (3, 7, 11, 15)  # the target's MLA layers; its KDA layers hold no KV pages
DRAFT_LAYER_IDS = tuple(range(61, 66))  # appended after the target's num_hidden_layers
HEAD_DIM = 576  # one MLA latent per token: kv_lora_rank + qk_rope_head_dim
PAGE = 32
PAGES = 6
LAYER_SLOT = 100  # encoded-offset distance between the K buffers of consecutive local layers


class _UnifiedK3Manager(KimiK3HybridCacheManager):
    """``KimiK3HybridCacheManager`` with the drafter's layers appended, minus its native pools."""

    def __init__(self, *, per_layer_page_tables=True, draft_dtype=torch.float8_e4m3fn, cp=None):
        # No super().__init__(): it builds the native pools. The layer bookkeeping below is what
        # MambaHybridCacheManagerV2.__init__ derives from draft_layer_specs.
        self.layer_specs = None
        self.pp_layers = list(TARGET_LAYER_IDS + DRAFT_LAYER_IDS)
        self.layer_offsets = {layer_id: local for local, layer_id in enumerate(self.pp_layers)}
        self._legacy_draft_layer_ids = ()
        self._appended_draft_layer_specs = {
            self.layer_offsets[layer_id]: KVCacheLayerSpec(
                global_layer_id=layer_id,
                owner=KVCacheLayerOwner.DRAFT,
                dtype=DataType.FP8,
                num_kv_heads=1,
                head_dim=HEAD_DIM,
                buffer_roles=(Role.KEY,),
            )
            for layer_id in DRAFT_LAYER_IDS
        }
        self.kv_factor = 1
        num_target = len(TARGET_LAYER_IDS)
        # Target layers in layer group 0 / pool 0, the drafter's in group 1 / pool 1.
        groups = [0] * num_target + [1] * len(DRAFT_LAYER_IDS)
        self.layer_to_pool_mapping_dict = dict(enumerate(groups))
        self.kv_cache_pool_mapping = torch.tensor([[group, 0] for group in groups])
        self.enable_swa_scratch_reuse = False
        self._use_per_layer_page_tables = per_layer_page_tables
        self.max_blocks_per_seq = 4
        self.kv_cache_map = {}
        self.mapping = cp or Mapping()
        self._pools = {
            layer_id: torch.zeros(PAGES, 1, 1, PAGE, HEAD_DIM, dtype=draft_dtype)
            for layer_id in DRAFT_LAYER_IDS
        }
        self.impl = SimpleNamespace(
            get_page_index_converter=lambda local, role: SimpleNamespace(
                scale=1, layer_offset=LAYER_SLOT * local
            ),
            get_page_index_scale=lambda local, role: 1,
        )

    def get_buffers(self, layer_idx, kv_layout="NHD"):
        assert kv_layout == "HND"
        return self._pools[layer_idx]


def _worker(manager=None):
    worker = K3DFlashWorker(
        SimpleNamespace(attention_backend="CUTEDSL", max_draft_len=7), Mapping()
    )
    # What _lazy_init_ctx_buffers sets before binding: a manager that is the target metadata's
    # own is the unified cache, and the MLA drafter keeps one latent per token.
    worker._ctx_unified_cache = manager is not None
    worker._ctx_kv_factor = 1
    return worker


def _resource_manager(manager):
    managers = {ResourceManagerType.KV_CACHE_MANAGER: manager}
    return SimpleNamespace(get_resource_manager=managers.get)


def _bind(worker, manager, dtype=torch.float8_e4m3fn):
    pool = worker._managed_ctx_pool(manager, len(DRAFT_LAYER_IDS), 1, HEAD_DIM, dtype, 1)
    return pool, pool is not None and worker._init_ctx_block_offsets(
        manager, pool, len(DRAFT_LAYER_IDS), 1, HEAD_DIM, PAGE
    )


@requires_appended_draft_layers
def test_the_target_manager_is_the_draft_cache():
    manager = _UnifiedK3Manager()
    assert manager.draft_layer_ids == DRAFT_LAYER_IDS
    assert manager.get_layer_kv_factor(DRAFT_LAYER_IDS[0]) == 1
    assert _worker().get_draft_kv_cache_manager(_resource_manager(manager)) is manager


@requires_appended_draft_layers
@pytest.mark.parametrize("per_layer_page_tables", [True, False])
def test_binds_the_appended_draft_layers(per_layer_page_tables):
    manager = _UnifiedK3Manager(per_layer_page_tables=per_layer_page_tables)
    worker = _worker(manager)
    pool, bound = _bind(worker, manager)

    assert bound
    assert worker._ctx_layer_ids == DRAFT_LAYER_IDS
    assert [layer.data_ptr() for layer in pool] == [
        manager._pools[layer_id].data_ptr() for layer_id in DRAFT_LAYER_IDS
    ]
    assert worker._ctx_pool_idx == 1
    # A V2 view's stride(0) is one layer slot, so an entry decodes as offset // kv_factor.
    assert worker._ctx_block_divisor == 1
    # Per-layer page tables address the slot origin; the view starts at the first draft layer's
    # K buffer (local layer 4).
    expected_offset = LAYER_SLOT * len(TARGET_LAYER_IDS) if per_layer_page_tables else 0
    assert worker._ctx_block_offset == expected_offset


@requires_appended_draft_layers
def test_a_unified_pool_of_the_wrong_dtype_raises_instead_of_falling_back():
    manager = _UnifiedK3Manager(draft_dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="unified draft KV pool does not match"):
        _bind(_worker(manager), manager)


@requires_appended_draft_layers
def test_a_separate_pool_of_the_wrong_dtype_falls_back_to_the_arena():
    manager = _UnifiedK3Manager(draft_dtype=torch.bfloat16)
    pool, bound = _bind(_worker(), manager)
    assert pool is None and not bound


@requires_appended_draft_layers
def test_draft_layers_of_mixed_lifecycles_cannot_share_one_table():
    manager = _UnifiedK3Manager()
    manager.layer_to_pool_mapping_dict[len(TARGET_LAYER_IDS) + 1] = 2
    with pytest.raises(NotImplementedError, match="same lifecycle"):
        _bind(_worker(manager), manager)


def _helix(cp_size, cp_rank):
    return SimpleNamespace(cp_size=cp_size, cp_rank=cp_rank, has_cp_helix=lambda: cp_size > 1)


@pytest.mark.parametrize(
    "target, drafter, ok",
    [
        ((1, 0), None, True),
        ((8, 3), (8, 3), True),
        ((8, 3), None, False),
        ((8, 3), (8, 4), False),
        ((1, 0), (8, 3), False),
    ],
)
def test_the_drafter_shards_its_pages_as_the_target_manager_does(target, drafter, ok):
    manager = _UnifiedK3Manager()
    manager.mapping = _helix(*target)
    worker = _worker(manager)
    worker._helix_mapping = None if drafter is None else _helix(*drafter)
    if ok:
        worker._check_unified_helix(manager)
    else:
        with pytest.raises(ValueError, match="helix-sharded"):
            worker._check_unified_helix(manager)


@pytest.mark.parametrize(
    "tokens_per_block, reads",
    [(32, True), (64, True), (128, True), (48, False), (256, False), (1, False)],
)
def test_the_drafter_declares_which_manager_pages_it_reads(tokens_per_block, reads):
    drafter = SimpleNamespace(dflash_attention_backend="CUTEDSL", _helix_mapping=None)
    assert K3DSparkDrafter.reads_managed_ctx_pages(drafter, tokens_per_block) is reads
    assert K3DSparkDrafter.reads_helix_sharded_ctx_pages.fget(drafter) is False
    drafter._helix_mapping = _helix(8, 0)
    assert K3DSparkDrafter.reads_helix_sharded_ctx_pages.fget(drafter) is True
