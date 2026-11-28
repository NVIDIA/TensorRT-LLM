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
"""The family KDA + DSA cache manager's GPU pools start zeroed where the target asks for it.

The CuTe DSL MLA decode loads whole 128-key tiles: for a request whose length is not a multiple of 128
it reads the rest of its last page and the pages the following page-table entries name, masks those
keys' scores, and multiplies their value rows by a zero probability. A NaN bit pattern in such a row
(FP8 E4M3 0x7F / 0xFF) makes the output NaN. The pools are mapped from memory that is never cleared,
so the manager zeroes them once at construction when its target binds ``zero_gpu_pools`` (every
Rubin target's servers do). The zero runs in the manager's ``_reserve_guard_page``,
before the engine reserves the guard page, so the guard page keeps its fill.

* CPU: the zero covers every pool region.
* GPU: every byte of every GPU pool, and of the MLA, index-key and draft layer views, reads zero after
  construction, also when the memory held 0xFF bytes before.
"""

import functools
from types import SimpleNamespace

import pytest
import torch

__extra_import_path__ = [".", "../.."]
from _k31_utils import cpu_or_sm
from _kda_ops_util import (  # noqa: E402 — needs the path declared above
    bare,
    dsa_sparse_config,
    dsa_text_config,
)

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.ssm import (
    rubin_kda_dsa_cache_manager as dsa_cache_manager,
)
from tensorrt_llm._torch.modeling_v2.models.kimi_k31.ops.ssm.rubin_kda_dsa_cache_manager import (
    RubinKdaDsaHybridCacheManager,
)

# The CPU cases run in l0_cpu and the GPU cases in l0_gr100; a GPU host other than sm_107 skips
# the file.
pytestmark = cpu_or_sm(107)

KV_LORA, ROPE = 512, 64
LATENT = KV_LORA + ROPE
MAX_SEQ_LEN = 2048

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA cache pools")


# CPU: what the zero covers


#: A manager without V2 pools, carrying the given attributes.
_bare_manager = functools.partial(bare, RubinKdaDsaHybridCacheManager)


class _Event:
    """``torch.cuda.Event`` on the CPU: the zero records and times nothing."""

    def __init__(self, enable_timing=False):
        pass

    def record(self, stream=None):
        pass

    def synchronize(self):
        pass

    def elapsed_time(self, other):
        return 0.0


@pytest.fixture
def cpu_pools(monkeypatch):
    """Two pool groups of 0xFF bytes in host memory, reached by base address as the zero reaches
    the device pools; ``impl.pool_group_descs`` describes them as V2 does."""
    pools = {
        1 << 20: torch.full((3 * 96,), 0xFF, dtype=torch.uint8),
        2 << 20: torch.full((2 * 40,), 0xFF, dtype=torch.uint8),
        3 << 20: torch.full((2 * 24,), 0xFF, dtype=torch.uint8),
    }
    descs = [
        SimpleNamespace(
            pool_group_index=0,
            num_slots=3,
            pools=[SimpleNamespace(pool_index=0, base_address=1 << 20, slot_bytes=96)],
        ),
        SimpleNamespace(
            pool_group_index=1,
            num_slots=2,
            pools=[
                SimpleNamespace(pool_index=0, base_address=2 << 20, slot_bytes=40),
                SimpleNamespace(pool_index=1, base_address=3 << 20, slot_bytes=24),
            ],
        ),
    ]
    monkeypatch.setattr(
        dsa_cache_manager, "TensorWrapper", lambda base, dtype, shape: (base, shape)
    )
    monkeypatch.setattr(
        dsa_cache_manager,
        "convert_to_torch_tensor",
        lambda wrapped: pools[wrapped[0]][: wrapped[1][0]],
    )
    monkeypatch.setattr(torch.cuda, "Event", _Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: None)
    return SimpleNamespace(pools=pools, impl=SimpleNamespace(pool_group_descs=descs))


@pytest.mark.cpu_only
def test_the_zero_covers_every_pool_region(cpu_pools) -> None:
    """Every pool of every pool group, ``num_slots * slot_bytes`` bytes from its base."""
    manager = _bare_manager(_zero_gpu_pools=True, impl=cpu_pools.impl)
    assert manager._gpu_pool_regions() == [
        ((0, 0), 1 << 20, 3 * 96),
        ((1, 0), 2 << 20, 2 * 40),
        ((1, 1), 3 << 20, 2 * 24),
    ]
    manager._zero_gpu_pool_memory()
    assert all(not pool.any() for pool in cpu_pools.pools.values())
    assert manager._zeroed_gpu_pool_bytes == {(0, 0): 288, (1, 0): 80, (1, 1): 48}


# GPU: a context server's cache with real pools


def _manager(page: int) -> RubinKdaDsaHybridCacheManager:
    """A KDA layer, a DSA MLA layer (576-wide FP8 latent, FP8 index keys) and an FP8 unified draft
    layer, as the specialized targets bind them."""
    from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, DataType
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig
    from tensorrt_llm.mapping import Mapping

    return RubinKdaDsaHybridCacheManager(
        mamba_d_state=8,
        mamba_d_conv=4,
        mamba_num_heads=4,
        mamba_n_groups=4,
        mamba_head_dim=8,
        mamba_num_layers=1,
        mamba_layer_mask=[True, False],
        mamba_cache_dtype=torch.bfloat16,
        mamba_ssm_cache_dtype=torch.float32,
        kv_cache_config=KvCacheConfig(max_tokens=16 * MAX_SEQ_LEN, enable_block_reuse=False),
        kv_cache_type=CacheTypeCpp.SELFKONLY,
        num_layers=1,
        num_kv_heads=1,
        head_dim=LATENT,
        tokens_per_block=page,
        max_seq_len=MAX_SEQ_LEN,
        max_batch_size=4,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.FP8,
        layer_mask=[False, True],
        sparse_attention_config=dsa_sparse_config(),
        # One KDA layer (1-indexed 1) and one DSA MLA layer (2) with a 512 + 64 latent.
        pretrained_config=dsa_text_config((2,), 2, KV_LORA, ROPE),
        draft_num_layers=1,
        draft_num_kv_heads=1,
        draft_head_dim=LATENT,
        draft_dtype=DataType.FP8,
        draft_kv_factor=1,
        draft_total_num_kv_heads=1,
        draft_scratch_tokens=9,
        draft_window_size=1024,
        zero_gpu_pools=True,
        kv_dtype="fp8",
        indexer_k_dtype="fp8",
        state_dtype=torch.float32,
    )


def _raw(base: int, size: int) -> torch.Tensor:
    return dsa_cache_manager.convert_to_torch_tensor(
        dsa_cache_manager.TensorWrapper(base, torch.uint8, [size])
    )


def _nonzero_bytes(view: torch.Tensor) -> int:
    try:
        raw = view.view(torch.uint8)
    except RuntimeError:
        raw = view.contiguous().view(torch.uint8)
    return int(torch.count_nonzero(raw).item())


def _layer_views(manager) -> list:
    """The MLA layer's K pool and index-key pool, and the draft layer's pool."""
    mla_local = manager._attention_local_layers[0]
    return [
        manager.get_primary_pool(mla_local),
        manager.get_indexer_k_cache_buffers(manager.pp_layers[mla_local]),
        manager.get_draft_buffers(0),
    ]


@needs_gpu
def test_gpu_pools_read_zero_before_the_first_request():
    page = 64
    torch.cuda.init()
    # The memory the pools map may have held anything before; leave 0xFF behind.
    stale = torch.full((1 << 30,), 0xFF, dtype=torch.uint8, device="cuda")
    del stale
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    manager = _manager(page)
    try:
        regions = manager._gpu_pool_regions()
        assert len(regions) >= 3 and all(size > 0 for _, _, size in regions)
        for _, base, size in regions:
            assert _nonzero_bytes(_raw(base, size)) == 0
        for view in _layer_views(manager):
            assert _nonzero_bytes(view) == 0
        # Every byte the layer views expose lies in the zeroed regions.
        for _, base, size in regions:
            _raw(base, size).fill_(0xFF)
        manager._zero_gpu_pool_memory()
        for view in _layer_views(manager):
            assert _nonzero_bytes(view) == 0
        assert not manager.check_gpu_pool_growth()
    finally:
        manager.shutdown()
