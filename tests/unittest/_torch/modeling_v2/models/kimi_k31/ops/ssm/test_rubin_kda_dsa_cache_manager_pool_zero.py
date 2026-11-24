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
so the manager zeroes them once at construction when its target binds ``zero_gpu_pools`` (the DEP16
servers and the helix8 generation server). The zero runs in the manager's ``_reserve_guard_page``,
before the engine reserves the guard page, so the guard page keeps its fill.

* CPU: the zero comes before the guard page and leaves its fill; it covers every pool region; the
  growth check compares against what it zeroed.
* GPU: every byte of every GPU pool, and of the MLA, index-key and draft layer views, reads zero after
  construction, also when the memory held 0xFF bytes before; with ``TRTLLM_KV_GUARD_PAGE`` set the
  guard page holds its fill; a decode over the MLA or the draft layer's pool whose last tile reads
  past the request returns exactly what it returns over a clean pool holding the same keys, and NaN
  once the unwritten bytes hold 0xFF.
* ``zero_gpu_pools`` off: construction never runs the zero and ``check_gpu_pool_growth`` is a no-op.
"""

import functools
import math
from types import SimpleNamespace

import pytest
import torch
from _kda_ops_util import bare

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm import (
    rubin_kda_dsa_cache_manager as dsa_cache_manager,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_kda_dsa_cache_manager import (
    RubinKdaDsaHybridCacheManager,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2

# The CPU cases run in l0_cpu and the GPU cases in l0_gr100; a GPU host other than sm_107 skips
# the file.
pytestmark = pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability(0) != (10, 7),
    reason="Kimi K3.1 Rubin: sm_107 or CPU only",
)

KV_LORA, ROPE = 512, 64
LATENT = KV_LORA + ROPE
TILE = 128
MAX_SEQ_LEN = 2048

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA cache pools")


# CPU: the order of the zero and the guard page, and what the zero covers


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
def test_the_zero_comes_before_the_guard_page(monkeypatch, cpu_pools) -> None:
    """The engine reserves the guard page at the end of its construction (``TRTLLM_KV_GUARD_PAGE``
    fills it); a target that zeroes its pools zeroes them first, so the fill stays."""
    events = []

    def reserve_guard_page(self):
        events.append("guard page")
        cpu_pools.pools[1 << 20][96 + 5] = 0x7E  # the guard page's fill

    monkeypatch.setattr(KVCacheManagerV2, "_reserve_guard_page", reserve_guard_page)
    zero = RubinKdaDsaHybridCacheManager._zero_gpu_pool_memory

    def counting_zero(self):
        events.append("zero")
        zero(self)

    monkeypatch.setattr(RubinKdaDsaHybridCacheManager, "_zero_gpu_pool_memory", counting_zero)
    manager = _bare_manager(_zero_gpu_pools=True, impl=cpu_pools.impl)
    manager._reserve_guard_page()
    assert events == ["zero", "guard page"]
    first = cpu_pools.pools[1 << 20]
    assert first[96 + 5] == 0x7E
    assert int(torch.count_nonzero(first)) == 1
    for base in (2 << 20, 3 << 20):
        assert not cpu_pools.pools[base].any()


@pytest.mark.cpu_only
def test_without_zero_gpu_pools_only_the_guard_page_is_reserved(monkeypatch, cpu_pools) -> None:
    events = []
    monkeypatch.setattr(
        KVCacheManagerV2, "_reserve_guard_page", lambda self: events.append("guard page")
    )
    monkeypatch.setattr(
        RubinKdaDsaHybridCacheManager, "_zero_gpu_pool_memory", lambda self: events.append("zero")
    )
    manager = _bare_manager(_zero_gpu_pools=False, impl=cpu_pools.impl)
    manager._reserve_guard_page()
    assert events == ["guard page"]
    assert not hasattr(manager, "_zeroed_gpu_pool_bytes")
    assert manager.check_gpu_pool_growth() is False


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


@pytest.mark.cpu_only
def test_the_growth_check_warns_once_for_unzeroed_memory(monkeypatch, cpu_pools) -> None:
    """Pool rebalancing can map memory after the zero; the executor's check names the pools that
    grew, once."""
    warnings = []
    monkeypatch.setattr(dsa_cache_manager.logger, "warning", warnings.append)
    manager = _bare_manager(_zero_gpu_pools=True, impl=cpu_pools.impl)
    manager._zero_gpu_pool_memory()
    assert manager.check_gpu_pool_growth() is False
    cpu_pools.impl.pool_group_descs[1].num_slots = 3
    assert manager.check_gpu_pool_growth() is True
    assert manager.check_gpu_pool_growth() is True
    assert len(warnings) == 1 and "[(1, 0), (1, 1)]" in warnings[0]


# GPU: a context server's cache with real pools


def _text_config() -> SimpleNamespace:
    """One KDA layer (1-indexed 1) and one DSA MLA layer (2) with a 512 + 64 latent."""
    return SimpleNamespace(
        architectures=["KimiLinearForCausalLM"],
        model_type="kimi_linear",
        num_hidden_layers=2,
        kv_lora_rank=KV_LORA,
        qk_rope_head_dim=ROPE,
        linear_attn_config={
            "head_dim": 8,
            "num_heads": 4,
            "short_conv_kernel_size": 4,
            "kda_layers": [1],
            "full_attn_layers": [2],
        },
        dsa_config={"dsa_layers": [2], "index_topk_freq": 1},
        index_topk_freq=1,
        index_skip_topk_offset=1,
        dtype=torch.bfloat16,
    )


def _manager(
    page: int, zero_gpu_pools: bool = True, kv_dtype: str = "fp8"
) -> RubinKdaDsaHybridCacheManager:
    """A KDA layer, a DSA MLA layer (576-wide FP8 latent, FP8 index keys) and an FP8 unified draft
    layer, as the specialized targets bind them."""
    from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp, DataType
    from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig, KvCacheConfig
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
        dtype={"fp8": DataType.FP8, "bf16": DataType.BF16}[kv_dtype],
        layer_mask=[False, True],
        sparse_attention_config=DeepSeekSparseAttentionConfig(
            index_n_heads=4, index_head_dim=128, index_topk=64, indexer_k_dtype="fp8"
        ),
        pretrained_config=_text_config(),
        draft_num_layers=1,
        draft_num_kv_heads=1,
        draft_head_dim=LATENT,
        draft_dtype=DataType.FP8,
        draft_kv_factor=1,
        draft_total_num_kv_heads=1,
        draft_scratch_tokens=9,
        draft_attention_backend="CUTEDSL",
        zero_gpu_pools=zero_gpu_pools,
        kv_dtype=kv_dtype,
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


def _layer_pool(manager, pool: str, page: int) -> torch.Tensor:
    """The layer's pool as ``[pages, page, 576]`` FP8 (one latent per token)."""
    if pool == "target":
        view = manager.get_primary_pool(manager._attention_local_layers[0])
    else:
        view = manager.get_draft_buffers(0)
    assert view.is_contiguous() and view.dtype == torch.float8_e4m3fn
    return view.view(-1, page, LATENT)


@needs_gpu
@pytest.mark.parametrize("page", [32, 64])
def test_gpu_pools_read_zero_before_the_first_request(page):
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


@needs_gpu
def test_the_guard_page_keeps_its_fill_after_the_zero(monkeypatch):
    """``TRTLLM_KV_GUARD_PAGE=7`` (read when the manager is built) with the pools zeroed: the zero
    runs first, so the guard page's fill is the only thing in the pools. A BF16 latent cache, since
    FP8 pages take no fill."""
    monkeypatch.setenv("TRTLLM_KV_GUARD_PAGE", "7")
    manager = _manager(32, kv_dtype="bf16")
    try:
        assert manager._guard_page_by_layer, "no guard page was reserved"
        regions = manager._gpu_pool_regions()
        assert sum(_nonzero_bytes(_raw(base, size)) for _, base, size in regions) > 0
    finally:
        manager.shutdown()
    monkeypatch.delenv("TRTLLM_KV_GUARD_PAGE")
    manager = _manager(32, kv_dtype="bf16")
    try:
        assert not manager._guard_page_by_layer
        regions = manager._gpu_pool_regions()
        assert sum(_nonzero_bytes(_raw(base, size)) for _, base, size in regions) == 0
    finally:
        manager.shutdown()


def _decode(cache, table, lengths, heads, seq_q, page):
    import cutlass

    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import CuteDSLNVMlaDecodeBlackwellRunner

    batch = lengths.shape[0]
    g = torch.Generator(device="cuda").manual_seed(11)
    q = (torch.randn(batch, seq_q, heads, LATENT, device="cuda", generator=g) * 0.1).to(
        torch.float8_e4m3fn
    )
    out = torch.full((batch, seq_q, heads, KV_LORA), float("nan"), device="cuda").bfloat16()
    workspace = torch.zeros(
        CuteDSLNVMlaDecodeBlackwellRunner.get_max_padded_workspace_size(
            heads, seq_q, KV_LORA, batch, cutlass.Float32
        ),
        dtype=torch.uint8,
        device="cuda",
    )
    torch.ops.trtllm.cute_dsl_mla_decode_fp8_blackwell(
        q[..., :KV_LORA].permute(2, 3, 1, 0),
        q[..., KV_LORA:].permute(2, 3, 1, 0),
        cache[..., :KV_LORA].permute(1, 2, 0),
        cache[..., KV_LORA:].permute(1, 2, 0),
        table.t(),
        lengths,
        out.permute(2, 3, 1, 0),
        workspace,
        heads,
        seq_q,
        page,
        torch.tensor([1.0 / math.sqrt(192)], device="cuda"),
        torch.tensor([1.0], device="cuda"),
        batch,
        None,
        None,
    )
    torch.cuda.synchronize()
    return out


@needs_gpu
@pytest.mark.parametrize("page", [32, 64])
@pytest.mark.parametrize(
    "pool,heads,seq_q", [("target", 128, 1), ("draft", 64, 8)], ids=["target_mla", "draft"]
)
def test_over_reading_decode_on_the_zeroed_pool(page, pool, heads, seq_q):
    manager = _manager(page)
    try:
        cache = _layer_pool(manager, pool, page)
        num_pages = cache.shape[0]
        # Lengths off the 128-key tile: the last tile covers the rest of the last page and the pages of the
        # next table entries, which point at pages no request wrote.
        lengths = [3 * TILE + 37, TILE + page // 2 + 1]
        width = max(-(-n // TILE) * TILE // page for n in lengths) + 1
        g = torch.Generator(device="cuda").manual_seed(page)
        perm = torch.randperm(num_pages, device="cuda", generator=g)
        assert perm.numel() >= len(lengths) * width
        table = perm[: len(lengths) * width].view(len(lengths), width).to(torch.int32).contiguous()
        written = torch.zeros(num_pages, page, dtype=torch.bool, device="cuda")
        for b, n in enumerate(lengths):
            for p in range(-(-n // page)):
                written[table[b, p], : min(page, n - p * page)] = True
        values = (torch.randn(num_pages, page, LATENT, device="cuda", generator=g) * 0.5).to(
            torch.float8_e4m3fn
        )
        # FP8 has no index_put kernel: write the bytes.
        rows = written.nonzero(as_tuple=True)
        cache.view(torch.uint8)[rows] = values.view(torch.uint8)[rows]
        clean = torch.zeros_like(cache)
        clean.view(torch.uint8)[rows] = values.view(torch.uint8)[rows]
        lens = torch.tensor(lengths, dtype=torch.int32, device="cuda")

        got = _decode(cache, table, lens, heads, seq_q, page)
        ref = _decode(clean, table, lens, heads, seq_q, page)
        assert torch.isfinite(got.float()).all()
        assert torch.equal(got, ref)
        # Control: the bytes no request wrote are read; 0xFF there poisons the output.
        cache.view(torch.uint8)[~written] = 0xFF
        poisoned = _decode(cache, table, lens, heads, seq_q, page)
        assert not torch.isfinite(poisoned.float()).all()
    finally:
        manager.shutdown()


@needs_gpu
def test_zero_gpu_pools_off_construction_does_not_zero(monkeypatch):
    """``zero_gpu_pools`` off (the TEP8 servers): construction never runs the zero (no pool byte is
    written and no synchronize is added) and the executor's growth check is a no-op."""
    calls = []
    monkeypatch.setattr(
        RubinKdaDsaHybridCacheManager, "_zero_gpu_pool_memory", lambda self: calls.append(self)
    )
    manager = _manager(32, zero_gpu_pools=False)
    try:
        assert calls == []
        assert not hasattr(manager, "_zeroed_gpu_pool_bytes")
        assert manager.check_gpu_pool_growth() is False
    finally:
        manager.shutdown()


@needs_gpu
def test_zero_gpu_pools_on_construction_zeroes_once(monkeypatch):
    """``zero_gpu_pools`` on: the zero runs exactly once, at construction."""
    calls = []
    original = RubinKdaDsaHybridCacheManager._zero_gpu_pool_memory

    def counting(self):
        calls.append(self)
        original(self)

    monkeypatch.setattr(RubinKdaDsaHybridCacheManager, "_zero_gpu_pool_memory", counting)
    manager = _manager(32)
    try:
        assert len(calls) == 1
        assert manager.check_gpu_pool_growth() is False
    finally:
        manager.shutdown()
