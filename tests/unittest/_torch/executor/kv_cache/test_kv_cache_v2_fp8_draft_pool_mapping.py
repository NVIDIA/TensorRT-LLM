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
"""Attention-op pool mapping of a unified target + draft KV cache, on the host only.

A DFlash drafter's KV layers share the target's sliding-window life cycle. As BF16 they have
another storage layout than the FP8 target and, by size, a physical pool of their own. As FP8
(``TRTLLM_DFLASH_FP8_DRAFT_KV``) their K/V buffers can be exactly the size of the target's;
their coalescing group still keeps them in a pool of their own, with another slot size. The
attention op addresses a page as ``pool pointer + (table entry + mapping offset * kv_factor) *
page stride``, so one pointer and one page-index scale per life cycle cannot address both pools.

``KVCacheManagerV2._prepare_page_table_tensor`` runs here against a fake allocator that lays
out life cycles, physical pools and slots the way the bindings describe them
(``getMemPoolBaseAddress``, ``getPageIndexConverter``): a pool per (coalescing group, buffer
size) of a life cycle, its buffers K, V, K, V, ... in a slot, one slot count per life cycle.
For the layouts of a TP4 rank and of the last stage of a PP4 context server it checks:

* with FP8 draft layers, every layer's K and V address from the attention-op tables equals the
  allocator's for slots 0-3, whatever the pool order and however far apart the pools lie;
* the FP8 mapping, pointers and tables are the BF16 ones but for the draft page stride;
* with BF16 draft layers or no draft layers, the result is the one without the FP8 rule;
* without the rule, one pointer per life cycle overflows int32 or misaddresses draft pages;
* with the knob unset, FP8 draft buffers take the coalescing group they had without the rule
  (0) and share the target's pool by size, with one table per life cycle.

``kv_cache_manager_v2.py`` and ``dflash_kv_storage.py`` (the knob) are executed from their files
with every other ``tensorrt_llm`` module stubbed, so the tests run without the compiled bindings.
The stubs are in ``sys.modules`` only while this module's tests run.
"""

from __future__ import annotations

import enum
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[5]
_KVM_FILE = "tensorrt_llm/_torch/pyexecutor/kv_cache/kv_cache_manager_v2.py"
_KVM = "tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2"
_STORAGE_FILE = "tensorrt_llm/_torch/speculative/dflash_kv_storage.py"
_STORAGE = "tensorrt_llm._torch.speculative.dflash_kv_storage"
_FP8_KNOB = "TRTLLM_DFLASH_FP8_DRAFT_KV"
_STUBBED_ROOTS = ("tensorrt_llm", "triton", "nvtx")

TOKENS_PER_BLOCK = 32
WINDOW = 2048
TARGET_LAYERS = 62
DRAFT_IDS = list(range(TARGET_LAYERS, TARGET_LAYERS + 5))
PP_PARTITION = [15, 16, 16, 15]
SLOTS = 4
# Pool bases far enough apart that a cross-pool offset in units of a page pair overflows int32.
FAR = 1 << 48
NEAR = 1 << 24


# --------------------------------------------------------------------------
# Stubs for the modules kv_cache_manager_v2.py imports
# --------------------------------------------------------------------------
class _StubMeta(type):
    """Metaclass of a stub: any attribute is another stub; a bare decorator passes through."""

    _cache: dict = {}

    def __getattr__(cls, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _stub(f"{cls.__name__}.{name}")

    def __call__(cls, *args, **kwargs):
        if len(args) == 1 and not kwargs and callable(args[0]):
            return args[0]
        return super().__call__(*args, **kwargs)

    def __or__(cls, other):
        return cls

    __ror__ = __or__

    def __getitem__(cls, item):
        return cls

    def __iter__(cls):
        return iter(())


class _StubBase(metaclass=_StubMeta):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and not kwargs and callable(args[0]):
            return args[0]
        return self

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _stub(f"{type(self).__name__}().{name}")


def _stub(qualname: str) -> type:
    cache = _StubMeta._cache
    if qualname not in cache:
        cache[qualname] = _StubMeta(qualname, (_StubBase,), {})
    return cache[qualname]


class _StubModule(types.ModuleType):
    def __init__(self, name: str):
        super().__init__(name)
        self.__path__ = []

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        submodule = sys.modules.get(f"{self.__name__}.{name}")
        if submodule is not None:
            return submodule
        return _stub(f"{self.__name__}.{name}")


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Serves a ``_StubModule`` for every stubbed name not already in ``sys.modules``."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in _STUBBED_ROOTS:
            return importlib.machinery.ModuleSpec(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        return _StubModule(spec.name)

    def exec_module(self, module):
        pass


class DataType(enum.Enum):
    """The binding's KV data types that the layouts here use."""

    FP8 = "fp8"
    BF16 = "bf16"
    HALF = "half"
    FLOAT = "float"
    NVFP4 = "nvfp4"


class PageIndexMode(enum.IntEnum):
    SHARED = 0
    PER_LAYER = 1


_DTYPE_BYTES = {DataType.FP8: 1, DataType.BF16: 2, DataType.HALF: 2, DataType.FLOAT: 4}


def _exact_div(x: int, y: int) -> int:
    assert x % y == 0, (x, y)
    return x // y


def _module(name: str, **attributes) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = _StubModule(name)
        sys.modules[name] = module
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _install_stubs() -> None:
    _module("tensorrt_llm")
    _module(
        "tensorrt_llm._utils",
        get_size_in_bytes=lambda elements, dtype: int(elements * _DTYPE_BYTES[dtype]),
        prefer_pinned=lambda: False,
    )
    _module("tensorrt_llm.bindings", DataType=DataType)
    _module(
        "tensorrt_llm.runtime.kv_cache_manager_v2",
        _KV_CACHE_ITERATION_STATS_DELTA_FIELDS=(),
        BAD_PAGE_INDEX=-1,
        DataRole=str,
        LayerId=int,
        PageIndexMode=PageIndexMode,
        exact_div=_exact_div,
        typed_range=range,
    )
    _module("tensorrt_llm._torch.pyexecutor.resource_manager", DataType=DataType)
    # The real knob reader: the coalescing group of FP8 draft layers depends on it.
    spec = importlib.util.spec_from_file_location(_STORAGE, _ROOT / _STORAGE_FILE)
    storage = importlib.util.module_from_spec(spec)
    sys.modules[_STORAGE] = storage
    spec.loader.exec_module(storage)


@pytest.fixture(scope="module")
def kvm():
    """``kv_cache_manager_v2.py`` executed against the stubs; ``sys.modules`` restored after."""
    saved = {
        name: module for name, module in sys.modules.items() if name.split(".")[0] in _STUBBED_ROOTS
    }
    for name in saved:
        del sys.modules[name]
    finder = _StubFinder()
    sys.meta_path.insert(0, finder)
    try:
        _install_stubs()
        spec = importlib.util.spec_from_file_location(_KVM, _ROOT / _KVM_FILE)
        module = importlib.util.module_from_spec(spec)
        sys.modules[_KVM] = module
        spec.loader.exec_module(module)
        yield module
    finally:
        sys.meta_path.remove(finder)
        for name in [name for name in sys.modules if name.split(".")[0] in _STUBBED_ROOTS]:
            del sys.modules[name]
        sys.modules.update(saved)


@pytest.fixture(autouse=True)
def _fp8_draft_kv_requested(monkeypatch):
    # The FP8 draft layer specs exist only under the knob; the tests below assume it.
    monkeypatch.setenv(_FP8_KNOB, "1")


# --------------------------------------------------------------------------
# A fake allocator with the bindings' address rules
# --------------------------------------------------------------------------
@dataclass
class _Pool:
    life_cycle: int
    buffer_size: int
    base: int = 0
    buffers: list = field(default_factory=list)  # (local layer, role) in slot order

    @property
    def scale(self) -> int:
        """Pages per slot."""
        return len(self.buffers)

    @property
    def slot_size(self) -> int:
        return self.scale * self.buffer_size


class _FakeImpl:
    """Life cycles by window; a pool per (coalescing group, buffer size) of a life cycle."""

    def __init__(self, kvm, specs, *, gap: int, reverse_pools: bool):
        windows = []
        for spec in specs:
            if spec.sliding_window_size not in windows:
                windows.append(spec.sliding_window_size)
        self.layer_grouping = [
            [i for i, spec in enumerate(specs) if spec.sliding_window_size == window]
            for window in windows
        ]
        self._life_cycle = {
            i: lc for lc, layer_ids in enumerate(self.layer_grouping) for i in layer_ids
        }
        self.pools: list[_Pool] = []
        self._where: dict = {}
        for lc, layer_ids in enumerate(self.layer_grouping):
            by_key: dict = {}
            for layer in layer_ids:
                spec = specs[layer]
                for role in spec.buffer_roles:
                    size = spec.bytes_per_token(role) * TOKENS_PER_BLOCK
                    key = (kvm._layer_buffer_coalescing_group(spec, role), size)
                    pool = by_key.setdefault(key, _Pool(lc, size))
                    self._where[(layer, role)] = (pool, len(pool.buffers))
                    pool.buffers.append((layer, role))
            pools = list(by_key.values())
            self.pools += pools[::-1] if reverse_pools else pools
        address = 1 << 40
        for pool in self.pools:
            pool.base = address
            address += gap

    def get_layer_group_id(self, layer):
        return self._life_cycle[int(layer)]

    def get_mem_pool_base_address(self, layer, role, mode):
        pool, index = self._where[(int(layer), role)]
        return pool.base + (index * pool.buffer_size if mode == PageIndexMode.SHARED else 0)

    def get_page_stride(self, layer, role):
        return self._where[(int(layer), role)][0].buffer_size

    def get_page_index_scale(self, layer, role):
        return self._where[(int(layer), role)][0].scale

    def get_page_index_converter(self, layer, role):
        pool, index = self._where[(int(layer), role)]
        return SimpleNamespace(scale=pool.scale, expansion=1, layer_offset=index)

    def pool_of(self, layer, role) -> _Pool:
        return self._where[(int(layer), role)][0]

    def address(self, layer, role, slot: int) -> int:
        """Where the bindings put ``layer``'s ``role`` page of ``slot``."""
        pool, index = self._where[(int(layer), role)]
        return pool.base + slot * pool.slot_size + index * pool.buffer_size


# --------------------------------------------------------------------------
# Layouts
# --------------------------------------------------------------------------
def _is_full(layer: int) -> bool:
    return layer < TARGET_LAYERS and layer % 4 == 1


def _specs(kvm, layout: str, draft_dtype):
    """Local layer specs of a TP4 rank (context or generation) or of a PP4 context stage 3."""
    if layout == "tp4":
        target, draft, tp = list(range(TARGET_LAYERS)), DRAFT_IDS, 4
    elif layout == "pp4_stage3":
        target, draft, tp = list(range(sum(PP_PARTITION[:3]), TARGET_LAYERS)), DRAFT_IDS, 1
    else:
        assert layout == "tp4_no_draft"
        target, draft, tp = list(range(TARGET_LAYERS)), [], 4
    roles = (kvm.Role.KEY, kvm.Role.VALUE)
    specs = [
        kvm.KVCacheLayerSpec(
            global_layer_id=layer,
            owner=kvm.KVCacheLayerOwner.TARGET,
            dtype=DataType.FP8,
            num_kv_heads=16 // tp,
            head_dim=64,
            buffer_roles=roles,
            sliding_window_size=None if _is_full(layer) else WINDOW,
        )
        for layer in target
    ]
    specs += [
        kvm.KVCacheLayerSpec(
            global_layer_id=layer,
            owner=kvm.KVCacheLayerOwner.DRAFT,
            dtype=draft_dtype,
            num_kv_heads=8 // tp,
            head_dim=128,
            buffer_roles=roles,
            sliding_window_size=WINDOW,
        )
        for layer in draft
    ]
    return specs


def _manager(kvm, layout: str, draft_dtype=DataType.FP8, *, gap=FAR, reverse_pools=False):
    """A ``KVCacheManagerV2`` with only the state ``_prepare_page_table_tensor`` reads."""
    specs = _specs(kvm, layout, draft_dtype)
    impl = _FakeImpl(kvm, specs, gap=gap, reverse_pools=reverse_pools)
    manager = object.__new__(kvm.KVCacheManagerV2)
    manager.impl = impl
    manager.layer_specs = specs
    manager.num_local_layers = len(specs)
    manager.num_pools = len(impl.layer_grouping)
    manager.enable_swa_scratch_reuse = False
    manager.tokens_per_block = TOKENS_PER_BLOCK
    manager.max_beam_width = 1
    manager.max_copy_beam_width = 1
    manager.max_blocks_per_seq = 8
    # As __init__ does from the pool descriptors: the first buffer of a role in a life cycle.
    manager._pool_layer_ids_by_role = {}
    for pool in impl.pools:
        for layer, role in pool.buffers:
            manager._pool_layer_ids_by_role.setdefault((pool.life_cycle, role), layer)
    manager.copies = []
    manager._prepare_batch_block_offsets_copy = lambda *args: manager.copies.append(args)
    manager._prepare_swa_scratch_copy_tensors = lambda *args: manager.copies.append(args)
    manager._prepare_page_table_tensor(index_mapper_capacity=2)
    return manager


def _op_address(kvm, manager, layer: int, role, slot: int) -> int:
    """The attention op's address of ``layer``'s ``role`` page for a life-cycle slot."""
    op_pool, offset = manager.kv_cache_pool_mapping[layer].tolist()
    pointer = int(manager.kv_cache_pool_pointers[op_pool][0])
    stride = manager.impl.get_page_stride(layer, kvm.Role.KEY)
    kv_factor = manager.layer_specs[layer].kv_factor
    if manager._attention_op_pool_groups is not None:
        group = manager._attention_op_pool_groups[op_pool]
        assert group.pool_id == manager.impl.get_layer_group_id(layer)
        scale, kv_offset = group.scale, group.kv_offset
    else:
        assert not manager._use_per_layer_page_tables
        assert op_pool == manager.impl.get_layer_group_id(layer)
        scale, kv_offset = int(manager.index_scales[op_pool]), int(manager.kv_offset[op_pool])
    page = slot * scale + (kv_offset if role == kvm.Role.VALUE else 0) + offset * kv_factor
    return pointer + page * stride


def _check_addresses(kvm, manager) -> None:
    for layer in range(manager.num_local_layers):
        for role in (kvm.Role.KEY, kvm.Role.VALUE):
            for slot in range(SLOTS):
                assert _op_address(kvm, manager, layer, role, slot) == manager.impl.address(
                    layer, role, slot
                ), (manager.layer_specs[layer].global_layer_id, role, slot)


def _no_fp8_rule(monkeypatch, kvm) -> None:
    # raising=False: the same tests then also run against a tree without the rule.
    monkeypatch.setattr(
        kvm.KVCacheManagerV2, "_has_fp8_draft_pool", lambda self: False, raising=False
    )


def _tables(manager):
    groups = manager._attention_op_pool_groups
    return (
        manager._use_per_layer_page_tables,
        manager.num_attention_op_pools,
        None if groups is None else [tuple(g) for g in groups],
        manager.kv_cache_pool_pointers.tolist(),
        manager.kv_cache_pool_mapping.tolist(),
        manager.index_scales.tolist(),
        manager.kv_offset.tolist(),
    )


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
_DRAFT_LAYOUTS = ["tp4", "pp4_stage3"]


@pytest.mark.parametrize("reverse_pools", [False, True], ids=["target-pool-first", "draft-first"])
@pytest.mark.parametrize("gap", [FAR, NEAR], ids=["far-pools", "near-pools"])
@pytest.mark.parametrize("layout", _DRAFT_LAYOUTS)
def test_fp8_draft_pages_are_addressed_in_their_own_pool(kvm, layout, gap, reverse_pools):
    manager = _manager(kvm, layout, DataType.FP8, gap=gap, reverse_pools=reverse_pools)
    draft = [i for i, s in enumerate(manager.layer_specs) if s.global_layer_id in DRAFT_IDS]
    target_sliding = next(
        i for i, s in enumerate(manager.layer_specs) if not _is_full(s.global_layer_id)
    )
    # The case: the draft buffers have the target's page stride and storage layout, and a
    # pool of their own in the target's sliding life cycle, with another slot size.
    key = kvm.Role.KEY
    assert manager.impl.get_page_stride(draft[0], key) == manager.impl.get_page_stride(
        target_sliding, key
    )
    assert {(s.dtype, s.kv_factor) for s in manager.layer_specs} == {(DataType.FP8, 2)}
    draft_pool = manager.impl.pool_of(draft[0], key)
    target_pool = manager.impl.pool_of(target_sliding, key)
    assert draft_pool is not target_pool
    assert draft_pool.life_cycle == target_pool.life_cycle
    assert draft_pool.scale == 2 * len(draft) != target_pool.scale

    assert manager._has_fp8_draft_pool()
    assert not manager._use_per_layer_page_tables
    # (sliding target pool), (full pool), (sliding draft pool).
    assert manager.num_attention_op_pools == 3
    assert manager.kv_cache_pool_mapping.dtype == torch.int32
    _check_addresses(kvm, manager)
    [(_, pool_ids, scales, kv_offsets)] = manager.copies
    assert pool_ids.tolist() == [[g.pool_id] * 2 for g in manager._attention_op_pool_groups]
    assert scales.tolist() == [[g.scale] * 2 for g in manager._attention_op_pool_groups]
    assert kv_offsets.tolist() == [[0, 1]] * 3


@pytest.mark.parametrize("layout", _DRAFT_LAYOUTS)
def test_fp8_draft_tables_are_the_bf16_tables(kvm, layout):
    """FP8 draft layers change the draft pool's page stride and nothing else in the mapping."""
    fp8 = _manager(kvm, layout, DataType.FP8)
    bf16 = _manager(kvm, layout, DataType.BF16)
    num_draft = len(DRAFT_IDS)
    num_target = fp8.num_local_layers - num_draft
    sliding = [i for i in range(num_target) if not _is_full(fp8.layer_specs[i].global_layer_id)]
    full = [i for i in range(num_target) if _is_full(fp8.layer_specs[i].global_layer_id)]
    expected_mapping = [None] * fp8.num_local_layers
    for op_pool, layers in enumerate((sliding, full, range(num_target, fp8.num_local_layers))):
        for offset, layer in enumerate(layers):
            expected_mapping[layer] = [op_pool, offset]
    for manager in (fp8, bf16):
        assert not manager._use_per_layer_page_tables
        assert manager.num_attention_op_pools == 3
        assert manager.kv_cache_pool_mapping.tolist() == expected_mapping
        groups = manager._attention_op_pool_groups
        assert [(g.pool_id, g.scale, g.kv_offset) for g in groups] == [
            (0, 2 * len(sliding), 1),
            (1, 2 * len(full), 1),
            (0, 2 * num_draft, 1),
        ]
        _check_addresses(kvm, manager)
    stride = {m: m.impl.get_page_stride(num_target, kvm.Role.KEY) for m in (fp8, bf16)}
    assert stride[bf16] == 2 * stride[fp8]
    assert fp8.impl.get_page_stride(sliding[0], kvm.Role.KEY) == stride[fp8]


@pytest.mark.parametrize("gap", [FAR, NEAR], ids=["far-pools", "near-pools"])
@pytest.mark.parametrize(
    "layout, draft_dtype",
    [("tp4", DataType.BF16), ("pp4_stage3", DataType.BF16), ("tp4_no_draft", None)],
    ids=["tp4-bf16", "pp4_stage3-bf16", "tp4-no-draft"],
)
def test_without_fp8_draft_layers_the_tables_are_unchanged(
    kvm, monkeypatch, layout, draft_dtype, gap
):
    manager = _manager(kvm, layout, draft_dtype, gap=gap)
    assert not manager._has_fp8_draft_pool()
    with monkeypatch.context() as patch:
        _no_fp8_rule(patch, kvm)
        reference = _manager(kvm, layout, draft_dtype, gap=gap)
    assert _tables(manager) == _tables(reference)
    _check_addresses(kvm, manager)
    if layout == "tp4_no_draft":
        # One table per life cycle, as before.
        assert manager._attention_op_pool_groups is None
        assert not manager._use_per_layer_page_tables
        assert manager.num_attention_op_pools == manager.num_pools == 2
        sliding = [i for i in range(TARGET_LAYERS) if not _is_full(i)]
        full = [i for i in range(TARGET_LAYERS) if _is_full(i)]
        expected = [None] * TARGET_LAYERS
        for life_cycle, layers in enumerate((sliding, full)):
            for offset, layer in enumerate(layers):
                expected[layer] = [life_cycle, offset]
        assert manager.kv_cache_pool_mapping.tolist() == expected
        assert manager.index_scales.tolist() == [2 * len(sliding), 2 * len(full)]
    else:
        # A BF16 draft differs in storage layout and so already had attention-op tables.
        assert manager.num_attention_op_pools == 3


@pytest.mark.parametrize("layout", _DRAFT_LAYOUTS)
def test_one_pointer_per_life_cycle_cannot_address_the_fp8_draft_pool(kvm, monkeypatch, layout):
    """Without the FP8 draft rule the manager takes one pointer per life cycle and fails."""
    _no_fp8_rule(monkeypatch, kvm)
    with pytest.raises(RuntimeError, match="int32"):
        _manager(kvm, layout, DataType.FP8, gap=FAR)
    manager = _manager(kvm, layout, DataType.FP8, gap=NEAR)
    assert manager._attention_op_pool_groups is None
    assert manager.num_attention_op_pools == manager.num_pools == 2
    draft = manager.num_local_layers - 1
    key = kvm.Role.KEY
    # Slot 0 lands on the draft page; every later slot is scaled by the target pool's slot.
    assert _op_address(kvm, manager, draft, key, 0) == manager.impl.address(draft, key, 0)
    for slot in range(1, SLOTS):
        assert _op_address(kvm, manager, draft, key, slot) != manager.impl.address(draft, key, slot)


def _group_without_fp8_rule(kvm, spec, role) -> int:
    """The coalescing group of explicit layer specs before the FP8 draft rule: NVFP4 only."""
    if spec.dtype != DataType.NVFP4:
        return 0
    if role in (kvm.Role.KEY_BLOCK_SCALE, kvm.Role.VALUE_BLOCK_SCALE):
        return kvm.NVFP4_SCALE_COALESCING_GROUP
    return kvm.NVFP4_DATA_COALESCING_GROUP


@pytest.mark.parametrize("knob", [None, "0", "true"], ids=["unset", "0", "true"])
@pytest.mark.parametrize("layout", _DRAFT_LAYOUTS)
def test_knob_off_fp8_draft_layers_keep_the_grouping_without_the_rule(
    kvm, monkeypatch, layout, knob
):
    """Without ``TRTLLM_DFLASH_FP8_DRAFT_KV=1`` FP8 draft layers get no group of their own."""
    if knob is None:
        monkeypatch.delenv(_FP8_KNOB, raising=False)
    else:
        monkeypatch.setenv(_FP8_KNOB, knob)
    specs = _specs(kvm, layout, DataType.FP8)
    nvfp4 = kvm.KVCacheLayerSpec(
        global_layer_id=TARGET_LAYERS + len(DRAFT_IDS),
        owner=kvm.KVCacheLayerOwner.DRAFT,
        dtype=DataType.NVFP4,
        num_kv_heads=2,
        head_dim=128,
        buffer_roles=(
            kvm.Role.KEY,
            kvm.Role.VALUE,
            kvm.Role.KEY_BLOCK_SCALE,
            kvm.Role.VALUE_BLOCK_SCALE,
        ),
        sliding_window_size=WINDOW,
    )
    for spec in specs + [nvfp4]:
        for role in spec.buffer_roles:
            assert kvm._layer_buffer_coalescing_group(spec, role) == _group_without_fp8_rule(
                kvm, spec, role
            ), (spec.global_layer_id, role)

    manager = _manager(kvm, layout, DataType.FP8, gap=FAR)
    assert not manager._has_fp8_draft_pool()
    draft = manager.num_local_layers - 1
    target_sliding = next(
        i for i, s in enumerate(manager.layer_specs) if not _is_full(s.global_layer_id)
    )
    # Same-size FP8 buffers of one life cycle coalesce into one pool, one table per life cycle.
    key = kvm.Role.KEY
    assert manager.impl.pool_of(draft, key) is manager.impl.pool_of(target_sliding, key)
    assert manager._attention_op_pool_groups is None
    assert not manager._use_per_layer_page_tables
    assert manager.num_attention_op_pools == manager.num_pools == 2
    _check_addresses(kvm, manager)
