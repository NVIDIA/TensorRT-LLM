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
"""Draft-layer handshake and layer routing of a PP4 DFlash context instance, on the host only.

The context instance holds the target's 62 layers as 15/16/16/15 over four pipeline stages and
the drafter's five KV layers (global ids 62-66) on the last stage, so its stages publish
``layer_num_per_pp`` [15, 16, 16, 20]. The generation instance is TP4 and holds all 67 layers on
every rank. The tests pin:

* the switch (``pp_dflash_context.py``), which the model construction sets;
* ``RankInfo``: construction and wire bytes do not depend on the switch, and
  ``publish_instance_draft_layers`` changes only ``num_draft_layers``;
* ``PeerRegistrar.validate_draft_layers``: the same outcomes and message on any server, and
  the per-rank pairs of the PP4 -> TP4 layout before and after publishing;
* ``get_peer_overlap``, ``get_pool_mapping``, the view head geometry and the aux owner for the
  PP4 -> TP4 pair, from both ends;
* the uneven PP fan-in bounce sizing (``pp_fanin_writer_block_bytes``) with the draft layers on
  the last stage, and why a per-stage split of the draft layers would mis-size it;
* the transceiver's rank-info exchange on the context server and on any other;
* FP8 draft KV layers (``TRTLLM_DFLASH_FP8_DRAFT_KV``): the page-table views a draft coalescing
  group gives them (a pool of their own, head counts 8 / 2 next to the target's 16 / 4), the
  refusal of one pool shared with the target's FP8 layers, and the peer check that fails a pair
  whose context and generation sides store the draft layers in different dtypes.

Without the compiled bindings ``tensorrt_llm`` does not import, so the disaggregation modules
are then loaded from their files with their heavy leaf dependencies stubbed for the load, and
``transceiver.py`` with every import but ``rank_info`` stubbed.
"""

from __future__ import annotations

import ast
import enum
import importlib
import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import msgpack
import numpy as np
import pytest

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[3]
_PKG = "tensorrt_llm._torch.disaggregation"


# --------------------------------------------------------------------------
# Module loading: the real package when it imports, else stubbed leaves
# --------------------------------------------------------------------------
class _Log:
    """Logger stand-in for the stubbed load."""

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


class _DataType(enum.IntEnum):
    FLOAT = 0
    HALF = 1
    INT8 = 2
    INT32 = 3
    BOOL = 4
    UINT8 = 5
    FP8 = 6
    BF16 = 7
    INT64 = 8
    NVFP4 = 9


_DTYPE_BYTES = {
    _DataType.FLOAT: 4,
    _DataType.HALF: 2,
    _DataType.BF16: 2,
    _DataType.FP8: 1,
    _DataType.INT8: 1,
    _DataType.UINT8: 1,
}


def _stub_get_size_in_bytes(num_elements, dtype):
    return num_elements * _DTYPE_BYTES[dtype]


def _stub_nvtx_range(_name, *args, **kwargs):
    return lambda fn: fn


def _package(name: str, path: Path | None = None) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = [str(path)] if path is not None else []
    return module


def _module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


class _Stand:
    """Placeholder for any name the transceiver imports: a base class, a value or a decorator."""

    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, fn=None, *args, **kwargs):
        return fn


class _AutoStub(types.ModuleType):
    """A module whose every attribute is a fresh placeholder class."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        value = type(name, (_Stand,), {})
        setattr(self, name, value)
        return value


def _load_transceiver_with_stubbed_imports(trt: Path) -> types.ModuleType:
    """Load ``transceiver.py`` from its file, with most of its imports stubbed.

    ``rank_info`` and the switch (``pp_dflash_context``) are the real modules.

    Only ``KvCacheTransceiverV2._exchange_rank_info`` is exercised; it reaches the stubs only
    through the fakes the test passes in.
    """
    path = trt / "_torch/disaggregation/transceiver.py"
    tree = ast.parse(path.read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    # The root package already carries the logger stand-in.
    keep = {"tensorrt_llm", f"{_PKG}.native.rank_info", f"{_PKG}.pp_dflash_context"}
    for name in sorted(names):
        if not name.startswith("tensorrt_llm") or name in keep:
            continue
        parts = name.split(".")
        for i in range(1, len(parts)):
            prefix = ".".join(parts[:i])
            if prefix not in sys.modules:
                sys.modules[prefix] = _AutoStub(prefix)
                sys.modules[prefix].__path__ = []
        sys.modules[name] = _AutoStub(name)
        sys.modules[name].__path__ = []
    spec = importlib.util.spec_from_file_location(f"{_PKG}.transceiver", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _stubbed_load() -> dict:
    """Import the modules under test with stubbed leaves, then restore ``sys.modules``."""
    trt = _ROOT / "tensorrt_llm"

    class _Placeholder:
        pass

    stubs = {
        "tensorrt_llm": _package("tensorrt_llm", trt),
        "tensorrt_llm._torch": _package("tensorrt_llm._torch", trt / "_torch"),
        "tensorrt_llm._torch.pyexecutor": _package("tensorrt_llm._torch.pyexecutor"),
        "tensorrt_llm._torch.pyexecutor.kv_cache": _package(
            "tensorrt_llm._torch.pyexecutor.kv_cache"
        ),
        "tensorrt_llm._torch.pyexecutor.llm_request": _module(
            "tensorrt_llm._torch.pyexecutor.llm_request", LlmRequest=_Placeholder
        ),
        "tensorrt_llm._torch.pyexecutor.resource_manager": _module(
            "tensorrt_llm._torch.pyexecutor.resource_manager", KVCacheManager=_Placeholder
        ),
        "tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2": _module(
            "tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2", Role=_Placeholder
        ),
        "tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager": _module(
            "tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager",
            MambaHybridCacheManager=type("MambaHybridCacheManager", (), {}),
            MambaHybridCacheManagerV2=type("MambaHybridCacheManagerV2", (), {}),
        ),
        "tensorrt_llm._utils": _module(
            "tensorrt_llm._utils",
            get_size_in_bytes=_stub_get_size_in_bytes,
            nvtx_range=_stub_nvtx_range,
        ),
        "tensorrt_llm.bindings": _module("tensorrt_llm.bindings", DataType=_DataType),
        "tensorrt_llm.runtime": _package("tensorrt_llm.runtime"),
        "tensorrt_llm.runtime.kv_cache_manager_v2": _module(
            "tensorrt_llm.runtime.kv_cache_manager_v2", DataRole=_Placeholder
        ),
        # The bounce package's __init__ pulls the CUDA transport; only its pure core is needed.
        f"{_PKG}.native.bounce": _package(
            f"{_PKG}.native.bounce", trt / "_torch/disaggregation/native/bounce"
        ),
    }
    stubs["tensorrt_llm"].logger = _Log()
    before = dict(sys.modules)
    sys.modules.update(stubs)
    try:
        loaded = {
            "switch": importlib.import_module(f"{_PKG}.pp_dflash_context"),
            "rank_info": importlib.import_module(f"{_PKG}.native.rank_info"),
            "peer": importlib.import_module(f"{_PKG}.native.peer"),
            "page": importlib.import_module(f"{_PKG}.resource.page"),
            "spec": importlib.import_module(f"{_PKG}.native.mixers.attention.spec"),
            "bounce_core": importlib.import_module(f"{_PKG}.native.bounce.core"),
            "extractor": importlib.import_module(f"{_PKG}.resource.kv_extractor"),
            "DataType": _DataType,
        }
        loaded["transceiver"] = _load_transceiver_with_stubbed_imports(trt)
    finally:
        for name in set(sys.modules) - set(before):
            del sys.modules[name]
        sys.modules.update(before)
    return loaded


def _load() -> dict:
    try:
        from tensorrt_llm.bindings import DataType
    except ImportError:
        return _stubbed_load()
    return {
        "switch": importlib.import_module(f"{_PKG}.pp_dflash_context"),
        "rank_info": importlib.import_module(f"{_PKG}.native.rank_info"),
        "peer": importlib.import_module(f"{_PKG}.native.peer"),
        "page": importlib.import_module(f"{_PKG}.resource.page"),
        "spec": importlib.import_module(f"{_PKG}.native.mixers.attention.spec"),
        "bounce_core": importlib.import_module(f"{_PKG}.native.bounce.core"),
        "extractor": importlib.import_module(f"{_PKG}.resource.kv_extractor"),
        "transceiver": importlib.import_module(f"{_PKG}.transceiver"),
        "DataType": DataType,
    }


_M = _load()
rank_info_module = _M["rank_info"]
RankInfo = rank_info_module.RankInfo
PeerRegistrar = _M["peer"].PeerRegistrar
AttentionInfo = _M["spec"].AttentionInfo
page = _M["page"]
pp_fanin_writer_block_bytes = _M["bounce_core"].pp_fanin_writer_block_bytes
stage_layer_ranges = _M["bounce_core"].stage_layer_ranges
DataType = _M["DataType"]
extractor = _M["extractor"]


# --------------------------------------------------------------------------
# The served layout
# --------------------------------------------------------------------------
TARGET_LAYERS = 62
DRAFT_LAYERS = 5
PP_PARTITION = [15, 16, 16, 15]
CTX_LPP = [15, 16, 16, 15 + DRAFT_LAYERS]
GEN_TP = 4
GEN_LPP = [TARGET_LAYERS + DRAFT_LAYERS]
TOKENS_PER_BLOCK = 32
# Target: FP8, 16 KV heads, head_dim 64. Drafter: BF16, 8 KV heads, head_dim 128.
TARGET_HEADS, TARGET_HEAD_DIM, TARGET_ELEM = 16, 64, 1
DRAFT_HEADS, DRAFT_HEAD_DIM, DRAFT_ELEM = 8, 128, 2
DRAFT_IDS = list(range(TARGET_LAYERS, TARGET_LAYERS + DRAFT_LAYERS))


def _is_full(layer: int) -> bool:
    """Full-attention target layers sit at 1 mod 4; every other layer, and the drafter, slides."""
    return layer < TARGET_LAYERS and layer % 4 == 1


def _stage_target_layers(stage: int) -> list[int]:
    start = sum(PP_PARTITION[:stage])
    return list(range(start, start + PP_PARTITION[stage]))


def _buffer_bytes(heads: int, head_dim: int, elem: int) -> int:
    return TOKENS_PER_BLOCK * heads * head_dim * elem


def _make_page_table(target_layers, draft_layers, tp_size, draft_elem=DRAFT_ELEM):
    """V2-shaped page table: a sliding group (target + draft pools) and a full group.

    Local layer ids follow the layer-spec order (target layers, then draft layers). Each pool
    slot holds K and V of every layer of its pool, one after the other. ``draft_elem`` is the
    draft layers' bytes per element (2: BF16, 1: FP8).
    """
    target_buf = _buffer_bytes(TARGET_HEADS // tp_size, TARGET_HEAD_DIM, TARGET_ELEM)
    draft_buf = _buffer_bytes(DRAFT_HEADS // tp_size, DRAFT_HEAD_DIM, draft_elem)
    local_of = {gid: lid for lid, gid in enumerate(list(target_layers) + list(draft_layers))}
    roles = frozenset({"key", "value"})

    def view(pool_idx, layers, buf, heads):
        entries = []
        for i, gid in enumerate(layers):
            entries.append((local_of[gid], i * 2 * buf, buf))
            entries.append((local_of[gid], i * 2 * buf + buf, buf))
        return page.PoolView(
            pool_idx=pool_idx,
            buffer_entries=np.array(entries, dtype=page.BUFFER_ENTRY_DTYPE),
            pool_role=roles,
            mapper_kind=page.MapperKind.INDEXED,
            bytes_per_layer=2 * buf,
            num_kv_heads=heads,
        )

    def pool(base, layers, buf):
        return page.PhysicalPool(base_address=base, slot_bytes=2 * buf * len(layers), num_slots=64)

    swa_target = [layer for layer in target_layers if not _is_full(layer)]
    full_target = [layer for layer in target_layers if _is_full(layer)]
    swa_views = [view(0, swa_target, target_buf, TARGET_HEADS // tp_size)]
    swa_pools = [pool(0x1000_0000, swa_target, target_buf)]
    if draft_layers:
        swa_views.append(view(1, list(draft_layers), draft_buf, DRAFT_HEADS // tp_size))
        swa_pools.append(pool(0x2000_0000, list(draft_layers), draft_buf))
    swa_group = page.AttentionLayerGroup(
        pool_group_idx=0,
        kv_head_num_per_rank=TARGET_HEADS // tp_size,
        sliding_window_size=2048,
        local_layers=[
            page.LocalLayer(local_layer_id=local_of[gid], global_layer_id=gid)
            for gid in swa_target + list(draft_layers)
        ],
        pool_views=swa_views,
    )
    full_group = page.AttentionLayerGroup(
        pool_group_idx=1,
        kv_head_num_per_rank=TARGET_HEADS // tp_size,
        sliding_window_size=None,
        local_layers=[
            page.LocalLayer(local_layer_id=local_of[gid], global_layer_id=gid)
            for gid in full_target
        ],
        pool_views=[view(0, full_target, target_buf, TARGET_HEADS // tp_size)],
    )
    return page.KVCachePageTable(
        tokens_per_block=TOKENS_PER_BLOCK,
        layer_groups=[swa_group, full_group],
        pool_groups=[
            page.PhysicalPoolGroup(pools=swa_pools),
            page.PhysicalPoolGroup(pools=[pool(0x3000_0000, full_target, target_buf)]),
        ],
    )


def _rank_info(*, name, rank, tp_size, tp_rank, pp_size, pp_rank, lpp, page_table, draft):
    return RankInfo(
        instance_name=name,
        instance_rank=rank,
        tp_size=tp_size,
        tp_rank=tp_rank,
        pp_size=pp_size,
        pp_rank=pp_rank,
        layer_num_per_pp=list(lpp),
        sender_endpoints=[],
        self_endpoint="",
        transfer_engine_info=b"",
        attention=AttentionInfo(
            kv_heads_per_rank=TARGET_HEADS // tp_size,
            tokens_per_block=TOKENS_PER_BLOCK,
            dims_per_head=TARGET_HEAD_DIM,
            element_bytes=TARGET_ELEM,
            enable_attention_dp=False,
            is_mla=False,
        ),
        page_table=page_table,
        num_draft_layers=draft,
    )


def _ctx_rank_info(stage: int, *, published: bool, draft_elem: int = DRAFT_ELEM) -> RankInfo:
    draft = DRAFT_IDS if stage == len(PP_PARTITION) - 1 else []
    ri = _rank_info(
        name="ctx",
        rank=stage,
        tp_size=1,
        tp_rank=0,
        pp_size=len(PP_PARTITION),
        pp_rank=stage,
        lpp=CTX_LPP,
        page_table=_make_page_table(_stage_target_layers(stage), draft, 1, draft_elem),
        draft=len(draft),
    )
    if published:
        ri.publish_instance_draft_layers([0, 0, 0, DRAFT_LAYERS])
    return ri


def _gen_rank_info(tp_rank: int, draft_elem: int = DRAFT_ELEM) -> RankInfo:
    return _rank_info(
        name="gen",
        rank=tp_rank,
        tp_size=GEN_TP,
        tp_rank=tp_rank,
        pp_size=1,
        pp_rank=0,
        lpp=GEN_LPP,
        page_table=_make_page_table(range(TARGET_LAYERS), DRAFT_IDS, GEN_TP, draft_elem),
        draft=DRAFT_LAYERS,
    )


def _registrar(self_ri: RankInfo) -> PeerRegistrar:
    return PeerRegistrar(self_ri, SimpleNamespace(page_table=self_ri.page_table))


# --------------------------------------------------------------------------
# The switch
# --------------------------------------------------------------------------
def _server(monkeypatch, enabled):
    """What the model construction of a pipeline-parallel DFlash context server records."""
    monkeypatch.setattr(_M["switch"], "_pp_dflash_context_server", bool(enabled))


@pytest.mark.parametrize("enabled", [False, True])
def test_switch_reader(monkeypatch, enabled):
    _server(monkeypatch, enabled)
    switch = _M["switch"]
    assert switch.pp_dflash_context_enabled() is enabled
    # The transceiver gates its publish on the one shared switch.
    assert _M["transceiver"].pp_dflash_context_enabled is switch.pp_dflash_context_enabled


# --------------------------------------------------------------------------
# RankInfo wire bytes
# --------------------------------------------------------------------------
# Field order of the msgpack map RankInfo puts on the wire.
_WIRE_FIELDS = [
    "instance_name",
    "instance_rank",
    "tp_size",
    "tp_rank",
    "pp_size",
    "pp_rank",
    "layer_num_per_pp",
    "sender_endpoints",
    "self_endpoint",
    "transfer_engine_info",
    "dp_size",
    "dp_rank",
    "cp_size",
    "cp_rank",
    "device_id",
    "attention",
    "aux_meta",
    "page_table",
    "num_draft_layers",
]


def _manager(*, pp_rank, pp_layers, draft_layer_ids, layer_specs=(object(),)):
    return SimpleNamespace(
        mapping=SimpleNamespace(
            rank=pp_rank,
            tp_size=1,
            tp_rank=0,
            pp_size=4,
            pp_rank=pp_rank,
            dp_size=1,
            cp_size=1,
            cp_rank=0,
            enable_attention_dp=False,
        ),
        num_kv_heads_per_layer=[TARGET_HEADS] * len(pp_layers),
        pp_layers=list(pp_layers),
        tokens_per_block=TOKENS_PER_BLOCK,
        head_dim=TARGET_HEAD_DIM,
        dtype=DataType.FP8,
        kv_factor=2,
        layer_specs=layer_specs,
        draft_layer_ids=tuple(draft_layer_ids),
    )


def _expected_wire(manager, num_draft_layers) -> bytes:
    """The ``_WIRE_FIELDS`` map ``from_kv_cache_manager`` + ``to_bytes`` put on the wire."""
    m = manager.mapping
    values = {
        "instance_name": "ctx",
        "instance_rank": m.rank,
        "tp_size": m.tp_size,
        "tp_rank": m.tp_rank,
        "pp_size": m.pp_size,
        "pp_rank": m.pp_rank,
        "layer_num_per_pp": [len(manager.pp_layers)],
        "sender_endpoints": [],
        "self_endpoint": "",
        "transfer_engine_info": b"",
        "dp_size": m.dp_size,
        "dp_rank": 0,
        "cp_size": m.cp_size,
        "cp_rank": m.cp_rank,
        "device_id": 0,
        "attention": {
            "kv_heads_per_rank": TARGET_HEADS,
            "tokens_per_block": TOKENS_PER_BLOCK,
            "dims_per_head": TARGET_HEAD_DIM,
            "element_bytes": 1,
            "enable_attention_dp": False,
            "is_mla": False,
        },
        "aux_meta": None,
        "page_table": None,
        "num_draft_layers": num_draft_layers,
    }
    return msgpack.packb({key: values[key] for key in _WIRE_FIELDS})


@pytest.mark.parametrize("server", [False, True], ids=["other", "pp-dflash-ctx"])
@pytest.mark.parametrize("stage", range(4))
def test_rank_info_construction_and_bytes_do_not_depend_on_the_switch(monkeypatch, server, stage):
    """Each stage's RankInfo is built from its own manager alone, on any server."""
    _server(monkeypatch, server)
    monkeypatch.setattr(rank_info_module, "build_page_table_from_manager", lambda _: None)
    draft = DRAFT_IDS if stage == 3 else []
    manager = _manager(
        pp_rank=stage,
        pp_layers=_stage_target_layers(stage) + draft,
        draft_layer_ids=draft,
    )
    info = RankInfo.from_kv_cache_manager("ctx", manager, device_id=0)
    assert info.num_draft_layers == len(draft)
    assert info.to_bytes() == _expected_wire(manager, len(draft))
    assert RankInfo.from_bytes(info.to_bytes()) == info


@pytest.mark.parametrize(
    ("layer_specs", "draft_ids", "expected"),
    [(None, (), 0), (None, (62,), 0), ((object(),), (), 0), ((object(),), tuple(DRAFT_IDS), 5)],
)
def test_unified_draft_layer_count(layer_specs, draft_ids, expected):
    manager = _manager(pp_rank=3, pp_layers=[], draft_layer_ids=draft_ids, layer_specs=layer_specs)
    assert rank_info_module.unified_draft_layer_count(manager) == expected


@pytest.mark.parametrize("stage", range(4))
def test_publish_changes_only_the_draft_count(stage):
    before = _ctx_rank_info(stage, published=False)
    after = _ctx_rank_info(stage, published=True)
    assert after.num_draft_layers == DRAFT_LAYERS
    unpacked_before = msgpack.unpackb(before.to_bytes(), strict_map_key=False)
    unpacked_after = msgpack.unpackb(after.to_bytes(), strict_map_key=False)
    assert list(unpacked_after) == _WIRE_FIELDS
    changed = {key for key in _WIRE_FIELDS if unpacked_before[key] != unpacked_after[key]}
    assert changed <= {"num_draft_layers"}
    restored = RankInfo.from_bytes(after.to_bytes())
    assert restored.num_draft_layers == DRAFT_LAYERS
    assert restored.to_bytes() == after.to_bytes()


def test_publish_on_a_single_stage_keeps_the_count():
    ri = _gen_rank_info(0)
    ri.publish_instance_draft_layers([DRAFT_LAYERS])
    assert ri.num_draft_layers == DRAFT_LAYERS


@pytest.mark.parametrize(
    ("stage", "counts", "match"),
    [
        (0, [0, 0, 5], "3 per-stage draft layer counts for pp_size=4"),
        (0, [5, 0, 0, 5], "stage 0 reports 5 draft layers but holds 0"),
        (3, [0, 0, 0, 4], "stage 3 reports 4 draft layers but holds 5"),
    ],
)
def test_publish_rejects_inconsistent_counts(stage, counts, match):
    ri = _ctx_rank_info(stage, published=False)
    with pytest.raises(ValueError, match=match):
        ri.publish_instance_draft_layers(counts)


def test_publish_rejects_draft_layers_before_the_last_stage():
    """A 2/1/1/1 split of the draft layers breaks the contiguous per-stage layer ranges."""
    ri = _ctx_rank_info(0, published=False)
    ri.num_draft_layers = 2
    with pytest.raises(ValueError, match=r"\[2, 1, 1, 1\].*all draft layers on the last stage"):
        ri.publish_instance_draft_layers([2, 1, 1, 1])


# --------------------------------------------------------------------------
# validate_draft_layers
# --------------------------------------------------------------------------
def _validation_outcome(local: int, peer: int):
    self_ri = _gen_rank_info(0)
    self_ri.num_draft_layers = local
    peer_ri = _gen_rank_info(1)
    peer_ri.num_draft_layers = peer
    try:
        PeerRegistrar.validate_draft_layers(self_ri, peer_ri)
    except ValueError as error:
        return str(error)
    return None


@pytest.mark.parametrize("local", [0, 1, 5])
@pytest.mark.parametrize("peer", [0, 1, 5])
def test_validate_draft_layers_matrix_does_not_depend_on_the_switch(monkeypatch, local, peer):
    _server(monkeypatch, False)
    unset = _validation_outcome(local, peer)
    _server(monkeypatch, True)
    assert _validation_outcome(local, peer) == unset
    if local == peer:
        assert unset is None
    else:
        assert unset == (
            "PeerRegistrar: unified draft KV layer count differs "
            f"(local={local}, peer={peer}), so "
            "the drafter's prompt KV would not reach the generation worker. Run the "
            "context and generation workers with the same speculative_config and the "
            "same TRTLLM_DISAGG_SHARED_DRAFT_KV setting. To draft on the generation "
            "worker alone, keep its draft KV private with TRTLLM_DISAGG_SHARED_DRAFT_KV=0 "
            "or attention_backend VANILLA."
        )


@pytest.mark.parametrize("gen_tp_rank", range(GEN_TP))
@pytest.mark.parametrize("stage", range(4))
def test_pp4_to_tp4_draft_check_per_rank_pair(stage, gen_tp_rank):
    """Unpublished, every pair but (stage 3, any) fails; published, every pair passes.

    The generation rank checks the first context rank's RankInfo, and each context rank checks
    every generation rank's RankInfo.
    """
    gen = _gen_rank_info(gen_tp_rank)
    unpublished = _ctx_rank_info(stage, published=False)
    if stage == 3:
        PeerRegistrar.validate_draft_layers(unpublished, gen)
    else:
        with pytest.raises(ValueError, match=r"local=0, peer=5"):
            PeerRegistrar.validate_draft_layers(unpublished, gen)
    with pytest.raises(ValueError, match=r"local=5, peer=0"):
        PeerRegistrar.validate_draft_layers(gen, _ctx_rank_info(0, published=False))

    published = _ctx_rank_info(stage, published=True)
    PeerRegistrar.validate_draft_layers(published, gen)
    PeerRegistrar.validate_draft_layers(gen, _ctx_rank_info(0, published=True))


@pytest.mark.parametrize("stage", range(4))
def test_context_sender_accepts_every_generation_rank_after_publish(stage):
    """The context side's full peer check (layer counts, view geometry, draft count)."""
    registrar = _registrar(_ctx_rank_info(stage, published=True))
    for gen_tp_rank in range(GEN_TP):
        assert registrar._check_peer_compatible(_gen_rank_info(gen_tp_rank))


# --------------------------------------------------------------------------
# Layer and rank routing, PP4 -> TP4
# --------------------------------------------------------------------------
def test_stage_layer_ranges_cover_the_draft_layers_on_the_last_stage():
    assert stage_layer_ranges(CTX_LPP) == [(0, 15), (15, 31), (31, 47), (47, 67)]
    assert stage_layer_ranges(GEN_LPP) == [(0, 67)]


@pytest.mark.parametrize("gen_tp_rank", range(GEN_TP))
def test_generation_rank_fans_in_from_every_stage(gen_tp_rank):
    gen = _gen_rank_info(gen_tp_rank)
    registrar = _registrar(gen)
    overlap = registrar.get_peer_overlap(_ctx_rank_info(0, published=True), peer_dp_rank=0)
    assert overlap.overlap_pp_size == 4
    assert overlap.overlap_tp_size == 1
    assert overlap.overlap_cp_size == 1
    assert overlap.ranks == [0, 1, 2, 3]
    assert (overlap.duplicate_head_factor, overlap.peer_duplicate_head_factor) == (1, 1)
    assert overlap.view_duplicate_head_factors == {}


@pytest.mark.parametrize("stage", range(4))
def test_context_stage_sends_to_every_generation_rank(stage):
    registrar = _registrar(_ctx_rank_info(stage, published=True))
    gen = _gen_rank_info(0)
    overlap = registrar.get_peer_overlap(gen, peer_dp_rank=0)
    assert overlap.overlap_pp_size == 1
    assert overlap.overlap_tp_size == GEN_TP
    assert overlap.ranks == [0, 1, 2, 3]
    assert (overlap.duplicate_head_factor, overlap.peer_duplicate_head_factor) == (1, 1)
    assert overlap.view_duplicate_head_factors == {}
    # Only the first stage sends the request's aux data to a generation rank.
    assert registrar.should_send_aux(gen) is (stage == 0)


def _view_layers(page_table, lg_idx, pool_idx):
    layer_group = page_table.layer_groups[lg_idx]
    local_to_global = {ll.local_layer_id: ll.global_layer_id for ll in layer_group.local_layers}
    entries = layer_group.pool_views[pool_idx].buffer_entries
    return sorted({local_to_global[int(e["local_layer_id"])] for e in entries})


@pytest.mark.parametrize("stage", range(4))
def test_pool_mapping_pairs_draft_views_on_the_last_stage_only(stage):
    ctx = _ctx_rank_info(stage, published=True)
    gen = _gen_rank_info(0)
    # Receiver side: the generation rank's draft view has a peer on the last stage only.
    gen_mapping = _registrar(gen).get_pool_mapping(ctx)
    draft_key = (0, 1)
    assert _view_layers(gen.page_table, *draft_key) == DRAFT_IDS
    if stage == 3:
        assert gen_mapping[draft_key] == (0, 1)
        assert _view_layers(ctx.page_table, 0, 1) == DRAFT_IDS
    else:
        assert draft_key not in gen_mapping
    assert gen_mapping[(0, 0)] == (0, 0)
    assert gen_mapping[(1, 0)] == (1, 0)
    # Sender side: the same pairs, seen from the context stage.
    ctx_registrar = _registrar(ctx)
    ctx_mapping = ctx_registrar.get_pool_mapping(gen)
    assert ctx_mapping == {value: key for key, value in gen_mapping.items()}
    ctx_registrar._validate_view_head_geometry(gen)


# --------------------------------------------------------------------------
# Uneven PP fan-in bounce sizing
# --------------------------------------------------------------------------
def _gen_layer_block_bytes():
    """Receiver bytes per block for each layer, per layer group (sliding, full), on a TP4 rank."""
    target = 2 * _buffer_bytes(TARGET_HEADS // GEN_TP, TARGET_HEAD_DIM, TARGET_ELEM)
    draft = 2 * _buffer_bytes(DRAFT_HEADS // GEN_TP, DRAFT_HEAD_DIM, DRAFT_ELEM)
    swa = {layer: target for layer in range(TARGET_LAYERS) if not _is_full(layer)}
    swa.update({layer: draft for layer in DRAFT_IDS})
    full = {layer: target for layer in range(TARGET_LAYERS) if _is_full(layer)}
    return [swa, full], target, draft


def test_pp_fanin_writer_bytes_with_the_draft_layers_on_the_last_stage():
    groups, target, draft = _gen_layer_block_bytes()
    assert (target, draft) == (16384, 32768)
    writers = pp_fanin_writer_block_bytes(groups, [0, 1, 2, 3], CTX_LPP)
    # Stage s holds 11/12/12/11 sliding and 4 full target layers; stage 3 adds the five draft
    # layers to its sliding group.
    assert writers == [
        [11 * target, 4 * target],
        [12 * target, 4 * target],
        [12 * target, 4 * target],
        [11 * target + 5 * draft, 4 * target],
    ]
    # Every receiver byte has exactly one writer.
    assert [sum(column) for column in zip(*writers)] == [sum(g.values()) for g in groups]
    # Each writer's share is what that stage's page table holds of the receiver's layers.
    for stage in range(4):
        held = set(_stage_target_layers(stage)) | (set(DRAFT_IDS) if stage == 3 else set())
        assert writers[stage] == [
            sum(b for layer, b in group.items() if layer in held) for group in groups
        ]


def test_a_per_stage_draft_split_would_mis_size_the_writers():
    """With the draft layers split 2/1/1/1, a stage's contiguous range is not the layers it holds."""
    groups, _, _ = _gen_layer_block_bytes()
    split_lpp = [15 + 2, 16 + 1, 16 + 1, 15 + 1]
    writers = pp_fanin_writer_block_bytes(groups, [0, 1, 2, 3], split_lpp)
    held_stage0 = set(_stage_target_layers(0)) | {62, 63}
    assert writers[0] != [sum(b for layer, b in g.items() if layer in held_stage0) for g in groups]


# --------------------------------------------------------------------------
# Transceiver rank-info exchange
# --------------------------------------------------------------------------
class _FakeDist:
    def __init__(self, stage: int, pp_values: list):
        self._stage = stage
        self._pp_values = pp_values
        self.pp_allgather_calls = []

    def allgather(self, obj):
        return [obj] * 4

    def pp_allgather(self, obj):
        self.pp_allgather_calls.append(obj)
        values = self._pp_values.pop(0)
        assert values[self._stage] == obj
        return values


@pytest.mark.parametrize("server", [False, True], ids=["other", "pp-dflash-ctx"])
@pytest.mark.parametrize("stage", range(4))
def test_exchange_rank_info_publishes_only_on_the_server(monkeypatch, server, stage):
    transceiver = _M["transceiver"]
    _server(monkeypatch, server)
    draft = DRAFT_IDS if stage == 3 else []
    ri = _ctx_rank_info(stage, published=False)
    ri.layer_num_per_pp = [len(_stage_target_layers(stage)) + len(draft)]
    populated = {}

    def populate(endpoints, layer_num_per_pp):
        populated.update(endpoints=endpoints, layer_num_per_pp=layer_num_per_pp)
        ri.sender_endpoints = endpoints
        ri.layer_num_per_pp = layer_num_per_pp

    tc = transceiver.KvCacheTransceiverV2.__new__(transceiver.KvCacheTransceiverV2)
    tc._dist = _FakeDist(stage, [list(CTX_LPP), [0, 0, 0, DRAFT_LAYERS]])
    tc._kv_cache_manager = SimpleNamespace(
        pp_layers=_stage_target_layers(stage) + draft,
        layer_specs=(object(),),
        draft_layer_ids=tuple(draft),
    )
    tc._transfer_worker = SimpleNamespace(
        sender_endpoint=f"tcp://ctx{stage}",
        populate_instance_and_rank_info=populate,
        _rank_info=ri,
    )
    tc._context_info_endpoint = "tcp://ctx-info"
    tc._exchange_rank_info()

    assert populated["layer_num_per_pp"] == CTX_LPP
    if not server:
        assert tc._dist.pp_allgather_calls == [len(_stage_target_layers(stage)) + len(draft)]
        assert ri.num_draft_layers == len(draft)
    else:
        assert tc._dist.pp_allgather_calls[1] == len(draft)
        assert ri.num_draft_layers == DRAFT_LAYERS


# --------------------------------------------------------------------------
# A drafter context window: the context peer's keep count, and the generation worker's reuse
# --------------------------------------------------------------------------
@pytest.mark.parametrize("keep", [None, 0, 2592])
@pytest.mark.parametrize("reuse", [False, True])
def test_window_keep_count_is_on_the_wire_only_when_set(monkeypatch, keep, reuse):
    """A window's keep count goes on the wire last, and only when set.

    A manager without a window (or with 0) puts only the ``_WIRE_FIELDS`` map on the wire.
    Whether this worker reuses blocks is never sent.
    """
    monkeypatch.setattr(rank_info_module, "build_page_table_from_manager", lambda _: None)
    manager = _manager(
        pp_rank=3, pp_layers=_stage_target_layers(3) + DRAFT_IDS, draft_layer_ids=DRAFT_IDS
    )
    manager.enable_block_reuse = reuse
    if keep is not None:
        manager.draft_context_keep_tokens = keep
    info = RankInfo.from_kv_cache_manager("ctx", manager, device_id=0)
    assert info.reuses_cached_blocks is reuse
    assert info.draft_context_keep_tokens == (keep or 0)
    wire = msgpack.unpackb(info.to_bytes(), strict_map_key=False)
    assert "reuses_cached_blocks" not in wire
    if keep:
        assert list(wire) == _WIRE_FIELDS + ["draft_context_keep_tokens"]
        assert wire["draft_context_keep_tokens"] == keep
    else:
        assert info.to_bytes() == _expected_wire(manager, DRAFT_LAYERS)
    restored = RankInfo.from_bytes(info.to_bytes())
    assert restored.draft_context_keep_tokens == (keep or 0)
    assert restored.reuses_cached_blocks is False


@pytest.mark.parametrize("peer_keep", [0, 2592])
@pytest.mark.parametrize("self_keep", [0, 2592])
@pytest.mark.parametrize("self_reuse", [False, True])
def test_validate_draft_context_window(peer_keep, self_keep, self_reuse):
    """A block-reusing worker that keeps every row refuses a context peer with a window.

    That is a generation worker with block reuse on; every other pair is accepted.
    """
    self_ri = _gen_rank_info(0)
    self_ri.draft_context_keep_tokens = self_keep
    self_ri.reuses_cached_blocks = self_reuse
    peer_ri = _gen_rank_info(1)
    peer_ri.draft_context_keep_tokens = peer_keep
    if peer_keep and not self_keep and self_reuse:
        with pytest.raises(ValueError, match="enable_block_reuse=false"):
            PeerRegistrar.validate_draft_context_window(self_ri, peer_ri)
    else:
        PeerRegistrar.validate_draft_context_window(self_ri, peer_ri)


# --------------------------------------------------------------------------
# FP8 draft KV layers (TRTLLM_DFLASH_FP8_DRAFT_KV)
# --------------------------------------------------------------------------
FP8_ELEM = 1
_DRAFT_GROUP = 3


def _sliding_variant(target_layers, tp_size, *, draft_group):
    """The sliding life cycle's coalesced buffers, as the manager's storage config forms them.

    A model of the C++ storage config: buffers of one life cycle coalesce into one physical
    pool when their (coalescing group, size) match, and the pools are ordered by slot size,
    largest first. Target layers are FP8 (group 0); the drafter's FP8 layers carry
    ``draft_group``. Local layer ids follow the layer-spec order (target, then draft).
    """
    target_buf = _buffer_bytes(TARGET_HEADS // tp_size, TARGET_HEAD_DIM, TARGET_ELEM)
    draft_buf = _buffer_bytes(DRAFT_HEADS // tp_size, DRAFT_HEAD_DIM, FP8_ELEM)
    sliding = [layer for layer in target_layers if not _is_full(layer)]
    local_of = {gid: lid for lid, gid in enumerate(list(target_layers) + DRAFT_IDS)}
    buffers = [(local_of[gid], role, target_buf, 0) for gid in sliding for role in ("key", "value")]
    buffers += [
        (local_of[gid], role, draft_buf, draft_group)
        for gid in DRAFT_IDS
        for role in ("key", "value")
    ]
    pools: dict = {}
    for layer_id, role, size, group in buffers:
        pools.setdefault((group, size), []).append(SimpleNamespace(layer_id=layer_id, role=role))
    coalesced = [
        SimpleNamespace(single_buffer_size=size, buffer_ids=ids)
        for (_, size), ids in sorted(pools.items(), key=lambda item: -item[0][1] * len(item[1]))
    ]
    heads = {local_of[gid]: TARGET_HEADS // tp_size for gid in sliding}
    heads.update({local_of[gid]: DRAFT_HEADS // tp_size for gid in DRAFT_IDS})
    return SimpleNamespace(coalesced_buffers=coalesced), heads


def _sliding_views(target_layers, tp_size, *, draft_group):
    variant, heads = _sliding_variant(target_layers, tp_size, draft_group=draft_group)
    return extractor._build_pool_views_for_variant(
        variant,
        {},
        page.MapperKind.INDEXED,
        layer_group_id=0,
        kv_heads_for_layer=heads.get,
    )


@pytest.mark.parametrize(
    "side,target_layers,tp_size",
    [("ctx_stage3", _stage_target_layers(3), 1), ("gen_tp4", list(range(TARGET_LAYERS)), GEN_TP)],
)
def test_fp8_draft_layers_get_views_of_their_own(side, target_layers, tp_size):
    target_buf = _buffer_bytes(TARGET_HEADS // tp_size, TARGET_HEAD_DIM, TARGET_ELEM)
    draft_buf = _buffer_bytes(DRAFT_HEADS // tp_size, DRAFT_HEAD_DIM, FP8_ELEM)
    # FP8 draft buffers are the size of the target's: 32 KiB on the context stage, 8 KiB on a
    # generation rank.
    assert target_buf == draft_buf == {"ctx_stage3": 32 << 10, "gen_tp4": 8 << 10}[side]
    views = _sliding_views(target_layers, tp_size, draft_group=_DRAFT_GROUP)
    assert [view.pool_idx for view in views] == [0, 1]
    assert [view.num_kv_heads for view in views] == [
        TARGET_HEADS // tp_size,
        DRAFT_HEADS // tp_size,
    ]
    assert [view.bytes_per_layer for view in views] == [2 * target_buf, 2 * draft_buf]
    # Bytes per KV head per layer (K and V): 64 per token for the target, 128 for the drafter.
    tokens = TOKENS_PER_BLOCK
    assert [view.bytes_per_layer // (2 * view.num_kv_heads * tokens) for view in views] == [64, 128]


@pytest.mark.parametrize(
    "target_layers,tp_size",
    [(_stage_target_layers(3), 1), (list(range(TARGET_LAYERS)), GEN_TP)],
    ids=["ctx_stage3", "gen_tp4"],
)
def test_fp8_draft_layers_in_the_target_pool_are_refused(target_layers, tp_size):
    """Without a group of their own the FP8 draft buffers coalesce into the target's pool."""
    variant, _ = _sliding_variant(target_layers, tp_size, draft_group=0)
    assert len(variant.coalesced_buffers) == 1
    with pytest.raises(ValueError, match="mixes layers with different per-rank KV-head counts"):
        _sliding_views(target_layers, tp_size, draft_group=0)


@pytest.mark.parametrize("gen_tp_rank", range(GEN_TP))
def test_fp8_draft_layers_on_both_sides_pass_the_head_geometry(gen_tp_rank):
    ctx = _ctx_rank_info(3, published=True, draft_elem=FP8_ELEM)
    gen = _gen_rank_info(gen_tp_rank, draft_elem=FP8_ELEM)
    gen_registrar = _registrar(gen)
    gen_registrar._validate_view_head_geometry(ctx)
    _registrar(ctx)._validate_view_head_geometry(gen)
    assert gen_registrar.get_pool_mapping(ctx) == _registrar(
        _gen_rank_info(gen_tp_rank)
    ).get_pool_mapping(_ctx_rank_info(3, published=True))


@pytest.mark.parametrize(
    "ctx_elem,gen_elem",
    [(FP8_ELEM, DRAFT_ELEM), (DRAFT_ELEM, FP8_ELEM)],
    ids=["ctx_fp8", "gen_fp8"],
)
def test_one_sided_fp8_draft_layers_fail_the_head_geometry(ctx_elem, gen_elem):
    ctx = _ctx_rank_info(3, published=True, draft_elem=ctx_elem)
    gen = _gen_rank_info(0, draft_elem=gen_elem)
    with pytest.raises(ValueError, match="bytes per KV head differ"):
        _registrar(gen)._validate_view_head_geometry(ctx)
    with pytest.raises(ValueError, match="bytes per KV head differ"):
        _registrar(ctx)._validate_view_head_geometry(gen)
