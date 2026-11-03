# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests for disaggregated serving on copy-based rebalancing (CoReba): bound-pool slot translation
(ArenaPoolBindings) and its wire format (RecvReqInfo.bound_pool_block_ids)."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.native.transfer import (
    ArenaPoolBindings,
    RecvReqInfo,
    _self_pool_bindings,
)
from tensorrt_llm._torch.disaggregation.resource.page import CacheKind


class _Impl:
    """Two pool groups: group 0 with pools 0, 1 (bound), group 1 with one pool."""

    def __init__(self):
        self.version = 0
        self.tables = {(0, 1): [5, 6, 7, -1, 9]}
        self.fetches = 0
        self.pool_group_descs = [
            SimpleNamespace(pool_group_index=0, pools=[object(), object()]),
            SimpleNamespace(pool_group_index=1, pools=[object()]),
        ]

    def get_page_index_binding_version(self):
        return self.version

    def get_pool_bindings(self, pg, pool):
        self.fetches += 1
        return self.tables.get((pg, pool))


def _page_table():
    kind = CacheKind.PAGED
    return SimpleNamespace(layer_groups=[
        SimpleNamespace(kind=kind, pool_group_idx=0,
                        pool_views=[SimpleNamespace(pool_idx=0), SimpleNamespace(pool_idx=1)]),
        SimpleNamespace(kind=kind, pool_group_idx=1, pool_views=[SimpleNamespace(pool_idx=0)]),
    ])


def test_translate_bound_pool_keeps_holes():
    pb = ArenaPoolBindings(_Impl())
    out = pb.translate(_page_table(), 0, 1, np.array([2, -1, 0, 4]))
    assert out.tolist() == [7, -1, 5, 9]


def test_unbound_pools_use_group_ids():
    pb = ArenaPoolBindings(_Impl())
    pt = _page_table()
    assert pb.translate(pt, 0, 0, [1, 2]) is None
    assert pb.translate(pt, 1, 0, [1, 2]) is None


def test_held_slot_without_binding_raises():
    pb = ArenaPoolBindings(_Impl())
    with pytest.raises(RuntimeError, match="without a binding"):
        pb.translate(_page_table(), 0, 1, [3])
    with pytest.raises(RuntimeError, match="beyond the binding table"):
        pb.translate(_page_table(), 0, 1, [10])


def test_tables_refetched_on_version_change():
    impl = _Impl()
    pb = ArenaPoolBindings(impl)
    pt = _page_table()
    pb.translate(pt, 0, 1, [0])
    pb.translate(pt, 0, 1, [1])
    assert impl.fetches == 1
    impl.tables[(0, 1)] = [8, 6, 7, -1, 9]
    impl.version = 1
    assert pb.translate(pt, 0, 1, [0]).tolist() == [8]
    assert impl.fetches == 2


def test_translate_all_lists_only_bound_views():
    pb = ArenaPoolBindings(_Impl())
    out = pb.translate_all(_page_table(), [np.array([0, 1]), np.array([3])])
    assert list(out) == ["0:1"]
    assert out["0:1"].tolist() == [5, 6]


def test_only_arena_pool_bindings_translate():
    """Registrars without bindings, or stand-ins for one, never translate."""
    pb = ArenaPoolBindings(_Impl())
    assert _self_pool_bindings(SimpleNamespace(self_pool_bindings=pb)) is pb
    assert _self_pool_bindings(SimpleNamespace(self_pool_bindings=None)) is None
    assert _self_pool_bindings(SimpleNamespace()) is None
    assert _self_pool_bindings(MagicMock()) is None


@pytest.mark.parametrize("bound", [None, {"0:1": np.array([5, -1, 9], dtype=np.int64)}])
def test_recv_req_info_round_trip(bound):
    info = RecvReqInfo(sender_req_id=1, instance_name="gen", instance_rank=0,
                       block_ids_per_layer_groups=[np.array([0, -1, 4], dtype=np.int64)], unique_rid=7,
                       bound_pool_block_ids=bound)
    back = RecvReqInfo.from_bytes(info.to_bytes())
    assert back.block_ids_per_layer_groups[0].tolist() == [0, -1, 4]
    if bound is None:
        assert back.bound_pool_block_ids is None
    else:
        assert {k: v.tolist() for k, v in back.bound_pool_block_ids.items()} == {"0:1": [5, -1, 9]}
