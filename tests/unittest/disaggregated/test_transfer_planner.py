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
"""Contract tests for the model-owned transfer planner seam."""

import numpy as np
import pytest

from tensorrt_llm._torch.disaggregation.base.planner import (
    CacheDescriber,
    LayerGroupRows,
    PeerIncompatibleError,
    PeerTargets,
    TransferPlan,
    TransferPlanner,
)


def test_transfer_plan_empty_is_three_int64_arrays():
    plan = TransferPlan.empty()
    for arr in (plan.src_ptrs, plan.dst_ptrs, plan.sizes):
        assert arr.dtype == np.int64 and arr.size == 0


def test_transfer_planner_is_abstract():
    with pytest.raises(TypeError):
        TransferPlanner()  # type: ignore[abstract]
    with pytest.raises(TypeError):
        CacheDescriber()  # type: ignore[abstract]


def test_peer_incompatible_error_is_value_error():
    assert issubclass(PeerIncompatibleError, ValueError)
    assert PeerTargets(ranks=[0], uniform_writers=True).ranks == [0]


# --------------------------------------------------------------------------- #
# Sender drives the planner
# --------------------------------------------------------------------------- #
from types import SimpleNamespace  # noqa: E402
from unittest.mock import Mock  # noqa: E402

from tensorrt_llm import DisaggregatedParams  # noqa: E402
from tensorrt_llm._torch.disaggregation.base import (  # noqa: E402
    CacheKind,
    Chunk,
    RecvReqInfo,
    TokenRange,
)
from tensorrt_llm._torch.disaggregation.native.transfer import KVSendTask, Sender  # noqa: E402


class _ToyPlanner(TransferPlanner):
    """Fixed ranks and a fixed plan; records what the framework handed it."""

    planner_id = "toy"

    def __init__(self, plan: TransferPlan, ranks=(0,)):
        self._plan = plan
        self._ranks = list(ranks)
        self.calls: list = []

    def check_peer(self, self_ri, peer_ri):
        return None

    def targets(self, self_ri, peer_ri, peer_dp_rank):
        return PeerTargets(ranks=list(self._ranks), uniform_writers=True)

    def pair(
        self, self_ri, peer_ri, self_rows, peer_rows, *, token_range, prompt_len, dst_start_token
    ):
        self.calls.append((self_rows, peer_rows, token_range, prompt_len, dst_start_token))
        return self._plan

    def receiver_payload_bytes(self, self_ri, peer_ri, req_info):
        return 0


class _ToyDescriber(CacheDescriber):
    """One row per held slot at address == slot; records what the framework handed it."""

    describer_id = "toy"

    def __init__(self):
        self.calls: list = []

    def rows(self, page_table, slot_table_per_layer_group, excluded_views):
        self.calls.append((page_table, slot_table_per_layer_group, excluded_views))
        table = np.asarray(slot_table_per_layer_group[0], dtype=np.int64)
        held = np.flatnonzero(table >= 0)
        return [
            LayerGroupRows(
                layer_group_index=0,
                view_index=0,
                kind=CacheKind.PAGED,
                block_ordinals=held,
                addresses=table[held],
                row_bytes=1,
                span_blocks=int(table.size),
            )
        ]


def _sender_with(planner: TransferPlanner, describer: CacheDescriber | None = None) -> Sender:
    peer_ri = SimpleNamespace(
        instance_name="gen",
        instance_rank=3,
        dp_rank=0,
        device_id=1,
        cp_size=1,
        self_endpoint="tcp://gen:1",
        page_table="peer page table",
    )
    sender = Sender.__new__(Sender)
    sender._registrar = Mock()
    sender._registrar.self_rank_info = SimpleNamespace(
        cp_size=1, instance_name="ctx", instance_rank=0, page_table="self page table"
    )
    sender._registrar.get_peer_rank_info.return_value = peer_ri
    sender._planner = planner
    sender._describer = describer or _ToyDescriber()
    return sender


def _task_and_req():
    task = KVSendTask(
        Chunk(
            block_ids_per_layer_groups=[np.array([5, 6], dtype=np.int64)],
            kind_per_layer_group=[CacheKind.PAGED],
            token_range=TokenRange(0, 64),
            is_last=True,
            excluded_pool_views={(0, 1)},
        ),
        DisaggregatedParams(disagg_request_id=1),
        slice_id=0,
        prompt_len=64,
    )
    req = RecvReqInfo(
        sender_req_id=1,
        instance_name="gen",
        instance_rank=3,
        block_ids_per_layer_groups=[np.array([9, 10], dtype=np.int64)],
        unique_rid=1,
        dst_start_token=32,
        slice_id=0,
    )
    return task, req


def test_build_kv_write_meta_describes_both_sides_then_pairs():
    plan = TransferPlan(
        src_ptrs=np.array([100, 200], dtype=np.int64),
        dst_ptrs=np.array([300, 400], dtype=np.int64),
        sizes=np.array([16, 16], dtype=np.int64),
    )
    planner = _ToyPlanner(plan, ranks=(3, 4))
    describer = _ToyDescriber()
    sender = _sender_with(planner, describer)
    task, req = _task_and_req()

    wm = sender._build_kv_write_meta(task, req)

    assert wm.src_ptrs is plan.src_ptrs and wm.dst_ptrs is plan.dst_ptrs and wm.sizes is plan.sizes
    assert wm.expected_transfers == 2 and task.expected_transfers == 2
    assert wm.dst_device_id == 1 and wm.peer_rank == 3 and wm.peer_endpoint == "tcp://gen:1"
    # The describer sees each side's page table and slot tables; only this side excludes views.
    (self_pt, self_tables, self_excluded), (peer_pt, peer_tables, peer_excluded) = describer.calls
    assert self_pt == "self page table" and self_tables is task._chunk.block_ids_per_layer_groups
    assert self_excluded == {(0, 1)}
    assert peer_pt == "peer page table" and peer_tables is req.block_ids_per_layer_groups
    assert not peer_excluded
    # The planner pairs the rows the describer produced, plus the request's bounds.
    ((self_rows, peer_rows, token_range, prompt_len, dst_start_token),) = planner.calls
    assert self_rows[0].addresses.tolist() == [5, 6] and peer_rows[0].addresses.tolist() == [9, 10]
    assert token_range == task._chunk.token_range and prompt_len == 64 and dst_start_token == 32


def test_build_kv_write_meta_empty_plan_still_delivers():
    sender = _sender_with(_ToyPlanner(TransferPlan.empty()))
    task, req = _task_and_req()

    wm = sender._build_kv_write_meta(task, req)

    assert wm.sizes.size == 0 and wm.src_ptrs.size == 0 and wm.dst_ptrs.size == 0
    assert wm.expected_transfers == 1 and wm.is_last_slice is True


# --------------------------------------------------------------------------- #
# Receiver drives the planner; transfer.py knows no cache kinds
# --------------------------------------------------------------------------- #
import inspect  # noqa: E402

import tensorrt_llm._torch.disaggregation.native.transfer as transfer_mod  # noqa: E402
from tensorrt_llm._torch.disaggregation.native.transfer import Receiver  # noqa: E402


def test_transfer_module_has_no_model_knowledge():
    src = inspect.getsource(transfer_mod)
    for forbidden in ("mixers", "CacheKind", "MapperKind", "MambaPolicy", "AttentionPolicy"):
        assert forbidden not in src, f"native/transfer.py mentions {forbidden}"


def _receiver_with(planner) -> tuple[Receiver, SimpleNamespace]:
    recv = Receiver.__new__(Receiver)
    recv._enforce_physical_ownership = False
    recv._registrar = SimpleNamespace(
        self_rank_info=SimpleNamespace(cp_size=1, instance_name="gen", instance_rank=0)
    )
    recv._planner = planner
    recv._bounce = Mock()
    req = SimpleNamespace(
        unique_rid=7,
        slice_id=0,
        block_ids_per_layer_groups=[np.array([1], dtype=np.int64)],
        bounce_dst_base=None,
        to_bytes=Mock(return_value=b"req"),
    )
    recv._build_recv_req_info = Mock(return_value=req)
    recv._get_sender_info = Mock(
        return_value=SimpleNamespace(
            sender_endpoints={0: "tcp://s0"}, page_table=None, dp_size=1, cp_size=1
        )
    )
    session = SimpleNamespace(mark_transferring=Mock(), _sender_endpoints=set())
    recv._get_session = Mock(return_value=session)
    recv._request_sender_data = Mock()
    task = SimpleNamespace(
        _params=SimpleNamespace(ctx_dp_rank=0), _unique_rid=7, slice_id=0, _perf_timer=None
    )
    return recv, task


def test_dispatch_task_payload_none_disables_bounce():
    planner = Mock()
    planner.targets.return_value = PeerTargets(ranks=[0], uniform_writers=True)
    planner.receiver_payload_bytes.return_value = None
    recv, task = _receiver_with(planner)

    recv.dispatch_task(task)

    recv._bounce.reserve.assert_not_called()
    recv._request_sender_data.assert_called_once_with("tcp://s0", b"req")
    assert task.expected_transfers == 1
    # The receiver hands its request description to the planner whole.
    (_self_ri, _peer_ri, req_info), _ = planner.receiver_payload_bytes.call_args
    assert req_info is recv._build_recv_req_info.return_value


# --------------------------------------------------------------------------- #
# Same planner on both sides; the worker takes the model's planner
# --------------------------------------------------------------------------- #
from tensorrt_llm._torch.disaggregation.native.peer import PeerRegistrar  # noqa: E402
from tensorrt_llm._torch.disaggregation.native.rank_info import RankInfo  # noqa: E402
from tensorrt_llm._torch.disaggregation.native.transfer import TransferWorker  # noqa: E402


def _minimal_rank_info(**overrides) -> RankInfo:
    fields = dict(
        instance_name="ctx",
        instance_rank=0,
        tp_size=1,
        tp_rank=0,
        pp_size=1,
        pp_rank=0,
        layer_num_per_pp=[2],
        sender_endpoints=[],
        self_endpoint="",
        transfer_engine_info=b"",
    )
    fields.update(overrides)
    return RankInfo(**fields)


def test_registrar_rejects_planner_id_mismatch():
    self_ri = _minimal_rank_info(planner_id="toy")
    peer_ri = _minimal_rank_info(instance_name="gen", planner_id="trtllm.default")
    reg = PeerRegistrar(self_ri, Mock(), _ToyPlanner(TransferPlan.empty()))

    with pytest.raises(PeerIncompatibleError, match="toy"):
        reg.register("gen", 0, peer_ri)
    with pytest.raises(KeyError):
        reg.get_peer_rank_info("gen", 0)


def test_registrar_rejects_describer_id_mismatch():
    self_ri = _minimal_rank_info(planner_id="toy", describer_id="toy")
    peer_ri = _minimal_rank_info(
        instance_name="gen", planner_id="toy", describer_id="trtllm.default"
    )
    reg = PeerRegistrar(self_ri, Mock(), _ToyPlanner(TransferPlan.empty()))

    with pytest.raises(PeerIncompatibleError, match="describes its cache"):
        reg.register("gen", 0, peer_ri)


def test_receiver_rejects_planner_id_mismatch(monkeypatch):
    recv = Receiver.__new__(Receiver)
    recv._incompatible_peers = {}
    recv._sender_ep_instance_map = {}
    recv._registrar = SimpleNamespace(self_rank_info=_minimal_rank_info(planner_id="toy"))
    recv._planner = _ToyPlanner(TransferPlan.empty())
    sender_info = _minimal_rank_info(
        instance_name="ctx", planner_id="trtllm.default", sender_endpoints=["tcp://ctx:1"]
    )
    messenger = Mock()
    messenger.receive.return_value = [sender_info.to_bytes()]
    monkeypatch.setattr(transfer_mod, "ZMQMessenger", Mock(return_value=messenger))
    recv._get_or_connect_dealer = Mock()
    params = SimpleNamespace(ctx_info_endpoint="tcp://ctx:9")

    with pytest.raises(PeerIncompatibleError, match="trtllm.default"):
        recv._get_sender_info(params)

    recv._get_or_connect_dealer.assert_not_called()  # REGISTER_RANK_INFO never sent
    assert "tcp://ctx:9" in recv._incompatible_peers  # verdict cached
    with pytest.raises(PeerIncompatibleError):  # and fails fast next time
        recv._get_sender_info(params)


def test_worker_uses_model_planner_factory(monkeypatch):
    worker = TransferWorker.__new__(TransferWorker)
    worker._rank_info = _minimal_rank_info()
    made = _ToyPlanner(TransferPlan.empty())
    seen = {}

    def factory(kvm, ri, ext):
        seen["args"] = (kvm, ri, ext)
        return made

    worker._config = SimpleNamespace(planner_factory=factory)
    kvm = Mock()
    kvm.mapping.rank = 1
    extractor = Mock()
    monkeypatch.setattr(transfer_mod, "KVRegionExtractorV1", Mock(return_value=extractor))
    monkeypatch.setattr(transfer_mod, "RankInfoServer", Mock())

    worker._setup_peer_infrastructure(kvm)

    assert worker._peer_registrar.planner is made
    assert seen["args"] == (kvm, worker._rank_info, extractor)
    assert worker._rank_info.planner_id == "toy"
    assert worker._rank_info.planner_blob == b""


def test_worker_defaults_to_default_planner(monkeypatch):
    from tensorrt_llm._torch.disaggregation.native.default_planner import DefaultTransferPlanner

    worker = TransferWorker.__new__(TransferWorker)
    worker._rank_info = _minimal_rank_info()
    worker._config = SimpleNamespace(planner_factory=None)
    kvm = Mock()
    kvm.mapping.rank = 0
    monkeypatch.setattr(transfer_mod, "KVRegionExtractorV1", Mock(return_value=Mock()))
    monkeypatch.setattr(transfer_mod, "RankInfoServer", Mock())

    worker._setup_peer_infrastructure(kvm)

    assert isinstance(worker._peer_registrar.planner, DefaultTransferPlanner)
    assert worker._rank_info.planner_id == DefaultTransferPlanner.planner_id


# --------------------------------------------------------------------------- #
# DefaultTransferPlanner contract pins (from the whole-branch review)
# --------------------------------------------------------------------------- #
from tensorrt_llm._torch.disaggregation.native.default_planner import (  # noqa: E402
    DefaultTransferPlanner,
)


def test_default_planner_statics_survive_a_wrapping_nvtx_range():
    """The planner's static helpers stay static under a wrapping nvtx_range.

    With a profiler attached, nvtx_range wraps the function it decorates. A decorator
    accidentally stacked on a @staticmethod then turns the gate into a bound method and
    every targets() call fails. Import the module with a wrapping nvtx_range (a fresh
    interpreter, so this test cannot disturb the others) and check the statics stay static.
    """
    import subprocess
    import sys

    code = """
import functools
import tensorrt_llm._utils as u
def wrapping_nvtx_range(*_a, **_k):
    def deco(func):
        @functools.wraps(func)
        def inner(*args, **kwargs):
            return func(*args, **kwargs)
        return inner
    return deco
u.nvtx_range = wrapping_nvtx_range
from tensorrt_llm._torch.disaggregation.native.default_planner import DefaultTransferPlanner as P
for name in ("_fanin_bounce_safe", "_pair_rows", "_get_mamba_slot", "_get_buffers_per_layer", "_find_overlap"):
    assert isinstance(P.__dict__[name], staticmethod), name
print("STATICS_OK")
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )
    assert "STATICS_OK" in result.stdout, result.stdout[-2000:] + result.stderr[-4000:]


def _default_planner_pair():
    """A default planner and a compatible peer over the small real page tables test_peer builds."""
    from tensorrt_llm._torch.disaggregation.resource.kv_extractor import KVRegionExtractorV1
    from tests.unittest.disaggregated.test_peer import make_page_table, make_rankinfo

    self_ri = make_rankinfo(instance_name="gen", tp_size=1, tp_rank=0, page_table=make_page_table())
    peer_ri = make_rankinfo(
        instance_name="ctx", instance_rank=0, tp_size=1, tp_rank=0, page_table=make_page_table()
    )
    planner = DefaultTransferPlanner(self_ri, KVRegionExtractorV1(self_ri.page_table))
    return planner, self_ri, peer_ri


def test_default_planner_check_peer_wraps_every_value_error(monkeypatch):
    """Every ValueError raised while vetting a peer becomes PeerIncompatibleError.

    That way the receiver fails only that peer's requests and caches the verdict.
    """
    planner, self_ri, peer_ri = _default_planner_pair()
    # A head-count mismatch is what sends check_peer into pool matching.
    peer_ri.attention.kv_heads_per_rank = 1

    def boom(_peer_ri):
        raise ValueError("pool view spans multiple peer layer groups")

    monkeypatch.setattr(planner, "get_pool_mapping", boom)
    with pytest.raises(PeerIncompatibleError, match="multiple peer layer groups"):
        planner.check_peer(self_ri, peer_ri)


def test_default_planner_targets_is_cached_per_peer_dp_group(monkeypatch):
    """targets() is computed once per (peer instance, DP group).

    The sender asks on every request and the fan-in gate walks both page tables,
    so the answer must be cached.
    """
    planner, self_ri, peer_ri = _default_planner_pair()
    calls = []
    real = DefaultTransferPlanner._fanin_bounce_safe

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    monkeypatch.setattr(planner, "_fanin_bounce_safe", counting)
    first = planner.targets(self_ri, peer_ri, 0)
    second = planner.targets(self_ri, peer_ri, 0)
    assert first == second and first.ranks == [0]
    assert len(calls) == 1
