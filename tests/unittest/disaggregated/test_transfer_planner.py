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


def test_peer_incompatible_error_is_value_error():
    assert issubclass(PeerIncompatibleError, ValueError)
    assert PeerTargets(ranks=[0], uniform_writers=True).ranks == [0]


# --------------------------------------------------------------------------- #
# Sender drives the planner
# --------------------------------------------------------------------------- #
from types import SimpleNamespace  # noqa: E402
from unittest.mock import Mock  # noqa: E402

from tensorrt_llm import DisaggregatedParams  # noqa: E402
from tensorrt_llm._torch.disaggregation.base import CacheKind, Chunk, TokenRange  # noqa: E402
from tensorrt_llm._torch.disaggregation.native.transfer import (  # noqa: E402
    KVSendTask,
    RecvReqInfo,
    Sender,
)


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

    def plan(
        self, self_ri, peer_ri, src, dst, token_range, prompt_len, dst_start_token, excluded
    ):
        self.calls.append((src, dst, token_range, prompt_len, dst_start_token, excluded))
        return self._plan

    def receiver_payload_bytes(self, self_ri, peer_ri, dst):
        return 0


def _sender_with(planner: TransferPlanner) -> Sender:
    peer_ri = SimpleNamespace(
        instance_name="gen",
        instance_rank=3,
        dp_rank=0,
        device_id=1,
        cp_size=1,
        self_endpoint="tcp://gen:1",
        page_table=None,
    )
    sender = Sender.__new__(Sender)
    sender._registrar = Mock()
    sender._registrar.self_rank_info = SimpleNamespace(
        cp_size=1, instance_name="ctx", instance_rank=0
    )
    sender._registrar.get_peer_rank_info.return_value = peer_ri
    sender._planner = planner
    return sender


def _task_and_req():
    task = KVSendTask(
        Chunk(
            block_ids_per_layer_groups=[np.array([5, 6], dtype=np.int64)],
            kind_per_layer_group=[CacheKind.PAGED],
            token_range=TokenRange(0, 64),
            is_last=True,
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


def test_build_kv_write_meta_uses_planner_plan():
    plan = TransferPlan(
        src_ptrs=np.array([100, 200], dtype=np.int64),
        dst_ptrs=np.array([300, 400], dtype=np.int64),
        sizes=np.array([16, 16], dtype=np.int64),
    )
    planner = _ToyPlanner(plan, ranks=(3, 4))
    sender = _sender_with(planner)
    task, req = _task_and_req()

    wm = sender._build_kv_write_meta(task, req)

    assert wm.src_ptrs is plan.src_ptrs and wm.dst_ptrs is plan.dst_ptrs and wm.sizes is plan.sizes
    assert wm.expected_transfers == 2 and task.expected_transfers == 2
    assert wm.dst_device_id == 1 and wm.peer_rank == 3 and wm.peer_endpoint == "tcp://gen:1"
    ((src, dst, token_range, prompt_len, dst_start, excluded),) = planner.calls
    assert src is task._chunk.block_ids_per_layer_groups
    assert dst is req.block_ids_per_layer_groups
    assert excluded is task._chunk.excluded_pool_views
    assert (token_range.start, token_range.end, prompt_len, dst_start) == (0, 64, 64, 32)


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
