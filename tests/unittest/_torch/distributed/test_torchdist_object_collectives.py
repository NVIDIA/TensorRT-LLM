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
"""TorchDist CUDA object collectives: lean transport contract.

The executor hot loop issues ~10 object all-gathers per iteration through
``TorchDist._cuda_allgather_object``; each costs two NCCL kernels and two host
syncs on every rank.  An empty list rides as size 0 and skips the payload
round, ``tp_gather`` unpickles only on the destination rank, and the size /
CPU-tensor staging uses persistent pinned buffers guarded by one lock.  These
tests pin the contract: results identical to a plain pickle round trip, the
NCCL call sequence identical for non-empty payloads on every rank, the staging
touched only under the lock, and ``allgather_ints`` riding exactly the
CPU-tensor transport.

Multi-process, no Ray: run inside a container with >= 2 GPUs as

    TLLM_DISABLE_MPI=1 torchrun --nproc_per_node=2 -m pytest \\
        tests/unittest/_torch/distributed/test_torchdist_object_collectives.py -q

The tests skip when not launched that way.
"""

import datetime
import os
import pickle
from contextlib import nullcontext
from unittest import mock

import pytest
import torch
import torch.distributed as dist

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
_LAUNCHED = (
    WORLD_SIZE >= 2
    and os.environ.get("TLLM_DISABLE_MPI") == "1"
    and torch.cuda.device_count() >= WORLD_SIZE
)

pytestmark = pytest.mark.skipif(
    not _LAUNCHED,
    reason="needs TLLM_DISABLE_MPI=1 torchrun --nproc_per_node>=2 on a multi-GPU node",
)


@pytest.fixture(scope="module")
def world():
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=180))
    yield dist.get_rank(), dist.get_world_size(), local_rank
    dist.barrier()
    dist.destroy_process_group()


def _build_torchdist(world):
    """Construct a real TorchDist under torchrun (no Ray).

    Only the Ray-specific node discovery and the C++ process-group registration
    are stubbed; ``_init_cuda_object_groups`` (group creation, WORLD consensus,
    warm-up collective) runs for real.
    """
    from tensorrt_llm._torch.distributed import communicator
    from tensorrt_llm.mapping import Mapping

    rank, world_size, _ = world
    mapping = Mapping(world_size=world_size, tp_size=world_size, rank=rank)

    def fake_setup_local_comm(self):
        self.local_comm = dist.group.WORLD

    with (
        mock.patch.object(communicator.TorchDist, "setup_local_comm", fake_setup_local_comm),
        mock.patch.object(communicator, "init_pg", lambda *args, **kwargs: None),
    ):
        return communicator.TorchDist(mapping)


@pytest.fixture(scope="module")
def td(world):
    return _build_torchdist(world)


def _pinning_expected():
    from tensorrt_llm._utils import prefer_pinned

    return prefer_pinned()


def _record_nccl(fn):
    """Run ``fn`` and return the ordered NCCL tensor-collective calls it made."""
    calls = []
    orig_into = dist.all_gather_into_tensor
    orig_ag = dist.all_gather

    def rec_into(output, input, *args, **kwargs):
        calls.append(("all_gather_into_tensor", tuple(output.shape), output.dtype))
        return orig_into(output, input, *args, **kwargs)

    def rec_ag(output_list, tensor, *args, **kwargs):
        calls.append(("all_gather", tuple(tensor.shape), tensor.dtype))
        return orig_ag(output_list, tensor, *args, **kwargs)

    with (
        mock.patch.object(dist, "all_gather_into_tensor", rec_into),
        mock.patch.object(dist, "all_gather", rec_ag),
    ):
        result = fn()
    return result, calls


def test_cuda_object_groups_are_created(td):
    assert set(td._cuda_obj_pgs) == {"world", "tp"}
    assert td._cuda_obj_pg("tp") is not None
    assert td._cuda_obj_guard() is td._cuda_obj_lock


def test_fresh_instance_round_trips_empty_list(world):
    fresh = _build_torchdist(world)
    assert fresh.tp_allgather([]) == [[] for _ in range(WORLD_SIZE)]
    assert fresh.tp_allgather([1]) == [[1] for _ in range(WORLD_SIZE)]


def test_all_empty_lists_skip_payload_round(td, world):
    _, world_size, _ = world
    out, calls = _record_nccl(lambda: td.tp_allgather([]))
    assert out == [[] for _ in range(world_size)]
    assert [c[0] for c in calls] == ["all_gather_into_tensor"], calls
    assert calls[0][1] == (world_size,) and calls[0][2] == torch.int64


def test_mixed_empty_and_nonempty_lists(td, world):
    rank, world_size, _ = world

    def payload(r):
        return [] if r == 0 else [("resp", r), {"ids": [r, r + 1]}]

    out, calls = _record_nccl(lambda: td.tp_allgather(payload(rank)))
    assert out == [payload(r) for r in range(world_size)]
    assert [c[0] for c in calls] == ["all_gather_into_tensor"] * 2


@pytest.mark.parametrize("obj", [(), {}, "", range(0), None], ids=type)
def test_empty_non_list_payloads_keep_two_rounds(td, world, obj):
    _, world_size, _ = world
    out, calls = _record_nccl(lambda: td.tp_allgather(obj))
    assert out == [obj] * world_size
    assert len(calls) == 2, "sentinel must be type-exact (list only)"


def test_object_payloads_round_trip_like_pickle(td, world):
    rank, world_size, _ = world

    def payload(r):
        return [
            {"error_ids": list(range(r * 3)), "blocked_ids": [r]},
            r * 7,
            bool(r % 2),
            [bool(r % 2 == 0), 16 + r],
            ("first", {"nested": (r, None, 2.5)}),
        ]

    for idx in range(len(payload(0))):
        out = td.tp_allgather(payload(rank)[idx])
        ref = [pickle.loads(pickle.dumps(payload(r)[idx])) for r in range(world_size)]
        assert out == ref
        assert [type(x) for x in out] == [type(x) for x in ref]


def test_tp_gather_only_destination_unpickles(td, world):
    rank, world_size, _ = world
    payload = [(rank, f"resp{rank}")]
    with mock.patch.object(pickle, "loads", wraps=pickle.loads) as loads:
        out, calls = _record_nccl(lambda: td.tp_gather(payload, dst=0))
    assert [c[0] for c in calls] == ["all_gather_into_tensor"] * 2, "wire protocol unchanged"
    if rank == 0:
        assert out == [[(r, f"resp{r}")] for r in range(world_size)]
        assert loads.call_count == world_size
    else:
        assert out is None
        assert loads.call_count == 0


def test_tp_gather_all_empty(td, world):
    rank, world_size, _ = world
    out, calls = _record_nccl(lambda: td.tp_gather([], dst=0))
    assert len(calls) == 1
    assert out == ([[] for _ in range(world_size)] if rank == 0 else None)


def test_allgather_ints_matches_object_reference(td, world):
    rank, world_size, _ = world

    def vector(r):
        # RankState-shaped: 13 ints incl. zero, negative and > 2**32 values.
        return [r, 5 + r, 0, -3 * r, 2**40 + r] + [r * 10 + i for i in range(8)]

    out = td.allgather_ints(vector(rank))
    assert out == td.tp_allgather(vector(rank))  # object path reference
    assert out == [vector(r) for r in range(world_size)]
    assert all(type(x) is int for row in out for x in row)
    assert all(type(row) is list for row in out)


def test_allgather_ints_same_nccl_sequence_as_tensor_allgather(td, world):
    rank, world_size, _ = world
    values = [rank + i for i in range(13)]
    _, seq_ints = _record_nccl(lambda: td.allgather_ints(values))
    _, seq_tensor = _record_nccl(lambda: td.tp_allgather(torch.tensor(values, dtype=torch.int64)))
    assert seq_ints == seq_tensor
    assert seq_ints == [("all_gather_into_tensor", (world_size, 13), torch.int64)]


def test_allgather_ints_rejects_width_mismatch_in_result(td, world):
    rank, world_size, _ = world
    # Simulate a transport returning rows of unequal width (a rank sending a
    # different count); the helper must not hand back a ragged result.
    ragged = [torch.zeros(13, dtype=torch.int64), torch.zeros(12, dtype=torch.int64)]
    with mock.patch.object(td, "tp_allgather", return_value=ragged):
        with pytest.raises(RuntimeError, match="fixed-width"):
            td.allgather_ints([0] * 13)


# The lock is asserted structurally: a spy on the staging accessor records
# whether ``_cuda_obj_lock`` is held when the shared buffers are touched.  (A
# two-thread race is not a usable oracle here -- the GIL and NCCL's own
# serialisation let it pass with the lock patched out.)
def test_staging_is_touched_only_under_lock(td, world):
    rank, _, _ = world
    seen = []
    orig = td._cuda_obj_staging_for

    def spy(pg):
        seen.append(td._cuda_obj_lock.locked())
        return orig(pg)

    with mock.patch.object(td, "_cuda_obj_staging_for", spy):
        td.tp_allgather({"vote": rank})  # object all-gather
        td.tp_allgather(torch.arange(4, dtype=torch.int64) + rank)  # CPU tensor
        td.tp_gather([rank], dst=0)  # gather semantics
        td.tp_allgather([])  # empty-list sentinel
    assert seen == [True] * 4, seen
    assert not td._cuda_obj_lock.locked()

    # Negative control: with the guard patched out the same spy observes the
    # lock free, so the assertion above has discriminating power.
    seen.clear()
    with (
        mock.patch.object(td, "_cuda_obj_staging_for", spy),
        mock.patch.object(td, "_cuda_obj_guard", lambda: nullcontext()),
    ):
        td.tp_allgather({"vote": rank})
    assert seen == [False], seen


def test_broadcast_and_scalar_allreduce_hold_lock(td, world):
    from tensorrt_llm._torch.distributed.communicator import ReduceOp

    rank, world_size, _ = world
    held = []
    orig_bc, orig_ar = dist.broadcast, dist.all_reduce

    def rec_bc(tensor, *args, **kwargs):
        held.append(("broadcast", td._cuda_obj_lock.locked()))
        return orig_bc(tensor, *args, **kwargs)

    def rec_ar(tensor, *args, **kwargs):
        held.append(("all_reduce", td._cuda_obj_lock.locked()))
        return orig_ar(tensor, *args, **kwargs)

    with (
        mock.patch.object(dist, "broadcast", rec_bc),
        mock.patch.object(dist, "all_reduce", rec_ar),
    ):
        assert td.broadcast({"src": rank}, root=0) == {"src": 0}
        assert td.allreduce(rank + 1, op=ReduceOp.SUM) == world_size * (world_size + 1) // 2
    # Size + payload broadcast, one all-reduce: the CUDA object paths ran,
    # every NCCL call under the lock.
    assert [op for op, _ in held] == ["broadcast", "broadcast", "all_reduce"]
    assert all(locked for _, locked in held), held
    assert not td._cuda_obj_lock.locked()

    # Negative control: without the guard the recorder sees the lock free.
    held.clear()
    with (
        mock.patch.object(dist, "broadcast", rec_bc),
        mock.patch.object(dist, "all_reduce", rec_ar),
        mock.patch.object(td, "_cuda_obj_guard", lambda: nullcontext()),
    ):
        td.broadcast({"src": rank}, root=0)
        td.allreduce(1, op=ReduceOp.MAX)
    assert [locked for _, locked in held] == [False, False, False], held


def test_broadcast_or_none_is_one_round_for_nothing_two_for_payload(td, world):
    rank, _, _ = world
    assert td.supports_single_round_broadcast
    shapes = []
    orig_bc = dist.broadcast

    def rec_bc(tensor, *args, **kwargs):
        shapes.append(tuple(tensor.shape))
        return orig_bc(tensor, *args, **kwargs)

    with mock.patch.object(dist, "broadcast", rec_bc):
        assert td.broadcast_or_none(None if rank == 0 else "ignored", root=0) is None
        assert shapes == [(1,)], shapes  # size exchange only
        shapes.clear()
        obj = {"payload": list(range(200)), "rank": 0}
        assert td.broadcast_or_none(obj if rank == 0 else None, root=0) == obj
        assert len(shapes) == 2 and shapes[0] == (1,), shapes  # size + payload
        shapes.clear()
        # Without the flag None is an ordinary object (pickle never yields 0 bytes).
        assert td.broadcast(None, root=0) is None
        assert len(shapes) == 2, shapes


def test_broadcast_root_stages_asynchronously_through_fenced_pinned_buffers(td, world):
    rank, _, _ = world
    pg = td._cuda_obj_pg("world")
    staging = td._cuda_obj_staging_for(pg)
    for i in range(3):
        obj = {"i": i, "blob": bytes([i]) * (1000 * (i + 1))}
        assert td.broadcast(obj if rank == 0 else None, root=0) == obj
    if rank == 0:
        pinned = staging["bcast_pinned"]
        assert pinned is not None and pinned.numel() >= 3000
        if _pinning_expected():
            assert pinned.is_pinned() and staging["bcast_pinned_size"].is_pinned()
        assert isinstance(staging["bcast_fence"], torch.cuda.Event)
        # Growth replaces the buffer; a smaller payload reuses it.
        td.broadcast({"small": 1}, root=0)
        assert staging["bcast_pinned"].data_ptr() == pinned.data_ptr()
    else:
        td.broadcast(None, root=0)
        assert staging["bcast_pinned"] is None  # non-root never stages
    # The broadcast staging is separate from the allgather staging, so an
    # allgather right after a broadcast cannot rewrite its pinned mirrors.
    assert staging["bcast_size"] is not staging["size_in"]
    assert staging["bcast_pinned_size"] is not staging["pinned_size"]
    assert td.tp_allgather({"vote": rank}) == [{"vote": r} for r in range(pg.size())]


def test_broadcast_root_runs_ahead_of_lagging_peers_without_corrupting_payloads(td, world):
    """The fence exists for the root running ahead: peers join late while the
    root issues back-to-back calls that reuse the same pinned staging (equal
    payload sizes force reuse). Every rank must still see the payloads in
    order -- a fence recorded before the copies, or no fence, corrupts them."""
    import time

    rank, _, _ = world
    payloads = [{"seq": i, "blob": bytes([i]) * 4096} for i in range(3)]
    dist.barrier()
    if rank != 0:
        time.sleep(0.5)
    got = [td.broadcast_or_none(payloads[i] if rank == 0 else None, root=0) for i in range(3)]
    assert got == payloads
    # And the sentinel interleaved with payloads, again with lagging peers.
    dist.barrier()
    if rank != 0:
        time.sleep(0.3)
    seq = [None, payloads[0], None, payloads[1]]
    got = [td.broadcast_or_none(obj if rank == 0 else "peer", root=0) for obj in seq]
    assert got == seq


def test_request_broadcaster_single_round_on_real_torchdist(td, world):
    from unittest.mock import MagicMock

    from tensorrt_llm._torch.pyexecutor.request_utils import RequestBroadcaster

    rank, _, _ = world
    broadcaster = RequestBroadcaster(td, MagicMock())
    assert broadcaster._single_round(prefer_cpu=False)
    assert not broadcaster._single_round(prefer_cpu=True)
    # Empty iteration: one size exchange, ([], None) everywhere.
    shapes = []
    orig_bc = dist.broadcast

    def rec_bc(tensor, *args, **kwargs):
        shapes.append(tuple(tensor.shape))
        return orig_bc(tensor, *args, **kwargs)

    with mock.patch.object(dist, "broadcast", rec_bc):
        assert broadcaster.broadcast([]) == ([], None)
        assert shapes == [(1,)]
        shapes.clear()
        # Busy iteration: picklable RequestQueueItems whose py_* attributes are
        # all None (only py_conversation_params is collected with include_none).
        from types import SimpleNamespace

        from tensorrt_llm._torch.pyexecutor.executor_request_queue import RequestQueueItem

        def item(req_id):
            return RequestQueueItem(
                req_id,
                SimpleNamespace(
                    py_logits_post_processors=None,
                    py_multimodal_data=None,
                    py_scheduling_params=None,
                    py_num_logprobs=None,
                    py_disaggregated_params=None,
                    py_conversation_params=None,
                    py_lora_path=None,
                ),
            )

        items = [item(1), item(2)] if rank == 0 else []
        new_requests, py_objects = broadcaster.broadcast(items)
        assert len(shapes) == 2  # size + payload on every rank
    assert [r.id for r in new_requests] == [1, 2]
    assert py_objects == (("py_conversation_params", {1: None, 2: None}),)


def test_pinned_staging_grows_and_is_reused(td, world):
    rank, world_size, _ = world
    pg = td._cuda_obj_pg("tp")

    def pinned():
        return td._cuda_obj_staging[pg]["pinned"][torch.int64]

    def check(width):
        out = td.tp_allgather(torch.arange(width, dtype=torch.int64) + rank * 100)
        assert len(out) == world_size
        for r, row in enumerate(out):
            assert not row.is_cuda
            assert torch.equal(row, torch.arange(width, dtype=torch.int64) + r * 100)

    check(13)
    b13 = pinned()
    assert b13.numel() >= 13
    if _pinning_expected():
        assert b13.is_pinned()
    check(20)
    b20 = pinned()
    assert b20.numel() >= 20 and b20.data_ptr() != b13.data_ptr()
    check(20)
    assert pinned().data_ptr() == b20.data_ptr(), "equal width must reuse"
    check(13)
    assert pinned().data_ptr() == b20.data_ptr(), "narrower call must not shrink"
    # The size-exchange tensors are persistent too.
    size_in, size_out = td._cuda_obj_staging[pg]["size_in"], td._cuda_obj_staging[pg]["size_out"]
    td.tp_allgather({"vote": rank})
    assert td._cuda_obj_staging[pg]["size_in"] is size_in
    assert td._cuda_obj_staging[pg]["size_out"] is size_out
    assert size_out.numel() == pg.size()


def test_cpu_tensor_allgather_2d_other_dtype(td, world):
    rank, world_size, _ = world
    t = torch.full((2, 3), rank + 1, dtype=torch.int32)
    out = td.tp_allgather(t)
    assert len(out) == world_size
    for r, row in enumerate(out):
        assert row.dtype == torch.int32 and tuple(row.shape) == (2, 3)
        assert torch.equal(row, torch.full((2, 3), r + 1, dtype=torch.int32))
    buf = td._cuda_obj_staging[td._cuda_obj_pg("tp")]["pinned"][torch.int32]
    assert buf.numel() >= 6
    if _pinning_expected():
        assert buf.is_pinned()
