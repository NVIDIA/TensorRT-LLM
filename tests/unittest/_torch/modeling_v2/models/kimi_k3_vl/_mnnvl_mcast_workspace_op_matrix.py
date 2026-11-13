# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3 family's ``mnnvl_workspace`` on W ranks of one GB200 tray (default 4): the TP-group communicator, a
multicast buffer whose exchanged handles let every rank write into every rank's memory, the agreement, the Lamport
reset and the shared workspace.

    CUDA_VISIBLE_DEVICES=0,1,2,3 python _mnnvl_mcast_workspace_op_matrix.py [--world-size 4] [--fabric-handle on]

Not a pytest module: one fixed sequence of collective steps inside one W-rank job. The collected entry point is
``test_modeling_v2_mnnvl_mcast_workspace_op_matrix.py``.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "comm"))
import _lockstep as ls  # noqa: E402

assert torch.cuda.is_available(), "multicast memory requires CUDA devices"

DEADLINE_S = 600
WORDS = 4096  # int32 words per rank
EMPTY = torch.iinfo(torch.int32).min  # the bits of fp32 -0.0

R = None
mcast = None
COMM = None
FABRIC = None


def check_group_comm():
    global COMM, FABRIC
    from tensorrt_llm._utils import mpi_disabled

    COMM = mcast.group_comm(R.mapping)
    assert not mpi_disabled()
    assert COMM.Get_size() == R.world and COMM.Get_rank() == R.mapping.tp_rank
    assert mcast.device_index(R.mapping) == torch.cuda.current_device()
    FABRIC = mcast.use_fabric_handle(R.mapping) if R.fabric is None else R.fabric
    # The other construction: only the TP group's ranks take part, same ranking.
    tp_comm = mcast.mnnvl_tp_group_comm(R.mapping)
    assert tp_comm.Get_size() == R.world and tp_comm.Get_rank() == R.mapping.tp_rank
    mcast.free_group_comm(tp_comm)


def check_agreement():
    assert mcast.all_ranks_succeeded(COMM, True)
    # One rank fails: every rank learns it.
    assert not mcast.all_ranks_succeeded(COMM, R.rank != R.world - 1)


def check_multicast_buffer():
    handle = mcast.make_mcast_buffer(COMM, WORDS * 4, R.mapping, FABRIC)
    uc = handle.get_uc_buffer(R.mapping.tp_rank, (WORDS,), torch.int32, 0)
    mc = handle.get_mc_buffer((WORDS,), torch.int32, 0)
    assert uc.device.index == torch.cuda.current_device() and uc.numel() == WORDS
    assert mc.numel() == WORDS and mc.data_ptr() != uc.data_ptr()
    uc.fill_(-1)
    R.barrier()
    # Each rank writes its own slice into every rank's memory through that rank's unicast view (the handles were
    # exchanged); every rank then reads every slice from its own memory. The multicast view is written only by
    # multimem instructions, so torch does not touch it.
    share = WORDS // R.world
    for peer in range(R.world):
        view = handle.get_uc_buffer(peer, (WORDS,), torch.int32, 0)
        view[R.rank * share : (R.rank + 1) * share].fill_(R.rank + 7)
    R.barrier()
    want = torch.arange(R.world, device="cuda", dtype=torch.int32).repeat_interleave(share) + 7
    assert torch.equal(uc[: share * R.world], want), "a rank's write did not reach every rank"
    R.barrier()
    del handle, uc, mc, view


def check_arm_lamport_buffers():
    buffer_bytes = 1 << 16
    handle = mcast.make_mcast_buffer(
        COMM, mcast.NUM_LAMPORT_BUFFERS * buffer_bytes, R.mapping, FABRIC
    )
    uc = handle.get_uc_buffer(
        R.mapping.tp_rank, (mcast.NUM_LAMPORT_BUFFERS * buffer_bytes // 4,), torch.float32, 0
    )
    uc.fill_(1.0)
    flags = torch.tensor([5] * 9, dtype=torch.uint32, device="cuda")
    mcast.arm_lamport_buffers(uc, flags, buffer_bytes, COMM)
    assert torch.equal(uc.view(torch.int32), torch.full_like(uc.view(torch.int32), EMPTY))
    want = torch.tensor([0, 2, buffer_bytes, 0, 0, 0, 0, 0, 0], dtype=torch.int32)
    assert torch.equal(flags.view(torch.int32).cpu(), want), flags.view(torch.int32)
    R.barrier()
    del handle, uc


def check_shared_workspace():
    from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce

    ws = mcast.shared_lamport_workspace(R.mapping, torch.bfloat16)
    assert mcast.has_shared_lamport_workspace(R.mapping)
    builtin = MNNVLAllReduce.allreduce_mnnvl_workspaces[R.mapping]
    assert ws.uc_buffer.data_ptr() == builtin["uc_buffer"].data_ptr()
    assert ws.buffer_flags.data_ptr() == builtin["buffer_flags"].data_ptr()
    assert ws.buffer_bytes == builtin["buffer_size_bytes"] >= 1024 * 8192 * 2
    assert ws.world_size == R.world
    view = ws.comm_buffer(torch.bfloat16)
    assert view.shape == (3, ws.buffer_bytes // 2) and view.data_ptr() == ws.uc_buffer.data_ptr()
    # Growing it is collective; the new buffer is again the built-in one.
    grown = mcast.shared_lamport_workspace(R.mapping, torch.bfloat16, ws.buffer_bytes + 1)
    assert grown.buffer_bytes >= ws.buffer_bytes + 1
    assert (
        grown.uc_buffer.data_ptr()
        == MNNVLAllReduce.allreduce_mnnvl_workspaces[R.mapping]["uc_buffer"].data_ptr()
    )


def check_mnnvl_decision():
    from tensorrt_llm._torch.distributed.ops import MNNVLAllReduce

    for requested in (False, True):
        assert mcast.mnnvl_supported(
            R.mapping, torch.bfloat16, requested
        ) == MNNVLAllReduce.is_mnnvl(R.mapping, torch.bfloat16, explicitly_requested=requested)
    # One tray: AUTO does not take MNNVL.
    assert not mcast.mnnvl_supported(R.mapping, torch.bfloat16, False)


def check_free_group_comm():
    mcast.free_group_comm(COMM)


CHECKS = [
    check_group_comm,
    check_agreement,
    check_multicast_buffer,
    check_arm_lamport_buffers,
    check_shared_workspace,
    check_mnnvl_decision,
    check_free_group_comm,
]


def _run_one_rank(args) -> int:
    global R, mcast
    R = ls.Rank(args)
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl import mnnvl_workspace

    mcast = mnnvl_workspace
    return ls.run_checks(R, CHECKS)


if __name__ == "__main__":
    ARGS = ls.parse_args(sys.argv[1:])
    if ARGS.rank_worker or ARGS.launcher == "srun":
        sys.exit(_run_one_rank(ARGS))
    ls.spawn(__file__, ARGS, DEADLINE_S)
    print("OK")
