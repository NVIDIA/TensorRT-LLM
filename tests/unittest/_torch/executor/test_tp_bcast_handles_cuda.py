# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Four-rank CUDA IPC correctness and retirement coverage.

Run with ``mpirun -n 4 python -m pytest <this file>`` with at least one GPU.
When multiple GPUs are visible, each rank also copies its view to its own GPU.
"""

import gc
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import test_tp_bcast_handles as transport
import torch


@pytest.mark.skipif(
    transport.COMM.Get_size() != 4 or not torch.cuda.is_available(),
    reason="Requires four MPI ranks and a visible CUDA device",
)
def test_cuda_transport() -> None:
    transport.ROOT_PID = transport.COMM.bcast(os.getpid() if transport.RANK == 0 else None, root=0)
    torch.cuda.set_device(transport.RANK % torch.cuda.device_count())
    original_tensors = transport._tensors
    original_clean = transport._assert_clean

    def clean() -> None:
        gc.collect()
        transport.COMM.Barrier()
        torch.cuda.synchronize(0)
        torch.cuda.ipc_collect()
        transport.COMM.Barrier()
        original_clean()

    with (
        patch.object(transport, "_tensors", lambda: original_tensors("cuda:0")),
        patch.object(transport, "_assert_clean", clean),
    ):
        for handles in (False, True):
            transport._roundtrip(handles, variants=True, delay=True)
        _overlapping_view_fallback()
        transport._abort(False)
        transport._abort(True)
        transport._serialization_abort("cuda:0")
        allocations, fds = [], []
        for cycle in range(20):
            transport._roundtrip(True, variants=True, delay=cycle % 5 == 0)
            gc.collect()
            transport.COMM.Barrier()
            torch.cuda.synchronize(0)
            torch.cuda.ipc_collect()
            allocations.append(torch.cuda.memory_allocated(0))
            fds.append(len(list(Path("/proc/self/fd").iterdir())))
        assert max(allocations[2:]) <= allocations[1], allocations
        assert max(fds[2:]) <= fds[1], fds


def _overlapping_view_fallback() -> None:
    tensor = torch.arange(4, device="cuda:0").expand(4, 4)
    broadcaster = transport.RequestBroadcaster(transport.DIST, transport.Pause())
    requests = transport._request(dict(image=tensor)) if transport.RANK == 0 else []
    requests, objects = broadcaster.broadcast(requests)
    assert torch.equal(dict(objects)["py_multimodal_data"][41]["image"], tensor)
