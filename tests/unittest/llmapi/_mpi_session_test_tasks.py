# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Importable tasks for launcher-driven MPI session tests."""

import torch

from tensorrt_llm.executor.ipc import ZeroMqQueue


def receive_logits_processor(addresses: list[tuple[str, bytes]]) -> None:
    """Deserialize and execute a client-defined processor on every MPI rank."""
    from mpi4py import MPI

    rank = MPI.COMM_WORLD.Get_rank()
    queue = ZeroMqQueue(addresses[rank], is_server=False)
    try:
        params = queue.get(timeout=15)
        assert params is not None
        logits = torch.zeros(1, 32)
        params.logits_processor(0, logits, [[]], None, None)
        queue.put((rank, logits.argmax(dim=-1).item()))
    finally:
        queue.close()
