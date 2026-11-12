# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-GPU byte check of the adaptive EPLB peer transport over MNNVL memory.

pytest runs it on four local GPUs; ``torchrun`` runs it across nodes as well
(``python -m torch.distributed.run --nnodes N --nproc-per-node G ... <this file>``), where every
rank also copies from the rank at its place on the next node.
"""

import math
import os
import subprocess
import sys
import threading
from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist

from tensorrt_llm._torch.moe.fused_moe.adaptive_eplb.transport import (
    LayerBlockLayout,
    PeerSlotTransport,
    SlotPlane,
    dedicated_copy_stream,
)

LAYERS = 2
SLOTS = 4
"""Slots per layer and rank: the first half hold the rank's pattern, the second half receive."""
HIDDEN, INTERMEDIATE = 7168, 3072
"""The expert shape of the MegaMoE layout the blocks take."""


def megamoe_planes() -> list[SlotPlane]:
    """The slot planes of the MegaMoE NVFP4 expert layout, as its backend declares them."""
    from tensorrt_llm._torch.moe.kernels.megamoe_scheduler.sami.geometry import BundleLayout

    bundle = BundleLayout.create(hidden=HIDDEN, intermediate=INTERMEDIATE)
    planes = []
    for plane in bundle.planes:
        shape, stride, dtype_name, _, storage_alignment = plane.storage_metadata(SLOTS)
        kernel_alignment = plane.kernel_metadata(SLOTS)[-1]
        alignment = math.lcm(int(plane.alignment), storage_alignment, kernel_alignment)
        dtype = getattr(torch, dtype_name)
        planes.append(SlotPlane(plane.name, tuple(shape), tuple(stride), dtype, alignment))
    return planes


def pattern(rank, layer, plane, rows, row_bytes, device) -> torch.Tensor:
    generator = torch.Generator(device=device)
    generator.manual_seed(1 + (rank * LAYERS + layer) * 16 + plane)
    return torch.randint(
        0, 256, (rows, row_bytes), dtype=torch.uint8, device=device, generator=generator
    )


def peers(rank: int, world: int, per_node: int) -> list[int]:
    """The ranks this rank copies from: the next one on its node, and the one at its place on the
    next node when there are several nodes."""
    node, local = divmod(rank, per_node)
    found = []
    if per_node > 1:
        found.append(node * per_node + (local + 1) % per_node)
    if world > per_node:
        found.append((rank + per_node) % world)
    return found


@dataclass
class Rig:
    """One rank's transport, filled with the source halves of its slots."""

    transport: PeerSlotTransport
    copy_stream: torch.cuda.Stream
    peers: list[int]


def make_rig(rank, world, per_node, device) -> Rig:
    """The transport over ``LAYERS`` blocks of the MegaMoE expert layout.

    The source half of this rank's slots is filled with its pattern.
    """
    from tensorrt_llm.mapping import Mapping

    mapping = Mapping(
        world_size=world, rank=rank, gpus_per_node=per_node, tp_size=world, moe_ep_size=world
    )
    layout = LayerBlockLayout(megamoe_planes(), SLOTS)
    found = peers(rank, world, per_node)
    if not found:
        raise RuntimeError("the check needs at least two GPUs per node or two nodes")
    transport = PeerSlotTransport(mapping)
    for layer in range(LAYERS):
        transport.add_layer(layer, layout)
    half = SLOTS // 2
    for layer in range(LAYERS):
        for plane, rows in enumerate(transport.local_planes(layer).values()):
            rows[:half].copy_(pattern(rank, layer, plane, half, rows.shape[1], device))
    torch.cuda.synchronize()
    dist.barrier()
    stream = dedicated_copy_stream(device)
    assert stream.cuda_stream != torch.cuda.current_stream(device).cuda_stream
    return Rig(transport, stream, found)


def copy_on_a_thread(rig: Rig, copies: list, events: tuple) -> None:
    """Issue the copies as the adaptive runtime's worker thread does: from a thread that never sets
    the CUDA device, waiting on the end event and reading the copy time there."""
    failures = []

    def job() -> None:
        try:
            rig.transport.copy(copies, rig.copy_stream, *events)
            events[1].synchronize()
            events[0].elapsed_time(events[1])
        except BaseException as exc:
            failures.append(exc)

    worker = threading.Thread(target=job)
    worker.start()
    worker.join()
    if failures:
        raise failures[0]


def verify(rig: Rig, device, on_thread: bool = False) -> int:
    """Copy into every destination slot from the peers in one batch, then compare the rows.

    With ``on_thread`` the destinations are cleared first and a worker thread issues the copies.
    Returns the number of rows that differ from their source's pattern.
    """
    half = SLOTS // 2
    copies = [
        (layer, half + j, rig.peers[(layer + j) % len(rig.peers)], j)
        for layer in range(LAYERS)
        for j in range(half)
    ]
    events = (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
    if on_thread:
        for layer in range(LAYERS):
            for rows in rig.transport.local_planes(layer).values():
                rows[half:].zero_()
        torch.cuda.synchronize()
        copy_on_a_thread(rig, copies, events)
    else:
        rig.transport.copy(copies, rig.copy_stream, *events)
    rig.copy_stream.synchronize()
    bad = 0
    for layer, dst, src_rank, src in copies:
        for plane, rows in enumerate(rig.transport.local_planes(layer).values()):
            expected = pattern(src_rank, layer, plane, half, rows.shape[1], device)
            bad += int(not torch.equal(rows[dst], expected[src]))
    return bad


def main() -> int:
    """One rank of the check under torchrun: prints its verdict; 0 when every copied byte matches."""
    os.environ["TLLM_DISABLE_MPI"] = "1"
    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    local_rank, per_node = int(os.environ["LOCAL_RANK"]), int(os.environ["LOCAL_WORLD_SIZE"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(backend="cuda:nccl,cpu:gloo", device_id=device)
    try:
        rig = make_rig(rank, world, per_node, device)
        bad = verify(rig, device)
        dist.barrier()
        bad += verify(rig, device, on_thread=True)
        torch.cuda.synchronize()
        dist.barrier()
    finally:
        dist.destroy_process_group()
    print(f"{'VERIFY OK' if bad == 0 else 'VERIFY FAIL'} rank {rank}: {bad} rows differ")
    return int(bad != 0)


@pytest.mark.gpu4
@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs 4 GPUs")
def test_peer_copies_between_four_local_ranks():
    command = [sys.executable, "-m", "torch.distributed.run", "--standalone"]
    command += ["--nproc-per-node", "4", __file__]
    env = {**os.environ, "TLLM_DISABLE_MPI": "1"}
    done = subprocess.run(command, capture_output=True, text=True, timeout=600, env=env)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
    assert done.stdout.count("VERIFY OK") == 4, done.stdout[-4000:]


if __name__ == "__main__":
    sys.exit(main())
