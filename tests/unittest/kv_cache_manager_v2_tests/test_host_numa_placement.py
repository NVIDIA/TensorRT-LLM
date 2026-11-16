# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Host-tier pages must land on the NUMA node of the GPU that will read them.

Only the mmap host backing places pages with ``mbind``; the VMM backing names
its node to the driver instead.  The backing is chosen from device attributes,
so this is meaningful only where mmap is selected -- Grace-class parts, where
the GPU reaches host memory through the CPU's page tables over a coherent link.
Elsewhere the probe cannot answer and these cases skip.

``mbind`` is not the only thing placing pages.  Where it cannot be applied the
range keeps ``MPOL_DEFAULT`` and each page is faulted onto the node of the CPU
that first touches it; the fill worker binds itself to the GPU's node, so the
common path still places correctly.  ``mbind`` is what makes placement hold for
a page faulted by some *other* thread.  This matters for running the test: the
memory-policy syscalls sit behind ``CAP_SYS_NICE`` under Docker's default
seccomp profile, so a container without it both loses ``mbind`` and cannot read
placement back -- the probe returns ``None`` and these cases skip.

The test pins itself to one NUMA node and allocates against a GPU on another.
Without that, first touch would place the pages on the right node by accident
and the cases could not fail.  Restoring the affinity is left to a fixture, and
the test is additionally expected to run isolated, so the change cannot leak
into whatever runs next in the same session.
"""

from __future__ import annotations

import os
from importlib.util import find_spec
from typing import TYPE_CHECKING

import pytest
import torch

if not TYPE_CHECKING and find_spec("kv_cache_manager_v2") is not None:
    from kv_cache_manager_v2 import _introspection
else:
    from tensorrt_llm.runtime.kv_cache_manager_v2 import _introspection

# Large enough that the fill commits several pages, small enough to stay cheap.
_PROBE_BYTES = 8 << 20


def _gpu_numa_node(device: int) -> int:
    """NUMA node a GPU attaches to, as the kernel reports it.

    ``get_device_properties`` exposes the PCI address as three integers, which
    sysfs wants formatted as ``domain:bus:device.function`` with a fixed width.
    Returns -1 when the platform reports no affinity.
    """
    props = torch.cuda.get_device_properties(device)
    bdf = f"{props.pci_domain_id:04x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
    try:
        with open(f"/sys/bus/pci/devices/{bdf}/numa_node", encoding="ascii") as handle:
            return int(handle.read().strip())
    except (OSError, ValueError):
        return -1


def _find_gpu_pair() -> tuple[int, int, int]:
    """A (local_node, remote_gpu, remote_node) triple on different NUMA nodes."""
    nodes = {device: _gpu_numa_node(device) for device in range(torch.cuda.device_count())}
    known = {device: node for device, node in nodes.items() if node >= 0}
    if not known:
        pytest.skip("no GPU reports a NUMA affinity")
    first_node = next(iter(known.values()))
    for device, node in known.items():
        if node != first_node:
            return first_node, device, node
    pytest.skip(f"every GPU is on NUMA node {first_node}; nothing to distinguish")


def _cpus_on_node(node: int) -> set[int]:
    try:
        with open(f"/sys/devices/system/node/node{node}/cpulist", encoding="ascii") as handle:
            spec = handle.read().strip()
    except OSError:
        return set()
    cpus: set[int] = set()
    for part in spec.split(","):
        if "-" in part:
            lo, hi = part.split("-")
            cpus.update(range(int(lo), int(hi) + 1))
        elif part:
            cpus.add(int(part))
    return cpus


@pytest.fixture
def restore_process_state():
    """Puts the process back where it was, however the test leaves.

    Covers the CUDA device as well as CPU affinity: both are process-wide, and a
    later test in the same session would otherwise inherit them.
    """
    saved_affinity = os.sched_getaffinity(0)
    saved_device = torch.cuda.current_device() if torch.cuda.is_available() else None
    yield
    os.sched_setaffinity(0, saved_affinity)
    if saved_device is not None:
        torch.cuda.set_device(saved_device)


@pytest.mark.parametrize(
    "allow_remote_numa_fallback", [True, False], ids=["remote_fallback", "strict"]
)
def test_host_pages_land_on_the_gpus_numa_node(restore_process_state, allow_remote_numa_fallback):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    local_node, remote_gpu, remote_node = _find_gpu_pair()

    # A restricted cpuset can exclude every CPU on that node; asking for one
    # outside the permitted mask fails with EINVAL rather than skipping.
    local_cpus = _cpus_on_node(local_node) & os.sched_getaffinity(0)
    if not local_cpus:
        pytest.skip(f"no permitted CPU on NUMA node {local_node}")
    # Fault from the other node, so an unbound range would land there.
    os.sched_setaffinity(0, local_cpus)

    torch.cuda.set_device(remote_gpu)
    # The probe reads device attributes and will not create a context itself.
    torch.zeros(1, device=f"cuda:{remote_gpu}")

    placement = _introspection.probe_host_mem_placement(
        size=_PROBE_BYTES,
        allow_remote_numa_fallback=allow_remote_numa_fallback,
    )
    if placement is None:
        pytest.skip(
            "host pages are not placed by policy here: the VMM backing is selected, or libnuma is absent"
        )

    assert placement.gpu_numa_node == remote_node, (
        f"probe reports GPU on node {placement.gpu_numa_node}, sysfs says {remote_node}"
    )

    counts = placement.node_page_counts
    assert counts, "no resident pages reported for the mapping"
    # Placement is asserted whether or not mbind applied: the commit runs on a
    # worker bound to the GPU's node, so first touch lands the pages there even
    # when the policy could not be set.
    assert set(counts) == {remote_node}, (
        f"pages resident on nodes {sorted(counts)}; expected all on node {remote_node}, "
        f"which GPU {remote_gpu} attaches to (counts: {counts})"
    )

    if placement.strict_binding is None:
        # get_mempolicy is gated with mbind, so an unreadable policy means the
        # placement above came from first touch alone -- still correct, but the
        # flag itself cannot be checked here.
        # First touch prefers the local node but takes a remote page rather than
        # failing, so it reproduces allow_remote_numa_fallback=True only. The
        # strict setting is simply not enforceable here.
        unenforceable = (
            "" if allow_remote_numa_fallback else " (strict binding is not enforceable without it)"
        )
        pytest.skip(
            f"placement verified on node {remote_node} via first touch; policy readback needs "
            f"CAP_SYS_NICE, so allow_remote_numa_fallback is unchecked{unenforceable}"
        )
    # The flag picks between a strict binding and a preference. Both place the
    # same way while the node has room, so the mode is what distinguishes them.
    assert placement.strict_binding is (not allow_remote_numa_fallback)
