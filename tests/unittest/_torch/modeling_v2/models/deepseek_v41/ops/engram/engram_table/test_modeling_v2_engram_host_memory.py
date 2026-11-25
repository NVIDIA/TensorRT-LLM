# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host memory of the targets' Engram table, on CPU with the CUDA calls monkeypatched.

The targets' table is the engine's (``modules/engram``), so it allocates through the engine's ``_host_memory``; each
test resolves the helpers through the table class, so it checks what the targets run. Proved here:
- ``TRTLLM_ENGRAM_EVICT_PAGE_CACHE=0`` keeps a NUMA node that is out of memory from having its page cache evicted
  (the default still evicts and retries);
- a device without host NUMA affinity (CUDA reports node -1) gets no NUMA node, so no host-NUMA allocation;
- a host-NUMA allocation that fails with a CUDA error falls back to the default pinned allocator (a private table)
  or to CUDA-registered memory (a node-shared table), instead of failing the load;
- the table carries the marker the loader reads, so a load format other than AUTO is refused up front.
"""

import contextlib
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.ops.engram.engram_table.table import (
    ShardedFp8MultiHeadEmbedding,
)

pytestmark = pytest.mark.cpu_only

_EVICT_ENV = "TRTLLM_ENGRAM_EVICT_PAGE_CACHE"
_NODE = 3


def _table_impl():
    """The module whose helpers the targets' table calls to place its rows in host memory."""
    return sys.modules[ShardedFp8MultiHeadEmbedding._materialize_pinned.__module__]


def _host_memory():
    """The host-memory module behind the targets' table."""
    return sys.modules[_table_impl().host_numa_mem.__module__]


def test_the_targets_table_uses_the_engines_host_memory():
    from tensorrt_llm._torch.modules.engram import _host_memory as engine_host_memory
    from tensorrt_llm._torch.modules.engram import engram as engine_table

    assert _table_impl() is engine_table
    assert _host_memory() is engine_host_memory


@pytest.fixture
def node_out_of_memory(monkeypatch):
    """NUMA node ``_NODE`` cannot hold another allocation; records allocations and evictions."""
    host_memory = _host_memory()
    allocations, evictions = [], []

    def full(size, device, numa_node, shareable=False):
        allocations.append(numa_node)
        raise host_memory.HostNumaOutOfMemoryError(f"NUMA node {numa_node} is full")

    @contextlib.contextmanager
    def lock(mode):
        yield -1, False  # taken without waiting; no flock on sysfs in a test

    def evict(node, lock_fd):
        evictions.append(node)
        return True

    monkeypatch.setattr(host_memory, "HostNumaMem", full)
    monkeypatch.setattr(host_memory, "_eviction_lock", lock)
    monkeypatch.setattr(host_memory, "evict_numa_node_page_cache", evict)
    return host_memory, allocations, evictions


@pytest.mark.parametrize("value", ["0", "false", "FALSE"])
def test_evict_page_cache_opt_out_is_honored(monkeypatch, node_out_of_memory, value):
    host_memory, allocations, evictions = node_out_of_memory
    monkeypatch.setenv(_EVICT_ENV, value)
    with pytest.raises(host_memory.HostNumaOutOfMemoryError):
        host_memory.host_numa_mem(1 << 20, 0, _NODE)
    assert evictions == []
    assert allocations == [_NODE]


def test_out_of_memory_node_is_evicted_by_default(monkeypatch, node_out_of_memory):
    host_memory, allocations, evictions = node_out_of_memory
    monkeypatch.delenv(_EVICT_ENV, raising=False)
    with pytest.raises(host_memory.HostNumaOutOfMemoryError):
        host_memory.host_numa_mem(1 << 20, 0, _NODE)
    assert evictions == [_NODE]
    assert allocations == [_NODE, _NODE]  # retried once after the eviction


@pytest.mark.parametrize(
    "vmm, reported, expected",
    [(1, -1, None), (1, 2, 2), (1, 0, 0), (0, 2, None)],
    ids=["no-affinity", "node-2", "node-0", "no-host-numa-vmm"],
)
def test_host_numa_node_maps_no_affinity_to_none(monkeypatch, vmm, reported, expected):
    host_memory = _host_memory()
    values = {"vmm": vmm, "numa": reported}
    fake = SimpleNamespace(
        CUdevice_attribute=SimpleNamespace(
            CU_DEVICE_ATTRIBUTE_HOST_NUMA_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED="vmm",
            CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID="numa",
        ),
        CUresult=SimpleNamespace(CUDA_SUCCESS=0),
        cuDeviceGetAttribute=lambda attribute, device: (0, values[attribute]),
    )
    monkeypatch.setattr(host_memory, "cuda", fake)
    assert host_memory.host_numa_node(0) == expected


@pytest.fixture
def host_numa_fails(monkeypatch):
    """The device has host NUMA node 0, but every host-NUMA allocation fails with a CUDA error."""
    impl = _table_impl()

    def fail(size, device, numa_node, shareable=False):
        raise RuntimeError("cuMemCreate failed: CUresult.CUDA_ERROR_NOT_SUPPORTED")

    monkeypatch.setattr(impl, "host_numa_node", lambda device: 0)
    monkeypatch.setattr(impl, "host_numa_mem", fail)
    monkeypatch.setattr(impl, "CUASSERT", lambda result: result)
    monkeypatch.setattr(impl, "cudart", SimpleNamespace(cudaSetDevice=lambda device: None))
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    return impl


def test_cuda_error_falls_back_to_the_default_pinned_allocator(monkeypatch, host_numa_fails):
    impl = host_numa_fails
    assert impl._host_numa_table((4, 64), torch.uint8) is None
    # The private table then comes from the default allocator (pageable here: no CUDA to pin with).
    monkeypatch.setattr(impl, "prefer_pinned", lambda: False)
    table = impl._pinned_host_table((4, 64), torch.uint8)
    assert table.device.type == "cpu" and table.shape == (4, 64)


def test_cuda_error_shares_registered_memory_instead(monkeypatch, host_numa_fails):
    impl = host_numa_fails
    registered = []

    def register(size, device, shareable=False, fd=None):
        registered.append((size, device, shareable))
        return SimpleNamespace(size=size)

    monkeypatch.setattr(impl, "_can_use_host_pointer", lambda device: True)
    monkeypatch.setattr(impl, "HostRegisteredMem", register)
    host = impl._shareable_host_mem(1 << 20, 0)
    assert host.size == 1 << 20
    assert registered == [(1 << 20, 0, True)]


def _shared_table(monkeypatch):
    """A node-shared table: its rows stay on the meta device until ``load_weights``, so it builds on CPU."""
    monkeypatch.setattr(_table_impl(), "prefer_pinned", lambda: True)
    return ShardedFp8MultiHeadEmbedding(
        list_of_N=[37, 41, 43, 47],
        D=128,
        block_size=32,
        dtype=torch.bfloat16,
        share_key="/ckpt/model-00002.safetensors:1:2:3:4:layer1",
        mxfp4=True,
    )


def test_table_carries_the_loaders_marker(monkeypatch):
    table = _shared_table(monkeypatch)
    assert table.weight.device.type == "meta" and table.weight.shape == (168, 64)
    assert table._requires_regular_checkpoint_loading is True
    assert not hasattr(table, "_requires_standard_hf_loading")


@pytest.mark.parametrize("load_format", ["DUMMY", "AUTO"])
def test_loader_refuses_a_non_auto_load_of_the_table(monkeypatch, load_format):
    from tensorrt_llm._errors import UnsupportedError
    from tensorrt_llm._torch.pyexecutor.model_loader import _check_post_init_model_compat
    from tensorrt_llm.llmapi.llm_args import LoadFormat

    model = nn.ModuleDict({"table": _shared_table(monkeypatch)})
    loader = SimpleNamespace(checkpoint_format="HF")
    if load_format == "AUTO":
        _check_post_init_model_compat(model, LoadFormat.AUTO, loader)
    else:
        with pytest.raises(UnsupportedError, match="load_format=AUTO"):
            _check_post_init_model_compat(model, LoadFormat[load_format], loader)
