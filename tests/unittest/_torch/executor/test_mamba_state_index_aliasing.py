# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""State indices reaching the device once instead of twice per step.

prepare_resources() already stages every scheduled request's recurrent-state
block index into a device buffer, so Mamba2Metadata.prepare() can read that
buffer in place rather than staging the same values again. The buffer is only
usable when the batch is asked for in the order it was staged in, and it is
offered through get_state_indices_device() rather than get_state_indices(),
whose per-request list of Python ints every other caller depends on.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modules.mamba.mamba2_metadata import Mamba2Metadata
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID
from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import MambaHybridCacheManagerV2

skip_no_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for the metadata buffers",
)

CAPACITY = 5

device_view = MambaHybridCacheManagerV2.get_state_indices_device
host_indices = MambaHybridCacheManagerV2.get_state_indices


def _manager_stub(order, values, dummies=()):
    """A stand-in carrying only what the two accessors read."""
    refreshed = []
    buffer = torch.zeros(CAPACITY, dtype=torch.int32)
    buffer[: len(values)] = torch.tensor(values, dtype=torch.int32)
    stub = SimpleNamespace(
        local_num_mamba_layers=2,
        _state_index_request_ids=list(order),
        _request_id_to_state_index=dict(zip(order, values)),
        _request_id_to_is_dummy={rid: rid in dummies for rid in order},
        cuda_state_indices=buffer,
        _refresh_dummy_request_mask=refreshed.append,
        refreshed=refreshed,
    )
    stub._dummy_flags_for = MambaHybridCacheManagerV2._dummy_flags_for.__get__(stub)
    return stub


@pytest.mark.parametrize("batch", [1, 2, 3])
def test_device_view_is_one_entry_per_request_id(batch):
    # The device buffer is capacity-sized, not batch-sized; a caller reads one
    # entry per request either way.
    order = list(range(batch))
    mgr = _manager_stub(order=order, values=[4, 1, 6][:batch])

    indices = device_view(mgr, order, [False] * batch)

    assert len(indices) == batch
    assert indices.tolist() == [4, 1, 6][:batch]
    # A view from element 0, so the alias keeps the buffer's identity.
    assert indices.data_ptr() == mgr.cuda_state_indices.data_ptr()


def test_host_accessor_still_returns_python_ints():
    # The contract every other caller depends on: one Python int per request
    # id, in the caller's order, comparable against a plain list.
    mgr = _manager_stub(order=[7, 8], values=[4, 1])

    indices = host_indices(mgr, [7, 8], [False, False])

    assert indices == [4, 1]
    assert all(isinstance(index, int) for index in indices)


def test_no_device_view_when_the_batch_was_reordered():
    # Disagg serving sorts generation_requests by py_batch_idx after
    # prepare_resources ran, so the device order no longer describes the batch.
    mgr = _manager_stub(order=[7, 8, 9], values=[4, 1, 6])

    assert device_view(mgr, [9, 7, 8], [False] * 3) is None
    assert host_indices(mgr, [9, 7, 8], [False] * 3) == [6, 4, 1]


def test_no_device_view_when_the_batch_was_padded():
    # CUDA graph padding appends dummy requests, so the staged order is a
    # different length from the one being asked about.
    mgr = _manager_stub(
        order=[7, CUDA_GRAPH_DUMMY_REQUEST_ID],
        values=[4, 0],
        dummies=(CUDA_GRAPH_DUMMY_REQUEST_ID,),
    )

    assert device_view(mgr, [7], [False]) is None
    assert host_indices(mgr, [7], [False]) == [4]


def test_no_device_view_before_construction_finishes():
    # Several of this manager's tests build it with object.__new__ and set
    # only the attributes the accessors read, so the staged order has to have
    # a class-level default rather than existing only after __init__.
    mgr = object.__new__(MambaHybridCacheManagerV2)
    mgr.local_num_mamba_layers = 2

    assert mgr.get_state_indices_device([7, 8], [False, False]) is None


def test_device_view_refreshes_the_dummy_mask_like_the_host_accessor():
    padded = _manager_stub(order=[7, 8], values=[4, 1])
    plain = _manager_stub(order=[7, 8], values=[4, 1])

    device_view(padded, [7, 8], [False, True])
    host_indices(plain, [7, 8], [False, True])

    assert padded.refreshed == [[False, True]]
    assert padded.refreshed == plain.refreshed


def _decode_attn_metadata(kv_cache_manager, request_ids, batch_size):
    seq_lens = torch.ones(batch_size, dtype=torch.int)
    return SimpleNamespace(
        seq_lens=seq_lens,
        seq_lens_cuda=seq_lens.cuda(),
        num_contexts=0,
        num_ctx_tokens=0,
        kv_cache_manager=kv_cache_manager,
        request_ids=request_ids,
        kv_cache_params=SimpleNamespace(
            num_cached_tokens_per_seq=torch.full((batch_size,), 8, dtype=torch.int),
        ),
    )


class _DeviceManager:
    """Offers a device view, as the V2 manager does for a stable batch."""

    def __init__(self, values, capacity=CAPACITY):
        self.buffer = torch.zeros(capacity, dtype=torch.int32, device="cuda")
        self.buffer[: len(values)] = torch.tensor(values, dtype=torch.int32, device="cuda")
        self.host_calls = 0

    def get_state_indices_device(self, request_ids, is_padding=None):
        return self.buffer[: len(request_ids)]

    def get_state_indices(self, request_ids, is_padding=None):
        self.host_calls += 1
        return self.buffer[: len(request_ids)].tolist()


class _HostOnlyManager:
    """A manager without the device accessor, i.e. every other manager."""

    def __init__(self, values):
        self.values = list(values)

    def get_state_indices(self, request_ids, is_padding=None):
        return self.values


@skip_no_cuda
def test_prepare_aliases_the_managers_device_buffer():
    manager = _DeviceManager([3, 1])
    metadata = Mamba2Metadata(max_batch_size=4, chunk_size=8)

    metadata.prepare(_decode_attn_metadata(manager, [11, 12], 2))

    # Aliased, not restaged: no copy into the metadata's own buffer, and the
    # host accessor is not consulted at all.
    assert metadata.state_indices.data_ptr() == manager.buffer.data_ptr()
    assert manager.host_calls == 0
    torch.testing.assert_close(
        metadata.state_indices_long, torch.tensor([3, 1], dtype=torch.long, device="cuda")
    )


@skip_no_cuda
def test_prepare_falls_back_for_a_manager_without_the_device_accessor():
    metadata = Mamba2Metadata(max_batch_size=4, chunk_size=8)

    metadata.prepare(_decode_attn_metadata(_HostOnlyManager([5, 2]), [21, 22], 2))

    torch.testing.assert_close(
        metadata.state_indices[:2], torch.tensor([5, 2], dtype=torch.int32).cuda()
    )


@skip_no_cuda
def test_prepare_writes_its_own_buffer_again_after_aliasing():
    manager = _DeviceManager([3, 1])
    metadata = Mamba2Metadata(max_batch_size=4, chunk_size=8)
    metadata.prepare(_decode_attn_metadata(manager, [11, 12], 2))
    assert metadata.state_indices.data_ptr() == manager.buffer.data_ptr()

    # A later batch the manager cannot describe on device must not be staged
    # through the alias, which would corrupt the manager's buffer.
    metadata.prepare(_decode_attn_metadata(_HostOnlyManager([5, 2]), [21, 22], 2))

    assert metadata.state_indices.data_ptr() != manager.buffer.data_ptr()
    torch.testing.assert_close(manager.buffer[:2], torch.tensor([3, 1], dtype=torch.int32).cuda())
    torch.testing.assert_close(
        metadata.state_indices[:2], torch.tensor([5, 2], dtype=torch.int32).cuda()
    )
    torch.testing.assert_close(
        metadata.state_indices_long, torch.tensor([5, 2], dtype=torch.long, device="cuda")
    )


@skip_no_cuda
def test_prepare_rejects_a_manager_buffer_that_moves():
    manager = _DeviceManager([3, 1])
    metadata = Mamba2Metadata(max_batch_size=4, chunk_size=8)
    metadata.prepare(_decode_attn_metadata(manager, [11, 12], 2))

    # A reallocated buffer would leave captured kernels reading the old
    # address, so it has to be caught rather than silently aliased.
    manager.buffer = torch.zeros(CAPACITY, dtype=torch.int32, device="cuda")
    with pytest.raises(AssertionError, match="stable data pointer"):
        metadata.prepare(_decode_attn_metadata(manager, [11, 12], 2))


@skip_no_cuda
def test_prepare_keeps_the_alias_stable_across_batch_sizes():
    # A shrinking batch hands back a shorter view of the same buffer; the
    # data-pointer stability assert must not read that as a reallocation.
    manager = _DeviceManager([3, 1, 2])
    metadata = Mamba2Metadata(max_batch_size=4, chunk_size=8)

    metadata.prepare(_decode_attn_metadata(manager, [11, 12, 13], 3))
    metadata.prepare(_decode_attn_metadata(manager, [11, 12], 2))

    assert metadata.state_indices.data_ptr() == manager.buffer.data_ptr()
    torch.testing.assert_close(
        metadata.state_indices_long, torch.tensor([3, 1], dtype=torch.long, device="cuda")
    )
