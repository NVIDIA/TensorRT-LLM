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
"""Ownership of the DLPack block behind create_dlpack_capsule.

Every block must be released exactly once, by whichever side owns it last: the
consumer's deleter once an imported tensor dies, or the CapsuleWrapper for a
capsule that was never imported.
"""

import ctypes
import gc

import pytest
import torch

import tensorrt_llm._dlpack_utils as dlpack_utils

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

_DTYPE = torch.float16
_ELEM_BYTES = 2


# Recording deleters handed to tensors; kept alive for the whole session in case a tensor
# outlives its test.
_KEEPALIVE = []


@pytest.fixture
def freed_blocks(monkeypatch):
    """Addresses of the DLPack blocks released so far, in release order.

    Swaps both release paths -- the consumer's deleter and the wrapper's free -- for
    recording versions that still free the block.
    """
    freed = []
    real_free = dlpack_utils._raw_free

    def wrapper_free(addr):
        freed.append(addr)
        real_free(addr)

    @ctypes.CFUNCTYPE(None, ctypes.POINTER(dlpack_utils.DLManagedTensor))
    def consumer_deleter(dmt_ptr):
        addr = ctypes.cast(dmt_ptr, ctypes.c_void_p).value
        freed.append(addr)
        real_free(addr)

    _KEEPALIVE.append(consumer_deleter)
    monkeypatch.setattr(dlpack_utils.CapsuleWrapper, "_raw_free", staticmethod(wrapper_free))
    monkeypatch.setattr(dlpack_utils, "_free_dlpack_block", consumer_deleter)
    return freed


@pytest.fixture
def buf():
    """Device memory the DLPack tensors view; owned by the test, never by the block."""
    return torch.arange(4 * 64, dtype=_DTYPE, device="cuda")


def _capsule(buf, num_segments=4, segment_elems=64, stride_elems=64):
    return dlpack_utils.create_dlpack_capsule(
        buf.data_ptr(),
        segment_elems * _ELEM_BYTES,
        stride_elems * _ELEM_BYTES,
        num_segments,
        _DTYPE,
        buf.device.index,
    )


def _collect():
    gc.collect()
    # Churn same-sized allocations so a block freed too early gets reused.
    churn = [dlpack_utils._DLPackBlock() for _ in range(1000)]
    del churn
    gc.collect()


def test_unconsumed_capsule_is_freed_by_wrapper(buf, freed_blocks):
    wrapper = _capsule(buf)
    addr = wrapper._block_addr

    del wrapper
    _collect()

    assert freed_blocks == [addr]


def test_consumed_block_outlives_wrapper(buf, freed_blocks):
    wrapper = _capsule(buf)
    addr = wrapper._block_addr
    tensor = torch.utils.dlpack.from_dlpack(wrapper.capsule)

    del wrapper
    _collect()
    # The consumer owns the block now; dropping the wrapper must not free it.
    assert freed_blocks == []
    assert tensor.shape == (4, 64)
    assert tensor.data_ptr() == buf.data_ptr()
    assert torch.equal(tensor.flatten(), buf)

    del tensor
    _collect()
    assert freed_blocks == [addr]


def test_pack_strided_memory_views_outlive_their_base(buf, freed_blocks):
    """The MNNVL caller: views keep the storage alive after the base (and its wrapper) die."""
    num_iters = 16
    views = []
    for _ in range(num_iters):
        base = dlpack_utils.pack_strided_memory(
            buf.data_ptr(), 32 * _ELEM_BYTES, 64 * _ELEM_BYTES, 4, _DTYPE, buf.device.index
        )
        assert base.shape == (4, 32)
        assert base.stride() == (64, 1)
        # Slicing drops the base and with it the _capsule_wrapper attribute.
        views.append(base[1:, 16:])
        del base
        _collect()

    assert freed_blocks == []
    expected = buf.view(4, 64)[1:, 16:32]
    for view in views:
        assert torch.equal(view, expected)

    del view, views
    _collect()
    assert len(freed_blocks) == num_iters
    assert len(set(freed_blocks)) == num_iters


def test_dwdp_tensor_from_ptr_single_segment(buf, freed_blocks):
    """The DWDP caller: one segment with a zero segment stride, then a reshape."""
    vmm = pytest.importorskip("tensorrt_llm._torch.modules.dwdp.vmm")

    tensor = vmm.tensor_from_ptr(buf.data_ptr(), (8, 32), _DTYPE, buf.device.index)
    _collect()

    assert tensor.shape == (8, 32)
    assert tensor.data_ptr() == buf.data_ptr()
    assert torch.equal(tensor.flatten(), buf)
    buf.zero_()
    assert not tensor.any()

    del tensor
    _collect()
    assert len(freed_blocks) == 1
