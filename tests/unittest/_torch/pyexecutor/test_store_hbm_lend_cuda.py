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
"""The GPU buffers of the HBM lend, on a real GPU."""

import os

import pytest

from tensorrt_llm._torch.pyexecutor import store_hbm_lend as lend_mod

torch = pytest.importorskip("torch")

SIZE = 64 << 20

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture(autouse=True)
def enough_memory():
    # A lent buffer plus two staging tensors, with room for allocator overhead.
    if lend_mod._default_free_memory(torch.cuda.current_device()) < 8 * SIZE:
        pytest.skip(f"needs {8 * SIZE} bytes of free GPU memory")


def _lend_with(allocate, protocol):
    return lend_mod.StoreHbmLend(
        size=SIZE,
        master="127.0.0.1:1",
        protocol=protocol,
        device_name="",
        metadata_server="P2PHANDSHAKE",
        rank=0,
        store_factory=lambda: None,
        allocate=allocate,
        reserve=0,
    )


def _copy(dst, src):
    from cuda.bindings import runtime as cudart

    (error,) = cudart.cudaMemcpy(dst, src, SIZE, cudart.cudaMemcpyKind.cudaMemcpyDeviceToDevice)
    assert error == cudart.cudaError_t.cudaSuccess, error


def _round_trip(pointer, device):
    source = torch.full((SIZE,), 5, dtype=torch.uint8, device=device)
    back = torch.zeros_like(source)
    _copy(pointer, source.data_ptr())
    _copy(back.data_ptr(), pointer)
    torch.cuda.synchronize()
    assert bool((back == 5).all())


def _require_fabric():
    """Skip unless the installed Mooncake and this GPU support FABRIC memory."""
    mooncake = pytest.importorskip("mooncake")
    path = os.path.join(
        os.path.dirname(os.path.abspath(mooncake.__file__)), lend_mod.NVLINK_ALLOCATOR
    )
    if not os.path.exists(path):
        pytest.skip(f"the installed Mooncake has no {lend_mod.NVLINK_ALLOCATOR}")
    from cuda.bindings import driver

    attribute = driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED
    error, supported = driver.cuDeviceGetAttribute(attribute, torch.cuda.current_device())
    assert error == driver.CUresult.CUDA_SUCCESS, error
    if not supported:
        pytest.skip("this GPU does not support FABRIC memory handles")


def test_cuda_buffer_round_trip():
    device = torch.cuda.current_device()
    buffer = lend_mod._default_allocate(SIZE, device, "rdma")
    try:
        assert isinstance(buffer, lend_mod.CudaBuffer) and buffer.pointer
        _round_trip(buffer.pointer, device)
    finally:
        buffer.free()
    assert buffer.pointer == 0


def test_fabric_buffer_round_trip():
    _require_fabric()
    device = torch.cuda.current_device()
    buffer = lend_mod._default_allocate(SIZE, device, "nvlink")
    try:
        assert isinstance(buffer, lend_mod.FabricBuffer) and buffer.pointer
        _round_trip(buffer.pointer, device)
    finally:
        buffer.free()
    assert buffer.pointer == 0
    buffer.free()


@pytest.mark.parametrize("protocol", ["rdma", "nvlink"])
def test_the_real_buffer_is_released_when_the_mount_fails(protocol):
    if protocol == "nvlink":
        _require_fabric()
    taken = []

    def allocate(size, device, proto):
        buffer = lend_mod._default_allocate(size, device, proto)
        taken.append(buffer)
        return buffer

    class Refusing:
        def setup(self, *args):
            return 0

        def mount_buffer(self, pointer, size, protocol="tcp", location="*"):
            return {"ret": -1, "segment_ids": []}

        def close(self):
            return 0

    lend = _lend_with(allocate, protocol)
    lend._store_factory = Refusing
    with pytest.raises(RuntimeError, match="status -1"):
        lend.mount("127.0.0.1")
    (buffer,) = taken
    assert buffer.pointer == 0 and lend.buffer is None
