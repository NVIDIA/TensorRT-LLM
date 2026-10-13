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

from contextlib import nullcontext
from types import SimpleNamespace

import cuda.bindings.driver as drv
import pytest

from tensorrt_llm.runtime.kv_cache_manager_v2 import _cuda_virt_mem as virt_mem
from tensorrt_llm.runtime.kv_cache_manager_v2._exceptions import CuError, CuOOMError


def _success() -> drv.CUresult:
    return drv.CUresult.CUDA_SUCCESS


_SUCCESS_CODE = int(_success())
_FABRIC = int(drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_FABRIC)
_NO_HANDLE = int(drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE)


class _LocalizedProbe:
    def __init__(self) -> None:
        self.gdr_supported = False
        self.granularities = {0: 64, 1: 64}
        self.granularity_errors: dict[tuple[int, bool, int], int] = {}
        self.create_errors: dict[tuple[int, bool, int], int] = {}
        self.create_calls: list[tuple[int, int, bool, int, int]] = []
        self.released: list[int] = []
        self._next_handle = 100

    def get_granularity(
        self,
        _handle: object,
        _device_id: int,
        ugpu_id: int,
        handle_types: int,
        gdr: bool,
        _usage: int,
    ) -> tuple[int, int]:
        error = self.granularity_errors.get((handle_types, gdr, ugpu_id), _SUCCESS_CODE)
        granularity = self.granularities[ugpu_id] if error == _SUCCESS_CODE else 0
        return error, granularity

    def create(
        self,
        _handle: object,
        size: int,
        _device_id: int,
        ugpu_id: int,
        handle_types: int,
        gdr: bool,
        usage: int,
    ) -> tuple[int, int]:
        self.create_calls.append((size, handle_types, gdr, usage, ugpu_id))
        error = self.create_errors.get((handle_types, gdr, ugpu_id), _SUCCESS_CODE)
        if error != _SUCCESS_CODE:
            return error, 0
        self._next_handle += 1
        return error, self._next_handle


@pytest.fixture
def localized_probe(monkeypatch: pytest.MonkeyPatch) -> _LocalizedProbe:
    probe = _LocalizedProbe()
    monkeypatch.setattr(
        virt_mem,
        "_query_localized_gdr_support",
        lambda _device_id: probe.gdr_supported,
    )
    monkeypatch.setattr(
        virt_mem,
        "_try_get_localized_allocation_granularity",
        probe.get_granularity,
    )
    monkeypatch.setattr(
        virt_mem,
        "_try_create_localized_allocation_handle",
        probe.create,
    )
    monkeypatch.setattr(
        virt_mem.drv,
        "cuMemRelease",
        lambda handle: probe.released.append(int(handle)) or _success(),
    )
    return probe


def _select_localized_config(size: int = 64) -> virt_mem._AllocationConfig:
    return virt_mem._select_localized_allocation_config(
        size=size,
        device_id=0,
        localization_handle=SimpleNamespace(),
    )


def test_memory_capability_is_independent_and_errors_are_not_hidden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeHandle:
        def supports_ugpu_memory_localization(self) -> bool:
            raise RuntimeError("driver initialization failed")

        def supports_ugpu_localization(self) -> bool:
            return False

    monkeypatch.delenv("TRT_LLM_MOCK_LOCALIZATION_SUPPORT", raising=False)
    monkeypatch.setattr(virt_mem, "get_current_device_id", lambda: 0)
    monkeypatch.setattr(virt_mem, "_get_ugpu_localization_handle", lambda: FakeHandle())

    with pytest.raises(RuntimeError, match="driver initialization failed"):
        virt_mem.is_device_localization_supported(0)


def test_localized_allocation_tuple_is_atomic_across_domains(
    localized_probe: _LocalizedProbe,
) -> None:
    invalid = int(drv.CUresult.CUDA_ERROR_INVALID_VALUE)
    localized_probe.granularities[1] = 128
    localized_probe.create_errors.update(
        {
            (_FABRIC, True, 1): invalid,
            (_NO_HANDLE, True, 1): invalid,
        }
    )

    config = _select_localized_config(size=1 << 40)

    assert config.requested_handle_types == _FABRIC
    assert not config.gpu_direct_rdma_capable
    assert config.usage == 0
    assert config.granularity == 128
    assert all(call[0] == virt_mem._PROBE_SIZE for call in localized_probe.create_calls)
    assert [call[1:] for call in localized_probe.create_calls] == [
        (_FABRIC, True, virt_mem._GPU_DIRECT_RDMA_OVER_PCIE_USAGE, 0),
        (_FABRIC, True, virt_mem._GPU_DIRECT_RDMA_OVER_PCIE_USAGE, 1),
        (_NO_HANDLE, True, virt_mem._GPU_DIRECT_RDMA_OVER_PCIE_USAGE, 0),
        (_NO_HANDLE, True, virt_mem._GPU_DIRECT_RDMA_OVER_PCIE_USAGE, 1),
        (_FABRIC, False, 0, 0),
        (_FABRIC, False, 0, 1),
    ]
    # The first-domain handles from both rejected tuples and both handles from
    # the accepted tuple are all released.
    assert localized_probe.released == [101, 102, 103, 104]


def test_localized_gdr_uses_pcie_usage_when_attribute_is_false(
    localized_probe: _LocalizedProbe,
) -> None:
    config = _select_localized_config()

    assert config.gpu_direct_rdma_capable
    assert config.usage == virt_mem._GPU_DIRECT_RDMA_OVER_PCIE_USAGE
    assert [call[3] for call in localized_probe.create_calls] == [
        virt_mem._GPU_DIRECT_RDMA_OVER_PCIE_USAGE
    ] * 2


def test_localized_probe_propagates_oom_and_releases_prior_domain(
    localized_probe: _LocalizedProbe,
) -> None:
    localized_probe.gdr_supported = True
    localized_probe.create_errors[(_FABRIC, True, 1)] = int(drv.CUresult.CUDA_ERROR_OUT_OF_MEMORY)

    with pytest.raises(CuOOMError):
        _select_localized_config()

    assert localized_probe.released == [101]


def test_localized_probe_propagates_invalid_context(
    localized_probe: _LocalizedProbe,
) -> None:
    localized_probe.gdr_supported = True
    localized_probe.granularity_errors[(_FABRIC, True, 0)] = int(
        drv.CUresult.CUDA_ERROR_INVALID_CONTEXT
    )

    with pytest.raises(CuError) as error:
        _select_localized_config()

    assert error.value.error_code == drv.CUresult.CUDA_ERROR_INVALID_CONTEXT


def test_native_allocator_aligns_size_to_selected_granularity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(virt_mem, "_get_current_context", lambda: drv.CUcontext(0x1234))
    monkeypatch.setattr(virt_mem, "get_current_device_id", lambda: 0)
    config = virt_mem._AllocationConfig(
        requested_handle_types=int(drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE),
        gpu_direct_rdma_capable=False,
        usage=0,
        granularity=256,
    )

    allocator = virt_mem.NativePhysMemAllocator(size=257, allocation_config=config)

    assert allocator.size == 512


class _FakePhysMem:
    def __init__(self, handle: int) -> None:
        self.handle = drv.CUmemGenericAllocationHandle(handle)
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _FakePhysMemAllocator:
    num_ugpus = 1
    phys_mem_size = 64
    device_id = 0
    context = drv.CUcontext(0x1234)

    def __init__(self) -> None:
        self.created: list[_FakePhysMem] = []

    def create(self, ugpu_id: int = 0) -> _FakePhysMem:
        phys_mem = _FakePhysMem(400 + len(self.created))
        self.created.append(phys_mem)
        return phys_mem


@pytest.fixture
def virtual_memory_driver(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    state = SimpleNamespace(
        set_access_results=[],
        reservations=[],
        unmapped=[],
        freed=[],
    )
    monkeypatch.setattr(
        virt_mem,
        "_activate_cuda_context",
        lambda _context, _device_id: nullcontext(),
    )
    monkeypatch.setattr(
        virt_mem.drv,
        "cuMemAddressReserve",
        lambda size, alignment, _address, _flags: (
            state.reservations.append((size, alignment)) or _success(),
            drv.CUdeviceptr(0x100000),
        ),
    )
    monkeypatch.setattr(
        virt_mem.drv,
        "cuMemMap",
        lambda _address, _size, _offset, _handle, _flags: _success(),
    )
    monkeypatch.setattr(
        virt_mem.drv,
        "cuMemSetAccess",
        lambda _address, _size, _access_desc, _count: state.set_access_results.pop(0),
    )
    monkeypatch.setattr(
        virt_mem.drv,
        "cuMemUnmap",
        lambda address, size: state.unmapped.append((int(address), size)) or _success(),
    )
    monkeypatch.setattr(virt_mem.drv, "cuCtxSynchronize", lambda: _success())
    monkeypatch.setattr(
        virt_mem.drv,
        "cuMemAddressFree",
        lambda address, size: state.freed.append((int(address), size)) or _success(),
    )
    return state


def test_set_access_failure_rolls_back_mapping_and_handle(
    virtual_memory_driver: SimpleNamespace,
) -> None:
    virtual_memory_driver.set_access_results.extend(
        [
            _success(),
            drv.CUresult.CUDA_ERROR_INVALID_VALUE,
        ]
    )
    allocator = _FakePhysMemAllocator()
    vm = virt_mem.VirtMem(256, allocator)
    assert virtual_memory_driver.reservations == [(256, 0)]

    with pytest.raises(CuError, match="CUDA driver error"):
        vm.extend(2)

    assert vm.num_phys_mem() == 0
    assert all(phys_mem.closed for phys_mem in allocator.created)
    assert virtual_memory_driver.unmapped == [(0x100040, 64), (0x100000, 64)]

    vm.destroy()
    assert virtual_memory_driver.freed == [(0x100000, 256)]


def test_constructor_failure_frees_exact_reservation_size(
    virtual_memory_driver: SimpleNamespace,
) -> None:
    virtual_memory_driver.set_access_results.append(drv.CUresult.CUDA_ERROR_INVALID_VALUE)
    allocator = _FakePhysMemAllocator()

    with pytest.raises(CuError, match="CUDA driver error"):
        virt_mem.VirtMem(256, allocator, init_num_phys_mem=1)

    assert virtual_memory_driver.reservations == [(256, 0)]
    assert allocator.created[0].closed
    assert virtual_memory_driver.freed == [(0x100000, 256)]


def test_native_release_keeps_tracking_when_driver_release_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allocator = virt_mem.NativePhysMemAllocator.__new__(virt_mem.NativePhysMemAllocator)
    allocator._context = drv.CUcontext(0x1234)
    allocator._device_id = 0
    allocator._outstanding_handles = {501}
    allocator._outstanding_handles_location = {501: virt_mem._UGPU_LOC_UGPU0}
    monkeypatch.setattr(
        virt_mem,
        "_activate_cuda_context",
        lambda _context, _device_id: nullcontext(),
    )
    monkeypatch.setattr(
        virt_mem.drv,
        "cuMemRelease",
        lambda handle: drv.CUresult.CUDA_ERROR_INVALID_CONTEXT,
    )

    with pytest.raises(CuError, match="CUDA driver error"):
        allocator.release(drv.CUmemGenericAllocationHandle(501))

    assert allocator._outstanding_handles == {501}
    assert allocator._outstanding_handles_location == {501: virt_mem._UGPU_LOC_UGPU0}
