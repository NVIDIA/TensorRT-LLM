# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GMS migration with real CUDA storage and simulated daemon allocations."""

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_llm._torch.memory.gpu_memory_backend import GMSBackend

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture
def migration_backend(monkeypatch):
    allocations = {}
    client = MagicMock()
    client._mappings = {}

    def create_mapping(size: int, tag: str) -> int:
        allocation = torch.zeros(size, dtype=torch.uint8, device="cuda")
        ptr = allocation.data_ptr()
        allocations[ptr] = allocation
        client._mappings[ptr] = SimpleNamespace(va=ptr, size=size)
        return ptr

    def tensor_from_pointer(data_ptr, shape, stride, dtype, device_index) -> torch.Tensor:
        # Match the upstream raw-pointer wrapper: the tensor does not own the allocation.
        storage = torch._C._construct_storage_from_data_pointer(
            data_ptr, torch.device("cuda", device_index), allocations[data_ptr].numel()
        )
        metadata = {
            "size": torch.Size(shape),
            "stride": stride,
            "storage_offset": 0,
            "dtype": dtype,
        }
        return torch._C._construct_CUDA_Tensor_From_Storage_And_Metadata(metadata, storage)

    def iter_tensors(model):
        for name, parameter in model.named_parameters(remove_duplicate=False):
            yield name, parameter, "parameter"
        for name, buffer in model.named_buffers():
            yield name, buffer, "buffer"

    module = ModuleType("gpu_memory_service.client.torch.module")
    module._iter_module_tensors = iter_tensors
    tensor_module = ModuleType("gpu_memory_service.client.torch.tensor")
    tensor_module._tensor_from_pointer = tensor_from_pointer
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setitem(sys.modules, tensor_module.__name__, tensor_module)
    client.create_mapping.side_effect = create_mapping
    backend = GMSBackend(socket_path="/tmp/fake-gms.sock", mapping=MagicMock())
    backend._client = client
    yield backend, allocations
    backend._client = None


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_migrate_shared_storage(migration_backend, dtype) -> None:
    backend, allocations = migration_backend
    source = torch.arange(1, 65, dtype=dtype, device="cuda")
    model = torch.nn.Module()
    model.first = torch.nn.Parameter(source[3:11], requires_grad=False)
    model.second = torch.nn.Parameter(source[20:40:2], requires_grad=False)
    model.overlap = torch.nn.Parameter(source[5:9], requires_grad=False)
    model.bytes_view = torch.nn.Parameter(source.view(torch.uint8)[7:13], requires_grad=False)
    model.tied = model.first
    model.cpu_parameter = torch.nn.Parameter(torch.ones(2), requires_grad=False)
    model.register_buffer("buffer", source)
    expected = {
        name: (parameter.clone(), parameter.stride(), parameter.storage_offset())
        for name, parameter in model.named_parameters()
    }
    original_ptr = source.data_ptr()
    backend.move_untracked_params(model)

    assert len(allocations) == 1
    backend._client.create_mapping.assert_called_once_with(
        size=source.untyped_storage().nbytes(), tag=backend._tag
    )
    mapped = next(iter(allocations.values()))
    # Includes prefix, gaps and tail that no parameter view covers.
    assert torch.equal(mapped, source.view(torch.uint8))
    for name, parameter in model.named_parameters():
        value, stride, offset = expected[name]
        assert torch.equal(parameter, value), name
        assert parameter.stride() == stride, name
        assert parameter.storage_offset() == offset, name
        if parameter.is_cuda:
            assert parameter.untyped_storage().data_ptr() == mapped.data_ptr()
            assert parameter.untyped_storage().nbytes() == mapped.numel()
    assert model.first is model.tied
    assert model.buffer.data_ptr() == original_ptr
    assert model.cpu_parameter.device.type == "cpu"
    model.first[2] = 123
    assert model.overlap[0].item() == 123
    backend.move_untracked_params(model)
    assert backend._client.create_mapping.call_count == 1


def test_reject_zero_byte_cuda_parameter(migration_backend) -> None:
    backend, allocations = migration_backend
    model = torch.nn.Module()
    model.empty_weight = torch.nn.Parameter(torch.empty(0, device="cuda"), requires_grad=False)
    with pytest.raises(
        ValueError, match="GMS cannot register CUDA parameter 'empty_weight'.*zero-byte"
    ):
        backend.move_untracked_params(model)
    backend._client.create_mapping.assert_not_called()
    assert not allocations
