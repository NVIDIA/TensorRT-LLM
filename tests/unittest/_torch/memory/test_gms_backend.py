# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import pytest
import torch

from tensorrt_llm._torch.memory.gpu_memory_backend import (
    GMSBackend,
    GPUMemoryBackend,
    _ptr_in_gms,
    _storage_nbytes,
)

_GMS_MODULE_NAMES = [
    "gpu_memory_service",
    "gpu_memory_service.client",
    "gpu_memory_service.client.torch",
    "gpu_memory_service.client.torch.allocator",
    "gpu_memory_service.client.torch.module",
    "gpu_memory_service.common",
    "gpu_memory_service.common.locks",
    "gpu_memory_service.common.utils",
    "gpu_memory_service.integrations",
    "gpu_memory_service.integrations.common",
    "gpu_memory_service.integrations.common.patches",
    "gpu_memory_service.integrations.common.utils",
]


def _make_backend(**kwargs):
    with patch("torch.cuda.current_device", return_value=0):
        return GMSBackend(mapping=MagicMock(), socket_path="/tmp/gms.sock", **kwargs)


class TestConstruction:

    @pytest.mark.parametrize("mode", ["rw", "ro", "auto"])
    def test_accepts_valid_modes(self, mode):
        backend = _make_backend(mode=mode)
        assert backend._mode == mode
        assert backend._client is None
        assert backend._is_rw is None

    @pytest.mark.parametrize(
        "bad_mode", ["RW", "Auto", "read", "write", "shadow", "", "rw "])
    def test_rejects_invalid_modes(self, bad_mode):
        with pytest.raises(ValueError, match="GMS mode must be"):
            _make_backend(mode=bad_mode)

    def test_default_tag(self):
        backend = _make_backend()
        assert backend._tag == "weights"
        assert GMSBackend.DEFAULT_TAG == "weights"

    def test_socket_path_none_resolved_lazily(self):
        with patch("torch.cuda.current_device", return_value=0):
            backend = GMSBackend(mapping=MagicMock(), socket_path=None)
        assert backend._socket_path is None


class TestPreConnectState:

    def test_is_rw_is_none(self):
        assert _make_backend().is_rw is None

    def test_total_bytes_is_zero(self):
        assert _make_backend().total_bytes() == 0

    def test_has_committed_weights_returns_false(self):
        assert _make_backend().has_committed_weights() is False

    def test_cleanup_safe_when_never_connected(self):
        _make_backend().cleanup()

    @pytest.mark.parametrize(
        "invoke",
        [
            lambda backend: backend.mem_pool_scope().__enter__(),
            lambda backend: backend.materialize_module(MagicMock()),
            lambda backend: backend.finalize_write(MagicMock()),
            lambda backend: backend.move_untracked_params(MagicMock()),
        ],
    )
    def test_method_raises_when_not_connected(self, invoke):
        with pytest.raises(RuntimeError, match="not connected"):
            invoke(_make_backend())


class TestConnectFailure:

    def test_returns_false_when_upstream_not_installed(self):
        backend = _make_backend()
        with _block_gms():
            assert backend.connect() is False
        assert backend._client is None
        assert backend._is_rw is None

    def test_returns_false_when_upstream_raises(self):
        backend = _make_backend()
        fake_gms = _build_fake_gms_lock_failure(RuntimeError("socket missing"))
        with _install_fake_gms(fake_gms):
            assert backend.connect() is False
        assert backend._client is None


class TestRwOnlyMethodsGated:

    def _ro_backend(self):
        backend = _make_backend()
        backend._client = MagicMock()
        backend._is_rw = False
        return backend

    @pytest.mark.parametrize(
        "invoke",
        [
            lambda backend: backend.mem_pool_scope().__enter__(),
            lambda backend: backend.finalize_write(MagicMock()),
        ],
    )
    def test_method_raises_when_granted_ro(self, invoke):
        with pytest.raises(RuntimeError, match="only valid in RW mode"):
            invoke(self._ro_backend())


class TestFinalizeWrite:

    def test_uses_two_step_gms_api(self):
        backend = _make_backend()
        backend._client = MagicMock(total_bytes=1234)
        backend._is_rw = True
        model = MagicMock()
        fake_gms = _build_fake_gms_success()

        with _install_fake_gms(fake_gms), patch("torch.cuda.synchronize"):
            assert backend.finalize_write(model) == 1234

        fake_gms.client.torch.module.register_module_tensors.assert_called_once_with(
            backend._client, model)
        fake_gms.integrations.common.utils.finalize_gms_write.assert_called_once_with(
            backend._client)
        assert backend._is_rw is False


class TestTotalBytes:

    def test_returns_client_total_bytes(self):
        backend = _make_backend()
        backend._client = MagicMock(total_bytes=5678)
        assert backend.total_bytes() == 5678


class TestPtrInGms:

    def test_returns_false_when_no_mappings_attr(self):
        client = MagicMock(spec=[])
        assert _ptr_in_gms(client, 0xABCDEF) is False

    def test_returns_false_when_mappings_empty(self):
        client = MagicMock()
        client._mappings = {}
        assert _ptr_in_gms(client, 0xABCDEF) is False

    @pytest.mark.parametrize(
        "ptr, expected",
        [
            (0x1080, True),
            (0x1000, True),
            (0x10FF, True),
            (0x1100, False),
            (0x0FFF, False),
        ],
    )
    def test_half_open_interval(self, ptr, expected):
        client = _make_gms_client_with_mapping(va=0x1000, size=0x100)
        assert _ptr_in_gms(client, ptr) is expected


class TestStorageNbytes:

    def test_cpu_tensor(self):
        assert _storage_nbytes(torch.empty(10, dtype=torch.float32)) >= 40

    def test_dtype_dependent(self):
        assert _storage_nbytes(torch.empty(10, dtype=torch.float32)) > _storage_nbytes(
            torch.empty(10, dtype=torch.bfloat16))

    def test_shape_independent_for_views(self):
        base = torch.empty(100, dtype=torch.float32)
        view = base.view(10, 10)
        assert _storage_nbytes(view) == _storage_nbytes(base)


class TestProtocolConformance:

    def test_gms_backend_implements_protocol(self):
        assert isinstance(_make_backend(), GPUMemoryBackend)

    def test_protocol_method_set(self):
        required = {
            "connect",
            "is_rw",
            "has_committed_weights",
            "total_bytes",
            "mem_pool_scope",
            "materialize_module",
            "finalize_write",
            "move_untracked_params",
            "cleanup",
        }
        for method in required:
            assert hasattr(GPUMemoryBackend, method)


def _make_gms_client_with_mapping(va: int, size: int):
    client = MagicMock()
    mapping = MagicMock()
    mapping.va = va
    mapping.size = size
    client._mappings = {"k": mapping}
    return client


@contextmanager
def _block_gms():
    saved = {name: sys.modules.get(name) for name in _GMS_MODULE_NAMES}
    try:
        for name in _GMS_MODULE_NAMES:
            sys.modules[name] = None
        yield
    finally:
        for name, prior in saved.items():
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior


@contextmanager
def _install_fake_gms(fake_pkg):
    saved = {name: sys.modules.get(name) for name in _GMS_MODULE_NAMES}
    try:
        sys.modules["gpu_memory_service"] = fake_pkg
        sys.modules["gpu_memory_service.client"] = fake_pkg.client
        sys.modules["gpu_memory_service.client.torch"] = fake_pkg.client.torch
        sys.modules["gpu_memory_service.client.torch.allocator"] = fake_pkg.client.torch.allocator
        sys.modules["gpu_memory_service.client.torch.module"] = fake_pkg.client.torch.module
        sys.modules["gpu_memory_service.common"] = fake_pkg.common
        sys.modules["gpu_memory_service.common.locks"] = fake_pkg.common.locks
        sys.modules["gpu_memory_service.common.utils"] = fake_pkg.common.utils
        sys.modules["gpu_memory_service.integrations"] = fake_pkg.integrations
        sys.modules["gpu_memory_service.integrations.common"] = fake_pkg.integrations.common
        sys.modules["gpu_memory_service.integrations.common.patches"] = fake_pkg.integrations.common.patches
        sys.modules["gpu_memory_service.integrations.common.utils"] = fake_pkg.integrations.common.utils
        yield
    finally:
        for name, prior in saved.items():
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior


def _build_fake_gms_lock_failure(error):
    fake_pkg = MagicMock(name="gpu_memory_service")
    fake_pkg.client = MagicMock()
    fake_pkg.client.torch = MagicMock()
    fake_pkg.client.torch.allocator = MagicMock()
    fake_pkg.client.torch.allocator.get_or_create_gms_client_memory_manager = MagicMock(
        side_effect=error)
    fake_pkg.client.torch.module = MagicMock()
    fake_pkg.common = MagicMock()
    fake_pkg.common.locks = MagicMock()
    fake_pkg.common.locks.RequestedLockType = MagicMock(RW="rw", RO="ro", RW_OR_RO="rw_or_ro")
    fake_pkg.common.locks.GrantedLockType = MagicMock(RW="rw", RO="ro")
    fake_pkg.common.utils = MagicMock()
    fake_pkg.common.utils.get_socket_path = MagicMock(return_value="/tmp/fake.sock")
    fake_pkg.integrations = MagicMock()
    fake_pkg.integrations.common = MagicMock()
    fake_pkg.integrations.common.patches = MagicMock()
    fake_pkg.integrations.common.patches.patch_empty_cache = MagicMock()
    fake_pkg.integrations.common.utils = MagicMock()
    return fake_pkg


def _build_fake_gms_success():
    fake_pkg = _build_fake_gms_lock_failure(None)
    fake_pkg.client.torch.allocator.get_or_create_gms_client_memory_manager = MagicMock(
        return_value=MagicMock(granted_lock_type="rw"))
    fake_pkg.client.torch.module.register_module_tensors = MagicMock()
    fake_pkg.integrations.common.utils.finalize_gms_write = MagicMock()
    return fake_pkg
