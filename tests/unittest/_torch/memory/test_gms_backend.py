# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
"""Unit tests for ``GMSBackend`` (``LoadFormat.GMS``).

These tests intentionally do NOT exercise the upstream
``gpu_memory_service`` library. Tests for the import-failure fallback
path mock ``gpu_memory_service.*`` symbols out of ``sys.modules`` so the
assertion is about *our* fallback behavior, not about the upstream API.

The only piece of real CUDA touched by ``GMSBackend.__init__`` is a
single ``torch.cuda.current_device()`` call to record the device index.
We monkeypatch that to a fixed integer for the whole module so every
control-path test (construction, pre-connect, connect-failure, RW/RO
gating, protocol conformance) runs on CPU CI without a GPU.
"""

import sys
from contextlib import contextmanager
from unittest.mock import MagicMock

import pytest
import torch

from tensorrt_llm._torch.memory.gpu_memory_backend import (
    GMSBackend,
    GPUMemoryBackend,
    _ptr_in_gms,
    _storage_nbytes,
)


@pytest.fixture(autouse=True)
def _stub_current_device(monkeypatch):
    """Make ``GMSBackend.__init__`` runnable on CPU CI.

    The only real-CUDA dependency in our backend's ``__init__`` is
    ``torch.cuda.current_device()`` — there are no kernel launches.
    Stubbing it to ``0`` lets every mocked happy/failure connect path
    actually execute under CPU CI.
    """
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """GMSBackend construction-time invariants."""

    @pytest.mark.parametrize("mode", ["rw", "ro", "auto"], ids=["rw", "ro", "auto"])
    def test_accepts_valid_modes(self, mode):
        backend = GMSBackend(socket_path="/tmp/gms.sock", mapping=MagicMock(), mode=mode)
        assert backend._mode == mode
        assert backend._client is None
        assert backend._is_rw is None

    @pytest.mark.parametrize(
        "bad_mode",
        ["RW", "Auto", "read", "write", "shadow", "", "rw ", "auto-detect"],
        ids=[
            "uppercase-rw",
            "mixedcase-auto",
            "read",
            "write",
            "shadow",
            "empty",
            "trailing-space",
            "auto-detect",
        ],
    )
    def test_rejects_invalid_modes(self, bad_mode):
        with pytest.raises(ValueError, match="GMS mode must be"):
            GMSBackend(socket_path="/tmp/gms.sock", mapping=MagicMock(), mode=bad_mode)

    def test_default_tag(self):
        # IMPORTANT: default tag must be ``weights``, NOT ``model_weights``.
        # Matches the GMS library convention (GMS_TAGS upstream).
        backend = GMSBackend(socket_path="/tmp/gms.sock", mapping=MagicMock())
        assert backend._tag == "weights"
        assert GMSBackend.DEFAULT_TAG == "weights"

    def test_custom_tag(self):
        backend = GMSBackend(socket_path="/tmp/gms.sock", mapping=MagicMock(), tag="custom")
        assert backend._tag == "custom"

    def test_socket_path_stored(self):
        backend = GMSBackend(socket_path="/tmp/custom.sock", mapping=MagicMock())
        assert backend._socket_path == "/tmp/custom.sock"

    def test_socket_path_none_resolved_lazily(self):
        # Socket-path resolution to upstream get_socket_path(device, tag)
        # happens inside connect(), not __init__. Construction with
        # socket_path=None must not raise.
        backend = GMSBackend(socket_path=None, mapping=MagicMock())
        assert backend._socket_path is None
        assert backend._client is None


# ---------------------------------------------------------------------------
# Properties before connect()
# ---------------------------------------------------------------------------


class TestPreConnectState:
    def _backend(self):
        return GMSBackend(socket_path="/tmp/gms.sock", mapping=MagicMock())

    def test_is_rw_is_none(self):
        assert self._backend().is_rw is None

    def test_has_committed_weights_returns_false(self):
        # No committed weights when not connected — return False (not raise).
        assert self._backend().has_committed_weights() is False

    def test_cleanup_safe_when_never_connected(self):
        # cleanup() must be idempotent and safe to call before connect().
        self._backend().cleanup()  # must not raise

    @pytest.mark.parametrize(
        "method_name, invoke",
        [
            # invoke(backend) -> trigger; we wrap mem_pool_scope so the
            # context-manager protocol gets exercised inside pytest.raises.
            ("mem_pool_scope", lambda backend: backend.mem_pool_scope().__enter__()),
            ("materialize_module", lambda backend: backend.materialize_module(MagicMock())),
            ("finalize_write", lambda backend: backend.finalize_write(MagicMock())),
            ("move_untracked_params", lambda backend: backend.move_untracked_params(MagicMock())),
        ],
        ids=["mem-pool-scope", "materialize-module", "finalize-write", "move-untracked-params"],
    )
    def test_method_raises_when_not_connected(self, method_name, invoke):
        # Every method that uses the GMS client must guard with a clear
        # "not connected" error before dereferencing self._client.
        with pytest.raises(RuntimeError, match="not connected"):
            invoke(self._backend())


# ---------------------------------------------------------------------------
# connect() — graceful failure on missing upstream library
# ---------------------------------------------------------------------------


class TestConnectFailure:
    def test_returns_false_when_upstream_not_installed(self):
        backend = GMSBackend(socket_path="/tmp/gms.sock", mapping=MagicMock())
        with _block_gms():
            assert backend.connect() is False
        assert backend._client is None
        assert backend._is_rw is None

    def test_returns_false_when_upstream_raises(self):
        # If get_or_create_gms_client_memory_manager raises (e.g. socket
        # doesn't exist), connect() must surface a False (not propagate).
        backend = GMSBackend(socket_path="/tmp/nonexistent.sock", mapping=MagicMock())

        fake_gms = _build_fake_gms_lock_failure(error=RuntimeError("socket missing"))
        with _install_fake_gms(fake_gms):
            assert backend.connect() is False
        assert backend._client is None


# ---------------------------------------------------------------------------
# RW-vs-RO method gating
# ---------------------------------------------------------------------------


class TestRwOnlyMethodsGated:
    """RW-only methods must raise when the granted lock is RO."""

    def _ro_backend(self):
        backend = GMSBackend(socket_path="/tmp/gms.sock", mapping=MagicMock())
        # Simulate "connected, granted RO" without going through real connect().
        backend._client = MagicMock()
        backend._is_rw = False
        return backend

    @pytest.mark.parametrize(
        "method_name, invoke",
        [
            ("mem_pool_scope", lambda backend: backend.mem_pool_scope().__enter__()),
            ("finalize_write", lambda backend: backend.finalize_write(MagicMock())),
        ],
        ids=["mem-pool-scope", "finalize-write"],
    )
    def test_method_raises_when_granted_ro(self, method_name, invoke):
        with pytest.raises(RuntimeError, match="only valid in RW mode"):
            invoke(self._ro_backend())


# ---------------------------------------------------------------------------
# Helper functions: _ptr_in_gms, _storage_nbytes
# ---------------------------------------------------------------------------


def _make_gms_client_with_mapping(va: int, size: int):
    """Build a fake GMS client that exposes one mapping at (va, size)."""
    client = MagicMock()
    mapping = MagicMock()
    mapping.va = va
    mapping.size = size
    client._mappings = {"k": mapping}
    return client


class TestPtrInGms:
    """``_ptr_in_gms`` semantics: half-open intervals + defensive guards."""

    def test_returns_false_when_no_mappings_attr(self):
        # Older GMS releases may not have ``_mappings`` at all.
        client = MagicMock(spec=[])
        assert _ptr_in_gms(client, 0xABCDEF) is False

    def test_returns_false_when_mappings_empty(self):
        client = MagicMock()
        client._mappings = {}
        assert _ptr_in_gms(client, 0xABCDEF) is False

    @pytest.mark.parametrize(
        "label, ptr, expected",
        [
            # Reference mapping is at va=0x1000, size=0x100 (range
            # [0x1000, 0x1100) — half-open).
            ("inside_mapping", 0x1080, True),
            ("at_mapping_start", 0x1000, True),
            ("just_below_end", 0x10FF, True),
            ("at_mapping_end", 0x1100, False),  # half-open
            ("just_below_start", 0x0FFF, False),
            ("far_outside", 0x2000, False),
        ],
        ids=[
            "inside-mapping",
            "at-mapping-start",
            "just-below-end",
            "at-mapping-end",
            "just-below-start",
            "far-outside",
        ],
    )
    def test_half_open_interval(self, label, ptr, expected):
        client = _make_gms_client_with_mapping(va=0x1000, size=0x100)
        assert _ptr_in_gms(client, ptr) is expected, (
            f"case={label}: ptr=0x{ptr:x} expected {expected}"
        )

    @pytest.mark.parametrize(
        "label, va, size",
        [
            ("zero_va_zero_size", 0, 0),
            ("zero_va_nonzero_size", 0, 0x100),
            ("nonzero_va_zero_size", 0x1000, 0),
        ],
        ids=["zero-va-zero-size", "zero-va-nonzero-size", "nonzero-va-zero-size"],
    )
    def test_zero_sentinels_never_match(self, label, va, size):
        # Defensive: a mapping with va=0 or size=0 must never match,
        # even when the queried pointer is 0.
        client = _make_gms_client_with_mapping(va=va, size=size)
        assert _ptr_in_gms(client, 0) is False, f"case={label}"
        assert _ptr_in_gms(client, va) is False, f"case={label}"


class TestStorageNbytes:
    def test_cpu_tensor(self):
        t = torch.empty(10, dtype=torch.float32)
        # 10 * 4 = 40 bytes minimum
        assert _storage_nbytes(t) >= 40

    def test_dtype_dependent(self):
        t_f32 = torch.empty(10, dtype=torch.float32)
        t_bf16 = torch.empty(10, dtype=torch.bfloat16)
        # 10 fp32 occupies more than 10 bf16
        assert _storage_nbytes(t_f32) > _storage_nbytes(t_bf16)

    def test_shape_independent_for_views(self):
        # Storage size should reflect the underlying buffer, not the view.
        base = torch.empty(100, dtype=torch.float32)
        view = base.view(10, 10)
        assert _storage_nbytes(view) == _storage_nbytes(base)


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_gms_backend_implements_protocol(self):
        # GMSBackend instances must satisfy the runtime-checkable
        # GPUMemoryBackend protocol.
        backend = GMSBackend(socket_path="/tmp/gms.sock", mapping=MagicMock())
        assert isinstance(backend, GPUMemoryBackend)

    def test_protocol_method_set(self):
        # The protocol must keep the methods that model_loader.py invokes
        # so a different backend (e.g. CudaIpcBackend) can be plugged in.
        required = {
            "connect",
            "is_rw",
            "has_committed_weights",
            "mem_pool_scope",
            "materialize_module",
            "finalize_write",
            "move_untracked_params",
            "cleanup",
        }
        for method in required:
            assert hasattr(GPUMemoryBackend, method), (
                f"GPUMemoryBackend protocol missing method: {method}"
            )


# ---------------------------------------------------------------------------
# Helpers — fake gpu_memory_service modules and import blockers
# ---------------------------------------------------------------------------


_GMS_MODULE_NAMES = [
    "gpu_memory_service",
    "gpu_memory_service.client",
    "gpu_memory_service.client.torch",
    "gpu_memory_service.client.torch.allocator",
    "gpu_memory_service.common",
    "gpu_memory_service.common.locks",
    "gpu_memory_service.common.utils",
    "gpu_memory_service.integrations",
    "gpu_memory_service.integrations.common",
    "gpu_memory_service.integrations.common.patches",
]


@contextmanager
def _block_gms():
    """Context manager that makes ``import gpu_memory_service.*`` raise."""
    saved = {name: sys.modules.get(name) for name in _GMS_MODULE_NAMES}
    try:
        for name in _GMS_MODULE_NAMES:
            sys.modules[name] = None  # forces ImportError on import
        yield
    finally:
        for name, prior in saved.items():
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior


def _build_fake_gms_lock_failure(*, error):
    """Build a fake gpu_memory_service whose factory raises on call."""
    fake_pkg = MagicMock(name="gpu_memory_service")
    fake_pkg.client = MagicMock(name="gpu_memory_service.client")
    fake_pkg.client.torch = MagicMock(name="gpu_memory_service.client.torch")
    fake_pkg.client.torch.allocator = MagicMock(name="gpu_memory_service.client.torch.allocator")
    fake_pkg.client.torch.allocator.get_or_create_gms_client_memory_manager = MagicMock(
        side_effect=error
    )

    # Lock-type and util modules
    fake_pkg.common = MagicMock()
    fake_pkg.common.locks = MagicMock()
    fake_pkg.common.locks.RequestedLockType = MagicMock(RW="rw", RO="ro", RW_OR_RO="rw_or_ro")
    fake_pkg.common.locks.GrantedLockType = MagicMock(RW="rw", RO="ro")
    fake_pkg.common.utils = MagicMock()
    fake_pkg.common.utils.get_socket_path = MagicMock(return_value="/tmp/fake.sock")

    # patch_empty_cache stub
    fake_pkg.integrations = MagicMock()
    fake_pkg.integrations.common = MagicMock()
    fake_pkg.integrations.common.patches = MagicMock()
    fake_pkg.integrations.common.patches.patch_empty_cache = MagicMock()
    return fake_pkg


@contextmanager
def _install_fake_gms(fake_pkg):
    """Install a fake ``gpu_memory_service`` tree into ``sys.modules``."""
    saved = {name: sys.modules.get(name) for name in _GMS_MODULE_NAMES}
    try:
        sys.modules["gpu_memory_service"] = fake_pkg
        sys.modules["gpu_memory_service.client"] = fake_pkg.client
        sys.modules["gpu_memory_service.client.torch"] = fake_pkg.client.torch
        sys.modules["gpu_memory_service.client.torch.allocator"] = fake_pkg.client.torch.allocator
        sys.modules["gpu_memory_service.common"] = fake_pkg.common
        sys.modules["gpu_memory_service.common.locks"] = fake_pkg.common.locks
        sys.modules["gpu_memory_service.common.utils"] = fake_pkg.common.utils
        sys.modules["gpu_memory_service.integrations"] = fake_pkg.integrations
        sys.modules["gpu_memory_service.integrations.common"] = fake_pkg.integrations.common
        sys.modules["gpu_memory_service.integrations.common.patches"] = (
            fake_pkg.integrations.common.patches
        )
        yield
    finally:
        for name, prior in saved.items():
            if prior is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = prior
