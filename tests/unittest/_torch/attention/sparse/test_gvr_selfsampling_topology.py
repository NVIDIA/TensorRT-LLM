# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only topology policy tests for self-sampling GVR prefill."""

import importlib.util
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_HOST_PATH = (
    Path(__file__).resolve().parents[5]
    / "tensorrt_llm/_torch/cute_dsl_kernels/blackwell/top_k/gvr_topk_decode_self_sampling_host.py"
)
_HOST_SPEC = importlib.util.spec_from_file_location("_gvr_selfsampling_topology_host", _HOST_PATH)
assert _HOST_SPEC is not None and _HOST_SPEC.loader is not None
ss_host = importlib.util.module_from_spec(_HOST_SPEC)
sys.modules[_HOST_SPEC.name] = ss_host
_HOST_SPEC.loader.exec_module(ss_host)


def _topology(
    active_num_sms: int,
    *,
    total_num_sms: int | None = None,
    locality_domain_id: int | None = None,
) -> ss_host._PrefillTopology:
    return ss_host._PrefillTopology(
        locality_domain_id,
        active_num_sms,
        total_num_sms or active_num_sms,
    )


def test_prefill_b200_tier_parity() -> None:
    """The parameterized policy reproduces the original 148-SM bands."""
    topology = _topology(148)
    assert ss_host._prefill_tier_rows(topology) == (75, 149, 297)
    assert [ss_host._prefill_tier(rows, topology) for rows in (1, 148, 149, 296, 297)] == [
        0,
        0,
        1,
        1,
        2,
    ]

    plans = [
        ss_host.route_streaming(
            rows,
            32768,
            32768,
            512,
            force_main=True,
            num_sms=topology.active_num_sms,
            force_r_one=True,
        )
        for rows in ss_host._prefill_tier_rows(topology)
    ]
    assert [plan["rt"]["R"] for plan in plans] == [1, 1, 1]
    assert [plan["block"] for plan in plans] == [1024, 512, 256]


@pytest.mark.parametrize("active_num_sms", [106, 212])
def test_prefill_tiers_follow_active_sm_count(active_num_sms: int) -> None:
    topology = _topology(
        active_num_sms,
        total_num_sms=212,
        locality_domain_id=0 if active_num_sms < 212 else None,
    )
    assert ss_host._prefill_tier(active_num_sms, topology) == 0
    assert ss_host._prefill_tier(active_num_sms + 1, topology) == 1
    assert ss_host._prefill_tier(2 * active_num_sms, topology) == 1
    assert ss_host._prefill_tier(2 * active_num_sms + 1, topology) == 2

    plans = [
        ss_host.route_streaming(
            rows,
            262144,
            262144,
            1024,
            force_main=True,
            num_sms=active_num_sms,
            force_r_one=True,
        )
        for rows in ss_host._prefill_tier_rows(topology)
    ]
    assert [plan["rt"]["R"] for plan in plans] == [1, 1, 1]
    assert [plan["block"] for plan in plans] == [1024, 512, 256]


def test_prefill_topology_keeps_b200_on_full_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(multi_processor_count=148, major=10, minor=0),
    )
    ss_host._DEVICE_COMPUTE_INFO.clear()

    assert ss_host._prefill_topology(0) == _topology(148)


def test_prefill_topology_uses_full_device_outside_locality_domain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(multi_processor_count=212, major=10, minor=7),
    )
    monkeypatch.setattr(ss_host, "_current_locality_domain", lambda: None)
    ss_host._DEVICE_COMPUTE_INFO.clear()

    assert ss_host._prefill_topology(0) == _topology(212)


def test_prefill_topology_uses_current_locality_partition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device_events: list[tuple[str, int]] = []

    @contextmanager
    def device_context(device: int) -> Iterator[None]:
        device_events.append(("enter", device))
        try:
            yield
        finally:
            device_events.append(("exit", device))

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(multi_processor_count=212, major=10, minor=7),
    )
    monkeypatch.setattr(torch.cuda, "device", device_context)
    monkeypatch.setattr(ss_host, "_current_locality_domain", lambda: 1)

    def get_topology() -> tuple[tuple[int, int], ...]:
        assert device_events == [("enter", 0)]
        return ((106, 212), (106, 212))

    monkeypatch.setattr(ss_host, "_locality_domain_topology", get_topology)
    ss_host._DEVICE_COMPUTE_INFO.clear()

    assert ss_host._prefill_topology(0) == _topology(
        106,
        total_num_sms=212,
        locality_domain_id=1,
    )
    assert device_events == [("enter", 0), ("exit", 0)]


def test_prefill_topology_rejects_a_different_device_total(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @contextmanager
    def device_context(_device: int) -> Iterator[None]:
        yield

    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _device: SimpleNamespace(multi_processor_count=212, major=10, minor=7),
    )
    monkeypatch.setattr(torch.cuda, "device", device_context)
    monkeypatch.setattr(ss_host, "_current_locality_domain", lambda: 0)
    monkeypatch.setattr(ss_host, "_locality_domain_topology", lambda: ((112, 224), (112, 224)))
    ss_host._DEVICE_COMPUTE_INFO.clear()

    with pytest.raises(RuntimeError, match="topology.*target device"):
        ss_host._prefill_topology(0)


def test_prefill_cache_key_includes_topology_identity() -> None:
    full = _topology(212)
    partition0 = _topology(106, total_num_sms=212, locality_domain_id=0)
    partition1 = _topology(106, total_num_sms=212, locality_domain_id=1)
    keys = {
        ss_host._prefill_cache_key(topology, 0, 512, 32768)
        for topology in (full, partition0, partition1)
    }
    assert len(keys) == 3


def test_prefill_launcher_cache_is_topology_specific(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakeDevice:
        def get_compiled(self, tpl: tuple, **_kwargs: object) -> tuple:
            return tpl

    monkeypatch.setattr(ss_host, "_PREFILL_CACHE", {})
    monkeypatch.setattr(ss_host, "_device", lambda: _FakeDevice())
    full = _topology(212)
    partition = _topology(106, total_num_sms=212, locality_domain_id=0)

    full_launcher = ss_host._prefill_launcher(full, 1, 512, 32768)
    partition_launcher = ss_host._prefill_launcher(partition, 1, 512, 32768)

    assert len(ss_host._PREFILL_CACHE) == 2
    assert full_launcher[1] == partition_launcher[1]


def test_prefill_capture_cache_miss_precedes_workspace_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _FakeTensor:
        def __init__(
            self,
            shape: tuple[int, ...],
            dtype: object,
            *,
            strides: tuple[int, ...] | None = None,
        ) -> None:
            self.shape = shape
            self.dtype = dtype
            self.is_cuda = True
            self._strides = strides or tuple(1 for _ in shape)

        def dim(self) -> int:
            return len(self.shape)

        def is_contiguous(self) -> bool:
            return True

        def data_ptr(self) -> int:
            return 16

        def stride(self, dim: int) -> int:
            return self._strides[dim]

        def get_device(self) -> int:
            return 0

    float32 = object()
    int32 = object()
    logits = _FakeTensor((1, 1024), float32, strides=(1024, 1))
    row_starts = _FakeTensor((1,), int32)
    row_ends = _FakeTensor((1,), int32)
    indices = _FakeTensor((1, 4), int32)
    allocation_attempted = False

    def allocate_workspace(_logits: _FakeTensor) -> object:
        nonlocal allocation_attempted
        allocation_attempted = True
        return object()

    monkeypatch.setattr(ss_host, "_TENSOR", _FakeTensor)
    monkeypatch.setattr(ss_host, "_F32", float32)
    monkeypatch.setattr(ss_host, "_I32", int32)
    monkeypatch.setattr(ss_host, "_PREFILL_CACHE", {})
    monkeypatch.setattr(ss_host, "_ws_hot", {})
    monkeypatch.setattr(ss_host, "_is_capturing", lambda: True)
    monkeypatch.setattr(ss_host, "_prefill_topology", lambda _device: _topology(212))
    monkeypatch.setattr(ss_host, "default_workspace", allocate_workspace)

    with pytest.raises(RuntimeError, match="warm up before CUDA graph capture"):
        ss_host.run_prefill(logits, row_starts, row_ends, indices, max_row_len=1024)

    assert not allocation_attempted
