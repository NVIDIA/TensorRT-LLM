# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the free-memory reading used when sizing the KV cache."""
from types import SimpleNamespace

from tensorrt_llm._torch.pyexecutor import _util

GB = 1 << 30


def _patch(monkeypatch, *, free, total, integrated, available):
    monkeypatch.setattr(_util.torch.cuda, "mem_get_info",
                        lambda: (free * GB, total * GB))
    monkeypatch.setattr(_util, "is_device_integrated", lambda: integrated)
    monkeypatch.setattr(_util.psutil, "virtual_memory",
                        lambda: SimpleNamespace(available=int(available * GB)))


def test_integrated_gpu_counts_reclaimable_page_cache(monkeypatch):
    # Checkpoint page cache: cudaMemGetInfo reports 60 GiB free, but the
    # kernel can hand out 110 GiB.
    _patch(monkeypatch, free=60, total=120, integrated=True, available=110)
    assert _util._mem_get_info_for_kv_sizing() == (110 * GB, 120 * GB)


def test_integrated_gpu_free_is_capped_at_total(monkeypatch):
    _patch(monkeypatch, free=60, total=120, integrated=True, available=125)
    assert _util._mem_get_info_for_kv_sizing() == (120 * GB, 120 * GB)


def test_integrated_gpu_never_lowers_free(monkeypatch):
    _patch(monkeypatch, free=60, total=120, integrated=True, available=40)
    assert _util._mem_get_info_for_kv_sizing() == (60 * GB, 120 * GB)


def test_discrete_gpu_is_unchanged(monkeypatch):
    _patch(monkeypatch, free=60, total=120, integrated=False, available=110)
    assert _util._mem_get_info_for_kv_sizing() == (60 * GB, 120 * GB)
