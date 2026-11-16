# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures shared by the coordinator behavior tests in this directory."""

import pytest

from tensorrt_llm._torch.disaggregation.orchestration import coordinator as coordinator_module


@pytest.fixture
def inflight_cancel(monkeypatch):
    monkeypatch.setattr(coordinator_module, "is_disagg_inflight_cancel_enabled", lambda: True)


@pytest.fixture
def clock(monkeypatch):
    """Freeze the coordinator's ``time.monotonic``; tests advance ``clock["t"]``."""
    now = {"t": 100.0}
    monkeypatch.setattr(coordinator_module.time, "monotonic", lambda: now["t"])
    return now
