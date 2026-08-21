# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tensorrt_llm._torch.models import modeling_utils


def test_timing_metric_accumulates_and_records_failures(monkeypatch) -> None:
    perf_counter_values = iter([1.0, 1.25, 2.0, 2.5])
    monkeypatch.setattr(modeling_utils.time, "perf_counter", lambda: next(perf_counter_values))
    metrics = {}

    with modeling_utils.timing_metric("load_seconds", metrics):
        pass

    with pytest.raises(RuntimeError, match="load failed"):
        with modeling_utils.timing_metric("load_seconds", metrics):
            raise RuntimeError("load failed")

    assert metrics["load_seconds"] == pytest.approx(0.75)
