# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import pytest


def warmup_requests():
    perf_dir = Path(__file__).resolve().parents[2] / "integration/defs/perf"
    module_path = perf_dir / "bench_warmup.py"
    if not module_path.is_file():
        return None
    spec = importlib.util.spec_from_file_location("bench_warmup", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.bench_warmup_requests


@pytest.mark.parametrize(
    "gpus,concurrency,requests,batch_size,reuse,env,expected",
    [
        (4, -1, 512, 128, False, {}, 128),
        (4, 16, 512, 128, False, {}, 16),
        (4, 64, 32, 128, False, {}, 32),
        (4, -1, 512, 128, True, {}, 8),
        (4, -1, 512, 128, False, {"TRTLLM_PERF_WARMUP_PASS": "0"}, 8),
        (1, -1, 512, 16, False, {}, 16),
        (1, -1, 512, 16, False, {"TRTLLM_PERF_WARMUP_PASS": "off"}, 0),
    ],
)
def test_bench_warmup_covers_one_measured_wave(
    gpus, concurrency, requests, batch_size, reuse, env, expected
):
    select = warmup_requests()
    assert select is not None, "benchmark warmup selection is missing"
    assert select(gpus, concurrency, requests, batch_size, reuse, env) == expected
