# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TRTLLM_AR_TUNE_NCCL_SYMMETRIC: the AllReduce tuner's candidate list with and without NCCL_SYMMETRIC."""

import pytest
import torch

from tensorrt_llm._torch.autotuner import OptimizationProfile
from tensorrt_llm._torch.custom_ops import torch_custom_ops
from tensorrt_llm.functional import AllReduceStrategy


def _runner():
    runner = object.__new__(torch_custom_ops.AllReduceRunner)
    runner.tp_size = 2
    return runner


@pytest.mark.parametrize("rows", [8, 4096, 16384])
@pytest.mark.parametrize("tune_symmetric", [True, False])
def test_nccl_symmetric_candidate_follows_knob(
    monkeypatch, rows: int, tune_symmetric: bool
) -> None:
    monkeypatch.setattr(torch_custom_ops, "_AR_TUNE_NCCL_SYMMETRIC", tune_symmetric)
    inputs = [torch.empty(rows, 6144, dtype=torch.bfloat16)]
    tactics = _runner().get_valid_tactics(inputs, OptimizationProfile())
    assert AllReduceStrategy.NCCL.value in tactics
    assert (AllReduceStrategy.NCCL_SYMMETRIC.value in tactics) is tune_symmetric
    # Off removes only NCCL_SYMMETRIC: the other candidates are unchanged.
    monkeypatch.setattr(torch_custom_ops, "_AR_TUNE_NCCL_SYMMETRIC", True)
    stock = _runner().get_valid_tactics(inputs, OptimizationProfile())
    assert [t for t in stock if t != AllReduceStrategy.NCCL_SYMMETRIC.value] == [
        t for t in tactics if t != AllReduceStrategy.NCCL_SYMMETRIC.value
    ]
