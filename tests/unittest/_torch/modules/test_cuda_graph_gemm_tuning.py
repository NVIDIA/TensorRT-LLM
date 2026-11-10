# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TunableOp state around the generation CUDA-graph passes (``TLLM_CUDA_GRAPH_GEMM_TUNING``)."""

import os

import pytest
import torch

from tensorrt_llm._torch.modules.linear import (
    CUDA_GRAPH_GEMM_TUNING_ENV,
    CUDA_GRAPH_GEMM_TUNING_FILE_ENV,
    cuda_graph_gemm_tuning,
)

pytestmark = pytest.mark.cpu_only


class _FakeTunable:
    """Process-wide TunableOp state, as ``torch.cuda.tunable`` exposes it."""

    def __init__(self):
        self.enabled = False
        self.tuning = False
        self.filename = "tunableop_results0.csv"

    def is_enabled(self):
        return self.enabled

    def tuning_is_enabled(self):
        return self.tuning

    def get_filename(self):
        return self.filename

    def set_filename(self, filename, insert_device_ordinal=False):
        self.filename = filename + ("0" if insert_device_ordinal else "")

    def enable(self, val=True):
        self.enabled = val

    def tuning_enable(self, val=True):
        self.tuning = val

    def state(self):
        return self.enabled, self.tuning, self.filename


@pytest.fixture
def tunable(monkeypatch):
    fake = _FakeTunable()
    monkeypatch.setattr(torch.cuda, "tunable", fake)
    for env in (
        CUDA_GRAPH_GEMM_TUNING_ENV,
        CUDA_GRAPH_GEMM_TUNING_FILE_ENV,
        "PYTORCH_TUNABLEOP_ENABLED",
        "PYTORCH_TUNABLEOP_FILENAME",
    ):
        monkeypatch.delenv(env, raising=False)
    return fake


def test_off_by_default(tunable):
    before = tunable.state()
    with cuda_graph_gemm_tuning(tune=True):
        assert tunable.state() == before
    assert tunable.state() == before


def test_warmup_tunes_capture_reuses_and_both_restore(tunable, monkeypatch):
    monkeypatch.setenv(CUDA_GRAPH_GEMM_TUNING_ENV, "1")
    before = tunable.state()
    with cuda_graph_gemm_tuning(tune=True):  # warmup pass
        assert (tunable.enabled, tunable.tuning) == (True, True)
    assert tunable.state() == before
    with cuda_graph_gemm_tuning(tune=False):  # capture pass
        assert (tunable.enabled, tunable.tuning) == (True, False)
    assert tunable.state() == before


@pytest.mark.parametrize("tune", [True, False])
def test_results_file_used_inside_and_filename_restored(tunable, monkeypatch, tmp_path, tune):
    monkeypatch.setenv(CUDA_GRAPH_GEMM_TUNING_ENV, "1")
    results = str(tmp_path / "gemm.csv")
    monkeypatch.setenv(CUDA_GRAPH_GEMM_TUNING_FILE_ENV, results)
    before = tunable.state()
    with cuda_graph_gemm_tuning(tune=tune):
        assert tunable.filename == results + "0"  # one file per device
    assert tunable.state() == before


def test_results_stay_in_memory_without_a_file(tunable, monkeypatch):
    monkeypatch.setenv(CUDA_GRAPH_GEMM_TUNING_ENV, "1")
    before = tunable.state()
    with cuda_graph_gemm_tuning(tune=True):
        assert tunable.filename == os.devnull
    assert tunable.state() == before


def test_state_restored_when_the_pass_raises(tunable, monkeypatch):
    monkeypatch.setenv(CUDA_GRAPH_GEMM_TUNING_ENV, "1")
    before = tunable.state()
    with pytest.raises(RuntimeError):
        with cuda_graph_gemm_tuning(tune=True):
            raise RuntimeError("capture failed")
    assert tunable.state() == before


@pytest.mark.parametrize("extra_env", [{}, {"PYTORCH_TUNABLEOP_ENABLED": "1"}])
def test_ignored_with_multiple_ranks_or_user_tunableop(tunable, monkeypatch, extra_env):
    monkeypatch.setenv(CUDA_GRAPH_GEMM_TUNING_ENV, "1")
    for key, value in extra_env.items():
        monkeypatch.setenv(key, value)
    world_size = 1 if extra_env else 2
    before = tunable.state()
    with cuda_graph_gemm_tuning(tune=True, world_size=world_size):
        assert tunable.state() == before
    assert tunable.state() == before
