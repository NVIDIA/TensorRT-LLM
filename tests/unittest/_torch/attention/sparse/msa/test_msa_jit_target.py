# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The GPU target fmha_sm100 JIT-builds for, and where it builds and loads.

Runs on CPU: torch.cuda is faked and nothing is compiled.
"""

import os
import re

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
    msa_package_available,
)

pytestmark = pytest.mark.skipif(
    not msa_package_available(), reason="fmha_sm100 (MSA submodule) required"
)

ARCH_ENV = "MINFER_FMHA_CUDA_ARCH"


@pytest.fixture
def jit(monkeypatch, tmp_path):
    """fmha_sm100.jit with no target resolved yet, an empty cache in tmp_path,
    and MINFER_FMHA_CUDA_ARCH empty, which counts as unset."""
    import fmha_sm100.jit as jit

    monkeypatch.setattr(jit, "_jit_target_value", jit._JIT_TARGET_UNSET)
    monkeypatch.setattr(jit, "CACHE_BASE", tmp_path)
    monkeypatch.setenv(ARCH_ENV, "")
    return jit


def fake_gpus(monkeypatch, *capabilities):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: bool(capabilities))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: len(capabilities))
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i: capabilities[i])


def gencodes(jit, cache_dir):
    return re.findall(r"-gencode=\S+", jit._get_nvcc_flags(cache_dir))


@pytest.mark.parametrize(
    "capabilities, target",
    [([(10, 3)] * 4, "103a"), ([(10, 0)], "100a")],
    ids=["four_sm103", "one_sm100"],
)
def test_one_architecture_is_the_target_and_is_exported(jit, monkeypatch, capabilities, target):
    fake_gpus(monkeypatch, *capabilities)
    assert jit._resolve_jit_target() == target
    assert os.environ[ARCH_ENV] == target
    assert jit._jit_base() == jit.CACHE_BASE / f"sm_{target}"
    assert gencodes(jit, jit._jit_base() / "plan") == [
        f"-gencode=arch=compute_{target},code=sm_{target}"
    ]
    # A build into the shared root, as scripts/warmup_fmha_sm100.py does, stays portable.
    assert len(gencodes(jit, jit.CACHE_BASE / "plan")) == 3


@pytest.mark.parametrize(
    "capabilities",
    [[(10, 0), (10, 3)], [(9, 0)], [(12, 0)], []],
    ids=["sm100_and_sm103", "sm90", "sm120", "no_gpu"],
)
def test_other_devices_keep_the_portable_image(jit, monkeypatch, capabilities):
    fake_gpus(monkeypatch, *capabilities)
    assert jit._resolve_jit_target() is None
    assert os.environ[ARCH_ENV] == ""
    assert jit._jit_base() == jit.CACHE_BASE
    assert len(gencodes(jit, jit._jit_base() / "plan")) == 3


def test_a_cuda_error_keeps_the_portable_image(jit, monkeypatch):
    fake_gpus(monkeypatch, (10, 3))

    def fail(device):
        raise RuntimeError("Cannot re-initialize CUDA in forked subprocess")

    monkeypatch.setattr(torch.cuda, "get_device_capability", fail)
    assert jit._resolve_jit_target() is None
    assert jit._jit_base() == jit.CACHE_BASE


def test_the_override_wins_over_the_devices(jit, monkeypatch):
    fake_gpus(monkeypatch, (10, 3))
    monkeypatch.setenv(ARCH_ENV, "100a")
    assert jit._resolve_jit_target() == "100a"
    assert jit._jit_base() == jit.CACHE_BASE / "sm_100a"


def test_a_shared_build_is_loaded_until_the_target_has_its_own(jit, monkeypatch):
    fake_gpus(monkeypatch, (10, 3))
    so_name = "fmha_sm100_plan.so"
    target_dir = jit.CACHE_BASE / "sm_103a" / "plan"
    shared_dir = jit.CACHE_BASE / "plan"
    assert jit._built_dir("plan", so_name) == target_dir
    shared_dir.mkdir(parents=True)
    (shared_dir / so_name).touch()
    assert jit._built_dir("plan", so_name) == shared_dir
    target_dir.mkdir(parents=True)
    (target_dir / so_name).touch()
    assert jit._built_dir("plan", so_name) == target_dir
