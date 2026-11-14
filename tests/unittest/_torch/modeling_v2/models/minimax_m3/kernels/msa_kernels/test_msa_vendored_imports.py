# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The family's vendored MSA packages (``kernels/msa_kernels``) and the MiniMax-M3.1-Flash targets' kernels import
without the top-level ``fmha_sm100`` / ``fmha_sm100_q8kv8`` packages they were vendored from.

An import of a top-level name would bind whatever this environment happens to install under it (or fail inside an
``except ImportError`` fallback, where it silently turns a feature off), so every such attempt is refused and
recorded here.
"""

import importlib
import importlib.abc
import importlib.util
import pkgutil
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("cutlass") is None, reason="the vendored MSA kernels need the CuTe DSL"
)

TOP_LEVEL = ("fmha_sm100", "fmha_sm100_q8kv8")
MINIMAX_M3 = "tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3"
MSA_KERNELS = f"{MINIMAX_M3}.kernels.msa_kernels"


def _top_level(name: str) -> bool:
    return name.split(".")[0] in TOP_LEVEL


class _Recorder(importlib.abc.MetaPathFinder):
    """Records (and refuses) every attempt to import a top-level MSA package name."""

    def __init__(self):
        self.attempts = []

    def find_spec(self, fullname, path=None, target=None):
        if _top_level(fullname):
            self.attempts.append(fullname)
            raise ModuleNotFoundError(f"blocked top-level MSA package {fullname}")
        return None


def _modules_to_sweep():
    """Every module under msa_kernels/ and under the MiniMax-M3.1-Flash targets' kernels/ directories."""
    package = importlib.import_module(MSA_KERNELS)
    yield from (
        info.name for info in pkgutil.walk_packages(package.__path__, prefix=f"{MSA_KERNELS}.")
    )
    family = importlib.import_module(MINIMAX_M3)
    for target in sorted(Path(family.__path__[0]).glob("minimax_m3_1_flash_*/kernels")):
        prefix = f"{MINIMAX_M3}.{target.parent.name}.kernels"
        yield prefix
        yield from (info.name for info in pkgutil.walk_packages([str(target)], prefix=f"{prefix}."))


@pytest.fixture
def blocked_top_level(monkeypatch):
    for name in list(sys.modules):
        if _top_level(name):
            monkeypatch.delitem(sys.modules, name)
    recorder = _Recorder()
    monkeypatch.setattr(sys, "meta_path", [recorder, *sys.meta_path])
    return recorder


def test_vendored_and_kernel_modules_import_without_the_top_level_packages(blocked_top_level):
    imported, skipped = [], []
    for name in _modules_to_sweep():
        if ".test_" in name or ".cute" in name or "build_k2q_csr" in name:
            continue  # the cute/ subtree loads through its own sys.path entry; build_k2q_csr compiles at import
        try:
            importlib.import_module(name)
            imported.append(name)
        except Exception as err:  # noqa: BLE001  (a module this environment cannot load, e.g. without a GPU)
            skipped.append(f"{name}: {type(err).__name__}: {err}")
        if blocked_top_level.attempts:
            break
    assert not blocked_top_level.attempts, f"{name} tried to import {blocked_top_level.attempts}"
    # The vendored packages import wherever the CuTe DSL does (the targets' kernels may need a device).
    vendored_failures = [entry for entry in skipped if entry.startswith(f"{MSA_KERNELS}.")]
    assert not vendored_failures, "\n".join(vendored_failures)
    assert any(name.endswith("fmha_sm100_q8kv8.q8_indexer_interface") for name in imported), skipped


def test_the_nvfp4_index_k_decode_reader_is_available(blocked_top_level):
    module = importlib.import_module(
        f"{MINIMAX_M3}.minimax_m3_1_flash_nvfp4__sm_107__tp2.kernels.m3_nvfp4_index_decode"
    )
    assert module.Nvfp4IndexDecodeScorer(2).available()
    assert not blocked_top_level.attempts
