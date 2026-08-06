# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from importlib import import_module

import pytest

cute = pytest.importorskip("cutlass.cute")

from tensorrt_llm._torch.visual_gen.attention_backend import flash_attn4, parallel  # noqa: E402
from tensorrt_llm._torch.visual_gen.attention_backend.flash_attn4 import (  # noqa: E402
    _install_cutlass_dsl_compatibility,
)


def test_cutlass_dsl_47_moved_names_are_restored(monkeypatch):
    monkeypatch.delattr(cute.core, "ThrCopy", raising=False)
    monkeypatch.delattr(cute.core, "ThrMma", raising=False)
    monkeypatch.delattr(cute, "make_fragment", raising=False)

    _install_cutlass_dsl_compatibility()

    assert cute.core.ThrCopy is cute.ThrCopy
    assert cute.core.ThrMma is cute.ThrMma
    assert cute.make_fragment is cute.make_rmem_tensor


def test_cutlass_dsl_existing_names_are_preserved(monkeypatch):
    existing_thr_copy = object()
    existing_thr_mma = object()
    existing_make_fragment = object()
    monkeypatch.setattr(cute.core, "ThrCopy", existing_thr_copy)
    monkeypatch.setattr(cute.core, "ThrMma", existing_thr_mma)
    monkeypatch.setattr(cute, "make_fragment", existing_make_fragment)

    _install_cutlass_dsl_compatibility()

    assert cute.core.ThrCopy is existing_thr_copy
    assert cute.core.ThrMma is existing_thr_mma
    assert cute.make_fragment is existing_make_fragment


def test_cutlass_dsl_47_aliases_allow_fa4_interface_import() -> None:
    _install_cutlass_dsl_compatibility()
    interface = import_module("flash_attn.cute.interface")

    assert callable(interface._flash_attn_fwd)
    assert callable(interface.flash_attn_combine)
    assert callable(flash_attn4._flash_attn_fwd)
    assert callable(parallel._flash_attn_combine)
