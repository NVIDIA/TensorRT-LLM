# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.visual_gen.runtime import native

pytestmark = pytest.mark.cpu_only


def test_sibling_scopes_load_one_family_library(tmp_path, monkeypatch):
    monkeypatch.setattr(native, "_LIBS_DIR", tmp_path)
    lib = "libth_visual_gen_family.so"
    manifest = {
        "models/family": {"library": lib, "namespace": "vg_family"},
        "models/family/target": {"library": lib, "namespace": "vg_family_target"},
    }
    (tmp_path / "visual_gen_native.json").write_text(json.dumps(manifest))
    shared_op, target_op = object(), object()
    monkeypatch.setattr(torch.ops, "vg_family", SimpleNamespace(apply=shared_op))
    monkeypatch.setattr(torch.ops, "vg_family_target", SimpleNamespace(apply=target_op))
    loads = []

    def open_library(path):
        loads.append(path)
        torch.ops.loaded_libraries.add(path)

    monkeypatch.setattr(torch.ops, "loaded_libraries", set())
    monkeypatch.setattr(torch.ops, "load_library", open_library)
    assert native.load_native_op("models/family/target", "apply") is target_op
    assert native.load_native_op("models/family", "apply") is shared_op
    assert loads == [str(tmp_path / lib)]


def test_scope_filtered_out_of_build(tmp_path, monkeypatch):
    monkeypatch.setattr(native, "_LIBS_DIR", tmp_path)
    (tmp_path / "visual_gen_native.json").write_text("{}")
    with pytest.raises(native.NativeScopeUnavailable, match="catalog/activation"):
        native.load_native_op("catalog/activation", "swiglu_block_fp8")


def test_installed_scope_with_missing_library(tmp_path, monkeypatch):
    monkeypatch.setattr(native, "_LIBS_DIR", tmp_path)
    manifest = {
        "catalog/activation": {"library": "libth_visual_gen_catalog.so", "namespace": "vg_catalog"}
    }
    (tmp_path / "visual_gen_native.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="Cannot load .*libth_visual_gen_catalog.so") as exc:
        native.load_native_op("catalog/activation", "swiglu_block_fp8")
    assert not isinstance(exc.value, native.NativeScopeUnavailable)
