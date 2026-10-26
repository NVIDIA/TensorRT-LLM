# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host-only contracts for the Push-Combine EP surface and bundled DSO."""

import ast
import importlib.util
import re
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[5]
COMM_DIR = REPO_ROOT / "tensorrt_llm/_torch/modules/fused_moe/communication"
PUSH_CPP_DIR = REPO_ROOT / "cpp/tensorrt_llm/kernels/communicationKernels/tileNvfp4"
PUSH_FLAGS = PUSH_CPP_DIR / "moeAlltoAllTileNvfp4V2PushFlags.h"
PUSH_OP = PUSH_CPP_DIR / "moeAlltoAllTileNvfp4V2PushOp.cpp"
PUSH_K4 = PUSH_CPP_DIR / "moeAlltoAllTileNvfp4V2PushK4.cu"
LOADER_PATH = COMM_DIR / "tile_tma_forced_loader.py"
LAYOUT_PATH = COMM_DIR / "tile_nvfp4_v2_layout.py"
SELECTOR_PATH = COMM_DIR / "nvlink_one_sided_tile_nvfp4_push.py"

EXPECTED_EP_RECORDS = ((2, 2), (4, 4), (8, 8), (16, 8), (32, 8))
EXPECTED_READY_CAPACITY = 32
EXPECTED_GENERATION_OFFSET = EXPECTED_READY_CAPACITY * 8
EXPECTED_GATE_OFFSET = EXPECTED_GENERATION_OFFSET + 8
EXPECTED_CONTROL_SLAB_BYTES = EXPECTED_GATE_OFFSET + 8


def _load_source_module(module_name: str, path: Path, stubs=None):
    stubs = {} if stubs is None else stubs
    previous_modules = {name: sys.modules.get(name) for name in (module_name, *stubs)}
    sys.modules.update(stubs)
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous
    return module


def _loader():
    package_name = "_push_contract_communication"
    fake_package = types.ModuleType(package_name)
    fake_package.__path__ = []
    fake_base = types.ModuleType(f"{package_name}.base")
    fake_base.Communication = type("Communication", (), {})

    fake_torch = types.ModuleType("torch")
    fake_torch.dtype = type("dtype", (), {})
    fake_torch.bfloat16 = object()
    fake_torch.ops = types.SimpleNamespace(load_library=lambda _path: None)

    fake_trtllm = types.ModuleType("tensorrt_llm")
    fake_trtllm.__path__ = []
    fake_logger_module = types.ModuleType("tensorrt_llm.logger")
    fake_logger_module.logger = types.SimpleNamespace(info=lambda _message: None)
    fake_mapping_module = types.ModuleType("tensorrt_llm.mapping")
    fake_mapping_module.Mapping = type("Mapping", (), {})

    return _load_source_module(
        f"{package_name}.tile_tma_forced_loader",
        LOADER_PATH,
        {
            package_name: fake_package,
            f"{package_name}.base": fake_base,
            "torch": fake_torch,
            "tensorrt_llm": fake_trtllm,
            "tensorrt_llm.logger": fake_logger_module,
            "tensorrt_llm.mapping": fake_mapping_module,
        },
    )


def _layout():
    return _load_source_module("_push_contract_layout", LAYOUT_PATH)


def _selector_ast():
    tree = ast.parse(SELECTOR_PATH.read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "NVLinkOneSidedTilePushCombine"
    )


def _selector_assignment(name: str):
    class_node = _selector_ast()
    for node in class_node.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == name:
            return ast.literal_eval(node.value)
    raise AssertionError(f"missing selector assignment {name}")


def _selector_records_getter():
    class_node = _selector_ast()
    method = next(
        node
        for node in class_node.body
        if isinstance(node, ast.FunctionDef) and node.name == "records_per_token"
    )
    namespace = {}
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    exec(compile(module, str(SELECTOR_PATH), "exec"), namespace)
    return namespace["records_per_token"].fget


def _constexpr_int(source: str, name: str) -> int:
    match = re.search(rf"constexpr int {name}\s*=\s*(\d+);", source)
    assert match is not None, f"missing integer constexpr {name}"
    return int(match.group(1))


def _static_assert_offset(source: str, name: str) -> int:
    match = re.search(rf"static_assert\({name}\s*==\s*(\d+),", source)
    assert match is not None, f"missing frozen offset assertion for {name}"
    return int(match.group(1))


@pytest.mark.parametrize(("ep_size", "records_per_token"), EXPECTED_EP_RECORDS)
def test_push_ep_surface_and_records_mapping(ep_size, records_per_token):
    loader = _loader()

    assert loader.PUSH_COMBINE_SUPPORTED_EP_SIZES == tuple(ep for ep, _ in EXPECTED_EP_RECORDS)
    assert _selector_assignment("SUPPORTED_EP_SIZES") == loader.PUSH_COMBINE_SUPPORTED_EP_SIZES
    assert loader.validate_push_combine_ep_size(ep_size) == ep_size

    selector = types.SimpleNamespace(REQUIRED_TOP_K=8)
    selector.ep_size = ep_size
    assert _selector_records_getter()(selector) == records_per_token


@pytest.mark.parametrize("ep_size", [0, 1, 3, 6, 64])
def test_push_rejects_ep_sizes_outside_the_published_surface(ep_size):
    loader = _loader()

    with pytest.raises(ValueError, match=r"supported K4 consumer surface"):
        loader.validate_push_combine_ep_size(ep_size)


def test_push_identity_pins_ep32_manifest_and_control_geometry():
    layout = _layout()

    assert layout.PUSH_FLAG_LAYOUT_ID == "compact_row_v2"
    assert layout.PUSH_READY_CAPACITY == EXPECTED_READY_CAPACITY
    assert layout.PUSH_CONTROL_SLAB_BYTES == EXPECTED_CONTROL_SLAB_BYTES
    assert layout.PUSH_SUPPORTED_PACKED_TUPLE_MANIFEST == EXPECTED_EP_RECORDS
    assert layout.push_identity_fields() == (
        "push_combine_identity_v2",
        "raw_u64_epoch_v1",
        "compact_row_v2",
        EXPECTED_READY_CAPACITY,
        EXPECTED_CONTROL_SLAB_BYTES,
        (224, 0, 0, 2, 2),
        0,
        4480,
        EXPECTED_EP_RECORDS,
    )


def test_push_control_plane_supports_ep32_and_rejects_larger_ep():
    layout = _layout()

    assert layout.push_control_plane_bytes(32) == layout.align128(32 * EXPECTED_CONTROL_SLAB_BYTES)
    with pytest.raises(ValueError, match=r"must be in \(0, 32\]"):
        layout.push_control_plane_bytes(33)


def test_cpp_control_layout_matches_the_python_ep32_contract():
    source = PUSH_FLAGS.read_text(encoding="utf-8")

    assert _constexpr_int(source, "kPushReadyCapacity") == EXPECTED_READY_CAPACITY
    assert _constexpr_int(source, "kPushControlSlabBytes") == EXPECTED_CONTROL_SLAB_BYTES
    assert _static_assert_offset(source, "kPushGenerationOffsetBytes") == EXPECTED_GENERATION_OFFSET
    assert _static_assert_offset(source, "kPushLocalGateOffsetBytes") == EXPECTED_GATE_OFFSET


def test_push_build_id_v2_matches_loader_selector_and_dso():
    loader = _loader()
    op_source = PUSH_OP.read_text(encoding="utf-8")

    assert loader.PUSH_COMBINE_EXPORTED_BUILD_ID == "tile_push_combine:v2"
    assert _selector_assignment("BUILD_ID_PREFIX") == "tile_push_combine:v2:"
    assert '#define TILE_PUSH_COMBINE_BUILD_ID "tile_push_combine:v2"' in op_source


def test_ep4_k4_scans_all_topk_positions_via_route8():
    source = PUSH_K4.read_text(encoding="utf-8")

    # Dispatch deduplicates routes in place, so an EP4 token can have a valid
    # first occurrence in top-k slots 4..7. Route8 scans every position;
    # Route4's no-fallback fast path would silently drop those records.
    assert "if (params.records_per_token != 2)" in source
    assert "moeA2APushK4Route8Kernel" in source
    assert "moeA2APushK4Route4Kernel" not in source


def test_bundled_push_dso_path_is_package_relative():
    loader = _loader()
    dso_path = Path(loader.get_packaged_push_combine_dso_path())

    assert dso_path.name == loader.PUSH_COMBINE_PACKAGED_DSO
    assert dso_path.parent.name == "libs"
    assert dso_path.parent.parent.name == "tensorrt_llm"


def test_push_dso_resolver_uses_bundled_wheel_by_default(monkeypatch, tmp_path):
    loader = _loader()
    bundled_dso = tmp_path / loader.PUSH_COMBINE_PACKAGED_DSO
    bundled_dso.write_bytes(b"host-only resolver fixture")
    monkeypatch.delenv(loader.PUSH_COMBINE_DSO_ENV, raising=False)
    monkeypatch.setattr(
        loader,
        "get_packaged_push_combine_dso_path",
        lambda: str(bundled_dso),
    )

    assert loader.resolve_push_combine_dso_path() == str(bundled_dso)


def test_push_dso_resolver_prefers_expert_override(monkeypatch, tmp_path):
    loader = _loader()
    override_dso = tmp_path / "override.so"
    override_dso.write_bytes(b"host-only resolver fixture")
    monkeypatch.setenv(loader.PUSH_COMBINE_DSO_ENV, str(override_dso))
    monkeypatch.setattr(
        loader,
        "get_packaged_push_combine_dso_path",
        lambda: pytest.fail("bundled DSO must not be resolved when override is set"),
    )

    assert loader.resolve_push_combine_dso_path() == str(override_dso)


@pytest.mark.parametrize("use_override", [False, True])
def test_push_dso_resolver_fails_closed_when_selected_file_is_missing(
    monkeypatch, tmp_path, use_override
):
    loader = _loader()
    missing_dso = tmp_path / "missing.so"
    if use_override:
        monkeypatch.setenv(loader.PUSH_COMBINE_DSO_ENV, str(missing_dso))
        expected_source = f"{loader.PUSH_COMBINE_DSO_ENV} override"
    else:
        monkeypatch.delenv(loader.PUSH_COMBINE_DSO_ENV, raising=False)
        monkeypatch.setattr(
            loader,
            "get_packaged_push_combine_dso_path",
            lambda: str(missing_dso),
        )
        expected_source = "bundled TensorRT-LLM wheel"

    with pytest.raises(RuntimeError, match=rf"{re.escape(expected_source)} is missing"):
        loader.resolve_push_combine_dso_path()
