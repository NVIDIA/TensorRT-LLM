# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP4 helper imports and fixture lifetimes do not require a CUDA runtime."""

from __future__ import annotations

import ast
import gc
import importlib.util
import sys
import weakref
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from importlib.machinery import PathFinder
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.cpu_only

_HELPER = Path(__file__).with_name("_fp4_moe_test_utils.py")
_CATALOG = "tensorrt_llm._torch._experimental.modeling_v2.catalog"


@pytest.fixture
def helpers(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load the ordinary helper with tensor creation unavailable."""
    for name in (_CATALOG + "._op", _CATALOG + ".moe.fp4_block_scale_moe_runner"):
        parts = name.split(".")
        for count in range(1, len(parts) + 1):
            qualified = ".".join(parts[:count])
            package = ModuleType(qualified)
            package.__path__ = []
            monkeypatch.setitem(sys.modules, qualified, package)

    @contextmanager
    def precision_scope() -> Iterator[None]:
        yield

    precision = ModuleType("_precision")
    precision.true_fp32_matmul = precision_scope
    monkeypatch.setitem(sys.modules, "_precision", precision)
    sys.modules[_CATALOG + ".moe.fp4_block_scale_moe_runner"].fp4_block_scale_moe_runner = object()

    torch = ModuleType("torch")

    def unexpected_allocation(*args, **kwargs) -> None:
        raise AssertionError("importing test helpers must not allocate a tensor")

    torch.tensor = unexpected_allocation
    monkeypatch.setitem(sys.modules, "torch", torch)
    validating = ModuleType("_validating")
    validating.validating = nullcontext
    monkeypatch.setitem(sys.modules, "_validating", validating)

    spec = importlib.util.spec_from_file_location("_fp4_test_helpers_import", _HELPER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_import_does_not_allocate_or_read_cuda(helpers: ModuleType) -> None:
    assert callable(helpers._build)
    assert callable(helpers.renorm_operands)
    assert not hasattr(sys.modules["torch"], "cuda")


def test_helper_resolves_its_own_validating_import(
    helpers: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _HELPER.parents[3] / "test_common/magic_import.py"
    spec = importlib.util.spec_from_file_location("_fp4_helpers_magic_finder", source)
    assert spec is not None and spec.loader is not None
    finder_module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, finder_module)
    monkeypatch.setattr(sys, "meta_path", list(sys.meta_path))
    spec.loader.exec_module(finder_module)

    # PathFinder searches the declared paths directly, bypassing the stub
    # _validating module installed by helpers and any other sys.modules cache.
    paths = finder_module.MagicFinder._search_paths(str(_HELPER), helpers.__dict__)
    validating = PathFinder.find_spec("_validating", paths)
    assert validating is not None
    assert Path(validating.origin).resolve() == _HELPER.with_name("_validating.py").resolve()


@pytest.mark.parametrize("failure", [False, True], ids=["normal-teardown", "failed-test"])
def test_many_expert_operands_are_lazy_reused_and_released(
    helpers: ModuleType, monkeypatch: pytest.MonkeyPatch, failure: bool
) -> None:
    calls = []

    class Layer:
        pass

    def build(*args, **kwargs) -> Layer:
        calls.append((args, kwargs))
        return Layer()

    monkeypatch.setattr(helpers, "_build", build)
    fixture = helpers.many_expert_layer.__wrapped__()
    get_layer = next(fixture)
    assert calls == []

    layer = get_layer()
    reference = weakref.ref(layer)
    assert get_layer() is layer
    assert calls == [((512, 2560, 640), {"seed": 512640})]
    del layer

    if failure:
        with pytest.raises(RuntimeError, match="evaluation failed"):
            fixture.throw(RuntimeError("evaluation failed"))
    else:
        fixture.close()
    gc.collect()
    assert reference() is None


def test_fixture_instances_do_not_retain_another_modules_operands(
    helpers: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def build(*args, **kwargs) -> object:
        calls.append((args, kwargs))
        return object()

    monkeypatch.setattr(helpers, "_build", build)
    first = helpers.many_expert_layer.__wrapped__()
    second = helpers.many_expert_layer.__wrapped__()
    try:
        first_layer = next(first)()
        second_layer = next(second)()
        assert first_layer is not second_layer
        assert len(calls) == 2
    finally:
        first.close()
        second.close()


@pytest.mark.parametrize("failure", [False, True], ids=["normal-teardown", "failed-test"])
def test_shared_geometry_caches_are_released_at_module_teardown(failure: bool) -> None:
    source = _HELPER.parent / "moe/test_modeling_v2_fp4_block_scale_moe_runner.py"
    cleanup = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "_release_reference_layers"
    )
    cleanup.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            cleanup,
        ],
        type_ignores=[],
    )

    class Layer:
        pass

    namespace = {"_DSV3": Layer(), "_DSR1": Layer(), "_DSR1_SLICE": {256: Layer()}}
    references = [
        weakref.ref(namespace["_DSV3"]),
        weakref.ref(namespace["_DSR1"]),
        weakref.ref(namespace["_DSR1_SLICE"][256]),
    ]
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    fixture = namespace["_release_reference_layers"]()
    next(fixture)
    assert all(reference() is not None for reference in references)
    if failure:
        with pytest.raises(RuntimeError, match="evaluation failed"):
            fixture.throw(RuntimeError("evaluation failed"))
    else:
        fixture.close()
    gc.collect()
    assert namespace["_DSV3"] is None and namespace["_DSR1"] is None
    assert namespace["_DSR1_SLICE"] == {}
    assert all(reference() is None for reference in references)
