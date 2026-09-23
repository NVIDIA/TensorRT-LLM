# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise setuptools' native-source collection without CUDA or a wheel build."""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

from setuptools import Distribution
from setuptools.command.build_py import build_py

REPO = Path(__file__).resolve().parents[3]
SCHEDULER = Path("_torch/cute_dsl_kernels/megamoe_scheduler_v2")
NATIVE = REPO / "tensorrt_llm" / SCHEDULER / "native.py"
LOCAL_INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.MULTILINE)


def shared_package_data() -> list[str]:
    """Read the actual cross-platform package_data passed to setup()."""
    tree = ast.parse((REPO / "setup.py").read_text())
    patterns = []
    for node in tree.body:
        if (
            isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "package_data"
            and isinstance(node.op, ast.Add)
        ):
            patterns.extend(ast.literal_eval(node.value))
    (setup_call,) = [
        node.value
        for node in tree.body
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "setup"
    ]
    (package_keyword,) = [item.value for item in setup_call.keywords if item.arg == "package_data"]
    if not isinstance(package_keyword, ast.Dict):
        raise AssertionError("setup() no longer uses the audited package_data mapping")
    (tensor_data,) = [
        value
        for key, value in zip(package_keyword.keys, package_keyword.values)
        if isinstance(key, ast.Constant) and key.value == "tensorrt_llm"
    ]
    if not isinstance(tensor_data, ast.Name) or tensor_data.id != "package_data":
        raise AssertionError("The extracted package_data is not passed to setuptools")
    return patterns


def load_native(path: Path) -> ModuleType:
    """Import only the stdlib native loader, bypassing the GPU package initializer."""
    spec = importlib.util.spec_from_file_location("_native_packaging_test", path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Cannot import native loader at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def loader_sources(native: ModuleType) -> dict[str, tuple[Path, ...]]:
    def capture(*, sources: tuple[Path, ...], **_kwargs: object) -> tuple[Path, ...]:
        return sources

    with patch.object(native, "_build_and_load", side_effect=capture):
        return {name: getattr(native, name)() for name in native.__all__}


def include_closure(sources: tuple[Path, ...], root: Path) -> set[Path]:
    """Follow quoted local includes, including headers absent from the JIT key."""
    pending, found = list(sources), set()
    while pending:
        source = pending.pop().resolve()
        source.relative_to(root)  # Native includes must stay inside this package.
        if source in found:
            continue
        found.add(source)
        text = source.read_text()
        pending.extend(source.parent / include for include in LOCAL_INCLUDE.findall(text))
    return found


class NativeSourcePackagingTests(unittest.TestCase):
    def test_setuptools_packages_every_native_compilation_input(self) -> None:
        native = load_native(NATIVE)
        required = {
            path
            for sources in loader_sources(native).values()
            for path in include_closure(sources, NATIVE.parent.resolve())
        }
        self.assertTrue(required)
        with tempfile.TemporaryDirectory(prefix="tekit-native-package-") as temporary:
            work = Path(temporary)
            package = work / "source/tensorrt_llm"
            scheduler = package / SCHEDULER
            scheduler.mkdir(parents=True)
            (package / "__init__.py").write_text("")
            (scheduler / "__init__.py").write_text("")
            shutil.copy2(NATIVE, scheduler / "native.py")
            for source in required:
                destination = scheduler / source.relative_to(NATIVE.parent)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)

            # Exercise the real build_py globbing and copies on an isolated source
            # tree. Exclude implicit SCM manifests, existing build caches and the
            # project's binary libraries so they cannot hide missing rules.
            distribution = Distribution(
                {
                    "name": "native-source-packaging-test",
                    "packages": ["tensorrt_llm", "tensorrt_llm." + ".".join(SCHEDULER.parts)],
                    "package_dir": {"tensorrt_llm": str(package)},
                    "package_data": {"tensorrt_llm": shared_package_data()},
                    "include_package_data": False,
                }
            )
            distribution.script_name = str(work / "setup.py")
            command = build_py(distribution)
            command.build_lib = str(work / "build")
            command.ensure_finalized()
            command.run()

            installed = Path(command.build_lib) / "tensorrt_llm" / SCHEDULER
            missing = sorted(
                str(source.relative_to(NATIVE.parent))
                for source in required
                if not (installed / source.relative_to(NATIVE.parent)).is_file()
            )
            self.assertEqual(missing, [], f"Native inputs missing from build_py output: {missing}")
            for source in required:
                self.assertEqual(
                    (installed / source.relative_to(NATIVE.parent)).read_bytes(),
                    source.read_bytes(),
                )

            # Run every real loader's source digest against the packaged tree.
            # Only the compiler/load boundary is mocked; no CUDA toolchain is used.
            packaged_native = load_native(installed / "native.py")

            def digest_only(**kwargs: object) -> str:
                return packaged_native._cache_key(kwargs["argv"], kwargs["sources"])

            with (
                patch.dict(os.environ, {"MEGAMOE_HALO_Q_ARCH": "sm_107"}),
                patch.object(packaged_native, "_build_and_load", side_effect=digest_only),
            ):
                for name in packaged_native.__all__:
                    with self.subTest(loader=name):
                        self.assertRegex(getattr(packaged_native, name)(), r"^[0-9a-f]{20}$")


if __name__ == "__main__":
    unittest.main(verbosity=2)
