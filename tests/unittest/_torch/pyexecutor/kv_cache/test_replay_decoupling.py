# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The generic KV cache runtime knows nothing of speculative-decoding replay state.

Replay state belongs to the model families (Mamba2, GDN, KDA): their providers declare it as a
KV cache region and their cache managers and metadata classes handle it. The shared managers, the
region model and the shared Mamba metadata must not name it, nor import a family's replay code.

The shared hybrid manager and Mamba metadata likewise leave the rest of the families' state to
them: the GDN state and in-forward snapshots, the FP4 MLA cache layout Kimi K3 selects.
"""

import ast
import importlib.util
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

_GENERIC_MODULES = (
    "_torch/pyexecutor/kv_cache/kv_cache_manager_v2.py",
    "_torch/pyexecutor/kv_cache/regions.py",
    "_torch/pyexecutor/kv_cache/mamba_cache_manager/mamba_cache_manager_v2.py",
    "_torch/pyexecutor/kv_cache/mamba_cache_manager/common.py",
    "_torch/modules/mamba/mamba2_metadata.py",
)
# Names that mention replay for another reason: a KV connector replays a request's prefix.
_UNRELATED_NAMES = frozenset({"context_replay_tokens"})
_FAMILY_PACKAGES = (
    "tensorrt_llm._torch.modules.fla.",
    "tensorrt_llm._torch.modules.kimi_kda.",
    "tensorrt_llm._torch.modules.mamba.",
    "tensorrt_llm._torch.modules.qwen4_exp.",
)
# The families' managers and metadata classes, which own their replay state. Any other
# family module counts as replay code when its name says so; kernels the generic code
# shares with a family (e.g. GDN chunk helpers) do not.
_FAMILY_REPLAY_OWNERS = (
    "tensorrt_llm._torch.modules.fla.cache_manager",
    "tensorrt_llm._torch.modules.fla.gdn_metadata",
    "tensorrt_llm._torch.modules.kimi_kda.cache_manager",
    "tensorrt_llm._torch.modules.kimi_kda.kimi_k3_mamba_metadata",
    "tensorrt_llm._torch.modules.mamba.cache_manager",
    "tensorrt_llm._torch.modules.qwen4_exp.cache_manager",
)


def _is_family_replay_code(module: str) -> bool:
    return module.startswith(_FAMILY_REPLAY_OWNERS) or (
        module.startswith(_FAMILY_PACKAGES) and "replay" in module.lower()
    )


def _package_dir() -> Path:
    spec = importlib.util.find_spec("tensorrt_llm")
    assert spec is not None and spec.origin is not None
    return Path(spec.origin).parent


def _names(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            yield node.id
        elif isinstance(node, ast.Attribute):
            yield node.attr
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield node.name
        elif isinstance(node, ast.arg):
            yield node.arg
        elif isinstance(node, ast.alias):
            yield node.asname or node.name
        elif isinstance(node, ast.keyword) and node.arg is not None:
            yield node.arg


def _imported_modules(tree: ast.AST, module: str):
    package = module.rsplit(".", 1)[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = package.split(".")
                base = ".".join(parts[: len(parts) - node.level + 1] + ([base] if base else []))
            yield base
            yield from (f"{base}.{alias.name}" for alias in node.names)


@pytest.mark.parametrize("relative_path", _GENERIC_MODULES)
def test_generic_module_names_no_replay_state(relative_path: str) -> None:
    path = _package_dir() / relative_path
    tree = ast.parse(path.read_text(), filename=str(path))
    named = sorted({name for name in _names(tree) if "replay" in name.lower()} - _UNRELATED_NAMES)
    assert not named, f"{relative_path} names replay state: {named}"


@pytest.mark.parametrize("relative_path", _GENERIC_MODULES)
def test_generic_module_imports_no_family_replay_code(relative_path: str) -> None:
    path = _package_dir() / relative_path
    tree = ast.parse(path.read_text(), filename=str(path))
    module = "tensorrt_llm." + relative_path[: -len(".py")].replace("/", ".")
    imported = sorted(
        name for name in set(_imported_modules(tree, module)) if _is_family_replay_code(name)
    )
    assert not imported, f"{relative_path} imports model-family replay code: {imported}"


# The Mamba modules every hybrid family shares.
_GENERIC_MAMBA_MODULES = (
    "_torch/pyexecutor/kv_cache/mamba_cache_manager/mamba_cache_manager_v2.py",
    "_torch/pyexecutor/kv_cache/mamba_cache_manager/common.py",
    "_torch/modules/mamba/mamba2_metadata.py",
)
# Family state the shared Mamba modules leave to the families, matched in the lowercased
# names: the families (GDN, KDA, Kimi, Qwen, Nemotron), the in-forward snapshots, and the
# FP4 MLA cache layout.
_FAMILY_NAME_PARTS = ("gdn", "kda", "kimi", "qwen", "nemotron", "inforward", "fp4")
# Matched as whole words of a name instead: "SsmLayerConfig" holds the letters of "mla".
_FAMILY_NAME_WORDS = ("mla",)
# The family packages whose modules the shared Mamba modules do not import at all.
_FAMILY_ONLY_PACKAGES = (
    "tensorrt_llm._torch.attention.backends.fp4_mla",
    "tensorrt_llm._torch.modules.fla",
    "tensorrt_llm._torch.modules.kimi_kda",
    "tensorrt_llm._torch.modules.qwen4_exp",
)


def _words(name: str) -> set[str]:
    """The lowercased words of a snake_case or CamelCase name."""
    return {
        word.lower()
        for part in name.split("_")
        for word in re.findall(r"[A-Z]?[a-z0-9]+|[A-Z0-9]+(?![a-z])", part)
    }


@pytest.mark.parametrize("relative_path", _GENERIC_MAMBA_MODULES)
def test_generic_mamba_module_names_no_family_state(relative_path: str) -> None:
    path = _package_dir() / relative_path
    tree = ast.parse(path.read_text(), filename=str(path))
    names = set(_names(tree))
    named = sorted(
        name
        for name in names
        if any(part in name.lower() for part in _FAMILY_NAME_PARTS)
        or _words(name).intersection(_FAMILY_NAME_WORDS)
    )
    assert not named, f"{relative_path} names model-family state: {named}"


@pytest.mark.parametrize("relative_path", _GENERIC_MAMBA_MODULES)
def test_generic_mamba_module_imports_no_family_package(relative_path: str) -> None:
    path = _package_dir() / relative_path
    tree = ast.parse(path.read_text(), filename=str(path))
    module = "tensorrt_llm." + relative_path[: -len(".py")].replace("/", ".")
    imported = sorted(
        name
        for name in set(_imported_modules(tree, module))
        if name.startswith(_FAMILY_ONLY_PACKAGES)
    )
    assert not imported, f"{relative_path} imports model-family modules: {imported}"


def test_family_name_words_split_snake_and_camel_case() -> None:
    assert _words("Fp4MlaV2CacheLayoutPolicy") >= {"fp4", "mla", "v2"}
    assert _words("MLA_HP_TAIL") == {"mla", "hp", "tail"}
    assert "mla" not in _words("SsmLayerConfig")
