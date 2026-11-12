# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3 family copies of the FP4 MLA backend code are verbatim.

``ops/attention/_k3_fp4_mla_{cache,kernels}`` copy definitions of the FP4 MLA backend modules outside the
KV-cache-manager contract; ``_k3_fp4_mla_{v_repack,cute_dsl_utils}`` copy whole modules. Every copied
definition must be AST-identical to its source definition (imports aside), and every name the copies import
from the contract modules must exist there. Parsing only: no GPU, no imports of the modules.
"""

import ast
from pathlib import Path

import pytest

pytestmark = pytest.mark.cpu_only

ROOT = Path(__file__).resolve().parents[8]
BACKEND = ROOT / "tensorrt_llm/_torch/attention/backends/fp4_mla"
FAMILY = ROOT / "tensorrt_llm/_torch/_experimental/modeling_v2/models/kimi_k3_vl/ops/attention"
BACKEND_PACKAGE = "tensorrt_llm._torch.attention.backends.fp4_mla"
CONTRACT = ("config", "layout", "metadata", "packed_plan", "state")
DEFINITION_COPIES = ("_k3_fp4_mla_cache", "_k3_fp4_mla_kernels")
SOURCES = (
    "cache_update",
    "cache_gather",
    "v_cache",
    "fp4_mla_context",
    "fp4_mla_kernels",
    "fp4_mla_triton",
)
# Copied definitions whose source reads an environment override the K3 targets do not take; the copy keeps
# the source's default path, so it is checked only for not reading the environment.
ENV_FREE_DEFINITIONS = {
    "_select_triton_block_v": "TRTLLM_FP4_MLA_BLOCK_V",
    "_triton_prepack_v_enabled": "TRTLLM_FP4_MLA_PREPACK_V / TRTLLM_FP4_MLA_TRITON_PREPACK_V",
    "_triton_v_packed_attr": "TRTLLM_FP4_MLA_SHARE_V_PACK_STORAGE",
    "_set_triton_v_packed_cache_valid": "TRTLLM_FP4_MLA_SHARE_V_PACK_STORAGE",
}
WHOLE_COPIES = {
    "_k3_fp4_mla_v_repack": "fp4_mla_cutedsl_v_repack",
    "_k3_fp4_mla_cute_dsl_utils": "cute_dsl_utils",
}


def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _strip_imports(node: ast.AST) -> str:
    """``ast.dump`` of ``node`` with every import statement (the only lines a copy rewrites) removed."""

    class _Drop(ast.NodeTransformer):
        def visit_Import(self, _node):
            return ast.Pass()

        def visit_ImportFrom(self, _node):
            return ast.Pass()

    return ast.dump(_Drop().visit(node), include_attributes=False)


def _definitions(tree: ast.Module) -> dict:
    out = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for name in ast.walk(target):
                    if isinstance(name, ast.Name):
                        out[name.id] = node
    return out


def _module_docstring_free_body(tree: ast.Module) -> list:
    body = list(tree.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
        body = body[1:]
    return [n for n in body if not isinstance(n, (ast.Import, ast.ImportFrom))]


@pytest.mark.parametrize("copy", DEFINITION_COPIES)
def test_copied_definitions_match_their_sources(copy):
    sources = {}
    for name in SOURCES:
        for def_name, node in _definitions(_parse(BACKEND / f"{name}.py")).items():
            sources.setdefault(def_name, []).append((name, node))
    copied = _definitions(_parse(FAMILY / f"{copy}.py"))
    assert copied
    for name, node in copied.items():
        assert name in sources, f"{copy}.{name} has no source definition"
        if name in ENV_FREE_DEFINITIONS:
            text = ast.unparse(node)
            assert "environ" not in text and "getenv" not in text and "_env_" not in text, (
                f"{copy}.{name} reads the environment"
            )
            continue
        matches = [
            src
            for src, src_node in sources[name]
            if _strip_imports(src_node) == _strip_imports(node)
        ]
        assert matches, f"{copy}.{name} differs from its source in {[s for s, _ in sources[name]]}"


@pytest.mark.parametrize("copy", list(WHOLE_COPIES))
def test_whole_module_copies_match_their_sources(copy):
    source = _module_docstring_free_body(_parse(BACKEND / f"{WHOLE_COPIES[copy]}.py"))
    copied = _module_docstring_free_body(_parse(FAMILY / f"{copy}.py"))
    assert [_strip_imports(n) for n in copied] == [_strip_imports(n) for n in source]


@pytest.mark.parametrize("copy", (*DEFINITION_COPIES, *WHOLE_COPIES))
def test_copies_import_only_the_contract_from_the_backend(copy):
    contract_defs = {m: _definitions(_parse(BACKEND / f"{m}.py")) for m in CONTRACT}
    for node in ast.walk(_parse(FAMILY / f"{copy}.py")):
        if not isinstance(node, ast.ImportFrom) or node.level:
            continue
        module = node.module or ""
        if not module.startswith(BACKEND_PACKAGE):
            continue
        submodule = module[len(BACKEND_PACKAGE) + 1 :]
        assert submodule in CONTRACT, f"{copy} imports backend module {module}"
        for alias in node.names:
            assert alias.name in contract_defs[submodule], f"{module}.{alias.name} not found"
