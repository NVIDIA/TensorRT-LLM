# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""VisualGenRule — narrows CI when VisualGen source paths change.

VisualGen is a self-contained subsystem under
`tensorrt_llm/_torch/visual_gen/` (Flux / LTX-2 / Wan diffusion models)
plus the public API surface at `tensorrt_llm/visual_gen/` and the media
helpers at `tensorrt_llm/media/`.

For families with a routing module, narrow to mirrored tests and importing targets.
Modeling V2 catalog entries are traced through the LLM and VisualGen source trees.

Block membership:
VisualGen does NOT have its own `condition.terms.backend`; VG test
entries live in `backend: pytorch` and `backend: tensorrt` blocks.
A block "belongs to VG" iff any of its `tests:` entries lives under a
dedicated `visual_gen/` test path or is the VisualGen perf-sanity entry.

Shared serving boundary:
Public VisualGen and media modules participate in shared serving imports.
Their changes keep the complete CPU suite, shared serving integration tests,
and package/performance sanity checks in addition to VG entries. Missing API,
schema, lazy-import, or serving validation keeps the baseline fallback.
"""

from __future__ import annotations

from functools import cached_property
from pathlib import Path, PurePosixPath
from typing import Optional

from cbts.blocks import Stage, YAMLIndex, _entry_target, _target_in_filter_subtree

from ._helpers import (
    block_filters_for_stem,
    is_perf_stem,
    resolve_affected_stages,
    stages_by_yaml_stem,
)
from .base import PRInputs, Rule, RuleResult
from .docs import CPU_TEST_YAML_STEM, is_docs_path
from .modeling_v2 import _ImportGraph, _kept_entries, _module_name, _parse
from .modeling_v2_source import PACKAGE, routed_targets

# VG source-path prefixes the rule may claim, mirroring the VisualGen
# section of `.github/CODEOWNERS`. Tests under tests/** are left to
# TestsDefRule; the two rules' scopes combine via _TESTSONLY_FAMILY.
# fused-DiT paths stay unclaimed: those diffs also touch `inplace_info()`
# in `tensorrt_llm/_torch/compilation/utils.py`, which no VG rule claims.
_VG_SRC_PREFIXES: tuple[str, ...] = (
    "examples/visual_gen/",
    "scripts/visualgen_eval/",
    "tensorrt_llm/_torch/visual_gen/",
    "tensorrt_llm/media/",
    "tensorrt_llm/visual_gen/",
)

_VG_SCHEMA_PATH = "tensorrt_llm/schemas/trtllm-serve-visual-gen-config.schema.json"
_CATALOG = "tensorrt_llm/_torch/modeling_v2/catalog/"
_VG_V2 = "tensorrt_llm/_torch/visual_gen/modeling_v2"
_VG_V2_PACKAGE = _VG_V2.replace("/", ".")
_MODELS = f"{_VG_V2}/models/"
_VG_UNIT = "unittest/_torch/visual_gen/modeling_v2"
_VG_QUALITY = "visual_gen/modeling_v2"
_VG_CATALOG = f"{_VG_V2}/catalog/"
_VG_CATALOG_PACKAGE = f"{_VG_V2_PACKAGE}.catalog"
_VG_CATALOG_UNIT = f"{_VG_UNIT}/catalog"
_CATALOG_CONTRACT_TEST = "unittest/_torch/modeling_v2/test_modeling_v2_contracts.py"
# Trace shared provider packages as units, including changes to private or deleted implementation files.
_PRIMS_PACKAGE = "tensorrt_llm._torch.attention.backends.prims_ts"
_SAGE_PACKAGE = "tensorrt_llm._torch.attention.kernels.sage_quant"
_LOW_M_RUNNER = "tensorrt_llm._torch.modules.low_m_gemm"
_LOW_M_RUNNER_SOURCES = {
    "tensorrt_llm/_torch/modules/low_m_gemm.py",
    "tensorrt_llm/_torch/modules/low_m_mma_gemm.py",
    "tensorrt_llm/_torch/kernels/blackwell/low_m_mma_gemm.py",
}
_BLACKWELL_UTILS = "tensorrt_llm/_torch/kernels/blackwell/utils.py"
_SHARED_PROVIDERS = (_PRIMS_PACKAGE, _SAGE_PACKAGE, _LOW_M_RUNNER)
_SHARED_VG_KERNELS = "tensorrt_llm/_torch/visual_gen/cute_dsl_kernels/"
_SOL_PREDICTOR_KERNEL = "tensorrt_llm/_torch/visual_gen/attention/backends/sparse/sol/kernels.py"
_H3_GEMM_PROVIDERS = frozenset(
    {
        "tensorrt_llm/_torch/custom_ops/cute_dsl_custom_ops.py",
        "tensorrt_llm/_torch/custom_ops/cutedsl_matmul_heuristics.py",
        "tensorrt_llm/_torch/kernels/blackwell/custom_pipeline.py",
        "tensorrt_llm/_torch/kernels/blackwell/dense_gemm_persistent.py",
        _BLACKWELL_UTILS,
        "tensorrt_llm/_torch/kernels/rubin/dense_bf16_gemm_persistent.py",
        "tensorrt_llm/_torch/kernels/rubin/dense_gemm_persistent.py",
        "tensorrt_llm/_torch/kernels/utils/gemm/sm100.py",
    }
)
_H3_QUANTIZED_GEMM_PROVIDERS = frozenset(
    {
        "tensorrt_llm/_torch/custom_ops/torch_custom_ops.py",
        "tensorrt_llm/_torch/kernels/blackwell/fp8_block_scale_low_m_gemm.py",
        "tensorrt_llm/quantization/utils/fp4_utils.py",
        "tensorrt_llm/quantization/utils/fp8_quantize.py",
        "tensorrt_llm/quantization/utils/fp8_utils.py",
    }
)
_SCHEMA_TEST_TARGET = "unittest/llmapi/test_trtllm_serve_config_schemas.py"
# VisualGen native libraries build with the Modeling V2 CMake helpers and export map.
_SHARED_NATIVE_BUILD = frozenset(
    {
        "cpp/cmake/modules/modeling_v2.cmake",
        "cpp/tensorrt_llm/modeling_v2/common/familyLibrary.version",
    }
)
_NATIVE_CMAKE_TEST = "unittest/_torch/visual_gen/test_native_cmake.py"

# These packages expose shared protocol and media dependencies to serving.
_VG_OUTWARD_PREFIXES: tuple[str, ...] = (
    "tensorrt_llm/media/",
    "tensorrt_llm/visual_gen/",
)

# Keep complete serving families: protocol/media consumers include normal,
# disaggregated, multimodal, and benchmark requests, not just server startup.
_SHARED_SERVING_PREFIXES = (
    "test_e2e.py",
    "examples/serve",
    "disaggregated",
    "llmapi",
    "unittest/llmapi",
    "unittest/disaggregated",
)
_PUBLIC_CPU_CHECKS = (
    "unittest/api_stability",
    _SCHEMA_TEST_TARGET,
    "unittest/llmapi/apps/test_serve_lazy_import.py",
    "unittest/others/test_lazy_model_zoo.py",
)
_PUBLIC_SERVING_CHECKS = (
    "test_e2e.py::test_openai_chat_example",
    "test_e2e.py::test_openai_completions_example",
    "examples/serve/test_serve.py::test_config_override_precedence",
)


def _covers_check(entry: str, check: str) -> bool:
    """Require an unfiltered test entry covering the complete validation target."""
    options = entry.split()[1:]
    if any(
        option == "SKIP"
        or option.split("=", 1)[0] in {"-k", "-m", "--ignore", "--ignore-glob", "--deselect"}
        for option in options
    ):
        return False
    target = _entry_target(entry)
    return bool(target and _target_in_filter_subtree(check, target))


# Substrings that mark a test entry as VG. VG tests are expected to live
# under dedicated visual_gen test directories, except the perf-sanity
# frontend which stays with the shared perf tests.
_VG_ENTRY_PATTERNS: tuple[str, ...] = (
    "visual_gen/",
    "perf/test_visual_gen_perf_sanity.py",
)


def _is_vg_claim(path: str) -> bool:
    """Decide whether VisualGenRule claims `path`.

    Documentation files are excluded so docs-only PRs (e.g.
    `examples/visual_gen/README.md`) don't force VG stages; `DocsRule`
    routes them to the docs build instead. Image suffixes are
    intentionally NOT excluded: VG ships reference images that are
    loaded as test fixtures (e.g. `examples/visual_gen/cat_piano.png`
    referenced by `tests/unittest/_torch/visual_gen/`), so edits to
    them must still force VG stages.
    """
    if path == _VG_SCHEMA_PATH:
        return True
    if not path.startswith(_VG_SRC_PREFIXES):
        return False
    if is_docs_path(path):
        return False
    return True


def _entry_is_vg(entry: str) -> bool:
    return any(p in entry for p in _VG_ENTRY_PATTERNS)


def _vg_entries(block) -> list[str]:
    return [t for t in block.tests if _entry_is_vg(t)]


def _catalog_module(path: str) -> str:
    return _module_name(PurePosixPath(path).relative_to("tensorrt_llm"), "tensorrt_llm")


def _external_provider(path: str) -> Optional[str]:
    if path.startswith(_PRIMS_PACKAGE.replace(".", "/") + "/"):
        return _PRIMS_PACKAGE
    if path == _SAGE_PACKAGE.replace(".", "/") + ".py":
        return _SAGE_PACKAGE
    if path in _LOW_M_RUNNER_SOURCES or path == _BLACKWELL_UTILS:
        return _LOW_M_RUNNER
    return None


def _native_catalog_category(path: str) -> Optional[str]:
    if path.startswith(_VG_CATALOG):
        rel = PurePosixPath(path.removeprefix(_VG_CATALOG))
        if len(rel.parts) >= 3 and rel.parts[1] == "csrc" and rel.suffix in {".cu", ".cpp"}:
            return rel.parts[0]
    return None


class VisualGenFamilyScope:
    """Source dependencies and mirrored test paths for a routed VisualGen family."""

    def __init__(self, root: Path, family: Path):
        self.source = family
        self.family = family.name
        self.unit = f"{_VG_UNIT}/models/{self.family}"
        self.quality = f"{_VG_QUALITY}/{self.family}"
        self.unit_root = root / "tests" / self.unit
        self.catalog_roots = {
            f"{PACKAGE}.catalog": root / _CATALOG,
            _VG_CATALOG_PACKAGE: root / _VG_CATALOG,
        }
        self.package = f"{_VG_V2_PACKAGE}.models.{self.family}"
        tree = _parse(family / "pipeline_registry.py")
        self.targets = routed_targets(tree, function_name="route") if tree else None

    @cached_property
    def graph(self) -> _ImportGraph:
        return _ImportGraph(
            self.source,
            self.unit_root,
            _parse,
            package=self.package,
            catalog_roots=self.catalog_roots,
            provider_packages=_SHARED_PROVIDERS,
        )

    def _target_tests(self, target: str) -> set[str]:
        return {f"{self.unit}/{target}", f"{self.quality}/{target}"}

    def select(self, path: str, provider: Optional[str] = None) -> set[str]:
        whole = {self.unit, self.quality}
        catalog = provider is not None or path.startswith((_CATALOG, _VG_CATALOG))
        if catalog:
            category = _native_catalog_category(path)
            if not self.targets or path == _CATALOG + "_op.py":
                return whole
            if provider:
                module = provider
            elif category:
                module = f"{_VG_CATALOG_PACKAGE}.{category}"
            elif path.endswith(".py"):
                module = _catalog_module(path)
            else:
                return whole
            prefixes = set()
        else:
            rel = PurePosixPath(path.removeprefix(f"{_MODELS}{self.family}/"))
            if self.targets and rel.parts[0] in self.targets:
                return self._target_tests(rel.parts[0])
            if rel.name == "pipeline_registry.py" or rel.suffix != ".py" or not self.targets:
                return whole
            module = _module_name(rel, self.package)
            prefixes = set()
            if len(rel.parts) > 1 and (self.unit_root / rel.parent).is_dir():
                prefixes.add(f"{self.unit}/{rel.parent}")
            else:
                prefixes.update(f"{self.unit}/{p.name}" for p in self.unit_root.glob("test_*.py"))
        if not self.graph.complete:
            return whole
        sources, tests = self.graph.importers(module)
        for src in sources:
            if src.parts[0] in self.targets:
                prefixes |= self._target_tests(src.parts[0])
        prefixes.update(f"{self.unit}/{test}" for test in tests)
        if not catalog and not any(p.startswith(self.quality + "/") for p in prefixes):
            return whole
        return prefixes


class VisualGenRule(Rule):
    name = "visualgen"
    needs_diff_for: tuple[str, ...] = ()

    def __init__(
        self, yaml_index: YAMLIndex, stages: dict[str, Stage], repo_root: Optional[Path] = None
    ) -> None:
        self.root = repo_root
        self.families = (
            {
                f"{_MODELS}{routing.parent.name}/": VisualGenFamilyScope(repo_root, routing.parent)
                for routing in (repo_root / _MODELS).glob("*/pipeline_registry.py")
            }
            if repo_root is not None
            else {}
        )
        self.yaml_index = yaml_index
        self._stages_by_yaml = stages_by_yaml_stem(stages)

    @cached_property
    def catalog_graph(self) -> _ImportGraph:
        assert self.root is not None
        return _ImportGraph(
            self.root / _VG_CATALOG,
            self.root / "tests" / _VG_CATALOG_UNIT,
            _parse,
            package=_VG_CATALOG_PACKAGE,
            catalog_roots={f"{PACKAGE}.catalog": self.root / _CATALOG},
            provider_packages=_SHARED_PROVIDERS,
        )

    def _catalog_tests(self, path: str, provider: Optional[str] = None) -> set[str]:
        category = _native_catalog_category(path)
        if not self.catalog_graph.complete or (
            not provider and not category and not path.endswith(".py")
        ):
            return {_VG_CATALOG_UNIT}
        if category:
            _, tests = self.catalog_graph.importers(f"{_VG_CATALOG_PACKAGE}.{category}")
            return {f"{_VG_CATALOG_UNIT}/{test}" for test in tests | {PurePosixPath(category)}}
        _, tests = self.catalog_graph.importers(provider or _catalog_module(path))
        if path.startswith(_VG_CATALOG):
            rel = PurePosixPath(path.removeprefix(_VG_CATALOG))
            if len(rel.parts) == 2 and not rel.name.startswith("_"):
                tests.add(rel.with_name(f"test_{rel.name}"))
        return {f"{_VG_CATALOG_UNIT}/{test}" for test in tests}

    def _native_build_tests(self) -> Optional[set[str]]:
        """Select the native CMake test and the consumers of every VisualGen native source.

        Returns None when a native source belongs to a family without a routing module.
        """
        assert self.root is not None
        selected = {_NATIVE_CMAKE_TEST}
        source_root = self.root / _VG_V2
        sources = [*source_root.glob("**/csrc/**/*.cu"), *source_root.glob("**/csrc/**/*.cpp")]
        for source in sorted(sources):
            path = source.relative_to(self.root).as_posix()
            if path.startswith(_VG_CATALOG):
                selected |= self._catalog_tests(path)
                for family in self.families.values():
                    selected |= family.select(path)
                continue
            family = next(
                (scope for prefix, scope in self.families.items() if path.startswith(prefix)),
                None,
            )
            if family is None:
                return None
            selected |= family.select(path)
        return selected

    def apply(self, pr: PRInputs) -> Optional[RuleResult]:
        claimed: set[str] = set()
        selected: set[str] = set()
        all_vg = False
        shared: set[str] = set()
        for path in pr.changed_files:
            provider = _external_provider(path)
            if path in _H3_GEMM_PROVIDERS or path in _H3_QUANTIZED_GEMM_PROVIDERS:
                # torch.ops dispatch carries no Python import edge to the registered GEMM.
                shared.add(path)
                h3 = self.families.get(_MODELS + "minimax_h3/")
                if h3:
                    ops = (
                        ("quantized_gemm",)
                        if path in _H3_QUANTIZED_GEMM_PROVIDERS
                        else ("cute_bf16_gemm", "quantized_gemm")
                    )
                    for op in ops:
                        selected |= h3.select(path, f"{h3.package}.ops.{op}")
                else:
                    all_vg = True
            if path in _SHARED_NATIVE_BUILD:
                claimed.add(path)
                native = None if self.root is None else self._native_build_tests()
                if native is None:
                    all_vg = True
                else:
                    selected |= native
            elif (provider or path.startswith((_VG_CATALOG, _CATALOG))) and not is_docs_path(path):
                if provider:
                    # Leave this path in the coverage residual for its non-VisualGen consumers.
                    shared.add(path)
                if self.root is None:
                    if provider:
                        all_vg = True
                    if path.startswith(_VG_CATALOG):
                        return RuleResult(
                            handled_files={path},
                            affected_stages=set(),
                            scope=None,
                            reason="visualgen: repository unavailable for catalog consumer selection",
                        )
                    continue
                consumers = self._catalog_tests(path, provider)
                for scope in self.families.values():
                    consumers |= scope.select(path, provider)
                if consumers or path.startswith(_VG_CATALOG):
                    selected |= consumers
                    if not provider:
                        claimed.add(path)
            elif _is_vg_claim(path):
                claimed.add(path)
                family = next(
                    (scope for prefix, scope in self.families.items() if path.startswith(prefix)),
                    None,
                )
                if family is not None:
                    selected |= family.select(path)
                elif path.startswith(f"{_VG_V2}/") and self.root is not None:
                    selected.update((_VG_UNIT, _VG_QUALITY))
                else:
                    all_vg = True
        if not claimed and not shared:
            return None

        outward = {f for f in claimed if f.startswith(_VG_OUTWARD_PREFIXES)}

        for prefix, family in self.families.items():
            if (family.unit_root / "test_dependencies.py").is_file() and any(
                path.endswith(".py") and path.startswith((_VG_CATALOG, prefix)) for path in claimed
            ):
                selected.add(f"{family.unit}/test_dependencies.py")

        block_filters: dict[tuple[str, int], dict[str, set[str]]] = {}
        required_checks = {}
        if _VG_SCHEMA_PATH in claimed:
            required_checks[_SCHEMA_TEST_TARGET] = "schema"
        if shared or any(
            path.startswith((_VG_CATALOG, _SHARED_VG_KERNELS))
            or path == _SOL_PREDICTOR_KERNEL
            or (path.startswith(tuple(self.families)) and "/ops/" in path)
            for path in claimed
        ):
            required_checks[_CATALOG_CONTRACT_TEST] = "catalog contract"
        check_filters_by_target: dict[str, dict[tuple[str, int], dict[str, set[str]]]] = {
            check: {} for check in required_checks
        }
        for block in self.yaml_index.blocks:
            entries = _vg_entries(block)
            if not all_vg:
                entries = _kept_entries(entries, selected)
            for check, filters in check_filters_by_target.items():
                for entry in block.tests:
                    if _covers_check(entry, check):
                        key = (block.yaml_stem, block.block_index)
                        filters.setdefault(key, {}).setdefault(_entry_target(entry), set()).add(
                            entry
                        )
            if not entries:
                continue
            key = (block.yaml_stem, block.block_index)
            prefix_dict = block_filters.setdefault(key, {})
            for entry in entries:
                target = _entry_target(entry)
                if target:
                    prefix_dict.setdefault(target, set()).add(entry)

        if not block_filters:
            # Defensive: VG source changed but no VG blocks exist in any
            # yaml. Don't fabricate stages — fall back to baseline so
            # the change still gets coverage.
            return RuleResult(
                handled_files=claimed,
                affected_stages=set(),
                scope=None,
                reason=(
                    f"visualgen: {len(claimed)} VG source file(s); "
                    "no VG block matched in any test-db yaml — fallback"
                ),
            )

        affected = resolve_affected_stages(block_filters, self.yaml_index, self._stages_by_yaml)
        if outward:
            shared_filters = block_filters_for_stem(self.yaml_index, CPU_TEST_YAML_STEM)
            required = set(_PUBLIC_CPU_CHECKS) | set(_PUBLIC_SERVING_CHECKS)
            for check in sorted(required):
                check_filters: dict[tuple[str, int], dict[str, set[str]]] = {}
                for block in self.yaml_index.blocks:
                    if check in _PUBLIC_CPU_CHECKS and block.yaml_stem != CPU_TEST_YAML_STEM:
                        continue
                    for entry in block.tests:
                        if _covers_check(entry, check):
                            key = (block.yaml_stem, block.block_index)
                            check_filters.setdefault(key, {}).setdefault(
                                _entry_target(entry), set()
                            ).add(entry)
                if not resolve_affected_stages(
                    check_filters, self.yaml_index, self._stages_by_yaml
                ):
                    return RuleResult(
                        handled_files=claimed,
                        affected_stages=set(),
                        scope=None,
                        reason=f"visualgen: required shared validation unavailable: {check}",
                    )
            for block in self.yaml_index.blocks:
                for entry in block.tests:
                    target = _entry_target(entry)
                    if any(
                        _target_in_filter_subtree(target, prefix)
                        for prefix in _SHARED_SERVING_PREFIXES
                    ):
                        key = (block.yaml_stem, block.block_index)
                        shared_filters.setdefault(key, {}).setdefault(target, set()).add(entry)
            for key, prefixes in shared_filters.items():
                for target, entries in prefixes.items():
                    block_filters.setdefault(key, {}).setdefault(target, set()).update(entries)
            affected |= resolve_affected_stages(
                shared_filters, self.yaml_index, self._stages_by_yaml
            )
        for check, filters in check_filters_by_target.items():
            check_stages = resolve_affected_stages(filters, self.yaml_index, self._stages_by_yaml)
            if not check_stages:
                return RuleResult(
                    handled_files=claimed,
                    affected_stages=set(),
                    scope=None,
                    reason=f"visualgen: {required_checks[check]} validation stages unavailable — fallback",
                )
            affected |= check_stages
            for key, prefixes in filters.items():
                for target, entries in prefixes.items():
                    block_filters.setdefault(key, {}).setdefault(target, set()).update(entries)
        perfsanity_relevant = bool(outward) or any(is_perf_stem(stem) for stem, _ in block_filters)

        return RuleResult(
            handled_files=claimed,
            affected_stages=affected,
            scope="visualgenonly",
            block_filters=block_filters,
            sanity_relevant=bool(outward),
            perfsanity_relevant=perfsanity_relevant,
            reason=(
                f"visualgen: {len(claimed)} VG source file(s), {len(shared)} shared provider file(s) → "
                f"{len(block_filters)} block(s), {len(affected)} stage(s)"
                + ("; shared serving + complete CPU validation" if outward else "")
            ),
        )
