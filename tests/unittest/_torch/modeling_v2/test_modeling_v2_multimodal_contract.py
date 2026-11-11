# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Every target of a multimodal checkpoint runs the shared multimodal runtime or is text only.

``modeling_v2_resolve`` enforces this when a deployment resolves a target. The source check
below enforces it for every target a ``route`` can return, without importing target code or
needing a GPU, so a target that would fail at resolution fails here first.
"""

from __future__ import annotations

import ast
import importlib.util
from pathlib import Path

import pytest
import torch
from transformers import PretrainedConfig

from tensorrt_llm._torch._experimental.modeling_v2 import _router_index
from tensorrt_llm._torch._experimental.modeling_v2._multimodal import (
    TextOnlyTarget,
    check_target_multimodal_contract,
    checkpoint_accepts_media,
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.multimodal.model import (
    MultimodalModelContractError,
    MultimodalRuntimeMixin,
)
from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

pytestmark = pytest.mark.cpu_only

_REPO = Path(__file__).resolve().parents[4]
_MODELS = _REPO / "tensorrt_llm/_torch/_experimental/modeling_v2/models"
_MULTIMODAL_ARCH = "Qwen4ExpForConditionalGeneration"
_TEXT_ARCH = "GptOssForCausalLM"


def _load_source_parser():
    path = _REPO / "jenkins/scripts/cbts/cbts/rules/modeling_v2_source.py"
    spec = importlib.util.spec_from_file_location("_cbts_modeling_v2_source", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_source = _load_source_parser()


def _routed_architectures(tree: ast.AST) -> list[str]:
    """The ``architectures[0]`` values the module's ``@routes(...)`` decorators name."""
    names = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        for decorator in node.decorator_list:
            if (
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Name)
                and decorator.func.id == "routes"
            ):
                names.extend(arg.value for arg in decorator.args if isinstance(arg, ast.Constant))
    return names


def _routed_target_classes() -> list[tuple[str, Path, ast.ClassDef]]:
    """``(architecture, modeling.py, class)`` for every target class a ``route`` returns."""
    found = []
    for routing in sorted(_MODELS.glob("*/routing.py")):
        tree = ast.parse(routing.read_text())
        architectures = _routed_architectures(tree)
        for node, target in _source.route_returns(tree):
            if target is None or not isinstance(node.value, ast.Name):
                continue
            modeling = routing.parent / target / "modeling.py"
            classes = {
                cls.name: cls
                for cls in ast.parse(modeling.read_text()).body
                if isinstance(cls, ast.ClassDef)
            }
            found.extend(
                (architecture, modeling, classes[node.value.id]) for architecture in architectures
            )
    return found


def _base_names(cls: ast.ClassDef) -> set[str]:
    names = set()
    for base in cls.bases:
        node = base.value if isinstance(base, ast.Subscript) else base
        if isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Name):
            names.add(node.id)
    return names


def _defined_members(cls: ast.ClassDef) -> set[str]:
    return {
        node.name for node in cls.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_route_returns_are_discovered():
    classes = {cls.name for _, _, cls in _routed_target_classes()}
    assert "ModelingV2Qwen38FlashNextNvfp4Sm103Default" in classes
    assert "ModelingV2GptOss120bSm10xDefault" in classes


def test_every_target_of_a_multimodal_checkpoint_declares_how_it_serves_media():
    problems = []
    for architecture, modeling, cls in _routed_target_classes():
        bases = _base_names(cls)
        runs_runtime = MultimodalRuntimeMixin.__name__ in bases
        text_only = TextOnlyTarget.__name__ in bases
        where = f"{modeling.relative_to(_REPO)}::{cls.name}"
        if runs_runtime and text_only:
            problems.append(f"{where} inherits both MultimodalRuntimeMixin and TextOnlyTarget")
        elif runs_runtime:
            missing = set(MultimodalRuntimeMixin.REQUIRED_MULTIMODAL_MEMBERS) - _defined_members(
                cls
            )
            if missing:
                problems.append(f"{where} does not implement {sorted(missing)}")
        elif not text_only and checkpoint_accepts_media(architecture):
            problems.append(f"{where} serves {architecture}, which accepts media")
    assert not problems, "\n".join(problems)


def test_checkpoint_media_detection_reads_the_registered_input_processor():
    assert checkpoint_accepts_media(_MULTIMODAL_ARCH)
    assert not checkpoint_accepts_media(_TEXT_ARCH)
    assert not checkpoint_accepts_media("NotARegisteredArchitecture")


class _Plain:
    pass


class _TextOnly(TextOnlyTarget):
    pass


class _Runtime(MultimodalRuntimeMixin):
    def encode_multimodal_inputs(self, multimodal_params):
        return torch.zeros(0, 1)

    @property
    def embedding_dim(self) -> int:
        return 1

    @property
    def embedding_dtype(self) -> torch.dtype:
        return torch.float32


class _IncompleteRuntime(MultimodalRuntimeMixin):
    def encode_multimodal_inputs(self, multimodal_params):
        return torch.zeros(0, 1)


class _Both(_Runtime, TextOnlyTarget):
    pass


@pytest.mark.parametrize(
    "architecture,target",
    [
        (_TEXT_ARCH, _Plain),
        (_MULTIMODAL_ARCH, _TextOnly),
        (_MULTIMODAL_ARCH, _Runtime),
        (_TEXT_ARCH, _Runtime),
    ],
)
def test_conforming_targets_pass(architecture, target):
    check_target_multimodal_contract(architecture, target)


@pytest.mark.parametrize(
    "target,match",
    [
        (_Plain, "must inherit MultimodalRuntimeMixin to serve media, or TextOnlyTarget"),
        (_IncompleteRuntime, "does not implement embedding_dim, embedding_dtype"),
        (_Both, "inherits both"),
    ],
)
def test_nonconforming_targets_fail(target, match):
    with pytest.raises(MultimodalModelContractError, match=match):
        check_target_multimodal_contract(_MULTIMODAL_ARCH, target)


@pytest.mark.parametrize("key", ["image", "video", "audio", "multimodal_embedding"])
def test_text_only_targets_refuse_media_at_admission(key):
    with pytest.raises(ValueError, match=f"language model only.*carries {key}"):
        _TextOnly().validate_multimodal_request_data({key: object()})


@pytest.mark.parametrize(
    "mm_data", [{"mrope_config": {"mrope_position_deltas": object()}}, {"image": None}]
)
def test_text_only_targets_accept_payloads_without_media(mm_data):
    _TextOnly().validate_multimodal_request_data(mm_data)


@pytest.mark.parametrize("mode", ["auto", "require"])
def test_resolution_rejects_a_nonconforming_target_in_every_mode(monkeypatch, mode):
    """A contract violation is a bug in the target, so ``auto`` does not fall back on it."""
    _router_index.registered_routes()
    monkeypatch.setitem(_router_index._ROUTES, _MULTIMODAL_ARCH, lambda *args: _Plain)
    monkeypatch.setattr(_router_index, "_device_compute_capability", lambda: (10, 3))
    config = ModelConfig(pretrained_config=PretrainedConfig(architectures=[_MULTIMODAL_ARCH]))
    llm_args = TorchLlmArgs(model="/tmp/dummy_model", modeling_v2=mode)
    with pytest.raises(MultimodalModelContractError, match="_Plain"):
        _router_index.modeling_v2_resolve(config, llm_args)
