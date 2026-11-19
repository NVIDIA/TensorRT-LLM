# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multimodal registrations and the target classes their routes return."""

from __future__ import annotations

import ast
import inspect
from collections import defaultdict
from dataclasses import dataclass
from functools import cache
from importlib import import_module
from importlib.util import resolve_name
from pathlib import Path

from tensorrt_llm._torch.modeling_v2._router_index import NoTarget, registered_routes
from tensorrt_llm._torch.models._arch_index import MODEL_ARCH_TO_MODULE, is_builtin_zoo_module
from tensorrt_llm._torch.models.modeling_mistral import Mistral3VLM, MistralNativeInputProcessor
from tensorrt_llm._torch.models.modeling_mllama import MllamaForConditionalGeneration
from tensorrt_llm._torch.models.modeling_multimodal_mixin import MultimodalModelMixin
from tensorrt_llm._torch.models.modeling_utils import (
    MODEL_CLASS_MAPPING,
    MODEL_CLASS_VISION_ENCODER_MAPPING,
)
from tensorrt_llm.inputs.registry import INPUT_PROCESSOR_REGISTRY

__extra_import_path__ = ["~/jenkins/scripts/cbts/cbts/rules"]
from modeling_v2_source import route_returns  # noqa: E402

Key = tuple[str, str | None]
_ROOT = Path(__file__).resolve().parents[5]
_BASES = {"MultimodalForConditionalGeneration", "ModelingV2ForConditionalGeneration"}
TINY_CONFIGS: dict[Key, str] = {}
_B300_TARGETS = {
    "qwen4_exp/qwen3_8_flash_next_fp8__sm_103__unified",
    "qwen4_exp/qwen3_8_flash_next_nvfp4__sm_103__default",
}


@dataclass(frozen=True)
class ModelEntry:
    model_cls: type | None
    processor_cls: type | None = None
    fixture: str | None = None
    media: bool = True
    class_name: str = ""
    # A text-only target defines validate_multimodal_request_data and raises in it.
    refuses_media: bool = False
    # A registered tower of the architecture declares an encoder graph modality.
    graph_tower: bool = False

    @property
    def on_base(self) -> bool:
        return self.model_cls is not None and any(
            cls.__name__ in _BASES for cls in self.model_cls.__mro__
        )

    @property
    def marked(self) -> bool:
        return getattr(self.model_cls, "supports_multimodal", False) is True

    @property
    def processor_classes(self) -> tuple[type, ...]:
        processors = (self.processor_cls,) if self.processor_cls else ()
        if self.model_cls is Mistral3VLM:
            return (*processors, MistralNativeInputProcessor)
        return processors


def _returns(source: str, where: str) -> dict[str, str]:
    """Map each routed directory to its class, without choosing a deployment."""
    fn = ast.parse(source)
    imports = defaultdict(set)
    for node in ast.walk(fn):
        if isinstance(node, ast.ImportFrom) and node.level == 1 and node.module:
            for alias in node.names:
                imports[alias.asname or alias.name].add((node.module, alias.name))
    targets = {}
    for node, directory in route_returns(fn):
        origins = imports.get(node.value.id, set()) if isinstance(node.value, ast.Name) else set()
        assert directory and len(origins) == 1, f"{where}:{node.lineno}: ambiguous target return"
        ((module, name),) = origins
        assert module == f"{directory}.modeling", f"{where}: return a target modeling class"
        assert directory not in targets or targets[directory] == name, (
            f"{where}: different classes returned from {directory}"
        )
        targets[directory] = name
    assert targets, f"{where}: media route has no target return"
    return targets


@cache
def _source(module: str) -> tuple[dict[str, ast.ClassDef], dict[str, tuple[str, str]]]:
    path = _ROOT / (module.replace(".", "/") + ".py")
    body = ast.parse(path.read_text()).body
    imports = {
        alias.asname or alias.name: (
            resolve_name("." * node.level + node.module, module.rpartition(".")[0]),
            alias.name,
        )
        for node in body
        if isinstance(node, ast.ImportFrom) and node.module
        for alias in node.names
    }
    for node in body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name if alias.asname else alias.name.partition(".")[0]
                imports[alias.asname or module_name] = ("", module_name)
    return {node.name: node for node in body if isinstance(node, ast.ClassDef)}, imports


def _contract_in_source(module: str, name: str) -> bool:
    """Follow local and imported base classes to the marker, including mixin adapters."""
    if name in _BASES or name == "MultimodalModelMixin":
        return True
    if (
        not module.startswith("tensorrt_llm.")
        or not (_ROOT / (module.replace(".", "/") + ".py")).is_file()
    ):
        return False
    classes, imports = _source(module)
    cls = classes.get(name)
    if cls is None:
        return False
    if any(
        isinstance(node, ast.Name) and node.id == "supports_multimodal"
        for statement in cls.body
        if isinstance(statement, (ast.Assign, ast.AnnAssign))
        for node in ast.walk(statement)
    ):
        return True
    for base in cls.bases:
        if isinstance(base, ast.Subscript):
            base = base.value
        if isinstance(base, ast.Name):
            origin, parent = imports.get(base.id, (module, base.id))
        elif isinstance(base, ast.Attribute):
            head, _, tail = ast.unparse(base.value).partition(".")
            package, imported = imports.get(head, ("", head))
            origin = f"{package}.{imported}.{tail}".strip(".")
            parent = base.attr
        else:
            continue
        if _contract_in_source(origin, parent):
            return True
    return False


def _target(module: str, name: str) -> tuple[type | None, bool, bool]:
    """The class when it declares the contract, whether it takes media, and whether it
    refuses media at admission."""
    if _contract_in_source(module, name):
        try:
            cls = getattr(import_module(module), name)
        except NoTarget as exc:
            raise AssertionError(
                f"{module}.{name}: expose the feature contract to CPU tests"
            ) from exc
        media = (
            any(base.__name__ in _BASES for base in cls.__mro__)
            or getattr(cls, "supports_multimodal", False) is True
        )
        return cls, media, False
    # Some targets run the tower and splice its embeddings in their own forward.
    methods = {
        method.name: method
        for method in _source(module)[0][name].body
        if isinstance(method, ast.FunctionDef)
    }
    media = any(
        isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "multimodal_embeds"
        for node in ast.walk(methods.get("forward", ast.Pass()))
    )
    hook = methods.get("validate_multimodal_request_data")
    refuses = hook is not None and any(isinstance(node, ast.Raise) for node in ast.walk(hook))
    return None, media, refuses


@cache
def inventory() -> dict[Key, ModelEntry]:
    # Importing every provider also runs MiniMax's processor factory.
    for module in sorted(set(MODEL_ARCH_TO_MODULE.values())):
        import_module(f"tensorrt_llm._torch.models.{module}")
    processors = INPUT_PROCESSOR_REGISTRY._input_processors_cls_by_model_type
    keys = {
        arch
        for arch, cls in MODEL_CLASS_MAPPING.items()
        if is_builtin_zoo_module(cls.__module__)
        and (
            cls in processors
            or issubclass(cls, MultimodalModelMixin)
            or cls in (Mistral3VLM, MllamaForConditionalGeneration)
        )
    }
    entries: dict[Key, ModelEntry] = {}
    routes = registered_routes()
    jobs = _ROOT / "tests/integration/test_lists/test-db/l0_b300.yml"
    listed = {
        line.split()[1] for line in jobs.read_text().splitlines() if line.lstrip().startswith("- ")
    }
    for arch in sorted(keys):
        cls = MODEL_CLASS_MAPPING[arch]
        processor = processors.get(cls)
        towers = MODEL_CLASS_VISION_ENCODER_MAPPING.get(arch) or ()
        entries[arch, None] = ModelEntry(
            cls,
            processor,
            TINY_CONFIGS.get((arch, None)),
            class_name=cls.__name__,
            graph_tower=any(
                getattr(tower, "encoder_cuda_graph_modality", None) for tower in towers
            ),
        )
        if not entries[arch, None].processor_classes or arch not in routes:
            continue
        route = routes[arch]
        family = route.__module__.rsplit(".", 2)[-2]
        for directory, name in _returns(inspect.getsource(route), route.__module__).items():
            target = f"{family}/{directory}"
            module = resolve_name(f".{directory}.modeling", route.__module__.rpartition(".")[0])
            target_cls, media, refuses = _target(module, name)
            fixture = TINY_CONFIGS.get((arch, target))
            job = f"unittest/_torch/modeling_v2/models/{target}"
            if target in _B300_TARGETS and job in listed:
                fixture = f"l0_b300.yml: {job}"
            entries[arch, target] = ModelEntry(
                target_cls, processor, fixture, media, name, refuses_media=refuses
            )
    _source.cache_clear()
    return entries


def inventory_or_none() -> dict[Key, ModelEntry] | None:
    """The inventory for test collection, or None when it cannot be built.

    A test then calls ``inventory()`` and reports the error as one failure, so the error
    does not stop the collection of the other multimodal tests.
    """
    try:
        return inventory()
    except Exception:
        return None
