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
from importlib.util import find_spec, resolve_name
from pathlib import Path

from tensorrt_llm._torch.modeling_v2._router_index import NoTarget, registered_routes
from tensorrt_llm._torch.models._arch_index import MODEL_ARCH_TO_MODULE, is_builtin_zoo_module
from tensorrt_llm._torch.models.modeling_mistral import Mistral3VLM, MistralNativeInputProcessor
from tensorrt_llm._torch.models.modeling_mllama import MllamaForConditionalGeneration
from tensorrt_llm._torch.models.modeling_multimodal_mixin import MultimodalModelMixin
from tensorrt_llm._torch.models.modeling_utils import MODEL_CLASS_MAPPING
from tensorrt_llm.inputs.registry import INPUT_PROCESSOR_REGISTRY

__extra_import_path__ = ["~/jenkins/scripts/cbts/cbts/rules"]
from modeling_v2_source import route_returns  # noqa: E402

Key = tuple[str, str | None]
_ROOT = Path(__file__).resolve().parents[5]
# #22450 supplies this base. A V1 wrapper or a standalone V2 tower can also
# inherit the mixin without using the V2 composite.
_V2_MODULE = "tensorrt_llm._torch.modeling_v2.multimodal"
_V2_BASE = import_module(_V2_MODULE).MultimodalModelForCausalLM if find_spec(_V2_MODULE) else None
TINY_CONFIGS: dict[Key, str] = {}
_MINIMAX_DEP = "minimax_m3/minimax_m3_1_flash_nvfp4__sm_107__dep"
_B300_TARGETS = {
    "qwen4_exp/qwen3_8_flash_next_fp8__sm_10x__default",
    "qwen4_exp/qwen3_8_flash_next_nvfp4__sm_10x__default",
}


@dataclass(frozen=True)
class ModelEntry:
    model_cls: type | None
    processor_cls: type | None = None
    fixture: str | None = None
    media: bool = True
    class_name: str = ""

    @property
    def on_base(self) -> bool:
        return (
            self.model_cls is not None
            and _V2_BASE is not None
            and issubclass(self.model_cls, _V2_BASE)
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
    if (module, name) == (MultimodalModelMixin.__module__, MultimodalModelMixin.__name__):
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


def _target(module: str, name: str, registered_cls: type | None = None) -> tuple[type | None, bool]:
    """The class when it declares the contract, and whether the routed target serves media."""
    if _contract_in_source(module, name):
        try:
            cls = getattr(import_module(module), name)
        except NoTarget as exc:
            raise AssertionError(
                f"{module}.{name}: expose the feature contract to CPU tests"
            ) from exc
        return cls, issubclass(cls, MultimodalModelMixin) or getattr(
            cls, "supports_multimodal", False
        ) is True
    # Some targets run the tower and splice its embeddings in their own forward.
    classes, imports = _source(module)
    target = classes[name]
    forward = next(
        (
            node
            for node in target.body
            if isinstance(node, ast.FunctionDef) and node.name == "forward"
        ),
        None,
    )
    calls = [node for node in ast.walk(forward or ast.Pass()) if isinstance(node, ast.Call)]
    media = any(getattr(call.func, "attr", None) == "multimodal_embeds" for call in calls)
    # This pre-marker MiniMax target passes its forward kwargs to the registered media model.
    if registered_cls is not None and forward and forward.args.kwarg:
        media = media or (
            any(
                isinstance(base, ast.Name)
                and imports.get(base.id) == (registered_cls.__module__, registered_cls.__name__)
                for base in target.bases
            )
            and any(
                ast.unparse(call.func) == "super().forward"
                and [ast.unparse(kw) for kw in call.keywords] == [f"**{forward.args.kwarg.arg}"]
                for call in calls
            )
        )
    return None, media


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
        entries[arch, None] = ModelEntry(
            cls,
            processor,
            TINY_CONFIGS.get((arch, None)),
            class_name=cls.__name__,
        )
        if not entries[arch, None].processor_classes or arch not in routes:
            continue
        route = routes[arch]
        family = route.__module__.rsplit(".", 2)[-2]
        for directory, name in _returns(inspect.getsource(route), route.__module__).items():
            target = f"{family}/{directory}"
            module = resolve_name(f".{directory}.modeling", route.__module__.rpartition(".")[0])
            target_cls, media = _target(module, name, cls if target == _MINIMAX_DEP else None)
            fixture = TINY_CONFIGS.get((arch, target))
            job = f"unittest/_torch/modeling_v2/models/{target}"
            if target in _B300_TARGETS and job in listed:
                fixture = f"l0_b300.yml: {job}"
            entries[arch, target] = ModelEntry(target_cls, processor, fixture, media, name)
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
