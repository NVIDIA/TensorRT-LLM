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

"""Load hook serialization helpers for the AutoDeploy pipeline cache."""

import importlib
import types
from collections.abc import Callable
from functools import partial
from typing import Any

import torch
import torch.nn as nn

from ...utils.logger import ad_logger
from ...utils.pipeline_cache_hooks import (
    callable_ref,
    get_pipeline_cache_hook_spec,
    json_dict,
    json_instance_payload,
)


def _resolve_qualified_attr(module_name: str, qualname: str) -> Any:
    obj: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _identify_importable_hook(
    hook: Callable, scope: str, target_module: nn.Module
) -> dict[str, Any] | None:
    def identify_bound_method(method: Callable) -> dict[str, Any] | None:
        owner = getattr(method, "__self__", None)
        func = getattr(method, "__func__", None)
        if owner is None or func is None:
            return None

        ref = callable_ref(func)
        if ref is None:
            return None

        spec = {
            "type": "importable_load_hook",
            "scope": scope,
            "callable": ref,
        }
        if owner is target_module:
            spec["bind_to_module"] = True
            return spec

        spec["owner_payload"] = json_instance_payload(owner)
        owner_class_ref = callable_ref(type(owner))
        if owner_class_ref is not None:
            spec["owner_class"] = owner_class_ref
        return spec

    if isinstance(hook, partial):
        if hook.args:
            return None
        keywords = json_dict(hook.keywords or {})
        if keywords is None:
            return None

        spec = identify_bound_method(hook.func)
        if spec is None:
            ref = callable_ref(hook.func)
            if ref is None:
                return None
            spec = {
                "type": "importable_load_hook",
                "scope": scope,
                "callable": ref,
            }
        spec["keywords"] = keywords
        return spec

    spec = identify_bound_method(hook)
    if spec is not None:
        return spec

    ref = callable_ref(hook)
    if ref is None:
        return None
    return {
        "type": "importable_load_hook",
        "scope": scope,
        "callable": ref,
    }


def _identify_hook(
    hook: Any,
    scope: str = "root",
    target_module: nn.Module | None = None,
    *,
    phase: str = "pre",
    with_module: bool = False,
) -> dict[str, Any] | None:
    spec = get_pipeline_cache_hook_spec(hook)
    if spec is not None:
        if spec.get("type") not in _HOOK_REBUILDERS:
            return None
        spec["scope"] = scope
    else:
        if target_module is None:
            raise ValueError("target_module must be provided when identifying importable hooks.")
        spec = _identify_importable_hook(hook, scope, target_module)
        if spec is None:
            return None

    spec.setdefault("phase", phase)
    if spec["phase"] == "pre":
        spec.setdefault("with_module", with_module)
    return spec


def collect_hook_specs(model: nn.Module) -> tuple[list[dict[str, Any]], bool]:
    """Collect declarative hook specs from load hooks on ``model`` and children."""
    specs: list[dict[str, Any]] = []

    def log_unknown(hook_obj: Any) -> None:
        qualname = getattr(hook_obj, "__qualname__", repr(hook_obj))
        ad_logger.warning(f"Pipeline cache: unrecognized hook of type {qualname}")

    def collect_from_module(mod: nn.Module, scope: str) -> bool:
        for hook in mod._load_state_dict_pre_hooks.values():
            with_module = bool(getattr(hook, "with_module", False))
            hook_obj = hook.hook if hasattr(hook, "hook") else hook
            spec = _identify_hook(hook_obj, scope, mod, phase="pre", with_module=with_module)
            if spec is None:
                log_unknown(hook_obj)
                return False
            specs.append(spec)

        for hook_obj in mod._load_state_dict_post_hooks.values():
            hook = hook_obj.hook if hasattr(hook_obj, "hook") else hook_obj
            spec = _identify_hook(hook, scope, mod, phase="post")
            if spec is None:
                log_unknown(hook)
                return False
            specs.append(spec)

        return True

    if not collect_from_module(model, "root"):
        return [], True
    for name, child in model.named_modules():
        if name and not collect_from_module(child, name):
            return [], True

    return specs, False


def _rebuild_shard_tp_hook(spec: dict[str, Any]) -> Callable:
    from ..library.sharding import _load_hook, _split_tensor_for_tp

    dim = spec["dim"]
    rank = spec["rank"]
    world_size = spec["world_size"]
    min_local_shape = spec["min_local_shape"]
    fused_weight_dims = spec.get("fused_weight_dims")
    if fused_weight_dims:

        def f_split(
            t: torch.Tensor, fused_dims: list = fused_weight_dims, d: int = dim
        ) -> torch.Tensor:
            return torch.cat(
                [
                    _split_tensor_for_tp(w, dim, rank, world_size, min_local_shape)
                    for w in torch.split(t, fused_dims, dim=d)
                ],
                dim=d,
            )

    else:
        f_split = partial(
            _split_tensor_for_tp,
            dim=dim,
            rank=rank,
            world_size=world_size,
            min_local_shape=min_local_shape,
        )
    return partial(
        _load_hook,
        f_split=f_split,
        param_key=spec["param_key"],
        param_shape=torch.Size(spec["param_shape"]),
    )


def _rebuild_dedup_hook(spec: dict[str, Any]) -> Callable:
    from ...export.export import _load_hook_for_deduplication

    return partial(
        _load_hook_for_deduplication,
        param_key_remaining=spec["param_key_remaining"],
        param_key_removed=spec["param_key_removed"],
    )


def _rebuild_alias_hook(spec: dict[str, Any]) -> Callable:
    from ...export.export import _build_aliasing_load_pre_hook

    return _build_aliasing_load_pre_hook(spec["aliased_groups"])


def _rebuild_hook_owner(spec: dict[str, Any]) -> object:
    owner_payload = spec.get("owner_payload", {})
    owner_class = spec.get("owner_class")
    if owner_class is None:
        return types.SimpleNamespace(**owner_payload)

    owner_cls = _resolve_qualified_attr(owner_class["module"], owner_class["qualname"])
    # Avoid running model or transform constructors when rebuilding load-hook owners.
    owner = owner_cls.__new__(owner_cls)
    for attr_name, value in owner_payload.items():
        try:
            owner.__dict__[attr_name] = value
        except AttributeError:
            setattr(owner, attr_name, value)
    return owner


def _rebuild_importable_hook(
    spec: dict[str, Any], target_module: nn.Module | None = None
) -> Callable:
    ref = spec["callable"]
    hook_fn = _resolve_qualified_attr(ref["module"], ref["qualname"])

    if bool(spec.get("bind_to_module")):
        if target_module is None:
            raise ValueError("Pipeline cache: module-bound hook restore requires target module.")
        hook_fn = types.MethodType(hook_fn, target_module)
    elif "owner_payload" in spec:
        owner_class = spec.get("owner_class")
        if target_module is not None and owner_class is not None:
            owner_cls = _resolve_qualified_attr(owner_class["module"], owner_class["qualname"])
            if isinstance(target_module, owner_cls):
                hook_fn = types.MethodType(hook_fn, target_module)
            else:
                hook_fn = types.MethodType(hook_fn, _rebuild_hook_owner(spec))
        else:
            hook_fn = types.MethodType(hook_fn, _rebuild_hook_owner(spec))

    keywords = spec.get("keywords", {}) or {}
    if keywords:
        hook_fn = partial(hook_fn, **keywords)
    return hook_fn


_HOOK_REBUILDERS = {
    "alias": _rebuild_alias_hook,
    "dedup": _rebuild_dedup_hook,
    "importable_load_hook": _rebuild_importable_hook,
    "shard_tp": _rebuild_shard_tp_hook,
}


def _rebuild_hook(spec: dict[str, Any], target_module: nn.Module | None = None) -> Callable:
    hook_type = spec["type"]
    rebuilder = _HOOK_REBUILDERS.get(hook_type)
    if rebuilder is not None:
        if hook_type == "importable_load_hook":
            return rebuilder(spec, target_module)
        return rebuilder(spec)
    raise ValueError(f"Pipeline cache: unknown hook spec type {hook_type!r}")


def reattach_hooks(model: nn.Module, specs: list[dict[str, Any]]) -> None:
    """Rebuild and register AD-managed load hooks on ``model``."""
    for spec in specs:
        scope = spec["scope"]
        target_mod = model if scope == "root" else model.get_submodule(scope)
        hook_fn = _rebuild_hook(spec, target_mod)
        phase = spec.get("phase", "pre")
        if phase == "pre":
            target_mod._register_load_state_dict_pre_hook(
                hook_fn, with_module=bool(spec.get("with_module", False))
            )
        elif phase == "post":
            target_mod.register_load_state_dict_post_hook(hook_fn)
        else:
            raise ValueError(f"Pipeline cache: unknown hook phase {phase!r}")


def snapshot_and_clear_load_hooks(model: nn.Module) -> list[tuple[nn.Module, Any, Any]]:
    records: list[tuple[nn.Module, Any, Any]] = []
    for module in model.modules():
        records.append(
            (
                module,
                module._load_state_dict_pre_hooks.copy(),
                module._load_state_dict_post_hooks.copy(),
            )
        )
        module._load_state_dict_pre_hooks.clear()
        module._load_state_dict_post_hooks.clear()
    return records


def restore_load_hooks(records: list[tuple[nn.Module, Any, Any]]) -> None:
    for module, pre_hooks, post_hooks in records:
        module._load_state_dict_pre_hooks.clear()
        module._load_state_dict_pre_hooks.update(pre_hooks)
        module._load_state_dict_post_hooks.clear()
        module._load_state_dict_post_hooks.update(post_hooks)
