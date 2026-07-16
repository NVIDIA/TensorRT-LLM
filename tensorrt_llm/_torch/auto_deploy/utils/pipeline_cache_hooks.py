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

"""Markers for AutoDeploy load hooks that need pipeline-cache reconstruction."""

import importlib
import json
from collections.abc import Mapping
from typing import Any

PIPELINE_CACHE_HOOK_SPEC_ATTR = "_auto_deploy_pipeline_cache_spec"


def _resolve_qualified_attr(module_name: str, qualname: str) -> Any:
    obj: Any = importlib.import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def callable_ref(func: Any) -> dict[str, str] | None:
    """Return an importable reference for a callable, or ``None`` if it is local."""
    module_name = getattr(func, "__module__", None)
    qualname = getattr(func, "__qualname__", None)
    if not isinstance(module_name, str) or not isinstance(qualname, str):
        return None
    if "<locals>" in qualname:
        return None

    try:
        resolved = _resolve_qualified_attr(module_name, qualname)
    except (AttributeError, ImportError, ValueError):
        return None
    if resolved is not func:
        return None
    return {"module": module_name, "qualname": qualname}


def json_dict(value: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return a JSON-serializable copy of ``value``, or ``None``."""
    result = dict(value)
    try:
        return json.loads(json.dumps(result))
    except (TypeError, ValueError):
        return None


def json_instance_payload(obj: Any) -> dict[str, Any]:
    """Return JSON-serializable public attributes needed to rebuild a hook owner."""
    payload: dict[str, Any] = {}
    for attr_name, value in getattr(obj, "__dict__", {}).items():
        if attr_name.startswith("_"):
            continue
        if callable(value):
            continue
        try:
            value = json.loads(json.dumps(value))
        except (TypeError, ValueError):
            continue
        payload[attr_name] = value
    return payload


def mark_pipeline_cache_hook(hook: Any, spec: Mapping[str, Any]) -> Any:
    """Attach an explicit pipeline-cache reconstruction spec to a load hook."""
    setattr(hook, PIPELINE_CACHE_HOOK_SPEC_ATTR, dict(spec))
    return hook


def get_pipeline_cache_hook_spec(hook: Any) -> dict[str, Any] | None:
    """Return the hook's explicit pipeline-cache spec, if one was attached."""
    spec = getattr(hook, PIPELINE_CACHE_HOOK_SPEC_ATTR, None)
    if spec is None:
        return None
    return dict(spec)
