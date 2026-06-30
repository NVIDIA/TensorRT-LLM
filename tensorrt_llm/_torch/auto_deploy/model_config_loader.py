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

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

__all__ = [
    "MODELS_YAML",
    "INTERNAL_CONFIGS_DIR",
    "USER_CONFIGS_DIR",
    "load_registry",
    "find_registry_entry",
    "resolve_registry_yaml_extra",
    "translate_parallel_fields",
    "inject_autodeploy_registry_defaults",
]

# tensorrt_llm/_torch/auto_deploy/model_config_loader.py -> tensorrt_llm/
_PACKAGE_ROOT: Path = Path(__file__).resolve().parents[2]
INTERNAL_REGISTRY_DIR: Path = Path(__file__).resolve().parent / "config" / "model_registry_internal"
INTERNAL_CONFIGS_DIR: Path = INTERNAL_REGISTRY_DIR / "configs"
MODELS_YAML: Path = INTERNAL_REGISTRY_DIR / "models.yaml"
# User configs live under examples/, absent in wheels; skipped when not present.
USER_CONFIGS_DIR: Path = (
    _PACKAGE_ROOT.parent / "examples" / "auto_deploy" / "model_registry" / "configs"
)

# AutoDeploy parallelizes via world_size, so translate the TRT-LLM equivalent.
_PARALLEL_FIELD_TRANSLATION = {"tensor_parallel_size": "world_size"}


def _resolve_config_file(
    filename: str, *, internal_only: bool = False, user_dirs: Optional[List[Path]] = None
) -> Optional[Path]:
    """Find a registry config file in the internal (then user) configs dirs."""
    search_dirs = [INTERNAL_CONFIGS_DIR]
    if not internal_only:
        search_dirs.extend(user_dirs if user_dirs is not None else [USER_CONFIGS_DIR])
    for directory in search_dirs:
        candidate = Path(directory) / filename
        if candidate.is_file():
            return candidate
    return None


def load_registry() -> Dict[str, Any]:
    """Load the registry index. Returns ``{"models": []}`` when absent."""
    if not MODELS_YAML.is_file():
        return {"models": []}
    with MODELS_YAML.open("r") as f:
        return yaml.safe_load(f) or {"models": []}


def find_registry_entry(
    model_name: str, config_id: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Return the registry entry for ``model_name``, or ``None`` if not found.

    Raises ``KeyError`` when multiple entries match and ``config_id`` does not
    disambiguate them.
    """
    if not model_name:
        return None
    registry = load_registry()
    matches = [entry for entry in registry.get("models", []) if entry.get("name") == model_name]

    if config_id is not None:
        matches = [entry for entry in matches if entry.get("config_id", "default") == config_id]
        if not matches:
            raise KeyError(
                f"Model '{model_name}' with config_id '{config_id}' not found "
                f"in the AutoDeploy model registry ({MODELS_YAML})."
            )
    elif len(matches) > 1:
        default_matches = [
            entry for entry in matches if entry.get("config_id", "default") == "default"
        ]
        if len(default_matches) == 1:
            matches = default_matches
        else:
            available = sorted({entry.get("config_id", "default") for entry in matches})
            raise KeyError(
                f"Model '{model_name}' has multiple registry entries with "
                f"config_id values {available}. Provide a config_id to select one."
            )

    return matches[0] if matches else None


def resolve_registry_yaml_extra(
    model_name: str,
    config_id: Optional[str] = None,
    *,
    internal_only: bool = False,
    required: bool = False,
    user_dirs: Optional[List[Path]] = None,
) -> List[str]:
    """Resolve a model's ``yaml_extra`` filenames to absolute config paths.

    ``internal_only`` keeps only AD-internal (package) configs. ``required``
    raises instead of returning ``[]`` when the model is missing. ``user_dirs``
    overrides the default user configs dir, for callers that know where the
    source-tree ``examples/auto_deploy/model_registry/configs`` lives (the
    default is wrong for installed packages, where ``examples/`` is absent).
    """
    entry = find_registry_entry(model_name, config_id)
    if entry is None:
        if required:
            raise KeyError(
                f"Model '{model_name}' not found in the AutoDeploy model registry ({MODELS_YAML})."
            )
        return []

    searched_user_dirs = user_dirs if user_dirs is not None else [USER_CONFIGS_DIR]
    resolved: List[str] = []
    for filename in entry.get("yaml_extra", []):
        path = _resolve_config_file(filename, internal_only=internal_only, user_dirs=user_dirs)
        if path is not None:
            resolved.append(str(path))
        elif required and not internal_only:
            raise FileNotFoundError(
                f"Registry config '{filename}' for model '{model_name}' was not "
                f"found in {INTERNAL_CONFIGS_DIR} or {searched_user_dirs}."
            )
    return resolved


def translate_parallel_fields(args: Dict[str, Any]) -> Dict[str, Any]:
    """Translate TRT-LLM parallel fields to AutoDeploy equivalents (in place)."""
    for src, dst in _PARALLEL_FIELD_TRANSLATION.items():
        value = args.pop(src, None)
        if value is not None:
            args.setdefault(dst, value)
    return args


def inject_autodeploy_registry_defaults(llm_args: Dict[str, Any]) -> Dict[str, Any]:
    """Prepare an AutoDeploy ``llm_args`` dict from the registry, in place.

    Translates parallel fields and prepends the model's hidden AD-internal configs
    to ``yaml_extra`` (lowest priority, so user configs win). Best-effort: a
    registry miss or ambiguity leaves ``llm_args`` unchanged.
    """
    translate_parallel_fields(llm_args)

    model_name = llm_args.get("model")
    if not model_name:
        return llm_args
    try:
        internal = resolve_registry_yaml_extra(model_name, internal_only=True)
    except KeyError:
        internal = []
    if internal:
        existing = list(llm_args.get("yaml_extra", []) or [])
        llm_args["yaml_extra"] = [*internal, *existing]
    return llm_args
