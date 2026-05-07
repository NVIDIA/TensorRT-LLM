# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
"""AutoDeploy model registry for serving.

This registry maps model names to their AutoDeploy-specific default configs.
These configs contain AD-internal knobs (transforms, compile_backend, etc.)
that are auto-injected when serving with ``--backend _autodeploy``, so users
only need to supply a config YAML with common knobs (max_batch_size,
kv_cache_config, etc.).
"""

from importlib.resources import files
from pathlib import Path
from typing import Dict, List, Optional

import yaml

_REGISTRY_PKG = files(__package__)
_REGISTRY_YAML = _REGISTRY_PKG / "models.yaml"
_REGISTRY_CONFIGS_DIR = Path(str(_REGISTRY_PKG / "configs"))


def get_serve_ad_defaults(
    model_name: str,
    config_id: Optional[str] = None,
) -> List[str]:
    """Look up a model in the registry and return its AD-specific default config paths.

    These configs contain AutoDeploy-internal settings (transforms, compile_backend,
    etc.) and are meant to be injected as ``yaml_extra`` when serving with
    ``--backend _autodeploy``.

    Args:
        model_name: HuggingFace model id (e.g. ``deepseek-ai/DeepSeek-R1-0528``).
        config_id: Optional config variant selector when a model has multiple entries.

    Returns:
        List of absolute paths to the AD default yaml config files.

    Raises:
        KeyError: If the model is not found or is ambiguous without ``config_id``.
    """
    registry_text = _REGISTRY_YAML.read_text()
    registry = yaml.safe_load(registry_text)

    matches = [entry for entry in registry.get("models", []) if entry.get("name") == model_name]

    if config_id is not None:
        matches = [entry for entry in matches if entry.get("config_id", "default") == config_id]
        if not matches:
            raise KeyError(
                f"Model '{model_name}' with config_id '{config_id}' not found "
                f"in the AutoDeploy serve registry."
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
                f"config_id values {available}. Provide config_id to select one."
            )

    if not matches:
        raise KeyError(f"Model '{model_name}' not found in the AutoDeploy serve registry.")

    selected = matches[0]
    return [str(_REGISTRY_CONFIGS_DIR / cfg) for cfg in selected.get("ad_defaults", [])]


def list_registered_models() -> List[Dict]:
    """Return all models in the serve registry."""
    registry_text = _REGISTRY_YAML.read_text()
    registry = yaml.safe_load(registry_text)
    return registry.get("models", [])
