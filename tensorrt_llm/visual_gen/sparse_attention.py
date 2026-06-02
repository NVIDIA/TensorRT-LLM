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
"""Skip-softmax sparse attention helpers for visual generation."""

import fnmatch
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import yaml
from pydantic import Field as PydanticField
from pydantic import PrivateAttr, model_validator

from tensorrt_llm.llmapi.llm_args import SkipSoftmaxAttentionConfig
from tensorrt_llm.llmapi.utils import StrictBaseModel

if TYPE_CHECKING:
    import torch


class SkipSoftmaxFormula(StrictBaseModel):
    """Exponential calibration formula: threshold = exp(log_a + b * sparsity).

    Equivalent to: threshold = a * exp(b * sparsity) where a = exp(log_a).
    Stored in log-space (log_a) to match ModelOpt diffusion format and
    avoid precision loss. Accepts either 'log_a' (diffusion format) or 'a'
    (LLM format) at construction; 'a' is normalized to log_a = log(a).
    """

    log_a: float = PydanticField(description="Log of coefficient a (log-space)")
    b: float = PydanticField(description="Coefficient b")

    @model_validator(mode="before")
    @classmethod
    def _accept_linear_a(cls, values):
        """Normalize LLM-format 'a' to diffusion-format 'log_a'."""
        if not isinstance(values, dict) or "a" not in values:
            return values
        if "log_a" in values:
            raise ValueError(
                "SkipSoftmaxFormula: specify either 'log_a' (diffusion format) "
                "or 'a' (LLM format), not both."
            )
        a = values["a"]
        if a <= 0:
            raise ValueError(
                f"SkipSoftmaxFormula: 'a' must be positive (got {a}). "
                "Use 'log_a' directly if you need log(a) of a non-positive value."
            )
        values = {**values}
        values["log_a"] = math.log(a)
        values.pop("a")
        return values


class SkipSoftmaxConfig(SkipSoftmaxAttentionConfig):
    """SkipSoftmax sparse attention configuration for visual generation.

    Extends the shared :class:`SkipSoftmaxAttentionConfig` from
    ``tensorrt_llm.llmapi.llm_args`` (used by the LLM backend) without adding
    any user-facing fields. The user-facing surface is exactly:

    - ``threshold_scale_factor`` — raw value, resolution-dependent.
    - ``target_sparsity`` — semantic target; needs calibration to resolve.

    Calibration state (formula coefficients and per-layer overrides) is loaded
    from ModelOpt-produced artifacts (``sparse.yaml`` or checkpoint
    ``config.json``) and stored in private attributes — it is *not* settable
    via the user-facing constructor or YAML config. Users wanting custom
    calibration should author a ModelOpt sparse YAML and point
    :attr:`AttentionConfig.sparse_config_path` at it.

    To attach calibration in code (loaders / tests), use
    :meth:`SkipSoftmaxConfig.with_calibration`.
    """

    _formula: Optional[SkipSoftmaxFormula] = PrivateAttr(default=None)
    _layer_overrides: Optional[Dict[str, float]] = PrivateAttr(default=None)
    _component_configs: Optional[Dict[str, "SkipSoftmaxConfig"]] = PrivateAttr(default=None)
    _resolved_threshold_prefill: Optional[float] = PrivateAttr(default=None)

    @classmethod
    def with_calibration(
        cls,
        *,
        threshold_scale_factor: Optional[Union[float, Dict[str, float]]] = None,
        target_sparsity: Optional[Union[float, Dict[str, float]]] = None,
        formula: Optional[SkipSoftmaxFormula] = None,
        layer_overrides: Optional[Dict[str, float]] = None,
        component_configs: Optional[Dict[str, "SkipSoftmaxConfig"]] = None,
    ) -> "SkipSoftmaxConfig":
        """Internal factory used by YAML/checkpoint loaders and tests."""
        kwargs = {}
        if threshold_scale_factor is not None:
            kwargs["threshold_scale_factor"] = threshold_scale_factor
        if target_sparsity is not None:
            kwargs["target_sparsity"] = target_sparsity
        cfg = cls(**kwargs)
        cfg._formula = formula
        cfg._layer_overrides = layer_overrides
        cfg._component_configs = component_configs
        return cfg

    def _with_public_overrides(self, user_cfg: "SkipSoftmaxConfig") -> "SkipSoftmaxConfig":
        """Copy user-facing sparse knobs onto calibration loaded from ModelOpt."""
        updates = {
            k: v
            for k, v in {
                "threshold_scale_factor": user_cfg.threshold_scale_factor,
                "target_sparsity": user_cfg.target_sparsity,
            }.items()
            if v is not None
        }
        if not updates:
            return self

        merged = self.model_copy(update=updates)
        if self._component_configs:
            merged._component_configs = {
                name: component.model_copy(update=updates)
                for name, component in self._component_configs.items()
            }
        return merged

    def get_or_resolve_threshold(
        self,
        checkpoint_formula: Optional[Dict[str, float]] = None,
    ) -> Optional[float]:
        """Return the resolved prefill threshold, caching after the first call."""
        if self._resolved_threshold_prefill is not None:
            return self._resolved_threshold_prefill
        if (
            self._component_configs
            and self._formula is None
            and self.threshold_scale_factor_prefill is None
        ):
            return None
        threshold = self.resolve_threshold_scale_factor(checkpoint_formula)
        if threshold is not None and threshold > 0:
            self._resolved_threshold_prefill = threshold
        return threshold

    def resolve_threshold(self, module_name: str) -> Optional[float]:
        """Resolve the threshold for a specific layer by module name."""
        if self._component_configs:
            component = self._component_config_for_module_name(module_name)
            if component is None:
                return None
            component_cfg, relative_name = component
            return component_cfg.resolve_threshold(relative_name)

        threshold = self.get_or_resolve_threshold()
        if threshold is None:
            return None
        if self._layer_overrides:
            candidate_names = self._layer_override_match_names(module_name)
            for pattern, override in self._layer_overrides.items():
                if any(fnmatch.fnmatch(name, pattern) for name in candidate_names):
                    threshold = override
                    break
        return threshold if threshold > 0 else None

    @staticmethod
    def _layer_override_match_names(module_name: str) -> tuple[str, ...]:
        """Return full and component-relative names for ModelOpt override matching."""
        candidate_names = {module_name, module_name.replace("._orig_mod.", ".")}
        for name in tuple(candidate_names):
            for prefix in ("transformer.", "transformer_2."):
                if name.startswith(prefix):
                    candidate_names.add(name[len(prefix) :])
        return tuple(candidate_names)

    def _component_config_for_module_name(
        self, module_name: str
    ) -> Optional[tuple["SkipSoftmaxConfig", str]]:
        if not self._component_configs:
            return None
        normalized_name = module_name.replace("._orig_mod.", ".")
        for component_name, component_cfg in sorted(
            self._component_configs.items(), key=lambda item: len(item[0]), reverse=True
        ):
            prefix = f"{component_name}."
            if normalized_name == component_name:
                return component_cfg, ""
            if normalized_name.startswith(prefix):
                return component_cfg, normalized_name[len(prefix) :]
        return None

    def resolve_threshold_scale_factor(
        self,
        checkpoint_formula: Optional[Dict[str, float]] = None,
    ) -> Optional[float]:
        """Resolve to a concrete prefill threshold using the shared LLM resolver."""
        if self.threshold_scale_factor_prefill is not None:
            return self.threshold_scale_factor_prefill

        sparsity = self.target_sparsity_prefill
        if sparsity is None:
            return None

        formula = self._shared_formula(checkpoint_formula)
        if formula is None:
            raise ValueError(
                "SkipSoftmaxConfig: target_sparsity requires calibration formula "
                "coefficients. Provide via a ModelOpt sparse YAML "
                "(sparse_config_path) or a checkpoint config.json carrying "
                "calibrated coefficients."
            )
        resolved = SkipSoftmaxAttentionConfig(
            algorithm=self.algorithm,
            target_sparsity=self.target_sparsity,
        ).resolve_for_target_sparsity(formula)
        return resolved.threshold_scale_factor_prefill

    def _shared_formula(
        self,
        checkpoint_formula: Optional[Dict[str, float]],
    ) -> Optional[Dict[str, Dict[str, float]]]:
        if self._formula:
            coeffs = {"a": math.exp(self._formula.log_a), "b": self._formula.b}
        elif checkpoint_formula:
            coeffs = self._shared_formula_coefficients(checkpoint_formula)
        else:
            return None
        if coeffs is None:
            return None
        return {"prefill": coeffs, "decode": coeffs}

    @staticmethod
    def _shared_formula_coefficients(
        formula: Dict[str, float],
    ) -> Optional[Dict[str, float]]:
        if "log_a" in formula and "b" in formula:
            return {"a": math.exp(formula["log_a"]), "b": formula["b"]}
        if "a" in formula and "b" in formula:
            return {"a": formula["a"], "b": formula["b"]}
        return None


def _load_sparse_config_group_container(data: Dict[str, Any]) -> Optional[SkipSoftmaxConfig]:
    """Load one component's skip-softmax config from a ``config_groups`` container."""
    config_groups = data.get("config_groups", {})
    if not isinstance(config_groups, dict):
        return None

    for group in config_groups.values():
        if not isinstance(group, dict) or group.get("sparse_algo") != "softmax_skip":
            continue

        tsf = group.get("threshold_scale_factor", {})
        prefill = tsf.get("prefill", {}) if isinstance(tsf, dict) else {}
        if "b" not in prefill or ("log_a" not in prefill and "a" not in prefill):
            continue

        disabled = group.get("disabled_layers", [])
        if not isinstance(disabled, list):
            disabled = []
        layer_overrides: Optional[Dict[str, float]] = (
            {str(name): 0.0 for name in disabled} if disabled else None
        )

        formula_kwargs = {k: prefill[k] for k in ("log_a", "a", "b") if k in prefill}
        return SkipSoftmaxConfig.with_calibration(
            formula=SkipSoftmaxFormula(**formula_kwargs),
            layer_overrides=layer_overrides,
        )

    return None


def load_sparse_config_from_yaml(yaml_path: str) -> Optional[SkipSoftmaxConfig]:
    """Load SkipSoftmaxConfig from a ModelOpt sparse attention YAML file."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        return None

    cfg = _load_sparse_config_group_container(data)
    if cfg is not None:
        return cfg

    component_configs = {}
    for component_name, component_data in data.items():
        if not isinstance(component_data, dict):
            continue
        cfg = _load_sparse_config_group_container(component_data)
        if cfg is not None:
            component_configs[component_name] = cfg

    if component_configs:
        return SkipSoftmaxConfig.with_calibration(component_configs=component_configs)

    return None


def auto_detect_sparse_yaml(checkpoint_dir: str) -> Optional[SkipSoftmaxConfig]:
    """Auto-detect the current consolidated ModelOpt sparse YAML at checkpoint root."""
    checkpoint_path = Path(checkpoint_dir)
    candidates = [checkpoint_path / "sparse.yaml"]
    candidates.extend(checkpoint_path.glob("sparse.*.yaml"))
    candidates = sorted({path for path in candidates if path.is_file()})
    if not candidates:
        return None
    if len(candidates) > 1:
        raise ValueError(
            "auto_detect_sparse_yaml: multiple sparse YAML files found at "
            f"{checkpoint_path}: {candidates}. Pass an explicit `sparse_config_path` "
            "to disambiguate."
        )
    return load_sparse_config_from_yaml(str(candidates[0]))


def auto_detect_sparse_attention_config(
    checkpoint_config: Dict[str, Any],
) -> Optional[SkipSoftmaxConfig]:
    """Auto-detect sparse attention calibration from ModelOpt checkpoint config.json."""
    sparse_cfg = checkpoint_config.get("sparse_attention_config")
    if not isinstance(sparse_cfg, dict):
        return None

    tsf = sparse_cfg.get("threshold_scale_factor")
    if not isinstance(tsf, dict):
        return None

    prefill = tsf.get("prefill")
    if not isinstance(prefill, dict):
        return None

    if "log_a" in prefill and "b" in prefill:
        return SkipSoftmaxConfig.with_calibration(
            formula=SkipSoftmaxFormula(log_a=prefill["log_a"], b=prefill["b"]),
        )
    if "a" in prefill and "b" in prefill:
        return SkipSoftmaxConfig.with_calibration(
            formula=SkipSoftmaxFormula(log_a=math.log(prefill["a"]), b=prefill["b"]),
        )

    return None


def apply_skip_softmax_overrides(model: "torch.nn.Module", skip_softmax: SkipSoftmaxConfig) -> int:
    """Apply component-specific skip-softmax calibration to constructed TRTLLM backends."""
    if skip_softmax._layer_overrides is None and skip_softmax._component_configs is None:
        return 0

    from tensorrt_llm._torch.visual_gen.attention_backend.trtllm import TrtllmAttention

    modified = 0
    for name, module in model.named_modules():
        threshold = skip_softmax.resolve_threshold(name)
        attn = getattr(module, "attn", None)
        targets = []
        if isinstance(attn, TrtllmAttention):
            targets.append(attn)
        inner = getattr(attn, "inner_backend", None)
        if isinstance(inner, TrtllmAttention):
            targets.append(inner)

        for target in targets:
            if threshold is not None:
                target.sparse_attention_config = SkipSoftmaxAttentionConfig(
                    threshold_scale_factor={"prefill": threshold, "decode": 0}
                )
            else:
                target.sparse_attention_config = None
            modified += 1

    return modified
