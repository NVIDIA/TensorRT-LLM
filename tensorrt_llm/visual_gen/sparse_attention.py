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
"""Skip-softmax sparse attention config for visual generation.

Scalar ``threshold_scale_factor`` / ``target_sparsity`` (no prefill/decode
split), plus calibration (formula, per-layer disables, per-component
sub-configs) loaded from ModelOpt artifacts into private attributes.
Shared calibration helpers live in
:mod:`tensorrt_llm._torch.attention_backend.sparse.skip_softmax`.
"""

import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Literal, Optional

import yaml
from pydantic import Field as PydanticField
from pydantic import PrivateAttr

from tensorrt_llm.llmapi.utils import StrictBaseModel

if TYPE_CHECKING:
    import torch

    from tensorrt_llm._torch.attention_backend.sparse.skip_softmax import SkipSoftmaxFormula


class BaseSparseAttentionConfig(StrictBaseModel):
    """Base for visual-generation sparse attention configs.

    Each algorithm subclasses this and pins a unique ``algorithm``
    discriminator for the config union.
    """

    algorithm: str


class SkipSoftmaxAttentionConfig(BaseSparseAttentionConfig):
    """SkipSoftmax sparse attention configuration for visual generation.

    User-facing surface:

    - ``threshold_scale_factor`` — raw scalar value, resolution-dependent.
    - ``target_sparsity`` — semantic target in ``[0, 1]``; needs a
      calibration formula to resolve.
    - ``warmup`` — reserved knob for future warmup-step counting. Has no
      runtime effect today; included so existing YAML/Python configs do
      not break when the wiring lands.

    Calibration state (formula coefficients, per-layer overrides, and
    per-component sub-configs) is loaded from ModelOpt artifacts
    (``sparse.yaml`` or the checkpoint's ``config.json``) into private
    attributes — it is *not* settable via the user-facing constructor
    or YAML config. To attach calibration in loaders/tests, use
    :meth:`SkipSoftmaxAttentionConfig.with_calibration`.
    """

    algorithm: Literal["skip_softmax"] = "skip_softmax"
    threshold_scale_factor: Optional[float] = PydanticField(
        default=None,
        description="Raw per-block threshold; takes precedence over target_sparsity.",
    )
    target_sparsity: Optional[float] = PydanticField(
        default=None,
        ge=0.0,
        le=1.0,
        description="Semantic target sparsity in [0, 1]; requires a calibration formula.",
    )
    warmup: Optional[int] = PydanticField(
        default=None,
        ge=0,
        description="Reserved: number of warmup steps before skip-softmax engages. "
        "No runtime effect yet.",
    )

    _formula: Optional["SkipSoftmaxFormula"] = PrivateAttr(default=None)
    _layer_overrides: Optional[Dict[str, float]] = PrivateAttr(default=None)
    _component_configs: Optional[Dict[str, "SkipSoftmaxAttentionConfig"]] = PrivateAttr(
        default=None
    )
    _resolved_threshold: Optional[float] = PrivateAttr(default=None)

    @classmethod
    def with_calibration(
        cls,
        *,
        threshold_scale_factor: Optional[float] = None,
        target_sparsity: Optional[float] = None,
        warmup: Optional[int] = None,
        formula: Optional["SkipSoftmaxFormula"] = None,
        layer_overrides: Optional[Dict[str, float]] = None,
        component_configs: Optional[Dict[str, "SkipSoftmaxAttentionConfig"]] = None,
    ) -> "SkipSoftmaxAttentionConfig":
        """Internal factory used by YAML/checkpoint loaders and tests."""
        kwargs: Dict[str, Any] = {}
        if threshold_scale_factor is not None:
            kwargs["threshold_scale_factor"] = threshold_scale_factor
        if target_sparsity is not None:
            kwargs["target_sparsity"] = target_sparsity
        if warmup is not None:
            kwargs["warmup"] = warmup
        cfg = cls(**kwargs)
        cfg._formula = formula
        cfg._layer_overrides = layer_overrides
        cfg._component_configs = component_configs
        return cfg

    def _with_public_overrides(
        self, user_cfg: "SkipSoftmaxAttentionConfig"
    ) -> "SkipSoftmaxAttentionConfig":
        """Copy user-facing sparse knobs onto calibration loaded from ModelOpt."""
        updates = {
            k: v
            for k, v in {
                "threshold_scale_factor": user_cfg.threshold_scale_factor,
                "target_sparsity": user_cfg.target_sparsity,
                "warmup": user_cfg.warmup,
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

    def to_kernel_params(self):
        """Resolve to the kernel-facing ``SkipSoftmaxKernelParams``.

        Visual generation has no decode phase: the resolved scalar
        threshold is the prefill value and decode stays 0.
        """
        from tensorrt_llm._torch.attention_backend.sparse.skip_softmax import (
            SkipSoftmaxKernelParams,
        )

        return SkipSoftmaxKernelParams(
            threshold_scale_factor_prefill=self.threshold_scale_factor,
            threshold_scale_factor_decode=0.0,
        )

    def get_or_resolve_threshold(
        self,
        checkpoint_formula: Optional[Dict[str, Any]] = None,
    ) -> Optional[float]:
        """Return the resolved scalar threshold, caching after the first call."""
        if self._resolved_threshold is not None:
            return self._resolved_threshold
        if (
            self._component_configs
            and self._formula is None
            and self.threshold_scale_factor is None
        ):
            return None
        threshold = self.resolve_threshold_scale_factor(checkpoint_formula)
        if threshold is not None and threshold > 0:
            self._resolved_threshold = threshold
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
    ) -> Optional[tuple["SkipSoftmaxAttentionConfig", str]]:
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
        checkpoint_formula: Optional[Dict[str, Any]] = None,
    ) -> Optional[float]:
        """Resolve to a concrete scalar threshold via shared formula helpers.

        Resolution order: user-supplied ``threshold_scale_factor`` (raw,
        wins) → ``target_sparsity`` + attached calibration ``_formula``
        → ``target_sparsity`` + runtime ``checkpoint_formula`` dict.
        """
        if self.threshold_scale_factor is not None:
            return self.threshold_scale_factor

        sparsity = self.target_sparsity
        if sparsity is None:
            return None

        from tensorrt_llm._torch.attention_backend.sparse.skip_softmax import (
            parse_skip_softmax_formula_from_dict,
        )

        formula = self._formula
        if formula is None and checkpoint_formula is not None:
            formula = parse_skip_softmax_formula_from_dict(checkpoint_formula)
        if formula is None:
            raise ValueError(
                "SkipSoftmaxAttentionConfig: target_sparsity requires calibration formula "
                "coefficients. Provide via a ModelOpt sparse YAML "
                "(sparse_config_path) or a checkpoint config.json carrying "
                "calibrated coefficients."
            )
        return formula.compute_threshold_scale_factor(sparsity)


def _load_sparse_config_group_container(
    data: Dict[str, Any],
) -> Optional[SkipSoftmaxAttentionConfig]:
    """Load one component's skip-softmax config from a ``config_groups`` container."""
    config_groups = data.get("config_groups", {})
    if not isinstance(config_groups, dict):
        return None

    from tensorrt_llm._torch.attention_backend.sparse.skip_softmax import (
        parse_skip_softmax_formula_from_dict,
    )

    for group in config_groups.values():
        if not isinstance(group, dict) or group.get("sparse_algo") != "softmax_skip":
            continue

        tsf = group.get("threshold_scale_factor", {})
        coefficients = tsf.get("coefficients") if isinstance(tsf, dict) else None
        formula_str = tsf.get("formula") if isinstance(tsf, dict) else None
        formula = parse_skip_softmax_formula_from_dict(coefficients, formula=formula_str)
        if formula is None:
            continue

        disabled = group.get("disabled_layers", [])
        if not isinstance(disabled, list):
            disabled = []
        layer_overrides: Optional[Dict[str, float]] = (
            {str(name): 0.0 for name in disabled} if disabled else None
        )

        return SkipSoftmaxAttentionConfig.with_calibration(
            formula=formula,
            layer_overrides=layer_overrides,
        )

    return None


def load_sparse_config_from_yaml(yaml_path: str) -> Optional[SkipSoftmaxAttentionConfig]:
    """Load SkipSoftmaxAttentionConfig from a ModelOpt sparse attention YAML file."""
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
        return SkipSoftmaxAttentionConfig.with_calibration(component_configs=component_configs)

    return None


def auto_detect_sparse_yaml(checkpoint_dir: str) -> Optional[SkipSoftmaxAttentionConfig]:
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
) -> Optional[SkipSoftmaxAttentionConfig]:
    """Auto-detect sparse attention calibration from ModelOpt checkpoint config.json.

    Visual generation uses a phase-neutral ``coefficients`` key under
    ``sparse_attention_config.threshold_scale_factor`` (diffusion has
    no prefill / decode distinction).
    """
    from tensorrt_llm._torch.attention_backend.sparse.skip_softmax import (
        parse_skip_softmax_formula_from_ckpt_config,
    )

    formula = parse_skip_softmax_formula_from_ckpt_config(
        checkpoint_config, coefficient_key="coefficients"
    )
    if formula is None:
        return None
    return SkipSoftmaxAttentionConfig.with_calibration(formula=formula)


def apply_skip_softmax_overrides(
    model: "torch.nn.Module", skip_softmax: SkipSoftmaxAttentionConfig
) -> int:
    """Apply component-specific skip-softmax calibration to constructed TRTLLM backends.

    For each attention backend VG stores a per-layer
    :class:`SkipSoftmaxAttentionConfig` carrying that layer's resolved
    threshold (or ``None`` to disable). The backend later converts it
    via ``to_kernel_params()``.
    """
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
                    threshold_scale_factor=threshold
                )
            else:
                target.sparse_attention_config = None
            modified += 1

    return modified
