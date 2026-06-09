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
"""Skip-softmax calibration helpers shared by the LLM and VisualGen pipelines.

The kernel consumes a scalar ``threshold_scale_factor`` (combined with
the sequence length at runtime) to decide which KV blocks to skip.
This module owns the calibration side that produces that scalar from a
semantic ``target_sparsity`` via a formula shipped with the checkpoint
(the formula model and checkpoint-parsing helpers), plus the
kernel-facing :class:`SkipSoftmaxKernelParams` carrier that the shared
``TrtllmAttention`` reads. All of it is shared by the LLM and VisualGen
pipelines.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Union

import numexpr
from pydantic import ConfigDict, model_validator
from pydantic import Field as PydanticField

from tensorrt_llm.llmapi.utils import StrictBaseModel

from .params import SparseParams

_RESERVED_FORMULA_KEYS = frozenset({"formula", "target_sparsity"})
_SKIP_SOFTMAX_ALGORITHMS = frozenset({"skip_softmax", "softmax_skip"})


def _sparse_attention_config_from_checkpoint(
    checkpoint_config: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not isinstance(checkpoint_config, dict):
        return None

    sparse_cfg = checkpoint_config.get("sparse_attention_config")
    if isinstance(sparse_cfg, dict):
        return sparse_cfg

    # Some call sites already pass the inner sparse_attention_config dict.
    if any(
        key in checkpoint_config
        for key in (
            "config_groups",
            "threshold_scale_factor",
            "target_sparsity",
            "disabled_layers",
            "ignore",
        )
    ):
        return checkpoint_config
    return None


def _is_skip_softmax_group(group: Dict[str, Any]) -> bool:
    algorithm = group.get("algorithm", group.get("sparse_algo"))
    return algorithm in _SKIP_SOFTMAX_ALGORITHMS


def _looks_like_skip_softmax_group(group: Dict[str, Any]) -> bool:
    return _is_skip_softmax_group(group) or any(
        key in group
        for key in (
            "threshold_scale_factor",
            "target_sparsity",
            "disabled_layers",
            "ignore",
            "initial_disabled_steps",
        )
    )


def skip_softmax_config_groups_from_checkpoint_config(
    checkpoint_config: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Return skip-softmax config groups from checkpoint metadata."""
    sparse_cfg = _sparse_attention_config_from_checkpoint(checkpoint_config)
    if not isinstance(sparse_cfg, dict):
        return []

    config_groups = sparse_cfg.get("config_groups")
    if isinstance(config_groups, dict):
        return [
            group
            for group in config_groups.values()
            if isinstance(group, dict) and _is_skip_softmax_group(group)
        ]

    if _looks_like_skip_softmax_group(sparse_cfg):
        return [sparse_cfg]
    return []


def skip_softmax_threshold_scale_factor_config_from_checkpoint_config(
    checkpoint_config: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Return the first skip-softmax calibration formula config."""
    for group in skip_softmax_config_groups_from_checkpoint_config(checkpoint_config):
        threshold_config = group.get("threshold_scale_factor")
        if isinstance(threshold_config, dict):
            return threshold_config
    return None


def skip_softmax_target_sparsity_from_checkpoint_config(
    checkpoint_config: Optional[Dict[str, Any]],
) -> Optional[Union[float, Dict[str, float]]]:
    """Return checkpoint-provided skip-softmax target sparsity, if present."""
    for group in skip_softmax_config_groups_from_checkpoint_config(checkpoint_config):
        target_sparsity = group.get("target_sparsity")
        if isinstance(target_sparsity, dict):
            return {str(k): float(v) for k, v in target_sparsity.items()}
        if isinstance(target_sparsity, (float, int)):
            return float(target_sparsity)
    return None


def skip_softmax_initial_disabled_steps_from_checkpoint_config(
    checkpoint_config: Optional[Dict[str, Any]],
) -> Optional[int]:
    """Return checkpoint-provided initial step disablement, if present."""
    for group in skip_softmax_config_groups_from_checkpoint_config(checkpoint_config):
        initial_disabled_steps = group.get("initial_disabled_steps")
        if isinstance(initial_disabled_steps, int):
            return initial_disabled_steps
    return None


class SkipSoftmaxFormula(StrictBaseModel):
    """Numexpr formula + coefficients that map target_sparsity to threshold_scale_factor.

    ``formula`` references ``target_sparsity`` and one or more named
    coefficients (e.g. ``"a * exp(b * target_sparsity)"``);
    ``coefficients`` supplies their values.
    """

    # Frozen: validated once at construction and never mutated, so the
    # construction-time formula/coefficient checks always hold.
    # NOTE: ``model_config`` here is the Pydantic model setting (this
    # class's own validation behavior) — unrelated to the LLM runtime's
    # ``ModelConfig`` / model config plumbing.
    model_config = ConfigDict(frozen=True)

    formula: str = PydanticField(
        description="Numexpr formula referencing target_sparsity and named coefficients."
    )
    coefficients: Dict[str, float] = PydanticField(
        description="Coefficient values referenced by the formula."
    )

    @model_validator(mode="after")
    def _validate_formula(self):
        try:
            parsed = numexpr.NumExpr(self.formula)
        except Exception as exc:
            raise ValueError(
                f"SkipSoftmaxFormula: invalid formula {self.formula!r}: {exc}"
            ) from exc

        names = set(parsed.input_names)
        if "target_sparsity" not in names:
            raise ValueError(
                f"SkipSoftmaxFormula: formula {self.formula!r} must reference 'target_sparsity'."
            )

        required = names - {"target_sparsity"}
        missing = required - set(self.coefficients)
        if missing:
            raise ValueError(
                f"SkipSoftmaxFormula: formula {self.formula!r} requires "
                f"coefficients {sorted(required)}; missing: {sorted(missing)}."
            )
        return self

    def compute_threshold_scale_factor(self, target_sparsity: float) -> float:
        """Evaluate the formula at ``target_sparsity`` to get threshold_scale_factor."""
        result = numexpr.evaluate(
            self.formula,
            local_dict={**self.coefficients, "target_sparsity": float(target_sparsity)},
        )
        return float(result.item())

    @classmethod
    def parse_from_dict(
        cls,
        data: Optional[Dict[str, Any]],
        *,
        formula: Optional[str] = None,
    ) -> Optional["SkipSoftmaxFormula"]:
        """Build a formula from a coefficient dict plus a formula string."""
        if not isinstance(data, dict):
            return None
        coefficients = {k: v for k, v in data.items() if k not in _RESERVED_FORMULA_KEYS}
        if not formula:
            nested = data.get("formula")
            if isinstance(nested, str) and nested.strip():
                formula = nested
        if not formula or not coefficients:
            return None
        return cls(
            formula=formula,
            coefficients={k: float(v) for k, v in coefficients.items()},
        )

    @classmethod
    def parse_from_ckpt_config(
        cls,
        checkpoint_config: Dict[str, Any],
        *,
        coefficient_key: str = "prefill",
    ) -> Optional["SkipSoftmaxFormula"]:
        """Lift the calibration formula out of a HuggingFace config dict."""
        tsf = (
            checkpoint_config
            if "formula" in checkpoint_config
            else skip_softmax_threshold_scale_factor_config_from_checkpoint_config(
                checkpoint_config
            )
        )
        if not isinstance(tsf, dict):
            return None
        return cls.parse_from_dict(tsf.get(coefficient_key), formula=tsf.get("formula"))


def skip_softmax_disabled_layers_from_checkpoint_config(
    checkpoint_config: Optional[Dict[str, Any]],
) -> Optional[list[str]]:
    """Read checkpoint-provided layer-disable patterns for skip-softmax."""
    disabled_layers: list[str] = []
    seen: set[str] = set()

    def _extend(patterns: Any) -> None:
        if not isinstance(patterns, list):
            return
        for pattern in patterns:
            pattern = str(pattern)
            if pattern not in seen:
                seen.add(pattern)
                disabled_layers.append(pattern)

    sparse_cfg = _sparse_attention_config_from_checkpoint(checkpoint_config)
    if isinstance(sparse_cfg, dict):
        for key in ("ignore", "disabled_layers"):
            _extend(sparse_cfg.get(key))
    for group in skip_softmax_config_groups_from_checkpoint_config(checkpoint_config):
        for key in ("ignore", "disabled_layers"):
            _extend(group.get(key))
    return disabled_layers or None


def skip_softmax_formula_from_checkpoint_config(
    checkpoint_config: Optional[Dict[str, Any]],
) -> Optional[SkipSoftmaxFormula]:
    """Read the scalar skip-softmax calibration formula from checkpoint config."""
    if not isinstance(checkpoint_config, dict):
        return None

    formula = SkipSoftmaxFormula.parse_from_ckpt_config(
        checkpoint_config, coefficient_key="prefill"
    )
    if formula is not None:
        return formula
    formula = SkipSoftmaxFormula.parse_from_ckpt_config(
        checkpoint_config, coefficient_key="coefficients"
    )
    if formula is not None:
        return formula
    threshold_config = skip_softmax_threshold_scale_factor_config_from_checkpoint_config(
        checkpoint_config
    )
    if isinstance(threshold_config, dict):
        return SkipSoftmaxFormula.parse_from_dict(
            threshold_config.get("coefficients"),
            formula=threshold_config.get("formula"),
        )
    return None


@dataclass
class SkipSoftmaxKernelParams:
    """Skip-softmax thresholds passed to attention backend kernels.

    Produced by ``SkipSoftmaxScheduler.get_kernel_params()``.
    """

    # The kernel divides this by the context length to get the skip threshold;
    # zero turns skip-softmax off.
    threshold_scale_factor_prefill: float = 0.0
    # Only autoregressive (LLM) decoding has a decode phase; diffusion and
    # visual generation leave this at zero.
    threshold_scale_factor_decode: float = 0.0


class SkipSoftmaxScheduler:
    """Layer runtime scheduler for skip-softmax kernel thresholds."""

    def __init__(
        self,
        threshold_scale_factor_prefill: float = 0.0,
        threshold_scale_factor_decode: float = 0.0,
        initial_disabled_steps: Optional[int] = None,
    ):
        self.threshold_scale_factor_prefill = threshold_scale_factor_prefill
        self.threshold_scale_factor_decode = threshold_scale_factor_decode
        self.initial_disabled_steps = initial_disabled_steps

    @staticmethod
    def _phase(
        value: Optional[Union[float, Dict[str, float]]],
        phase: str,
    ) -> Optional[float]:
        if isinstance(value, dict):
            return value.get(phase, None)
        return value

    @staticmethod
    def _threshold_scale_factor_config(
        checkpoint_config: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(checkpoint_config, dict):
            return None
        if "formula" in checkpoint_config:
            return checkpoint_config
        return skip_softmax_threshold_scale_factor_config_from_checkpoint_config(checkpoint_config)

    @classmethod
    def from_threshold_scale_factor(
        cls,
        threshold_scale_factor: Optional[Union[float, Dict[str, float]]],
        *,
        initial_disabled_steps: Optional[int] = None,
    ) -> "SkipSoftmaxScheduler":
        prefill = cls._phase(threshold_scale_factor, "prefill")
        decode = cls._phase(threshold_scale_factor, "decode")
        return cls(
            threshold_scale_factor_prefill=0.0 if prefill is None else prefill,
            threshold_scale_factor_decode=0.0 if decode is None else decode,
            initial_disabled_steps=initial_disabled_steps,
        )

    @classmethod
    def from_target_sparsity(
        cls,
        target_sparsity: Optional[Union[float, Dict[str, float]]],
        *,
        checkpoint_config: Optional[Dict[str, Any]] = None,
        initial_disabled_steps: Optional[int] = None,
    ) -> "SkipSoftmaxScheduler":
        """Build a scheduler by resolving target sparsity through checkpoint formula."""
        if target_sparsity is None:
            return cls(initial_disabled_steps=initial_disabled_steps)

        threshold_config = cls._threshold_scale_factor_config(checkpoint_config)
        shared_formula = (
            threshold_config.get("formula") if isinstance(threshold_config, dict) else None
        )

        def _compute(phase: str, sparsity: Optional[float]) -> Optional[float]:
            if sparsity is None:
                return None
            if threshold_config is None:
                phase_formula = None
            else:
                phase_formula = SkipSoftmaxFormula.parse_from_dict(
                    threshold_config.get(phase), formula=shared_formula
                )
            if phase_formula is None:
                raise ValueError(
                    f"SkipSoftmaxAttentionConfig: config.json must carry a "
                    f"top-level 'formula' string and a '{phase}' coefficient "
                    f"dictionary under sparse_attention_config.config_groups.*."
                    f"threshold_scale_factor "
                    f"to resolve target_sparsity."
                )
            return phase_formula.compute_threshold_scale_factor(sparsity)

        prefill = _compute("prefill", cls._phase(target_sparsity, "prefill"))
        decode = _compute("decode", cls._phase(target_sparsity, "decode"))
        return cls(
            threshold_scale_factor_prefill=0.0 if prefill is None else prefill,
            threshold_scale_factor_decode=0.0 if decode is None else decode,
            initial_disabled_steps=initial_disabled_steps,
        )

    @staticmethod
    def get_graph_phase_for_step_index(
        step_index: Optional[int],
        *,
        initial_disabled_steps: Optional[int],
    ) -> Optional[int]:
        """Return the stable graph-key phase for the initial-disabled boundary."""
        if step_index is None:
            return None
        if initial_disabled_steps is None or initial_disabled_steps <= 0:
            return None
        return int(step_index >= initial_disabled_steps)

    def get_kernel_params(self, *, step_index: Optional[int] = None):
        """Return kernel params, applying step-based disablement when provided."""
        if (
            self.get_graph_phase_for_step_index(
                step_index,
                initial_disabled_steps=self.initial_disabled_steps,
            )
            == 0
        ):
            return SkipSoftmaxKernelParams()
        return SkipSoftmaxKernelParams(
            threshold_scale_factor_prefill=self.threshold_scale_factor_prefill,
            threshold_scale_factor_decode=self.threshold_scale_factor_decode,
        )


@dataclass(frozen=True)
class SkipSoftmaxParams(SparseParams):
    """Skip-softmax backend parameters."""

    algorithm: Literal["skip_softmax"] = field(init=False, default="skip_softmax")
    scheduler: SkipSoftmaxScheduler = field(default_factory=SkipSoftmaxScheduler)
