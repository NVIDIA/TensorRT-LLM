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
semantic ``target_sparsity`` via a formula shipped with the checkpoint.
The helpers are shared by the LLM and VisualGen pipelines.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, Union

import numexpr
import torch
from pydantic import ConfigDict, model_validator
from pydantic import Field as PydanticField

from tensorrt_llm.llmapi.utils import StrictBaseModel

from .params import SkipSoftmaxKernelParams, SparseParams

_RESERVED_FORMULA_KEYS = frozenset({"formula", "target_sparsity"})
_SKIP_SOFTMAX_ALGORITHMS = frozenset({"skip_softmax", "softmax_skip"})


def _is_skip_softmax_group(group: Dict[str, Any]) -> bool:
    algorithm = group.get("algorithm", group.get("sparse_algo"))
    return algorithm in _SKIP_SOFTMAX_ALGORITHMS


def _looks_like_skip_softmax_group(group: Dict[str, Any]) -> bool:
    return _is_skip_softmax_group(group) or any(
        key in group
        for key in (
            "threshold_scale_factor",
            "target_sparsity",
            "ignore",
            "disabled_until_timestep",
        )
    )


def skip_softmax_config_from_ckpt_sparse_attention_config(
    ckpt_sparse_attention_config: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Return the skip-softmax config from checkpoint ``sparse_attention_config``."""
    if not isinstance(ckpt_sparse_attention_config, dict):
        return None

    config_groups = ckpt_sparse_attention_config.get("config_groups")
    if isinstance(config_groups, dict):
        # ModelOpt may emit many sparse-attention groups. The shared
        # skip-softmax helpers operate on the single skip-softmax group.
        groups = [
            group
            for group in config_groups.values()
            if isinstance(group, dict) and _is_skip_softmax_group(group)
        ]
        if len(groups) > 1:
            raise ValueError(
                "checkpoint sparse_attention_config contains multiple skip-softmax "
                "config groups; expected at most one."
            )
        return groups[0] if groups else None

    if _looks_like_skip_softmax_group(ckpt_sparse_attention_config):
        return ckpt_sparse_attention_config
    return None


def skip_softmax_threshold_scale_factor_config_from_ckpt_sparse_attention_config(
    ckpt_sparse_attention_config: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Return the skip-softmax calibration formula config."""
    config = skip_softmax_config_from_ckpt_sparse_attention_config(ckpt_sparse_attention_config)
    if isinstance(config, dict):
        threshold_config = config.get("threshold_scale_factor")
        if isinstance(threshold_config, dict):
            return threshold_config
    return None


def skip_softmax_target_sparsity_from_ckpt_sparse_attention_config(
    ckpt_sparse_attention_config: Optional[Dict[str, Any]],
) -> Optional[Union[float, Dict[str, float]]]:
    """Return checkpoint-provided skip-softmax target sparsity, if present."""
    config = skip_softmax_config_from_ckpt_sparse_attention_config(ckpt_sparse_attention_config)
    if isinstance(config, dict):
        target_sparsity = config.get("target_sparsity")
        if isinstance(target_sparsity, dict):
            return {str(k): float(v) for k, v in target_sparsity.items()}
        if isinstance(target_sparsity, (float, int)):
            return float(target_sparsity)
    return None


def skip_softmax_disabled_until_timestep_from_ckpt_sparse_attention_config(
    ckpt_sparse_attention_config: Optional[Dict[str, Any]],
) -> Optional[float]:
    """Return checkpoint-provided normalized timestep cutoff, if present."""
    config = skip_softmax_config_from_ckpt_sparse_attention_config(ckpt_sparse_attention_config)
    if isinstance(config, dict):
        disabled_until_timestep = config.get("disabled_until_timestep")
        if isinstance(disabled_until_timestep, (float, int)):
            return float(disabled_until_timestep)
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
    def parse_from_ckpt_sparse_attention_config(
        cls,
        ckpt_sparse_attention_config: Dict[str, Any],
        *,
        coefficient_key: str = "prefill",
    ) -> Optional["SkipSoftmaxFormula"]:
        """Lift the calibration formula out of checkpoint sparse metadata."""
        threshold_config = (
            ckpt_sparse_attention_config
            if "formula" in ckpt_sparse_attention_config
            else skip_softmax_threshold_scale_factor_config_from_ckpt_sparse_attention_config(
                ckpt_sparse_attention_config
            )
        )
        if not isinstance(threshold_config, dict):
            return None
        return cls.parse_from_dict(
            threshold_config.get(coefficient_key),
            formula=threshold_config.get("formula"),
        )


def skip_softmax_ignore_from_ckpt_sparse_attention_config(
    ckpt_sparse_attention_config: Optional[Dict[str, Any]],
) -> Optional[list[str]]:
    """Read ModelOpt checkpoint ``ignore`` patterns for skip-softmax."""
    ignore: list[str] = []
    seen: set[str] = set()

    def _extend(patterns: Any) -> None:
        if not isinstance(patterns, list):
            return
        for pattern in patterns:
            pattern = str(pattern)
            if pattern not in seen:
                seen.add(pattern)
                ignore.append(pattern)

    if isinstance(ckpt_sparse_attention_config, dict):
        # The input may already be the skip-softmax group rather than the
        # outer sparse_attention_config.
        _extend(ckpt_sparse_attention_config.get("ignore"))
    config = skip_softmax_config_from_ckpt_sparse_attention_config(ckpt_sparse_attention_config)
    if isinstance(config, dict) and config is not ckpt_sparse_attention_config:
        _extend(config.get("ignore"))
    return ignore or None


def skip_softmax_formula_from_ckpt_sparse_attention_config(
    ckpt_sparse_attention_config: Optional[Dict[str, Any]],
) -> Optional[SkipSoftmaxFormula]:
    """Read the scalar skip-softmax calibration formula from checkpoint config."""
    if not isinstance(ckpt_sparse_attention_config, dict):
        return None

    threshold_config = skip_softmax_threshold_scale_factor_config_from_ckpt_sparse_attention_config(
        ckpt_sparse_attention_config
    )
    if not isinstance(threshold_config, dict) and "formula" in ckpt_sparse_attention_config:
        threshold_config = ckpt_sparse_attention_config
    if isinstance(threshold_config, dict):
        formula = SkipSoftmaxFormula.parse_from_dict(
            threshold_config.get("coefficients"),
            formula=threshold_config.get("formula"),
        )
        if formula is not None:
            return formula
        # Scalar VisualGen configs may also keep coefficients next to formula.
        return SkipSoftmaxFormula.parse_from_dict(threshold_config)
    return None


class SkipSoftmaxScheduler:
    """Layer runtime scheduler for skip-softmax kernel thresholds."""

    def __init__(
        self,
        threshold_scale_factor_prefill: float = 0.0,
        threshold_scale_factor_decode: float = 0.0,
        disabled_until_timestep: Optional[float] = None,
    ):
        self.threshold_scale_factor_prefill = threshold_scale_factor_prefill
        self.threshold_scale_factor_decode = threshold_scale_factor_decode
        self.disabled_until_timestep = disabled_until_timestep

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
        ckpt_sparse_attention_config: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if not isinstance(ckpt_sparse_attention_config, dict):
            return None
        if "formula" in ckpt_sparse_attention_config:
            return ckpt_sparse_attention_config
        return skip_softmax_threshold_scale_factor_config_from_ckpt_sparse_attention_config(
            ckpt_sparse_attention_config
        )

    @classmethod
    def from_threshold_scale_factor(
        cls,
        threshold_scale_factor: Optional[Union[float, Dict[str, float]]],
        *,
        disabled_until_timestep: Optional[float] = None,
    ) -> "SkipSoftmaxScheduler":
        prefill = cls._phase(threshold_scale_factor, "prefill")
        decode = cls._phase(threshold_scale_factor, "decode")
        return cls(
            threshold_scale_factor_prefill=0.0 if prefill is None else prefill,
            threshold_scale_factor_decode=0.0 if decode is None else decode,
            disabled_until_timestep=disabled_until_timestep,
        )

    @classmethod
    def from_target_sparsity(
        cls,
        target_sparsity: Optional[Union[float, Dict[str, float]]],
        *,
        ckpt_sparse_attention_config: Optional[Dict[str, Any]] = None,
        disabled_until_timestep: Optional[float] = None,
    ) -> "SkipSoftmaxScheduler":
        """Build a scheduler by resolving target sparsity through checkpoint formula."""
        if target_sparsity is None:
            return cls(disabled_until_timestep=disabled_until_timestep)

        threshold_config = cls._threshold_scale_factor_config(ckpt_sparse_attention_config)
        shared_formula = (
            threshold_config.get("formula") if isinstance(threshold_config, dict) else None
        )

        def _compute(phase: str, sparsity: Optional[float]) -> Optional[float]:
            if sparsity is None:
                return None
            if threshold_config is None:
                phase_formula = None
            else:
                # Prefer phase-specific coefficients for LLM, then shared
                # coefficients, then inline coefficients in the threshold dict.
                coefficient_config = threshold_config.get(phase)
                if coefficient_config is None:
                    coefficient_config = threshold_config.get("coefficients")
                if coefficient_config is None and shared_formula is not None:
                    coefficient_config = threshold_config
                phase_formula = SkipSoftmaxFormula.parse_from_dict(
                    coefficient_config, formula=shared_formula
                )
            if phase_formula is None:
                raise ValueError(
                    f"SkipSoftmaxAttentionConfig: config.json must carry a "
                    f"top-level 'formula' string and coefficient values under "
                    f"sparse_attention_config.config_groups.*.threshold_scale_factor "
                    f"to resolve target_sparsity for {phase}."
                )
            return phase_formula.compute_threshold_scale_factor(sparsity)

        prefill = _compute("prefill", cls._phase(target_sparsity, "prefill"))
        decode = _compute("decode", cls._phase(target_sparsity, "decode"))
        return cls(
            threshold_scale_factor_prefill=0.0 if prefill is None else prefill,
            threshold_scale_factor_decode=0.0 if decode is None else decode,
            disabled_until_timestep=disabled_until_timestep,
        )

    @staticmethod
    def _as_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return None
            return float(value.flatten()[0].item())
        return float(value)

    @classmethod
    def get_graph_phase_for_timestep(
        cls,
        timestep: Any,
        *,
        disabled_until_timestep: Optional[float],
    ) -> Optional[int]:
        """Return the graph-key phase for the timestep disablement boundary.

        VisualGen denoising timesteps descend from 1 to 0, so skip-softmax is
        disabled while ``timestep >= disabled_until_timestep``.
        """
        if disabled_until_timestep is None:
            return None
        timestep_value = cls._as_float(timestep)
        if timestep_value is None:
            return None
        return int(timestep_value < disabled_until_timestep)

    def get_kernel_params(self, *, timestep: Any = None):
        """Return kernel params, applying timestep-based disablement when provided."""
        if (
            self.get_graph_phase_for_timestep(
                timestep,
                disabled_until_timestep=self.disabled_until_timestep,
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
