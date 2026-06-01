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

from dataclasses import dataclass
from typing import Any, Dict, Optional

import numexpr
from pydantic import ConfigDict, model_validator
from pydantic import Field as PydanticField

from tensorrt_llm.llmapi.utils import StrictBaseModel

_RESERVED_FORMULA_KEYS = frozenset({"formula", "target_sparsity"})


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


def parse_skip_softmax_formula_from_dict(
    data: Optional[Dict[str, Any]],
    *,
    formula: Optional[str] = None,
) -> Optional[SkipSoftmaxFormula]:
    """Build a :class:`SkipSoftmaxFormula` from a coefficient dict + formula string.

    ``formula`` may be passed explicitly or carried inside ``data``
    under a ``formula`` key. It is required — there is no synthesized
    fallback. Returns ``None`` when the block is *absent* (``data`` is
    not a dict, or no formula / no coefficients). A block that *is*
    present but malformed (un-parseable formula, missing coefficients)
    raises from :class:`SkipSoftmaxFormula` validation rather than being
    silently dropped. The reserved keys ``formula``/``target_sparsity``
    are not treated as coefficients.
    """
    if not isinstance(data, dict):
        return None
    coefficients = {k: v for k, v in data.items() if k not in _RESERVED_FORMULA_KEYS}
    if not formula:
        nested = data.get("formula")
        if isinstance(nested, str) and nested.strip():
            formula = nested
    if not formula or not coefficients:
        return None
    return SkipSoftmaxFormula(
        formula=formula,
        coefficients={k: float(v) for k, v in coefficients.items()},
    )


def parse_skip_softmax_formula_from_ckpt_config(
    checkpoint_config: Dict[str, Any],
    *,
    coefficient_key: str = "prefill",
) -> Optional[SkipSoftmaxFormula]:
    """Lift the calibration formula out of a HuggingFace ``config.json`` dict.

    Reads ``sparse_attention_config.threshold_scale_factor`` (the
    ``formula`` string plus coefficients under ``coefficient_key``).
    ``coefficient_key`` defaults to ``"prefill"`` (LLM convention);
    pipelines without a prefill/decode split (e.g. visual generation)
    pass a phase-neutral key like ``"coefficients"``. Returns ``None``
    when the block is absent; a present-but-malformed block raises (see
    :func:`parse_skip_softmax_formula_from_dict`).
    """
    sparse_cfg = checkpoint_config.get("sparse_attention_config")
    if not isinstance(sparse_cfg, dict):
        return None
    tsf = sparse_cfg.get("threshold_scale_factor")
    if not isinstance(tsf, dict):
        return None
    return parse_skip_softmax_formula_from_dict(
        tsf.get(coefficient_key), formula=tsf.get("formula")
    )


@dataclass
class SkipSoftmaxKernelParams:
    """Skip-softmax parameters consumed directly by the attention backend.

    The LLM and visual-gen ``SkipSoftmaxAttentionConfig`` classes resolve
    to this via ``to_kernel_params()``.
    """

    threshold_scale_factor_prefill: float
    # Decode phase only applies to autoregressive (LLM) pipelines;
    # diffusion / visual generation has no decode phase and leaves it 0.
    threshold_scale_factor_decode: float = 0.0
