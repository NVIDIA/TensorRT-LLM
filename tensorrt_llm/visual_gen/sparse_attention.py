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
"""User-facing sparse attention configs for visual generation."""

import fnmatch
from types import SimpleNamespace
from typing import Any, Dict, Literal, Optional

from pydantic import Field as PydanticField

from tensorrt_llm.llmapi.utils import StrictBaseModel


class BaseSparseAttentionConfig(StrictBaseModel):
    """Base for visual-generation sparse attention configs.

    Each algorithm subclasses this and pins a unique ``algorithm``
    discriminator for the config union.
    """

    algorithm: str

    def to_sparse_params(self, **kwargs):
        """Lower user-facing config into SparseParams."""
        raise NotImplementedError(
            f"{type(self).__name__} does not implement SparseParams lowering."
        )

    def to_sparse_metadata_params(self, **kwargs):
        """Lower user-facing config into SparseMetadataParams."""
        return None


class SkipSoftmaxAttentionConfig(BaseSparseAttentionConfig):
    """SkipSoftmax sparse attention configuration for visual generation.

    This class is only the Python/YAML user surface. Checkpoint calibration
    data is passed to ``to_sparse_params()`` when SparseParams are created.
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
    disabled_until_timestep: Optional[float] = PydanticField(
        default=None,
        ge=0.0,
        le=1.0,
        description="Normalized timestep cutoff below which skip-softmax is enabled.",
    )

    def to_sparse_params(self, **kwargs):
        from tensorrt_llm._torch.attention_backend.sparse.skip_softmax import (
            SkipSoftmaxParams,
            SkipSoftmaxScheduler,
            skip_softmax_disabled_until_timestep_from_checkpoint_config,
            skip_softmax_target_sparsity_from_checkpoint_config,
        )

        module_name = kwargs.get("module_name", None)
        checkpoint_config = self._checkpoint_config_from_kwargs(kwargs)

        if module_name is not None and self._is_disabled(module_name, checkpoint_config):
            return None

        disabled_until_timestep = self.disabled_until_timestep
        if disabled_until_timestep is None:
            disabled_until_timestep = skip_softmax_disabled_until_timestep_from_checkpoint_config(
                checkpoint_config
            )

        if self.threshold_scale_factor is not None:
            scheduler = SkipSoftmaxScheduler.from_threshold_scale_factor(
                self.threshold_scale_factor,
                disabled_until_timestep=disabled_until_timestep,
            )
        else:
            target_sparsity = self.target_sparsity
            if target_sparsity is None:
                target_sparsity = skip_softmax_target_sparsity_from_checkpoint_config(
                    checkpoint_config
                )
            if target_sparsity is None:
                return None
            scheduler = SkipSoftmaxScheduler.from_target_sparsity(
                target_sparsity,
                checkpoint_config=checkpoint_config,
                disabled_until_timestep=disabled_until_timestep,
            )

        if (
            scheduler.threshold_scale_factor_prefill <= 0
            and scheduler.threshold_scale_factor_decode <= 0
        ):
            return None
        return SkipSoftmaxParams(scheduler=scheduler)

    def _is_disabled(
        self,
        module_name: str,
        checkpoint_config: Optional[Dict[str, Any]],
    ) -> bool:
        """Return whether skip-softmax should be disabled for this layer."""
        from tensorrt_llm._torch.attention_backend.sparse.skip_softmax import (
            skip_softmax_ignore_patterns_from_checkpoint_config,
        )

        candidate_names = self._layer_pattern_match_names(module_name)
        ignore_patterns = (
            skip_softmax_ignore_patterns_from_checkpoint_config(checkpoint_config) or ()
        )
        return any(
            fnmatch.fnmatch(name, pattern)
            for pattern in ignore_patterns
            for name in candidate_names
        )

    @staticmethod
    def _layer_pattern_match_names(module_name: str) -> tuple[str, ...]:
        """Return full and component-relative names for layer pattern matching."""
        candidate_names = {module_name, module_name.replace("._orig_mod.", ".")}
        for name in tuple(candidate_names):
            for prefix in ("transformer.", "transformer_2."):
                if name.startswith(prefix):
                    candidate_names.add(name[len(prefix) :])
        return tuple(candidate_names)

    def resolve_threshold_scale_factor(
        self,
        checkpoint_config: Optional[Dict[str, Any]] = None,
    ) -> Optional[float]:
        """Resolve to a concrete scalar threshold via shared formula helpers.

        Resolution order: user-supplied ``threshold_scale_factor`` (raw,
        wins) → ``target_sparsity`` + checkpoint calibration formula.
        """
        if self.threshold_scale_factor is not None:
            return self.threshold_scale_factor

        sparsity = self.target_sparsity
        if sparsity is None:
            from tensorrt_llm._torch.attention_backend.sparse.skip_softmax import (
                skip_softmax_target_sparsity_from_checkpoint_config,
            )

            checkpoint_sparsity = skip_softmax_target_sparsity_from_checkpoint_config(
                checkpoint_config
            )
            if isinstance(checkpoint_sparsity, dict):
                sparsity = checkpoint_sparsity.get("prefill")
            else:
                sparsity = checkpoint_sparsity
        if sparsity is None:
            return None

        from tensorrt_llm._torch.attention_backend.sparse.skip_softmax import (
            skip_softmax_formula_from_checkpoint_config,
        )

        formula = skip_softmax_formula_from_checkpoint_config(checkpoint_config)
        if formula is None:
            raise ValueError(
                "SkipSoftmaxAttentionConfig: target_sparsity requires calibration formula "
                "coefficients. Provide checkpoint config.json calibrated coefficients."
            )
        return formula.compute_threshold_scale_factor(sparsity)

    @staticmethod
    def _checkpoint_config_from_kwargs(kwargs: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        checkpoint_config = kwargs.get("checkpoint_config", None)
        if isinstance(checkpoint_config, dict):
            return checkpoint_config

        pretrained_config = kwargs.get("pretrained_config", None)
        if isinstance(pretrained_config, dict):
            return pretrained_config
        if isinstance(pretrained_config, SimpleNamespace):
            return vars(pretrained_config)
        if pretrained_config is not None:
            sparse_config = getattr(pretrained_config, "sparse_attention_config", None)
            if isinstance(sparse_config, dict):
                return {"sparse_attention_config": sparse_config}
        return None
