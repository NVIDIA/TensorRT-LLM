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
from pydantic import field_validator

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
        from tensorrt_llm._torch.attention.backends.sparse.skip_softmax import (
            SkipSoftmaxParams,
            SkipSoftmaxScheduler,
        )

        module_name = kwargs.get("module_name", None)
        ckpt_sparse_attention_config = self._ckpt_sparse_attention_config_from_kwargs(kwargs)

        if module_name is not None and self._is_disabled(module_name, ckpt_sparse_attention_config):
            return None

        disabled_until_timestep = self.resolve_disabled_until_timestep(
            checkpoint_config=ckpt_sparse_attention_config,
        )

        threshold_scale_factor = self.resolve_threshold_scale_factor(ckpt_sparse_attention_config)
        if threshold_scale_factor is None:
            return None
        scheduler = SkipSoftmaxScheduler.from_threshold_scale_factor(
            threshold_scale_factor,
            disabled_until_timestep=disabled_until_timestep,
        )

        if (
            scheduler.threshold_scale_factor_prefill <= 0
            and scheduler.threshold_scale_factor_decode <= 0
        ):
            return None
        return SkipSoftmaxParams(scheduler=scheduler)

    def resolve_disabled_until_timestep(
        self,
        *,
        checkpoint_config: Optional[Dict[str, Any]] = None,
        pretrained_config: Any = None,
    ) -> Optional[float]:
        """Resolve the user override or checkpoint-provided timestep cutoff.

        Exactly one of ``checkpoint_config`` (a raw checkpoint config dict) or
        ``pretrained_config`` (the model's pretrained config object/dict) is
        expected per call site; both default to ``None`` so a mistyped keyword
        raises ``TypeError`` here instead of silently resolving to ``None``.
        """
        if self.disabled_until_timestep is not None:
            return self.disabled_until_timestep

        from tensorrt_llm._torch.attention.backends.sparse.skip_softmax import (
            skip_softmax_disabled_until_timestep_from_ckpt_sparse_attention_config,
        )

        ckpt_sparse_attention_config = self._ckpt_sparse_attention_config_from_kwargs(
            {"checkpoint_config": checkpoint_config, "pretrained_config": pretrained_config}
        )
        return skip_softmax_disabled_until_timestep_from_ckpt_sparse_attention_config(
            ckpt_sparse_attention_config
        )

    def _is_disabled(
        self,
        module_name: str,
        ckpt_sparse_attention_config: Optional[Dict[str, Any]],
    ) -> bool:
        """Return whether skip-softmax should be disabled for this layer."""
        from tensorrt_llm._torch.attention.backends.sparse.skip_softmax import (
            skip_softmax_ignore_from_ckpt_sparse_attention_config,
        )

        candidate_names = self._layer_pattern_match_names(module_name)
        ignore = (
            skip_softmax_ignore_from_ckpt_sparse_attention_config(ckpt_sparse_attention_config)
            or ()
        )
        return any(fnmatch.fnmatch(name, pattern) for pattern in ignore for name in candidate_names)

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
        ckpt_sparse_attention_config: Optional[Dict[str, Any]] = None,
    ) -> Optional[float]:
        """Resolve to a concrete scalar threshold via shared formula helpers.

        Resolution order: user-supplied ``threshold_scale_factor`` (raw,
        wins) → ``target_sparsity`` + checkpoint calibration formula.
        """
        if self.threshold_scale_factor is not None:
            return self.threshold_scale_factor

        sparsity = self.target_sparsity
        if sparsity is None:
            from tensorrt_llm._torch.attention.backends.sparse.skip_softmax import (
                skip_softmax_target_sparsity_from_ckpt_sparse_attention_config,
            )

            checkpoint_sparsity = skip_softmax_target_sparsity_from_ckpt_sparse_attention_config(
                ckpt_sparse_attention_config
            )
            if isinstance(checkpoint_sparsity, dict):
                raise ValueError(
                    "VisualGen skip-softmax checkpoint target_sparsity must be "
                    "a scalar; prefill/decode phase dictionaries are LLM-only."
                )
            else:
                sparsity = checkpoint_sparsity
        if sparsity is None:
            return None

        from tensorrt_llm._torch.attention.backends.sparse.skip_softmax import (
            skip_softmax_formula_from_ckpt_sparse_attention_config,
        )

        formula = skip_softmax_formula_from_ckpt_sparse_attention_config(
            ckpt_sparse_attention_config
        )
        if formula is None:
            raise ValueError(
                "SkipSoftmaxAttentionConfig: target_sparsity requires calibration formula "
                "coefficients. Provide checkpoint config.json calibrated coefficients."
            )
        return formula.compute_threshold_scale_factor(sparsity)

    @staticmethod
    def _ckpt_sparse_attention_config_from_kwargs(
        kwargs: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        checkpoint_config = kwargs.get("checkpoint_config", None)
        if isinstance(checkpoint_config, dict):
            sparse_cfg = checkpoint_config.get("sparse_attention_config")
            return sparse_cfg if isinstance(sparse_cfg, dict) else checkpoint_config

        pretrained_config = kwargs.get("pretrained_config", None)
        if isinstance(pretrained_config, dict):
            sparse_cfg = pretrained_config.get("sparse_attention_config")
            return sparse_cfg if isinstance(sparse_cfg, dict) else pretrained_config
        if isinstance(pretrained_config, SimpleNamespace):
            sparse_cfg = getattr(pretrained_config, "sparse_attention_config", None)
            return sparse_cfg if isinstance(sparse_cfg, dict) else None
        if pretrained_config is not None:
            sparse_config = getattr(pretrained_config, "sparse_attention_config", None)
            if isinstance(sparse_config, dict):
                return sparse_config
        return None


class SolAttentionConfig(BaseSparseAttentionConfig):
    """Sol-Attn sparse attention configuration for visual generation.

    Dynamic block routing + sparse computation + approximation correction in
    one online-softmax pass (arXiv:2607.24027). Kernel is CuTeDSL on
    datacenter Blackwell -- sm100 (B200/GB200) and sm103 (B300/GB300) --
    head_dim=128, bf16, MHA.

    On an unsupported *shape, dtype, or architecture* the kernel falls back to
    dense attention -- the configured backend's dense kernel where available,
    torch SDPA otherwise -- and counts the fallback, so setting this config on the wrong GPU
    degrades rather than fails. Two cases are not covered by that fallback and do raise: GQA/MQA
    (num_kv_heads != num_heads) here at construction, and context parallelism
    (cp_size > 1), rejected in visual_gen/modules/attention.py.
    """

    algorithm: Literal["sol_attn"] = "sol_attn"
    tau: float = PydanticField(
        1.0,
        description="Per-block routing threshold; higher tau routes more blocks sparse.",
    )
    thresh_type: Literal["diag", "exact"] = PydanticField(
        "diag",
        description="Threshold policy forwarded to the kernel (kernel default: 'diag').",
    )
    disabled_until_timestep: Optional[float] = PydanticField(
        None,
        gt=0.0,
        le=1.0,
        description=(
            "Dense-prefix cutoff on the normalized denoising timestep, with the "
            "same sense as skip_softmax's field of the same name: the layer runs "
            "dense while timestep >= this value and switches to the sparse kernel "
            "below it. Larger timesteps are earlier, noisier steps, so this "
            "protects the high-noise prefix. Use None (not 0.0) to disable the "
            "prefix; 0.0 is rejected because it would run dense on every step "
            "and silently turn Sol-Attn off entirely. "
            "Read from the `timestep` forward kwarg, which must be the normalized "
            "scheduler time (larger = noisier). WAN and LTX-2 pass it; a pipeline "
            "that does not, or that passes something else, gets a one-time warning "
            "and runs sparse on every step (fail-open)."
        ),
    )
    dense_layers: Optional[list[int]] = PydanticField(
        None,
        description=(
            "Layer indices forced dense regardless of the dense prefix, e.g. "
            "[0, 2, 3]. Evaluated per layer at construction time; no pipeline "
            "wiring required."
        ),
    )

    @field_validator("dense_layers")
    @classmethod
    def _validate_dense_layers(cls, layers: Optional[list[int]]) -> Optional[list[int]]:
        if layers is None:
            return None
        for index in layers:
            if index < 0:
                raise ValueError(f"dense_layers contains a negative layer index: {index}")
        return sorted(set(layers))

    def to_sparse_params(self, **kwargs):
        # Sol-Attn's knobs are consumed directly by SolAttention.__init__
        # (constructed via CUTEDSL backend dispatch in create_attention), not
        # lowered into a shared SparseParams -- the vendored kernel has no
        # checkpoint-calibration step to resolve here, unlike skip_softmax.
        return None


class VideoSparseAttentionConfig(StrictBaseModel):
    """Video Sparse Attention (VSA) sparse-attention recipe.

    Two-stage hybrid attention: a coarse mean-pooled stage over (4,4,4) cubes
    and a block-sparse fine stage over the top-K cubes selected per head.
    vsa_sparsity controls the fraction of cubes dropped on the fine stage.
    The fine stage may run on either the CuTeDSL backend or the TRTLLM PrimTS
    backend, while the user-facing sparsity semantics stay the same.
    """

    algorithm: Literal["vsa"] = PydanticField(
        "vsa",
        description="Sparse attention algorithm discriminator.",
    )
    vsa_sparsity: float = PydanticField(
        0.9,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of cubes dropped on the fine stage. 0.0 keeps all cubes "
            "(dense fine stage); values closer to 1.0 keep fewer cubes."
        ),
    )
