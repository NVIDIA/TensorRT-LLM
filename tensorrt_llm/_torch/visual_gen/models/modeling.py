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
"""Base classes for VisualGen model components."""

from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from tensorrt_llm._torch.attention.backends.sparse.timestep_phase import graph_phase_for_timestep
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.token_order import require_native_token_order
from tensorrt_llm.visual_gen.sparse_attention import BaseSparseAttentionConfig

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.cuda_graph_runner import CUDAGraphRunner


class BaseDiffusionModel(nn.Module):
    """Base class for TRT-LLM VisualGen model components."""

    # Set to True by models that implement parallel_config.tp_layout='token_sharded'.
    _supports_token_sharded_tp = False
    # Set to True by models that apply ``TokenReorder`` around their block stack, so
    # ``sparse_attention_config.token_order`` takes effect.
    _supports_token_order = False

    def __init__(self, model_config: DiffusionModelConfig):
        super().__init__()
        if model_config.parallel.token_sharded_tp and not self._supports_token_sharded_tp:
            raise ValueError(
                "parallel_config.tp_layout='token_sharded' is not implemented for "
                f"{type(self).__name__}; unset tp_layout to use all-reduce tensor parallelism."
            )
        if not self._supports_token_order:
            require_native_token_order(
                model_config.attention.sparse_attention_config, type(self).__name__
            )
        self.model_config = model_config
        self.component_name = model_config.component_name
        self.pretrained_config = model_config.pretrained_config

    def _apply_tp_layout(self) -> None:
        """Apply ``parallel_config.tp_layout``; call at the end of ``__init__``."""
        if not self.model_config.parallel.token_sharded_tp:
            return
        from ..parallel.token_sharded_modules import convert_to_token_sharded_tp
        from ..parallel.token_sharded_tp import TokenShardedSequenceSharder, TokenShardedTP

        tp = TokenShardedTP.from_model_config(self.model_config)
        self.sharder = TokenShardedSequenceSharder(tp)
        convert_to_token_sharded_tp(self, tp)

    def forward(self, *args, timestep: torch.Tensor | None = None, **kwargs):
        """Run the diffusion transformer.

        Concrete VisualGen models own their full forward signatures. This base
        method defines the common arguments that every forward should accept.

        Args:
            timestep: Normalized denoising-time coordinate in ``[0, 1]``.
                Larger values correspond to earlier, noisier denoising steps.
                It may be ``None`` only for model paths that do not need a
                timestep-dependent model-forward decision.
                Model definers must pass the normalized value required by this
                contract and perform any conversion needed inside modules that
                reference this value. This TRT-LLM VisualGen contract
                intentionally differs from Diffusers' ``ModelMixin`` subclasses,
                where transformer ``timestep`` is model-specific. For example,
                WAN forwards raw integer scheduler timesteps in ``[0, 999]``,
                while FLUX forwards ``t / 1000``.
        """
        raise NotImplementedError("Diffusion model subclasses must implement forward().")

    def register_cuda_graph_extra_key_fns(self, runner: "CUDAGraphRunner") -> None:
        """Register CUDA graph key contributors that are not tensor shapes.

        Override this hook when a model.forward input changes captured
        execution without changing tensor shapes. Implementations should call
        ``runner.register_extra_key_fn(name, fn)``, where ``fn`` is the
        callback.

        The callback receives the wrapped model.forward ``*args`` and
        ``**kwargs`` and returns either a hashable key value or ``None``.
        If the callback returns ``None``, the runner omits that key part for
        the current call.

        Subclasses should call ``super()`` unless they intentionally replace
        the shared registrations.
        """
        sparse_config = self.model_config.attention.sparse_attention_config
        if not isinstance(sparse_config, BaseSparseAttentionConfig):
            return
        disabled_until_timestep = sparse_config.resolve_disabled_until_timestep(
            pretrained_config=self.model_config.pretrained_config,
        )
        if disabled_until_timestep is None:
            return

        # Timestep-scheduled sparse attention switches graph-visible behavior at
        # the cutoff while tensor shapes stay unchanged, and the backends prepare
        # the matching phase during warmup. Key each live timestep so dense and
        # sparse graphs stay separate; when no timestep is present the callback
        # returns None and the runner omits this key part.
        runner.register_extra_key_fn(
            "sparse_attn_phase",
            lambda *args, **kwargs: graph_phase_for_timestep(
                kwargs.get("timestep"),
                disabled_until_timestep=disabled_until_timestep,
            ),
        )
