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

from typing import TYPE_CHECKING, ClassVar

import torch
import torch.nn as nn

from tensorrt_llm._torch.attention.backends.sparse.skip_softmax import SkipSoftmaxScheduler
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm.visual_gen.sparse_attention import SkipSoftmaxAttentionConfig, SolAttentionConfig

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.cuda_graph_runner import CUDAGraphRunner


class BaseDiffusionModel(nn.Module):
    """Base class for TRT-LLM VisualGen model components."""

    # Models that implement parallel_config.tp_layout='token_sharded' (token-sharded residual
    # stream inside the TP group, see parallel/token_sharded_tp.py) set this to True.
    _supports_token_sharded_tp: ClassVar[bool] = False
    # For that layout (parallel/token_sharded_modules.py): the module lists holding the
    # transformer blocks, and {module-name pattern: kind, adapter or "keep"} for what the
    # conversion rules cannot decide (relative to a block).
    _token_sharded_tp_blocks: ClassVar[tuple[str, ...]] = ("blocks",)
    _token_sharded_tp_exceptions: ClassVar[dict] = {}

    def __init__(self, model_config: DiffusionModelConfig):
        super().__init__()
        parallel = getattr(model_config, "parallel", None)
        if (
            getattr(parallel, "token_sharded_tp", False)
            and not type(self)._supports_token_sharded_tp
        ):
            raise ValueError(
                "parallel_config.tp_layout='token_sharded' is not implemented for "
                f"{type(self).__name__} (the model does not set "
                "_supports_token_sharded_tp = True). Unset tp_layout to use "
                "all-reduce tensor parallelism."
            )
        self.model_config = model_config
        self.component_name = model_config.component_name
        self.pretrained_config = model_config.pretrained_config

    def _apply_tp_layout(self) -> None:
        """Set up ``parallel_config.tp_layout``; call at the end of ``__init__``.

        Token-sharded: switches ``self.sharder`` to the layout and converts the TP modules of
        ``_token_sharded_tp_blocks`` in place (the model is built exactly as for plain TP).
        Replicated (plain TP): nothing to do.
        """
        from ..parallel.token_sharded_modules import convert_to_token_sharded_tp
        from ..parallel.token_sharded_tp import TokenShardedTP

        if not self._wants_token_sharded_tp():
            return
        sharder = getattr(self, "sharder", None)
        if sharder is None:
            raise AttributeError(
                f"{type(self).__name__}._apply_tp_layout(): token-sharded TP enters and exits "
                "through self.sharder (a SequenceSharder); build it before this call."
            )
        tp = TokenShardedTP.from_model_config(self.model_config)
        sharder.use_token_sharded_tp(tp)
        convert_to_token_sharded_tp(
            self,
            tp,
            containers=type(self)._token_sharded_tp_blocks,
            exceptions=type(self)._token_sharded_tp_exceptions,
        )

    def check_tp_layout_applied(self) -> None:
        """Raise if ``parallel_config.tp_layout`` is token-sharded but this model did not set
        it up (it would silently run plain TP); the pipeline loader calls this."""
        if self._wants_token_sharded_tp() and not getattr(
            getattr(self, "sharder", None), "token_sharded_tp", False
        ):
            raise RuntimeError(
                f"{type(self).__name__} sets _supports_token_sharded_tp but did not call "
                "self._apply_tp_layout() at the end of __init__."
            )

    def _wants_token_sharded_tp(self) -> bool:
        return bool(
            getattr(getattr(self.model_config, "parallel", None), "token_sharded_tp", False)
        )

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

        if isinstance(sparse_config, SkipSoftmaxAttentionConfig):
            disabled_until_timestep = sparse_config.resolve_disabled_until_timestep(
                pretrained_config=self.model_config.pretrained_config,
            )
            if disabled_until_timestep is None:
                return

            # Skip Softmax switches graph-visible attention behavior at the
            # timestep boundary while tensor shapes stay unchanged. Key the dense
            # and sparse phases separately; if timestep is absent or None, the
            # scheduler returns None and the runner omits this key part.
            runner.register_extra_key_fn(
                "skip_softmax_phase",
                lambda *args, **kwargs: SkipSoftmaxScheduler.get_graph_phase_for_timestep(
                    kwargs.get("timestep"),
                    disabled_until_timestep=disabled_until_timestep,
                ),
            )
            return

        if isinstance(sparse_config, SolAttentionConfig):
            disabled_until_timestep = sparse_config.disabled_until_timestep
            if disabled_until_timestep is None:
                # dense_layers is fixed per layer at construction, so it is
                # already baked into each captured graph and needs no key.
                return

            # Sol-Attn switches between dense and sparse attention at the
            # dense-prefix boundary, again without changing tensor shapes, so
            # the two phases must not share a captured graph.
            runner.register_extra_key_fn(
                "sol_attn_phase",
                lambda *args, **kwargs: SkipSoftmaxScheduler.get_graph_phase_for_timestep(
                    kwargs.get("timestep"),
                    disabled_until_timestep=disabled_until_timestep,
                ),
            )
            return
