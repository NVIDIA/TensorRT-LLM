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

from typing import TYPE_CHECKING, Any, Optional

import torch
import torch.nn as nn

from tensorrt_llm._torch.attention_backend.interface import AttentionMetadata
from tensorrt_llm._torch.attention_backend.sparse.skip_softmax import SkipSoftmaxScheduler
from tensorrt_llm._torch.visual_gen.attention_backend.utils import get_visual_gen_attention_backend
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm.visual_gen.sparse_attention import SkipSoftmaxAttentionConfig

if TYPE_CHECKING:
    from tensorrt_llm._torch.visual_gen.cuda_graph_runner import CUDAGraphRunner


def _collect_attn_metadata(name: str, value: Any, sites: dict[str, AttentionMetadata]) -> None:
    """Flatten a metadata argument into ``sites``, keyed by its position."""
    if isinstance(value, AttentionMetadata):
        sites[name] = value
    elif isinstance(value, dict):
        for key, item in value.items():
            _collect_attn_metadata(f"{name}.{key}", item, sites)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _collect_attn_metadata(f"{name}[{index}]", item, sites)


def _attn_metadata_shape_key(*args: Any, **kwargs: Any) -> Optional[tuple]:
    """CUDA graph key from every ``attn_metadata*`` keyword, or ``None`` if absent.

    Accepts a single site, or a dict or list of them (Cosmos3 passes a per-sample
    list); the tensor-shape key cannot see metadata, so sites of differing length
    must not share a graph.
    """
    sites: dict[str, AttentionMetadata] = {}
    for name, value in kwargs.items():
        if name.startswith("attn_metadata"):
            _collect_attn_metadata(name, value, sites)

    key = []
    for name in sorted(sites):
        metadata = sites[name]
        seq_lens = metadata.seq_lens
        seq_lens_kv = metadata.seq_lens_kv
        key.append(
            (
                name,
                tuple(seq_lens.tolist()) if seq_lens is not None else None,
                # Only a cross site's KV length is independent of the Q length.
                tuple(seq_lens_kv.tolist()) if metadata.is_cross else None,
            )
        )
    return tuple(key) or None


class BaseDiffusionModel(nn.Module):
    """Base class for TRT-LLM VisualGen model components."""

    def __init__(self, model_config: DiffusionModelConfig):
        super().__init__()
        self.model_config = model_config
        self.component_name = model_config.component_name
        self.pretrained_config = model_config.pretrained_config

    @property
    def attn_backend_metadata_cls(self) -> type[AttentionMetadata]:
        """Metadata type this component's attention backend expects."""
        return get_visual_gen_attention_backend(self.model_config.attention.backend).Metadata

    @property
    def attn_requires_metadata(self) -> bool:
        """Whether any attention site in this model needs prepared metadata."""
        from tensorrt_llm._torch.visual_gen.modules.attention import Attention

        return any(m.requires_metadata for m in self.modules() if isinstance(m, Attention))

    def create_attn_metadata(self, **kwargs) -> dict[str, AttentionMetadata]:
        """Build one prepared metadata object per attention site."""
        raise NotImplementedError(
            "Diffusion model subclasses must implement create_attn_metadata()."
        )

    def forward(self, *args: Any, timestep: torch.Tensor | None = None, **kwargs: Any) -> Any:
        """Run the diffusion transformer.

        Concrete VisualGen models own their full forward signatures. This base
        method defines the common arguments that every forward should accept.

        Attention metadata is a required argument, one object per attention
        site. Callers build it from ``create_attn_metadata()``; a model must
        never construct it, which is illegal under CUDA graph capture.

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

        Override when a forward input changes captured execution without
        changing tensor shapes, calling ``runner.register_extra_key_fn(name,
        fn)``; ``fn`` takes the forward args and returns a hashable key or
        ``None`` to omit it. Subclasses should call ``super()``.
        """
        runner.register_extra_key_fn("attn_metadata_shape", _attn_metadata_shape_key)

        sparse_config = self.model_config.attention.sparse_attention_config
        if not isinstance(sparse_config, SkipSoftmaxAttentionConfig):
            return

        disabled_until_timestep = sparse_config.disabled_until_timestep
        if disabled_until_timestep is None:
            return

        # Skip Softmax switches attention behavior at the timestep boundary
        # without changing shapes, so key its dense and sparse phases apart.
        runner.register_extra_key_fn(
            "skip_softmax_phase",
            lambda *args, **kwargs: SkipSoftmaxScheduler.get_graph_phase_for_timestep(
                kwargs.get("timestep"),
                disabled_until_timestep=disabled_until_timestep,
            ),
        )
