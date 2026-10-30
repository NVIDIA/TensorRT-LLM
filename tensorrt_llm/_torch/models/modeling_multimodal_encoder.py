# Copyright 2024-2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# This file is based on official VILA: https://github.com/NVlabs/VILA/blob/main/llava/model/multimodal_encoder/

import os
from typing import TYPE_CHECKING, Optional, Type, cast

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

from ..._utils import prefer_pinned
from ..attention.backends.interface import AttentionMetadata
from .modeling_multimodal_utils import multiscale_forward

if TYPE_CHECKING:
    from ...llmapi.llm_args import (MultimodalConfig,
                                    MultimodalEncoderCudaGraphConfig)
    from .multimodal_encoder_graph import MultimodalEncoderGraphRunner

# Legacy fallback for encoders that do not yet opt into atomic-item scheduling
# or provide a model-specific item/token-to-segment mapping. Their encoder
# forward is not runtime-budgeted, so configured item/token limits cannot
# safely size fixed ``AttentionMetadata`` per-segment buffers.
#
# TODO: Replace this fallback as the remaining encoders provide model-specific
# conversions from atomic item/token budgets to attention segments. The exact
# conversion requires each encoder's minimum tokens per tile/window/segment, so
# retain this worst-case capacity for models without that contract.
# Runtime-scheduled Qwen and Mistral/Pixtral encoders override
# ``get_encoder_attention_metadata_capacity`` and do not consume this fallback.
_ENCODER_FALLBACK_MAX_NUM_REQUESTS = 8192


def prepare_encoder_attn_metadata(seq_lens: list[int],
                                  attn_metadata: AttentionMetadata, *,
                                  max_seq_len: int) -> AttentionMetadata:
    """Refresh packed no-KV encoder lengths while preserving captured buffers.

    Args:
        seq_lens: Nonempty list of nonnegative lengths for independent attention segments,
            including the isolated padding segment when using a padded graph.
        attn_metadata: Initialized encoder metadata. Captured length buffers are updated in place.
        max_seq_len: Positive, stable per-segment capacity, at least the largest live length.

    Returns:
        The supplied metadata, prepared for context-only attention without a KV cache.
    """
    seq_lens = [int(seq_len) for seq_len in seq_lens]
    if not seq_lens:
        raise ValueError("Encoder attention requires at least one segment")

    min_seq_len = min(seq_lens)
    if min_seq_len < 0:
        raise ValueError(
            f"Encoder attention segment length must be nonnegative, got {min_seq_len}"
        )

    if max_seq_len <= 0:
        raise ValueError(
            f"Encoder attention max_seq_len must be positive, got {max_seq_len}"
        )
    actual_max_seq_len = max(seq_lens)
    if actual_max_seq_len > max_seq_len:
        raise ValueError(
            "Encoder attention segment length exceeds the configured "
            f"maximum: actual={actual_max_seq_len}, maximum={max_seq_len}. "
            "Increase encoder_max_num_tokens or reduce the input max_pixels.")

    num_segments = len(seq_lens)
    seq_lens_torch = torch.tensor(seq_lens,
                                  dtype=torch.int,
                                  pin_memory=prefer_pinned())
    cu_seqlens = torch.empty(num_segments + 1,
                             dtype=torch.int32,
                             pin_memory=prefer_pinned())
    cu_seqlens[0] = 0
    torch.cumsum(seq_lens_torch, dim=0, out=cu_seqlens[1:])

    attn_metadata.num_contexts = num_segments
    attn_metadata.request_ids = list(range(1, num_segments + 1))
    prepare_graph_replay = getattr(attn_metadata,
                                   "prepare_encoder_cuda_graph_replay", None)
    is_cuda_graph = attn_metadata.is_cuda_graph
    if is_cuda_graph and callable(prepare_graph_replay):
        prepare_graph_replay(seq_lens, sum(seq_lens))
        # The replay helper refreshes only the host lengths, but the captured
        # attention op also reads the device lengths; update them in place.
        attn_metadata.seq_lens_cuda[:num_segments].copy_(seq_lens_torch,
                                                         non_blocking=True)
    else:
        attn_metadata.seq_lens = seq_lens_torch
    cu_seqlens = cu_seqlens.to(device=attn_metadata.seq_lens_cuda.device,
                               non_blocking=True)
    if (is_cuda_graph and attn_metadata.cu_q_seqlens is not None
            and attn_metadata.cu_q_seqlens.shape == cu_seqlens.shape):
        # CUDA graph capture bakes these addresses into attention kernels.
        # Refresh their contents without replacing the captured buffers.
        attn_metadata.cu_q_seqlens.copy_(cu_seqlens, non_blocking=True)
        if (attn_metadata.cu_kv_seqlens.data_ptr()
                != attn_metadata.cu_q_seqlens.data_ptr()):
            attn_metadata.cu_kv_seqlens.copy_(cu_seqlens, non_blocking=True)
    else:
        attn_metadata.cu_q_seqlens = cu_seqlens
        attn_metadata.cu_kv_seqlens = cu_seqlens
    # Keep the native attention-op cache key stable across image resolutions.
    # Actual segment lengths remain available through seq_lens / cu_seqlens.
    attn_metadata.max_seq_len = max_seq_len
    # The vision tower runs no-cache, context-only attention and supplies its
    # own `cu_seqlens` above, so the heavy KV-oriented `prepare()` (kv_lens /
    # prompt_lens / host_request_types setup) is unnecessary host work.
    # `prepare_encoder_only` runs the lean path on backends that have one
    # (TRTLLM) and falls back to the full `prepare()` elsewhere.
    attn_metadata.prepare_encoder_only()
    return attn_metadata


class MultimodalEncoderMixin:
    """Encoder-side counterpart to ``MultimodalModelMixin``.

    Marker + default ``setup_attn_metadata`` for multimodal encoders whose
    ``AttentionMetadata`` is built by ``ModelEngine`` after model load
    using the runtime encoder token budget (``max_num_tokens``).

    Subclasses set ``metadata_cls`` in their own ``__init__`` (typically from
    ``get_attention_backend(model_config.attn_backend).Metadata``) and either
    use the default ``setup_attn_metadata`` below or override it for custom
    Metadata kwargs (e.g. FlashInfer ``kv_layout``, multi-metadata encoders).
    """
    metadata_cls: Type[AttentionMetadata]
    attn_metadata: Optional[AttentionMetadata] = None
    encoder_cuda_graph_modality: str | None = None
    _encoder_cuda_graph_config: "MultimodalEncoderCudaGraphConfig | None" = None
    _blocks_graph_runner: "MultimodalEncoderGraphRunner | None" = None

    @property
    def encoder_cuda_graph_config(
            self) -> "MultimodalEncoderCudaGraphConfig | None":
        """CUDA graph configuration selected for this encoder tower."""
        return self._encoder_cuda_graph_config

    @encoder_cuda_graph_config.setter
    def encoder_cuda_graph_config(
            self, config: "MultimodalEncoderCudaGraphConfig | None") -> None:
        self._encoder_cuda_graph_config = config

    def configure_encoder_cuda_graph(
            self, multimodal_config: "MultimodalConfig | None") -> None:
        """Select this tower's configuration without rejecting other towers' modalities.

        Subclasses declare ``encoder_cuda_graph_modality`` when their graph factory is supported.
        The engine validates the requested keys against all local encoder capabilities before
        capturing any graphs. Explicit configs for standalone encoders remain supported.

        Args:
            multimodal_config: Model-wide configuration shared by its encoder towers.
        """
        graph_configs = (multimodal_config.encoder_cuda_graph
                         if multimodal_config is not None else None)
        self._encoder_cuda_graph_config = (
            graph_configs.get(self.encoder_cuda_graph_modality)
            if graph_configs is not None
            and self.encoder_cuda_graph_modality is not None else None)

    def _build_blocks_graph_runner(
        self, config: "MultimodalEncoderCudaGraphConfig"
    ) -> "MultimodalEncoderGraphRunner":
        """Describe the block function, tensor specs, and metadata provider.

        Encoders opting into graphs override this factory. Model-specific
        preprocessing and output assembly stay in the encoder forward.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support encoder CUDA graphs")

    def enable_blocks_cuda_graph(
        self,
        config: "MultimodalEncoderCudaGraphConfig | None" = None,
        *,
        device: torch.device | None = None,
        max_num_tokens: int | None = None,
    ) -> None:
        """Capture configured block graphs once, after weights and metadata are ready.

        The engine drives this lifecycle for every local multimodal encoder.
        Standalone callers may provide an explicit config and device. Automatic
        configs use their own token cap or the engine's original encoder budget;
        raising the scheduling budget for an oversized item does not enlarge captures.
        """
        config = config if config is not None else self._encoder_cuda_graph_config
        if config is None or self._blocks_graph_runner is not None:
            return
        if config.buckets is None and config.max_num_tokens is None:
            if max_num_tokens is None or max_num_tokens <= 0:
                raise ValueError("Automatic encoder graphs need max_num_tokens "
                                 "from the config or engine.")
            config = config.model_copy(
                update={"max_num_tokens": max_num_tokens})
        runner = self._build_blocks_graph_runner(config)
        if device is None:
            device = next(cast(nn.Module, self).parameters()).device
        runner.capture_all(device)
        self._blocks_graph_runner = runner

    def clear_blocks_cuda_graph(self,
                                *,
                                release_nccl_window_owners: bool = True
                                ) -> None:
        """Release block graphs at the engine's coordinated teardown boundary."""
        if self._blocks_graph_runner is not None:
            self._blocks_graph_runner.clear(
                release_nccl_window_owners=release_nccl_window_owners)
            self._blocks_graph_runner = None

    def get_encoder_attention_metadata_capacity(
            self, max_num_tokens: int) -> dict[str, int]:
        """Map the token budget to model-internal attention sequences.

        Keys name this encoder's attention metadata objects, so they are
        model-specific: this default declares a single `attention` object,
        while a windowed encoder such as Qwen2.5-VL declares `full_attention`
        and `window_attention`. There is no fixed superset to enumerate.

        The default conservatively allows one attention sequence per token.
        Encoders with tighter geometry-aware bounds should override this.
        """
        return {
            "attention": max(max_num_tokens, _ENCODER_FALLBACK_MAX_NUM_REQUESTS)
        }

    def setup_attn_metadata(
        self,
        max_num_tokens: int,
        attention_metadata_capacity: Optional[dict[str, int]] = None,
    ) -> None:
        """Map encoder item/token budgets to attention metadata capacity."""
        capacities = (
            attention_metadata_capacity
            if attention_metadata_capacity is not None else
            self.get_encoder_attention_metadata_capacity(max_num_tokens))
        self.attn_metadata = self.metadata_cls(  # type: ignore[call-arg]
            max_num_requests=capacities["attention"],
            max_num_tokens=max_num_tokens,
            kv_cache_manager=None,
        )

    def set_attn_max_seq_len(self, max_seq_len: int) -> None:
        """Set an optional stable per-segment attention capacity.

        Specialized encoders may override this hook. Keeping it separate from
        `setup_attn_metadata` preserves that method's existing override
        contract for external encoders.
        """


class VisionTower(nn.Module):

    def __init__(self, model_name_or_path, config):
        super().__init__()

        assert os.path.exists(
            model_name_or_path
        ), f"Pretrained vision tower path {model_name_or_path} does not exist!"
        vision_tower_cfg = AutoConfig.from_pretrained(model_name_or_path,
                                                      trust_remote_code=True)
        self.name = vision_tower_cfg.architectures[0].lower()

        if "clip" in self.name:
            self.vision_tower = AutoModel.from_pretrained(
                model_name_or_path, dtype=config.model_dtype)
        elif "siglip" in self.name:
            self.vision_tower = AutoModel.from_pretrained(
                model_name_or_path,
                attn_implementation="flash_attention_2",
                dtype="auto")
        else:
            raise ValueError(f"Unsupported vision tower: {self.name}")

        self.select_layer = getattr(config, "mm_vision_select_layer", -2)
        self.select_feature = getattr(config, "mm_vision_select_feature",
                                      "patch")

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == "patch":
            image_features = image_features[:, 1:]
        elif self.select_feature == "cls_patch":
            image_features = image_features
        else:
            raise ValueError(
                f"Unexpected select feature: {self.select_feature}")
        return image_features

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.vision_tower(
                    image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                    output_hidden_states=True,
                )
                image_feature = self.feature_select(image_forward_out).to(
                    image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(
                images.to(device=self.device, dtype=self.dtype),
                output_hidden_states=True,
            )
            image_features = self.feature_select(image_forward_outs).to(
                images.dtype)

        return image_features

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        return self.vision_tower.config

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size)**2


class VisionTowerS2(VisionTower):

    def __init__(self, model_name_or_path, config):
        super().__init__(model_name_or_path, config)

        self.scales = list(map(int, config.s2_scales.split(",")))
        self.scales.sort()
        self.max_split_size = config.s2_max_split_size
        self.resize_output_to_scale_idx = getattr(
            config, "s2_resize_output_to_scale_idx", 0)

    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device,
                                                         dtype=self.dtype),
                                               output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(
            images.dtype)
        return image_features

    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = multiscale_forward(
                    self.forward_feature,
                    image.unsqueeze(0),
                    img_sizes=self.scales,
                    max_split_size=self.max_split_size,
                    resize_output_to_idx=self.resize_output_to_scale_idx,
                )
                image_features.append(image_feature)
        else:
            image_features = multiscale_forward(
                self.forward_feature,
                images,
                img_sizes=self.scales,
                max_split_size=self.max_split_size,
                resize_output_to_idx=self.resize_output_to_scale_idx,
            )

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.scales)


class VisionTowerDynamicS2(VisionTowerS2):

    def __init__(self, model_name_or_path, config):
        super().__init__(model_name_or_path, config)

    def forward(self, images):
        assert type(images) is not list

        image_features = self.forward_feature(images)

        return image_features
