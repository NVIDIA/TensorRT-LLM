# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cosmos3-Edge Reasoner.

The Edge checkpoint is a unified Reasoner + Generator model. This module loads
only the understanding path: a SigLIP2 vision tower, a 4608 -> 11520 -> 2048
projector, and a dense decoder (attention, RMSNorm, ReLU-squared MLP) with
interleaved multimodal RoPE. Generator experts are not constructed.
"""

import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import safetensors.torch
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.transforms import InterpolationMode
from transformers import AutoProcessor, AutoTokenizer, PretrainedConfig
from transformers.feature_extraction_utils import BatchFeature
from transformers.image_utils import ChannelDimension, PILImageResampling, SizeDict, get_image_size
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
    Qwen3VLVideoProcessor,
    smart_resize,
)
from transformers.models.siglip2.image_processing_siglip2 import (
    Siglip2ImageProcessor,
    convert_image_to_patches,
)
from transformers.utils import TensorType
from transformers.video_processing_utils import BaseVideoProcessor
from transformers.video_utils import group_videos_by_shape, reorder_videos

from tensorrt_llm._torch.configs.cosmos3_edge import _as_torch_dtype
from tensorrt_llm.functional import PositionEmbeddingType, RotaryScalingType
from tensorrt_llm.mapping import Mapping

from ...inputs import (
    ContentFormat,
    MultimodalPlaceholderMetadata,
    MultimodalPlaceholderPlacement,
    register_input_processor,
    support_multimodal_disaggregated,
)
from ...inputs.multimodal import MultimodalParams
from ..attention.attention import Attention, maybe_allgather_for_helix_cp, maybe_slice_for_helix_cp
from ..attention.backends import AttentionMetadata
from ..attention.backends.interface import PositionalEmbeddingParams, RopeParams
from ..attention.backends.utils import get_attention_backend
from ..attention.rotary_embedding import MRotaryEmbedding
from ..distributed import AllReduceParams
from ..model_config import ModelConfig
from ..modules.decoder_layer import DecoderLayer
from ..modules.embedding import Embedding
from ..modules.layer_norm import LayerNorm
from ..modules.linear import TensorParallelMode
from ..modules.mlp import MLP
from ..modules.rms_norm import RMSNorm
from ..speculative import SpecMetadata
from ..utils import relu2
from .checkpoints.base_weight_mapper import BaseWeightMapper
from .checkpoints.hf.cosmos3_edge_weight_mapper import Cosmos3EdgeHfWeightMapper
from .modeling_cosmos3 import _get_cosmos3_model_paths
from .modeling_multimodal_encoder import MultimodalEncoderMixin
from .modeling_multimodal_mixin import (
    EncoderGroup,
    MultimodalModelMixin,
    encode_multimodal_by_groups,
)
from .modeling_qwen2vl import Qwen2_5_VLVisionAttention, _prepare_qwen_vl_vision_attn_metadata
from .modeling_qwen3vl import (
    Qwen3VLInputProcessorBase,
    Qwen3VLModelBase,
    Qwen3VLVisionMLP,
    Qwen3VLVisionPatchMerger,
    _qwen3vl_build_batched_input,
)
from .modeling_speculative import SpecDecOneEngineForCausalLM
from .modeling_utils import (
    DecoderModel,
    QuantConfig,
    _load_weights_impl,
    filter_weights,
    register_auto_model,
    register_vision_encoder,
)

_VISION_NORM_EPS = 1e-6


def _config_dtype(config: PretrainedConfig) -> torch.dtype:
    value = getattr(config, "dtype", None)
    if not isinstance(value, (torch.dtype, str)):
        value = getattr(config, "torch_dtype", None)
    if value is None:
        raise ValueError(f"{type(config).__name__} is missing dtype.")
    return _as_torch_dtype(value)


def _cpu_int_rows(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.device.type == "cpu" and tensor.dtype == torch.int64:
        return tensor
    return tensor.detach().to(device="cpu", dtype=torch.int64)


_VISION_KEY_RENAMES = (
    (re.compile(r"(.*)\.self_attn\.out_proj\.(.*)"), r"\1.self_attn.o_proj.\2"),
    (re.compile(r"(.*)\.mlp\.fc1\.(.*)"), r"\1.mlp.up_proj.\2"),
    (re.compile(r"(.*)\.mlp\.fc2\.(.*)"), r"\1.mlp.down_proj.\2"),
)


def _rename_vision_module_keys(weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Rename SigLIP2 module names. Keys that already use the module names stay put."""
    renamed = {}
    for key, value in weights.items():
        new_key = key
        for pattern, replacement in _VISION_KEY_RENAMES:
            updated = pattern.sub(replacement, key)
            if updated != key:
                new_key = updated
                break
        renamed[new_key] = value
    return renamed


def resize_positional_embeddings_packed(
    positional_embeddings: torch.Tensor,
    spatial_shapes: torch.Tensor,
    lengths: List[int],
) -> torch.Tensor:
    """Bilinearly resample a square position grid onto each packed frame.

    ``positional_embeddings`` is ``(grid, grid, hidden)``. Each ``(height, width)``
    in ``spatial_shapes`` is a patch grid, not pixels. Interpolation uses
    ``align_corners=False`` and ``antialias=True``, which is the SigLIP2 rule
    rather than Qwen3-VL's ``(n - 1) / (h - 1)`` grid.
    """
    spatial_shapes = _cpu_int_rows(spatial_shapes)
    embed_dim = positional_embeddings.shape[-1]
    source_dtype = positional_embeddings.dtype
    total_tokens = int(sum(lengths))
    packed = torch.empty(
        (total_tokens, embed_dim),
        device=positional_embeddings.device,
        dtype=source_dtype,
    )
    pos_4d = positional_embeddings.permute(2, 0, 1).unsqueeze(0)
    if pos_4d.device.type == "cpu":
        pos_4d = pos_4d.to(torch.float32)
    rows = spatial_shapes.tolist()
    if len(rows) != len(lengths):
        raise ValueError(
            f"spatial_shapes has {len(rows)} rows but lengths has {len(lengths)} entries."
        )
    resized_by_shape: Dict[Tuple[int, int], torch.Tensor] = {}
    offset = 0
    index = 0
    while index < len(rows):
        height, width = int(rows[index][0]), int(rows[index][1])
        run = 1
        while (
            index + run < len(rows)
            and int(rows[index + run][0]) == height
            and int(rows[index + run][1]) == width
        ):
            run += 1
        key = (height, width)
        if key not in resized_by_shape:
            resized = F.interpolate(
                pos_4d,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
                antialias=True,
            )
            resized_by_shape[key] = (
                resized.reshape(embed_dim, height * width).transpose(0, 1).to(source_dtype)
            )
        frame = resized_by_shape[key]
        block = frame.repeat(run, 1) if run > 1 else frame
        length = int(sum(lengths[index : index + run]))
        packed[offset : offset + length] = block
        offset += length
        index += run
    return packed


def merge_spatial_patches(
    image_embeds: torch.Tensor,
    grid_thw: torch.Tensor,
    merge_size: int = 2,
) -> torch.Tensor:
    """Pack each ``merge_size x merge_size`` spatial block into the channel axis."""
    merged: List[torch.Tensor] = []
    current = 0
    hidden_size = image_embeds.shape[-1]
    grids = _cpu_int_rows(grid_thw).tolist()
    for grid in grids:
        frames, height, width = (int(v) for v in grid)
        if height % merge_size != 0 or width % merge_size != 0:
            raise ValueError(
                "Grid height and width must be divisible by merge_size: "
                f"got grid={(frames, height, width)} and merge_size={merge_size}."
            )
        num_patches = frames * height * width
        media = image_embeds[current : current + num_patches]
        if media.shape[0] != num_patches:
            raise ValueError("image_embeds contains fewer patches than grid_thw describes.")
        current += num_patches
        media = media.view(frames, height, width, hidden_size)
        media = media.view(
            frames,
            height // merge_size,
            merge_size,
            width // merge_size,
            merge_size,
            hidden_size,
        )
        media = media.permute(0, 1, 3, 2, 4, 5).reshape(-1, merge_size * merge_size * hidden_size)
        merged.append(media)
    if current != image_embeds.shape[0]:
        raise ValueError(
            "image_embeds contains more patches than grid_thw describes: "
            f"got {image_embeds.shape[0]}, expected {current}."
        )
    if not merged:
        return image_embeds.new_empty((0, merge_size * merge_size * hidden_size))
    if len(merged) == 1:
        return merged[0]
    return torch.cat(merged, dim=0)


class Cosmos3EdgeVisionAttention(Qwen2_5_VLVisionAttention):
    """Packed full attention for SigLIP2. Positions are added as embeddings, not RoPE."""

    def __init__(self, model_config: ModelConfig[PretrainedConfig], layer_idx: int):
        super().__init__(
            model_config,
            layer_idx=layer_idx,
            reduce_output=(
                not model_config.mapping.enable_attention_dp and model_config.mapping.tp_size > 1
            ),
        )

    def apply_rope(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        position_ids: Optional[torch.IntTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        packed_qk: Optional[torch.Tensor] = None,
    ):
        del position_ids, position_embeddings
        return q, k, v


class Cosmos3EdgeVisionBlock(nn.Module):
    def __init__(self, model_config: ModelConfig[PretrainedConfig], layer_idx: int):
        super().__init__()
        config = model_config.pretrained_config.vision_config
        dtype = _config_dtype(config)
        eps = getattr(config, "layer_norm_eps", _VISION_NORM_EPS)
        self.layer_norm1 = LayerNorm(hidden_size=config.hidden_size, eps=eps, dtype=dtype)
        self.layer_norm2 = LayerNorm(hidden_size=config.hidden_size, eps=eps, dtype=dtype)
        self.self_attn = Cosmos3EdgeVisionAttention(model_config, layer_idx)
        self.mlp = Qwen3VLVisionMLP(model_config, layer_idx)

    def forward(
        self, hidden_states: torch.Tensor, attn_metadata: AttentionMetadata
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.self_attn(
            hidden_states=self.layer_norm1(hidden_states),
            attn_metadata=attn_metadata,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.mlp(self.layer_norm2(hidden_states))
        return residual + hidden_states


class Cosmos3EdgeSiglip(nn.Module, MultimodalEncoderMixin):
    """SigLIP2 tower: linear patch embed, interpolated positions, per-frame attention."""

    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__()
        self.model_config = model_config
        self.config = model_config.pretrained_config.vision_config
        text_dtype = _config_dtype(self.config)
        patch_dim = self.config.num_channels * self.config.patch_size * self.config.patch_size
        self.embeddings = nn.Module()
        self.embeddings.patch_embedding = nn.Linear(patch_dim, self.config.hidden_size, bias=True)
        self.embeddings.position_embedding = nn.Embedding(
            self.config.num_patches, self.config.hidden_size
        )
        self.position_embedding_size = int(self.config.num_patches**0.5)
        self.encoder = nn.Module()
        self.encoder.layers = nn.ModuleList(
            [
                Cosmos3EdgeVisionBlock(model_config, layer_idx)
                for layer_idx in range(self.config.num_hidden_layers)
            ]
        )
        self.post_layernorm = LayerNorm(
            hidden_size=self.config.hidden_size,
            eps=getattr(self.config, "layer_norm_eps", _VISION_NORM_EPS),
            dtype=text_dtype,
        )
        self.metadata_cls = get_attention_backend(model_config.attn_backend).Metadata
        self.attn_metadata: Optional[AttentionMetadata] = None
        self._fixed_max_seq_len = model_config.max_num_tokens
        self.spatial_merge_size = self.config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

    @property
    def device(self) -> torch.device:
        return self.embeddings.patch_embedding.weight.device

    def setup_attn_metadata(
        self,
        max_num_tokens: int,
        attention_metadata_capacity: Optional[Dict[str, int]] = None,
    ) -> None:
        if attention_metadata_capacity is None:
            attention_metadata_capacity = {
                "attention": max(1, max_num_tokens // self.spatial_merge_unit)
            }
        self.attn_metadata = self.metadata_cls(
            max_num_requests=attention_metadata_capacity["attention"],
            max_num_tokens=max_num_tokens,
            kv_cache_manager=None,
        )
        self.set_attn_max_seq_len(max_num_tokens)

    def set_attn_max_seq_len(self, max_seq_len: int) -> None:
        if max_seq_len <= 0:
            raise ValueError(
                f"Cosmos3-Edge vision attention max_seq_len must be positive, got {max_seq_len}"
            )
        self._fixed_max_seq_len = max_seq_len

    def get_encoder_attention_metadata_capacity(self, max_num_tokens: int) -> Dict[str, int]:
        return {"attention": max(1, max_num_tokens // self.spatial_merge_unit)}

    def _spatial_shapes(
        self, grid_thw: torch.Tensor, grid_cpu: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[int]]:
        grid_cpu = _cpu_int_rows(grid_thw if grid_cpu is None else grid_cpu)
        shapes = torch.repeat_interleave(grid_cpu[:, 1:], grid_cpu[:, 0], dim=0)
        lengths = (shapes[:, 0] * shapes[:, 1]).tolist()
        return shapes, [int(length) for length in lengths]

    def embed(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        grid_cpu: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if pixel_values.dim() == 3:
            if pixel_values.shape[0] != 1:
                raise ValueError(
                    f"Packed pixel_values batch must be 1, got {tuple(pixel_values.shape)}."
                )
            pixel_values = pixel_values[0]
        shapes, lengths = self._spatial_shapes(grid_thw, grid_cpu=grid_cpu)
        if sum(lengths) != pixel_values.shape[0]:
            raise ValueError(
                "Packed pixel_values token count does not match grid_thw: "
                f"{pixel_values.shape[0]} vs {sum(lengths)}."
            )
        weight = self.embeddings.patch_embedding.weight
        hidden = self.embeddings.patch_embedding(pixel_values.to(dtype=weight.dtype))
        grid = self.position_embedding_size
        positional = self.embeddings.position_embedding.weight.reshape(grid, grid, -1)
        return hidden + resize_positional_embeddings_packed(positional, shapes, lengths)

    def forward(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        grid_cpu: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.attn_metadata is None:
            raise RuntimeError(
                "Vision encoder AttentionMetadata is not initialized. "
                "It must be set up before the encoder forward runs."
            )
        grid_cpu = _cpu_int_rows(grid_thw if grid_cpu is None else grid_cpu)
        _, lengths = self._spatial_shapes(grid_thw, grid_cpu=grid_cpu)
        self.attn_metadata = _prepare_qwen_vl_vision_attn_metadata(
            lengths, self.attn_metadata, max_seq_len=self._fixed_max_seq_len
        )
        hidden = self.embed(pixel_values, grid_thw, grid_cpu=grid_cpu)
        for block in self.encoder.layers:
            hidden = block(hidden, self.attn_metadata)
        return self.post_layernorm(hidden)


class Cosmos3EdgeVisionEncoder(nn.Module):
    """SigLIP2 encoder plus the Edge projector. Returns language-model embeddings."""

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        model_class: type = Cosmos3EdgeSiglip,
    ):
        super().__init__()
        self.model_config = model_config
        self.model_config.quant_config = QuantConfig()
        pretrained = model_config.pretrained_config
        self.config = pretrained.vision_config
        encoder_dtype = _config_dtype(self.config)
        self.visual = MultimodalModelMixin._cast_multimodal_encoder_dtype(
            model_class(model_config), encoder_dtype
        )
        projector = pretrained.projector_config
        self.projector = Qwen3VLVisionPatchMerger(
            model_config=model_config,
            use_postshuffle_norm=bool(getattr(projector, "use_postshuffle_norm", False)),
            merger_intermediate_size=int(projector.merger_intermediate_size),
            norm_eps=_VISION_NORM_EPS,
        )
        self.projector = MultimodalModelMixin._cast_multimodal_encoder_dtype(
            self.projector, encoder_dtype
        )
        self.spatial_merge_size = int(projector.spatial_merge_size)
        self._vision_encoder_path = None
        if getattr(pretrained, "_name_or_path", None):
            _, _, self._vision_encoder_path = _get_cosmos3_model_paths(pretrained)

    @property
    def llm_checkpoint_dir(self) -> str:
        """Directory whose safetensors are the SigLIP2 tower and projector."""
        if not self._vision_encoder_path:
            raise ValueError(
                "Cosmos3EdgeVisionEncoder requires pretrained_config._name_or_path "
                "to locate vision_encoder/."
            )
        return self._vision_encoder_path

    def setup_attn_metadata(self, max_num_tokens: int, attention_metadata_capacity=None) -> None:
        self.visual.setup_attn_metadata(max_num_tokens, attention_metadata_capacity)

    def load_weights(self, weights: Dict[str, torch.Tensor], allow_partial_loading: bool = False):
        visual_weights = _rename_vision_module_keys(filter_weights("model.visual", weights))
        projector_weights = filter_weights("model.projector", weights)
        combined = {f"visual.{name}": value for name, value in visual_weights.items()}
        combined.update({f"projector.{name}": value for name, value in projector_weights.items()})
        _load_weights_impl(self, combined, allow_partial_loading=allow_partial_loading)

    @torch.inference_mode()
    def encode_batched(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        grid_cpu = _cpu_int_rows(grid_thw)
        hidden = self.visual(pixel_values, grid_thw, grid_cpu=grid_cpu)
        merged = merge_spatial_patches(hidden, grid_cpu, merge_size=self.spatial_merge_size)
        merged = merged.view(-1, self.spatial_merge_size**2, hidden.shape[-1])
        return self.projector(merged)

    @property
    def mm_encoder_groups(self) -> Tuple[EncoderGroup, ...]:
        return (
            EncoderGroup(
                modalities=("image", "video"),
                encoder_fn=self.encode_batched,
                build_batched_input=_qwen3vl_build_batched_input,
            ),
        )

    def forward(self, multimodal_params: List[MultimodalParams]) -> List[torch.Tensor]:
        return [encode_multimodal_by_groups(self.mm_encoder_groups, multimodal_params)]


class Cosmos3EdgeAttention(Attention):
    """Dense attention with interleaved mRoPE and no Q/K norm."""

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: Optional[int] = None,
        reduce_output: bool = True,
        mapping_with_cp: Optional[Mapping] = None,
    ):
        config = model_config.pretrained_config
        pos_embd_params = _edge_positional_embedding_params(config)
        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            max_position_embeddings=config.max_position_embeddings,
            bias=bool(getattr(config, "attention_bias", False)),
            pos_embd_params=pos_embd_params,
            rope_fusion=not getattr(config, "disable_fuse_rope", False),
            layer_idx=layer_idx,
            dtype=_config_dtype(config),
            dense_bias=bool(getattr(config, "attention_bias", False)),
            config=model_config,
            reduce_output=reduce_output,
            mapping_with_cp=mapping_with_cp,
            head_dim=getattr(config, "head_dim", None),
        )


class Cosmos3EdgeDecoderLayer(DecoderLayer):
    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        layer_idx: int,
        mapping_with_cp: Optional[Mapping] = None,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        config = model_config.pretrained_config
        self.mapping = model_config.mapping
        self.mapping_with_cp = mapping_with_cp
        self.enable_attention_dp = self.mapping.enable_attention_dp
        needs_tp_reduce = not self.enable_attention_dp and self.mapping.tp_size > 1
        needs_cp_reduce = mapping_with_cp is not None and mapping_with_cp.has_cp_helix()
        self.self_attn = Cosmos3EdgeAttention(
            model_config,
            layer_idx=layer_idx,
            mapping_with_cp=mapping_with_cp,
            reduce_output=needs_tp_reduce or needs_cp_reduce,
        )
        has_cp = mapping_with_cp is not None and mapping_with_cp.cp_size > 1
        can_skip_attn_for_dp = self.enable_attention_dp and not has_cp
        self.disable_attn_allreduce = self.mapping.tp_size == 1 or can_skip_attn_for_dp
        self.disable_mlp_allreduce = self.mapping.tp_size == 1 or self.enable_attention_dp
        dtype = _config_dtype(config)
        self.mlp = MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=bool(getattr(config, "mlp_bias", False)),
            activation=relu2,
            dtype=dtype,
            config=model_config,
            layer_idx=layer_idx,
            reduce_output=not self.disable_mlp_allreduce,
            overridden_tp_size=1 if self.enable_attention_dp else None,
        )
        self.input_layernorm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=dtype
        )
        self.post_attention_layernorm = RMSNorm(
            hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=dtype
        )

    def forward(
        self,
        position_ids: torch.IntTensor,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        residual: Optional[torch.Tensor],
        spec_metadata: Optional[SpecMetadata] = None,
        mrope_config: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            position_ids=position_ids,
            hidden_states=hidden_states,
            attn_metadata=attn_metadata,
            all_reduce_params=AllReduceParams(enable_allreduce=not self.disable_attn_allreduce),
            mrope_config=mrope_config,
            **kwargs,
        )
        residual = maybe_slice_for_helix_cp(
            residual, attn_metadata, self.mapping_with_cp, self.layer_idx
        )
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states, lora_params=kwargs.get("lora_params"))
        if spec_metadata is not None:
            spec_metadata.maybe_capture_hidden_states(self.layer_idx, hidden_states, residual)
        return hidden_states, residual


class Cosmos3EdgeDecoderModel(DecoderModel):
    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        mapping_with_cp: Optional[Mapping] = None,
    ):
        super().__init__(model_config)
        config = self.model_config.pretrained_config
        self.mapping_with_cp = mapping_with_cp
        dtype = _config_dtype(config)
        self.embed_tokens = Embedding(
            config.vocab_size,
            config.hidden_size,
            dtype=dtype,
            mapping=self.model_config.mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            gather_output=True,
        )
        self.layers = nn.ModuleList(
            [
                Cosmos3EdgeDecoderLayer(model_config, layer_idx, mapping_with_cp=mapping_with_cp)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)

    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        spec_metadata: Optional[SpecMetadata] = None,
        mrope_config: Optional[dict] = None,
        **kwargs,
    ) -> torch.Tensor:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, "
                "and must specify either one"
            )
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds
        residual = None
        for decoder_layer in self.layers:
            hidden_states, residual = decoder_layer(
                position_ids=position_ids,
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                residual=residual,
                spec_metadata=spec_metadata,
                mrope_config=mrope_config,
                **kwargs,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return maybe_allgather_for_helix_cp(hidden_states, attn_metadata, self.mapping_with_cp)


@register_auto_model("Cosmos3EdgeForCausalLM")
class Cosmos3EdgeForCausalLM(SpecDecOneEngineForCausalLM):
    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        self.mapping_with_cp = None
        if model_config.mapping.has_cp_helix():
            self.mapping_with_cp = model_config.mapping
            model_config._frozen = False
            model_config.mapping = model_config.mapping.repurpose_helix_cp_to_tp()
            model_config._frozen = True
        super().__init__(
            Cosmos3EdgeDecoderModel(model_config, mapping_with_cp=self.mapping_with_cp),
            model_config,
        )
        if self.mapping_with_cp is not None:
            model_config._frozen = False
            model_config.mapping = self.mapping_with_cp
            model_config._frozen = True


def round_by_factor(number: int, factor: int) -> int:
    """Return the integer closest to ``number`` that is divisible by ``factor``."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Return the smallest multiple of ``factor`` that is at least ``number``."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Return the largest multiple of ``factor`` that is at most ``number``."""
    return math.floor(number / factor) * factor


def _cacheable_stats(value):
    """Turn mean/std lists into tuples so rescale helpers can cache them."""
    if isinstance(value, list):
        return tuple(value)
    return value


class Cosmos3EdgeImagesKwargs(Siglip2ImageProcessor.valid_kwargs, total=False):
    max_pixels: int | None
    min_pixels: int | None
    per_image_kwargs: list[dict] | None
    merge_size: int | None


class Cosmos3EdgeImageProcessor(Siglip2ImageProcessor):
    resample = PILImageResampling.BICUBIC
    valid_kwargs = Cosmos3EdgeImagesKwargs

    def __init__(self, **kwargs):
        merge_size = kwargs.get("merge_size", 2)
        super().__init__(**kwargs)
        self.merge_size = merge_size or 2

    def _resize_image(
        self,
        image: torch.Tensor,
        max_ratio=200,
        resample: PILImageResampling | InterpolationMode | int | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Resize so both sides are divisible by ``merge_size * patch_size``.

        Each side stays at least one factor. A long image can otherwise floor
        its short side to zero while still passing the aspect-ratio check.
        """
        image_min_pixels = kwargs.get("min_pixels") or self.size.get("shortest_edge", None)
        image_max_pixels = kwargs.get("max_pixels") or self.size.get("longest_edge", None)
        if image_min_pixels is None or image_max_pixels is None:
            raise ValueError("When do_resize is True, min_pixels and max_pixels must be provided.")
        if image_max_pixels < image_min_pixels:
            raise ValueError("The max_pixels of image must be greater than or equal to min_pixels.")

        _, height, width = image.shape
        aspect_ratio = max(height, width) / min(height, width)
        if aspect_ratio > max_ratio:
            raise ValueError(
                f"absolute aspect ratio must be smaller than {max_ratio}, got {aspect_ratio}"
            )
        factor = self.merge_size * self.patch_size
        h_bar = max(factor, round_by_factor(height, factor))
        w_bar = max(factor, round_by_factor(width, factor))
        if h_bar * w_bar > image_max_pixels:
            beta = math.sqrt((height * width) / image_max_pixels)
            h_bar = max(factor, floor_by_factor(height / beta, factor))
            w_bar = max(factor, floor_by_factor(width / beta, factor))
        elif h_bar * w_bar < image_min_pixels:
            beta = math.sqrt(image_min_pixels / (height * width))
            h_bar = ceil_by_factor(height * beta, factor)
            w_bar = ceil_by_factor(width * beta, factor)

        return self.resize(
            image,
            size=SizeDict(height=h_bar, width=w_bar),
            resample=resample,
        )

    def _preprocess(
        self,
        images: list["torch.Tensor"],
        do_resize: bool,
        patch_size: int,
        max_num_patches: int,
        resample: PILImageResampling | InterpolationMode | int | None,
        do_rescale: bool,
        rescale_factor: float,
        do_normalize: bool,
        image_mean: float | list[float] | None,
        image_std: float | list[float] | None,
        return_tensors: str | TensorType | None,
        **kwargs,
    ) -> BatchFeature:
        del max_num_patches  # Edge sizes images by min/max pixels, not a patch cap.
        per_image_kwargs = kwargs.pop("per_image_kwargs", None)
        pixel_values = []
        spatial_shapes = []
        for idx, image in enumerate(images):
            image_kwargs = kwargs.copy()
            if (
                per_image_kwargs is not None
                and idx < len(per_image_kwargs)
                and per_image_kwargs[idx]
            ):
                image_kwargs.update(per_image_kwargs[idx])
            if do_resize:
                image = self._resize_image(
                    image,
                    max_ratio=200,
                    resample=resample,
                    **image_kwargs,
                )

            image = self.rescale_and_normalize(
                image,
                do_rescale,
                rescale_factor,
                do_normalize,
                _cacheable_stats(image_mean),
                _cacheable_stats(image_std),
            )
            patches = convert_image_to_patches(image, patch_size)
            num_patches_height = image.shape[1] // patch_size
            num_patches_width = image.shape[2] // patch_size
            spatial_shapes.append((num_patches_height, num_patches_width))
            pixel_values.append(patches)

        pixel_values = torch.cat(pixel_values, dim=0)
        spatial_shapes = torch.tensor(spatial_shapes)
        t_dim = torch.ones(
            (spatial_shapes.shape[0], 1),
            dtype=spatial_shapes.dtype,
            device=spatial_shapes.device,
        )
        image_grid_thw = torch.cat([t_dim, spatial_shapes], dim=1)
        return BatchFeature(
            data={"pixel_values": pixel_values, "image_grid_thw": image_grid_thw},
            tensor_type=return_tensors,
        )


class Cosmos3EdgeVideoProcessor(Qwen3VLVideoProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _preprocess(
        self,
        videos: list[torch.Tensor],
        do_convert_rgb: bool = True,
        do_resize: bool = True,
        size: SizeDict | None = None,
        resample: PILImageResampling | InterpolationMode | int | None = PILImageResampling.BICUBIC,
        do_rescale: bool = True,
        rescale_factor: float = 1 / 255.0,
        do_normalize: bool = True,
        image_mean: float | list[float] | None = None,
        image_std: float | list[float] | None = None,
        patch_size: int | None = None,
        return_tensors: str | TensorType | None = None,
        **kwargs,
    ):
        del do_convert_rgb, kwargs
        if patch_size is None:
            raise ValueError("Cosmos3-Edge video preprocessing requires patch_size.")

        merge_size = self.merge_size
        grouped_videos, grouped_videos_index = group_videos_by_shape(videos)
        resized_videos_grouped = {}

        for shape, stacked_videos in grouped_videos.items():
            batch, frames, channels, height, width = stacked_videos.shape
            if do_resize:
                if size is None:
                    raise ValueError("Cosmos3-Edge video resize requires a size dict.")
                resized_height, resized_width = smart_resize(
                    num_frames=frames,
                    height=height,
                    width=width,
                    temporal_factor=1,
                    factor=patch_size * merge_size,
                    min_pixels=size.shortest_edge,
                    max_pixels=size.longest_edge,
                )
                stacked_videos = stacked_videos.view(batch * frames, channels, height, width)
                stacked_videos = BaseVideoProcessor.resize(
                    self,
                    image=stacked_videos,
                    size=SizeDict(height=resized_height, width=resized_width),
                    resample=resample,
                )
                stacked_videos = stacked_videos.view(
                    batch, frames, channels, resized_height, resized_width
                )
            resized_videos_grouped[shape] = stacked_videos
        resized_videos = reorder_videos(resized_videos_grouped, grouped_videos_index)

        grouped_videos, grouped_videos_index = group_videos_by_shape(resized_videos)
        processed_videos_grouped = {}
        processed_grids = {}
        for shape, stacked_videos in grouped_videos.items():
            resized_height, resized_width = get_image_size(
                stacked_videos[0], channel_dim=ChannelDimension.FIRST
            )
            stacked_videos = self.rescale_and_normalize(
                stacked_videos,
                do_rescale,
                rescale_factor,
                do_normalize,
                _cacheable_stats(image_mean),
                _cacheable_stats(image_std),
            )
            batch_size, grid_t, channel = stacked_videos.shape[:3]
            grid_h, grid_w = resized_height // patch_size, resized_width // patch_size
            patches = stacked_videos.view(
                batch_size,
                grid_t,
                channel,
                grid_h,
                patch_size,
                grid_w,
                patch_size,
            )
            patches = patches.permute(0, 1, 3, 5, 4, 6, 2)
            flatten_patches = patches.reshape(
                batch_size,
                grid_t * grid_h * grid_w,
                patch_size * patch_size * channel,
            )
            processed_videos_grouped[shape] = flatten_patches
            processed_grids[shape] = [[grid_t, grid_h, grid_w]] * batch_size

        processed_videos = reorder_videos(processed_videos_grouped, grouped_videos_index)
        processed_grids = reorder_videos(processed_grids, grouped_videos_index)
        return BatchFeature(
            data={
                "pixel_values_videos": torch.cat(processed_videos, dim=0),
                "video_grid_thw": torch.tensor(processed_grids),
            },
            tensor_type=return_tensors,
        )


class Cosmos3EdgeProcessor(Qwen3VLProcessor):
    """Qwen3-VL chat template with Cosmos3-Edge image and video processors."""

    attributes = ["image_processor", "tokenizer", "video_processor"]

    @classmethod
    def _get_arguments_from_pretrained(
        cls,
        pretrained_model_name_or_path,
        processor_dict=None,
        **kwargs,
    ):
        del processor_dict
        component_kwargs = dict(kwargs)
        component_kwargs.pop("trust_remote_code", None)
        image_processor = Cosmos3EdgeImageProcessor.from_pretrained(
            pretrained_model_name_or_path,
            **component_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=False,
            **component_kwargs,
        )
        video_processor = Cosmos3EdgeVideoProcessor.from_pretrained(
            pretrained_model_name_or_path,
            **component_kwargs,
        )
        return [image_processor, tokenizer, video_processor]


def register_cosmos3_edge_processor() -> None:
    """Register the Edge processor so ``AutoProcessor.from_pretrained`` finds it."""
    from tensorrt_llm._torch.configs.cosmos3_edge import Cosmos3EdgeConfig

    try:
        AutoProcessor.register(Cosmos3EdgeConfig, Cosmos3EdgeProcessor, exist_ok=True)
    except TypeError:
        try:
            AutoProcessor.register(Cosmos3EdgeConfig, Cosmos3EdgeProcessor)
        except ValueError:
            return
    except ValueError:
        return


PLACEHOLDER_METADATA = MultimodalPlaceholderMetadata(
    placeholder_map={
        "image": "<|vision_start|><|image_pad|><|vision_end|>",
        "video": "<|vision_start|><|video_pad|><|vision_end|>",
    },
    placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
    placeholders_separator="",
    content_format=ContentFormat.STRING,
)


class Cosmos3EdgeInputProcessor(Qwen3VLInputProcessorBase):
    def __init__(self, *args, **kwargs):
        register_cosmos3_edge_processor()
        super().__init__(*args, **kwargs)

    def get_preferred_media_io_kwargs(self) -> Dict[str, Dict[str, Any]]:
        # Benchmark clips are already sampled. Do not resample them to the
        # shared ten-frame default.
        media_io_kwargs = super().get_preferred_media_io_kwargs()
        media_io_kwargs["video"].update({"num_frames": -1, "fps": -1})
        return media_io_kwargs


def _edge_positional_embedding_params(config) -> PositionalEmbeddingParams:
    """Build mRoPE params whose scaling matches ``rope_type``.

    ``type`` stays ``mrope`` for positions. ``RopeParams.from_config`` would
    otherwise read ``type`` and build a different table from the decoder.
    """
    rope_scaling = getattr(config, "rope_scaling", None)
    if rope_scaling is None:
        raise ValueError("Cosmos3-Edge text config is missing rope_scaling.")
    if "type" in rope_scaling:
        pos_type = rope_scaling["type"]
    elif "rope_type" in rope_scaling:
        pos_type = rope_scaling["rope_type"]
    else:
        raise ValueError("rope_scaling must have type or rope_type field")
    rope = RopeParams.from_config(config)
    rope.scale_type = RotaryScalingType.from_string(rope_scaling.get("rope_type", "default"))
    if "factor" in rope_scaling:
        rope.scale = rope_scaling["factor"]
    return PositionalEmbeddingParams(
        type=PositionEmbeddingType.from_string(pos_type),
        rope=rope,
        mrope_section=rope_scaling.get("mrope_section", None),
        mrope_interleaved=rope_scaling.get("mrope_interleaved", False),
    )


@support_multimodal_disaggregated
@register_vision_encoder(Cosmos3EdgeVisionEncoder, vlm_base_model=Cosmos3EdgeSiglip)
@register_auto_model("Cosmos3EdgeForConditionalGeneration")
@register_input_processor(
    Cosmos3EdgeInputProcessor,
    model_type="cosmos3_edge",
    placeholder_metadata=PLACEHOLDER_METADATA,
)
class Cosmos3EdgeModel(Qwen3VLModelBase):
    supports_encoder_cache = True

    def __init__(self, model_config: ModelConfig[PretrainedConfig], *args, **kwargs):
        omni_config = model_config.pretrained_config
        if omni_config is None or not getattr(omni_config, "_name_or_path", None):
            raise ValueError(
                "Cosmos3EdgeModel requires pretrained_config._name_or_path to locate "
                "transformer/ and vision_encoder/."
            )
        self._checkpoint_root, self.llm_path, self._vision_encoder_path = _get_cosmos3_model_paths(
            omni_config
        )
        kwargs["vision_encoder_class"] = Cosmos3EdgeVisionEncoder
        super().__init__(model_config, *args, **kwargs)

    def init_mrope_embedding(self, model_config: ModelConfig[PretrainedConfig]):
        config = model_config.pretrained_config.text_config
        pos_embd_params = _edge_positional_embedding_params(config)
        head_dim = getattr(config, "head_dim", None)
        if not isinstance(head_dim, int):
            head_dim = config.hidden_size // config.num_attention_heads
        self.rotary_emb = MRotaryEmbedding(
            pos_embd_params.rope,
            head_dim=head_dim,
            is_neox=pos_embd_params.is_neox,
            mrope_section=pos_embd_params.mrope_section,
            mrope_interleaved=pos_embd_params.mrope_interleaved,
        ).to("cuda")

    @property
    def multimodal_data_device_paths(self) -> List[str]:
        # ``image_grid_thw`` and ``video_grid_thw`` stay on CPU. The vision
        # forward only needs them as host shapes.
        return [
            "image.pixel_values",
            "video.pixel_values_videos",
            "multimodal_embedding",
            "mrope_config.mrope_position_ids",
            "mrope_config.mrope_position_deltas",
        ]

    @property
    def llm_checkpoint_dir(self) -> str:
        return self.llm_path

    def load_weights(self, weights: Dict[str, torch.Tensor], weight_mapper: BaseWeightMapper):
        if self.mm_encoder is not None:
            vision_weights_file = os.path.join(self._vision_encoder_path, "model.safetensors")
            if not os.path.isfile(vision_weights_file):
                raise FileNotFoundError(
                    f"Cosmos3-Edge vision encoder weights not found at {vision_weights_file}."
                )
            weights.update(safetensors.torch.load_file(vision_weights_file))
        if not isinstance(weight_mapper, Cosmos3EdgeHfWeightMapper):
            weight_mapper = Cosmos3EdgeHfWeightMapper()
        weights = weight_mapper.preprocess_weights(weights)
        if self.mm_encoder is not None:
            self.mm_encoder.load_weights(weights)
        text_weights = {
            key: value
            for key, value in weights.items()
            if not key.startswith("model.visual.") and not key.startswith("model.projector.")
        }
        weight_mapper.init_model_and_config(self.llm, self.llm.model_config)
        self.llm.load_weights(text_weights, weight_mapper)
