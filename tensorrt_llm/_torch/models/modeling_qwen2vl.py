# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
import math
import re
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import (AutoProcessor, AutoTokenizer, PretrainedConfig,
                          PreTrainedModel)
from transformers.modeling_outputs import BaseModelOutputWithPooling
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionPatchEmbed, Qwen2_5_VisionTransformerPretrainedModel)
from transformers.models.qwen2_vl.modeling_qwen2_vl import \
    Qwen2VisionTransformerPretrainedModel

from tensorrt_llm._torch.attention_backend.interface import \
    PredefinedAttentionMask
from tensorrt_llm._torch.models.checkpoints.base_weight_mapper import \
    BaseWeightMapper
from tensorrt_llm._torch.models.checkpoints.hf.qwen2vl_weight_mapper import \
    Qwen2VLHfWeightMapper
from tensorrt_llm._torch.models.modeling_multimodal_utils import _is_mm_disagg
from tensorrt_llm._torch.modules.attention import Attention
from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.inputs.multimodal import (DisaggPrefillMultimodalInputs,
                                            MultimodalParams)

from ..._utils import async_tensor_h2d, nvtx_range, prefer_pinned
from ...inputs import (BaseMultimodalDummyInputsBuilder,
                       BaseMultimodalInputProcessor, ContentFormat,
                       ExtraProcessedInputs, MultimodalPlaceholderMetadata,
                       MultimodalPlaceholderPlacement, TextPrompt,
                       register_input_processor,
                       support_multimodal_disaggregated)
from ...logger import logger
from ...sampling_params import SamplingParams
from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import PositionalEmbeddingParams, RopeParams
from ..attention_backend.utils import get_attention_backend
from ..flashinfer_utils import IS_FLASHINFER_AVAILABLE

# Guarded module-level import: `flashinfer_apply_rope_with_cos_sin_cache_inplace`
# is only exported from `custom_ops` when FlashInfer is installed (see
# `_torch/custom_ops/__init__.py`). Unconditional import would break loading
# this module in FlashInfer-less environments; importing inside the guard mirrors
# the pattern used in `custom_ops` itself.
if IS_FLASHINFER_AVAILABLE:
    from ..custom_ops import flashinfer_apply_rope_with_cos_sin_cache_inplace

# Vision RoPE on Qwen2-VL/2.5-VL/3-VL uses head_dim 72/80 which doesn't satisfy
# FlashInfer's "head_size % 64 == 0" precondition. Prefer flash_attn's fused
# Triton rotary kernel when available; SBSA CI images may not package flash_attn.
try:
    from flash_attn.ops.triton.rotary import \
        apply_rotary as _flash_attn_apply_rotary  # type: ignore
except ImportError:
    _flash_attn_apply_rotary = None

from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from ..modules.gated_mlp import GatedMLP
from ..modules.rotary_embedding import MRotaryEmbedding, RotaryEmbedding
from .modeling_auto import AutoModelForCausalLM
from .modeling_multimodal_encoder import (_ENCODER_FALLBACK_MAX_NUM_REQUESTS,
                                          MultimodalEncoderMixin)
from .modeling_multimodal_mixin import MultimodalModelMixin
from .modeling_multimodal_utils import (
    _install_processor_output_validation_filter, find_input_mm_embeds,
    fuse_input_embeds, get_attached_multimodal_embeddings,
    get_multimodal_embeddings)
from .modeling_utils import (ModelConfig, QuantConfig, _load_weights_impl,
                             filter_weights, register_auto_model,
                             register_vision_encoder)

PAD_INDEX = -100  # NOTE: refer to https://github.com/huggingface/transformers/blob/main/src/transformers/models/qwen2_5_vl/modular_qwen2_5_vl.py#L269


def _prepare_qwen_vl_vision_attn_metadata(
        seq_lens: List[int],
        attn_metadata: AttentionMetadata) -> AttentionMetadata:
    seq_lens = [int(seq_len) for seq_len in seq_lens]
    if not seq_lens:
        raise ValueError(
            "Qwen VL vision attention requires at least one segment")

    min_seq_len = min(seq_lens)
    if min_seq_len < 0:
        raise ValueError(
            f"Qwen VL vision attention segment length must be nonnegative, got {min_seq_len}"
        )

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
    attn_metadata.seq_lens = seq_lens_torch
    cu_seqlens = cu_seqlens.to(device=attn_metadata.seq_lens_cuda.device,
                               non_blocking=True)
    attn_metadata.cu_q_seqlens = cu_seqlens
    attn_metadata.cu_kv_seqlens = cu_seqlens
    attn_metadata.max_seq_len = max(seq_lens)
    # The vision tower runs no-cache, context-only attention and supplies its
    # own `cu_seqlens` above, so the heavy KV-oriented `prepare()` (kv_lens /
    # prompt_lens / host_request_types setup) is unnecessary host work.
    # `prepare_encoder_only` runs the lean path on backends that have one
    # (TRTLLM) and falls back to the full `prepare()` elsewhere.
    attn_metadata.prepare_encoder_only()
    return attn_metadata


def _prepare_qwen_vl_mrope_config(
    *,
    multimodal_params: List[MultimodalParams],
    num_generation_requests: int,
    position_ids: Optional[torch.Tensor],
    rotary_emb: MRotaryEmbedding,
    mrope_position_deltas_cache: torch.Tensor,
    mrope_rotary_cos_sin_workspace: torch.Tensor,
    mrope_delta_write_seq_slots: Optional[torch.Tensor] = None,
    mrope_delta_read_seq_slots: Optional[torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    mrope_config: Dict[str, torch.Tensor] = {}

    if mrope_delta_write_seq_slots is not None:
        seq_slots = mrope_delta_write_seq_slots
        num_seq_slots = seq_slots.numel()
        delta_tensors = []
        for multimodal_param in multimodal_params:
            request_mrope_config = multimodal_param.multimodal_data.get(
                'mrope_config', {})
            mrope_position_delta = request_mrope_config.get(
                'mrope_position_deltas')
            if mrope_position_delta is None:
                continue
            delta_tensors.append(mrope_position_delta.reshape(1))
            if len(delta_tensors) == num_seq_slots:
                break
        if len(delta_tensors) != num_seq_slots:
            raise RuntimeError(
                "Missing MRoPE position deltas for seq-slot cache update")
        deltas = torch.cat(delta_tensors, dim=0)
        mrope_position_deltas_cache.index_copy_(0, seq_slots, deltas)

    if position_ids is not None \
            and position_ids.shape[-1] > num_generation_requests:
        cos, sin = rotary_emb.get_cos_sin(position_ids)
        num_packed_values = cos.numel() * 2
        packed = mrope_rotary_cos_sin_workspace[:, :num_packed_values]
        packed_view = packed.view(*cos.shape, 2)
        torch.stack((cos, sin), dim=-1, out=packed_view)
        mrope_config['mrope_rotary_cos_sin'] = packed

    if mrope_delta_read_seq_slots is not None:
        mrope_config[
            'mrope_position_deltas'] = mrope_position_deltas_cache.index_select(
                0, mrope_delta_read_seq_slots).unsqueeze(1)

    return mrope_config


# A token budget larger than any real ``encoder_max_num_tokens`` so that
# ``get_size_for_max_tokens`` falls through to its ``max_pixels`` cap, yielding
# the largest single-image size (used to report the per-item token demand).
_MAX_PIXELS_TOKEN_PROBE = 1 << 31


class Qwen2VLInputProcessorBase(BaseMultimodalInputProcessor,
                                BaseMultimodalDummyInputsBuilder):

    def __init__(self,
                 model_path: str,
                 config: PretrainedConfig,
                 tokenizer: AutoTokenizer,
                 trust_remote_code: bool = True,
                 **kwargs):
        _install_processor_output_validation_filter()
        super().__init__(model_path=model_path,
                         config=config,
                         tokenizer=tokenizer,
                         trust_remote_code=trust_remote_code,
                         **kwargs)
        self._dtype = self._config.torch_dtype
        self._tokenizer = tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(
            model_path)
        self._model_path = model_path
        self._processor = AutoProcessor.from_pretrained(
            model_path,
            use_fast=self.use_fast,
            trust_remote_code=trust_remote_code)

        # temporal patch size for video frames
        self.temporal_patch_size = getattr(self.config.vision_config,
                                           'temporal_patch_size', 1)

    def get_vocab_size(self) -> int:
        """Return the vocab size of the model."""
        return self.config.text_config.vocab_size

    @property
    def config(self) -> PretrainedConfig:
        return self._config

    @property
    def tokenizer(self) -> AutoTokenizer:
        return self._tokenizer

    @property
    def model_path(self) -> str:
        return self._model_path

    @property
    def processor(self) -> AutoProcessor:
        return self._processor

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    # ------------------------------------------------------------------
    # Deterministic dummy-input sizing for multimodal profiling.
    #
    # `_num_vision_tokens` / `get_size_for_max_tokens` are the encoder-
    # side counterpart to the LLM's `max_num_tokens`: they report (and
    # invert) the exact number of attention tokens the vision encoder will
    # process for a given input size. The unit is **pre-merger patches** so
    # that values are directly comparable with ``encoder_max_num_tokens``
    # and ``AttentionMetadata.max_num_tokens``. Callers working in
    # LLM-visible (post-merger) units multiply/divide by
    # ``spatial_merge_unit`` at the boundary.
    # ------------------------------------------------------------------
    @property
    def spatial_merge_unit(self) -> int:
        """Encoder→LLM token ratio. Qwen2/2.5-VL applies a ``merge_size`` × ``merge_size`` spatial merger."""
        merge_size = self.config.vision_config.spatial_merge_size
        return merge_size * merge_size

    def _vision_pixel_bounds(self) -> Tuple[int, int]:
        """``(min_pixels, max_pixels)`` the HF image processor clamps to in
        ``smart_resize``. Read from the live processor's ``size`` config
        (transformers 5.x ``SizeDict`` with ``shortest_edge`` / ``longest_edge``)
        so per-model overrides are honored; falls back to the HF ``smart_resize``
        defaults when unavailable.
        """
        processor = getattr(self, "_processor", None)
        image_processor = getattr(processor, "image_processor", None)
        size = getattr(image_processor, "size", None)
        if size is not None:
            try:
                min_pixels = size["shortest_edge"]
                max_pixels = size["longest_edge"]
                if min_pixels and max_pixels:
                    return int(min_pixels), int(max_pixels)
            except (KeyError, TypeError):
                pass
        # HF `smart_resize` defaults.
        return 3136, 1003520

    def _num_vision_tokens(
        self,
        *,
        width: int,
        height: int,
        num_frames: int = 1,
    ) -> int:
        """Return the encoder attention tokens (pre-merger) that result from
        an image/video of the given pixel dimensions.

        Defers to HF Qwen-VL's ``smart_resize`` (the exact resize the processor
        applies) so the predicted token count matches what the processor
        actually produces, **including** the ``[min_pixels, max_pixels]``
        clamping that this method's previous inline rounding ignored -- which
        matters because ``get_num_tokens_per_image`` / ``..._video`` use this to
        size real requests, not just the profiling dummy.
        """
        grid_t, grid_h, grid_w = self._grid_thw_for_size(width=width,
                                                         height=height,
                                                         num_frames=num_frames)
        return grid_t * grid_h * grid_w

    def get_num_tokens_per_image(self, *, image, **kwargs) -> int:
        """Prompt-side image token count: the encoder tokens for the image size
        divided by ``spatial_merge_unit`` (the post-merger placeholder count)."""
        if isinstance(image, torch.Tensor):
            image_h, image_w = int(image.shape[-2]), int(image.shape[-1])
        else:
            image_h, image_w = image.height, image.width
        encoder_tokens = self._num_vision_tokens(width=image_w,
                                                 height=image_h,
                                                 num_frames=1)
        return encoder_tokens // self.spatial_merge_unit

    def get_num_tokens_per_video(self, *, video, **kwargs) -> int:
        """Prompt-side video token count: the encoder tokens for the frame stack
        divided by ``spatial_merge_unit``."""
        num_frames = len(video)
        first_frame = video[0]
        if isinstance(first_frame, torch.Tensor):
            frame_h = int(first_frame.shape[-2])
            frame_w = int(first_frame.shape[-1])
        else:
            frame_h, frame_w = first_frame.height, first_frame.width
        encoder_tokens = self._num_vision_tokens(width=frame_w,
                                                 height=frame_h,
                                                 num_frames=num_frames)
        return encoder_tokens // self.spatial_merge_unit

    def _grid_thw_for_size(
        self,
        *,
        width: int,
        height: int,
        num_frames: int = 1,
    ) -> Tuple[int, int, int]:
        """``(grid_t, grid_h, grid_w)`` patch-grid dimensions the processor would
        produce for an image/video of the given pixel size, after HF
        ``smart_resize`` (so the ``[min_pixels, max_pixels]`` clamp is honored).
        Shared by :meth:`_num_vision_tokens` (product) and
        :meth:`get_dummy_mm_data_for_size` (tensor shapes)."""
        cfg = self.config.vision_config
        patch_size = cfg.patch_size
        merge_size = cfg.spatial_merge_size
        temporal_patch_size = getattr(cfg, "temporal_patch_size", 1)
        factor = patch_size * merge_size

        min_pixels, max_pixels = self._vision_pixel_bounds()
        resized_h, resized_w = smart_resize(height=height,
                                            width=width,
                                            factor=factor,
                                            min_pixels=min_pixels,
                                            max_pixels=max_pixels)
        grid_h = resized_h // patch_size
        grid_w = resized_w // patch_size

        padded_frames = ((num_frames + temporal_patch_size - 1) //
                         temporal_patch_size) * temporal_patch_size
        grid_t = max(padded_frames // temporal_patch_size, 1)

        return grid_t, grid_h, grid_w

    def get_size_for_max_tokens(
        self,
        *,
        max_tokens: int,
    ) -> Dict[str, int]:
        """Invert ``_num_vision_tokens``: pick the ``(width, height)`` whose
        attention-token count is the largest value ≤ ``max_tokens`` while
        keeping the aspect ratio bounded.

        ``max_tokens`` is in the same unit as ``_num_vision_tokens`` /
        ``encoder_max_num_tokens`` (pre-merger).

        Returns a single-image geometry (``num_frames=1``). This is a valid
        worst case for the *whole* vision encoder regardless of the runtime
        modality: the ViT cost is a function of the total pre-merger patch
        count (the token unit), and ``_num_vision_tokens`` already folds video
        frames into that same count — so an image saturating ``max_tokens``
        hits the same attention workspace as any video with the same token
        count. Non-visual modalities (e.g. audio) live on a different input
        processor and size their own dummy.
        """
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")

        def closest_factor_pair(n: int) -> Tuple[int, int]:
            """Closest ``h*w=n`` to square; keeps dummy aspect ratio near 1:1."""
            for d in range(math.isqrt(n), 0, -1):
                if n % d == 0:
                    return d, n // d
            return 1, n

        cfg = self.config.vision_config
        patch_size = cfg.patch_size
        merge_size = cfg.spatial_merge_size
        unit = patch_size * merge_size

        # Pre-merger tokens factor into (grid_h * grid_w) and each grid
        # dimension is ``merge_size`` × the post-merger factor. Searching in
        # post-merger units bounds the inner loop and lets us reuse the
        # familiar near-square factor pair for aspect ratio bounds.
        post_merger_budget = max(max_tokens // (merge_size * merge_size), 1)
        # A single image can't exceed the processor's ``max_pixels`` -- past
        # that, ``smart_resize`` (in ``_num_vision_tokens``) clamps the image
        # back down, so an uncapped budget would produce a size whose real
        # token count falls short of ``max_tokens``. Cap so the chosen size
        # round-trips exactly (size the worst case by ``max_pixels``).
        _, max_pixels = self._vision_pixel_bounds()
        max_post_merger = max(max_pixels // (unit * unit), 1)
        post_merger_budget = min(post_merger_budget, max_post_merger)
        h_factor, w_factor = closest_factor_pair(post_merger_budget)
        for seq_len in range(post_merger_budget, 0, -1):
            h_f, w_f = closest_factor_pair(seq_len)
            if w_f / max(h_f, 1) <= 200:
                h_factor, w_factor = h_f, w_f
                break

        return {
            "width": unit * w_factor,
            "height": unit * h_factor,
            "num_frames": 1,
        }

    def get_mm_max_tokens_per_item(self) -> Dict[str, int]:
        """Qwen-VL runs image and video through one shared ViT, so the image
        worst case already covers the vision encoder — only ``"image"`` is
        declared. The value is the largest single image's encoder tokens (the
        ``max_pixels``-capped size), used to weight the shared-budget split."""
        size = self.get_size_for_max_tokens(max_tokens=_MAX_PIXELS_TOKEN_PROBE)
        return {
            "image":
            self._num_vision_tokens(width=size["width"], height=size["height"])
        }

    def get_dummy_mm_data_for_tokens(
        self,
        *,
        max_tokens_per_modality: Dict[str, int],
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Any]:
        """Vision implementation of the modality-agnostic profiler entry: for the
        ``"image"`` budget, pick the worst-case image size, fill it with identical
        copies, and materialize the encoder tensors.

        ``num_images`` is computed from the *realized* token count of the chosen
        size (which the ``max_pixels`` cap may make smaller than the budget), so
        the batch saturates the budget rather than assuming one image fills it.
        """
        max_tokens = max_tokens_per_modality.get("image")
        if not max_tokens:
            return {}
        size = self.get_size_for_max_tokens(max_tokens=max_tokens)
        tokens_per_image = max(
            1,
            self._num_vision_tokens(width=size["width"],
                                    height=size["height"],
                                    num_frames=size.get("num_frames", 1)))
        num_images = max(1, max_tokens // tokens_per_image)
        return self.get_dummy_mm_data_for_size(width=size["width"],
                                               height=size["height"],
                                               num_frames=size.get(
                                                   "num_frames", 1),
                                               num_images=num_images,
                                               dtype=dtype)

    def get_dummy_mm_data_for_size(
        self,
        *,
        width: int,
        height: int,
        num_frames: int = 1,
        num_images: int = 1,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Any]:
        """Build the *processed* multimodal tensors for ``num_images`` identical
        ``(width, height)`` images directly, skipping PIL image creation and the
        HF processor (the encoder profiler's worst-case dummy batch).

        The vision encoder forward consumes only ``pixel_values`` (shape
        ``[num_patches, in_dim]``) and ``image_grid_thw`` (shape
        ``[num_images, 3]``); the pixel *content* is irrelevant for memory
        profiling, so zero tensors of the exact shape the processor would emit
        suffice. Returns the ``multimodal_data`` dict consumed by
        :meth:`Qwen2VisionModelBase._parse_and_batch_multimodal_data`.
        """
        cfg = self.config.vision_config
        grid_t, grid_h, grid_w = self._grid_thw_for_size(width=width,
                                                         height=height,
                                                         num_frames=num_frames)
        num_images = max(num_images, 1)
        patches_per_image = grid_t * grid_h * grid_w
        in_channels = getattr(cfg, "in_channels", None) or getattr(
            cfg, "in_chans", 3)
        temporal_patch_size = getattr(cfg, "temporal_patch_size", 1)
        in_dim = in_channels * temporal_patch_size * cfg.patch_size * cfg.patch_size

        pixel_values = torch.zeros(
            (num_images * patches_per_image, in_dim),
            dtype=dtype or self.dtype,
        )
        image_grid_thw = torch.tensor([[grid_t, grid_h, grid_w]] * num_images,
                                      dtype=torch.long)
        return {
            "image": {
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
            }
        }

    @classmethod
    def _build_temporal_block(
        cls,
        config: PretrainedConfig,
        llm_grid_t: int,
        llm_grid_h: int,
        llm_grid_w: int,
        second_per_grid_t: float,
    ) -> np.ndarray:
        """Per-grid (3, llm_grid_t * llm_grid_h * llm_grid_w) position block.

        Qwen2VL / Qwen2_5_VL: when ``vision_config.tokens_per_second`` is set,
        scale the temporal axis by ``second_per_grid_t * tokens_per_second``
        (Qwen2_5 style); otherwise fall back to a plain ``np.indices`` grid.

        Subclasses with a different temporal scheme (e.g. Qwen3-VL uses
        separate timestamp tokens) override this hook.
        """
        tokens_per_second = getattr(config.vision_config, 'tokens_per_second',
                                    None)
        if tokens_per_second is not None:
            t_index = (np.arange(llm_grid_t).reshape(-1, 1) *
                       (second_per_grid_t * tokens_per_second))
            t_index = np.broadcast_to(
                t_index.astype(np.int64),
                (llm_grid_t, llm_grid_h * llm_grid_w),
            ).reshape(-1)
            h_idx = np.broadcast_to(
                np.arange(llm_grid_h).reshape(1, -1, 1),
                (llm_grid_t, llm_grid_h, llm_grid_w),
            ).reshape(-1)
            w_idx = np.broadcast_to(
                np.arange(llm_grid_w).reshape(1, 1, -1),
                (llm_grid_t, llm_grid_h, llm_grid_w),
            ).reshape(-1)
            return np.stack([t_index, h_idx, w_idx])
        return np.indices((llm_grid_t, llm_grid_h, llm_grid_w)).reshape(3, -1)

    @classmethod
    def get_rope_index(
        cls,
        config: PretrainedConfig,
        input_ids: Optional[torch.IntTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Shared implementation for the Qwen2-VL family; the per-grid temporal
        block is built by ``cls._build_temporal_block``, which subclasses
        override when their temporal scheme differs (e.g. Qwen3-VL uses
        per-frame timestamp tokens instead of ``tokens_per_second``-scaled
        positions).

        Args:
            config: The HF's PretrainedConfig model configuration
            input_ids: Indices of input sequence tokens in the vocabulary
            image_grid_thw: The temporal, height and width of feature shape of each image in LLM
            video_grid_thw: The temporal, height and width of feature shape of each video in LLM
            attention_mask: Mask to avoid performing attention on padding token indices
            second_per_grid_ts: The time interval (in seconds) for each grid along the temporal dimension

        Returns:
            position_ids: A tensor of shape (3, batch_size, sequence_length)
            mrope_position_deltas: A tensor of shape (batch_size)
        """
        spatial_merge_size = config.vision_config.spatial_merge_size
        image_token_id = config.image_token_id
        video_token_id = config.video_token_id
        vision_start_token_id = config.vision_start_token_id

        # Handle case with no vision inputs
        if image_grid_thw is None and video_grid_thw is None:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = position_ids.unsqueeze(0).expand(3, -1, -1).to(
                    input_ids.device)
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                    -1, keepdim=True)[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[
                    -1]
            else:
                position_ids = (torch.arange(input_ids.shape[1],
                                             device=input_ids.device).view(
                                                 1, 1, -1).expand(
                                                     3, input_ids.shape[0], -1))
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )
            return position_ids, mrope_position_deltas

        # numpy-based vectorized impl: ~2-3× faster than the torch loop
        # version on CPU since per-image/video work is small tensor
        # allocations dominated by Python+dispatch overhead. Output tensors
        # are placed back on `input_ids.device`.
        input_device = input_ids.device
        input_dtype = input_ids.dtype
        ids_np = input_ids.detach().cpu().numpy()
        attn_np = (np.ones_like(ids_np) if attention_mask is None else
                   attention_mask.detach().cpu().numpy())
        image_grid_np = (image_grid_thw.detach().cpu().numpy()
                         if image_grid_thw is not None else None)
        video_grid_np = (video_grid_thw.detach().cpu().numpy()
                         if video_grid_thw is not None else None)
        if second_per_grid_ts is not None and isinstance(
                second_per_grid_ts, torch.Tensor):
            second_per_grid_ts_np = second_per_grid_ts.detach().cpu().numpy()
        else:
            second_per_grid_ts_np = second_per_grid_ts
        B, S = ids_np.shape
        position_ids_np = np.ones((3, B, S), dtype=ids_np.dtype)
        deltas = []
        image_index = 0
        video_index = 0
        for i in range(B):
            mask_i = attn_np[i] == 1
            seq = ids_np[i][mask_i]
            vision_start_indices = np.flatnonzero(seq == vision_start_token_id)
            vision_tokens = (seq[vision_start_indices +
                                 1] if vision_start_indices.size else seq[:0])
            image_nums = int((vision_tokens == image_token_id).sum())
            video_nums = int((vision_tokens == video_token_id).sum())
            seq_list = seq.tolist()
            has_image = image_token_id in seq_list
            has_video = video_token_id in seq_list
            llm_pos_ids_list = []
            st = 0
            remain_images, remain_videos = image_nums, video_nums
            for _ in range(image_nums + video_nums):
                if has_image and remain_images > 0:
                    ed_image = seq_list.index(image_token_id, st)
                else:
                    ed_image = len(seq_list) + 1
                if has_video and remain_videos > 0:
                    ed_video = seq_list.index(video_token_id, st)
                else:
                    ed_video = len(seq_list) + 1
                if ed_image < ed_video:
                    t = int(image_grid_np[image_index][0])
                    h = int(image_grid_np[image_index][1])
                    w = int(image_grid_np[image_index][2])
                    second_per_grid_t = 0
                    image_index += 1
                    remain_images -= 1
                    ed = ed_image
                else:
                    t = int(video_grid_np[video_index][0])
                    h = int(video_grid_np[video_index][1])
                    w = int(video_grid_np[video_index][2])
                    if second_per_grid_ts_np is not None:
                        second_per_grid_t = float(
                            second_per_grid_ts_np[video_index])
                    else:
                        second_per_grid_t = 1.0
                    video_index += 1
                    remain_videos -= 1
                    ed = ed_video
                llm_grid_t = t
                llm_grid_h = h // spatial_merge_size
                llm_grid_w = w // spatial_merge_size
                text_len = ed - st
                st_idx = llm_pos_ids_list[-1].max(
                ) + 1 if llm_pos_ids_list else 0
                if text_len > 0:
                    llm_pos_ids_list.append(
                        np.broadcast_to(np.arange(text_len), (3, text_len)) +
                        st_idx)
                block = cls._build_temporal_block(config, llm_grid_t,
                                                  llm_grid_h, llm_grid_w,
                                                  second_per_grid_t)
                llm_pos_ids_list.append(block + text_len + st_idx)
                st = ed + llm_grid_t * llm_grid_h * llm_grid_w
            if st < len(seq_list):
                st_idx = llm_pos_ids_list[-1].max(
                ) + 1 if llm_pos_ids_list else 0
                text_len = len(seq_list) - st
                llm_pos_ids_list.append(
                    np.broadcast_to(np.arange(text_len), (3, text_len)) +
                    st_idx)
            llm_positions = np.concatenate(llm_pos_ids_list,
                                           axis=1).reshape(3, -1)
            position_ids_np[:, i, mask_i] = llm_positions
            deltas.append(
                int(llm_positions.max()) + 1 - int(ids_np[i].shape[0]))
        position_ids = torch.from_numpy(position_ids_np).to(device=input_device,
                                                            dtype=input_dtype)
        mrope_position_deltas = torch.tensor(deltas,
                                             device=input_device).unsqueeze(1)
        return position_ids, mrope_position_deltas

    def _preprocess(self, text: Dict[str, any], mm_data: Dict[str, any],
                    mm_processor_kwargs: Dict[str, Any]):
        images = mm_data.get("image")
        video_datas = mm_data.get("video")
        if video_datas is not None:
            videos = [video_data.frames for video_data in video_datas]
        else:
            videos = None
        do_rescale = True
        if images and isinstance(images[0], torch.Tensor):
            do_rescale = False
        if videos and isinstance(videos[0][0], torch.Tensor):
            do_rescale = False
        # transformers 5.x's ``ProcessorMixin._merge_kwargs`` strictly
        # validates per-modality kwargs against the processor's TypedDict.
        # Processor *output* keys (``video_grid_thw``, ``pixel_values``, ...)
        # round-trip into the validator via tokenizer ``init_kwargs`` /
        # ``model_input_names`` and would trip ``TypeError:
        # merged_typed_dict.__init__() got an unexpected keyword argument
        # 'video_grid_thw'``. ``_install_processor_output_validation_filter``
        # (called from ``__init__``) installs a process-wide filter that drops
        # those keys before the validator sees them.
        return self.processor(text=[text],
                              images=images,
                              videos=videos,
                              padding=True,
                              do_rescale=do_rescale,
                              return_tensors='pt',
                              **mm_processor_kwargs)

    def _postprocess(self, input_ids: torch.IntTensor) -> torch.IntTensor:
        # Keep image / vision / video placeholders in-vocab; the model engine
        # locates mm positions via ``torch.isin(input_ids, mm_token_ids)`` and
        # the previous ``vocab_size + 1`` OOV remap is no longer needed.
        return input_ids

    def get_mm_token_ids(self) -> Optional[torch.Tensor]:
        """Surface the in-vocab image / vision / video placeholder IDs so that
        ``maybe_compute_mm_embed_cumsum`` builds ``embed_mask_cumsum`` via
        ``torch.isin(input_ids, mm_token_ids)`` instead of the OOV
        ``>= vocab_size`` fallback (which would miss all positions now that the
        ``_postprocess`` OOV remap is gone).
        """
        ids = [
            tid for tid in (
                getattr(self.config, 'image_token_id', None),
                getattr(self.config, 'vision_token_id', None),
                getattr(self.config, 'video_token_id', None),
            ) if tid is not None
        ]
        return torch.tensor(ids, dtype=torch.int32) if ids else None

    def get_mrope_config(
            self,
            input_ids: torch.IntTensor,
            image_grid_thw: torch.LongTensor,
            video_grid_thw: torch.LongTensor,
            attention_mask: torch.Tensor,
            second_per_grid_ts: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # Dispatch via ``type(self)`` so subclasses (e.g. Qwen3-VL) get their
        # overridden ``get_rope_index`` instead of the base implementation.
        mrope_position_ids, mrope_position_deltas = type(self).get_rope_index(
            self.config, input_ids, image_grid_thw, video_grid_thw,
            attention_mask, second_per_grid_ts)

        mrope_config = {}
        mrope_config['mrope_position_ids'] = mrope_position_ids.to(
            'cpu').clone()
        mrope_config['mrope_position_deltas'] = mrope_position_deltas.to(
            'cpu').to(torch.int32).clone()
        return mrope_config

    @staticmethod
    def _infer_image_grid_thw(num_tokens: int,
                              spatial_merge_size: int) -> List[int]:
        if num_tokens <= 0:
            raise ValueError(
                f"Image embedding must contain at least one token, got {num_tokens}"
            )
        llm_grid_h = int(num_tokens**0.5)
        while llm_grid_h > 1 and num_tokens % llm_grid_h != 0:
            llm_grid_h -= 1
        llm_grid_w = num_tokens // llm_grid_h
        return [
            1,
            llm_grid_h * spatial_merge_size,
            llm_grid_w * spatial_merge_size,
        ]

    def _attach_multimodal_embeddings_impl(
        self,
        inputs: TextPrompt,
        multimodal_embedding: Dict[str, List[torch.Tensor]],
        sampling_params: SamplingParams,
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        if not isinstance(multimodal_embedding, dict):
            raise ValueError("multimodal_embedding must be a dictionary")
        if set(multimodal_embedding) != {"image"}:
            raise ValueError(
                "Only image modality is supported for external multimodal embedding"
            )

        image_embeddings = multimodal_embedding["image"]
        if isinstance(image_embeddings, torch.Tensor):
            image_embeddings = [image_embeddings]
        if not image_embeddings:
            raise ValueError("At least one image embedding is required")
        for index, image_embedding in enumerate(image_embeddings):
            if image_embedding.dim() != 2:
                raise ValueError(
                    f"Image embedding {index} must be rank 2, got shape {tuple(image_embedding.shape)}"
                )

        build_disagg_prefill_multimodal_inputs = getattr(
            self, "build_disagg_prefill_multimodal_inputs", None)
        if not callable(build_disagg_prefill_multimodal_inputs):
            raise NotImplementedError(
                f"{type(self).__name__} does not support external multimodal embeddings"
            )

        mm_handles = [{
            "tensor_size": tuple(image_embedding.shape)
        } for image_embedding in image_embeddings]
        prompt_token_ids = build_disagg_prefill_multimodal_inputs(
            inputs, mm_handles).prompt_token_ids

        # ``build_disagg_prefill_multimodal_inputs`` already emits the in-vocab
        # ``image_token_id`` at mm positions (no legacy OOV remap), so
        # ``mrope_input_ids`` is fed to ``get_mrope_config`` as-is.
        mrope_input_ids = torch.tensor(prompt_token_ids,
                                       dtype=torch.long).unsqueeze(0)
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_grid_thw = torch.tensor(
            [
                self._infer_image_grid_thw(image_embedding.shape[0],
                                           spatial_merge_size)
                for image_embedding in image_embeddings
            ],
            dtype=torch.long,
        )
        attention_mask = torch.ones_like(mrope_input_ids)
        mrope_config = self.get_mrope_config(mrope_input_ids, image_grid_thw,
                                             None, attention_mask, None)

        multimodal_data = {
            "multimodal_embedding": image_embeddings,
            "mrope_config": mrope_config,
        }
        return prompt_token_ids, {"multimodal_data": multimodal_data}

    @nvtx_range("Qwen2VLInputProcessorBase forward()")
    @torch.inference_mode()
    def call_with_text_prompt(
        self,
        inputs: TextPrompt,
        sampling_params: SamplingParams,
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        text_prompt, mm_data, mm_processor_kwargs = inputs.get("prompt"), \
                        inputs.get("multi_modal_data", {}), inputs.get("mm_processor_kwargs", {})

        # Text-only fast path: skip the multi-modal HF processor (tokenizer
        # output matches it bit-exactly when `images` / `videos` are `None`)
        # while still populating mrope_config since the LM is M-RoPE.
        if not mm_data:
            input_ids = self.tokenizer(text_prompt,
                                       return_tensors="pt").input_ids
            attention_mask = torch.ones_like(input_ids)
            mrope_config = self.get_mrope_config(input_ids, None, None,
                                                 attention_mask, None)
            return input_ids[0].to(torch.int32).tolist(), {
                "multimodal_data": {
                    "mrope_config": mrope_config
                },
            }

        processed_inputs = self._preprocess(text_prompt, mm_data,
                                            mm_processor_kwargs)

        multimodal_data = {}
        pixel_values = processed_inputs.get('pixel_values', None)
        if pixel_values is not None:
            multimodal_data["image"] = {
                "pixel_values": pixel_values.to(self.dtype),
                "image_grid_thw": processed_inputs.get('image_grid_thw')
            }

        pixel_values_videos = processed_inputs.get('pixel_values_videos', None)
        if pixel_values_videos is not None:
            multimodal_data["video"] = {
                "pixel_values_videos": pixel_values_videos.to(self.dtype),
                "video_grid_thw": processed_inputs.get('video_grid_thw')
            }

        # NOTE: Even on the text-only prompts, we still need 'mrope_position_ids'.
        mrope_config = self.get_mrope_config(
            processed_inputs['input_ids'],
            processed_inputs.get('image_grid_thw', None),
            processed_inputs.get('video_grid_thw', None),
            processed_inputs.get('attention_mask', None),
            processed_inputs.get('second_per_grid_ts', None))
        multimodal_data["mrope_config"] = mrope_config

        fused_input_ids = processed_inputs['input_ids'][0]
        if mm_data:
            fused_input_ids = self._postprocess(fused_input_ids)

        return fused_input_ids.to(torch.int32).tolist(), {
            "multimodal_data": multimodal_data,
        }


class Qwen2VisionModelBase(nn.Module):

    def __init__(self, model_config: ModelConfig[PretrainedConfig],
                 model_class: Union[type[PreTrainedModel],
                                    type[torch.nn.Module]]):
        super().__init__()
        self.model_config = model_config
        self.model_dtype = self.model_config.pretrained_config.torch_dtype
        self.config = self.model_config.pretrained_config.vision_config
        self.config.num_attention_heads = self.config.num_heads

        # NOTE: Re-setting QuantConfig to exclude vision encoder weights from quantization load.
        self.model_config.quant_config = QuantConfig(
            kv_cache_quant_algo=self.model_config.quant_config.
            kv_cache_quant_algo)

        if model_class in [
                Qwen2VisionTransformerPretrainedModel,
                Qwen2_5_VisionTransformerPretrainedModel
        ]:
            # NOTE: For Qwen2VL, we use flash_attention_2 for attention implementation to avoid OOM issue.
            self.config._attn_implementation = 'flash_attention_2'
            self.visual = MultimodalModelMixin._cast_multimodal_encoder_dtype(
                model_class(model_config.pretrained_config.vision_config),
                self.model_dtype).eval()
        elif model_class == Qwen2_5_VisionModel:
            self.visual = MultimodalModelMixin._cast_multimodal_encoder_dtype(
                model_class(self.model_config), self.model_dtype)
        else:
            raise NotImplementedError(
                f"Model class {model_class} not implemented")

    def _split_fused_vision_qkv_tensor(
        self, tensor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split HF fused `attn.qkv` along output dim (dim 0 for Linear).

        Qwen2.5-VL vision is **MHA** (`num_key_value_heads == num_heads`): Q, K, and V
        each occupy `num_heads * head_dim` — three equal blocks.

        EXAONE-4.5 vision is **GQA**: Q uses `num_heads * head_dim`, K and V each use
        `num_key_value_heads * head_dim` (asymmetric split).
        """
        cfg = self.config
        num_heads = cfg.num_heads
        num_kv_heads = getattr(cfg, "num_key_value_heads", None)
        if num_kv_heads is None:
            num_kv_heads = num_heads
        head_dim, rem = divmod(cfg.hidden_size, num_heads)
        if rem != 0:
            raise ValueError(
                f"vision hidden_size {cfg.hidden_size} not divisible by "
                f"num_heads {num_heads}")
        q_dim = num_heads * head_dim
        kv_dim = num_kv_heads * head_dim
        # Fused Linear out_features = Q + K + V along dim 0 of the weight (or bias).
        fused_out_features = q_dim + 2 * kv_dim
        leading_dim = tensor.shape[0]
        if leading_dim == fused_out_features:
            # GQA (e.g. EXAONE-4.5 vision) or MHA with fused length matching config.
            return (tensor[:q_dim], tensor[q_dim:q_dim + kv_dim],
                    tensor[q_dim + kv_dim:])
        if num_kv_heads == num_heads and leading_dim % 3 == 0:
            # MHA (e.g. Qwen2.5-VL vision): three equal Q/K/V blocks; used if fused
            # leading dim is a triple split but does not match `fused_out_features`.
            dim_shape = leading_dim // 3
            return (tensor[:dim_shape], tensor[dim_shape:2 * dim_shape],
                    tensor[2 * dim_shape:])
        raise ValueError(
            f"Fused vision qkv leading dim is {leading_dim}, "
            f"want {fused_out_features} from config (q_dim={q_dim}, kv_dim={kv_dim}) "
            f"or for MHA a length divisible by 3; "
            f"num_heads={num_heads}, num_key_value_heads={num_kv_heads}, "
            f"head_dim={head_dim}.")

    def load_weights(self, weights: Dict):
        visual_weights = filter_weights("visual", weights)
        converted_weights = dict()
        if isinstance(self.visual, (Qwen2VisionTransformerPretrainedModel,
                                    Qwen2_5_VisionTransformerPretrainedModel)):
            self.visual.load_state_dict(visual_weights, strict=True)
            return

        qkv_pattern = re.compile(r'(.*?)attn\.qkv\.(.*)')
        for name in visual_weights:
            # Handle with weights and bias for vision transformer's qkv projection.
            match = qkv_pattern.match(name)
            if match:
                prefix, suffix = match.groups()
                q_name = f"{prefix}attn.q_proj.{suffix}"
                k_name = f"{prefix}attn.k_proj.{suffix}"
                v_name = f"{prefix}attn.v_proj.{suffix}"
                q_part, k_part, v_part = self._split_fused_vision_qkv_tensor(
                    visual_weights[name])
                converted_weights[q_name] = q_part
                converted_weights[k_name] = k_part
                converted_weights[v_name] = v_part
            else:
                converted_weights[name] = visual_weights[name]
        pattern_mapping = {
            r'(.*?)attn.proj.(.*)': r'\1attn.o_proj.\2',
            r'(.*?)mlp.fc1.(.*)': r'\1mlp.up_proj.\2',
            r'(.*?)mlp.fc2.(.*)': r'\1mlp.down_proj.\2',
        }
        _load_weights_impl(self.visual,
                           converted_weights,
                           params_map=pattern_mapping)

    def _parse_and_batch_multimodal_data(
        self, multimodal_params: List[MultimodalParams]
    ) -> Tuple[Dict[str, Any], Dict[str, List[Any]]]:

        pixel_values_list = []
        pixel_values_videos_list = []
        image_grid_thw_list = []
        video_grid_thw_list = []

        for multimodal_param in multimodal_params:
            # Process images if present
            if multimodal_param.multimodal_data.get("image") is not None:
                pixel_values_list.append(
                    multimodal_param.multimodal_data["image"]["pixel_values"])
                image_grid_thw_list.append(
                    multimodal_param.multimodal_data["image"]["image_grid_thw"])

            # Process videos if present
            if multimodal_param.multimodal_data.get("video") is not None:
                pixel_values_videos_list.append(
                    multimodal_param.multimodal_data["video"]
                    ["pixel_values_videos"])
                video_grid_thw_list.append(
                    multimodal_param.multimodal_data["video"]["video_grid_thw"])

        # Concatenate tensors
        mm_content_dict = {}
        if pixel_values_list:
            mm_content_dict["pixel_values"] = torch.cat(
                pixel_values_list,
                dim=0) if len(pixel_values_list) > 1 else pixel_values_list[0]
        if pixel_values_videos_list:
            mm_content_dict["pixel_values_videos"] = torch.cat(
                pixel_values_videos_list,
                dim=0) if len(pixel_values_videos_list
                              ) > 1 else pixel_values_videos_list[0]

        # Prepare extra data
        mm_extra_data = {}
        if image_grid_thw_list:
            mm_extra_data["image_grid_thw"] = torch.cat(
                image_grid_thw_list, dim=0) if len(
                    image_grid_thw_list) > 1 else image_grid_thw_list[0]
        if video_grid_thw_list:
            mm_extra_data["video_grid_thw"] = torch.cat(
                video_grid_thw_list, dim=0) if len(
                    video_grid_thw_list) > 1 else video_grid_thw_list[0]

        return mm_content_dict, mm_extra_data

    @nvtx_range("Qwen2VisionModelBase forward()")
    @torch.inference_mode()
    def forward(self, multimodal_params: List[MultimodalParams]):

        mm_content_data, mm_extra_data = self._parse_and_batch_multimodal_data(
            multimodal_params)
        pixel_values = mm_content_data.get("pixel_values", None)
        pixel_values_videos = mm_content_data.get("pixel_values_videos", None)

        image_grid_thw = mm_extra_data.get("image_grid_thw", None)
        video_grid_thw = mm_extra_data.get("video_grid_thw", None)

        embeds = []
        if pixel_values is not None:
            embed = self.visual(pixel_values, grid_thw=image_grid_thw)
            if isinstance(embed, BaseModelOutputWithPooling):
                embed = embed.pooler_output
            embeds.append(embed)

        if pixel_values_videos is not None:
            embed = self.visual(pixel_values_videos, grid_thw=video_grid_thw)
            if isinstance(embed, BaseModelOutputWithPooling):
                embed = embed.pooler_output
            embeds.append(embed)
        return embeds


class Qwen2_5_VLVisionAttention(Attention):

    def __init__(self,
                 model_config: ModelConfig[PretrainedConfig],
                 layer_idx: int,
                 reduce_output: bool = True) -> None:

        config = model_config.pretrained_config.vision_config
        # Composite VLM configs (transformers 5.x strict mode) keep
        # `max_position_embeddings` inside `text_config` rather than at
        # the top level; fall back to it when the parent doesn't expose it.
        max_position_embeddings = getattr(model_config.pretrained_config,
                                          "max_position_embeddings", None)
        if max_position_embeddings is None:
            text_config = getattr(model_config.pretrained_config, "text_config",
                                  None)
            max_position_embeddings = getattr(text_config,
                                              "max_position_embeddings", None)
        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_heads,
            num_key_value_heads=getattr(config, "num_key_value_heads", None)
            or config.num_heads,
            max_position_embeddings=max_position_embeddings,
            bias=True,
            pos_embd_params=None,
            rope_fusion=False,
            layer_idx=layer_idx,
            dtype=config.torch_dtype,
            config=model_config,
            reduce_output=reduce_output,
            # The vision encoder's head_dim is derived from its own
            # hidden_size; don't inherit head_dim mirrored from the text
            # sub-config onto the top-level pretrained_config.
            head_dim=config.hidden_size // config.num_heads,
        )
        # Vision attention runs from the outer VL wrapper (outside the
        # compiled LM region). When `set_torch_compiling(True)` is set
        # for the LM, `forward_impl` would otherwise dispatch vision
        # through `attn_custom_op_inplace`, which looks
        # `attention_metadata` up in the global `extra_attrs` --
        # that slot is populated by `model_engine.model_forward` with
        # the LM decoder's metadata, so vision FMHA receives the LM's
        # S/num_contexts with vision's head_dim and dispatch fails
        # (`FMHA kernels are not found ... D: <vision_head_dim>`).
        # Unregister so `forward_impl` falls back to the eager path
        # that uses vision's own `attn_metadata`.
        if self.register_to_config:
            model_config.extra_attrs.get("attn_layers",
                                         {}).pop(self.layer_idx_str, None)
            self.register_to_config = False

    def apply_rope(self,
                   q: torch.Tensor,
                   k: Optional[torch.Tensor],
                   v: Optional[torch.Tensor],
                   position_ids: Optional[torch.IntTensor] = None,
                   position_embeddings: Optional[Tuple[torch.Tensor,
                                                       torch.Tensor]] = None):
        seq_len, _ = q.size()
        cos, sin = position_embeddings

        # FlashInfer fused RoPE assumes head_size is a multiple of 64 (see
        # auto_deploy custom op rope docs / flashinfer tests). Qwen2.5-VL vision
        # uses head_dim=80 (e.g. 1280 hidden / 16 heads), so use PyTorch RoPE.
        if IS_FLASHINFER_AVAILABLE and self.head_dim % 64 == 0 and position_ids is not None:
            try:
                cos_sin_cache = torch.cat([cos, sin], dim=-1).contiguous()
                flashinfer_apply_rope_with_cos_sin_cache_inplace(
                    position_ids,
                    q,
                    k,
                    self.head_dim,
                    cos_sin_cache,
                    is_neox=True,
                )
                return q, k, v
            except RuntimeError as err:
                logger.warning(
                    "Qwen2.5-VL vision RoPE: FlashInfer failed (%s); "
                    "falling back to PyTorch RotaryEmbedding.apply_rotary_pos_emb.",
                    err,
                )

        # cos/sin are typically already in `q.dtype` upstream; `.to`
        # short-circuits when the dtype already matches, so this is a
        # safety net for callers / quantization paths that didn't pre-cast.
        cos = cos.to(dtype=q.dtype)
        sin = sin.to(dtype=q.dtype)
        q = q.view(seq_len, -1, self.head_dim)
        k = k.view(seq_len, -1, self.head_dim)
        v = v.view(seq_len, -1, self.head_dim)
        if _flash_attn_apply_rotary is None:
            q = RotaryEmbedding.apply_rotary_pos_emb(q.unsqueeze(0),
                                                     cos,
                                                     sin,
                                                     unsqueeze_dim=1).squeeze(0)
            k = RotaryEmbedding.apply_rotary_pos_emb(k.unsqueeze(0),
                                                     cos,
                                                     sin,
                                                     unsqueeze_dim=1).squeeze(0)
        else:
            # flash_attn Triton kernel: single launch per tensor. cos/sin
            # are expected as `[seqlen, head_dim/2]`. The PyTorch path
            # built `RotaryEmbedding.apply_rotary_pos_emb` with cos/sin
            # already in that layout (see `get_rotary_pos_emb_window_data`).
            # The kernel takes 4D `(batch, seq, nheads, headdim)`; add a
            # batch dim, run in-place, then drop it.
            q4 = q.unsqueeze(0)
            k4 = k.unsqueeze(0)
            _flash_attn_apply_rotary(q4,
                                     cos,
                                     sin,
                                     interleaved=False,
                                     inplace=True)
            _flash_attn_apply_rotary(k4,
                                     cos,
                                     sin,
                                     interleaved=False,
                                     inplace=True)
        q, k, v = q.reshape(seq_len, -1), k.reshape(seq_len,
                                                    -1), v.reshape(seq_len, -1)
        return q, k, v

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        position_ids: Optional[torch.IntTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        # NOTE: Qwen2.5-VL vision attention needs a custom forward: the generic
        # Attention path does not accept precomputed (cos, sin) position_embeddings,
        # and vision RoPE may use FlashInfer with explicit position_ids.

        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv, None, None
        q, k, v = self.split_qkv(q, k, v)

        q, k, v = self.apply_rope(q, k, v, position_ids, position_embeddings)
        q, k, v = self.convert_qkv(q, k, v)

        output = self.forward_impl(
            q=q,
            k=k,
            v=v,
            attn_metadata=attn_metadata,
            attention_mask=PredefinedAttentionMask.FULL,
            attention_window_size=None,
            attention_mask_data=None,
            mrope_config=None,
            attention_sinks=None,
        )
        attn_output = self.o_proj(output, layer_idx=self.layer_idx)
        return attn_output


class Qwen2_5_VLMLP(GatedMLP):

    def __init__(self, model_config: ModelConfig[PretrainedConfig],
                 layer_idx: int):
        config = model_config.pretrained_config.vision_config
        super().__init__(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=True,
            activation=F.silu,
            dtype=model_config.pretrained_config.torch_dtype,
            config=model_config,
            layer_idx=layer_idx,
        )


class Qwen2_5_VLVisionBlock(torch.nn.Module):

    def __init__(self, model_config: ModelConfig[PretrainedConfig],
                 layer_idx: int):
        super().__init__()
        config = model_config.pretrained_config.vision_config
        self.norm1 = RMSNorm(
            hidden_size=config.hidden_size,
            eps=model_config.pretrained_config.text_config.rms_norm_eps,
            dtype=model_config.pretrained_config.torch_dtype)
        self.norm2 = RMSNorm(
            hidden_size=config.hidden_size,
            eps=model_config.pretrained_config.text_config.rms_norm_eps,
            dtype=model_config.pretrained_config.torch_dtype)
        self.attn = Qwen2_5_VLVisionAttention(model_config, layer_idx)
        self.mlp = Qwen2_5_VLMLP(model_config, layer_idx)

    @torch.inference_mode()
    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        position_ids: Optional[torch.IntTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Collapse the post-attn residual add into `norm2`'s residual path
        # (`RMSNorm.forward` with `residual=` uses the FlashInfer
        # `fused_add_rmsnorm` kernel). Saves one `add<c10::BFloat16>`
        # launch per vision block (= 32 launches per executor iter at
        # full-batch).
        x_attn = self.attn(
            hidden_states=self.norm1(hidden_states),
            attn_metadata=attn_metadata,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        x_fused_norm, residual = self.norm2(hidden_states, residual=x_attn)
        return residual + self.mlp(x_fused_norm)


class Qwen2_5_VLPatchMerger(torch.nn.Module):

    def __init__(self,
                 model_config: ModelConfig[PretrainedConfig],
                 spatial_merge_size: int = 2) -> None:
        super().__init__()
        config = model_config.pretrained_config.vision_config
        dim = config.out_hidden_size
        context_dim = config.hidden_size
        self.hidden_size = context_dim * (spatial_merge_size**2)
        self.ln_q = RMSNorm(
            hidden_size=context_dim,
            eps=model_config.pretrained_config.text_config.rms_norm_eps,
            dtype=model_config.pretrained_config.torch_dtype)
        self.mlp = torch.nn.Sequential(
            Linear(in_features=self.hidden_size,
                   out_features=self.hidden_size,
                   bias=True,
                   dtype=model_config.pretrained_config.torch_dtype,
                   mapping=model_config.mapping,
                   tensor_parallel_mode=TensorParallelMode.COLUMN,
                   allreduce_strategy=model_config.allreduce_strategy),
            torch.nn.GELU(),
            Linear(in_features=self.hidden_size,
                   out_features=dim,
                   bias=True,
                   dtype=model_config.pretrained_config.torch_dtype,
                   mapping=model_config.mapping,
                   tensor_parallel_mode=TensorParallelMode.ROW,
                   allreduce_strategy=model_config.allreduce_strategy),
        )

    @torch.inference_mode()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.ln_q(hidden_states)
        hidden_states = hidden_states.view(-1, self.hidden_size)
        hidden_states = self.mlp(hidden_states)
        return hidden_states


class Qwen2_5_VisionModel(torch.nn.Module, MultimodalEncoderMixin):

    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__()
        self.model_config = model_config
        self.config = self.model_config.pretrained_config.vision_config

        self.spatial_merge_size = self.config.spatial_merge_size
        self.patch_size = self.config.patch_size
        self.fullatt_block_indexes = self.config.fullatt_block_indexes
        self.window_size = self.config.window_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = Qwen2_5_VisionPatchEmbed(
            patch_size=self.config.patch_size,
            temporal_patch_size=self.config.temporal_patch_size,
            in_channels=self.config.in_channels,
            embed_dim=self.config.hidden_size,
        )

        text_config = getattr(model_config.pretrained_config, "text_config",
                              model_config.pretrained_config)
        self.config.max_position_embeddings = text_config.max_position_embeddings
        self.config.partial_rotary_factor = 0.5
        self.head_dim = self.config.hidden_size // self.config.num_heads
        self.pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.rope_gpt_neox,
            rope=RopeParams.from_config(self.config),
        )
        self.rotary_pos_emb = RotaryEmbedding(
            self.pos_embd_params.rope,
            head_dim=self.head_dim,
            is_neox=self.pos_embd_params.is_neox,
        )

        self.blocks = torch.nn.ModuleList([
            Qwen2_5_VLVisionBlock(model_config, layer_idx=layer_idx)
            for layer_idx in range(self.config.depth)
        ])
        self.merger = Qwen2_5_VLPatchMerger(self.model_config, )
        self.metadata_cls = get_attention_backend(
            self.model_config.attn_backend).Metadata
        self.full_attn_metadata: Optional[AttentionMetadata] = None
        self.window_attn_metadata: Optional[AttentionMetadata] = None
        # Pre-allocated `arange` for the vision block's `rope_position_ids`;
        # per-call code slices `[:seq_len]` instead of a fresh `(seq_len,) int32`
        # + H->D copy. Sized by `setup_attn_metadata` (engine-driven); `forward`
        # grows it on the rare miss above the budget.
        self.register_buffer("_rope_position_ids_buffer",
                             None,
                             persistent=False)

    def setup_attn_metadata(self, max_num_requests: int,
                            max_num_tokens: int) -> None:
        # Override: Qwen2/2.5-VL uses two metadata objects (full + window
        # attention) instead of the mixin's single ``attn_metadata``.
        #
        # Windowed attention splits each image into many attention sequences
        # (one per window grid cell), so ``max_num_requests`` here is the
        # **window** count, not the image count, and can far exceed the
        # LLM-side ``max_batch_size`` that ``encoder_max_batch_size`` falls back
        # to. Floor it at the same legacy fallback the mixin uses (see the TODO
        # there: derive from ``encoder_max_num_tokens`` once the scheduler caps
        # encoder forwards at it).
        max_num_requests = max(max_num_requests,
                               _ENCODER_FALLBACK_MAX_NUM_REQUESTS)
        kwargs = dict(max_num_requests=max_num_requests,
                      max_num_tokens=max_num_tokens,
                      kv_cache_manager=None)
        self.full_attn_metadata = self.metadata_cls(**kwargs)
        self.window_attn_metadata = self.metadata_cls(**kwargs)
        # Size the vision-block ``rope_position_ids`` scratch to the encoder
        # token budget; ``forward`` grows it on the rare miss above the budget.
        self._rope_position_ids_buffer = torch.arange(max_num_tokens,
                                                      dtype=torch.int32,
                                                      device=self.device)

    def get_rotary_pos_emb_window_data(
        self, grid_rows: List[List[int]]
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor],
               List[int]]:
        window_index_id = 0
        rotary_pos_emb_cos: List[torch.Tensor] = []
        rotary_pos_emb_sin: List[torch.Tensor] = []
        window_indices: List[torch.Tensor] = []
        window_seq_lens: List[int] = []
        for row in grid_rows:
            t, h, w = int(row[0]), int(row[1]), int(row[2])
            llm_h = h // self.spatial_merge_size
            llm_w = w // self.spatial_merge_size
            (cos_thw, sin_thw, window_index_thw,
             window_seq_lens_thw) = self.get_rope_and_window_index_by_thw(
                 t, h, w)

            window_indices.append(window_index_thw + window_index_id)
            window_index_id += t * llm_h * llm_w

            rotary_pos_emb_cos.append(cos_thw)
            rotary_pos_emb_sin.append(sin_thw)

            window_seq_lens.extend(window_seq_lens_thw)

        return (rotary_pos_emb_cos, rotary_pos_emb_sin, window_indices,
                window_seq_lens)

    def get_window_index_by_thw(self, grid_t: int, grid_h: int,
                                grid_w: int) -> Tuple[torch.Tensor, List[int]]:
        vit_merger_window_size = (self.window_size // self.spatial_merge_size //
                                  self.patch_size)
        llm_grid_h = grid_h // self.spatial_merge_size
        llm_grid_w = grid_w // self.spatial_merge_size
        index = torch.arange(grid_t * llm_grid_h * llm_grid_w,
                             dtype=torch.long).reshape(grid_t, llm_grid_h,
                                                       llm_grid_w)
        pad_h = vit_merger_window_size - llm_grid_h % vit_merger_window_size
        pad_w = vit_merger_window_size - llm_grid_w % vit_merger_window_size
        num_windows_h = (llm_grid_h + pad_h) // vit_merger_window_size
        num_windows_w = (llm_grid_w + pad_w) // vit_merger_window_size
        index_padded = F.pad(index, (0, pad_w, 0, pad_h), "constant", PAD_INDEX)
        index_padded = index_padded.reshape(
            grid_t,
            num_windows_h,
            vit_merger_window_size,
            num_windows_w,
            vit_merger_window_size,
        )
        index_padded = index_padded.permute(0, 1, 3, 2, 4).reshape(
            grid_t,
            num_windows_h * num_windows_w,
            vit_merger_window_size,
            vit_merger_window_size,
        )
        seqlens = (index_padded != PAD_INDEX).sum([2, 3]).reshape(-1)
        index_padded = index_padded.reshape(-1)
        index_new = index_padded[index_padded != PAD_INDEX]
        seqlens = seqlens * self.spatial_merge_unit
        return index_new, seqlens.tolist()

    @lru_cache(maxsize=1024)  # noqa: B019
    def get_rope_and_window_index_by_thw(
        self, t: int, h: int, w: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Tuple[int, ...]]:
        """GPU (cos_thw, sin_thw); CPU (window_index_thw, seq_lens_thw).

        Cached per `(t, h, w)`. pos_ids and window_index are built on CPU,
        then moved to device once per unique tile. The gather
        (`cos[pos_ids]`) and window reorder
        (`cos_thw[window_index_thw_dev]`) run on device, so the cached
        cos/sin tensors are already-on-device -- `forward`'s `torch.cat`
        on them is a device-side cat with no H->D transfer. `window_index`
        stays on CPU because `forward` cats all per-tile window_index
        tensors and ships them with a single H->D transfer.

        GPU memory cost (measured on H200, `head_dim=80`,
        `rotary_cos_sin` fp32; cos_thw and sin_thw each shape
        `(t*h*w, 2*head_dim)` after gather+flatten+window reorder):

          ============================  =======  =========
          tile (t, h, w)                tokens   per entry
          ============================  =======  =========
          (1, 16, 16)  -- 224**2          256      160 KB
          (1, 32, 32)  -- 448**2         1024      640 KB
          (1, 48, 48)  -- 672**2         2304     1.41 MB
          (1, 64, 64)  -- 1024**2        4096     2.50 MB
          (8, 32, 32)  -- 8-frame 448    8192     5.00 MB
          ============================  =======  =========

        Per-token cost is 640 bytes (= 2 (cos, sin) x 80 (head_dim) x 4
        bytes fp32). Typical production VLM serving has 10-30 unique tile
        shapes, so the cache settles around 10-20 MB. `maxsize=1024` is
        a safety cap; reaching it requires >1024 distinct (t, h, w).
        """
        hpos_ids = torch.arange(h, dtype=torch.long).unsqueeze(1).expand(-1, w)
        wpos_ids = torch.arange(w, dtype=torch.long).unsqueeze(0).expand(h, -1)
        hpos_ids = (hpos_ids.reshape(h // self.spatial_merge_size,
                                     self.spatial_merge_size,
                                     w // self.spatial_merge_size,
                                     self.spatial_merge_size).permute(
                                         0, 2, 1, 3).flatten())
        wpos_ids = (wpos_ids.reshape(h // self.spatial_merge_size,
                                     self.spatial_merge_size,
                                     w // self.spatial_merge_size,
                                     self.spatial_merge_size).permute(
                                         0, 2, 1, 3).flatten())
        pos_ids = torch.stack([hpos_ids, wpos_ids], dim=-1).repeat(t, 1)

        # Pos_ids -> device for the freq_table gather. rotary_cos_sin is
        # created on CUDA in RopeParams.create_rope_const_params; this is
        # the one H->D copy per unique (t, h, w). `async_tensor_h2d` is
        # the project-wide helper that guarantees pinned-host + async DMA
        # (a bare `.to(..., non_blocking=True)` on pageable memory
        # silently degrades to a staging copy).
        device = self.rotary_pos_emb.rotary_cos_sin.device
        pos_ids_dev = async_tensor_h2d(pos_ids,
                                       dtype=pos_ids.dtype,
                                       device=device)
        max_grid_size = max(h, w)
        cos_sin = self.rotary_pos_emb.rotary_cos_sin[:max_grid_size]
        cos, sin = cos_sin[:, 0, :], cos_sin[:, 1, :]
        cos_flattened = cos[pos_ids_dev].flatten(1)
        sin_flattened = sin[pos_ids_dev].flatten(1)

        cos_thw = cos_flattened.reshape(
            cos_flattened.shape[0] // self.spatial_merge_unit,
            self.spatial_merge_unit,
            -1,
        )
        sin_thw = sin_flattened.reshape(
            sin_flattened.shape[0] // self.spatial_merge_unit,
            self.spatial_merge_unit,
            -1,
        )

        window_index_thw, seq_lens_thw = self.get_window_index_by_thw(t, h, w)
        # Device-side window reorder; cached cos/sin stay on device. Same
        # pinned-host + async DMA pattern as the pos_ids transfer above.
        window_index_thw_dev = async_tensor_h2d(window_index_thw,
                                                dtype=window_index_thw.dtype,
                                                device=device)
        cos_thw = cos_thw[window_index_thw_dev, :, :].reshape(
            -1, cos_thw.shape[-1])
        sin_thw = sin_thw[window_index_thw_dev, :, :].reshape(
            -1, sin_thw.shape[-1])

        # Cast once (per cached (t, h, w)) to the vision-tower dtype so the
        # per-block `cos.to(dtype=q.dtype)` cast in `apply_rope` becomes
        # a no-op on the hot path.
        target_dtype = (
            self.model_config.pretrained_config.vision_config.torch_dtype
            if hasattr(self.model_config.pretrained_config.vision_config,
                       "torch_dtype") else
            self.model_config.pretrained_config.torch_dtype)
        cos_thw = cos_thw.to(target_dtype)
        sin_thw = sin_thw.to(target_dtype)

        return cos_thw, sin_thw, window_index_thw, tuple(seq_lens_thw)

    def prepare_attn_metadata(
            self,
            seq_lens: List[int],
            attn_metadata: Optional[AttentionMetadata] = None):
        if attn_metadata is None:
            raise RuntimeError(
                "Vision encoder AttentionMetadata is not initialized. "
                "It must be set up before the encoder forward runs.")
        return _prepare_qwen_vl_vision_attn_metadata(seq_lens, attn_metadata)

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    @torch.inference_mode()
    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor,
                **kwargs) -> torch.Tensor:

        hidden_states = self.patch_embed(pixel_values)

        seq_len, _ = hidden_states.size()
        grid_rows = grid_thw.tolist()

        (rotary_pos_emb_cos, rotary_pos_emb_sin, window_indices,
         window_seq_lens) = self.get_rotary_pos_emb_window_data(grid_rows)

        # cos/sin are already device tensors -- get_rope_and_window_index_by_thw
        # caches them on `self.rotary_pos_emb.rotary_cos_sin.device`, so the
        # cat runs on device with no H->D transfer.
        cos = torch.cat(rotary_pos_emb_cos)
        sin = torch.cat(rotary_pos_emb_sin)
        position_embeddings = (cos, sin)

        # window_index is built per tile on CPU (lru_cached). The cat
        # result is a fresh pageable tensor, so route the H->D copy
        # through async_tensor_h2d to land it on a pinned host buffer
        # first -- otherwise `non_blocking=True` silently stages.
        window_index = async_tensor_h2d(torch.cat(window_indices),
                                        dtype=torch.long,
                                        device=self.device)

        # window_index is a permutation: it maps original token order ->
        # window order. Its inverse permutation is exactly argsort(window_index),
        # which torch implements as a fused gpu sort rather than the
        # alloc + scatter pair the previous code used.
        reverse_indices = torch.argsort(window_index)

        # Pre-allocated 0..seq_len-1 positions (sized in setup_attn_metadata);
        # slice instead of a fresh arange + H->D copy per forward.
        if (self._rope_position_ids_buffer is None
                or seq_len > self._rope_position_ids_buffer.numel()):
            self._rope_position_ids_buffer = torch.arange(seq_len,
                                                          dtype=torch.int32,
                                                          device=self.device)
        rope_position_ids = self._rope_position_ids_buffer[:seq_len]
        seq_lens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2],
                                           grid_thw[:, 0]).tolist()

        hidden_states = hidden_states.reshape(
            seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :].reshape(seq_len, -1)

        self.full_attn_metadata = self.prepare_attn_metadata(
            seq_lens, self.full_attn_metadata)
        self.window_attn_metadata = self.prepare_attn_metadata(
            window_seq_lens, self.window_attn_metadata)

        for layer_num, block in enumerate(self.blocks):
            if layer_num in self.fullatt_block_indexes:
                attn_metadata = self.full_attn_metadata
            else:
                attn_metadata = self.window_attn_metadata
            hidden_states = block(
                hidden_states=hidden_states,
                attn_metadata=attn_metadata,
                position_embeddings=position_embeddings,
                position_ids=rope_position_ids,
            )
        hidden_states = self.merger(hidden_states)
        hidden_states = hidden_states[reverse_indices, :]

        return hidden_states


class Qwen2VLModelBase(PreTrainedModel, MultimodalModelMixin):

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        *args,
        **kwargs,
    ) -> None:
        self.original_arch = model_config.pretrained_config.architectures[0]

        # NOTE: Setting disable_fuse_rope to True to do mrope fusion in the model engine by pre-computing rotary_cos_sin in the model engine
        disable_fuse_rope = kwargs.get('disable_fuse_rope', False)
        model_config.pretrained_config.disable_fuse_rope = disable_fuse_rope
        # In transformers 5.x, rope_scaling is a property that reads
        # rope_parameters, which may not exist on Qwen2_5_VLConfig.
        # Use getattr to safely check and initialize.
        rope_scaling = getattr(model_config.pretrained_config, 'rope_scaling',
                               None)
        if rope_scaling is None or not isinstance(rope_scaling, dict):
            rope_scaling = {}
            model_config.pretrained_config.rope_scaling = rope_scaling
        rope_scaling['type'] = 'mrope'
        config = model_config.pretrained_config

        self._supports_sdpa = True
        super().__init__(config)

        self.model_config = model_config
        self.config = model_config.pretrained_config

        if model_config.attn_backend != 'TRTLLM':
            raise ValueError("Qwen2/2.5-VL only supports TRTLLM backend now")
        if not disable_fuse_rope:
            self.init_mrope_embedding(model_config)
            # Extra slot is reserved for CUDA graph / warmup dummy requests.
            max_mrope_delta_slots = (
                model_config.max_num_tokens * model_config.mapping.pp_size + 1)
            self.register_buffer('mrope_position_deltas_cache',
                                 torch.zeros(max_mrope_delta_slots,
                                             dtype=torch.int32,
                                             device='cuda'),
                                 persistent=False)
            rotary_dim = self.rotary_emb.rotary_cos_sin.shape[-1]
            self.register_buffer(
                'mrope_rotary_cos_sin_workspace',
                torch.empty((1, model_config.max_num_tokens * rotary_dim * 2),
                            dtype=torch.float32,
                            device='cuda'),
                persistent=False)

        llm_model_config = copy.deepcopy(model_config)
        text_config = getattr(llm_model_config.pretrained_config, 'text_config',
                              None)
        if text_config is not None:
            llm_model_config.pretrained_config = text_config
        llm_model_config.pretrained_config.disable_fuse_rope = disable_fuse_rope
        llm_model_config.pretrained_config.architectures = ["Qwen2ForCausalLM"]
        # The LM's attention modules look themselves up in the global
        # `extra_attrs` that `model_engine.model_forward` binds via
        # `with_model_extra_attrs(self.model.extra_attrs)` -- the
        # outer wrapper's dict. Without sharing, `llm_model_config`
        # carries a deep-copied dict, so LM `attn_custom_op_inplace`
        # (used under `set_torch_compiling(True)`) fails its layer
        # lookup and the piecewise-CUDA-graph dynamo trace blows up
        # at the LM's `o_proj` call. Vision attention unregisters
        # itself from this dict in `Qwen2_5_VLVisionAttention.__init__`
        # so it does not poison LM lookups.
        llm_model_config.extra_attrs = model_config.extra_attrs
        self.llm = AutoModelForCausalLM.from_config(llm_model_config)

        # Normal worker owns encoder. MM E/P prefill worker gets attached embeddings.
        if not _is_mm_disagg():
            mm_encoder_config = copy.deepcopy(model_config)
            self.mm_encoder = Qwen2VisionModelBase(
                mm_encoder_config, kwargs.get('vision_model_class', None))
        else:
            self.mm_encoder = None

        # Surface the in-vocab image / vision / video placeholder IDs to the
        # model engine's ``_prepare_multimodal_indices`` so it selects the
        # ``torch.isin`` predicate. Filter out None — some Qwen2-VL variants
        # only define a subset of these.
        _mm_ids = [
            tid for tid in (
                getattr(self.config, 'image_token_id', None),
                getattr(self.config, 'vision_token_id', None),
                getattr(self.config, 'video_token_id', None),
            ) if tid is not None
        ]
        self._mm_token_ids = torch.tensor(_mm_ids, dtype=torch.int32)

    @property
    def mm_token_ids(self) -> torch.Tensor:
        return self._mm_token_ids

    def init_mrope_embedding(self, model_config: ModelConfig[PretrainedConfig]):
        config = model_config.pretrained_config
        # For VL configs (Qwen2_5_VLConfig), hidden_size etc. live in
        # text_config. Use text_config for RoPE params if available.
        rope_config = getattr(config, 'text_config', config)
        # mrope_section may be in text_config.rope_scaling for VL configs
        rope_scaling_for_mrope = getattr(rope_config, 'rope_scaling',
                                         None) or {}
        mrope_section = rope_scaling_for_mrope.get('mrope_section', None)
        pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.from_string(
                rope_config.rope_scaling["type"]),
            rope=RopeParams.from_config(rope_config),
            mrope_section=mrope_section)
        self.rotary_emb = MRotaryEmbedding(
            pos_embd_params.rope,
            head_dim=rope_config.hidden_size // rope_config.num_attention_heads,
            is_neox=pos_embd_params.is_neox,
            mrope_section=pos_embd_params.mrope_section,
        ).to('cuda')

    def load_weights(self, weights, weight_mapper: BaseWeightMapper):
        pass

    @property
    def vocab_size_padded(self) -> int:
        return self.llm.vocab_size_padded

    def infer_max_seq_len(self) -> int:
        return self.llm.infer_max_seq_len()

    def encode_multimodal_inputs(self,
                                 multimodal_params: List[MultimodalParams],
                                 **encoder_kwargs: Any) -> torch.Tensor:
        """Uniform encoder entry (``MultimodalModelMixin`` contract).

        Runs the vision encoder over ``multimodal_params`` and returns the
        embeddings as a single tensor (Qwen folds any deepstack streams into
        the hidden dim, so the single-tensor contract holds). Used by the
        startup memory profiler to invoke the encoder directly; the model's
        own ``forward`` keeps its custom deepstack fusion path.
        """
        mm_embeds = get_multimodal_embeddings(
            encoder_forward_fn=self.mm_encoder.forward,
            multimodal_params=list(multimodal_params))
        return mm_embeds[0]

    def apply_llm_torch_compile(self, *, backend: Any, fullgraph: bool) -> None:
        # TODO: Move this hook to MultimodalModelMixin once multimodal models
        # consistently expose an LLM compile contract.
        """Compile only the LLM decoder; the vision encoder stays eager."""
        self.llm.model = torch.compile(self.llm.model,
                                       backend=backend,
                                       fullgraph=fullgraph)

    @nvtx_range("Qwen2.5-VL prepare_mrope_config")
    def prepare_mrope_config(
            self,
            multimodal_params: List[MultimodalParams],
            num_generation_requests: int,
            position_ids: torch.Tensor,
            mrope_delta_write_seq_slots: Optional[torch.Tensor] = None,
            mrope_delta_read_seq_slots: Optional[torch.Tensor] = None):
        return _prepare_qwen_vl_mrope_config(
            multimodal_params=multimodal_params,
            num_generation_requests=num_generation_requests,
            position_ids=position_ids,
            rotary_emb=self.rotary_emb,
            mrope_position_deltas_cache=self.mrope_position_deltas_cache,
            mrope_rotary_cos_sin_workspace=self.mrope_rotary_cos_sin_workspace,
            mrope_delta_write_seq_slots=mrope_delta_write_seq_slots,
            mrope_delta_read_seq_slots=mrope_delta_read_seq_slots)

    @torch.inference_mode()
    def forward(
        self,
        attn_metadata: AttentionMetadata,
        input_ids: Optional[torch.IntTensor] = None,
        position_ids: Optional[torch.IntTensor] = None,
        input_embeds: Optional[torch.Tensor] = None,
        return_context_logits: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        """
        VLM forward logic with inflight batching support.
        """
        num_context_requests, num_generation_requests = (
            attn_metadata.num_contexts, attn_metadata.num_generations)

        multimodal_params = kwargs.get("multimodal_params", [])
        mm_embeds = []
        mrope_config = {}
        # NOTE: Qwen*-VL series has mrope_config even on the text-only prompts, so we need to separate
        # the entries that do have multimodal data from those that correspond to text-only prompts.
        if num_context_requests > 0:
            mm_multimodal_params = self._get_requests_with_mm_data(
                multimodal_params[:num_context_requests])
        else:
            mm_multimodal_params = []
        if len(mm_multimodal_params) > 0:
            # Local encoder present: raw pixels/videos become embeddings here.
            if self.mm_encoder is not None:
                mm_embeds = get_multimodal_embeddings(
                    encoder_forward_fn=self.mm_encoder.forward,
                    multimodal_params=mm_multimodal_params)
            elif not getattr(self, "support_mm_disagg", False):
                raise NotImplementedError(
                    "Qwen2VLModel does not support disaggregated inference yet. Please unset "
                    "the TLLM_MULTIMODAL_DISAGGREGATED environment variable, or set it to '0'."
                )
            # E/P prefill: encoder already ran; use attached embeddings.
            else:
                mm_embeds = get_attached_multimodal_embeddings(
                    mm_multimodal_params)
            mm_embeds = find_input_mm_embeds(mm_embeds, mm_multimodal_params)

        if not self.model_config.pretrained_config.disable_fuse_rope:
            mrope_config = self.prepare_mrope_config(
                multimodal_params,
                num_generation_requests,
                position_ids,
                mrope_delta_write_seq_slots=kwargs.get(
                    "mrope_delta_write_seq_slots"),
                mrope_delta_read_seq_slots=kwargs.get(
                    "mrope_delta_read_seq_slots"))

        input_ids, input_embeds = fuse_input_embeds(
            self.llm.model.embed_tokens,
            input_ids,
            mm_embeds,
            mm_token_ids=self.mm_token_ids,
            mm_token_indices=kwargs.get("mm_token_indices"),
            text_token_indices=kwargs.get("text_token_indices"),
        )
        output_prob = self.llm.forward(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=input_embeds,
            return_context_logits=return_context_logits,
            mrope_config=mrope_config)

        logger.debug(f'output shape: {output_prob.shape}')
        return output_prob

    def _get_requests_with_mm_data(self, multimodal_params):
        mm_multimodal_params = []
        for multimodal_param in multimodal_params:
            data = multimodal_param.multimodal_data
            if (
                    # The first 2 conditions check whether there is input on which inference should be run.
                    data.get("image", {}).get("pixel_values") is not None or
                    data.get("video", {}).get("pixel_values_videos") is not None
                    # This condition corresponds to when the embeddings are already populated, as is e.g.
                    # the case in EPD disagg in the prefill worker.
                    or data.get("multimodal_embedding") is not None):
                mm_multimodal_params.append(multimodal_param)

        return mm_multimodal_params


@register_vision_encoder(Qwen2VisionModelBase,
                         vlm_base_model=Qwen2VisionTransformerPretrainedModel)
@register_auto_model("Qwen2VLForConditionalGeneration")
@register_input_processor(
    Qwen2VLInputProcessorBase,
    model_type="qwen2_vl",
    placeholder_metadata=MultimodalPlaceholderMetadata(
        placeholder_map={
            "image": "<|vision_start|><|image_pad|><|vision_end|>",
            "video": "<|vision_start|><|video_pad|><|vision_end|>"
        },
        placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
        content_format=ContentFormat.STRING,
    ))
class Qwen2VLModel(Qwen2VLModelBase):

    def __init__(self, model_config: ModelConfig[PretrainedConfig], *args,
                 **kwargs):
        # NOTE: Since Qwen2-VL is outdated model, we leave it as HF implementation.
        kwargs['vision_model_class'] = Qwen2VisionTransformerPretrainedModel
        super().__init__(model_config, *args, **kwargs)

    @property
    def multimodal_data_device_paths(self) -> List[str]:
        return [
            "image.pixel_values", "image.image_grid_thw",
            "video.pixel_values_videos", "video.video_grid_thw",
            "multimodal_embedding", "mrope_config.mrope_position_ids",
            "mrope_config.mrope_position_deltas"
        ]

    def load_weights(self, weights, weight_mapper: BaseWeightMapper):
        if self.mm_encoder is not None:
            self.mm_encoder.load_weights(weights)

        self.llm.load_weights(weights, weight_mapper)


class Qwen2_5VLInputProcessorBase(Qwen2VLInputProcessorBase):

    def build_disagg_prefill_multimodal_inputs(
            self, inputs: TextPrompt,
            mm_handles: List[Dict[str, Any]]) -> DisaggPrefillMultimodalInputs:
        """
        Build disaggregated prefill inputs from multimodal embedding handles.

        Args:
            inputs: Text prompt input container. Must contain a non-empty prompt string.
            mm_handles: List of multimodal embedding handles.

        Returns:
            DisaggPrefillMultimodalInputs containing expanded token IDs,
            prompt-side MM positions/lengths, exact runs, and encoder-output
            embedding lengths.
        """
        # TODO: Move this function to the base input processor class when extending for more models
        text_prompt = inputs.get("prompt")
        if not text_prompt:
            raise ValueError("Text prompt is required but not provided")

        if not isinstance(mm_handles, list):
            raise TypeError("mm_handles must be a list")

        expected_hidden_size = self.config.text_config.hidden_size
        for i, mm_handle in enumerate(mm_handles):
            hidden_size = mm_handle['tensor_size'][1]
            if hidden_size != expected_hidden_size:
                raise RuntimeError(
                    f"Multimodal embedding {i} hidden size {hidden_size} must match model hidden size {expected_hidden_size}"
                )
        input_ids = self.tokenizer(text_prompt,
                                   return_tensors="pt").input_ids[0]

        image_token_index = self.config.image_token_id

        image_mask = input_ids == image_token_index
        image_positions = torch.where(image_mask)[0]
        num_images = len(image_positions)
        assert num_images == len(
            mm_handles), "Number of images must match number of mm_handles"
        total_mm_tokens = sum(mm_handle["tensor_size"][0]
                              for mm_handle in mm_handles)
        final_length = len(input_ids) - num_images + total_mm_tokens
        # Create output tensor
        expanded_ids = torch.empty(final_length, dtype=input_ids.dtype)
        # Use the in-vocab image_token_id as the placeholder repeated mm_token_num
        # times, so the model engine can locate mm positions via
        # ``torch.isin(input_ids, mm_token_ids)`` without the legacy OOV remap.
        placeholder_id = image_token_index

        # Fill the expanded sequence
        write_pos = 0
        image_cnt = 0
        mm_token_length = []
        mm_token_offsets = []
        for read_pos in range(len(input_ids)):
            if input_ids[read_pos] == image_token_index:
                # Replace with placeholder id
                mm_token_num = mm_handles[image_cnt]["tensor_size"][0]
                expanded_ids[write_pos:write_pos + mm_token_num] = \
                    placeholder_id
                mm_token_offsets.append(write_pos)
                mm_token_length.append(mm_token_num)
                write_pos += mm_token_num
                image_cnt += 1
            else:
                # Copy text token as-is
                expanded_ids[write_pos] = input_ids[read_pos]
                write_pos += 1

        assert write_pos == final_length, f"Write position mismatch: {write_pos} != {final_length}"
        assert mm_token_length[-1] + mm_token_offsets[
            -1] <= final_length, f"mm_token_length[-1] + mm_token_offsets[-1] ({mm_token_length[-1] + mm_token_offsets[-1]}) should be less than or equal to final_length ({final_length})"
        return DisaggPrefillMultimodalInputs(
            prompt_token_ids=expanded_ids.to(torch.int32).tolist(),
            multimodal_lengths=mm_token_length,
            multimodal_positions=mm_token_offsets,
            multimodal_embedding_lengths=[
                mm_handle["tensor_size"][0] for mm_handle in mm_handles
            ],
            multimodal_item_run_cu_offsets=list(range(len(mm_token_length) +
                                                      1)),
            multimodal_run_positions=mm_token_offsets,
            multimodal_run_lengths=mm_token_length,
        )


@support_multimodal_disaggregated
@register_vision_encoder(Qwen2VisionModelBase,
                         vlm_base_model=Qwen2_5_VisionModel)
@register_auto_model("Qwen2_5_VLForConditionalGeneration")
@register_input_processor(
    Qwen2_5VLInputProcessorBase,
    model_type="qwen2_5_vl",
    placeholder_metadata=MultimodalPlaceholderMetadata(
        placeholder_map={
            "image": "<|vision_start|><|image_pad|><|vision_end|>",
            "video": "<|vision_start|><|video_pad|><|vision_end|>"
        },
        placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
        content_format=ContentFormat.STRING,
    ))
class Qwen2_5_VLModel(Qwen2VLModelBase):

    def __init__(self, model_config: ModelConfig[PretrainedConfig], *args,
                 **kwargs):
        kwargs['vision_model_class'] = Qwen2_5_VisionModel
        kwargs['disable_fuse_rope'] = kwargs.get(
            'disable_fuse_rope',
            False)  # TODO: Make this ModelConfig's argument
        super().__init__(model_config, *args, **kwargs)

    @property
    def multimodal_data_device_paths(self) -> List[str]:
        return [
            "image.pixel_values", "video.pixel_values_videos",
            "multimodal_embedding", "mrope_config.mrope_position_ids",
            "mrope_config.mrope_position_deltas"
        ]

    def load_weights(self, weights, weight_mapper: BaseWeightMapper):
        if isinstance(weight_mapper, Qwen2VLHfWeightMapper):
            weights = weight_mapper.preprocess_weights(weights)

        if self.mm_encoder is not None:
            self.mm_encoder.load_weights(weights)

        self.llm.load_weights(weights)
