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
import triton
import triton.language as tl
from PIL import Image
from transformers import AutoTokenizer, PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN as HF_ACT2FN
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLVisionPatchEmbed as HFQwen3VLVisionPatchEmbed,
)

from tensorrt_llm._torch.models.modeling_multimodal_utils import _is_mm_disagg
from tensorrt_llm.functional import PositionEmbeddingType
from tensorrt_llm.mapping import Mapping

from ..._utils import async_tensor_h2d
from ...inputs import (
    ContentFormat,
    MultimodalPlaceholderMetadata,
    MultimodalPlaceholderPlacement,
    TextPrompt,
    register_input_processor,
    support_multimodal_disaggregated,
)
from ...inputs.multimodal import DisaggPrefillMultimodalInputs, MultimodalParams
from ...logger import logger
from ..attention_backend import AttentionMetadata
from ..attention_backend.interface import PositionalEmbeddingParams, RopeParams
from ..attention_backend.utils import get_attention_backend
from ..modules.layer_norm import LayerNorm
from ..modules.linear import Linear, TensorParallelMode
from ..modules.mlp import MLP
from ..modules.rotary_embedding import MRotaryEmbedding, RotaryEmbedding
from .checkpoints.base_weight_mapper import BaseWeightMapper
from .checkpoints.hf.qwen3vl_weight_mapper import Qwen3VLHfWeightMapper
from .modeling_auto import AutoModelForCausalLM
from .modeling_multimodal_encoder import MultimodalEncoderMixin
from .modeling_multimodal_mixin import MultimodalModelMixin
from .modeling_multimodal_utils import (
    filter_mm_token_from_input_ids,
    find_input_mm_embeds,
    fuse_input_embeds,
    get_attached_multimodal_embeddings,
    get_multimodal_embeddings,
)
from .modeling_qwen2vl import (
    Qwen2_5_VLVisionAttention,
    Qwen2VLInputProcessorBase,
    _prepare_qwen_vl_mrope_config,
    _prepare_qwen_vl_vision_attn_metadata,
)
from .modeling_utils import (
    ModelConfig,
    QuantConfig,
    _load_weights_impl,
    filter_weights,
    register_auto_model,
    register_vision_encoder,
)


def _expand_prompt_token_ids_for_mm_handoff(
    input_ids: torch.Tensor,
    mm_handles: List[Dict[str, Any]],
    *,
    image_token_id: int,
    video_token_id: int,
    vision_start_token_id: int,
) -> DisaggPrefillMultimodalInputs:
    """Expand Qwen3-VL image/video placeholders and emit sparse MM layout.

    Qwen handoff has one coarse <image_pad> or <video_pad> token per item.
    This helper expands that one token to the number of embedding rows in the
    handoff handle using the original in-vocab placeholder token, then returns
    the sparse layout metadata.

    Agg gets this expansion from Qwen's HF processor taking raw images/videos
    as inputs. Reusing that would be wasteful here, hence this helper that
    expands based on the embedding handles row count.

    """
    placeholder_positions = [
        pos
        for pos, token in enumerate(input_ids.tolist())
        if token in (image_token_id, video_token_id)
    ]
    if len(placeholder_positions) != len(mm_handles):
        raise ValueError(
            "Number of multimodal placeholders must match number of mm_handles: "
            f"placeholders={len(placeholder_positions)}, "
            f"mm_handles={len(mm_handles)}"
        )

    total_mm_embed_tokens = sum(mm_handle["tensor_size"][0] for mm_handle in mm_handles)
    final_length = len(input_ids) - len(placeholder_positions) + total_mm_embed_tokens
    expanded_ids = torch.empty(final_length, dtype=input_ids.dtype)

    mm_token_lengths: List[int] = []
    mm_token_offsets: List[int] = []
    item_types: List[int] = []
    item_run_cu_offsets: List[int] = [0]
    run_positions: List[int] = []
    run_lengths: List[int] = []
    multimodal_embedding_lengths: List[int] = []
    special_token_offsets: List[int] = []

    write_pos = 0
    mm_handle_idx = 0
    flat_mm_offset = 0
    for read_pos, token_id in enumerate(input_ids.tolist()):
        if token_id not in (image_token_id, video_token_id):
            expanded_ids[write_pos] = token_id
            write_pos += 1
            continue

        mm_token_num = mm_handles[mm_handle_idx]["tensor_size"][0]
        has_leading_special = (
            read_pos > 0 and int(input_ids[read_pos - 1].item()) == vision_start_token_id
        )
        run_start = write_pos - 1 if has_leading_special else write_pos
        prompt_mm_length = mm_token_num + int(has_leading_special)

        expanded_ids[write_pos : write_pos + mm_token_num] = token_id
        mm_token_offsets.append(run_start)
        mm_token_lengths.append(prompt_mm_length)
        multimodal_embedding_lengths.append(mm_token_num)
        item_types.append(0 if token_id == image_token_id else 1)
        run_positions.append(run_start)
        run_lengths.append(prompt_mm_length)
        item_run_cu_offsets.append(len(run_positions))

        if has_leading_special:
            special_token_offsets.append(flat_mm_offset)

        write_pos += mm_token_num
        flat_mm_offset += prompt_mm_length
        mm_handle_idx += 1

    if write_pos != final_length:
        raise RuntimeError(f"Write position mismatch: {write_pos} != {final_length}")

    return DisaggPrefillMultimodalInputs(
        prompt_token_ids=expanded_ids.to(torch.int32).tolist(),
        multimodal_lengths=mm_token_lengths,
        multimodal_positions=mm_token_offsets,
        multimodal_embedding_lengths=multimodal_embedding_lengths,
        multimodal_item_run_cu_offsets=item_run_cu_offsets,
        multimodal_run_positions=run_positions,
        multimodal_run_lengths=run_lengths,
        special_token_offsets=special_token_offsets,
        item_types=item_types,
    )


def _decide_do_sample_frames(
    video_datas: Optional[List[Any]],
    mm_processor_kwargs: Dict[str, Any],
) -> bool:
    """Pick a single `do_sample_frames` flag for the HF processor call.

    HF's video processor takes a scalar `do_sample_frames` that applies to
    every video in the request. Decide it as follows:

      1. If `mm_processor_kwargs.do_sample_frames` is explicitly set
         (True or False), honor it.
      2. If the caller supplies no frame target (`num_frames` / `fps`),
         match HF's class default, which samples frames (returns True).
      3. Otherwise, for each video compute the target frame count from the
         kwargs (`num_frames` directly, or `floor(duration * fps)` if
         `fps` is given) and compare to `len(vd.frames)`. If any video
         needs a different count, the batch is sampled (returns True).

    Per-video targets that match the IO-decoded count don't need HF
    sampling; the all-or-nothing reduction over the batch means a single
    video needing resampling pulls the rest along through a no-op
    identity `np.linspace`.
    """
    if "do_sample_frames" in mm_processor_kwargs:
        return bool(mm_processor_kwargs["do_sample_frames"])

    if not video_datas:
        return False

    user_num_frames = mm_processor_kwargs.get("num_frames")
    user_fps = mm_processor_kwargs.get("fps")
    has_num_frames = user_num_frames is not None and user_num_frames != -1
    has_fps = user_fps is not None and user_fps != -1

    # No explicit frame target from the caller: defer to HF's class-default
    # sampling (the stock processor sets `do_sample_frames=True` when neither
    # `num_frames` nor `fps` is given). Returning False here would hand the
    # IO-decoded frames straight to HF unchanged and diverge from stock HF
    # whenever the IO loader decoded a different number of frames than HF's
    # default sampler would select.
    if not has_num_frames and not has_fps:
        return True

    for vd in video_datas:
        n_decoded = len(vd.frames)
        if has_num_frames:
            n_target = user_num_frames
        else:  # has_fps
            duration = (vd.metadata or {}).get("duration") or 0
            n_target = math.floor(duration * user_fps)
        if n_target != n_decoded:
            return True
    return False


class Qwen3VLInputProcessorBase(Qwen2VLInputProcessorBase):
    """Qwen3-VL input processor.

    Reuses the Qwen2-VL implementation for tokenization, multimodal processor
    invocation, and rope-index construction. Only the dtype source and the
    per-grid temporal block differ — Qwen3-VL encodes video temporal info via
    separate timestamp tokens, so each frame is its own (1, h, w) block rather
    than a ``tokens_per_second``-scaled stretch.
    """

    def __init__(
        self,
        model_path: str,
        config: PretrainedConfig,
        tokenizer: AutoTokenizer,
        trust_remote_code: bool = True,
        **kwargs,
    ):
        super().__init__(
            model_path=model_path,
            config=config,
            tokenizer=tokenizer,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        # Qwen3-VL keeps ``torch_dtype`` only on ``text_config`` under transformers 5.x.
        self._dtype = self.config.text_config.dtype

    @classmethod
    def _build_temporal_block(
        cls,
        config: PretrainedConfig,
        llm_grid_t: int,
        llm_grid_h: int,
        llm_grid_w: int,
        second_per_grid_t: float,
    ) -> np.ndarray:
        # Qwen3-VL encodes video temporal info via separate timestamp tokens,
        # so the per-grid block is a plain ``np.indices`` lattice (no
        # ``tokens_per_second`` scaling).
        return np.indices((llm_grid_t, llm_grid_h, llm_grid_w)).reshape(3, -1)

    # Deterministic dummy-input sizing (`spatial_merge_unit`,
    # `_num_vision_tokens`, `get_size_for_max_tokens`) and the
    # `get_num_tokens_per_image` override are inherited unchanged from
    # `Qwen2VLInputProcessorBase` -- the grid math and the HF `smart_resize`
    # it defers to are identical for Qwen3-VL.

    @classmethod
    def get_rope_index(
        cls,
        config: PretrainedConfig,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Qwen3-VL splits videos with timestamp tokens like
        # ``<t1> <vision_start> <frame1> <vision_end> <t2> ...``, so
        # ``video_grid_thw`` must be expanded to one row per frame before the
        # shared Qwen2-VL traversal.
        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1
        return super().get_rope_index(
            config,
            input_ids,
            image_grid_thw,
            video_grid_thw,
            attention_mask,
            second_per_grid_ts,
        )

    def _preprocess(
        self,
        text: Dict[str, Any],
        mm_data: Dict[str, Any],
        mm_processor_kwargs: Dict[str, Any],
    ):
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

        do_sample_frames = _decide_do_sample_frames(video_datas, mm_processor_kwargs)

        # Pass `do_sample_frames` plus, when sampling is needed, the
        # caller's `num_frames` / `fps` target. Everything else the caller
        # supplied (resize, normalize knobs, etc.) flows through unchanged.
        proc_kwargs: Dict[str, Any] = {"do_sample_frames": do_sample_frames}
        for k, v in mm_processor_kwargs.items():
            if k in ("num_frames", "fps", "do_sample_frames"):
                continue
            proc_kwargs[k] = v
        if do_sample_frames:
            if "num_frames" in mm_processor_kwargs:
                proc_kwargs["num_frames"] = mm_processor_kwargs["num_frames"]
            if "fps" in mm_processor_kwargs:
                proc_kwargs["fps"] = mm_processor_kwargs["fps"]
            elif "num_frames" in mm_processor_kwargs:
                # HF's `sample_frames` honors `num_frames` only when `fps` is
                # not also set; the class-default `fps=2` would otherwise cap
                # the returned count below the caller's requested
                # `num_frames` for short clips. Null `fps` so `num_frames` is
                # respected verbatim.
                proc_kwargs["fps"] = None

        # Forward per-video metadata with `total_num_frames` rewritten to the
        # actual decoded frame count. HF's `sample_frames` computes indices
        # via `np.linspace(0, total_num_frames - 1, num_frames)` and indexes
        # the frame tensor with them; the rewrite keeps those indices in
        # range and the no-sampling path consistent for downstream qwen3vl
        # code that consults the metadata.
        video_metadata: Optional[List[Dict[str, Any]]] = None
        if video_datas:
            video_metadata = []
            for vd in video_datas:
                m = dict(vd.metadata or {})
                m["total_num_frames"] = len(vd.frames)
                video_metadata.append(m)

        return self.processor(
            text=[text],
            images=images,
            videos=videos,
            padding=True,
            do_rescale=do_rescale,
            return_tensors="pt",
            video_metadata=video_metadata,
            **proc_kwargs,
        )

    def get_num_tokens_per_video(
        self,
        *,
        video: List[Image.Image],
        video_grid_thw: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> int:
        merge = self.config.vision_config.spatial_merge_size
        if video_grid_thw is not None:
            if video_grid_thw.dim() == 1:
                t, h, w = (int(x) for x in video_grid_thw)
                return t * (h // merge) * (w // merge)
            token_counts = (
                video_grid_thw[:, 0]
                * (video_grid_thw[:, 1] // merge)
                * (video_grid_thw[:, 2] // merge)
            )
            return int(token_counts.sum().item())

        # HF's ``Qwen3VLProcessor._get_num_multimodal_tokens`` (what the base
        # class default delegates to) raises on video-only calls and returns a
        # wrong-formula fallback that would break chunked prefill, so we must
        # run the full processor here.
        do_rescale = not (video and isinstance(video[0], torch.Tensor))
        processed = self._processor(
            text=["<|vision_start|><|video_pad|><|vision_end|>"],
            videos=[video],
            padding=True,
            do_rescale=do_rescale,
            return_tensors="pt",
            **kwargs,
        )
        vgt = processed.get("video_grid_thw")
        if vgt is None or len(vgt) == 0:
            raise RuntimeError(
                "get_num_tokens_per_video: HF processor returned no "
                "video_grid_thw for the provided video."
            )
        return self.get_num_tokens_per_video(video=video, video_grid_thw=vgt)

    def get_preferred_media_io_kwargs(self) -> Dict[str, Dict[str, Any]]:
        # uint8 HWC frames let the HF processor rescale/permute once, skipping
        # the per-frame CHW-float conversion in the IO loader.
        return {"video": {"format": "np"}}

    def build_disagg_prefill_multimodal_inputs(
        self, inputs: TextPrompt, mm_handles: List[Dict[str, Any]]
    ) -> DisaggPrefillMultimodalInputs:
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

        num_deepstack_levels = len(self.config.vision_config.deepstack_visual_indexes)
        # This is because, unlike previous Qwen VL models, the embeddings are concatenated with
        # feature maps from deepstack layers.
        expected_size = self.config.text_config.hidden_size * (1 + num_deepstack_levels)
        for i, mm_handle in enumerate(mm_handles):
            hidden_size = mm_handle["tensor_size"][1]
            if hidden_size != expected_size:
                raise RuntimeError(
                    f"Expected multimodal embedding {i} to have hidden size {expected_size}, got {hidden_size}."
                )

        input_ids = self.tokenizer(text_prompt, return_tensors="pt").input_ids[0]

        return _expand_prompt_token_ids_for_mm_handoff(
            input_ids,
            mm_handles,
            image_token_id=self.config.image_token_id,
            video_token_id=self.config.video_token_id,
            vision_start_token_id=self.config.vision_start_token_id,
        )


class Qwen3VLVisionAttention(Qwen2_5_VLVisionAttention):
    def __init__(self, model_config, layer_idx):
        # Qwen3-VL keeps `torch_dtype` only on `text_config` under transformers 5.x
        # strict mode; mirror it onto `vision_config` so the parent picks it up.
        # `max_position_embeddings` lives on `text_config` only, so propagate it
        # to the top-level pretrained_config that the parent inspects.
        model_config.pretrained_config.max_position_embeddings = (
            model_config.pretrained_config.text_config.max_position_embeddings
        )
        model_config.pretrained_config.vision_config.torch_dtype = (
            model_config.pretrained_config.text_config.dtype
        )
        super().__init__(
            model_config,
            layer_idx=layer_idx,
            reduce_output=(
                not model_config.mapping.enable_attention_dp and model_config.mapping.tp_size > 1
            ),
        )


class Qwen3VLVisionMLP(MLP):
    def __init__(self, model_config: ModelConfig[PretrainedConfig], layer_idx: int):
        config = model_config.pretrained_config.vision_config
        super().__init__(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=True,
            activation=HF_ACT2FN[config.hidden_act],
            dtype=model_config.pretrained_config.text_config.dtype,
            config=model_config,
            layer_idx=layer_idx,
            overridden_tp_size=1 if model_config.mapping.enable_attention_dp else None,
        )


class Qwen3VLVisionBlock(torch.nn.Module):
    def __init__(self, model_config: ModelConfig[PretrainedConfig], layer_idx: int):
        super().__init__()
        self.model_config = model_config
        config = model_config.pretrained_config.vision_config

        self.norm1 = LayerNorm(
            hidden_size=config.hidden_size,
            eps=model_config.pretrained_config.text_config.rms_norm_eps,
            dtype=model_config.pretrained_config.text_config.dtype,
        )
        self.norm2 = LayerNorm(
            hidden_size=config.hidden_size,
            eps=model_config.pretrained_config.text_config.rms_norm_eps,
            dtype=model_config.pretrained_config.text_config.dtype,
        )
        self.attn = Qwen3VLVisionAttention(model_config, layer_idx)
        self.mlp = Qwen3VLVisionMLP(model_config, layer_idx)

    @torch.inference_mode()
    def forward(
        self,
        hidden_states: torch.Tensor,
        rotary_pos_emb: Optional[torch.Tensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Collapse the post-attn `residual + x_attn` add into `norm2`'s
        # residual path. `LayerNorm.forward` with `residual=` is
        # torch.compile-fused so both the `+` and the LN run inside one
        # Triton kernel -- saves one elementwise_kernel `add<c10::BFloat16>`
        # launch per vision block (32 launches per executor iter at full-batch).
        x_attn = self.attn(
            hidden_states=self.norm1(hidden_states),
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        x_fused_norm, residual = self.norm2(hidden_states, residual=x_attn)
        return residual + self.mlp(x_fused_norm)


class Qwen3VLVisionPatchMerger(torch.nn.Module):
    def __init__(
        self, model_config: ModelConfig[PretrainedConfig], use_postshuffle_norm: bool = False
    ) -> None:
        super().__init__()
        config = model_config.pretrained_config.vision_config
        self.hidden_size = config.hidden_size * (config.spatial_merge_size**2)
        self.use_postshuffle_norm = use_postshuffle_norm
        self.norm = LayerNorm(
            hidden_size=self.hidden_size if use_postshuffle_norm else config.hidden_size,
            eps=model_config.pretrained_config.text_config.rms_norm_eps,
            dtype=model_config.pretrained_config.text_config.dtype,
        )

        self.mapping = model_config.mapping
        overridden_tp_size = 1 if model_config.mapping.enable_attention_dp else None
        if overridden_tp_size is not None:
            assert self.mapping.tp_size % overridden_tp_size == 0
            tp_size = overridden_tp_size
            # "Misuse" pp_size here to perform all-reduce within smaller groups
            pp_size = self.mapping.pp_size * self.mapping.tp_size // overridden_tp_size
            mapping = Mapping(
                world_size=tp_size * pp_size,
                rank=self.mapping.rank,
                gpus_per_node=self.mapping.gpus_per_node,
                tp_size=tp_size,
                pp_size=pp_size,
            )
        else:
            mapping = self.mapping

        self.linear_fc1 = Linear(
            in_features=self.hidden_size,
            out_features=self.hidden_size,
            bias=True,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            allreduce_strategy=model_config.allreduce_strategy,
        )
        self.act_fn = nn.GELU()
        self.linear_fc2 = Linear(
            in_features=self.hidden_size,
            out_features=config.out_hidden_size,
            bias=True,
            mapping=mapping,
            tensor_parallel_mode=TensorParallelMode.ROW,
            allreduce_strategy=model_config.allreduce_strategy,
        )

    @torch.inference_mode()
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            hidden_states = hidden_states.view(-1, self.hidden_size)

        hidden_states = self.norm(hidden_states).view(-1, self.hidden_size)
        hidden_states = self.linear_fc1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        hidden_states = self.linear_fc2(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# Fused bilinear position-embedding interpolation for the Qwen3-VL vision
# tower.
#
# `fast_pos_embed_interpolate` resamples the learned grid of positional
# embeddings onto each (t, h, w) image grid using bilinear interpolation and
# then reorders the spatial axis to match the spatial-merge layout used by
# the rest of the vision tower. The Triton path fuses the
# bilinear-interp + spatial-merge reorder into a single kernel so the
# embedding gather, the 4 corner reads, and the permute are all one fused
# pass instead of separate ops.
# ---------------------------------------------------------------------------


@triton.jit
def _bilinear_pos_embed_kernel(
    embed_ptr,
    output_ptr,
    H,
    W,
    h_scale,
    w_scale,
    NUM_GRID: tl.constexpr,
    M_SIZE: tl.constexpr,
    HIDDEN_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Fused bilinear pos-embed interpolation with spatial-merge reorder."""
    pid = tl.program_id(0)
    total_spatial = H * W
    spatial_idx = pid % total_spatial

    num_blocks_w = W // M_SIZE
    block_idx = spatial_idx // (M_SIZE * M_SIZE)
    local_idx = spatial_idx % (M_SIZE * M_SIZE)
    br = block_idx // num_blocks_w
    bc = block_idx % num_blocks_w
    lr = local_idx // M_SIZE
    lc = local_idx % M_SIZE
    row = br * M_SIZE + lr
    col = bc * M_SIZE + lc

    h_frac = row.to(tl.float32) * h_scale
    w_frac = col.to(tl.float32) * w_scale

    hf = tl.math.floor(h_frac).to(tl.int32)
    wf = tl.math.floor(w_frac).to(tl.int32)
    hc = tl.minimum(hf + 1, NUM_GRID - 1)
    wc = tl.minimum(wf + 1, NUM_GRID - 1)

    dh = h_frac - hf.to(tl.float32)
    dw = w_frac - wf.to(tl.float32)
    w11 = dh * dw
    w10 = dh - w11
    w01 = dw - w11
    w00 = 1.0 - dh - w01

    off00 = (hf * NUM_GRID + wf) * HIDDEN_DIM
    off01 = (hf * NUM_GRID + wc) * HIDDEN_DIM
    off10 = (hc * NUM_GRID + wf) * HIDDEN_DIM
    off11 = (hc * NUM_GRID + wc) * HIDDEN_DIM
    out_off = pid * HIDDEN_DIM

    # Cast weights to output dtype so the multiply-accumulate stays in
    # the same precision as the reference PyTorch implementation used in
    # the unit test.
    out_dtype = output_ptr.dtype.element_ty
    w00_c = w00.to(out_dtype)
    w01_c = w01.to(out_dtype)
    w10_c = w10.to(out_dtype)
    w11_c = w11.to(out_dtype)

    for d in tl.range(0, HIDDEN_DIM, BLOCK_D):
        cols = d + tl.arange(0, BLOCK_D)
        mask = cols < HIDDEN_DIM

        e00 = tl.load(embed_ptr + off00 + cols, mask=mask)
        e01 = tl.load(embed_ptr + off01 + cols, mask=mask)
        e10 = tl.load(embed_ptr + off10 + cols, mask=mask)
        e11 = tl.load(embed_ptr + off11 + cols, mask=mask)

        val = w00_c * e00 + w01_c * e01 + w10_c * e10 + w11_c * e11

        tl.store(output_ptr + out_off + cols, val, mask=mask)


def _triton_pos_embed_interpolate(
    embed_weight: torch.Tensor,
    t: int,
    h: int,
    w: int,
    num_grid_per_side: int,
    m_size: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Launch the fused Triton kernel for one (t, h, w) grid.

    Returns a tensor of shape ``(t * h * w, hidden_dim)`` with the
    bilinearly-interpolated position embeddings already in spatial-merge
    order.
    """
    assert h % m_size == 0 and w % m_size == 0, (
        f"h={h} and w={w} must be divisible by m_size={m_size}"
    )
    hidden_dim = embed_weight.shape[1]
    total_out = t * h * w
    output = torch.empty(
        total_out,
        hidden_dim,
        device=embed_weight.device,
        dtype=dtype,
    )

    h_scale = (num_grid_per_side - 1) / (h - 1) if h > 1 else 0.0
    w_scale = (num_grid_per_side - 1) / (w - 1) if w > 1 else 0.0

    BLOCK_D = triton.next_power_of_2(hidden_dim)

    _bilinear_pos_embed_kernel[(total_out,)](
        embed_weight,
        output,
        h,
        w,
        h_scale,
        w_scale,
        num_grid_per_side,
        m_size,
        hidden_dim,
        BLOCK_D,
    )
    return output


class Qwen3VisionModel(torch.nn.Module, MultimodalEncoderMixin):
    def __init__(self, model_config: ModelConfig[PretrainedConfig]):
        super().__init__()
        self.model_config = model_config
        self.config = self.model_config.pretrained_config.vision_config

        self.spatial_merge_size = self.config.spatial_merge_size
        self.patch_size = self.config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.patch_embed = HFQwen3VLVisionPatchEmbed(
            config=self.config,
        )

        self.pos_embed = nn.Embedding(self.config.num_position_embeddings, self.config.hidden_size)
        self.num_grid_per_side = int(self.config.num_position_embeddings**0.5)

        # 2D rotary positional embedding for the vision tower. Reuse the
        # generic `RotaryEmbedding` cos/sin buffer (sized to the text
        # config's `max_position_embeddings` via `RopeParams`) so
        # `forward` only gathers `rotary_cos_sin[pos_ids]` -- no
        # per-forward `torch.outer` or `.cos()` / `.sin()` kernels
        # and no hard-coded upper bound on the per-axis image grid.
        text_config = getattr(
            model_config.pretrained_config, "text_config", model_config.pretrained_config
        )
        self.config.max_position_embeddings = text_config.max_position_embeddings
        # Vision RoPE uses half of head_dim (partial_rotary_factor=0.5),
        # so the cos/sin tables hold `head_dim/2` columns -- which
        # matches the per-token (h_pos, w_pos) layout produced by
        # `rot_pos_ids`.
        self.config.partial_rotary_factor = 0.5
        self.config.num_attention_heads = self.config.num_heads
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

        self.blocks = nn.ModuleList(
            [
                Qwen3VLVisionBlock(model_config, layer_idx=layer_idx)
                for layer_idx in range(self.config.depth)
            ]
        )
        self.merger = Qwen3VLVisionPatchMerger(
            model_config=model_config,
            use_postshuffle_norm=False,
        )
        self.deepstack_visual_indexes = self.config.deepstack_visual_indexes
        self.deepstack_merger_list = nn.ModuleList(
            [
                Qwen3VLVisionPatchMerger(
                    model_config=model_config,
                    use_postshuffle_norm=True,
                )
                for _ in range(len(self.deepstack_visual_indexes))
            ]
        )
        # O(1) lookup table: layer_idx -> merger position. Avoids the
        # per-layer `deepstack_visual_indexes.index(layer_num)` linear
        # scan in `forward`. (Not a parameter; just a Python dict.)
        self._deepstack_layer_to_merger_idx = {
            layer_idx: i for i, layer_idx in enumerate(self.deepstack_visual_indexes)
        }
        self.metadata_cls = get_attention_backend(self.model_config.attn_backend).Metadata

        self.attn_metadata: Optional[AttentionMetadata] = None

        # Vision block's `rope_position_ids` scratch. Registered empty here;
        # `setup_attn_metadata` allocates it as an `arange` (see there).
        self.register_buffer("_rope_position_ids_buffer", None, persistent=False)

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    def setup_attn_metadata(self, max_num_requests: int, max_num_tokens: int) -> None:
        # Override the mixin default: each image / video frame is its own
        # attention segment (``seq_lens.extend([h * w] * t)`` in ``forward``),
        # so a single multi-image or video request can produce many more
        # segments than ``max_batch_size``. The number of segments in one
        # encoder forward is bounded by the token budget (every segment holds
        # at least one token), NOT by the request count -- so floor the
        # metadata's request capacity at ``max_num_tokens`` to keep the
        # per-request buffers (prompt_lens / host_request_types / kv_lens) from
        # overflowing when ``num_contexts`` is set to the segment count.
        max_num_requests = max(max_num_requests, max_num_tokens)
        self.attn_metadata = self.metadata_cls(
            max_num_requests=max_num_requests,
            max_num_tokens=max_num_tokens,
            kv_cache_manager=None,
        )
        # Pre-allocate the vision-block ``rope_position_ids`` as an ``arange``
        # sized to the encoder's ``max_num_tokens`` (engine-driven) so per-call
        # code just slices ``[:seq_len]`` instead of allocating a fresh
        # ``(seq_len,) int32`` + H->D copy; ``forward`` still grows it on the
        # rare miss above the budget (e.g. packed multi-video batches).
        self._rope_position_ids_buffer = torch.arange(
            max_num_tokens, dtype=torch.int32, device=self.device
        )

    @staticmethod
    @lru_cache(maxsize=1024)
    def rot_pos_ids(h: int, w: int, spatial_merge_size: int) -> torch.Tensor:
        # CPU-side numpy build; identical (h, w) hits the lru_cache, so the
        # per-image torch.arange/expand/stack chain only runs on first sight.
        hpos_ids = np.broadcast_to(np.arange(h).reshape(h, 1), (h, w))
        h_div = h // spatial_merge_size
        w_div = w // spatial_merge_size
        hpos_ids = hpos_ids.reshape(
            h_div,
            spatial_merge_size,
            w_div,
            spatial_merge_size,
        )
        hpos_ids = hpos_ids.transpose(0, 2, 1, 3).flatten()

        wpos_ids = np.broadcast_to(np.arange(w).reshape(1, w), (h, w))
        wpos_ids = wpos_ids.reshape(
            h_div,
            spatial_merge_size,
            w_div,
            spatial_merge_size,
        )
        wpos_ids = wpos_ids.transpose(0, 2, 1, 3).flatten()

        return torch.from_numpy(np.stack([hpos_ids, wpos_ids], axis=-1))

    @lru_cache(maxsize=1024)  # noqa: B019
    def _rotary_pos_emb_thw(self, t: int, h: int, w: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Per-(t, h, w) rotary (cos, sin) pair, cached as device tensors.

        Gathers from `self.rotary_pos_emb.rotary_cos_sin` (sized to the
        text config's `max_position_embeddings`) -- so no `.cos()` /
        `.sin()` kernels fire in forward. pos_ids are built CPU-side via
        `rot_pos_ids` (also lru_cached), the H->D copy happens here on
        the first miss, and subsequent calls return the cached on-device
        cos/sin tuple. `rot_pos_emb` over a multi-image grid becomes a
        list lookup + per-half `torch.cat` -- no transfer, no elementwise
        trig.

        GPU memory cost (measured on H200, head_dim=72 -> freq_dim=18,
        cos and sin each cached as (t*h*w, 36) after gather+flatten;
        per-token = 2 * 36 floats * 4 bytes = 288 B, fp32):

          ============================  =======  =========
          tile (t, h, w)                tokens   per entry
          ============================  =======  =========
          (1, 16, 16)  -- 224**2          256       72 KB
          (1, 32, 32)  -- 448**2         1024      288 KB
          (1, 48, 48)  -- 672**2         2304      648 KB
          (1, 64, 64)  -- 1024**2        4096     1.13 MB
          (8, 32, 32)  -- 8-frame 448    8192     2.25 MB
          ============================  =======  =========

        Typical production VLM serving has 10-30 unique tile shapes, so
        the cache settles around 4-10 MB. `maxsize=1024` is a safety
        cap; reaching it would require >1024 distinct (t, h, w) and
        even then LRU evicts.
        """
        pos_ids = self.rot_pos_ids(h, w, self.spatial_merge_size)
        if t > 1:
            pos_ids = pos_ids.repeat(t, 1)
        # Pinned-host + async DMA via the project helper; a bare
        # `.to(..., non_blocking=True)` on pageable memory silently
        # degrades to a staging copy.
        pos_ids = async_tensor_h2d(pos_ids, dtype=pos_ids.dtype, device=self.device)
        # Gather pre-computed cos/sin from the standard `RotaryEmbedding`
        # buffer. `rotary_cos_sin` has shape (max_pos, 2, freq_dim);
        # index 0 holds cos, index 1 holds sin. pos_ids has shape
        # (total, 2); the last-dim 2 holds the (h-pos, w-pos) freq
        # indices, so after gather + flatten the per-token layout is
        # `[cos(freq_h), cos(freq_w)]` of size `2*freq_dim`.
        max_grid_size = max(h, w)
        cos_sin = self.rotary_pos_emb.rotary_cos_sin[:max_grid_size]
        cos = cos_sin[:, 0, :][pos_ids].flatten(1)
        sin = cos_sin[:, 1, :][pos_ids].flatten(1)
        return cos, sin

    def rot_pos_emb(self, grid_thw: list[list[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return concatenated (cos, sin) for the whole multi-image batch."""
        cos_pieces: List[torch.Tensor] = []
        sin_pieces: List[torch.Tensor] = []
        for t, h, w in grid_thw:
            c, s = self._rotary_pos_emb_thw(t, h, w)
            cos_pieces.append(c)
            sin_pieces.append(s)
        if len(cos_pieces) == 1:
            return cos_pieces[0], sin_pieces[0]
        return torch.cat(cos_pieces, dim=0), torch.cat(sin_pieces, dim=0)

    def fast_pos_embed_interpolate(self, grid_thw: list[list[int]]) -> torch.Tensor:
        """Per-image fused Triton bilinear pos-embed interpolation.

        Launches `_triton_pos_embed_interpolate` once per image with
        `(h, w, h_scale, w_scale)` passed as scalar kernel args, so the
        function performs zero H<->D transfers. `torch.cat` joins the
        per-image outputs at the end.
        """
        pieces = [
            _triton_pos_embed_interpolate(
                self.pos_embed.weight,
                t,
                h,
                w,
                self.num_grid_per_side,
                self.spatial_merge_size,
                self.pos_embed.weight.dtype,
            )
            for t, h, w in grid_thw
        ]
        if len(pieces) == 1:
            return pieces[0]
        return torch.cat(pieces, dim=0)

    def prepare_attn_metadata(
        self,
        seq_lens: List[int],
        attn_metadata: Optional[AttentionMetadata] = None,
    ):
        if attn_metadata is None:
            raise RuntimeError(
                "Vision encoder AttentionMetadata is not initialized. "
                "It must be set up before the encoder forward runs."
            )
        return _prepare_qwen_vl_vision_attn_metadata(seq_lens, attn_metadata)

    @torch.inference_mode()
    def forward(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        grid_rows = grid_thw.tolist()
        seq_lens: List[int] = []
        for t, h, w in grid_rows:
            seq_lens.extend([h * w] * t)
        self.attn_metadata = self.prepare_attn_metadata(seq_lens, self.attn_metadata)

        # `rot_pos_emb` returns (cos, sin) gathered from the pre-computed
        # cos/sin buffers -- no `.cos()`/`.sin()` kernels in forward.
        # Each half has shape (total_tokens, 2*freq_dim) = (total_tokens,
        # rotary_dim), which is what `Qwen2_5_VLVisionAttention.apply_rope`
        # / `RotaryEmbedding.apply_rotary_pos_emb` expects (it computes
        # `rot_dim = cos.shape[-1] * 2 = head_dim` and chunks q/k into
        # halves of size `cos.shape[-1]`).
        cos, sin = self.rot_pos_emb(grid_rows)
        pos_embeds = self.fast_pos_embed_interpolate(grid_rows)

        hidden_states = self.patch_embed(pixel_values)
        hidden_states += pos_embeds
        hidden_states = hidden_states.flatten(1)

        # Vision RoPE backend (FlashInfer path) gates on `position_ids is
        # not None`; supply trivial 0..seq_len-1 positions on device so
        # the gate clears when `head_dim % 64 == 0`. Keep the pre-allocated
        # buffer large enough for packed multi-video batches.
        seq_len = hidden_states.shape[0]
        if (
            self._rope_position_ids_buffer is None
            or seq_len > self._rope_position_ids_buffer.numel()
        ):
            self._rope_position_ids_buffer = torch.arange(
                seq_len, dtype=torch.int32, device=self.device
            )
        rope_position_ids = self._rope_position_ids_buffer[:seq_len]
        position_embeddings = (cos, sin)

        deepstack_feature_lists = []
        for layer_num, block in enumerate(self.blocks):
            hidden_states = block(
                hidden_states,
                position_ids=rope_position_ids,
                attn_metadata=self.attn_metadata,
                position_embeddings=position_embeddings,
            )
            merger_idx = self._deepstack_layer_to_merger_idx.get(layer_num)
            if merger_idx is not None:
                deepstack_feature = self.deepstack_merger_list[merger_idx](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)
        hidden_states = self.merger(hidden_states)

        return hidden_states, deepstack_feature_lists


class Qwen3VisionModelBase(nn.Module):
    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        model_class: Union[type[PreTrainedModel], type[torch.nn.Module]],
    ):
        super().__init__()
        self.model_config = model_config
        self.model_dtype = self.model_config.pretrained_config.text_config.dtype

        # NOTE: Re-setting QuantConfig to exclude vision encoder from quantization,
        # including KV cache quantization (vision encoder head dims may not be
        # supported by FP8 FMHA kernels).
        self.model_config.quant_config = QuantConfig()

        self.visual = MultimodalModelMixin._cast_multimodal_encoder_dtype(
            model_class(self.model_config), self.model_dtype
        )

        self.post_config()

    def post_config(self):
        self.config = self.model_config.pretrained_config.vision_config

    def load_weights(self, weights: Dict[str, torch.Tensor]):
        visual_weights = filter_weights("model.visual", weights)
        converted_weights = {}

        qkv_pattern = re.compile(r"(.*?)attn\.qkv\.(.*)")
        for name in visual_weights:
            # Handle with weights and bias for vision transformer's qkv projection.
            match = qkv_pattern.match(name)
            if match:
                prefix, suffix = match.groups()
                q_name = f"{prefix}attn.q_proj.{suffix}"
                k_name = f"{prefix}attn.k_proj.{suffix}"
                v_name = f"{prefix}attn.v_proj.{suffix}"
                dim_shape = visual_weights[name].shape[0] // 3
                converted_weights[q_name] = visual_weights[name][:dim_shape]
                converted_weights[k_name] = visual_weights[name][dim_shape : 2 * dim_shape]
                converted_weights[v_name] = visual_weights[name][2 * dim_shape :]
            else:
                converted_weights[name] = visual_weights[name]
        pattern_mapping = {
            r"(.*?)attn.proj.(.*)": r"\1attn.o_proj.\2",
            r"(.*?)mlp.linear_fc1.(.*)": r"\1mlp.up_proj.\2",
            r"(.*?)mlp.linear_fc2.(.*)": r"\1mlp.down_proj.\2",
        }
        self.visual.config.num_attention_heads = self.visual.config.num_heads
        _load_weights_impl(self.visual, converted_weights, params_map=pattern_mapping)

    def _parse_and_batch_multimodal_data(
        self, multimodal_params: List[MultimodalParams]
    ) -> Tuple[Dict[str, Any], Dict[str, List[Any]]]:
        pixel_values_list = []
        pixel_values_videos_list = []
        image_grid_thw_list = []
        video_grid_thw_list = []

        for multimodal_param in multimodal_params:
            multimodal_data = multimodal_param.multimodal_data
            # Process images if present
            if multimodal_data.get("image") is not None:
                pixel_values_list.append(multimodal_data["image"]["pixel_values"])
                image_grid_thw_list.append(multimodal_data["image"]["image_grid_thw"])

            # Process videos if present
            if multimodal_data.get("video") is not None:
                pixel_values_videos_list.append(multimodal_data["video"]["pixel_values_videos"])
                video_grid_thw_list.append(multimodal_data["video"]["video_grid_thw"])

        # Concatenate tensors
        mm_content_dict = {}
        if pixel_values_list:
            mm_content_dict["pixel_values"] = (
                torch.cat(pixel_values_list, dim=0)
                if len(pixel_values_list) > 1
                else pixel_values_list[0]
            )
        if pixel_values_videos_list:
            mm_content_dict["pixel_values_videos"] = (
                torch.cat(pixel_values_videos_list, dim=0)
                if len(pixel_values_videos_list) > 1
                else pixel_values_videos_list[0]
            )

        # Prepare extra data
        mm_extra_data = {}
        if image_grid_thw_list:
            mm_extra_data["image_grid_thw"] = (
                torch.cat(image_grid_thw_list, dim=0)
                if len(image_grid_thw_list) > 1
                else image_grid_thw_list[0]
            )
        if video_grid_thw_list:
            mm_extra_data["video_grid_thw"] = (
                torch.cat(video_grid_thw_list, dim=0)
                if len(video_grid_thw_list) > 1
                else video_grid_thw_list[0]
            )

        return mm_content_dict, mm_extra_data

    @torch.inference_mode()
    def forward(self, multimodal_params: List[MultimodalParams]) -> List[torch.Tensor]:
        mm_content_data, mm_extra_data = self._parse_and_batch_multimodal_data(multimodal_params)
        pixel_values = mm_content_data.get("pixel_values", None)
        pixel_values_videos = mm_content_data.get("pixel_values_videos", None)

        if pixel_values is not None and pixel_values_videos is not None:
            raise ValueError("Currently only support single modality per request")

        image_grid_thw = mm_extra_data.get("image_grid_thw", None)
        video_grid_thw = mm_extra_data.get("video_grid_thw", None)

        embeds = []
        if pixel_values is not None:
            pixel_values = pixel_values.to(self.model_dtype)
            image_embeds, deepstack_image_embeds = self.visual(
                pixel_values, grid_thw=image_grid_thw
            )
            # NOTE: We concatenate deepstack_embeds to mm_embeds
            # The shape will be [seq_len, hidden_dim * (num_deepstack_layers + 1)]
            mixed_image_embeds = torch.cat([image_embeds] + deepstack_image_embeds, dim=1)
            embeds.append(mixed_image_embeds)

        if pixel_values_videos is not None:
            pixel_values_videos = pixel_values_videos.to(self.model_dtype)
            video_embeds, deepstack_video_embeds = self.visual(
                pixel_values_videos, grid_thw=video_grid_thw
            )
            # NOTE: We concatenate deepstack_embeds to mm_embeds
            # The shape will be [seq_len, hidden_dim * (num_deepstack_layers + 1)]
            mixed_video_embeds = torch.cat([video_embeds] + deepstack_video_embeds, dim=1)
            embeds.append(mixed_video_embeds)
        return embeds


class Qwen3VLModelBase(PreTrainedModel, MultimodalModelMixin):
    def encode_multimodal_inputs(
        self, multimodal_params: List[MultimodalParams], **encoder_kwargs: Any
    ) -> torch.Tensor:
        """Uniform encoder entry (``MultimodalModelMixin`` contract).

        Runs the vision encoder over ``multimodal_params`` and returns the
        embeddings as a single tensor (Qwen3-VL folds deepstack streams into
        the hidden dim, so the single-tensor contract holds). Used by the
        startup memory profiler to invoke the encoder directly; the model's
        own ``forward`` keeps its custom deepstack fusion path.
        """
        mm_embeds = get_multimodal_embeddings(
            encoder_forward_fn=self.mm_encoder.forward, multimodal_params=list(multimodal_params)
        )
        return mm_embeds[0]

    def _check_and_adjust_experts_implementation(self, *args, **kwargs):
        """No-op override.

        Transformers 5.x's ``PreTrainedModel.__init__`` calls this method
        (with an ``experts_implementation`` argument) which fails for VL
        wrapper models that do not directly contain MoE layers.  TRT-LLM
        manages expert implementations independently, so skip the check.
        """
        return None

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        *args,
        **kwargs,
    ) -> None:
        self.original_arch = model_config.pretrained_config.architectures[0]

        disable_fuse_rope = kwargs.get("disable_fuse_rope", False)
        model_config.pretrained_config.disable_fuse_rope = disable_fuse_rope
        model_config.pretrained_config.text_config.disable_fuse_rope = disable_fuse_rope
        # transformers>=5.5 flipped the ``Qwen3VL[Moe]TextConfig`` class-level
        # default for ``tie_word_embeddings`` from inherited ``False`` to
        # ``True``. Real checkpoints set it only on the top-level config, so
        # the nested ``text_config`` silently picks up the new default and
        # ties lm_head to embed_tokens, producing systematic logits drift.
        # Mirror the top-level value onto ``text_config`` to stay aligned.
        model_config.pretrained_config.text_config.tie_word_embeddings = (
            model_config.pretrained_config.tie_word_embeddings
        )
        # In transformers 5.x, rope_scaling may delegate to rope_parameters which
        # can be None.  Ensure the dict exists before setting the type key.
        if model_config.pretrained_config.text_config.rope_scaling is None:
            model_config.pretrained_config.text_config.rope_scaling = {}
        model_config.pretrained_config.text_config.rope_scaling["type"] = "mrope"
        config = model_config.pretrained_config

        self._supports_sdpa = True
        self._supports_flash_attn = True
        super().__init__(config)
        if not disable_fuse_rope:
            self.init_mrope_embedding(model_config)
            # Extra slot is reserved for CUDA graph / warmup dummy requests.
            max_mrope_delta_slots = model_config.max_num_tokens * model_config.mapping.pp_size + 1
            self.register_buffer(
                "mrope_position_deltas_cache",
                torch.zeros(
                    max_mrope_delta_slots,
                    dtype=torch.int32,
                    device="cuda",
                ),
                persistent=False,
            )
            rotary_dim = self.rotary_emb.rotary_cos_sin.shape[-1]
            self.register_buffer(
                "mrope_rotary_cos_sin_workspace",
                torch.empty(
                    (1, model_config.max_num_tokens * rotary_dim * 2),
                    dtype=torch.float32,
                    device="cuda",
                ),
                persistent=False,
            )

        self.model_config = model_config

        vlm_to_llm_arch = {
            "Qwen3VLForConditionalGeneration": "Qwen3ForCausalLM",
            "Qwen3VLMoeForConditionalGeneration": "Qwen3MoeForCausalLM",
            "QwenImageBenchForConditionalGeneration": "Qwen3_5ForCausalLM",
            "Cosmos3ForConditionalGeneration": "Qwen3ForCausalLM",
            "Qwen3_5MoeForConditionalGeneration": "Qwen3_5MoeForCausalLM",
            "Qwen3_5ForConditionalGeneration": "Qwen3_5ForCausalLM",
        }
        llm_arch = vlm_to_llm_arch.get(self.original_arch)
        if llm_arch is None:
            raise ValueError(f"Unsupported architecture: {self.original_arch}")
        llm_model_config = copy.deepcopy(model_config)
        llm_model_config.pretrained_config = config.text_config
        # The LM attention modules use extra_attrs through the outer wrapper.
        # Share the dict after deepcopy so compiled LM attention lookups see
        # the same per-layer metadata as model_engine.model_forward.
        # Vision attention unregisters itself from this dict during init,
        # so it does not pollute LM lookups.
        llm_model_config.extra_attrs = model_config.extra_attrs
        llm_model_config.pretrained_config.architectures = [llm_arch]
        # Qwen3ForCausalLM.
        self.llm = AutoModelForCausalLM.from_config(llm_model_config)

        self.mm_encoder = None
        # Normal workers own the encoder. MM E/P handoff uses attached embeddings.
        if not _is_mm_disagg():
            self.mm_encoder = Qwen3VisionModelBase(
                copy.deepcopy(model_config), kwargs.get("vision_model_class", None)
            ).eval()

        self.use_deepstack = hasattr(config.vision_config, "deepstack_visual_indexes")
        self.deepstack_num_level = (
            len(config.vision_config.deepstack_visual_indexes) if self.use_deepstack else 0
        )
        if self.use_deepstack:
            # Pre-allocated `(L, max_num_tokens, hidden)` scratch buffer for
            # per-layer deepstack embeddings; replaces `L` fresh
            # `torch.zeros` + `L` scatters per prefill.
            # `persistent=False` keeps it out of `state_dict`.
            self.register_buffer(
                "deepstack_input_embeds",
                torch.zeros(
                    self.deepstack_num_level,
                    model_config.max_num_tokens,
                    config.text_config.hidden_size,
                    device="cuda",
                    dtype=config.text_config.torch_dtype,
                ),
                persistent=False,
            )

        # Surface the in-vocab image / video placeholder IDs to the model
        # engine's ``_prepare_multimodal_indices`` so it selects the
        # ``torch.isin`` predicate.
        _mm_ids = [
            tid
            for tid in (
                getattr(config, "image_token_id", None),
                getattr(config, "video_token_id", None),
            )
            if tid is not None
        ]
        self._mm_token_ids = torch.tensor(_mm_ids, dtype=torch.int32)
        self.post_config()

    @property
    def mm_token_ids(self) -> torch.Tensor:
        return self._mm_token_ids

    def post_config(self):
        # use llm.config as config for pytorch model engine
        self.model_config.pretrained_config = self.llm.config
        self.config = self.model_config.pretrained_config

    @property
    def vocab_size_padded(self) -> int:
        return self.llm.vocab_size_padded

    def infer_max_seq_len(self) -> int:
        return self.llm.infer_max_seq_len()

    # Draft-model (two-model speculative decoding, e.g. DFlash / Eagle3)
    # delegation: `ModelLoader.load` reads `draft_config` / `draft_model` and
    # calls `load_draft_weights` on the *outer* model it resolved, but the
    # spec-decoding wrapper (`SpecDecOneEngineForCausalLM`) is applied to the
    # inner `self.llm` when this VLM composes it. Composite checkpoints
    # (e.g. Qwen3.5-4B publishes text_config + vision_config) route text-only
    # spec tests through this wrapper, so surface the inner LM's draft state.
    # Note: `load_draft_weights` must keep an explicit signature — the loader
    # dispatches kwargs via `inspect.getfullargspec`.
    @property
    def draft_config(self):
        return self.llm.draft_config

    @property
    def draft_model(self):
        return self.llm.draft_model

    def load_draft_weights(self, weights: Dict, weight_mapper: Optional[BaseWeightMapper] = None):
        return self.llm.load_draft_weights(weights, weight_mapper=weight_mapper)

    def apply_llm_torch_compile(self, *, backend: Any, fullgraph: bool) -> None:
        # TODO: Move this hook to MultimodalModelMixin once multimodal models
        # consistently expose an LLM compile contract.
        """Compile only the LLM decoder; the vision encoder stays eager."""
        self.llm.model = torch.compile(self.llm.model, backend=backend, fullgraph=fullgraph)

    def init_mrope_embedding(self, model_config: ModelConfig[PretrainedConfig]):
        config = model_config.pretrained_config.text_config
        pos_embd_params = PositionalEmbeddingParams(
            type=PositionEmbeddingType.from_string(config.rope_scaling["type"]),
            rope=RopeParams.from_config(config),
            mrope_section=config.rope_scaling.get("mrope_section", None),
            mrope_interleaved=config.rope_scaling.get("mrope_interleaved", False),
        )
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

    def prepare_mrope_config(
        self,
        multimodal_params: List[MultimodalParams],
        num_generation_requests: int,
        position_ids: torch.Tensor,
        mrope_delta_write_seq_slots: Optional[torch.Tensor] = None,
        mrope_delta_read_seq_slots: Optional[torch.Tensor] = None,
    ):
        return _prepare_qwen_vl_mrope_config(
            multimodal_params=multimodal_params,
            num_generation_requests=num_generation_requests,
            position_ids=position_ids,
            rotary_emb=self.rotary_emb,
            mrope_position_deltas_cache=self.mrope_position_deltas_cache,
            mrope_rotary_cos_sin_workspace=self.mrope_rotary_cos_sin_workspace,
            mrope_delta_write_seq_slots=mrope_delta_write_seq_slots,
            mrope_delta_read_seq_slots=mrope_delta_read_seq_slots,
        )

    def split_mm_embeds(self, mm_embed, deepstack_num_level):
        num_elements = mm_embed.shape[1] // (deepstack_num_level + 1)
        mm_embed_chunks = torch.split(mm_embed, [num_elements] * (deepstack_num_level + 1), dim=1)
        return mm_embed_chunks[0], list(mm_embed_chunks[1:])

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
            attn_metadata.num_contexts,
            attn_metadata.num_generations,
        )

        multimodal_params = kwargs.get("multimodal_params", [])
        mm_embeds = []
        mrope_config = {}
        deepstack_embeds = []

        # NOTE: Qwen*-VL series has mrope_config even on the text-only prompts,
        # so we need to separate the mm_multimodal_params from the text-only prompts.
        if num_context_requests > 0:
            mm_multimodal_params, has_raw_image_or_video_data = self._get_requests_with_mm_data(
                multimodal_params[:num_context_requests]
            )
        else:
            mm_multimodal_params = []
            has_raw_image_or_video_data = False
        if len(mm_multimodal_params) > 0:
            # Raw image/video tensors: run local encoder.
            if has_raw_image_or_video_data and self.mm_encoder is not None:
                mm_embeds = get_multimodal_embeddings(
                    encoder_forward_fn=self.mm_encoder.forward,
                    multimodal_params=mm_multimodal_params,
                )
            # Raw image/video tensors on a worker with no encoder: bad route.
            elif has_raw_image_or_video_data:
                raise ValueError(
                    "Raw multimodal inputs require a local multimodal encoder on this "
                    "worker, or multimodal_embedding handles from an encoder handoff."
                )
            # support_mm_disagg is only set in subclasses of Qwen3VLModelBase that support EPD
            elif not getattr(self, "support_mm_disagg", False):
                raise NotImplementedError(
                    f"{type(self)} does not support disaggregated inference yet. Please unset "
                    "the TLLM_MULTIMODAL_DISAGGREGATED environment variable, or set it to '0'."
                )
            # E/P prefill: encoder already ran; use attached embeddings.
            else:
                mm_embeds = get_attached_multimodal_embeddings(mm_multimodal_params)
            mm_embeds = find_input_mm_embeds(mm_embeds, mm_multimodal_params)

            if self.use_deepstack:
                for i, mm_embed in enumerate(mm_embeds):
                    mm_embed, deepstack_embed = self.split_mm_embeds(
                        mm_embed, self.deepstack_num_level
                    )
                    mm_embeds[i] = mm_embed
                    deepstack_embeds.extend(deepstack_embed)

        if not self.model_config.pretrained_config.disable_fuse_rope:
            mrope_config = self.prepare_mrope_config(
                multimodal_params,
                num_generation_requests,
                position_ids,
                mrope_delta_write_seq_slots=kwargs.get("mrope_delta_write_seq_slots"),
                mrope_delta_read_seq_slots=kwargs.get("mrope_delta_read_seq_slots"),
            )

        # Prefer the indices the executor already computed (CPU-side
        # `filter_mm_token_from_input_ids` + async H2D) and forwarded via
        # kwargs; fall back to filtering only on engine-bypass paths
        # (e.g., direct `forward` calls in unit tests).
        text_token_indices = kwargs.get("text_token_indices")
        mm_token_indices = kwargs.get("mm_token_indices")
        if len(mm_embeds) > 0 and (text_token_indices is None or mm_token_indices is None):
            text_token_indices, mm_token_indices = filter_mm_token_from_input_ids(
                input_ids,
                vocab_size=self.llm.model.embed_tokens.num_embeddings,
                mm_token_ids=self.mm_token_ids,
            )

        # Expand the per-level deepstack mm embeddings into the pre-allocated
        # `(L, max_num_tokens, H)` buffer with a single packed scatter,
        # avoiding `L` fresh `torch.zeros` + `L` scatters inside
        # `fuse_input_embeds`.
        if self.use_deepstack and len(deepstack_embeds) > 0:
            num_tokens = input_ids.shape[0]
            deepstack_buffer = self.deepstack_input_embeds[:, :num_tokens, :]
            deepstack_buffer.zero_()
            packed_deepstack = torch.stack(deepstack_embeds, dim=0)
            deepstack_buffer[:, mm_token_indices, :] = packed_deepstack.to(
                dtype=deepstack_buffer.dtype, device=deepstack_buffer.device
            )
            deepstack_embeds = list(deepstack_buffer.unbind(0))

        # Preserve the pre-fusion token IDs. `fuse_input_embeds` collapses
        # input_ids -> None when MM embeddings are fused in, but spec
        # decoding (MTP / Eagle) still needs the original prompt token
        # IDs for drafter context preparation; pass them through as a
        # dedicated kwarg consumed by `SpecDecOneEngineForCausalLM.forward`.
        orig_input_ids = input_ids

        input_ids, input_embeds = fuse_input_embeds(
            self.llm.model.embed_tokens,
            input_ids,
            mm_embeds,
            text_token_indices=text_token_indices,
            mm_token_indices=mm_token_indices,
        )

        output_prob = self.llm.forward(
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            position_ids=position_ids,
            inputs_embeds=input_embeds,
            return_context_logits=return_context_logits,
            deepstack_embeds=deepstack_embeds,
            mrope_config=mrope_config,
            spec_metadata=kwargs.get("spec_metadata"),
            resource_manager=kwargs.get("resource_manager"),
            orig_input_ids=orig_input_ids,
        )
        # Spec-decoding (MTP / Eagle) returns a dict (accepted tokens,
        # draft tokens, logits); plain forward returns a tensor.
        if hasattr(output_prob, "shape"):
            logger.debug(f"output shape: {output_prob.shape}")
        return output_prob

    def _get_requests_with_mm_data(self, multimodal_params):
        mm_multimodal_params = []
        # TODO: This returns one batch-wide "has raw pixels/video" flag. That is
        # safe only when a batch is all raw-MM or all attached embeddings. If a
        # scheduler can mix both, split raw requests from attached-embedding
        # requests and merge outputs back by request index.
        has_raw_image_or_video_data = False
        for multimodal_param in multimodal_params:
            data = multimodal_param.multimodal_data
            has_raw_data = (
                data.get("image", {}).get("pixel_values") is not None
                or data.get("video", {}).get("pixel_values_videos") is not None
            )
            has_raw_image_or_video_data |= has_raw_data
            if has_raw_data or data.get("multimodal_embedding") is not None:
                mm_multimodal_params.append(multimodal_param)

        return mm_multimodal_params, has_raw_image_or_video_data


@support_multimodal_disaggregated
@register_vision_encoder(Qwen3VisionModelBase, vlm_base_model=Qwen3VisionModel)
@register_auto_model("Qwen3VLForConditionalGeneration")
@register_input_processor(
    Qwen3VLInputProcessorBase,
    model_type="qwen3_vl",
    placeholder_metadata=MultimodalPlaceholderMetadata(
        placeholder_map={
            "image": "<|vision_start|><|image_pad|><|vision_end|>",
            "video": "<|vision_start|><|video_pad|><|vision_end|>",
        },
        placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
        placeholders_separator="",
        content_format=ContentFormat.STRING,
    ),
)
class Qwen3VLModel(Qwen3VLModelBase):
    def __init__(self, model_config: ModelConfig[PretrainedConfig], *args, **kwargs):
        # NOTE: HF implementation.
        kwargs["vision_model_class"] = Qwen3VisionModel
        kwargs["disable_fuse_rope"] = kwargs.get(
            "disable_fuse_rope", False
        )  # TODO: Make this ModelConfig's argument
        super().__init__(model_config, *args, **kwargs)

    @property
    def multimodal_data_device_paths(self) -> List[str]:
        return [
            "image.pixel_values",
            "video.pixel_values_videos",
            "multimodal_embedding",
            "mrope_config.mrope_position_ids",
            "mrope_config.mrope_position_deltas",
        ]

    def load_weights(self, weights: Dict[str, torch.Tensor], weight_mapper: BaseWeightMapper):
        if self.mm_encoder is not None:
            self.mm_encoder.load_weights(weights)

        weight_mapper = Qwen3VLHfWeightMapper()
        weight_mapper.init_model_and_config(self.llm, self.model_config)
        filtered_weights = {k: v for k, v in weights.items() if not k.startswith("model.visual.")}
        params_map = {
            r"^model\.language_model\.(.*)$": r"model.\1",
        }
        self.llm.load_weights(filtered_weights, weight_mapper, params_map=params_map)
