# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared Pixtral vision tower and image-row splicing for Mistral Large 4 targets."""

from __future__ import annotations

from types import SimpleNamespace
from typing import NamedTuple, Optional

import torch
import torch.nn as nn

from tensorrt_llm._torch.modeling_v2 import step_contract_enabled
from tensorrt_llm._torch.modeling_v2.catalog.activation.flashinfer_silu_and_mul import (
    flashinfer_silu_and_mul,
)
from tensorrt_llm._torch.modeling_v2.catalog.attention.thop_attention import thop_attention
from tensorrt_llm._torch.modeling_v2.catalog.gemm.cublas_mm import cublas_mm
from tensorrt_llm._torch.modeling_v2.catalog.norm.flashinfer_fused_add_rmsnorm import (
    flashinfer_fused_add_rmsnorm,
)
from tensorrt_llm._torch.modeling_v2.catalog.norm.flashinfer_rmsnorm import flashinfer_rmsnorm
from tensorrt_llm._torch.modeling_v2.models.mistral_large4.ops.activation.gelu_erf import gelu_erf
from tensorrt_llm._torch.modeling_v2.models.mistral_large4.ops.vision.pixtral_rope_2d import (
    pixtral_rope_2d,
)
from tensorrt_llm._torch.models.modeling_multimodal_utils import (
    find_input_mm_embeds,
    get_multimodal_embeddings,
)
from tensorrt_llm.inputs.multimodal import MultimodalParams
from tensorrt_llm.logger import logger

from .weight_table import checkpoint_dims

#: eps 1e-5, the reference's value; params.json's `vision_encoder` states none.
_VISION_EPS = 1e-5

#: The thop attention arguments the vision tower's call holds fixed: the no-KV-cache
#: mode (an encoder: the three cache-addressing arguments None), one context sequence
#: per image, the fused `[q | k | v]` rows already rotated (no rope in the op),
#: bidirectional within each sequence (mask_type 0), softmax scale head_dim^-0.5.
_VISION_ATTN = dict(
    k=None,
    v=None,
    output_sf=None,
    max_context_q_len_override=None,
    kv_cache_block_offsets=None,
    host_kv_cache_pool_pointers=None,
    host_kv_cache_pool_mapping=None,
    cache_indirection=None,
    kv_scale_orig_quant=None,
    kv_scale_quant_orig=None,
    out_scale=None,
    rotary_inv_freq=None,
    rotary_cos_sin=None,
    latent_cache=None,
    q_pe=None,
    block_ids_per_seq=None,
    attention_sinks=None,
    is_fused_qkv=True,
    update_kv_cache=True,
    predicted_tokens_per_seq=1,
    local_layer_idx=0,
    tokens_per_block=None,
    beam_width=1,
    mask_type=0,
    quant_mode=0,
    q_scaling=1.0,
    position_embedding_type=0,
    rope_dim=0,
    rope_base=10000.0,
    rope_scale_type=0,
    rope_scale=1.0,
    rope_short_m_scale=1.0,
    rope_long_m_scale=1.0,
    rope_max_positions=1024,
    rope_original_max_positions=1024,
    use_paged_context_fmha=False,
    attention_input_type=0,
    is_mla_enable=False,
    chunked_prefill_buffer_batch_size=1,
    q_lora_rank=None,
    kv_lora_rank=None,
    qk_nope_head_dim=None,
    qk_rope_head_dim=None,
    v_head_dim=None,
    rope_append=None,
    mrope_rotary_cos_sin=None,
    mrope_position_deltas=None,
    helix_position_offsets=None,
    helix_is_inactive_rank=None,
    attention_chunk_size=None,
    softmax_stats_tensor=None,
    is_spec_decoding_enabled=False,
    use_spec_decoding=False,
    is_spec_dec_tree=False,
    spec_decoding_generation_lengths=None,
    spec_decoding_position_offsets=None,
    spec_decoding_packed_mask=None,
    spec_decoding_bl_tree_mask_offset=None,
    spec_decoding_bl_tree_mask=None,
    spec_bl_tree_first_sparse_mask_offset_kv=None,
    sparse_kv_indices=None,
    sparse_kv_offsets=None,
    sparse_attn_indices=None,
    sparse_attn_offsets=None,
    sparse_attn_indices_block_size=0,
)


class _VisionBlock(NamedTuple):
    """One vision transformer block's weights, as the views the ops read: the GEMM
    operands column-major (`.t()` of the stored rows)."""

    attn_norm: torch.Tensor
    qkv: torch.Tensor
    wo: torch.Tensor
    ffn_norm: torch.Tensor
    w13: torch.Tensor
    w2: torch.Tensor


def _vision_step(seg_lens: list[int], device, workspace: torch.Tensor) -> dict:
    """The batch state of one vision attention call: one context sequence per image,
    no KV cache. Every value is host-known from the images' patch counts."""
    total = sum(seg_lens)
    longest = max(seg_lens)
    lens_dev = torch.tensor(seg_lens, dtype=torch.int32, device=device)
    lens_host = torch.tensor(seg_lens, dtype=torch.int32)
    return dict(
        workspace=workspace,
        sequence_length=lens_dev,
        host_past_key_value_lengths=lens_host,
        host_total_kv_lens=torch.tensor([total, 0], dtype=torch.int32),
        context_lengths=lens_dev,
        host_context_lengths=lens_host,
        host_request_types=torch.zeros(len(seg_lens), dtype=torch.int32),
        max_num_requests=len(seg_lens),
        max_context_length=longest,
        max_seq_len=longest,
        attention_window_size=longest,
        num_contexts=len(seg_lens),
        num_ctx_tokens=total,
    )


def _vision_attention(
    x: torch.Tensor,
    B: _VisionBlock,
    positions: torch.Tensor,
    step: dict,
    heads: int,
    head_dim: int,
    theta: float,
) -> torch.Tensor:
    """One vision block's attention over the patch rows of every image of the call:
    `[wq ; wk ; wv]` as one GEMM, q and k rotated by each patch's (row, col) in its
    own image, attention bidirectional within each image, then `wo`."""
    rows = x.shape[0]
    width = heads * head_dim
    qkv = cublas_mm(x, B.qkv)
    q, k = pixtral_rope_2d(
        torch.reshape(qkv[:, :width], [rows, heads, head_dim]),
        torch.reshape(qkv[:, width : 2 * width], [rows, heads, head_dim]),
        positions,
        theta,
    )
    qkv = torch.cat(
        [torch.reshape(q, [rows, width]), torch.reshape(k, [rows, width]), qkv[:, 2 * width :]], -1
    )
    out = torch.empty([rows, width], dtype=x.dtype, device=x.device)
    out = thop_attention(
        q=qkv,
        output=out,
        num_heads=heads,
        num_kv_heads=heads,
        head_size=head_dim,
        **step,
        **_VISION_ATTN,
    )
    return cublas_mm(out, B.wo)


def _merge_tiles(x: torch.Tensor, grid_h: int, grid_w: int, merge: int) -> torch.Tensor:
    """One image's `[grid_h * grid_w, d]` patch rows -> `[grid_h * grid_w / merge^2,
    d * merge^2]`: each merge x merge tile of the grid as one row, tiles row-major (the
    order of the image's [IMG] tokens), each row laid out (channel, tile row, tile
    column) as the merging layer's input columns are."""
    d = x.shape[-1]
    grid = torch.permute(torch.reshape(x, [grid_h, grid_w, d]), [2, 0, 1])
    tiles = nn.functional.unfold(torch.unsqueeze(grid, 0), kernel_size=merge, stride=merge)
    return torch.t(tiles[0])


class VisionTower:
    """The image input path: a request's pixels -> one row per [IMG] token, in the
    language model's hidden width. Run by the shell, in context steps only, never
    captured; every rank runs the whole tower (all heads) over its own requests.

    Per image: the 14x14 patch embedding as an unfold and one GEMM, `ln_pre`, the
    transformer blocks (RMSNorm, attention with 2-D RoPE, RMSNorm, SwiGLU, each with
    its residual), the pre-projector RMSNorm, the 2x2 patch merger and the adapter
    (`w_in`, exact GELU, `w_out`). The images of one call are concatenated along the
    rows; only the attention needs their boundaries."""

    def __init__(self, core) -> None:
        """Take the tower's weight views and geometry, once, off the real weights."""
        d = SimpleNamespace(**checkpoint_dims(core))
        w = core.w
        self.hidden = d.v_hidden
        self.heads = d.v_heads
        self.head_dim = d.v_head_dim
        self.patch = d.v_patch
        self.merge = d.v_merge
        self.theta = d.v_theta
        self.dtype = d.dtype
        self.out_hidden = d.hidden
        self._validate = step_contract_enabled()
        self._rank = core.model_config.mapping.rank
        self._target_name = core.__class__.__module__.rsplit(".", 1)[0]
        self.patch_w = w["vis_patch"].t()
        self.ln_pre = w["vis_ln_pre"]
        self.blocks = [
            _VisionBlock(
                attn_norm=w[f"v{j}_attn_norm"],
                qkv=w[f"v{j}_qkv"].t(),
                wo=w[f"v{j}_wo"].t(),
                ffn_norm=w[f"v{j}_ffn_norm"],
                w13=w[f"v{j}_w13"].t(),
                w2=w[f"v{j}_w2"].t(),
            )
            for j in range(d.v_layers)
        ]
        self.proj_norm = w["vis_proj_norm"]
        self.merge_w = w["vis_merge"].t()
        self.w_in = w["vis_w_in"].t()
        self.w_out = w["vis_w_out"].t()
        # The prompt token the tower's rows replace, on the host: the executor finds a
        # step's image positions with it before anything reaches the device.
        self.image_token_ids = torch.tensor([d.image_token_id], dtype=torch.int32)
        # The attention op sizes its workspace in place per call and never shrinks it.
        self._workspace = torch.empty(0, dtype=torch.int8, device=self.ln_pre.device)

    def _images(self, multimodal_params: list[MultimodalParams]) -> list[tuple]:
        """`(pixels [C, H, W], grid rows, grid cols)` of every image, in order. The
        mistral-native processor pads the batch and preserves each image's true size."""
        images = []
        unit = self.patch * self.merge
        for param in multimodal_params:
            image = (param.multimodal_data or {}).get("image")
            if image is None:
                continue
            pixels = image["pixel_values"]
            sizes = image.get("image_sizes")
            if pixels.ndim != 4 or pixels.shape[1] != 3:
                raise ValueError("Image pixels must have shape [N, 3, H, W]")
            if sizes is None or len(sizes) != pixels.shape[0]:
                raise ValueError("Image sizes must describe every image in the padded batch")
            for b in range(pixels.shape[0]):
                if len(sizes[b]) != 2:
                    raise ValueError("Each image size must be [height, width]")
                height, width = int(sizes[b][0]), int(sizes[b][1])
                if not (0 < height <= pixels.shape[-2] and 0 < width <= pixels.shape[-1]):
                    raise ValueError("Image sizes must fit within their padded pixels")
                if height % unit or width % unit:
                    raise ValueError(
                        f"image {height}x{width} is not a whole number of {unit}-pixel "
                        "merge tiles; the processor resizes to them"
                    )
                images.append(
                    (pixels[b, :, :height, :width], height // self.patch, width // self.patch)
                )
        return images

    def forward(self, multimodal_params: list[MultimodalParams]) -> torch.Tensor:
        """The rows of every image of `multimodal_params`, request by request, image by
        image: `[sum of merged tiles, text hidden]`."""
        images = self._images(multimodal_params)
        dev = self.ln_pre.device
        if not images:
            return torch.empty([0, self.out_hidden], dtype=self.dtype, device=dev)
        seg = [gh * gw for _, gh, gw in images]
        if self._validate:
            logger.info(
                "modeling_v2 image forward target=%s rank=%d grids=%s merged_rows=%d",
                self._target_name,
                self._rank,
                [(gh, gw) for _, gh, gw in images],
                sum(seg) // self.merge**2,
            )
        # Each patch as one row, (channel, row, col) within it, patches row-major.
        patches = torch.cat(
            [
                torch.t(
                    nn.functional.unfold(
                        torch.unsqueeze(px.to(device=dev, dtype=self.dtype), 0),
                        kernel_size=self.patch,
                        stride=self.patch,
                    )[0]
                )
                for px, _, _ in images
            ],
            0,
        )
        positions = torch.cat(
            [
                torch.reshape(
                    torch.stack(
                        torch.meshgrid(
                            torch.arange(gh, device=dev),
                            torch.arange(gw, device=dev),
                            indexing="ij",
                        ),
                        -1,
                    ),
                    [-1, 2],
                )
                for _, gh, gw in images
            ],
            0,
        )
        step = _vision_step(seg, dev, self._workspace)

        h = flashinfer_rmsnorm(cublas_mm(patches, self.patch_w), self.ln_pre, _VISION_EPS)
        x = flashinfer_rmsnorm(h, self.blocks[0].attn_norm, _VISION_EPS)
        residual = h
        for j, B in enumerate(self.blocks):
            attn = _vision_attention(x, B, positions, step, self.heads, self.head_dim, self.theta)
            x, residual = flashinfer_fused_add_rmsnorm(attn, residual, B.ffn_norm, _VISION_EPS)
            mlp = cublas_mm(flashinfer_silu_and_mul(cublas_mm(x, B.w13)), B.w2)
            norm_next = self.blocks[j + 1].attn_norm if j + 1 < len(self.blocks) else self.proj_norm
            x, residual = flashinfer_fused_add_rmsnorm(mlp, residual, norm_next, _VISION_EPS)

        # x is the pre-projector-normed tower output.
        tiles = torch.cat(
            [
                _merge_tiles(xi, gh, gw, self.merge)
                for xi, (_, gh, gw) in zip(torch.split(x, seg, 0), images)
            ],
            0,
        )
        merged = cublas_mm(tiles, self.merge_w)
        return cublas_mm(gelu_erf(cublas_mm(merged, self.w_in)), self.w_out)


def _image_rows(
    vision: Optional[VisionTower], multimodal_params: list[MultimodalParams]
) -> Optional[torch.Tensor]:
    """This step's image rows, one per [IMG] position its context chunks cover, in
    the order those positions appear; None when the step covers none.

    The tower runs once per request, on the first step that needs it; its rows stay
    on the request's multimodal data for the later chunks. A chunk -- under chunked
    prefill, or resuming after a reused prefix -- takes rows `[cached, cached + in
    chunk)` of its request's, counted by the request's [IMG] cumsum."""
    if vision is None:
        if any((p.multimodal_data or {}).get("image") is not None for p in multimodal_params):
            raise ValueError(
                "an image reached a Mistral Large 4 target whose checkpoint has no vision encoder"
            )
        return None
    # Filter before encoding and gathering so inactive requests neither run the
    # tower nor shift the embedding offsets of active requests in a mixed batch.
    multimodal_params = [
        param
        for param in multimodal_params
        if param.multimodal_runtime is None or param.multimodal_runtime.num_mm_tokens_in_chunk > 0
    ]
    embeds = find_input_mm_embeds(
        get_multimodal_embeddings(vision.forward, multimodal_params), multimodal_params
    )
    if not embeds:
        return None
    rows = embeds[0] if len(embeds) == 1 else torch.cat(embeds, 0)
    return rows if rows.shape[0] else None


def _splice_image_rows(
    embed: torch.Tensor,
    input_ids: torch.Tensor,
    rows: torch.Tensor,
    mm_token_indices: Optional[torch.Tensor],
) -> torch.Tensor:
    """The step's input embeddings with `rows` written over its [IMG] positions,
    `mm_token_indices` (which the executor computes on the host from the shell's
    `multimodal_token_ids`); every other position keeps its token embedding."""
    if mm_token_indices is None:
        raise ValueError("an image step arrived without the executor's mm_token_indices")
    if mm_token_indices.shape[0] != rows.shape[0]:
        raise ValueError(
            f"{mm_token_indices.shape[0]} [IMG] positions in this step but "
            f"{rows.shape[0]} image rows for them"
        )
    h = nn.functional.embedding(input_ids, embed)
    return h.index_copy_(0, mm_token_indices, rows.to(h.dtype))
