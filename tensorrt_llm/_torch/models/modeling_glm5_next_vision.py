# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""GLM-5.3-Flash vision encoder and conditional-generation wrapper.

Images and video frame pairs share a BF16 tower: patch embedding, pre-norm
blocks, post-norm, spatial downsampling and a projector into text embeddings.
Blocks use per-head Q/K RMSNorm, FP32 two-axis RoPE and clamped SwiGLU.

Vision attention uses the TRTLLM backend with per-item full-attention segments
and no KV cache, reusing Qwen-VL metadata preparation. The HF Glm5NextProcessor
handles preprocessing and placeholder expansion; the required Transformers
version is documented in the deployment guide.
"""

import copy
import re
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, PretrainedConfig, PreTrainedModel

from tensorrt_llm._torch.models.modeling_multimodal_utils import _is_mm_disagg
from tensorrt_llm._utils import prefer_pinned

from ...inputs import (
    ContentFormat,
    ExtraProcessedInputs,
    MultimodalPlaceholderMetadata,
    TextPrompt,
    register_input_processor,
)
from ...inputs.multimodal import MultimodalParams
from ...inputs.registry import BaseMultimodalDummyInputsBuilder, BaseMultimodalInputProcessor
from ...logger import logger
from ...sampling_params import SamplingParams
from ..attention.attention import Attention
from ..attention.backends import AttentionMetadata
from ..attention.backends.interface import PredefinedAttentionMask
from ..attention.backends.trtllm import TrtllmAttention
from ..attention.rotary_embedding import RotaryEmbedding
from ..flashinfer_utils import IS_FLASHINFER_AVAILABLE
from ..modules.gated_mlp import GatedMLP
from ..modules.layer_norm import LayerNorm
from ..modules.linear import Linear, TensorParallelMode, WeightMode, WeightsLoadingConfig
from ..modules.rms_norm import RMSNorm
from ..modules.swiglu import swiglu
from ..pyexecutor.config_utils import unwrap_glm5_next_text_config
from ..utils import torch_compiling

if IS_FLASHINFER_AVAILABLE:
    from ..custom_ops import flashinfer_apply_rope_with_cos_sin_cache_inplace
else:  # pragma: no cover - CPU / no-flashinfer environments
    flashinfer_apply_rope_with_cos_sin_cache_inplace = None
from .modeling_auto import AutoModelForCausalLM
from .modeling_glm5_next import Glm5NextForCausalLM, glm5_next_attention_mapping
from .modeling_multimodal_encoder import MultimodalEncoderMixin
from .modeling_multimodal_mixin import (
    EncoderGroup,
    MultimodalModelMixin,
    encode_multimodal_by_groups,
)
from .modeling_qwen2vl import _prepare_qwen_vl_vision_attn_metadata
from .modeling_utils import (
    ModelConfig,
    QuantConfig,
    _load_weights_impl,
    filter_weights,
    register_auto_model,
    register_vision_encoder,
)
from .multimodal_encoder_graph import (
    EncoderGraphKey,
    EncoderGraphTensorSpec,
    EncoderMetadataProvider,
    MultimodalEncoderGraphRunner,
)

if TYPE_CHECKING:
    from ...llmapi.llm_args import MultimodalEncoderCudaGraphConfig

_VISION_WEIGHT_PREFIX = "model.visual"


def _image_encoder_cuda_graph_config(
    model_config: ModelConfig[PretrainedConfig],
) -> Optional["MultimodalEncoderCudaGraphConfig"]:
    mm_config = model_config.multimodal_config
    if mm_config is None or mm_config.encoder_cuda_graph is None:
        return None
    unknown = set(mm_config.encoder_cuda_graph) - {"image"}
    if unknown:
        raise ValueError(
            "glm5_next: unsupported multimodal encoder CUDA graph modalities "
            f"{sorted(unknown)}; only 'image' is supported"
        )
    return mm_config.encoder_cuda_graph.get("image")


def _text_dtype(model_config: ModelConfig[PretrainedConfig]) -> torch.dtype:
    text_config = unwrap_glm5_next_text_config(model_config.pretrained_config)
    dtype = getattr(text_config, "dtype", None)
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    return dtype or torch.bfloat16


def _require_trtllm_vision_backend(model_config: ModelConfig[PretrainedConfig], where: str) -> None:
    """Require full-mask TRTLLM vision attention without sparse backend wrappers."""
    backend = model_config.attn_backend
    if not isinstance(backend, str) or backend.upper() != "TRTLLM":
        raise ValueError(
            f"{where}: the glm5_next vision tower supports only the TRTLLM "
            f"attention backend, got attn_backend={backend!r}. There is no "
            "VANILLA/FlashInfer/SDPA vision attention path."
        )
    if model_config.sparse_attention_config is not None:
        raise ValueError(
            f"{where}: the glm5_next vision tower runs plain full-mask TRTLLM "
            "attention; a sparse_attention_config would swap in a sparse "
            "backend wrapper and is not supported on the vision path."
        )


def _create_linear_weights(*modules: nn.Module) -> None:
    """The tower is not a ``DecoderModelForCausalLM``, so nothing runs
    ``create_weights`` for it after construction; materialize any shared
    module that deferred its weights (``skip_create_weights_in_init``)."""
    for module in modules:
        for sub in module.modules():
            if isinstance(sub, Linear) and not sub._weights_created:
                sub.create_weights()


class Glm5NextVisionAttention(Attention):
    """GLM vision attention on the standard ``Attention`` module.

    Module-side additions relative to the base flow: per-head Q/K RMSNorm
    followed by table-driven two-axis rotary embedding. Core execution stays on the
    configured backend (TRTLLM) with ``PredefinedAttentionMask.FULL`` per
    image segment and no KV cache.
    """

    def __init__(self, model_config: ModelConfig[PretrainedConfig], layer_idx: int) -> None:
        _require_trtllm_vision_backend(model_config, type(self).__name__)
        config = model_config.pretrained_config.vision_config
        text_config = unwrap_glm5_next_text_config(model_config.pretrained_config)
        dtype = _text_dtype(model_config)
        super().__init__(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            max_position_embeddings=int(text_config.max_position_embeddings),
            bias=bool(config.attention_bias),
            pos_embd_params=None,
            rope_fusion=False,
            layer_idx=layer_idx,
            dtype=dtype,
            config=model_config,
            reduce_output=(
                not model_config.mapping.enable_attention_dp and model_config.mapping.tp_size > 1
            ),
            head_dim=config.hidden_size // config.num_heads,
        )
        # Backend resolution can substitute an implementation; require plain
        # TRTLLM metadata and full-attention semantics after construction too.
        if type(self.attn) is not TrtllmAttention:
            raise ValueError(
                f"{type(self).__name__}: constructed attention backend is "
                f"{type(self.attn).__name__}, expected TrtllmAttention. The "
                "glm5_next vision tower admits no other backend."
            )
        # Per-head Q/K RMSNorm over head_dim, weights replicated across TP.
        self.q_norm = RMSNorm(hidden_size=self.head_dim, eps=config.rms_norm_eps, dtype=dtype)
        self.k_norm = RMSNorm(hidden_size=self.head_dim, eps=config.rms_norm_eps, dtype=dtype)
        # Vision attention runs from the outer VL wrapper, outside the
        # compiled LM region. Unregister from the shared attn-layer metadata
        # map so compiled LM attention lookups never resolve a vision layer
        # (same contract as Qwen2_5_VLVisionAttention).
        if self.register_to_config:
            model_config.extra_attrs.get("attn_layers", {}).pop(self.layer_idx_str, None)
            self.register_to_config = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = self.split_qkv(qkv, None, None)
        seq_len = q.shape[0]
        # Normalize per head, then apply the two-axis RoPE tables by token row.
        q = self.q_norm(q.reshape(-1, self.head_dim)).reshape(seq_len, -1)
        k = self.k_norm(k.reshape(-1, self.head_dim)).reshape(seq_len, -1)
        cos, sin = position_embeddings
        half = self.head_dim // 2
        cos_half, sin_half = cos[:, :half], sin[:, :half]
        if flashinfer_apply_rope_with_cos_sin_cache_inplace is not None and self.head_dim % 64 == 0:
            q = q.contiguous()
            k = k.contiguous()
            cos_sin_cache = torch.cat([cos_half, sin_half], dim=-1).to(torch.float32).contiguous()
            positions = torch.arange(seq_len, device=q.device, dtype=torch.int32)
            flashinfer_apply_rope_with_cos_sin_cache_inplace(
                positions, q, k, self.head_dim, cos_sin_cache, is_neox=True
            )
        else:
            q = (
                RotaryEmbedding.apply_rotary_pos_emb(
                    q.view(1, seq_len, -1, self.head_dim), cos_half, sin_half, unsqueeze_dim=1
                )
                .to(q.dtype)
                .reshape(seq_len, -1)
            )
            k = (
                RotaryEmbedding.apply_rotary_pos_emb(
                    k.view(1, seq_len, -1, self.head_dim), cos_half, sin_half, unsqueeze_dim=1
                )
                .to(k.dtype)
                .reshape(seq_len, -1)
            )
        # The TRTLLM backend consumes the packed [q|k|v] projection directly.
        qkv = torch.cat([q, k, v], dim=-1)
        output = self.forward_impl(
            q=qkv,
            k=None,
            v=None,
            attn_metadata=attn_metadata,
            attention_mask=PredefinedAttentionMask.FULL,
            attention_window_size=None,
            attention_mask_data=None,
            mrope_config=None,
            attention_sinks=None,
        )
        return self.o_proj(output, layer_idx=self.layer_idx)


class Glm5NextVisionBlock(nn.Module):
    """Pre-norm residual block: ``x += attn(norm1(x)); x += mlp(norm2(x))``.

    The first residual add is fused into ``norm2``; the block still returns
    the full hidden state so boundary hooks see the source activation.
    """

    def __init__(self, model_config: ModelConfig[PretrainedConfig], layer_idx: int) -> None:
        super().__init__()
        config = model_config.pretrained_config.vision_config
        dtype = _text_dtype(model_config)
        self.norm1 = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)
        self.norm2 = RMSNorm(hidden_size=config.hidden_size, eps=config.rms_norm_eps, dtype=dtype)
        self.attn = Glm5NextVisionAttention(model_config, layer_idx)
        # Biased clamped-SwiGLU MLP: the shared GatedMLP already runs one fused
        # gate|up column-sharded GEMM, the clamped swiglu kernel and a
        # row-sharded down projection (source ``Glm5NextVisionMLP``).
        self.mlp = GatedMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            bias=bool(config.attention_bias),
            dtype=dtype,
            config=model_config,
            overridden_tp_size=1 if model_config.mapping.enable_attention_dp else None,
            layer_idx=layer_idx,
            swiglu_limit=float(config.swiglu_limit),
        )
        _create_linear_weights(self.mlp, self.attn)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attn_metadata: AttentionMetadata,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        attn_out = self.attn(
            self.norm1(hidden_states),
            attn_metadata=attn_metadata,
            position_embeddings=position_embeddings,
        )
        normed, hidden_states = self.norm2(attn_out, hidden_states)
        return hidden_states + self.mlp(normed)


class Glm5NextVisionPatchEmbed(nn.Module):
    """``Conv3d`` patchifier over packed ``(N, 3*t*p*p)`` rows (replicated)."""

    def __init__(self, config: PretrainedConfig, dtype: torch.dtype) -> None:
        super().__init__()
        self.patch_size = config.patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.in_channels = config.in_channels
        self.embed_dim = config.hidden_size
        kernel = (self.temporal_patch_size, self.patch_size, self.patch_size)
        self.kernel = kernel
        # Meta-safe parameter creation (no fill-style init; the checkpoint
        # always overwrites); forward is F.conv3d — Conv3d-exact.
        self.proj = nn.Module()
        self.proj.weight = nn.Parameter(
            torch.empty(self.embed_dim, self.in_channels, *kernel, dtype=dtype)
        )
        self.proj.bias = nn.Parameter(torch.empty(self.embed_dim, dtype=dtype))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        weight = self.proj.weight
        # kernel == stride == the packed patch extent, so the Conv3d is exactly
        # one GEMM over the flattened (C, T, P, P) patch rows.
        hidden_states = hidden_states.reshape(-1, weight.shape[1:].numel()).to(dtype=weight.dtype)
        return nn.functional.linear(hidden_states, weight.view(self.embed_dim, -1), self.proj.bias)


class Glm5NextVisionPatchMerger(nn.Module):
    """Projector: ``proj → LayerNorm → GELU → clamped SwiGLU → down``.

    All linears are bias-free (source ``Glm5NextVisionPatchMerger``); the
    first projection and the LayerNorm stay replicated because the norm needs
    the full 4096-wide row, gate/up shard column-wise and down row-wise.
    """

    def __init__(self, model_config: ModelConfig[PretrainedConfig]) -> None:
        super().__init__()
        config = model_config.pretrained_config.vision_config
        dtype = _text_dtype(model_config)
        dim = config.out_hidden_size
        context_dim = config.projection_intermediate_size
        self.swiglu_limit = float(config.swiglu_limit)
        # Each attention-DP rank encodes its own images, so the complete
        # merger must be local too (the Qwen-VL ownership convention).
        mapping = glm5_next_attention_mapping(model_config.mapping)
        # Replicated projection + LayerNorm (the norm needs the full row).
        self.proj = Linear(
            dim,
            dim,
            bias=False,
            dtype=dtype,
            mapping=mapping,
            quant_config=None,
            allreduce_strategy=model_config.allreduce_strategy,
        )
        self.post_projection_norm = LayerNorm(hidden_size=dim, eps=1e-5, dtype=dtype)
        self.act1 = nn.GELU()
        common = dict(
            dtype=dtype,
            mapping=mapping,
            quant_config=None,
            allreduce_strategy=model_config.allreduce_strategy,
        )
        # One fused gate|up GEMM + the clamped-swiglu kernel, as in the blocks.
        self.gate_up_proj = Linear(
            dim,
            context_dim * 2,
            bias=False,
            tensor_parallel_mode=TensorParallelMode.COLUMN,
            weights_loading_config=WeightsLoadingConfig(
                weight_mode=WeightMode.FUSED_GATE_UP_LINEAR
            ),
            **common,
        )
        self.down_proj = Linear(
            context_dim, dim, bias=False, tensor_parallel_mode=TensorParallelMode.ROW, **common
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.proj(hidden_states)
        hidden_states = self.act1(self.post_projection_norm(hidden_states))
        gate_up = self.gate_up_proj(hidden_states)
        return self.down_proj(swiglu(gate_up, swiglu_limit=self.swiglu_limit))


class Glm5NextVisionModel(nn.Module, MultimodalEncoderMixin):
    """The GLM vision tower: patch embed → 24 blocks → post-norm →
    downsample → projector. Context-only, ``kv_cache_manager=None``."""

    def __init__(self, model_config: ModelConfig[PretrainedConfig]) -> None:
        super().__init__()
        _require_trtllm_vision_backend(model_config, type(self).__name__)
        self.model_config = model_config
        self.config = model_config.pretrained_config.vision_config
        dtype = _text_dtype(model_config)

        self.spatial_merge_size = self.config.spatial_merge_size
        self.spatial_merge_unit = self.spatial_merge_size**2
        self.head_dim = self.config.hidden_size // self.config.num_heads

        self.patch_embed = Glm5NextVisionPatchEmbed(self.config, dtype)
        self.blocks = nn.ModuleList(
            [Glm5NextVisionBlock(model_config, layer_idx) for layer_idx in range(self.config.depth)]
        )
        self.post_layernorm = RMSNorm(
            hidden_size=self.config.hidden_size, eps=self.config.rms_norm_eps, dtype=dtype
        )
        # Conv2d(kernel = stride = merge) over one merge tile is a GEMM over the
        # tile's flattened (kh, kw, C) row; the weight keeps the checkpoint's
        # conv layout and is re-viewed once for the GEMM (see patch embed).
        self.downsample = nn.Module()
        self._downsample_gemm_weight: Optional[torch.Tensor] = None
        self.downsample.weight = nn.Parameter(
            torch.empty(
                self.config.out_hidden_size,
                self.config.hidden_size,
                self.spatial_merge_size,
                self.spatial_merge_size,
                dtype=dtype,
            )
        )
        self.downsample.bias = nn.Parameter(torch.empty(self.config.out_hidden_size, dtype=dtype))
        self.merger = Glm5NextVisionPatchMerger(model_config)

        # Two-axis Neox RoPE uses head_dim//4 frequencies per axis in FP32.
        # Keep inv_freq outside module buffers so encoder dtype casts preserve it.
        freq_dim = self.head_dim // 2
        # device='cpu' pins the table out of any meta-device construction
        # context (it is data, not a weight to materialize later).
        self._rope_inv_freq_cpu = 1.0 / (
            10000.0 ** (torch.arange(0, freq_dim, 2, dtype=torch.float32, device="cpu") / freq_dim)
        )
        self._rope_inv_freq_by_device: Dict[torch.device, torch.Tensor] = {}
        # Per-grid (cos, sin) tables on the model device; image grids repeat
        # heavily in practice and the numpy position build + H2D copy per call
        # is pure host latency otherwise.
        self._rope_cache: Dict[
            Tuple[Tuple[int, int, int], ...], Tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._rope_cache_limit = 64

        # The vision backend is fixed to plain TRTLLM.
        self.metadata_cls = TrtllmAttention.Metadata
        self.attn_metadata: Optional[AttentionMetadata] = None
        self._fixed_max_seq_len = model_config.max_num_tokens

        # Optional CUDA-graph replay of the block stack (see
        # `enable_blocks_cuda_graph`); opted in through
        # `multimodal_config.encoder_cuda_graph["image"]`.
        self._encoder_cuda_graph_config = _image_encoder_cuda_graph_config(model_config)
        self._blocks_graph_runner: Optional[MultimodalEncoderGraphRunner] = None

    @property
    def device(self) -> torch.device:
        return self.patch_embed.proj.weight.device

    # -- block-stack CUDA graphs ---------------------------------------------
    def enable_blocks_cuda_graph(
        self,
        config: Optional["MultimodalEncoderCudaGraphConfig"] = None,
        *,
        device: Optional[torch.device] = None,
    ) -> None:
        """Capture the 24-block loop for every configured bucket.

        Weights must already live on the CUDA device. The eager prelude
        (patch embed, rotary tables) and tail (post norm, downsample, merger)
        stay outside the graph; a request whose (num_images, total_patches)
        matches no bucket falls back to the eager block loop.
        """
        config = config if config is not None else self._encoder_cuda_graph_config
        if config is None or self._blocks_graph_runner is not None:
            return
        dtype = self.patch_embed.proj.weight.dtype
        runner = MultimodalEncoderGraphRunner(
            encoder_fn=self._encoder_graph_fn,
            metadata_provider=_Glm5NextVisionGraphMetadataProvider(self),
            input_specs={
                "x": EncoderGraphTensorSpec(shape=(self.config.hidden_size,), dtype=dtype),
                "cos": EncoderGraphTensorSpec(shape=(self.head_dim,), dtype=torch.float32),
                "sin": EncoderGraphTensorSpec(shape=(self.head_dim,), dtype=torch.float32),
            },
            output_specs={"x": 0},
            config=config,
        )
        runner.capture_all(device if device is not None else self.device)
        self._blocks_graph_runner = runner

    def _encoder_graph_fn(
        self, inputs: Mapping[str, torch.Tensor], attn_metadata: AttentionMetadata
    ) -> Dict[str, torch.Tensor]:
        position_embeddings = (inputs["cos"], inputs["sin"])
        return {"x": self._run_blocks_eager(inputs["x"], position_embeddings, attn_metadata)}

    def _run_blocks_eager(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attn_metadata: AttentionMetadata,
    ) -> torch.Tensor:
        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                attn_metadata=attn_metadata,
                position_embeddings=position_embeddings,
            )
        return hidden_states

    def _run_blocks(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        seq_lens: List[int],
    ) -> torch.Tensor:
        # The tower runs outside the compiled LM region with explicit
        # metadata; keep both graph replay and eager off the custom-op path.
        with torch_compiling(False):
            if self._blocks_graph_runner is not None:
                cos, sin = position_embeddings
                out = self._blocks_graph_runner.maybe_run(
                    seq_lengths=seq_lens,
                    inputs={"x": hidden_states, "cos": cos, "sin": sin},
                )
                if out is not None:
                    return out["x"]
            self.attn_metadata = _prepare_qwen_vl_vision_attn_metadata(
                seq_lens, self.attn_metadata, max_seq_len=self._fixed_max_seq_len
            )
            return self._run_blocks_eager(hidden_states, position_embeddings, self.attn_metadata)

    # -- engine setup contract (MultimodalEncoderMixin) --------------------
    def setup_attn_metadata(
        self,
        max_num_tokens: int,
        attention_metadata_capacity: Optional[Dict[str, int]] = None,
    ) -> None:
        capacities = (
            attention_metadata_capacity
            if attention_metadata_capacity is not None
            else self.get_encoder_attention_metadata_capacity(max_num_tokens)
        )
        self.attn_metadata = self.metadata_cls(
            max_num_requests=capacities["attention"],
            max_num_tokens=max_num_tokens,
            kv_cache_manager=None,
        )
        self.set_attn_max_seq_len(max_num_tokens)

    def set_attn_max_seq_len(self, max_seq_len: int) -> None:
        if max_seq_len <= 0:
            raise ValueError(
                f"GLM vision attention max_seq_len must be positive, got {max_seq_len}"
            )
        self._fixed_max_seq_len = max_seq_len

    def get_encoder_attention_metadata_capacity(self, max_num_tokens: int) -> Dict[str, int]:
        # Every image segment holds at least one spatial-merge block.
        return {"attention": max(1, max_num_tokens // self.spatial_merge_unit)}

    # -- rotary -------------------------------------------------------------
    @staticmethod
    def rot_pos_ids(t: int, h: int, w: int, spatial_merge_size: int) -> torch.Tensor:
        """(h, w) coordinates per patch in spatial-merge-block order, repeated
        over the temporal axis — source ``get_vision_position_ids``."""
        hpos = np.broadcast_to(np.arange(h).reshape(h, 1), (h, w))
        wpos = np.broadcast_to(np.arange(w).reshape(1, w), (h, w))
        block = (
            h // spatial_merge_size,
            spatial_merge_size,
            w // spatial_merge_size,
            spatial_merge_size,
        )
        hpos = hpos.reshape(block).transpose(0, 2, 1, 3).flatten()
        wpos = wpos.reshape(block).transpose(0, 2, 1, 3).flatten()
        pos = torch.from_numpy(np.stack([hpos, wpos], axis=-1).copy())
        if t > 1:
            pos = pos.repeat(t, 1)
        return pos

    def rot_pos_emb(self, grid_rows: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        key = tuple(tuple(int(v) for v in row) for row in grid_rows)
        cached = self._rope_cache.get(key)
        if cached is not None and cached[0].device == self.device:
            return cached
        pos = torch.cat(
            [self.rot_pos_ids(t, h, w, self.spatial_merge_size) for t, h, w in grid_rows], dim=0
        ).to(self.device)
        inv_freq = self._rope_inv_freq_by_device.get(pos.device)
        if inv_freq is None:
            inv_freq = self._rope_inv_freq_cpu.to(pos.device)
            self._rope_inv_freq_by_device[pos.device] = inv_freq
        freqs = (pos.unsqueeze(-1).float() * inv_freq).flatten(1)
        emb = torch.cat((freqs, freqs), dim=-1)
        result = (emb.cos(), emb.sin())
        if len(self._rope_cache) >= self._rope_cache_limit:
            self._rope_cache.pop(next(iter(self._rope_cache)))
        self._rope_cache[key] = result
        return result

    # -- forward -------------------------------------------------------------
    @torch.inference_mode()
    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        grid_rows = [[int(t), int(h), int(w)] for t, h, w in grid_thw.tolist()]
        seq_lens: List[int] = []
        for t, h, w in grid_rows:
            seq_lens.extend([h * w] * t)

        position_embeddings = self.rot_pos_emb(grid_rows)
        hidden_states = self.patch_embed(pixel_values.to(device=self.device))
        hidden_states = self._run_blocks(hidden_states, position_embeddings, seq_lens)
        return self.project_merged(hidden_states)

    def project_merged(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Post-blocks tail: post-norm → 2×2 downsample → projector."""
        hidden_states = self.post_layernorm(hidden_states)
        merge = self.spatial_merge_size
        rows = hidden_states.reshape(-1, merge * merge * hidden_states.shape[-1])
        weight = self.downsample.weight
        cached = self._downsample_gemm_weight
        if cached is None or cached.device != weight.device or weight.is_meta:
            # [out, C, kh, kw] -> [out, kh, kw, C] to match the tile row order.
            cached = weight.permute(0, 2, 3, 1).reshape(weight.shape[0], -1).contiguous()
            if not weight.is_meta:
                self._downsample_gemm_weight = cached
        hidden_states = nn.functional.linear(rows, cached, self.downsample.bias)
        return self.merger(hidden_states)


class _Glm5NextVisionGraphMetadataProvider(EncoderMetadataProvider):
    """Graph-owned TRTLLM metadata for the block-stack runner.

    Per bucket one metadata instance with ``is_cuda_graph=True`` so the
    ``seq_lens`` setter copies into the captured ``_seq_lens_cuda`` buffer
    instead of reallocating; the packed cumulative lengths the FMHA reads
    live in a fixed ``cu_q_seqlens`` buffer refreshed in place. Host-side
    views are rebound per refresh — the C++ op reads them only at capture.
    """

    graph_critical_attrs: Sequence[str] = ("_seq_lens_cuda", "cu_q_seqlens")

    def __init__(self, vit: "Glm5NextVisionModel") -> None:
        self._vit = vit

    def build(self, key: EncoderGraphKey) -> AttentionMetadata:
        vit = self._vit
        metadata = vit.metadata_cls(
            max_num_requests=key.num_contexts,
            max_num_tokens=key.total_tokens,
            kv_cache_manager=None,
        )
        metadata.is_cuda_graph = True
        metadata.cu_q_seqlens = torch.zeros(
            key.num_contexts + 1, dtype=torch.int32, device=vit.device
        )
        metadata.cu_kv_seqlens = metadata.cu_q_seqlens
        return metadata

    def refresh_in_place(
        self, metadata: AttentionMetadata, padded_seq_lengths: Sequence[int]
    ) -> None:
        n = len(padded_seq_lengths)
        seq_lens = torch.tensor(padded_seq_lengths, dtype=torch.int32, pin_memory=prefer_pinned())
        metadata.num_contexts = n
        metadata.request_ids = list(range(1, n + 1))
        metadata.seq_lens = seq_lens
        metadata.bind_encoder_cuda_graph_seq_lens(metadata.seq_lens, n)
        cu_seqlens = torch.zeros(n + 1, dtype=torch.int32, pin_memory=prefer_pinned())
        torch.cumsum(seq_lens, dim=0, out=cu_seqlens[1:])
        metadata.cu_q_seqlens.copy_(cu_seqlens, non_blocking=True)
        metadata.max_seq_len = self._vit._fixed_max_seq_len
        metadata.prepare_encoder_only()


class Glm5NextVisionModelBase(nn.Module):
    """Encoder wrapper: dtype/quant isolation, weight routing, batching."""

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        vlm_base_model: Optional[type[nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.model_config = model_config
        self.model_dtype = _text_dtype(model_config)
        # The tower is excluded from checkpoint quantization (BF16 in
        # ``modules_to_not_convert``); scrub the quant config so no Linear
        # picks up FP8 behavior.
        self.model_config.quant_config = QuantConfig()
        # Pin vision TP reductions to NCCL to avoid runtime autotuning.
        # This wrapper owns a config copy; preserve its incoming frozen state.
        from ..distributed import AllReduceStrategy

        was_frozen = self.model_config._frozen
        self.model_config._frozen = False
        self.model_config.allreduce_strategy = AllReduceStrategy.NCCL
        self.model_config._frozen = was_frozen
        self.visual = MultimodalModelMixin._cast_multimodal_encoder_dtype(
            (vlm_base_model or Glm5NextVisionModel)(self.model_config), self.model_dtype
        )

    def load_weights(
        self,
        weights: Dict[str, torch.Tensor],
        allow_partial_loading: bool = False,
    ) -> None:
        """Load ``model.visual.*`` through the shared loader (Qwen2-VL pattern).

        The checkpoint stores one fused ``attn.qkv`` projection and names the
        output projection ``attn.proj``; the shared ``Attention`` takes split
        q/k/v inputs and ``o_proj``. ``_load_weights_impl`` fuses the block
        and merger ``gate_proj``/``up_proj`` pairs into ``gate_up_proj`` and
        raises on any parameter left without a source.
        """
        visual_weights = filter_weights(_VISION_WEIGHT_PREFIX, weights)
        converted: Dict[str, torch.Tensor] = {}
        qkv_pattern = re.compile(r"(.*?)attn\.qkv\.(.*)")
        for name, tensor in visual_weights.items():
            match = qkv_pattern.match(name)
            if match:
                prefix, suffix = match.groups()
                q, k, v = tensor[:].chunk(3, dim=0)
                converted[f"{prefix}attn.q_proj.{suffix}"] = q
                converted[f"{prefix}attn.k_proj.{suffix}"] = k
                converted[f"{prefix}attn.v_proj.{suffix}"] = v
            else:
                converted[name] = tensor[:]
        self.visual.config.num_attention_heads = self.visual.config.num_heads
        _load_weights_impl(
            self.visual,
            converted,
            params_map={r"(.*?)attn\.proj\.(.*)": r"\1attn.o_proj.\2"},
            allow_partial_loading=allow_partial_loading,
        )

    def enable_blocks_cuda_graph(self, *, device: Optional[torch.device] = None) -> None:
        """Capture the configured encoder CUDA graphs (no-op when not configured)."""
        self.visual.enable_blocks_cuda_graph(device=device)

    @torch.inference_mode()
    def encode_batched(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """Run the tower over HF-processor pixel rows (images and video frame
        pairs share one layout: ``[patches, C * T * P * P]``)."""
        pixel_values = pixel_values.to(device=self.visual.device, dtype=self.model_dtype)
        return self.visual(pixel_values, grid_thw=grid_thw)

    @property
    def mm_encoder_groups(self) -> Tuple[EncoderGroup, ...]:
        # Images and videos share the tower; the framework lays the output out
        # as all image rows then all video rows and reorders into prompt order.
        return (
            EncoderGroup(
                modalities=("image", "video"),
                encoder_fn=self.encode_batched,
                build_batched_input=_glm5_next_build_batched_input,
            ),
        )

    def forward(self, multimodal_params: List[MultimodalParams]) -> List[torch.Tensor]:
        return [encode_multimodal_by_groups(self.mm_encoder_groups, multimodal_params)]


def _flatten_video_grid_thw(video_grid_thw: torch.Tensor) -> torch.Tensor:
    """Source ``Glm5NextModel.get_video_features``: every temporal frame pair of
    a video is encoded as its own ``(1, h, w)`` item, so ``(t, h, w)`` becomes
    ``t`` rows of ``(1, h, w)``."""
    t = video_grid_thw[:, 0]
    hw = torch.repeat_interleave(video_grid_thw[:, 1:], t, dim=0)
    return torch.cat([hw.new_ones(hw.shape[0], 1), hw], dim=1)


def _glm5_next_build_batched_input(multimodal_params: List[MultimodalParams]) -> Dict[str, Any]:
    pixels: List[torch.Tensor] = []
    grids: List[torch.Tensor] = []
    for param in multimodal_params:
        bucket = param.multimodal_data.get("image")
        if bucket is not None:
            pixels.append(bucket["pixel_values"])
            grids.append(bucket["image_grid_thw"])
    for param in multimodal_params:
        bucket = param.multimodal_data.get("video")
        if bucket is not None:
            pixels.append(bucket["pixel_values_videos"])
            grids.append(_flatten_video_grid_thw(bucket["video_grid_thw"]))
    device = pixels[0].device
    return {
        "pixel_values": torch.cat(pixels, dim=0),
        "grid_thw": torch.cat([g.to(device) for g in grids], dim=0),
    }


class Glm5NextInputProcessor(BaseMultimodalInputProcessor, BaseMultimodalDummyInputsBuilder):
    """Adapt HF image/video preprocessing and placeholder expansion to engine inputs.

    Requires the Transformers version specified in the deployment guide.
    """

    def __init__(
        self,
        model_path: str,
        config: PretrainedConfig,
        tokenizer: Optional[AutoTokenizer] = None,
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
        text_config = unwrap_glm5_next_text_config(config)
        dtype = getattr(text_config, "dtype", None) or torch.bfloat16
        self._dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self._tokenizer = (
            tokenizer if tokenizer is not None else AutoTokenizer.from_pretrained(model_path)
        )
        self._processor = None
        vision = config.vision_config
        self._merge_size = int(vision.spatial_merge_size)
        self._patch_size = int(vision.patch_size)

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
        if self._processor is None:
            from transformers.models.auto.processing_auto import PROCESSOR_MAPPING_NAMES

            if "glm5_next" not in PROCESSOR_MAPPING_NAMES:
                raise RuntimeError(
                    "GLM-5.3-Flash image/video processing requires transformers==5.17.0. "
                    "For text-only inference on older Transformers, set disable_mm_encoder=True."
                )
            self._processor = AutoProcessor.from_pretrained(
                self.model_path, use_fast=self._use_fast, trust_remote_code=self._trust_remote_code
            )
        return self._processor

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def get_vocab_size(self) -> int:
        return int(unwrap_glm5_next_text_config(self._config).vocab_size)

    def get_preferred_media_io_kwargs(self) -> Dict[str, Dict[str, Any]]:
        # PIL is the HF processor's native input; the server's default float
        # CHW tensor would be decoded, hashed and converted back for nothing.
        return {"image": {"format": "pil"}}

    def get_mm_token_ids(self) -> torch.Tensor:
        """In-vocab ``<|image|>`` / ``<|video|>`` ids: the frontend's
        out-of-vocabulary fallback (``ids >= vocab_size``) would find no
        multimodal positions and the engine would skip fusion (Qwen2-VL
        precedent). The begin/end delimiters stay ordinary text."""
        ids = [
            int(tid)
            for tid in (
                getattr(self._config, "image_token_id", None),
                getattr(self._config, "video_token_id", None),
            )
            if tid is not None
        ]
        return torch.tensor(ids, dtype=torch.int32)

    @property
    def spatial_merge_unit(self) -> int:
        return self._merge_size**2

    def get_num_tokens_per_video(
        self,
        *,
        video: List[Any],
        video_metadata: Optional[dict] = None,
        video_grid_thw: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> int:
        """Embedding rows one video contributes: ``video_grid_thw.prod() /
        merge_size**2`` after the HF video processor's temporal pairing.

        The framework passes the processor-produced ``video_grid_thw`` when it
        has it (Qwen3-VL precedent); otherwise the frames are run through the
        HF video processor. (The processor's own
        ``_get_num_multimodal_tokens(video_sizes=...)`` counts frames before
        pairing and would overstate the video by ``temporal_patch_size``.)
        """
        if video_grid_thw is None:
            meta = dict(video_metadata or {})
            meta["total_num_frames"] = len(video)
            video_grid_thw = self.processor.video_processor(
                videos=[video], video_metadata=[meta], do_sample_frames=False, return_tensors="pt"
            )["video_grid_thw"]
        grid = torch.as_tensor(video_grid_thw)
        return int(grid.prod(dim=-1).sum().item()) // self.spatial_merge_unit

    # -- profiling dummies ----------------------------------------------------
    def _max_grid_side(self, max_patches: int) -> int:
        side = int(max_patches**0.5)
        side -= side % self._merge_size
        while side > 0 and side * side > max_patches:
            side -= self._merge_size
        return max(side, self._merge_size)

    def _processor_max_patches(self) -> int:
        """HF's image-token budget expressed in pre-merge patch rows."""
        processor = self.processor.image_processor
        return int(processor.max_image_tokens) * int(processor.merge_size) ** 2

    def get_mm_max_tokens_per_item(
        self, max_num_encoder_tokens: Optional[int] = None
    ) -> Dict[str, int]:
        max_patches = self._processor_max_patches()
        if max_num_encoder_tokens is not None:
            max_patches = min(max_patches, int(max_num_encoder_tokens))
        side = self._max_grid_side(max_patches)
        return {"image": side * side}

    def get_dummy_mm_data(
        self,
        *,
        max_num_encoder_tokens: int,
        mm_counts: Mapping[str, int],
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Any]:
        num_images = int(mm_counts.get("image", 0))
        if num_images <= 0:
            return {}
        per_item = max(self.spatial_merge_unit, int(max_num_encoder_tokens) // num_images)
        side = self._max_grid_side(min(per_item, self._processor_max_patches()))
        pixels_side = side * self._patch_size
        image = Image.new("RGB", (pixels_side, pixels_side), (127, 127, 127))
        out = self.processor.image_processor(images=[image] * num_images, return_tensors="pt")
        return {
            "image": {
                "pixel_values": out["pixel_values"].to(dtype or self._dtype),
                "image_grid_thw": out["image_grid_thw"],
            }
        }

    # -- request processing ----------------------------------------------------
    def call_with_text_prompt(
        self, inputs: TextPrompt, sampling_params: SamplingParams
    ) -> Tuple[List[int], Optional[ExtraProcessedInputs]]:
        text_prompt = inputs.get("prompt")
        mm_data = inputs.get("multi_modal_data") or {}
        mm_processor_kwargs = dict(inputs.get("mm_processor_kwargs") or {})
        images = mm_data.get("image") or None
        video_datas = mm_data.get("video") or None
        if not images and not video_datas:
            # Text-only fast path: identical to the tokenizer and no
            # multimodal payload, so the tower is never touched.
            input_ids = self._tokenizer(text_prompt, return_tensors="pt").input_ids
            return input_ids[0].to(torch.int32).tolist(), None

        # Media loaders may deliver pre-rescaled float tensors (the server
        # default) instead of PIL / uint8; tell the HF processor not to
        # rescale those a second time (Qwen2-VL precedent).
        images_kwargs = dict(mm_processor_kwargs.get("images_kwargs") or {})
        videos_kwargs = dict(mm_processor_kwargs.get("videos_kwargs") or {})
        if images and "do_rescale" not in mm_processor_kwargs:
            images_kwargs.setdefault("do_rescale", not isinstance(images[0], torch.Tensor))
        videos = None
        video_metadata = None
        if video_datas:
            videos = [video_data.frames for video_data in video_datas]
            if "do_rescale" not in mm_processor_kwargs:
                videos_kwargs.setdefault("do_rescale", not isinstance(videos[0][0], torch.Tensor))
            # Frames are already sampled by the media loader; the processor
            # only needs fps / frame indices to lay out the per-frame
            # timestamps (``VideoMetadata.timestamps``).
            video_metadata = []
            for video_data in video_datas:
                meta = dict(video_data.metadata or {})
                meta["total_num_frames"] = len(video_data.frames)
                video_metadata.append(meta)
            if "do_sample_frames" not in mm_processor_kwargs:
                videos_kwargs.setdefault("do_sample_frames", False)

        mm_processor_kwargs["images_kwargs"] = images_kwargs
        mm_processor_kwargs["videos_kwargs"] = videos_kwargs

        # Fail closed before any pixel work: a placeholder / item count
        # mismatch must surface as a ValueError (an HTTP error at the serving
        # boundary), not as the HF processor's iterator exhaustion.
        for placeholder, items, what in (
            (self.processor.image_token, images or [], "image"),
            (self.processor.video_token, video_datas or [], "video"),
        ):
            n_placeholders = text_prompt.count(placeholder)
            if n_placeholders != len(items):
                raise ValueError(
                    f"glm5_next multimodal request carries {n_placeholders} "
                    f"{placeholder!r} placeholder(s) but {len(items)} {what}(s); "
                    "refusing the mismatched request"
                )

        processed = self.processor(
            text=[text_prompt],
            images=images,
            videos=videos,
            video_metadata=video_metadata,
            return_tensors="pt",
            **mm_processor_kwargs,
        )
        multimodal_data: Dict[str, Any] = {}
        if processed.get("pixel_values") is not None:
            multimodal_data["image"] = {
                "pixel_values": processed["pixel_values"].to(self._dtype),
                "image_grid_thw": processed["image_grid_thw"],
            }
        if processed.get("pixel_values_videos") is not None:
            multimodal_data["video"] = {
                "pixel_values_videos": processed["pixel_values_videos"].to(self._dtype),
                "video_grid_thw": processed["video_grid_thw"],
            }
        fused_input_ids = processed["input_ids"][0]
        return fused_input_ids.to(torch.int32).tolist(), {"multimodal_data": multimodal_data}


@register_vision_encoder(Glm5NextVisionModelBase, vlm_base_model=Glm5NextVisionModel)
@register_auto_model("Glm5NextForConditionalGeneration")
@register_input_processor(
    Glm5NextInputProcessor,
    model_type="glm5_next",
    placeholder_metadata=MultimodalPlaceholderMetadata(
        # Preserve each image/video placeholder at its original prompt position.
        placeholder_map={
            "image": "<|begin_of_image|><|image|><|end_of_image|>",
            "video": "<|begin_of_video|><|video|><|end_of_video|>",
        },
        placeholders_separator="",
        content_format=ContentFormat.OPENAI,
        interleave_placeholders=True,
    ),
)
class Glm5NextVLM(MultimodalModelMixin, PreTrainedModel):
    """Compose the text decoder with an optional image/video encoder.

    The inner decoder owns recurrent/KV state and speculative decoding. Text-only
    requests bypass the vision tower, which runs full attention without KV cache.
    """

    _supports_flash_attn = True
    _supports_sdpa = True

    def _check_and_adjust_experts_implementation(self, *args, **kwargs):
        # Transformers 5.x PreTrainedModel.__init__ probes MoE expert
        # implementations; the wrapper holds no MoE modules itself (the text
        # decoder manages its own experts), so skip the check.
        return None

    def __init__(self, model_config: ModelConfig[PretrainedConfig], *args, **kwargs) -> None:
        config = model_config.pretrained_config
        self.original_arch = config.architectures[0]
        super().__init__(config)
        self.model_config = model_config

        llm_model_config = copy.deepcopy(model_config)
        # Share the live extra_attrs dict: the LM attention layers register
        # their per-layer metadata there and model_engine reads the same dict.
        llm_model_config.extra_attrs = model_config.extra_attrs
        llm_model_config.pretrained_config.architectures = ["Glm5NextForCausalLM"]
        self.llm = AutoModelForCausalLM.from_config(llm_model_config)

        self.mm_encoder = None
        if not (_is_mm_disagg() or model_config.disable_mm_encoder):
            self.mm_encoder = Glm5NextVisionModelBase(copy.deepcopy(model_config)).eval()
            self.mm_encoder_groups = self.mm_encoder.mm_encoder_groups
        else:
            logger.info("Glm5NextVLM: multimodal encoder disabled; serving text-only requests.")

        # device='cpu' keeps the constant real under meta-device construction.
        self._mm_token_ids = torch.tensor(
            [int(config.image_token_id), int(config.video_token_id)],
            dtype=torch.int32,
            device="cpu",
        )
        self.post_config()

    # -- engine contracts -----------------------------------------------------
    @classmethod
    def get_model_defaults(cls, llm_args) -> dict:
        return Glm5NextForCausalLM.get_model_defaults(llm_args)

    @classmethod
    def get_preferred_kv_cache_manager_version(cls, pretrained_config=None) -> str:
        return Glm5NextForCausalLM.get_preferred_kv_cache_manager_version(pretrained_config)

    @classmethod
    def get_preferred_transceiver_runtime(cls, pretrained_config=None) -> str:
        return Glm5NextForCausalLM.get_preferred_transceiver_runtime(pretrained_config)

    @property
    def mamba_metadata_cls(self):
        # The engine resolves the Mamba metadata class from the top-level model.
        return self.llm.mamba_metadata_cls

    @property
    def multimodal_token_ids(self) -> torch.Tensor:
        return self._mm_token_ids

    @property
    def multimodal_data_device_paths(self) -> List[str]:
        return ["image.pixel_values", "video.pixel_values_videos", "multimodal_embedding"]

    @property
    def language_model(self) -> torch.nn.Module:
        return self.llm

    # Speculative decoding lives on the inner decoder (``SpecDecOneEngineForCausalLM``
    # via the deep-copied model_config), but ``ModelLoader.load`` reads the draft
    # state from the outer model it resolved for the checkpoint's architecture.
    # ``load_draft_weights`` keeps an explicit signature: the loader dispatches
    # kwargs via ``inspect.getfullargspec``.
    @property
    def draft_config(self):
        return self.llm.draft_config

    @property
    def draft_model(self):
        return self.llm.draft_model

    def load_draft_weights(
        self, weights: Dict[str, torch.Tensor], weight_mapper: Optional[Any] = None
    ):
        return self.llm.load_draft_weights(weights, weight_mapper=weight_mapper)

    def get_language_model_extra_forward_kwargs(
        self,
        *,
        raw_input_ids: Optional[torch.Tensor],
        position_ids: Optional[torch.Tensor],
        mm_inputs: Any,
        spec_metadata: Any = None,
        resource_manager: Any = None,
        **forward_kwargs: Any,
    ) -> Dict[str, Any]:
        # The mixin's default forwards only the five common arguments. The
        # decoder's speculative worker indexes spec_metadata unconditionally,
        # and MTP drafting needs the pre-fusion token ids once fused
        # inputs_embeds replace input_ids for image prompts (Qwen3-VL contract).
        del position_ids, mm_inputs, forward_kwargs
        return {
            "spec_metadata": spec_metadata,
            "resource_manager": resource_manager,
            "orig_input_ids": raw_input_ids,
        }

    @property
    def text_embedding_layer(self):
        return self.llm.model.embed_tokens

    @property
    def embedding_dim(self) -> int:
        return self.text_embedding_layer.embedding_dim

    @property
    def embedding_dtype(self) -> torch.dtype:
        return self.text_embedding_layer.weight.dtype

    def post_config(self):
        self.model_config.pretrained_config = self.llm.config
        self.config = self.model_config.pretrained_config

    def encode_multimodal_inputs(
        self, multimodal_params: List[MultimodalParams], **encoder_kwargs: Any
    ) -> torch.Tensor:
        if self.mm_encoder is None:
            raise ValueError("Raw multimodal inputs require a local multimodal encoder.")
        mm_embeds = self.mm_encoder.forward(list(multimodal_params), **encoder_kwargs)
        if len(mm_embeds) != 1:
            raise ValueError(
                "glm5_next multimodal encoder must return one packed "
                f"embedding tensor, but returned {len(mm_embeds)} tensors."
            )
        return mm_embeds[0]

    def load_weights(self, weights: Dict[str, torch.Tensor], **_ignored: Any) -> None:
        """Route ``model.visual.*`` to the tower, everything else to the
        audited text loader (which itself allowlists the visual namespace).

        The wrapper does not declare ``weight_mapper`` because the text
        decoder initializes its own registered mapper. An explicitly passed
        wrapper-level mapper is rejected.
        """
        if _ignored.pop("weight_mapper", None) is not None:
            raise ValueError(
                "glm5_next uses its audited exact-placement loader; a "
                "checkpoint-format weight_mapper is not supported"
            )
        if self.mm_encoder is not None:
            visual_keys = any(key.startswith(_VISION_WEIGHT_PREFIX) for key in weights)
            if not visual_keys:
                raise ValueError(
                    "glm5_next multimodal load: the checkpoint holds no "
                    f"'{_VISION_WEIGHT_PREFIX}.*' weights but the vision "
                    "encoder is enabled; refusing to leave an uninitialized "
                    "tower (serve text-only with disable_mm_encoder instead)"
                )
            self.mm_encoder.load_weights(weights)
            # Weights are on device now; capture the (opt-in) encoder graphs.
            self.mm_encoder.enable_blocks_cuda_graph()
        self.llm.load_weights(weights)
