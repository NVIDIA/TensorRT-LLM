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
"""Kimi K3 multimodal model for TensorRT-LLM PyTorch backend.

Wires the Kimi K3 vision-language model on top of the already-brought-up
text-only core (:class:`KimiLinearForCausalLM`):

- MoonViT3d vision encoder (native, no ``trust_remote_code`` for the model)
- PatchMergerMLPV2 vision->text projector
- ``KimiK3ForConditionalGeneration`` that fuses vision embeddings into the
  KimiLinear text backbone.

The structure closely mirrors the in-tree K2.5 model
(``modeling_kimi_k25.py``); this file only carries the K3-specific deltas and
reuses everything numerically identical from the K2.5 implementation:

    delta                 | K2.5                       | K3
    ----------------------+----------------------------+-----------------------------
    vision norms          | LayerNorm                  | RMSNorm (torch.nn.RMSNorm)
    attention head_dim    | vt_hidden_size // heads    | qkv_hidden_size // heads
    patch-embed conv bias | True                       | False
    vision qkv/o/MLP bias | True                       | False
    projector             | PatchMergerMLP (pre_norm)  | PatchMergerMLPV2 (post_norm)
    text backbone         | DeepseekV3ForCausalLM      | KimiLinearForCausalLM
"""

import copy
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

from ...inputs import (
    MultimodalPlaceholderMetadata,
    MultimodalPlaceholderPlacement,
    register_input_processor,
)
from ...logger import logger  # noqa: E402
from ..attention_backend import AttentionMetadata
from ..attention_backend.utils import get_attention_backend
from ..model_config import ModelConfig
from ..modules.linear import Linear, TensorParallelMode
from ..modules.mlp import MLP
from ..pyexecutor.config_utils import resolve_hf_torch_dtype
from .modeling_kimi_k25 import (
    _MEDIA_PLACEHOLDER_TOKEN_ID,
    DISAGG,
    KimiK25ForConditionalGeneration,
    KimiK25InputProcessor,
    KimiK25VisionAttention,
    KimiK25VisionModel,
    Learnable2DPosEmb,
    MoonViT3dEncoder,
    Rope2D,
    _gelu_tanh,
    _get_vision_tp_mapping,
    _has_meta_tensors,
)
from .modeling_kimi_linear import KimiLinearForCausalLM
from .modeling_utils import (
    MetaInitException,
    QuantConfig,
    register_auto_model,
    register_vision_encoder,
)

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

# ---------------------------------------------------------------------------
# Native MoonViT3d Vision Encoder Components (K3 deltas)
# ---------------------------------------------------------------------------


class K3PatchEmbed3d(nn.Module):
    """Conv2d patch embedding (bias-free) + learnable 2D position embedding.

    Mirrors ``MoonVision3dPatchEmbed`` with ``patch_embed_proj_bias=False``.
    ``Learnable2DPosEmb`` is numerically identical to the reference
    ``Learnable2DInterpPosEmbDivided_fixed`` for image inputs (bicubic interp).
    """

    def __init__(
        self,
        hidden_dim: int,
        patch_size: int,
        pos_emb_height: int,
        pos_emb_width: int,
        pos_emb_time: int,
    ) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.pos_emb = Learnable2DPosEmb(pos_emb_height, pos_emb_width, pos_emb_time, hidden_dim)

    def forward(self, x: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        x = self.proj(x).view(x.size(0), -1)
        return self.pos_emb(x, grid_thws)


class K3VisionMLP(MLP):
    """Bias-free MoonViT3d MLP2 (fc0 -> gelu_tanh -> fc1)."""

    def __init__(
        self, model_config: ModelConfig, layer_idx: int, hidden_dim: int, mlp_dim: int
    ) -> None:
        super().__init__(
            hidden_size=hidden_dim,
            intermediate_size=mlp_dim,
            bias=False,
            activation=_gelu_tanh,
            dtype=model_config.torch_dtype,
            config=model_config,
            layer_idx=layer_idx,
            overridden_tp_size=1 if model_config.mapping.enable_attention_dp else None,
        )


class K3EncoderLayer(nn.Module):
    """Single MoonViT3d encoder layer with RMSNorm + bias-free attention/MLP."""

    def __init__(
        self,
        model_config: ModelConfig,
        layer_idx: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
    ) -> None:
        super().__init__()
        # Reference uses torch.nn.RMSNorm(hidden_dim) with default eps for the
        # per-layer norms; match it exactly (created in fp32, cast with the rest
        # of the vision tower to the model dtype in load_weights()). NOTE:
        # eps=None resolves at *runtime* to torch.finfo(input.dtype).eps
        # (~7.8e-3 for bf16 vs ~1.2e-7 for fp32), so exact parity with the
        # reference holds only while both towers run in the same dtype (bf16
        # today). If the vision tower dtype ever changes, pin eps explicitly on
        # both sides together. Same applies to final_layernorm below.
        self.norm0 = nn.RMSNorm(hidden_dim)
        self.norm1 = nn.RMSNorm(hidden_dim)
        # head_dim is taken from model_config.pretrained_config.head_dim, which
        # KimiK3VisionModel sets to qkv_hidden_size // num_heads (128).
        self.attn = KimiK25VisionAttention(
            model_config,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            layer_idx=layer_idx,
            attn_bias=False,
        )
        self.mlp = K3VisionMLP(model_config, layer_idx, hidden_dim, mlp_dim)

    def forward(
        self,
        x: torch.Tensor,
        attn_metadata: AttentionMetadata,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        residual = x
        x = self.norm0(x)
        x = self.attn(x, attn_metadata, freqs_cis)
        x = residual + x
        residual = x
        x = self.norm1(x)
        x = residual + self.mlp(x)
        return x


class K3MoonViT3dEncoder(MoonViT3dEncoder):
    """MoonViT3d encoder stack with K3 deltas (RMSNorm + qkv-sized head_dim).

    Reuses :meth:`MoonViT3dEncoder.prepare_attn_metadata` and
    :meth:`MoonViT3dEncoder.forward`; only the sub-module construction differs,
    so ``__init__`` builds the K3 blocks directly and skips the K2.5 parent
    ``__init__`` (which would build LayerNorm blocks with the wrong head_dim).
    """

    def __init__(
        self,
        model_config: ModelConfig,
        hidden_dim: int,
        num_layers: int,
        num_heads: int,
        mlp_dim: int,
        head_dim: int,
    ) -> None:
        nn.Module.__init__(self)
        self.rope_2d = Rope2D(head_dim)
        self.blocks = nn.ModuleList(
            [
                K3EncoderLayer(
                    model_config,
                    layer_idx=layer_idx,
                    num_heads=num_heads,
                    hidden_dim=hidden_dim,
                    mlp_dim=mlp_dim,
                )
                for layer_idx in range(num_layers)
            ]
        )
        # Default-eps RMSNorm to match the reference; the eps value is
        # runtime-dtype-dependent — see the note on K3EncoderLayer.norm0.
        self.final_layernorm = nn.RMSNorm(hidden_dim)
        self.metadata_cls = get_attention_backend(model_config.attn_backend).Metadata
        self.attn_metadata: Optional[AttentionMetadata] = None


class PatchMergerMLPV2(nn.Module):
    """K3 vision->text projector: view-merge -> Linear -> GELU -> Linear -> RMSNorm.

    Matches the reference ``PatchMergerMLPV2``: no ``pre_norm``, bias-free
    projections, exact (erf) ``nn.GELU``, and a trailing ``RMSNorm`` over the
    text hidden size. Reuses TRT-LLM ``Linear`` (tp_size==1 under attention_dp)
    so the projector stays TP-composable like the K2.5 projector.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        mm_hidden_size: int,
        text_hidden_size: int,
        num_heads: int,
        merge_kernel_size: Tuple[int, int] = (2, 2),
        ln_eps: float = 1e-5,
    ) -> None:
        super().__init__()
        kh, kw = merge_kernel_size
        self.merged_dim = mm_hidden_size * kh * kw
        mapping = _get_vision_tp_mapping(model_config, num_heads)
        self.proj = nn.Sequential(
            Linear(
                self.merged_dim,
                self.merged_dim,
                bias=False,
                dtype=model_config.torch_dtype,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                quant_config=model_config.get_quant_config(),
                skip_create_weights_in_init=model_config.skip_create_weights_in_init,
                allreduce_strategy=model_config.allreduce_strategy,
            ),
            nn.GELU(),
            Linear(
                self.merged_dim,
                text_hidden_size,
                bias=False,
                dtype=model_config.torch_dtype,
                mapping=mapping,
                tensor_parallel_mode=TensorParallelMode.ROW,
                quant_config=model_config.get_quant_config(),
                skip_create_weights_in_init=model_config.skip_create_weights_in_init,
                allreduce_strategy=model_config.allreduce_strategy,
            ),
        )
        self.post_norm = nn.RMSNorm(text_hidden_size, eps=ln_eps)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            lengths = [item.shape[0] for item in x]
            merged = torch.cat([item.reshape(item.shape[0], -1) for item in x], dim=0)
            out = self.post_norm(self.proj(merged))
            return list(torch.split(out, lengths, dim=0))
        batch = x.shape[0]
        return self.post_norm(self.proj(x.view(batch, -1, self.merged_dim)))


# ---------------------------------------------------------------------------
# MoonViT3d Vision Encoder (top-level wrapper)
# ---------------------------------------------------------------------------


class KimiK3VisionModel(KimiK25VisionModel):
    """Native MoonViT3d encoder + PatchMergerMLPV2 projector for Kimi K3.

    Reuses :meth:`KimiK25VisionModel.load_weights`, ``_extract_features`` and
    ``forward`` (the HF weight names and the merge/projection pipeline are
    identical); only the sub-module construction differs, so ``__init__`` builds
    the K3 tower and skips the K2.5 parent ``__init__``.
    """

    def __init__(self, model_config: ModelConfig[PretrainedConfig]) -> None:
        nn.Module.__init__(self)
        self.model_config = copy.copy(model_config)
        self.model_config.extra_attrs = copy.copy(model_config.extra_attrs)
        self.model_config._frozen = False
        # Vision tower is not quantized (checkpoint quant ignore list covers
        # vision_tower.* / mm_projector.*), and its encoder layers keep no KV
        # cache — so do not carry the text model's kv_cache_quant_algo over:
        # for cache-less vision attention it could only steer kernel selection
        # (a past instance of that failure mode is #12851).
        self.model_config.quant_config = QuantConfig()
        self.model_config.pretrained_config = copy.copy(model_config.pretrained_config)

        # Extract vision config dict (num_heads is resolved here, with this
        # model's default of 12 heads, because the TP-replication decision
        # below must use the same head count the tower is built with).
        vision_cfg = getattr(self.model_config.pretrained_config, "vision_config", {})
        if vision_cfg is None:
            vision_cfg = {}
        if not isinstance(vision_cfg, dict):
            vision_cfg = (
                vision_cfg.to_dict() if hasattr(vision_cfg, "to_dict") else vars(vision_cfg)
            )
        num_heads = vision_cfg.get(
            "vt_num_attention_heads", vision_cfg.get("num_attention_heads", 12)
        )

        # The MoonViT tower cannot be tensor-parallel sharded when its attention
        # head count is not divisible by the attention-TP degree (e.g. Kimi K3's
        # 12 heads under TP16); run the whole tower replicated (tp=1) in that
        # case so module construction and weight loading agree. Attention-DP
        # already replicates the vision tower via its own path, so leave it be.
        if not model_config.mapping.enable_attention_dp:
            self.model_config.mapping = _get_vision_tp_mapping(model_config, num_heads)
        pretrained_config = self.model_config.pretrained_config
        # Normalize the checkpoint dtype once (covers both the modern `dtype`
        # and legacy `torch_dtype` field names, string forms, and "auto") and
        # write the concrete torch.dtype back — after this, every access in
        # this file can simply read model_config.torch_dtype, whose property
        # assumes pretrained_config.torch_dtype is already concrete.
        model_dtype = resolve_hf_torch_dtype(pretrained_config) or torch.bfloat16
        pretrained_config.torch_dtype = model_dtype

        hidden_dim = vision_cfg.get("vt_hidden_size", vision_cfg.get("hidden_size", 1024))
        num_layers = vision_cfg.get("vt_num_hidden_layers", vision_cfg.get("num_hidden_layers", 27))
        # K3 delta: the attention head_dim is qkv_hidden_size // num_heads (128),
        # NOT vt_hidden_size // num_heads. wqkv projects hidden_dim -> 3*qkv and
        # wo projects qkv -> hidden_dim, so q/k/v live in the qkv space.
        qkv_hidden_size = vision_cfg.get("qkv_hidden_size", hidden_dim)
        head_dim = qkv_hidden_size // num_heads
        self.model_config.pretrained_config.head_dim = head_dim
        self.model_config._frozen = True

        mlp_dim = vision_cfg.get("vt_intermediate_size", vision_cfg.get("intermediate_size", 4096))
        mm_hidden_size = vision_cfg.get("mm_hidden_size", hidden_dim)
        text_hidden_size = vision_cfg.get("text_hidden_size", 7168)
        patch_size = vision_cfg.get("patch_size", 14)
        ln_eps = vision_cfg.get("projector_ln_eps", 1e-5)
        pos_h = vision_cfg.get("init_pos_emb_height", 64)
        pos_w = vision_cfg.get("init_pos_emb_width", 64)
        pos_t = vision_cfg.get("init_pos_emb_time", 4)

        merge_ks = vision_cfg.get("merge_kernel_size", [2, 2])
        if isinstance(merge_ks, int):
            self.merge_kernel_size = (merge_ks, merge_ks)
        elif isinstance(merge_ks, (list, tuple)):
            self.merge_kernel_size = tuple(merge_ks)
        else:
            self.merge_kernel_size = (2, 2)
        self.merge_type = vision_cfg.get("merge_type", "sd2_tpool")

        # This tower hardcodes the released K3 vision architecture
        # (K3PatchEmbed3d: bias-free conv; K3EncoderLayer: RMSNorm +
        # bias-free attention; K3VisionMLP: bias-free, gelu_tanh). Reject a
        # checkpoint that asks for a variant these modules do not build, so
        # the load fails loudly instead of silently producing wrong outputs.
        for field, supported in (
            ("norm_type", "rmsnorm"),
            ("mlp_type", "mlp2"),
            ("activation_func", "gelu_pytorch_tanh"),
            ("pos_emb_type", "divided_fixed"),
            ("attn_bias", False),
            ("patch_embed_proj_bias", False),
            ("linear_bias", False),
        ):
            value = vision_cfg.get(field, supported)
            if value != supported:
                raise ValueError(
                    f"Kimi K3 vision tower supports {field}={supported!r}, got {value!r}"
                )

        text_config = getattr(pretrained_config, "text_config", pretrained_config)
        self.model_dtype = model_dtype
        self.text_hidden_size = (
            text_config.get("hidden_size", text_hidden_size)
            if isinstance(text_config, dict)
            else getattr(text_config, "hidden_size", text_hidden_size)
        )
        self.config = PretrainedConfig(
            num_attention_heads=num_heads,
            num_key_value_heads=num_heads,
            tie_word_embeddings=False,
        )

        self.patch_embed = K3PatchEmbed3d(hidden_dim, patch_size, pos_h, pos_w, pos_t)
        self.encoder = K3MoonViT3dEncoder(
            self.model_config,
            hidden_dim,
            num_layers,
            num_heads,
            mlp_dim,
            head_dim,
        )
        self.mm_projector = PatchMergerMLPV2(
            self.model_config,
            mm_hidden_size,
            self.text_hidden_size,
            num_heads,
            self.merge_kernel_size,
            ln_eps,
        )


# ---------------------------------------------------------------------------
# Input Processor
# ---------------------------------------------------------------------------


class KimiK3InputProcessor(KimiK25InputProcessor):
    """Image-only input processor for Kimi K3.

    Reuses the K2.5 processor: the K3 ``AutoProcessor`` (``KimiK3Processor``,
    loaded via ``trust_remote_code`` from the checkpoint) exposes the same
    ``image_processor.media_tokens_calculator`` and ``(medias=, text=)`` call
    contract, and returns ``grid_thws`` / ``pixel_values``. The framework
    injects the K3 ``<|kimi_image_placeholder|>`` marker (see the registration
    below), which ``KimiK3Processor.update_raw_text`` expands into the
    ``<|media_pad|>`` (id 163605) run that ``call_with_text_prompt`` then
    duplicates to ``(h // merge_kh) * (w // merge_kw)`` tokens per image.
    """


# ---------------------------------------------------------------------------
# Full VLM Model
# ---------------------------------------------------------------------------


@register_vision_encoder(KimiK3VisionModel)
@register_auto_model("KimiK3ForConditionalGeneration")
@register_input_processor(
    KimiK3InputProcessor,
    model_type="kimi_k3",
    placeholder_metadata=MultimodalPlaceholderMetadata(
        placeholder_map={
            "image": "<|kimi_image_placeholder|>",
        },
        placeholder_placement=MultimodalPlaceholderPlacement.BEFORE_TEXT,
        # K3's reference renderer concatenates content parts with no
        # separator; the default "\n" join skews prompt-token parity.
        placeholders_separator="",
    ),
)
class KimiK3ForConditionalGeneration(KimiK25ForConditionalGeneration):
    """Kimi K3 vision-language model: MoonViT3d + KimiLinear text backbone.

    Reuses the K2.5 wrapper's spec-dec / weight-loading property forwarding,
    :meth:`forward`, and :meth:`load_weights` (which builds the tower through
    ``_VISION_MODEL_CLS``); only ``__init__`` differs (vision encoder + text
    backbone classes).
    """

    _VISION_MODEL_CLS = KimiK3VisionModel

    @classmethod
    def get_model_defaults(cls, llm_args: "TorchLlmArgs") -> Dict[str, str]:
        """Default this model to the lazy safetensors load format.

        The K3 checkpoint (~1.5 TB) must be streamed shard-by-shard, so K3
        declares the lazy ``LoadFormat`` as its default now that the shared
        loader no longer sniffs K3 by model type. This wrapper subclasses HF
        ``PreTrainedModel``, not the TRT-LLM base, so there is no inherited
        ``get_model_defaults`` to extend. A user-set ``load_format`` still wins.
        """
        return {"load_format": "lazy_safetensors"}

    def __init__(
        self,
        model_config: ModelConfig[PretrainedConfig],
        *args,
        **kwargs,
    ) -> None:
        config = model_config.pretrained_config
        # PreTrainedModel.__init__ resolves config._attn_implementation and
        # rejects the default "sdpa" unless the class declares support. This
        # wrapper never runs HF attention itself (TRT-LLM attention is wired
        # inside the sub-modules), so declare support to keep HF init happy.
        self._supports_sdpa = True
        # Skip the K2.5 parent __init__ (it wires DeepSeek-V3 + K2.5 vision);
        # initialize the HF PreTrainedModel machinery directly, then build the
        # K3 components below.
        PreTrainedModel.__init__(self, config)

        # Re-entry guard (mirrors the K2.5 wrapper): the tail of this __init__
        # repoints model_config.pretrained_config at the text_config and remaps
        # the quant exclude list, so running the body twice on the same
        # instance would fail the text_config assert below. If the text
        # backbone already exists this is a re-init on a built instance: no-op.
        if hasattr(self, "llm"):
            return

        self.model_config = model_config
        self._vlm_pretrained_config = config

        # --- Vision encoder (deferred under MetaInitMode, recreated in
        # load_weights) ---
        self.mm_encoder = None
        if not DISAGG:
            try:
                mm_encoder = self._VISION_MODEL_CLS(model_config)
                if _has_meta_tensors(mm_encoder):
                    logger.info("Vision encoder deferred to load_weights() (MetaInitMode active)")
                else:
                    self.mm_encoder = mm_encoder
            except MetaInitException:
                logger.info("Vision encoder deferred to load_weights() (MetaInitMode active)")

        text_model_config = copy.copy(model_config)
        assert hasattr(config, "text_config"), "Kimi K3 config must have text_config"
        text_model_config._frozen = False
        text_model_config.pretrained_config = config.text_config

        # Remap quant exclude_modules: language_model.X -> model.X
        if text_model_config.quant_config.exclude_modules:
            text_model_config.quant_config = copy.copy(text_model_config.quant_config)
            p = self._LANG_PREFIX
            mapped = []
            for m in text_model_config.quant_config.exclude_modules:
                if m.startswith(p):
                    rest = m[len(p) :]
                    if rest.startswith("layers."):
                        rest = "model." + rest
                    mapped.append(rest)
                else:
                    mapped.append(m)
            text_model_config.quant_config.exclude_modules = mapped

        if not text_model_config.skip_create_weights_in_init:
            text_model_config.skip_create_weights_in_init = True
        text_model_config._frozen = True

        self.llm = KimiLinearForCausalLM(text_model_config)

        self._media_placeholder_token_id = getattr(
            config, "media_placeholder_token_id", _MEDIA_PLACEHOLDER_TOKEN_ID
        )
        self._mm_token_ids = torch.tensor([self._media_placeholder_token_id], dtype=torch.int32)

        # Point model_config at the text_config so the executor reads generation
        # params (eos_token_id, ...) from the text backbone, matching K2.5.
        self.config = self.llm.config
        model_config._frozen = False
        model_config.pretrained_config = self.llm.config
        model_config._frozen = True

    # load_weights is inherited from KimiK25ForConditionalGeneration: the
    # MetaInitMode deferral/recreation logic is identical and constructs the
    # tower via _VISION_MODEL_CLS, which this class overrides above.
