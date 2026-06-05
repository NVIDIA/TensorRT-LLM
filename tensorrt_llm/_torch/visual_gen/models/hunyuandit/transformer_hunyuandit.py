# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HunyuanDiT 2D transformer wrapper with Ulysses sequence parallelism.

Architecture overview
---------------------
The wrapper exposes two classes to the pipeline:

``HunyuanDiT2DModelWrapper``
    Thin ``nn.Module`` around diffusers' ``HunyuanDiT2DModel``.  Selects
    ``HunyuanDiT2DModelUlysses`` (a subclass with Ulysses support) when
    ``visual_gen_mapping.ulysses_size > 1``, otherwise falls back to the
    vanilla diffusers model for single-GPU usage.

``HunyuanDiT2DModelUlysses``
    Subclass of ``HunyuanDiT2DModel`` that overrides ``forward()`` to shard
    the latent sequence across Ulysses ranks AFTER the patch-embed and gather
    it back BEFORE the final norm/proj.  Self-attention blocks use
    ``HunyuanDiTUlyssesAttnProcessor`` which injects an all-to-all before and
    after ``F.scaled_dot_product_attention``.  Cross-attention is standard
    SDPA — text tokens are replicated on every rank so no all-to-all is needed.

``HunyuanDiTUlyssesAttnProcessor``
    Drop-in replacement for ``HunyuanAttnProcessor2_0``.  When called for
    self-attention it wraps SDPA with

        all_to_all(q/k/v,  scatter_dim=heads, gather_dim=seq)   # before
        SDPA([B, S, H/U, D])
        all_to_all(output, scatter_dim=seq,   gather_dim=heads)  # after

    so each rank computes a head-sharded slice of the full-sequence attention.

References
----------
- DeepSpeed Ulysses: https://arxiv.org/abs/2309.14509
- diffusers HunyuanDiT: ``diffusers.models.transformers.hunyuan_transformer_2d``
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from tensorrt_llm.logger import logger


# ---------------------------------------------------------------------------
# Ulysses attention processor
# ---------------------------------------------------------------------------


class HunyuanDiTUlyssesAttnProcessor:
    """Custom attention processor injecting Ulysses all-to-all for self-attention.

    Compatible with diffusers' attention processor protocol
    (``processor.__call__(attn, hidden_states, ...)``) and is a drop-in
    replacement for ``HunyuanAttnProcessor2_0``.

    For *cross-attention* (``encoder_hidden_states is not None``) the processor
    falls back to standard SDPA because text K/V tensors are already replicated
    on every rank — no all-to-all is required.

    Args:
        ulysses_group: ``torch.distributed.ProcessGroup`` spanning Ulysses ranks.
        ulysses_size:  Number of Ulysses ranks (must divide ``num_attention_heads``).
    """

    def __init__(self, ulysses_group: dist.ProcessGroup, ulysses_size: int):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("HunyuanDiTUlyssesAttnProcessor requires PyTorch ≥ 2.0.")
        self.ulysses_group = ulysses_group
        self.ulysses_size = ulysses_size

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        temb: Optional[torch.Tensor] = None,
        image_rotary_emb=None,
    ) -> torch.Tensor:
        from tensorrt_llm._torch.distributed import all_to_all_4d

        try:
            from diffusers.models.embeddings import apply_rotary_emb
        except ImportError:
            apply_rotary_emb = None

        is_cross = encoder_hidden_states is not None
        B = hidden_states.shape[0]

        # ---- Q / K / V projections ----------------------------------------
        query = attn.to_q(hidden_states)
        kv_src = encoder_hidden_states if is_cross else hidden_states
        key = attn.to_k(kv_src)
        value = attn.to_v(kv_src)

        # Derive head_dim from key projection output
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        # Reshape → [B, S, H, D]
        query = query.view(B, -1, attn.heads, head_dim)
        key = key.view(B, -1, attn.heads, head_dim)
        value = value.view(B, -1, attn.heads, head_dim)

        # ---- QK-norm (LayerNorm, applied per-head) -------------------------
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)

        # ---- RoPE (expects [B, H, S, D]) -----------------------------------
        if image_rotary_emb is not None and apply_rotary_emb is not None:
            query = apply_rotary_emb(query.transpose(1, 2), image_rotary_emb)[0]
            query = query.transpose(1, 2)
            if not is_cross:
                key = apply_rotary_emb(key.transpose(1, 2), image_rotary_emb)[0]
                key = key.transpose(1, 2)

        # ---- Ulysses all-to-all (self-attention only) ----------------------
        if not is_cross and self.ulysses_size > 1:
            # [B, S/U, H, D] → [B, S, H/U, D]
            query = all_to_all_4d(
                query, scatter_dim=2, gather_dim=1, process_group=self.ulysses_group
            )
            key = all_to_all_4d(
                key, scatter_dim=2, gather_dim=1, process_group=self.ulysses_group
            )
            value = all_to_all_4d(
                value, scatter_dim=2, gather_dim=1, process_group=self.ulysses_group
            )

            # SDPA expects [B, H, S, D]
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

            out = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)

            # Reverse: [B, H/U, S, D] → [B, S, H/U, D] → [B, S/U, H, D]
            out = out.transpose(1, 2).contiguous()
            out = all_to_all_4d(
                out, scatter_dim=1, gather_dim=2, process_group=self.ulysses_group
            )
        else:
            # Standard SDPA (cross-attention or single-GPU fallback)
            if attention_mask is not None:
                seq_len_kv = key.shape[1]
                attention_mask = attn.prepare_attention_mask(attention_mask, seq_len_kv, B)
                attention_mask = attention_mask.view(B, attn.heads, -1, attention_mask.shape[-1])

            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            out = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0
            )
            out = out.transpose(1, 2)

        # ---- Output projection ---------------------------------------------
        out = out.reshape(B, -1, attn.heads * head_dim).to(query.dtype)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out)
        return out


# ---------------------------------------------------------------------------
# Ulysses-capable HunyuanDiT2DModel subclass
# ---------------------------------------------------------------------------


class HunyuanDiT2DModelUlysses:
    """Mixin that overrides ``forward()`` to add Ulysses sequence sharding.

    We implement this as a plain class whose ``forward()`` replaces the
    diffusers model's ``forward()`` via attribute assignment after construction,
    because subclassing diffusers ``ModelMixin`` (which uses ``@register_to_config``)
    reliably breaks the config serialization when additional ``__init__`` kwargs are added.

    Usage (inside ``HunyuanDiT2DModelWrapper``)::

        model = HunyuanDiT2DModel(...)
        HunyuanDiT2DModelUlysses.patch(model, ulysses_group, ulysses_size)
    """

    @staticmethod
    def patch(
        model,
        ulysses_group: dist.ProcessGroup,
        ulysses_size: int,
    ) -> None:
        """Attach Ulysses sharding behaviour to an existing ``HunyuanDiT2DModel``.

        1. Stores ``ulysses_group`` / ``ulysses_size`` on the model instance.
        2. Replaces ``forward`` with our sequence-sharding variant.
        3. Replaces the self-attention processor on every block with
           ``HunyuanDiTUlyssesAttnProcessor``.
        """
        model._ulysses_group = ulysses_group
        model._ulysses_size = ulysses_size

        processor = HunyuanDiTUlyssesAttnProcessor(ulysses_group, ulysses_size)
        for block in model.blocks:
            # attn1 = self-attention; attn2 = cross-attention (keep default)
            block.attn1.processor = processor

        # Bind the new forward as a bound method
        import types

        model.forward = types.MethodType(HunyuanDiT2DModelUlysses._forward, model)

    @staticmethod
    def _forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states=None,
        text_embedding_mask=None,
        encoder_hidden_states_t5=None,
        text_embedding_mask_t5=None,
        image_meta_size=None,
        style=None,
        image_rotary_emb=None,
        controlnet_block_samples=None,
        return_dict=True,
    ):
        """Ulysses-aware forward.

        Identical to ``HunyuanDiT2DModel.forward`` except that it shards the
        patch-embedded sequence across Ulysses ranks before the transformer
        blocks and gathers it back before ``norm_out`` / ``proj_out``.
        """
        height, width = hidden_states.shape[-2:]

        # 1. PatchEmbed → [B, S, D]
        hidden_states = self.pos_embed(hidden_states)

        # 2. Timestep + text conditioning
        temb = self.time_extra_emb(
            timestep,
            encoder_hidden_states_t5,
            image_meta_size,
            style,
            hidden_dtype=timestep.dtype,
        )

        # 3. T5 text projection
        batch_size, seq_len_t5, _ = encoder_hidden_states_t5.shape
        encoder_hidden_states_t5 = self.text_embedder(
            encoder_hidden_states_t5.view(-1, encoder_hidden_states_t5.shape[-1])
        )
        encoder_hidden_states_t5 = encoder_hidden_states_t5.view(batch_size, seq_len_t5, -1)

        # Concatenate CLIP + T5 text embeddings
        encoder_hidden_states = torch.cat(
            [encoder_hidden_states, encoder_hidden_states_t5], dim=1
        )
        text_embedding_mask = torch.cat(
            [text_embedding_mask, text_embedding_mask_t5], dim=-1
        )
        text_embedding_mask = text_embedding_mask.unsqueeze(2).bool()
        encoder_hidden_states = torch.where(
            text_embedding_mask, encoder_hidden_states, self.text_embedding_padding
        )

        # 4. Ulysses: shard image sequence across ranks
        S = hidden_states.shape[1]
        ulysses_size = self._ulysses_size
        ulysses_group = self._ulysses_group
        rank = dist.get_rank(ulysses_group)
        if S % ulysses_size != 0:
            raise ValueError(
                f"HunyuanDiT Ulysses: sequence length {S} is not divisible by "
                f"ulysses_size={ulysses_size}. Adjust the image resolution so that "
                f"(height // vae_scale_factor // patch_size)^2 is divisible by ulysses_size."
            )
        shard_size = S // ulysses_size
        hidden_states = hidden_states[:, rank * shard_size : (rank + 1) * shard_size, :].contiguous()

        # Shard the RoPE frequencies to match the sequence shard
        if image_rotary_emb is not None:
            freqs_cos, freqs_sin = image_rotary_emb
            freqs_cos = freqs_cos[rank * shard_size : (rank + 1) * shard_size]
            freqs_sin = freqs_sin[rank * shard_size : (rank + 1) * shard_size]
            image_rotary_emb = (freqs_cos, freqs_sin)

        # 5. Transformer blocks (U-Net-style skip connections)
        skips = []
        for layer, block in enumerate(self.blocks):
            if layer > self.config.num_layers // 2:
                if controlnet_block_samples is not None:
                    skip = skips.pop() + controlnet_block_samples.pop()
                else:
                    skip = skips.pop()
                hidden_states = block(
                    hidden_states,
                    temb=temb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_rotary_emb=image_rotary_emb,
                    skip=skip,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    temb=temb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_rotary_emb=image_rotary_emb,
                )

            if layer < (self.config.num_layers // 2 - 1):
                skips.append(hidden_states)

        if controlnet_block_samples is not None and len(controlnet_block_samples) != 0:
            raise ValueError(
                "The number of controls is not equal to the number of skip connections."
            )

        # 6. Ulysses: gather sequence shards → [B, S, D]
        gathered = [torch.zeros_like(hidden_states) for _ in range(ulysses_size)]
        dist.all_gather(gathered, hidden_states.contiguous(), group=ulysses_group)
        hidden_states = torch.cat(gathered, dim=1)

        # 7. Final norm + projection
        hidden_states = self.norm_out(hidden_states, temb.to(torch.float32))
        hidden_states = self.proj_out(hidden_states)

        # 8. Unpatchify → [B, out_channels, H, W]
        patch_size = self.pos_embed.patch_size
        h_out = height // patch_size
        w_out = width // patch_size
        hidden_states = hidden_states.reshape(
            hidden_states.shape[0], h_out, w_out, patch_size, patch_size, self.out_channels
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            hidden_states.shape[0], self.out_channels, h_out * patch_size, w_out * patch_size
        )

        if not return_dict:
            return (output,)

        from diffusers.models.modeling_outputs import Transformer2DModelOutput

        return Transformer2DModelOutput(sample=output)


# ---------------------------------------------------------------------------
# Wrapper
# ---------------------------------------------------------------------------


class HunyuanDiT2DModelWrapper(nn.Module):
    """Thin TRT-LLM wrapper around diffusers ``HunyuanDiT2DModel``.

    Selects ``HunyuanDiT2DModelUlysses``-patched model when
    ``visual_gen_mapping.ulysses_size > 1``.

    Args:
        model_config: ``DiffusionModelConfig`` from the pipeline.
        **transformer_kwargs: Config overrides for ``HunyuanDiT2DModel``.
    """

    # Published HunyuanDiT-v1.2 config defaults
    _DEFAULTS: Dict[str, Any] = {
        "num_attention_heads": 16,
        "attention_head_dim": 88,
        "in_channels": 4,
        "patch_size": 2,
        "activation_fn": "gelu-approximate",
        "num_layers": 40,
        "use_linear_projection": False,
        "cross_attention_dim": 1024,
        "cross_attention_dim_t5": 2048,
        "pooled_projection_dim": 1024,
        "text_len": 77,
        "text_len_t5": 256,
        "norm_type": "ada_norm_continous",
        "sample_size": 128,
    }

    def __init__(self, model_config, **transformer_kwargs):
        super().__init__()
        self.model_config = model_config

        # Merge defaults → pretrained_config → caller overrides
        cfg: Dict[str, Any] = dict(self._DEFAULTS)
        pretrained = getattr(model_config, "pretrained_config", None)
        if pretrained is not None:
            src = pretrained if isinstance(pretrained, dict) else vars(pretrained)
            for k in self._DEFAULTS:
                if k in src:
                    cfg[k] = src[k]
        cfg.update(transformer_kwargs)

        try:
            from diffusers.models import HunyuanDiT2DModel
        except ImportError as exc:
            raise ImportError(
                "HunyuanDiT requires diffusers >= 0.26 (`pip install -U diffusers`)."
            ) from exc

        # Read Ulysses config
        vgm = getattr(model_config, "visual_gen_mapping", None)
        ulysses_size = vgm.ulysses_size if vgm is not None else 1

        num_heads = cfg["num_attention_heads"]
        if ulysses_size > 1 and num_heads % ulysses_size != 0:
            raise ValueError(
                f"HunyuanDiT: num_attention_heads ({num_heads}) must be divisible by "
                f"ulysses_size ({ulysses_size})."
            )

        logger.info(
            "Building HunyuanDiT2DModel: %d layers, %d heads, head_dim=%d, ulysses=%d",
            cfg["num_layers"],
            num_heads,
            cfg["attention_head_dim"],
            ulysses_size,
        )
        self.transformer = HunyuanDiT2DModel(**cfg)
        self.in_channels = cfg["in_channels"]

        # Patch model with Ulysses-aware forward when requested
        if ulysses_size > 1:
            ulysses_group = vgm.ulysses_group
            if ulysses_group is None:
                raise RuntimeError(
                    "HunyuanDiT Ulysses requires vgm.ulysses_group to be initialised "
                    "(call VisualGenMapping.init_device_mesh first)."
                )
            HunyuanDiT2DModelUlysses.patch(self.transformer, ulysses_group, ulysses_size)
            logger.info("HunyuanDiT: Ulysses sequence parallelism enabled (size=%d)", ulysses_size)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    def load_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        result = self.transformer.load_state_dict(weights, strict=False)
        if result.missing_keys:
            logger.warning(
                "HunyuanDiT: %d missing keys (first 10: %s)",
                len(result.missing_keys),
                result.missing_keys[:10],
            )
        if result.unexpected_keys:
            logger.warning(
                "HunyuanDiT: %d unexpected keys (first 10: %s)",
                len(result.unexpected_keys),
                result.unexpected_keys[:10],
            )

    def to_inference_dtype(self):
        dtype = getattr(self.model_config, "torch_dtype", torch.bfloat16)
        self.transformer.to(dtype)
        return self

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        text_embedding_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states_t5: Optional[torch.Tensor] = None,
        text_embedding_mask_t5: Optional[torch.Tensor] = None,
        image_meta_size: Optional[torch.Tensor] = None,
        style: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_dict: bool = True,
        **kwargs,
    ):
        return self.transformer(
            hidden_states=hidden_states,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            text_embedding_mask=text_embedding_mask,
            encoder_hidden_states_t5=encoder_hidden_states_t5,
            text_embedding_mask_t5=text_embedding_mask_t5,
            image_meta_size=image_meta_size,
            style=style,
            image_rotary_emb=image_rotary_emb,
            return_dict=return_dict,
        )
