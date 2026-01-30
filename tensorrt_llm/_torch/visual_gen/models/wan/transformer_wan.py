import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps
from tqdm import tqdm
from transformers.modeling_utils import get_parameter_device

from tensorrt_llm._torch.modules.layer_norm import LayerNorm
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.modules.mlp import MLP
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.visual_gen.config import DiffusionModelConfig
from tensorrt_llm._torch.visual_gen.modules.attention import Attention, QKVMode
from tensorrt_llm._torch.visual_gen.parallelism import setup_sequence_parallelism
from tensorrt_llm._torch.visual_gen.quantization.loader import DynamicLinearWeightLoader
from tensorrt_llm.logger import logger
from tensorrt_llm.models.modeling_utils import QuantConfig

# =========================================================================
# 1. Rotary Positional Embeddings
# =========================================================================


class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size: Tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.patch_size = patch_size

        # Split logic matches Hugging Face exactly
        self.h_dim = 2 * (attention_head_dim // 6)
        self.w_dim = 2 * (attention_head_dim // 6)
        self.t_dim = attention_head_dim - self.h_dim - self.w_dim

        freqs_cos, freqs_sin = [], []

        # Order: Time, Height, Width
        for dim in [self.t_dim, self.h_dim, self.w_dim]:
            # High precision generation
            freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
            t = torch.arange(max_seq_len, dtype=torch.float64)
            freqs = torch.outer(t, freqs)

            # Interleaved Pattern [c0, c0, c1, c1]
            freqs_cos.append(freqs.cos().repeat_interleave(2, dim=-1).float())
            freqs_sin.append(freqs.sin().repeat_interleave(2, dim=-1).float())

        self.register_buffer("freqs_cos", torch.cat(freqs_cos, dim=1), persistent=False)
        self.register_buffer("freqs_sin", torch.cat(freqs_sin, dim=1), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Robust shape unpacking
        b, c, f, h, w = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = f // p_t, h // p_h, w // p_w

        split_sizes = [self.t_dim, self.h_dim, self.w_dim]
        freqs_cos = self.freqs_cos.split(split_sizes, dim=1)
        freqs_sin = self.freqs_sin.split(split_sizes, dim=1)

        # Broadcast frequencies to 3D grid: [Time, Height, Width]
        f_cos_t = freqs_cos[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        f_sin_t = freqs_sin[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)

        f_cos_h = freqs_cos[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        f_sin_h = freqs_sin[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)

        f_cos_w = freqs_cos[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)
        f_sin_w = freqs_sin[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        # Concatenate and flatten for Attention [1, SeqLen, 1, Dim] (SHD format)
        # New Attention module applies RoPE in [B, S, H, D] layout before reshaping to [B, H, S, D]
        return (
            torch.cat([f_cos_t, f_cos_h, f_cos_w], dim=-1).flatten(0, 2).unsqueeze(0).unsqueeze(2),
            torch.cat([f_sin_t, f_sin_h, f_sin_w], dim=-1).flatten(0, 2).unsqueeze(0).unsqueeze(2),
        )


# =========================================================================
# 2. Embeddings & Attention
# =========================================================================


class WanImageEmbedding(nn.Module):
    """Image embedding for I2V models (Wan 2.1/2.2)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        pos_embed_seq_len: int = None,
        model_config: DiffusionModelConfig = None,
    ):
        super().__init__()
        dtype = model_config.torch_dtype if model_config else None

        self.norm1 = LayerNorm(
            hidden_size=in_features, eps=1e-6, dtype=dtype, has_weights=True, has_bias=True
        )

        # Match HF FeedForward structure: Linear(in, in) → GELU → Linear(in, out)
        self.ff_in = Linear(
            in_features,
            in_features,
            bias=True,
            dtype=dtype,
            mapping=model_config.mapping if model_config else None,
            quant_config=model_config.quant_config if model_config else None,
            skip_create_weights_in_init=model_config.skip_create_weights_in_init
            if model_config
            else False,
            force_dynamic_quantization=model_config.force_dynamic_quantization
            if model_config
            else False,
        )
        self.ff_out = Linear(
            in_features,
            out_features,
            bias=True,
            dtype=dtype,
            mapping=model_config.mapping if model_config else None,
            quant_config=model_config.quant_config if model_config else None,
            skip_create_weights_in_init=model_config.skip_create_weights_in_init
            if model_config
            else False,
            force_dynamic_quantization=model_config.force_dynamic_quantization
            if model_config
            else False,
        )

        self.norm2 = LayerNorm(
            hidden_size=out_features, eps=1e-6, dtype=dtype, has_weights=True, has_bias=True
        )

        if pos_embed_seq_len is not None:
            self.pos_embed = nn.Parameter(torch.zeros(1, pos_embed_seq_len, in_features))
        else:
            self.pos_embed = None

    def forward(self, encoder_hidden_states_image: torch.Tensor) -> torch.Tensor:
        if self.pos_embed is not None:
            batch_size, seq_len, embed_dim = encoder_hidden_states_image.shape
            encoder_hidden_states_image = encoder_hidden_states_image.view(
                -1, 2 * seq_len, embed_dim
            )
            encoder_hidden_states_image = encoder_hidden_states_image + self.pos_embed

        hidden_states = self.norm1(encoder_hidden_states_image)
        hidden_states = self.ff_in(hidden_states)
        hidden_states = F.gelu(hidden_states)
        hidden_states = self.ff_out(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states


class WanTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        time_freq_dim,
        time_proj_dim,
        text_embed_dim,
        model_config: DiffusionModelConfig,
        image_embed_dim: int = None,
        pos_embed_seq_len: int = None,
    ):
        super().__init__()
        dtype = model_config.torch_dtype
        quant_config = model_config.quant_config
        skip_create_weights = model_config.skip_create_weights_in_init
        force_dynamic_quant = model_config.force_dynamic_quantization

        self.timesteps_proj = Timesteps(
            num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()

        self.time_proj = Linear(
            dim,
            time_proj_dim,
            dtype=dtype,
            mapping=model_config.mapping,
            quant_config=quant_config,
            skip_create_weights_in_init=skip_create_weights,
            force_dynamic_quantization=force_dynamic_quant,
        )
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = WanImageEmbedding(
                image_embed_dim, dim, pos_embed_seq_len=pos_embed_seq_len, model_config=model_config
            )

    def forward(self, timestep, encoder_hidden_states, encoder_hidden_states_image=None):
        timestep = self.timesteps_proj(timestep)

        # Get time_embedder dtype
        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype not in [
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ]:
            timestep = timestep.to(time_embedder_dtype)

        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)

        temb_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)

        if encoder_hidden_states_image is not None and self.image_embedder is not None:
            encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)

        return temb, temb_proj, encoder_hidden_states, encoder_hidden_states_image


class WanBlock(nn.Module):
    def __init__(
        self,
        model_config: DiffusionModelConfig,
        _layer_idx: int,
        added_kv_proj_dim: int = None,
    ):
        super().__init__()
        config = model_config.pretrained_config

        if hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
        elif hasattr(config, "attention_head_dim") and hasattr(config, "num_attention_heads"):
            hidden_size = config.attention_head_dim * config.num_attention_heads
        else:
            hidden_size = 1536

        # Wan 2.1 1.3B defaults
        num_heads = getattr(config, "num_attention_heads", 12)
        head_dim = getattr(config, "attention_head_dim", 128)
        ffn_dim = getattr(config, "ffn_dim", 8960)
        eps = getattr(config, "eps", 1e-6)

        dtype = model_config.torch_dtype
        quant_config = model_config.quant_config
        skip_create_weights = model_config.skip_create_weights_in_init
        force_dynamic_quant = model_config.force_dynamic_quantization

        # Store for I2V reshaping logic
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.norm1 = LayerNorm(
            hidden_size=hidden_size, eps=eps, dtype=dtype, has_weights=False, has_bias=False
        )

        # Self-attention with fused QKV
        self.attn1 = Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            head_dim=head_dim,
            qkv_mode=QKVMode.FUSE_QKV,
            qk_norm=True,
            eps=eps,
            config=model_config,
            layer_idx=_layer_idx,
        )

        # Cross-attention with separate Q, K, V
        self.attn2 = Attention(
            hidden_size=hidden_size,
            num_attention_heads=num_heads,
            head_dim=head_dim,
            qkv_mode=QKVMode.SEPARATE_QKV,
            qk_norm=True,
            eps=eps,
            config=model_config,
            layer_idx=_layer_idx,
        )

        self.norm2 = LayerNorm(
            hidden_size=hidden_size, eps=eps, dtype=dtype, has_weights=True, has_bias=True
        )
        self.norm3 = LayerNorm(
            hidden_size=hidden_size, eps=eps, dtype=dtype, has_weights=False, has_bias=False
        )

        self.ffn = MLP(
            hidden_size=hidden_size,
            intermediate_size=ffn_dim,
            bias=True,
            activation=lambda x: F.gelu(x, approximate="tanh"),
            dtype=dtype,
            config=model_config,
            layer_idx=_layer_idx,
            reduce_output=False,
        )

        # I2V: Additional K/V projections for image embeddings
        self.add_k_proj = self.add_v_proj = None
        self.norm_added_k = None
        if added_kv_proj_dim is not None:
            self.add_k_proj = Linear(
                added_kv_proj_dim,
                hidden_size,
                dtype=dtype,
                mapping=model_config.mapping,
                quant_config=quant_config,
                skip_create_weights_in_init=skip_create_weights,
                force_dynamic_quantization=force_dynamic_quant,
            )
            self.add_v_proj = Linear(
                added_kv_proj_dim,
                hidden_size,
                dtype=dtype,
                mapping=model_config.mapping,
                quant_config=quant_config,
                skip_create_weights_in_init=skip_create_weights,
                force_dynamic_quantization=force_dynamic_quant,
            )
            self.norm_added_k = RMSNorm(
                hidden_size=hidden_size, eps=eps, dtype=dtype, has_weights=True
            )

        # Use torch.empty().normal_(std=...) instead of torch.randn()/scale for MetaInitMode compatibility
        self.scale_shift_table = nn.Parameter(
            torch.empty(1, 6, hidden_size).normal_(std=hidden_size**-0.5)
        )

    def forward(
        self,
        x,
        encoder_hidden_states,
        temb,
        freqs_cos,
        freqs_sin,
    ):
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
            self.scale_shift_table.float() + temb.float()
        ).chunk(6, dim=1)

        normed = self.norm1(x.float()) * (1 + scale_msa) + shift_msa
        normed = normed.to(x.dtype)

        # Prepare frequencies for Attention
        freqs = (freqs_cos, freqs_sin) if freqs_cos is not None and freqs_sin is not None else None

        # Self-attention with RoPE
        x = (
            x.float()
            + self.attn1(
                normed,
                freqs=freqs,
            ).float()
            * gate_msa
        ).to(x.dtype)

        norm_x = self.norm2(x.float()).to(x.dtype)

        # I2V: Split encoder_hidden_states into image and text parts if needed
        encoder_hidden_states_img = None
        encoder_hidden_states_text = encoder_hidden_states
        if self.add_k_proj is not None:
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states_text = encoder_hidden_states[:, image_context_length:]

        # Text cross-attention
        attn2_output = self.attn2(norm_x, encoder_hidden_states=encoder_hidden_states_text)

        # I2V: Additional image cross-attention if image embeddings are present
        if encoder_hidden_states_img is not None:
            batch_size, seq_len = norm_x.shape[:2]

            query = self.attn2.get_qkv(norm_x, None)[0]  # Q only
            query, _ = self.attn2.apply_qk_norm(query, query)

            key_img = self.add_k_proj(encoder_hidden_states_img)
            value_img = self.add_v_proj(encoder_hidden_states_img)
            key_img = self.norm_added_k(key_img)

            query = query.view(batch_size, seq_len, self.num_heads, self.head_dim)
            key_img = key_img.view(
                batch_size, encoder_hidden_states_img.shape[1], self.num_heads, self.head_dim
            )
            value_img = value_img.view(
                batch_size, encoder_hidden_states_img.shape[1], self.num_heads, self.head_dim
            )

            attn_img_output = self.attn2._attn_impl(
                query,
                key_img,
                value_img,
                batch_size=batch_size,
                seq_len=seq_len,
                kv_seq_len=encoder_hidden_states_img.shape[1],
            )

            attn2_output = attn2_output + attn_img_output

        x = x + attn2_output

        # 3. Feed-forward
        normed = self.norm3(x.float()) * (1 + c_scale_msa) + c_shift_msa
        normed = normed.to(x.dtype)

        x = (x.float() + self.ffn(normed).float() * c_gate_msa).to(x.dtype)

        return x


class WanTransformer3DModel(nn.Module):
    _supports_gradient_checkpointing = True

    def __init__(
        self,
        model_config: DiffusionModelConfig,
    ):
        super().__init__()

        self.model_config = model_config

        # Validate no tensor parallelism
        if model_config.parallel.dit_tp_size > 1:
            raise ValueError(
                f"WAN does not support tensor parallelism. "
                f"Got dit_tp_size={model_config.parallel.dit_tp_size}"
            )

        # Setup sequence parallelism (Ulysses)
        num_heads = getattr(model_config.pretrained_config, "num_attention_heads", 12)
        self.use_ulysses, self.ulysses_size, self.ulysses_pg, self.ulysses_rank = (
            setup_sequence_parallelism(
                model_config=model_config,
                num_attention_heads=num_heads,
            )
        )

        config = model_config.pretrained_config

        dtype = model_config.torch_dtype
        quant_config = model_config.quant_config
        skip_create_weights = model_config.skip_create_weights_in_init
        force_dynamic_quant = model_config.force_dynamic_quantization

        if hasattr(config, "hidden_size"):
            hidden_size = config.hidden_size
        elif hasattr(config, "attention_head_dim") and hasattr(config, "num_attention_heads"):
            hidden_size = config.attention_head_dim * config.num_attention_heads
        else:
            hidden_size = 1536  # Wan 1.3B default

        num_layers = getattr(config, "num_layers", 30)
        attention_head_dim = getattr(config, "attention_head_dim", 128)
        in_channels = getattr(config, "in_channels", 16)
        out_channels = getattr(config, "out_channels", 16)
        text_dim = getattr(config, "text_dim", 4096)
        freq_dim = getattr(config, "freq_dim", 256)
        patch_size = getattr(config, "patch_size", [1, 2, 2])
        image_embed_dim = getattr(config, "image_dim", None)  # e.g., 1280 for I2V
        added_kv_proj_dim = getattr(config, "added_kv_proj_dim", None)
        pos_embed_seq_len = getattr(config, "pos_embed_seq_len", None)

        # Calculate FFN Dim
        ffn_dim = getattr(config, "ffn_dim", None)
        if ffn_dim is None:
            ffn_dim = (
                13824
                if hidden_size == 5120
                else (8960 if hidden_size == 1536 else int(hidden_size * 4))
            )

        # Store config for unpatchify and pipeline compatibility
        self.config = type(
            "Config",
            (),
            {
                "patch_size": patch_size,
                "hidden_size": hidden_size,
                "image_dim": image_embed_dim,
                "in_channels": in_channels,
                "out_channels": out_channels,
                "num_layers": num_layers,
            },
        )()

        self.patch_embedding = nn.Conv3d(
            in_channels,
            hidden_size,
            kernel_size=patch_size,
            stride=patch_size,
            dtype=dtype,  # use model's target dtype (bf16)
        )

        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=hidden_size,
            time_freq_dim=freq_dim,
            time_proj_dim=hidden_size * 6,
            text_embed_dim=text_dim,
            model_config=model_config,
            image_embed_dim=image_embed_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )

        self.blocks = nn.ModuleList(
            [
                WanBlock(
                    model_config=model_config,
                    _layer_idx=i,
                    added_kv_proj_dim=added_kv_proj_dim,
                )
                for i in range(num_layers)
            ]
        )

        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, max_seq_len=1024)

        self.norm_out = LayerNorm(
            hidden_size=hidden_size, eps=1e-6, dtype=dtype, has_weights=False, has_bias=False
        )

        self.proj_out = Linear(
            hidden_size,
            out_channels * math.prod(patch_size),
            dtype=dtype,
            mapping=model_config.mapping,
            quant_config=quant_config,
            skip_create_weights_in_init=skip_create_weights,
            force_dynamic_quantization=force_dynamic_quant,
        )
        # Use torch.empty().normal_(std=...) instead of torch.randn()/scale for MetaInitMode compatibility
        self.scale_shift_table = nn.Parameter(
            torch.empty(1, 2, hidden_size).normal_(std=hidden_size**-0.5)
        )

        self.__post_init__()

    @property
    def device(self):
        return get_parameter_device(self)

    def __post_init__(self):
        self.apply_quant_config_exclude_modules()

        for _, module in self.named_modules():
            if callable(getattr(module, "create_weights", None)):
                module.create_weights()

    def apply_quant_config_exclude_modules(self):
        quant_config = self.model_config.quant_config
        if quant_config is None or quant_config.exclude_modules is None:
            return

        kv_cache_quant_algo = quant_config.kv_cache_quant_algo if quant_config else None
        no_quant_config = QuantConfig(kv_cache_quant_algo=kv_cache_quant_algo)

        for name, module in self.named_modules():
            if isinstance(module, Linear):
                is_excluded = quant_config.is_module_excluded_from_quantization(name)
                if is_excluded and getattr(module, "quant_config", None) is not None:
                    module.quant_config = no_quant_config

    def unpatchify(self, x, original_shape):
        N, C, T, H, W = original_shape
        pt, ph, pw = self.config.patch_size
        gt, gh, gw = T // pt, H // ph, W // pw
        # Use output channels instead of input channels for unpatchifying
        out_channels = self.proj_out.out_features // (pt * ph * pw)
        return (
            x.view(N, gt, gh, gw, pt, ph, pw, out_channels)
            .permute(0, 7, 1, 4, 2, 5, 3, 6)
            .reshape(N, out_channels, T, H, W)
        )

    def forward(
        self,
        hidden_states,
        timestep,
        encoder_hidden_states,
        encoder_hidden_states_image=None,
        **kwargs,
    ):
        """
        Forward pass with optional Ulysses sequence parallelism.

        With Ulysses enabled (ulysses_size > 1):
            1. Shard input sequence across ranks: [B, S] -> [B, S/P]
            2. Each block's attention does internal all-to-all for full sequence
            3. Gather output sequence: [B, S/P] -> [B, S]

        When TeaCache is enabled, TeaCacheHook intercepts and replaces this call.
        """
        original_shape = hidden_states.shape
        B, C, T, H, W = original_shape
        pt, ph, pw = self.config.patch_size

        # Generate WAN RoPE frequencies
        freqs_cos, freqs_sin = self.rope(hidden_states)

        # Patchify and flatten: [B, C, T, H, W] -> [B, S, hidden_size]
        x = self.patch_embedding(hidden_states).flatten(2).transpose(1, 2)

        # Shard sequence for Ulysses parallelism: [B, S] -> [B, S/P]
        if self.use_ulysses:
            seq_len = x.shape[1]
            if seq_len % self.ulysses_size != 0:
                raise ValueError(
                    f"Sequence length ({seq_len}) is not divisible by ulysses_size ({self.ulysses_size}). "
                    f"Adjust video dimensions or use a different ulysses_size."
                )

            chunk_size = seq_len // self.ulysses_size
            x = x[:, self.ulysses_rank * chunk_size : (self.ulysses_rank + 1) * chunk_size, :]

            # Shard RoPE frequencies to match sequence sharding
            # RoPE freqs shape: [B, S, ...], so shard along dim 1 (sequence dimension)
            if freqs_cos is not None and freqs_sin is not None:
                freqs_cos = freqs_cos[
                    :, self.ulysses_rank * chunk_size : (self.ulysses_rank + 1) * chunk_size
                ]
                freqs_sin = freqs_sin[
                    :, self.ulysses_rank * chunk_size : (self.ulysses_rank + 1) * chunk_size
                ]

        # Time and text/image embeddings
        temb, temb_proj, encoder_hidden_states, encoder_hidden_states_image = (
            self.condition_embedder(timestep, encoder_hidden_states, encoder_hidden_states_image)
        )
        temb_proj = temb_proj.view(-1, 6, self.config.hidden_size)

        # I2V: Concatenate image and text embeddings if image embeddings are provided
        if encoder_hidden_states_image is not None:
            # Handle CFG: duplicate image embeddings if batch dimension is doubled
            if encoder_hidden_states_image.shape[0] != encoder_hidden_states.shape[0]:
                batch_multiplier = (
                    encoder_hidden_states.shape[0] // encoder_hidden_states_image.shape[0]
                )
                encoder_hidden_states_image = encoder_hidden_states_image.repeat(
                    batch_multiplier, 1, 1
                )
            encoder_hidden_states = torch.cat(
                [encoder_hidden_states_image, encoder_hidden_states], dim=1
            )

        # Transformer blocks (attention handles all-to-all internally for Ulysses)
        for block in self.blocks:
            x = block(
                x,
                encoder_hidden_states,
                temb_proj,
                freqs_cos,
                freqs_sin,
            )

        # Gather sequence from all ranks: [B, S/P] -> [B, S]
        if self.use_ulysses:
            # Ensure tensor is contiguous before all_gather
            x = x.contiguous()
            x_list = [torch.zeros_like(x) for _ in range(self.ulysses_size)]
            torch.distributed.all_gather(x_list, x, group=self.ulysses_pg)
            x = torch.cat(x_list, dim=1)

        # Output projection and unpatchify
        shift, scale = (self.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        x = self.norm_out(x) * (1 + scale) + shift
        x = x.to(hidden_states.dtype)

        return self.unpatchify(self.proj_out(x), original_shape)

    def load_weights(self, weights: dict) -> None:
        # Remap checkpoint keys to match model structure
        remapped_weights = {}
        for key, value in weights.items():
            # Remap transformer block FFN keys
            if ".ffn.net.0.proj." in key:
                new_key = key.replace(".ffn.net.0.proj.", ".ffn.up_proj.")
                remapped_weights[new_key] = value
            elif ".ffn.net.2." in key:
                new_key = key.replace(".ffn.net.2.", ".ffn.down_proj.")
                remapped_weights[new_key] = value
            # Remap image embedder FF keys
            elif ".image_embedder.ff.net.0.proj." in key:
                new_key = key.replace(".image_embedder.ff.net.0.proj.", ".image_embedder.ff_in.")
                remapped_weights[new_key] = value
            elif ".image_embedder.ff.net.2." in key:
                new_key = key.replace(".image_embedder.ff.net.2.", ".image_embedder.ff_out.")
                remapped_weights[new_key] = value
            # Remap I2V attention keys
            elif ".attn2.add_k_proj." in key:
                new_key = key.replace(".attn2.add_k_proj.", ".add_k_proj.")
                remapped_weights[new_key] = value
            elif ".attn2.add_v_proj." in key:
                new_key = key.replace(".attn2.add_v_proj.", ".add_v_proj.")
                remapped_weights[new_key] = value
            elif ".attn2.norm_added_k." in key:
                new_key = key.replace(".attn2.norm_added_k.", ".norm_added_k.")
                remapped_weights[new_key] = value
            else:
                remapped_weights[key] = value

        weights = remapped_weights

        # Handle root-level parameters (filter_weights doesn't work for empty prefix)
        for param_name, param in self._parameters.items():
            if param is not None and param_name in weights:
                param.data.copy_(weights[param_name].to(self.model_config.torch_dtype))

        params_map = {
            "qkv_proj": ["to_q", "to_k", "to_v"],
        }
        loader = DynamicLinearWeightLoader(self.model_config, params_map=params_map)

        for name, module in tqdm(self.named_modules(), desc="Loading weights"):
            if len(module._parameters) == 0:
                continue

            if isinstance(module, Linear):
                weight_dicts = loader.get_linear_weights(module, name, weights)

                if weight_dicts:
                    loader.load_linear_weights(module, name, weight_dicts)
                elif "add_k_proj" in name or "add_v_proj" in name:
                    logger.info(f"[Weight Loading] No weights found for I2V module: {name}")
            else:
                module_weights = loader.filter_weights(name, weights)
                for param_name, param in module._parameters.items():
                    if param is not None and param_name in module_weights:
                        param.data.copy_(
                            module_weights[param_name].to(self.model_config.torch_dtype)
                        )

    def post_load_weights(self) -> None:
        """Call post_load_weights on all Linear modules and convert embedders to target dtype."""
        # Convert condition_embedder components to target dtype
        target_dtype = self.model_config.torch_dtype
        if hasattr(self.condition_embedder, "time_embedder"):
            self.condition_embedder.time_embedder.to(target_dtype)
        if hasattr(self.condition_embedder, "text_embedder"):
            self.condition_embedder.text_embedder.to(target_dtype)

        # Call post_load_weights on all Linear modules
        for _, module in self.named_modules():
            if isinstance(module, Linear):
                module.post_load_weights()
