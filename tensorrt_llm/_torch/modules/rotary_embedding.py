import math
from typing import List, Literal, Optional, Tuple

import torch
from torch import nn
from transformers import LlamaConfig, PretrainedConfig

from ..attention_backend import IS_FLASHINFER_AVAIABLE, AttentionMetadata


def compute_default_rope_parameters(
    config: PretrainedConfig,
    device: Optional[torch.device] = None,
) -> "Tuple[torch.Tensor, float]":
    """
    Computes the inverse frequencies according to the original RoPE implementation
    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
        seq_len (`int`, *optional*):
            The current sequence length. Unused for this type of RoPE.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
    """
    base = config.rope_theta
    partial_rotary_factor = config.partial_rotary_factor if hasattr(
        config, "partial_rotary_factor") else 1.0
    dim = int((config.hidden_size // config.num_attention_heads) *
              partial_rotary_factor)

    attention_factor = 1.0  # Unused in this type of RoPE

    # Compute the inverse frequencies
    inv_freq = 1.0 / (base**(
        torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim))
    return inv_freq, attention_factor


def compute_llama3_parameters(
        config: LlamaConfig,
        device: "torch.device") -> "Tuple[torch.Tensor, float]":
    """
    Computes the inverse frequencies for llama 3.1.

    Args:
        config ([`~transformers.PretrainedConfig`]):
            The model configuration.
        device (`torch.device`):
            The device to use for initialization of the inverse frequencies.
    Returns:
        Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
        post-processing scaling factor applied to the computed cos/sin.
    """
    # Gets the default RoPE parameters
    inv_freq, attention_factor = compute_default_rope_parameters(config, device)

    factor: int = config.rope_scaling[
        "factor"]  # `8` in the original implementation
    low_freq_factor: int = config.rope_scaling[
        "low_freq_factor"]  # `1` in the original implementation
    high_freq_factor: int = config.rope_scaling[
        "high_freq_factor"]  # `4` in the original implementation
    old_context_len: int = config.rope_scaling[
        "original_max_position_embeddings"]  # `8192` in the original implementation

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / inv_freq
    # wavelen < high_freq_wavelen: do nothing
    # wavelen > low_freq_wavelen: divide by factor
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor,
                                 inv_freq)
    # otherwise: interpolate between the two, using a smooth factor
    smooth_factor = (old_context_len / wavelen -
                     low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed_inv_freq = (
        1 - smooth_factor
    ) * inv_freq_llama / factor + smooth_factor * inv_freq_llama
    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen
                                                        > low_freq_wavelen)
    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq,
                                 inv_freq_llama)

    return inv_freq_llama, attention_factor


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        config: PretrainedConfig,
        *,
        head_dim: int,
        num_attention_heads: int,
        max_position_embeddings: int,
        device: Optional[torch.device] = None,
        rope_type: Literal['default', 'llama3'] = "default",
    ):
        super().__init__()
        self.config = config
        self.head_dim = head_dim
        self.num_attention_heads = num_attention_heads
        self.rope_type = rope_type
        self.max_seq_len_cached = max_position_embeddings
        self.original_max_seq_len = max_position_embeddings

        rope_init_fn = compute_llama3_parameters if rope_type == "llama3" else compute_default_rope_parameters

        inv_freq, self.attention_scaling = rope_init_fn(config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

        rope_percentage = (getattr(config, 'rotary_pct', None)
                           or getattr(config, 'partial_rotary_factor', None)
                           or 1.0)
        self.rotary_dim = (getattr(config, 'rotary_dim', None)
                           or getattr(config, 'rotary_emb_base', None)
                           or int(head_dim * rope_percentage))

    def forward(
        self,
        position_ids: torch.Tensor,
        targets: List[torch.Tensor],
        attn_metadata: Optional[AttentionMetadata] = None
    ) -> List[torch.Tensor]:
        """
        Apply RoPE to any number of target tensors with the same position_ids.
        This is useful if q_len = k_len, in which case we may RoPE q and k with the same cos and sin values.
        However, if k is cached without positional embedding, we need to apply rope to q and k with different values, so we need a separate call for each.
        """
        use_gptj_style_rope = self.config.model_type == "llama4_text"
        if IS_FLASHINFER_AVAIABLE:
            from ..attention_backend import FlashInferAttentionMetadata
            if attn_metadata is not None:
                from ..attention_backend import (FlashInferAttentionMetadata,
                                                 StarAttentionMetadata)
                if isinstance(attn_metadata, StarAttentionMetadata):
                    pass
                elif isinstance(attn_metadata, FlashInferAttentionMetadata):
                    from ..custom_ops import flashinfer_apply_rope_inplace
                    assert len(targets) == 2
                    q = targets[0]
                    seq_len = q.size()[0]
                    q = targets[0].view(seq_len, -1, self.head_dim)
                    k = targets[1].view(seq_len, -1, self.head_dim)
                    rope_theta = self.config.rope_theta
                    if self.rope_type in ['default', 'llama3']:
                        flashinfer_apply_rope_inplace(
                            q,
                            k,
                            attn_metadata.qo_indptr,
                            attn_metadata.cached_token_lens,
                            rope_theta=rope_theta,
                            rotary_dim=self.rotary_dim,
                            interleave=use_gptj_style_rope)
                    else:
                        # TODO(qijun): support apply_llama31_rope_inplace
                        raise ValueError(
                            f'unsupported rope type {self.rope_type}')
                        # rope_scale = self.config.rope_scaling
                        # flashinfer.apply_llama31_rope_inplace(
                        #     q,
                        #     k,
                        #     attn_metadata.qo_indptr,
                        #     attn_metadata.cached_token_lens,
                        #     rope_theta=rope_theta,
                        #     rope_scale=rope_scale["factor"],
                        #     low_freq_factor=rope_scale["low_freq_factor"],
                        #     high_freq_factor=rope_scale["high_freq_factor"],
                        #     old_context_len=rope_scale[
                        #         "original_max_position_embeddings"],
                        # )

                    q = q.view(seq_len, -1)
                    k = k.view(seq_len, -1)
                    return [q, k]

        if use_gptj_style_rope:
            raise ValueError(
                "gptj style RoPE has to go through flashinfer route for correct results."
            )
        # it is assumed all targets are of the same rank
        q_or_k = targets[0]
        remove_input_padding = (len(q_or_k.size()) == 2)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(
            position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = q_or_k.device.type
        device_type = device_type if isinstance(
            device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float()
                     @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        cos = cos.to(dtype=q_or_k.dtype)
        sin = sin.to(dtype=q_or_k.dtype)

        if remove_input_padding:
            bsz = 1
            seq_len, _ = q_or_k.size()
        else:
            bsz, seq_len, _ = q_or_k.size()

        def rope_target(target):
            target = target.view(bsz, seq_len, -1,
                                 self.head_dim).transpose(1, 2)
            target = RotaryEmbedding.apply_rotary_pos_emb(target, cos, sin)
            target = target.transpose(1, 2).contiguous()
            if remove_input_padding:
                target = target.view(seq_len, -1)
            else:
                target = target.view(bsz, seq_len, -1)
            return target

        return [rope_target(target) for target in targets]

    @staticmethod
    def apply_rotary_pos_emb(q_or_k: torch.Tensor,
                             cos: torch.Tensor,
                             sin: torch.Tensor,
                             unsqueeze_dim: int = 1) -> torch.Tensor:
        """Applies Rotary Position Embedding to the query and key tensors.

        Args:
            q_or_k (`torch.Tensor`): The query/key tensor.
            cos (`torch.Tensor`): The cosine part of the rotary embedding.
            sin (`torch.Tensor`): The sine part of the rotary embedding.
            unsqueeze_dim (`int`, *optional*, defaults to 1):
                The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
                sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
                that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
                k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
                cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
                the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
        Returns:
            `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
        """
        cos = cos.unsqueeze(unsqueeze_dim)
        sin = sin.unsqueeze(unsqueeze_dim)

        rot_dim = cos.shape[-1]
        # If q_or_k_pass is empty, rotary pos embedding is applied to all tensor
        q_or_k, q_or_k_pass = q_or_k[..., :rot_dim], q_or_k[..., rot_dim:]

        embed = (q_or_k * cos) + (RotaryEmbedding.rotate_half(q_or_k) * sin)
        return torch.cat((embed, q_or_k_pass), dim=-1)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dims of the input."""
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)
