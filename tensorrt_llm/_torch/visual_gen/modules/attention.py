from enum import Enum
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn

from ...modules.linear import Linear, WeightMode, WeightsLoadingConfig
from ...modules.rms_norm import RMSNorm
from ..attention_backend.interface import AttentionTensorLayout
from ..attention_backend.utils import create_attention

if TYPE_CHECKING:
    from ..config import DiffusionModelConfig


class QKVMode(str, Enum):
    FUSE_QKV = "fuse_qkv"
    FUSE_KV = "fuse_kv"
    SEPARATE_QKV = "separate"


# TODO: torch compile
def apply_rotary_emb(
    x: torch.Tensor, freqs_cos: torch.Tensor, freqs_sin: torch.Tensor
) -> torch.Tensor:
    freqs_cos = freqs_cos.to(x.dtype)
    freqs_sin = freqs_sin.to(x.dtype)
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)  # [B, S, H, D/2]

    cos = freqs_cos[..., 0::2]
    sin = freqs_sin[..., 1::2]

    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1).flatten(-2)


class Attention(nn.Module):
    """Attention module for visual generation models."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        qkv_mode: QKVMode = QKVMode.FUSE_QKV,
        qk_norm: bool = True,
        eps: float = 1e-6,  # TODO: remove this, we should add this to the config
        config: Optional["DiffusionModelConfig"] = None,
        layer_idx: Optional[int] = None,
    ):
        super().__init__()

        config = config or DiffusionModelConfig()
        self.dtype = config.torch_dtype
        self.quant_config = config.quant_config
        self.skip_create_weights_in_init = config.skip_create_weights_in_init
        self.force_dynamic_quantization = config.force_dynamic_quantization
        self.mapping = getattr(config, "mapping", None)

        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads or num_attention_heads
        self.head_dim = head_dim or (hidden_size // num_attention_heads)
        self.qkv_mode = QKVMode(qkv_mode) if isinstance(qkv_mode, str) else qkv_mode

        # Select compute backend (orthogonal to parallelism)
        ulysses_size = config.parallel.dit_ulysses_size
        base_backend = config.attention.backend

        if self.qkv_mode == QKVMode.SEPARATE_QKV:
            backend_name = "VANILLA"  # Cross-attention requires VANILLA
        else:
            backend_name = base_backend
        self.attn_backend = backend_name
        self.qk_norm = qk_norm
        self.layer_idx = layer_idx if layer_idx is not None else 0
        self.eps = eps

        self.q_dim = self.num_attention_heads * self.head_dim
        self.kv_dim = self.num_key_value_heads * self.head_dim

        self._init_qkv_proj()

        if self.qk_norm:
            self.norm_q = RMSNorm(
                hidden_size=self.q_dim, eps=self.eps, dtype=self.dtype, has_weights=True
            )
            self.norm_k = RMSNorm(
                hidden_size=self.kv_dim, eps=self.eps, dtype=self.dtype, has_weights=True
            )

        # TODO: Use weight mapper to create just a Linear module
        self.to_out = nn.ModuleList(
            [
                Linear(
                    self.q_dim,
                    self.hidden_size,
                    dtype=self.dtype,
                    mapping=self.mapping,
                    quant_config=self.quant_config,
                    skip_create_weights_in_init=self.skip_create_weights_in_init,
                    force_dynamic_quantization=self.force_dynamic_quantization,
                )
            ]
        )

        # Compute head counts for the backend
        # Ulysses shards heads across workers; inner backend sees sharded count
        if ulysses_size > 1 and self.qkv_mode != QKVMode.SEPARATE_QKV:
            backend_num_heads = self.num_attention_heads // ulysses_size
            backend_num_kv_heads = self.num_key_value_heads // ulysses_size
        else:
            backend_num_heads = self.num_attention_heads
            backend_num_kv_heads = self.num_key_value_heads

        # Create compute backend
        self.attn = create_attention(
            backend=backend_name,
            layer_idx=self.layer_idx,
            num_heads=backend_num_heads,
            head_dim=self.head_dim,
            num_kv_heads=backend_num_kv_heads,
            quant_config=self.quant_config,
            dtype=self.dtype,
        )

        # Wrap with parallelism strategy (orthogonal to backend choice)
        if ulysses_size > 1 and self.qkv_mode != QKVMode.SEPARATE_QKV:
            from ..attention_backend.parallel import UlyssesAttention

            process_group = getattr(config, "ulysses_process_group", None)
            self.attn = UlyssesAttention(
                inner_backend=self.attn,
                process_group=process_group,
            )

    def _init_qkv_proj(self) -> None:
        if self.qkv_mode == QKVMode.FUSE_QKV:
            qkv_out_dim = self.q_dim + 2 * self.kv_dim
            self.qkv_proj = Linear(
                self.hidden_size,
                qkv_out_dim,
                dtype=self.dtype,
                mapping=self.mapping,
                quant_config=self.quant_config,
                skip_create_weights_in_init=self.skip_create_weights_in_init,
                force_dynamic_quantization=self.force_dynamic_quantization,
                weights_loading_config=WeightsLoadingConfig(
                    weight_mode=WeightMode.FUSED_QKV_LINEAR
                ),
                fused_weight_shard_indices_mapping={
                    "q": (0, self.q_dim),
                    "k": (self.q_dim, self.kv_dim),
                    "v": (self.q_dim + self.kv_dim, self.kv_dim),
                },
            )
        else:
            self.to_q = Linear(
                self.hidden_size,
                self.q_dim,
                dtype=self.dtype,
                mapping=self.mapping,
                quant_config=self.quant_config,
                skip_create_weights_in_init=self.skip_create_weights_in_init,
                force_dynamic_quantization=self.force_dynamic_quantization,
            )
            self.to_k = Linear(
                self.hidden_size,
                self.kv_dim,
                dtype=self.dtype,
                mapping=self.mapping,
                quant_config=self.quant_config,
                skip_create_weights_in_init=self.skip_create_weights_in_init,
                force_dynamic_quantization=self.force_dynamic_quantization,
            )
            self.to_v = Linear(
                self.hidden_size,
                self.kv_dim,
                dtype=self.dtype,
                mapping=self.mapping,
                quant_config=self.quant_config,
                skip_create_weights_in_init=self.skip_create_weights_in_init,
                force_dynamic_quantization=self.force_dynamic_quantization,
            )

    def get_qkv(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.qkv_mode == QKVMode.FUSE_QKV:
            qkv = self.qkv_proj(hidden_states)
            q, k, v = qkv.split([self.q_dim, self.kv_dim, self.kv_dim], dim=-1)
        else:
            kv_source = (
                encoder_hidden_states if encoder_hidden_states is not None else hidden_states
            )
            q = self.to_q(hidden_states)
            k = self.to_k(kv_source)
            v = self.to_v(kv_source)
        return q, k, v

    def apply_qk_norm(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.qk_norm:
            q = self.norm_q(q)
            k = self.norm_k(k)
        return q, k

    def _attn_impl(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        batch_size: Optional[int] = None,
        seq_len: Optional[int] = None,
        kv_seq_len: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Call attention backend with appropriate tensor layout.

        Two layout paths:
        1. HND backends (VANILLA): [B, S, H*D] -> [B, H, S, D]
        2. NHD backends (TRTLLM, UlyssesAttention): [B, S, H*D] -> [B, S, H, D]
        """
        backend_layout = getattr(self.attn, "preferred_layout", AttentionTensorLayout.NHD)

        batch_size = batch_size or q.shape[0]
        seq_len = seq_len or q.shape[1]
        kv_seq_len = kv_seq_len or k.shape[1]

        # Reshape inputs: [B, S, H*D] -> backend's preferred 4D layout
        if backend_layout == AttentionTensorLayout.HND:
            q = q.view(batch_size, -1, self.num_attention_heads, self.head_dim).transpose(1, 2)
            k = k.view(batch_size, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            v = v.view(batch_size, -1, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        else:
            q = q.view(batch_size, -1, self.num_attention_heads, self.head_dim)
            k = k.view(batch_size, -1, self.num_key_value_heads, self.head_dim)
            v = v.view(batch_size, -1, self.num_key_value_heads, self.head_dim)

        # Call backend
        out = self.attn.forward(
            q=q,
            k=k,
            v=v,
            batch_size=batch_size,
            seq_len=seq_len,
            seq_len_kv=kv_seq_len if kv_seq_len != seq_len else None,
        )

        # Flatten back to [B, S, H*D]
        if backend_layout == AttentionTensorLayout.HND:
            return out.transpose(1, 2).flatten(2)
        else:
            return out.flatten(2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        freqs: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        assert hidden_states.ndim == 3, "hidden_states must be a 3D tensor"
        batch_size, seq_len = hidden_states.shape[:2]
        kv_seq_len = (
            encoder_hidden_states.shape[1] if encoder_hidden_states is not None else seq_len
        )

        q, k, v = self.get_qkv(hidden_states, encoder_hidden_states)
        q, k = self.apply_qk_norm(q, k)

        # Apply RoPE if provided (model handles RoPE, not attention backend)
        if freqs is not None:
            freqs_cos, freqs_sin = freqs
            q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim)  # [B, S, H, D]
            k = k.view(batch_size, kv_seq_len, self.num_key_value_heads, self.head_dim)
            q = apply_rotary_emb(q, freqs_cos, freqs_sin)
            k = apply_rotary_emb(k, freqs_cos, freqs_sin)
            q = q.flatten(2)
            k = k.flatten(2)

        out = self._attn_impl(q, k, v, batch_size, seq_len, kv_seq_len)
        out = self.to_out[0](out)
        return out
