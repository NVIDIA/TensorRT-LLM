from typing import Optional, Type

from ...models.modeling_utils import QuantConfig
from . import IS_FLASHINFER_AVAIABLE
from .interface import AttentionBackend, PositionalEmbeddingParams
from .trtllm import TrtllmAttention
from .vanilla import VanillaAttention


def get_attention_backend(backend_name: str) -> Type[AttentionBackend]:
    if backend_name == "VANILLA":
        return VanillaAttention
    elif backend_name == "TRTLLM":
        return TrtllmAttention
    elif backend_name == "FLASHINFER" and IS_FLASHINFER_AVAIABLE:
        from .flashinfer import FlashInferAttention
        return FlashInferAttention
    elif backend_name == "FLASHINFER_STAR_ATTENTION" and IS_FLASHINFER_AVAIABLE:
        from .star_flashinfer import StarAttention
        return StarAttention

    return TrtllmAttention


def create_attention(
        backend_name: str,
        layer_idx: int,
        num_heads: int,
        head_dim: int,
        num_kv_heads: Optional[int] = None,
        pos_embd_params: Optional[PositionalEmbeddingParams] = None,
        quant_config: Optional[QuantConfig] = None):
    attn_cls = get_attention_backend(backend_name)
    if attn_cls == TrtllmAttention:
        return TrtllmAttention(layer_idx, num_heads, head_dim, num_kv_heads,
                               pos_embd_params, quant_config)

    return attn_cls(layer_idx, num_heads, head_dim, num_kv_heads, quant_config)
