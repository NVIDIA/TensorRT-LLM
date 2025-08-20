from typing import Optional, Type

from ...models.modeling_utils import QuantConfig
from ..flashinfer_utils import IS_FLASHINFER_AVAILABLE
from .interface import AttentionBackend, MLAParams, PositionalEmbeddingParams
from .trtllm import TrtllmAttention
from .vanilla import VanillaAttention


def get_attention_backend(backend_name: str) -> Type[AttentionBackend]:
    if backend_name == "VANILLA":
        return VanillaAttention
    elif backend_name == "TRTLLM":
        return TrtllmAttention
    elif backend_name == "FLASHINFER" and IS_FLASHINFER_AVAILABLE:
        from .flashinfer import FlashInferAttention

        return FlashInferAttention
    elif backend_name == "FLASHINFER_STAR_ATTENTION" and IS_FLASHINFER_AVAILABLE:
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
    quant_config: Optional[QuantConfig] = None,
    q_scaling: Optional[float] = None,
    is_mla_enable: bool = False,
    q_lora_rank: Optional[int] = None,
    kv_lora_rank: Optional[int] = None,
    qk_rope_head_dim: Optional[int] = None,
    qk_nope_head_dim: Optional[int] = None,
    v_head_dim: Optional[int] = None,
    predicted_tokens_per_seq: Optional[int] = 1,
    skip_create_weights_in_init: bool = False,
    attention_chunk_size: Optional[int] = None,
):
    if attention_chunk_size is not None and backend_name.upper() != "TRTLLM":
        raise ValueError(
            f"Backend {backend_name} does not support chunked attention.")

    attn_cls = get_attention_backend(backend_name)

    if is_mla_enable:
        assert attn_cls.support_mla(
        ), f"MLA is not supported for {backend_name} backend"
        assert (q_lora_rank > 0 and kv_lora_rank > 0 and qk_rope_head_dim > 0
                and qk_nope_head_dim > 0 and v_head_dim > 0)
        mla_params = MLAParams(
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            qk_nope_head_dim=qk_nope_head_dim,
            v_head_dim=v_head_dim,
            predicted_tokens_per_seq=predicted_tokens_per_seq,
        )
    else:
        mla_params = None

    return attn_cls(
        layer_idx,
        num_heads,
        head_dim,
        num_kv_heads,
        quant_config=quant_config,
        q_scaling=q_scaling,
        pos_embd_params=pos_embd_params,
        mla_params=mla_params,
        skip_create_weights_in_init=skip_create_weights_in_init,
        attention_chunk_size=attention_chunk_size,
    )
