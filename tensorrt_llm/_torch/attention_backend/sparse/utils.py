from typing import TYPE_CHECKING, Type, Union

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.interface import AttentionBackend
    from tensorrt_llm.llmapi.llm_args import \
        SparseAttentionConfig as LlmSparseAttentionConfig
    from tensorrt_llm.visual_gen.args import \
        SparseAttentionConfig as VisualGenSparseAttentionConfig

    from .params import SparseParams

    SparseAttentionConfig = Union[LlmSparseAttentionConfig,
                                  VisualGenSparseAttentionConfig]

# Imports of the concrete backends and cache managers are kept local to
# each function: they pull in `trtllm` and `resource_manager`, which
# import `interface` and would otherwise form an import cycle when this
# package is loaded.


def get_sparse_attn_kv_cache_manager(
        sparse_attention_config: "SparseAttentionConfig") -> type:
    from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager

    from .deepseek_v4 import DeepseekV4CacheManager
    from .dsa import DSACacheManager
    from .minimax_m3 import MiniMaxM3KVCacheManagerV2
    from .rocket import RocketKVCacheManager
    if sparse_attention_config.algorithm == "rocket":
        return RocketKVCacheManager
    elif sparse_attention_config.algorithm == "dsa":
        return DSACacheManager
    elif sparse_attention_config.algorithm == "deepseek_v4":
        return DeepseekV4CacheManager
    elif sparse_attention_config.algorithm == "skip_softmax":
        return KVCacheManager
    elif sparse_attention_config.algorithm == "minimax_m3":
        return MiniMaxM3KVCacheManagerV2
    else:
        raise ValueError(
            f"Unsupported sparse attention algorithm: {sparse_attention_config.algorithm}"
        )


def _resolve_minimax_m3_backend_cls(
        sparse_params: "SparseParams") -> Type["AttentionBackend"]:
    """Pick the Triton or MSA-backed M3 backend class.

    Honours `use_msa` on the lowered `MiniMaxM3SparseParams` (populated
    from the user-facing `sparse_use_msa` flag). Falls back to the Triton
    reference path when the flag is unset.
    """
    from .minimax_m3 import (get_minimax_m3_attention_backend_cls,
                             get_minimax_m3_msa_attention_backend_cls)
    if getattr(sparse_params, "use_msa", False):
        return get_minimax_m3_msa_attention_backend_cls()
    return get_minimax_m3_attention_backend_cls()


def get_vanilla_sparse_attn_attention_backend(
        sparse_params: "SparseParams") -> Type["AttentionBackend"]:
    from .rocket import RocketVanillaAttention
    if sparse_params.algorithm == "rocket":
        return RocketVanillaAttention
    elif sparse_params.algorithm == "minimax_m3":
        return _resolve_minimax_m3_backend_cls(sparse_params)
    else:
        raise ValueError(
            f"Unsupported sparse attention algorithm in vanilla attention backend: {sparse_params.algorithm}"
        )


def get_trtllm_sparse_attn_attention_backend(
        sparse_params: "SparseParams") -> Type["AttentionBackend"]:
    from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention

    from .deepseek_v4 import DeepseekV4TrtllmAttention
    from .dsa import DSATrtllmAttention
    from .rocket import RocketTrtllmAttention
    if sparse_params.algorithm == "rocket":
        return RocketTrtllmAttention
    elif sparse_params.algorithm == "dsa":
        return DSATrtllmAttention
    elif sparse_params.algorithm == "deepseek_v4":
        return DeepseekV4TrtllmAttention
    elif sparse_params.algorithm == "skip_softmax":
        return TrtllmAttention
    elif sparse_params.algorithm == "minimax_m3":
        # The MiniMax-M3 sparse algorithm runs in Python through the
        # model-layer override; this backend exists so the standard
        # `create_attention(...)` dispatch in `Attention.__init__`
        # returns an instantiable AttentionBackend under the trtllm
        # attention backend slot.
        return _resolve_minimax_m3_backend_cls(sparse_params)
    else:
        raise ValueError(
            f"Unsupported sparse attention algorithm in trtllm attention backend: {sparse_params.algorithm}"
        )


def get_flashinfer_sparse_attn_attention_backend(
        sparse_params: "SparseParams") -> Type["AttentionBackend"]:
    if sparse_params.algorithm == "minimax_m3":
        return _resolve_minimax_m3_backend_cls(sparse_params)
    raise ValueError(
        f"Unsupported sparse attention algorithm in flashinfer attention backend: {sparse_params.algorithm}"
    )
