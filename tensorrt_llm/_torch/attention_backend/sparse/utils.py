from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from tensorrt_llm.llmapi.llm_args import \
        SparseAttentionConfig as LlmSparseAttentionConfig
    from tensorrt_llm.visual_gen.args import \
        SparseAttentionConfig as VisualGenSparseAttentionConfig

    SparseAttentionConfig = Union[LlmSparseAttentionConfig,
                                  VisualGenSparseAttentionConfig]

# Imports of the concrete backends / cache managers are kept local to each
# function: they pull in ``trtllm`` and ``resource_manager``, which import
# ``interface`` and would otherwise form an import cycle when this package is
# loaded.


def get_sparse_attn_kv_cache_manager(
        sparse_attention_config: "SparseAttentionConfig"):
    from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager

    from .dsa import DSACacheManager
    from .rocket import RocketKVCacheManager
    if sparse_attention_config.algorithm == "rocket":
        return RocketKVCacheManager
    elif sparse_attention_config.algorithm == "dsa":
        return DSACacheManager
    elif sparse_attention_config.algorithm == "skip_softmax":
        return KVCacheManager
    else:
        raise ValueError(
            f"Unsupported sparse attention algorithm: {sparse_attention_config.algorithm}"
        )


def get_vanilla_sparse_attn_attention_backend(
        sparse_attention_config: "SparseAttentionConfig"):
    from .rocket import RocketVanillaAttention
    if sparse_attention_config.algorithm == "rocket":
        return RocketVanillaAttention
    else:
        raise ValueError(
            f"Unsupported sparse attention algorithm in vanilla attention backend: {sparse_attention_config.algorithm}"
        )


def get_trtllm_sparse_attn_attention_backend(
        sparse_attention_config: "SparseAttentionConfig"):
    from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttention

    from .dsa import DSATrtllmAttention
    from .rocket import RocketTrtllmAttention
    if sparse_attention_config.algorithm == "rocket":
        return RocketTrtllmAttention
    elif sparse_attention_config.algorithm == "dsa":
        return DSATrtllmAttention
    elif sparse_attention_config.algorithm == "skip_softmax":
        return TrtllmAttention
    else:
        raise ValueError(
            f"Unsupported sparse attention algorithm in trtllm attention backend: {sparse_attention_config.algorithm}"
        )


def get_flashinfer_sparse_attn_attention_backend(
        sparse_attention_config: "SparseAttentionConfig"):
    raise ValueError(
        f"Unsupported sparse attention algorithm in flashinfer attention backend: {sparse_attention_config.algorithm}"
    )
