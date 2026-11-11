"""Blackwell CuTe DSL pathway for DiffVAE (deferred stage-4 + fused stage-5)."""

from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.dsl_kernels.apply_dsl import enable_blackwell_dsl
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.dsl_kernels.attn import (
    NA_SOFTMAX_BOUND_BUFFER,
    DSLAttention,
    na_attention_dsl,
    na_dsl_available,
    require_softmax_bound,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.dsl_kernels.block import (
    FUSED_CTX_BIAS_BUFFER,
    FUSED_CTX_WEIGHT_BUFFER,
    DSLDiffusionNABlock,
    fused_na_block_supported,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.dsl_kernels.chain import (
    DSLChannelLinear,
    DSLDiffusionBlockChain,
    alloc_block_volume,
    linear_into_block_volume,
)
from tensorrt_llm._torch.visual_gen.models.ltx_core.model.video_vae.transformer.dsl_kernels.joint_attn import (
    DSLJointAttention,
    dsl_joint_supported,
    na_attention_joint_dsl,
)

__all__ = [
    "FUSED_CTX_BIAS_BUFFER",
    "FUSED_CTX_WEIGHT_BUFFER",
    "NA_SOFTMAX_BOUND_BUFFER",
    "DSLAttention",
    "DSLChannelLinear",
    "DSLDiffusionBlockChain",
    "DSLDiffusionNABlock",
    "DSLJointAttention",
    "alloc_block_volume",
    "dsl_joint_supported",
    "enable_blackwell_dsl",
    "fused_na_block_supported",
    "linear_into_block_volume",
    "na_attention_dsl",
    "na_attention_joint_dsl",
    "na_dsl_available",
    "require_softmax_bound",
]
