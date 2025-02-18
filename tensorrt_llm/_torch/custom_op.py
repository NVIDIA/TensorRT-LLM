import platform
import traceback
from typing import List, Optional, Tuple

import torch

IS_FLASHINFER_AVAIABLE = False

if platform.system() != "Windows":
    try:
        import flashinfer
        IS_FLASHINFER_AVAIABLE = True
    except ImportError:
        traceback.print_exc()
        print(
            "flashinfer is not installed properly, please try pip install or building from source codes"
        )

# TO-DO: Register the custom op by ourselves


def _register_fake():

    @torch.library.register_fake("trtllm::allreduce")
    def _(
        input,
        workspace,
        reduce_fusion_inputs,
        group,
        strategy,
        config,
        op,
        eps,
        affine,
        bias,
        scale,
    ):
        final_output = torch.empty_like(
            input, dtype=torch.float8_e4m3fn if scale else input.dtype)
        inter_output = torch.empty_like(input)
        return final_output, inter_output

    @torch.library.register_fake("trtllm::allgather")
    def _(input, group):
        output_shape = (len(group), *input.shape)
        return input.new_empty(output_shape)

    @torch.library.register_fake("trtllm::cublas_scaled_mm")
    def _(
        mat_a: torch.Tensor,
        mat_b: torch.Tensor,
        scale_a: torch.Tensor,
        scale_b: torch.Tensor,
        bias,
        out_dtype,
        userbuffers_id,
    ):
        shape = [i for i in mat_a.shape]
        shape[-1] = mat_b.shape[-1]
        ret = mat_a.new_empty(shape, dtype=out_dtype)
        return ret

    @torch.library.register_fake("trtllm::attention")
    def _(
        q,
        k,
        v,
        out_dtype,
        workspace,
        sequence_length,
        host_past_key_value_lengths,
        context_lengths,
        host_context_lengths,
        host_request_types,
        kv_cache_block_offsets,
        host_kv_cache_block_offsets,
        host_kv_cache_pool_pointers,
        host_kv_cache_pool_mapping,
        cache_indirection,
        kv_scale_orig_quant,
        kv_scale_quant_orig,
        out_scale,
        rotary_inv_freq,
        rotary_cos_sin,
        q_b_proj,
        kv_b_proj,
        k_b_proj_trans,
        q_b_proj_scale,
        kv_b_proj_scale,
        k_b_proj_trans_scale,
        is_fused_qkv,
        update_kv_cache,
        layer_idx,
        num_heads,
        num_kv_heads,
        head_size,
        tokens_per_block,
        max_num_requests,
        max_context_length,
        attention_window_size,
        sink_token_length,
        beam_width,
        mask_type,
        quant_mode,
        position_embedding_type,
        rotary_embedding_dim,
        rotary_embedding_base,
        rotary_embedding_scale_type,
        rotary_embedding_scale,
        rotary_embedding_short_m_scale,
        rotary_embedding_long_m_scale,
        rotary_embedding_max_positions,
        rotary_embedding_original_max_positions,
        use_paged_context_fmha,
        is_mla_enable,
        q_lora_rank,
        kv_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        is_ptp128c_enabled,
    ):
        output_shape = (q.shape[0], num_heads * head_size)
        return q.new_empty(output_shape, dtype=out_dtype or q.dtype)

    @torch.library.register_fake("trtllm::userbuffers_allreduce_finalize")
    def _(input):
        return torch.empty_like(input)

    @torch.library.register_fake("trtllm::fp8_block_scaling_gemm")
    def _(a, b, a_scale, b_scale):
        m = a.shape[0]
        n = b.shape[0]
        return a.new_empty((m, n))

    @torch.library.register_fake(
        "tensorrt_llm::static_quantize_e4m3_per_tensor")
    def _(input: torch.Tensor, scale: torch.Tensor):
        return torch.empty_like(input).to(torch.float8_e4m3fn), scale


@torch.library.custom_op("trtllm::ub_scaled_mm_allreduce_quant_scaled_mm_op",
                         mutates_args=())
def ub_scaled_mm_allreduce_quant_scaled_mm_op(
    mm0_a: torch.Tensor,
    mm0_b: torch.Tensor,
    mm0_a_scale: torch.Tensor,
    mm0_b_scale: torch.Tensor,
    mm0_bias: Optional[torch.Tensor],
    mm_dtype: torch.dtype,
    residual_in: torch.Tensor,
    gamma: torch.Tensor,
    groups: List[int],
    eps: float,
    scale: torch.Tensor,
    mm1_b: torch.Tensor,
    mm1_b_scale: torch.Tensor,
    mm1_bias: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    mm0_res = torch.ops.trtllm.cublas_scaled_mm(
        mm0_a,
        mm0_b,
        mm0_a_scale,
        mm0_b_scale,
        bias=mm0_bias,
        out_dtype=mm_dtype,
        userbuffers_id=0,
    )
    from tensorrt_llm.functional import AllReduceFusionOp, AllReduceStrategy
    hidden, residual = torch.ops.trtllm.allreduce(
        mm0_res,
        None,
        [residual_in, gamma, scale],
        groups,
        int(AllReduceStrategy.UB),
        0,  # UB ar does not care about AllReduceConfig
        int(AllReduceFusionOp.RESIDUAL_RMS_NORM),
        eps,
        True,
        False,
        True,
    )
    mm1_res = torch.ops.trtllm.cublas_scaled_mm(
        hidden,
        mm1_b.t(),
        scale,
        mm1_b_scale,
        bias=mm1_bias,
        out_dtype=mm_dtype,
        userbuffers_id=-1,
    )
    return mm1_res, residual


@ub_scaled_mm_allreduce_quant_scaled_mm_op.register_fake
def _(
    mm0_a: torch.Tensor,
    mm0_b: torch.Tensor,
    mm0_a_scale: torch.Tensor,
    mm0_b_scale: torch.Tensor,
    mm0_bias: Optional[torch.Tensor],
    mm_dtype: torch.dtype,
    residual_in: torch.Tensor,
    gamma: torch.Tensor,
    groups: List[int],
    eps: float,
    scale: torch.Tensor,
    mm1_b: torch.Tensor,
    mm1_b_scale: torch.Tensor,
    mm1_bias: Optional[torch.Tensor],
):
    shape = [i for i in mm0_a.shape]
    shape[-1] = mm1_b.shape[-1]
    ret = mm0_a.new_empty(shape, dtype=mm_dtype)
    residual = torch.empty_like(residual_in)
    return ret, residual


if IS_FLASHINFER_AVAIABLE:
    from flashinfer.activation import silu_and_mul
    from flashinfer.norm import fused_add_rmsnorm, rmsnorm

    # Warp this into custom op since flashinfer didn't warp it properly and we want to avoid graph break between mlp layer for user buffer optimization
    @torch.library.custom_op("trtllm::flashinfer_silu_and_mul", mutates_args=())
    def flashinfer_silu_and_mul(x: torch.Tensor) -> torch.Tensor:
        return silu_and_mul(x)

    @flashinfer_silu_and_mul.register_fake
    def _(x: torch.Tensor) -> torch.Tensor:
        return torch.empty_like(x).chunk(2, dim=-1)[1]

    # Warp this into custom op since flashinfer provides default value for eps with would produce two different graphs depends on the eps value.
    @torch.library.custom_op("trtllm::flashinfer_rmsnorm", mutates_args=())
    def flashinfer_rmsnorm(input: torch.Tensor, weight: torch.Tensor,
                           eps: float) -> torch.Tensor:
        return rmsnorm(input, weight, eps)

    @flashinfer_rmsnorm.register_fake
    def rmsnorm_fake(input: torch.Tensor, weight: torch.Tensor,
                     eps: float) -> torch.Tensor:
        return torch.empty_like(input)

    @torch.library.custom_op("trtllm::flashinfer_fused_add_rmsnorm",
                             mutates_args=("input", "residual"))
    def flashinfer_fused_add_rmsnorm(input: torch.Tensor,
                                     residual: torch.Tensor,
                                     weight: torch.Tensor, eps: float) -> None:
        fused_add_rmsnorm(input, residual, weight, eps)

    @torch.library.custom_op("trtllm::flashinfer_apply_rope_inplace",
                             mutates_args=("q", "k"))
    def flashinfer_apply_rope_inplace(
        q: torch.Tensor,
        k: torch.Tensor,
        indptr: torch.Tensor,
        offsets: torch.Tensor,
        rotary_dim: Optional[int] = None,
        interleave: bool = False,
        rope_scale: float = 1,
        rope_theta: float = 1e4,
    ) -> None:
        flashinfer.apply_rope_inplace(q, k, indptr, offsets, rotary_dim,
                                      interleave, rope_scale, rope_theta)

    @flashinfer_apply_rope_inplace.register_fake
    def apply_rope_inplace_fake(
        q: torch.Tensor,
        k: torch.Tensor,
        indptr: torch.Tensor,
        offsets: torch.Tensor,
        rotary_dim: Optional[int] = None,
        interleave: bool = False,
        rope_scale: float = 1,
        rope_theta: float = 1e4,
    ):
        return
