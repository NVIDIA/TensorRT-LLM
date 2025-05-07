from typing import List

import torch

import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils


def _register_fake():

    @torch.library.register_fake("trtllm::allreduce")
    def _(
        input,
        residual,
        norm_weight,
        scale,
        bias,
        workspace,
        group,
        strategy,
        op,
        eps,
    ):
        from tensorrt_llm.functional import AllReduceFusionOp
        if op == int(AllReduceFusionOp.NONE):
            return [torch.empty_like(input)]
        elif op == int(AllReduceFusionOp.RESIDUAL_RMS_NORM):
            norm_out = torch.empty_like(input)
            residual_out = torch.empty_like(input)
            return [norm_out, residual_out]
        elif op == int(AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8):
            quant_out = torch.empty_like(input, dtype=torch.float8_e4m3fn)
            residual_out = torch.empty_like(input)
            return [quant_out, residual_out]
        elif op == int(AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_FP8):
            norm_out = torch.empty_like(input)
            quant_out = torch.empty_like(input, dtype=torch.float8_e4m3fn)
            residual_out = torch.empty_like(input)
            return [norm_out, quant_out, residual_out]
        elif op == int(AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4):
            fp4_shape, scale_shape = fp4_utils.get_fp4_shape(input.shape, 16)
            quant_fp4 = input.new_empty(fp4_shape, dtype=torch.uint8)
            scale_fp4 = input.new_empty(scale_shape, dtype=torch.uint8)
            residual_out = torch.empty_like(input)
            return [quant_fp4, scale_fp4, residual_out]
        elif op == int(AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4):
            fp4_shape, scale_shape = fp4_utils.get_fp4_shape(input.shape, 16)
            quant_fp4 = input.new_empty(fp4_shape, dtype=torch.uint8)
            scale_fp4 = input.new_empty(scale_shape, dtype=torch.uint8)
            norm_out = torch.empty_like(input)
            residual_out = torch.empty_like(input)
            return [norm_out, quant_fp4, scale_fp4, residual_out]
        else:
            return [torch.empty_like(input)]

    @torch.library.register_fake("trtllm::moe_allreduce")
    def _(residual, norm_weight, device_num_experts, scale_input,
          active_experts_token_input, token_input, workspace, rank, nranks,
          eps):
        norm_out = torch.empty_like(token_input)
        residual_out = torch.empty_like(residual)
        return [norm_out, residual_out]

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
        userbuffers_id=False,
    ):
        shape = [i for i in mat_a.shape]
        shape[-1] = mat_b.shape[-1]
        ret = mat_a.new_empty(shape, dtype=out_dtype)
        return ret

    @torch.library.register_fake("trtllm::cublas_mm")
    def _(mat_a, mat_b, bias, out_dtype):
        shape = list(mat_a.shape)
        shape[-1] = mat_b.shape[-1]
        ret = mat_a.new_empty(
            shape, dtype=out_dtype if out_dtype is not None else mat_a.dtype)
        return ret

    @torch.library.register_fake("trtllm::fp4_gemm")
    def _(
        mat1: torch.Tensor,
        mat2: torch.Tensor,
        mat1_scale: torch.Tensor,
        mat2_scale: torch.Tensor,
        global_scale: torch.Tensor,
        sf_use_ue8m0: bool,
        out_dtype=None,
    ):
        shape = list(mat1.shape)
        shape[-1] = mat2.shape[0]
        ret = mat1.new_empty(shape, dtype=out_dtype)
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
        latent_cache,
        q_pe,
        block_ids_per_seq,
        is_fused_qkv,
        update_kv_cache,
        predicted_tokens_per_seq,
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
        q_scaling,
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
        attention_input_type,
        is_mla_enable,
        q_lora_rank,
        kv_lora_rank,
        qk_nope_head_dim,
        qk_rope_head_dim,
        v_head_dim,
        mrope_rotary_cos_sin,
        mrope_position_deltas,
    ):
        output_shape = (q.shape[0], num_heads *
                        v_head_dim if is_mla_enable else num_heads * head_size)
        return q.new_empty(output_shape, dtype=out_dtype or q.dtype)

    @torch.library.register_fake("trtllm::noaux_tc_op")
    def _(scores, scores_with_bias, n_group, topk_group, topk,
          routed_scaling_factor):
        shape = list(scores.shape)
        shape[-1] = topk
        return scores.new_empty(shape,
                                dtype=scores_with_bias.dtype), scores.new_empty(
                                    shape, dtype=torch.int32)

    @torch.library.register_fake("trtllm::userbuffers_allreduce_finalize")
    def _(input, force_applying_finalize):
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

    @torch.library.register_fake("trtllm::deepseek_allreduce_fusion")
    def _(
        input,
        workspace,
        reduce_fusion_inputs,
        rank,
        nranks,
        eps,
        fusion_op,
    ):
        from tensorrt_llm.functional import AllReduceFusionOp
        residual = reduce_fusion_inputs[0]
        if fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4 or fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4:
            sf_vec_size = 16
            quant_shape, scale_shape = fp4_utils.get_fp4_shape(
                input.shape, sf_vec_size)
            if fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_NVFP4:
                return [
                    input.new_empty(quant_shape, dtype=torch.uint8),
                    input.new_empty(scale_shape, dtype=torch.uint8),
                    torch.empty_like(residual)
                ]
            else:
                return [
                    torch.empty_like(input),
                    input.new_empty(quant_shape, dtype=torch.uint8),
                    input.new_empty(scale_shape, dtype=torch.uint8),
                    torch.empty_like(residual)
                ]

        elif fusion_op == AllReduceFusionOp.RESIDUAL_RMS_NORM:
            return [torch.empty_like(input), torch.empty_like(residual)]
        elif fusion_op == AllReduceFusionOp.MOE_ALLREDUCE_RESIDUAL_RMS_NORM:
            return [torch.empty_like(residual), torch.empty_like(residual)]
        else:
            raise ValueError(f"Unsupported fusion op: {fusion_op}")

    @torch.library.register_fake("trtllm::logits_bitmask")
    def _(logits: List[torch.Tensor], bitmask: List[torch.Tensor]):
        pass

    @torch.library.register_fake("trtllm::fp4_quantize")
    def _(
        input: torch.Tensor,
        global_scale: torch.Tensor,
        sf_vec_size: int,
        sf_use_ue8m0=False,
    ):
        output_shape, scale_shape = fp4_utils.get_fp4_shape(
            input.shape, sf_vec_size)

        return (input.new_empty(output_shape, dtype=torch.uint8),
                global_scale.new_empty(scale_shape, dtype=torch.uint8))

    @torch.library.register_fake("trtllm::moe_comm_prepare_indices")
    def _(
        gathered_target_rank_ids: torch.Tensor,
        real_rank_token_count_cum_sum,
        max_token_count_per_rank: int,
        expert_count: int,
        top_k: int,
        ep_rank: int,
        ep_size: int,
    ):
        max_send_ranks_per_token = max(ep_size, top_k)
        local_gather_indices_shape = (max_token_count_per_rank * ep_size, )
        rank_count_cum_sum_shape = (ep_size, )
        send_rank_local_indices_shape = (max_token_count_per_rank *
                                         max_send_ranks_per_token, )
        recv_rank_local_indices_shape = (max_token_count_per_rank * ep_size, )
        backward_recv_rank_local_indices_shape = (max_token_count_per_rank *
                                                  max_send_ranks_per_token, )

        local_gather_indices = gathered_target_rank_ids.new_empty(
            local_gather_indices_shape, dtype=torch.int32)
        send_rank_count_cum_sum = gathered_target_rank_ids.new_empty(
            rank_count_cum_sum_shape, dtype=torch.int32)
        send_rank_local_indices = gathered_target_rank_ids.new_empty(
            send_rank_local_indices_shape, dtype=torch.int32)
        recv_rank_count_cum_sum = gathered_target_rank_ids.new_empty(
            rank_count_cum_sum_shape, dtype=torch.int32)
        recv_rank_local_indices = gathered_target_rank_ids.new_empty(
            recv_rank_local_indices_shape, dtype=torch.int32)
        backward_recv_rank_local_indices = gathered_target_rank_ids.new_empty(
            backward_recv_rank_local_indices_shape, dtype=torch.int32)

        return (local_gather_indices, send_rank_count_cum_sum,
                send_rank_local_indices, recv_rank_count_cum_sum,
                recv_rank_local_indices, backward_recv_rank_local_indices)

    @torch.library.register_fake("trtllm::moe_local_gather")
    def _(
        recv_rank_cum_sum: torch.Tensor,
        local_gather_indices: torch.Tensor,
        gathered_expert_ids: torch.Tensor,
        gathered_scales: torch.Tensor,
        local_expert_ids: torch.Tensor,
        local_scales: torch.Tensor,
        max_token_count_per_rank: int,
        expert_count: int,
        top_k: int,
        ep_rank: int,
        ep_size: int,
    ):
        pass

    @torch.library.register_fake("trtllm::moe_comm")
    def _(
        input: torch.Tensor,
        send_rank_cum_sum: torch.Tensor,
        send_indices: torch.Tensor,
        output: torch.Tensor,
        recv_rank_cum_sum: torch.Tensor,
        recv_indices: torch.Tensor,
        all_workspaces: torch.Tensor,
        ep_rank: int,
        ep_size: int,
    ):
        pass

    @torch.library.register_fake("trtllm::get_moe_commworkspace_size_per_rank")
    def _(ep_size: int):
        return 0

    @torch.library.register_fake("trtllm::set_moe_max_usable_sm_count")
    def _(max_sm_count: int):
        pass

    @torch.library.custom_op("trtllm::group_rms_norm",
                             mutates_args=("outputs", ))
    def group_rms_norm(
        inputs: List[torch.Tensor],
        outputs: List[torch.Tensor],
        weights: List[torch.Tensor],
        eps: float,
        weight_bias: float,
    ) -> None:
        pass

    @group_rms_norm.register_fake
    def _(
        inputs: List[torch.Tensor],
        outputs: List[torch.Tensor],
        weights: List[torch.Tensor],
        eps: float,
        weight_bias: float,
    ) -> List[torch.Tensor]:
        return outputs

    @torch.library.custom_op("trtllm::group_rms_norm_large_batch",
                             mutates_args=("outputs", ))
    def group_rms_norm_large_batch(
        inputs: List[torch.Tensor],
        outputs: List[torch.Tensor],
        weights: List[torch.Tensor],
        eps: float,
        weight_bias: float,
    ) -> None:
        pass

    @group_rms_norm_large_batch.register_fake
    def _(
        inputs: List[torch.Tensor],
        outputs: List[torch.Tensor],
        weights: List[torch.Tensor],
        eps: float,
        weight_bias: float,
    ) -> List[torch.Tensor]:
        return outputs
