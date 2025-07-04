from typing import List, Optional

import torch

import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils

from ..._utils import get_sm_version


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
        trigger_completion_at_end,
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

    #MNNVL Allreduce
    @torch.library.register_fake("trtllm::mnnvl_twoshot_allreduce")
    def _(input, buffer, buffer_flags, wait_for_results):
        output = input.new_empty(input.shape)
        return output

    @torch.library.register_fake("trtllm::mnnvl_twoshot_rmsnorm")
    def _(comm_buf, gamma, eps, residual, buffer_flags):
        output = residual.new_empty(residual.shape)
        residual_out = residual.new_empty(residual.shape)
        return [output, residual_out]

    @torch.library.register_fake("trtllm::moe_allreduce")
    def _(residual, norm_weight, device_num_experts, scale_input,
          active_experts_token_input, token_input, workspace, rank, nranks,
          eps):
        norm_out = torch.empty_like(token_input)
        residual_out = torch.empty_like(residual)
        return [norm_out, residual_out]

    @torch.library.register_fake("trtllm::allgather")
    def _(input, sizes, group):
        if sizes is None:
            output_shape = (len(group) * input.shape[0], *input.shape[1:])
        else:
            output_shape = (sum(sizes), *input.shape[1:])
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

    @torch.library.register_fake("trtllm::dsv3_router_gemm_op")
    def _(mat_a, mat_b, bias, out_dtype):
        shape = list(mat_a.shape)
        shape[-1] = mat_b.shape[-1]
        ret = mat_a.new_empty(
            shape, dtype=out_dtype if out_dtype is not None else mat_a.dtype)
        return ret

    @torch.library.register_fake("trtllm::dsv3_fused_a_gemm_op")
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
        return a.new_empty((m, n), dtype=torch.bfloat16)

    @torch.library.register_fake(
        "tensorrt_llm::static_quantize_e4m3_per_tensor")
    def _(input: torch.Tensor, scale: torch.Tensor):
        return torch.empty_like(input).to(torch.float8_e4m3fn), scale

    @torch.library.register_fake("trtllm::logits_bitmask")
    def _(logits: List[torch.Tensor], bitmask: List[torch.Tensor]):
        pass

    @torch.library.register_fake("trtllm::fp4_quantize")
    def _(
        input: torch.Tensor,
        global_scale: torch.Tensor,
        sf_vec_size: int,
        sf_use_ue8m0=False,
        swizzled_layout=True,
    ):
        output_shape, scale_shape = fp4_utils.get_fp4_shape(
            input.shape, sf_vec_size, swizzled_layout)

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

    @torch.library.register_fake("trtllm::moe_load_balance_wait_gpu_stage")
    def _(single_layer_load_balancer_ptr: int):
        return torch.empty((1, ),
                           dtype=torch.int32,
                           device=torch.device("cuda"))

    @torch.library.register_fake("trtllm::moe_load_balance_set_cpu_stage")
    def _(single_layer_load_balancer_ptr: int):
        pass

    @torch.library.register_fake("trtllm::moe_load_balance_statistic")
    def _(gathered_raw_expert_ids: torch.Tensor, enabled: torch.Tensor,
          single_layer_load_balancer_ptr: int, is_first_stage: bool,
          is_last_stage: bool):
        pass

    @torch.library.register_fake(
        "trtllm::moe_hierarchical_statistic_local_device")
    def _(local_raw_expert_ids: torch.Tensor,
          local_expert_token_count: torch.Tensor, enabled: torch.Tensor,
          single_layer_load_balancer_ptr: int, is_first_stage: bool,
          is_last_stage: bool):
        pass

    @torch.library.register_fake("trtllm::moe_hierarchical_statistic_update")
    def _(global_expert_token_count: torch.Tensor, enabled: torch.Tensor,
          single_layer_load_balancer_ptr: int):
        pass

    @torch.library.register_fake("trtllm::moe_load_balance_routing")
    def _(single_layer_load_balancer_ptr: int,
          token_selected_experts: torch.Tensor, offset_by_ep_rank: bool):
        return torch.empty_like(token_selected_experts)

    @torch.library.custom_op("trtllm::group_rms_norm_base",
                             mutates_args=("outputs", ))
    def group_rms_norm_base(
        inputs: List[torch.Tensor],
        outputs: List[torch.Tensor],
        weights: List[torch.Tensor],
        eps: float,
        weight_bias: float,
    ) -> None:
        pass

    @group_rms_norm_base.register_fake
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

    # Use groupRMSNormHeuristic which automatically selects between regular and large batch kernels
    @torch.library.custom_op("trtllm::group_rms_norm_heuristic",
                             mutates_args=("outputs", ))
    def group_rms_norm_heuristic(
        inputs: List[torch.Tensor],
        outputs: List[torch.Tensor],
        weights: List[torch.Tensor],
        eps: float,
        weight_bias: float,
    ) -> None:
        pass

    @group_rms_norm_heuristic.register_fake
    def _(
        inputs: List[torch.Tensor],
        outputs: List[torch.Tensor],
        weights: List[torch.Tensor],
        eps: float,
        weight_bias: float,
    ) -> List[torch.Tensor]:
        return outputs

    @torch.library.register_fake(
        "trtllm::mtp_sampling_and_accepted_draft_tokens_op")
    def _(logits: torch.Tensor, draft_tokens: torch.Tensor,
          target_tokens: torch.Tensor, num_mtp_modules: int, batch_size: int,
          num_context_request: int, vocab_size: int):
        return logits.new_empty((batch_size, num_mtp_modules + 1),
                                dtype=torch.int32), logits.new_empty(
                                    (batch_size, ), dtype=torch.int32)

    @torch.library.register_fake("trtllm::fp8_quantize_1x128")
    def _(input: torch.Tensor):
        pad_m = fp4_utils.pad_up(input.shape[0], 4)
        blocked_n = (input.shape[1] + 127) // 128
        if get_sm_version() >= 100:
            sz = (blocked_n, input.shape[0])
        else:
            sz = (fp4_utils.pad_up(pad_m * blocked_n * 4, 128) // 4, )
        return torch.empty_like(input,
                                dtype=torch.float8_e4m3fn), input.new_empty(
                                    sz, dtype=torch.float)

    @torch.library.register_fake("trtllm::causal_conv1d_fwd")
    def _(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias_: Optional[torch.Tensor],
        conv_states: Optional[torch.Tensor],
        query_start_loc: Optional[torch.Tensor],
        cache_indices: Optional[torch.Tensor],
        has_initial_state: Optional[torch.Tensor],
        silu_activation: bool,
        pad_slot_id: int,
    ) -> None:
        pass

    @torch.library.register_fake("trtllm::causal_conv1d_update")
    def _(
        x: torch.Tensor,
        conv_state: torch.Tensor,
        weight: torch.Tensor,
        bias_: Optional[torch.Tensor],
        silu_activation: bool,
        cache_seqlens_: Optional[torch.Tensor],
        conv_state_indices_: Optional[torch.Tensor],
        pad_slot_id: int,
    ) -> None:
        pass

    @torch.library.register_fake("trtllm::moe_permute_op")
    def _(
        input: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: torch.Tensor,
        fc1_expert_weights: torch.Tensor,
        fc2_expert_weights: torch.Tensor,
        quant_scales: List[torch.Tensor],
        input_sf: Optional[torch.Tensor],
        num_experts_per_node: int,
        tp_size: int,
        tp_rank: int,
        ep_size: int,
        ep_rank: int,
        cluster_size: int,
        cluster_rank: int,
        min_latency_mode: bool,
        use_fp8_block_scaling: bool,
    ):

        experts_per_token = token_selected_experts.shape[1]
        num_rows = input.shape[0]
        hidden_size = input.shape[1]

        num_moe_inputs = experts_per_token * num_rows

        unpermuted_token_selected_experts_tensor = token_selected_experts.new_empty(
            (num_moe_inputs, ), dtype=torch.int32)
        unpermuted_source_token_ids_tensor = token_selected_experts.new_empty(
            (num_moe_inputs, ), dtype=torch.int32)
        permuted_source_token_ids_tensor = token_selected_experts.new_empty(
            (num_moe_inputs, ), dtype=torch.int32)
        permuted_token_selected_experts_tensor = token_selected_experts.new_empty(
            (num_moe_inputs, ), dtype=torch.int32)
        permuted_data_tensor = input.new_empty((num_moe_inputs, hidden_size),
                                               dtype=torch.float32)
        expert_first_token_offset_tensor = token_selected_experts.new_empty(
            (num_experts_per_node + 1, ), dtype=torch.int64)
        permuted_token_final_scales_tensor = token_selected_experts.new_empty(
            (num_moe_inputs, ), dtype=torch.float32)
        src_to_dest_map_tensor = token_selected_experts.new_empty(
            (num_moe_inputs, ), dtype=torch.int32)

        return (
            unpermuted_token_selected_experts_tensor,
            unpermuted_source_token_ids_tensor,
            permuted_source_token_ids_tensor,
            permuted_token_selected_experts_tensor,
            permuted_data_tensor,
            expert_first_token_offset_tensor,
            permuted_token_final_scales_tensor,
            src_to_dest_map_tensor,
        )

    @torch.library.register_fake("trtllm::moe_finalize_scale_op")
    def _(
        gemm2_output: torch.Tensor,
        fc2_expert_biases: torch.Tensor,
        unpermuted_final_scales: torch.Tensor,
        expanded_source_row_to_expanded_dest_row: torch.Tensor,
        expert_for_source_row: torch.Tensor,
        expert_first_token_offset_tensor: torch.Tensor,
        num_rows: torch.SymInt,
        hidden_size: torch.SymInt,
        experts_per_token: int,
        num_experts_per_node: int,
        tp_size: int,
        tp_rank: int,
        ep_size: int,
        ep_rank: int,
    ):
        num_rows_val = int(num_rows)
        hidden_size_val = int(hidden_size)
        return gemm2_output.new_empty((num_rows_val, hidden_size_val),
                                      dtype=gemm2_output.dtype)

    @torch.library.register_fake("trtllm::allgather_list")
    def _(input_list, sizes, group):
        assert len(input_list) > 0

        def create_output_tensor(i):
            shape = list(i.shape)
            if sizes is None:
                shape[0] *= len(group)
            else:
                shape[0] = sum(sizes)
            return i.new_empty(shape)

        return [create_output_tensor(i) for i in input_list]

    @torch.library.register_fake("trtllm::reducescatter")
    def _(input, sizes, group):
        import tensorrt_llm
        local_rank = tensorrt_llm.mpi_rank()

        shape = list(input.shape)
        if sizes is None:
            shape[0] = shape[0] // len(group)
        else:
            shape[0] = sizes[local_rank]
        return input.new_empty(shape)

    @torch.library.register_fake("trtllm::fp4_block_scale_moe_runner")
    def _(
        routing_logits,
        routing_bias,
        hidden_states,
        hidden_states_scale,
        gemm1_weights,
        gemm1_weights_scale,
        gemm2_weights,
        gemm2_weights_scale,
        output1_scale_scalar,
        output1_scale_gate_scalar,
        output2_scale_scalar,
        num_experts,
        top_k,
        n_group,
        topk_group,
        intermediate_size,
        local_expert_offset,
        local_num_experts,
        routed_scaling_factor,
        tile_tokens_dim,
        routing_method_type,
        do_finalize,
    ) -> List[torch.Tensor]:
        num_tokens = hidden_states.shape[0]
        hidden_size = hidden_states.shape[1] * 2
        if do_finalize:
            return [
                hidden_states.new_empty((num_tokens, hidden_size),
                                        dtype=torch.bfloat16)
            ]

        expanded_row_count = num_tokens * top_k
        max_padding_required = (tile_tokens_dim - 1) * num_experts
        max_num_padded_tokens = fp4_utils.pad_up(
            expanded_row_count + max_padding_required, tile_tokens_dim)
        wt_dtype = routing_bias.dtype if routing_bias is not None else torch.bfloat16
        return [
            hidden_states.new_empty((max_num_padded_tokens, hidden_size),
                                    dtype=torch.bfloat16),
            hidden_states.new_empty((num_tokens, top_k), dtype=wt_dtype),
            hidden_states.new_empty((num_tokens, top_k), dtype=torch.int32)
        ]

    @torch.library.register_fake("trtllm::nvfp4_block_scale_interleave")
    def _(sf: torch.Tensor):
        rows = sf.shape[-2]
        cols = sf.shape[-1]
        expert_out_size = fp4_utils.pad_up(rows, 128) * fp4_utils.pad_up(
            cols, 4)
        num_experts = sf.shape[0] if len(sf.shape) == 3 else 1
        return sf.new_empty((num_experts * expert_out_size, ),
                            dtype=torch.uint8)

    @torch.library.register_fake("trtllm::nvfp4_block_scale_interleave_reverse")
    def _(sf: torch.Tensor):
        return torch.empty_like(sf, dtype=torch.uint8)
