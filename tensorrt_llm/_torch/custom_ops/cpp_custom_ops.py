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
