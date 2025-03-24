from typing import List, Optional, Tuple

import torch


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
        int(AllReduceFusionOp.RESIDUAL_RMS_NORM_QUANT_FP8),
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
