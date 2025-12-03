# Adapted from https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/delta_rule/chunk.py

from typing import Optional

import torch

from tensorrt_llm._torch.modules.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h
from tensorrt_llm._torch.modules.fla.chunk_o import chunk_fwd_o

from .wy_fast import prepare_wy_repr_fwd


def chunk_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: Optional[torch.LongTensor] = None,
):
    # obtain WY representation. u is actually the new v.
    w, u, A = prepare_wy_repr_fwd(
        k=k,
        v=v,
        beta=beta,
        cu_seqlens=cu_seqlens,
    )
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=None,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
    )

    o = chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=None, scale=scale, cu_seqlens=cu_seqlens)

    return o, A, final_state
