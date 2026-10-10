# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FLA's chunked KDA (``fla.ops.kda.chunk_kda``): the prefill path of batches below four 64-token chunks."""

from typing import Optional

import torch


def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    use_gate_in_kernel: bool = False,
    use_beta_sigmoid_in_kernel: bool = False,
    safe_gate: bool = False,
    lower_bound: Optional[float] = None,
    state_v_first: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Chunked KDA over a batch of sequences from dense initial states: ``(o, final_state)``.

    The inference arguments of ``fla.ops.kda.chunk_kda``, passed through by name; ``A_log`` and ``dt_bias`` reach
    it as the keyword arguments it reads them from.
    """
    # Imported on the first call, so that importing the catalog does not need FLA.
    from fla.ops.kda import chunk_kda as fla_chunk_kda

    gate_args = {}
    if A_log is not None:
        gate_args["A_log"] = A_log
    if dt_bias is not None:
        gate_args["dt_bias"] = dt_bias
    return fla_chunk_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=output_final_state,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        safe_gate=safe_gate,
        lower_bound=lower_bound,
        state_v_first=state_v_first,
        cu_seqlens=cu_seqlens,
        **gate_args,
    )
