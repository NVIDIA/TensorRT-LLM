# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FLA's chunked KDA (``fla.ops.kda.chunk_kda``): the prefill path of batches below four 64-token chunks."""

from typing import Optional

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Arch, Cell, OpWrapper


class ChunkKda(OpWrapper):
    """Chunked KDA over a batch of sequences from dense initial states: ``(o, final_state)``.

    The inference arguments of ``fla.ops.kda.chunk_kda``, passed through by name; ``A_log`` and ``dt_bias`` reach
    it as the keyword arguments it reads them from.
    """

    ARCHS = frozenset({Arch.SM_100, Arch.SM_103})
    CELLS: tuple[Cell, ...] = (
        Cell(
            why="TP16: two ragged sequences mixing fresh and continuing states below four chunks",
            spec=dict(heads=6, seq_lens=(100, 37), fresh=(True, False)),
        ),
        Cell(
            why="TP4: a continuing 150-token sequence at 24 heads",
            spec=dict(heads=24, seq_lens=(150,), fresh=(False,)),
        ),
        Cell(
            why="one fresh token exercises the smallest prefill",
            spec=dict(heads=6, seq_lens=(1,), fresh=(True,)),
        ),
        Cell(
            why="three sequences including a whole 64-token chunk and two short tails",
            spec=dict(heads=6, seq_lens=(17, 5, 64), fresh=(False, True, False)),
        ),
    )
    note = """
    Kimi K3 supplies contiguous bf16 L2-normalized q/k and bf16 v/g with
    K=V=128, fp32 beta/A_log/dt_bias, scale=128**-0.5, safe_gate=True,
    lower_bound=-5, both gate and beta activation enabled, and V-first fp32
    states. cu_seqlens is the caller's int64 cumulative sequence metadata.
    initial_state is not mutated; the caller gathers pool rows, zeroes fresh
    rows (or passes None when all are fresh), and scatters the returned states.
    Projections, convolution, q/k normalization and gated output norm are not
    fused here. The layer sends fewer than four 64-token chunks here and larger
    batches to kda_prefill; the tests also compare their common four-chunk case,
    128 + 64 continuation and its wrong-order control, and None against zero
    initial states.

    FLA is imported on first call, not catalog import. Use inference_mode to
    avoid retaining backward activations. Training/context-parallel arguments,
    intermediate states, allow_neg_eigval and chunk_size are not exposed.
    Kernels autotune once per process: repeated calls are bit-identical within
    one process, but choices can differ between processes. The indexed
    kda_prefill path accepts resident BF16/FP32 pools directly and normalizes
    its raw Q/K inputs internally. This path carries FP32 state; comparison
    bands are 2e-2 for outputs and 1e-2 for final states.
    """

    def __call__(
        self,
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

    def reference(
        self,
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
        qf, kf, vf = q.float(), k.float(), v.float()
        if use_qk_l2norm_in_kernel:
            qf = qf * torch.rsqrt(qf.square().sum(-1, keepdim=True) + 1e-6)
            kf = kf * torch.rsqrt(kf.square().sum(-1, keepdim=True) + 1e-6)
        qf = qf * (q.shape[-1] ** -0.5 if scale is None else scale)
        gate = g.float()
        if use_gate_in_kernel:
            if dt_bias is not None:
                gate = gate + dt_bias.float().reshape(q.shape[-2:])
            rate = A_log.float().exp().reshape(-1, 1)
            gate = (
                lower_bound * torch.sigmoid(rate * gate)
                if safe_gate
                else -rate * torch.nn.functional.softplus(gate)
            )
        decay = gate.exp()
        bf = beta.float().sigmoid() if use_beta_sigmoid_in_kernel else beta.float()
        if cu_seqlens is None:
            spans = [(row, 0, q.shape[1]) for row in range(q.shape[0])]
        else:
            starts = cu_seqlens.tolist()
            spans = [(0, lo, hi) for lo, hi in zip(starts, starts[1:])]
        out = torch.zeros_like(v, dtype=torch.float32)
        final = []
        for seq, (row, lo, hi) in enumerate(spans):
            if initial_state is None:
                state = vf.new_zeros(q.shape[2], v.shape[-1], q.shape[-1])
            else:
                state = initial_state[seq].float()
                if not state_v_first:
                    state = state.transpose(-1, -2)
            for token in range(lo, hi):
                state = state * decay[row, token].unsqueeze(-2)
                key = kf[row, token]
                err = vf[row, token] - (state * key.unsqueeze(-2)).sum(-1)
                state = state + (
                    bf[row, token, :, None, None] * err.unsqueeze(-1) * key.unsqueeze(-2)
                )
                out[row, token] = (state * qf[row, token].unsqueeze(-2)).sum(-1)
            final.append(state if state_v_first else state.transpose(-1, -2))
        return out.to(q.dtype), torch.stack(final) if output_final_state else None


chunk_kda = ChunkKda()
