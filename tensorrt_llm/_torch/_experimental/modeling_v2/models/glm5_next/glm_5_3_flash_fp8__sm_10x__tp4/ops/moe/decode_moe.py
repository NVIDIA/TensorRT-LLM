# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The MoE branch of a decode step of at most 4 rows in one persistent CUDA launch (after the router GEMM).

Every CTA (one per SM) computes the routing itself (sigmoid scores, FP32 correction bias, top-8,
normalized weights times the routed scaling factor) and derives the same static schedule from it,
while its producer warp already streams the shared expert, whose weights do not depend on the
routing. FC1 tiles (4 up and 4 gate rows of one expert over H, 32 KB) go round robin over the CTAs,
the CTAs without FC2 rows taking the tail; the compute warps multiply them with the rows, converted
to f16, on the tensor cores (FP32 accumulation, each 128 x 128 block's partial times its scale) and
the epilogue warps apply the clamped SwiGLU and the routing weight and publish the activation. e4m3
converts to f16 exactly; a BF16 element does for magnitudes in [2^-14, 65504], while a smaller one
rounds to f16's subnormal grid (an absolute error of at most 2^-25) and a larger one overflows.
Both callers pass RMS-normalized layer inputs (the decoder layer's and the MTP layer's), far inside
that range. For FC2, each of 128 CTAs owns 32 output columns and streams every distinct expert's
down-projection tile, adding the expert's activation once its FC1 tiles are complete, so FC2
overlaps the FC1 tail; the experts are summed in a fixed order in FP32 before the single BF16
rounding. The rank's result is the moe_tp partial the caller all-reduces. The kernel is in
``csrc/decodeMoe.cu``.

The BF16 activations are never quantized: the FC1 -> FC2 activations travel as an f16 hi / lo
pair, so this computes the reference product more closely than the FP8-activation grouped GEMM
it replaces at these sizes. With programmatic dependent launch (``TRTLLM_ENABLE_PDL``) only the
weights are read before the grid-dependency wait.
"""

from __future__ import annotations

import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import (
    Arch,
    Cell,
    OpWrapper,
    true_fp32_matmul,
)
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl

from ... import native_ops

MAX_ROWS = 4
_TOPK = 8


class DecodeMoe(OpWrapper):
    """``scaling * sum_i w_i * expert_{e_i}(x) + shared(x)`` for each of at most 4 rows.

    ``logits`` are the router's FP32 ``[M, E]`` outputs; ``bias`` the FP32 correction bias.
    ``w31`` / ``s31`` are the routed experts' ``[E, 2I, H]`` FP8 ``[up; gate]`` weights and
    ``[E, 2I/128, H/128]`` scales, ``w2`` / ``s2`` the ``[E, H, I]`` down weights and scales;
    ``sw31``, ``ss31``, ``sw2``, ``ss2`` the shared expert in the same per-expert layout.
    Returns BF16 ``[M, H]``.
    """

    ARCHS = frozenset({Arch.SM_100})

    # GLM-5.3-Flash at moe_tp4: hidden 4096, intermediate 2048 / 4 = 512, 288 experts.
    CELLS: tuple[Cell, ...] = (
        Cell(why="MTP draft row", spec=dict(m=1, e=288, h=4096, i=512)),
        Cell(why="MTP-3 verify of one request", spec=dict(m=4, e=288, h=4096, i=512)),
    )

    note = """
    Top-8 ties go to the lower expert index. The scores are computed as the routing kernels of the
    larger steps' grouped GEMMs compute them (the tanh form of the sigmoid, plus the FP32 bias), so
    near-equal experts order alike on both sides of the 4-row switch. The kernel is compiled for
    H = 4096, I = 512 and E = 288 (moe_tp4) and holds at most 4 rows; larger steps are the caller's
    to route to the grouped GEMMs. Its FC1 -> FC2 scratch and counters are device-global (the
    counters double-buffered by call parity), so calls on one device must be ordered on one stream.
    Its CTAs, one per SM, wait on each other, so they must all be resident at once: the op raises
    when the device cannot hold the grid, and work that keeps SMs busy while waiting on this launch
    would hang it. Each distinct expert of the step is read from HBM once, so the time follows the
    number of distinct experts the rows select (one request's MTP-3 verify rows choose 14-33 of
    them, shared expert included).
    """

    def __call__(self, x, logits, bias, scaling, limit, w31, s31, w2, s2, sw31, ss31, sw2, ss2):
        out = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
        native_ops.decode_moe(
            x,
            logits,
            bias,
            scaling,
            limit,
            w31,
            s31,
            w2,
            s2,
            sw31,
            ss31,
            sw2,
            ss2,
            out,
            get_env_enable_pdl(),
        )
        return out

    def reference(self, x, logits, bias, scaling, limit, w31, s31, w2, s2, sw31, ss31, sw2, ss2):
        def dequant(w, s):
            r, c = w.shape
            return (w.float().view(r // 128, 128, c // 128, 128) * s[:, None, :, None]).view(r, c)

        def expert(w_up_gate, s_up_gate, w_down, s_down, row):
            h = dequant(w_up_gate, s_up_gate) @ row
            inter = h.shape[0] // 2
            u, g = h[:inter].clamp(-limit, limit), h[inter:].clamp(max=limit)
            return dequant(w_down, s_down) @ (torch.nn.functional.silu(g) * u)

        out = []
        with true_fp32_matmul():
            score = torch.sigmoid(logits.float())
            ids = torch.topk(score + bias.float(), _TOPK, dim=-1).indices
            wts = score.gather(1, ids)
            wts = wts / wts.sum(-1, keepdim=True) * scaling
            for t in range(x.shape[0]):
                row = x[t].float()
                acc = expert(sw31, ss31, sw2, ss2, row)
                for i in range(_TOPK):
                    e = int(ids[t, i])
                    acc = acc + wts[t, i] * expert(w31[e], s31[e], w2[e], s2[e], row)
                out.append(acc)
        return torch.stack(out).to(torch.bfloat16)

    def is_valid(self, x, logits, bias, scaling, limit, w31, s31, w2, s2, sw31, ss31, sw2, ss2):
        m, h = x.shape
        assert 1 <= m <= MAX_ROWS, f"{m} rows; at most {MAX_ROWS}"
        assert x.dtype == torch.bfloat16 and x.is_contiguous()
        assert (
            logits.dtype == torch.float32 and logits.is_contiguous() and bias.dtype == torch.float32
        )
        for t in (w31, w2, sw31, sw2):
            assert t.dtype == torch.float8_e4m3fn and t.is_contiguous()
        for t in (s31, s2, ss31, ss2):
            assert t.dtype == torch.float32 and t.is_contiguous()
        assert h == 4096 and logits.shape[1] == 288 and w2.shape[2] == 512, (
            "the kernel is compiled for H 4096, E 288, I 512"
        )


decode_moe = DecodeMoe()
