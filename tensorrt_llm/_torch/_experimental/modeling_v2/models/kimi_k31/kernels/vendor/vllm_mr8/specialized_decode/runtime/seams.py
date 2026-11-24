# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""Glue of the stock seams the specialized decode runs at wide decode shapes.

``StockAttnRes``: the stock decoder layer's AttnRes op (``nvidia.ops.attn_res``, the
native fused kernel on SM100-family devices) on the decode's prefix / blocks rows, with
the stock layer's prefix semantics, for shapes whose AttnRes runs stock
(``attn_res.prepared.choose``). It returns the op's own output tensor, which the next
role reads directly (no staging copy).

``gate_sigmoid_mul``: the MLA output gate of the stock attention seam (S2) as one
kernel: ``core * sigmoid(gate)`` computed in FP32 and rounded once, the arithmetic of
the stock layer's compiled gate, read from the strided gate columns of the
in-projection rows and written into the rows the o-projection reads.
"""

import torch
import triton
import triton.language as tl

from ...ops.attn_res import attn_res


class StockAttnRes:
    """One AttnRes site of the stock decoder layer (``_pre_attn_norm``,
    ``_post_attn_norm`` or the model's final one), on ``prefix`` [M, H] and ``blocks``
    [M, B, H]. Without a delta (the first layer, a block-write layer's post site) the
    prefix starts from ``delta`` instead, as the stock layer rebinds ``prefix_sum``."""

    def __init__(self, norm, query, output, blocks, write, has_delta, out_eps):
        self.norm, self.query = norm.weight, query.weight.reshape(-1)
        self.output = None if output is None else output.weight
        self.blocks, self.write, self.has_delta = blocks, write, has_delta
        self.eps, self.out_eps = norm.variance_epsilon, out_eps

    @classmethod
    def for_target(cls, decoder, *, has_delta=True):
        write = decoder.block_write_idx if decoder.is_block_write_layer else -1
        output = decoder.input_layernorm
        return cls(
            decoder.self_attention_res_norm, decoder.self_attention_res_proj, output,
            decoder.prev_valid_blocks, write, has_delta, output.variance_epsilon,
        )  # fmt: skip

    @classmethod
    def for_post(cls, decoder):
        write = decoder.is_block_write_layer
        output = decoder.post_attention_layernorm
        return cls(
            decoder.mlp_res_norm, decoder.mlp_res_proj, output,
            decoder.prev_valid_blocks + write, -1, not write, output.variance_epsilon,
        )  # fmt: skip

    @classmethod
    def for_final(cls, backbone):
        return cls(
            backbone.output_attn_res_norm, backbone.output_attn_res_proj, None,
            backbone.num_attn_res_blocks, -1, True, 0.0,
        )  # fmt: skip

    def run(self, delta, prefix, blocks, aux=None):
        """The AttnRes output [M, H] (a new tensor); ``aux``: also store the updated
        prefix rows (the stock aux hidden-state stream)."""
        if not self.has_delta:
            prefix.copy_(delta)
        out = attn_res(
            prefix,
            delta if self.has_delta else None,
            blocks,
            self.norm,
            self.query,
            self.output,
            num_blocks=self.blocks,
            block_write_idx=self.write,
            eps=self.eps,
            output_norm_eps=self.out_eps,
        )
        if aux is not None:
            aux.copy_(prefix)
        return out


@triton.jit
def _gate_kernel(
    core, gate, out, core_stride, gate_stride, out_stride,
    WIDTH: tl.constexpr, BLOCK: tl.constexpr,
):  # fmt: skip
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < WIDTH
    c = tl.load(core + row * core_stride + cols, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(gate + row * gate_stride + cols, mask=mask, other=0.0).to(tl.float32)
    y = c * tl.sigmoid(g)
    tl.store(out + row * out_stride + cols, y.to(out.dtype.element_ty), mask=mask)


def gate_sigmoid_mul(core, gate, out):
    """``out = core * sigmoid(gate)`` (FP32, one rounding); rows may be strided."""
    rows, width = core.shape
    block = triton.next_power_of_2(width)
    _gate_kernel[(rows,)](
        core, gate, out, core.stride(0), gate.stride(0), out.stride(0),
        WIDTH=width, BLOCK=block,
    )  # fmt: skip
    return out


def warm_gate(pitch, width, device):
    """Compile the gate kernel for in-projection rows of ``pitch`` columns outside any
    CUDA graph capture (the seam only runs inside captures)."""
    rows = torch.zeros((2, pitch), dtype=torch.bfloat16, device=device)
    core = torch.zeros((2, width), dtype=torch.bfloat16, device=device)
    gate_sigmoid_mul(core, rows[:, pitch - width :], torch.empty_like(core))
