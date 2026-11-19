# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Explicit native-control/fused attention-output projection + gate/residual boundary."""

import types

import breuse_down
import breuse_gateup
import fp8_producers as producers
import round4_attention_kernel as candidate
import torch

STATE = dict(
    enabled=False,
    capture=False,
    calls=0,
    case=None,
    validate=False,
    checks=[],
    fused_calls=0,
    fallback_calls=0,
)


def quantize(raw, scale):
    return torch.ops.tensorrt_llm.static_quantize_e4m3_per_tensor(raw, scale)[0].reshape(
        -1, raw.shape[-1]
    )


def native(a, w, sa, sw, x, norm, mod, idx, sg, eps):
    out = torch.ops.trtllm.cublas_scaled_mm(
        a, w.t(), scale_a=sa, scale_b=sw, bias=None, out_dtype=torch.bfloat16
    ).reshape_as(x)
    return producers.gate_norm(x, out, norm, mod, idx, sg, eps)


def fused(a, w, sa, sw, x, norm, mod, idx, sg, eps):
    if a.shape[0] not in (37820, 37776, 38169):
        STATE["fallback_calls"] += 1
        return native(a, w, sa, sw, x, norm, mod, idx, sg, eps)
    STATE["fused_calls"] += 1
    residual = candidate.run(a, w, sa, sw, x, mod, idx).reshape_as(x)
    codes = producers._launch(residual, None, norm, mod, idx, sg, eps, 0, 4, 3)[1]
    return residual, codes


@torch.library.custom_op("h3_fp8_round4_attention::projection_gate_norm", mutates_args=())
def projection_gate_norm(
    raw: torch.Tensor,
    w: torch.Tensor,
    sa: torch.Tensor,
    sw: torch.Tensor,
    x: torch.Tensor,
    norm: torch.Tensor,
    mod: torch.Tensor,
    idx: torch.Tensor,
    sg: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        raw.shape[-1] != 7168
        or w.shape != (5376, 7168)
        or x.shape[-1] != 5376
        or raw.numel() // 7168 != x.numel() // 5376
    ):
        raise ValueError(
            f"Expected raw attention [*,7168], got raw={tuple(raw.shape)}, "
            f"weight={tuple(w.shape)}, hidden={tuple(x.shape)}"
        )
    STATE["calls"] += 1
    a = quantize(raw, sa)
    if STATE["capture"] and STATE["case"] is None:
        STATE["case"] = dict(
            a=a.detach().clone(),
            w=w,
            sa=sa,
            sw=sw,
            x=x.detach().clone(),
            norm=norm,
            mod=mod.detach().clone(),
            idx=idx.detach().clone(),
            sg=sg,
            eps=eps,
        )
    if STATE["validate"]:
        reference = native(a, w, sa, sw, x, norm, mod, idx, sg, eps)
        value = fused(a, w, sa, sw, x, norm, mod, idx, sg, eps)
        check = dict(
            index=len(STATE["checks"]),
            byte_equal_outputs=[
                bool(
                    torch.equal(p.contiguous().view(torch.uint8), q.contiguous().view(torch.uint8))
                )
                for p, q in zip(reference, value)
            ],
        )
        STATE["checks"].append(check)
        assert all(check["byte_equal_outputs"]), check
        return reference
    return (
        fused(a, w, sa, sw, x, norm, mod, idx, sg, eps)
        if STATE["enabled"]
        else native(a, w, sa, sw, x, norm, mod, idx, sg, eps)
    )


@projection_gate_norm.register_fake
def _(raw, w, sa, sw, x, norm, mod, idx, sg, eps):
    return torch.empty_like(x), torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)


def raw_output(self, hidden_states):
    return hidden_states


def forward(
    self, hidden_states, temb, adaln_indices, rotary_emb, key_padding_mask, timestep, valid_seq_len
):
    mod = self.adaln_proj.table(temb)
    quantized = producers.norm(
        hidden_states,
        self.norm1.weight,
        mod,
        adaln_indices,
        self.attn.qkv_proj.input_scale,
        self.norm1.variance_epsilon,
    )
    raw = self.attn(quantized, rotary_emb, key_padding_mask, timestep, valid_seq_len)
    out = self.attn.to_out[0]
    hidden_states, quantized = projection_gate_norm(
        raw,
        out.weight,
        out.input_scale,
        out.weight_scale,
        hidden_states,
        self.norm2.weight,
        mod,
        adaln_indices,
        self.ff.gate_up_proj.input_scale,
        float(self.norm2.variance_epsilon),
    )
    up, down = self.ff.gate_up_proj, self.ff.down_proj
    codes = breuse_gateup.gateup_swiglu(
        quantized.reshape(-1, quantized.shape[-1]),
        up.weight,
        up.input_scale,
        up.weight_scale,
        down.input_scale,
    )
    return breuse_down.down_residual(
        codes, down.weight, down.input_scale, down.weight_scale, hidden_states, mod, adaln_indices
    )


class Patch:
    def __init__(self, blocks):
        from tensorrt_llm._torch.visual_gen.models.minimax_h3.minimax_h3_nvfp4__sm_107__u1_u4 import (
            modeling,
        )

        self.blocks = blocks
        self.scenario = modeling
        self.original_block = modeling.ScenarioBlock._forward_fused_adaln
        self.original_instance_out = [
            (("_out_proj" in vars(b.attn)), vars(b.attn).get("_out_proj")) for b in blocks
        ]
        self.raw_outputs = [types.MethodType(raw_output, b.attn) for b in blocks]
        for b in blocks:
            assert (
                b.ff.mapping.tp_size == 1
                and b.attn.to_out[0].weight.shape == (5376, 7168)
                and b.attn.to_out[0].bias is None
            )

    def configure(self, enabled):
        # Stable instance-bound methods make the module guard observe projection
        # selection. A class-only _out_proj patch reused a stale compiled projection.
        for b, raw, previous in zip(self.blocks, self.raw_outputs, self.original_instance_out):
            if enabled:
                b.attn._out_proj = raw
            elif previous[0]:
                b.attn._out_proj = previous[1]
            elif "_out_proj" in vars(b.attn):
                delattr(b.attn, "_out_proj")
        self.scenario.ScenarioBlock._forward_fused_adaln = (
            forward if enabled else self.original_block
        )
