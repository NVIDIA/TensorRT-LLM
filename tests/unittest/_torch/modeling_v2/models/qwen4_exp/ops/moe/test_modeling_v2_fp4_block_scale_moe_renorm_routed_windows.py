# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the fp4_block_scale_moe_renorm_routed entry's expert windows (the whole-layer cells
are in test_modeling_v2_fp4_block_scale_moe_renorm_routed.py).

Two claims per cell: the logits-routed call over a rank's expert window is **bitwise**
the two-entry composition (`renorm_moe_routing_op` -> pre-routed
`fp4_block_scale_moe_runner` over the same window), and it sits inside the pre-routed
entry's band against this entry's own fp32 reference. The NVFP4 operand construction
(layout, the 512-expert stack, windows, activations, scalars, the band) is the pre-routed
entry's test's, loaded by path so the two tests cannot drift apart on what a valid operand
is; the tp4 zero-padded stack is built here from the same helpers.
"""

import importlib.util
import zlib
from pathlib import Path

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe.fp4_block_scale_moe_renorm_routed import (  # noqa: E501
    fp4_block_scale_moe_renorm_routed as op,
)
from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.ops.moe.renorm_moe_routing_op import (
    renorm_moe_routing_op,
)

__extra_import_path__ = ["../../../.."]
from _fp4_moe_test_utils import many_expert_layer  # noqa: E402,F401 — fixture by name
from _precision import true_fp32_matmul  # noqa: E402
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "fp4_block_scale_moe_renorm_routed requires a CUDA device"

DEV = "cuda"
G1 = 137.0  # activation global scale, as the pre-routed entry's test draws it


def _helpers():
    path = (
        Path(__file__).resolve().parents[4]
        / "moe"
        / "test_modeling_v2_fp4_block_scale_moe_runner.py"
    )
    spec = importlib.util.spec_from_file_location("_fp4_moe_runner_r4_test_helpers", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H = _helpers()

# tp4 holds each rank's 160-wide slice of the 640 intermediate, stored zero-padded to 256.
_TP4_LOGICAL_INTER = 160
_TP4_PADDED_INTER = 256
_TP4 = None


def _pad_zero(codes: torch.Tensor, sf: torch.Tensor, dim: int, size: int):
    """Extend a `(codes, scales)` stack with zero codes and zero (e4m3 +0.0) block scales
    along `dim` (1 = rows, 2 = the K columns) to `size`."""
    pad_codes = list(codes.shape)
    pad_codes[dim] = size - codes.shape[dim]
    pad_sf = list(sf.shape)
    pad_sf[dim] = pad_codes[dim] if dim == 1 else pad_codes[dim] // H.SV
    codes = torch.cat([codes, torch.zeros(pad_codes, dtype=codes.dtype, device=DEV)], dim=dim)
    sf_u8 = torch.cat(
        [sf.view(torch.uint8), torch.zeros(pad_sf, dtype=torch.uint8, device=DEV)], dim=dim
    )
    return codes.contiguous(), sf_u8.view(torch.float8_e4m3fn).contiguous()


def _pack(codes: torch.Tensor) -> torch.Tensor:
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()


def _tp4_padded():
    """The tp4 layer, built once: kernel-ready operands at I = 256 and reference operands
    at I = 256 (the padded ones the kernel reads)."""
    global _TP4
    if _TP4 is None:
        e, h, i, ip = 512, 2560, _TP4_LOGICAL_INTER, _TP4_PADDED_INTER
        gen = torch.Generator(device=DEV).manual_seed(512160)
        _, up_s, up_c = H._rand_nvfp4(e, i, h, gen)
        _, gt_s, gt_c = H._rand_nvfp4(e, i, h, gen)
        _, dn_s, dn_c = H._rand_nvfp4(e, h, i, gen)
        up_cp, up_sp = _pad_zero(up_c, up_s, 1, ip)
        gt_cp, gt_sp = _pad_zero(gt_c, gt_s, 1, ip)
        dn_cp, dn_sp = _pad_zero(dn_c, dn_s, 2, ip)
        up_p, gt_p, dn_p = _pack(up_cp), _pack(gt_cp), _pack(dn_cp)
        w1, s1 = H._prep_fc1(up_p, gt_p, up_sp, gt_sp)
        w2, s2 = H._prep_fc2(dn_p, dn_sp)
        args = dict(
            gemm1_weights=w1,
            gemm1_weights_scale=s1,
            gemm2_weights=w2,
            gemm2_weights_scale=s2,
            intermediate_size=ip,
        )
        ref = dict(
            up=(up_cp, up_sp),
            gate=(gt_cp, gt_sp),
            down=(dn_cp, dn_sp),
            raw=(up_p, gt_p, up_sp, gt_sp, dn_p, dn_sp),
            hidden=h,
            inter=ip,
            num_experts=e,
            gen=gen,
        )
        _TP4 = (args, ref)
    return _TP4


def _stack(spec, layer):
    """`(args, ref)` of the whole 512-expert stack at the cell's intermediate width."""
    if spec["intermediate"] == 256:
        return _tp4_padded()
    assert spec["intermediate"] == 640, spec["intermediate"]
    return layer()


def _operands(spec, seed, layer):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    args, ref = _stack(spec, layer)
    off, n, e = spec["local_expert_offset"], spec["local_num_experts"], spec["num_experts"]
    data, sf, x = H._rand_activation(spec["tokens"], spec["hidden"], G1, gen)
    logits = torch.randn(spec["tokens"], e, generator=gen, device=DEV).to(spec["logits_dtype"])
    ids, wts = renorm_moe_routing_op(logits, spec["top_k"], torch.bfloat16)
    g2 = H._calibrate_g2(x, ids, ref)
    s1, sg, s2 = H._scalars(e, G1, g2)
    win_args = args if n == e else H._window_args(args, off, n)
    win_ref = ref if n == e else H._window_ref(ref, off, n)
    scalars = H._window_scalars(s1, sg, s2, off, n)
    return win_args, win_ref, data, sf, x, logits, ids, wts, g2, scalars


def _routed(win_args, data, sf, logits, scalars, spec, top_k=None, weights=None):
    w = win_args if weights is None else weights
    return op(
        logits,
        data,
        sf.view(torch.float8_e4m3fn),
        w["gemm1_weights"],
        w["gemm1_weights_scale"],
        w["gemm2_weights"],
        w["gemm2_weights_scale"],
        scalars["output1_scale_scalar"],
        scalars["output1_scale_gate_scalar"],
        scalars["output2_scale_scalar"],
        spec["num_experts"],
        spec["top_k"] if top_k is None else top_k,
        spec["intermediate"],
        spec["local_expert_offset"],
        spec["local_num_experts"],
    )


#: The window cells; the whole-layer cells have their own test file.
_WINDOWS = [c for c in op.CELLS if "local_expert_offset" in c.spec]


@pytest.mark.parametrize("cell", _WINDOWS, ids=[c.why[:44] for c in _WINDOWS])
def test_certified_cells(cell, certifying, many_expert_layer):  # noqa: F811
    certifying(op)
    spec = cell.spec
    seed = zlib.crc32(cell.why.encode()) % 2**31
    win_args, win_ref, data, sf, x, logits, ids, wts, g2, scalars = _operands(
        spec, seed, many_expert_layer
    )
    off, n = spec["local_expert_offset"], spec["local_num_experts"]

    y = _routed(win_args, data, sf, logits, scalars, spec)
    assert y.shape == (spec["tokens"], spec["hidden"]) and y.dtype == torch.bfloat16

    # 1. Bitwise the two-entry composition, on identical operands.
    y_pre = H._call(
        data,
        sf,
        win_args,
        spec["num_experts"],
        spec["top_k"],
        local_expert_offset=off,
        local_num_experts=n,
        topk_ids=ids,
        topk_weights=wts,
        **scalars,
    )[0]
    differing = (y.view(torch.int16) != y_pre.view(torch.int16)).any(dim=1).sum().item()
    assert differing == 0, f"{differing} of {spec['tokens']} rows differ from the composition"

    # 2. Its own reference, through the pre-routed entry's gate.
    with true_fp32_matmul():
        exp = op.reference(
            logits,
            x,
            spec["top_k"],
            win_ref["up"],
            win_ref["gate"],
            win_ref["down"],
            g2,
            local_expert_offset=off,
            local_num_experts=n,
        )
    H._compare(y, exp.to(torch.bfloat16))
    elt, rms = H._dev(y, exp.to(torch.bfloat16))
    print(f"  cell {cell.why[:64]!r}: {elt:.2f} elt / {rms:.2f} rms ulp")


def test_guard_refuses_the_whole_stack_beside_a_window(many_expert_layer):  # noqa: F811
    """The whole 512-expert weight stack passed beside a window's scales: the kernel sizes the
    window from the scales and would read the stack's first experts as the window's;
    `is_valid` refuses it."""
    cell = next(
        c
        for c in _WINDOWS
        if c.spec["intermediate"] == 640 and c.spec["local_num_experts"] < c.spec["num_experts"]
    )
    spec = cell.spec
    win_args, _, data, sf, _, logits, _, _, _, scalars = _operands(spec, 7, many_expert_layer)
    whole, _ = _stack(spec, many_expert_layer)
    weights = dict(
        win_args, gemm1_weights=whole["gemm1_weights"], gemm2_weights=whole["gemm2_weights"]
    )
    message = (
        f"gemm1_weights holds {spec['num_experts']} experts but the window has "
        f"{spec['local_num_experts']}"
    )
    with validating(op), pytest.raises(AssertionError, match=message):
        _routed(win_args, data, sf, logits, scalars, spec, weights=weights)
