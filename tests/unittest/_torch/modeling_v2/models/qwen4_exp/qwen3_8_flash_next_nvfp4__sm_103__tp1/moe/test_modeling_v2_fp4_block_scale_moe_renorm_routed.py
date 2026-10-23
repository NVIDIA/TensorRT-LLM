# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for ops/moe/fp4_block_scale_moe_renorm_routed.

Two claims per cell: the logits-routed call is **bitwise** the composition it
replaces (`renorm_moe_routing_op` -> pre-routed `fp4_block_scale_moe_runner`),
and it sits inside the pre-routed entry's band against this entry's own
fp32 reference. The NVFP4 operand construction (layout, activations, scalars,
the band) is the pre-routed entry's test's, loaded by path so the two tests
cannot drift apart on what a valid operand is. Beyond the cells: the refusal
of an uncertified (num_experts, top_k) pair, and per `DECODE_TACTICS` bucket
that the pinned tactic is a valid runner config and answers bitwise what the
AutoTuner path's untuned (runner-default) tactic answers.
"""

import importlib.util
import zlib
from pathlib import Path

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.moe.fp4_block_scale_moe_renorm_routed import (  # noqa: E501
    ACT_SWIGLU,
    CERTIFIED_PAIRS,
    DECODE_TACTICS,
    DECODE_TACTICS_GEOMETRY,
    ROUTING_RENORMALIZE,
    _ckpt,
    _pinned_tactic,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.moe.fp4_block_scale_moe_renorm_routed import (  # noqa: E501
    fp4_block_scale_moe_renorm_routed as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1.ops.moe.renorm_moe_routing_op import (  # noqa: E501
    renorm_moe_routing_op,
)

__extra_import_path__ = ["../../../.."]
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
    spec = importlib.util.spec_from_file_location("_fp4_moe_runner_test_helpers", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


H = _helpers()


def _logits(spec, gen):
    t, e = spec["tokens"], spec["num_experts"]
    if spec["dist"] == "coarse":
        v = torch.randint(-1, 2, (t, e), generator=gen, device=DEV).float()
    else:
        v = torch.randn(t, e, generator=gen, device=DEV)
    return v.to(spec["logits_dtype"]).contiguous()


def _operands(spec, seed):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    args, ref = H._many_experts()
    data, sf, x = H._rand_activation(spec["tokens"], spec["hidden"], G1, gen)
    logits = _logits(spec, gen)
    ids, wts = renorm_moe_routing_op(logits, spec["top_k"], torch.bfloat16)
    g2 = H._calibrate_g2(x, ids, ref)
    scalars = H._scalars(spec["num_experts"], G1, g2)
    return args, ref, data, sf, x, logits, ids, wts, g2, scalars


def _routed(args, data, sf, logits, scalars, spec, top_k=None):
    return op(
        logits,
        data,
        sf.view(torch.float8_e4m3fn),
        args["gemm1_weights"],
        args["gemm1_weights_scale"],
        args["gemm2_weights"],
        args["gemm2_weights_scale"],
        *scalars,
        spec["num_experts"],
        spec["top_k"] if top_k is None else top_k,
        spec["intermediate"],
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying):  # noqa: F811
    certifying(op)
    spec = cell.spec
    seed = zlib.crc32(cell.why.encode()) % 2**31
    args, ref, data, sf, x, logits, ids, wts, g2, (s1, sg, s2) = _operands(spec, seed)

    y = _routed(args, data, sf, logits, (s1, sg, s2), spec)
    assert y.shape == (spec["tokens"], spec["hidden"]) and y.dtype == torch.bfloat16

    # 1. Bitwise the composition it replaces, on identical operands.
    y_pre = H._call(
        data,
        sf,
        args,
        spec["num_experts"],
        spec["top_k"],
        topk_ids=ids,
        topk_weights=wts,
        output1_scale_scalar=s1,
        output1_scale_gate_scalar=sg,
        output2_scale_scalar=s2,
    )[0]
    differing = (y.view(torch.int16) != y_pre.view(torch.int16)).any(dim=1).sum().item()
    assert differing == 0, f"{differing} of {spec['tokens']} rows differ from the composition"

    # 2. Its own reference, through the pre-routed entry's gate.
    exp = op.reference(logits, x, spec["top_k"], ref["up"], ref["gate"], ref["down"], g2)
    H._compare(y, exp.to(torch.bfloat16))


def test_uncertified_pair_refused():
    spec = op.CELLS[0].spec
    assert (spec["num_experts"], 9) not in CERTIFIED_PAIRS
    args, _, data, sf, _, logits, _, _, _, scalars = _operands(spec, 13)
    with validating(op), pytest.raises(AssertionError, match="certified Renormalize"):
        _routed(args, data, sf, logits, scalars, spec, top_k=9)


@pytest.mark.parametrize("bucket", sorted(DECODE_TACTICS))
def test_decode_tactic_valid_and_bitwise(bucket):
    """The pinned tactic of each decode bucket is what a call there runs, is a valid config of
    the runner, and gives bitwise the output of the AutoTuner path outside tuning mode (the
    runner's default tactic for the shape)."""
    spec = _ckpt(bucket)
    num_experts, top_k, hidden, inter = DECODE_TACTICS_GEOMETRY
    assert (spec["num_experts"], spec["top_k"], spec["hidden"], spec["intermediate"]) == (
        num_experts,
        top_k,
        hidden,
        inter,
    )
    # Raises if the runner does not list the tactic as valid at this row count.
    assert _pinned_tactic(bucket, num_experts, top_k, hidden, inter) == DECODE_TACTICS[bucket]
    args, _, data, sf, _, logits, _, _, _, scalars = _operands(spec, 7000 + bucket)
    y = _routed(args, data, sf, logits, scalars, spec)
    y_tuner = torch.ops.trtllm.fp4_block_scale_moe_runner(
        logits,
        None,
        data,
        sf.view(torch.float8_e4m3fn),
        args["gemm1_weights"],
        args["gemm1_weights_scale"],
        None,
        None,
        None,
        None,
        args["gemm2_weights"],
        args["gemm2_weights_scale"],
        None,
        *scalars,
        num_experts,
        top_k,
        None,
        None,
        inter,
        0,
        num_experts,
        None,
        ROUTING_RENORMALIZE,
        True,
        ACT_SWIGLU,
    )[0]
    differing = (y.view(torch.int16) != y_tuner.view(torch.int16)).any(dim=1).sum().item()
    assert differing == 0, f"{differing} of {bucket} rows differ from the AutoTuner path"
