# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for ops/moe/fp4_block_scale_moe_renorm_routed.

Two claims per cell: the logits-routed call is **bitwise** the two-entry composition
(`renorm_moe_routing_op` -> pre-routed `fp4_block_scale_moe_runner`),
and it sits inside the pre-routed entry's band against this entry's own
fp32 reference. The NVFP4 operand construction (layout, activations, scalars,
the band) uses ordinary shared test helpers. Beyond the cells: the refusal
of an uncertified (num_experts, top_k) pair, the prefill tactic resolution
bitwise the custom op's, and per `DECODE_TACTICS` bucket
that the pinned tactic is a valid runner config and answers bitwise what the
AutoTuner path's untuned (runner-default) tactic answers.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.moe.fp4_block_scale_moe_renorm_routed import (  # noqa: E501
    ACT_SWIGLU,
    CERTIFIED_PAIRS,
    DECODE_TACTICS,
    DECODE_TACTICS_GEOMETRY,
    ROUTING_RENORMALIZE,
    _ckpt,
    _pinned_tactic,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.moe.fp4_block_scale_moe_renorm_routed import (  # noqa: E501
    fp4_block_scale_moe_renorm_routed as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.ops.moe.renorm_moe_routing_op import (  # noqa: E501
    renorm_moe_routing_op,
)

__extra_import_path__ = ["../../../.."]
import _fp4_moe_test_utils as H  # noqa: E402
from _fp4_moe_test_utils import many_expert_layer  # noqa: E402,F401 — fixture by name
from _precision import true_fp32_matmul  # noqa: E402
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "fp4_block_scale_moe_renorm_routed requires a CUDA device"

DEV = "cuda"
G1 = 137.0  # activation global scale, as the pre-routed entry's test draws it


def _routed(args, data, sf, logits, scalars, spec, top_k=None):
    call_output1_scale_scalar, call_output1_scale_gate_scalar, call_output2_scale_scalar = scalars
    return op(
        logits,
        data,
        sf.view(torch.float8_e4m3fn),
        args["gemm1_weights"],
        args["gemm1_weights_scale"],
        args["gemm2_weights"],
        args["gemm2_weights_scale"],
        call_output1_scale_scalar,
        call_output1_scale_gate_scalar,
        call_output2_scale_scalar,
        spec["num_experts"],
        spec["top_k"] if top_k is None else top_k,
        spec["intermediate"],
    )


#: The whole-layer cells (tp1); the windows have their own test file.
_WHOLE = [c for c in op.CELLS if "local_expert_offset" not in c.spec]


@pytest.mark.parametrize("cell", _WHOLE, ids=[c.why[:44] for c in _WHOLE])
def test_certified_cells(cell, certifying, many_expert_layer):  # noqa: F811
    certifying(op)
    spec = cell.spec
    seed = zlib.crc32(cell.why.encode()) % 2**31
    args, ref, data, sf, x, logits, ids, wts, g2, (s1, sg, s2) = H.renorm_operands(
        many_expert_layer(), spec, seed, renorm_moe_routing_op
    )

    y = _routed(args, data, sf, logits, (s1, sg, s2), spec)
    assert y.shape == (spec["tokens"], spec["hidden"]) and y.dtype == torch.bfloat16

    # 1. Bitwise the two-entry composition, on identical operands.
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
    with true_fp32_matmul():
        exp = op.reference(logits, x, spec["top_k"], ref["up"], ref["gate"], ref["down"], g2)
    H._compare(y, exp.to(torch.bfloat16))


def test_uncertified_pair_refused(many_expert_layer):  # noqa: F811
    spec = op.CELLS[0].spec
    assert (spec["num_experts"], 9) not in CERTIFIED_PAIRS
    args, _, data, sf, _, logits, _, _, _, scalars = H.renorm_operands(
        many_expert_layer(), spec, 13, renorm_moe_routing_op
    )
    with validating(op), pytest.raises(AssertionError, match="certified Renormalize"):
        _routed(args, data, sf, logits, scalars, spec, top_k=9)


def test_prefill_tactic_resolution_matches_the_custom_op_bitwise(
    monkeypatch,
    many_expert_layer,  # noqa: F811
):
    """A non-pinned prefill bucket keeps the direct op's routing and tactic."""
    spec = _ckpt(2048)
    args, _, data, sf, _, logits, _, _, _, scalars = H.renorm_operands(
        many_expert_layer(), spec, 821, renorm_moe_routing_op
    )
    module = __import__(op.__class__.__module__, fromlist=["run_renorm_moe"])
    calls = []
    catalog = module._pre_routed

    def recorded(*args, **kwargs):
        calls.append(True)
        return catalog(*args, **kwargs)

    monkeypatch.setattr(module, "_pre_routed", recorded)
    got = _routed(args, data, sf, logits, scalars, spec)
    expected = torch.ops.trtllm.fp4_block_scale_moe_runner(
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
        spec["num_experts"],
        spec["top_k"],
        None,
        None,
        spec["intermediate"],
        0,
        spec["num_experts"],
        None,
        ROUTING_RENORMALIZE,
        True,
        ACT_SWIGLU,
    )[0]
    assert calls == [True]
    assert torch.equal(got, expected)
    assert torch.equal(_routed(args, data, sf, logits, scalars, spec), expected)


@pytest.mark.parametrize("bucket", sorted(DECODE_TACTICS))
def test_decode_tactic_valid_and_bitwise(bucket, many_expert_layer):  # noqa: F811
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
    # None (the AutoTuner path, with a warning) if the runner does not list it as valid here.
    assert _pinned_tactic(bucket, num_experts, top_k, hidden, inter) == DECODE_TACTICS[bucket]
    args, _, data, sf, _, logits, _, _, _, scalars = H.renorm_operands(
        many_expert_layer(), spec, 7000 + bucket, renorm_moe_routing_op
    )
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
