# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the sigmoid_mul_nvfp4 entry.

Cells against the reference (torch's sigmoid, native-torch NVFP4) in the band of
`_nvfp4_checks.assert_nvfp4_matches_reference`, with the gate read through its
strided view into a q/gate-interleaved qkv output (saturated, infinite and zeroed
gate values included). The exactness claim on its own: bitwise the in-tree
`fused_sigmoid_mul_inplace` + `fp4_quantize`, whole scale buffer included under a
poisoned allocation; the attention output is not modified; graph replay; the guards.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.activation.fused_sigmoid_mul_inplace import (  # noqa: E501
    fused_sigmoid_mul_inplace,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.activation.sigmoid_mul_nvfp4 import (  # noqa: E501
    sigmoid_mul_nvfp4 as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../..", ".."]
from _nvfp4_checks import (  # noqa: E402
    assert_nvfp4_matches_reference,
    capture,
    corrupt_a_rescaled_block,
    seed_of,
)
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    get_sm_version() != 120, reason="the Qwen3.8-27B sm_120 tp1 entries are certified on sm_120"
)

DEV = "cuda"
HEADS, HEAD_DIM, QKV_WIDTH = 24, 256, 14336
HIDDEN = HEADS * HEAD_DIM

#: As for the SwiGLU kernel: Triton's sigmoid against torch's moves the BF16 product
#: in roughly 1e-4 of the elements, and a block scale only through its maximum.
MAX_SF_FRAC = 1e-3


def _operands(t: int, seed: int):
    """``(attention_output [t, H], gate view [t, heads, head_dim], global_scale)``; the
    gate is the second half of each head's 512 columns of a ``[t, 14336]`` qkv output,
    with saturated (+-30), infinite and zero gates and a zeroed attention block."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    qkv = torch.randn(t, QKV_WIDTH, generator=g, device=DEV) * 3.0
    qkv[:, 256:260] = 30.0
    qkv[:, 768:772] = -30.0
    qkv[0, 1280] = float("inf")
    qkv[:, 1792:1808] = 0.0
    attn = torch.randn(t, HIDDEN, generator=g, device=DEV) * 2.0
    attn[:, 512:528] = 0.0
    qkv, attn = qkv.to(torch.bfloat16), attn.to(torch.bfloat16)
    gate = qkv[:, : 2 * HIDDEN].unflatten(1, (HEADS, 2 * HEAD_DIM))[:, :, HEAD_DIM:]
    assert gate.stride() == (QKV_WIDTH, 2 * HEAD_DIM, 1)
    gs = torch.tensor([(134.4, 6.0 / 0.04)[seed % 2]], dtype=torch.float32, device=DEV)
    return attn, gate, gs


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811
    certifying(op)
    s = cell.spec
    attn, gate, gs = _operands(s["m"], seed_of(cell.why))
    before = attn.clone()
    out = op(attn, gate, gs)
    assert torch.equal(attn, before), "the attention output must not be modified"
    assert out[0].shape == (s["m"], HIDDEN // 2)
    ref = op.reference(attn, gate, gs)
    assert_nvfp4_matches_reference(out, ref, s["m"], HIDDEN, max_sf_frac=MAX_SF_FRAC)


def test_the_check_rejects_corrupt_codes_in_a_rescaled_block() -> None:
    """The NVFP4 check exempts a block whose scale moved from its code gates; its value
    gate must still catch codes that moved with it."""
    attn, gate, gs = _operands(33, seed=3)
    ref = op.reference(attn, gate, gs)
    assert_nvfp4_matches_reference(ref, ref, 33, HIDDEN, max_sf_frac=1.0)
    bad = corrupt_a_rescaled_block(ref, 33, HIDDEN)
    with pytest.raises(AssertionError, match="two-scale bound"):
        assert_nvfp4_matches_reference(bad, ref, 33, HIDDEN, max_sf_frac=1.0)


@pytest.mark.parametrize("t", [33, 129, 2048], ids=lambda t: f"t{t}")
def test_bitwise_sigmoid_mul_then_fp4_quantize(t) -> None:
    attn, gate, gs = _operands(t, seed=t)
    torch.empty(64 << 20, dtype=torch.uint8, device=DEV).fill_(0xAB)
    fp4, sf = op(attn, gate, gs)
    gated = attn.clone()
    fused_sigmoid_mul_inplace(gated, gate)
    want_fp4, want_sf = torch.ops.trtllm.fp4_quantize(gated, gs, 16, False, True)
    assert torch.equal(fp4, want_fp4.view(torch.uint8))
    assert torch.equal(sf, want_sf.view(torch.uint8))


def test_graph_replay_matches_eager() -> None:
    attn, gate, gs = _operands(64, seed=1)
    graph, (fp4, sf) = capture(lambda: op(attn, gate, gs))
    for trial in range(3):
        fresh = _operands(64, seed=30 + trial)
        attn.copy_(fresh[0])
        gate.copy_(fresh[1])
        fp4.fill_(0xFF)
        sf.fill_(0xFF)
        graph.replay()
        torch.cuda.synchronize()
        want = op(attn, gate, gs)
        assert torch.equal(fp4, want[0]) and torch.equal(sf, want[1])


def test_guard_refuses_a_gate_of_another_geometry() -> None:
    attn, _, gs = _operands(40, seed=2)
    gate = torch.zeros(40, 12, HEAD_DIM, dtype=torch.bfloat16, device=DEV)
    with validating(op), pytest.raises(AssertionError, match="heads \\* head_dim"):
        op(attn, gate, gs)
