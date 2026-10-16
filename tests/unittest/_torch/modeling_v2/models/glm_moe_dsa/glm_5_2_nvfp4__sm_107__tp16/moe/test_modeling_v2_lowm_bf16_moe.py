# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the glm_lowm_bf16_moe entry."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_107__tp16.ops.moe.lowm_bf16_moe import (  # noqa: E501
    glm_lowm_bf16_moe as op,
)
from tensorrt_llm._utils import get_sm_version

__extra_import_path__ = ["../../../.."]
from _validating import (  # noqa: E402,F401 — needs the path declared above
    assert_within_ulp,
    certifying,
    validating,
)

pytestmark = pytest.mark.skipif(
    get_sm_version() != 107, reason="the GLM-5.2 tp16 target's ops are certified on sm_107"
)

NUM_EXPERTS = 256
TOP_K = 8
ROUTED_SCALING_FACTOR = 2.5


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """One accumulation order's worth of BF16 rounding flips.

    The reference rounds at the op's points (FC1, SwiGLU, each scaled expert output, the sum), so the two differ
    only where the FP32 accumulation order of a dot product moves a BF16 rounding of an intermediate.

    Over 6144-term FC1 dots of unit-scale operands the order changes the FP32 sum by a few 1e-6 relative, which
    flips the BF16 rounding of about one gate or up value in a thousand; a flipped activation moves its expert's
    FC2 outputs by about one BF16 ulp of the activation times a 1/sqrt(128)-scale weight, far below one ulp of the
    output. What reaches the output is then the occasional flip of a term's or the sum's own rounding: one ulp of
    that element, which is at most one ulp of the row's scale.

    So: 2 ulp element-wise against the row scale, 0.5 ulp of relative RMS. A wrong expert or weight moves the
    output by about its own size, hundreds of ulp.
    """
    assert_within_ulp(out, ref, element_ulp=2.0, rms_ulp=0.5)


def _build(spec, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    hidden, inter = spec["hidden"], spec["inter"]
    x = torch.randn(1, hidden, generator=g, device="cuda", dtype=torch.bfloat16)
    w3_w1 = torch.randn(
        NUM_EXPERTS, 2 * inter, hidden, generator=g, device="cuda", dtype=torch.bfloat16
    ).mul_(hidden**-0.5)
    w2 = torch.randn(
        NUM_EXPERTS, hidden, inter, generator=g, device="cuda", dtype=torch.bfloat16
    ).mul_(inter**-0.5)
    routing = spec["routing"]
    if routing == "random":
        logits = 2.0 * torch.randn(1, NUM_EXPERTS, generator=g, device="cuda")
        bias = 0.1 * torch.randn(NUM_EXPERTS, generator=g, device="cuda")
    elif routing == "ties":
        # Half-integer logits in [-2, 2] and no bias: about 28 experts share the top key, so which 8 of them are
        # selected, and in which order, is the tie-break.
        logits = torch.randint(-4, 5, (1, NUM_EXPERTS), generator=g, device="cuda").float() / 2.0
        bias = torch.zeros(NUM_EXPERTS, device="cuda")
    else:
        assert routing == "tiny"
        # sigmoid(-16 +- 1) is about 1e-7, so the 8 selected scores sum to about 1e-6 < 2^-13.
        logits = -16.0 + 0.5 * torch.randn(1, NUM_EXPERTS, generator=g, device="cuda")
        bias = 0.1 * torch.randn(NUM_EXPERTS, generator=g, device="cuda")
    return x, logits.contiguous(), bias.float().contiguous(), w3_w1, w2


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — fixture by name
    args = _build(cell.spec, seed=zlib.crc32(cell.why.encode()) % 2**31)
    certifying(op)
    out = op(*args, TOP_K, ROUTED_SCALING_FACTOR)
    torch.cuda.synchronize()
    _compare(out, op.reference(*args, TOP_K, ROUTED_SCALING_FACTOR))


def test_cuda_graph_replay_matches_eager() -> None:
    """The draft loop runs inside the decode graph: a replay with new inputs gives the eager result bitwise."""
    spec = op.CELLS[0].spec
    x, logits, bias, w3_w1, w2 = _build(spec, seed=1)
    static_x, static_logits = x.clone(), logits.clone()
    out = op(static_x, static_logits, bias, w3_w1, w2, TOP_K, ROUTED_SCALING_FACTOR)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_out = op(static_x, static_logits, bias, w3_w1, w2, TOP_K, ROUTED_SCALING_FACTOR)
    x2, logits2, _, _, _ = _build(spec, seed=2)
    static_x.copy_(x2)
    static_logits.copy_(logits2)
    graph.replay()
    eager = op(x2, logits2, bias, w3_w1, w2, TOP_K, ROUTED_SCALING_FACTOR)
    torch.cuda.synchronize()
    assert torch.equal(graph_out, eager)
    assert not torch.equal(out, eager), "the replay must have read the new inputs"


def _small(hidden=2048, inter=64, tokens=1):
    """Operands of the right layout at a small supported shape.

    The weights stay uninitialized: only the refusals, which happen before any launch, are exercised.
    """
    x = torch.zeros(tokens, hidden, dtype=torch.bfloat16, device="cuda")
    logits = torch.zeros(tokens, NUM_EXPERTS, dtype=torch.float32, device="cuda")
    bias = torch.zeros(NUM_EXPERTS, dtype=torch.float32, device="cuda")
    w3_w1 = torch.empty(NUM_EXPERTS, 2 * inter, hidden, dtype=torch.bfloat16, device="cuda")
    w2 = torch.empty(NUM_EXPERTS, hidden, inter, dtype=torch.bfloat16, device="cuda")
    return x, logits, bias, w3_w1, w2


def test_refuses_more_than_one_token() -> None:
    """Draft step 0 without the row select has 8 rows: the caller must take another MoE path.

    The op says so rather than computing token 0 alone.
    """
    with pytest.raises(RuntimeError, match="takes one token"):
        with validating(op):
            op(*_small(tokens=2), TOP_K, ROUTED_SCALING_FACTOR)


def test_refuses_an_expert_slice_it_does_not_serve() -> None:
    """The FC2 kernel walks the expert slice in 64-element K steps."""
    with pytest.raises(RuntimeError, match="unsupported hidden size"):
        with validating(op):
            op(*_small(inter=96), TOP_K, ROUTED_SCALING_FACTOR)


def test_refuses_another_top_k() -> None:
    with pytest.raises(RuntimeError, match="top-8"):
        with validating(op):
            op(*_small(), 6, ROUTED_SCALING_FACTOR)


def test_refuses_a_misaligned_token() -> None:
    """The FC1 kernel reads x in 16-byte pieces."""
    _, logits, bias, w3_w1, w2 = _small()
    storage = torch.zeros(1, 2048 + 1, dtype=torch.bfloat16, device="cuda")
    misaligned = storage[:, 1:]
    assert misaligned.is_contiguous() and misaligned.data_ptr() % 16 != 0
    with pytest.raises(RuntimeError, match="16-byte aligned"):
        with validating(op):
            op(misaligned, logits, bias, w3_w1, w2, TOP_K, ROUTED_SCALING_FACTOR)
