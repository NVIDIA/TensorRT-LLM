# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the attn_res_add_rmsnorm_persistent_fwd family op.

A cell is (T, N, addend or not, sm_reserve). Every cell checks both outputs' shapes and dtypes, the bf16 sum bit for
bit (or, without an addend, that the first result is layer_residual itself), bit-identical results from two identical
calls, untouched inputs, and closeness of the normalized output to an fp32 torch evaluation under the metric of
test_modeling_v2_attn_res_rmsnorm_fwd.py.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.norm.attn_res_add_rmsnorm_persistent_fwd import (  # noqa: E501
    attn_res_add_rmsnorm_persistent_fwd,
)

__extra_import_path__ = [".", "../../../.."]
from _native_attn_res_inputs import HIDDEN_SIZE, bits, make_inputs, similarity
from _validating import validating

assert torch.cuda.is_available(), "attn_res_add_rmsnorm_persistent_fwd requires a CUDA device"
pytestmark = pytest.mark.skipif(
    torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="certified on sm_100 and sm_103",
)

MIN_COSINE = 0.9999
MAX_RELATIVE_L2 = 5e-3
OP = attn_res_add_rmsnorm_persistent_fwd


@pytest.mark.parametrize("cell", OP.CELLS, ids=[c.why[:48] for c in OP.CELLS])
def test_certified_cells(cell) -> None:
    spec = cell.spec
    x = make_inputs(cell)
    addend = x["addend"] if spec["add"] else None
    inputs = (
        x["layer_residual"],
        addend,
        x["block_residual"],
        x["res_weight"],
        x["rms_weight"],
        x["output_rms_weight"],
    )
    tensors = [t for t in inputs if t is not None]
    before = [t.clone() for t in tensors]
    rest = (spec["eps"], spec["output_eps"], spec["sm_reserve"])
    with validating(OP):
        updated, output = OP(*inputs, *rest)
        updated_again, output_again = OP(*inputs, *rest)
    expected_updated, expected = OP.reference(*inputs, *rest)

    shape = (spec["tokens"], 1, HIDDEN_SIZE)
    for name, t in (("updated", updated), ("output", output)):
        assert t.shape == shape and t.dtype == torch.bfloat16, f"{cell.why}: {name}"
        assert t.is_contiguous() and t.device == inputs[0].device, f"{cell.why}: {name}"
    if addend is None:
        assert updated.data_ptr() == x["layer_residual"].data_ptr(), f"{cell.why}: not the input"
    else:
        assert torch.equal(bits(updated), bits(expected_updated)), f"{cell.why}: the sum differs"
        assert torch.equal(bits(updated), bits(updated_again)), f"{cell.why}: two calls disagree"
    assert torch.equal(bits(output), bits(output_again)), f"{cell.why}: two calls disagree"
    cosine, relative_l2 = similarity(output, expected)
    assert cosine > MIN_COSINE and relative_l2 < MAX_RELATIVE_L2, (
        f"{cell.why}: cosine {cosine:.6f}, relative L2 {relative_l2:.3e}"
    )
    for index, (old, new) in enumerate(zip(before, tensors)):
        assert torch.equal(bits(old), bits(new)), f"{cell.why}: input {index} was mutated"


def test_cuda_graph_replay_matches_eager() -> None:
    """A call captured in a CUDA graph replays the eager call's bits."""
    cell = OP.CELLS[0]
    x = make_inputs(cell)
    args = (
        x["layer_residual"],
        x["addend"],
        x["block_residual"],
        x["res_weight"],
        x["rms_weight"],
        x["output_rms_weight"],
        1e-6,
        1e-6,
    )
    eager = OP(*args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = OP(*args)
    graph.replay()
    for t in captured:
        t.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    for expected, actual in zip(eager, captured):
        assert torch.equal(bits(actual), bits(expected))
    del captured


def test_rejects_out_of_contract() -> None:
    x = make_inputs(OP.CELLS[0])
    weights = (x["res_weight"], x["rms_weight"], x["output_rms_weight"], 1e-6, 1e-6)
    single = x["block_residual"][:0]
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    bad_calls = {
        "N == 1": (x["layer_residual"], None, single, *weights, 0),
        "sm_reserve == SM count": (
            x["layer_residual"],
            None,
            x["block_residual"],
            *weights,
            sm_count,
        ),
        "negative sm_reserve": (x["layer_residual"], None, x["block_residual"], *weights, -1),
    }
    for name, args in bad_calls.items():
        try:
            OP(*args)
        except RuntimeError:
            continue
        raise AssertionError(f"{name}: the op accepted an out-of-contract call")
