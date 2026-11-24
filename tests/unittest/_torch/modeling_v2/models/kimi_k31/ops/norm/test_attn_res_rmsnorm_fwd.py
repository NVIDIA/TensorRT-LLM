# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 attn_res_rmsnorm_fwd entry.

A cell is (T, N) with T == 1, the only token count the target fuses the norm at, and N = K + 1
candidates. Every cell checks shapes and dtypes, bit-identical repeated calls, untouched inputs and
closeness to the entry's reference, which rounds where the kernel rounds.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.norm.attn_res_rmsnorm_fwd import (
    attn_res_rmsnorm_fwd,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 3), (10, 7)),
    reason="certified on sm_103 and sm_107",
)

HIDDEN_SIZE = 7168
RMS_EPS = 1e-5
# The reference rounds at the kernel's three rounding points, so what remains is the fast-math
# softmax weights and the summation order inside each candidate's norm. Measured on GB300 over every
# cell: relative L2 at most 6e-6, at most 0.14 bf16 ulp of a row's largest magnitude per element,
# cosine >= 0.9999998. A dropped candidate moves the output by tens of percent.
MIN_COSINE = 0.9999
MAX_RELATIVE_L2 = 1e-3


def _make_inputs(num_tokens: int, num_candidates: int, seed: int) -> tuple[torch.Tensor, ...]:
    gen = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape):
        return torch.randn(*shape, generator=gen, device="cuda")

    layer_residual = (randn(num_tokens, 1, HIDDEN_SIZE) * 0.05).to(torch.bfloat16)
    layer_residual_add = (randn(num_tokens, 1, HIDDEN_SIZE) * 0.05).to(torch.bfloat16)
    block_residual = (randn(num_candidates - 1, num_tokens, 1, HIDDEN_SIZE) * 0.05).to(
        torch.bfloat16
    )
    res_weight = (randn(HIDDEN_SIZE) * 0.02).to(torch.bfloat16)
    rms_weight = (1 + randn(HIDDEN_SIZE) * 0.02).to(torch.bfloat16)
    output_rms_weight = (1 + randn(HIDDEN_SIZE) * 0.02).to(torch.bfloat16)
    return (
        layer_residual,
        layer_residual_add,
        block_residual,
        res_weight,
        rms_weight,
        output_rms_weight,
    )


def _bits(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.view(torch.int16)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, what: str) -> None:
    a, e = actual.float().flatten(), expected.float().flatten()
    cosine = torch.nn.functional.cosine_similarity(a, e, dim=0).item()
    relative_l2 = ((a - e).norm() / (e.norm() + 1e-12)).item()
    assert cosine > MIN_COSINE and relative_l2 < MAX_RELATIVE_L2, (
        f"{what}: cosine {cosine:.7f}, relative L2 {relative_l2:.3e}"
    )


def _seed(cell) -> int:
    return zlib.crc32(cell.why.encode()) % 2**31


@pytest.mark.parametrize(
    "cell", attn_res_rmsnorm_fwd.CELLS, ids=[c.why[:48] for c in attn_res_rmsnorm_fwd.CELLS]
)
def test_certified_cells_rmsnorm(cell, certifying) -> None:  # noqa: F811 — fixture resolved by name
    certifying(attn_res_rmsnorm_fwd)
    spec = cell.spec
    assert spec["hidden"] == HIDDEN_SIZE
    layer, _, block, res_w, rms_w, out_w = _make_inputs(
        spec["tokens"], spec["candidates"], _seed(cell)
    )
    inputs = (layer, block, res_w, rms_w, out_w)
    before = [t.clone() for t in inputs]
    args = (*inputs, spec["eps"], spec["output_eps"])
    out = attn_res_rmsnorm_fwd(*args)
    again = attn_res_rmsnorm_fwd(*args)
    assert out.shape == layer.shape and out.dtype == torch.bfloat16
    assert torch.equal(_bits(out), _bits(again)), "two identical calls disagree"
    _assert_close(out, attn_res_rmsnorm_fwd.reference(*args), "output")
    for old, new in zip(before, inputs):
        assert torch.equal(_bits(old), _bits(new)), "an input was mutated"


def test_gate_refuses_a_dropped_candidate() -> None:
    layer, _, block, res_w, rms_w, out_w = _make_inputs(1, 6, seed=5)
    out = attn_res_rmsnorm_fwd(layer, block, res_w, rms_w, out_w, RMS_EPS, RMS_EPS)
    wrong = attn_res_rmsnorm_fwd.reference(layer, block[1:], res_w, rms_w, out_w, RMS_EPS, RMS_EPS)
    with pytest.raises(AssertionError):
        _assert_close(out, wrong, "dropped candidate")


def test_cuda_graph_replay_matches_eager() -> None:
    for num_candidates in (3, 9):
        layer, _, block, res_w, rms_w, out_w = _make_inputs(1, num_candidates, seed=7)
        eager = attn_res_rmsnorm_fwd(layer, block, res_w, rms_w, out_w, RMS_EPS, RMS_EPS)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = attn_res_rmsnorm_fwd(layer, block, res_w, rms_w, out_w, RMS_EPS, RMS_EPS)
        graph.replay()
        captured.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(_bits(captured), _bits(eager)), f"N={num_candidates}"
        del captured
