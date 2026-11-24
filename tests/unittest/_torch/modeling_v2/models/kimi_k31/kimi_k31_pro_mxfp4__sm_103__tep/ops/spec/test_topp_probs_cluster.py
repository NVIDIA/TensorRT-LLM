# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 k31_topp_probs_cluster entry (top-p alone, one cluster per row).

Kept sets are compared by mass at the cut: a token sitting on a top-p boundary may land on either
side here and in the reference or an engine sampler, which moves that token's mass and nothing
else; a wrong cut moves whole tokens far from it. Values are bounded per row in L1, with the fused
entry's band.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.spec.topp_probs_cluster import (  # noqa: E501
    k31_topp_probs_cluster as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_103__tep.ops.spec.topp_probs_cluster import (  # noqa: E501
    serves,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.flashinfer_compute_probs_from_logits import (  # noqa: E501
    flashinfer_compute_probs_from_logits,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.spec.fused_compute_probs_from_logits import (  # noqa: E501
    fused_compute_probs_from_logits,
)

__extra_import_path__ = ["../../../../..", "../../../ops/spec"]
from _k31_spec_inputs import DISABLE_TOP_K, on_sm103, row_l1, sampling_inputs, seed_of  # noqa: E402
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

pytestmark = on_sm103

#: The fused entry's band (same reference; its derivation: fp32 softmax over 163840 entries reduced
#: in different orders, plus one boundary flip of a ~2e-5 token).
_ROW_L1 = 5e-5
#: A token whose side of the cut differs sits within this relative distance of the reference's cut
#: probability. Measured on GB300 against filtered_probs and the fused sampler: <= 3.2e-4 at
#: top-p <= 0.99 (fp32 cumulative sums of up to 1.6e5 terms on the reference side).
_CUT_REL = 2e-3
#: An engine sampler against this entry: each within _ROW_L1 of the reference.
_ENGINE_ROW_L1 = 2 * _ROW_L1


def topp_inputs(spec: dict, seed: int):
    """``(logits, temperatures, top_p)`` of a cell; ``spec["logits"]`` picks the logits."""
    logits, temperatures, _, top_p, _ = sampling_inputs(spec, seed, fused=False)
    rows, vocab = logits.shape
    gen = torch.Generator(device="cuda").manual_seed(seed + 1)
    kind = spec.get("logits", "randn")
    if kind == "peaked":
        # Three distinct tokens 6 / 5 / 4 above the row's maximum: a top-p 0.95 set of about three.
        hot = torch.rand(rows, vocab, generator=gen, device="cuda").topk(3, dim=-1).indices
        top = logits.amax(dim=-1, keepdim=True) + torch.tensor([6.0, 5.0, 4.0], device="cuda")
        logits = logits.scatter(1, hot, top)
    elif kind == "flat":
        logits = torch.randn(rows, vocab, generator=gen, device="cuda") * 0.5
    else:
        assert kind == "randn", kind
    return logits.contiguous(), temperatures, top_p


def assert_same_nucleus(probs, other, logits, temperatures, keep_ref, what: str) -> None:
    """Every token kept by one side and not the other sits at the reference's cut (by mass)."""
    softmax = torch.softmax(logits.float() / temperatures.view(-1, 1), dim=-1)
    diff = (probs > 0) ^ (other > 0)
    for r in torch.nonzero(diff.any(dim=-1)).flatten().tolist():
        cut = softmax[r][keep_ref[r]].min()
        moved = softmax[r][diff[r]]
        rel = ((moved - cut).abs() / cut).max().item()
        mass = moved.double().sum().item()
        assert rel <= _CUT_REL and mass <= _ROW_L1, (
            f"{what}: row {r} keeps {int(diff[r].sum())} token(s) differently, "
            f"{rel:.2e} from the cut, mass {mass:.2e}"
        )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    logits, temperatures, top_p = topp_inputs(cell.spec, seed_of(cell))
    assert serves(logits, temperatures, top_p)
    probs = op(logits, temperatures, top_p)
    expected = op.reference(logits, temperatures, top_p)
    assert probs.shape == expected.shape and probs.dtype == torch.float32
    l1 = row_l1(probs, expected)
    assert l1.max().item() <= _ROW_L1, f"max row L1 {l1.max().item():.3e}"
    torch.testing.assert_close(
        probs.sum(-1), torch.ones(probs.shape[0], device="cuda"), atol=1e-5, rtol=0
    )
    assert_same_nucleus(probs, expected, logits, temperatures, expected > 0, "vs filtered_probs")
    greedy = temperatures[: logits.shape[0]] <= 1e-4
    assert torch.equal(probs[greedy].argmax(-1), logits[greedy].argmax(-1))
    assert bool(((probs[greedy] > 0).sum(-1) == 1).all()), "greedy rows must be one-hot"


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:48] for c in op.CELLS])
def test_nucleus_matches_the_engine_samplers(cell) -> None:
    """Against the flashinfer chain this entry replaces (NO_TOPK: top-k None) and the fused sampler
    (FULL, top-k and min-p disabled), on the certified cells: kept sets by mass at the cut on the rows
    with a cut (top-p <= 0.95), L1 on every row."""
    logits, temperatures, top_p = topp_inputs(cell.spec, seed_of(cell))
    rows = logits.shape[0]
    probs = op(logits, temperatures, top_p)
    keep_ref = op.reference(logits, temperatures, top_p) > 0
    chain = flashinfer_compute_probs_from_logits(logits, temperatures, None, top_p)
    top_ks = torch.full((rows,), DISABLE_TOP_K, dtype=torch.int32, device="cuda")
    min_ps = torch.zeros(rows, dtype=torch.float32, device="cuda")
    fused = fused_compute_probs_from_logits(logits, temperatures, top_ks, top_p, min_ps)
    # Rows at top-p 1.0 have no cut: the flashinfer chain's renorm may drop a float-rounding tail
    # there (measured: 86 tokens of total mass 6.4e-8), which the L1 bound covers.
    cut = top_p[:rows] < 1.0
    for name, other in (("flashinfer NO_TOPK chain", chain), ("fused sampler", fused)):
        assert_same_nucleus(
            probs[cut],
            other[cut],
            logits[cut],
            temperatures[:rows][cut],
            keep_ref[cut],
            f"vs the {name}",
        )
        l1 = row_l1(probs, other)
        assert l1.max().item() <= _ENGINE_ROW_L1, f"vs the {name}: max row L1 {l1.max().item():.3e}"


def test_repeatable_graph_replay_and_pdl(monkeypatch) -> None:
    """Bit-identical run to run, under CUDA-graph replay (also of new logits in the captured buffer),
    and with PDL on or off."""
    cell = next(c for c in op.CELLS if c.spec["mixed"])  # per-row top-p, greedy-sentinel rows
    logits, temperatures, top_p = topp_inputs(cell.spec, seed_of(cell))
    monkeypatch.setenv("TRTLLM_ENABLE_PDL", "1")
    eager = op(logits, temperatures, top_p)
    assert torch.equal(op(logits, temperatures, top_p), eager)
    monkeypatch.setenv("TRTLLM_ENABLE_PDL", "0")
    assert torch.equal(op(logits, temperatures, top_p), eager)
    monkeypatch.setenv("TRTLLM_ENABLE_PDL", "1")

    static = logits.clone()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = op(static, temperatures, top_p)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, eager)
    other, _, _ = topp_inputs(cell.spec, seed_of(cell) + 7)
    static.copy_(other)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(captured, op(other, temperatures, top_p))


def test_serves() -> None:
    cell = op.CELLS[0]
    logits, temperatures, top_p = topp_inputs(cell.spec, seed_of(cell))
    assert serves(logits, temperatures, top_p)
    assert not serves(logits.bfloat16(), temperatures, top_p)
    assert not serves(logits[:, 4:], temperatures, top_p)  # not row-contiguous
    assert not serves(logits[:, :1000].contiguous(), temperatures, top_p)  # vocab does not tile
    assert not serves(logits, temperatures[:-1], top_p)
    assert not serves(logits, temperatures, top_p.double())
    assert not serves(logits[:0], temperatures, top_p)
