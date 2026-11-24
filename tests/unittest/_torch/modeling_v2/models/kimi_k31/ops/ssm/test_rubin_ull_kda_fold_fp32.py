# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ULL target's kda_fold_fp32 entry (the K3.1-owned folded KDA pair of three or
four requests x q8 with a bounded F_A wait), built as ``FlatKda._prepare_fold`` builds it: every
cell against the FP64 reference (output rows, state checkpoints, conv pages), and bit for bit
against the plain pair at the same in-projection tactic (``SplitKTmemGemm64`` at MR8's 64-row Rubin
tactic -> ``kda_verify``) on a random (non-dyadic) layer.

The reference cells use small dyadic weights and inputs whose projection sums are exact in FP32 in
any order (``test_rubin_ull_kda_head_verify.py``)."""

import types
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell

__extra_import_path__ = [".", "../../../.."]
import _kda_case as case  # noqa: E402 — needs the path declared above
from _cells import certified_cells  # noqa: E402
from _validating import validating  # noqa: E402

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.gemm.rubin_ull_prepared_projection import (  # noqa: E402, E501
    prepared_projection,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_ull_kda_fold_fp32 import (  # noqa: E402, E501
    kda_fold_fp32 as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_ull_kda_verify import (  # noqa: E402, E501
    kda_verify,
)

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
        reason="the folded KDA pair needs an SM10x GPU (tensor memory, clusters)",
    ),
    # spin_trap.record_address() starts the trap-record watcher, a daemon thread that lives as
    # long as the process (kernels/common/rubin_ull_spin_trap.py).
    pytest.mark.threadleak(enabled=False),
]

HIDDEN = 7168
TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(op, TARGET_CELLS)


def _dyadic_weight(heads: int, g: torch.Generator) -> torch.Tensor:
    """BF16 [pitch, hidden]: q / k / v / g and beta rows of multiples of 1/64 in [-1/32, 1/32],
    one +-1 per F_A row, zero padding rows."""
    dim, pitch = heads * 128, case.pitch_of(heads)
    weight = torch.zeros(pitch, HIDDEN, device="cuda")
    dense = list(range(4 * dim)) + list(range(4 * dim + 128, 4 * dim + 128 + heads))
    weight[dense] = torch.randint(-2, 3, (len(dense), HIDDEN), generator=g, device="cuda") / 64
    columns = torch.randint(0, HIDDEN, (128,), generator=g, device="cuda")
    signs = torch.randint(0, 2, (128,), generator=g, device="cuda") * 2 - 1
    weight[4 * dim + torch.arange(128, device="cuda"), columns] = signs.float()
    return weight.bfloat16().contiguous()


def _fold(weight, core, caches, rows):
    """``FlatKda._prepare_fold``: the layer's own armed projection buffer, a prepared scratch."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.ssm import (
        rubin_ull_kda_fold_fp32 as fold,
    )

    tactic = case.module("runtime.providers").WIDE_B64[weight.shape[0]][:4]
    projected = torch.empty((32, weight.shape[0]), dtype=torch.bfloat16, device="cuda")
    fold.PreparedKdaFold.arm(projected, core.dim)
    pitch = case.module("cutedsl.kda.inproj64").PREPARED_PITCH
    prepared = torch.empty((fold.TILE_ROWS, pitch), dtype=torch.float32, device="cuda")
    core.readers = torch.zeros(4, dtype=torch.int32, device="cuda")
    return fold.PreparedKdaFold(weight, core, tactic, projected, prepared).prepare_caches(
        caches, (rows,)
    )


def _pair(weight, weights, heads, tokens, requests, caches, x, slots, accepted, out):
    """The plain pair at the fold's tactic: SplitKTmemGemm64 (WIDE_B64) -> kda_verify."""
    providers = case.module("runtime.providers")
    rows = requests * tokens
    projection = providers.PreparedProjection(
        weight, {rows: providers.WIDE_B64[weight.shape[0]]}
    ).prepare()
    projected = torch.empty((rows, weight.shape[0]), dtype=torch.bfloat16, device="cuda")
    prepared_projection(projection, x, projected)
    core = case.make_core(weights, heads, {(requests, tokens): (case.pitch_of(heads), 0)})
    core.prepare_caches(caches)
    kda_verify(core, projected, out, caches, slots, accepted)


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    spec = cell.spec
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    heads, tokens = spec["heads"], spec["tokens"]
    requests = len(spec["accepted"])
    rows, dim = requests * tokens, heads * 128
    slot_count = rows + 3
    caches = case.checkpoint_caches(slot_count, heads, tokens + 2, g)
    weights = case.layer_weights(heads, g)
    core = case.make_core(weights, heads, {})
    weight = _dyadic_weight(heads, g)
    x = (torch.randint(-2, 3, (rows, HIDDEN), generator=g, device="cuda") / 4).bfloat16()
    slots = case.verify_slots(requests, tokens, spec["padded"], slot_count, g)
    accepted = torch.tensor(spec["accepted"], dtype=torch.int32, device="cuda")
    out = torch.full((rows, dim), 7.0, dtype=torch.bfloat16, device="cuda")
    fold = _fold(weight, core, caches, rows)
    args = (fold, x, out, caches, slots, accepted)
    ref = op.reference(*args)
    before = tuple(t.clone() for t in caches)
    with validating(op):
        result = op(*args)
    torch.cuda.synchronize()
    assert result is out
    case.assert_core_matches(out, caches, before, slots, accepted, tokens, ref)


@pytest.mark.parametrize("requests", [3, 4])
def test_bitwise_the_plain_pair(requests) -> None:
    """A random layer (N(0, 0.02) weight, N(0, 1) input): outputs, checkpoints and conv pages bit
    for bit the plain pair's at the same in-projection tactic, three steps in a row on the same
    buffers (the F_A rows re-armed by the verify's last reader)."""
    g = torch.Generator(device="cuda").manual_seed(2400 + requests)
    heads, tokens = 6, 8
    rows, dim = requests * tokens, heads * 128
    slot_count = rows + 3
    caches = case.checkpoint_caches(slot_count, heads, tokens + 2, g)
    pair_caches = tuple(t.clone() for t in caches)
    weights = case.layer_weights(heads, g)
    weight = torch.zeros(case.pitch_of(heads), HIDDEN, dtype=torch.bfloat16, device="cuda")
    live = 4 * dim + 128 + heads
    weight[:live] = (torch.randn(live, HIDDEN, generator=g, device="cuda") * 0.02).bfloat16()
    fold = _fold(weight, case.make_core(weights, heads, {}), caches, rows)
    for step in range(3):
        x = torch.randn(rows, HIDDEN, generator=g, device="cuda").bfloat16()
        slots = case.verify_slots(requests, tokens, (), slot_count, g)
        accepted = torch.randint(1, tokens + 1, (requests,), generator=g, device="cuda")
        accepted = accepted.to(torch.int32)
        out = torch.zeros(rows, dim, dtype=torch.bfloat16, device="cuda")
        pair_out = out.clone()
        op(fold, x, out, caches, slots, accepted)
        _pair(weight, weights, heads, tokens, requests, pair_caches, x, slots, accepted, pair_out)
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int16), pair_out.view(torch.int16)), f"step {step}"
        for mine, theirs in zip(caches, pair_caches):
            assert torch.equal(mine.view(torch.uint8), theirs.view(torch.uint8)), f"step {step}"


def test_graph_replays_the_eager_step() -> None:
    """One 32-row step captured in a CUDA graph gives the eager step's outputs and checkpoints
    bit for bit, twice in a row (the F_A rows re-armed by the verify's last reader)."""
    g = torch.Generator(device="cuda").manual_seed(32)
    heads, tokens, requests = 6, 8, 4
    rows, dim = requests * tokens, heads * 128
    slot_count = rows + 3
    caches = case.checkpoint_caches(slot_count, heads, tokens + 2, g)
    core = case.make_core(case.layer_weights(heads, g), heads, {})
    weight = _dyadic_weight(heads, g)
    x = (torch.randint(-2, 3, (rows, HIDDEN), generator=g, device="cuda") / 4).bfloat16()
    slots = case.verify_slots(requests, tokens, (), slot_count, g)
    accepted = torch.tensor([3, 8, 1, 5], dtype=torch.int32, device="cuda")
    out = torch.zeros(rows, dim, dtype=torch.bfloat16, device="cuda")
    fold = _fold(weight, core, caches, rows)
    snapshot = tuple(t.clone() for t in caches)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        fold.run_fold(x, out, caches=caches, slots=slots, accepted=accepted)
    torch.cuda.synchronize()
    eager = (out.clone(), *(t.clone() for t in caches))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        fold.run_fold(x, out, caches=caches, slots=slots, accepted=accepted)
    for _ in range(2):
        for t, s in zip(caches, snapshot):
            t.copy_(s)
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out.view(torch.int16), eager[0].view(torch.int16))
        for t, e in zip(caches, eager[1:]):
            assert torch.equal(t.view(torch.uint8), e.view(torch.uint8))


def test_is_valid_refuses_what_the_kernel_would_misplace() -> None:
    fold = types.SimpleNamespace(
        core=types.SimpleNamespace(dim=768),
        weight=torch.zeros(case.pitch_of(6), HIDDEN),
        codes={24: None, 32: None},
    )
    x = torch.zeros(32, HIDDEN, dtype=torch.bfloat16)
    out = torch.zeros(32, 768, dtype=torch.bfloat16)
    slots = torch.arange(1, 33, dtype=torch.int32).view(4, 8)
    four = torch.tensor([1, 8, 2, 3], dtype=torch.int32)
    fp32 = (torch.zeros(1), torch.zeros(1, dtype=torch.float32))
    op.is_valid(fold, x, out, fp32, slots, four)
    op.is_valid(fold, x[:24], out[:24], fp32, slots[:3], four)
    with pytest.raises(ValueError, match="must be FP32"):
        op.is_valid(fold, x, out, (fp32[0], fp32[1].bfloat16()), slots, four)
    with pytest.raises(ValueError, match="3 or 4 requests"):
        op.is_valid(fold, x[:16], out[:16], fp32, slots[:2], four)
    with pytest.raises(ValueError, match="outside 1..q"):
        op.is_valid(fold, x, out, fp32, slots, torch.tensor([1, 9, 1, 1], dtype=torch.int32))
    shared = slots.clone()
    shared[1, 3] = shared[1, 0]
    with pytest.raises(ValueError, match="share a slot"):
        op.is_valid(fold, x, out, fp32, shared, four)
