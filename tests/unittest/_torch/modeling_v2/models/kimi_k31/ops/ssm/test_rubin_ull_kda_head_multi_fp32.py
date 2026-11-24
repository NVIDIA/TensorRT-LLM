# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ULL target's kda_head_multi_fp32 entry (the K3.1-owned fused KDA head of two
requests x q8 with bounded F_A waits), built as ``FlatKda._prepare_multi`` builds it: every cell
against the FP64 reference (output rows, state checkpoints, conv pages), bit for bit against MR8's
``PreparedKdaHead`` (``KdaHeadMulti``) on the same inputs (the bound changes no output bit), and bit
for bit against the 16-row split pair's recurrence (``kda_verify``) on the same projected rows.

The layer weight and input are small dyadic values whose projection sums are exact in FP32 in any
order (``test_rubin_ull_kda_head_verify.py``), so the head's four-way split-K sums and the split
pair's projection give the same BF16 rows."""

import types
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell

__extra_import_path__ = [".", "../../../.."]
import _kda_case as case  # noqa: E402 — needs the path declared above
from _cells import certified_cells  # noqa: E402
from _validating import validating  # noqa: E402

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_ull_kda_head_multi_fp32 import (  # noqa: E402, E501
    HIDDEN,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_ull_kda_head_multi_fp32 import (  # noqa: E402, E501
    kda_head_multi_fp32 as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_ull_kda_verify import (  # noqa: E402, E501
    kda_verify,
)

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
        reason="the fused KDA head needs an SM10x GPU (tensor memory, 16-CTA clusters)",
    ),
    # spin_trap.record_address() starts the trap-record watcher, a daemon thread that lives as
    # long as the process (kernels/common/rubin_ull_spin_trap.py).
    pytest.mark.threadleak(enabled=False),
]

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(op, TARGET_CELLS)


def _layer_weight(heads: int, g: torch.Generator) -> torch.Tensor:
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


def _head(cls, weight, core, caches, rows, tokens, **order):
    """``FlatKda._prepare_multi``: the head's own side stream, counter pair and F_A buffer.
    ``order``: the launch order, set directly (one process, nothing else resident)."""
    head_module = case.module("cutedsl.kda.head")
    dim = core.dim
    side = head_module.KdaSide(weight.device)
    counter = torch.zeros(2, dtype=torch.int32, device="cuda")
    fa_out = torch.empty((rows, weight.shape[0] - 4 * dim), dtype=torch.bfloat16, device="cuda")
    head = cls(weight, core, (), side, counter, fa_out, verify=((rows, tokens),), **order)
    return head.prepare_caches(caches), side


@pytest.mark.parametrize("order", ["event", "sibling"])
@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, order) -> None:
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.ssm.rubin_ull_kda_head_multi_fp32 import (  # noqa: E501
        PreparedKdaHeadMultiFp32,
    )

    head_module = case.module("cutedsl.kda.head")
    spec = cell.spec
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    heads, tokens, requests = spec["heads"], spec["tokens"], spec["requests"]
    rows, dim = requests * tokens, heads * 128
    slot_count = rows + 3
    caches = case.checkpoint_caches(slot_count, heads, tokens + 2, g)
    weights = case.layer_weights(heads, g)
    core = case.make_core(weights, heads, {})
    weight = _layer_weight(heads, g)
    x = (torch.randint(-2, 3, (rows, HIDDEN), generator=g, device="cuda") / 4).bfloat16()
    slots = case.verify_slots(requests, tokens, spec["padded"], slot_count, g)
    accepted = torch.tensor(spec["accepted"], dtype=torch.int32, device="cuda")
    out = torch.full((rows, dim), 7.0, dtype=torch.bfloat16, device="cuda")
    mr8_caches = tuple(t.clone() for t in caches)
    pair_caches = tuple(t.clone() for t in caches)
    mr8_out, pair_out = out.clone(), out.clone()

    head, side = _head(PreparedKdaHeadMultiFp32, weight, core, caches, rows, tokens, order=order)
    args = (head, x, out, caches, slots, accepted)
    ref = op.reference(*args)
    before = tuple(t.clone() for t in caches)
    with validating(op):
        result = op(*args)
    side.join()
    torch.cuda.synchronize()
    assert result is out
    case.assert_core_matches(out, caches, before, slots, accepted, tokens, ref)

    # MR8's head (KdaHeadMulti, FP32 state from the caches) on the same inputs: the bounded
    # waits change no output or checkpoint bit.
    mr8, mr8_side = _head(head_module.PreparedKdaHead, weight, core, mr8_caches, rows, tokens)
    mr8.run_verify(x, mr8_out, caches=mr8_caches, slots=slots, accepted=accepted)
    mr8_side.join()
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), mr8_out.view(torch.int16))
    for mine, theirs in zip(caches, mr8_caches):
        assert torch.equal(mine.view(torch.uint8), theirs.view(torch.uint8))

    # The 16-row split pair's recurrence (kda_verify) on the same BF16 projected rows (exact in
    # any summation order here): the same outputs, checkpoints and conv pages bit for bit.
    pair_core = case.make_core(weights, heads, {(requests, tokens): (case.pitch_of(heads), 0)})
    pair_core.prepare_caches(pair_caches)
    projected = (x.double() @ weight.double().T).float().bfloat16()
    kda_verify(pair_core, projected, pair_out, pair_caches, slots, accepted)
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), pair_out.view(torch.int16))
    for mine, theirs in zip(caches, pair_caches):
        assert torch.equal(mine.view(torch.uint8), theirs.view(torch.uint8))


def test_graph_replays_the_eager_step() -> None:
    """One step captured in a CUDA graph (the head and its SignalGemm on the side stream) gives
    the eager step's outputs and checkpoints bit for bit, twice in a row (the F_A buffer and the
    counter pair re-armed by the last reader)."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.ssm.rubin_ull_kda_head_multi_fp32 import (  # noqa: E501
        PreparedKdaHeadMultiFp32,
    )

    g = torch.Generator(device="cuda").manual_seed(16)
    heads, tokens, requests = 6, 8, 2
    rows, dim = requests * tokens, heads * 128
    slot_count = rows + 3
    caches = case.checkpoint_caches(slot_count, heads, tokens + 2, g)
    core = case.make_core(case.layer_weights(heads, g), heads, {})
    weight = _layer_weight(heads, g)
    x = (torch.randint(-2, 3, (rows, HIDDEN), generator=g, device="cuda") / 4).bfloat16()
    slots = case.verify_slots(requests, tokens, (), slot_count, g)
    accepted = torch.tensor([3, 8], dtype=torch.int32, device="cuda")
    out = torch.zeros(rows, dim, dtype=torch.bfloat16, device="cuda")
    head, side = _head(PreparedKdaHeadMultiFp32, weight, core, caches, rows, tokens, order="event")
    snapshot = tuple(t.clone() for t in caches)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        head.run_verify(x, out, caches=caches, slots=slots, accepted=accepted)
        side.join()
    torch.cuda.synchronize()
    eager = (out.clone(), *(t.clone() for t in caches))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        head.run_verify(x, out, caches=caches, slots=slots, accepted=accepted)
        side.join()
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
    head = types.SimpleNamespace(
        core=types.SimpleNamespace(dim=768), weight=torch.zeros(case.pitch_of(6), HIDDEN)
    )
    x = torch.zeros(16, HIDDEN, dtype=torch.bfloat16)
    out = torch.zeros(16, 768, dtype=torch.bfloat16)
    slots = torch.arange(1, 17, dtype=torch.int32).view(2, 8)
    two = torch.tensor([1, 8], dtype=torch.int32)
    fp32 = (torch.zeros(1), torch.zeros(1, dtype=torch.float32))
    op.is_valid(head, x, out, fp32, slots, two)
    with pytest.raises(ValueError, match="must be FP32"):
        op.is_valid(head, x, out, (fp32[0], fp32[1].bfloat16()), slots, two)
    with pytest.raises(ValueError, match="two requests"):
        op.is_valid(head, x[:8], out, fp32, slots, two)
    with pytest.raises(ValueError, match="no polled output"):
        op.is_valid(head, x, out, fp32, slots, two, out)
    with pytest.raises(ValueError, match="outside 1..q"):
        op.is_valid(head, x, out, fp32, slots, torch.tensor([1, 9], dtype=torch.int32))
    shared = slots.clone()
    shared[1, 3] = shared[1, 0]
    with pytest.raises(ValueError, match="share a slot"):
        op.is_valid(head, x, out, fp32, shared, two)
