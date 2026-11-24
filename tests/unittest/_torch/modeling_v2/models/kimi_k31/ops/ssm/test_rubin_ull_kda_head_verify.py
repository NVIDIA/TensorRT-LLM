# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ULL target's kda_head_verify entry: the fused in-projection + q8 recurrence of
every cell against the FP64 reference (output rows, state checkpoints, conv page), and the polled
variant's canonical output and re-armed buffer.

The layer weight and input are small dyadic values whose projection sums are exact in FP32 in any
order, so the kernels' BF16 operand rows are the reference's bit for bit and the band is the
recurrence's alone (``_kda_case``)."""

import types
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell

__extra_import_path__ = [".", "../../../.."]
import _kda_case as case  # noqa: E402 — needs the path declared above
from _cells import certified_cells  # noqa: E402
from _validating import validating  # noqa: E402

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_ull_kda_head_verify import (  # noqa: E402, E501
    HIDDEN,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_ull_kda_head_verify import (  # noqa: E402, E501
    kda_head_verify as op,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="MR8's fused KDA head needs an SM10x GPU (tensor memory, 16-CTA clusters)",
)

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


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell) -> None:
    head_module = case.module("cutedsl.kda.head")
    spec = cell.spec
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    heads, tokens = spec["heads"], spec["tokens"]
    dim = heads * 128
    slot_count = tokens + 3
    caches = case.checkpoint_caches(slot_count, heads, tokens + 2, g)
    core = case.make_core(case.layer_weights(heads, g), heads, {})
    weight = _layer_weight(heads, g)
    side = head_module.KdaSide(weight.device)
    counter = torch.zeros(2, dtype=torch.int32, device="cuda")
    fa_out = torch.empty((tokens, weight.shape[0] - 4 * dim), dtype=torch.bfloat16, device="cuda")
    # FlatKda._prepare_head
    head = head_module.PreparedKdaHead(
        weight, core, (), side, counter, fa_out, verify=((tokens, tokens),)
    )
    out, rearm = torch.full((tokens, dim), 7.0, dtype=torch.bfloat16, device="cuda"), None
    if spec["polled"]:
        # FlatKda._prepare_head's POLLED_OUT keying; the buffer starts armed (OUT_SENT, 0xFFFF).
        assert head_module.OUT_SENT == 0xFFFF
        head.polled = (tokens,)
        out = torch.full((tokens, dim), -1, dtype=torch.int16, device="cuda").view(torch.bfloat16)
        rearm = torch.randn(tokens, dim, generator=g, device="cuda").bfloat16()
    head = head.prepare_caches(caches)
    x = (torch.randint(-2, 3, (tokens, HIDDEN), generator=g, device="cuda") / 4).bfloat16()
    slots = case.verify_slots(1, tokens, (), slot_count, g)
    accepted = torch.tensor([spec["accepted"]], dtype=torch.int32, device="cuda")
    args = (head, x, out, caches, slots, accepted, rearm)
    ref = op.reference(*args)
    before = tuple(t.clone() for t in caches)
    with validating(op):
        result = op(*args)
    side.join()
    torch.cuda.synchronize()
    assert result is out
    case.assert_core_matches(out, caches, before, slots, accepted, tokens, ref)
    if rearm is not None:
        assert bool((rearm.view(torch.int16) == -1).all()), "the previous buffer is not re-armed"
        assert not bool((out.view(torch.int16) == -1).any()), "a polled element left unwritten"


def test_is_valid_refuses_what_the_kernel_would_misplace() -> None:
    head = types.SimpleNamespace(
        core=types.SimpleNamespace(dim=768), weight=torch.zeros(case.pitch_of(6), HIDDEN)
    )
    x = torch.zeros(8, HIDDEN, dtype=torch.bfloat16)
    out = torch.zeros(8, 768, dtype=torch.bfloat16)
    slots = torch.arange(1, 9, dtype=torch.int32).view(1, 8)
    one = torch.tensor([1], dtype=torch.int32)
    op.is_valid(head, x, out, None, slots, one)
    with pytest.raises(ValueError, match="one request"):
        op.is_valid(head, torch.zeros(16, HIDDEN, dtype=torch.bfloat16), out, None, slots, one)
    with pytest.raises(ValueError, match="outside 1..q"):
        op.is_valid(head, x, out, None, slots, torch.tensor([9], dtype=torch.int32))
    with pytest.raises(ValueError, match="share a slot"):
        op.is_valid(head, x, out, None, torch.ones(1, 8, dtype=torch.int32), one)
