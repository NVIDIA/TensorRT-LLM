# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The shared one-kernel speculative decode step head on its ``.CELLS``.

Every tensor it writes matches the reference, including the ranges outside the active
step, and the input offsets are untouched.
"""

import importlib
import zlib

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.catalog._op import Arch

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402, F401 — path above; fixture by name

_OPS = "tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl.ops.spec"
k3_spec_decode_head = importlib.import_module(f"{_OPS}.k3_spec_decode_head").k3_spec_decode_head

_ARCH = {(10, 0): Arch.SM_100, (10, 3): Arch.SM_103, (10, 7): Arch.SM_107}
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or _ARCH.get(torch.cuda.get_device_capability()) not in k3_spec_decode_head.ARCHS,
    reason="k3_spec_decode_head is certified on its ARCHS",
)


def _step(cell):
    """The positional arguments and keywords of one cell's step (fresh tensors per call)."""
    spec = cell.spec
    num_seqs, tokens_per_seq, tokens_per_block = spec["num_seqs"], spec["tokens_per_seq"], 128
    n = num_seqs * tokens_per_seq
    gen = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)

    def ints(low, high, size):
        return torch.randint(low, high, (size,), device="cuda", generator=gen).to(torch.int32)

    # Groups start a few tokens past a page boundary, so a group straddles owners.
    base = ints(0, 1 << 11, num_seqs).long() * tokens_per_block - ints(0, tokens_per_seq, num_seqs)
    base = base.clamp(min=0)
    positions = (base.view(-1, 1) + torch.arange(tokens_per_seq, device="cuda")).view(-1)
    capacity = n + 16
    helix_positions = torch.zeros(capacity, dtype=torch.int32, device="cuda")
    helix_positions[:n] = positions.to(torch.int32)
    args = (
        ints(0, 1 << 18, capacity),  # position_ids
        ints(-2, 1, capacity),  # pos_offsets
        n,
        ints(0, 1 << 16, num_seqs + 8),  # kv_lens
        ints(-2, 1, num_seqs + 8),  # kv_offsets
        0,
        num_seqs,
    )
    kwargs = dict(
        has_overlap=spec["overlap"],
        helix_positions=helix_positions,
        helix_slots=torch.full((capacity,), -7, dtype=torch.int32, device="cuda"),
        helix_bounds=torch.full((capacity,), -7, dtype=torch.int32, device="cuda"),
        helix_zero_kv=torch.ones(capacity, dtype=torch.bool, device="cuda"),
        num_helix=n,
        tokens_per_block=tokens_per_block,
        cp_size=spec["cp_size"],
        cp_rank=spec["cp_rank"],
        kv_bound_start=0,
        tokens_per_seq=tokens_per_seq,
    )
    return args, kwargs


def _written(args, kwargs):
    return dict(position_ids=args[0], kv_lens=args[3], **kwargs)


@pytest.mark.parametrize(
    "cell", k3_spec_decode_head.CELLS, ids=[c.why[:48] for c in k3_spec_decode_head.CELLS]
)
def test_k3_spec_decode_head_ull_cells(cell, certifying):  # noqa: F811 — fixture by name
    op = k3_spec_decode_head
    certifying(op)
    args, kwargs = _step(cell)
    expected = op.reference(*args, **kwargs)
    pos_offsets, kv_offsets = args[1].clone(), args[4].clone()
    op(*args, **kwargs)
    torch.cuda.synchronize()
    written = _written(args, kwargs)
    for name, value in expected.items():
        assert torch.equal(written[name], value), name
    assert torch.equal(args[1], pos_offsets) and torch.equal(args[4], kv_offsets)
