# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test of the fused_kda_conv_prefill entry: every cell of ``fused_kda_conv_prefill.CELLS``,
built the way the tp1 KDA layer calls it (strided QKVG column view, transposed weight copy, padded
conv pool slots), against the fp64 reference: Q/K/V within the stated bf16 band, beta and the
written conv states bitwise, untouched slots and slot padding unchanged."""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.ssm.fused_kda_conv_prefill import (  # noqa: E501
    TP1_HEADS,
    fused_kda_conv_prefill,
)

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, validating

HEAD_DIM = 128
WIDTH = 4
SLOTS = 8
SLOT_PAD = 64  # elements past each slot's [3 * H * D, W - 1] state, as the cache manager pads


def _driven_here() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return f"sm_{major}{minor}" in {arch.value for arch in fused_kda_conv_prefill.ARCHS}


pytestmark = pytest.mark.skipif(not _driven_here(), reason="fused_kda_conv_prefill.ARCHS")


def _inputs(spec: dict, seed: int) -> dict:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    heads, seq_lens = TP1_HEADS, spec["seq_lens"]
    channels = 3 * heads * HEAD_DIM
    tokens = sum(seq_lens)

    def randn(*shape, scale=1.0):
        return torch.randn(*shape, generator=gen, device="cuda") * scale

    qkvg = randn(tokens, 4 * heads * HEAD_DIM).bfloat16()
    weight_t = randn(WIDTH, channels, scale=0.5).bfloat16()
    pool = randn(SLOTS, channels * (WIDTH - 1) + SLOT_PAD).bfloat16()
    conv_states = pool[:, : channels * (WIDTH - 1)].view(SLOTS, channels, WIDTH - 1)
    starts = [0]
    for n in seq_lens:
        starts.append(starts[-1] + n)
    slots = torch.randperm(SLOTS, generator=torch.Generator().manual_seed(seed))[: len(seq_lens)]
    args = dict(
        qkv=qkvg[:, :channels],
        conv_weight=weight_t.t(),
        conv_states=conv_states,
        query_start_loc=torch.tensor(starts, dtype=torch.int32, device="cuda"),
        cache_indices=slots.to(torch.int32).cuda(),
        has_initial_state=torch.tensor(spec["has_init"], dtype=torch.bool, device="cuda"),
        num_heads=heads,
        head_dim=HEAD_DIM,
    )
    if spec["prepad"]:
        bfa = randn(1, tokens, HEAD_DIM + heads).bfloat16()
        args.update(
            padded_tokens=-(-tokens // 64) * 64,
            beta=bfa[..., HEAD_DIM : HEAD_DIM + heads],
        )
    return args, pool


@pytest.mark.parametrize(
    "cell", fused_kda_conv_prefill.CELLS, ids=[c.why[:60] for c in fused_kda_conv_prefill.CELLS]
)
@torch.no_grad()
def test_cells(cell) -> None:
    args, pool = _inputs(cell.spec, zlib.crc32(cell.why.encode()) % 2**31)
    pool_before = pool.clone()
    ref_out, ref_states = fused_kda_conv_prefill.reference(**args)
    with validating(fused_kda_conv_prefill):
        out = fused_kda_conv_prefill(**args)

    assert len(out) == len(ref_out), cell.why
    for name, got, want in zip("qkvb", out, ref_out):
        assert got.shape == want.shape and got.dtype == want.dtype, (cell.why, name)
        if name == "b":
            assert torch.equal(got, want), (cell.why, name)
            continue
        # The convolution accumulates in fp32 (here fp64) before one bf16 rounding, and Q/K are
        # normalized from that rounded value: an element can land one bf16 ulp away, and a
        # normalized row inherits that ulp, so two ulps of the row scale bound both.
        assert_within_ulp(got, want, element_ulp=2.0, rms_ulp=0.25)

    state_size = args["conv_states"].shape[1] * (WIDTH - 1)
    assert torch.equal(args["conv_states"], ref_states), f"{cell.why}: conv states"
    assert torch.equal(pool[:, state_size:], pool_before[:, state_size:]), "slot padding written"
    written = set(args["cache_indices"].tolist())
    for slot in range(SLOTS):
        if slot not in written:
            assert torch.equal(pool[slot], pool_before[slot]), f"slot {slot} written"
