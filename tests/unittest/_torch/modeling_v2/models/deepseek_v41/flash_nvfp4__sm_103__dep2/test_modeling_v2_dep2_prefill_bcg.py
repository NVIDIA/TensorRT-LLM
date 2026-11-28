# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dep2's breakable two-pass prefill graphs (`prefill_graphs.py`): bucket selection as stock's, and a pass graph that
captures every bucket on first use and replays with eager breaks over the live rows."""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.deepseek_v41.flash_nvfp4__sm_103__dep2 import (
    prefill_graphs as pg,
)
from tensorrt_llm._torch.pyexecutor.breakable_cuda_graph import eager_on_graph


def test_decoder_buckets_match_stock():
    from tensorrt_llm._torch.models.deepseekv41_decoder_graph import decoder_pass_buckets as stock

    for window, top in ((140, 8064), (140, 560), (128, 5000)):
        assert pg.decoder_pass_buckets(window, top) == stock(window, top)


@pytest.mark.parametrize(
    "counts,want",
    [
        ([11000, 10500], 12288),  # both ranks run the largest rank's bucket
        ([2048, 1500], 2048),
        ([32768, 31000], 32768),
        ([8000, 100], 8192),  # any padding: the MoE routes padding rows nowhere
        ([4000, 0], None),  # a rank without rows runs eagerly
        ([], None),
    ],
)
def test_encoder_select(counts, want):
    g = pg.EncoderPassGraph(lambda *a: None, pg.PassGraphPool(), 2)
    assert g.select(counts, 32768) == want
    assert g.buckets[-1] == 32768


def test_decoder_select():
    g = pg.DecoderPassGraph(lambda *a: None, pg.PassGraphPool(), 2, 140)
    assert g.select([700, 560], 32768, 256) == 840
    assert g.select([0, 0], 32768, 256) is None
    assert g.select([9000, 10], 32768, 256) is None  # past the largest bucket
    assert g.buckets[-1] == 8064


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs")
def test_capture_all_then_replay_with_live_breaks():
    """Graph segments run over the bucket's rows; the eager break between them sees only the live rows of the replay
    (none during the warm runs and captures). Every bucket is captured on first use."""
    seen = []

    def body(x, live, counts, padding):
        y = torch.where(
            padding, 0.0, x * 2
        )  # a graph segment; padding rows (past the replay's live rows) zeroed

        def bump(t):
            seen.append((live.rows, t.shape[0], list(counts)))
            if live.rows:
                t[: live.rows] += 1000

        eager_on_graph(bump)(y)
        return (y + 1,)  # another segment

    g = pg.EncoderPassGraph(body, pg.PassGraphPool(), 2)
    g.buckets = (4, 8)
    x = torch.arange(6, dtype=torch.float32, device="cuda")[:, None].repeat(1, 3)
    (out,) = g.replay(8, (x,), 6, None, None, [6, 5])
    assert g.captured and set(g._graphs) == {4, 8}
    # warm runs and captures: no live rows; then the replay's break with the 6 live rows of the 8-row bucket
    assert [s[0] for s in seen[:-1]] == [0] * 4 and seen[-1][:2] == (6, 8)
    assert torch.equal(out, x * 2 + 1001)
    full = g._graphs[8].outputs[
        0
    ]  # the bucket's whole output: the 2 padding rows saw padding = True
    assert torch.equal(full[6:], torch.ones_like(full[6:]))
    seen.clear()
    (out,) = g.replay(4, (x[:3],), 3, None, None, [3, 2])
    assert torch.equal(out, x[:3] * 2 + 1001) and seen == [(3, 4, [4, 4])]
