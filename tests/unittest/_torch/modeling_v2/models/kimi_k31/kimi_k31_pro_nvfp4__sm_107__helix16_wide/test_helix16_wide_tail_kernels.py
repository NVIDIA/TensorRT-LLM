# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The wide tail's kernel choice per row range (CPU only, extracted from ``modeling.py`` on
stand-in tails): ``LatentRowsLoaded`` (H8b, every load of a work item before its first push)
replaces LatentRows at 113-128 rows but the experts' own push rows, and ``NormRowsVote`` (the
gathered-row poll exits by a warp vote) replaces NormRows at 65-128 rows, each compiled with its
row count's carve-out and poll decisions."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from _helix16_wide_cpu import modeling_path, source_objects

pytestmark = pytest.mark.cpu_only


def _tail_namespace(built):
    carve_rows, polled_rows = {96, 120}, {96, 125}
    tables = SimpleNamespace(ROWS_CARVEOUT={16: "spec"}, norm_polled=lambda tp, b: b in polled_rows)
    policy = SimpleNamespace(batch_table=lambda spec: {b: 1 for b in carve_rows} if spec else {})
    mods = {"cutedsl.moe.tail_rows": tables, "cutedsl.common.policy": policy}
    cuda = SimpleNamespace(
        get_device_properties=lambda d: SimpleNamespace(multi_processor_count=212)
    )
    ns = {
        "module": mods.__getitem__, "torch": SimpleNamespace(cuda=cuda), "Tuple": tuple,
        "compile_latent_rows_loaded": lambda *a: built.append(("latent", *a)) or ("loaded",) + a,
        "compile_norm_rows_vote": lambda *a: built.append(("norm", *a)) or ("vote",) + a,
    }  # fmt: skip
    names = ("LATENT_LOADS_ROWS", "NORM_VOTE_ROWS", "load_latent_first", "vote_norm_rows")
    return source_objects(modeling_path(), names, ns)


def test_the_tail_codes_are_replaced_at_their_rows():
    built = []
    ns = _tail_namespace(built)
    assert ns["LATENT_LOADS_ROWS"] == range(113, 129) and ns["NORM_VOTE_ROWS"] == range(65, 129)
    codes = {b: (f"push{b}", f"latent{b}", f"norm{b}", f"out{b}") for b in (96, 120, 125, 128)}
    dev = torch.zeros(1).device
    box = SimpleNamespace(
        cap=128, xnorm=torch.zeros(1), latent=SimpleNamespace(recv=torch.zeros(1))
    )
    tail = SimpleNamespace(tp=16, codes=dict(codes), box=box, experts_push={128}, deferred={})
    assert ns["load_latent_first"](tail) == (120, 125)  # 113-128, not the experts' own push rows
    assert ns["vote_norm_rows"](tail) == (96, 120, 125, 128)
    assert tail.vote_rows == frozenset({96, 120, 125, 128})
    for b in (120, 125):
        assert tail.codes[b][1][0] == "loaded" and tail.codes[b][2][0] == "vote"
        assert (tail.codes[b][0], tail.codes[b][3]) == (codes[b][0], codes[b][3])
    assert tail.codes[96][:2] == codes[96][:2] and tail.codes[128][1] == "latent128"
    # (tp, rows, cap, device index, carve) / (tp, rows, cap, device index, polled, carve)
    assert built[:2] == [
        ("latent", 16, 120, 128, dev.index, True),
        ("latent", 16, 125, 128, dev.index, False),
    ]
    assert ("norm", 16, 96, 128, dev.index, True, True) in built
    assert ("norm", 16, 125, 128, dev.index, True, False) in built
