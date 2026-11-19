# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""rubin_megamoe on the CPU: the stateless staging (a replay after any size, the tail a larger one
left reset).

The family launcher runs with the engine's MegaMoE op and symmetric-buffer provider replaced by
fakes that record what the kernel reads at each launch. The cells of the EP entries
(``rubin_megamoe.CELLS``, ``rubin_megamoe_in_bucket.CELLS``, ``rubin_moe_a2a_dispatch.CELLS`` and
``rubin_moe_a2a_combine.CELLS``) need four SM107 ranks of one tray, which no CI stage has.
"""

from __future__ import annotations

import sys
import types
from typing import NamedTuple

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_nvfp4__sm_107__dep16.kernels.moe import (
    rubin_megamoe as K,
)

__extra_import_path__ = ["../../.."]
from _k31_utils import cpu_or_sm

pytestmark = [pytest.mark.cpu_only, cpu_or_sm(107)]

H, TOP_K, LOCAL, INTER = 64, 4, 2, 32
SF_COLS = H // 16


class _Regions(NamedTuple):
    activation: torch.Tensor
    activation_sf: torch.Tensor
    topk_weights: torch.Tensor
    combine_output: torch.Tensor
    shared_workspace: torch.Tensor
    peer_offsets: list


class _Engine:
    """The engine surface the launcher calls, recording the providers and the launches."""

    def __init__(self, direct=()):
        self.direct = set(direct)
        self.providers = []
        self.launches = []
        self.tuning = False
        self.in_op = False  # the fake op's own writes stand for the kernel's

    def module(self):
        m = types.ModuleType("cute_dsl_megamoe_custom_op")
        m.query_megamoe_kernel_properties = lambda **kw: (
            64,
            kw["max_tokens_per_rank"] in self.direct,
        )
        m.get_megamoe_symm_provider = self.provider
        m.megamoe_activation_sf_bytes_per_row = lambda hidden: hidden // 16
        return m

    def provider(self, **kw):
        t, k = kw["max_tokens_per_rank"], kw["num_topk"]
        self.providers.append((t, kw["rank"], kw["world_size"], kw["combine_format"]))
        regions = _Regions(
            torch.full((t, H // 2), 0xAB, dtype=torch.uint8),
            torch.full((t, SF_COLS), 0xCD, dtype=torch.uint8),
            torch.full((t, k), 9.0),
            torch.zeros(t, 1, H, dtype=torch.bfloat16),
            torch.zeros(kw["shared_workspace_bytes"], dtype=torch.uint8),
            [0, 1, 2, 3],
        )
        return types.SimpleNamespace(get_regions=lambda: regions)

    def op(self, **kw):
        self.in_op = True
        try:
            self._op(**kw)
        finally:
            self.in_op = False

    def _op(self, **kw):
        n = kw["num_tokens"]
        self.launches.append(
            dict(
                bucket=kw["max_tokens_per_rank"],
                n=n,
                ids=kw["topk_idx"].clone(),
                weights=kw["topk_weights"].clone(),
                ids_tensor=kw["topk_idx"],
                weights_tensor=kw["topk_weights"],
                activation=kw["activation"],
                activation_sf=kw["activation_sf"],
                activation_rows=kw["activation"].view(torch.uint8)[:n].clone(),
                sf_rows=kw["activation_sf"].view(torch.uint8)[:n].clone(),
                fc1_sf_shape=tuple(kw["fc1_weight_sf"].shape),
                fc2_sf_shape=tuple(kw["fc2_weight_sf"].shape),
                situ=(kw["act_alpha"], kw["act_beta"]),
                apply_topk_in_fc1=kw["apply_topk_in_fc1"],
                combine_format=kw["combine_format"],
            )
        )
        # The routed rows: their activation bytes, so a caller can see which rows it got.
        kw["combine_output"][:n, 0, : H // 2] = (
            kw["activation"].view(torch.uint8)[:n].to(torch.bfloat16)
        )


@pytest.fixture
def engine(monkeypatch):
    e = _Engine(direct=(96,))
    monkeypatch.setitem(
        sys.modules, "tensorrt_llm._torch.moe.custom_ops.cute_dsl_megamoe_custom_op", e.module()
    )
    tuner = types.SimpleNamespace(get=lambda: types.SimpleNamespace(is_tuning_mode=e.tuning))
    monkeypatch.setitem(
        sys.modules,
        "tensorrt_llm._torch.autotuner",
        types.SimpleNamespace(AutoTuner=tuner),
    )
    monkeypatch.setattr(torch.ops.trtllm, "cute_dsl_megamoe_nvfp4_blackwell", e.op, raising=False)
    K._PRIMED.clear()
    return e


def _ws(ep_size=4, rank=1, rungs=K.SM107_RUNGS, max_num_tokens=1024, chunk=16384):
    return K.build_workspace(
        object(),
        rank,
        ep_size,
        LOCAL,
        H,
        INTER,
        TOP_K,
        max_num_tokens,
        chunk,
        torch.device("cpu"),
        situ=(4.0, 25.0),
        rungs=rungs,
    )


def _experts():
    e = LOCAL
    return (
        torch.zeros(e, 2 * INTER, H // 2, dtype=torch.uint8),
        torch.zeros(e, 2 * INTER, H // 16, dtype=torch.uint8),
        torch.zeros(e, H, INTER // 2, dtype=torch.uint8),
        torch.zeros(e, H, INTER // 16, dtype=torch.uint8),
        torch.ones(e),
        torch.ones(e),
        torch.ones(e),
    )


def _rows(n, seed):
    g = torch.Generator().manual_seed(seed)
    x4 = torch.randint(0, 256, (n, H // 2), dtype=torch.uint8, generator=g)
    sf = torch.randint(0, 256, (n, SF_COLS), dtype=torch.uint8, generator=g)
    ids = torch.randint(0, 4 * LOCAL, (n, TOP_K), dtype=torch.int32, generator=g)
    weights = torch.rand(n, TOP_K, generator=g)
    return x4, sf, ids, weights


@pytest.mark.parametrize("order", [(40, 96, 40, 96), (96, 40, 96, 40)])
def test_replay_after_any_size(engine, order):
    """A call's writes depend only on its own rows: a graph captured at one row count replays
    the same staging after any other. A tail a replay of a larger size left behind is reset."""
    ws, ex = _ws(), _experts()
    rows = {40: _rows(40, 3), 96: _rows(96, 4)}
    seen = {}
    for n in order:
        bucket = K.select_bucket(ws, 96)
        bucket.topk_idx.fill_(7)  # what another graph's replay may have left
        bucket.topk_weights.fill_(3.0)
        K.run(ws, *rows[n], ex, 96, n)
        launch = engine.launches[-1]
        snap = (launch["ids"], launch["weights"], launch["activation_rows"])
        if n in seen:
            assert all(torch.equal(a, b) for a, b in zip(seen[n], snap))
        seen[n] = snap
        assert bool((launch["ids"][n:] == -1).all()) and not launch["weights"][n:].any()


# --- rubin_megamoe_in_bucket: the launch on a bucket its producers filled ---
