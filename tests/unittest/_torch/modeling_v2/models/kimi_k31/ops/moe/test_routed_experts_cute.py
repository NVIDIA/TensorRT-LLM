# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the Kimi K3.1 pre-routed routed experts (``k31_routed_experts``): the calls of the generation shapes
(TEP split experts, DEP16 full-width experts) run the CuTe DSL grouped GEMMs and come back bitwise equal to the
runner; prefill-size calls and padded intermediate shards take the runner before any tiling choice; under per-layer
dynamic EPLB a rank's window of physical slots past the 896 experts computes what the same slots at the front of the
expert id space compute, on both kernels."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe import (
    routed_experts as cute_backend,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe.w4a8_moe_runner import (  # noqa: E501
    HIDDEN,
    NUM_EXPERTS,
    TOP_K,
    k31_w4a8_moe_runner,
)
from tensorrt_llm._utils import is_sm_100f

if not torch.cuda.is_available() or not is_sm_100f():
    pytest.skip("the W4A8 CuTe DSL MoE needs an SM10x GPU", allow_module_level=True)

_GATE_CAP, _LINEAR_CAP = 4.0, 25.0


def _problem(rows: int, local: int, inter: int, seed: int = 0) -> dict:
    """Random prepared operands of one rank (local experts [0, local)) and random distinct top-16 ids."""
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(seed)

    def nib(*shape):
        return torch.randint(0, 256, shape, generator=g, device=dev, dtype=torch.uint8)

    def sf(*shape):
        return torch.randint(118, 124, shape, generator=g, device=dev, dtype=torch.uint8)

    x = (torch.randn(rows, HIDDEN, generator=g, device=dev) * 2).to(torch.bfloat16)
    xq, xsf = torch.ops.trtllm.mxfp8_quantize(x, False, alignment=32)
    ids = torch.argsort(torch.rand(rows, NUM_EXPERTS, generator=g, device=dev), dim=1)[:, :TOP_K]
    w = torch.rand(rows, TOP_K, generator=g, device=dev) + 0.5
    return dict(
        x=xq,
        x_sf=xsf.view(rows, -1),
        ids=ids.to(torch.int32).contiguous(),
        w=(w / w.sum(1, keepdim=True)).to(torch.bfloat16).contiguous(),
        w1=nib(local, 2 * inter, HIDDEN // 2),
        w1s=sf(local, 2 * inter, HIDDEN // 32),
        w2=nib(local, HIDDEN, inter // 2),
        w2s=sf(local, HIDDEN, inter // 32),
        alpha=torch.full((local,), _GATE_CAP, dtype=torch.float32, device=dev),
        beta=torch.full((local,), _LINEAR_CAP, dtype=torch.float32, device=dev),
        local=local,
        inter=inter,
    )


def _call(p: dict, valid_inter=None) -> torch.Tensor:
    return cute_backend.k31_routed_experts(
        p["x"], p["x_sf"], p["ids"], p["w"], p["w1"], p["w1s"], p["w2"], p["w2s"], p["alpha"], p["beta"],
        p["inter"], p["inter"] if valid_inter is None else valid_inter, 0, p["local"], _GATE_CAP, _LINEAR_CAP, 8192,
        False,
    )  # fmt: skip


def _runner(p: dict) -> torch.Tensor:
    return k31_w4a8_moe_runner(
        p["x"], p["x_sf"], p["ids"], p["w"], p["w1"], p["w1s"], p["alpha"], p["beta"], p["w2"], p["w2s"],
        p["inter"], p["inter"], 0, p["local"], None,
    )  # fmt: skip


@pytest.mark.parametrize(
    "rows,local,inter",
    [
        (8, 112, 768),  # TEP32 decode step, 1-CTA kernel
        (72, 224, 768),  # TEP16 verify step, 1-CTA kernel
        (1536, 56, 3072),  # DEP16 decode, 96 rows per rank x 16, CTA pairs
        (3584, 56, 3072),  # DEP16 decode, 224 rows per rank x 16, CTA pairs
    ],
)
def test_generation_calls_are_bitwise_to_the_runner(rows, local, inter, monkeypatch) -> None:
    p = _problem(rows, local, inter, seed=rows)
    runner_calls = []
    real_runner = cute_backend.k31_runner_moe
    monkeypatch.setattr(
        cute_backend,
        "k31_runner_moe",
        lambda *a, **k: runner_calls.append(1) or real_runner(*a, **k),
    )
    out = _call(p)
    assert not runner_calls, "a generation call took the runner"
    assert out.shape == (rows, HIDDEN) and out.dtype is torch.bfloat16
    assert torch.equal(out.view(torch.int16), _runner(p).view(torch.int16)), (
        "not bitwise vs the runner"
    )


@pytest.mark.parametrize("rows", [cute_backend.MAX_ROWS + 1, 8192])
def test_prefill_calls_reach_the_runner_before_any_tiling(rows, monkeypatch) -> None:
    def no_tiling(*args, **kwargs):
        raise AssertionError("pick_tiling ran for a prefill-size call")

    marker = object()
    monkeypatch.setattr(cute_backend, "pick_tiling", no_tiling)
    monkeypatch.setattr(cute_backend, "k31_runner_moe", lambda *a, **k: marker)
    p = _problem(rows, 8, 3072)  # full-width experts; the stand-in runner never reads them
    assert _call(p) is marker


def test_padded_intermediate_shards_reach_the_runner(monkeypatch) -> None:
    marker = object()
    monkeypatch.setattr(cute_backend, "k31_runner_moe", lambda *a, **k: marker)
    p = _problem(72, 224, 768)
    assert _call(p, valid_inter=640) is marker


@pytest.mark.parametrize("rows", [1536, 8192], ids=["cute_pairs", "runner"])
def test_physical_slot_window_matches_the_resident_window(rows) -> None:
    """DEP16 under per-layer dynamic EPLB with 1024 slots: rank 15 computes slots [960, 1024) (56 resident + 8
    helper slots), past the 896 experts. The same slots rotated to the front of the slot space (the window to [0, 64),
    every other slot one-to-one past it) give the same bits."""
    slots, local, offset = 1024, 64, 960
    p = _problem(rows, local, 3072, seed=rows + 1)
    g = torch.Generator(device="cuda").manual_seed(rows)
    ids = torch.argsort(torch.rand(rows, slots, generator=g, device="cuda"), dim=1)[:, :TOP_K]
    # About a sixteenth of the picks land on the window, as on one of 16 ranks.
    p["ids"] = ids.to(torch.int32).contiguous()
    window = (p["ids"] >= offset) & (p["ids"] < offset + local)
    assert bool(window.any()) and not bool(window.all())
    front = torch.where(window, p["ids"] - offset, p["ids"] + local)

    def call(ids: torch.Tensor, window_offset: int) -> torch.Tensor:
        return cute_backend.k31_routed_experts(
            p["x"], p["x_sf"], ids.to(torch.int32).contiguous(), p["w"], p["w1"], p["w1s"], p["w2"], p["w2s"],
            p["alpha"], p["beta"], 3072, 3072, window_offset, local, _GATE_CAP, _LINEAR_CAP, 8192, True,
            num_experts=slots,
        )  # fmt: skip

    got = call(p["ids"], offset)
    want = call(front, 0)
    assert torch.equal(got.view(torch.int16), want.view(torch.int16))
