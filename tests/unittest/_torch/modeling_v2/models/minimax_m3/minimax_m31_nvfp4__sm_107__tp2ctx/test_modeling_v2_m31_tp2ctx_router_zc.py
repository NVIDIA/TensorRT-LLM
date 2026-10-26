# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP2CTX_ROUTER_ZC: the pre-MoE boundary's fp32 router rows written in place into the router's full-step buffer.

route_owned_full_shape must return the owned rows' logits bit for bit as the per-call zero + copy path did, whatever
the rows outside this rank's own hold (earlier router inputs, or anything else), at the production row counts and on
both ranks. The workspace's IPC memory is not needed for the router buffer, so the tests build the object bare.
"""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm import tp2ctx_ipc

_DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
_HIDDEN = 6144


def _ws(rank: int, max_tokens: int = 16384) -> tp2ctx_ipc.Tp2ctxIpcWorkspace:
    ws = object.__new__(tp2ctx_ipc.Tp2ctxIpcWorkspace)
    ws.rank, ws.peer = rank, 1 - rank
    ws.max_tokens, ws.hidden, ws.device = max_tokens, _HIDDEN, _DEVICE
    ws._router_buf = None
    return ws


def _reference(ws, x_owned, weight, rows):
    """The pre-item path: a zeroed [T, K] buffer, the owned rows copied in, F.linear at M = T."""
    first, count = ws.own_rows(rows)
    full = torch.zeros((rows, x_owned.shape[1]), dtype=torch.float32, device=x_owned.device)
    full[first : first + count].copy_(x_owned)
    return torch.nn.functional.linear(full, weight)[first : first + count]


def _stress_rows(n, k, gen):
    """Router-input-like rows with a 2^+-12 per-row scale spread (normed hidden rows times a wide gain)."""
    x = torch.randn((n, k), generator=gen, dtype=torch.float32, device=_DEVICE)
    scale = torch.pow(2.0, torch.randint(-12, 13, (n, 1), generator=gen, device=_DEVICE).float())
    return x * scale


@pytest.mark.parametrize("tf32", [False, True])
@pytest.mark.parametrize("rows", [16384, 8000, 5632])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("experts", [128, 256])
def test_in_place_rows_are_bitwise_the_zero_copy_path(monkeypatch, tf32, rows, rank, experts):
    if _DEVICE.type == "cpu" and rows > 5632:
        pytest.skip("CPU stand-in: the smallest row count only")
    monkeypatch.setattr(tp2ctx_ipc, "_ROUTER_ZC", True)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", tf32)
    gen = torch.Generator(device=_DEVICE).manual_seed(rows + 7 * rank + experts)
    ws = _ws(rank)
    weight = torch.randn((experts, _HIDDEN), generator=gen, dtype=torch.float32, device=_DEVICE)
    first, count = ws.own_rows(rows)
    # Earlier calls leave other rows in the buffer: fill everything with garbage (incl. inf / nan / huge) first.
    junk = ws._router_full(ws.max_tokens, _HIDDEN)
    junk.copy_(_stress_rows(ws.max_tokens, _HIDDEN, gen))
    junk[::97, 5] = float("inf")
    junk[::89, 11] = float("nan")
    junk[::83, 17] = 3.0e38
    for step in range(2):  # two boundaries in a row: the second sees the first one's rows
        x = _stress_rows(count, _HIDDEN, gen)
        view = ws.router_owned_input(rows, _HIDDEN)
        assert view is not None and view.is_contiguous() and view.shape == (count, _HIDDEN)
        view.copy_(x)  # what the NORMQ op's f32_out does
        got = ws.route_owned_full_shape(view, weight, rows)
        ref = _reference(ws, x, weight, rows)
        assert got.shape == (count, experts)
        assert torch.equal(got.view(torch.int32), ref.view(torch.int32)), f"step {step}"


@pytest.mark.parametrize("rank", [0, 1])
def test_a_separate_input_is_still_copied(monkeypatch, rank):
    monkeypatch.setattr(tp2ctx_ipc, "_ROUTER_ZC", True)
    rows = 5632
    gen = torch.Generator(device=_DEVICE).manual_seed(11 + rank)
    ws = _ws(rank)
    weight = torch.randn((128, _HIDDEN), generator=gen, device=_DEVICE)
    first, count = ws.own_rows(rows)
    x = _stress_rows(count, _HIDDEN, gen)
    got = ws.route_owned_full_shape(x, weight, rows)
    assert torch.equal(got.view(torch.int32), _reference(ws, x, weight, rows).view(torch.int32))
    assert torch.equal(ws._router_buf[first : first + count], x)


def test_kill_switch_restores_the_zero_copy_path(monkeypatch):
    monkeypatch.setattr(tp2ctx_ipc, "_ROUTER_ZC", False)
    rows = 5632
    gen = torch.Generator(device=_DEVICE).manual_seed(3)
    ws = _ws(1)
    assert ws.router_owned_input(rows, _HIDDEN) is None
    weight = torch.randn((128, _HIDDEN), generator=gen, device=_DEVICE)
    first, count = ws.own_rows(rows)
    ws._router_full(rows, _HIDDEN).fill_(5.0)
    x = _stress_rows(count, _HIDDEN, gen)
    got = ws.route_owned_full_shape(x, weight, rows)
    full = ws._router_buf[:rows]
    assert int((full[:first] != 0).sum()) == 0 and int((full[first + count :] != 0).sum()) == 0
    assert torch.equal(got.view(torch.int32), _reference(ws, x, weight, rows).view(torch.int32))


def test_switch_reads_the_documented_variable():
    import inspect

    assert "TRTLLM_M31_DISABLE_TP2CTX_ROUTER_ZC" in inspect.getsource(tp2ctx_ipc)
    assert tp2ctx_ipc.router_zc_enabled() is tp2ctx_ipc._ROUTER_ZC
