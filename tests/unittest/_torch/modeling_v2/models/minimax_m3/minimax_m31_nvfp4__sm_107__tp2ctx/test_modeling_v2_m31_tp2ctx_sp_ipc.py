# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP2CTX_SP on MPI IPC, single-GPU parts: the device-flag barrier op (epoch bump, release to the peer slot, passes
once the peer's slot reaches the epoch), and the layout fact the sequence-parallel all-gather relies on: the
swizzled MXFP8 scale factors of the owned half of a T % 256 == 0 step are exactly the contiguous row-tile block of
the full step's scale factors, and the e4m3 rows are the full step's rows. Also the workspace gate from source. The
two-rank collectives are in the gate kit (bitwise vs NCCL). GPU (any sm_10x build carrying the ops)."""

from pathlib import Path

import pytest
import torch

import tensorrt_llm  # noqa: F401  (registers torch.ops.trtllm)
from tensorrt_llm._torch._experimental.modeling_v2.catalog.comm import tp2ctx_ipc

H = 6144
_BAR = getattr(torch.ops.trtllm, "minimax_m31_tp2ctx_ipc_barrier", None)
pytestmark = pytest.mark.skipif(
    _BAR is None or not torch.cuda.is_available(),
    reason="needs a build carrying minimax_m31_tp2ctx_ipc_barrier",
)


@pytest.mark.parametrize("rank", [0, 1])
def test_barrier_bumps_epoch_and_publishes(rank):
    flags = torch.zeros(2, dtype=torch.int32, device="cuda")
    flags[1 - rank] = 1 << 30  # the "peer" is already far ahead, so the spin passes
    state = torch.zeros(4, dtype=torch.int32, device="cuda")
    for epoch in (1, 2, 3):
        _BAR(flags.data_ptr(), flags.data_ptr(), state, rank)
        torch.cuda.synchronize()
        assert int(state[0]) == epoch
        assert int(flags[rank]) == epoch


def test_barrier_in_a_cuda_graph_advances_per_replay():
    flags = torch.zeros(2, dtype=torch.int32, device="cuda")
    flags[1] = 1 << 30
    state = torch.zeros(4, dtype=torch.int32, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _BAR(flags.data_ptr(), flags.data_ptr(), state, 0)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    assert int(state[0]) == 3 and int(flags[0]) == 3


@pytest.mark.parametrize("num_tokens", [6144, 8192, 16384])
@pytest.mark.parametrize("kind", ["randn", "stress"])
def test_owned_half_quantize_is_the_full_steps_block(num_tokens, kind):
    g = torch.Generator(device="cuda").manual_seed(num_tokens)
    x = torch.randn(num_tokens, H, generator=g, device="cuda")
    if kind == "stress":
        e = torch.randint(-12, 13, (num_tokens, H // 32, 1), generator=g, device="cuda").float()
        x = (x.view(num_tokens, H // 32, 32) * torch.exp2(e)).view(num_tokens, H)
    x = x.to(torch.bfloat16)
    q_full, sf_full = torch.ops.trtllm.mxfp8_quantize(x, True)
    half = num_tokens // 2
    sf_full = sf_full.view(torch.uint8).reshape(-1)
    per_half = sf_full.numel() // 2
    for r in (0, 1):
        q, sf = torch.ops.trtllm.mxfp8_quantize(x[r * half : (r + 1) * half].contiguous(), True)
        assert torch.equal(q.view(torch.uint8), q_full.view(torch.uint8)[r * half : (r + 1) * half])
        assert torch.equal(
            sf.view(torch.uint8).reshape(-1), sf_full[r * per_half : (r + 1) * per_half]
        )


@pytest.mark.parametrize("num_tokens", [5632, 6144, 8192, 12288, 16384])
@pytest.mark.parametrize("kind", ["randn", "stress"])
def test_owned_router_rows_are_the_full_steps_rows(num_tokens, kind):
    """route_owned_full_shape (zero-padded full-T call) == today's full-T router rows, bitwise, for both halves, as
    served (TF32 on); F.linear at M = T / 2 is NOT bitwise to it below 8192 (PR-CTX 838135: split-K at M <= 4096)."""
    torch.backends.cuda.matmul.allow_tf32 = True
    g = torch.Generator(device="cuda").manual_seed(num_tokens + 1)
    x = torch.randn(num_tokens, H, generator=g, device="cuda")
    if kind == "stress":
        e = torch.randint(-12, 13, (num_tokens, H // 32, 1), generator=g, device="cuda").float()
        x = (x.view(num_tokens, H // 32, 32) * torch.exp2(e)).view(num_tokens, H)
    x = x.to(torch.bfloat16).float()
    w = (torch.randn(128, H, generator=g, device="cuda") * 0.02).float()
    full = torch.nn.functional.linear(x, w)
    ws = tp2ctx_ipc.Tp2ctxIpcWorkspace.__new__(
        tp2ctx_ipc.Tp2ctxIpcWorkspace
    )  # router helper only: no IPC needed
    ws.max_tokens, ws.device, ws._router_buf = 16384, torch.device("cuda"), None
    for rank in (0, 1):
        ws.rank = rank
        first, count = ws.own_rows(num_tokens)
        got = ws.route_owned_full_shape(x[first : first + count], w, num_tokens)
        assert torch.equal(got.view(torch.int32), full[first : first + count].view(torch.int32))


def test_workspace_gate_and_kill_switch(monkeypatch):
    monkeypatch.delenv("TRTLLM_M31_DISABLE_TP2CTX_SP", raising=False)
    assert tp2ctx_ipc.enabled()
    monkeypatch.setenv("TRTLLM_M31_DISABLE_TP2CTX_SP", "1")
    assert not tp2ctx_ipc.enabled()
    src = Path(tp2ctx_ipc.__file__).read_text()
    assert "2 * _SF_TILE_ROWS <= num_tokens <= self.max_tokens" in src
    assert "num_tokens * self.hidden * 2 > _nccl_fallback_bytes()" in src
    assert "from tensorrt_llm._ipc_utils import IpcMemory" in src
    assert "return torch.nn.functional.linear(full, weight)[first : first + count]" in src
