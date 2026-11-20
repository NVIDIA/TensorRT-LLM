# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP2CTX_SP on MPI IPC, single-GPU parts: the device-flag barrier op (epoch bump, release to the peer slot, passes
once the peer's slot reaches the epoch), and the layout fact the sequence-parallel all-gather relies on: the
swizzled MXFP8 scale factors of the owned half of a T % 256 == 0 step are exactly the contiguous row-tile block of
the full step's scale factors, and the e4m3 rows are the full step's rows. Also the workspace kill switch. The
two-rank collectives need two GPUs. GPU: SM107, where the family library carries the ops."""

from types import SimpleNamespace

import pytest
import torch

import tensorrt_llm  # noqa: F401  (registers torch.ops.trtllm)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.native import family_ops
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.comm import tp2ctx_ipc
from tensorrt_llm._utils import get_sm_version

H = 6144
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107, reason="SM107 (Rubin) only"
)


def _barrier():
    return family_ops().minimax_m31_tp2ctx_ipc_barrier


@pytest.mark.parametrize("rank", [0, 1])
def test_barrier_bumps_epoch_and_publishes(rank):
    flags = torch.zeros(2, dtype=torch.int32, device="cuda")
    flags[1 - rank] = 1 << 30  # the "peer" is already far ahead, so the spin passes
    state = torch.zeros(4, dtype=torch.int32, device="cuda")
    for epoch in (1, 2, 3):
        _barrier()(flags.data_ptr(), flags.data_ptr(), state, rank)
        torch.cuda.synchronize()
        assert int(state[0]) == epoch
        assert int(flags[rank]) == epoch


@pytest.mark.parametrize("num_tokens", [6144, 16384])
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


class _FakeIpcMemory:
    """IpcMemory stand-in: a local device buffer; the peer pointer is linked by ``_ws_pair``."""

    def __init__(self, mapping, size: int):
        self.size = size
        self.buf = torch.zeros(size, dtype=torch.uint8, device="cuda")
        self.local_ptr = self.buf.data_ptr()
        self.peer_ptrs = [0, 0]
        self.peer_ptrs[mapping.tp_rank] = self.local_ptr


def _ws_pair(monkeypatch, max_tokens: int = 1024):
    """Rank 0 and rank 1 workspaces built by the real __init__ over linked fake IPC buffers (one GPU)."""
    import tensorrt_llm._ipc_utils as ipc_utils

    monkeypatch.setattr(ipc_utils, "IpcMemory", _FakeIpcMemory)
    dev = torch.device("cuda")
    ws = [
        tp2ctx_ipc.Tp2ctxIpcWorkspace(SimpleNamespace(tp_size=2, tp_rank=r), max_tokens, H, dev)
        for r in (0, 1)
    ]
    for k in ws[0]._mem:
        ws[0]._mem[k].peer_ptrs[1] = ws[1]._mem[k].local_ptr
        ws[1]._mem[k].peer_ptrs[0] = ws[0]._mem[k].local_ptr
    return ws


def _stub_peer_ops(monkeypatch):
    """The peer push as a plain device copy into the peer pointer; the barrier as a no-op (one process)."""
    pushes = []
    # Bound now: a test may later wrap the module's _view to count the workspace's own calls.
    raw_view = tp2ctx_ipc._view

    def peer_push(src, dst):
        pushes.append((src.data_ptr(), src.numel() * src.element_size(), dst))
        raw_view(dst, (src.numel(),), torch.uint8, src.device).copy_(src.view(torch.uint8))

    ops = SimpleNamespace(
        minimax_m31_tp2ctx_peer_push=peer_push,
        minimax_m31_tp2ctx_ipc_barrier=lambda *args: None,
    )
    monkeypatch.setattr(tp2ctx_ipc, "family_ops", lambda: ops)
    return pushes


# kind -> (dtype, columns) of the [T, columns] view the boundary producers and consumers address.


@pytest.mark.parametrize("engine", ["sync", "sm", "ce"])
@pytest.mark.parametrize("num_tokens", [256, 1000])
def test_push_owned_moves_each_ranks_owned_bytes_to_the_peer(monkeypatch, engine, num_tokens):
    """The real push paths: each rank's owned bytes of every kind land at the same offsets of the peer's buffer,
    and every push source is this rank's owned range of its own buffer."""
    if engine == "ce":
        pytest.importorskip("cuda.bindings")
    ws = _ws_pair(monkeypatch)
    pushes = _stub_peer_ops(monkeypatch)
    kinds = list(tp2ctx_ipc._KINDS)
    want = {}
    for w in ws:
        for kind in kinds:
            view = w.gather_view(kind, num_tokens)
            flat = view.reshape(-1) if view.dim() == 1 else view.view(torch.uint8).reshape(-1)
            if tp2ctx_ipc._KINDS[kind] == "sf_swizzled":
                sl = w.own_sf_slice(num_tokens)
                lo, hi = sl.start, sl.stop
            else:
                first, count = w.own_rows(num_tokens)
                per = w._row_bytes[tp2ctx_ipc._KINDS[kind]]
                lo, hi = first * per, (first + count) * per
            data = torch.randint(0, 256, (hi - lo,), dtype=torch.uint8, device="cuda")
            flat[lo:hi].copy_(data)
            want[(w.rank, kind)] = (lo, hi, data)
    for w in ws:
        if engine == "sync":
            w.push_owned(kinds, num_tokens)
        else:
            torch.cuda.current_stream().wait_event(
                w.push_owned_async(kinds, num_tokens, engine=engine)
            )
    torch.cuda.synchronize()
    for w in ws:
        peer = ws[w.peer]
        for kind in kinds:
            lo, hi, data = want[(w.rank, kind)]
            assert torch.equal(peer._local(kind)[lo:hi], data), (w.rank, kind)
    if engine != "ce":  # the copy-engine path copies by raw pointer, not through peer_push
        assert len(pushes) == 2 * len(kinds)
        for src_ptr, nbytes, dst in pushes:
            owner = next(
                w for w in ws for k in kinds if w._ptr(k) <= src_ptr < w._ptr(k) + w._sizes[k]
            )
            kind = next(
                k for k in kinds if owner._ptr(k) <= src_ptr < owner._ptr(k) + owner._sizes[k]
            )
            lo, hi, _ = want[(owner.rank, kind)]
            assert (src_ptr, nbytes, dst) == (
                owner._ptr(kind) + lo,
                hi - lo,
                owner._ptr(kind, peer=True) + lo,
            )
