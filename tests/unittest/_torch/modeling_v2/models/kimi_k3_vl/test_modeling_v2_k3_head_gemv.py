# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the k3_head_gemv catalog entry."""

import functools
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.ops.gemm.k3_head_gemv import (
    K3HeadGemvWorkspace,
    k3_head_gemv,
)

__extra_import_path__ = ["../.."]
from _validating import validating

assert torch.cuda.is_available(), "k3_head_gemv requires a CUDA device"
# The receipts are sm_100 ones; CI's other architectures skip this file.
pytestmark = pytest.mark.skipif(
    torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="certified on sm_100 and sm_103",
)

TOL = 8e-3
VOCAB_SHARD, HIDDEN = 10240, 7168  # Kimi K3's 163840-row LM head over 16 ranks
TILES = VOCAB_SHARD // 128


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16)


def _same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and torch.equal(_bits(a), _bits(b))


def _rel(y: torch.Tensor, ref: torch.Tensor) -> float:
    return (y.double() - ref).abs().max().item() / ref.abs().max().item()


@functools.lru_cache(maxsize=None)
def _inputs(
    seed: int = 20261001, n: int = VOCAB_SHARD, k: int = HIDDEN
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Two shard weights (e.g. the target's LM head and DSpark's draft head) and 8 activation rows."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    weight = (torch.randn(n, k, generator=gen, device="cuda") * 0.02).bfloat16()
    weight2 = (torch.randn(n, k, generator=gen, device="cuda") * 0.02).bfloat16()
    x8 = torch.randn(8, k, generator=gen, device="cuda").bfloat16()
    # The kernel reads the weight before its PDL grid-dependency wait: it is written well before any call.
    torch.cuda.synchronize()
    return weight, weight2, x8


def _workspace() -> K3HeadGemvWorkspace:
    return K3HeadGemvWorkspace.create(VOCAB_SHARD, HIDDEN, torch.device("cuda"))


def _at_rest(ws: K3HeadGemvWorkspace) -> bool:
    """Whether every flag / count word of the workspace is back at zero."""
    return not bool(ws.flags.any()) and not bool(ws.claim.any())


@pytest.mark.parametrize("cell", k3_head_gemv.CELLS, ids=[c.why[:40] for c in k3_head_gemv.CELLS])
def test_certified_cells(cell) -> None:
    """Every M on one workspace: shape, dtype, error, rerun and 8-row-call bits, the words at zero after each call;
    keep_tiles 40 and 80 bit-identical to 0."""
    n, k = (cell.spec[key] for key in ("n", "k"))
    weight, weight2, x8 = _inputs(zlib.crc32(cell.why.encode()) % 2**31, n, k)
    weight = (weight, weight2)[cell.spec["weight_index"]]
    assert weight.shape == (n, k)
    ws = K3HeadGemvWorkspace.create(n, k, weight.device)
    assert ws.schedule == "streamk" and ws.chunk_tiles == 0
    assert ws.partials.dtype == torch.float32 and ws.flags.dtype == torch.int32
    assert ws.claim.dtype == torch.int32 and ws.claim.numel() == 1
    # One flag word and one [128 x 8] fp32 partial slot per (tile, piece).
    assert ws.flags.numel() % TILES == 0 and ws.partials.numel() == ws.flags.numel() * 128 * 8
    assert _at_rest(ws)
    ref8 = k3_head_gemv.reference(x8, weight, ws)
    with validating(k3_head_gemv):
        y8 = k3_head_gemv(x8, weight, ws)
        for m in cell.spec["ms"]:
            x = x8[:m]
            y = k3_head_gemv(x, weight, ws)
            again = k3_head_gemv(x, weight, ws)
            assert y.shape == (m, n) and y.dtype == torch.bfloat16 and y.device == x.device
            assert y.is_contiguous()
            err = _rel(y, ref8[:m])
            assert err <= TOL, f"M={m}: rel {err:.3e}"
            assert _same(y, again), f"M={m}: a rerun differs"
            assert _same(y, y8[:m]), f"M={m}: rows differ from the 8-row call"
            assert _at_rest(ws), f"M={m}: words left raised"
        for m in (1, 8):
            for keep in cell.spec["keep_tiles"]:
                y = k3_head_gemv(x8[:m], weight, ws, keep_tiles=keep)
                ref = k3_head_gemv.reference(x8[:m], weight, ws, keep_tiles=keep)
                assert _rel(y, ref) <= TOL, f"M={m} keep_tiles={keep}"
                assert _same(y, y8[:m]), f"M={m} keep_tiles={keep}: result differs"
        assert _at_rest(ws)


def test_m_dipping_and_growing_on_one_workspace() -> None:
    """M 8, 8, 2, 7, 8, 1, 1, 8, 3, 8 back to back on one stream and one workspace: every call bit-identical to the
    same call on a fresh workspace, the words at zero after."""
    weight, _, x8 = _inputs()
    seq = (8, 8, 2, 7, 8, 1, 1, 8, 3, 8)
    want = {m: k3_head_gemv(x8[:m], weight, _workspace()) for m in set(seq)}
    ws = _workspace()
    got = [k3_head_gemv(x8[:m], weight, ws) for m in seq]
    for i, (m, y) in enumerate(zip(seq, got)):
        assert _same(y, want[m]), f"call {i} (M={m}) differs"
    assert _at_rest(ws)


def test_two_workspaces_interleaved_and_two_weights_on_one() -> None:
    """Two workspaces of the shard's shape, each with its own weight, their calls alternating on one stream; then both
    weights alternating on one workspace: every call bit-identical to the same call on a fresh workspace, every
    workspace's words at zero after."""
    weight, weight2, x8 = _inputs()
    weights = (weight, weight2)
    want = {}
    for m in (1, 4, 8):
        for i in (0, 1):
            want[(m, i)] = k3_head_gemv(x8[:m], weights[i], _workspace())
    ws_a, ws_b = _workspace(), _workspace()
    for m in (8, 1, 4, 1, 8):
        for i, ws in ((0, ws_a), (1, ws_b)):
            y = k3_head_gemv(x8[:m], weights[i], ws)
            assert _same(y, want[(m, i)]), f"interleaved: M={m} weight {i}"
    for m, i in ((8, 1), (8, 0), (1, 1), (4, 0), (4, 1), (1, 0)):
        y = k3_head_gemv(x8[:m], weights[i], ws_a)
        assert _same(y, want[(m, i)]), f"one workspace: M={m} weight {i}"
    assert _at_rest(ws_a) and _at_rest(ws_b)


def test_graph_replays_between_eager_calls() -> None:
    """A CUDA graph of M 1, 8 and 3 calls on one workspace, captured once and replayed three times on the stream of the
    eager calls, with the activation rewritten in place before each replay and an eager M 8 call on the same workspace
    after it: every replayed and eager result bit-identical to the same call on a fresh workspace, the words at zero
    after each round."""
    weight, _, x8 = _inputs()
    ws = _workspace()
    x_buf = x8.clone()
    for m in (1, 3, 8):
        k3_head_gemv(x_buf[:m], weight, ws)  # compiles outside the capture
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = {m: k3_head_gemv(x_buf[:m], weight, ws) for m in (1, 8, 3)}
    gen = torch.Generator(device="cuda").manual_seed(4)
    for rep in range(3):
        x_buf.copy_(torch.randn(8, HIDDEN, generator=gen, device="cuda").bfloat16())
        graph.replay()
        between = k3_head_gemv(x_buf, weight, ws)
        fresh = _workspace()
        for m in (1, 3, 8):
            assert _same(out[m], k3_head_gemv(x_buf[:m], weight, fresh)), f"replay {rep}, M={m}"
        assert _same(between, out[8]), f"replay {rep}: the eager call differs"
        assert _at_rest(ws), f"replay {rep}: words left raised"


def test_refusals() -> None:
    """Negative controls, none of which launches anything: a workspace created for another weight shape ([5120, 7168]:
    other flag-word and partial sizes) is refused and stays at rest, and a call right after on a right workspace is
    unaffected; creating a workspace under CUDA-graph capture, an unknown schedule, M 0 or 9, and N not a multiple of
    128 are refused."""
    weight, _, x8 = _inputs()
    other = K3HeadGemvWorkspace.create(VOCAB_SHARD // 2, HIDDEN, torch.device("cuda"))
    with pytest.raises(ValueError, match="workspace"):
        k3_head_gemv(x8[:1], weight, other)
    assert _at_rest(other)
    ws = _workspace()
    assert _same(k3_head_gemv(x8[:1], weight, ws), k3_head_gemv(x8[:1], weight, _workspace()))
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="capture"):
        with torch.cuda.graph(graph):
            _workspace()
    with pytest.raises(ValueError, match="unknown schedule"):
        K3HeadGemvWorkspace.create(VOCAB_SHARD, HIDDEN, torch.device("cuda"), schedule="splitk")
    for m in (0, 9):
        x = torch.zeros(m, HIDDEN, dtype=torch.bfloat16, device="cuda")
        with pytest.raises(ValueError, match="unsupported call"):
            k3_head_gemv(x, weight, ws)
    odd = torch.zeros(200, HIDDEN, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="unsupported call"):
        k3_head_gemv(x8[:1], odd, ws)
    assert _at_rest(ws)


@pytest.mark.parametrize("operand", ("x", "weight"))
def test_guard_refuses_misaligned_inputs(operand) -> None:
    weight, _, x8 = _inputs()
    x = x8[:1]
    source = x if operand == "x" else weight
    flat = torch.empty(source.numel() + 1, dtype=source.dtype, device=source.device)
    misaligned = flat[1:].view(source.shape)
    if operand == "x":
        x = misaligned
    else:
        weight = misaligned
    with pytest.raises(AssertionError, match="16-byte-aligned"):
        with validating(k3_head_gemv):
            k3_head_gemv(x, weight, _workspace())


def test_guard_refuses_weight_on_another_device() -> None:
    _, _, x8 = _inputs()
    weight = torch.empty(VOCAB_SHARD, HIDDEN, dtype=torch.bfloat16)
    with pytest.raises(AssertionError, match="weight must be on"):
        with validating(k3_head_gemv):
            k3_head_gemv(x8[:1], weight, _workspace())
