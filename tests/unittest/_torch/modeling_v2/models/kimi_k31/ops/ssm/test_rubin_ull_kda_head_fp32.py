# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the ULL target's kda_head_verify_fp32 entry (the K3.1-owned fused KDA head with
bounded F_A waits), built as ``FlatKda._prepare_head`` builds it: every cell against the FP64
reference (output rows, state checkpoints, conv page), and bit for bit against MR8's
``PreparedKdaHead`` on the same inputs (the bound changes no output bit).

The layer weight and input are small dyadic values whose projection sums are exact in FP32 in any
order (``test_rubin_ull_kda_head_verify.py``)."""

import types
import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell

__extra_import_path__ = [".", "../../../.."]
import _kda_case as case  # noqa: E402 — needs the path declared above
from _cells import certified_cells  # noqa: E402
from _validating import validating  # noqa: E402

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_ull_kda_head_fp32 import (  # noqa: E402, E501
    HIDDEN,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.ssm.rubin_ull_kda_head_fp32 import (  # noqa: E402, E501
    kda_head_verify_fp32 as op,
)

pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
        reason="the fused KDA head needs an SM10x GPU (tensor memory, 16-CTA clusters)",
    ),
    # spin_trap.record_address() starts the trap-record watcher, a daemon thread that lives as
    # long as the process (kernels/common/rubin_ull_spin_trap.py).
    pytest.mark.threadleak(enabled=False),
]

TARGET_CELLS: dict[str, tuple[Cell, ...]] = {}
CELLS = certified_cells(op, TARGET_CELLS)


def _layer_weight(heads: int, g: torch.Generator) -> torch.Tensor:
    """BF16 [pitch, hidden]: q / k / v / g and beta rows of multiples of 1/64 in [-1/32, 1/32],
    one +-1 per F_A row, zero padding rows."""
    dim, pitch = heads * 128, case.pitch_of(heads)
    weight = torch.zeros(pitch, HIDDEN, device="cuda")
    dense = list(range(4 * dim)) + list(range(4 * dim + 128, 4 * dim + 128 + heads))
    weight[dense] = torch.randint(-2, 3, (len(dense), HIDDEN), generator=g, device="cuda") / 64
    columns = torch.randint(0, HIDDEN, (128,), generator=g, device="cuda")
    signs = torch.randint(0, 2, (128,), generator=g, device="cuda") * 2 - 1
    weight[4 * dim + torch.arange(128, device="cuda"), columns] = signs.float()
    return weight.bfloat16().contiguous()


def _head(cls, weight, core, caches, tokens, polled, **order):
    """``FlatKda._prepare_head``: the head's own side stream, counter pair and F_A buffer.
    ``order``: ``PreparedKdaHeadFp32``'s launch order, set directly (a test-only bypass of
    ``FlatKda.head_order``'s SM-count fallback: one process, nothing else resident)."""
    head_module = case.module("cutedsl.kda.head")
    dim = core.dim
    side = head_module.KdaSide(weight.device)
    counter = torch.zeros(2, dtype=torch.int32, device="cuda")
    fa_out = torch.empty((tokens, weight.shape[0] - 4 * dim), dtype=torch.bfloat16, device="cuda")
    head = cls(weight, core, (), side, counter, fa_out, verify=((tokens, tokens),), **order)
    if polled:
        head.polled = (tokens,)
    return head.prepare_caches(caches), side


def _buffers(tokens, dim, polled, g):
    """(out, rearm): ``out`` armed to ``OUT_SENT`` and a previous buffer to re-arm when polled."""
    if not polled:
        return torch.full((tokens, dim), 7.0, dtype=torch.bfloat16, device="cuda"), None
    out = torch.full((tokens, dim), -1, dtype=torch.int16, device="cuda").view(torch.bfloat16)
    return out, torch.randn(tokens, dim, generator=g, device="cuda").bfloat16()


@pytest.mark.parametrize("order", ["event", "sibling"])
@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:44] for c in CELLS])
def test_certified_cells(cell, order) -> None:
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.ssm.rubin_ull_kda_head_fp32 import (  # noqa: E501
        PreparedKdaHeadFp32,
    )

    head_module = case.module("cutedsl.kda.head")
    assert head_module.OUT_SENT == 0xFFFF
    spec = cell.spec
    g = torch.Generator(device="cuda").manual_seed(zlib.crc32(cell.why.encode()) % 2**31)
    heads, tokens, polled = spec["heads"], spec["tokens"], spec["polled"]
    dim = heads * 128
    slot_count = tokens + 3
    caches = case.checkpoint_caches(slot_count, heads, tokens + 2, g)
    core = case.make_core(case.layer_weights(heads, g), heads, {})
    weight = _layer_weight(heads, g)
    x = (torch.randint(-2, 3, (tokens, HIDDEN), generator=g, device="cuda") / 4).bfloat16()
    slots = case.verify_slots(1, tokens, (), slot_count, g)
    accepted = torch.tensor([spec["accepted"]], dtype=torch.int32, device="cuda")
    out, rearm = _buffers(tokens, dim, polled, g)
    mr8_caches = tuple(t.clone() for t in caches)
    mr8_out, mr8_rearm = out.clone(), None if rearm is None else rearm.clone()

    head, side = _head(PreparedKdaHeadFp32, weight, core, caches, tokens, polled, order=order)
    args = (head, x, out, caches, slots, accepted, rearm)
    ref = op.reference(*args)
    before = tuple(t.clone() for t in caches)
    with validating(op):
        result = op(*args)
    side.join()
    torch.cuda.synchronize()
    assert result is out
    case.assert_core_matches(out, caches, before, slots, accepted, tokens, ref)
    if rearm is not None:
        assert bool((rearm.view(torch.int16) == -1).all()), "the previous buffer is not re-armed"
        assert not bool((out.view(torch.int16) == -1).any()), "a polled element left unwritten"

    # MR8's head on the same inputs: the bounded waits change no output or checkpoint bit.
    mr8, mr8_side = _head(head_module.PreparedKdaHead, weight, core, mr8_caches, tokens, polled)
    mr8.run_verify(x, mr8_out, caches=mr8_caches, slots=slots, accepted=accepted, rearm=mr8_rearm)
    mr8_side.join()
    torch.cuda.synchronize()
    assert torch.equal(out.view(torch.int16), mr8_out.view(torch.int16))
    for mine, theirs in zip(caches, mr8_caches):
        assert torch.equal(mine.view(torch.uint8), theirs.view(torch.uint8))
    if rearm is not None:
        assert torch.equal(rearm.view(torch.int16), mr8_rearm.view(torch.int16))


def test_is_valid_refuses_what_the_kernel_would_misplace() -> None:
    head = types.SimpleNamespace(
        core=types.SimpleNamespace(dim=768), weight=torch.zeros(case.pitch_of(6), HIDDEN)
    )
    x = torch.zeros(8, HIDDEN, dtype=torch.bfloat16)
    out = torch.zeros(8, 768, dtype=torch.bfloat16)
    slots = torch.arange(1, 9, dtype=torch.int32).view(1, 8)
    one = torch.tensor([1], dtype=torch.int32)
    fp32 = (torch.zeros(1), torch.zeros(1, dtype=torch.float32))
    op.is_valid(head, x, out, fp32, slots, one)
    with pytest.raises(ValueError, match="must be FP32"):
        op.is_valid(head, x, out, (fp32[0], fp32[1].bfloat16()), slots, one)
    with pytest.raises(ValueError, match="one request"):
        op.is_valid(head, torch.zeros(16, HIDDEN, dtype=torch.bfloat16), out, fp32, slots, one)
    with pytest.raises(ValueError, match="outside 1..q"):
        op.is_valid(head, x, out, fp32, slots, torch.tensor([9], dtype=torch.int32))


def _kernel_edges(graph):
    """(kernel node names, edges (from, to, type)) of a captured graph kept in debug mode
    (cuGraphGetEdges_v2: type 0 full, 1 programmatic)."""
    import ctypes

    class Edge(ctypes.Structure):
        _fields_ = [("from_port", ctypes.c_ubyte), ("to_port", ctypes.c_ubyte),
                    ("type", ctypes.c_ubyte), ("reserved", ctypes.c_ubyte * 5)]  # fmt: skip

    class Params(ctypes.Structure):  # CUDA_KERNEL_NODE_PARAMS_v2
        _fields_ = [("func", ctypes.c_void_p), ("grid", ctypes.c_uint * 3),
                    ("block", ctypes.c_uint * 3), ("smem", ctypes.c_uint),
                    ("params", ctypes.c_void_p), ("extra", ctypes.c_void_p),
                    ("kern", ctypes.c_void_p), ("ctx", ctypes.c_void_p)]  # fmt: skip

    cu = ctypes.CDLL("libcuda.so.1")
    g = ctypes.c_void_p(int(graph.raw_cuda_graph()))
    n = ctypes.c_size_t(0)
    assert cu.cuGraphGetNodes(g, None, ctypes.byref(n)) == 0
    nodes = (ctypes.c_void_p * n.value)()
    assert cu.cuGraphGetNodes(g, nodes, ctypes.byref(n)) == 0
    names = {}
    for node in nodes:
        kind = ctypes.c_int(-1)
        cu.cuGraphNodeGetType(ctypes.c_void_p(node), ctypes.byref(kind))
        name = ""
        if kind.value == 0:  # CU_GRAPH_NODE_TYPE_KERNEL
            p, s = Params(), ctypes.c_char_p()
            assert cu.cuGraphKernelNodeGetParams_v2(ctypes.c_void_p(node), ctypes.byref(p)) == 0
            if not (p.func and cu.cuFuncGetName(ctypes.byref(s), ctypes.c_void_p(p.func)) == 0):
                cu.cuKernelGetName(ctypes.byref(s), ctypes.c_void_p(p.kern))
            name = (s.value or b"?").decode(errors="replace")
        names[node] = name
    e = ctypes.c_size_t(0)
    assert cu.cuGraphGetEdges_v2(g, None, None, None, ctypes.byref(e)) == 0
    if e.value == 0:  # the arrays below would be empty: the call rejects them
        return list(names.values()), []
    frm, to, data = (ctypes.c_void_p * e.value)(), (ctypes.c_void_p * e.value)(), (Edge * e.value)()
    assert cu.cuGraphGetEdges_v2(g, frm, to, data, ctypes.byref(e)) == 0
    edges = [(names[frm[k]], names[to[k]], data[k].type) for k in range(e.value)]
    return list(names.values()), edges


@pytest.mark.parametrize("order", ["event", "sibling"])
def test_launch_order_edges(order) -> None:
    """The captured head / SignalGemm pair: "event" gives the head a programmatic in-edge from
    SignalGemm (fix (a)); "sibling" (MR8's order) none, the two only share their predecessor.
    One process with nothing else resident (``_head``'s bypass of the SM-count fallback)."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kernels.ssm.rubin_ull_kda_head_fp32 import (  # noqa: E501
        PreparedKdaHeadFp32,
    )

    g = torch.Generator(device="cuda").manual_seed(5)
    heads, tokens = 6, 8
    slot_count = tokens + 3
    caches = case.checkpoint_caches(slot_count, heads, tokens + 2, g)
    core = case.make_core(case.layer_weights(heads, g), heads, {})
    weight = _layer_weight(heads, g)
    x = (torch.randint(-2, 3, (tokens, HIDDEN), generator=g, device="cuda") / 4).bfloat16()
    slots = case.verify_slots(1, tokens, (), slot_count, g)
    accepted = torch.tensor([1], dtype=torch.int32, device="cuda")
    out, _ = _buffers(tokens, heads * 128, False, g)
    head, side = _head(PreparedKdaHeadFp32, weight, core, caches, tokens, False, order=order)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):  # warm: compile and first launch outside the capture
        head.run_verify(x, out, caches=caches, slots=slots, accepted=accepted)
        side.join()
    torch.cuda.synchronize()
    try:  # the cudaGraph_t outlives capture_end
        graph = torch.cuda.CUDAGraph(keep_graph=True)
    except TypeError:
        graph = torch.cuda.CUDAGraph()
        graph.enable_debug_mode()
    with torch.cuda.graph(graph, stream=stream):
        head.run_verify(x, out, caches=caches, slots=slots, accepted=accepted)
        side.join()
    names, edges = _kernel_edges(graph)
    signal = [n for n in names if "SignalGemm" in n]
    waiter = [n for n in names if "KdaHeadVerify" in n]
    assert len(signal) == 1 and len(waiter) == 1, names
    into_head = [(a, t) for a, b, t in edges if b == waiter[0]]
    from_signal = [t for a, t in into_head if a == signal[0]]
    if order == "event":
        assert from_signal == [1], into_head  # one programmatic edge
    else:
        assert from_signal == [], into_head
