# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The op-level MoE all-to-all off the GPU: the builder's layout is the engine strategy's, the
builder's setup calls, the dispatch / combine entries' op calls equal the engine
``NVLinkOneSided`` strategy's on the same inputs (ops recorded), the counted-write rule, the
guards, the construction check of the strategy's variables, and the two collective references.

The entries' cells run over 4 ranks on SM107 in rubin_multirank/test_rubin_moe_a2a.py
(this file carries ``_op`` so the two share no basename in one pytest session).
"""

import itertools

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe import (
    rubin_moe_a2a as mod,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe.rubin_moe_a2a import (
    RubinA2aWorkspace,
    a2a_use_cft,
    check_a2a_lever_env,
    rubin_moe_a2a_combine,
    rubin_moe_a2a_dispatch,
    workspace_payload_bytes,
)

__extra_import_path__ = ["."]
from _rubin_a2a_kernel_split import kernel_target_rank  # noqa: E402

_SM107 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)
pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.skipif(torch.cuda.is_available() and not _SM107, reason="certified on SM107 only"),
]

HIDDEN, TOP_K, EXPERTS = 3584, 16, 896


class _Ops:
    """A stand-in ``torch.ops.trtllm`` that records every call and returns fixed results."""

    def __init__(self, results):
        self.calls = []
        self.results = results

    def __getattr__(self, name):
        def call(*args, **kwargs):
            self.calls.append((name, args, kwargs))
            result = self.results.get(name)
            return result(*args, **kwargs) if callable(result) else result

        return call


def _same(a, b):
    """Two recorded arguments are the same: the same tensor, or equal values."""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and (a.data_ptr(), a.shape, a.dtype) == (b.data_ptr(), b.shape, b.dtype)
        )
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b and type(a) is type(b)


@pytest.mark.parametrize(
    "ep, top_k, tokens, hidden, cft, low",
    [
        (16, 16, 16384, 3584, False, True),
        (16, 16, 16384, 3584, True, True),
        (16, 16, 512, 3584, True, False),
        (12, 16, 16384, 3584, False, True),
        (12, 16, 96, 3584, True, True),
        (4, 8, 33, 7168, False, False),
    ],
)
def test_layout_arguments_are_the_engine_strategys(
    monkeypatch, ep, top_k, tokens, hidden, cft, low
):
    """The builder asks ``moe_a2a_get_workspace_layout`` for exactly the regions the engine
    strategy asks for at the same shapes (no EPLB, no extra payload), so the layout op returns the
    same metainfo (compared on the GPU image too)."""
    from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import NVLinkOneSided

    ops = _Ops({"moe_a2a_get_workspace_layout": torch.arange(4)})
    monkeypatch.setattr(torch.ops, "trtllm", ops, raising=False)
    NVLinkOneSided._make_workspace_layout(
        ep, top_k, tokens, hidden, torch.bfloat16, None, 0, cft, low
    )
    mod.workspace_layout(ep, top_k, tokens, hidden, torch.bfloat16, cft, low)
    (n1, a1, k1), (n2, a2, k2) = ops.calls
    assert n1 == n2 == "moe_a2a_get_workspace_layout" and a1 == a2 and k1 == k2 == {}
    assert a2[3:6] == workspace_payload_bytes(ep, top_k, tokens, hidden, torch.bfloat16, cft, low)


def test_layout_tensor_is_the_engine_strategys_with_the_real_op():
    """On a tree with the built extension, the metainfo tensors themselves (a CPU-only op)."""
    from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import NVLinkOneSided

    try:
        want = NVLinkOneSided._make_workspace_layout(
            16, 16, 1024, HIDDEN, torch.bfloat16, None, 0, False, True
        )
    except Exception as e:  # the CPU test env has no compiled ops
        pytest.skip(f"no moe_a2a_get_workspace_layout here: {type(e).__name__}")
    got = mod.workspace_layout(16, 16, 1024, HIDDEN, torch.bfloat16, False, True)
    assert torch.equal(got, want)


class _Memory:
    """MNNVL memory stand-in: a [ep, bytes] workspace and a handle."""

    made = []

    def __init__(self, mapping, size):
        self.size = size
        self.local_mem_handle = 77
        _Memory.made.append((type(self).__name__, size))

    @staticmethod
    def initialize():
        _Memory.made.append(("initialize", None))

    def as_torch_strided_tensor(self, dtype):
        return torch.zeros(4, self.size, dtype=dtype)


class _CftMemory(_Memory):
    pass


def _mapping(ep=4, rank=1):
    from tensorrt_llm.mapping import Mapping

    return Mapping(
        world_size=ep,
        rank=rank,
        tp_size=ep,
        enable_attention_dp=True,
        moe_ep_size=ep,
        moe_tp_size=1,
    )


@pytest.mark.parametrize("low", [False, True], ids=["bf16-combine", "fp8-combine"])
@pytest.mark.parametrize("cft", [False, True])
def test_builder_sets_up_the_workspace_as_the_engine(monkeypatch, cft, low):
    """Layout, MNNVL allocation of the layout's size (counted-write memory with CFT), initialize
    on every rank, the CFT endpoints only with CFT, the warmup-aware timeout registered; the
    workspace keeps the combine wire its receive region was sized for."""
    from tensorrt_llm._torch.distributed import mnnvl_memory
    from tensorrt_llm._torch.moe.fused_moe import moe_comm_timeout_guard
    from tensorrt_llm._torch.moe.fused_moe.communication import nvlink_one_sided as engine

    _Memory.made = []
    # The kernels' limits, as the compiled module exports them (moeAlltoAllKernels.h).
    monkeypatch.setattr(engine.NVLinkOneSided, "MAX_RANKS", 256)
    monkeypatch.setattr(engine.NVLinkOneSided, "MAX_TOP_K", 22)
    size = 4096
    metainfo = torch.zeros(32, dtype=torch.int64)
    monkeypatch.setattr(mod, "_workspace_size_index", lambda: 5)
    metainfo[5] = size
    ops = _Ops({"moe_a2a_get_workspace_layout": metainfo})
    monkeypatch.setattr(torch.ops, "trtllm", ops, raising=False)
    monkeypatch.setattr(mnnvl_memory, "MnnvlMemory", _Memory)
    monkeypatch.setattr(mnnvl_memory, "CftMnnvlMemory", _CftMemory)
    monkeypatch.setattr(engine, "select_cft_counted_writes", lambda force: cft)
    proxies = []
    monkeypatch.setattr(moe_comm_timeout_guard, "register_moe_comm_timeout_proxy", proxies.append)
    ws = mod.rubin_moe_a2a_workspace(
        _mapping(),
        num_experts=EXPERTS,
        top_k=TOP_K,
        max_num_tokens_per_rank=1024,
        hidden_size=HIDDEN,
        dtype=torch.bfloat16,
        use_low_precision_combine=low,
        force_cft=None,
    )
    assert proxies == [engine._TIMEOUT_PROXY]
    want = workspace_payload_bytes(4, TOP_K, 1024, HIDDEN, torch.bfloat16, cft, low)
    assert ops.calls[0][1][3:6] == want and ws.use_low_precision_combine is low
    assert _Memory.made == [("initialize", None), ("_CftMemory" if cft else "_Memory", size)]
    names = [c[0] for c in ops.calls]
    assert names == ["moe_a2a_get_workspace_layout", "moe_a2a_initialize"] + (
        ["moe_a2a_cft_initialize"] if cft else []
    )
    _, init_args, _ = ops.calls[1]
    assert init_args[0] is ws.workspace and init_args[1] is metainfo and init_args[2:] == (1, 4)
    if cft:
        assert ops.calls[2][1][1:] == (77, size, 1, 4)
    assert (ws.ep_rank, ws.ep_size, ws.top_k, ws.num_experts, ws.can_use_cft) == (
        1,
        4,
        TOP_K,
        EXPERTS,
        cft,
    )


def test_builder_refuses_what_the_strategy_refuses(monkeypatch):
    from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import NVLinkOneSided
    from tensorrt_llm.mapping import Mapping

    monkeypatch.setattr(NVLinkOneSided, "MAX_RANKS", 256)
    monkeypatch.setattr(NVLinkOneSided, "MAX_TOP_K", 22)

    kw = dict(
        num_experts=EXPERTS,
        top_k=TOP_K,
        max_num_tokens_per_rank=64,
        hidden_size=HIDDEN,
        dtype=torch.bfloat16,
        use_low_precision_combine=True,
        force_cft=False,
    )
    tp2 = Mapping(world_size=4, rank=0, tp_size=4, moe_ep_size=2, moe_tp_size=2)
    with pytest.raises(ValueError, match="pure EP"):
        mod.rubin_moe_a2a_workspace(tp2, **kw)
    with pytest.raises(ValueError, match="kernels' limits"):
        mod.rubin_moe_a2a_workspace(_mapping(), **{**kw, "top_k": 64})


# The entries' calls against the engine strategy's


def _engine_strategy(ws, can_use_cft, force_cft=None):
    """An engine ``NVLinkOneSided`` over the same workspace, without its allocation (the fields
    its dispatch and combine read)."""
    from tensorrt_llm._torch.alltoall_watchdog import ActiveRankMaskSnapshot
    from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import NVLinkOneSided

    class _Coordinator:
        def capture_active_rank_mask(self, mask):
            return ActiveRankMaskSnapshot(active_rank_mask=None, committed_generation=None)

        def active_rank_mask_for_combine(self, snapshot, mask):
            return None

        def watch_collective(self, *args):
            pass

    class _Lifecycle:
        coordinator = _Coordinator()
        metainfo = ws.metainfo

        def watchdog_for(self, owner):
            return None

    s = object.__new__(NVLinkOneSided)
    s.__dict__.update(
        ep_rank=ws.ep_rank,
        ep_size=ws.ep_size,
        top_k=ws.top_k,
        num_experts=ws.num_experts,
        workspace=ws.workspace,
        mnnvl_mem=type("M", (), {"mapped": True})(),
        payload_in_workspace=True,
        use_low_precision_combine=ws.use_low_precision_combine,
        enable_eplb=False,
        can_use_cft_counted_writes=can_use_cft,
        _force_cft=force_cft,
        cft_max_batch_for_dispatch=None if force_cft is not None else mod.CFT_MAX_BATCH,
        cft_max_batch_for_combine=None if force_cft is not None else mod.CFT_MAX_BATCH,
        _rank_mask_enabled=False,
        _workspace_lifecycle=_Lifecycle(),
        _dispatch_state={"phase": "idle"},
        invalid_token_expert_id=-1,
    )
    return s


@pytest.mark.parametrize("low", [False, True], ids=["bf16-combine", "fp8-combine"])
@pytest.mark.parametrize(
    "can_use_cft, rows", [(False, (64, 1, 0, 17)), (True, (64, 1, 0, 17)), (True, (200, 3, 7, 9))]
)
def test_dispatch_and_combine_are_the_engine_strategys_calls(monkeypatch, can_use_cft, rows, low):
    """On the same inputs and workspace, the entries make the engine strategy's op calls (same
    tensors, same scalars, same order) and return its tensors: dispatch with or without fused
    sanitize, the combine payload view, the combine on the workspace's wire; counted writes by the
    engine's token rule."""
    ep, rank, runtime_max = 4, 2, max(rows)
    ws = RubinA2aWorkspace(
        workspace=torch.zeros(ep, 64, dtype=torch.uint8),
        metainfo=torch.zeros(32, dtype=torch.int64),
        ep_rank=rank,
        ep_size=ep,
        top_k=TOP_K,
        num_experts=EXPERTS,
        max_num_tokens_per_rank=256,
        hidden_size=HIDDEN,
        can_use_cft=can_use_cft,
        use_low_precision_combine=low,
        memory=None,
    )
    local = rows[rank]
    g = torch.Generator().manual_seed(sum(rows))
    x4 = torch.randint(0, 255, (local, HIDDEN // 2), generator=g, dtype=torch.uint8)
    sf = torch.randint(0, 255, (local, HIDDEN // 16), generator=g, dtype=torch.uint8)
    ids = torch.randint(0, EXPERTS, (local, TOP_K), generator=g, dtype=torch.int32)
    wts = torch.rand(local, TOP_K, generator=g)
    recv = [
        torch.zeros(ep, runtime_max, HIDDEN // 2, dtype=torch.uint8),
        torch.zeros(ep, runtime_max, HIDDEN // 16, dtype=torch.uint8),
        torch.zeros(ep, runtime_max, TOP_K, dtype=torch.int32),
        torch.zeros(ep, runtime_max, TOP_K),
    ]
    payload = torch.zeros(ep * runtime_max, HIDDEN, dtype=torch.bfloat16)
    out = torch.zeros(local, HIDDEN, dtype=torch.bfloat16)
    ops = _Ops(
        {
            "moe_a2a_dispatch": (recv, 4096, torch.empty(0)),
            "moe_a2a_get_combine_payload_tensor": payload,
            "moe_a2a_combine": out,
        }
    )
    monkeypatch.setattr(torch.ops, "trtllm", ops, raising=False)

    engine = _engine_strategy(ws, can_use_cft)
    e_out = engine.dispatch(x4, sf, ids, wts, all_rank_num_tokens=list(rows))
    e_payload = engine.get_combine_payload_tensor_in_workspace(runtime_max, HIDDEN, torch.bfloat16)
    e_final = engine.combine(e_payload, all_rank_max_num_tokens=runtime_max)
    engine_calls, ops.calls = ops.calls, []

    use_cft = a2a_use_cft(can_use_cft, None, mod.CFT_MAX_BATCH, runtime_max)
    *o_out, offset, o_payload = rubin_moe_a2a_dispatch(
        x4, sf, ids, wts, ws, runtime_max, use_cft, -1, torch.bfloat16, HIDDEN
    )
    o_final = rubin_moe_a2a_combine(
        o_payload, local, ws, runtime_max, offset, True, ws.use_low_precision_combine, use_cft
    )
    ours = ops.calls

    assert [c[0] for c in ours] == [c[0] for c in engine_calls]
    assert ("moe_a2a_sanitize_expert_ids" in [c[0] for c in ours]) == (not use_cft)
    for (name, a1, k1), (_, a2, k2) in zip(engine_calls, ours):
        assert k1 == k2 == {} and len(a1) == len(a2), name
        for i, (v1, v2) in enumerate(zip(a1, a2)):
            if name == "moe_a2a_combine" and i == 0:
                assert v1.shape == (ep, runtime_max, HIDDEN) and v2.shape == v1.shape
                assert v1.data_ptr() == v2.data_ptr() == payload.data_ptr()
            else:
                assert _same(v1, v2), (name, i, v1, v2)
    assert offset == 4096
    for e, o in zip(e_out, o_out):
        assert _same(e, o)
    assert o_payload is payload and e_final is o_final is out


def test_cft_rule_is_the_engines():
    from tensorrt_llm._torch.moe.fused_moe.communication.nvlink_one_sided import should_use_cft

    for can, force, limit, tokens in itertools.product(
        (False, True), (None, False, True), (None, 128), (1, 128, 129, 16384)
    ):
        assert a2a_use_cft(can, force, limit, tokens) == should_use_cft(can, force, limit, tokens)


def test_a2a_lever_env():
    """The strategy's variables: unset passes; the engine's default limits pass; a forced mode
    must match ``force_cft``; a workspace size or other limits raise."""
    check_a2a_lever_env({}, force_cft=None)
    check_a2a_lever_env(
        {"TRTLLM_NVLINK_ONE_SIDED_A2A_CFT_MAX_BATCH_FOR_DISPATCH": "128"}, force_cft=None
    )
    check_a2a_lever_env({"TRTLLM_NVLINK_ONE_SIDED_A2A_FORCE_CFT": "0"}, force_cft=False)
    for env, force in (
        ({"TRTLLM_NVLINK_ONE_SIDED_A2A_FORCE_CFT": "1"}, None),
        ({"TRTLLM_NVLINK_ONE_SIDED_A2A_FORCE_CFT": "1"}, False),
        ({"TRTLLM_NVLINK_ONE_SIDED_A2A_WORKSPACE_MB": "4096"}, None),
        ({"TRTLLM_NVLINK_ONE_SIDED_A2A_CFT_MAX_BATCH_FOR_COMBINE": "64"}, None),
    ):
        with pytest.raises(ValueError, match=next(iter(env))):
            check_a2a_lever_env(env, force_cft=force)


def test_guards():
    ws = RubinA2aWorkspace(
        torch.zeros(4, 8, dtype=torch.uint8),
        torch.zeros(1),
        0,
        4,
        TOP_K,
        EXPERTS,
        128,
        HIDDEN,
        False,
        True,
        None,
    )
    x4, sf = (
        torch.zeros(3, HIDDEN // 2, dtype=torch.uint8),
        torch.zeros(3, HIDDEN // 16, dtype=torch.uint8),
    )
    ids, wts = torch.zeros(3, TOP_K, dtype=torch.int32), torch.zeros(3, TOP_K)
    rubin_moe_a2a_dispatch.is_valid(x4, sf, ids, wts, ws, 64, False, -1, torch.bfloat16, HIDDEN)
    for bad, match in (
        (dict(token_selected_experts=ids.long()), "int32"),
        (dict(invalid_token_expert_id=0), "-1"),
        (dict(runtime_max_tokens_per_rank=129), "exceed"),
        (dict(use_cft_counted_writes=True), "fence workspace"),
        (dict(combine_hidden=7168), "activation width"),
    ):
        args = dict(
            hidden_states=x4,
            hidden_states_sf=sf,
            token_selected_experts=ids,
            token_final_scales=wts,
            ws=ws,
            runtime_max_tokens_per_rank=64,
            use_cft_counted_writes=False,
            invalid_token_expert_id=-1,
            combine_dtype=torch.bfloat16,
            combine_hidden=HIDDEN,
        )
        with pytest.raises(AssertionError, match=match):
            rubin_moe_a2a_dispatch.is_valid(**{**args, **bad})
    payload = torch.zeros(4 * 64, HIDDEN, dtype=torch.bfloat16)
    rubin_moe_a2a_combine.is_valid(payload, 3, ws, 64, 0, True, True, False)
    with pytest.raises(AssertionError, match="more local tokens"):
        rubin_moe_a2a_combine.is_valid(payload, 65, ws, 64, 0, True, True, False)
    with pytest.raises(AssertionError, match="ep_size x runtime_max"):
        rubin_moe_a2a_combine.is_valid(payload, 3, ws, 32, 0, True, True, False)


_SPLITS = [(896, 16), (896, 12), (896, 8), (896, 7), (299, 4), (129, 2), (8, 3), (10, 4), (5, 5)]


@pytest.mark.parametrize("num_experts, ep", _SPLITS)
def test_expert_window_is_the_kernels_split(num_experts, ep):
    """``a2a_expert_window`` is the set of experts the dispatch kernel sends to each rank, and
    ``a2a_expert_owner`` its inverse on every expert id."""
    owner = [kernel_target_rank(e, num_experts, ep) for e in range(num_experts)]
    for rank in range(ep):
        first, count = mod.a2a_expert_window(num_experts, ep, rank)
        assert [e for e in range(num_experts) if owner[e] == rank] == list(
            range(first, first + count)
        ), rank
    got = mod.a2a_expert_owner(torch.arange(num_experts, dtype=torch.int32), num_experts, ep)
    assert got.dtype == torch.int64 and got.tolist() == owner


@pytest.mark.parametrize("num_experts, ep", _SPLITS)
def test_expert_window_is_the_engines_partition(num_experts, ep):
    """``a2a_expert_window`` is the engine MoE's partition (``_compute_ep_partition``, which the
    engine's modules load their expert slots by), rank by rank."""
    from tensorrt_llm._torch.moe.fused_moe.interface import _compute_ep_partition

    for rank in range(ep):
        size, start, end = _compute_ep_partition(num_experts, ep, rank)
        assert mod.a2a_expert_window(num_experts, ep, rank) == (start, size) and end == start + size


def test_expert_windows_of_dep16_and_dep12():
    """896 experts: 56 per rank over 16 ranks; over 12, ranks 0-7 hold 75 and ranks 8-11 74."""
    assert [mod.a2a_expert_window(896, 16, r) for r in range(16)] == [
        (56 * r, 56) for r in range(16)
    ]
    assert [mod.a2a_expert_window(896, 12, r) for r in range(12)] == [
        *((75 * r, 75) for r in range(8)),
        *((600 + 74 * r, 74) for r in range(4)),
    ]
    with pytest.raises(AssertionError):
        mod.a2a_expert_window(896, 12, 12)
    with pytest.raises(AssertionError):
        mod.a2a_expert_window(8, 16, 0)


@pytest.mark.parametrize("ep, num_experts", [(4, 8), (3, 8)], ids=["even", "uneven"])
def test_references_compose_to_the_moe_sum(ep, num_experts):
    """Dispatch's reference, a per-rank expert sum over each received row's local experts, then
    combine's reference: every token gets the sum over all its experts, as one rank holding them
    all would compute; also when the experts split unevenly (3, 3 and 2 experts over 3 ranks)."""
    top_k, hidden = 3, 6
    rows = (5, 1, 0, 3)[:ep]
    g = torch.Generator().manual_seed(3)
    xs = [torch.randn(n, hidden, generator=g) for n in rows]
    sfs = [torch.zeros(n, 1) for n in rows]
    ids = [
        torch.stack([torch.randperm(num_experts, generator=g)[:top_k] for _ in range(n)]).int()
        if n
        else torch.zeros(0, top_k, dtype=torch.int32)
        for n in rows
    ]
    wts = [torch.rand(n, top_k, generator=g) for n in rows]
    expert = torch.randn(num_experts, hidden, hidden, generator=g)
    ws = type("W", (), {"ep_size": ep, "num_experts": num_experts})()
    got = rubin_moe_a2a_dispatch.reference(
        xs, sfs, ids, wts, ws, max(rows), False, -1, torch.float32, hidden
    )
    payload = []
    for r in range(ep):
        sent = {
            (s, t)
            for s in range(ep)
            for t in range(rows[s])
            if any(kernel_target_rank(int(e), num_experts, ep) == r for e in ids[s][t])
        }
        assert set(got[r]) == sent, r
        part = {}
        for key, (x, _, i, w) in got[r].items():
            mine = [kernel_target_rank(int(e), num_experts, ep) == r for e in i]
            part[key] = sum(w[k] * (expert[i[k]] @ x) for k in range(top_k) if mine[k])
        payload.append(part)
    out = rubin_moe_a2a_combine.reference(payload, list(rows), ws, max(rows), 0, True, False, False)
    for r in range(ep):
        for t in range(rows[r]):
            want = sum(wts[r][t, k] * (expert[ids[r][t, k]] @ xs[r][t]) for k in range(top_k))
            torch.testing.assert_close(out[r][t], want)


def test_the_cells_run_dep12s_uneven_windows():
    """Some dispatch and combine cells split their experts as DEP12 does, wide ranks first and one
    expert apart (75 / 75 / 75 / 74 over the 4 ranks the cells run on), with fewer ranks than
    top-k, as at EP12; the rest split evenly."""
    uneven = []
    for cell in (*rubin_moe_a2a_dispatch.CELLS, *rubin_moe_a2a_combine.CELLS):
        spec = cell.spec
        ranks = spec["ranks"]
        widths = [mod.a2a_expert_window(spec["num_experts"], ranks, r)[1] for r in range(ranks)]
        if len(set(widths)) > 1:
            assert widths == [75, 75, 75, 74] and ranks < spec["top_k"], cell.why
            uneven.append(spec["cft"])
    assert sorted(uneven, key=repr) == [False, False, None]
