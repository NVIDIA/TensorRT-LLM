# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``replicated_experts``' routing and expert hooks on stand-in modules (GPU).

The hooks are driven as the Muse Spark Open MoE layer drives them -- routing
first, then the experts' ``do_finalize=False`` call, then the memo clear --
both under ``torch.inference_mode()`` (how serving runs the forward) and in
grad mode. The stacked expert call itself is replaced by a recorder; its
numerics are gated in ``test_modeling_v2_ring_buddy_moe.py``.

Also here: the routing hook over the built-in ``MuseSparkOpenRoutingMethod``
(its Triton routing kernel and memo) at 1..8 rows, ``install`` called twice
(a weight reload refreshes the stacks in place), the agreement of its
refusals over the TP group (a stand-in for ``Distributed``).
"""

import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.replicated_experts import (  # noqa: E501
    ring_buddy,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.moe.ring_buddy_assign import (  # noqa: E501
    ring_buddy_assign,
)

assert torch.cuda.is_available(), "the placement kernel requires a CUDA device"


class _Routing:
    routing_method_type = 1

    def __init__(self):
        self.cleared = 0

    def apply(self, router_logits, input_ids=None):
        top = torch.topk(router_logits, 8, dim=-1)
        return top.indices.to(torch.int32).contiguous(), torch.softmax(top.values, -1)

    def clear_routing_memo(self):
        self.cleared += 1


class _Experts:
    def __init__(self):
        self.routing_method = _Routing()

    def forward(self, x, router_logits, *args, do_finalize=True, **kwargs):
        return "built-in"


def _layer(monkeypatch, rank, balance=True):
    calls = []

    def stacked_call_factory(backend, routing):
        def call(stacks, x, local_ids, weights_bf16):
            calls.append((local_ids.clone(), weights_bf16.clone()))
            return "stacked"

        return call

    monkeypatch.setattr(ring_buddy, "_nvfp4_call", stacked_call_factory)
    moe = types.SimpleNamespace(experts=_Experts())
    state = ring_buddy._State("nvfp4", {}, rank, balance)
    ring_buddy._patch_routing(moe, state)
    ring_buddy._patch_experts(moe, state)
    return moe, state, calls


@pytest.mark.parametrize("inference", [True, False], ids=["inference_mode", "grad_mode"])
@pytest.mark.parametrize("balance", [True, False], ids=["balanced", "owner_only"])
def test_layer_call(monkeypatch, inference, balance):
    rank = 5
    moe, state, calls = _layer(monkeypatch, rank, balance)
    routing = moe.experts.routing_method
    g = torch.Generator(device="cuda").manual_seed(7)
    mode = torch.inference_mode() if inference else torch.no_grad()
    with mode:
        logits = torch.randn(8, 256, device="cuda", generator=g)
        x = torch.randn(8, 64, device="cuda", generator=g)
        # The layer's preparation: routing on the main stream fills the placement.
        selected, weights = routing.apply(logits)
        placement = state.current(logits)
        assert placement is not None
        out = moe.experts.forward(x, logits, do_finalize=False, output_dtype=torch.bfloat16)
        assert out == "stacked"
        # The experts reuse the placement made from the same logits, not a new one.
        assert state.current(logits) is placement
        want_ids, want_w, _ = ring_buddy_assign(selected, weights, rank, balance)
        assert torch.equal(calls[-1][0], want_ids)
        assert torch.equal(calls[-1][1], want_w)
        routing.clear_routing_memo()
        assert state.placement is None and routing.cleared == 1

        # Calls the stacked path does not serve keep the built-in forward.
        assert moe.experts.forward(x, logits, do_finalize=True) == "built-in"
        big = torch.randn(17, 256, device="cuda", generator=g)
        assert moe.experts.forward(torch.randn(17, 64, device="cuda"), big, do_finalize=False) == (
            "built-in"
        )
        state.active = False
        assert moe.experts.forward(x, logits, do_finalize=False) == "built-in"
        state.active = True
    if not inference:
        # An in-place write to the same logits object invalidates the placement.
        with torch.no_grad():
            routing.apply(logits)
            first = state.current(logits)
            logits.mul_(-1.0)
            assert state.current(logits) is None
            routing.apply(logits)
            assert state.current(logits) is not first


def _partition(selected: torch.Tensor, balance: bool) -> None:
    """Every (token, slot) runs on exactly one rank, as the expert it selected."""
    flat = selected.reshape(-1).tolist()
    weights = torch.zeros_like(selected, dtype=torch.float32)
    owner = {}
    for rank in range(ring_buddy.RANKS):
        ids, _, _ = ring_buddy_assign(selected, weights, rank, balance)
        for slot, local in enumerate(ids.reshape(-1).tolist()):
            if local >= 2 * ring_buddy.BLOCK:
                continue
            block = rank if local < ring_buddy.BLOCK else (rank - 1) % ring_buddy.RANKS
            assert ring_buddy.BLOCK * block + local % ring_buddy.BLOCK == flat[slot]
            assert slot not in owner, f"slot {slot} on ranks {owner[slot]} and {rank}"
            owner[slot] = rank
    assert sorted(owner) == list(range(len(flat)))


@pytest.mark.parametrize("rows", list(range(1, 9)))
def test_real_routing_method(rows):
    """The hook over the built-in ``MuseSparkOpenRoutingMethod`` (Triton routing, memo on)."""
    from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenRoutingMethod

    rank = 6
    g = torch.Generator(device="cuda").manual_seed(100 + rows)
    weight = torch.randn(256, 512, device="cuda", generator=g) * 0.05
    bias = (torch.randn(256, device="cuda", generator=g) * 0.01).contiguous()
    routing = MuseSparkOpenRoutingMethod(
        top_k=8, num_experts=256, callable_e_score_correction_bias=lambda: bias, renorm_eps=1e-15
    )
    moe = types.SimpleNamespace(experts=types.SimpleNamespace(routing_method=routing))
    state = ring_buddy._State("nvfp4", {}, rank, True)
    ring_buddy._patch_routing(moe, state)
    with torch.inference_mode():
        for _ in range(6):
            x = torch.randn(rows, 512, device="cuda", generator=g)
            logits = (x @ weight.t()).contiguous()
            selected, weights = routing.apply(logits)
            assert selected.dtype == torch.int32, "the Triton routing path did not run"
            placement = state.current(logits)
            assert placement is not None, "no placement under inference_mode"
            # The model's second apply returns the memo and keeps the placement.
            again, _ = routing.apply(logits)
            assert again is selected and state.current(logits) is placement
            want_ids, want_w, want_stats = ring_buddy_assign(selected, weights, rank, True)
            assert torch.equal(placement[2], want_ids)
            assert torch.equal(placement[3], want_w)
            assert torch.equal(placement[4], want_stats)
            _partition(selected, True)
            routing.clear_routing_memo()
            assert state.placement is None and routing._routing_memo is None


class _TpGroup:
    """Stands in for ``Distributed`` over the TP group: this rank, then ``peers``."""

    def __init__(self, peers):
        self.peers = list(peers)
        self.gathered = []

    def get(self, mapping):
        return self

    def tp_allgather(self, obj, small_payload=False):
        assert small_payload
        self.gathered.append(obj)
        return [obj] + self.peers


class _Param:
    """Stand-in for a leaf's per-expert parameter: only ``.data`` is read and re-bound."""

    def __init__(self, data):
        self.data = data


def _installable_model(monkeypatch, layers=2):
    """Stand-in text model whose MoE layers ``install`` treats as eligible NVFP4 leaves."""
    model = types.SimpleNamespace(layers=[])
    for index in range(layers):
        backend = types.SimpleNamespace(
            quant_method=types.SimpleNamespace(transform_weights=lambda leaf: None)
        )
        for k, name in enumerate(ring_buddy.STACKED["nvfp4"]):
            data = torch.full((ring_buddy.BLOCK, 3), float(100 * index + k), device="cuda")
            setattr(backend, name, _Param(data))
        experts = _Experts()
        experts.backend = backend
        moe = types.SimpleNamespace(experts=experts)
        model.layers.append(types.SimpleNamespace(feed_forward=moe))
    monkeypatch.setattr(ring_buddy, "eligible", lambda moe, mapping: None)
    monkeypatch.setattr(ring_buddy, "expert_format", lambda moe: "nvfp4")
    # The predecessor's block as the all-gather would deliver it; here own + 1000.
    monkeypatch.setattr(ring_buddy, "_predecessor_block", lambda local, mapping: local + 1000)
    monkeypatch.setattr(ring_buddy, "_nvfp4_call", lambda backend, routing: None)
    # The refusals' agreement: every peer agrees (no refusal) unless a test says otherwise.
    monkeypatch.setattr(
        ring_buddy._tp_agreement, "Distributed", _TpGroup([None] * (ring_buddy.RANKS - 1))
    )
    mapping = types.SimpleNamespace(tp_rank=2, tp_size=ring_buddy.RANKS)
    return model, mapping


def test_a_peers_refusal_before_the_first_all_gather_is_raised_here(monkeypatch):
    """A refusal only a peer sees (here: its refresh would widen a layer) stops this rank
    before its first all-gather, and nothing is replicated or patched."""
    model, mapping = _installable_model(monkeypatch)
    peer = "three_window_placement: layer 0 holds 2 ring windows and this install stacks 3 on it"
    group = _TpGroup([None] * (ring_buddy.RANKS - 2) + [peer])
    monkeypatch.setattr(ring_buddy._tp_agreement, "Distributed", group)
    gathered = []
    monkeypatch.setattr(
        ring_buddy, "_predecessor_block", lambda local, mapping: gathered.append(1) or local
    )
    with pytest.raises(ValueError, match=r"stacks 3 on it \(on TP ranks \[7\] of 8\)"):
        ring_buddy.install(model, mapping)
    assert group.gathered == [None] and gathered == []
    assert all(not hasattr(layer.feed_forward, "_replicated_experts") for layer in model.layers)


def test_a_failed_expert_compile_is_agreed_after_every_layers_all_gather(monkeypatch):
    """A compile that fails on this rank between the per-layer all-gathers is held: every
    later layer still takes its all-gather, and every rank raises after the last one."""
    model, mapping = _installable_model(monkeypatch, layers=3)
    group = _TpGroup([None] * (ring_buddy.RANKS - 1))
    monkeypatch.setattr(ring_buddy._tp_agreement, "Distributed", group)
    gathered = []
    monkeypatch.setattr(
        ring_buddy,
        "_predecessor_block",
        lambda local, mapping: gathered.append(1) or local + 1000,
    )
    calls = []

    def nvfp4_call(backend, routing):
        calls.append(backend)
        if len(calls) == 1:
            raise RuntimeError("compile failed")

    monkeypatch.setattr(ring_buddy, "_nvfp4_call", nvfp4_call)
    with pytest.raises(
        ValueError, match=r"replicated_experts could not install: layer 0: RuntimeError"
    ):
        ring_buddy.install(model, mapping)
    per_layer = len(gathered) // 3
    assert per_layer >= 1 and len(gathered) == 3 * per_layer, "a layer skipped its all-gather"
    assert len(calls) == 1, "patching went on after the failure"
    assert len(group.gathered) == 2 and group.gathered[0] is None
    assert "compile failed" in group.gathered[1]


def test_install_twice_refreshes_in_place(monkeypatch):
    """A second ``install`` (a weight reload) refreshes the stacks; no second stack or hook."""
    model, mapping = _installable_model(monkeypatch)
    assert ring_buddy.install(model, mapping) == [0, 1]
    moe = model.layers[1].feed_forward
    state = moe._replicated_experts
    backend = moe.experts.backend
    hooks = (moe.experts.routing_method.apply, moe.experts.forward)
    ptrs = {name: t.data_ptr() for name, t in state.stacks.items()}
    half = ring_buddy.BLOCK
    for name, stack in state.stacks.items():
        assert getattr(backend, name).data.data_ptr() == stack.data_ptr()
        assert torch.equal(stack[half:], stack[:half] + 1000)

    # The reload writes one parameter in place and replaces another's storage.
    backend.w2_weight.data.add_(7.0)
    backend.fc2_alpha.data = torch.full_like(backend.fc2_alpha.data, -5.0)
    assert ring_buddy.install(model, mapping) == [0, 1]
    assert moe._replicated_experts is state
    assert (moe.experts.routing_method.apply, moe.experts.forward) == hooks, "hooks wrapped again"
    for name, stack in state.stacks.items():
        assert stack.data_ptr() == ptrs[name], f"{name}: a second stack was allocated"
        assert getattr(backend, name).data.data_ptr() == stack.data_ptr()
        assert torch.equal(stack[half:], stack[:half] + 1000)
    assert bool((state.stacks["w2_weight"][:half] == 109.0).all())
    assert bool((state.stacks["fc2_alpha"][:half] == -5.0).all())

    backend.w2_weight.data = torch.zeros(half, 4, device="cuda")
    with pytest.raises(RuntimeError, match="cannot refresh in place"):
        ring_buddy.install(model, mapping)


def test_three_windows_install_and_refresh(monkeypatch):
    """Three windows: the stack is [own | b-1 | b-2], refreshed in place; a refresh adds no window."""
    narrow, _ = _installable_model(monkeypatch)
    model, mapping = _installable_model(monkeypatch)
    # The predecessor's block stays the two-window stand-in (own + 1000); two back is own + 2000.
    monkeypatch.setattr(ring_buddy, "_ring_block", lambda local, mapping, back: local + 1000 * back)
    monkeypatch.setattr(ring_buddy, "persistent_eligible", lambda backend: True)
    monkeypatch.setattr(
        ring_buddy, "_nvfp4_call", lambda backend, routing, persistent=False, stacks=None: None
    )
    assert ring_buddy.install(model, mapping, persistent=True, windows=3) == [0, 1]
    moe = model.layers[1].feed_forward
    state = moe._replicated_experts
    backend = moe.experts.backend
    block = ring_buddy.BLOCK
    assert state.windows == 3 and sorted(state.by_window) == [2 * block, 3 * block]

    def check():
        for name, stack in state.stacks.items():
            assert stack.shape[0] == 3 * block
            assert getattr(backend, name).data.data_ptr() == stack.data_ptr()
            assert torch.equal(stack[block : 2 * block], stack[:block] + 1000)
            assert torch.equal(stack[2 * block :], stack[:block] + 2000)
            prefix = state.by_window[2 * block][name]
            assert prefix.data_ptr() == stack.data_ptr() and prefix.shape[0] == 2 * block
            assert state.by_window[3 * block][name] is stack

    check()
    ptrs = {name: t.data_ptr() for name, t in state.stacks.items()}
    backend.w2_weight.data.add_(7.0)
    assert ring_buddy.install(model, mapping, persistent=True, windows=3) == [0, 1]
    assert moe._replicated_experts is state
    assert {name: t.data_ptr() for name, t in state.stacks.items()} == ptrs
    check()
    assert bool((state.stacks["w2_weight"][:block] == 109.0).all())

    # Layers installed with two windows cannot be widened by a later install.
    assert ring_buddy.install(narrow, mapping, persistent=True) == [0, 1]
    assert narrow.layers[0].feed_forward._replicated_experts.windows == 2
    with pytest.raises(ValueError, match="a refresh cannot add one"):
        ring_buddy.install(narrow, mapping, persistent=True, windows=3)
    # Nor narrowed: a three-window layer refreshed by a two-window install.
    with pytest.raises(ValueError, match="a refresh cannot add one or drop one"):
        ring_buddy.install(model, mapping, persistent=True)


def test_three_windows_refresh_keeps_ineligible_layers_at_two(monkeypatch):
    """Three windows: an NVFP4 leaf the persistent kernel does not compute keeps two windows,
    on the first install and on every refresh with ``windows=3`` (no refusal)."""
    model, mapping = _installable_model(monkeypatch)
    monkeypatch.setattr(ring_buddy, "_ring_block", lambda local, mapping, back: local + 1000 * back)
    narrow_backend = model.layers[0].feed_forward.experts.backend
    monkeypatch.setattr(
        ring_buddy, "persistent_eligible", lambda backend: backend is not narrow_backend
    )
    monkeypatch.setattr(
        ring_buddy, "_nvfp4_call", lambda backend, routing, persistent=False, stacks=None: None
    )
    assert ring_buddy.install(model, mapping, persistent=True, windows=3) == [0, 1]
    narrow = model.layers[0].feed_forward._replicated_experts
    wide = model.layers[1].feed_forward._replicated_experts
    assert (narrow.windows, wide.windows) == (2, 3)
    assert sorted(narrow.by_window) == [2 * ring_buddy.BLOCK]
    ptrs = {name: t.data_ptr() for name, t in narrow.stacks.items()}
    narrow_backend.w2_weight.data.add_(7.0)
    assert ring_buddy.install(model, mapping, persistent=True, windows=3) == [0, 1]
    assert model.layers[0].feed_forward._replicated_experts is narrow and narrow.windows == 2
    assert {name: t.data_ptr() for name, t in narrow.stacks.items()} == ptrs
    for name, stack in narrow.stacks.items():
        assert stack.shape[0] == 2 * ring_buddy.BLOCK
        assert torch.equal(stack[ring_buddy.BLOCK :], stack[: ring_buddy.BLOCK] + 1000)
    assert bool((narrow.stacks["w2_weight"][: ring_buddy.BLOCK] == 9.0).all())


# --- tep4 and the owner-only window -------------------------------------------------------------


def _tep4_model(monkeypatch, layers=2):
    """Stand-in tep4 text model: 64-expert own blocks, every peer agreeing."""
    model = types.SimpleNamespace(layers=[])
    for index in range(layers):
        backend = types.SimpleNamespace(
            quant_method=types.SimpleNamespace(transform_weights=lambda leaf: None)
        )
        for k, name in enumerate(ring_buddy.STACKED["nvfp4"]):
            data = torch.full((64, 3), float(100 * index + k), device="cuda")
            setattr(backend, name, _Param(data))
        experts = _Experts()
        experts.backend = backend
        moe = types.SimpleNamespace(experts=experts)
        model.layers.append(types.SimpleNamespace(feed_forward=moe))
    monkeypatch.setattr(ring_buddy, "eligible", lambda moe, mapping: None)
    monkeypatch.setattr(ring_buddy, "expert_format", lambda moe: "nvfp4")
    monkeypatch.setattr(ring_buddy._tp_agreement, "Distributed", _TpGroup([None] * 3))
    return model, types.SimpleNamespace(tp_rank=1, tp_size=4)


def test_owner_window_install_replicates_nothing_at_tep4(monkeypatch):
    """``windows=1`` at tep4: no all-gather, the stacks are the parameters themselves, the
    window is the 64-expert own block, and the routing hook places owner-only on the
    4-rank ring."""
    model, mapping = _tep4_model(monkeypatch)
    gathered = []
    monkeypatch.setattr(
        ring_buddy, "_ring_block", lambda local, mapping, back: gathered.append(back) or local
    )
    seen = {}

    def nvfp4_call(backend, routing, persistent=False, stacks=None):
        seen["persistent"], seen["windows"] = persistent, sorted(stacks or {})

        def call(stacks, x, local_ids, weights_bf16):
            seen["window"] = stacks["w3_w1_weight"].shape[0]
            seen["ids"] = local_ids
            return "stacked"

        return call

    monkeypatch.setattr(ring_buddy, "_nvfp4_call", nvfp4_call)
    assert ring_buddy.install(model, mapping, balance=False, persistent=True, windows=1) == [0, 1]
    assert gathered == [], "the owner-only window gathered a peer's block"
    moe = model.layers[1].feed_forward
    state = moe._replicated_experts
    backend = moe.experts.backend
    assert (state.windows, state.ranks, state.block, state.default_window) == (1, 4, 64, 64)
    assert state.static_from == 64 and not state.balance
    for name, stack in state.stacks.items():
        assert stack is getattr(backend, name).data, f"{name}: the window must be the parameter"
    assert seen["persistent"] and seen["windows"] == [64]
    g = torch.Generator(device="cuda").manual_seed(41)
    with torch.inference_mode():
        logits = torch.randn(8, 256, device="cuda", generator=g)
        x = torch.randn(8, 64, device="cuda", generator=g)
        selected, weights = moe.experts.routing_method.apply(logits)
        assert moe.experts.forward(x, logits, do_finalize=False) == "stacked"
    assert seen["window"] == 64
    want, _, _ = ring_buddy_assign(selected, weights, 1, False, ranks=4)
    assert torch.equal(seen["ids"], want)
    for g_id, local in zip(selected.reshape(-1).tolist(), want.reshape(-1).tolist()):
        assert local == g_id - 64 if g_id // 64 == 1 else local >= 128


def test_owner_window_refusals(monkeypatch):
    """The owner-only window has no copy to balance onto; three windows do not fit at tep4."""
    model, mapping = _tep4_model(monkeypatch)
    with pytest.raises(ValueError, match="owner-only window holds no copy"):
        ring_buddy.install(model, mapping, balance=True, windows=1)
    with pytest.raises(ValueError, match="owner-only window holds no copy"):
        ring_buddy.install(model, mapping, balance=False, static_from=56, windows=1)
    with pytest.raises(ValueError, match="three_window_placement at tep4"):
        ring_buddy.install(model, mapping, persistent=True, windows=3)
    assert all(not hasattr(layer.feed_forward, "_replicated_experts") for layer in model.layers)


def test_two_windows_at_tep4_stack_the_predecessors_64_experts(monkeypatch):
    """``replicated_experts`` at tep4 (not composed by default): 128-expert stacks."""
    model, mapping = _tep4_model(monkeypatch)
    monkeypatch.setattr(ring_buddy, "_ring_block", lambda local, mapping, back: local + 1000)
    monkeypatch.setattr(ring_buddy, "_nvfp4_call", lambda backend, routing: None)
    assert ring_buddy.install(model, mapping) == [0, 1]
    state = model.layers[0].feed_forward._replicated_experts
    assert (state.windows, state.block, state.default_window) == (2, 64, 128)
    for stack in state.stacks.values():
        assert stack.shape[0] == 128 and torch.equal(stack[64:], stack[:64] + 1000)


def test_eligible_reads_the_ring_from_the_mapping():
    """tep8 needs 32-expert own blocks, tep4 64-expert ones; any other size is not served."""
    from tensorrt_llm._torch.moe.fused_moe.moe_op_backend import TRTLLMOpBackend

    def moe_of(block, rank):
        backend = types.SimpleNamespace(
            has_nvfp4=True,
            op_backend=object.__new__(TRTLLMOpBackend),
            num_slots=256,
            expert_size_per_partition=block,
            slot_start=block * rank,
            bias=False,
        )
        for name in ring_buddy.STACKED["nvfp4"]:
            setattr(backend, name, torch.zeros(block, 1))
        return types.SimpleNamespace(experts=types.SimpleNamespace(backend=backend, comm=None))

    def mapping_of(tp, rank):
        return types.SimpleNamespace(
            tp_size=tp, moe_ep_size=tp, moe_tp_size=1, moe_ep_rank=rank, tp_rank=rank
        )

    assert ring_buddy.eligible(moe_of(32, 5), mapping_of(8, 5)) is None
    assert ring_buddy.eligible(moe_of(64, 3), mapping_of(4, 3)) is None
    assert "64-expert block" in ring_buddy.eligible(moe_of(32, 3), mapping_of(4, 3))
    assert "neither tp8/ep8 nor tp4/ep4" in ring_buddy.eligible(moe_of(128, 1), mapping_of(2, 1))
