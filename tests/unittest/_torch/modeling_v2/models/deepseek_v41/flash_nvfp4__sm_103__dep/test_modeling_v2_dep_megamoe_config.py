# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MegaMoE build-time decisions of the dep target (`megamoe.py`), on CPU with synthetic engine configs.

Each test runs at both deployed worlds of the target: 8 ranks (384 slots, max_num_tokens 1280, no copies) and
32 ranks (416 slots, max_num_tokens 150, 32 copies).

Proved here: only `moe_config.backend: MEGAMOE_CUTEDSL` selects the path; the slot count, per-rank slots, bucket
ladder and fused shared expert are the deployment's; the static
placement holds every expert, never twice on a rank, and routes every pick to a slot of the
expert it named, on every rank; a configured `initial_global_assignments` replaces the default; a learned or online
placement request runs the static one and says so, a dynamic mode or a layout the kernel cannot serve is refused, and
a MoE token cap below the engine's per-rank cap (which stock would run in chunks) is refused. Every refusal is a
`NoTarget`, so `modeling_v2: auto` falls back to the built-in model (routing declines the same settings first).
"""

import importlib
from types import SimpleNamespace as NS

import pytest
import torch

__extra_import_path__ = [".."]  # the family test directory (_megamoe_reference)
from _megamoe_reference import megamoe_slots, megamoe_slots_by_placement  # noqa: E402

from tensorrt_llm._torch.modeling_v2._router_index import NoTarget

pytestmark = pytest.mark.cpu_only

_FAMILY = "tensorrt_llm._torch.modeling_v2.models.deepseek_v41"
mega = importlib.import_module(f"{_FAMILY}.flash_nvfp4__sm_103__dep.megamoe")
slots_op = importlib.import_module(f"{_FAMILY}.ops.moe.megamoe_expert_slots")
ws_op = importlib.import_module(f"{_FAMILY}.ops.moe.megamoe_workspace")

EXPERTS = 384
# (world size, slots, engine max_num_tokens, bucket ladder) of the deployments the target serves
_DEPLOYMENTS = {"dep8": (8, 384, 1280, [256, 1024, 1280]), "dep32": (32, 416, 150, [150])}
WS = SLOTS = MNT = LADDER = None


@pytest.fixture(autouse=True, params=sorted(_DEPLOYMENTS))
def deployment(request):
    """Run every test at each world the target serves (the module constants are the deployment's)."""
    global WS, SLOTS, MNT, LADDER
    WS, SLOTS, MNT, LADDER = _DEPLOYMENTS[request.param]
    yield request.param


CFG = NS(
    n_routed_experts=EXPERTS, hidden_size=5120, moe_intermediate_size=2304, num_experts_per_tok=6,
    swiglu_limit=10.0, n_shared_experts=1, routed_scaling_factor=1.5,
)  # fmt: skip


def _balancer(**kw):
    return NS(
        mode="standard",
        num_slots=SLOTS,
        layer_updates_per_iter=0,
        warmup_placement=None,
        initial_global_assignments=None,
        **kw,
    )


def _model_config(rank=0, backend="MEGAMOE_CUTEDSL", balancer="default", fuse=True, **mapping):
    m = dict(
        enable_attention_dp=True, moe_tp_size=1, moe_ep_size=WS, world_size=WS, moe_ep_rank=rank
    )
    m.update(mapping)
    return NS(
        moe_backend=backend, mapping=NS(**m), moe_load_balancer=_balancer() if balancer == "default" else balancer,
        megamoe_fuse_shared_expert=fuse, moe_max_num_tokens=MNT * WS, max_num_tokens=MNT,
    )  # fmt: skip


def test_only_the_megamoe_backend_selects_the_path():
    for backend in ("TRTLLM", "CUTEDSL", "CUTLASS", "AUTO"):
        assert mega.resolve_megamoe_config(_model_config(backend=backend), CFG) is None
    assert mega.resolve_megamoe_config(_model_config(), CFG) is not None


def test_the_deployment_numbers():
    mc = mega.resolve_megamoe_config(_model_config(rank=WS - 1), CFG)
    assert (mc.ep_size, mc.ep_rank, mc.num_experts, mc.num_slots, mc.top_k) == (
        WS,
        WS - 1,
        EXPERTS,
        SLOTS,
        6,
    )
    assert mc.local_slots == SLOTS // WS
    assert (mc.hidden, mc.inter, mc.clamp, mc.combine_format, mc.fused_shared) == (
        5120,
        2304,
        10.0,
        "bf16",
        True,
    )
    # the ladder is cut from min(MoE cap, engine cap): the MoE cap defaults to cap * dp_size, the engine's wins
    assert mc.max_tokens == MNT
    assert ws_op.megamoe_token_buckets(mc.max_tokens) == LADDER


def test_a_moe_cap_below_the_engine_cap_is_refused():
    """Stock runs a step past `moe_config.max_num_tokens` as several lockstep launches; this target launches each step
    whole, in a bucket of a ladder that ends at the cap, so a cap below the engine's per-rank `max_num_tokens` would
    leave the larger steps without a bucket: refused at resolution, on every rank alike. A cap at or above it runs."""
    low = _model_config()
    low.moe_max_num_tokens = MNT - 1
    with pytest.raises(NoTarget, match="below the engine's per-rank max_num_tokens"):
        mega.resolve_megamoe_config(low, CFG)
    for moe_cap in (MNT, MNT * WS):
        ok = _model_config()
        ok.moe_max_num_tokens = moe_cap
        assert mega.resolve_megamoe_config(ok, CFG).max_tokens == MNT


def test_no_balancer_means_one_copy_per_expert():
    mc = mega.resolve_megamoe_config(_model_config(balancer=None), CFG)
    assert mc.num_slots == EXPERTS


def test_the_static_placement_of_every_rank():
    mc = mega.resolve_megamoe_config(_model_config(), CFG)
    assign = mega.layer_assignments(mc, 7)
    assert len(assign) == SLOTS
    held = torch.bincount(torch.tensor(assign), minlength=EXPERTS)
    assert int(held.min()) >= 1 and int(held.sum()) == SLOTS
    assert int((held - 1).sum()) == SLOTS - EXPERTS  # copies only from the extra slots
    local = SLOTS // WS
    for rank in range(WS):
        mine = assign[rank * local : (rank + 1) * local]
        assert len(set(mine)) == local, f"rank {rank} holds an expert twice"
    if SLOTS == EXPERTS:
        assert assign == list(range(EXPERTS))
    else:  # rank r: experts 12r .. 12r + 12, the last one a copy of the next rank's first
        assert assign[:13] == list(range(13)) and assign[-13:] == [
            (12 * (WS - 1) + j) % EXPERTS for j in range(13)
        ]


@pytest.mark.parametrize("which", ["first", "second", "middle", "last"])
def test_every_pick_reaches_a_slot_of_its_expert(which):
    rank = {"first": 0, "second": 1, "middle": WS // 2, "last": WS - 1}[which]
    mc = mega.resolve_megamoe_config(_model_config(rank=rank), CFG)
    assign = mega.layer_assignments(mc, 3)
    table = slots_op.megamoe_slot_table(assign, EXPERTS, rank, "cpu")
    gen = torch.Generator().manual_seed(rank)
    ids = torch.randint(0, EXPERTS, (1280, 6), generator=gen, dtype=torch.int32)
    got = megamoe_slots(
        table, ids, EXPERTS
    )  # the routing launch's lookup of the table, in plain torch
    assert torch.equal(torch.tensor(assign)[got.long()].int(), ids)
    assert torch.equal(got, megamoe_slots_by_placement(assign, ids, rank))


def test_a_configured_placement_replaces_the_default():
    default = slots_op.megamoe_default_placement(EXPERTS, SLOTS, WS)
    local = SLOTS // WS
    # swap two experts between the first and the last rank, neither of which holds the other's expert: a valid
    # placement, so a rejection fails the test
    shuffled = list(default)
    a, b = 1, SLOTS - 2
    shuffled[a], shuffled[b] = shuffled[b], shuffled[a]
    slots_op.megamoe_check_placement(shuffled, EXPERTS, WS)
    balancer = _balancer()
    balancer.initial_global_assignments = {5: shuffled}
    mc = mega.resolve_megamoe_config(_model_config(balancer=balancer), CFG)
    assert mega.layer_assignments(mc, 5) == shuffled
    assert local == mc.local_slots
    with pytest.raises(KeyError):  # a layer without an entry is an error, not a silent default
        mega.layer_assignments(mc, 6)


def test_a_bad_configured_placement_is_refused():
    balancer = _balancer()
    balancer.initial_global_assignments = {1: list(range(SLOTS - 1))}
    mc = mega.resolve_megamoe_config(_model_config(balancer=balancer), CFG)
    with pytest.raises(ValueError, match="assignments"):
        mega.layer_assignments(mc, 1)
    balancer.initial_global_assignments = {1: [0] * SLOTS}
    mc = mega.resolve_megamoe_config(_model_config(balancer=balancer), CFG)
    with pytest.raises(ValueError):
        mega.layer_assignments(mc, 1)


def test_a_warmup_placement_request_is_served_and_the_online_balancer_refused(monkeypatch):
    """`warmup_placement` resolves to the same deployment (the layers join the engine's controller, see
    `test_modeling_v2_megamoe_placement`); `layer_updates_per_iter > 0` (the online balancer) is refused."""
    seen = []
    monkeypatch.setattr(mega.logger, "warning", lambda msg, *a, **k: seen.append(msg))
    plain = mega.resolve_megamoe_config(_model_config(), CFG)
    balancer = _balancer()
    balancer.warmup_placement = NS(planner="lpt")
    assert mega.resolve_megamoe_config(_model_config(balancer=balancer), CFG) == plain
    assert not seen
    balancer = _balancer()
    balancer.layer_updates_per_iter = 4
    with pytest.raises(NoTarget, match="online balancer"):
        mega.resolve_megamoe_config(_model_config(balancer=balancer), CFG)


@pytest.mark.parametrize(
    "mapping, balancer_kw, error, match",
    [
        (dict(enable_attention_dp=False), {}, NoTarget, "pure DEP"),
        (dict(moe_ep_size="half", moe_tp_size=2), {}, NoTarget, "pure DEP"),
        (dict(), dict(mode="per_layer_dynamic"), NoTarget, "static placement only"),
        (dict(), dict(num_slots="slots+1"), NoTarget, "multiple of EP"),
        (dict(), dict(num_slots="experts-ws"), NoTarget, "multiple of EP"),
    ],
    ids=["no-adp", "moe-tp", "dynamic-mode", "slots-not-multiple", "fewer-slots-than-experts"],
)  # fmt: skip
def test_what_the_kernel_cannot_serve_is_refused(mapping, balancer_kw, error, match):
    symbolic = {"half": WS // 2, "slots+1": SLOTS + 1, "experts-ws": EXPERTS - WS}
    mapping = {k: symbolic.get(v, v) for k, v in mapping.items()}
    balancer = _balancer()
    for k, v in balancer_kw.items():
        setattr(balancer, k, symbolic.get(v, v))
    with pytest.raises(error, match=match):
        mega.resolve_megamoe_config(_model_config(balancer=balancer, **mapping), CFG)


def test_the_fused_shared_expert_follows_the_engine_switch_and_its_conditions(monkeypatch):
    monkeypatch.setattr(mega.logger, "warning", lambda *a, **k: None)
    assert mega.resolve_megamoe_config(_model_config(fuse=False), CFG).fused_shared is False
    two = NS(**{**vars(CFG), "n_shared_experts": 2})
    assert mega.resolve_megamoe_config(_model_config(), two).fused_shared is False
    narrow = NS(**{**vars(CFG), "moe_intermediate_size": 2000})
    assert mega.resolve_megamoe_config(_model_config(), narrow).fused_shared is False
    # the combine wire format is the production one (bf16, which the fused shared expert needs); no environment switch
    monkeypatch.setenv("MEGAMOE_COMBINE_FORMAT", "32e4m3xe8m0")
    mc = mega.resolve_megamoe_config(_model_config(), CFG)
    assert mc.combine_format == "bf16" and mc.fused_shared is True
