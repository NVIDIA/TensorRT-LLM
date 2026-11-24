# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The cep target's static expert placement (``kimi_k31_expert_placement``), on the CPU.

* The placement file (``ops/moe/_rubin_expert_placement``): the generic Kimi K3 work's
  ``kimi_k3_expert_placement/v1`` format, v2 as its v1 part, and every malformed file refused.
* The relabel equivalence on a small model (16 experts, top-4, EP4 over a cep4-shaped group,
  three MoE layers, one placed): the cep weight table loads every rank twice from one synthetic
  NVFP4 checkpoint, without and with the placement. Each placed slot holds the bytes of the
  checkpoint expert the file names, the router rows and score-correction bias follow it, the rank
  reads exactly the experts its slots hold, and the MoE layer computed from the loaded tables
  (router, top-k over slots, the slots' experts) gives the same bits under both loads.
* An EP16-balanced placement stays balanced at EP8: a cep8 rank holds two whole cep16 ranks.
"""

from __future__ import annotations

import importlib
import json
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.config import KimiK31TextConfig
from tensorrt_llm.mapping import CpType, Mapping

pytestmark = pytest.mark.cpu_only

_FAMILY = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31"
_PLACEMENT = f"{_FAMILY}.ops.moe._rubin_expert_placement"
_WEIGHTS = f"{_FAMILY}.kimi_k31_pro_nvfp4__sm_107__cep.weights"

#: The small model: layer 0 dense, layers 1..3 MoE; 16 experts of an NVFP4 latent 128 wide.
E, EP, TOP_K, LAYERS = 16, 4, 4, 4
LATENT, INTER, HIDDEN = 128, 128, 256
_TEXT = dict(
    num_hidden_layers=LAYERS,
    hidden_size=HIDDEN,
    intermediate_size=256,
    num_attention_heads=4,
    q_lora_rank=64,
    kv_lora_rank=64,
    qk_nope_head_dim=16,
    qk_rope_head_dim=16,
    v_head_dim=16,
    vocab_size=512,
    num_experts=E,
    num_experts_per_token=TOP_K,
    moe_intermediate_size=INTER,
    # 48 x 128: the shared experts' 6144-row intermediate the table is written for.
    num_shared_experts=48,
    routed_expert_hidden_size=LATENT,
    first_k_dense_replace=1,
    attn_res_block_size=2,
    linear_attn_config=dict(
        full_attn_layers=[4],
        kda_layers=[1, 2, 3],
        num_heads=8,
        head_dim=16,
        short_conv_kernel_size=4,
    ),
    dsa_config=dict(
        dsa_layers=[4],
        index_topk=64,
        index_num_heads=4,
        index_head_dim=32,
        index_k_norm_type="layernorm",
    ),
)
#: Layer 1 placed by the file, layer 2 listed as the identity (dropped), layer 3 left out.
_PLACED = 1


def _placement_module():
    return importlib.import_module(_PLACEMENT)


def _file(**changes) -> dict:
    g = torch.Generator().manual_seed(7)
    data = {
        "format": "kimi_k3_expert_placement/v1",
        "num_experts": E,
        "ep_size": EP,
        "expert_of_slot": {
            str(_PLACED): torch.randperm(E, generator=g).tolist(),
            "2": list(range(E)),
        },
    }
    data.update(changes)
    return data


# --- the file ---


def test_the_file_loads_its_placed_layers(tmp_path):
    mod = _placement_module()
    path = tmp_path / "placement.json"
    path.write_text(json.dumps(_file()))
    tables, file_ep = mod.read_expert_placement(str(path), E, range(1, LAYERS), EP)
    assert file_ep == EP
    assert sorted(tables) == [_PLACED]  # the identity layer is dropped
    assert tables[_PLACED] == tuple(_file()["expert_of_slot"][str(_PLACED)])
    assert mod.checkpoint_expert(tables, _PLACED, 3) == tables[_PLACED][3]
    assert mod.checkpoint_expert(tables, 3, 3) == 3
    assert mod.checkpoint_expert(None, _PLACED, 3) == 3


def test_a_v2_file_loads_as_its_v1_part():
    mod = _placement_module()
    data = _file(format="kimi_k3_expert_placement/v2", replica_slots=2, replicas={"1": [[0, 1]]})
    tables, _ = mod.parse_expert_placement(data, E, range(1, LAYERS), EP)
    assert tables[_PLACED] == tuple(data["expert_of_slot"][str(_PLACED)])


@pytest.mark.parametrize(
    "changes, named",
    [
        (dict(format="kimi_k3_expert_placement/v9"), "format"),
        (dict(num_experts=E * 2), "a placement of 32 experts"),
        (dict(expert_of_slot={"1": [0] * E}), "not a permutation"),
        (dict(expert_of_slot={"0": list(range(E))}), "layer 0 is not a MoE layer"),
        (dict(expert_of_slot={"9": list(range(E))}), "layer 9 is not a MoE layer"),
        (dict(ep_size=3), "ep_size 3 does not split"),
    ],
    ids=["format", "experts", "permutation", "dense-layer", "past-the-model", "file-ep"],
)
def test_a_malformed_file_is_refused(changes, named):
    mod = _placement_module()
    with pytest.raises(ValueError, match=named):
        mod.parse_expert_placement(_file(**changes), E, range(1, LAYERS), EP)


def test_an_ep_size_that_does_not_split_the_experts_is_refused():
    mod = _placement_module()
    with pytest.raises(ValueError, match="EP 3 does not split"):
        mod.parse_expert_placement(_file(), E, range(1, LAYERS), 3)


def test_a_balanced_placement_stays_balanced_on_a_divisor_ep():
    """An EP16 placement balanced greedily on skewed counts: each EP8 rank holds two whole EP16
    ranks, so its load is at most the EP16 imbalance, and well below the identity's."""
    mod = _placement_module()
    assert mod.balanced_for(16, 8) and mod.balanced_for(16, 16)
    assert not mod.balanced_for(8, 16) and not mod.balanced_for(None, 8)
    g = torch.Generator().manual_seed(3)
    counts = (torch.rand(896, generator=g) ** 4 * 1000).tolist()
    per_rank, load, held = 896 // 16, [0.0] * 16, [[] for _ in range(16)]
    for e in sorted(range(896), key=lambda e: -counts[e]):
        r = min((r for r in range(16) if len(held[r]) < per_rank), key=lambda r: load[r])
        held[r].append(e)
        load[r] += counts[e]
    table = tuple(e for r in range(16) for e in sorted(held[r]))

    def imbalance(t, ep):
        loads = mod.rank_loads(counts, t, ep)
        return max(loads) / (sum(loads) / ep)

    identity = tuple(range(896))
    assert imbalance(table, 16) < 1.01 < imbalance(identity, 16)
    assert imbalance(table, 8) <= imbalance(table, 16) + 1e-9
    assert imbalance(table, 8) < imbalance(identity, 8)


# --- the relabel on a small model ---


def _checkpoint() -> dict:
    """A synthetic NVFP4 export of the small model's MoE layers: per expert, E2M1 codes, E4M3
    block scales, weight_scale_2 and input_scale (1.0, one per layer); the router and its bias."""
    g = torch.Generator().manual_seed(11)
    ckpt = {}
    for i in range(1, LAYERS):
        base = f"language_model.model.layers.{i}.block_sparse_moe."
        ckpt[base + "gate.weight"] = torch.randn(E, HIDDEN, generator=g).to(torch.bfloat16)
        ckpt[base + "gate.e_score_correction_bias"] = torch.rand(E, generator=g) * 0.1
        for e in range(E):
            for w, (rows, cols) in (("w1", (INTER, LATENT)), ("w3", (INTER, LATENT)),
                                    ("w2", (LATENT, INTER))):  # fmt: skip
                key = f"{base}experts.{e}.{w}."
                ckpt[key + "weight"] = torch.randint(
                    0, 256, (rows, cols // 2), generator=g, dtype=torch.uint8
                )
                steps = torch.randint(120, 132, (rows, cols // 16), generator=g).float() / 126.0
                ckpt[key + "weight_scale"] = steps.to(torch.float8_e4m3fn)
                ckpt[key + "weight_scale_2"] = torch.tensor(0.01 + 0.001 * e)
                ckpt[key + "input_scale"] = torch.tensor(1.0)
    return ckpt


def _rows(weights, d) -> tuple:
    """The table rows the relabel touches: the router and its bias, every routed-expert row."""
    return tuple(
        r
        for r in (*weights.w1_rows(d), *weights.w2_rows(d))
        if r.name in ("moe_router", "moe_router_bias") or r.name.startswith("moe_fc")
    )


def _load(rank: int, placement, ckpt: dict):
    """Rank ``rank`` of a cep4 group loaded from ``ckpt`` through the weight table."""
    weights = importlib.import_module(_WEIGHTS)
    core = SimpleNamespace(
        role="context",
        kda_split="ulysses",
        expert_placement=placement,
        model_config=SimpleNamespace(
            pretrained_config=KimiK31TextConfig(**_TEXT),
            mapping=Mapping(
                world_size=EP,
                rank=rank,
                tp_size=1,
                cp_size=EP,
                cp_config={"cp_type": CpType.ULYSSES},
                moe_tp_size=1,
                moe_ep_size=EP,
            ),
            torch_dtype=torch.bfloat16,
        ),
    )
    core.dims = weights.dims(core)
    assert core.dims.expert_of_slot is placement
    rows = _rows(weights, core.dims)
    core.w = weights.declare(core.dims, rows)
    params = {f"w.{k}": p for k, p in core.w.items()}
    consumed = weights.fill_table(core, ckpt, rows, params)
    weights.place_routers(core)
    return core, consumed


_PLANES = (
    "moe_fc1_w",
    "moe_fc1_s",
    "moe_fc2_w",
    "moe_fc2_s",
    "moe_fc31_alpha",
    "moe_fc2_alpha",
    "moe_fc1_norm_const",
)
_E2M1 = torch.tensor(
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
)


def _unswizzle(s: torch.Tensor) -> torch.Tensor:
    m, c = s.shape
    return s.reshape(m // 128, c // 4, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(m, c)


def _dequant(codes: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """FP32 values of E2M1 pairs ``[rows, k / 2]`` under 128x4-swizzled E4M3 scales."""
    lo, hi = _E2M1[(codes & 0xF).long()], _E2M1[(codes >> 4).long()]
    vals = torch.stack((lo, hi), -1).reshape(codes.shape[0], -1)
    sf = _unswizzle(scales.view(torch.uint8)).view(torch.float8_e4m3fn).float()
    return vals * sf.repeat_interleave(16, dim=1)


def _layer(cores, i: int, x: torch.Tensor) -> torch.Tensor:
    """Layer ``i``'s routed MoE from the loaded tables of every rank (slot-major): sigmoid router
    with the score-correction bias, top-4 slots (ties to the lower slot), weights renormalized,
    each slot's FC1 (16 gate / 16 up rows), SiLU-gated, FC2."""
    router = cores[0].w[f"l{i}_moe_router"].float()
    bias = cores[0].w[f"l{i}_moe_router_bias"].float()
    for c in cores[1:]:  # the router is whole on every rank
        assert torch.equal(c.w[f"l{i}_moe_router"], cores[0].w[f"l{i}_moe_router"])
    scores = torch.sigmoid(x.float() @ router.t())
    ids = torch.sort(scores + bias, dim=-1, descending=True, stable=True).indices[:, :TOP_K]
    picked = torch.gather(scores, 1, ids)
    weights = picked / picked.sum(-1, keepdim=True)
    xin = x.float()[:, :LATENT]
    out = torch.zeros(x.shape[0], LATENT)
    per_rank = E // EP
    for k in range(TOP_K):
        for slot in ids[:, k].unique().tolist():
            core, j = cores[slot // per_rank], slot % per_rank
            w1 = _dequant(core.w[f"l{i}_moe_fc1_w"][j], core.w[f"l{i}_moe_fc1_s"][j])
            w1 = w1 * core.w[f"l{i}_moe_fc31_alpha"][j]
            gate_up = w1.view(-1, 2, 16, LATENT)
            gate, up = gate_up[:, 0].reshape(INTER, LATENT), gate_up[:, 1].reshape(INTER, LATENT)
            w2 = _dequant(core.w[f"l{i}_moe_fc2_w"][j], core.w[f"l{i}_moe_fc2_s"][j])
            w2 = w2 * core.w[f"l{i}_moe_fc2_alpha"][j]
            rows = ids[:, k] == slot
            h = torch.nn.functional.silu(xin[rows] @ gate.t()) * (xin[rows] @ up.t())
            out[rows] += weights[rows, k : k + 1] * (h @ w2.t())
    return out


def test_the_placement_relabels_the_load_and_keeps_the_layer(tmp_path):
    mod = _placement_module()
    path = tmp_path / "placement.json"
    path.write_text(json.dumps(_file()))
    placement, _ = mod.read_expert_placement(str(path), E, range(1, LAYERS), EP)
    table = placement[_PLACED]
    ckpt = _checkpoint()
    plain = [_load(r, None, ckpt) for r in range(EP)]
    placed = [_load(r, placement, ckpt) for r in range(EP)]
    per_rank = E // EP
    for i in range(1, LAYERS):
        expert_of_slot = table if i == _PLACED else tuple(range(E))
        # Each slot holds the bytes of its checkpoint expert, as the identity load holds them.
        for slot in range(E):
            e = expert_of_slot[slot]
            got, ref = placed[slot // per_rank][0], plain[e // per_rank][0]
            for name in _PLANES:
                a, b = got.w[f"l{i}_{name}"][slot % per_rank], ref.w[f"l{i}_{name}"][e % per_rank]
                assert torch.equal(a, b), (i, slot, e, name)
        # The router rows follow the slots.
        for name in ("moe_router", "moe_router_bias"):
            got, ref = placed[0][0].w[f"l{i}_{name}"], plain[0][0].w[f"l{i}_{name}"]
            assert torch.equal(got, ref[list(expert_of_slot)]), (i, name)
        # Each rank reads exactly the experts its slots hold.
        for r, (core, consumed) in enumerate(placed):
            held = {expert_of_slot[s] for s in range(r * per_rank, (r + 1) * per_rank)}
            base = f"language_model.model.layers.{i}.block_sparse_moe.experts."
            read = {int(k[len(base) :].split(".")[0]) for k in consumed if k.startswith(base)}
            weights_read = {
                int(k[len(base) :].split(".")[0])
                for k in consumed
                if k.startswith(base) and k.endswith(".weight")
            }
            assert weights_read == held, (i, r)
            assert read == set(range(E))  # every expert's input_scale, on every rank
    assert table != tuple(range(E)) and placed[0][0].w["l1_moe_router"].shape == (E, HIDDEN)
    # The MoE layer of the loaded tables gives the same bits under both loads.
    x = torch.randn(64, HIDDEN, generator=torch.Generator().manual_seed(5)).to(torch.bfloat16)
    for i in range(1, LAYERS):
        a = _layer([c for c, _ in plain], i, x)
        b = _layer([c for c, _ in placed], i, x)
        assert torch.equal(a, b), (i, (a - b).abs().max())
        assert a.abs().max() > 0


def test_the_offwindow_keys_follow_the_placement(tmp_path, monkeypatch):
    """The checkpoint keys a rank leaves unread: the experts no slot of its window holds."""
    mod = _placement_module()
    weights = importlib.import_module(_WEIGHTS)
    placement, _ = mod.parse_expert_placement(_file(), E, range(1, LAYERS), EP)
    table, per_rank = placement[_PLACED], E // EP
    ckpt = _checkpoint()
    cores = [_load(r, placement, ckpt)[0] for r in range(EP)]
    for r, core in enumerate(cores):
        monkeypatch.setattr(weights, "dims", lambda core, d=core.dims: d)
        left = weights.w2_offwindow_keys(core, None)
        held = set(table[r * per_rank : (r + 1) * per_rank])
        base = f"language_model.model.layers.{_PLACED}.block_sparse_moe.experts."
        unread = {int(k[len(base) :].split(".")[0]) for k in left if k.startswith(base)}
        assert unread == set(range(E)) - held
        assert not any(k.endswith("input_scale") for k in left)
