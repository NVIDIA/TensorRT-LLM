# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The chain's checkpoint reads and packing (``_chain_weights``), on a synthetic checkpoint (CPU).

Each test checks a reader against an independent computation from the whole checkpoint tensors:
the rank's slices, the FP8 block dequantization (FP32 product, one BF16 rounding), the
in-projection / front composition and the NVFP4 TP16 expert packing (row orders, UMMA scale
atoms, the w1 / w3 global-scale requantization, alphas).
"""

import functools
import importlib.util
import os
import types

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl.ops.chain import k3_ull_chain as cw

pytestmark = pytest.mark.cpu_only


def _load_synthetic():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_synthetic_checkpoint.py")
    spec = importlib.util.spec_from_file_location("k3_ull_chain_synthetic_checkpoint", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


synth = _load_synthetic()
P = "language_model."


def bits(t: torch.Tensor) -> torch.Tensor:
    t = t.contiguous()
    return t.view(torch.uint8) if t.element_size() == 1 else t.view(-1).view(torch.uint8)


def same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(bits(a), bits(b))


def dequant_full(ckpt, key, rows=slice(None), cols=slice(None)) -> torch.Tensor:
    """The BF16 block ``[rows, cols]`` of a checkpoint matrix: each FP8 code times its 128 x 128
    block's scale in FP32, rounded once."""
    source = ckpt[key]
    value = source[rows, cols]
    if value.dtype == torch.bfloat16:
        return value
    grid = ckpt[key + "_scale_inv"].full().float()
    grid = grid.reshape(grid.shape[0], grid.shape[2])
    n, k = source.get_shape()
    r = torch.arange(n)[rows] // 128
    c = torch.arange(k)[cols] // 128
    return (value.float() * grid[r][:, c]).to(torch.bfloat16)


@pytest.fixture
def few_experts(monkeypatch):
    """The routed-expert reader over 2 experts (the synthetic checkpoint holds 2)."""
    reader = functools.partial(cw.read_routed_experts_tp16, experts=2)
    monkeypatch.setattr(cw, "read_routed_experts_tp16", reader)


def test_dequant_block_offsets():
    gen = torch.Generator().manual_seed(0)
    codes = torch.randint(0, 0x7E, (300, 260), generator=gen, dtype=torch.uint8)
    codes = codes.view(torch.float8_e4m3fn)
    scale = torch.rand(3, 1, 3, 1, generator=gen) + 0.5
    full = cw.dequant_block(codes, scale)
    grid = scale.reshape(3, 3)
    for r in range(0, 300, 37):
        for c in range(0, 260, 41):
            want = (codes[r, c].float() * grid[r // 128, c // 128]).to(torch.bfloat16)
            assert same(full[r, c], want)
    # A block starting inside a scale block reads the grid from that block on.
    part = cw.dequant_block(codes[70:250, 130:200], scale.reshape(3, 3)[0:, 1:], 70, 130)
    assert same(part, full[70:250, 130:200])


@pytest.mark.parametrize("rank", [0, 5, 15])
def test_read_kda(rank):
    ckpt = synth.synthetic_checkpoint([1], experts=0)
    w = cw.read_kda(ckpt, 1, rank, prefix=P)
    a = f"{P}model.layers.1.self_attn."
    d, heads = 768, 6
    hs = slice(rank * d, (rank + 1) * d)
    rows, pitch = cw.inproj_rows()
    assert (rows, pitch) == (3206, 3216)
    for k, name in enumerate(("q_proj", "k_proj", "v_proj", "g_proj")):
        assert same(w.in_proj[k * d : (k + 1) * d], dequant_full(ckpt, a + name + ".weight", hs))
    assert same(w.in_proj[4 * d : 4 * d + 128], dequant_full(ckpt, a + "f_a_proj.weight"))
    beta = dequant_full(ckpt, a + "b_proj.weight", slice(rank * heads, (rank + 1) * heads))
    assert same(w.in_proj[4 * d + 128 : rows], beta)
    assert not w.in_proj[rows:].view(torch.int16).any()
    assert same(w.f_b, dequant_full(ckpt, a + "f_b_proj.weight", hs))
    assert same(w.o_proj, dequant_full(ckpt, a + "o_proj.weight", slice(None), hs))
    conv = torch.cat([ckpt[a + f"{x}_conv1d.weight"].full()[hs] for x in "qkv"])
    assert same(w.conv, conv.reshape(3 * d, 1, 4))
    a_log = ckpt[a + "A_log"].full().reshape(-1)[rank * heads : (rank + 1) * heads]
    assert same(w.a_log, a_log)
    assert same(w.dt_bias, ckpt[a + "dt_bias"].full()[hs])
    assert same(w.o_norm, ckpt[a + "o_norm.weight"].full())
    assert cw.read_kda(ckpt, 1, rank, prefix=P, o_proj=False).o_proj is None


@pytest.mark.parametrize("fp8", [False, True])
def test_read_moe_dense(fp8, few_experts):
    rank = 7
    ckpt = synth.synthetic_checkpoint([3], experts=2, fp8_moe_dense=fp8)
    w = cw.read_moe(ckpt, 3, rank, prefix=P)
    m = f"{P}model.layers.3.block_sparse_moe."
    router = dequant_full(ckpt, m + "gate.weight")
    down = dequant_full(ckpt, m + "routed_expert_down_proj.weight")
    up = dequant_full(ckpt, m + "routed_expert_up_proj.weight")
    # The front GEMM's rows: router rows of the rank, 8 zero rows, its latent-down shard.
    front = torch.cat((router[56 * rank : 56 * rank + 56], router.new_zeros(8, 7168),
                       down[224 * rank : 224 * rank + 224]))  # fmt: skip
    assert same(w.front, front)
    assert same(w.latent_up, up)
    assert same(w.up_shard, up[448 * rank : 448 * rank + 448])
    assert w.up_shard.untyped_storage().data_ptr() == w.latent_up.untyped_storage().data_ptr()
    assert same(w.score_bias, ckpt[m + "gate.e_score_correction_bias"].full())
    assert same(w.latent_norm, ckpt[m + "routed_expert_norm.weight"].full().to(torch.bfloat16))
    ss = slice(384 * rank, 384 * rank + 384)
    gate = dequant_full(ckpt, m + "shared_experts.gate_proj.weight")[ss]
    upp = dequant_full(ckpt, m + "shared_experts.up_proj.weight")[ss]
    assert same(w.shared_gate_up, torch.cat((gate, upp)))
    down_shared = dequant_full(ckpt, m + "shared_experts.down_proj.weight")[:, ss]
    assert same(w.shared_down, down_shared.contiguous())
    assert w.routed["w13"].shape == (2, 384, 1792)


def test_read_routed_experts_slices():
    rank, experts = 3, 2
    ckpt = synth.synthetic_checkpoint([5], experts=experts)
    raw = cw.read_routed_experts_tp16(ckpt, 5, rank, prefix=P, experts=experts)
    base = f"{P}model.layers.5.block_sparse_moe.experts."
    rows = slice(192 * rank, 192 * rank + 192)
    for e in range(experts):
        for w in ("w1", "w3"):
            assert same(raw[w][e], ckpt[f"{base}{e}.{w}.weight"].full()[rows])
            sf = ckpt[f"{base}{e}.{w}.weight_scale"].full()[rows].view(torch.uint8)
            assert same(raw[w + "_sf"][e], sf)
        w2 = ckpt[f"{base}{e}.w2.weight"].full()[:, 96 * rank : 96 * rank + 96]
        assert same(raw["w2"][e], w2.contiguous())
        sf = ckpt[f"{base}{e}.w2.weight_scale"].full()[:, 12 * rank : 12 * rank + 12]
        assert same(raw["w2_sf"][e], sf.contiguous().view(torch.uint8))
        s = {w: float(ckpt[f"{base}{e}.{w}.input_scale"].full()) for w in ("w1", "w2", "w3")}
        assert float(raw["fc1_input_scale"][e]) == pytest.approx(max(s["w1"], s["w3"]), rel=0)
        assert float(raw["fc2_input_scale"][e]) == pytest.approx(s["w2"], rel=0)
        for w in ("w1", "w2", "w3"):
            want = float(ckpt[f"{base}{e}.{w}.weight_scale_2"].full())
            assert float(raw[f"{w}_scale2"][e]) == pytest.approx(want, rel=0)


def _random_experts(experts: int, gen: torch.Generator, mismatched: bool):
    u8 = functools.partial(torch.randint, 0, 256, generator=gen, dtype=torch.uint8)
    scales = functools.partial(torch.randint, 0x28, 0x48, generator=gen, dtype=torch.uint8)
    s1 = torch.rand(experts, generator=gen) + 0.5
    s3 = s1 * torch.tensor([2.0 if mismatched and e % 2 else 1.0 for e in range(experts)])
    return dict(
        w1=u8((experts, 192, 1792)), w3=u8((experts, 192, 1792)), w2=u8((experts, 3584, 96)),
        w1_sf=scales((experts, 192, 224)), w3_sf=scales((experts, 192, 224)),
        w2_sf=scales((experts, 3584, 12)), w1_scale2=s1, w3_scale2=s3,
        w2_scale2=torch.rand(experts, generator=gen) + 0.5,
        fc1_input_scale=torch.rand(experts, generator=gen) + 0.1,
        fc2_input_scale=torch.rand(experts, generator=gen) + 0.1,
    )  # fmt: skip


def _requant(sf: torch.Tensor, ratio: torch.Tensor) -> torch.Tensor:
    value = sf.view(torch.float8_e4m3fn).float() * ratio.view(-1, 1, 1)
    return value.to(torch.float8_e4m3fn).view(torch.uint8)


@pytest.mark.parametrize("mismatched", [False, True])
def test_pack_routed_experts(mismatched):
    experts = 3
    raw = _random_experts(experts, torch.Generator().manual_seed(1), mismatched)
    packed = cw.pack_routed_experts_tp16(**raw)
    cw.check_routed(packed)
    order = cw.w13_row_order()
    assert sorted(order.tolist()) == list(range(384))
    inverse = torch.empty_like(order)
    inverse[order] = torch.arange(384)
    # Rows ``channel + 192 * is_gate``: up (w3) channels first, then gate (w1).
    assert same(packed["w13"][:, inverse], torch.cat((raw["w3"], raw["w1"]), 1))
    s13 = torch.maximum(raw["w1_scale2"], raw["w3_scale2"])
    sf1, sf3 = raw["w1_sf"], raw["w3_sf"]
    if mismatched:
        sf1, sf3 = _requant(sf1, raw["w1_scale2"] / s13), _requant(sf3, raw["w3_scale2"] / s13)
    sf13 = cw._unatoms(packed["w13_sf"], 3, 3584 // 64)[:, inverse]
    assert same(sf13, torch.cat((sf3, sf1), 1))
    columns = cw.w2_row_order(True)
    where = torch.empty_like(columns)
    where[columns] = torch.arange(3584)
    assert same(packed["w2"][:, where], raw["w2"])
    assert same(cw._unatoms(packed["w2_sf"], 28, 3)[:, where], raw["w2_sf"])
    in1, in2 = raw["fc1_input_scale"].max(), raw["fc2_input_scale"].max()
    assert same(packed["alphas"][:experts, 0], (in1 * s13).float())
    assert same(packed["alphas"][:experts, 1], (in2 * raw["w2_scale2"]).float())
    assert same(packed["alphas"][experts], torch.stack((1.0 / in2, torch.zeros(()))))
    assert same(packed["x_gsf"], (1.0 / in1).reshape(1))
    assert same(packed["h_gsf"], (1.0 / in2).reshape(1))
    # repack_w2 into the publish order is the identity; direct_columns is a permutation.
    w2, w2_sf = cw.repack_w2(packed, columns)
    assert same(w2, packed["w2"]) and same(w2_sf, packed["w2_sf"])
    assert sorted(cw.direct_columns().tolist()) == list(range(3584))


def test_scale_atoms():
    gen = torch.Generator().manual_seed(2)
    scales = torch.randint(0, 256, (2, 256, 8), generator=gen, dtype=torch.uint8)
    atoms = cw._atoms(scales, 2, 2)
    assert same(cw._unatoms(atoms, 2, 2), scales)
    # Byte (m % 32) * 16 + (m // 32) * 4 + k of the 512 B atom (tile, K64 block).
    for e, tile, kb, m, k in ((0, 0, 0, 5, 1), (1, 1, 1, 100, 3), (1, 0, 1, 127, 0)):
        offset = (tile * 2 + kb) * 512 + (m % 32) * 16 + (m // 32) * 4 + k
        assert atoms[e, offset] == scales[e, 128 * tile + m, 4 * kb + k]


def test_chain_config():
    assert cw.chain_config(synth.text_config()) == cw.ChainConfig()
    cfg = cw.chain_config(synth.text_config(rms_norm_eps=1e-6, routed_scaling_factor=2.5))
    assert (cfg.rms_norm_eps, cfg.routed_scaling_factor) == (1e-6, 2.5)
    text = types.SimpleNamespace(
        linear_attn_config={"gate_lower_bound": -4.0}, rms_norm_eps=1e-5, num_hidden_layers=93
    )
    model_config = types.SimpleNamespace(pretrained_config=types.SimpleNamespace(text_config=text))
    assert cw.chain_config(model_config).gate_lower_bound == -4.0


def test_load_chain_weights(few_experts):
    rank = 9
    ckpt = synth.synthetic_checkpoint([0, 1, 3, 12], experts=2)
    w = cw.load_chain_weights(ckpt, synth.text_config(), rank, "cpu", layers=[0, 1, 3, 12])
    assert (w.tp, w.rank, sorted(w.layers)) == (16, rank, [0, 1, 3, 12])
    assert w.layers[0].moe is None and w.layers[0].kda.o_proj is None
    assert w.layers[1].kda.o_proj is not None and w.layers[1].moe is not None
    assert w.layers[3].kda is None and w.layers[3].moe is not None
    assert w.output.num_attn_res_blocks == 8
    for i in (0, 1, 3, 12):
        a = w.layers[i].attn_res
        base = f"{P}model.layers.{i}."
        assert same(
            a.input_layernorm.weight,
            ckpt[base + "input_layernorm.weight"].full().to(torch.bfloat16),
        )
        assert same(a.mlp_res_proj.weight, ckpt[base + "mlp_res_proj.weight"].full())
        assert a.mlp_res_norm.variance_epsilon == 1e-5
        assert a.prev_valid_blocks == -(-i // 12)
        assert a.is_block_write_layer == (i % 12 == 0) and a.block_write_idx == i // 12
    assert same(w.output.norm.weight, ckpt[f"{P}model.norm.weight"].full().to(torch.bfloat16))
    assert w.nbytes() == sum(t.untyped_storage().nbytes() for t in {
        t.untyped_storage().data_ptr(): t for t in w.tensors()}.values())  # fmt: skip


def test_load_chain_weights_rejects_other_models():
    with pytest.raises(ValueError, match="layers"):
        cw.load_chain_weights({}, synth.text_config(num_hidden_layers=61), 0, "cpu")
