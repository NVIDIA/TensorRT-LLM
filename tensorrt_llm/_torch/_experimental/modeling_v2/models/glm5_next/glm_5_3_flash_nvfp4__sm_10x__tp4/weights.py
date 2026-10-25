# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight manifest and loading of glm-5.3-flash-nvfp4 / sm_10x / tp4 (rank r of 4).

The parameter table (`param_specs`) is the single declaration both sides use: modeling.py
declares exactly these parameters (meta-init), and `load` fills exactly these from the
checkpoint, asserting shapes, dtypes and both directions of coverage.

Per rank: KDA heads [16 r, 16 r + 16), MLA heads [16 r, 16 r + 16), the dense MLPs' intermediate
features [3072 r, +3072), the shared expert's features [512 r, +512), the LM head's vocabulary
rows [38720 r, +38720). Embedding, norms, mHC, the MLA down projections and the indexer are
replicated. The routed experts follow the mapping (`Layout`): expert parallel (moe_ep 4: experts
[72 r, 72 r + 72), whole) or tensor parallel (moe_tp 4: every expert's features [512 r, +512)).
The MTP layer's BF16 experts are tensor parallel in both: its passes route a few tokens each, so
a slice of every expert spreads them evenly over the ranks. For the token-split prefill attention
every MLA layer also holds every head's q_b, absorption weights and (decoder layers) o_proj,
~0.2 GB per layer: a sequence-parallel prefill step attends its quarter of the rows with all 64
heads, and the rank's own heads are a slice of them.

Kernel-ready layouts built here (the kernels' contracts):
- NVFP4 scales in the CUTLASS SM100 K-major layout (`swizzle_sf`);
- expert, shared-expert and dense-MLP gate and up rows interleaved by 64 (a 128-row tile holds
  both halves of 64 features);
- KDA in_proj fused [q | k | v | b | f_a | g_a]; conv weights bf16 (the checkpoint's fp32 values
  are bf16 values: exact); MLA [q_a ; kv_a] fused, kv_b split per head into the absorption
  operands w_kc [h, 256, 512] / w_vc [h, 256, 512] and w_kc transposed per head (w_kcT
  [h, 512, 256], the decode absorption's GEMV rows); indexer [wk ; gate ; weights].
- NVFP4 scalars: alpha = input_scale * weight_scale_2 (GEMM), gs = input_scale (quantizer);
  experts share one input scale per layer (asserted).
"""

from typing import NamedTuple

import torch

PRE = "model.language_model."
H = 4096
V = 154880
L = 45
TP = 4
KDA_HEADS, KDA_D = 64, 128
KDA_HL = KDA_HEADS // TP
KDA_HD = KDA_HL * KDA_D  # 2048
KDA_NIN = 3 * KDA_HD + KDA_HL + 2 * KDA_D  # 6416
MLA_HEADS = 64
MLA_HL = MLA_HEADS // TP  # 16
NOPE, VH, LORA, QLORA = 256, 256, 512, 1536
IDX_HEADS, IDX_D = 32, 128
INTER = 12288
INTER_L = INTER // TP  # 3072
E = 288
MOE_I = 2048
SH_IL = MOE_I // TP  # 512 shared-expert features per rank
HC_MIX = 24


class Layout(NamedTuple):
    """This rank's share of the weights."""

    rank: int
    moe_tp: int  # the routed experts: moe_tp x moe_ep = 4
    moe_ep: int
    mtp: bool  # the MTP layer is loaded
    mtp_moe_tp: bool  # the MTP layer's experts tensor parallel (all of them, a slice each)
    ts: bool  # every head's MLA weights too (the token-split prefill attention)

    @staticmethod
    def of(rank: int, moe_tp: int, moe_ep: int, mtp: bool) -> "Layout":
        assert moe_tp * moe_ep == TP and moe_ep in (1, TP), (moe_tp, moe_ep)
        return Layout(rank, moe_tp, moe_ep, mtp, mtp_moe_tp=mtp, ts=True)

    def experts(self, mtp: bool = False) -> tuple[int, int, int, int]:
        """(first expert, experts, intermediate features per expert, this rank's first feature) of
        the main layers' (or the MTP layer's) routed experts on this rank."""
        if self.moe_ep == 1 or (mtp and self.mtp_moe_tp):
            il = MOE_I // TP
            return 0, E, il, self.rank * il
        n = E // self.moe_ep
        return self.rank * n, n, MOE_I, 0


def sf_bytes(rows: int, k: int) -> int:
    """Bytes of the NVFP4 scales of a [rows, k] operand in the CUTLASS SM100 layout."""
    r = (rows + 127) // 128 * 128
    nb = (k // 16 + 3) // 4 * 4
    return r * nb


def swizzle_sf(s: torch.Tensor) -> torch.Tensor:
    """Row-major e4m3 scales [rows, k/16] (one per 16 values) -> the CUTLASS SM100 K-major layout
    (512-byte atoms of 128 rows x 4 blocks: row m at (m % 32) * 16 + (m / 32 % 4) * 4, block kb
    at + kb % 4; atoms along K first). Rows padded to 128, blocks to 4, with zeros."""
    s = s.view(torch.uint8)
    rows, nb = s.shape
    R = (rows + 127) // 128 * 128
    NB = (nb + 3) // 4 * 4
    p = torch.zeros(R, NB, dtype=torch.uint8, device=s.device)
    p[:rows, :nb] = s
    return p.view(R // 128, 4, 32, NB // 4, 4).permute(0, 3, 2, 1, 4).reshape(-1)


def interleave64(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """[gate rows; up rows] interleaved by 64: block c = [gate 64c..64c+63 | up 64c..64c+63]."""
    n = gate.shape[0]
    assert up.shape == gate.shape and n % 64 == 0
    return torch.stack([gate.reshape(n // 64, 64, -1), up.reshape(n // 64, 64, -1)], 1).reshape(
        2 * n, *gate.shape[1:]
    )


def kda_layers(cfg) -> list[int]:
    """The KDA layers of the text config."""
    return [i for i, t in enumerate(cfg.layer_types) if t == "linear_attention"]


def mla_layers(cfg) -> list[int]:
    """The sparse MLA layers of the text config."""
    return [i for i, t in enumerate(cfg.layer_types) if t != "linear_attention"]


def param_specs(cfg, lay: Layout) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """name -> (per-rank shape, dtype) of every core parameter."""
    bf, f32, u8 = torch.bfloat16, torch.float32, torch.uint8
    p: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
        "embed": ((V, H), bf),
        "final_norm": ((H,), bf),
    }

    def mla(n: str, o_proj_all: bool) -> None:
        heads = MLA_HEADS if lay.ts else MLA_HL
        p.update(
            {
                f"{n}_qkv_a": ((QLORA + LORA, H), bf),
                f"{n}_q_a_norm": ((QLORA,), bf),
                f"{n}_kv_a_norm": ((LORA,), bf),
                # (token-split: every head's; this rank's are rows [16 r, 16 r + 16) of them)
                f"{n}_q_b": ((heads * NOPE, QLORA), bf),
                f"{n}_w_kc": ((heads, NOPE, LORA), bf),
                f"{n}_w_vc": ((heads, VH, LORA), bf),
                f"{n}_w_kcT": ((MLA_HL, LORA, NOPE), bf),
                f"{n}_o_proj": ((H, MLA_HL * VH), bf),
                f"{n}_idx_wq": ((IDX_HEADS * IDX_D, QLORA), bf),
                f"{n}_idx_kgw": ((2 * IDX_D + IDX_HEADS, H), bf),
                f"{n}_idx_kn_w": ((IDX_D,), bf),
                f"{n}_idx_kn_b": ((IDX_D,), bf),
                f"{n}_idx_ape": ((4, IDX_D), bf),
            }
        )
        if lay.ts and o_proj_all:
            p[f"{n}_o_proj_all"] = ((H, MLA_HEADS * VH), bf)

    def shared(n: str) -> None:
        p.update(
            {
                f"{n}_router": ((E, H), bf),
                f"{n}_bias": ((E,), f32),
                f"{n}_sh_gu": ((2 * SH_IL, H), bf),
                f"{n}_sh_dn": ((H, SH_IL), bf),
            }
        )

    _, ne, il, _ = lay.experts()
    for i in range(L):
        n = f"l{i}"
        for w in ("attn", "ffn"):
            p[f"{n}_hc_{w}_fn"] = ((HC_MIX, 4 * H), bf)
            p[f"{n}_hc_{w}_base"] = ((HC_MIX,), f32)
            p[f"{n}_hc_{w}_scale"] = ((3,), f32)
        p[f"{n}_in_norm"] = ((H,), bf)
        p[f"{n}_post_norm"] = ((H,), bf)
        if cfg.layer_types[i] == "linear_attention":
            p.update(
                {
                    f"{n}_in_proj": ((KDA_NIN, H), bf),
                    f"{n}_conv": ((3 * KDA_HD, 4), bf),
                    f"{n}_f_b": ((KDA_HD, KDA_D), bf),
                    f"{n}_g_b": ((KDA_HD, KDA_D), bf),
                    f"{n}_a_log": ((KDA_HL,), f32),
                    f"{n}_dt_bias": ((KDA_HD,), f32),
                    f"{n}_o_norm": ((KDA_D,), bf),
                    f"{n}_kda_o_proj": ((H, KDA_HD), bf),
                }
            )
        else:
            mla(n, o_proj_all=True)
        if cfg.mlp_layer_types[i] == "dense":
            p.update(
                {
                    f"{n}_gu_w": ((2 * INTER_L, H // 2), u8),
                    f"{n}_gu_sf": ((sf_bytes(2 * INTER_L, H),), u8),
                    f"{n}_dn_w": ((H, INTER_L // 2), u8),
                    f"{n}_dn_sf": ((sf_bytes(H, INTER_L),), u8),
                    f"{n}_alpha4": ((2,), f32),
                }
            )
        else:
            shared(n)
            p.update(
                {
                    f"{n}_w13": ((ne, 2 * il, H // 2), u8),
                    f"{n}_sf13": ((ne, sf_bytes(2 * il, H)), u8),
                    f"{n}_alpha13": ((ne,), f32),
                    f"{n}_w2": ((ne, H, il // 2), u8),
                    f"{n}_sf2": ((ne, sf_bytes(H, il)), u8),
                    f"{n}_alpha2": ((ne,), f32),
                }
            )
    if lay.mtp:
        n = "mtp"
        _, nm, ilm, _ = lay.experts(mtp=True)
        p.update(
            {
                f"{n}_enorm": ((H,), bf),
                f"{n}_hnorm": ((H,), bf),
                f"{n}_eh_proj": ((H, 2 * H), bf),
                f"{n}_in_norm": ((H,), bf),
                f"{n}_post_norm": ((H,), bf),
                f"{n}_head_norm": ((H,), bf),
                f"{n}_w13_16": ((nm, 2 * ilm, H), bf),
                f"{n}_w2_16": ((nm, H, ilm), bf),
            }
        )
        # (the MTP layer's o_proj is never token-split: its pass sums the heads by all-reduce)
        mla(n, o_proj_all=False)
        shared(n)
    return p


class _Reader:
    """Checkpoint access that records every key read (coverage). The runtime hands lazy
    safetensors slices: a rank reads only its shard (`rows` / `cols`)."""

    def __init__(self, weights):
        self.w = weights
        self.used: set[str] = set()

    def _get(self, key: str):
        self.used.add(key)
        return self.w[key]

    def __call__(self, key: str) -> torch.Tensor:
        t = self._get(key)
        return t if isinstance(t, torch.Tensor) else t[...]

    def rows(self, key: str, a: int, b: int) -> torch.Tensor:
        return self._get(key)[a:b]

    def cols(self, key: str, a: int, b: int) -> torch.Tensor:
        t = self._get(key)[:, a:b]
        return t.contiguous()

    def scalar(self, key: str) -> float:
        t = self(key)
        assert t.numel() == 1 and t.dtype == torch.float32, key
        return float(t.reshape(()).item())


def load(core, weights) -> None:
    """Fill the core's parameters (and NVFP4 scalars) from the checkpoint dict for this rank."""
    cfg = core.cfg
    lay: Layout = core.layout
    r = lay.rank
    w = core.w
    rd = _Reader(weights)
    filled: set[str] = set()
    dev = w["embed"].device

    def put(name: str, value: torch.Tensor) -> None:
        dst = w[name]
        value = value.to(dev)
        assert tuple(value.shape) == tuple(dst.shape), (name, tuple(value.shape), tuple(dst.shape))
        assert value.dtype == dst.dtype, (name, value.dtype, dst.dtype)
        dst.data.copy_(value)
        filled.add(name)

    def rows(key: str, i: int, n: int) -> torch.Tensor:
        return rd.rows(key, i * n, (i + 1) * n)

    def cols(key: str, i: int, n: int) -> torch.Tensor:
        return rd.cols(key, i * n, (i + 1) * n)

    def fp4(prefix: str, row_slice=None, col_slice=None):
        """(packed weight, row-major scales, weight_scale_2, input_scale) of an NVFP4 linear,
        rows / columns (in elements) sliced."""
        if row_slice is not None:
            wt = rd.rows(prefix + ".weight", row_slice.start, row_slice.stop)
            sc = rd.rows(prefix + ".weight_scale", row_slice.start, row_slice.stop)
        elif col_slice is not None:
            a, b = col_slice
            assert a % 64 == 0 and b % 64 == 0
            wt = rd.cols(prefix + ".weight", a // 2, b // 2)
            sc = rd.cols(prefix + ".weight_scale", a // 16, b // 16)
        else:
            wt, sc = rd(prefix + ".weight"), rd(prefix + ".weight_scale")
        sc = sc.view(torch.uint8)
        assert wt.dtype == torch.uint8
        # sliced on the host, re-laid on the device
        return (
            wt.contiguous().to(dev, non_blocking=True),
            sc.contiguous().to(dev, non_blocking=True),
            rd.scalar(prefix + ".weight_scale_2"),
            rd.scalar(prefix + ".input_scale"),
        )

    put("embed", rd(PRE + "embed_tokens.weight"))
    put("final_norm", rd(PRE + "norm.weight"))

    def load_mla(n: str, p: str, o_proj_all: bool) -> None:
        put(
            f"{n}_qkv_a",
            torch.cat([rd(p + "q_a_proj.weight"), rd(p + "kv_a_proj_with_mqa.weight")]),
        )
        put(f"{n}_q_a_norm", rd(p + "q_a_layernorm.weight"))
        put(f"{n}_kv_a_norm", rd(p + "kv_a_layernorm.weight"))
        h0, nh = (0, MLA_HEADS) if lay.ts else (r * MLA_HL, MLA_HL)
        put(f"{n}_q_b", rd.rows(p + "q_b_proj.weight", h0 * NOPE, (h0 + nh) * NOPE))
        kvb = rd.rows(p + "kv_b_proj.weight", h0 * (NOPE + VH), (h0 + nh) * (NOPE + VH))
        kvb = kvb.to(dev).view(nh, NOPE + VH, LORA)
        put(f"{n}_w_kc", kvb[:, :NOPE].contiguous())
        put(f"{n}_w_vc", kvb[:, NOPE:].contiguous())
        own = kvb[r * MLA_HL - h0 : (r + 1) * MLA_HL - h0, :NOPE]
        put(f"{n}_w_kcT", own.transpose(1, 2).contiguous())
        put(f"{n}_o_proj", cols(p + "o_proj.weight", r, MLA_HL * VH))
        if lay.ts and o_proj_all:
            put(f"{n}_o_proj_all", rd(p + "o_proj.weight"))
        ip = p + "indexer."
        put(f"{n}_idx_wq", rd(ip + "wq_b.weight"))
        put(
            f"{n}_idx_kgw",
            torch.cat(
                [
                    rd(ip + "wk.weight"),
                    rd(ip + "index_kpool_compress_gate"),
                    rd(ip + "weights_proj.weight"),
                ]
            ),
        )
        put(f"{n}_idx_kn_w", rd(ip + "k_norm.weight"))
        put(f"{n}_idx_kn_b", rd(ip + "k_norm.bias"))
        put(f"{n}_idx_ape", rd(ip + "index_kpool_compress_ape"))

    def load_shared(n: str, p: str) -> None:
        put(f"{n}_router", rd(p + "gate.weight"))
        put(f"{n}_bias", rd(p + "gate.e_score_correction_bias").float())
        g = rows(p + "shared_experts.gate_proj.weight", r, SH_IL)
        u = rows(p + "shared_experts.up_proj.weight", r, SH_IL)
        put(f"{n}_sh_gu", interleave64(g, u))
        put(f"{n}_sh_dn", cols(p + "shared_experts.down_proj.weight", r, SH_IL))

    e0, ne, il, f0 = lay.experts()
    sliced = il != MOE_I  # (tensor parallel: features [f0, f0 + il) of every expert)
    for i in range(L):
        n, p = f"l{i}", f"{PRE}layers.{i}."
        for wname in ("attn", "ffn"):
            put(f"{n}_hc_{wname}_fn", rd(p + f"hc_{wname}_fn"))
            put(f"{n}_hc_{wname}_base", rd(p + f"hc_{wname}_base").float())
            put(f"{n}_hc_{wname}_scale", rd(p + f"hc_{wname}_scale").float())
        put(f"{n}_in_norm", rd(p + "input_layernorm.weight"))
        put(f"{n}_post_norm", rd(p + "post_attention_layernorm.weight"))
        a = p + "self_attn."
        if cfg.layer_types[i] == "linear_attention":
            put(
                f"{n}_in_proj",
                torch.cat(
                    [
                        rows(a + "q_proj.weight", r, KDA_HD),
                        rows(a + "k_proj.weight", r, KDA_HD),
                        rows(a + "v_proj.weight", r, KDA_HD),
                        rows(a + "b_proj.weight", r, KDA_HL),
                        rd(a + "f_a_proj.weight"),
                        rd(a + "g_a_proj.weight"),
                    ]
                ),
            )
            conv = torch.cat([rows(a + f"{x}_conv1d.weight", r, KDA_HD) for x in "qkv"])
            conv = conv.reshape(3 * KDA_HD, 4)
            conv16 = conv.to(torch.bfloat16)
            assert torch.equal(conv16.float(), conv.float()), "KDA conv weights are bf16 values"
            put(f"{n}_conv", conv16)
            put(f"{n}_f_b", rows(a + "f_b_proj.weight", r, KDA_HD))
            put(f"{n}_g_b", rows(a + "g_b_proj.weight", r, KDA_HD))
            put(f"{n}_a_log", rows(a + "A_log", r, KDA_HL).float())
            put(f"{n}_dt_bias", rows(a + "dt_bias", r, KDA_HD).float())
            put(f"{n}_o_norm", rd(a + "o_norm.weight"))
            put(f"{n}_kda_o_proj", cols(a + "o_proj.weight", r, KDA_HD))
        else:
            load_mla(n, a, o_proj_all=True)
        m = p + "mlp."
        if cfg.mlp_layer_types[i] == "dense":
            sl = slice(r * INTER_L, (r + 1) * INTER_L)
            gw, gs, g2, gi = fp4(m + "gate_proj", row_slice=sl)
            uw, us, u2, ui = fp4(m + "up_proj", row_slice=sl)
            assert (g2, gi) == (u2, ui), f"{m}: gate / up NVFP4 global scales differ"
            put(f"{n}_gu_w", interleave64(gw, uw))
            put(f"{n}_gu_sf", swizzle_sf(interleave64(gs, us)))
            dw, ds, d2, di = fp4(m + "down_proj", col_slice=(r * INTER_L, (r + 1) * INTER_L))
            put(f"{n}_dn_w", dw)
            put(f"{n}_dn_sf", swizzle_sf(ds))
            core.scalars[f"{n}_gu_alpha"] = gi * g2
            core.scalars[f"{n}_gu_gs"] = gi
            core.scalars[f"{n}_dn_alpha"] = di * d2
            core.scalars[f"{n}_dn_gs"] = di
            put(f"{n}_alpha4", torch.tensor([gi * g2, di * d2], dtype=torch.float32))
        else:
            load_shared(n, m)
            w13, sf13, w2, sf2, a13, a2 = [], [], [], [], [], []
            gs13 = gs2 = None
            for e in range(e0, e0 + ne):
                q = f"{m}experts.{e}."
                if sliced:
                    fs = slice(f0, f0 + il)
                    gw, gs, g2, gi = fp4(q + "gate_proj", row_slice=fs)
                    uw, us, u2, ui = fp4(q + "up_proj", row_slice=fs)
                    dw, ds, d2, di = fp4(q + "down_proj", col_slice=(f0, f0 + il))
                else:
                    gw, gs, g2, gi = fp4(q + "gate_proj")
                    uw, us, u2, ui = fp4(q + "up_proj")
                    dw, ds, d2, di = fp4(q + "down_proj")
                assert (g2, gi) == (u2, ui), f"{q}: gate / up NVFP4 global scales differ"
                gs13 = gi if gs13 is None else gs13
                gs2 = di if gs2 is None else gs2
                assert gi == gs13 and di == gs2, f"{q}: per-expert input scales differ"
                w13.append(interleave64(gw, uw))
                sf13.append(swizzle_sf(interleave64(gs, us)))
                w2.append(dw)
                sf2.append(swizzle_sf(ds))
                a13.append(gi * g2)
                a2.append(di * d2)
            put(f"{n}_w13", torch.stack(w13))
            put(f"{n}_sf13", torch.stack(sf13))
            put(f"{n}_w2", torch.stack(w2))
            put(f"{n}_sf2", torch.stack(sf2))
            put(f"{n}_alpha13", torch.tensor(a13, dtype=torch.float32))
            put(f"{n}_alpha2", torch.tensor(a2, dtype=torch.float32))
            core.scalars[f"{n}_gs13"] = gs13
            core.scalars[f"{n}_gs2"] = gs2
        if dev.type == "cuda":
            torch.cuda.empty_cache()

    if lay.mtp:
        n, p = "mtp", f"{PRE}layers.{L}."
        put(f"{n}_enorm", rd(p + "enorm.weight"))
        put(f"{n}_hnorm", rd(p + "hnorm.weight"))
        put(f"{n}_eh_proj", rd(p + "eh_proj.weight"))
        put(f"{n}_in_norm", rd(p + "input_layernorm.weight"))
        put(f"{n}_post_norm", rd(p + "post_attention_layernorm.weight"))
        put(f"{n}_head_norm", rd(p + "shared_head.norm.weight"))
        load_mla(n, p + "self_attn.", o_proj_all=False)
        m = p + "mlp."
        load_shared(n, m)
        m0, nm, ilm, fm = lay.experts(mtp=True)
        w13, w2 = [], []
        for e in range(m0, m0 + nm):
            q = f"{m}experts.{e}."
            if ilm != MOE_I:
                g = rd.rows(q + "gate_proj.weight", fm, fm + ilm)
                u = rd.rows(q + "up_proj.weight", fm, fm + ilm)
                d = rd.cols(q + "down_proj.weight", fm, fm + ilm)
            else:
                g, u, d = (
                    rd(q + "gate_proj.weight"),
                    rd(q + "up_proj.weight"),
                    rd(q + "down_proj.weight"),
                )
            w13.append(interleave64(g.to(dev, non_blocking=True), u.to(dev, non_blocking=True)))
            w2.append(d.to(dev, non_blocking=True))
        put(f"{n}_w13_16", torch.stack(w13))
        put(f"{n}_w2_16", torch.stack(w2))

    # coverage: every declared parameter filled; every checkpoint key read except the predicted
    # leftovers (other ranks' experts, the vision tower, the MTP layer without MTP)
    declared = set(w.keys())
    assert filled == declared, (sorted(declared - filled)[:8], sorted(filled - declared)[:8])
    leftover = set(weights.keys()) - rd.used
    predicted = set()
    for k in leftover:
        if k.startswith("model.visual.") or "lm_head" in k:
            predicted.add(k)
        elif f"{PRE}layers.{L}." in k and not lay.mtp:
            predicted.add(k)
        elif ".mlp.experts." in k:
            e = int(k.split(".mlp.experts.")[1].split(".")[0])
            a, cnt, _, _ = lay.experts(mtp=f"{PRE}layers.{L}." in k)
            if not a <= e < a + cnt:
                predicted.add(k)
    assert leftover == predicted, sorted(leftover - predicted)[:16]


def load_lm_head(model, weights) -> None:
    """The vocabulary-parallel lm_head: this rank's contiguous vocabulary rows."""
    dst = model.lm_head.weight
    v = V // TP
    r = model.model.layout.rank
    src = weights["lm_head.weight"][r * v : (r + 1) * v]  # (lazy slice: this rank's rows only)
    assert tuple(dst.shape) == tuple(src.shape), (dst.shape, src.shape)
    dst.data.copy_(src.to(dst.device, dst.dtype))
