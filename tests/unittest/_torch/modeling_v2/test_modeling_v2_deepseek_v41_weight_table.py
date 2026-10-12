# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DeepSeek-V4.1's weight tables declare and source what the hand-written code did.

The two references below are transcriptions of what the target stated before it
moved onto ``ModelWeights``: the declaration block of the core's ``__init__`` and
the hand-written manifests (trunk and DSpark drafter), whose sources were proven on
GB300 by the target's gates. They are copies rather than imports on purpose -- the
point is to hold the tables to what the model used to say. One renaming was
deliberate: the drafter's rows moved from the trunk's ``ds{s}_*`` / ``ds_*`` names
into their own table as stage ``l{s}_*`` and plain ``*``.

Every parallel member is checked at a non-zero rank, so a slice that ignores its
rank cannot pass, along with a two-stage pipeline split and both Engram residencies.

No GPU: nothing here allocates (meta tensors), and ``dims`` reads ``model_config``
alone.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

__extra_import_path__ = [".."]  # noqa: F841 -- repo's file-scoped import hook

from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified import (  # noqa: E501
    weights as W,
)

# DeepSeek-V4.1-Flash's own text_config numbers, so the comparison is against the
# shapes this target actually ships rather than a toy configuration.
_TC = dict(
    num_hidden_layers=40,
    hidden_size=5120,
    vocab_size=129280,
    num_attention_heads=64,
    head_dim=512,
    q_lora_rank=1280,
    o_groups=8,
    o_lora_rank=1024,
    n_routed_experts=384,
    moe_intermediate_size=2304,
    compress_ratios=[0, 0] + [2] * 18 + [1] * 20 + [0] * 3,
    kv_source_layer_ids=[2, 8, 14, 20],
    index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
    engram_layer_ids=[1, 14],
    engram_num_embeddings=[384006168, 384016682],
    engram_max_ngram_size=4,
    engram_n_heads=8,
    engram_head_dim=256,
    num_nextn_predict_layers=3,
    dspark_n_routed_experts=128,
    dspark_markov_rank=256,
    dspark_target_layer_ids=[37, 38, 39],
)

_HOST_ENV = "TRTLLM_DSV41_ENGRAM_HOST_OFFLOAD"

# (tp, pp_rank, pp, ep, moe_tp, attention_dp, tp_rank): every member at a non-zero rank.
_MEMBERS = {
    "tep4": (4, 0, 1, 4, 1, False, 1),
    "dep4": (4, 0, 1, 4, 1, True, 2),
    "tp4": (4, 0, 1, 1, 4, False, 3),
    "tep8": (8, 0, 1, 8, 1, False, 5),
    "pp2xtep4-stage1": (4, 1, 2, 4, 1, False, 2),
}


def _mapping(tp, pp_rank, pp, ep, moe_tp, dp, tp_rank):
    n = _TC["num_hidden_layers"]
    per = n // pp
    return SimpleNamespace(
        tp_size=tp,
        tp_rank=tp_rank,
        pp_size=pp,
        pp_rank=pp_rank,
        enable_attention_dp=dp,
        moe_ep_size=ep,
        moe_ep_rank=tp_rank if ep > 1 else 0,
        moe_tp_size=moe_tp,
        moe_tp_rank=tp_rank if moe_tp > 1 else 0,
        pp_layers=lambda num_layers: list(range(pp_rank * per, (pp_rank + 1) * per)),
        is_last_pp_rank=lambda: pp_rank == pp - 1,
    )


def _core(member: str):
    tc = SimpleNamespace(**_TC)
    return SimpleNamespace(
        model_config=SimpleNamespace(
            pretrained_config=SimpleNamespace(text_config=tc),
            mapping=_mapping(*_MEMBERS[member]),
            torch_dtype=torch.bfloat16,
        )
    )


def _reference_core(member: str, host: bool) -> SimpleNamespace:
    """The partition the core's ``__init__`` derived, transcribed."""
    tp, pp_rank, pp, ep, moe_tp, dp, tp_rank = _MEMBERS[member]
    c = SimpleNamespace(**_TC)
    attn_tp = 1 if dp else tp
    attn_rank = 0 if dp else tp_rank
    r = SimpleNamespace()
    r.hidden, r.vocab, r.head_dim, r.q_lora = c.hidden_size, c.vocab_size, c.head_dim, c.q_lora_rank
    r.o_lora, r.num_experts, r.moe_inter = (
        c.o_lora_rank,
        c.n_routed_experts,
        c.moe_intermediate_size,
    )
    r.h_loc = c.num_attention_heads // attn_tp
    r.g_loc = c.o_groups // attn_tp
    r.attn_rank = attn_rank
    r.group_in = c.num_attention_heads * c.head_dim // c.o_groups
    if ep == tp:
        r.e_loc = c.n_routed_experts // tp
        r.e_off = r.e_loc * tp_rank
        r.i_loc, r.i_off = c.moe_intermediate_size, 0
    else:
        r.e_loc, r.e_off = c.n_routed_experts, 0
        r.i_loc = c.moe_intermediate_size // tp
        r.i_off = r.i_loc * tp_rank
    per = c.num_hidden_layers // pp
    r.pp_layer_ids = list(range(pp_rank * per, (pp_rank + 1) * per))
    r.local = frozenset(r.pp_layer_ids)
    r.pp_prev = r.pp_layer_ids[0] - 1 if r.pp_layer_ids[0] > 0 else None
    r.ratios = c.compress_ratios[: c.num_hidden_layers]
    r.kv_sources = tuple(c.kv_source_layer_ids)
    r.index_sources = tuple(c.index_source_layer_ids)
    r.engram_layers = tuple(c.engram_layer_ids)
    r.eng_rows = list(c.engram_num_embeddings)
    r.eng_part = [(n + tp - 1) // tp for n in r.eng_rows]
    r.eng_start = [p * tp_rank for p in r.eng_part]
    r.eng_end = [min(s + p, n) for s, p, n in zip(r.eng_start, r.eng_part, r.eng_rows)]
    r.eng_host = host
    if host:
        r.eng_start, r.eng_end = [0] * len(r.eng_rows), list(r.eng_rows)
    r.eng_dim = c.engram_head_dim
    r.eng_cols = (c.engram_max_ngram_size - 1) * c.engram_n_heads
    r.eng_in = r.eng_cols * r.eng_dim
    r.eng_wkv_n = 5 * c.hidden_size
    r.hc_k = 4 * c.hidden_size
    return r


def _mx_sf(n, k):
    return ((n + 127) // 128 * 128) * ((k // 32 + 3) // 4 * 4)


def _reference_declaration(member: str, host: bool) -> dict:
    """The trunk + vision declaration block of the core's ``__init__``, transcribed:
    name -> (shape, dtype)."""
    r = _reference_core(member, host)
    bf, f8, u8, f32 = torch.bfloat16, torch.float8_e4m3fn, torch.uint8, torch.float32
    H, mix = r.hidden, 24
    w = {"embed": ((r.vocab, H), bf), "final_norm": ((H,), bf)}
    for i in range(40):
        if i not in r.local:
            if i == r.pp_prev:
                w[f"l{i}_hc_ffn_fn"] = ((mix, r.hc_k), f32)
                w[f"l{i}_hc_ffn_base"] = ((mix,), f32)
                w[f"l{i}_hc_ffn_scale"] = ((3,), f32)
            continue
        for s in ("attn", "ffn"):
            w[f"l{i}_hc_{s}_fn"] = ((mix, r.hc_k), f32)
            w[f"l{i}_hc_{s}_base"] = ((mix,), f32)
            w[f"l{i}_hc_{s}_scale"] = ((3,), f32)
        w[f"l{i}_attn_norm"] = ((H,), bf)
        w[f"l{i}_ffn_norm"] = ((H,), bf)
        w[f"l{i}_wqkv_a"] = ((r.q_lora + r.head_dim, H), f8)
        w[f"l{i}_wqkv_a_sf"] = ((_mx_sf(r.q_lora + r.head_dim, H),), u8)
        w[f"l{i}_q_norm"] = ((r.q_lora,), bf)
        w[f"l{i}_kv_norm"] = ((r.head_dim,), bf)
        w[f"l{i}_wq_b"] = ((r.h_loc * r.head_dim, r.q_lora), f8)
        w[f"l{i}_wq_b_sf"] = ((_mx_sf(r.h_loc * r.head_dim, r.q_lora),), u8)
        w[f"l{i}_sink"] = ((r.h_loc,), f32)
        w[f"l{i}_wo_a"] = ((r.g_loc * r.o_lora, r.group_in), bf)
        w[f"l{i}_wo_b"] = ((H, r.g_loc * r.o_lora), f8)
        w[f"l{i}_wo_b_sf"] = ((_mx_sf(H, r.g_loc * r.o_lora),), u8)
        if i in r.kv_sources:
            if r.ratios[i] == 2:
                w[f"l{i}_comp_w"] = ((2 * r.head_dim, H), f32)
            else:
                w[f"l{i}_comp_w"] = ((r.head_dim, H), bf)
            w[f"l{i}_comp_norm"] = ((r.head_dim,), bf)
            w[f"l{i}_idx_wk"] = ((128, r.head_dim), bf)
            w[f"l{i}_idx_k_norm"] = ((128,), bf)
        if i in r.index_sources:
            w[f"l{i}_idx_wq_b"] = ((32 * 128, r.q_lora), f8)
            w[f"l{i}_idx_wq_b_sf"] = ((_mx_sf(32 * 128, r.q_lora),), u8)
            w[f"l{i}_idx_wproj"] = ((32, H), bf)
        w[f"l{i}_gate_w"] = ((r.num_experts, H), bf)
        w[f"l{i}_gate_bias"] = ((r.num_experts,), f32)
        w[f"l{i}_gate_bias_vl"] = ((r.num_experts,), f32)
        w[f"l{i}_fc1_w"] = ((r.e_loc, 2 * r.i_loc, H // 2), u8)
        w[f"l{i}_fc1_s"] = ((r.e_loc, 2 * r.i_loc, H // 16), u8)
        w[f"l{i}_fc2_w"] = ((r.e_loc, H, r.i_loc // 2), u8)
        w[f"l{i}_fc2_s"] = ((r.e_loc, H, r.i_loc // 16), u8)
        for name in ("isc1", "isc3", "isc2", "ws1", "ws3", "ws2"):
            w[f"l{i}_e_{name}"] = ((r.num_experts,), f32)
        w[f"l{i}_sh_w13"] = ((2 * r.moe_inter, H), f8)
        w[f"l{i}_sh_w13_sf"] = ((_mx_sf(2 * r.moe_inter, H),), u8)
        w[f"l{i}_sh_w2"] = ((H, r.moe_inter), f8)
        w[f"l{i}_sh_w2_sf"] = ((_mx_sf(H, r.moe_inter),), u8)
        if i in r.engram_layers:
            li = r.engram_layers.index(i)
            if not host:
                w[f"l{i}_eng_table"] = ((r.eng_part[li], r.eng_dim), f8)
                w[f"l{i}_eng_scale"] = ((r.eng_part[li], r.eng_dim // 32), u8)
            w[f"l{i}_eng_wkv"] = ((r.eng_wkv_n, r.eng_in), f8)
            w[f"l{i}_eng_wkv_sf"] = ((_mx_sf(r.eng_wkv_n, r.eng_in),), u8)
            w[f"l{i}_eng_q"] = ((4, H), bf)
            w[f"l{i}_eng_k"] = ((4, H), bf)
    D, inter = 1024, 2816
    w.update(
        vis_patch_w=((D, 588), bf),
        vis_patch_b=((D,), bf),
        vis_norm=((D,), bf),
        vis_al_w1=((H, D * 9), bf),
        vis_al_b1=((H,), bf),
        vis_al_w2=((H, H), bf),
        vis_al_b2=((H,), bf),
        vis_span_tail=((3, H), bf),
    )
    for L in range(32):
        p = f"vis{L}_"
        w[p + "norm1"] = ((D,), bf)
        w[p + "wqkv_w"] = ((3 * D, D), bf)
        w[p + "wqkv_b"] = ((3 * D,), bf)
        w[p + "wo_w"] = ((D, D), bf)
        w[p + "wo_b"] = ((D,), bf)
        w[p + "norm2"] = ((D,), bf)
        w[p + "w1"] = ((2 * inter, D), bf)
        w[p + "w2"] = ((D, inter), bf)
    return w


def _reference_draft_declaration() -> dict:
    """The drafter's parameter specs, transcribed, under the table's names."""
    f8, u8, f32, bf = torch.float8_e4m3fn, torch.uint8, torch.float32, torch.bfloat16
    H, hd, ql, inter, E, heads, og, ol = 5120, 512, 1280, 2304, 128, 64, 8, 1024
    out = {}
    for s in range(3):
        p = f"l{s}_"
        for x in ("attn", "ffn"):
            out[p + f"hc_{x}_fn"] = ((24, 4 * H), f32)
            out[p + f"hc_{x}_base"] = ((24,), f32)
            out[p + f"hc_{x}_scale"] = ((3,), f32)
        out[p + "attn_norm"] = ((H,), bf)
        out[p + "ffn_norm"] = ((H,), bf)
        out[p + "wqkv_a"] = ((ql + hd, H), f8)
        out[p + "wqkv_a_sf"] = ((_mx_sf(ql + hd, H),), u8)
        out[p + "q_norm"] = ((ql,), bf)
        out[p + "kv_norm"] = ((hd,), bf)
        out[p + "wq_b"] = ((heads * hd, ql), f8)
        out[p + "wq_b_sf"] = ((_mx_sf(heads * hd, ql),), u8)
        out[p + "sink"] = ((heads,), f32)
        out[p + "wo_a"] = ((og * ol, heads * hd // og), bf)
        out[p + "wo_b"] = ((H, og * ol), f8)
        out[p + "wo_b_sf"] = ((_mx_sf(H, og * ol),), u8)
        out[p + "gate_w"] = ((E, H), f32)
        out[p + "gate_bias"] = ((E,), f32)
        out[p + "fc1_w"] = ((E, 2 * inter, H // 2), u8)
        out[p + "fc1_s"] = ((E, 2 * inter, H // 32), u8)
        out[p + "fc2_w"] = ((E, H, inter // 2), u8)
        out[p + "fc2_s"] = ((E, H, inter // 32), u8)
        out[p + "sh_w13"] = ((2 * inter, H), f8)
        out[p + "sh_w13_sf"] = ((_mx_sf(2 * inter, H),), u8)
        out[p + "sh_w2"] = ((H, inter), f8)
        out[p + "sh_w2_sf"] = ((_mx_sf(H, inter),), u8)
    out["main_proj"] = ((H, 3 * H), f8)
    out["main_proj_sf"] = ((_mx_sf(H, 3 * H),), u8)
    out["main_norm"] = ((H,), bf)
    out["wkv3"] = ((3 * hd, H), f8)
    out["wkv3_sf"] = ((_mx_sf(3 * hd, H),), u8)
    out["norm"] = ((H,), bf)
    out["mk_embed"] = ((129280, 256), bf)
    out["mk_head"] = ((129280, 256), f32)
    return out


def _reference_manifest(member: str, host: bool) -> dict:
    """The hand-written trunk manifest, transcribed: param -> [(keys, srcs, dst,
    has_transform)]. Host-resident Engram rows are no parameter and are left out."""
    r = _reference_core(member, host)
    rows = {}

    def row(key, src=None, dst=None, tf=False):
        keys = key if isinstance(key, tuple) else (key,)
        if src is None:
            srcs = (None,) * len(keys)
        elif isinstance(key, tuple):
            srcs = src
        else:
            srcs = (src,)
        return (keys, srcs, dst, tf)

    rows["embed"] = [row("embed.weight")]
    rows["final_norm"] = [row("norm.weight")]
    h0 = r.attn_rank * r.h_loc
    q_rows = slice(h0 * r.head_dim, (h0 + r.h_loc) * r.head_dim)
    q_sf_rows = slice(q_rows.start // 32, q_rows.stop // 32)
    g0 = r.attn_rank * r.g_loc
    o_rows = slice(g0 * r.o_lora, (g0 + r.g_loc) * r.o_lora)
    o_sf_rows = slice(o_rows.start // 32, o_rows.stop // 32)
    all_ = slice(None)
    for i in range(40):
        p = f"layers.{i}"
        if i not in r.local:
            if i == r.pp_prev:
                for f in ("fn", "base", "scale"):
                    rows[f"l{i}_hc_ffn_{f}"] = [row(f"{p}.hc_ffn_{f}")]
            continue
        for s in ("attn", "ffn"):
            for f in ("fn", "base", "scale"):
                rows[f"l{i}_hc_{s}_{f}"] = [row(f"{p}.hc_{s}_{f}")]
        rows[f"l{i}_attn_norm"] = [row(f"{p}.attn_norm.weight")]
        rows[f"l{i}_ffn_norm"] = [row(f"{p}.ffn_norm.weight")]
        a = f"{p}.attn"
        rows[f"l{i}_wqkv_a"] = [
            row(f"{a}.wq_a.weight", None, slice(0, r.q_lora)),
            row(f"{a}.wkv.weight", None, slice(r.q_lora, r.q_lora + r.head_dim)),
        ]
        rows[f"l{i}_wqkv_a_sf"] = [row((f"{a}.wq_a.scale", f"{a}.wkv.scale"), None, None, True)]
        rows[f"l{i}_q_norm"] = [row(f"{a}.q_norm.weight")]
        rows[f"l{i}_kv_norm"] = [row(f"{a}.kv_norm.weight")]
        rows[f"l{i}_wq_b"] = [row(f"{a}.wq_b.weight", (q_rows,))]
        rows[f"l{i}_wq_b_sf"] = [row(f"{a}.wq_b.scale", (q_sf_rows,), None, True)]
        rows[f"l{i}_sink"] = [row(f"{a}.attn_sink", (slice(h0, h0 + r.h_loc),))]
        rows[f"l{i}_wo_a"] = [
            row((f"{a}.wo_a.weight", f"{a}.wo_a.scale"), ((o_rows,), (o_sf_rows,)), None, True)
        ]
        rows[f"l{i}_wo_b"] = [row(f"{a}.wo_b.weight", (all_, o_rows))]
        rows[f"l{i}_wo_b_sf"] = [row(f"{a}.wo_b.scale", (all_, o_sf_rows), None, True)]
        if i in r.kv_sources:
            c = f"{a}.compressor"
            if r.ratios[i] == 2:
                rows[f"l{i}_comp_w"] = [
                    row((f"{c}.wkv.weight", f"{c}.wgate.weight"), None, None, True)
                ]
            else:
                rows[f"l{i}_comp_w"] = [row(f"{c}.wkv.weight")]
            rows[f"l{i}_comp_norm"] = [row(f"{c}.norm.weight")]
            rows[f"l{i}_idx_wk"] = [row(f"{a}.indexer.wk.weight")]
            rows[f"l{i}_idx_k_norm"] = [row(f"{a}.indexer.k_norm.weight")]
        if i in r.index_sources:
            x = f"{a}.indexer"
            rows[f"l{i}_idx_wq_b"] = [row(f"{x}.wq_b.weight")]
            rows[f"l{i}_idx_wq_b_sf"] = [row(f"{x}.wq_b.scale", None, None, True)]
            rows[f"l{i}_idx_wproj"] = [row(f"{x}.weights_proj.weight", None, None, True)]
        f_ = f"{p}.ffn"
        rows[f"l{i}_gate_w"] = [row(f"{f_}.gate.weight")]
        rows[f"l{i}_gate_bias"] = [row(f"{f_}.gate.bias")]
        rows[f"l{i}_gate_bias_vl"] = [row(f"{f_}.gate.bias_vl")]
        up_rows = (slice(r.i_off, r.i_off + r.i_loc),)
        dn_w_cols = (all_, slice(r.i_off // 2, (r.i_off + r.i_loc) // 2))
        dn_s_cols = (all_, slice(r.i_off // 16, (r.i_off + r.i_loc) // 16))
        fc1_w, fc1_s, fc2_w, fc2_s = [], [], [], []
        for local in range(r.e_loc):
            q = f"{f_}.experts.{r.e_off + local}"
            fc1_w.append(
                row((f"{q}.w3.weight", f"{q}.w1.weight"), (up_rows, up_rows), (local,), True)
            )
            fc1_s.append(
                row(
                    (f"{q}.w3.weight_scale", f"{q}.w1.weight_scale"),
                    (up_rows, up_rows),
                    (local,),
                    True,
                )
            )
            fc2_w.append(row(f"{q}.w2.weight", dn_w_cols, (local,), True))
            fc2_s.append(row(f"{q}.w2.weight_scale", dn_s_cols, (local,), True))
        rows[f"l{i}_fc1_w"] = fc1_w
        rows[f"l{i}_fc1_s"] = fc1_s
        rows[f"l{i}_fc2_w"] = fc2_w
        rows[f"l{i}_fc2_s"] = fc2_s
        for name, (w_, s_) in {
            "isc1": ("w1", "input_scale"),
            "isc3": ("w3", "input_scale"),
            "isc2": ("w2", "input_scale"),
            "ws1": ("w1", "weight_scale_2"),
            "ws3": ("w3", "weight_scale_2"),
            "ws2": ("w2", "weight_scale_2"),
        }.items():
            rows[f"l{i}_e_{name}"] = [
                row(f"{f_}.experts.{e}.{w_}.{s_}", None, (e,)) for e in range(r.num_experts)
            ]
        sh = f"{f_}.shared_experts"
        rows[f"l{i}_sh_w13"] = [
            row(f"{sh}.w1.weight", None, slice(0, r.moe_inter)),
            row(f"{sh}.w3.weight", None, slice(r.moe_inter, 2 * r.moe_inter)),
        ]
        rows[f"l{i}_sh_w13_sf"] = [row((f"{sh}.w1.scale", f"{sh}.w3.scale"), None, None, True)]
        rows[f"l{i}_sh_w2"] = [row(f"{sh}.w2.weight")]
        rows[f"l{i}_sh_w2_sf"] = [row(f"{sh}.w2.scale", None, None, True)]
        if i in r.engram_layers:
            li = r.engram_layers.index(i)
            e = f"{p}.engram"
            if not host:
                n = r.eng_end[li] - r.eng_start[li]
                shard = (slice(r.eng_start[li], r.eng_end[li]),)
                rows[f"l{i}_eng_table"] = [row(f"{e}.embed.weight", shard, slice(0, n))]
                rows[f"l{i}_eng_scale"] = [row(f"{e}.embed.scale", shard, slice(0, n), True)]
            rows[f"l{i}_eng_wkv"] = [row(f"{e}.wkv.weight")]
            rows[f"l{i}_eng_wkv_sf"] = [row(f"{e}.wkv.scale", None, None, True)]
            rows[f"l{i}_eng_q"] = [row(f"{e}.q_weight")]
            rows[f"l{i}_eng_k"] = [row(f"{e}.k_weight")]
    rows.update(
        vis_patch_w=[row("vision.patch_embed.proj.weight")],
        vis_patch_b=[row("vision.patch_embed.proj.bias")],
        vis_norm=[row("vision.norm.weight")],
        vis_al_w1=[row("aligner.w1.weight", None, None, True)],
        vis_al_b1=[row("aligner.w1.bias")],
        vis_al_w2=[row("aligner.w2.weight")],
        vis_al_b2=[row("aligner.w2.bias")],
        vis_span_tail=[row(("image_start", "image_end", "image_newline"), None, None, True)],
    )
    for L in range(32):
        p, v = f"vis{L}_", f"vision.blocks.{L}."
        rows[p + "norm1"] = [row(v + "norm1.weight")]
        rows[p + "wqkv_w"] = [row(v + "attn.wqkv.weight")]
        rows[p + "wqkv_b"] = [row(v + "attn.wqkv.bias")]
        rows[p + "wo_w"] = [row(v + "attn.wo.weight")]
        rows[p + "wo_b"] = [row(v + "attn.wo.bias")]
        rows[p + "norm2"] = [row(v + "norm2.weight")]
        rows[p + "w1"] = [row(v + "mlp.w1.weight")]
        rows[p + "w2"] = [row(v + "mlp.w2.weight")]
    return rows


def _normalize(manifest: dict) -> dict:
    """The table's manifest in the reference's row form."""
    out = {}
    for key, sources in manifest.items():
        rows = []
        for ckpt, index, transform in sources:
            keys, srcs, _ = W._split_source(ckpt)
            rows.append((keys, srcs, index, transform is not None))
        out[key] = rows
    return out


# Every member with the Engram tables in the node's host copy (the default), and the
# device row shard on the members whose shard boundaries differ.
_CASES = [(m, True) for m in sorted(_MEMBERS)] + [("tep4", False), ("tep8", False)]
_IDS = [f"{m}-{'engram_host' if h else 'engram_device'}" for m, h in _CASES]


def _residency(monkeypatch, host: bool) -> None:
    monkeypatch.setenv(_HOST_ENV, "1" if host else "0")


@pytest.mark.parametrize("member,host", _CASES, ids=_IDS)
def test_the_trunk_table_declares_what_the_init_block_did(member, host, monkeypatch):
    """Shape and dtype per key, so a failure names the weight that drifted."""
    _residency(monkeypatch, host)
    d = W.MODEL_WEIGHTS.dims(_core(member))
    with torch.device("meta"):
        got = W.MODEL_WEIGHTS.declare(d)
    want = _reference_declaration(member, host)
    assert set(got.keys()) == set(want.keys()), sorted(set(got.keys()) ^ set(want.keys()))[:8]
    for key in sorted(want):
        assert (tuple(got[key].shape), got[key].dtype) == want[key], key


@pytest.mark.parametrize("member,host", _CASES, ids=_IDS)
def test_the_trunk_table_sources_what_the_manifest_did(member, host, monkeypatch):
    """Every parameter's checkpoint keys, this rank's part of each, the destination
    index and whether a transform runs -- per row, in order."""
    _residency(monkeypatch, host)
    got = _normalize(W.MODEL_WEIGHTS.manifest(_core(member)))
    want = _reference_manifest(member, host)
    assert set(got) == set(want), sorted(set(got) ^ set(want))[:8]
    for key in sorted(want):
        assert got[key] == want[key], key


@pytest.mark.parametrize("member,host", _CASES, ids=_IDS)
def test_dims_carries_the_partition_the_core_derived(member, host, monkeypatch):
    _residency(monkeypatch, host)
    d = W.MODEL_WEIGHTS.dims(_core(member))
    r = _reference_core(member, host)
    for name in (
        "h_loc",
        "g_loc",
        "attn_rank",
        "e_loc",
        "e_off",
        "i_loc",
        "i_off",
        "pp_layer_ids",
        "local",
        "pp_prev",
        "eng_start",
        "eng_end",
        "eng_host",
    ):
        assert getattr(d, name) == getattr(r, name), name


def test_the_draft_table_declares_the_drafter():
    d = W.DRAFT_WEIGHTS.dims(_core("tep4"))
    with torch.device("meta"):
        got = W.DRAFT_WEIGHTS.declare(d)
    want = _reference_draft_declaration()
    assert set(got.keys()) == set(want.keys()), sorted(set(got.keys()) ^ set(want.keys()))[:8]
    for key in sorted(want):
        assert (tuple(got[key].shape), got[key].dtype) == want[key], key


def test_the_draft_table_reads_every_stage_from_mtp():
    """Every drafter source is an ``mtp.*`` key, each stage reads its own, and the
    three-stage ring operand stacks the stages' ``wkv`` in order."""
    manifest = W.DRAFT_WEIGHTS.manifest(_core("tep4"))
    for key, sources in manifest.items():
        assert sources, key
        for ckpt, _, _ in sources:
            keys, _, _ = W._split_source(ckpt)
            assert all(k.startswith("mtp.") for k in keys), (key, keys)
            if key.startswith("l") and key[1].isdigit():
                stage = key[1 : key.index("_")]
                assert all(k.startswith(f"mtp.{stage}.") for k in keys), (key, keys)
    wkv3 = manifest["wkv3"]
    assert [ckpt for ckpt, _, _ in wkv3] == [f"mtp.{s}.attn.wkv.weight" for s in range(3)]
    assert [index for _, index, _ in wkv3] == [slice(s * 512, (s + 1) * 512) for s in range(3)]


def test_every_declared_weight_has_a_source():
    """A declared weight with no source allocates memory nothing ever fills."""
    for key, sources in W.MODEL_WEIGHTS.manifest(_core("tep4")).items():
        assert sources, key


def test_the_engram_vocabulary_reads_either_config_form():
    """The checkpoint's integer, or a per-n-gram list of the same bucket size."""
    assert W._engram_vocab_size(SimpleNamespace(engram_vocab_size=16000000)) == 16000000
    assert (
        W._engram_vocab_size(
            SimpleNamespace(engram_vocab_size=[16000000] * 3, engram_bucket_vocab_size=16000000)
        )
        == 16000000
    )
    assert W._engram_vocab_size(SimpleNamespace(engram_vocab_size=[16000000] * 3)) == 16000000
