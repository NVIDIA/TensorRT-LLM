# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""qwen4_exp's weight table declares what the hand-written code did.

The declaration reference below is a transcription of the `__init__` block the table
replaced, keyed the way the table keys it: vision block `b`'s weights are `l{b}_vis_*`
and the MTP module's are `l{num_hidden_layers}_mtp_*`, its layer index in the checkpoint
and the KV pool. The n-gram table is not in either: its height is a checkpoint fact,
declared at load.

No GPU: nothing here allocates, and `dims` reads the configs alone.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

__extra_import_path__ = ["../../.."]  # noqa: F841 -- repo's file-scoped import hook

from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1 import (  # noqa: E501
    modeling as M,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_103__tp1 import (  # noqa: E501
    weights as W,
)

pytestmark = pytest.mark.cpu_only

# Qwen3.8-Flash-Next's own numbers, so the comparison is against the shapes this target
# actually ships rather than a toy configuration.
_TEXT = dict(
    num_hidden_layers=48,
    layer_types=["full_attention" if i % 4 == 3 else "linear_attention" for i in range(48)],
    hidden_size=2560,
    hc_count=4,
    hc_lowrank=320,
    head_dim=256,
    num_attention_heads=24,
    num_key_value_heads=2,
    indexer_n_heads=4,
    indexer_kv_heads=1,
    indexer_head_dim=128,
    linear_num_key_heads=16,
    linear_num_value_heads=48,
    linear_key_head_dim=128,
    linear_value_head_dim=128,
    linear_conv_kernel_dim=4,
    num_experts=512,
    moe_intermediate_size=640,
    shared_expert_intermediate_size=640,
    vocab_size=248320,
    ple_layer_ids=[2],
    ple_embed_dim=2560,
    ple_conv_kernel_size=4,
    ngram_size=3,
    heads_per_ngram=8,
)
_VISION = dict(
    depth=27,
    hidden_size=1152,
    intermediate_size=4304,
    in_channels=3,
    temporal_patch_size=2,
    patch_size=16,
    num_position_embeddings=2304,
    spatial_merge_size=2,
)


def _core(mtp: bool, backend: str = "TRTLLM", prefill_copy: bool | None = None):
    """The configuration surface the table reads; the second expert copy defaults to kept
    under trtllm-gen."""
    return SimpleNamespace(
        model_config=SimpleNamespace(
            pretrained_config=SimpleNamespace(**_TEXT),
            torch_dtype=torch.bfloat16,
            spec_config=object() if mtp else None,
        ),
        vision_config=SimpleNamespace(**_VISION),
        mtp_enabled=mtp,
        trunk_moe_backend=backend,
        moe_prefill_copy=backend == "TRTLLM" if prefill_copy is None else prefill_copy,
    )


def test_every_declared_weight_has_a_source():
    """A declared weight with no source allocates memory nothing ever fills -- silently
    zero rather than an error, which is what `load`'s coverage assert catches from the
    other side. With MTP live: the rows without it are a subset
    (`test_mtp_rows_appear_only_when_mtp_is_live`)."""
    core = _core(True)
    manifest = W.MODEL_WEIGHTS.manifest(core)
    for key, sources in manifest.items():
        assert sources, f"{key} is declared with no checkpoint source"
    with torch.device("meta"):
        declared = W.MODEL_WEIGHTS.declare(W.MODEL_WEIGHTS.dims(core))
    assert set(manifest) == set(declared)


def test_mtp_rows_appear_only_when_mtp_is_live():
    off = W.MODEL_WEIGHTS.manifest(_core(False))
    on = W.MODEL_WEIGHTS.manifest(_core(True))
    added = set(on) - set(off)
    assert set(off) <= set(on)
    m = f"l{_TEXT['num_hidden_layers']}_mtp_"
    assert added and all(k.startswith(m) for k in added)
    # The draft layer runs QSA over its own indexer, read from the module's keys.
    indexer = {f"{m}idx_w", f"{m}idx_qn", f"{m}idx_kn"}
    assert indexer <= added
    assert all(
        str(src).startswith("mtp.layers.0.self_attn.indexer.")
        for k in indexer
        for src, _, _ in on[k]
    )


def _reference_declaration(mtp: bool) -> dict:
    """The declaration block from modeling.py, transcribed."""
    t, v = SimpleNamespace(**_TEXT), SimpleNamespace(**_VISION)
    bf16, u8, f8, f32, i64 = (
        torch.bfloat16,
        torch.uint8,
        torch.float8_e4m3fn,
        torch.float32,
        torch.int64,
    )
    hidden, stream, rank = t.hidden_size, t.hc_count * t.hidden_size, t.hc_lowrank
    q_width = t.num_attention_heads * t.head_dim
    kv_width = t.num_key_value_heads * t.head_dim
    index_width = (t.indexer_n_heads + t.indexer_kv_heads) * t.indexer_head_dim
    key_dim = t.linear_num_key_heads * t.linear_key_head_dim
    value_dim = t.linear_num_value_heads * t.linear_value_head_dim
    conv_dim = 2 * key_dim + value_dim
    experts, inter, shared = (
        t.num_experts,
        t.moe_intermediate_size,
        t.shared_expert_intermediate_size,
    )
    ngram_heads = t.heads_per_ngram * (t.ngram_size - 1)
    vh, vi = v.hidden_size, v.intermediate_size
    vm = vh * v.spatial_merge_size**2
    nl = t.num_hidden_layers
    w = {}

    def P(name, *shape, dtype=bf16):
        w[name] = (tuple(shape), dtype)

    def hyper_connection(prefix):
        P(f"{prefix}_norm", stream)
        P(f"{prefix}_down", rank, stream)
        P(f"{prefix}_up", stream, rank)
        P(f"{prefix}_inject", t.hc_count, stream)

    def gated_attention(prefix, fused_in=False):
        if fused_in:
            # [q | k | v | gate | index q, k | one index head of zero pad]
            rows = 2 * q_width + 2 * kv_width + index_width + t.indexer_head_dim
            P(f"{prefix}attn_in_w", rows, hidden)
        else:
            P(f"{prefix}q_w", 2 * q_width, hidden)
            P(f"{prefix}k_w", kv_width, hidden)
            P(f"{prefix}v_w", kv_width, hidden)
        P(f"{prefix}o_w", hidden, q_width)
        P(f"{prefix}q_norm", t.head_dim)
        P(f"{prefix}k_norm", t.head_dim)

    def shared_expert(prefix):
        # gate|up, then the per-token gate's row and zero rows (`weights._cat_rows`).
        P(f"{prefix}sh_gu", 2 * shared + 8, hidden)
        P(f"{prefix}sh_dn", hidden, shared)

    for i in range(nl):
        hyper_connection(f"l{i}_ahc")
        hyper_connection(f"l{i}_mhc")
        if t.layer_types[i] == "full_attention":
            gated_attention(f"l{i}_", fused_in=True)
            P(f"l{i}_idx_qn", t.indexer_head_dim)
            P(f"l{i}_idx_kn", t.indexer_head_dim)
        else:
            # [in_proj_qkv | in_proj_z | in_proj_a | in_proj_b], one row block each.
            P(f"l{i}_in_proj_w", conv_dim + value_dim + 2 * t.linear_num_value_heads, hidden)
            P(f"l{i}_A_log", t.linear_num_value_heads)
            P(f"l{i}_dt_bias", t.linear_num_value_heads)
            P(f"l{i}_conv_w", conv_dim, 1, t.linear_conv_kernel_dim)
            P(f"l{i}_gnorm", t.linear_value_head_dim)
            P(f"l{i}_out_w", hidden, value_dim)
        P(f"l{i}_router", experts, hidden)
        P(f"l{i}_fc1_w", experts, 2 * inter, hidden // 2, dtype=u8)
        P(f"l{i}_fc1_s", experts, 2 * inter, hidden // 16, dtype=u8)
        P(f"l{i}_fc2_w", experts, hidden, inter // 2, dtype=u8)
        P(f"l{i}_fc2_s", experts, hidden, inter // 16, dtype=u8)
        # The trtllm-gen backend's second, CuteDSL-layout copy of the experts.
        P(f"l{i}_fc1_w_cd", experts, 2 * inter, hidden // 2, dtype=u8)
        P(f"l{i}_fc1_s_cd", experts, 2 * inter, hidden // 16, dtype=u8)
        P(f"l{i}_fc2_w_cd", experts, hidden, inter // 2, dtype=u8)
        P(f"l{i}_fc2_s_cd", experts, hidden, inter // 16, dtype=u8)
        for name in ("e_isc1", "e_isc1_up", "e_ws2_1", "e_ws2_1_up", "e_isc2", "e_ws2_2"):
            P(f"l{i}_{name}", experts, dtype=f32)
        shared_expert(f"l{i}_")
    P("fhc_norm", stream)
    P("fhc_down", rank, stream)
    P("fhc_up", stream, rank)
    P("embed", t.vocab_size, hidden)
    P("ple_key_w", stream, hidden)
    P("ple_val_w", t.ple_embed_dim, hidden)
    P("ple_nk", stream)
    P("ple_nq", stream)
    P("ple_nc", stream)
    P("ple_conv_w", stream, 1, t.ple_conv_kernel_size)
    P("ple_mult", t.ngram_size, dtype=i64)
    P("ple_vocab", ngram_heads, dtype=i64)
    P("ple_offs", ngram_heads, dtype=i64)
    P("ple_tscale", 1)
    P("vis_patch_w", vh, v.in_channels, v.temporal_patch_size, v.patch_size, v.patch_size)
    P("vis_patch_b", vh)
    P("vis_pos", v.num_position_embeddings, vh)
    for b in range(v.depth):
        P(f"l{b}_vis_n1_w", vh)
        P(f"l{b}_vis_n1_b", vh)
        P(f"l{b}_vis_qkv_w", 3 * vh, vh)
        P(f"l{b}_vis_qkv_b", 3 * vh)
        P(f"l{b}_vis_proj_w", vh, vh)
        P(f"l{b}_vis_proj_b", vh)
        P(f"l{b}_vis_n2_w", vh)
        P(f"l{b}_vis_n2_b", vh)
        P(f"l{b}_vis_fc1_w", vi, vh)
        P(f"l{b}_vis_fc1_b", vi)
        P(f"l{b}_vis_fc2_w", vh, vi)
        P(f"l{b}_vis_fc2_b", vh)
    P("vis_m_n_w", vh)
    P("vis_m_n_b", vh)
    P("vis_m_fc1_w", vm, vm)
    P("vis_m_fc1_b", vm)
    P("vis_m_fc2_w", hidden, vm)
    P("vis_m_fc2_b", hidden)
    if mtp:
        m = f"l{nl}_mtp_"
        P(f"{m}pre_e_norm", hidden)
        P(f"{m}pre_h_norm", stream)
        P(f"{m}fc_e", hidden, hidden)
        P(f"{m}fc_h", hidden, hidden)
        hyper_connection(f"{m}ahc")
        hyper_connection(f"{m}mhc")
        gated_attention(m)
        P(f"{m}idx_w", index_width, hidden)
        P(f"{m}idx_qn", t.indexer_head_dim)
        P(f"{m}idx_kn", t.indexer_head_dim)
        P(f"{m}router", experts, hidden)
        P(f"{m}fc1_w", experts, 2 * inter, hidden, dtype=f8)
        P(f"{m}fc1_s", experts, (2 * inter) // 128, hidden // 128, dtype=f32)
        P(f"{m}fc2_w", experts, hidden, inter, dtype=f8)
        P(f"{m}fc2_s", experts, hidden // 128, inter // 128, dtype=f32)
        shared_expert(m)
        P(f"{m}fhc_norm", stream)
        P(f"{m}fhc_down", rank, stream)
        P(f"{m}fhc_up", stream, rank)
    return w


def test_the_table_declares_what_the_init_block_did():
    """With MTP live; without it the table drops exactly the MTP module's rows
    (`test_mtp_rows_appear_only_when_mtp_is_live`)."""
    core = _core(True)
    with torch.device("meta"):
        declared = W.MODEL_WEIGHTS.declare(W.MODEL_WEIGHTS.dims(core))
    got = {k: (tuple(p.shape), p.dtype) for k, p in declared.items()}
    want = _reference_declaration(True)
    assert set(got) == set(want), (
        sorted(set(got) - set(want))[:8],
        sorted(set(want) - set(got))[:8],
    )
    wrong = {k: (got[k], want[k]) for k in got if got[k] != want[k]}
    assert not wrong, list(wrong.items())[:8]


def test_gdn_input_projections_fill_adjacent_row_blocks():
    """The concatenated GDN input projection: each checkpoint tensor lands in its own
    row block, in the column order the forward splits the GEMM output by, and the
    blocks tile the declared rows exactly."""
    t = SimpleNamespace(**_TEXT)
    core = _core(False)
    manifest = W.MODEL_WEIGHTS.manifest(core)
    key_dim = t.linear_num_key_heads * t.linear_key_head_dim
    value_dim = t.linear_num_value_heads * t.linear_value_head_dim
    widths = (
        2 * key_dim + value_dim,
        value_dim,
        t.linear_num_value_heads,
        t.linear_num_value_heads,
    )
    gdn = [i for i, kind in enumerate(t.layer_types) if kind == "linear_attention"]
    assert len(gdn) == 36
    for i in gdn:
        rows = manifest[f"l{i}_in_proj_w"]
        names = [ckpt.rsplit(".", 2)[-2] for ckpt, _, _ in rows]
        assert names == ["in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"], names
        assert all(ckpt.startswith(f"{W._LM}.layers.{i}.linear_attn.") for ckpt, _, _ in rows)
        assert all(transform is None for _, _, transform in rows)
        spans = [(index[0].start, index[0].stop) for _, index, _ in rows]
        starts = [0, widths[0], widths[0] + widths[1], widths[0] + widths[1] + widths[2]]
        assert spans == [(s, s + w) for s, w in zip(starts, widths)], spans
        assert spans[-1][1] == sum(widths) == 16480


def test_attention_input_projection_relayout():
    """`_attn_in_w` puts q, k, v, the output gate and the index projection side by side
    so that one GEMM's columns are exactly what the four separate projections gave:
    q and gate de-interleaved from q_proj's per-head `[q | gate]` rows, then a zero pad
    of one index head. Checked at the checkpoint's head geometry, a narrow hidden."""
    t = SimpleNamespace(**_TEXT)
    heads, d, kvh, hidden = t.num_attention_heads, t.head_dim, t.num_key_value_heads, 64
    idx_rows = (t.indexer_n_heads + t.indexer_kv_heads) * t.indexer_head_dim
    g = torch.Generator().manual_seed(0)
    q = torch.randn(2 * heads * d, hidden, generator=g, dtype=torch.float64)
    k = torch.randn(kvh * d, hidden, generator=g, dtype=torch.float64)
    v = torch.randn(kvh * d, hidden, generator=g, dtype=torch.float64)
    idx = torch.randn(idx_rows, hidden, generator=g, dtype=torch.float64)
    core = SimpleNamespace(heads_q=heads, head_dim=d, index_head_dim=t.indexer_head_dim)
    fused = W._attn_in_w((q, k, v, idx), core)
    rows = 2 * heads * d + 2 * kvh * d + idx_rows + t.indexer_head_dim
    assert fused.shape == (rows, hidden) and fused.is_contiguous()

    x = torch.randn(3, hidden, generator=g, dtype=torch.float64)
    y = x @ fused.t()
    qg = (x @ q.t()).reshape(3, heads, 2 * d)
    want = torch.cat(
        [
            qg[..., :d].reshape(3, -1),
            x @ k.t(),
            x @ v.t(),
            qg[..., d:].reshape(3, -1),
            x @ idx.t(),
            torch.zeros(3, t.indexer_head_dim, dtype=torch.float64),
        ],
        1,
    )
    torch.testing.assert_close(y, want, rtol=0, atol=1e-12)
    # The row is a whole number of head_dim heads: the main norm+rope addresses it so.
    assert rows % d == 0


def test_cutedsl_fc1_pairs_up_and_gate_rows_in_64_row_groups():
    """Destination rows `[128k, 128k + 64)` hold up rows `[64k, 64k + 64)` and the next 64
    hold the matching gate rows, which sit `I` rows later in the `[up ; gate]` stack."""
    inter = _TEXT["moe_intermediate_size"]
    core = SimpleNamespace(trunk_moe_backend="CUTEDSL")
    perm = W._fc1_perm(2 * inter, "cpu", core)
    assert sorted(perm.tolist()) == list(range(2 * inter))
    for k in range(inter // 64):
        assert perm[128 * k : 128 * k + 64].tolist() == list(range(64 * k, 64 * k + 64))
        assert perm[128 * k + 64 : 128 * k + 128].tolist() == list(
            range(inter + 64 * k, inter + 64 * k + 64)
        )


def test_cutedsl_fc2_keeps_the_checkpoint_row_order_and_trtllm_gen_shuffles_it():
    """trtllm-gen's 32-row block shuffle moves source row `4u + v` to destination row
    `8v + u`; CuteDSL reads `down_proj` as the checkpoint stores it."""
    rows = torch.arange(64).unsqueeze(1).expand(64, 8).contiguous()
    kept = W._expert_fc2_w(rows, SimpleNamespace(trunk_moe_backend="CUTEDSL"))
    assert torch.equal(kept, rows)
    shuffled = W._expert_fc2_w(rows, SimpleNamespace(trunk_moe_backend="TRTLLM"))
    for u in range(8):
        for v in range(4):
            assert int(shuffled[8 * v + u, 0]) == 4 * u + v
            assert int(shuffled[32 + 8 * v + u, 0]) == 32 + 4 * u + v


@pytest.mark.parametrize("backend, prefill_copy", [("TRTLLM", True), ("CUTEDSL", False)])
def test_cute_dsl_expert_copy_rows_only_when_kept(backend, prefill_copy):
    """With the second expert copy (trtllm-gen, a step able to reach the CuteDSL dispatch
    threshold) every layer also carries its experts in the CuteDSL layout (`fc*_cd`);
    without it, and under CuteDSL whose one copy is that layout, no `_cd` row is declared."""
    manifest = W.MODEL_WEIGHTS.manifest(_core(False, backend, prefill_copy))
    rows = [k for k in manifest if k.endswith(("_fc1_w_cd", "_fc1_s_cd", "_fc2_w_cd", "_fc2_s_cd"))]
    layers = _TEXT["num_hidden_layers"]
    assert len(rows) == (4 * layers if prefill_copy else 0), len(rows)


@pytest.mark.parametrize(
    "backend, max_num_tokens, env, kept",
    [
        ("TRTLLM", 4096, None, True),
        ("TRTLLM", 4095, None, False),
        ("TRTLLM", 16384, "0", False),
        ("CUTEDSL", 16384, None, False),
    ],
)
def test_second_expert_copy_kept_only_where_its_launches_can_happen(
    monkeypatch, backend, max_num_tokens, env, kept
):
    """The copy costs about 1.3 GiB a layer: it is kept only under trtllm-gen, only when a
    step's rows can reach the CuteDSL dispatch threshold, and not when the env var is off."""
    if env is None:
        monkeypatch.delenv(M._MOE_PREFILL_COPY_ENV, raising=False)
    else:
        monkeypatch.setenv(M._MOE_PREFILL_COPY_ENV, env)
    assert M._moe_prefill_copy(backend, max_num_tokens) is kept
