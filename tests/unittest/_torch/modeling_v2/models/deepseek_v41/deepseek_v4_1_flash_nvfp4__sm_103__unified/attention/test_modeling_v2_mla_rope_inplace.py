# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the mla_rope_inplace catalog entry.

The entry owns what is certified: `CELLS` is the list, `reference` and
`compare` are the gate. This file builds the DeepSeek-V4.1 tables, turns a
cell's spec into tensors, and keeps the checks a cell cannot carry: the
tables against the checkpoint's recipe, the forward/inverse round trip, the
wrong-variant controls, the op's rejections, CUDA-graph replay, every row
count 1..32768 for each core call form, the capture union, every envelope
engine's own table with positions at its far end, and the packed ViT form.
Every call writes between two guard rows that must come back unchanged, and
the `nope` channels must come back bitwise.
"""

import math
import zlib

import pytest
import torch
from torch.profiler import ProfilerActivity, profile

from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.attention.mla_rope_inplace import (  # noqa: E501
    mla_rope_inplace as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "mla_rope_inplace requires a CUDA device"

DEV = "cuda"
ROPE = 64
# The tables have the row count of the largest ones the target builds. The target sizes
# both tables to md.max_seq_len + 64 rows, where md.max_seq_len is the manager length Lm:
# the engine's max_seq_len L, plus 14 with DSpark on. The envelope's per-request lengths
# are below; the largest is the checkpoint's max_position_embeddings. Positions are drawn
# over the whole table, up to 16x the YaRN original context (65536), so the interpolated
# low-frequency channels rotate by angles a no-YaRN table cannot reproduce.
ENVELOPE_L = (16384, 32768, 65536, 131072, 262144, 1048576)
DSPARK_EXTRA_ROWS = 14
TABLE_PAD_ROWS = 64
MAX_POS = max(ENVELOPE_L) + DSPARK_EXTRA_ROWS + TABLE_PAD_ROWS  # 1048654 table rows
YARN = dict(factor=16.0, original=65536, beta_fast=32, beta_slow=1)


def _table(base: float, yarn: bool, max_pos: int = MAX_POS, mscale: float = 1.0) -> torch.Tensor:
    """[max_pos, 2, ROPE/2] fp32 (cos block, sin block) built with the checkpoint's recipe."""
    angles = torch.outer(torch.arange(max_pos, dtype=torch.float32, device=DEV), _freqs(base, yarn))
    return (torch.stack([angles.cos(), angles.sin()], dim=1) * mscale).contiguous()


def _freqs(base: float, yarn: bool) -> torch.Tensor:
    """[ROPE/2] fp32 rotation frequencies: base^(-2i/ROPE), YaRN frequency interpolation only."""
    freqs = 1.0 / (base ** (torch.arange(0, ROPE, 2, dtype=torch.float32, device=DEV) / ROPE))
    if yarn:

        def corrected_dim(rotations: float) -> float:
            return (
                ROPE * math.log(YARN["original"] / (rotations * 2 * math.pi)) / (2 * math.log(base))
            )

        low = max(math.floor(corrected_dim(YARN["beta_fast"])), 0)
        high = min(math.ceil(corrected_dim(YARN["beta_slow"])), ROPE - 1)
        ramp = (
            (torch.arange(ROPE // 2, dtype=torch.float32, device=DEV) - low) / max(high - low, 1e-3)
        ).clamp(0, 1)
        smooth = 1 - ramp
        freqs = freqs / YARN["factor"] * (1 - smooth) + freqs * smooth
    return freqs


_TABLE_SPECS = {
    # ratio-0 layers (0, 1) and the DSpark stages: rope_theta 10000, no YaRN
    "ratio0_theta1e4": (10000.0, False),
    # compressed layers (ratio 1 and 2): compress_rope_theta 160000 with YaRN, no mscale
    "compressed_theta160k_yarn": (160000.0, True),
}
_TABLES: dict = {}


def _get_table(name: str) -> torch.Tensor:
    if name not in _TABLES:
        base, yarn = _TABLE_SPECS[name]
        _TABLES[name] = _table(base, yarn)
    return _TABLES[name]


@pytest.mark.parametrize("table_name", list(_TABLE_SPECS))
def test_table_is_the_checkpoint_reference_recipe_at_full_size(table_name):
    """Every row of the MAX_POS-row table equals the checkpoint reference's
    precompute_freqs_cis rebuilt here as it builds it (int64 positions promoted to
    fp32 angles, then torch.polar), on CUDA as the reference does, bitwise."""
    base, yarn = _TABLE_SPECS[table_name]
    angles = torch.outer(torch.arange(MAX_POS, device=DEV), _freqs(base, yarn))
    assert angles.dtype == torch.float32
    cis = torch.polar(torch.ones_like(angles), angles)
    table = _get_table(table_name)
    assert table.shape == (MAX_POS, 2, ROPE // 2)
    assert torch.equal(table[:, 0], cis.real) and torch.equal(table[:, 1], cis.imag)


def _reference(
    x: torch.Tensor,
    pos: torch.Tensor,
    table: torch.Tensor,
    nope: int,
    inverse: bool,
    is_neox: bool,
    first: bool = False,
) -> torch.Tensor:
    """`op.reference` in this file's argument order, `rope_dim` = ROPE.

    `first=True` is for a control only: it rotates the leading ROPE channels instead, by
    moving them to the tail, rotating there, and moving them back.
    """
    if first:
        moved = torch.cat([x[..., ROPE:], x[..., :ROPE]], dim=-1)
        y = op.reference(moved, pos, table, x.size(1), nope, ROPE, inverse, is_neox)
        return torch.cat([y[..., -ROPE:], y[..., :-ROPE]], dim=-1)
    return op.reference(x, pos, table, x.size(1), nope, ROPE, inverse, is_neox)


def _positions(num_tokens: int, gen: torch.Generator) -> torch.Tensor:
    if num_tokens == 1:
        return torch.tensor([MAX_POS - 1], dtype=torch.int32, device=DEV)
    if num_tokens <= 8:  # a decode / DSpark-verify run of consecutive positions
        start = int(torch.randint(0, MAX_POS - num_tokens, (1,), generator=gen).item())
        return torch.arange(start, start + num_tokens, dtype=torch.int32).to(DEV)
    pos = torch.randint(0, MAX_POS, (num_tokens,), generator=gen, dtype=torch.int32)
    pos[0], pos[-1] = 0, MAX_POS - 1
    return pos.to(DEV)


def _run(
    x: torch.Tensor,
    pos: torch.Tensor,
    table: torch.Tensor,
    nope: int,
    inverse: bool,
    is_neox: bool,
) -> torch.Tensor:
    """Call the op on x placed between two guard rows; lock both sides of the write."""
    num_tokens, num_heads, head_dim = x.shape
    buf = torch.randn(num_tokens + 2, num_heads, head_dim, dtype=x.dtype, device=DEV)
    buf[1:-1].copy_(x)
    before = buf.clone()
    data = buf[1:-1]
    assert data.is_contiguous()
    result = op(data, pos, table, num_heads, nope, ROPE, inverse, is_neox)
    assert result is data, "the op rotates in place and returns data"
    assert torch.equal(buf[0], before[0]) and torch.equal(buf[-1], before[-1]), "out-of-row write"
    assert torch.equal(data[..., :nope], x[..., :nope]), "nope channels were modified"
    return data.clone()


def _cell_inputs(spec, gen: torch.Generator):
    """(x, positions, table) for one cell, built the way the target builds them."""
    form = spec["form"]
    if form == "table":
        # T = 1 at the table's last row, T = 6 a consecutive run, T = 8192 random with 0 and
        # the last row included.
        table = _get_table(spec["table"])
        assert table.shape[0] == spec["rows"] == MAX_POS
        num_tokens = spec["T"]
        pos = _positions(num_tokens, gen)
    elif form == "group_first":
        # Latent j of a ratio-r layer at position j * r (the group's first token), for every
        # group whose first position is in the table.
        table = _get_table(spec["table"])
        assert table.shape[0] == spec["rows"] == MAX_POS
        num_tokens = spec["rows"] // spec["ratio"]
        pos = (torch.arange(num_tokens, dtype=torch.int32) * spec["ratio"]).to(DEV)
    elif form == "random_angles":
        # NeoX half-split pairs over a whole 64-wide head, an arbitrary per-row table.
        num_tokens = spec["T"]
        angles = (torch.rand(num_tokens, ROPE // 2, generator=gen) * 64.0).to(DEV)
        table = torch.stack([angles.cos(), angles.sin()], dim=1).contiguous()
        pos = torch.arange(num_tokens, dtype=torch.int32, device=DEV)
    else:
        assert form == "vit_packed", form
        # The vision tower's per-request call: its own [P + 1] table with the identity row
        # at P, positions (t, t, P).
        table = _vit_table(spec["P"], gen)
        pos = _vit_positions(spec["P"])
        num_tokens = 3 * spec["P"]
    x = torch.randn(num_tokens, spec["H"], spec["D"], generator=gen).to(DEV, torch.bfloat16)
    return x, pos, table


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    certifying(op)
    spec = cell.spec
    gen = torch.Generator().manual_seed(zlib.crc32(cell.why.encode()))
    x, pos, table = _cell_inputs(spec, gen)
    out = _run(x, pos, table, spec["nope"], inverse=spec["inverse"], is_neox=spec["is_neox"])
    op.compare(
        out,
        op.reference(
            x, pos, table, spec["H"], spec["nope"], ROPE, spec["inverse"], spec["is_neox"]
        ),
    )
    if spec["form"] == "vit_packed":
        assert torch.equal(out[2::3], x[2::3]), "a v row read through the identity row changed"


def test_inverse_undoes_forward():
    gen = torch.Generator().manual_seed(5000)
    x = torch.randn(8192, 16, 512, generator=gen).to(DEV, torch.bfloat16)
    pos = _positions(8192, gen)
    table = _get_table("compressed_theta160k_yarn")
    y = _run(x, pos, table, 448, inverse=False, is_neox=False)
    back = _run(y, pos, table, 448, inverse=True, is_neox=False)
    # Two bf16 roundings (after the forward and after the inverse rotation), each at most half
    # an ulp = 2^-9 of the pair's magnitude; the rotation preserves the pair norm, so the error
    # of either channel is bounded by 2 * sqrt(2) * 2^-9 = 5.5e-3 of the pair norm. Gate at 2^-6.
    pair = x.float()[..., 448:].unflatten(-1, (-1, 2)).norm(dim=-1).repeat_interleave(2, dim=-1)
    err = (back.float()[..., 448:] - x.float()[..., 448:]).abs()
    assert bool((err <= pair * 2**-6).all()), f"max err / pair norm {(err / pair).max().item()}"
    twice = _run(y, pos, table, 448, inverse=False, is_neox=False)
    assert not bool(((twice.float()[..., 448:] - x.float()[..., 448:]).abs() <= pair * 2**-6).all())


# Deliberately wrong references. Each must be rejected by the same gate that accepts the
# correct reference, which is what shows the gate can see the checkpoint's RoPE mistakes.
_MSCALE = 0.1 * math.log(YARN["factor"]) + 1.0
CONTROLS = {
    # (table the op runs with, op inverse, positions scale, wrong reference builder)
    "rotate_half_pairing": (
        "compressed_theta160k_yarn",
        False,
        1,
        lambda x, p: (x, p, "same", 448, False, True, False),
    ),
    "first_64_dims": (
        "compressed_theta160k_yarn",
        False,
        1,
        lambda x, p: (x, p, "same", 448, False, False, True),
    ),
    "yarn_with_mscale": (
        "compressed_theta160k_yarn",
        False,
        1,
        lambda x, p: (x, p, "mscale", 448, False, False, False),
    ),
    "compressed_without_yarn": (
        "compressed_theta160k_yarn",
        False,
        1,
        lambda x, p: (x, p, "noyarn160k", 448, False, False, False),
    ),
    "compressed_with_theta1e4": (
        "compressed_theta160k_yarn",
        False,
        1,
        lambda x, p: (x, p, "ratio0_theta1e4", 448, False, False, False),
    ),
    "ratio0_with_yarn_theta160k": (
        "ratio0_theta1e4",
        False,
        1,
        lambda x, p: (x, p, "compressed_theta160k_yarn", 448, False, False, False),
    ),
    "ratio0_with_yarn_theta1e4": (
        "ratio0_theta1e4",
        False,
        1,
        lambda x, p: (x, p, "yarn1e4", 448, False, False, False),
    ),
    "forward_instead_of_inverse": (
        "compressed_theta160k_yarn",
        True,
        1,
        lambda x, p: (x, p, "same", 448, False, False, False),
    ),
    "latent_at_group_last_position": (
        "compressed_theta160k_yarn",
        False,
        2,
        lambda x, p: (x, p + 1, "same", 448, False, False, False),
    ),
}


def _control_table(name: str, run_table: str) -> torch.Tensor:
    if name == "same":
        return _get_table(run_table)
    if name == "mscale":
        return _table(160000.0, True, mscale=_MSCALE)
    if name == "noyarn160k":
        return _table(160000.0, False)
    if name == "yarn1e4":
        return _table(10000.0, True)
    return _get_table(name)


@pytest.mark.parametrize("control", list(CONTROLS))
def test_wrong_variant_is_rejected(control):
    run_table, inverse, pos_scale, wrong = CONTROLS[control]
    gen = torch.Generator().manual_seed(6000)
    x = torch.randn(4096, 16, 512, generator=gen).to(DEV, torch.bfloat16)
    pos = _positions(4096, gen) // pos_scale * pos_scale  # ratio-2 group-first positions
    table = _get_table(run_table)
    out = _run(x, pos, table, 448, inverse=inverse, is_neox=False)
    op.compare(out, _reference(x, pos, table, 448, inverse=inverse, is_neox=False))
    wx, wp, wtab, wnope, winv, wneox, wfirst = wrong(x, pos)
    bad = _reference(wx, wp, _control_table(wtab, run_table), wnope, winv, wneox, wfirst)
    close = torch.isclose(out.float(), bad.float(), rtol=1.6e-2, atol=1e-5)
    frac_bad = 1.0 - close.float().mean().item()
    max_err = (out.float() - bad.float()).abs().max().item()
    print(
        f"control {control}: {frac_bad:.4f} of elements outside the gate, max abs err {max_err:.3e}"
    )
    assert not bool(close.all()), f"control {control} was not rejected"


def _raises(msg, fn):
    with pytest.raises(RuntimeError, match=msg):
        fn()


@pytest.mark.parametrize(
    "case", ["strided_tail_view", "int64_positions", "fp32_data", "flat_cos_sin_table"]
)
def test_rejected_inputs_leave_data_untouched(case):
    x = torch.randn(4, 16, 512, device=DEV, dtype=torch.bfloat16)
    before = x.clone()
    pos = torch.arange(4, dtype=torch.int32, device=DEV)
    table = _table(10000.0, False, max_pos=16)
    if case == "strided_tail_view":
        _raises(
            "data must be contiguous",
            lambda: op(x[..., 448:], pos, table, 16, 0, ROPE, False, False),
        )
    elif case == "int64_positions":
        _raises(
            "position_ids must be int32",
            lambda: op(x, pos.long(), table, 16, 448, ROPE, False, False),
        )
    elif case == "fp32_data":
        xf = x.float()
        _raises(
            "unsupported dtype, expected bf16 or fp16",
            lambda: op(xf, pos, table, 16, 448, ROPE, False, False),
        )
    else:
        _raises(
            r"cos_sin_cache must have shape \[max_positions, 2, rope_dim / 2\]",
            lambda: op(x, pos, table.view(16, ROPE), 16, 448, ROPE, False, False),
        )
    torch.cuda.synchronize()
    assert torch.equal(x, before)


def test_cuda_graph_replay_reads_positions_and_data_at_replay():
    table = _get_table("compressed_theta160k_yarn")
    data = torch.zeros(8, 16, 512, device=DEV, dtype=torch.bfloat16)
    pos = torch.zeros(8, dtype=torch.int32, device=DEV)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        op(data, pos, table, 16, 448, ROPE, False, False)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op(data, pos, table, 16, 448, ROPE, False, False)
    gen = torch.Generator().manual_seed(7000)
    for _ in range(3):
        x = torch.randn(8, 16, 512, generator=gen).to(DEV, torch.bfloat16)
        p = _positions(8, gen)
        data.copy_(x)
        pos.copy_(p)
        graph.replay()
        torch.cuda.synchronize()
        op.compare(data.clone(), _reference(x, p, table, 448, inverse=False, is_neox=False))


# ─── Every call form of the core target at every row count 1..32768, and the capture union ──
#
# DeepSeek-V4.1's core target rotates, per attention layer: q [T, H, 512] and the
# window kv [T, 1, 512] with the layer's table (ratio-0 on layers 0-1, the
# compressed table otherwise), the attention output [T, H, 512] inversely, the
# index q [T, 32, 128] with the compressed table, and on KV-source layers the
# index k [n, 1, 128] and the compressed latent [n, 1, 512] at compressed
# positions. H is the rank's share of the 64 heads, 64 / tp under attention TP:
# 32 (tp 2: the pp2 x tep2 / tp2 stages), 16 (tp 4: tep4, tp4 and their pp2
# stages) or 8 (tp 8: tep8); it is 64 under attention DP (dep4, dep8 and their
# pp2 stages) and in the DSpark drafter's q and output at every layout (ratio-0
# table). Every H meets both tables: a pp2 stage 0 holds layers 0-19 (ratio 0
# and 2), stage 1 layers 20-39 (compressed table). T and n are at most the
# engine's max_num_tokens, 16384 per rank; in layers 21-39 the decoder bounded
# replay's pass adds the finishing requests' 128-row tails to the generation
# rows, up to 128 x 256 = 32768. The launch grid is (T, ceil(H / heads per
# block)): block (t, g) reads position_ids[t] and rotates row t only, and the
# kernel variant depends on H, not T. The tests below check that on the GPU at
# every row count from 1 to 32768, for every form.

_N_MAX = 32768
# (H, head_dim, nope_dim, inverse) of every call the core target makes
TARGET_FORMS = {
    "q_h8": (8, 512, 448, False),
    "q_h16": (16, 512, 448, False),
    "q_h32": (32, 512, 448, False),
    "q_h64": (64, 512, 448, False),
    "window_kv": (1, 512, 448, False),
    "attn_out_h8": (8, 512, 448, True),
    "attn_out_h16": (16, 512, 448, True),
    "attn_out_h32": (32, 512, 448, True),
    "attn_out_h64": (64, 512, 448, True),
    "index_q": (32, 128, 64, False),
    "index_k": (1, 128, 64, False),
    "compressed_latent": (1, 512, 448, False),
}
# q, kv and the output use the layer's table; the indexer and the latent always use the compressed one
_LAYER_TABLE_FORMS = (
    "q_h8", "q_h16", "q_h32", "q_h64", "window_kv", "attn_out_h8", "attn_out_h16", "attn_out_h32", "attn_out_h64",
)  # fmt: skip
_FORM_TABLES = [
    (f, t)
    for f in TARGET_FORMS
    for t in _TABLE_SPECS
    if f in _LAYER_TABLE_FORMS or t != "ratio0_theta1e4"
]
assert len(_FORM_TABLES) == 21
# T values in the core's runtime-value dump (boot gates tep4, dep4, tp4 and tep8)
_DUMPED_T = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    12,
    14,
    16,
    22,
    24,
    48,
    128,
    140,
    888,
    890,
    1018,
    1030,
    1935,
    1942,
    2048,
)
# The capture union per rank. G is every CUDA-graph batch the envelope
# captures: every batch 1..32, plus TRT-LLM's padded list for
# cuda_graph_config {enable_padding: true, max_batch_size: 256}
# (_generate_cuda_graph_batch_sizes in llmapi/llm_args.py: 1, 2, 4, 8, 16..128
# in steps of 8, 192, 256). The padding-off list at 256 (1..32, 64, 128, 256)
# is a subset. Per G, the target's captured calls rotate these row counts:
#   G  -- a decode step (next_n 1), every form; a ratio-2 layer reserves one
#         compressed row per request, so its latent and index k are G rows too;
#         also the DSpark drafter's bonus-row window kv (ratio-0 table);
#   3G -- the compressed latent and index k of a ratio-2 layer in the DSpark
#         target-verify step (next_n 6): each request reserves ceil(6 / 2) = 3
#         compressed rows (num_total_compressed_tokens[2] = G * 3);
#   5G -- the DSpark drafter's block: q and the output at 64 heads, and the
#         window kv, on the ratio-0 table;
#   6G -- the rest of the verify step: q, window kv, index q, the inverse
#         output, and the ratio-1 latent and index k (one row per token).
# Every form and table is captured at every T of the union (126 values, up to
# 1536), a superset of the row counts each form reaches.
_BATCHES = tuple(range(1, 33)) + tuple(range(40, 129, 8)) + (192, 256)
_CAPTURE_T = tuple(sorted({m * b for b in _BATCHES for m in (1, 3, 5, 6)}))
assert len(_BATCHES) == 46 and len(_CAPTURE_T) == 126 and max(_CAPTURE_T) == 1536


def _sweep_positions(num_tokens: int, gen: torch.Generator) -> torch.Tensor:
    """Random positions in [1, MAX_POS), with 1 and MAX_POS - 1 included.

    Position 0 is the identity rotation, so a row at position 0 that the kernel
    skipped would look correct; the sweep keeps every row's position nonzero.
    """
    pos = torch.randint(1, MAX_POS, (num_tokens,), generator=gen, dtype=torch.int32)
    pos[0], pos[-1] = MAX_POS - 1, 1
    return pos.to(DEV)


@pytest.mark.parametrize("form,table_name", _FORM_TABLES)
def test_every_row_count_is_the_same_per_row_function(form, table_name):
    """Every prefix T = 1..32768 returns exactly the first T rows of the 32768-row
    result, which is held to the fp32 reference.

    Each call runs in place on the first T rows of a buffer whose row T is a
    guard row holding finite data that must come back bit-identical. A row the
    kernel skipped would keep its unrotated value, which differs from the
    rotated one because no position is 0.
    """
    num_heads, head_dim, nope = TARGET_FORMS[form][:3]
    inverse = TARGET_FORMS[form][3]
    gen = torch.Generator().manual_seed(8000 + num_heads + head_dim)
    x = torch.randn(_N_MAX + 1, num_heads, head_dim, generator=gen).to(DEV, torch.bfloat16)
    guard_row = x[_N_MAX].clone()
    x = x[:_N_MAX]
    pos = _sweep_positions(_N_MAX, gen)
    table = _get_table(table_name)
    full = _run(x, pos, table, nope, inverse=inverse, is_neox=False)
    ref = _reference(x, pos, table, nope, inverse=inverse, is_neox=False)
    op.compare(full, ref)
    exact = float((full[..., nope:] == ref[..., nope:]).float().mean())
    print(
        f"\n[{form} {table_name}] T={_N_MAX}: rope channels bit-exact {exact:.6f} vs the fp32 reference"
    )
    del ref
    fb, gb = full.view(torch.int16), guard_row.view(torch.int16)
    buf = torch.empty(_N_MAX + 1, num_heads, head_dim, dtype=torch.bfloat16, device=DEV)
    mismatch = torch.zeros(_N_MAX + 1, dtype=torch.int64, device=DEV)
    guard = torch.zeros(_N_MAX + 1, dtype=torch.int64, device=DEV)
    differs = torch.empty(
        _N_MAX, num_heads, head_dim, dtype=torch.bool, device=DEV
    )  # one buffer; .any() keeps the reduction free of a widened copy
    for t in range(1, _N_MAX + 1):
        buf[:t].copy_(x[:t])
        buf[t].copy_(guard_row)
        op(buf[:t], pos[:t], table, num_heads, nope, ROPE, inverse, False)
        mismatch[t] = torch.ne(buf[:t].view(torch.int16), fb[:t], out=differs[:t]).any()
        guard[t] = (buf[t].view(torch.int16) != gb).sum()
    bad = mismatch.nonzero().flatten().tolist()
    over = guard.nonzero().flatten().tolist()
    assert not bad, (
        f"{len(bad)} row counts differ from the {_N_MAX}-row result's rows, first {bad[:10]}"
    )
    assert not over, f"{len(over)} row counts wrote the guard row, first {over[:10]}"


def test_row_sweep_harness_sees_a_one_row_shift():
    """Control for the sweep: its bitwise per-row check rejects a read one row off (data and position)."""
    num_heads, head_dim, nope = 16, 512, 448
    gen = torch.Generator().manual_seed(8100)
    x = torch.randn(2049, num_heads, head_dim, generator=gen).to(DEV, torch.bfloat16)
    pos = _sweep_positions(2049, gen)
    table = _get_table("compressed_theta160k_yarn")
    full = _run(x[:2048], pos[:2048], table, nope, inverse=False, is_neox=False)
    for t in (1, 2, 7, 1024, 2048):
        shifted = _run(x[1 : t + 1], pos[1 : t + 1], table, nope, inverse=False, is_neox=False)
        n_bad = int((shifted.view(torch.int16) != full.view(torch.int16)[:t]).sum().item())
        print(f"[control row shift] T={t}: {n_bad} / {t * num_heads * head_dim} elements differ")
        assert n_bad > t * num_heads * head_dim // 2
    # the rotation alone: the right data at the neighbouring row's position
    for t in (1, 7, 2048):
        wrong_pos = _run(x[:t], pos[1 : t + 1], table, nope, inverse=False, is_neox=False)
        n_bad = int((wrong_pos.view(torch.int16) != full.view(torch.int16)[:t]).sum().item())
        print(
            f"[control position shift] T={t}: {n_bad} / {t * num_heads * ROPE} rope elements differ"
        )
        assert n_bad > t * num_heads * ROPE // 2


@pytest.mark.parametrize("form,table_name", _FORM_TABLES)
def test_dumped_values_against_reference(form, table_name):
    """Fresh inputs at every dumped T, each call form and table, against the fp32 reference."""
    num_heads, head_dim, nope, inverse = TARGET_FORMS[form]
    table = _get_table(table_name)
    for num_tokens in _DUMPED_T:
        gen = torch.Generator().manual_seed(9000 + num_tokens)
        x = torch.randn(num_tokens, num_heads, head_dim, generator=gen).to(DEV, torch.bfloat16)
        pos = _positions(num_tokens, gen)
        out = _run(x, pos, table, nope, inverse=inverse, is_neox=False)
        op.compare(out, _reference(x, pos, table, nope, inverse=inverse, is_neox=False))


def test_one_compiled_kernel_per_form_at_every_row_count():
    """Probe: each call form launches the same single kernel at every T (compiled ahead of time, no JIT)."""
    table = _get_table("compressed_theta160k_yarn")
    gen = torch.Generator().manual_seed(9500)
    pos = _sweep_positions(_N_MAX, gen)
    counts = sorted(
        set(_DUMPED_T)
        | set(_CAPTURE_T)
        | {9, 64, 65, 129, 4096, 8191, 8192, 8193, 16383, 16384, 16385, _N_MAX - 1, _N_MAX}
    )
    for form, (num_heads, head_dim, nope, inverse) in TARGET_FORMS.items():
        data = torch.randn(_N_MAX, num_heads, head_dim, generator=gen).to(DEV, torch.bfloat16)
        names = set()
        for t in counts:
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                op(data[:t], pos[:t], table, num_heads, nope, ROPE, inverse, False)
                torch.cuda.synchronize()
            launched = [
                e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
            ]
            assert len(launched) == 1, (form, t, launched)
            names.add(launched[0])
        print(f"{form}: {sorted(names)}")
        assert len(names) == 1 and "mlaRoPEInplaceKernel" in next(iter(names)), (form, names)


@pytest.mark.parametrize("num_tokens", _CAPTURE_T)
def test_capture_union_replay_is_bitwise_eager(num_tokens):
    """Every call form and table captured in one graph at T, after an eager call at that
    shape, replayed on new data and positions: each output bitwise equal to an eager call
    on that data.

    The replay on the
    capture-time data must differ from the replay on new data for every form,
    so a replay that reused stale inputs would fail.
    """
    t = num_tokens
    gen = torch.Generator().manual_seed(10000 + t)
    calls = [(f, tn) for f, tn in _FORM_TABLES]
    data = {
        c: torch.randn(t, TARGET_FORMS[c[0]][0], TARGET_FORMS[c[0]][1], generator=gen).to(
            DEV, torch.bfloat16
        )
        for c in calls
    }
    pos = _positions(t, gen)
    # The engine runs each graph shape eagerly (its warm-up) before it captures it.
    for c in calls:
        num_heads, _, nope, inverse = TARGET_FORMS[c[0]]
        op(data[c], pos, _get_table(c[1]), num_heads, nope, ROPE, inverse, False)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for c in calls:
            num_heads, _, nope, inverse = TARGET_FORMS[c[0]]
            op(data[c], pos, _get_table(c[1]), num_heads, nope, ROPE, inverse, False)
    graph.replay()
    torch.cuda.synchronize()
    old = {c: v.clone() for c, v in data.items()}
    new = {c: torch.randn(v.shape, generator=gen).to(DEV, torch.bfloat16) for c, v in data.items()}
    new_pos = _positions(t, gen)
    for c in calls:
        data[c].copy_(new[c])
    pos.copy_(new_pos)
    graph.replay()
    torch.cuda.synchronize()
    for c in calls:
        num_heads, _, nope, inverse = TARGET_FORMS[c[0]]
        table = _get_table(c[1])
        eager = _run(new[c], new_pos, table, nope, inverse=inverse, is_neox=False)
        assert torch.equal(data[c], eager), (c, t)
        assert not torch.equal(data[c], old[c]), (c, t, "the replay did not read the new data")
        op.compare(
            data[c], _reference(new[c], new_pos, table, nope, inverse=inverse, is_neox=False)
        )


# ─── Every engine's table size, with positions at its far end ──
#
# The target builds both tables once per engine with md.max_seq_len + 64 rows,
# md.max_seq_len = Lm = L (DSpark off) or L + 14 (DSpark on), for each L in
# ENVELOPE_L. Positions reach about Lm + 5: the drafts of a verify step past a
# request's last token, and the drafter's block.
_ENGINE_TABLES = [(seq_len, dspark) for seq_len in ENVELOPE_L for dspark in (False, True)]
_E4_ROWS = 4096


def _far_end_positions(lm: int, rows: int) -> torch.Tensor:
    """Lm - 6 .. Lm + 5 (a full-length request's last tokens and the drafts past
    them), the table's last 64 rows, and 2^k - 1, 2^k below the table size."""
    far = set(range(lm - 6, lm + 6)) | set(range(rows - TABLE_PAD_ROWS, rows))
    far |= {v for k in range(10, 21) for v in (2**k - 1, 2**k)}
    return torch.tensor(sorted(p for p in far if 0 <= p < rows), dtype=torch.int32)


@pytest.mark.parametrize(
    "seq_len,dspark",
    _ENGINE_TABLES,
    ids=[f"L{s}-dspark_{'on' if d else 'off'}" for s, d in _ENGINE_TABLES],
)
def test_engine_table_far_end_positions(seq_len, dspark):
    """One engine's own tables (md.max_seq_len + 64 rows), built on the spot: every form and
    table call at positions over the whole table with its far end included, eager against the
    fp32 reference, and captured in one graph at 6G for G = 256 that replays bitwise equal
    to eager on new data and new positions."""
    lm = seq_len + (DSPARK_EXTRA_ROWS if dspark else 0)
    rows = lm + TABLE_PAD_ROWS
    tables = {}
    for name, (base, yarn) in _TABLE_SPECS.items():
        tables[name] = _table(base, yarn, max_pos=rows)
        # a shorter table is the first rows of the longest one, bitwise
        assert torch.equal(tables[name], _get_table(name)[:rows]), (name, rows)
    far = _far_end_positions(lm, rows)
    gen = torch.Generator().manual_seed(13000 + rows)

    def positions(num_tokens: int) -> torch.Tensor:
        """Random over [0, rows), with the far-end set at the start and, reversed, at the end."""
        pos = torch.randint(0, rows, (num_tokens,), generator=gen, dtype=torch.int32)
        pos[: far.numel()] = far
        pos[-far.numel() :] = far.flip(0)
        return pos.to(DEV)

    pos = positions(_E4_ROWS)
    far_rows = torch.cat(
        [torch.arange(far.numel()), torch.arange(_E4_ROWS - far.numel(), _E4_ROWS)]
    ).to(DEV)
    for form, table_name in _FORM_TABLES:
        num_heads, head_dim, nope, inverse = TARGET_FORMS[form]
        table = tables[table_name]
        x = torch.randn(_E4_ROWS, num_heads, head_dim, generator=gen).to(DEV, torch.bfloat16)
        out = _run(x, pos, table, nope, inverse=inverse, is_neox=False)
        ref = _reference(x, pos, table, nope, inverse=inverse, is_neox=False)
        op.compare(out, ref)
        op.compare(out[far_rows], ref[far_rows])
        if form == "q_h16" and table_name == "compressed_theta160k_yarn":
            # Control: the same gate rejects the far-end rows rotated at the neighbouring position.
            wrong = _reference(
                x, (pos - 1).clamp(min=0), table, nope, inverse=inverse, is_neox=False
            )
            close = torch.isclose(
                out[far_rows].float(), wrong[far_rows].float(), rtol=1.6e-2, atol=1e-5
            )
            frac_bad = 1.0 - close.float().mean().item()
            print(
                f"\n[L {seq_len}, dspark {dspark}] rows {rows}: far-end rows at position - 1: "
                f"{frac_bad:.4f} of elements outside the gate"
            )
            assert frac_bad > 0.01, frac_bad

    t = max(_CAPTURE_T)
    data = {
        c: torch.randn(t, TARGET_FORMS[c[0]][0], TARGET_FORMS[c[0]][1], generator=gen).to(
            DEV, torch.bfloat16
        )
        for c in _FORM_TABLES
    }
    cap_pos = positions(t)
    for c in _FORM_TABLES:
        num_heads, _, nope, inverse = TARGET_FORMS[c[0]]
        op(data[c], cap_pos, tables[c[1]], num_heads, nope, ROPE, inverse, False)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for c in _FORM_TABLES:
            num_heads, _, nope, inverse = TARGET_FORMS[c[0]]
            op(data[c], cap_pos, tables[c[1]], num_heads, nope, ROPE, inverse, False)
    new = {c: torch.randn(v.shape, generator=gen).to(DEV, torch.bfloat16) for c, v in data.items()}
    new_pos = positions(t)
    assert not torch.equal(new_pos, cap_pos)
    for c in _FORM_TABLES:
        data[c].copy_(new[c])
    cap_pos.copy_(new_pos)
    graph.replay()
    torch.cuda.synchronize()
    for c in _FORM_TABLES:
        num_heads, _, nope, inverse = TARGET_FORMS[c[0]]
        eager = _run(new[c], new_pos, tables[c[1]], nope, inverse=inverse, is_neox=False)
        assert torch.equal(data[c], eager), (c, rows)
        op.compare(
            data[c], _reference(new[c], new_pos, tables[c[1]], nope, inverse=inverse, is_neox=False)
        )


# The DeepSeek-V4.1 vision tower rotates q and k of its packed qkv projection in ONE call: the
# [P, 3072] bf16 output viewed as [3P, 16, 64] (row 3t = q of patch t, 3t+1 =
# k, 3t+2 = v), nope_dim 0, rope_dim 64, NeoX pairs, forward. The fp32 table is
# [P + 1, 2, 32]: rows 0..P-1 are the per-patch 2D RoPE cos/sin, row P is the
# identity (cos 1, sin 0). Positions are int32 (t, t, P) per patch, so q and k
# of patch t read row t and v reads the identity row and comes back bit-exact.
# P is every patch of one request: at most 17037 over the 900 MMMU val
# documents, so T = 3P reaches 51111.
_VIT_P_MAX = 17037
_VIT_T_MAX = 3 * _VIT_P_MAX
_VIT_HEADS, _VIT_HEAD_DIM = 16, 64
# The prototype's whole-tower CUDA graph ran at the golden cases' patch counts;
# 17037 is the MMMU maximum.
_VIT_GRAPH_P = (1517, 4680, 5325, 8772, 17037)


def _vit_table(num_patches: int, gen: torch.Generator) -> torch.Tensor:
    """[P + 1, 2, 32] fp32: random per-patch angles, then the identity row."""
    angles = torch.rand(num_patches, ROPE // 2, generator=gen) * 64.0
    table = torch.stack([angles.cos(), angles.sin()], dim=1)
    identity = torch.stack([torch.ones(ROPE // 2), torch.zeros(ROPE // 2)])[None]
    return torch.cat([table, identity]).contiguous().to(DEV)


def _vit_positions(num_patches: int) -> torch.Tensor:
    t = torch.arange(num_patches, dtype=torch.int32)
    return torch.stack([t, t, torch.full_like(t, num_patches)], dim=1).reshape(-1).to(DEV)


def _vit_call(data: torch.Tensor, pos: torch.Tensor, table: torch.Tensor) -> None:
    op(data, pos, table, _VIT_HEADS, 0, ROPE, False, True)


def test_vit_packed_qkv_every_row_count():
    """T = 1..51111 in the packed [3P, 16, 64] form: the full call is held to the fp32
    reference with every v row bitwise unchanged, and every prefix T is bitwise the
    first T rows of it; the guard row after T is never written."""
    gen = torch.Generator().manual_seed(12000)
    table = _vit_table(_VIT_P_MAX, gen)
    pos = _vit_positions(_VIT_P_MAX)
    x = torch.randn(_VIT_T_MAX + 1, _VIT_HEADS, _VIT_HEAD_DIM, generator=gen).to(
        DEV, torch.bfloat16
    )
    guard_row = x[_VIT_T_MAX].clone()
    x = x[:_VIT_T_MAX]
    full = _run(x, pos, table, 0, inverse=False, is_neox=True)
    ref = _reference(x, pos, table, 0, inverse=False, is_neox=True)
    op.compare(full, ref)
    assert torch.equal(full[2::3], x[2::3]), "a v row read through the identity row changed"
    exact_qk = float(
        (
            full.view(-1, 3, _VIT_HEADS, _VIT_HEAD_DIM)[:, :2]
            == ref.view(-1, 3, _VIT_HEADS, _VIT_HEAD_DIM)[:, :2]
        )
        .float()
        .mean()
    )
    print(
        f"\n[vit packed] T={_VIT_T_MAX}: q/k bit-exact {exact_qk:.6f} vs fp32 reference, v rows bitwise unchanged"
    )
    fb, gb = full.view(torch.int16), guard_row.view(torch.int16)
    buf = torch.empty(_VIT_T_MAX + 1, _VIT_HEADS, _VIT_HEAD_DIM, dtype=torch.bfloat16, device=DEV)
    mismatch = torch.zeros(_VIT_T_MAX + 1, dtype=torch.bool, device=DEV)
    guard = torch.zeros(_VIT_T_MAX + 1, dtype=torch.int64, device=DEV)
    differs = torch.empty(_VIT_T_MAX, _VIT_HEADS, _VIT_HEAD_DIM, dtype=torch.bool, device=DEV)
    for t in range(1, _VIT_T_MAX + 1):
        buf[:t].copy_(x[:t])
        buf[t].copy_(guard_row)
        _vit_call(buf[:t], pos[:t], table)
        mismatch[t] = torch.ne(buf[:t].view(torch.int16), fb[:t], out=differs[:t]).any()
        guard[t] = (buf[t].view(torch.int16) != gb).sum()
    bad = mismatch.nonzero().flatten().tolist()
    over = guard.nonzero().flatten().tolist()
    assert not bad, (
        f"{len(bad)} row counts differ from the {_VIT_T_MAX}-row result's rows, first {bad[:10]}"
    )
    assert not over, f"{len(over)} row counts wrote the guard row, first {over[:10]}"
    print(f"[vit packed] T = 1..{_VIT_T_MAX}: every prefix bitwise equal, guard row untouched")


def test_vit_packed_qkv_controls():
    """The same gates reject: v read at the last patch's row instead of the identity row,
    GPT-J pairs instead of NeoX, and a read one row off."""
    num_patches = 4680
    gen = torch.Generator().manual_seed(12200)
    table = _vit_table(num_patches, gen)
    pos = _vit_positions(num_patches)
    x = torch.randn(3 * num_patches, _VIT_HEADS, _VIT_HEAD_DIM, generator=gen).to(
        DEV, torch.bfloat16
    )
    good = _run(x, pos, table, 0, inverse=False, is_neox=True)
    wrong_pos = pos.clone()
    wrong_pos[2::3] = num_patches - 1
    wrong_v = _run(x, wrong_pos, table, 0, inverse=False, is_neox=True)
    v_changed = float((wrong_v[2::3] != x[2::3]).float().mean())
    ref = _reference(x, pos, table, 0, inverse=False, is_neox=True)
    gptj = _run(x, pos, table, 0, inverse=False, is_neox=False)
    outside = float(
        ((gptj.float() - ref.float()).abs() > 1e-5 + 1.6e-2 * ref.float().abs()).float().mean()
    )
    shifted = _run(x[1:], pos[1:], table, 0, inverse=False, is_neox=True)
    n_bad = float((shifted != good[:-1]).float().mean())
    print(
        f"\n[vit packed controls] v at row P-1: {v_changed:.4f} of v elements changed; "
        f"GPT-J pairs: {outside:.4f} outside the bf16 gate; one row off: {n_bad:.4f} differ"
    )
    assert v_changed > 0.5 and outside > 0.05 and n_bad > 0.5


def test_vit_packed_qkv_one_kernel_at_every_row_count():
    """Probe: the packed form launches the same single kernel at every T probed up to 51111."""
    gen = torch.Generator().manual_seed(12300)
    table = _vit_table(_VIT_P_MAX, gen)
    pos = _vit_positions(_VIT_P_MAX)
    data = torch.randn(_VIT_T_MAX, _VIT_HEADS, _VIT_HEAD_DIM, generator=gen).to(DEV, torch.bfloat16)
    counts = (
        1,
        2,
        3,
        4,
        63,
        64,
        65,
        4551,
        8191,
        8192,
        8193,
        14040,
        15975,
        16383,
        16384,
        16385,
        26316,
        32768,
        51109,
        51110,
        51111,
    )
    names = set()
    for t in counts:
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            _vit_call(data[:t], pos[:t], table)
            torch.cuda.synchronize()
        launched = [
            e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
        ]
        assert len(launched) == 1, (t, launched)
        names.add(launched[0])
    print(f"\n[vit packed] {len(counts)} row counts, kernels: {sorted(names)}")
    assert len(names) == 1 and "mlaRoPEInplaceKernel" in next(iter(names)), names


def test_vit_packed_qkv_cuda_graph_at_the_prototype_sizes():
    """Captured at T = 3P for the prototype's graph sizes; the replay reads the data, table and
    positions present at replay time and is bitwise equal to an eager call on them."""
    gen = torch.Generator().manual_seed(12400)
    pool = torch.cuda.graph_pool_handle()
    graphs = {}
    for p in sorted(_VIT_GRAPH_P, reverse=True):
        table = _vit_table(p, gen)
        pos = _vit_positions(p)
        data = torch.randn(3 * p, _VIT_HEADS, _VIT_HEAD_DIM, generator=gen).to(DEV, torch.bfloat16)
        _vit_call(data, pos, table)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool):
            _vit_call(data, pos, table)
        graphs[p] = (graph, data, pos, table)
    for p in _VIT_GRAPH_P:
        graph, data, pos, table = graphs[p]
        new = torch.randn(data.shape, generator=gen).to(DEV, torch.bfloat16)
        new_table = _vit_table(p, gen)
        data.copy_(new)
        table.copy_(new_table)
        graph.replay()
        torch.cuda.synchronize()
        eager = _run(new, pos, new_table, 0, inverse=False, is_neox=True)
        assert torch.equal(data, eager), p
        assert torch.equal(data[2::3], new[2::3]), p
        op.compare(data, _reference(new, pos, new_table, 0, inverse=False, is_neox=True))
    print(
        f"\n[vit packed] captured at P = {_VIT_GRAPH_P}: replays bitwise equal to eager on new data and tables"
    )
