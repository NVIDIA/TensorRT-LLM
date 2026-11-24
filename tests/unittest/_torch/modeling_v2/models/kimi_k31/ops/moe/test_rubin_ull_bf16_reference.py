# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The BF16 low-latency MoE entries' host side on the CPU: rank geometry, tactics, Lamport offsets,
references, guards, and the push kernel's epilogue against the base kernel's. Their cells run over
the TP group in rubin_multirank/test_rubin_ull_bf16_ranks.py; the kernel on one GPU in
test_rubin_ull_push_gemm.py."""

import ast
import re
from pathlib import Path
from types import SimpleNamespace

import _rubin_ull_reference as uref  # noqa: E402 -- the test-side references next to this file
import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe import (
    _rubin_ull_common as c,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe.rubin_ull_cp_tail import (  # noqa: E501
    rubin_ull_cp_tail,
    rubin_ull_cp_tail_share,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.moe.rubin_ull_front_bf16 import (  # noqa: E501
    rubin_ull_front_bf16,
    rubin_ull_front_share,
)

_SM107 = torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7)
pytestmark = [
    pytest.mark.cpu_only,
    pytest.mark.skipif(torch.cuda.is_available() and not _SM107, reason="certified on SM107 only"),
]

_FAMILY = Path(c.__file__).resolve().parents[2]
#: The ULL BF16 kernel and the modules of its entries.
_ULL_FILES = (
    _FAMILY / "kernels/moe/rubin_ull_push_gemm.py",
    _FAMILY / "ops/moe/_rubin_ull_common.py",
    _FAMILY / "ops/moe/rubin_ull_cp_tail.py",
    _FAMILY / "ops/moe/rubin_ull_front_bf16.py",
)


def test_tep8_shares():
    """TP8: latent 4 blocks on ranks 0-3 and 3 on 4-7; hidden 7 blocks each."""
    latent = [c.block_slice(c.LATENT, 8, r) for r in range(8)]
    assert latent == [
        (0, 512),
        (512, 512),
        (1024, 512),
        (1536, 512),
        (2048, 384),
        (2432, 384),
        (2816, 384),
        (3200, 384),
    ]
    assert [c.block_slice(c.HIDDEN, 8, r) for r in range(8)] == [(896 * r, 896) for r in range(8)]
    assert [uref.front_slot_cols(w) for w in (2, 4, 8)] == [1792, 896, 512]


@pytest.mark.parametrize("world", [2, 3, 4, 8])
@pytest.mark.parametrize("extent", [c.LATENT, c.HIDDEN])
def test_block_slices_tile_the_extent(world, extent):
    cuts = [c.block_slice(extent, world, r) for r in range(world)]
    assert cuts[0][0] == 0 and sum(n for _, n in cuts) == extent
    assert all(a + n == b for (a, n), (b, _) in zip(cuts, cuts[1:]))
    assert all(n % c.BLOCK == 0 for _, n in cuts)
    assert max(n for _, n in cuts) - min(n for _, n in cuts) <= c.BLOCK
    if extent == c.LATENT:
        assert max(n for _, n in cuts) == uref.front_slot_cols(world)
    with pytest.raises(ValueError):
        c.block_slice(extent, world, world)


def test_smem_formula():
    """FlashInfer's allocator layout, by hand: (64, 8, 4, 12) needs 227,592 bytes, the push
    kernel's staging tile 1,024 more after 16-byte alignment."""
    assert c.gemm_smem_bytes((64, 8, 4, 12)) == 227592
    assert c.push_smem_bytes((64, 8, 4, 12)) == 228624
    assert c.gemm_smem_bytes((128, 8, 1, 6)) == 209020


@pytest.mark.parametrize("m", [1, 8, 16, 17, 32, 33, 48, 54, 64, 65, 66, 72])
@pytest.mark.parametrize("n,k", [(512, 7168), (384, 7168), (896, 3584), (1792, 3584)])
def test_push_tactic_serves_every_call(m, n, k):
    t = c.push_tactic(m, n, k)
    assert c.tactic_ok(t, m, n, k)
    # The smallest kernel-N that holds the rows, up to 32; past 32 rows N tiles of 32.
    assert t[1] == next((v for v in (8, 16, 32) if m <= v), 32)
    assert t[3] <= k // c.CTA_K // t[2]
    cands = uref.tactic_candidates(m, n, k)
    assert cands and len(set(cands)) == len(cands) and all(c.tactic_ok(x, m, n, k) for x in cands)


def test_push_tactic_at_r():
    """TEP8 b1 k7 (8 rows): split-K 4 on latent_down's 4 / 3 tiles and latent_up's 7 per rank."""
    assert c.push_tactic(8, 512, 7168)[:3] == (64, 8, 4)
    assert c.push_tactic(8, 896, 3584)[:3] == (64, 8, 4)
    with pytest.raises(ValueError):
        c.push_tactic(73, 896, 3584)
    assert not c.tactic_ok((64, 8, 3, 4), 8, 896, 3584)  # 28 K tiles do not split in 3
    assert not c.tactic_ok((64, 8, 3, 4), 8, 512, 7168)  # 56 K tiles do not split in 3
    assert not c.tactic_ok((64, 128, 4, 2), 72, 896, 3584)  # kernel-N stops at 64


def test_push_tactic_past_32_rows():
    """helix8's k5 verify rows past 32 (b6 to b12, 36 to 72 rows) over TP8: N tiles of 32 rows
    (kernel-N 32, two or three tiles), split-K 4 on latent_up's 7 blocks, one stage per K tile of
    a CTA (7). Kernel-N 64 (one tile up to 64 rows) is served for an explicit tactic: 5 stages
    fit beside its split-K mailbox and staging tile."""
    for m in (33, 36, 48, 54, 60, 64, 65, 66, 72):
        assert c.push_tactic(m, 896, c.LATENT) == (64, 32, 4, 7)
    for m in (33, 54, 64, 66, 72):
        assert c.tactic_ok((64, 64, 4, 5), m, 896, c.LATENT)
    assert c.push_smem_bytes((64, 64, 4, 5)) <= c.SMEM_CAPACITY < c.push_smem_bytes((64, 64, 4, 6))


def test_cp_tail_share():
    """Each TP8 rank's 7 blocks of latent_up's rows, and a tactic for every row count; the
    tactics at 1 to 32 rows are the ones the SM107 tactic sweep picked or confirmed."""
    for rank in range(8):
        c0, n, tactics = rubin_ull_cp_tail_share(8, rank)
        assert (c0, n) == (896 * rank, 896)
        assert len(tactics) == c.MAX_M + 1 == 73 and tactics[0] is None
        assert all(tactics[m] == c.push_tactic(m, 896, c.LATENT) for m in range(1, c.MAX_M + 1))
    _, _, tactics = rubin_ull_cp_tail_share(8, 0)
    assert (tactics[1], tactics[8], tactics[16], tactics[32]) == (
        (64, 8, 4, 7),
        (64, 8, 4, 7),
        (64, 16, 4, 7),
        (64, 32, 4, 7),
    )
    assert len(rubin_ull_cp_tail_share(4, 1, 16)[2]) == 17
    assert len(rubin_ull_cp_tail_share(8, 0, 32)[2]) == 33  # TEP8's 32-row tail
    with pytest.raises(ValueError):
        rubin_ull_cp_tail_share(8, 0, 73)


def test_tail_workspace_at_72_rows():
    """helix8's tail call of 72 rows over TP8: 72 x (8 + 1) rows of the hidden size per buffer,
    rounded up to whole MiB (9 MiB), three buffers (27 MiB)."""
    need = c.required_bytes(72, c.HIDDEN, 8, True, ag=True)
    assert need == 72 * 7168 * 9 * 2 == 9289728
    assert -(-need // (1 << 20)) << 20 == 9 << 20


def test_front_share():
    """Each TP8 rank's latent_down tiles (4 on ranks 0-3, 3 on 4-7) and a tactic for every row
    count; rank 0's tactics at 1 to 32 rows are the ones the SM107 tactic sweep picked or
    confirmed. The front stays at 32 rows (its route kernel's ``M_MAX``)."""
    for rank in range(8):
        r0, n, tactics = rubin_ull_front_share(8, rank)
        assert (r0, n) == c.block_slice(c.LATENT, 8, rank)
        assert len(tactics) == c.FRONT_MAX_M + 1 == 33 and tactics[0] is None
        assert all(tactics[m] == c.push_tactic(m, n, c.HIDDEN) for m in range(1, c.FRONT_MAX_M + 1))
    _, _, tactics = rubin_ull_front_share(8, 0)
    assert (tactics[1], tactics[8], tactics[16], tactics[32]) == (
        (64, 8, 4, 12),
        (64, 8, 4, 12),
        (64, 16, 4, 10),
        (64, 32, 4, 8),
    )
    with pytest.raises(ValueError):
        rubin_ull_front_share(8, 0, 33)


def test_lamport_offsets():
    """The kernel's addresses (kernels/comm/rubin_lamport_ar.py push_slot / ag_row_addr)."""
    w, row, nb = 8, 7168, 1 << 20
    assert uref.slot_offset(3, 2, w, row, 16, True, 1, nb) == (-1, nb + (3 * w + 2) * row * 2 + 32)
    assert uref.slot_offset(11, 2, w, row, 16, False, 1, nb) == (3, nb + (1 * w + 2) * row * 2 + 32)
    assert uref.ag_offset(8, 3, w, row, 16, 2, nb) == 2 * nb + (8 * w + 3) * row * 2 + 32
    assert c.required_bytes(32, row, w, True, ag=True) == 32 * row * 9 * 2
    assert c.required_bytes(9, row, w, False) == 2 * 2 * w * row * 2


def _exact(rows, cols, seed, scale=1.0):
    """Small integers times a power of two: every FP32 sum below is exact."""
    g = torch.Generator().manual_seed(seed)
    return (torch.randint(-3, 4, (rows, cols), generator=g).float() * scale).to(torch.bfloat16)


def test_push_payload_rounds_once_and_sanitizes():
    x, w = _exact(4, 256, 1), _exact(64, 256, 2, 0.25)
    assert torch.equal(c.push_payload(x, w), (x.float() @ w.float().t()).to(torch.bfloat16))
    v = torch.tensor([-0.0, 0.0, -1.5, 2.0], dtype=torch.bfloat16)
    s = c.sanitize_bf16(v)
    assert not torch.signbit(s[:2]).any() and torch.equal(s[2:], v[2:])


def test_cp_tail_is_the_replicated_tail():
    """With exact sums the column-parallel tail equals bf16(latent_up + bf16(all-reduced shared))
    of the replicated latent_up, bit for bit."""
    world = 4
    latent, up = _exact(8, c.LATENT, 4, 0.125), _exact(c.HIDDEN, c.LATENT, 5, 2.0**-6)
    shared = [_exact(8, c.HIDDEN, 6 + r, 0.5) for r in range(world)]
    got = uref.cp_tail(latent, up, shared, cpp_one_shot=True)
    y = (latent.float() @ up.float().t()).to(torch.bfloat16)
    sh = sum(s.float() for s in shared).to(torch.bfloat16)
    assert torch.equal(got, (y.float() + sh.float()).to(torch.bfloat16))


def test_tail_finish_in_the_mix_is_the_replicated_fold():
    """The next mix finishes the tail's call (``finish_ag``, the round-to-nearest add): its
    updated residual is ``bf16(prefix + bf16(latent_up + bf16(all-reduced shared)))``, the
    replicated tail's fused MoE residual update; the finish kernel and a torch add give the same
    bits."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.comm.rubin_lamport_push import (  # noqa: E501
        rubin_lamport_finish,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.ops.norm.rubin_attn_res_lamport_add_rmsnorm import (  # noqa: E501
        rubin_attn_res_lamport_add_rmsnorm,
    )

    g = torch.Generator().manual_seed(21)
    world, rows = 8, 8

    def mk(*shape, scale=1.0):
        return (torch.randn(*shape, generator=g) * scale).to(torch.bfloat16)

    prefix, y = mk(rows, c.HIDDEN, scale=4.0), mk(rows, c.HIDDEN)
    shared = [mk(rows, c.HIDDEN, scale=0.5) for _ in range(world)]
    bank = mk(2, rows, c.HIDDEN)
    proj = mk(c.HIDDEN, scale=0.1)
    res_norm = (1.0 + mk(c.HIDDEN, scale=0.1).float()).to(torch.bfloat16)
    out_norm = (1.0 + mk(c.HIDDEN, scale=0.1).float()).to(torch.bfloat16)
    sh = uref.allreduce(shared, cpp_one_shot=True)
    fold = (prefix.float() + (y.float() + sh.float()).to(torch.bfloat16).float()).to(torch.bfloat16)
    chunk = c.reduce_chunk(world, True)
    updated, _ = rubin_attn_res_lamport_add_rmsnorm.reference(
        prefix, shared, bank, proj, res_norm, out_norm, 1e-5, 1e-5, chunk, True, y
    )
    assert torch.equal(updated, fold)
    finished = rubin_lamport_finish.reference(shared, True, True, ag_rows=y)
    assert torch.equal(prefix + finished, fold)
    assert torch.equal(finished, uref.ag_finish(y, shared, cpp_one_shot=True))


def _ws(world, rank, nbytes=1 << 23):
    return SimpleNamespace(world=world, rank=rank, bytes_per_buffer=nbytes)


def test_cp_tail_guard():
    latent = torch.zeros(8, c.LATENT, dtype=torch.bfloat16)
    w = torch.zeros(896, c.LATENT, dtype=torch.bfloat16)
    rubin_ull_cp_tail.is_valid(latent, w, _ws(8, 1), (64, 8, 4, 7), c0=896)
    for bad in (
        dict(ws=_ws(8, 2), c0=896),
        dict(ws=_ws(8, 1), c0=0),
        dict(ws=_ws(8, 1, 1 << 16), c0=896),
    ):
        with pytest.raises(AssertionError):
            rubin_ull_cp_tail.is_valid(latent, w, bad["ws"], (64, 8, 4, 7), c0=bad["c0"])
    with pytest.raises(AssertionError):
        rubin_ull_cp_tail.is_valid(
            torch.zeros(73, c.LATENT, dtype=torch.bfloat16), w, _ws(8, 1), (64, 8, 4, 7), c0=896
        )
    ref = rubin_ull_cp_tail.reference(
        _exact(8, c.LATENT, 1), _exact(896, c.LATENT, 2, 0.25), None, None, c0=896
    )
    assert ref.shape == (8, 896)


def test_front_guard_and_rows():
    down = torch.zeros(c.LATENT, c.HIDDEN, dtype=torch.bfloat16)
    r0, n = c.block_slice(c.LATENT, 8, 5)
    rows = down[r0 : r0 + n]
    assert rows.shape == (384, c.HIDDEN) and rows.data_ptr() == down[2432].data_ptr()
    x = torch.zeros(8, c.HIDDEN, dtype=torch.bfloat16)
    logits = torch.zeros(8, c.NUM_EXPERTS)
    views = (torch.zeros(32, c.NUM_EXPERTS), torch.zeros(32, c.NUM_EXPERTS))
    args = (x, logits, rows, views, torch.zeros(c.NUM_EXPERTS), 1.0, torch.ones(1), 560, 112)
    rubin_ull_front_bf16.is_valid(*args, _ws(8, 5), (64, 8, 4, 12))
    with pytest.raises(AssertionError):
        rubin_ull_front_bf16.is_valid(*args, _ws(8, 1), (64, 8, 4, 12))
    with pytest.raises(AssertionError):
        rubin_ull_front_bf16.is_valid(x, logits[:4], *args[2:], _ws(8, 5), (64, 8, 4, 12))


def test_front_reference_latent():
    """latent_down row j reads hidden column 2j: the BF16 latent is exact, so its NVFP4 too."""
    g = torch.Generator().manual_seed(11)
    grid = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
    sign = torch.randint(0, 2, (3, c.LATENT), generator=g) * 2 - 1
    even = grid[torch.randint(0, 8, (3, c.LATENT), generator=g)] * sign
    even[:, ::16] = 6.0
    x = torch.randn(3, c.HIDDEN, generator=g)
    x[:, 0::2] = even
    down = torch.zeros(c.LATENT, c.HIDDEN)
    down[torch.arange(c.LATENT), 2 * torch.arange(c.LATENT)] = 1.0
    logits = torch.randn(3, c.NUM_EXPERTS, generator=g)
    bias = (torch.rand(c.NUM_EXPERTS, generator=g) - 0.5) * 0.19
    routing, latent = rubin_ull_front_bf16.reference(
        x.bfloat16(), logits, down.bfloat16(), bias, 1.0, 1.0, 112, 112
    )
    assert latent.shape == (3, c.LATENT) and torch.equal(latent, even)
    assert routing[0].shape == (3, 16)


def test_cells_fit_their_trays():
    """The front's cells fit one tray (2 or 4 ranks) and 32 rows; the tail's also cover helix8's
    MoE TP8 group (two trays) at its k5 verify rows past 32, up to 72."""
    assert rubin_ull_front_bf16.CELLS
    assert {cell.spec["world"] for cell in rubin_ull_front_bf16.CELLS} <= {2, 4}
    assert all(1 <= cell.spec["rows"] <= c.FRONT_MAX_M for cell in rubin_ull_front_bf16.CELLS)
    cells = rubin_ull_cp_tail.CELLS
    assert {cell.spec["world"] for cell in cells} == {2, 4, 8}
    assert all(1 <= cell.spec["rows"] <= c.MAX_M for cell in cells)
    for world in (4, 8):
        rows = {cell.spec["rows"] for cell in cells if cell.spec["world"] == world}
        assert {33, 48, 54, 66, 72} <= rows, (world, rows)


def _func(path, cls, name):
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls)
    return next(n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == name)


def test_epilogue_differs_from_the_base_only_in_the_push():
    """The push kernel's epilogue is the base kernel's (TMEM load, split-K reduction, BF16
    rounding) up to the owner's staging; only the plain store and the push loop differ."""
    base = _func(
        _FAMILY / "kernels/spec/rubin_draft_gemm_push.py",
        "SplitKArPushGemmKernel",
        "push_epilog_warp",
    )
    mine = _func(
        _FAMILY / "kernels/moe/rubin_ull_push_gemm.py", "UllPushGemmKernel", "push_epilog_warp"
    )
    assert [a.arg for a in base.args.args] == [a.arg for a in mine.args.args]

    def split(fn):
        i = next(
            i
            for i, s in enumerate(fn.body)
            if isinstance(s, ast.If)
            and "OWNER_RANK" in ast.unparse(s.test)
            and "!=" not in ast.unparse(s.test)
            and s is fn.body[-1]
        )
        return fn.body[:i], fn.body[i].body

    (b_pre, b_owner), (m_pre, m_owner) = split(base), split(mine)
    body = [
        s for s in b_pre if not isinstance(s, ast.Expr) or not isinstance(s.value, ast.Constant)
    ]
    mbody = [
        s for s in m_pre if not isinstance(s, ast.Expr) or not isinstance(s.value, ast.Constant)
    ]
    assert [ast.dump(s) for s in body] == [ast.dump(s) for s in mbody]
    assert ast.dump(b_owner[0]) == ast.dump(m_owner[0])  # rD.store(rAcc.load().to(c_dtype))


def test_no_op_registration_or_env_in_the_ull_files():
    """No torch op is registered and no environment variable is read, anywhere in these files;
    the entry modules call no torch op."""
    pat = re.compile(r"torch\.library|TORCH_LIBRARY|register_fake|custom_op|os\.environ|getenv")
    for path in _ULL_FILES:
        assert not pat.search(path.read_text()), path.name
    for path in _ULL_FILES[1:]:
        assert "torch.ops." not in path.read_text(), path.name
