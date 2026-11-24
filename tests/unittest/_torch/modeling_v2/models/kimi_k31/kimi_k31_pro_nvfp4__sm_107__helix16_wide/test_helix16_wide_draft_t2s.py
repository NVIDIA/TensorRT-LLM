# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""T2S, N6's drafter partial of ``kimi_k31_pro_nvfp4__sm_107__helix16_wide`` (CPU only, on the
sources; the kernels stood in by recorders): the kernels are the ones validated bitwise on GPU and
wait for their PDL predecessor first; the launch at N6's requests of every draft length and its
refusals; the drafter's bind; the workspace covers every row once; graph capture safety."""

from __future__ import annotations

import ast
import copy
import hashlib
from types import SimpleNamespace

import pytest
import torch
from _helix16_wide_cpu import family_dir, function_node, modeling_path, source_objects

pytestmark = pytest.mark.cpu_only

FAMILY_OPS = family_dir() / "ops/spec/rubin_ull_dflash2_headtp_decode.py"
#: sha256 of each kernel's ``ast.unparse`` (its name normalized, its docstring dropped): the kernel
#: text the bitwise validation ran (the same PTX and SASS but for the names).
VERIFIED = {
    "_partial_ranked_t2s": "e4229dfa75c4f3afdc8b56f089b6c5c01f6b8db76ed66c726544202a57086c7c",
    "_wuv_ranked_t2s": "b1c9f500e270c3adeb57aaee4e8eb061fcac1a7dcd9e3d159a3832dddbaae0cc",
}
HG = {4: 3, 5: 3, 6: 2, 8: 2}  # heads a stage-1 program packs: HG * block <= 16


def _kernel(name: str) -> ast.FunctionDef:
    fn = copy.deepcopy(function_node(modeling_path(), name))
    fn.name = "kernel"
    first = fn.body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
        fn.body = fn.body[1:]
    return fn


def test_t2s_kernels_are_the_verified_kernels_and_wait_first():
    for name, digest in VERIFIED.items():
        text = ast.unparse(_kernel(name))
        assert hashlib.sha256(text.encode()).hexdigest() == digest, name
        decorators = [ast.unparse(d) for d in function_node(modeling_path(), name).decorator_list]
        assert decorators == ["triton.jit"]
        assert [ast.unparse(s) for s in _kernel(name).body[:2]] == [
            "tl.extra.cuda.gdc_wait()",
            "tl.extra.cuda.gdc_launch_dependents()",
        ], name


class _Recorder:
    def __init__(self):
        self.calls = []

    def __getitem__(self, grid):
        return lambda *args, **kwargs: self.calls.append((grid, args, kwargs))


def _t2s_namespace():
    family = source_objects(FAMILY_OPS, ("RANKED_CP", "headtp_ranked_workspace"),
                            {"torch": torch, "SimpleNamespace": SimpleNamespace})  # fmt: skip
    ns = {
        "torch": torch, "Any": object, "RANKED_CP": family["RANKED_CP"],
        "RANKED_PAGE": family["RANKED_PAGE"], "headtp_ranked_workspace": family["headtp_ranked_workspace"],
        "_partial_ranked_t2s": _Recorder(), "_wuv_ranked_t2s": _Recorder(), "triton": None, "tl": None,
    }  # fmt: skip
    names = ("T2S_BM", "T2S_WARPS", "t2s_hg", "headtp_decode_ranked_t2s", "draft_ranked_workspace")
    return source_objects(modeling_path(), names, ns)


def _inputs(block: int, requests: int):
    """The drafter's arguments at one layer: query, the layer's ring rows (``pool.view(-1, 576)``
    of an FP8 pool), slots and context lengths, W_UV as ``v_b_t``'s transposed view."""
    rows = block * requests
    query = torch.empty(rows, 6, 576, dtype=torch.bfloat16)
    pool = torch.empty(2, 17, 1, 1, 64, 576, dtype=torch.float8_e4m3fn)[1][:, 0, 0]
    slots, starts = (
        torch.empty(requests, dtype=torch.int64),
        torch.empty(requests, dtype=torch.int64),
    )
    kv_b = torch.empty(6 * 128 * 2, 512, dtype=torch.bfloat16)
    v_b_t = kv_b[6 * 128 :].view(6, -1, 512).transpose(1, 2)
    return query, pool.view(-1, 576), slots, starts, v_b_t


def test_t2s_launch_at_n6s_requests():
    """At every draft length's N6 requests: stage 1 over (16 page groups, head groups, requests)
    with the arguments the partial took and stage 2 over head-major 64-row pairs, both PDL at 8
    warps; the views returned are the workspace's."""
    ns = _t2s_namespace()
    s1, s2 = ns["_partial_ranked_t2s"], ns["_wuv_ranked_t2s"]
    tiles, scale, kv_scale = (64, 64, 8, 1, None), 0.1352337788608801, 1.0
    for block, top in ((8, 16), (6, 21), (5, 25), (4, 32)):
        assert ns["t2s_hg"](block) == HG[block]
        rws = ns["draft_ranked_workspace"](128 // block, block, 6, 512, "cpu")
        for b in range(9, top + 1):
            query, ring, slots, starts, v_b_t = _inputs(block, b)
            s1.calls.clear()
            s2.calls.clear()
            part, lse = ns["headtp_decode_ranked_t2s"](
                query, ring, slots, starts, 1088, scale, kv_scale, 1024, v_b_t, rws, tiles, True, 64
            )
            (grid1, args1, kw1), (grid2, args2, kw2) = s1.calls + s2.calls
            nrows_h = 16 * b * block
            assert grid1 == (16, 6 // HG[block], b)
            assert all(
                x is y for x, y in zip(args1[:6], (query, ring, slots, starts, rws.o16, rws.lse))
            )
            assert args1[6:] == (scale * kv_scale, kv_scale, 1088, b) and len(args1) == 10
            assert kw1 == dict(W=1024, BLOCK=block, HEADS=6, HG=HG[block], BR=16, BK=64, BC=64, LAT=512,
                               ROPE=64, PAGE=64, CP=16, num_warps=8, launch_pdl=True, maxnreg=64)  # fmt: skip
            assert grid2 == (-(-nrows_h // 64), 6) and (grid2[0] - 1) * 64 < nrows_h
            assert all(x is y for x, y in zip(args2[:3], (rws.o16, v_b_t, rws.part16)))
            assert args2[3:] == (nrows_h, 65536, 1, 512)
            assert kw2 == dict(HEADS=6, BM=32, BC=64, LAT=512, V=128, num_warps=8, launch_pdl=True)
            assert part.shape == (1, 16, block * b, 6, 128) and lse.shape == (16, block * b, 6)
            assert part.data_ptr() == rws.part16.data_ptr() and lse.data_ptr() == rws.lse.data_ptr()
    # T2S replaces N6 (one split, rolled, 64 x 64, 8 warps) at its pre-conditions only.
    query, ring, slots, starts, v_b_t = _inputs(5, 9)
    rws = ns["draft_ranked_workspace"](25, 5, 6, 512, "cpu")
    good = (query, ring, slots, starts, 1088, scale, kv_scale, 1024, v_b_t, rws, tiles, True, 64)
    for at, bad in ((10, (64, 64, 8, 2, None)), (11, False), (8, v_b_t.contiguous()), (4, 1000)):
        with pytest.raises(AssertionError):
            ns["headtp_decode_ranked_t2s"](*good[:at], bad, *good[at + 1 :])
    small = SimpleNamespace(part16=rws.part16, lse=rws.lse, o16=rws.o16[: 6 * 16 * 9 * 5 - 1])
    with pytest.raises(AssertionError):
        ns["headtp_decode_ranked_t2s"](*good[:9], small, *good[10:])


def test_the_drafter_takes_t2s_at_n6():
    attention = ast.unparse(function_node(modeling_path(), "_draft_attention_headtp"))
    call = attention[attention.index("if rolled or maxnreg:") :]
    assert call.index("headtp_decode_ranked_t2s(query, pool.view(-1, latent), step.slots") < call.index(
        "return headtp_ranked_lse_merge(part, lse, st.block)"
    ) < call.index("headtp_decode_ranked(")  # fmt: skip
    assert "flags += f' t2s=hg{t2s_hg(st.block)}.w2' if rolled else ''" in attention
    assert "headtp_decode_ranked_n6" not in modeling_path().read_text()
    assert "_partial_ranked_rolled" not in modeling_path().read_text()


def test_t2s_workspace_holds_every_row_once():
    """o16: one 512-column BF16 row per ``part16`` row, 12.6 MB at 128 rows. Stage 1's o16 rows
    (head-major) and LSE entries, stage 2's o16 reads and ``part16`` rows each cover their range
    once at every block and request count; stage 2's part row is the partial's."""
    ns = _t2s_namespace()
    for block in HG:
        rws = ns["draft_ranked_workspace"](128 // block, block, 6, 512, "cpu")
        assert rws.o16.dtype == torch.bfloat16 and rws.o16.shape == (rws.part16.shape[0], 512)
        assert rws.o16.shape[0] == 6 * 16 * (128 // block) * block
        assert rws.o16.numel() * 2 <= 6 * 16 * 128 * 512 * 2 == 12582912
    for block, requests in ((4, 32), (5, 25), (6, 21), (8, 9)):
        hg, nq, nrows_h = HG[block], 6 * block, 16 * requests * block
        r, hgi, g, rho = torch.meshgrid(torch.arange(16), torch.arange(6 // hg),
                                        torch.arange(requests), torch.arange(16), indexing="ij")  # fmt: skip
        j, hh = rho % block, hgi * hg + rho // block
        real = (rho < hg * block) & (hh < 6)
        orow = (hh * nrows_h + (r * requests + g) * block + j)[real]
        assert torch.equal(orow.sort().values, torch.arange(6 * nrows_h))
        lse = ((r * requests + g) * nq + j * 6 + hh)[real]
        assert torch.equal(lse.sort().values, torch.arange(6 * nrows_h))
        t, h, i = torch.meshgrid(torch.arange(-(-nrows_h // 64)), torch.arange(6), torch.arange(64),
                                 indexing="ij")  # fmt: skip
        rows = 2 * t * 32 + i  # tile 2t's rows, then 2t + 1's
        read = rows < nrows_h
        assert torch.equal((h * nrows_h + rows)[read].sort().values, torch.arange(6 * nrows_h))
        part = (rows * 6 + h)[read]
        assert torch.equal(part.sort().values, torch.arange(6 * nrows_h))
        # the partial's part row (r * groups + g) * NQ + j * HEADS + h at R = (r * groups + g) * block + j
        rg, jj = rows // block, rows % block
        assert torch.equal((rg * nq + jj * 6 + h)[read], part)


def test_t2s_is_capture_safe():
    """The workspace is the runtime's (bound before capture, one for the five layers); the launch
    allocates nothing, reads nothing back to the host and returns the workspace's views."""
    runtime = ast.unparse(function_node(modeling_path(), "DraftModel._headtp_runtime"))
    assert (
        "headtp=draft_ranked_workspace(max_rows // block, block, heads, latent, device)" in runtime
    )
    assert "if positions + 2 * RANKED_CP * RANKED_PAGE >= 2 ** 31:" in runtime  # int32 positions
    launcher = function_node(modeling_path(), "headtp_decode_ranked_t2s")
    banned = {"empty", "zeros", "ones", "full", "new_empty", "new_zeros", "empty_like", "zeros_like",
              "clone", "contiguous", "item", "cpu", "tolist", "synchronize", "to", "copy_"}  # fmt: skip
    called = {
        n.func.attr for n in ast.walk(launcher)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }  # fmt: skip
    assert called and not called & banned, called & banned
