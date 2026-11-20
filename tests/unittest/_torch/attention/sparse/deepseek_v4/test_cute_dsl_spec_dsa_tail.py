# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""specDSA tail attention (``tail_op``, CuTe DSL, SM100) against an fp64 reference of the decode sparse MLA.

The tail attention takes the exact top-512 selection of any top-K kernel (here an input, so the tests also feed
selections the real top-K never produces: -1 padded, fewer than 512 positions, a whole tile of -1, no selection) and
runs the sparse MLA over the sliding window plus the selected compressed rows with the attention sink. One cluster of
5 CTAs serves a row, up to 26 rows (the clusters of 5 that fit one wave of the 148 SMs; clusters of 4 and 3 for 27 to 45
rows are not offered: they leave only 0.5 to 0.8 % of numerical margin and win nothing over the large-batch kernels).
The attention does not depend on the indexer K dtype: one compile serves both specs (FP8 and FP4), and both are
launched and checked. Rows with a window shorter than 128 rows (kv_len 1, 127, 128, 129, ...) and rows with fewer than
512 compressed positions are part of every mixture; the reference takes the window as the last ``min(128, kv_len)``
positions and ignores the -1 entries.

Three attention modes, each with the bar of its kind:

- ``realistic``: random queries and KV rows, calc_diff < 1e-3 as in the other specDSA tests. The FP8 quantization of P
  (E4M3 at the running maximum) sets the floor of this mode: 2.5e-4 on this data (the real-capture gates, whose
  attention is more peaked, run at 6.0e-5).
- ``gather_exact``: every score of a row is bit-equal, so P is exact and one wrong, missing or unmasked KV row is
  visible far above the bf16 noise (calc_diff < 2e-5 on the 448 columns that carry the KV rows; measured 3e-6).
- ``pow2``: with the softmax scale ln 2 every score is an integer number of bits (a query with one non-zero element,
  KV rows with an element that is -2 m), so every softmax weight is an exact power of two and P is exact in E4M3 *with
  different row maxima in the tiles of a row*: the combine of the cluster (the weights 2^(m_i - m*) of the partial
  outputs) is checked at the tight bar too (calc_diff < 2e-5 on all columns; only the bf16 roundings remain).
"""

import dataclasses
import inspect
import math

import pytest
import torch

from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa.config import DS_V4_FLASH as SPEC
from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa.config import (
    DS_V4_FLASH_FP4 as SPEC_FP4,
)
from tensorrt_llm._utils import get_sm_version

from . import test_cute_dsl_spec_dsa_large_batch as H

K_TOP, H_ATT, D_ATT = SPEC.topk, SPEC.attn_heads, SPEC.kv_dim
TPB_C, TPB_SWA, WINDOW, RATIO = (
    SPEC.kv_page_rows,
    SPEC.swa_page_rows,
    SPEC.window,
    SPEC.compress_ratio,
)
SM_SCALE = SPEC.default_softmax_scale
LN2 = math.log(2.0)
# An indexer shape without kernels: top-1024 instead of top-512.
PRO = dataclasses.replace(SPEC, name="DS-V4-Pro", topk=1024)

# The tail attention serves 1 to 26 rows with one cluster of 5 CTAs per row; 27 rows and more are refused.
MAX_ROWS, CLUSTER = 26, 5
BATCHES = [1, 2, 5, 8, 16, 24, 25, 26]
REFUSED = [27, 33, 34, 45, 46, 148, 1024]
ATTN_MODES = ["realistic", "gather_exact", "pow2"]
CD_REALISTIC = 1e-3
CD_EXACT = 2e-5


def _scale_of(attn):
    return LN2 if attn == "pow2" else SM_SCALE


def _tail():
    from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa import tail_op

    return tail_op


def _supported() -> bool:
    return torch.cuda.is_available() and get_sm_version() == 100 and _tail().is_supported()[0]


requires_b200 = pytest.mark.skipif(
    not _supported(), reason="clusters of 5 CTAs on 148 SMs, qualified on B200 only"
)


def _spec_of(name):
    return SPEC_FP4 if name == "fp4" else SPEC


@pytest.fixture(params=["fp4", "fp8"])
def spec_name(request):
    """The spec under test: the attention is the same, one compile serves both."""
    return request.param


# (kv_len, selection pattern). nc = kv_len // 4 compressed positions exist. The patterns:
#   dense: every position that exists (at most 512) at random slots of the 512 entries;
#   holes: dense with a quarter of the entries replaced by -1;
#   short: at most 300 positions at the front, the rest -1;
#   tile_gap: dense with one whole 128-entry tile of -1;
#   sorted: dense in ascending order;
#   one / last: a single position at slot 255 / 511;
#   none: no selection, the window and the sink only.
ROW_KINDS = [
    (1048579, "dense"),  # 2**18 compressed positions, the longest row
    (131075, "holes"),
    (262147, "tile_gap"),
    (
        129,
        "dense",
    ),  # a window of 128 rows over two pages (positions 1 to 128), 32 compressed positions
    (1, "none"),  # a window of one row, no compressed position
    (100003, "dense"),
    (127, "dense"),  # a window of 127 rows, 31 compressed positions
    (65539, "short"),
    (128, "holes"),  # a window of exactly 128 rows (positions 0 to 127)
    (8193, "sorted"),
    (257, "dense"),  # a window from position 129: two pages
    (2, "none"),  # a window of two rows
    (4100, "dense"),
    (256, "dense"),  # a window of exactly one page (positions 128 to 255)
    (2051, "one"),
    (513, "dense"),  # 128 compressed positions: fewer than 512 exist
    (100000, "last"),
    (4, "one"),  # a window of 4 rows, a single compressed position
    (40001, "tile_gap"),
    (255, "dense"),
]


def _rows(batch, rot=0):
    return [ROW_KINDS[(b + rot) % len(ROW_KINDS)] for b in range(batch)]


def _selection(pattern, nc, gen):
    """int32 [512]: local compressed positions of one row in no particular order, -1 for an unused entry."""
    sel = torch.full((K_TOP,), -1, dtype=torch.int32)
    take = min(nc, K_TOP)
    if pattern == "none" or take == 0:
        return sel
    pos = torch.randperm(nc, generator=gen)[:take].to(torch.int32)
    if pattern == "sorted":
        sel[:take] = torch.sort(pos).values
    elif pattern == "short":
        m = min(take, 300)
        sel[:m] = pos[:m]
    elif pattern == "one":
        sel[255] = pos[0]
    elif pattern == "last":
        sel[K_TOP - 1] = pos[0]
    else:  # dense, holes, tile_gap: at random slots
        sel[torch.randperm(K_TOP, generator=gen)[:take]] = pos
        if pattern == "holes":
            sel[torch.rand(K_TOP, generator=gen) < 0.25] = -1
        elif pattern == "tile_gap":
            tile = int(torch.randint(0, K_TOP // 128, (1,), generator=gen))
            sel[tile * 128 : (tile + 1) * 128] = -1
    return sel


@torch.no_grad()
def _make_case(rows, seed, attn="realistic"):
    """Attention inputs for ``len(rows)`` rows, rows = [(kv_len, pattern), ...], and the selections [B, 512].

    Rows with the same kv_len share their compressed KV pages (random physical page ids); every row has its own
    window pages. The pages a row does not read (window rows after its last position, the spare pages) hold
    poison: the largest values, so that a row that is not masked shows up in the output.
    """
    dev = torch.device("cuda")
    batch = len(rows)
    gh = torch.Generator().manual_seed(seed)
    gd = torch.Generator(device=dev).manual_seed(seed + 1)

    def rnd(*shape):
        return torch.randn(*shape, generator=gd, device=dev)

    kv_lens = [kv for kv, _ in rows]
    base_of = {}
    for kv in kv_lens:
        base_of.setdefault(kv, len(base_of))
    nb = len(base_of)
    if attn == "gather_exact":
        # scores depend only on dims [0, 64), shared by every KV row of a base
        q_att = torch.randint(-2, 3, (batch, H_ATT, D_ATT), generator=gd, device=dev).float() * 0.25
        q_att[:, :, 64:] = 0.0
        key = torch.randint(-2, 3, (nb, 64), generator=gd, device=dev).float() * 0.25
    elif attn == "pow2":
        # score = q[0] * k[0] with q[0] in {1/2, 1, 2} per head and k[0] = -2 m, m in 0..3 per KV row: with the softmax
        # scale ln 2 the exponent of every weight is an integer
        q_att = torch.zeros(batch, H_ATT, D_ATT, device=dev)
        q_att[:, :, 0] = torch.tensor([0.5, 1.0, 2.0], device=dev).repeat(H_ATT // 3 + 1)[:H_ATT]
    else:
        q_att = rnd(batch, H_ATT, D_ATT)
    q_att = H._e4m3(q_att).reshape(batch, H_ATT * D_ATT)

    def kv_rows(j, r, m=None):
        x = rnd(r, D_ATT)
        if attn == "gather_exact":
            x[:, :64] = key[j]
        elif attn == "pow2":
            x[:, 0] = -2.0 * m
        return H._e4m3(x).view(torch.uint8)

    poison = torch.full((D_ATT,), 448.0, device=dev)
    poison[:64] = -448.0
    poison = H._e4m3(poison).view(torch.uint8)
    # two window pages per row (the window spans at most two), then a poison spare
    n_swa = 2 * batch + 1
    swa_pool = torch.empty((n_swa, TPB_SWA, D_ATT), dtype=torch.uint8, device=dev)
    swa_pool[n_swa - 1] = poison
    perm_s = torch.randperm(2 * batch, generator=gh)
    bt_swa = torch.full((batch, max(kv_lens) // TPB_SWA + 1), n_swa - 1, dtype=torch.int32)
    for b in range(batch):
        hi = (kv_lens[b] - 1) // TPB_SWA
        for t, blk in enumerate((hi - 1, hi)):
            page = int(perm_s[2 * b + t])
            if blk >= 0:
                bt_swa[b, blk] = page
            # pow2: the window rows have m = 1, 2, 3, 1, ...: a window maximum below the maximum of the selected rows
            m_win = 1.0 + torch.arange(TPB_SWA, device=dev) % 3
            swa_pool[page] = kv_rows(base_of[kv_lens[b]], TPB_SWA, m_win)
    # compressed KV pages of every base (at least one page: the table row is never empty)
    ncs = {kv: kv // RATIO for kv in base_of}
    pg_c = {kv: max(1, H._cdiv(nc, TPB_C)) for kv, nc in ncs.items()}
    n_c = sum(pg_c.values()) + 1
    c_pool = torch.empty((n_c, TPB_C, D_ATT), dtype=torch.uint8, device=dev)
    c_pool[n_c - 1] = poison
    perm_c = torch.randperm(n_c - 1, generator=gh)
    bt_row, off = {}, 0
    for kv, j in base_of.items():
        pages = perm_c[off : off + pg_c[kv]]
        off += pg_c[kv]
        # pow2: m grows with the position, 0 in the first quarter of the row to 3 in the last: the tiles of a selection
        # in ascending order have different maxima
        m_pos = (torch.arange(pg_c[kv] * TPB_C, device=dev) * 4 // max(ncs[kv], 1)).clamp(max=3)
        c_pool[pages.to(dev)] = kv_rows(j, pg_c[kv] * TPB_C, m_pos.float()).view(
            pg_c[kv], TPB_C, D_ATT
        )
        bt_row[kv] = pages.to(torch.int32)
    bt_c = torch.full((batch, H._cdiv(max(pg_c.values()), 4) * 4), n_c - 1, dtype=torch.int32)
    for b in range(batch):
        pages = bt_row[kv_lens[b]]
        bt_c[b, : pages.numel()] = pages
    sel = torch.stack([_selection(p, ncs[kv], gh) for kv, p in rows])
    sinks = (torch.linspace(-6.0, 9.0, H_ATT, device=dev) + 0.1 * rnd(H_ATT)).float()
    inp = {
        "q_att": q_att,
        "swa_pool": swa_pool.view(torch.float8_e4m3fn),
        "bt_swa": bt_swa.to(dev),
        "c_pool": c_pool.view(torch.float8_e4m3fn),
        "bt_c": bt_c.to(dev),
        "kv_lens": torch.tensor(kv_lens, dtype=torch.int32, device=dev),
        "sinks": sinks.contiguous(),
        "topk": sel.to(dev),
    }
    return inp


@torch.no_grad()
def _ref_attention(inp, b, scale):
    """fp64 sparse MLA of row b over its window (the last min(128, kv_len) positions) and the selected rows (not -1)."""
    pos = int(inp["kv_lens"][b]) - 1
    p = torch.arange(max(0, pos - WINDOW + 1), pos + 1, device=inp["topk"].device)
    swa_rows = inp["bt_swa"][b, p // TPB_SWA].long() * TPB_SWA + p % TPB_SWA
    sel = inp["topk"][b].long()
    sel = sel[sel >= 0]
    c_rows = inp["bt_c"][b, sel // TPB_C].long() * TPB_C + sel % TPB_C
    swa = inp["swa_pool"].view(torch.uint8).reshape(-1, D_ATT)
    cpl = inp["c_pool"].view(torch.uint8).reshape(-1, D_ATT)
    kv = torch.cat([swa[swa_rows], cpl[c_rows]], 0).view(torch.float8_e4m3fn).double()
    q = inp["q_att"][b].view(H_ATT, D_ATT).double()
    s = (q @ kv.t()) * scale
    m = s.amax(-1, keepdim=True)
    e = torch.exp(s - m)
    den = e.sum(-1, keepdim=True) + torch.exp(inp["sinks"].double().view(-1, 1) - m)
    return (e @ kv) / den


def _check_rows(inp, out, attn, scale=None):
    """Every row of ``out`` against the reference (the scale of the mode unless given); the largest calc_diff."""
    scale = _scale_of(attn) if scale is None else scale
    cols = (
        slice(64, None) if attn == "gather_exact" else slice(None)
    )  # gather_exact: dims 0-63 are the score key
    worst = 0.0
    for b in range(out.shape[0]):
        ref = _ref_attention(inp, b, scale)
        o = out[b].view(H_ATT, D_ATT).double()
        assert bool(torch.isfinite(o).all()), f"row {b}: non-finite output"
        cd = H._calc_diff(o[:, cols], ref[:, cols])
        if attn == "realistic":
            assert cd < CD_REALISTIC, f"row {b}: calc_diff {cd:.3e}"
        else:
            assert cd < CD_EXACT, f"row {b}: calc_diff {cd:.3e}"
            assert float((o - ref).abs().max()) <= 2.0**-6 * float(ref.abs().max()), f"row {b}"
        worst = max(worst, cd)
    return worst


def _launch(inp, out, spec, scale=SM_SCALE):
    _tail().run_attention(
        inp["topk"],
        inp["q_att"],
        inp["swa_pool"],
        inp["bt_swa"],
        inp["c_pool"],
        inp["bt_c"],
        inp["kv_lens"],
        inp["sinks"],
        out,
        scale,
        config=spec,
    )


def _run_and_check(inp, out, spec, attn):
    out.zero_()
    _launch(inp, out, spec, _scale_of(attn))
    torch.cuda.synchronize()
    return _check_rows(inp, out, attn)


def _new_out(batch):
    return torch.zeros((batch, H_ATT * D_ATT), dtype=torch.bfloat16, device="cuda")


def _compiled_clusters():
    """Cluster sizes the op has compiled on the current device (one compile serves both specs)."""
    dev = torch.cuda.current_device()
    return {g for (d, g) in _tail()._compiled if d == dev}


# --------------------------------------------------------------------------------------------------------------------
# host side (no GPU)


def test_cluster_size_table():
    tail = _tail()
    assert tail.TAIL_MAX_BATCH == MAX_ROWS
    assert [tail.tail_cluster_size(b) for b in range(1, MAX_ROWS + 1)] == [CLUSTER] * MAX_ROWS
    for bad in (-1, 0, *REFUSED):
        assert tail.tail_cluster_size(bad) is None
    # a cluster of 5 CTAs per row, all rows resident in one wave of the 148 SMs
    assert MAX_ROWS * CLUSTER <= 148


def test_cluster_sizes_of_the_op_are_the_ones_of_the_kernel():
    """The op offers the cluster sizes the kernel is built and validated for, and the kernel refuses the others."""
    pytest.importorskip("cutlass.cute")  # CuTe DSL; no GPU needed
    from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa import tail_attention as TA

    tail = _tail()
    assert {tail.tail_cluster_size(b) for b in range(1, MAX_ROWS + 1)} == set(
        TA.SUPPORTED_CLUSTER_SIZES
    )
    assert TA.SUPPORTED_CLUSTER_SIZES == (CLUSTER,)


def test_batch_size_support():
    tail = _tail()
    for b in range(1, MAX_ROWS + 1):
        assert tail.is_batch_size_supported(b) == (True, "")
    for bad in (-1, 0, *REFUSED):
        ok, why = tail.is_batch_size_supported(bad)
        assert not ok and f"batch size {bad} not in [1, {MAX_ROWS}]" in why


def test_is_supported_has_the_signature_of_the_other_ops():
    """is_supported(device, config) like the megakernel's and the large batch's (the batch size has its own helper)."""
    from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa import (
        large_batch_op,
        megakernel_op,
    )

    def params(fn):
        return list(inspect.signature(fn).parameters)

    assert params(_tail().is_supported) == ["device", "config"]
    assert params(_tail().is_supported) == params(megakernel_op.is_supported)
    assert params(_tail().is_supported) == params(large_batch_op.is_supported)


def test_is_supported_host_checks():
    tail = _tail()
    ok, why = tail.is_supported(config=PRO)
    assert not ok and "DS-V4-Pro" in why
    # a supported spec: the answer is the device check
    for spec in (None, SPEC, SPEC_FP4):
        ok, why = tail.is_supported(config=spec)
        assert ok == (not why)


def test_default_config_of_every_entry_point_is_the_same():
    """precompile, run_attention and run_hybrid take ``config=None`` (the default spec): none of them has its own."""
    tail = _tail()
    for fn in (tail.precompile, tail.run_attention, tail.run_hybrid):
        assert inspect.signature(fn).parameters["config"].default is None, fn.__name__


@requires_b200
def test_is_supported_on_b200():
    tail = _tail()
    for spec in (None, SPEC, SPEC_FP4):
        assert tail.is_supported(torch.device("cuda"), spec) == (True, "")
        assert tail.is_supported(config=spec) == (True, "")


def _host_args(batch, topk_cols=K_TOP):
    """Valid CPU arguments of run_attention (the checks run before any device work)."""
    i32 = torch.int32
    return dict(
        topk=torch.full((batch, topk_cols), -1, dtype=i32),
        q_att=torch.zeros((batch, H_ATT * D_ATT), dtype=torch.float8_e4m3fn),
        swa_pool=torch.zeros((2 * batch, TPB_SWA, D_ATT), dtype=torch.float8_e4m3fn),
        bt_swa=torch.zeros((batch, 4), dtype=i32),
        c_pool=torch.zeros((4, TPB_C, D_ATT), dtype=torch.float8_e4m3fn),
        bt_c=torch.zeros((batch, 4), dtype=i32),
        kv_lens=torch.full((batch,), 1000, dtype=i32),
        sinks=torch.zeros(H_ATT),
        out=torch.zeros((batch, H_ATT * D_ATT), dtype=torch.bfloat16),
    )


def test_rejects_batches_above_26():
    tail = _tail()
    for bad in (0, *REFUSED):
        with pytest.raises(ValueError, match=rf"batch size {bad} not in \[1, {MAX_ROWS}\]"):
            tail.run_attention(**_host_args(bad))
        for config in (SPEC, SPEC_FP4):
            with pytest.raises(ValueError, match=rf"batch size {bad} not in \[1, {MAX_ROWS}\]"):
                tail.precompile(torch.device("cuda"), bad, config)


def _transposed(t):
    return t.t().contiguous().t()


def _rows_of_256_bytes(t):
    return t[..., :256].contiguous()


# (argument, a variant of it, the error). topk gives the row count B: with another row count the other arguments fail.
BAD_ARGUMENTS = [
    ("topk", lambda t: t.to(torch.int64), r"topk must be a contiguous int32 \[B, 512\] tensor"),
    (
        "topk",
        lambda t: t[:, :-1].contiguous(),
        r"topk must be a contiguous int32 \[B, 512\] tensor",
    ),
    ("topk", _transposed, r"topk must be a contiguous int32 \[B, 512\] tensor"),
    ("topk", lambda t: t.reshape(-1), r"topk must be a contiguous int32 \[B, 512\] tensor"),
    (
        "topk",
        lambda t: t[:3].contiguous(),
        "q_att must be a contiguous FP8 tensor with 3 x 64 x 512",
    ),
    (
        "q_att",
        lambda t: t.to(torch.bfloat16),
        "q_att must be a contiguous FP8 tensor with 4 x 64 x 512",
    ),
    (
        "q_att",
        lambda t: t[:3].contiguous(),
        "q_att must be a contiguous FP8 tensor with 4 x 64 x 512",
    ),
    ("q_att", _transposed, "q_att must be a contiguous FP8 tensor with 4 x 64 x 512"),
    ("swa_pool", _rows_of_256_bytes, "swa_pool must be a contiguous FP8 tensor with 512-byte rows"),
    (
        "swa_pool",
        lambda t: t.reshape(-1),
        "swa_pool must be a contiguous FP8 tensor with 512-byte rows",
    ),
    (
        "swa_pool",
        lambda t: t.to(torch.bfloat16),
        "swa_pool must be a contiguous FP8 tensor with 512-byte",
    ),
    (
        "swa_pool",
        lambda t: t.transpose(0, 1),
        "swa_pool must be a contiguous FP8 tensor with 512-byte",
    ),
    ("c_pool", _rows_of_256_bytes, "c_pool must be a contiguous FP8 tensor with 512-byte rows"),
    (
        "c_pool",
        lambda t: t.reshape(-1),
        "c_pool must be a contiguous FP8 tensor with 512-byte rows",
    ),
    (
        "c_pool",
        lambda t: t.to(torch.bfloat16),
        "c_pool must be a contiguous FP8 tensor with 512-byte",
    ),
    ("c_pool", lambda t: t.transpose(0, 1), "c_pool must be a contiguous FP8 tensor with 512-byte"),
    ("bt_swa", lambda t: t.to(torch.int64), "bt_swa must be a contiguous int32"),
    ("bt_swa", lambda t: t[0], "bt_swa must be a contiguous int32"),
    ("bt_swa", lambda t: t[:3].contiguous(), "bt_swa must be a contiguous int32"),
    ("bt_swa", _transposed, "bt_swa must be a contiguous int32"),
    ("bt_c", lambda t: t.to(torch.int16), "bt_c must be a contiguous int32"),
    ("bt_c", lambda t: t[0], "bt_c must be a contiguous int32"),
    ("bt_c", lambda t: t[:3].contiguous(), "bt_c must be a contiguous int32"),
    ("bt_c", _transposed, "bt_c must be a contiguous int32"),
    (
        "kv_lens",
        lambda t: t.to(torch.int64),
        "kv_lens must be a contiguous int32 tensor with 4 elements",
    ),
    (
        "kv_lens",
        lambda t: t[:3].contiguous(),
        "kv_lens must be a contiguous int32 tensor with 4 elements",
    ),
    (
        "kv_lens",
        lambda t: torch.arange(8, dtype=t.dtype)[::2],
        "kv_lens must be a contiguous int32",
    ),
    (
        "sinks",
        lambda t: t.to(torch.float16),
        "sinks must be a contiguous float32 tensor with 64 elements",
    ),
    (
        "sinks",
        lambda t: t[:32].contiguous(),
        "sinks must be a contiguous float32 tensor with 64 elements",
    ),
    (
        "sinks",
        lambda t: torch.zeros(128)[::2],
        "sinks must be a contiguous float32 tensor with 64 elements",
    ),
    ("out", lambda t: t.float(), "out must be a contiguous bfloat16 tensor with 4 x 64 x 512"),
    (
        "out",
        lambda t: t[:3].contiguous(),
        "out must be a contiguous bfloat16 tensor with 4 x 64 x 512",
    ),
    ("out", _transposed, "out must be a contiguous bfloat16 tensor with 4 x 64 x 512"),
]


@pytest.mark.parametrize(
    "name, variant, match",
    [pytest.param(*c, id=f"{i}-{c[0]}") for i, c in enumerate(BAD_ARGUMENTS)],
)
def test_rejects_bad_arguments(name, variant, match):
    args = _host_args(4)
    args[name] = variant(args[name])
    with pytest.raises(ValueError, match=match):
        _tail().run_attention(**args)


def test_rejects_tensors_that_are_not_on_a_cuda_device():
    """Valid arguments on the CPU: the kernel reads device memory."""
    with pytest.raises(ValueError, match="must be CUDA tensors, got q_att on cpu"):
        _tail().run_attention(**_host_args(4))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_rejects_tensors_on_other_devices():
    """Every tensor must be on the device of q_att: a tensor on the CPU is refused by name."""
    tail = _tail()
    args = {k: v.cuda() for k, v in _host_args(4).items()}
    for name in args:
        if name == "q_att":
            continue
        bad = dict(args)
        bad[name] = args[name].cpu()
        with pytest.raises(
            ValueError, match=rf"{name} must be on the device of q_att \(cuda:\d+\), got cpu"
        ):
            tail.run_attention(**bad)
    bad = dict(args)
    bad["q_att"] = args["q_att"].cpu()
    with pytest.raises(ValueError, match="must be CUDA tensors, got q_att on cpu"):
        tail.run_attention(**bad)


def test_unsupported_spec_is_refused_before_any_device_work():
    pytest.importorskip("cutlass.cute")  # CuTe DSL; no GPU needed
    tail = _tail()
    with pytest.raises(NotImplementedError, match="DS-V4-Pro"):
        tail.precompile(torch.device("cpu"), 4, PRO)
    with pytest.raises(NotImplementedError, match="DS-V4-Pro"):
        tail.run_attention(**_host_args(4, topk_cols=PRO.topk), config=PRO)
    from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa import tail_attention as TA

    # only the cluster size 5 is built and validated: 4 and 3 (27 to 45 rows) are refused, as is anything else
    for g in (0, 1, 2, 3, 4, 6, 8):
        with pytest.raises(ValueError, match="cluster size"):
            TA.TailAttention(g, SPEC)
    with pytest.raises(NotImplementedError, match="specialized for"):
        TA.TailAttention(5, PRO)
    for spec in (SPEC, SPEC_FP4):
        assert TA.TailAttention(CLUSTER, spec).g == CLUSTER


def test_run_hybrid_calls_the_stages_in_order(monkeypatch):
    """K1 logits, then the top-K kernel on those logits, then the tail attention with the caller's tensors."""
    tail = _tail()
    calls = []
    args = _host_args(3)
    logits = object()

    def k1():
        calls.append("k1")
        return logits

    def topk_fn(x):
        assert x is logits, "the top-K stage gets the logits the K1 stage returned"
        calls.append("topk")

    def fake_attention(*a, **kw):
        calls.append(("attention", a, kw))

    monkeypatch.setattr(tail, "run_attention", fake_attention)
    names = ("topk", "q_att", "swa_pool", "bt_swa", "c_pool", "bt_c", "kv_lens", "sinks", "out")
    tail.run_hybrid(k1, topk_fn, *(args[n] for n in names), 0.125, config=SPEC)
    assert calls[:2] == ["k1", "topk"] and len(calls) == 3
    _, a, kw = calls[2]
    assert len(a) == 10 and all(x is args[n] for x, n in zip(a, names)) and a[9] == 0.125
    assert kw == {"config": SPEC}
    # no config: the default of run_attention (the kernel is the same for both specs)
    calls.clear()
    tail.run_hybrid(k1, topk_fn, *(args[n] for n in names))
    assert calls[2][1][9] is None and calls[2][2]["config"] is None


# --------------------------------------------------------------------------------------------------------------------
# kernel


@requires_b200
@pytest.mark.parametrize("attn", ATTN_MODES)
@pytest.mark.parametrize("rot", [0, 7])
@pytest.mark.parametrize("batch", BATCHES)
def test_matches_reference(batch, rot, attn):
    """Batch sizes on both sides of every cluster-size boundary, rows of every kind (FP4 spec)."""
    inp = _make_case(_rows(batch, rot), seed=1000 * batch + 10 * rot + len(attn), attn=attn)
    _run_and_check(inp, _new_out(batch), SPEC_FP4, attn)
    assert _compiled_clusters() == {CLUSTER}


@requires_b200
@pytest.mark.parametrize("attn", ATTN_MODES)
@pytest.mark.parametrize("batch", [1, 8, MAX_ROWS])
def test_fp8_spec_matches_reference(batch, attn):
    """The same attention behind the FP8 spec (the compile of the FP4 spec serves it)."""
    inp = _make_case(_rows(batch, 3), seed=77 * batch + len(attn), attn=attn)
    _run_and_check(inp, _new_out(batch), SPEC, attn)
    assert _compiled_clusters() == {CLUSTER}


# Windows of 1 to 5 rows, around 32 / 64 rows, around the 128-row window and around the page edge at 256 positions.
SHORT_KV_LENS = [
    1,
    2,
    3,
    4,
    5,
    8,
    31,
    32,
    33,
    63,
    64,
    65,
    126,
    127,
    128,
    129,
    130,
    131,
    255,
    256,
    257,
]


@requires_b200
@pytest.mark.parametrize("attn", ATTN_MODES)
@pytest.mark.parametrize("batch", [21, MAX_ROWS])
def test_short_windows(batch, attn):
    """Every kv_len of SHORT_KV_LENS in the batch (rows with fewer than 4 positions have no compressed position)."""
    kv_lens = (SHORT_KV_LENS * 3)[:batch]
    rows = [(kv, "dense" if b % 3 else "holes") for b, kv in enumerate(kv_lens)]
    inp = _make_case(rows, seed=500 + batch + len(attn), attn=attn)
    _run_and_check(inp, _new_out(batch), SPEC_FP4, attn)


@requires_b200
@pytest.mark.parametrize("attn", ATTN_MODES)
@pytest.mark.parametrize("batch", [1, 8, MAX_ROWS])
def test_empty_and_sparse_selections(batch, attn):
    """No selection (the window and the sink), a single position, and every pattern at the longest lengths."""
    kinds = ["none", "one", "last", "short", "tile_gap", "holes", "sorted", "dense"]
    rows = [(1048579 if b % 2 else 524291, kinds[b % len(kinds)]) for b in range(batch)]
    inp = _make_case(rows, seed=600 + batch + len(attn), attn=attn)
    out = _new_out(batch)
    _run_and_check(inp, out, SPEC_FP4, attn)
    inp["topk"].fill_(-1)  # all rows without a selection
    _run_and_check(inp, out, SPEC_FP4, attn)


@requires_b200
def test_softmax_scale_and_sinks_are_read():
    """The scale is a launch argument, the sinks a tensor: both change the result as the reference says."""
    inp = _make_case(_rows(8, 2), seed=700)
    out = _new_out(8)
    for scale in (SM_SCALE, 0.03, 0.2):
        out.zero_()
        _launch(inp, out, SPEC_FP4, scale)
        torch.cuda.synchronize()
        _check_rows(inp, out, "realistic", scale)
    inp["sinks"].add_(3.0)
    _run_and_check(inp, out, SPEC_FP4, "realistic")


@requires_b200
@pytest.mark.parametrize("batch", [2, 16, MAX_ROWS])
def test_deterministic_and_batch_invariant(batch):
    """Launches repeat bit for bit (a race inside the cluster would show as a differing launch); a row gives the
    same bits alone as inside a batch."""
    inp = _make_case(_rows(batch, 1), seed=800 + batch)
    first = _new_out(batch)
    _launch(inp, first, SPEC_FP4)
    for launch in range(60):
        out = _new_out(batch)  # zeros: a row that is not written shows as a difference
        _launch(inp, out, SPEC_FP4)
        assert torch.equal(out, first), f"launch {launch} differs from the first launch"
    _check_rows(inp, first, "realistic")
    # the first row alone (batch 1) equals that row inside the batch
    row_keys = ("q_att", "bt_swa", "bt_c", "kv_lens", "topk")
    one = {k: (v[:1] if k in row_keys else v) for k, v in inp.items()}
    out1 = _new_out(1)
    _launch(one, out1, SPEC_FP4)
    torch.cuda.synchronize()
    assert torch.equal(out1[0], first[0])


@requires_b200
def test_batch_sizes_up_and_down_do_not_leak():
    """The op keeps nothing but the compiled kernel: batch sizes up and down in one process give the right result each
    time."""
    for batch in (MAX_ROWS, 2, 24, 1, 16):
        inp = _make_case(_rows(batch, batch), seed=900 + batch, attn="pow2")
        _run_and_check(inp, _new_out(batch), SPEC_FP4, "pow2")


def _capture(fn):
    """Warm up on a side stream (as the other specDSA tests do), then capture ``fn`` into a CUDA graph."""
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        fn()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


@requires_b200
@pytest.mark.parametrize("attn", ["realistic", "pow2"])
@pytest.mark.parametrize("batch", [8, MAX_ROWS])
def test_cuda_graph_replay(batch, attn, spec_name):
    """One captured launch replays over several steps; the inputs change in place (selection, lengths, pages)."""
    spec = _spec_of(spec_name)
    rows = _rows(batch, 2)
    steps = [rows[r:] + rows[:r] for r in (0, 3, 5)]  # same pool sizes, other rows get other kinds
    inp = _make_case(steps[0], seed=1100 + batch, attn=attn)
    _tail().precompile(torch.device("cuda"), batch, spec)
    out = _new_out(batch)
    graph = _capture(lambda: _launch(inp, out, spec, _scale_of(attn)))
    for step, rows in enumerate(steps):
        if step:
            new = _make_case(rows, seed=1100 + batch + step, attn=attn)
            for key, buf in inp.items():
                assert buf.shape == new[key].shape, key
                buf.copy_(new[key])
        out.zero_()
        graph.replay()
        torch.cuda.synchronize()
        _check_rows(inp, out, attn)


@requires_b200
@pytest.mark.parametrize("index", [None, 0], ids=["device-cuda", "device-cuda-0"])
def test_precompile_covers_the_device_of_the_tensors(monkeypatch, index):
    """precompile(device, ...) compiles what run_attention needs for tensors on that device, however the device is
    spelled (``cuda`` or ``cuda:0``): the launch after it compiles nothing."""
    from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa import tail_attention as TA

    tail = _tail()
    compiles = []
    real_compile = TA.compile_tail_attention
    monkeypatch.setattr(
        TA, "compile_tail_attention", lambda t: (compiles.append(t.g), real_compile(t))[1]
    )
    monkeypatch.setattr(tail, "_compiled", {})  # as in a fresh process
    inp = _make_case(_rows(8, 4), seed=1250)
    out = _new_out(8)
    device = torch.device("cuda", index) if index is not None else torch.device("cuda")
    tail.precompile(device, 8, SPEC_FP4)
    assert compiles == [5]
    _run_and_check(inp, out, SPEC_FP4, "realistic")
    assert compiles == [5], (
        "the launch compiled again: precompile did not cover the tensors' device"
    )


@requires_b200
def test_one_compile_serves_both_specs_and_every_batch_size(monkeypatch):
    """The kernel does not depend on the indexer K dtype or on the batch size: the FP8 spec, the FP4 spec and the
    default config share one compile per device."""
    from tensorrt_llm._torch.attention.kernels.blackwell.spec_dsa import tail_attention as TA

    tail = _tail()
    compiles = []
    real_compile = TA.compile_tail_attention
    monkeypatch.setattr(
        TA, "compile_tail_attention", lambda t: (compiles.append(t.g), real_compile(t))[1]
    )
    monkeypatch.setattr(tail, "_compiled", {})
    for batch, config in ((8, SPEC_FP4), (8, SPEC), (26, None), (1, SPEC_FP4), (16, SPEC)):
        inp = _make_case(_rows(batch, 2), seed=1260 + batch)
        _run_and_check(inp, _new_out(batch), config, "realistic")
    tail.precompile(torch.device("cuda"), 24, SPEC)
    tail.precompile(torch.device("cuda"), 3)
    assert compiles == [5]
    assert _compiled_clusters() == {CLUSTER}


@requires_b200
@pytest.mark.parametrize(
    "run_config", [None, SPEC, SPEC_FP4], ids=["run-default", "run-fp8", "run-fp4"]
)
@pytest.mark.parametrize(
    "pre_config", [None, SPEC, SPEC_FP4], ids=["pre-default", "pre-fp8", "pre-fp4"]
)
def test_precompile_then_captured_hybrid(monkeypatch, pre_config, run_config):
    """precompile(device, batch) followed by a CUDA-graph capture of run_hybrid compiles nothing during the capture,
    whichever spec each call names (``None``: the default of both): there is no warm-up launch in between."""
    tail = _tail()
    batch = 8
    monkeypatch.setattr(tail, "_compiled", {})  # as in a fresh process
    inp = _make_case(_rows(batch, 5), seed=1500 + batch)
    sel = inp["topk"].clone()
    topk = torch.full_like(sel, -1)
    logits = torch.zeros(batch, 4, device="cuda")
    out = _new_out(batch)

    def hybrid():
        tail.run_hybrid(
            lambda: logits.add_(1.0),
            lambda lg: topk.copy_(sel),
            topk,
            inp["q_att"],
            inp["swa_pool"],
            inp["bt_swa"],
            inp["c_pool"],
            inp["bt_c"],
            inp["kv_lens"],
            inp["sinks"],
            out,
            config=run_config,
        )

    tail.precompile(torch.device("cuda"), batch, pre_config)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        hybrid()
    graph.replay()
    torch.cuda.synchronize()
    _check_rows({**inp, "topk": topk}, out, "realistic")


@requires_b200
@pytest.mark.parametrize("name", ["q_att", "swa_pool", "c_pool", "sinks", "out"])
def test_misaligned_tensors_are_refused(name):
    """The kernel reads these through TMA and 16-byte accesses: a base address that is not a multiple of 16 bytes is a
    ValueError of the compiled function, not a CUDA fault."""
    inp = _make_case(_rows(4, 0), seed=1280)
    out = _new_out(4)
    t = out if name == "out" else inp[name]
    nbytes = t.numel() * t.element_size()
    buf = torch.zeros(nbytes + 64, dtype=torch.uint8, device="cuda")
    shifted = buf[t.element_size() : t.element_size() + nbytes].view(t.dtype).view(t.shape)
    shifted.copy_(t)
    assert shifted.data_ptr() % 16 == t.element_size()
    if name == "out":
        out = shifted
    else:
        inp[name] = shifted
    with pytest.raises(ValueError, match="Misaligned"):
        _launch(inp, out, SPEC_FP4)
    torch.cuda.synchronize()  # nothing was launched


@requires_b200
def test_arguments_of_other_shapes_are_accepted():
    """The contract says q_att and out are any shape with B * 64 * 512 elements and sinks any with 64: same bits."""
    batch = 5
    inp = _make_case(_rows(batch, 1), seed=1270)
    first = _new_out(batch)
    _launch(inp, first, SPEC_FP4)
    other = {
        **inp,
        "q_att": inp["q_att"].view(batch, H_ATT, D_ATT),
        "sinks": inp["sinks"].view(1, H_ATT),
    }
    out = torch.zeros((batch, H_ATT, D_ATT), dtype=torch.bfloat16, device="cuda")
    _launch(other, out, SPEC_FP4)
    torch.cuda.synchronize()
    assert torch.equal(out.view(batch, -1), first)


@pytest.mark.filterwarnings("ignore:The CUDA Graph is empty")
@requires_b200
def test_compile_during_capture_is_refused(monkeypatch):
    """A kernel that is not compiled yet cannot be compiled inside a capture: precompile first."""
    inp = _make_case(_rows(4, 0), seed=1200)
    out = _new_out(4)
    _launch(inp, out, SPEC_FP4)  # warm-up (scratch, compile)
    monkeypatch.setattr(_tail(), "_compiled", {})  # as in a fresh process
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(RuntimeError, match="precompile"):
        with torch.cuda.graph(graph):
            _launch(inp, out, SPEC_FP4)
    monkeypatch.undo()
    _run_and_check(inp, out, SPEC_FP4, "realistic")


@requires_b200
@pytest.mark.parametrize("attn", ["realistic", "pow2"])
@pytest.mark.parametrize("batch", [5, 16, MAX_ROWS])
def test_run_hybrid_with_stand_in_stages(batch, attn):
    """run_hybrid with stand-ins for the K1 logits kernel and the top-K kernel, eagerly and inside a CUDA graph."""
    tail = _tail()
    inp = _make_case(_rows(batch, 6), seed=1300 + batch, attn=attn)
    sel = inp["topk"].clone()  # what the stand-in top-K kernel "selects"
    topk = torch.full_like(sel, -1)  # the buffer the tail attention reads
    logits = torch.zeros(batch, 4, device="cuda")
    events = []

    def k1():
        events.append("k1")
        logits.add_(1.0)  # a device op: the capture records it too
        return logits

    def topk_fn(lg):
        events.append("topk")
        assert lg is logits
        topk.copy_(sel)

    def hybrid():
        tail.run_hybrid(
            k1,
            topk_fn,
            topk,
            inp["q_att"],
            inp["swa_pool"],
            inp["bt_swa"],
            inp["c_pool"],
            inp["bt_c"],
            inp["kv_lens"],
            inp["sinks"],
            out,
            _scale_of(attn),
            config=SPEC_FP4,
        )

    out = _new_out(batch)
    hybrid()
    torch.cuda.synchronize()
    assert events == ["k1", "topk"]
    _check_rows({**inp, "topk": topk}, out, attn)
    tail.precompile(torch.device("cuda"), batch, SPEC_FP4)
    graph = _capture(hybrid)
    # the replayed graph runs the stand-in stages again: new selection, same buffers
    rows = _rows(batch, 6)
    inp2 = _make_case(rows[2:] + rows[:2], seed=1400 + batch, attn=attn)
    for key, buf in inp.items():
        assert buf.shape == inp2[key].shape, key
        buf.copy_(inp2[key])
    sel.copy_(inp2["topk"])
    topk.fill_(-1)
    out.zero_()
    before = float(logits[0, 0])
    graph.replay()
    torch.cuda.synchronize()
    assert float(logits[0, 0]) == before + 1.0, "the K1 stage is part of the graph"
    _check_rows({**inp, "topk": topk}, out, attn)
