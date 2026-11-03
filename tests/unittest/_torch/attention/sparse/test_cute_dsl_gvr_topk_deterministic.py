# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Deterministic self-sampling GVR top-K: fixed selected set, fixed order, optional ascending rows."""

import pytest
import torch

from tensorrt_llm._torch.kernels.blackwell.top_k import gvr_topk_decode_self_sampling_host as ss
from tensorrt_llm._utils import get_sm_version

skip_not_sm100 = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() not in (100, 103),
    reason="self-sampling GVR kernels target sm_100a",
)

# (batch, n, k): one shape per kernel family (reg, reg_clus, clus, main R=1, main SPLIT)
_SHAPES = [
    (1, 8192, 512),
    (64, 32768, 1024),
    (64, 131072, 512),
    (128, 32768, 2048),
    (1, 524288, 1024),
    (256, 32783, 1024),
    (512, 32783, 2048),
    (256, 131087, 2048),
    (512, 131075, 1024),
    (4, 32771, 2048),  # reg_clus with K=2048
]
_PATTERNS = ["unique", "quantized", "equal", "boundary", "clustered"]


def _row(n: int, k: int, pattern: str) -> torch.Tensor:
    gen = torch.Generator().manual_seed(20261011 + n)
    base = torch.randperm(n, generator=gen).float() / n
    if pattern == "unique":
        return base
    if pattern == "quantized":
        return base.bfloat16().float()
    if pattern == "equal":
        return torch.ones(n)
    if pattern == "clustered":
        # Concentrate high scores in the chunks owned by a single warp.
        indices = torch.arange(n)
        return base + 2 * ((indices // 4) % 512 < 32).float()
    perm = torch.randperm(n, generator=gen)  # 64 ties straddling the boundary, 16 of them selected
    row = torch.zeros(n)
    row[perm[: k - 16]] = 2
    row[perm[k - 16 : k + 48]] = 1
    return row


def _oracle(row: torch.Tensor, k: int) -> set:
    """{score > tau} + the smallest indices among {score == tau}: stable sort by (-score, index)."""
    order = torch.sort(-row, stable=True).indices
    return set(order[:k].tolist())


def _run(scores, k, n, **kw):
    b = scores.shape[0]
    if scores.shape[1] % 16:
        scores = torch.nn.functional.pad(scores, (0, -scores.shape[1] % 16))
    lengths = torch.full((b,), n, dtype=torch.int32, device="cuda")
    out = torch.empty((b, k), dtype=torch.int32, device="cuda")
    ss.run_varlen(scores, lengths, out, max_seq_len=n, **kw)
    torch.cuda.synchronize()
    return out


@skip_not_sm100
@pytest.mark.parametrize("pattern", _PATTERNS)
@pytest.mark.parametrize("b,n,k", _SHAPES)
def test_deterministic_set_and_order(b, n, k, pattern):
    row = _row(n, k, pattern)
    scores = row.cuda().unsqueeze(0).repeat(b, 1)
    oracle = _oracle(row, k)
    outs = [_run(scores, k, n, deterministic=True) for _ in range(4)]
    for out in outs:
        assert torch.equal(out, outs[0])  # fixed order across launches
        for r in (0, b - 1):
            assert set(out[r].tolist()) == oracle  # fixed selected set, smaller index wins ties


@skip_not_sm100
@pytest.mark.parametrize("b,n,k", _SHAPES[::2])
def test_deterministic_sorted(b, n, k):
    row = _row(n, k, "quantized")
    scores = row.cuda().unsqueeze(0).repeat(b, 1)
    out = _run(scores, k, n, deterministic=True, sorting="ascending")
    assert torch.equal(out, out.sort(dim=1).values)
    assert set(out[0].tolist()) == _oracle(row, k)


@skip_not_sm100
@pytest.mark.parametrize("rows,n,k", [(1, 8192, 512), (149, 32768, 1024), (297, 32768, 2048)])
def test_deterministic_prefill(rows, n, k):
    """Prefill windows [ks, ke) per row: oracle set in the local frame, fixed order, ascending option."""
    gen = torch.Generator().manual_seed(20261012 + n)
    scores = torch.randperm(rows * n, generator=gen).float().view(rows, n).bfloat16().float().cuda()
    ks = torch.randint(0, n // 4, (rows,), generator=gen, dtype=torch.int32)
    ke = n - torch.randint(0, n // 4, (rows,), generator=gen, dtype=torch.int32)
    out = torch.empty((rows, k), dtype=torch.int32, device="cuda")
    runs = []
    for _ in range(3):
        ss.run_prefill(scores, ks.cuda(), ke.cuda(), out, deterministic=True)
        torch.cuda.synchronize()
        runs.append(out.clone())
    assert torch.equal(runs[0], runs[1]) and torch.equal(runs[1], runs[2])
    for r in (0, rows - 1):
        row = scores[r, int(ks[r]) : int(ke[r])].cpu()
        assert set(runs[0][r].tolist()) == _oracle(row, k)  # local frame: column - ks
    ss.run_prefill(scores, ks.cuda(), ke.cuda(), out, deterministic=True, sorting="ascending")
    torch.cuda.synchronize()
    assert torch.equal(out, out.sort(dim=1).values)
    assert torch.equal(out.sort(dim=1).values, runs[0].sort(dim=1).values)


@skip_not_sm100
@pytest.mark.parametrize(
    "rows,n,k,parts", [(4096, 65536, 2048, 16), (1024, 131072, 2048, 16), (600, 32768, 2048, 4)]
)
def test_prefill_slices_with_tier_rows_match_the_tile(rows, n, k, parts):
    """Row slices of a prefill tile run with tier_rows=<tile rows> write the tile's rows bit for bit (the
    plan, and with it each row's index order, otherwise follows the slice's row count)."""
    gen = torch.Generator().manual_seed(20261030 + rows)
    scores = torch.randperm(rows * n, generator=gen).float().view(rows, n).bfloat16().float().cuda()
    ks = torch.randint(0, n // 4, (rows,), generator=gen, dtype=torch.int32).cuda()
    ke = (n - torch.randint(0, n // 4, (rows,), generator=gen, dtype=torch.int32)).cuda()
    full = torch.empty((rows, k), dtype=torch.int32, device="cuda")
    ss.run_prefill(scores, ks, ke, full, deterministic=True)
    sliced = torch.empty_like(full)
    plain = torch.empty_like(full)
    for p in range(parts):
        lo, hi = rows * p // parts, rows * (p + 1) // parts
        ss.run_prefill(
            scores[lo:hi], ks[lo:hi], ke[lo:hi], sliced[lo:hi], deterministic=True, tier_rows=rows
        )
        ss.run_prefill(scores[lo:hi], ks[lo:hi], ke[lo:hi], plain[lo:hi], deterministic=True)
    torch.cuda.synchronize()
    assert torch.equal(sliced, full)
    # without the hint every row keeps its selected set (the order may differ)
    assert torch.equal(plain.sort(dim=1).values, full.sort(dim=1).values)


@skip_not_sm100
def test_default_path_is_the_original_kernel():
    b, n, k = _SHAPES[2]
    row = _row(n, k, "unique")
    scores = row.cuda().unsqueeze(0).repeat(b, 1)
    out = _run(scores, k, n)
    assert set(out[0].tolist()) == _oracle(row, k)  # exact set on tie-free data
    with pytest.raises(ValueError):
        _run(scores, k, n, sorting="ascending")  # sorting needs deterministic=True


@skip_not_sm100
@pytest.mark.parametrize(
    "kv_lens,next_n",
    [
        ((150000,), 6),
        ((1048576,), 6),
        ((1000, 600000), 6),
        ((4096, 150000, 600000, 1048576), 6),
        ((131072,), 7),
    ],
)
def test_varlen_split_tile_rows_bitwise(monkeypatch, kv_lens, next_n):
    """Under a 1M envelope the deterministic SPLIT main family takes a shorter tile for decode-sized batches
    (_varlen_split_tile); every row (set and order) equals the envelope-sized tile's, over a kv_len mix."""
    k, env = 2048, 1 << 20
    b = len(kv_lens)
    rows = b * next_n
    gen = torch.Generator(device="cuda").manual_seed(20261031 + rows)
    scores = torch.randn(rows, env, device="cuda", generator=gen)
    scores[0, 100:3000] = scores[0, 100:3000].bfloat16().float()  # some exact ties near the top
    lengths = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    outs = []
    for short in (False, True):
        monkeypatch.setattr(ss, "_VARLEN_SHORT_TILE", short)
        ss._VARLEN_CACHE.clear()
        out = torch.empty((rows, k), dtype=torch.int32, device="cuda")
        ss.run_varlen(scores, lengths, out, next_n=next_n, max_seq_len=env, deterministic=True)
        torch.cuda.synchronize()
        outs.append(out)
    ss._VARLEN_CACHE.clear()
    assert torch.equal(outs[0], outs[1])
    for r in (0, rows - 1):
        n = kv_lens[r // next_n] - next_n + r % next_n + 1
        assert {i for i in outs[1][r].tolist() if i >= 0} == _oracle(scores[r, :n].cpu(), k)


@skip_not_sm100
@pytest.mark.parametrize(
    "kv_lens,next_n",
    [
        ((150000,), 6),
        ((150000,), 8),
        ((65536, 262144), 8),
        ((1000, 600000), 6),
        ((4096, 150000, 600000, 1048576), 6),
        ((150000, 150000, 150000, 150000), 8),
    ],
)
def test_varlen_bitmap_emit_rows_bitwise(monkeypatch, kv_lens, next_n):
    """Under a 1M envelope the deterministic main family compiles the bitmap (LBM) emit in (TRTLLM_GVR_VARLEN_LBM,
    rows that fit take it in-kernel, longer rows keep the sort): every row (set and order) equals the
    sort-only variant's, over a kv_len mix."""
    from tensorrt_llm._torch.kernels.blackwell.top_k import gvr_topk_decode_self_sampling as dev

    k, env = 2048, 1 << 20
    rows = len(kv_lens) * next_n
    gen = torch.Generator(device="cuda").manual_seed(20261031 + 7 * rows)
    scores = torch.randn(rows, env, device="cuda", generator=gen)
    scores[0, 100:3000] = scores[0, 100:3000].bfloat16().float()  # some exact ties near the top
    lengths = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    outs = []
    for lbm in (False, True):
        monkeypatch.setattr(dev, "_VARLEN_LBM", lbm)
        ss._VARLEN_CACHE.clear()
        out = torch.empty((rows, k), dtype=torch.int32, device="cuda")
        ss.run_varlen(scores, lengths, out, next_n=next_n, max_seq_len=env, deterministic=True)
        torch.cuda.synchronize()
        outs.append(out)
    ss._VARLEN_CACHE.clear()
    assert torch.equal(outs[0], outs[1])
    for r in (0, rows - 1):
        n = kv_lens[r // next_n] - next_n + r % next_n + 1
        assert {i for i in outs[1][r].tolist() if i >= 0} == _oracle(scores[r, :n].cpu(), k)


def _paged_case(kv_lens, next_n, env, seed):
    k, bs = 2048, 64
    b = len(kv_lens)
    rows = b * next_n
    gen = torch.Generator(device="cuda").manual_seed(seed)
    scores = torch.randn(rows, env, device="cuda", generator=gen)
    scores[0, 100:3000] = scores[0, 100:3000].bfloat16().float()  # some exact ties near the top
    lengths = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    max_blocks = (env + bs - 1) // bs + 3
    # every request owns distinct pool blocks in a scrambled order; unused table entries are -1 padding
    perm = torch.randperm(b * max_blocks, device="cuda", generator=gen).to(torch.int32)
    table = perm.view(b, max_blocks).contiguous()
    for i, n in enumerate(kv_lens):
        table[i, (n + bs - 1) // bs + 1 :] = -1
    return k, bs, rows, scores, lengths, table


@skip_not_sm100
@pytest.mark.parametrize(
    "kv_lens,next_n",
    [
        ((150000,), 6),
        ((150000,), 8),
        ((1000, 600000), 6),
        ((4096, 150000, 600000, 1048576), 6),
        ((150000, 150000, 150000, 150000), 8),
    ],
)
@pytest.mark.parametrize("layer_offset", [0, 5])
def test_varlen_paged_remap_matches_convert(kv_lens, next_n, layer_offset):
    """remap: the deterministic main family writes convertReqIndexToGlobal's integers of its own local rows (request
    r // next_n, -1 for -1 and for padded table entries) and reports it."""
    import tensorrt_llm  # noqa: F401  (loads the trtllm ops)

    env = 1 << 20
    seed = 20261031 + 13 * len(kv_lens) * next_n + layer_offset
    k, bs, rows, scores, lengths, table = _paged_case(kv_lens, next_n, env, seed)
    stride_factor = 24 * bs  # e.g. 24 layers per pool block
    local = torch.empty((rows, k), dtype=torch.int32, device="cuda")
    assert not ss.run_varlen(
        scores, lengths, local, next_n=next_n, max_seq_len=env, deterministic=True
    )
    paged = torch.empty((rows, k), dtype=torch.int32, device="cuda")
    assert ss.run_varlen(
        scores,
        lengths,
        paged,
        next_n=next_n,
        max_seq_len=env,
        deterministic=True,
        remap=(table, bs, stride_factor, layer_offset),
    )
    req = (torch.arange(rows, device="cuda", dtype=torch.int32) // next_n).contiguous()
    ref = torch.ops.trtllm.convert_req_index_to_global(
        req, table, local, bs, k, stride_factor, layer_offset
    )
    torch.cuda.synchronize()
    assert torch.equal(paged, ref)


@skip_not_sm100
def test_varlen_paged_remap_declined_off_main_family():
    """A launch the main family does not serve (small envelope: register families) keeps local indices and says so."""
    env, next_n = 8192, 6
    k, bs, rows, scores, lengths, table = _paged_case((8000,), next_n, env, 7)
    plain = torch.empty((rows, k), dtype=torch.int32, device="cuda")
    ss.run_varlen(scores, lengths, plain, next_n=next_n, max_seq_len=env, deterministic=True)
    out = torch.empty((rows, k), dtype=torch.int32, device="cuda")
    took = ss.run_varlen(
        scores,
        lengths,
        out,
        next_n=next_n,
        max_seq_len=env,
        deterministic=True,
        remap=(table, bs, 24 * bs, 0),
    )
    torch.cuda.synchronize()
    assert not took and torch.equal(out, plain)


@skip_not_sm100
@pytest.mark.parametrize(
    "kv_lens,next_n,env",
    [
        ((150000,), 8, 1 << 20),  # main SPLIT (B1)
        ((65536, 262144), 8, 1 << 20),
        ((1000, 600000), 6, 1 << 20),
        ((140000,) * 32, 7, 1 << 18),  # main R=1 (224 rows)
        ((140000,) * 36, 7, 1 << 18),  # main R=1 (252 rows)
        ((4000, 8000), 7, 1 << 13),  # register family: BF16 rows are widened
    ],
)
@pytest.mark.parametrize("remap", [False, True])
def test_varlen_bf16_rows_match_fp32_widened(kv_lens, next_n, env, remap):
    """BF16 logits (det): the launch writes the bits of the same launch on the rows widened to FP32 -- the main family
    reads BF16 directly (same set: values and ties unchanged; same order: canonical ascending emit), other families get
    widened rows -- with and without the paged remap. Rows carry dense BF16 ties at the selection boundary."""
    import tensorrt_llm  # noqa: F401  (loads the trtllm ops)

    k, bs = 2048, 64
    b = len(kv_lens)
    rows = b * next_n
    gen = torch.Generator(device="cuda").manual_seed(20261101 + rows + env)
    scores = torch.randn(rows, env, device="cuda", generator=gen).bfloat16()
    # row 0: the top ~3k values squeezed onto a few BF16 codes (many exact ties straddling the boundary)
    top = scores[0].float().topk(min(3000, env // 2)).indices
    scores[0, top] = (scores[0, top].float() * 4).round().bfloat16() / 4
    wide = scores.float()
    lengths = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    kw = dict(next_n=next_n, max_seq_len=env, deterministic=True)
    if remap:
        max_blocks = (env + bs - 1) // bs + 3
        perm = torch.randperm(b * max_blocks, device="cuda", generator=gen).to(torch.int32)
        table = perm.view(b, max_blocks).contiguous()
        for i, n in enumerate(kv_lens):
            table[i, (n + bs - 1) // bs + 1 :] = -1
        kw["remap"] = (table, bs, 24 * bs, 3)
    out16 = torch.empty((rows, k), dtype=torch.int32, device="cuda")
    out32 = torch.empty((rows, k), dtype=torch.int32, device="cuda")
    took16 = ss.run_varlen(scores, lengths, out16, **kw)
    took32 = ss.run_varlen(wide, lengths, out32, **kw)
    torch.cuda.synchronize()
    assert took16 == took32
    assert torch.equal(out16, out32)
    if not remap:
        for r in (0, rows - 1):
            n = kv_lens[r // next_n] - next_n + r % next_n + 1
            assert {i for i in out16[r].tolist() if i >= 0} == _oracle(wide[r, :n].cpu(), k)


def test_varlen_split_tile_rule():
    tpl = (1024, 8, 1, 256, 2, True, False)
    assert ss._varlen_split_tile(tpl, 24)[1] == 2
    assert ss._varlen_split_tile(tpl, 12)[1] == 4
    assert ss._varlen_split_tile(tpl, 6) == tpl
    assert ss._varlen_split_tile((1024, 1, 1, 256, 2, True, False), 24)[1] == 1


def _row_split_kv(case: int) -> tuple:
    gen = torch.Generator().manual_seed(20261101 + case)
    if case == 0:  # one 1M row and a long tail of mid / short rows (the e2e top-end shape)
        rest = torch.randint(1000, 200000, (21,), generator=gen).tolist()
        return (1 << 20, 600000, 300000, *rest), 7
    if case == 1:  # uniform rows plus one long row
        return (880000, *([140000] * 24)), 7
    if case == 2:  # rows at and below k (short-row path), tile-boundary lengths, one long row
        tiles = torch.randint(1, 50, (24,), generator=gen).tolist()
        edges = [16384 * t + d for t, d in zip(tiles, [-5, -4, 0, 1, 3, 4] * 4)]
        return (5, 900, 2048, 2054, 2055, 700000, *edges), 6
    if case == 3:  # 84 rows: the 1024-thread family; one 1M row, 32k-tile edges, short rows
        tiles = torch.randint(1, 20, (7,), generator=gen).tolist()
        edges = [32768 * t + d for t, d in zip(tiles, [-5, -4, 0, 1, 3, 4, 7])]
        return (1 << 20, 490000, 2048, 1000, 140000, *edges), 7
    # 140 rows (1024 threads): heavy-tailed mix
    rest = torch.randint(3000, 160000, (17,), generator=gen).tolist()
    return (720000, 330000, 250000, *rest), 7


@skip_not_sm100
@pytest.mark.parametrize(
    "case,pattern,dtype",
    [
        (0, "randn", torch.bfloat16),
        (0, "dense", torch.float32),
        (1, "clustered", torch.bfloat16),
        (2, "squeezed", torch.float32),
        (3, "randn", torch.bfloat16),
        (3, "squeezed", torch.float32),
        (4, "dense", torch.float32),
        (4, "clustered", torch.bfloat16),
    ],
)
@pytest.mark.parametrize("remap", [False, True])
def test_varlen_row_split_rows_bitwise(monkeypatch, case, pattern, dtype, remap):
    """ROW SPLIT: the one-CTA-per-row main families (75..256 rows) run long rows over several CTAs; every
    row (set, order, paged remap) equals the one-CTA launch's, for several launch CTA counts / part sizes, over rows
    with exact ties at the selection boundary, a dense top band (warp-segment overflow -> incomplete-list re-sweep)
    and clustered top scores."""
    import tensorrt_llm  # noqa: F401  (loads the trtllm ops)

    k, bs, env = 2048, 64, 1 << 20
    kv_lens, next_n = _row_split_kv(case)
    b = len(kv_lens)
    rows = b * next_n
    gen = torch.Generator(device="cuda").manual_seed(20261102 + rows + case)
    scores = torch.randn(rows, env, device="cuda", generator=gen)
    if (
        pattern == "dense"
    ):  # one score in five in a narrow top band: far more candidates than the staging holds
        scores = scores * 0.01
        scores[:, ::5] += 1.0
    elif (
        pattern == "squeezed"
    ):  # each row's top ~3k scores on a 0.25 grid: exact ties straddle the boundary
        top = scores.topk(3000, dim=1).indices
        scores.scatter_(1, top, (scores.gather(1, top) * 4).round() / 4)
    elif pattern == "clustered":  # the top scores of every long row inside one part's tiles
        scores[:, 300000:324576] += 4.0
    scores = scores.to(dtype)
    lengths = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
    kw = dict(next_n=next_n, max_seq_len=env, deterministic=True)
    if remap:
        max_blocks = (env + bs - 1) // bs + 3
        perm = torch.randperm(b * max_blocks, device="cuda", generator=gen).to(torch.int32)
        table = perm.view(b, max_blocks).contiguous()
        for i, n in enumerate(kv_lens):
            table[i, (n + bs - 1) // bs + 1 :] = -1
        kw["remap"] = (table, bs, 24 * bs, 3)
    outs = []
    for split, ctas, min_tiles, min_row in (
        (False, 0, 1, -1),
        (True, 0, 1, -1),
        (True, rows + 16, 3, 0),
    ):
        monkeypatch.setattr(ss, "_ROW_SPLIT", split)
        monkeypatch.setattr(ss, "_ROW_SPLIT_MIN_ROWS", 0)
        monkeypatch.setattr(ss, "_ROW_SPLIT_CTAS", ctas)
        monkeypatch.setattr(ss, "_ROW_SPLIT_MIN_TILES", min_tiles)
        monkeypatch.setattr(ss, "_ROW_SPLIT_MIN_ROW", min_row)
        out = torch.full((rows, k), -7, dtype=torch.int32, device="cuda")
        ss.run_varlen(scores, lengths, out, **kw)
        torch.cuda.synchronize()
        outs.append(out)
    assert any(lc[0] == "main_ps" for lc in ss._VARLEN_CACHE.values())
    ss._VARLEN_CACHE.clear()
    assert torch.equal(outs[1], outs[0])
    assert torch.equal(outs[2], outs[0])
    if not remap:
        for r in (0, next_n, rows - 1):
            n = max(min(kv_lens[r // next_n] - next_n + r % next_n + 1, env), 0)
            want = _oracle(scores[r, :n].float().cpu(), k) if n > k else set(range(n))
            assert {i for i in outs[0][r].tolist() if i >= 0} == want


@skip_not_sm100
@pytest.mark.parametrize("batch,next_n,kv", [(12, 7, 250000), (24, 7, 150000)])
def test_varlen_row_split_guard_keeps_one_cta_rows(monkeypatch, batch, next_n, kv):
    """Launches whose longest row is below the split guard keep one CTA per row inside the ROW SPLIT kernel; rows
    equal the one-CTA launch's, with the guard at its default and forced off (MIN_ROW 0: the same rows split)."""
    k, env = 2048, 1 << 20
    rows = batch * next_n
    gen = torch.Generator(device="cuda").manual_seed(20261103 + rows)
    scores = torch.randn(rows, env, device="cuda", generator=gen).bfloat16()
    lengths = torch.full((batch,), kv, dtype=torch.int32, device="cuda")
    outs = []
    for split, min_row in ((False, -1), (True, -1), (True, 0)):
        monkeypatch.setattr(ss, "_ROW_SPLIT", split)
        monkeypatch.setattr(ss, "_ROW_SPLIT_MIN_ROWS", 0)
        monkeypatch.setattr(ss, "_ROW_SPLIT_CTAS", 0)
        monkeypatch.setattr(ss, "_ROW_SPLIT_MIN_TILES", 1)
        monkeypatch.setattr(ss, "_ROW_SPLIT_MIN_ROW", min_row)
        out = torch.empty((rows, k), dtype=torch.int32, device="cuda")
        ss.run_varlen(scores, lengths, out, next_n=next_n, max_seq_len=env, deterministic=True)
        torch.cuda.synchronize()
        outs.append(out)
    ss._VARLEN_CACHE.clear()
    assert torch.equal(outs[1], outs[0])
    assert torch.equal(outs[2], outs[0])


def test_row_split_guard_rule(monkeypatch):
    monkeypatch.setattr(ss, "_ROW_SPLIT_MIN_ROW", -1)
    assert ss._row_split_guard((512, 8, 2, 256, 4, False, False)) == 12
    assert ss._row_split_guard((1024, 8, 1, 256, 2, False, False)) == 10
    monkeypatch.setattr(ss, "_ROW_SPLIT_MIN_ROW", 0)
    assert ss._row_split_guard((512, 8, 2, 256, 4, False, False)) == 1
    monkeypatch.setattr(ss, "_ROW_SPLIT_MIN_ROW", 300000)
    assert ss._row_split_guard((1024, 8, 1, 256, 2, False, False)) == 10  # ceil(300000 / 32768)


def test_row_split_ctas_rule(monkeypatch):
    monkeypatch.setattr(ss, "_ROW_SPLIT", True)
    monkeypatch.setattr(ss, "_ROW_SPLIT_CTAS", 0)
    monkeypatch.setattr(ss, "_ROW_SPLIT_MIN_ROWS", 0)
    assert ss._row_split_ctas(168, 152, det=True) == 304
    assert ss._row_split_ctas(84, 152, det=True) == 152  # 1024-thread family: one CTA per SM
    assert ss._row_split_ctas(168, 152, det=False) == 0  # deterministic engine only
    assert ss._row_split_ctas(305, 152, det=True) == 0  # more rows than CTAs
    assert ss._row_split_ctas(257, 160, det=True) == 0  # beyond the in-kernel row map
    monkeypatch.setattr(ss, "_ROW_SPLIT_MIN_ROWS", 149)
    assert ss._row_split_ctas(148, 152, det=True) == 0  # 1024-thread family left to one CTA per row
    assert ss._row_split_ctas(149, 152, det=True) == 304
    monkeypatch.setattr(ss, "_ROW_SPLIT_MIN_ROWS", 0)
    monkeypatch.setattr(ss, "_ROW_SPLIT_CTAS", 200)
    assert ss._row_split_ctas(168, 152, det=True) == 200
    monkeypatch.setattr(ss, "_ROW_SPLIT", False)
    assert ss._row_split_ctas(168, 152, det=True) == 0


@skip_not_sm100
@pytest.mark.parametrize(
    "batch,next_n,kv",
    [
        (24, 7, 301055),  # 512-thread main family; (n - 1) % 2048 >= the tie need
        (24, 7, 600000),  # (n - 1) % 4096 >= the tie need
        (11, 7, 600000),  # 1024-thread main family
        (2, 6, 600000),  # SPLIT main family
    ],
)
def test_varlen_large_boundary_tie_group_exact(batch, next_n, kv):
    """Nearly constant rows: the selection boundary sits inside an exact-tie group of ~n scores, so the deterministic
    degenerate path ranks the group by index (u = ~index from 2^32 - n). The rows must hold exactly the smallest tied
    indices and be the same on every launch, also where the top index bin would run past 2^32 - 1."""
    k, env = 2048, 1 << 20
    rows = batch * next_n
    gen = torch.Generator(device="cuda").manual_seed(20261101 + kv + rows)
    scores = torch.full((rows, env), float("-inf"), device="cuda")
    body = torch.zeros(rows, kv, device="cuda")
    body[torch.rand(rows, kv, device="cuda", generator=gen) < 2e-4] = 3.0
    scores[:, :kv] = body
    for r in range(rows):
        scores[r, kv - next_n + r % next_n + 1 :] = float("-inf")
    lengths = torch.full((batch,), kv, dtype=torch.int32, device="cuda")
    outs = []
    for _ in range(3):
        out = torch.empty((rows, k), dtype=torch.int32, device="cuda")
        ss.run_varlen(scores, lengths, out, next_n=next_n, max_seq_len=env, deterministic=True)
        torch.cuda.synchronize()
        outs.append(out)
    assert torch.equal(outs[0], outs[1]) and torch.equal(outs[0], outs[2])
    for r in (0, rows // 2, rows - 1):
        n = kv - next_n + r % next_n + 1
        assert set(outs[0][r].tolist()) == _oracle(scores[r, :n].cpu(), k)
