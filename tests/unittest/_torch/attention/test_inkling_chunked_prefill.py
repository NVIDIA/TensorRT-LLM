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
"""GPU parity tests for Inkling's chunked-context prefill attention.

The oracle is the existing packed prefill kernel, which is what every Inkling
accuracy number on record was produced with. Two properties matter:

1. **Degeneracy.** With ``num_cached == 0`` the chunked kernel reads the same
   keys in the same tile order as the packed kernel, just through the page
   table. It must agree to within accumulation noise -- if it does not, the
   paged indexing or the absolute-position arithmetic is wrong.
2. **Chunk invariance.** Splitting a prompt must not change the answer. The
   output for tokens ``[c:]`` computed as a second chunk on top of ``c`` cached
   tokens must match the tail of the one-shot result. This is the property the
   feature exists to provide, and the one that silently failed before: the
   packed kernel simply ignored everything before the chunk.

Both run over the local (sliding-window) and global (full-causal) layer shapes
and with the relative-position bias on, since the bias indexing is the part
most likely to be wrong across a chunk boundary.
"""

import pytest
import torch

pytest.importorskip("triton")

from tensorrt_llm._torch.attention_backend.sparse.inkling.kernels import (  # noqa: E402
    build_page_table,
    inkling_chunked_prefill_attention,
    inkling_prefill_attention,
    write_kv_cache_hnd,
)

requires_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

# Inkling's text tower: head_dim 128, 64 q heads / 8 kv heads global, 16 local.
HEAD_DIM = 128
NUM_HEADS = 8
PAGE_SIZE = 32
DTYPE = torch.bfloat16
# The real model uses 512; a smaller window keeps the tests fast while still
# forcing the kernel's whole-tile skip path (lo > 0) to run.
WINDOW = 96
REL_EXTENT = 8


def _rand(*shape, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(*shape, generator=g, device="cuda", dtype=DTYPE)


def _make_case(total_len, num_kv_heads, *, has_rel, seed=0):
    q = _rand(total_len, NUM_HEADS, HEAD_DIM, seed=seed)
    k = _rand(total_len, num_kv_heads, HEAD_DIM, seed=seed + 1)
    v = _rand(total_len, num_kv_heads, HEAD_DIM, seed=seed + 2)
    rel = None
    if has_rel:
        rel = _rand(total_len, NUM_HEADS, REL_EXTENT, seed=seed + 3).float().contiguous()
    return q, k, v, rel


def _empty_cache(num_pages, num_kv_heads):
    shape = (num_pages, num_kv_heads, PAGE_SIZE, HEAD_DIM)
    return torch.zeros(shape, device="cuda", dtype=DTYPE), torch.zeros(
        shape, device="cuda", dtype=DTYPE
    )


def _run_chunk(q, k, v, rel, num_cached, k_cache, v_cache, block_ids, window):
    """Write this chunk's K/V into the pages, then run the chunked kernel."""
    new_len = q.shape[0]
    write_kv_cache_hnd(k_cache, v_cache, k, v, block_ids, num_cached, PAGE_SIZE)
    cu = torch.tensor([0, new_len], dtype=torch.int32, device="cuda")
    nc = torch.tensor([num_cached], dtype=torch.int32, device="cuda")
    total = num_cached + new_len
    max_pages = (total + PAGE_SIZE - 1) // PAGE_SIZE
    # build_page_table writes len(blocks) entries into a row of width
    # max_pages, so hand it exactly the pages this request spans.
    page_table = build_page_table([block_ids[:max_pages]], max_pages, "cuda")
    return inkling_chunked_prefill_attention(
        q,
        k_cache,
        v_cache,
        cu,
        nc,
        page_table,
        PAGE_SIZE,
        new_len,
        HEAD_DIM**-1.0,
        rel,
        REL_EXTENT if rel is not None else 0,
        window,
    )


def _packed(q, k, v, rel, window):
    cu = torch.tensor([0, q.shape[0]], dtype=torch.int32, device="cuda")
    return inkling_prefill_attention(
        q,
        k,
        v,
        cu,
        q.shape[0],
        HEAD_DIM**-1.0,
        rel,
        REL_EXTENT if rel is not None else 0,
        window,
    )


def _assert_close(got, want, what, rtol=2e-2, atol=2e-2):
    got_f, want_f = got.float(), want.float()
    max_abs = (got_f - want_f).abs().max().item()
    cos = torch.nn.functional.cosine_similarity(got_f.flatten(), want_f.flatten(), dim=0).item()
    assert torch.allclose(got_f, want_f, rtol=rtol, atol=atol), (
        f"{what}: max_abs={max_abs:.4g} cos={cos:.6f}"
    )
    assert cos > 0.999, f"{what}: cos={cos:.6f}"


@requires_gpu
@pytest.mark.parametrize("window", [-1, WINDOW], ids=["global", "local"])
@pytest.mark.parametrize("has_rel", [False, True], ids=["norel", "rel"])
@pytest.mark.parametrize("total_len", [1, 63, 64, 65, 200])
def test_degenerates_to_the_packed_kernel_with_no_cached_tokens(window, has_rel, total_len):
    """num_cached == 0 must reproduce the packed kernel.

    Same keys, same tile order, only the indirection differs -- so any gap here
    is a paged-indexing or absolute-position bug, not accumulation noise.
    """
    num_kv_heads = 8
    q, k, v, rel = _make_case(total_len, num_kv_heads, has_rel=has_rel)
    k_cache, v_cache = _empty_cache(16, num_kv_heads)
    block_ids = list(range(16))

    got = _run_chunk(q, k, v, rel, 0, k_cache, v_cache, block_ids, window)
    want = _packed(q, k, v, rel, window)
    _assert_close(got, want, f"len={total_len} window={window} rel={has_rel}", rtol=1e-3, atol=1e-3)


@requires_gpu
@pytest.mark.parametrize("window", [-1, WINDOW], ids=["global", "local"])
@pytest.mark.parametrize("has_rel", [False, True], ids=["norel", "rel"])
@pytest.mark.parametrize("split", [1, 2, 7, 32, 64, 65, 128])
def test_a_split_prompt_matches_one_shot_prefill(window, has_rel, split):
    """The property the feature exists for.

    Splitting at ``split`` must not change the answer for the tokens after the
    split. Before the chunked kernel these differed wildly: the packed kernel
    saw only the second chunk's own tokens.

    ``total_len`` 200 with WINDOW 96 and REL_EXTENT 8 puts splits on both sides
    of the window edge and of the bias extent, and 64/65 straddle BLOCK_M.
    """
    total_len = 200
    num_kv_heads = 8
    q, k, v, rel = _make_case(total_len, num_kv_heads, has_rel=has_rel, seed=11)

    want = _packed(q, k, v, rel, window)[split:]

    k_cache, v_cache = _empty_cache(16, num_kv_heads)
    block_ids = list(range(16))
    # Chunk 1 seeds the cache; its own output is not under test here.
    _run_chunk(
        q[:split],
        k[:split],
        v[:split],
        None if rel is None else rel[:split],
        0,
        k_cache,
        v_cache,
        block_ids,
        window,
    )
    got = _run_chunk(
        q[split:],
        k[split:],
        v[split:],
        None if rel is None else rel[split:].contiguous(),
        split,
        k_cache,
        v_cache,
        block_ids,
        window,
    )
    _assert_close(got, want, f"split={split} window={window} rel={has_rel}")


@requires_gpu
def test_three_chunks_are_the_same_as_one():
    """More than one boundary, and boundaries that are not page-aligned: the
    page table has to be walked, not just offset once."""
    total_len, num_kv_heads, window = 200, 8, WINDOW
    q, k, v, rel = _make_case(total_len, num_kv_heads, has_rel=True, seed=23)
    want = _packed(q, k, v, rel, window)

    k_cache, v_cache = _empty_cache(16, num_kv_heads)
    block_ids = list(range(16))
    outs, lo = [], 0
    for hi in (37, 100, total_len):  # 37 and 100 are not multiples of PAGE_SIZE
        outs.append(
            _run_chunk(
                q[lo:hi],
                k[lo:hi],
                v[lo:hi],
                rel[lo:hi].contiguous(),
                lo,
                k_cache,
                v_cache,
                block_ids,
                window,
            )
        )
        lo = hi
    _assert_close(torch.cat(outs, dim=0), want, "three chunks")


@requires_gpu
def test_grouped_query_attention_maps_heads_correctly():
    """Inkling's local layers carry 16 KV heads against 64 query heads; the
    chunked kernel must fold q->kv the same way (cur_head // kv_group_num) or
    the split output is wrong in a way parity at kv_group_num == 1 cannot see.
    """
    total_len, num_kv_heads, split, window = 128, 2, 33, -1
    q, k, v, rel = _make_case(total_len, num_kv_heads, has_rel=True, seed=31)
    want = _packed(q, k, v, rel, window)[split:]

    k_cache, v_cache = _empty_cache(16, num_kv_heads)
    block_ids = list(range(16))
    _run_chunk(
        q[:split],
        k[:split],
        v[:split],
        rel[:split].contiguous(),
        0,
        k_cache,
        v_cache,
        block_ids,
        window,
    )
    got = _run_chunk(
        q[split:],
        k[split:],
        v[split:],
        rel[split:].contiguous(),
        split,
        k_cache,
        v_cache,
        block_ids,
        window,
    )
    _assert_close(got, want, "gqa split")


# ---------------------------------------------------------------------------
# The conv half of the same property. Attention is only one of the two places
# a split prompt can lose its history; the four depthwise short convs are the
# other. causal_conv1d_fn writes the trailing kernel-1 window into the state
# pool on every context call, so a later chunk only has to DECLARE that it has
# one (has_initial_state) for it to be consumed.
# ---------------------------------------------------------------------------
SCONV_KERNEL = 4
SCONV_CHANNELS = 256


def _conv_state_pool(rows=1):
    return torch.zeros(rows, SCONV_CHANNELS, SCONV_KERNEL - 1, device="cuda", dtype=DTYPE)


def _run_conv_chunk(x, conv_w, state, has_initial):
    """One varlen context call through causal_conv1d_fn, state updated in place."""
    from tensorrt_llm._torch.modules.mamba.causal_conv1d import causal_conv1d_fn

    n = x.shape[0]
    xt = x.transpose(0, 1).contiguous()  # [channels, tokens]
    y = causal_conv1d_fn(
        xt,
        conv_w,
        None,
        query_start_loc=torch.tensor([0, n], dtype=torch.int32, device="cuda"),
        cache_indices=torch.tensor([0], dtype=torch.int32, device="cuda"),
        has_initial_state=torch.tensor([has_initial], dtype=torch.bool, device="cuda"),
        conv_states=state,
        activation=None,
    )
    return y.transpose(0, 1).contiguous()


@requires_gpu
@pytest.mark.parametrize("split", [1, 2, 3, 4, 17, 64])
def test_the_short_conv_carries_its_window_across_a_chunk_boundary(split):
    """Splitting must not change the conv output either.

    Splits at 1..3 are the interesting ones: they are shorter than the
    kernel-1 window, so the second chunk's first outputs depend on tokens the
    first chunk owned. With has_initial_state=False these were convolved
    against zeros -- the defect the review named.
    """
    total_len = 128
    x = _rand(total_len, SCONV_CHANNELS, seed=41)
    conv_w = _rand(SCONV_CHANNELS, SCONV_KERNEL, seed=42)

    state = _conv_state_pool()
    want = _run_conv_chunk(x, conv_w, state, False)

    state = _conv_state_pool()
    _run_conv_chunk(x[:split], conv_w, state, False)
    got = _run_conv_chunk(x[split:], conv_w, state, True)
    _assert_close(got, want[split:], f"conv split={split}")


@requires_gpu
def test_declaring_no_initial_state_on_a_later_chunk_is_visibly_wrong():
    """The negative control: without has_initial_state the second chunk really
    does differ. If this ever passes, the test above proves nothing."""
    total_len, split = 128, 2
    x = _rand(total_len, SCONV_CHANNELS, seed=43)
    conv_w = _rand(SCONV_CHANNELS, SCONV_KERNEL, seed=44)

    state = _conv_state_pool()
    want = _run_conv_chunk(x, conv_w, state, False)[split:]

    state = _conv_state_pool()
    _run_conv_chunk(x[:split], conv_w, state, False)
    wrong = _run_conv_chunk(x[split:], conv_w, state, False)  # the old behaviour

    assert not torch.allclose(wrong.float(), want.float(), rtol=2e-2, atol=2e-2), (
        "dropping the carried window changed nothing -- the parity test above "
        "is not exercising what it claims"
    )


def test_the_package_re_exports_the_chunked_entry_point():
    """modeling_inkling imports from the PACKAGE, not from .kernels.

    This test exists because the tests above import from
    ``...inkling.kernels`` directly and so could not see that
    ``inkling/__init__.py`` never re-exported the new entry point. The whole
    unit suite passed while the model could not import it -- an end-to-end run
    (job 6044664) was what surfaced it. No GPU needed: it is an import.
    """
    import tensorrt_llm._torch.attention_backend.sparse.inkling as pkg

    assert hasattr(pkg, "inkling_chunked_prefill_attention")
    assert "inkling_chunked_prefill_attention" in pkg.__all__


# ---------------------------------------------------------------------------
# The gap the rest of this file leaves open. Every test above hands the kernel
# block_ids = range(16): contiguous, zero-based, ascending. A real
# KVCacheManagerV2 hands out whatever pages are free -- sparse, unordered, and
# never starting at 0 in a warm pool. The end-to-end runs (jobs 6046462 /
# 6047277) show the head of the distribution shifting by ~0.35 logprob when a
# prompt is actually split, and the real page table is the prime suspect
# precisely because nothing here has ever exercised it.
# ---------------------------------------------------------------------------
@requires_gpu
@pytest.mark.parametrize(
    "blocks",
    [
        [9, 2, 14, 5, 11, 0, 7, 3],  # unordered
        [31, 30, 29, 28, 27, 26, 25, 24],  # high, descending, never touches 0
        [4, 12, 20, 28, 36, 44, 52, 60],  # strided, sparse
    ],
    ids=["unordered", "high_descending", "strided"],
)
def test_chunk_invariance_holds_on_a_realistic_page_layout(blocks):
    """Split a prompt across pages the KV manager could plausibly hand out.

    write_kv_cache_hnd and the kernel must agree on the SAME mapping from
    absolute token position to (page, offset). A disagreement shows up here and
    nowhere else in this file: with range(16) a bug that ignores the page table
    and treats pages as contiguous still passes everything above.
    """
    total_len, num_kv_heads, split, window = 200, 8, 37, WINDOW
    q, k, v, rel = _make_case(total_len, num_kv_heads, has_rel=True, seed=71)
    want = _packed(q, k, v, rel, window)[split:]

    k_cache, v_cache = _empty_cache(64, num_kv_heads)
    _run_chunk(
        q[:split],
        k[:split],
        v[:split],
        rel[:split].contiguous(),
        0,
        k_cache,
        v_cache,
        blocks,
        window,
    )
    got = _run_chunk(
        q[split:],
        k[split:],
        v[split:],
        rel[split:].contiguous(),
        split,
        k_cache,
        v_cache,
        blocks,
        window,
    )
    _assert_close(got, want, f"pages={blocks[:4]}...")


@requires_gpu
def test_a_contiguous_page_assumption_would_be_caught():
    """Negative control for the test above: writing with one layout and reading
    with another must NOT agree. If it does, the kernel is ignoring the page
    table and the test above proves nothing."""
    total_len, num_kv_heads, split, window = 200, 8, 37, -1
    q, k, v, rel = _make_case(total_len, num_kv_heads, has_rel=False, seed=73)
    k_cache, v_cache = _empty_cache(64, num_kv_heads)
    write_blocks = [9, 2, 14, 5, 11, 0, 7, 3]

    _run_chunk(q[:split], k[:split], v[:split], None, 0, k_cache, v_cache, write_blocks, window)
    honest = _run_chunk(
        q[split:], k[split:], v[split:], None, split, k_cache, v_cache, write_blocks, window
    )
    # Same cache, but the reader is told a different page order.
    cu = torch.tensor([0, total_len - split], dtype=torch.int32, device="cuda")
    nc = torch.tensor([split], dtype=torch.int32, device="cuda")
    max_pages = (total_len + PAGE_SIZE - 1) // PAGE_SIZE
    wrong_table = build_page_table([list(range(max_pages))], max_pages, "cuda")
    wrong = inkling_chunked_prefill_attention(
        q[split:],
        k_cache,
        v_cache,
        cu,
        nc,
        wrong_table,
        PAGE_SIZE,
        total_len - split,
        HEAD_DIM**-1.0,
        None,
        0,
        window,
    )
    assert not torch.allclose(honest.float(), wrong.float(), rtol=2e-2, atol=2e-2), (
        "reading with the wrong page order changed nothing -- the kernel is not "
        "using the page table"
    )


@requires_gpu
def test_report_the_per_layer_split_divergence_magnitude():
    """Measure, do not just bound.

    The end-to-end runs show the head of the distribution moving ~0.35 logprob
    when a prompt is split (jobs 6046462 / 6047277), and the standing
    explanation is that a correct chunked kernel still re-aligns query tiles at
    the chunk boundary, so the online-softmax accumulation order differs and
    bf16 rounding compounds over 66 layers.

    That explanation is only credible if the PER-LAYER difference is actually
    rounding-sized. The other tests bound it at 2e-2 and stop; this one prints
    it, so the claim rests on a number. Compare against the same measurement
    with num_cached == 0, where the kernels are bit-identical -- that is the
    floor for this measurement.
    """
    total_len, num_kv_heads, window = 200, 8, WINDOW
    q, k, v, rel = _make_case(total_len, num_kv_heads, has_rel=True, seed=91)
    want = _packed(q, k, v, rel, window)

    # Floor: same kernel, no split.
    k_cache, v_cache = _empty_cache(16, num_kv_heads)
    fresh = _run_chunk(q, k, v, rel, 0, k_cache, v_cache, list(range(16)), window)
    f_abs = (fresh.float() - want.float()).abs().max().item()

    print(f"\n  num_cached==0 floor : max_abs={f_abs:.3e}")
    for split in (37, 64, 100):
        k_cache, v_cache = _empty_cache(16, num_kv_heads)
        blocks = list(range(16))
        _run_chunk(
            q[:split],
            k[:split],
            v[:split],
            rel[:split].contiguous(),
            0,
            k_cache,
            v_cache,
            blocks,
            window,
        )
        got = _run_chunk(
            q[split:],
            k[split:],
            v[split:],
            rel[split:].contiguous(),
            split,
            k_cache,
            v_cache,
            blocks,
            window,
        )
        tail = want[split:].float()
        max_abs = (got.float() - tail).abs().max().item()
        rel_rms = ((got.float() - tail).pow(2).mean().sqrt() / tail.pow(2).mean().sqrt()).item()
        cos = torch.nn.functional.cosine_similarity(
            got.float().flatten(), tail.flatten(), dim=0
        ).item()
        print(
            f"  split={split:4d}          : max_abs={max_abs:.3e} rel_rms={rel_rms:.3e} cos={cos:.8f}"
        )
        # Loose: this test reports, the strict bounds live in the tests above.
        assert cos > 0.999, f"split={split} cos={cos}"
