# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""M3/M4 validation: in-tree decode driver vs MSA api path, bit-exact.

Runs the same JIT-compiled SM100 kernel binaries through (a) MSA's
host-centric ``fmha_sm100_plan`` / ``fmha_sm100`` driver and (b) the
in-tree graph-safe ``dispatch.M3DecodeKernelDriver``, on identical
inputs, and asserts bit-equality:

* proxy MQA max-score pass — uniform and heterogeneous KV lens;
* top-k block selection — bit-diff vs ``sparse_topk_select`` on
  uniform lens (global == per-row there), reference-diff vs a pure
  Python implementation on heterogeneous lens;
* sparse block-GQA pass — same ``kv_block_indexes`` fed to both;
* full pipeline under CUDA graph capture/replay with mutated lengths
  and contents between replays (the property MSA's driver lacks).

Requires SM100 + the ``fmha_sm100`` package (see requirements.txt).
"""

import math

import pytest
import torch

# Full-model per-rank M3 geometry.
NUM_Q_HEADS = 64
NUM_KV_HEADS = 4
NUM_INDEX_HEADS = 4
HEAD_DIM = 128
PAGE_SIZE = 128
TOPK = 16
INIT_BLOCKS = 0
LOCAL_BLOCKS = 1
MAX_KV_LEN = 2048  # 16 pages -> max_k_tiles rounds to 128, same as MSA's

SM_SCALE = HEAD_DIM**-0.5
IDX_SM_SCALE = HEAD_DIM**-0.5


def _require_env():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    major, _ = torch.cuda.get_device_capability()
    if major != 10:
        pytest.skip("SM100 (Blackwell) required")
    try:
        import fmha_sm100  # noqa: F401
    except ImportError:
        pytest.skip("fmha_sm100 (MSA) not importable")


def _geometry(max_batch):
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.decode_wrapper.dispatch import (  # noqa: E501
        M3DecodeGeometry,
    )

    return M3DecodeGeometry(
        num_q_heads=NUM_Q_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        num_index_heads=NUM_INDEX_HEADS,
        head_dim=HEAD_DIM,
        page_size=PAGE_SIZE,
        topk=TOPK,
        init_blocks=INIT_BLOCKS,
        local_blocks=LOCAL_BLOCKS,
        max_batch=max_batch,
        max_kv_len=MAX_KV_LEN,
    )


def _make_inputs(kv_lens, seed=0, pool_pages=None):
    """Build synthetic paged caches + decode Q for the given KV lengths.

    ``pool_pages`` fixes the physical page-pool size so tensors keep
    identical shapes across calls (required by the CUDA graph test's
    in-place refreshes).
    """
    device = torch.device("cuda")
    gen = torch.Generator(device="cuda").manual_seed(seed)
    batch = len(kv_lens)
    kv_lens_t = torch.tensor(kv_lens, dtype=torch.int32)
    num_pages = [(kv_len + PAGE_SIZE - 1) // PAGE_SIZE for kv_len in kv_lens]
    total_pages = sum(num_pages)
    if pool_pages is None:
        pool_pages = max(total_pages, 1)
    assert pool_pages >= total_pages, "pool too small for requested kv_lens"

    def r(*shape, dtype=torch.bfloat16, scale=1.0):
        return (torch.randn(*shape, generator=gen, device=device, dtype=torch.float32) * scale).to(
            dtype
        )

    q = r(batch, NUM_Q_HEADS, HEAD_DIM, scale=0.5)
    idx_q = r(batch, NUM_INDEX_HEADS, HEAD_DIM, scale=0.5)
    k_paged = r(pool_pages, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, scale=0.5)
    v_paged = r(pool_pages, NUM_KV_HEADS, PAGE_SIZE, HEAD_DIM, scale=0.5)
    idx_k_paged = r(pool_pages, 1, PAGE_SIZE, HEAD_DIM, scale=0.5)

    # Shuffled physical page assignment (exercises kv_indices gather).
    perm = torch.randperm(pool_pages, generator=gen, device=device)[:total_pages]
    kv_indices = perm.to(torch.int32)
    kv_page_indptr = torch.zeros(batch + 1, dtype=torch.int32, device=device)
    kv_page_indptr[1:] = torch.cumsum(torch.tensor(num_pages, dtype=torch.int32, device=device), 0)

    return {
        "batch": batch,
        "q": q,
        "idx_q": idx_q,
        "k_paged": k_paged,
        "v_paged": v_paged,
        "idx_k_paged": idx_k_paged,
        "kv_indices": kv_indices,
        "kv_page_indptr": kv_page_indptr,
        "kv_lens_cpu": kv_lens_t,
        "seq_lens_dev": kv_lens_t.to(device),
    }


# ---------------------------------------------------------------------------
# MSA reference path (mirrors msa_backend.py decode exactly)
# ---------------------------------------------------------------------------


def _msa_proxy_max_score(inp):
    import fmha_sm100

    batch = inp["batch"]
    qo_lens_cpu = torch.ones(batch, dtype=torch.int32)
    qo_offset_cpu = (inp["kv_lens_cpu"] - 1).to(torch.int32)
    plan = fmha_sm100.fmha_sm100_plan(
        qo_lens_cpu,
        inp["kv_lens_cpu"],
        NUM_INDEX_HEADS,
        num_kv_heads=1,
        qo_offset=qo_offset_cpu,
        page_size=PAGE_SIZE,
        output_maxscore=True,
        causal=False,
        num_kv_splits=1,
    )
    _, max_score = fmha_sm100.fmha_sm100(
        inp["idx_q"],
        inp["idx_k_paged"],
        inp["idx_k_paged"],
        plan,
        kv_indices=inp["kv_indices"],
        output_o=False,
        output_maxscore=True,
        sm_scale=IDX_SM_SCALE,
    )
    return max_score


def _msa_topk(max_score, kv_lens_cpu):
    import fmha_sm100

    max_valid = int(((kv_lens_cpu + PAGE_SIZE - 1) // PAGE_SIZE).max().item())
    return fmha_sm100.sparse_topk_select(
        max_score.contiguous(),
        TOPK,
        num_valid_pages=max_valid,
        force_begin_blocks=INIT_BLOCKS,
        force_end_blocks=LOCAL_BLOCKS,
    )


def _msa_sparse(inp, kv_block_indexes, causal=False):
    """MSA reference sparse pass.

    ``causal=False`` is the production decode configuration: MSA then
    overwrites ``qo_offset`` with the *global* ``max_kv_len``, which in
    sparse mode (no secondary seqlen clip) lets short requests attend
    stale positions inside forced/OOB blocks — a real MSA hetero-batch
    defect. ``causal=True`` keeps the per-request ``kv_len - 1``
    offsets and is the exact-semantics reference the in-tree driver
    implements; both agree on uniform batches.
    """
    import fmha_sm100

    batch = inp["batch"]
    qo_lens_cpu = torch.ones(batch, dtype=torch.int32)
    qo_offset_cpu = (inp["kv_lens_cpu"] - 1).to(torch.int32)
    plan = fmha_sm100.fmha_sm100_plan(
        qo_lens_cpu,
        inp["kv_lens_cpu"],
        NUM_Q_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        qo_offset=qo_offset_cpu,
        page_size=PAGE_SIZE,
        kv_block_num=TOPK,
        causal=causal,
        num_kv_splits=1,
    )
    out, _ = fmha_sm100.fmha_sm100(
        inp["q"],
        inp["k_paged"],
        inp["v_paged"],
        plan,
        kv_indices=inp["kv_indices"],
        kv_block_indexes=kv_block_indexes,
        sm_scale=SM_SCALE,
        output_maxscore=False,
    )
    return out


# ---------------------------------------------------------------------------
# In-tree driver path
# ---------------------------------------------------------------------------


def _driver(max_batch):
    from tensorrt_llm._torch.attention_backend.sparse.minimax_m3.decode_wrapper.dispatch import (  # noqa: E501
        get_decode_driver,
    )

    return get_decode_driver(_geometry(max_batch), torch.device("cuda"))


def _intree_proxy(driver, inp):
    return driver.proxy_max_score(
        inp["idx_q"],
        inp["idx_k_paged"],
        seq_lens=inp["seq_lens_dev"],
        kv_page_indptr=inp["kv_page_indptr"],
        kv_indices=inp["kv_indices"],
        sm_scale=IDX_SM_SCALE,
    )


def _intree_sparse(driver, inp, kv_block_indexes):
    return driver.sparse_attention(
        inp["q"],
        inp["k_paged"],
        inp["v_paged"],
        kv_block_indexes,
        seq_lens=inp["seq_lens_dev"],
        kv_page_indptr=inp["kv_page_indptr"],
        kv_indices=inp["kv_indices"],
        sm_scale=SM_SCALE,
    )


# ---------------------------------------------------------------------------
# Pure-Python top-k reference (per-row semantics)
# ---------------------------------------------------------------------------


def _reference_topk(max_score_kv, kv_lens):
    """Per-(token, kv-head) reference: force init/local, top-k, ascending."""
    num_kv_heads, max_k_tiles, total_q = max_score_kv.shape
    scores = max_score_kv.float().cpu().numpy()
    out = torch.full((total_q, num_kv_heads, TOPK), -1, dtype=torch.int32)
    for t in range(total_q):
        valid = (int(kv_lens[t]) + PAGE_SIZE - 1) // PAGE_SIZE
        for h in range(num_kv_heads):
            row = scores[h, :, t].copy()
            row[valid:] = -math.inf
            for k in range(min(INIT_BLOCKS, valid)):
                row[k] = math.inf
            for k in range(max(valid - LOCAL_BLOCKS, 0), valid):
                row[k] = math.inf
            order = sorted(range(max_k_tiles), key=lambda i: (-row[i], i))[:TOPK]
            picked = sorted(i for i in order if row[i] != -math.inf)
            for j, blk in enumerate(picked):
                out[t, h, j] = blk
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

UNIFORM_LENS = [512] * 8
HETERO_LENS = [1, 130, 257, 128, 511, 1024, 33, 900]


@pytest.mark.parametrize("kv_lens", [UNIFORM_LENS, HETERO_LENS], ids=["uniform", "hetero"])
def test_proxy_max_score_bitdiff(kv_lens):
    _require_env()
    inp = _make_inputs(kv_lens)
    driver = _driver(max_batch=len(kv_lens))

    ms_ref = _msa_proxy_max_score(inp)
    ms_new = _intree_proxy(driver, inp)
    torch.cuda.synchronize()

    assert ms_ref.shape == ms_new.shape, f"{ms_ref.shape} vs {ms_new.shape}"
    same = ms_ref == ms_new
    both_neginf = torch.isinf(ms_ref) & torch.isinf(ms_new) & (ms_ref == ms_new)
    mismatch = (~same & ~both_neginf).sum().item()
    assert mismatch == 0, f"proxy max_score mismatches: {mismatch}/{ms_ref.numel()}"


def test_topk_bitdiff_uniform():
    _require_env()
    inp = _make_inputs(UNIFORM_LENS)
    driver = _driver(max_batch=len(UNIFORM_LENS))

    ms = _msa_proxy_max_score(inp)
    blocks_ref = _msa_topk(ms, inp["kv_lens_cpu"])
    blocks_new = driver.select_blocks(ms, seq_lens=inp["seq_lens_dev"])
    torch.cuda.synchronize()

    assert torch.equal(blocks_ref, blocks_new), (
        f"topk mismatch rows: {(blocks_ref != blocks_new).any(dim=-1).nonzero()[:8].tolist()}"
    )


def test_topk_reference_hetero():
    _require_env()
    inp = _make_inputs(HETERO_LENS)
    driver = _driver(max_batch=len(HETERO_LENS))

    ms = _intree_proxy(driver, inp)
    blocks_new = driver.select_blocks(ms, seq_lens=inp["seq_lens_dev"]).cpu()
    torch.cuda.synchronize()
    blocks_ref = _reference_topk(ms, HETERO_LENS)

    assert torch.equal(blocks_ref, blocks_new), (
        f"per-row topk mismatch, first rows: ref={blocks_ref[:2].tolist()} "
        f"new={blocks_new[:2].tolist()}"
    )


@pytest.mark.parametrize(
    "kv_lens",
    [UNIFORM_LENS, [130] * 8, HETERO_LENS],
    ids=["uniform", "uniform_partial_page", "hetero"],
)
def test_sparse_gqa_bitdiff(kv_lens):
    _require_env()
    inp = _make_inputs(kv_lens)
    driver = _driver(max_batch=len(kv_lens))

    # Use MSA's own block selection for both sides to isolate the
    # sparse kernel + driver comparison.  Heterogeneous batches need
    # the causal=True reference (per-request offsets — see _msa_sparse
    # docstring); uniform batches match the production causal=False
    # path bit-exactly as well.
    ms = _msa_proxy_max_score(inp)
    blocks = _msa_topk(ms, inp["kv_lens_cpu"])

    # causal=True reference whenever MSA's causal=False global-offset
    # shortcut would attend stale positions (hetero lens, or a
    # partially filled last page).
    needs_exact_ref = len(set(kv_lens)) > 1 or any(kv_len % PAGE_SIZE for kv_len in kv_lens)
    out_ref = _msa_sparse(inp, blocks, causal=needs_exact_ref)
    out_new = _intree_sparse(driver, inp, blocks)
    torch.cuda.synchronize()

    assert out_ref.shape == out_new.shape
    assert torch.equal(out_ref, out_new), (
        f"sparse out mismatch: max abs diff "
        f"{(out_ref.float() - out_new.float()).abs().max().item()}"
    )


def test_full_pipeline_bitdiff_uniform():
    _require_env()
    inp = _make_inputs(UNIFORM_LENS)
    driver = _driver(max_batch=len(UNIFORM_LENS))

    ms_ref = _msa_proxy_max_score(inp)
    blocks_ref = _msa_topk(ms_ref, inp["kv_lens_cpu"])
    out_ref = _msa_sparse(inp, blocks_ref)

    ms_new = _intree_proxy(driver, inp)
    blocks_new = driver.select_blocks(ms_new, seq_lens=inp["seq_lens_dev"])
    out_new = _intree_sparse(driver, inp, blocks_new)
    torch.cuda.synchronize()

    assert torch.equal(blocks_ref, blocks_new)
    assert torch.equal(out_ref, out_new)


def test_cuda_graph_replay_tracks_device_state():
    """Capture once, mutate lengths + contents, replay: must match eager.

    This is exactly the failure mode of the MSA driver (frozen plan) —
    the in-tree driver must produce bit-identical results to its own
    eager execution for every replay.
    """
    _require_env()
    batch = 8
    driver = _driver(max_batch=batch)
    driver.warmup_shapes(batch)

    # Persistent input buffers the graph will read.
    pool_pages = batch * (MAX_KV_LEN // PAGE_SIZE)
    inp0 = _make_inputs([256] * batch, seed=1, pool_pages=pool_pages)
    seq_lens = inp0["seq_lens_dev"].clone()
    kv_page_indptr = inp0["kv_page_indptr"].clone()
    kv_indices_buf = torch.zeros(
        batch * (MAX_KV_LEN // PAGE_SIZE), dtype=torch.int32, device="cuda"
    )
    n0 = inp0["kv_indices"].shape[0]
    kv_indices_buf[:n0] = inp0["kv_indices"]
    q = inp0["q"].clone()
    idx_q = inp0["idx_q"].clone()
    k_paged = inp0["k_paged"].clone()
    v_paged = inp0["v_paged"].clone()
    idx_k_paged = inp0["idx_k_paged"].clone()

    def run_pipeline():
        ms = driver.proxy_max_score(
            idx_q,
            idx_k_paged,
            seq_lens=seq_lens,
            kv_page_indptr=kv_page_indptr,
            kv_indices=kv_indices_buf,
            sm_scale=IDX_SM_SCALE,
        )
        blocks = driver.select_blocks(ms, seq_lens=seq_lens)
        return driver.sparse_attention(
            q,
            k_paged,
            v_paged,
            blocks,
            seq_lens=seq_lens,
            kv_page_indptr=kv_page_indptr,
            kv_indices=kv_indices_buf,
            sm_scale=SM_SCALE,
        )

    # Warm up (JIT + shape caches + allocator) then capture.
    out_view = run_pipeline()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out_view = run_pipeline()

    for step, lens in enumerate(
        [[256] * batch, [384] * batch, [128, 512, 1920, 256, 640, 64, 2048, 300]]
    ):
        # Refresh device state in place (as prepare() would).
        inp = _make_inputs(lens, seed=10 + step, pool_pages=pool_pages)
        seq_lens.copy_(inp["seq_lens_dev"])
        kv_page_indptr.copy_(inp["kv_page_indptr"])
        n = inp["kv_indices"].shape[0]
        kv_indices_buf.zero_()
        kv_indices_buf[:n] = inp["kv_indices"]
        q.copy_(inp["q"])
        idx_q.copy_(inp["idx_q"])
        k_paged.copy_(inp["k_paged"])
        v_paged.copy_(inp["v_paged"])
        idx_k_paged.copy_(inp["idx_k_paged"])

        graph.replay()
        torch.cuda.synchronize()
        replay_out = out_view.clone()

        eager_out = run_pipeline()
        torch.cuda.synchronize()

        assert torch.equal(replay_out, eager_out), (
            f"replay step {step} (lens={lens[:4]}...) diverges from eager: "
            f"max abs diff "
            f"{(replay_out.float() - eager_out.float()).abs().max().item()}"
        )


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-x"]))
