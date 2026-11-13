# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The family MSA backend's NVFP4 index-K reader: pure-decode steps score the NVFP4 pages (eager and under CUDA-graph
replay) and the native page layout selects as the HPR layout."""

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.msa.kernels.msa_utils import (
    select_blocks_from_maxscore,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.msa.msa_backend import (
    MiniMaxM31MsaSparseAttention,
    MiniMaxM31MsaSparseAttentionMetadata,
    MsaDecodeSpan,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.native import family_ops


def _nvfp4_index_k_kernels_available() -> bool:
    """Whether this GPU runs the family's NVFP4 index-K kernels (built for SM107)."""
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() == (10, 7)
        and all(
            hasattr(family_ops(), f"m31idxk_{name}")
            for name in ("nvfp4_idxk_write", "nvfp4_idxk_write_from_sink", "nvfp4_idxk_gather_fp8")
        )
    )


@pytest.mark.skipif(
    not _nvfp4_index_k_kernels_available(), reason="the NVFP4 index-K kernels are built for SM107"
)
@torch.inference_mode()
def test_msa_indexer_nvfp4_pure_decode_reads_pages_under_capture() -> None:
    """A pure-decode NVFP4 step hands the selector the Q8KV4 reader's scores of the NVFP4 pages, eager and replayed
    from a CUDA graph whose captured forward rebuilds the reader's history scheduler from the replayed lengths."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.msa.kernels import (
        nvfp4_index_k,
    )
    from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.common import (
        MiniMaxM3SparseConfig,
    )

    heads, qlen, batch, tiles, num_pages = 4, 8, 3, 6, 24
    attention = MiniMaxM31MsaSparseAttention.__new__(MiniMaxM31MsaSparseAttention)
    attention.m3_config = MiniMaxM3SparseConfig(
        num_q_heads=4,
        num_kv_heads=4,
        head_dim=128,
        num_index_heads=heads,
        sparse_index_dim=128,
        block_size=128,
        topk=16,
    )
    attention.layer_idx = 3
    attention.indexer_kv_dtype = "nvfp4"
    scores = torch.empty((heads, tiles, batch * qlen), dtype=torch.float32, device="cuda")
    table = torch.zeros((batch * qlen, 4, 16), dtype=torch.int32, device="cuda")
    seen = {}

    class FakeIndexer:
        def select_blocks(self, idx_q, idx_k_cache, **kwargs):
            seen.update(kwargs)
            scores.copy_(kwargs["gen_scores"])
            return table

    attention.indexer = FakeIndexer()
    generator = torch.Generator(device="cuda").manual_seed(11)
    cache = torch.randint(
        0,
        256,
        (num_pages, 1, 128, nvfp4_index_k.BYTES_PER_TOKEN),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    # E4M3 16-element scales in [0.25, 2], the last 1024 bytes of each page.
    page_scales = torch.rand((num_pages, 1024), device="cuda", generator=generator) * 1.75 + 0.25
    cache.view(num_pages, -1)[:, 128 * 64 :] = page_scales.to(torch.float8_e4m3fn).view(torch.uint8)

    class FakeMetadata:
        # No KV-manager row bound (the select bound; the real metadata always sets it).
        msa_max_blocks_per_seq = 0
        msa_decode_span = MsaDecodeSpan(row_first=0, query_len=qlen)
        msa_prefill_proxy_plan = None
        msa_prefill_proxy_plan_ref = None
        msa_prefill_proxy_kv_indices = None
        msa_prefill_n_valid_blocks = None
        _msa_prefill_identity = None
        msa_policy = None
        msa_qo_lens_cpu = None
        msa_kv_lens_cpu = None
        msa_qo_offset_cpu = None
        msa_worst_case_max_k_tiles = tiles
        num_seqs = batch
        _msa_nvfp4_decode_ws = None
        msa_nvfp4_decode_workspace = MiniMaxM31MsaSparseAttentionMetadata.msa_nvfp4_decode_workspace

        def __init__(self) -> None:
            self.msa_block_table = torch.zeros((batch, tiles + 2), dtype=torch.int32, device="cuda")
            self.msa_seq_lens_cuda = torch.zeros((batch,), dtype=torch.int32, device="cuda")
            self.msa_n_valid_blocks = torch.zeros((batch * qlen,), dtype=torch.int32, device="cuda")
            self.store = torch.empty((heads * tiles * batch * qlen,), device="cuda")

        def msa_idx_k_cache(self, layer_idx: int) -> torch.Tensor:
            return cache

        def msa_proxy_max_score_view(self, num_heads, max_k_tiles, num_tokens) -> torch.Tensor:
            return self.store[: num_heads * max_k_tiles * num_tokens].view(
                num_heads, max_k_tiles, num_tokens
            )

    metadata = FakeMetadata()
    idx_q = torch.randn((batch * qlen, heads * 128), device="cuda", generator=generator) * 0.5
    idx_q = idx_q.to(torch.float8_e4m3fn)

    def stage(lens):
        """Stage a step's pages and lengths; return the reader's own scores of them."""
        pages = torch.randperm(num_pages, device="cuda", generator=generator)[: batch * (tiles + 2)]
        metadata.msa_block_table.copy_(pages.view(batch, tiles + 2).to(torch.int32))
        metadata.msa_seq_lens_cuda.copy_(torch.tensor(lens, dtype=torch.int32))
        return nvfp4_index_k.score_decode_rows(
            idx_q.view(batch * qlen, heads, 128),
            cache,
            metadata.msa_block_table[:, :tiles],
            metadata.msa_seq_lens_cuda,
            qlen,
            heads,
        ).clone()

    def assert_history_scores(expected, lens):
        # Each query row's history blocks, the ones before the block holding its own position: the reader writes
        # them, and they are all the selector ranks (it forces the local block).
        for row, length in enumerate(lens):
            for offset in range(qlen):
                history = min((length - qlen + offset) // 128, tiles)
                col = row * qlen + offset
                torch.testing.assert_close(
                    scores[:, :history, col], expected[:, :history, col], rtol=0, atol=0
                )

    def forward():
        # on_update_kv_lens opens every forward by dropping the reader's scheduler.
        metadata._msa_nvfp4_decode_ws = None
        return attention.run_indexer(idx_q, None, metadata)

    lens = [300, 700, 129]
    expected = stage(lens)
    assert forward() is table
    assert_history_scores(expected, lens)
    assert seen["gen_token_first"] == 0 and seen["ctx_rows"] == 0
    assert seen["block_table"].data_ptr() == metadata.msa_block_table.data_ptr()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        forward()
    # History page counts 4 / 1 / 4 and 1 / 2 / 5, against the captured step's 2 / 5 / 1.
    for lens in ([520, 129, 640], [129, 300, 700]):
        expected = stage(lens)
        scores.fill_(float("nan"))
        graph.replay()
        assert_history_scores(expected, lens)


@pytest.mark.skipif(
    not _nvfp4_index_k_kernels_available(), reason="the NVFP4 index-K kernels are built for SM107"
)
@torch.inference_mode()
@pytest.mark.parametrize("query_len", [8, 1])
def test_nvfp4_reader_native_layout_selects_as_the_hpr_layout(monkeypatch, query_len) -> None:
    """At query length 8 the reader writes its native [batch, 8 * heads, cols] layout and hands back a strided
    [heads, cols, rows] view: the history scores and the selected table equal the hpr layout's bit for bit, and the
    kill (or another query length) keeps the hpr buffer."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.msa.kernels import (
        nvfp4_index_k,
    )

    heads, batch, cols, num_pages = 4, 3, 40, 160
    generator = torch.Generator(device="cuda").manual_seed(23)
    cache = torch.randint(
        0,
        256,
        (num_pages, 1, 128, nvfp4_index_k.BYTES_PER_TOKEN),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    page_scales = torch.rand((num_pages, 1024), device="cuda", generator=generator) * 1.75 + 0.25
    cache.view(num_pages, -1)[:, 128 * 64 :] = page_scales.to(torch.float8_e4m3fn).view(torch.uint8)
    block_table = (
        torch.randperm(num_pages, device="cuda", generator=generator)[: batch * cols]
        .view(batch, cols)
        .to(torch.int32)
    )
    lens = [3000, 129, 5100]
    seq_lens = torch.tensor(lens, dtype=torch.int32, device="cuda")
    idx_q = (
        torch.randn((batch * query_len, heads, 128), device="cuda", generator=generator) * 0.5
    ).to(torch.float8_e4m3fn)
    n_valid = torch.tensor(
        [-(-(n - query_len + 1 + t) // 128) for n in lens for t in range(query_len)],
        dtype=torch.int32,
        device="cuda",
    )

    def read(native: bool):
        monkeypatch.setattr(nvfp4_index_k, "_NATIVE_LAYOUT", native)
        scores = nvfp4_index_k.score_decode_rows(
            idx_q, cache, block_table, seq_lens, query_len, heads
        )
        return scores, select_blocks_from_maxscore(
            scores,
            topk=16,
            n_valid_blocks=n_valid,
            init_blocks=0,
            local_blocks=1,
            head_major_output=True,
        )

    native_scores, native_table = read(True)
    hpr_scores, hpr_table = read(False)
    assert hpr_scores.is_contiguous()
    if query_len == 8:
        assert native_scores.stride() == (cols, 1, heads * cols)
    else:
        assert native_scores.is_contiguous()
    for row, length in enumerate(lens):
        for offset in range(query_len):
            history = min((length - query_len + offset) // 128, cols)
            col = row * query_len + offset
            torch.testing.assert_close(
                native_scores[:, :history, col].view(torch.int32),
                hpr_scores[:, :history, col].view(torch.int32),
                rtol=0,
                atol=0,
            )
    assert torch.equal(native_table, hpr_table)
