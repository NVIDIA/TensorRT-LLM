# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Token-major row expansion for the DSv4 MLA generation ops.

The kernel-level tests in ``test_deepseek_v4_sparse_mla.py`` call the ops with
hand-built tensors, so they pass whatever the metadata layer produces -- they
cannot see a view that is assembled wrongly. That gap let a block table shaped
``[rows, max_blocks]`` reach an op that reads
``[num_pools, num_seqs, 2, max_blocks]``: the gather collapsed the leading axes
until only two were left, which stripped ``num_seqs`` rather than the pool axis
and left request ids indexing a length-2 K/V axis. It surfaced only on an
8-GPU end-to-end run, as a device-side ``vectorized gather kernel index out of
bounds``.

These tests exercise the expansion itself against a stub cache manager, so the
same mistake fails in seconds on one GPU.
"""

import pytest
import torch

from tensorrt_llm._torch.attention_backend.sparse.dsa import DSAtrtllmAttentionMetadata


class _StubCacheManager:
    """Only the attributes the row expansion reads."""

    def __init__(self, num_pools: int, max_blocks_per_seq: int) -> None:
        self.num_pools = num_pools
        self.num_attention_op_pools = num_pools
        self.max_blocks_per_seq = max_blocks_per_seq


def _metadata(
    *,
    num_pools: int,
    max_blocks: int,
    max_seqs: int,
    draft: int = 5,
    enable_ragged_verification: bool = True,
) -> DSAtrtllmAttentionMetadata:
    """A metadata object with just enough state to build its own buffers.

    Constructed through ``__new__`` rather than the real constructor: the real
    one wants a runtime, a resource manager and a model config, none of which
    the row expansion touches. What it does touch is set explicitly below, so a
    new dependency shows up as a clear AttributeError here instead of being
    silently satisfied by a mock.
    """
    meta = DSAtrtllmAttentionMetadata.__new__(DSAtrtllmAttentionMetadata)
    meta.kv_cache_manager = _StubCacheManager(num_pools, max_blocks)
    meta.max_num_sequences = max_seqs
    meta.max_num_requests = max_seqs
    meta.max_draft_tokens = draft
    meta.enable_ragged_verification = enable_ragged_verification
    meta.cuda_graph_buffers = None
    meta.num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    meta.runtime_tokens_per_gen_step = 0
    meta.create_expanded_buffers(capture_graph=False)
    meta.prompt_lens_cpu = torch.zeros(max_seqs, dtype=torch.int32)
    return meta


def test_feature_off_skips_ragged_row_buffers() -> None:
    """Static DSA does not pay for confidence-only row metadata."""
    torch.cuda.init()
    meta = _metadata(
        num_pools=2,
        max_blocks=8,
        max_seqs=4,
        enable_ragged_verification=False,
    )

    assert meta.gen_token_repeats_cuda is None
    assert meta.row_kv_lens_cuda is None
    assert meta.row_req_idx_cuda is None
    assert meta.attn_row_kv_lens_cuda is None
    assert meta.attn_row_block_offsets is None


def test_feature_off_rejects_ragged_layout() -> None:
    """An unexpected ragged update fails instead of dereferencing absent buffers."""
    meta = DSAtrtllmAttentionMetadata.__new__(DSAtrtllmAttentionMetadata)
    meta.enable_ragged_verification = False
    meta.ragged_verify_lens = [2]

    with pytest.raises(RuntimeError, match="without enabling ragged-verification buffers"):
        _ = meta.is_ragged_verify


def _expand(
    meta: DSAtrtllmAttentionMetadata,
    *,
    num_contexts: int,
    row_req_idx: list[int],
    num_seqs: int,
    block_offsets: torch.Tensor,
) -> int:
    """Drive the private expansion with an explicit row->request map."""
    ngt = len(row_req_idx)
    meta.num_contexts = num_contexts
    meta._ragged_num_rows = ngt
    meta.ragged_verify_lens = [1] * ngt
    meta.kv_cache_block_offsets = block_offsets
    meta.row_req_idx_host[:ngt].copy_(torch.tensor(row_req_idx, dtype=torch.long))
    meta.row_kv_lens_host[:ngt].fill_(7)
    meta.row_kv_correction_host[:ngt].zero_()
    meta.prompt_lens_cpu[:num_seqs].fill_(1)
    meta._prepare_token_major_gen_rows(torch.full((num_seqs,), 7, dtype=torch.int32))
    return meta._attn_num_rows


@pytest.mark.parametrize("num_pools", [1, 2])
def test_row_block_table_matches_op_layout(num_pools: int) -> None:
    """The expanded table keeps the op's rank, axis order and pool/KV axes."""
    torch.cuda.init()
    max_blocks, max_seqs = 8, 4
    meta = _metadata(num_pools=num_pools, max_blocks=max_blocks, max_seqs=max_seqs)

    src = torch.arange(
        num_pools * max_seqs * 2 * max_blocks, dtype=torch.int32, device="cuda"
    ).reshape(num_pools, max_seqs, 2, max_blocks)
    # Request 2 verifies three positions, request 0 one: a ragged batch, which
    # is the whole point -- rows per request differ.
    row_req_idx = [0, 2, 2, 2]
    rows = _expand(
        meta, num_contexts=0, row_req_idx=row_req_idx, num_seqs=max_seqs, block_offsets=src
    )
    assert rows == len(row_req_idx)

    got = meta.token_major_gen_view().kv_cache_block_offsets
    assert got.dim() == src.dim(), (
        f"expanded table has rank {got.dim()} but the attention op reads rank "
        f"{src.dim()}; substituting it wholesale would misread every axis"
    )
    assert got.shape[0] == num_pools and got.shape[2] == 2
    assert got.shape[1] == rows
    # Every row must carry its own request's blocks, on every pool and both
    # directions.
    for row, req in enumerate(row_req_idx):
        torch.testing.assert_close(got[:, row], src[:, req])


def test_row_indices_stay_in_bounds_of_the_sequence_axis() -> None:
    """A request id must index num_seqs, never a shorter axis.

    This is the assertion the device-side gather was making. Checking it here
    means the failure is a readable Python error rather than a CUDA abort that
    takes the whole 8-rank job with it.
    """
    torch.cuda.init()
    meta = _metadata(num_pools=1, max_blocks=8, max_seqs=4)

    src = torch.zeros(1, 4, 2, 8, dtype=torch.int32, device="cuda")
    rows = _expand(meta, num_contexts=0, row_req_idx=[0, 1, 3, 3], num_seqs=4, block_offsets=src)
    idx = meta.attn_row_req_idx_cuda[:rows]
    assert int(idx.min()) >= 0
    assert int(idx.max()) < src.shape[1], (
        f"row->request map reaches {int(idx.max())} but the sequence axis is {src.shape[1]} long"
    )


def test_context_rows_precede_generation_rows() -> None:
    """Contexts keep row 0..nc-1 so the ops' generation offset still lands."""
    torch.cuda.init()
    meta = _metadata(num_pools=1, max_blocks=8, max_seqs=6)

    src = torch.arange(1 * 6 * 2 * 8, dtype=torch.int32, device="cuda").reshape(1, 6, 2, 8)
    nc = 2
    rows = _expand(meta, num_contexts=nc, row_req_idx=[0, 1, 1], num_seqs=6, block_offsets=src)
    assert rows == nc + 3
    idx = meta.attn_row_req_idx_host[:rows].tolist()
    assert idx[:nc] == list(range(nc)), "context rows must be identity-mapped"
    # Generation row->request ids are batch-relative and must be shifted past
    # the contexts, or a generation row would read a context's blocks.
    assert idx[nc:] == [nc + 0, nc + 1, nc + 1]
    # host_context_lengths is the ops' shape oracle: one token per generation
    # row, real length for contexts.
    assert meta.attn_row_prompt_lens_cpu[nc:rows].tolist() == [1, 1, 1]


def test_device_layout_preserves_mixed_context_row_and_token_prefixes() -> None:
    """Device-selected generation windows start after both context prefixes."""
    torch.cuda.init()
    meta = _metadata(num_pools=1, max_blocks=8, max_seqs=4)
    block_offsets = torch.arange(1 * 4 * 2 * 8, dtype=torch.int32, device="cuda").reshape(
        1, 4, 2, 8
    )
    rows = _expand(
        meta,
        num_contexts=2,
        row_req_idx=[0, 1, 1],
        num_seqs=4,
        block_offsets=block_offsets,
    )
    assert rows == 5

    meta.num_generations = 2
    meta._num_ctx_tokens = 5
    meta.seq_lens = torch.tensor([2, 3, 1, 2], dtype=torch.int32)
    meta._seq_lens_cuda = meta.seq_lens.to(device="cuda")
    meta.req_idx_per_token = torch.full((32,), -1, dtype=torch.int32, device="cuda")
    context_token_requests = torch.tensor([0, 0, 1, 1, 1], dtype=torch.int32, device="cuda")
    meta.req_idx_per_token[:5].copy_(context_token_requests)
    context_row_requests = torch.tensor([0, 1], dtype=torch.long, device="cuda")
    context_row_corrections = torch.tensor([10, 20], dtype=torch.int32, device="cuda")
    meta.attn_row_req_idx_cuda[:2].copy_(context_row_requests)
    meta.attn_row_kv_correction_cuda[:2].copy_(context_row_corrections)
    meta.indexer_k_cache_block_offsets = torch.arange(
        4 * 8, dtype=torch.int32, device="cuda"
    ).reshape(4, 8)

    meta.apply_device_ragged_layout(
        verify_lens=torch.tensor([1, 2], dtype=torch.int32, device="cuda"),
        req_idx=torch.tensor([0, 1, 1], dtype=torch.long, device="cuda"),
        kv_correction=torch.tensor([-2, -1, 0], dtype=torch.int32, device="cuda"),
    )

    assert torch.equal(meta.attn_row_req_idx_cuda[:5], torch.tensor([0, 1, 2, 3, 3], device="cuda"))
    assert torch.equal(
        meta.attn_row_kv_correction_cuda[:5],
        torch.tensor([10, 20, -2, -1, 0], dtype=torch.int32, device="cuda"),
    )
    assert torch.equal(meta.req_idx_per_token[:5], context_token_requests)
    assert torch.equal(
        meta.req_idx_per_token[5:8],
        torch.tensor([2, 3, 3], dtype=torch.int32, device="cuda"),
    )


def test_every_runtime_view_shares_the_row_count() -> None:
    """All views the op reads must agree on batch_size, which is the row count.

    ``TrtllmAttention.forward`` asserts this directly, and the C++ op indexes
    ``request_types`` over its whole ``num_seqs`` -- so a view left at the
    request count is an out-of-bounds read there rather than a shape error.
    """
    torch.cuda.init()
    meta = _metadata(num_pools=1, max_blocks=8, max_seqs=6)
    src = torch.zeros(1, 6, 2, 8, dtype=torch.int32, device="cuda")
    nc = 1
    rows = _expand(meta, num_contexts=nc, row_req_idx=[0, 0, 1], num_seqs=6, block_offsets=src)

    view = meta.token_major_gen_view()
    for name in (
        "sequence_length",
        "host_past_key_value_lengths",
        "host_context_lengths",
        "prompt_lens_cuda",
        "host_request_types",
    ):
        got = getattr(view, name).shape[0]
        assert got == rows, f"{name} has {got} entries but the ops will read {rows} rows"
    assert view.kv_cache_block_offsets.shape[1] == rows

    # kCONTEXT is 0, kGENERATION is 1; the op checks every row at or past
    # num_contexts is a generation row.
    types = view.host_request_types.tolist()
    assert types[:nc] == [0] * nc
    assert types[nc:] == [1] * (rows - nc)
