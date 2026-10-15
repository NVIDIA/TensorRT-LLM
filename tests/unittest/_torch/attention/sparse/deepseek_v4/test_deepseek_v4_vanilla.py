# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the DeepSeek-V4 Vanilla golden.

They cover what the end-to-end comparisons (``test_deepseek_v4_sparse_mla.py``
and the backend sweep) cannot isolate: the window and selection masks against an
independent dense formulation on shuffled pages, RoPE and SWA-append ownership
per phase, and the checks that run before any cache write.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from tensorrt_llm._torch.attention.backends.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    MLAParams,
    PositionalEmbeddingParams,
    RopeParams,
)
from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import (
    DeepseekV4AttentionType,
    DeepSeekV4Params,
    DeepseekV4VanillaAttention,
    vanilla_backend,
)
from tensorrt_llm._torch.attention.backends.sparse.params import SparseBackendForwardArgs
from tensorrt_llm._torch.attention.rotary_embedding import RotaryEmbedding
from tensorrt_llm.functional import PositionEmbeddingType

NUM_HEADS = 2
KV_LORA_RANK = 6
ROPE_DIM = 2
HEAD_DIM = KV_LORA_RANK + ROPE_DIM
WINDOW = 4
SWA_TOKENS_PER_PAGE = 4
COMPRESSED_TOKENS_PER_PAGE = 2
NUM_POOL_PAGES = 32
DEVICE = "cuda"
ROPE_PARAMS = RopeParams(dim=ROPE_DIM, max_positions=512)


@pytest.fixture(autouse=True)
def _exact_fp32_matmul(monkeypatch):
    """Compare FP32 results at 1e-5 with TF32 matmuls off, whatever the process default."""
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)


def _make_backend(compress_ratio: int) -> DeepseekV4VanillaAttention:
    """Build the golden for layer 0 on any GPU.

    The SM check guards model runs, and the production indexer and compressor
    the backend owns serve only the DeepSeek-V4 module, so both are stubbed.
    """
    with (
        patch.object(vanilla_backend, "get_sm_version", return_value=100),
        patch.object(vanilla_backend, "DeepseekV4Indexer"),
        patch.object(vanilla_backend, "Compressor"),
    ):
        return DeepseekV4VanillaAttention(
            layer_idx=0,
            num_heads=NUM_HEADS,
            head_dim=HEAD_DIM,
            num_kv_heads=1,
            q_scaling=1.0,
            pos_embd_params=PositionalEmbeddingParams(
                type=PositionEmbeddingType.rope_gptj, rope=ROPE_PARAMS, is_neox=False
            ),
            mla_params=MLAParams(
                q_lora_rank=8,
                kv_lora_rank=KV_LORA_RANK,
                qk_rope_head_dim=ROPE_DIM,
                qk_nope_head_dim=KV_LORA_RANK,
                v_head_dim=KV_LORA_RANK,
                hidden_size=16,
            ),
            sparse_params=DeepSeekV4Params(compress_ratios=[compress_ratio], window_size=WINDOW),
        )


def _shuffled_table(needed_blocks: range, num_blocks: int, free_pages: list[int]) -> torch.Tensor:
    """Return an int32 [num_blocks] table mapping needed blocks to free pages.

    ``free_pages`` is a shuffled pool shared by all requests, so requests never
    share a page. Every other block maps to -1, so a read outside the needed
    range fails.
    """
    table = torch.full((num_blocks,), -1, dtype=torch.int32)
    for block in needed_blocks:
        table[block] = free_pages.pop()
    return table.to(DEVICE)


def _write_rows(pool: torch.Tensor, table: torch.Tensor, start: int, rows: torch.Tensor) -> None:
    """Store ``rows`` at request-local indices ``start, start + 1, ...``."""
    indices = torch.arange(start, start + rows.shape[0], device=DEVICE)
    tokens_per_page = pool.shape[1]
    pool[table[indices // tokens_per_page].long(), indices % tokens_per_page] = rows


def _read_rows(pool: torch.Tensor, table: torch.Tensor, start: int, count: int) -> torch.Tensor:
    indices = torch.arange(start, start + count, device=DEVICE)
    tokens_per_page = pool.shape[1]
    return pool[table[indices // tokens_per_page].long(), indices % tokens_per_page]


def _rotate(x: torch.Tensor, positions: list[int]) -> torch.Tensor:
    """Rotate [T, H * R] RoPE slices with RotaryEmbedding's single-target torch path."""
    emb = RotaryEmbedding(ROPE_PARAMS, head_dim=ROPE_DIM, is_neox=False)
    return emb(torch.tensor(positions, device=DEVICE), [x])[0]


class _Scenario:
    """One phase of a two-pool batch, with full logical histories kept on the side.

    Each request owns shuffled SWA and COMPRESS pages for exactly the rows its
    tokens may read; every other block maps to -1. Rows outside a request's
    windows are never stored, so the dense reference can only agree with the
    golden if the golden reads windows alone.
    """

    def __init__(
        self,
        compress_ratio: int,
        is_generation: bool,
        seq_lens: list[int],
        past_lens: list[int],
        *,
        topk_width: int = 3,
        seed: int = 0,
    ) -> None:
        generator = torch.Generator().manual_seed(seed)
        self.compress_ratio = compress_ratio
        self.is_generation = is_generation
        self.seq_lens = seq_lens
        self.past_lens = past_lens
        self.totals = [past + new for past, new in zip(past_lens, seq_lens, strict=True)]
        self.swa = torch.zeros(NUM_POOL_PAGES, SWA_TOKENS_PER_PAGE, HEAD_DIM, device=DEVICE)
        self.compressed = torch.zeros(
            NUM_POOL_PAGES, COMPRESSED_TOKENS_PER_PAGE, HEAD_DIM, device=DEVICE
        )

        free_swa_pages = torch.randperm(NUM_POOL_PAGES, generator=generator).tolist()
        free_compressed_pages = torch.randperm(NUM_POOL_PAGES, generator=generator).tolist()
        swa_tables, compressed_tables = [], []
        self.histories, self.compressed_rows = [], []
        for past, total in zip(past_lens, self.totals, strict=True):
            # Pages for the rows any new token may read: its window and the new rows.
            window_start = max(0, past - WINDOW + 1)
            num_blocks = -(-total // SWA_TOKENS_PER_PAGE)
            table = _shuffled_table(
                range(window_start // SWA_TOKENS_PER_PAGE, num_blocks), num_blocks, free_swa_pages
            )
            history = torch.randn(past, HEAD_DIM, generator=generator).to(DEVICE)
            _write_rows(self.swa, table, window_start, history[window_start:past])
            swa_tables.append(table)
            self.histories.append(history)

            num_entries = total // compress_ratio if compress_ratio > 1 else 0
            num_blocks = -(-num_entries // COMPRESSED_TOKENS_PER_PAGE)
            table = _shuffled_table(range(num_blocks), max(1, num_blocks), free_compressed_pages)
            entries = torch.randn(num_entries, HEAD_DIM, generator=generator).to(DEVICE)
            _write_rows(self.compressed, table, 0, entries)
            compressed_tables.append(table)
            self.compressed_rows.append(entries)

        width = max(t.numel() for t in swa_tables)
        self.swa_tables = torch.stack(
            [torch.nn.functional.pad(t, (0, width - t.numel()), value=-1) for t in swa_tables]
        )
        width = max(t.numel() for t in compressed_tables)
        self.compressed_tables = torch.stack(
            [
                torch.nn.functional.pad(t, (0, width - t.numel()), value=-1)
                for t in compressed_tables
            ]
        )

        num_tokens = sum(seq_lens)
        self.positions = [
            past + i for past, new in zip(past_lens, seq_lens, strict=True) for i in range(new)
        ]
        self.q = torch.randn(num_tokens, NUM_HEADS * HEAD_DIM, generator=generator).to(DEVICE)
        self.latent = torch.randn(num_tokens, HEAD_DIM, generator=generator).to(DEVICE)
        self.topk_width = topk_width
        self.topk = None
        if compress_ratio == 4:
            # Valid-first rows of distinct visible entries, as the indexer emits.
            self.topk = torch.full((num_tokens, topk_width), -1, dtype=torch.int32)
            for row, position in enumerate(self.positions):
                visible = (position + 1) // compress_ratio
                count = min(visible, topk_width)
                picks = torch.randperm(visible, generator=generator)[:count]
                self.topk[row, :count] = picks.to(torch.int32)
            self.topk = self.topk.to(DEVICE)
        self.sink = torch.randn(NUM_HEADS, generator=generator).to(DEVICE)

    def zero_rope(self) -> None:
        """Zero every RoPE slice so rotation leaves all tensors unchanged."""
        self.q.view(-1, NUM_HEADS, HEAD_DIM)[..., KV_LORA_RANK:] = 0
        self.latent[:, KV_LORA_RANK:] = 0
        for rows in (*self.histories, *self.compressed_rows):
            rows[:, KV_LORA_RANK:] = 0
        self.swa[..., KV_LORA_RANK:] = 0
        self.compressed[..., KV_LORA_RANK:] = 0

    def q_pe(self) -> torch.Tensor:
        """Return a [T, H, R] copy of the query RoPE slices."""
        return self.q.view(-1, NUM_HEADS, HEAD_DIM)[..., KV_LORA_RANK:].clone()

    def metadata(self) -> SimpleNamespace:
        manager = SimpleNamespace(
            layer_offsets={0: 0},
            get_buffers=lambda layer_idx, attention_type: {
                DeepseekV4AttentionType.SWA: self.swa,
                DeepseekV4AttentionType.COMPRESS: self.compressed,
            }[attention_type],
        )
        num_contexts = 0 if self.is_generation else len(self.seq_lens)
        num_tokens = sum(self.seq_lens)
        return SimpleNamespace(
            kv_cache_manager=manager,
            window_size=WINDOW,
            multi_item_part_lens=None,
            num_contexts=num_contexts,
            num_seqs=len(self.seq_lens),
            num_ctx_tokens=0 if self.is_generation else num_tokens,
            num_tokens=num_tokens,
            seq_lens=torch.tensor(self.seq_lens, dtype=torch.int32),
            kv_lens_cuda=torch.tensor(self.totals, dtype=torch.int32, device=DEVICE),
            sliding_block_tables=self.swa_tables[None, None],
            compress_block_tables={self.compress_ratio: self.compressed_tables},
            max_compressed_indices={1: 0, 4: self.topk_width, 128: 8},
        )

    def forward_args(self, **overrides) -> AttentionForwardArgs:
        kwargs = dict(
            attention_input_type=(
                AttentionInputType.generation_only
                if self.is_generation
                else AttentionInputType.context_only
            ),
            latent_cache=self.latent,
            q_pe=self.q_pe(),
            attention_sinks=self.sink,
            sparse_backend_args=SparseBackendForwardArgs(topk_indices=self.topk),
        )
        kwargs.update(overrides)
        return AttentionForwardArgs(**kwargs)

    def dense_reference(self) -> torch.Tensor:
        """Masked dense attention over each request's full logical history.

        The attention sink is an extra logit column whose probability is
        dropped, and every key outside the token's window or selection is
        masked to -inf.
        """
        outputs = []
        token = 0
        for request, (past, new) in enumerate(zip(self.past_lens, self.seq_lens, strict=True)):
            begin = sum(self.seq_lens[:request])
            raw = torch.cat([self.histories[request], self.latent[begin : begin + new]])
            entries = self.compressed_rows[request]
            keys = torch.cat([raw, entries]).float()
            for position in range(past, past + new):
                window = torch.zeros(raw.shape[0], dtype=torch.bool, device=DEVICE)
                window[max(0, position - WINDOW + 1) : position + 1] = True
                chosen = torch.zeros(entries.shape[0], dtype=torch.bool, device=DEVICE)
                if self.compress_ratio == 4:
                    row = self.topk[token]
                    chosen[row[row >= 0].long()] = True
                elif self.compress_ratio == 128:
                    chosen[: (position + 1) // 128] = True
                mask = torch.cat([window, chosen])
                q = self.q[token].view(NUM_HEADS, HEAD_DIM).float()
                logits = (q @ keys.T) / HEAD_DIM**0.5
                logits = logits.masked_fill(~mask, float("-inf"))
                logits = torch.cat([logits, self.sink.float().unsqueeze(1)], dim=1)
                probabilities = torch.softmax(logits, dim=1)[:, :-1]
                outputs.append((probabilities @ keys).reshape(-1))
                token += 1
        return torch.stack(outputs)


@pytest.mark.parametrize(
    ("compress_ratio", "is_generation", "seq_lens", "past_lens"),
    [
        pytest.param(1, False, [6, 5], [3, 0], id="r1-context-cached-prefix"),
        pytest.param(4, False, [6, 5], [3, 0], id="r4-context-cached-prefix"),
        pytest.param(4, True, [2, 1], [9, 13], id="r4-generation-multi-token"),
        pytest.param(128, True, [2, 1], [254, 300], id="r128-generation-entry-boundary"),
    ],
)
def test_vanilla_attention_matches_dense_masked_reference(
    compress_ratio, is_generation, seq_lens, past_lens
):
    # With every RoPE slice zero, rotation is the identity, so this test checks
    # the window, selection, sink and paging logic in isolation.
    scenario = _Scenario(compress_ratio, is_generation, seq_lens, past_lens)
    scenario.zero_rope()
    backend = _make_backend(compress_ratio)
    metadata = scenario.metadata()
    output = torch.empty_like(scenario.q)
    if is_generation:
        backend.mla_rope_generation(
            scenario.q,
            scenario.q_pe(),
            scenario.latent,
            metadata,
            None,
            None,
            None,
            None,
            None,
            None,
        )

    result = backend.forward(scenario.q, None, None, metadata, scenario.forward_args(output=output))

    assert result.data_ptr() == output.data_ptr()
    torch.testing.assert_close(result, scenario.dense_reference(), atol=1e-5, rtol=1e-5)
    begin = 0
    for request, (past, new) in enumerate(zip(past_lens, seq_lens, strict=True)):
        stored = _read_rows(scenario.swa, scenario.swa_tables[request], past, new)
        torch.testing.assert_close(stored, scenario.latent[begin : begin + new])
        begin += new


def test_vanilla_rope_and_swa_append_ownership():
    # Context: forward rotates the query in place and appends rotated rows.
    context = _Scenario(1, False, [3], [2], seed=1)
    backend = _make_backend(1)
    q_pe, latent = context.q_pe(), context.latent.clone()
    backend.forward(context.q, None, None, context.metadata(), context.forward_args(q_pe=q_pe))

    q_heads = context.q.view(-1, NUM_HEADS, HEAD_DIM)
    torch.testing.assert_close(
        q_heads[..., KV_LORA_RANK:].reshape(3, -1),
        _rotate(q_pe.reshape(3, -1), [2, 3, 4]),
    )
    torch.testing.assert_close(context.latent, latent)  # the caller's rows stay unrotated
    expected = latent.clone()
    expected[:, KV_LORA_RANK:] = _rotate(latent[:, KV_LORA_RANK:], [2, 3, 4])
    torch.testing.assert_close(_read_rows(context.swa, context.swa_tables[0], 2, 3), expected)

    # Generation: mla_rope_generation owns RoPE and the append; forward does neither.
    generation = _Scenario(1, True, [1], [5], seed=2)
    metadata = generation.metadata()
    q_pe, latent = generation.q_pe(), generation.latent.clone()
    backend.mla_rope_generation(
        generation.q, q_pe, generation.latent, metadata, None, None, None, None, None, None
    )
    torch.testing.assert_close(
        generation.q.view(1, NUM_HEADS, HEAD_DIM)[..., KV_LORA_RANK:].reshape(1, -1),
        _rotate(q_pe.reshape(1, -1), [5]),
    )
    torch.testing.assert_close(generation.latent, latent)
    expected = latent.clone()
    expected[:, KV_LORA_RANK:] = _rotate(latent[:, KV_LORA_RANK:], [5])
    torch.testing.assert_close(_read_rows(generation.swa, generation.swa_tables[0], 5, 1), expected)

    q_before, swa_before = generation.q.clone(), generation.swa.clone()
    backend.forward(generation.q, None, None, metadata, generation.forward_args(q_pe=q_pe))
    torch.testing.assert_close(generation.q, q_before)
    torch.testing.assert_close(generation.swa, swa_before)


def _future_entry(scenario: _Scenario) -> None:
    scenario.topk[0, 0] = (scenario.positions[0] + 1) // 4


def _entry_past_readable_columns(scenario: _Scenario) -> None:
    # Position 3 sees one entry, so the kernel reads only column 0.
    scenario.topk[0] = torch.tensor([-1, 0, -1], dtype=torch.int32)


def _hole_in_window(scenario: _Scenario) -> None:
    scenario.swa_tables[0, 0] = -1


@pytest.mark.parametrize(
    ("corrupt", "overrides", "message"),
    [
        pytest.param(_future_entry, {}, "future compressed entry", id="future-entry"),
        pytest.param(
            _entry_past_readable_columns, {}, "first min", id="entry-past-readable-columns"
        ),
        pytest.param(_hole_in_window, {}, "invalid page", id="hole-in-window"),
        pytest.param(
            None, {"skip_mla_rope_generation": True}, "skip_mla_rope_generation", id="skip-rope"
        ),
    ],
)
def test_vanilla_attention_validates_before_writing(corrupt, overrides, message):
    scenario = _Scenario(4, False, [6], [3], seed=3)
    if corrupt is not None:
        corrupt(scenario)
    backend = _make_backend(4)
    q_before, swa_before = scenario.q.clone(), scenario.swa.clone()

    with pytest.raises(ValueError, match=message):
        backend.forward(
            scenario.q, None, None, scenario.metadata(), scenario.forward_args(**overrides)
        )
    torch.testing.assert_close(scenario.q, q_before)
    torch.testing.assert_close(scenario.swa, swa_before)
