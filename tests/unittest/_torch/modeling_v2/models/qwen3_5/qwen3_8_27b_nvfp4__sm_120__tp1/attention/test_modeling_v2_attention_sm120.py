# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the qwen3.8-27b sm_120 tp1 attention op (`ops/attention/attention_sm120`).

The target's 16 attention layers run the op over the KV cache manager's paged fp8 pool.
The cases below drive it through the catalog attention test's paged harness
(`test_modeling_v2_thop_attention._PagedAttnEnv`): a single-layer fp8 pool with explicit
page tables, history written by earlier calls of the op itself, and the catalog entry's
`reference` over the K/V the harness mirrored. `_AsTargetOp` stands in for the catalog
entry's call (it maps the entry's keyword arguments onto the op), so each case is checked
as a catalog cell is:

* against `thop_attention.reference` with e4m3 inputs: `_compare`, or past
  `FP8_ROW_MAX_MODEL_KEYS` keys `_compare_fp8_long_history`;
* every sequence's cache pages hold its K/V, and no unwritten page was touched.

The cases are the ones the target runs: decode with the one-wave multi-block cap,
decode batches of 8 with and without a shared prefix (the shared-prefix pass), fresh and
chunked prefill contexts of 1 to 3 sequences (the last q-tile KV split), a mixed step,
and the final layer's skipped leading q-tiles.
"""

import zlib

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12,
    reason="the target's attention kernels are built for SM120",
)

from _torch.modeling_v2.attention import test_modeling_v2_thop_attention as harness  # noqa: E402
from _torch.modeling_v2.attention.test_modeling_v2_thop_attention import (  # noqa: E402
    MASK_CAUSAL,
    ULP,
    _compare,
    _op_reference,
    _PagedAttnEnv,
    assert_within_ulp,
)

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import Cell  # noqa: E402
from tensorrt_llm._torch._experimental.modeling_v2.catalog.attention.thop_attention import (  # noqa: E402,E501
    QM_FP8_KV_CACHE,
    STANDARD,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.qwen3_5.qwen3_8_27b_nvfp4__sm_120__tp1.ops.attention.attention_sm120 import (  # noqa: E402,E501
    AttentionSm120Buffers,
    attention_sm120,
)

#: The target's decode shared-prefix threshold (modeling._XQA_SHARED_PREFIX_MIN_BATCH).
SHARED_PREFIX_MIN_BATCH = 8
#: The context FMHA's q-tile height.
FMHA_Q_TILE = 64

#: The longest history `_compare` holds an e4m3 call to; see `_compare_fp8_long_history`.
FP8_ROW_MAX_MODEL_KEYS = 256


def _compare_fp8_long_history(out: torch.Tensor, ref: torch.Tensor) -> None:
    """An e4m3 call attending over more than `FP8_ROW_MAX_MODEL_KEYS` keys.

    The kernels round the softmax probabilities to e4m3 against each KV tile's running
    max; `reference(e4m3_probabilities=True)` rounds against the row's max, which
    reproduces that within one tile and drifts as tiles accumulate. Measured on sm_120
    over the cases below (histories of 300 to 2047 keys, decode and paged-context
    prefill): up to 16.8 ulp element-wise and 10.4 ulp relative RMS. 24 and 14 ulp keep
    that headroom; a misaddressed page or a wrong merge partition is off by the values
    themselves, hundreds of ulp.
    """
    assert_within_ulp(out, ref, element_ulp=24.0, rms_ulp=14.0, ulp=ULP[torch.bfloat16])


class _AsTargetOp(type(harness.op)):
    """The catalog attention entry with its call replaced by the target's op: the
    entry's `reference` still applies, so the harness checks the op as it checks the
    entry."""

    def __init__(self, skip_leading_q_tiles: bool = False) -> None:
        super().__init__()
        self.skip_leading_q_tiles = skip_leading_q_tiles
        self._buffers: dict = {}

    def _buffers_for(self, kw) -> AttentionSm120Buffers:
        key = (kw["max_num_requests"], kw["q"].device)
        if key not in self._buffers:
            self._buffers[key] = AttentionSm120Buffers(
                max_tokens=8192, max_seqs=kw["max_num_requests"], device=kw["q"].device
            )
        return self._buffers[key]

    def is_valid(self, **kw) -> None:
        assert kw["is_fused_qkv"] and kw["update_kv_cache"] and kw["k"] is None
        assert kw["num_heads"] == 24 and kw["num_kv_heads"] == 4 and kw["head_size"] == 256
        assert kw["use_paged_context_fmha"], "the target runs the paged-context path"

    def __call__(self, **kw):
        attention_sm120(
            kw["q"],
            kw["output"],
            num_contexts=kw["num_contexts"],
            num_ctx_tokens=kw["num_ctx_tokens"],
            sequence_length=kw["sequence_length"],
            host_past_key_value_lengths=kw["host_past_key_value_lengths"],
            context_lengths=kw["context_lengths"],
            host_context_lengths=kw["host_context_lengths"],
            kv_cache_block_offsets=kw["kv_cache_block_offsets"],
            host_kv_cache_pool_pointers=kw["host_kv_cache_pool_pointers"],
            host_kv_cache_pool_mapping=kw["host_kv_cache_pool_mapping"],
            layer_idx=kw["local_layer_idx"],
            tokens_per_block=kw["tokens_per_block"],
            max_seq_len=kw["max_seq_len"],
            kv_scale_orig_quant=kw["kv_scale_orig_quant"],
            kv_scale_quant_orig=kw["kv_scale_quant_orig"],
            buffers=self._buffers_for(kw),
            q_scaling=kw["q_scaling"],
            skip_leading_q_tiles=self.skip_leading_q_tiles and kw["num_contexts"] > 0,
            shared_prefix_min_batch=SHARED_PREFIX_MIN_BATCH,
        )


#: Qwen3.8-27B's full-attention call on sm_120: GQA 24/4/256 over an fp8-e4m3 pool at
#: the production kv scale 1.0, on the paged-context path. Pools sized for 2048-token
#: histories and contexts; `max_blocks_per_seq` is a test-side page-table width.
_QWEN35 = dict(
    input_type=STANDARD,
    num_heads=24,
    num_kv_heads=4,
    head_size=256,
    quant_mode=QM_FP8_KV_CACHE,
    kv_scale=1.0,
    mask_type=1,
    tokens_per_block=32,
    num_blocks=256,
    max_num_requests=8,
    max_blocks_per_seq=96,
)

# `shared_prefix` tokens (whole pages) are one physical copy in every sequence's page
# table, as block reuse leaves them.
CELLS = (
    Cell(
        why="qwen3_5 decode batch 8, one sequence over 2047 keys: the one-wave multi-block "
        "cap (11 blocks per sequence by the history rule, capped at 188 // (8 * 4) = 5), "
        "two steps on one set of semaphores",
        spec=dict(
            _QWEN35,
            cached=(2047, 300, 31, 128, 512, 7, 200, 96),
            new=(1,) * 8,
            num_contexts=0,
            decode_steps=2,
        ),
    ),
    Cell(
        why="qwen3_5 decode, 4 sequences over 1500 to 31 keys: multi-block per sequence, "
        "two steps on one set of semaphores",
        spec=dict(
            _QWEN35,
            cached=(1500, 900, 300, 31),
            new=(1, 1, 1, 1),
            num_contexts=0,
            decode_steps=2,
        ),
    ),
    Cell(
        why="qwen3_5 decode batch 8 sharing a 512-token prefix: the shared-prefix pass "
        "over two 256-token tiles, then each sequence's own suffix",
        spec=dict(
            _QWEN35,
            shared_prefix=512,
            cached=(552, 612, 519, 576, 513, 712, 545, 640),
            new=(1,) * 8,
            num_contexts=0,
        ),
    ),
    Cell(
        why="qwen3_5 decode batch 8 with nothing shared: the plain decode after the "
        "shared-prefix scan",
        spec=dict(
            _QWEN35,
            cached=(64, 300, 31, 128, 512, 7, 200, 96),
            new=(1,) * 8,
            num_contexts=0,
        ),
    ),
    Cell(
        why="qwen3_5 prefill, one fresh 2048-row context: the target context FMHA",
        spec=dict(
            _QWEN35,
            cached=(0,),
            new=(2048,),
            num_contexts=1,
        ),
    ),
    Cell(
        why="qwen3_5 prefill, a 1024-row chunk over a 1024-token cached prefix",
        spec=dict(
            _QWEN35,
            cached=(1024,),
            new=(1024,),
            num_contexts=1,
        ),
    ),
    Cell(
        why="qwen3_5 prefill, three contexts of 1024, 1000 and 970 rows",
        spec=dict(
            _QWEN35,
            cached=(0, 0, 0),
            new=(1024, 1000, 970),
            num_contexts=3,
        ),
    ),
    Cell(
        why="qwen3_5 mixed step: a 700-row context ahead of two decode rows",
        spec=dict(
            _QWEN35,
            cached=(0, 300, 200),
            new=(700, 1, 1),
            num_contexts=1,
        ),
    ),
    Cell(
        why="qwen3_5 last layer, one 2048-row context: leading q-tiles skipped",
        spec=dict(
            _QWEN35,
            cached=(0,),
            new=(2048,),
            num_contexts=1,
            skip_leading_q_tiles=True,
        ),
    ),
    Cell(
        why="qwen3_5 last layer, three contexts ending in one q-tile: leading q-tiles skipped",
        spec=dict(
            _QWEN35,
            cached=(0, 0, 0),
            new=(1024, 1000, 970),
            num_contexts=3,
            skip_leading_q_tiles=True,
        ),
    ),
    Cell(
        why="qwen3_5 last layer, a mixed step: contexts of 2048 and 2016 rows (the second "
        "over a cached 64) and a decode row, leading q-tiles skipped",
        spec=dict(
            _QWEN35,
            cached=(0, 64, 300),
            new=(2048, 2016, 1),
            num_contexts=2,
            skip_leading_q_tiles=True,
        ),
    ),
)


def _seed(cell) -> None:
    """A cell's own seed, stable across processes (str hash is not)."""
    torch.manual_seed(zlib.crc32(cell.why.encode()))


def _fill_history(env, rids, lens) -> None:
    """One uncertified context call appending `lens` tokens to each of `rids`."""
    env.call_op(
        env.random_qkv(sum(lens)),
        lens,
        len(rids),
        rids,
        MASK_CAUSAL,
        ctx_lens_override=lens,
        use_paged_context_fmha=True,
    )


def _share_prefix(env, rids, tokens) -> None:
    """Point every sequence after the first at the first one's leading `tokens` (whole
    pages) and mirror them into its history: one physical prefix in every page table,
    as KV-cache block reuse leaves it."""
    assert tokens % env.tokens_per_block == 0, "a shared prefix is whole pages"
    first = rids[0]
    k, v = torch.cat(env.k_history[first]), torch.cat(env.v_history[first])
    for rid in rids[1:]:
        env.pages[rid] = env.pages[first][: tokens // env.tokens_per_block]
        env.k_history[rid] = [k[:tokens]]
        env.v_history[rid] = [v[:tokens]]


@pytest.mark.parametrize("cell", CELLS, ids=[c.why[:60] for c in CELLS])
def test_attention_sm120(cell, monkeypatch) -> None:
    """A first, unchecked context call fills each sequence's `cached` tokens (with
    `shared_prefix`, the first sequence's history first, then its leading pages become
    every other sequence's prefix); the checked call then appends `new` rows per
    sequence (`decode_steps` times, on one set of the op's buffers and semaphores). With
    `skip_leading_q_tiles`, the context rows before the batch's last q-tile must stay
    unwritten (the output starts as a sentinel) and are not compared."""
    spec = cell.spec
    _seed(cell)
    skip = spec.get("skip_leading_q_tiles", False)
    monkeypatch.setattr(harness, "op", _AsTargetOp())
    env = _PagedAttnEnv(
        num_heads=spec["num_heads"],
        num_kv_heads=spec["num_kv_heads"],
        head_dim=spec["head_size"],
        tokens_per_block=spec["tokens_per_block"],
        num_blocks=spec["num_blocks"],
        max_batch=spec["max_num_requests"],
        max_blocks_per_seq=spec["max_blocks_per_seq"],
        pool_dtype=torch.float8_e4m3fn,
        quant_mode=spec["quant_mode"],
        kv_scaling_factor=spec["kv_scale"],
    )
    cached, new, nc = spec["cached"], spec["new"], spec["num_contexts"]
    assert all(n == 1 for n in new[nc:]), "one row per generation sequence"
    rids = list(range(len(new)))
    for rid in rids:
        env.add_request(rid, cached[rid] + (new[rid] if rid < nc else 0))
    shared = spec.get("shared_prefix", 0)
    if shared:
        assert all(c > shared for c in cached), "every sequence extends the prefix"
        _fill_history(env, rids[:1], [cached[0]])
        _share_prefix(env, rids, shared)
        _fill_history(env, rids[1:], [cached[rid] - shared for rid in rids[1:]])
    else:
        prior = [rid for rid in rids if cached[rid]]
        if prior:
            _fill_history(env, prior, [cached[rid] for rid in prior])
    monkeypatch.setattr(harness, "op", _AsTargetOp(skip_leading_q_tiles=skip))
    steps = spec.get("decode_steps", 1)
    assert steps == 1 or nc == 0, "repeated steps are decode steps"
    for step in range(steps):
        qkv = env.random_qkv(sum(new))
        sentinel = torch.full(
            (sum(new), spec["num_heads"] * spec["head_size"]), -7.0, dtype=qkv.dtype,
            device=qkv.device,
        )  # fmt: skip
        out = env.call_op(
            qkv,
            list(new),
            nc,
            rids,
            spec["mask_type"],
            ctx_lens_override=[new[i] if i < nc else env.prompt_lens[i] for i in rids],
            use_paged_context_fmha=True,
            op_overrides=dict(output=sentinel),
        )
        out = sentinel
        ref = _op_reference(env, qkv, list(new), rids, causal=True, e4m3=True)
        defined = torch.ones(out.shape[0], dtype=torch.bool, device=out.device)
        if skip:
            first_kept = FMHA_Q_TILE * ((max(new[:nc]) + FMHA_Q_TILE - 1) // FMHA_Q_TILE - 1)
            start = 0
            for n in new[:nc]:
                defined[start : start + min(first_kept, n)] = False
                start += n
            # The split ran: the rows it may skip were left as they were.
            assert (out[~defined] == -7.0).all(), "a skipped leading q-tile row was written"
        longest = max(c + n * (step + 1) for c, n in zip(cached, new))
        compare = _compare_fp8_long_history if longest > FP8_ROW_MAX_MODEL_KEYS else _compare
        compare(out[defined], ref[defined])
        for rid in rids:
            env.check_cache(rid)
        env.check_unwritten_pool_zero()
