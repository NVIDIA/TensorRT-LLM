# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the compressor_prefill_reduction catalog entry.

The entry owns what is certified: `CELLS` is the list, `reference` and `compare` are the gate, and
`test_certified_cells` drives every cell through them, each call of each cell gated against
`op.reference` taken from the pools as they were before the call. The cells are the DeepSeek-V4.1
ratio-2 compressor of the KV-source layers 2/8/14 -- compress_ratio 2, head_dim 512, fp32 kv_score
rows ``[kv || score]``, fp32 paged state, zero APE (V4.1 has none), fp32 or bf16 output -- over
state page sizes 128 and 256 (the two values DeepseekV4CacheManager accepts for tokens_per_block).
A nonzero APE is covered once so where it enters is measured, not read.

This file keeps the checks a cell cannot carry: aliased pages behind the window (only the pending
token is read back), the wrong-variant controls, the silent-misuse guards, the undersized
max_outputs hazard, and the engine-size coverage (every row count on the engine's pool, the
target's call on the real managers, offset hazards, the multi-request envelope, every dumped
prefill step, pool-size invariance, zero-row requests).

Those engine-size checks, and the controls, gate on ``_reference_pool`` / ``_env_ref``: the
checkpoint's own formula (its inference/model.py, Compressor.forward): per channel, the sum over
each consecutive token pair of ``kv * softmax_over_the_pair(score)``, built in fp32 over the WHOLE
sequence at once, independent of the pool layout -- which is what lets it run on the engine's
pools (up to 65 GB) that `op.reference`, which returns whole expected pools, cannot copy. The op
instead runs chunk by chunk over a shared paged pool, so a chunk boundary at an odd position has
to carry the pending token through the pool.

Pool sizing: the semantics tests use POOL_PAGES, plainly larger than any batch
here needs; the engine's own pools, the multi-request envelope and pool-size
invariance up to the largest pool the gate settings allow are the sections at the end.
Physical pages are a random permutation of the pool,
distinct between the kv and the score tables, and every unused table column
points at a trap page, so slot addressing, a swapped table and an out-of-range
write are all visible.
"""

import math
import zlib
from typing import Dict, List, Sequence, Tuple

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.deepseek_v41.deepseek_v4_1_flash_nvfp4__sm_103__unified.ops.attention.compressor_prefill_reduction import (  # noqa: E501
    compressor_prefill_reduction as op,
)

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401 — path above; fixture by name

assert torch.cuda.is_available(), "compressor_prefill_reduction requires a CUDA device"

DEV = torch.device("cuda")
HEAD_DIM = 512
RATIO = 2
POOL_PAGES = 1024
TRAP_PAGE = 0
KV_SENTINEL = 7777.0
SCORE_SENTINEL = -7777.0

# Chunk plans: sequence -> new-token count per call. Every call is a mixed
# batch whose members sit at different start-position parities, including
# 1-token chunks that only park a pending token (zero outputs for that member).
CHUNK_PLANS: Dict[str, List[List[int]]] = {
    "mixed": [
        [1, 2, 97, 1, 200],
        [33, 31, 1, 1, 64],
        [128, 129, 256, 257, 257],
        [1, 1, 1, 1, 1],
        [4095, 1, 1, 4094, 1],
    ],
}


def _gen(seed: int) -> torch.Generator:
    g = torch.Generator(device=DEV)
    g.manual_seed(seed)
    return g


def _make_sequence(length: int, gen: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
    """fp32 kv and score rows. Scores span near-uniform to near-one-hot pairs;
    channel 0 is an exact tie in every pair and channel 1 a gap of 240, where
    exp underflows to exactly 0 in fp32."""
    kv = torch.randn(length, HEAD_DIM, generator=gen, device=DEV) * 2.0
    score = torch.randn(length, HEAD_DIM, generator=gen, device=DEV) * 4.0
    score[:, 0] = 1.5
    parity = torch.arange(length, device=DEV) % 2
    score[:, 1] = torch.where(parity == 0, 120.0, -120.0)
    return kv, score


def _reference_pool(kv: torch.Tensor, score: torch.Tensor, ape: torch.Tensor) -> torch.Tensor:
    """model.py's ratio-2 pooling over a whole sequence, fp32: [L, D] -> [L // 2, D]."""
    n = kv.shape[0] // RATIO
    pos = torch.arange(kv.shape[0], device=DEV)
    s = score + ape[pos % RATIO]
    kv_g = kv[: n * RATIO].unflatten(0, (n, RATIO))
    s_g = s[: n * RATIO].unflatten(0, (n, RATIO))
    return (kv_g * s_g.softmax(dim=1)).sum(dim=1)


class _StatePool:
    """Paged fp32 compressor-state pools (kv and score) plus per-sequence block tables."""

    def __init__(self, lengths: Sequence[int], page_size: int, seed: int):
        self.page_size = page_size
        self.kv = torch.full((POOL_PAGES, page_size, HEAD_DIM), KV_SENTINEL, device=DEV)
        self.score = torch.full((POOL_PAGES, page_size, HEAD_DIM), SCORE_SENTINEL, device=DEV)
        n_logical = max(math.ceil(max(lengths) / page_size), 1)
        need = len(lengths) * n_logical
        assert need < POOL_PAGES, "raise POOL_PAGES"
        cpu = torch.Generator().manual_seed(seed)
        self.bt_kv = self._table(len(lengths), n_logical, cpu)
        self.bt_score = self._table(len(lengths), n_logical, cpu)

    @staticmethod
    def _table(batch: int, n_logical: int, cpu: torch.Generator) -> torch.Tensor:
        pages = torch.randperm(POOL_PAGES - 1, generator=cpu)[: batch * n_logical] + 1
        table = torch.full((batch, n_logical + 1), TRAP_PAGE, dtype=torch.int32)
        table[:, :n_logical] = pages.view(batch, n_logical).to(torch.int32)
        return table.to(DEV)

    def slot(self, table: torch.Tensor, b: int, pos: int) -> Tuple[int, int]:
        return int(table[b, pos // self.page_size]), pos % self.page_size


def _run_call(
    pool: _StatePool,
    kvs: Sequence[torch.Tensor],
    scores: Sequence[torch.Tensor],
    starts: Sequence[int],
    news: Sequence[int],
    ape: torch.Tensor,
    out_dtype: torch.dtype,
    certify: bool = False,
) -> List[torch.Tensor]:
    """One op call over a batch of context chunks; returns each sequence's output rows.

    ``certify`` gates the call on the entry's own ``reference`` (taken from the pools before the
    call) and ``compare``, and checks the op returns the three tensors it wrote."""
    rows = [
        torch.cat([kv[s : s + n], sc[s : s + n]], dim=1)
        for kv, sc, s, n in zip(kvs, scores, starts, news)
    ]
    kv_score = torch.cat(rows, dim=0).contiguous()
    n_out = [(s + n) // RATIO - s // RATIO for s, n in zip(starts, news)]
    cu_seq = [0]
    cu_out = [0]
    for n, o in zip(news, n_out):
        cu_seq.append(cu_seq[-1] + n)
        cu_out.append(cu_out[-1] + o)
    total = cu_out[-1]
    # One guard row past the last output: an out-of-range write lands there.
    buf = torch.full((total + 1, HEAD_DIM), float("nan"), device=DEV, dtype=out_dtype)
    output = buf[:total]

    def i32(values):
        return torch.tensor(values, device=DEV, dtype=torch.int32)

    args = (
        kv_score,
        ape,
        pool.kv,
        pool.score,
        pool.bt_kv,
        pool.bt_score,
        output,
        i32([s + n for s, n in zip(starts, news)]),
        i32(list(starts)),
        i32(cu_seq),
        i32(cu_out),
        len(news),
        pool.page_size,
        HEAD_DIM,
        RATIO,
        max(n_out),
    )
    expected = op.reference(*args) if certify else None
    result = op(*args)
    torch.cuda.synchronize()
    if certify:
        assert result[0] is output and result[1] is pool.kv and result[2] is pool.score, (
            "the op writes in place and returns (output, paged_kv, paged_score)"
        )
        op.compare(result, expected)
    assert torch.isnan(buf[total]).all(), "op wrote past the last output row"
    return [output[cu_out[b] : cu_out[b + 1]] for b in range(len(news))]


def _run_plan(
    plan: List[List[int]],
    page_size: int,
    ape: torch.Tensor,
    out_dtype: torch.dtype,
    seed: int,
    certify: bool = False,
):
    gen = _gen(seed)
    lengths = [sum(p) for p in plan]
    seqs = [_make_sequence(L, gen) for L in lengths]
    kvs = [s[0] for s in seqs]
    scores = [s[1] for s in seqs]
    pool = _StatePool(lengths, page_size, seed)
    outs: List[List[torch.Tensor]] = [[] for _ in plan]
    starts = [0] * len(plan)
    for step in range(len(plan[0])):
        news = [p[step] for p in plan]
        got = _run_call(pool, kvs, scores, starts, news, ape, out_dtype, certify=certify)
        for b, g in enumerate(got):
            outs[b].append(g)
        starts = [s + n for s, n in zip(starts, news)]
    return pool, kvs, scores, [torch.cat(o, dim=0) for o in outs]


def _check_state(pool: _StatePool, kvs, scores, ape: torch.Tensor) -> None:
    """Every position of every sequence holds its kv and score + ape bit for bit;
    every other slot of both pools (the trap page included) is untouched."""
    written_kv = torch.zeros(POOL_PAGES, pool.page_size, dtype=torch.bool, device=DEV)
    written_sc = torch.zeros_like(written_kv)
    for b, (kv, sc) in enumerate(zip(kvs, scores)):
        pos = torch.arange(kv.shape[0], device=DEV)
        page_kv = pool.bt_kv[b].long()[pos // pool.page_size]
        page_sc = pool.bt_score[b].long()[pos // pool.page_size]
        off = pos % pool.page_size
        assert torch.equal(pool.kv[page_kv, off], kv), f"seq {b}: paged kv state"
        assert torch.equal(pool.score[page_sc, off], sc + ape[pos % RATIO]), (
            f"seq {b}: paged score state"
        )
        written_kv[page_kv, off] = True
        written_sc[page_sc, off] = True
    assert (pool.kv[~written_kv] == KV_SENTINEL).all(), (
        "kv pool written outside the sequences' slots"
    )
    assert (pool.score[~written_sc] == SCORE_SENTINEL).all(), (
        "score pool written outside the sequences' slots"
    )


def _zero_ape() -> torch.Tensor:
    return torch.zeros(RATIO, HEAD_DIM, device=DEV)


# ---------------------------------------------------------------------------
# certified surface
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by this name
    """Every call of every cell gated on the entry's reference and compare: the chunk's rows, both
    pools bitwise, and the guard row past the last output. A chunk after the first reads the token
    the op itself parked. At the end every position of every sequence holds its kv and score + ape
    and no other slot of either pool (the trap page included) changed."""
    certifying(op)
    spec = cell.spec
    seed = zlib.crc32(cell.why.encode()) % 2**31
    if spec["ape"] == "zero":
        ape = _zero_ape()
    else:
        ape = torch.randn(RATIO, HEAD_DIM, generator=_gen(seed + 1), device=DEV) * 2.0
    plan = [list(chunks) for chunks in spec["plan"]]
    pool, kvs, scores, _ = _run_plan(
        plan, spec["page_size"], ape, spec["out_dtype"], seed=seed, certify=True
    )
    _check_state(pool, kvs, scores, ape)


def test_only_the_pending_token_is_read_back():
    """Full windows are pooled from the input rows, not re-read from the pool: with
    every logical page but the last one aliased to a single scratch page, the
    outputs are still exact, and a follow-up call still finds the pending token."""
    ape = _zero_ape()
    gen = _gen(59)
    length, page_size = 1001, 128
    kv, sc = _make_sequence(length + 3, gen)
    pool = _StatePool([length + 3], page_size, seed=59)
    last_logical = length // page_size
    scratch = int(pool.bt_kv[0, 0])
    pool.bt_kv[0, :last_logical] = scratch
    pool.bt_score[0, :last_logical] = int(pool.bt_score[0, 0])
    first = _run_call(pool, [kv], [sc], [0], [length], ape, torch.float32)[0]
    second = _run_call(pool, [kv], [sc], [length], [3], ape, torch.float32)[0]
    ref = _reference_pool(kv, sc, ape)
    torch.testing.assert_close(first, ref[: length // RATIO])
    torch.testing.assert_close(second, ref[length // RATIO :])


# ---------------------------------------------------------------------------
# discrimination: wrong variants must fail the same gate
# ---------------------------------------------------------------------------


def test_wrong_variants_fail_the_gate():
    ape = _zero_ape()
    plan = CHUNK_PLANS["mixed"]
    _, kvs, scores, outs = _run_plan(plan, 128, ape, torch.float32, seed=67)
    got = torch.cat(outs, dim=0)
    ref = torch.cat([_reference_pool(kv, sc, ape) for kv, sc in zip(kvs, scores)], dim=0)
    torch.testing.assert_close(got, ref)
    print(f"correct reference: max abs err {(got - ref).abs().max().item():.3e}")

    def pooled(fn):
        return torch.cat([fn(kv, sc) for kv, sc in zip(kvs, scores)], dim=0)

    def pairs(kv, sc):
        n = kv.shape[0] // RATIO
        return kv[: n * RATIO].unflatten(0, (n, RATIO)), sc[: n * RATIO].unflatten(0, (n, RATIO))

    def softmax_over_channels(kv, sc):
        k, s = pairs(kv, sc)
        return (k * s.softmax(dim=-1)).sum(dim=1)

    def bf16_operands(kv, sc):
        return _reference_pool(kv.bfloat16().float(), sc.bfloat16().float(), ape)

    def uniform_mean(kv, sc):
        return pairs(kv, sc)[0].mean(dim=1)

    def pending_token_dropped():
        # A kernel that does not read the pool pools a straddling pair from its
        # new token alone (a softmax over one element is 1).
        rows = []
        for kv, sc, chunks in zip(kvs, scores, plan):
            r = _reference_pool(kv, sc, ape).clone()
            start = 0
            for n in chunks[:-1]:
                start += n
                if start % RATIO == 1:
                    r[start // RATIO] = kv[start]
            rows.append(r)
        return torch.cat(rows, dim=0)

    variants = {
        "softmax_over_channels": pooled(softmax_over_channels),
        "bf16_operands": pooled(bf16_operands),
        "uniform_mean": pooled(uniform_mean),
        "pending_token_dropped": pending_token_dropped(),
    }
    for name, wrong in variants.items():
        diff = (got - wrong).abs().max().item()
        print(f"control {name}: max abs diff {diff:.3e}")
        with pytest.raises(AssertionError):
            torch.testing.assert_close(got, wrong)
    # The bf16 gate's bit-exact fraction must also see the bf16-operand control.
    _, kvs16, scores16, outs16 = _run_plan(plan, 128, ape, torch.bfloat16, seed=67)
    got16 = torch.cat(outs16, dim=0)
    wrong16 = variants["bf16_operands"].to(torch.bfloat16)
    frac = (got16 == wrong16).float().mean().item()
    print(f"control bf16_operands vs bf16 output: bit-exact fraction {frac:.6f}")
    assert frac < 0.999


# ---------------------------------------------------------------------------
# misuse: what the op accepts silently, and what the wrapper therefore rejects
# ---------------------------------------------------------------------------


def _raw_call(kv_score, output, length, page_size=128):
    """The bare op, one sequence at start_pos 0, fresh pools."""
    pages = math.ceil(length / page_size) + 1
    pkv = torch.zeros(pages, page_size, HEAD_DIM, device=DEV)
    psc = torch.zeros_like(pkv)
    bt = torch.arange(pages, device=DEV, dtype=torch.int32).view(1, pages)
    n_out = length // RATIO

    def i32(values):
        return torch.tensor(values, device=DEV, dtype=torch.int32)

    args = (
        kv_score,
        _zero_ape(),
        pkv,
        psc,
        bt,
        bt,
        output,
        i32([length]),
        i32([0]),
        i32([0, length]),
        i32([0, n_out]),
        1,
        page_size,
        HEAD_DIM,
        RATIO,
        n_out,
    )
    return args, pkv


def test_silent_misuse_is_rejected_by_the_wrapper():
    gen = _gen(71)
    length = 257
    kv, sc = _make_sequence(length, gen)
    kv_score = torch.cat([kv, sc], dim=1).contiguous()
    ref = _reference_pool(kv, sc, _zero_ape())
    n_out = length // RATIO

    wide_in = torch.zeros(length, 3 * HEAD_DIM, device=DEV)
    wide_in[:, : 2 * HEAD_DIM] = kv_score
    wide_out = torch.zeros(n_out, 2 * HEAD_DIM, device=DEV)
    cases = {
        "fp16_kv_score": (kv_score.half(), torch.zeros(n_out, HEAD_DIM, device=DEV)),
        "fp16_output": (kv_score, torch.zeros(n_out, HEAD_DIM, device=DEV, dtype=torch.float16)),
        "strided_kv_score": (wide_in[:, : 2 * HEAD_DIM], torch.zeros(n_out, HEAD_DIM, device=DEV)),
        "strided_output": (kv_score, wide_out[:, :HEAD_DIM]),
    }
    for name, (ks, out) in cases.items():
        args, _ = _raw_call(ks, out, length)
        op.raw_call(*args)  # outside validating: accepted, no error
        torch.cuda.synchronize()
        diff = (out.float() - ref).abs().max().item()
        print(f"raw op, {name}: returned normally, max abs diff vs reference {diff:.3e}")
        assert diff > 1.0, f"{name}: expected a silently wrong result from the bare op"
        args, _ = _raw_call(ks, torch.zeros_like(out) if out.is_contiguous() else out, length)
        match = "float32 or bfloat16" if name.startswith("fp16") else "must be contiguous"
        with pytest.raises(AssertionError, match=match):
            with validating(op):
                op(*args)


def test_undersized_max_outputs_silently_skips_rows_and_pending_state():
    """The grid is sized by max_outputs; a sequence with more windows loses the
    rows past it AND its trailing pending token (written by its last block)."""
    gen = _gen(83)
    length = 9
    kv, sc = _make_sequence(length, gen)
    kv_score = torch.cat([kv, sc], dim=1).contiguous()
    out = torch.full((length // RATIO, HEAD_DIM), float("nan"), device=DEV)
    args, pkv = _raw_call(kv_score, out, length)
    args = args[:-1] + (2,)
    op(*args)
    torch.cuda.synchronize()
    ref = _reference_pool(kv, sc, _zero_ape())
    torch.testing.assert_close(out[:2], ref[:2])
    assert torch.isnan(out[2:]).all(), "rows past max_outputs were written"
    assert torch.equal(pkv[0, 3], kv[3]), "state of a covered window"
    assert (pkv[0, 8] == 0).all(), "pending token state was written despite max_outputs"


# ---------------------------------------------------------------------------
# The engine's own pool, every row count, the target's call on the
# engine-size manager
# ---------------------------------------------------------------------------
#
# The engine is the DeepSeek-V4.1 core target's accuracy gate: tep4,
# max_seq_len 16384, max_batch_size 32, max_num_tokens 8192, tokens_per_block
# 128, block reuse on, a per-rank KV quota of 72.68959879875183 GiB (the gate's
# log line "KV cache manager v2 device quota set to ..."). The manager built
# below at that quota reproduces the pool shapes the gate's runtime-value dump
# names (asserted in _engine_manager), so its compressor-state pool -- one fp32
# tensor for the kv and score state of owners 2, 8 and 14 -- is the engine's.

_ENGINE_QUOTA_BYTES = int(round(72.68959879875183 * (1 << 30)))
_ENGINE_MAX_SEQ = 16384
# DSpark (max_draft_len 5, overlap on) grows the manager's max_seq_len by 5 + 4 + 5.
_ENGINE_MAX_SEQ_DSPARK = 16398
# Pool shapes of the gate's runtime-value dump (rank 0).
_ENGINE_SWA_PAGES = 137440
_ENGINE_COMPRESS_R2_PAGES = {453118, 453119, 453120}
_ENGINE_COMPRESS_R1_PAGES = 151040
_SENTINEL = 7777.0
_ENGINE_MANAGERS = {}

# The engine envelope (per rank): max_num_tokens up to 16384, max_batch_size up to 256
# (so up to 256 context requests in one call, and max_ctx_compressed_tokens up to 8192),
# max_seq_len L in {16384, 32768, 65536, 131072, 262144, 1048576}, DSpark on (the manager's
# length Lm = L + 14) and off. A context request's kv length reaches Lm (1,048,590 at L 1M with
# DSpark). The compressor-state tables are the manager's max_blocks_per_seq columns wide: 132 /
# 260 / 516 / 1028 / 2052 / 8196 for those L, with and without DSpark (measured on managers of
# the envelope). The largest shared compressor-state pool over the envelope at the 0.5 x
# total-memory quota is the PP2 stage-0 manager's (see _BMAX_PAGES), 247806 pages.
_ENVELOPE_TOKENS = 16384
_ENVELOPE_REQS = 256
_ENVELOPE_L = (16384, 32768, 65536, 131072, 262144, 1048576)
_ENVELOPE_WIDTHS = (132, 260, 516, 1028, 2052, 8196)
_ENVELOPE_WIDTH_OF_L = dict(zip(_ENVELOPE_L, _ENVELOPE_WIDTHS))
_ENVELOPE_LM_1M = 1_048_576 + 14  # the longest context kv length: L 1M with DSpark


def _envelope_lm(seq: int, dspark: bool) -> int:
    return seq + 14 if dspark else seq


def _v41_pretrained_config():
    from types import SimpleNamespace

    text = dict(
        model_type="deepseek_v41_text",
        num_hidden_layers=40,
        num_nextn_predict_layers=3,
        compress_ratios=[0, 0] + [2] * 18 + [1] * 20 + [0, 0, 0],
        kv_source_layer_ids=[2, 8, 14, 20],
        index_source_layer_ids=[2, 8, 14, 20, 24, 28, 32, 36],
        candidate_source_layer_id=20,
        engram_layer_ids=[1, 14],
        engram_max_ngram_size=4,
        sliding_window=128,
        head_dim=HEAD_DIM,
        index_head_dim=128,
        index_topk=512,
        kv_lora_rank=448,
        qk_rope_head_dim=64,
    )
    return SimpleNamespace(
        architectures=["DeepseekV41ForCausalLM"],
        model_type="deepseek_v41",
        text_config=SimpleNamespace(**text),
        **{k: v for k, v in text.items() if k != "model_type"},
    )


def _engine_manager(
    max_draft_len: int,
    max_batch_size: int = 32,
    quota: int = _ENGINE_QUOTA_BYTES,
    storage: str = "bf16",
    max_num_tokens: int = 8192,
    max_seq_len: int = None,
    pp_stage: int = None,
):
    """The engine-size V4.1 DeepseekV4CacheManager (one alive at a time; released at module end).

    ``max_batch_size`` 32 is the accuracy gate; 8 is the boot gate's batch size at the
    same quota, where the compressor-state pool is larger. ``quota`` defaults to the
    accuracy gate's logged per-rank quota; the envelope tests pass the largest quota the
    device can give at kv_cache_fraction 0.5. ``storage`` "native" builds it with
    TRTLLM_DSV41_KV_STORAGE=native (set for the construction only). ``max_seq_len`` is the
    manager's length (default 16384, or 16398 with DSpark). ``pp_stage`` 0 builds the stage-0
    manager of PP2 (``Mapping(world_size=4, rank=0, tp_size=2, pp_size=2)``; the pools do not
    depend on tp), built alone in this process, so the manager's
    cross-rank allreduce(MIN) of its token capacity runs over this rank only -- the stage-alone
    pools, an upper bound on what a PP2 engine's stage allocates. Host tier off
    (``host_cache_size=0``): it only adds a host copy for suspend/resume and does not
    change the device pools."""
    import os

    if max_seq_len is None:
        max_seq_len = _ENGINE_MAX_SEQ_DSPARK if max_draft_len else _ENGINE_MAX_SEQ
    key = (max_draft_len, max_batch_size, quota, storage, max_num_tokens, max_seq_len, pp_stage)
    if key in _ENGINE_MANAGERS:
        return _ENGINE_MANAGERS[key]
    _release_engine_managers()
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import DeepseekV4CacheManager
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
        DeepseekV4AttentionType as T,
    )
    from tensorrt_llm.bindings import DataType
    from tensorrt_llm.bindings.internal.batch_manager import CacheType
    from tensorrt_llm.llmapi.llm_args import (
        DeepSeekV4SparseAttentionConfig,
        DSparkDecodingConfig,
        KvCacheConfig,
    )
    from tensorrt_llm.mapping import Mapping

    env_before = os.environ.get("TRTLLM_DSV41_KV_STORAGE")
    os.environ["TRTLLM_DSV41_KV_STORAGE"] = storage
    try:
        mgr = DeepseekV4CacheManager(
            kv_cache_config=KvCacheConfig(
                enable_block_reuse=True,
                max_gpu_total_bytes=quota,
                host_cache_size=0,
                event_buffer_max_size=0,
            ),
            kv_cache_type=CacheType.SELFKONLY,
            num_layers=40,
            num_kv_heads=1,
            head_dim=HEAD_DIM,
            tokens_per_block=128,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)
            if pp_stage is None
            else Mapping(world_size=4, rank=2 * pp_stage, tp_size=2, pp_size=2, gpus_per_node=4),
            dtype=DataType.BF16,
            compressor_dtype=DataType.FLOAT,
            vocab_size=129280,
            max_num_tokens=max_num_tokens,
            sparse_attn_config=DeepSeekV4SparseAttentionConfig(
                index_head_dim=128, window_size=128, compress_ratios=[1] * 43
            ),
            pretrained_config=_v41_pretrained_config(),
            spec_config=DSparkDecodingConfig(max_draft_len=max_draft_len, block_size=max_draft_len)
            if max_draft_len
            else None,
            max_cuda_graph_batch_size=max_batch_size,
            disable_overlap_scheduler=False,
        )
    finally:
        if env_before is None:
            os.environ.pop("TRTLLM_DSV41_KV_STORAGE")
        else:
            os.environ["TRTLLM_DSV41_KV_STORAGE"] = env_before
    if pp_stage is not None:
        assert pp_stage == 0, "only PP2 stage 0 holds the compressor state (owners 2 / 8 / 14)"
        assert list(mgr.pp_layers) == list(range(20)), list(mgr.pp_layers)
    pool = mgr.get_buffers(2, T.COMPRESSOR_KV)
    shapes = {
        "swa": tuple(mgr.get_buffers(0, T.SWA).shape),
        "compress_r2": sorted({mgr.get_buffers(o, T.COMPRESS).shape[0] for o in (2, 8, 14)}),
        "compress_r1": tuple(mgr.get_buffers(20, T.COMPRESS).shape) if pp_stage is None else None,
        "compressor_state": tuple(pool.shape),
        "swa_dtype": str(mgr.get_buffers(0, T.SWA).dtype),
    }
    print(
        f"[engine manager max_draft_len={max_draft_len} max_batch_size={max_batch_size} quota={quota} "
        f"storage={storage} max_num_tokens={max_num_tokens} max_seq_len={max_seq_len} pp_stage={pp_stage} "
        f"max_blocks_per_seq={mgr.max_blocks_per_seq}] {shapes}"
    )
    assert (shapes["swa_dtype"] == "torch.uint8") == (storage == "native"), shapes
    seq = max_seq_len - (14 if max_draft_len else 0)
    if seq in _ENVELOPE_WIDTH_OF_L:
        assert mgr.max_blocks_per_seq == _ENVELOPE_WIDTH_OF_L[seq], mgr.max_blocks_per_seq
    if key == (0, 32, _ENGINE_QUOTA_BYTES, "bf16", 8192, _ENGINE_MAX_SEQ, None):
        assert shapes["swa"][0] == _ENGINE_SWA_PAGES, shapes
        assert set(shapes["compress_r2"]) == _ENGINE_COMPRESS_R2_PAGES, shapes
        assert shapes["compress_r1"][0] == _ENGINE_COMPRESS_R1_PAGES, shapes
    for owner in (2, 8, 14):
        for role in (T.COMPRESSOR_KV, T.COMPRESSOR_SCORE):
            b = mgr.get_buffers(owner, role)
            assert (
                b.data_ptr() == pool.data_ptr()
                and b.shape == pool.shape
                and b.dtype == torch.float32
            )
    assert pool.is_contiguous() and tuple(pool.shape[1:]) == (128, HEAD_DIM)
    _ENGINE_MANAGERS[key] = mgr
    return mgr


def _release_engine_managers():
    for mgr in _ENGINE_MANAGERS.values():
        mgr.shutdown()
    _ENGINE_MANAGERS.clear()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module", autouse=True)
def _engine_managers_released_at_module_end():
    yield
    _release_engine_managers()


def _pool_range(
    kv: torch.Tensor, score: torch.Tensor, w0: int, w1: int, lo: int = 0
) -> torch.Tensor:
    """model.py's ratio-2 pooling of windows [w0, w1) of per-position arrays that start at
    position ``lo``, fp32 (zero ape)."""
    p = torch.arange(w0 * RATIO, w1 * RATIO, device=DEV) - lo
    k = kv[p].unflatten(0, (w1 - w0, RATIO))
    s = score[p].unflatten(0, (w1 - w0, RATIO))
    return (k * s.softmax(dim=1)).sum(dim=1)


def _hazard_pages(num_pages: int) -> List[int]:
    """Pages on both sides of every 32-bit offset hazard of an fp32 [pages, 128, 512]
    pool (65536 elements, 256 KiB per page): byte offsets 2^31 (page 8192) and 2^32
    (16384), element offsets 2^31 (32768), 2^32 (65536) and 2^33 (131072); those the
    pool reaches."""
    return sorted(
        c
        for c in (8191, 8192, 16383, 16384, 32767, 32768, 65535, 65536, 131071, 131072)
        if c < num_pages
    )


def _one_sequence_tables(
    num_pages: int, ps: int, lo: int, hi: int, seed: int, pages=None, width=None
):
    """kv / score tables of one sequence mapping positions [lo, hi) to distinct pages
    taken (shuffled) from ``pages`` -- by default the pool's last pages -- and every
    other column to a trap page; ``width`` columns (default: one past the last page).
    Returns (bt_kv, bt_sc, trap, mapped page ids)."""
    logical = list(range(lo // ps, (hi - 1) // ps + 1))
    need = 2 * len(logical)
    if pages is None:
        pages = list(range(num_pages - need, num_pages))
    pages = list(pages)[:need]
    assert len(set(pages)) == need and max(pages) < num_pages
    trap = next(t for t in range(num_pages - 1, -1, -1) if t not in set(pages))
    cpu = torch.Generator().manual_seed(seed)
    far = [pages[i] for i in torch.randperm(need, generator=cpu).tolist()]
    width = logical[-1] + 2 if width is None else width
    assert width > logical[-1]
    bt_kv = torch.full((1, width), trap, dtype=torch.int32)
    bt_sc = torch.full_like(bt_kv, trap)
    for q in logical:
        bt_kv[0, q], bt_sc[0, q] = far.pop(), far.pop()
    mapped = torch.tensor(sorted(set(pages) | {trap}), device=DEV)
    return bt_kv.to(DEV), bt_sc.to(DEV), trap, mapped


# (start, table width) of the row sweep: from 0 and from an odd 129 on the gate engine's
# 132-column tables, and a chunk ending exactly at the envelope's longest kv length Lm =
# 1,048,590 (L 1,048,576 with DSpark), from an even and an odd start, on its 8196-column tables.
_ROW_SWEEP_N = _ENVELOPE_TOKENS
_ROW_SWEEP = [
    (0, 132),
    (129, 132),
    (_ENVELOPE_LM_1M - _ROW_SWEEP_N, 8196),
    (_ENVELOPE_LM_1M - _ROW_SWEEP_N - 1, 8196),
]


@pytest.mark.parametrize("start,width", _ROW_SWEEP, ids=[f"start{s}-w{w}" for s, w in _ROW_SWEEP])
def test_every_row_count_1_to_16384_on_the_engine_pool(start: int, width: int):
    """A context chunk of r new tokens sizes the launch grid (r // 2 windows,
    max_outputs = r // 2). On the engine's compressor-state pool, with the sequence's
    kv and score pages the pool's LAST pages (shuffled, distinct) and a trap page below
    them, every chunk length r = 1..16384 (max_num_tokens 16384) from ``start`` (0; 129,
    odd, so the first window pairs the parked token 128 with the chunk's first token; and
    1,032,206 / 1,032,205, whose 16384-token chunk ends at the envelope's longest kv length
    1,048,590 / 1,048,589 on 8196-column tables) returns exactly the first rows of the
    r = 16384 call bit for bit, writes nothing past them (NaN guard), and leaves the pool
    holding exactly positions [start, start + r) plus the parked token (the calls restore
    the same state, so they are idempotent). The 16384-token call matches the reference;
    the whole pool outside the named slots is unchanged."""
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
        DeepseekV4AttentionType as T,
    )

    pool_t = _engine_manager(0).get_buffers(2, T.COMPRESSOR_KV)
    num_pages, ps = pool_t.shape[0], pool_t.shape[1]
    N = _ROW_SWEEP_N
    lo = start - 1 if start % RATIO else start
    gen = _gen(9100 + start % 1000)
    # Per-position rows over [lo, start + N), indexed from lo.
    kv, sc = _make_sequence(start + N - lo, gen)
    # Far-end tables: the pool's last pages cover [lo, start + N) for kv and for score.
    bt_kv, bt_sc, trap, mapped = _one_sequence_tables(
        num_pages, ps, lo, start + N, seed=9100 + start % 1000, width=width
    )
    assert bt_kv.shape[1] == width
    assert int(mapped[0]) == trap
    region = pool_t[trap:]
    region.fill_(_SENTINEL)
    outside_sum = pool_t[:trap].view(torch.int32).sum(dtype=torch.int64)
    if start % RATIO:
        p = start - 1
        pool_t[int(bt_kv[0, p // ps]), p % ps] = kv[p - lo]
        pool_t[int(bt_sc[0, p // ps]), p % ps] = sc[p - lo]
    parked = 1 if start % RATIO else 0

    kv_score = torch.cat([kv[start - lo :], sc[start - lo :]], dim=1).contiguous()
    ape = _zero_ape()
    r_all = torch.arange(1, N + 1, device=DEV, dtype=torch.int32)
    kv_lens_all = (start + r_all).view(N, 1).contiguous()
    n_all = [(start + r) // RATIO - start // RATIO for r in range(1, N + 1)]
    cu_seq_all = torch.stack([torch.zeros_like(r_all), r_all], 1).contiguous()
    cu_out_all = torch.stack(
        [torch.zeros_like(r_all), torch.tensor(n_all, device=DEV, dtype=torch.int32)], 1
    )
    cu_out_all = cu_out_all.contiguous()
    start_pos = _i32_tensor([start])
    buf = torch.full((n_all[-1] + 1, HEAD_DIM), float("nan"), device=DEV)

    def call(r):
        n = n_all[r - 1]
        op(
            kv_score[:r],
            ape,
            pool_t,
            pool_t,
            bt_kv,
            bt_sc,
            buf[:n],
            kv_lens_all[r - 1],
            start_pos,
            cu_seq_all[r - 1],
            cu_out_all[r - 1],
            1,
            ps,
            HEAD_DIM,
            RATIO,
            n,
        )
        return n

    n = call(N)
    torch.cuda.synchronize()
    full = buf[:n].clone()
    torch.testing.assert_close(full, _pool_range(kv, sc, start // RATIO, (start + N) // RATIO, lo))
    buf.fill_(float("nan"))
    bad = torch.zeros((), dtype=torch.int64, device=DEV)
    for r in range(1, N + 1):
        n = call(r)
        bad += (buf[:n].view(torch.int32) != full[:n].view(torch.int32)).sum()
        bad += ((~torch.isnan(buf)).sum() - n * HEAD_DIM).abs()
        buf[:n] = float("nan")
        # State so far: kv + score slots of positions [start, start + N) (the first call) and the parked token.
        bad += ((region != _SENTINEL).any(-1).sum() - 2 * (N + parked)).abs()
    torch.cuda.synchronize()
    assert int(bad) == 0, f"{int(bad)} elements / slots differ from the {N}-token call"
    pos = torch.arange(lo, start + N, device=DEV)
    assert torch.equal(pool_t[bt_kv[0].long()[pos // ps], pos % ps], kv[pos - lo]), "kv state"
    assert torch.equal(pool_t[bt_sc[0].long()[pos // ps], pos % ps], sc[pos - lo]), "score state"
    assert (region[0] == _SENTINEL).all(), "the trap page was written"
    assert int(pool_t[:trap].view(torch.int32).sum(dtype=torch.int64)) == int(outside_sum), (
        "pool written below the region"
    )
    print(
        f"[row sweep start={start} width={width}] engine pool {tuple(pool_t.shape)}, pages {trap + 1}..{num_pages - 1}"
        f", positions up to {start + N - 1}: r 1..{N} bitwise equal to the r = {N} rows"
    )


def _i32_tensor(values) -> torch.Tensor:
    return torch.tensor(list(values), device=DEV, dtype=torch.int32)


# Mixed and context-only steps on the engine-size manager, handed to the op exactly
# as the target's _compress hands them. The dump's prefill calls (T, nc,
# max_ctx_compressed_tokens[2]) are (1018, 1, 508), (1935, 2, 960), (2048, 1, 1024),
# (48, 1, 24), (2, 1, 1), (2, 1, 0) and (1, 1, 0); the other steps add chunks with
# an odd cached count (the parked token read back from the manager's pool), a full
# 8192-token step of four contexts, and a DSpark (max_draft_len 5) step.
# scenario: ([(cached, new)] of the contexts, histories of the generation rows,
# max_draft_len, expected max_ctx_compressed_tokens[2]).
_ENGINE_PREFILL_STEPS = {
    "dump_T1018_nc1": ([(0, 1016)], [2047, 16000], 0, 508),
    "dump_T1935_nc2": ([(0, 1920), (0, 14)], [5003], 0, 960),
    "dump_T2048_nc1": ([(0, 2048)], [], 0, 1024),
    "dump_T48_nc1": ([(0, 48)], [], 0, 24),
    "dump_T2_nc1_max1": ([(0, 2)], [], 0, 1),
    "dump_T2_nc1_max0": ([(0, 1)], [7], 0, 0),
    "dump_T1_nc1": ([(0, 1)], [], 0, 0),
    "chunks_odd_cached": ([(1023, 1017), (4095, 2001), (1, 1)], [128, 129], 0, 1001),
    "T8192_nc4": ([(0, 3000), (511, 2001), (1, 1), (4096, 3190)], [], 0, 1595),
    "dspark_T1028_nc1": ([(0, 1016)], [2047, 16000], 5, 508),
}


@pytest.mark.parametrize("scenario", list(_ENGINE_PREFILL_STEPS))
def test_engine_size_manager_target_call(scenario):
    """The target's exact call on a step prepared by the engine-size manager: the step's
    whole kv_score [T, 1024] (its generation rows NaN: they belong to the decode op),
    ``sliding_block_tables[layer_offsets[owner], COMPRESSOR_KV / SCORE][:nc]``,
    ``kv_lens_cuda_runtime[:nc]``, ``cached_token_lens_cuda[:nc]`` as start_pos,
    ``cu_seq_lens_cuda[:nc + 1]``, ``cu_new_comp_kv_cuda[2][:nc + 1]``,
    ``max_ctx_compressed_tokens[2]`` as max_outputs, and the zero-filled output of
    ``num_total_compressed_tokens[2]`` rows, bf16 (the target's) and fp32, owners 2/8/14.
    The whole pool is NaN except each odd-cached context's parked token, which a
    previous chunk leaves. Checks: the context rows match the reference (fp32 default
    tolerances; bf16 >= 0.999 bit-exact); the generation rows' reserved output rows stay
    zero; the manager gives every new position a slot of its own (asserted: no shared
    scratch page in these steps) and each holds its kv / score bit for bit; and no slot
    outside the tables' pages for these positions changed."""
    ctx, gen_cached, max_draft_len, expect_max = _ENGINE_PREFILL_STEPS[scenario]
    _target_call_on_manager(
        _engine_manager(max_draft_len), scenario, ctx, gen_cached, max_draft_len, expect_max
    )


def _target_call_on_manager(
    manager,
    scenario,
    ctx,
    gen_cached,
    max_draft_len,
    expect_max,
    max_num_tokens=8192,
    allow_alias=False,
):
    """One step on ``manager`` handed to the op exactly as the target hands it, with the
    checks of test_engine_size_manager_target_call. ``allow_alias``: the manager may map
    positions of a chunk behind its last two to a shared scratch slot; those slots are then
    only counted as written (their content is whichever aliased position the op stored last),
    while every slot of the last two positions of each chunk must be the position's own."""
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4 import (
        DeepseekV4TrtllmAttentionMetadata,
    )
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
        DeepseekV4AttentionType as T,
    )
    from tensorrt_llm._torch.metadata import KVCacheParams
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest
    from tensorrt_llm.bindings import SamplingConfig
    from tensorrt_llm.llmapi.llm_args import DeepSeekV4SparseAttentionConfig
    from tensorrt_llm.mapping import Mapping

    next_n = max_draft_len + 1
    sparse = DeepSeekV4SparseAttentionConfig(
        index_head_dim=128, window_size=128, compress_ratios=[1] * 43
    )
    cached = [c for c, _ in ctx] + list(gen_cached)
    new = [n for _, n in ctx] + [next_n] * len(gen_cached)
    nc, B = len(ctx), len(cached)
    requests = []
    try:
        for i, (c, n) in enumerate(zip(cached, new)):
            req = LlmRequest(
                request_id=i,
                max_new_tokens=64,
                input_tokens=list(range(c + n)),
                sampling_config=SamplingConfig(),
                is_streaming=False,
            )
            assert manager.prepare_context(req), f"manager could not hold request {i}"
            assert manager.resize_context(req, req.context_chunk_size)
            requests.append(req)
        md = DeepseekV4TrtllmAttentionMetadata(
            seq_lens=torch.tensor(new, dtype=torch.int),
            request_ids=list(range(B)),
            max_num_requests=manager.max_batch_size,
            num_contexts=nc,
            prompt_lens=[c + n for c, n in zip(cached, new)],
            max_num_tokens=max_num_tokens,
            kv_cache_manager=manager,
            kv_cache_params=KVCacheParams(use_cache=True, num_cached_tokens_per_seq=list(cached)),
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            sparse_attention_config=sparse,
        )
        md.prepare()
        torch.cuda.synchronize()
        ctx_tokens = sum(n for _, n in ctx)
        num_tokens = sum(new)
        ctx_rows = sum((c + n) // RATIO - c // RATIO for c, n in ctx)
        total = md.num_total_compressed_tokens[RATIO]
        max_out = md.max_ctx_compressed_tokens[RATIO]
        assert (md.num_contexts, md.num_seqs, md.num_ctx_tokens) == (nc, B, ctx_tokens)
        assert max_out == expect_max, (scenario, max_out)
        assert max_out == max((c + n) // RATIO - c // RATIO for c, n in ctx)
        assert total == ctx_rows + (B - nc) * math.ceil(next_n / RATIO), (total, ctx_rows)
        kv_lens = md.kv_lens_cuda_runtime[:nc]
        start_pos = md.cached_token_lens_cuda[:nc]
        cu_seq = md.cu_seq_lens_cuda[: nc + 1]
        cu_out = md.cu_new_comp_kv_cuda[RATIO][: nc + 1]
        assert kv_lens.tolist() == [c + n for c, n in ctx] and start_pos.tolist() == [
            c for c, _ in ctx
        ]
        assert cu_seq.tolist() == [0] + torch.tensor([n for _, n in ctx]).cumsum(0).tolist()
        cu = cu_out.tolist()
        assert cu[nc] == ctx_rows

        gen = _gen(9300 + len(scenario))
        truth = []
        kv_score = torch.full((num_tokens, 2 * HEAD_DIM), float("nan"), device=DEV)
        row = 0
        for c, n in ctx:
            lo = c - 1 if c % RATIO else c
            kv, sc = _make_sequence(c + n - lo, gen)  # positions [lo, c + n), indexed from lo
            kv_score[row : row + n] = torch.cat([kv[c - lo :], sc[c - lo :]], 1)
            truth.append((kv, sc, lo))
            row += n
        ape = _zero_ape()
        page = manager.tokens_per_block
        pool = manager.get_buffers(2, T.COMPRESSOR_KV)
        pages_seen = set()
        aliased = 0
        for owner in (2, 8, 14):
            p_kv = manager.get_buffers(owner, T.COMPRESSOR_KV)
            p_sc = manager.get_buffers(owner, T.COMPRESSOR_SCORE)
            lo_ = manager.layer_offsets[owner]
            bt_kv = md.sliding_block_tables[lo_, T.COMPRESSOR_KV.value]
            bt_sc = md.sliding_block_tables[lo_, T.COMPRESSOR_SCORE.value]
            # Flat slot ids (page * 128 + offset) every new position of every context writes.
            writes = []
            for b, (c, n) in enumerate(ctx):
                pos = torch.arange(c, c + n, device=DEV)
                for tbl in (bt_kv, bt_sc):
                    pg = tbl[b].long()[pos // page]
                    assert (pg >= 0).all() and (pg < pool.shape[0]).all(), (
                        f"{scenario} owner {owner}: table entry"
                    )
                    writes.append((b, tbl is bt_sc, pos, pg * page + pos % page))
            all_ids = torch.cat([w[3] for w in writes])
            uniq, counts = torch.unique(all_ids, return_counts=True)
            shared = int((counts > 1).sum())
            aliased = max(aliased, shared)
            # SWA scratch reuse could alias pages behind a window; the gate-size steps get none.
            assert allow_alias or shared == 0, (
                f"{scenario} owner {owner}: slots shared between positions"
            )
            for b, (c, n) in enumerate(ctx):
                for last in range(max(c, c + n - 2), c + n):
                    for tbl in (bt_kv, bt_sc):
                        sid = int(tbl[b, last // page]) * page + last % page
                        assert int(counts[uniq == sid].sum()) == 1, (
                            f"{scenario} owner {owner}: last position shared"
                        )
            pages_seen |= set((uniq // page).tolist())
            for out_dtype in (torch.bfloat16, torch.float32):
                pool.fill_(float("nan"))
                parked = []
                for b, (c, _) in enumerate(ctx):
                    if c % RATIO:
                        p = c - 1
                        kv, sc, lo = truth[b]
                        p_kv[bt_kv[b, p // page].long(), p % page] = kv[p - lo]
                        p_sc[bt_sc[b, p // page].long(), p % page] = sc[p - lo]
                        parked += [
                            int(bt_kv[b, p // page]) * page + p % page,
                            int(bt_sc[b, p // page]) * page + p % page,
                        ]
                pooled = torch.zeros(total + 1, HEAD_DIM, device=DEV, dtype=out_dtype)
                op(
                    kv_score,
                    ape,
                    p_kv,
                    p_sc,
                    bt_kv[:nc],
                    bt_sc[:nc],
                    pooled[:total],
                    kv_lens,
                    start_pos,
                    cu_seq,
                    cu_out,
                    nc,
                    page,
                    HEAD_DIM,
                    RATIO,
                    max_out,
                )
                torch.cuda.synchronize()
                got = torch.cat([pooled[cu[b] : cu[b + 1]].float() for b in range(nc)])
                ref = torch.cat(
                    [
                        _pool_range(kv, sc, c // RATIO, (c + n) // RATIO, lo)
                        for (kv, sc, lo), (c, n) in zip(truth, ctx)
                    ]
                )
                assert got.shape == ref.shape
                if out_dtype == torch.float32:
                    torch.testing.assert_close(
                        got, ref, msg=lambda m: f"{scenario} owner {owner}: {m}"
                    )
                elif got.numel():
                    torch.testing.assert_close(
                        got.bfloat16(),
                        ref.bfloat16(),
                        msg=lambda m: f"{scenario} owner {owner}: {m}",
                    )
                    exact = (got.bfloat16() == ref.bfloat16()).float().mean().item()
                    # Same gate as test_bf16_output.
                    assert exact >= 0.999, (
                        f"{scenario} owner {owner}: bf16 bit-exact fraction {exact}"
                    )
                assert (pooled[cu[nc] :] == 0).all(), (
                    f"{scenario} owner {owner}: rows past the contexts' outputs"
                )
                flat = pool.view(-1, HEAD_DIM)
                for b, is_score, pos, ids in writes:
                    kv, sc, lo = truth[b]
                    single = counts[torch.searchsorted(uniq, ids)] == 1
                    want = (sc if is_score else kv)[pos[single] - lo]
                    assert torch.equal(flat[ids[single]], want), (
                        f"{scenario} owner {owner} b {b}: state"
                    )
                written = int((~torch.isnan(flat)).any(-1).sum())
                expect = int(
                    torch.unique(
                        torch.cat([all_ids, torch.tensor(parked, device=DEV, dtype=torch.int64)])
                    ).numel()
                )
                assert written == expect, (
                    f"{scenario} owner {owner}: {written} slots written, {expect} named"
                )
        print(
            f"[engine target call {scenario}] nc={nc} T={num_tokens} max_outputs={max_out} ctx_rows={ctx_rows} "
            f"total={total} pool={tuple(pool.shape)} table_width={bt_kv.shape[-1]} "
            f"max_kv_len={max(c + n for c, n in ctx)} distinct pages written={len(pages_seen)} "
            f"highest page={max(pages_seen)} slots shared between positions={aliased}"
        )
    finally:
        for req in requests:
            manager.free_resources(req)


# Steps on managers of the engine envelope: bf16 storage at 0.5 x total memory
# (_ENVELOPE_QUOTA_BYTES), max_num_tokens 16384, max_batch_size 256, the manager's length Lm (L + 14
# with DSpark), so the metadata's tables are 260 / 8196 columns wide. The requests are allocated
# at their full length in one resize_context, so their lengths stay short: a 1,048,576-token
# request allocated that way does not fit (resize_context returned False), and at L 1,048,576
# with max_batch_size 256 the compressor-state pool has 1662 pages. Contexts near position 1M on
# 8196-column tables are the synthetic tests' (the row sweep, the envelope steps). scenario:
# (contexts [(cached, new)], generation histories, max_draft_len, L, expected
# max_ctx_compressed_tokens[2]). The batch-256 engines' dump step (T 16340, nc 39, max 431) at L
# 32768 with DSpark: 39 contexts (one of 862 tokens at the odd cached 1 owning the 431 rows) ahead
# of 217 verify rows, 256 requests; 256 contexts of 64 tokens at odd / even prefixes (a
# 16384-token step of 256 requests); at L 1,048,576 without DSpark the dump's (16384, 1, 8192);
# and with DSpark two contexts (9000 tokens, and 3001 at an odd prefix) ahead of two verify rows.
_ENVELOPE_QUOTA_BYTES = 148_510_474_240
_ENVELOPE_RNG = torch.Generator().manual_seed(2027)
_ENVELOPE_CTX39 = [(1, 862)] + [
    ((int(c) // 2) * 2 + i % 2, n)
    for i, (c, n) in enumerate(
        zip(
            torch.randint(0, 400, (38,), generator=_ENVELOPE_RNG).tolist(),
            [373 + (7 * i) % 13 - 6 + 2 * (i == 0) for i in range(38)],
        )
    )
]
_ENVELOPE_CTX256 = [
    ((int(c) // 2) * 2 + i % 2, 64)
    for i, c in enumerate(torch.randint(0, 120, (256,), generator=_ENVELOPE_RNG).tolist())
]
_ENVELOPE_PREFILL_STEPS = {
    "gc_L32k_dump_T16340_nc39": (
        _ENVELOPE_CTX39,
        torch.randint(1, 300, (256 - 39,), generator=_ENVELOPE_RNG).tolist(),
        5,
        32768,
        431,
    ),
    "gc_L32k_nc256_T16384": (_ENVELOPE_CTX256, [], 5, 32768, 32),
    "gc_L1m_nods_T16384_nc1": ([(0, 16384)], [], 0, 1_048_576, 8192),
    "gc_L1m_T12013_nc2": ([(0, 9000), (1, 3001)], [100, 201], 5, 1_048_576, 4500),
}


@pytest.mark.parametrize("scenario", list(_ENVELOPE_PREFILL_STEPS))
def test_envelope_manager_target_call(scenario):
    """The target's exact call (as in test_engine_size_manager_target_call) on steps prepared by
    managers of the engine envelope (max_batch_size 256, max_num_tokens 16384, max_seq_len
    32768 + 14 and 1,048,576 (+ 14); the metadata's tables 260 / 8196 columns wide): up to 256
    contexts, T up to 16384, max_ctx_compressed_tokens up to 8192. The manager may map chunk
    positions behind the last two to a shared scratch slot (counted and printed); every other
    check is the gate-size test's."""
    ctx, gen_cached, max_draft_len, L, expect_max = _ENVELOPE_PREFILL_STEPS[scenario]
    lm = _envelope_lm(L, max_draft_len > 0)
    next_n = max_draft_len + 1
    assert sum(n for _, n in ctx) + len(gen_cached) * next_n <= _ENVELOPE_TOKENS, scenario
    assert len(ctx) + len(gen_cached) <= _ENVELOPE_REQS and all(c + n <= lm for c, n in ctx)
    mgr = _engine_manager(
        max_draft_len,
        _ENVELOPE_REQS,
        _ENVELOPE_QUOTA_BYTES,
        "bf16",
        max_num_tokens=_ENVELOPE_TOKENS,
        max_seq_len=lm,
    )
    assert mgr.max_blocks_per_seq == _ENVELOPE_WIDTH_OF_L[L]
    _target_call_on_manager(
        mgr,
        scenario,
        ctx,
        gen_cached,
        max_draft_len,
        expect_max,
        max_num_tokens=_ENVELOPE_TOKENS,
        allow_alias=True,
    )


# The four engine-size pools: DSpark off / on (max_draft_len 5) at the accuracy gate's
# max_batch_size 32 and at the boot gate's 8, one quota. Their compressor-state pools
# are [22908 | 41088 | 21246 | 39678, 128, 512] fp32; the two at batch 8 pass element
# offset 2^31. Then the engine envelope's largest unsplit pools at 0.5 x total memory: the
# bf16 storage's [120192, 128, 512] (max_num_tokens 16384, batch 1, DSpark) and the native
# [172416, 128, 512] (max_num_tokens 16384, batch 2, DSpark off), past element offset 2^33; and
# the PP2 stage-0 managers at the same tuples: bf16 [188796, 128, 512] and B_max, native
# [247806, 128, 512]. (max_draft_len, max_batch_size, quota, storage, max_num_tokens, pp_stage,
# pages)
_ENGINE_POOLS = [
    (0, 32, _ENGINE_QUOTA_BYTES, "bf16", 8192, None, 22908),
    (0, 8, _ENGINE_QUOTA_BYTES, "bf16", 8192, None, 41088),
    (5, 32, _ENGINE_QUOTA_BYTES, "bf16", 8192, None, 21246),
    (5, 8, _ENGINE_QUOTA_BYTES, "bf16", 8192, None, 39678),
    (5, 1, _ENVELOPE_QUOTA_BYTES, "bf16", 16384, None, 120192),
    (0, 2, _ENVELOPE_QUOTA_BYTES, "native", 16384, None, 172416),
    (5, 1, _ENVELOPE_QUOTA_BYTES, "bf16", 16384, 0, 188796),
    (0, 2, _ENVELOPE_QUOTA_BYTES, "native", 16384, 0, 247806),
]


@pytest.mark.parametrize(
    "max_draft_len,max_batch_size,quota,storage,max_num_tokens,pp_stage,want_pages",
    _ENGINE_POOLS,
    ids=[
        f"D{d}-bs{b}-{st}-mnt{t}" + (f"-pp2stage{pp}" if pp is not None else "") + f"-p{p}"
        for d, b, q, st, t, pp, p in _ENGINE_POOLS
    ],
)
def test_offset_hazard_pages_of_the_engine_pools(
    max_draft_len: int,
    max_batch_size: int,
    quota: int,
    storage: str,
    max_num_tokens: int,
    pp_stage,
    want_pages: int,
):
    """At the far end, run: a 16384-token chunk from start 129 (a parked token
    read back) whose kv / score pages are the pages on both sides of every 32-bit offset
    hazard the pool reaches (_hazard_pages) and otherwise its last pages. Outputs match
    the reference, the pool holds exactly the chunk's state and the parked token, and no
    other byte of the pool changed. Control: the same call with every page id at or past
    the pool's highest hazard wrapped below it (as a 32-bit offset would wrap) fails that
    check."""
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
        DeepseekV4AttentionType as T,
    )

    pool_t = _engine_manager(
        max_draft_len,
        max_batch_size,
        quota,
        storage,
        max_num_tokens=max_num_tokens,
        pp_stage=pp_stage,
    ).get_buffers(2, T.COMPRESSOR_KV)
    num_pages, ps = pool_t.shape[0], pool_t.shape[1]
    assert num_pages == want_pages, pool_t.shape
    start, N = 129, _ENVELOPE_TOKENS
    lo = start - 1
    need = 2 * ((start + N - 1) // ps - lo // ps + 1)
    hz = _hazard_pages(num_pages)
    pages = hz + [p for p in range(num_pages - need, num_pages) if p not in hz][: need - len(hz)]
    gen = _gen(9500 + num_pages)
    kv = torch.zeros(start + N, HEAD_DIM, device=DEV)
    sc = torch.zeros_like(kv)
    kv[lo:], sc[lo:] = _make_sequence(start + N - lo, gen)
    kv_score = torch.cat([kv[start:], sc[start:]], dim=1).contiguous()
    n = (start + N) // RATIO - start // RATIO
    ref = _pool_range(kv, sc, start // RATIO, (start + N) // RATIO)
    pos = torch.arange(lo, start + N, device=DEV)

    def run(wrap: int):
        bt_kv, bt_sc, trap, mapped = _one_sequence_tables(
            num_pages, ps, lo, start + N, seed=9500, pages=pages
        )
        pool_t[mapped] = _SENTINEL
        outside = _pool_sum(pool_t) - int(pool_t[mapped].view(torch.int32).sum(dtype=torch.int64))
        pool_t[int(bt_kv[0, lo // ps]), lo % ps] = kv[lo]
        pool_t[int(bt_sc[0, lo // ps]), lo % ps] = sc[lo]
        op_kv = torch.where(bt_kv >= wrap, bt_kv % wrap, bt_kv) if wrap else bt_kv
        op_sc = torch.where(bt_sc >= wrap, bt_sc % wrap, bt_sc) if wrap else bt_sc
        out = torch.full((n + 1, HEAD_DIM), float("nan"), device=DEV)
        op(
            kv_score,
            _zero_ape(),
            pool_t,
            pool_t,
            op_kv,
            op_sc,
            out[:n],
            _i32_tensor([start + N]),
            _i32_tensor([start]),
            _i32_tensor([0, N]),
            _i32_tensor([0, n]),
            1,
            ps,
            HEAD_DIM,
            RATIO,
            n,
        )
        torch.cuda.synchronize()
        assert torch.isnan(out[n]).all(), "op wrote past the last output row"

        def check():
            torch.testing.assert_close(out[:n], ref)
            assert torch.equal(pool_t[bt_kv[0].long()[pos // ps], pos % ps], kv[pos]), "kv state"
            assert torch.equal(pool_t[bt_sc[0].long()[pos // ps], pos % ps], sc[pos]), "score state"
            assert int((pool_t[mapped] != _SENTINEL).any(-1).sum()) == 2 * pos.numel(), (
                "slot outside the positions"
            )
            assert (pool_t[trap] == _SENTINEL).all(), "the trap page was written"
            after = _pool_sum(pool_t) - int(pool_t[mapped].view(torch.int32).sum(dtype=torch.int64))
            assert after == outside, "pool written outside the mapped pages"

        return check

    check = run(0)
    check()
    wrap = max(hz)
    print(
        f"[hazard pages D={max_draft_len} bs={max_batch_size} {storage} pp_stage={pp_stage}] pool "
        f"{tuple(pool_t.shape)} "
        f"({pool_t.numel() / 2**31:.2f} x 2^31 elements), hazard pages {hz}: outputs and state exact"
    )
    check = run(wrap)
    with pytest.raises(AssertionError):
        check()
    print(f"[hazard pages control] page ids >= {wrap} wrapped mod {wrap}: the check fails")


# ---------------------------------------------------------------------------
# The engine configuration's envelope -- multi-request steps, the
# kernel-variant map, and pool-size invariance
# ---------------------------------------------------------------------------
#
# The engine envelope bounds a step at max_num_tokens 16384 tokens and max_batch_size
# 256 requests, each at most the manager's length Lm (max_seq_len L up to 1,048,576, + 14 with
# DSpark). A prefill step hands the op num_contexts = k in 1..256 context chunks packed first
# in kv_score, then the step's generation rows (never read); max_outputs is the target's
# md.max_ctx_compressed_tokens[2] = max_b(kv_len_b // 2 - sp_b // 2), up to 8192; the tables
# are the manager's max_blocks_per_seq (132 .. 8196) columns wide. The row axis of one chunk
# (1..16384 new tokens) is test_every_row_count_1_to_16384_on_the_engine_pool.

_ENV_TOKENS = _ENVELOPE_TOKENS
_ENV_REQS = _ENVELOPE_REQS
_ENV_PAGES = (
    4096  # plainly more than a step's pages; pool size itself is test_pool_size_invariance's axis
)
_ENV_SENTINEL = 7777.0

# Every (T, num_contexts, max_ctx_compressed_tokens[2]) the core target's runtime-value
# dump recorded for this op: the boot gates' logs and the TEP4 / DEP4 / TP4 accuracy-gate
# previews' logs, parsed line by line (a line is kept only if it is exactly one tuple of
# arity 4). Five garbles were dropped: nc 18 and 4096, two with max above what T allows, and
# (1767, 2, 3) -- once in the TP4 preview's log, where the other ranks print (1767, 2, 465),
# and impossible for two contexts with at most 30 one-token generation rows. T counts the
# step's generation rows too.
_DUMP_PREFILL_CALLS = [
    (1, 1, 0),
    (2, 1, 0),
    (2, 1, 1),
    (14, 1, 7),
    (16, 1, 8),
    (48, 1, 24),
    (524, 1, 246),
    (525, 1, 247),
    (534, 1, 252),
    (560, 1, 265),
    (580, 1, 274),
    (580, 1, 275),
    (581, 1, 275),
    (586, 1, 277),
    (587, 1, 278),
    (593, 1, 281),
    (600, 1, 285),
    (602, 1, 286),
    (604, 1, 286),
    (609, 1, 289),
    (610, 1, 289),
    (612, 1, 290),
    (617, 1, 293),
    (621, 1, 295),
    (624, 1, 296),
    (625, 1, 297),
    (626, 1, 297),
    (628, 1, 298),
    (629, 1, 299),
    (630, 1, 299),
    (632, 1, 301),
    (634, 1, 301),
    (637, 1, 303),
    (640, 1, 304),
    (643, 1, 306),
    (645, 1, 307),
    (647, 1, 308),
    (648, 1, 308),
    (650, 1, 309),
    (654, 1, 311),
    (656, 1, 312),
    (657, 1, 313),
    (658, 1, 313),
    (660, 1, 314),
    (660, 1, 315),
    (661, 1, 315),
    (662, 1, 315),
    (663, 1, 316),
    (665, 1, 317),
    (666, 1, 317),
    (668, 1, 318),
    (671, 1, 320),
    (672, 1, 320),
    (673, 1, 321),
    (674, 1, 321),
    (675, 1, 322),
    (677, 1, 323),
    (680, 1, 324),
    (681, 1, 325),
    (682, 1, 325),
    (683, 1, 326),
    (684, 1, 326),
    (685, 1, 327),
    (687, 1, 328),
    (691, 1, 330),
    (692, 1, 330),
    (693, 1, 331),
    (694, 1, 331),
    (696, 1, 332),
    (697, 1, 333),
    (698, 1, 333),
    (699, 1, 334),
    (701, 1, 335),
    (703, 1, 336),
    (705, 1, 337),
    (706, 1, 337),
    (706, 1, 338),
    (707, 1, 338),
    (710, 1, 339),
    (711, 1, 340),
    (712, 1, 340),
    (713, 1, 341),
    (714, 1, 341),
    (715, 1, 342),
    (716, 1, 342),
    (718, 1, 343),
    (719, 1, 344),
    (720, 1, 344),
    (721, 1, 345),
    (722, 1, 345),
    (723, 1, 346),
    (724, 1, 347),
    (725, 1, 347),
    (727, 1, 348),
    (728, 1, 348),
    (729, 1, 349),
    (730, 1, 349),
    (731, 1, 350),
    (732, 1, 350),
    (733, 1, 351),
    (734, 1, 351),
    (735, 1, 352),
    (737, 1, 353),
    (739, 1, 354),
    (741, 1, 355),
    (742, 1, 355),
    (743, 1, 356),
    (744, 1, 356),
    (745, 1, 357),
    (746, 1, 357),
    (749, 1, 359),
    (750, 1, 359),
    (751, 1, 360),
    (752, 1, 360),
    (753, 1, 361),
    (754, 1, 361),
    (755, 1, 362),
    (756, 1, 362),
    (757, 1, 363),
    (758, 1, 363),
    (759, 1, 364),
    (760, 1, 364),
    (761, 1, 365),
    (763, 1, 366),
    (764, 1, 366),
    (764, 1, 367),
    (765, 1, 367),
    (766, 1, 367),
    (767, 1, 368),
    (768, 1, 368),
    (768, 1, 369),
    (769, 1, 369),
    (770, 1, 369),
    (771, 1, 370),
    (772, 1, 370),
    (773, 1, 371),
    (774, 1, 371),
    (776, 1, 372),
    (776, 1, 373),
    (777, 1, 373),
    (778, 1, 373),
    (779, 1, 374),
    (780, 1, 374),
    (782, 1, 375),
    (783, 1, 376),
    (784, 1, 376),
    (785, 1, 377),
    (786, 1, 377),
    (787, 1, 378),
    (788, 1, 378),
    (789, 1, 379),
    (790, 1, 379),
    (791, 1, 380),
    (792, 1, 380),
    (793, 1, 381),
    (794, 1, 381),
    (795, 1, 382),
    (796, 1, 382),
    (797, 1, 383),
    (798, 1, 383),
    (799, 1, 384),
    (800, 1, 384),
    (801, 1, 385),
    (802, 1, 385),
    (803, 1, 386),
    (804, 1, 386),
    (805, 1, 387),
    (806, 1, 387),
    (807, 1, 388),
    (808, 1, 388),
    (809, 1, 389),
    (810, 1, 389),
    (811, 1, 390),
    (812, 1, 390),
    (813, 1, 391),
    (814, 1, 391),
    (815, 1, 392),
    (816, 1, 392),
    (817, 1, 393),
    (818, 1, 393),
    (819, 1, 394),
    (821, 1, 395),
    (822, 1, 395),
    (823, 1, 396),
    (824, 1, 396),
    (825, 1, 397),
    (826, 1, 397),
    (827, 1, 398),
    (828, 1, 398),
    (829, 1, 399),
    (830, 1, 399),
    (831, 1, 400),
    (832, 1, 400),
    (833, 1, 401),
    (834, 1, 401),
    (835, 1, 402),
    (836, 1, 402),
    (837, 1, 403),
    (838, 1, 403),
    (839, 1, 404),
    (840, 1, 404),
    (841, 1, 405),
    (842, 1, 405),
    (843, 1, 406),
    (844, 1, 406),
    (845, 1, 407),
    (846, 1, 407),
    (847, 1, 408),
    (848, 1, 408),
    (849, 1, 409),
    (850, 1, 409),
    (851, 1, 410),
    (853, 1, 411),
    (854, 1, 411),
    (855, 1, 412),
    (856, 1, 412),
    (856, 1, 413),
    (857, 1, 413),
    (858, 1, 413),
    (859, 1, 414),
    (860, 1, 414),
    (861, 1, 415),
    (862, 1, 415),
    (863, 1, 416),
    (864, 1, 416),
    (865, 1, 417),
    (866, 1, 417),
    (867, 1, 418),
    (868, 1, 418),
    (869, 1, 419),
    (870, 1, 419),
    (871, 1, 420),
    (872, 1, 420),
    (874, 1, 421),
    (875, 1, 422),
    (876, 1, 422),
    (877, 1, 423),
    (878, 1, 423),
    (879, 1, 424),
    (880, 1, 424),
    (881, 1, 425),
    (882, 1, 425),
    (883, 1, 426),
    (884, 1, 426),
    (885, 1, 427),
    (886, 1, 427),
    (887, 1, 428),
    (888, 1, 428),
    (888, 1, 444),
    (889, 1, 429),
    (890, 1, 429),
    (890, 1, 445),
    (891, 1, 430),
    (892, 1, 430),
    (893, 1, 431),
    (894, 1, 431),
    (895, 1, 432),
    (897, 1, 433),
    (898, 1, 433),
    (899, 1, 434),
    (900, 1, 434),
    (901, 1, 435),
    (902, 1, 435),
    (903, 1, 436),
    (904, 1, 436),
    (905, 1, 437),
    (906, 1, 437),
    (907, 1, 438),
    (908, 1, 438),
    (909, 1, 439),
    (910, 1, 439),
    (911, 1, 440),
    (912, 1, 440),
    (913, 1, 441),
    (914, 1, 441),
    (915, 1, 442),
    (916, 1, 442),
    (917, 1, 443),
    (918, 1, 443),
    (919, 1, 444),
    (920, 1, 444),
    (921, 1, 445),
    (922, 1, 445),
    (923, 1, 446),
    (924, 1, 446),
    (925, 1, 447),
    (926, 1, 447),
    (927, 1, 448),
    (928, 1, 448),
    (929, 1, 449),
    (930, 1, 449),
    (931, 1, 450),
    (932, 1, 450),
    (933, 1, 451),
    (934, 1, 451),
    (935, 1, 452),
    (937, 1, 453),
    (938, 1, 453),
    (939, 1, 454),
    (940, 1, 454),
    (941, 1, 455),
    (942, 1, 455),
    (943, 1, 456),
    (944, 1, 456),
    (945, 1, 457),
    (946, 1, 457),
    (946, 1, 458),
    (947, 1, 458),
    (948, 1, 458),
    (949, 1, 459),
    (950, 1, 459),
    (951, 1, 460),
    (952, 1, 460),
    (952, 1, 461),
    (953, 1, 461),
    (954, 1, 461),
    (955, 1, 462),
    (956, 1, 462),
    (957, 1, 463),
    (958, 1, 463),
    (959, 1, 464),
    (960, 1, 464),
    (961, 1, 465),
    (962, 1, 465),
    (963, 1, 466),
    (964, 1, 466),
    (965, 1, 467),
    (966, 1, 467),
    (967, 1, 468),
    (968, 1, 468),
    (969, 1, 469),
    (970, 1, 469),
    (971, 1, 470),
    (972, 1, 470),
    (973, 1, 471),
    (974, 1, 471),
    (975, 1, 472),
    (976, 1, 472),
    (977, 1, 473),
    (978, 1, 473),
    (980, 1, 474),
    (981, 1, 475),
    (982, 1, 475),
    (983, 1, 476),
    (984, 1, 476),
    (985, 1, 477),
    (986, 1, 477),
    (987, 1, 478),
    (988, 1, 478),
    (989, 1, 479),
    (990, 1, 479),
    (991, 1, 480),
    (992, 1, 480),
    (993, 1, 481),
    (994, 1, 481),
    (995, 1, 482),
    (996, 1, 482),
    (997, 1, 483),
    (998, 1, 483),
    (999, 1, 484),
    (1000, 1, 484),
    (1001, 1, 485),
    (1003, 1, 486),
    (1004, 1, 486),
    (1004, 1, 487),
    (1005, 1, 487),
    (1006, 1, 487),
    (1007, 1, 488),
    (1008, 1, 488),
    (1010, 1, 489),
    (1011, 1, 490),
    (1013, 1, 491),
    (1014, 1, 491),
    (1015, 1, 492),
    (1017, 1, 493),
    (1018, 1, 493),
    (1018, 1, 508),
    (1019, 1, 494),
    (1020, 1, 494),
    (1021, 1, 495),
    (1022, 1, 495),
    (1023, 1, 496),
    (1024, 1, 496),
    (1025, 1, 497),
    (1027, 1, 498),
    (1028, 1, 498),
    (1029, 1, 499),
    (1030, 1, 499),
    (1031, 1, 500),
    (1033, 1, 501),
    (1034, 1, 501),
    (1035, 1, 502),
    (1036, 1, 502),
    (1038, 1, 503),
    (1039, 1, 504),
    (1042, 1, 505),
    (1043, 1, 506),
    (1044, 1, 506),
    (1045, 1, 507),
    (1046, 1, 507),
    (1047, 1, 508),
    (1049, 1, 509),
    (1050, 1, 509),
    (1051, 1, 510),
    (1052, 1, 510),
    (1053, 1, 511),
    (1054, 1, 511),
    (1056, 1, 512),
    (1057, 1, 513),
    (1059, 1, 514),
    (1062, 1, 515),
    (1063, 1, 516),
    (1065, 1, 517),
    (1067, 1, 518),
    (1069, 1, 519),
    (1070, 1, 519),
    (1071, 1, 520),
    (1073, 1, 521),
    (1074, 1, 521),
    (1076, 1, 522),
    (1077, 1, 523),
    (1079, 1, 524),
    (1081, 1, 525),
    (1086, 1, 527),
    (1088, 1, 528),
    (1089, 1, 529),
    (1093, 1, 531),
    (1095, 1, 532),
    (1096, 1, 532),
    (1097, 1, 533),
    (1098, 1, 533),
    (1099, 1, 534),
    (1101, 1, 535),
    (1103, 1, 536),
    (1107, 1, 538),
    (1110, 1, 539),
    (1111, 1, 540),
    (1112, 1, 540),
    (1113, 1, 541),
    (1115, 1, 542),
    (1116, 1, 542),
    (1117, 1, 543),
    (1118, 1, 543),
    (1120, 1, 544),
    (1121, 1, 545),
    (1122, 1, 545),
    (1123, 1, 546),
    (1124, 1, 546),
    (1125, 1, 547),
    (1126, 1, 547),
    (1129, 1, 549),
    (1130, 1, 549),
    (1131, 1, 550),
    (1132, 1, 550),
    (1135, 1, 552),
    (1137, 1, 553),
    (1140, 1, 554),
    (1142, 1, 555),
    (1146, 1, 557),
    (1149, 1, 559),
    (1150, 1, 559),
    (1151, 1, 560),
    (1153, 1, 561),
    (1156, 1, 562),
    (1157, 1, 563),
    (1159, 1, 564),
    (1160, 1, 564),
    (1161, 1, 565),
    (1166, 1, 567),
    (1167, 1, 568),
    (1168, 1, 568),
    (1170, 1, 569),
    (1171, 1, 570),
    (1172, 1, 570),
    (1173, 1, 571),
    (1174, 1, 571),
    (1176, 1, 572),
    (1177, 1, 573),
    (1180, 1, 574),
    (1181, 1, 575),
    (1185, 1, 577),
    (1187, 1, 593),
    (1188, 1, 578),
    (1191, 1, 580),
    (1201, 1, 585),
    (1202, 1, 585),
    (1203, 1, 586),
    (1208, 1, 588),
    (1209, 1, 589),
    (1213, 1, 591),
    (1216, 1, 592),
    (1229, 1, 599),
    (1231, 1, 600),
    (1232, 1, 600),
    (1235, 1, 602),
    (1242, 1, 605),
    (1248, 1, 608),
    (1257, 1, 613),
    (1267, 1, 618),
    (1268, 1, 618),
    (1271, 2, 314),
    (1292, 1, 630),
    (1304, 1, 636),
    (1305, 2, 348),
    (1313, 2, 390),
    (1329, 1, 649),
    (1332, 1, 650),
    (1343, 2, 382),
    (1363, 2, 335),
    (1374, 2, 349),
    (1375, 2, 373),
    (1385, 2, 349),
    (1404, 2, 384),
    (1418, 2, 378),
    (1421, 2, 365),
    (1425, 2, 369),
    (1427, 2, 405),
    (1433, 2, 370),
    (1437, 2, 429),
    (1438, 2, 369),
    (1440, 2, 427),
    (1449, 2, 361),
    (1453, 2, 367),
    (1454, 2, 365),
    (1462, 2, 469),
    (1463, 2, 393),
    (1469, 2, 364),
    (1478, 2, 378),
    (1492, 2, 385),
    (1492, 2, 387),
    (1494, 2, 376),
    (1494, 2, 391),
    (1496, 2, 412),
    (1497, 2, 438),
    (1502, 2, 371),
    (1503, 2, 388),
    (1507, 2, 396),
    (1511, 2, 388),
    (1512, 2, 394),
    (1514, 2, 417),
    (1524, 2, 447),
    (1526, 2, 427),
    (1527, 2, 409),
    (1528, 2, 389),
    (1532, 1, 750),
    (1535, 2, 441),
    (1537, 2, 419),
    (1538, 2, 423),
    (1540, 2, 410),
    (1541, 2, 390),
    (1542, 2, 392),
    (1543, 2, 394),
    (1544, 2, 380),
    (1544, 2, 422),
    (1548, 2, 388),
    (1549, 2, 397),
    (1550, 2, 393),
    (1553, 2, 416),
    (1554, 2, 412),
    (1555, 2, 415),
    (1555, 2, 476),
    (1558, 2, 385),
    (1563, 2, 390),
    (1565, 2, 424),
    (1567, 2, 419),
    (1568, 2, 387),
    (1569, 2, 429),
    (1572, 2, 399),
    (1573, 2, 427),
    (1576, 2, 431),
    (1576, 2, 453),
    (1577, 2, 392),
    (1577, 2, 459),
    (1580, 2, 407),
    (1581, 2, 404),
    (1581, 2, 447),
    (1584, 2, 399),
    (1584, 2, 425),
    (1591, 2, 400),
    (1592, 2, 414),
    (1592, 2, 420),
    (1596, 2, 392),
    (1596, 2, 427),
    (1597, 2, 394),
    (1597, 2, 405),
    (1597, 2, 455),
    (1598, 2, 407),
    (1599, 2, 444),
    (1600, 2, 425),
    (1601, 2, 415),
    (1604, 2, 398),
    (1604, 2, 405),
    (1604, 2, 417),
    (1605, 2, 424),
    (1609, 2, 398),
    (1610, 2, 427),
    (1613, 2, 517),
    (1614, 2, 399),
    (1614, 2, 413),
    (1615, 2, 403),
    (1615, 2, 407),
    (1621, 2, 402),
    (1623, 2, 431),
    (1624, 2, 400),
    (1624, 2, 437),
    (1624, 2, 459),
    (1625, 2, 424),
    (1625, 2, 435),
    (1626, 2, 430),
    (1631, 2, 424),
    (1631, 2, 454),
    (1635, 2, 412),
    (1635, 2, 421),
    (1636, 2, 418),
    (1636, 2, 423),
    (1636, 2, 457),
    (1638, 2, 436),
    (1640, 2, 426),
    (1644, 2, 410),
    (1645, 2, 429),
    (1646, 2, 470),
    (1650, 2, 439),
    (1650, 2, 440),
    (1651, 2, 483),
    (1654, 2, 433),
    (1655, 2, 434),
    (1656, 2, 460),
    (1657, 2, 417),
    (1657, 2, 465),
    (1659, 2, 412),
    (1660, 2, 444),
    (1666, 2, 441),
    (1667, 2, 411),
    (1667, 2, 444),
    (1667, 2, 456),
    (1669, 2, 477),
    (1670, 2, 457),
    (1671, 2, 434),
    (1672, 2, 502),
    (1674, 2, 434),
    (1676, 2, 436),
    (1677, 2, 443),
    (1679, 2, 435),
    (1684, 2, 462),
    (1685, 2, 482),
    (1686, 2, 422),
    (1691, 2, 471),
    (1691, 2, 532),
    (1692, 2, 416),
    (1692, 2, 471),
    (1693, 2, 429),
    (1693, 2, 433),
    (1693, 2, 435),
    (1693, 2, 454),
    (1694, 2, 451),
    (1694, 2, 481),
    (1695, 2, 435),
    (1696, 2, 471),
    (1697, 2, 455),
    (1698, 2, 424),
    (1698, 2, 518),
    (1699, 2, 448),
    (1700, 2, 450),
    (1702, 2, 420),
    (1702, 2, 427),
    (1702, 2, 444),
    (1705, 2, 428),
    (1705, 2, 435),
    (1705, 2, 463),
    (1706, 2, 440),
    (1708, 2, 452),
    (1710, 2, 454),
    (1710, 2, 469),
    (1712, 2, 465),
    (1713, 2, 462),
    (1714, 2, 458),
    (1715, 2, 455),
    (1715, 2, 458),
    (1715, 2, 459),
    (1716, 2, 462),
    (1717, 2, 453),
    (1717, 2, 461),
    (1718, 2, 425),
    (1719, 2, 482),
    (1721, 2, 453),
    (1723, 2, 498),
    (1724, 2, 424),
    (1725, 2, 438),
    (1725, 2, 554),
    (1726, 2, 504),
    (1729, 2, 509),
    (1730, 2, 432),
    (1731, 2, 441),
    (1732, 2, 501),
    (1733, 2, 459),
    (1734, 2, 439),
    (1735, 2, 428),
    (1735, 2, 438),
    (1736, 2, 479),
    (1737, 2, 428),
    (1739, 2, 428),
    (1739, 2, 493),
    (1740, 2, 467),
    (1740, 2, 564),
    (1742, 2, 438),
    (1742, 2, 455),
    (1743, 2, 453),
    (1744, 2, 429),
    (1746, 2, 438),
    (1746, 2, 456),
    (1749, 2, 503),
    (1750, 2, 459),
    (1750, 2, 460),
    (1753, 2, 495),
    (1754, 2, 450),
    (1755, 2, 462),
    (1755, 2, 486),
    (1756, 2, 459),
    (1758, 2, 435),
    (1758, 2, 447),
    (1761, 2, 438),
    (1761, 2, 451),
    (1764, 2, 464),
    (1764, 2, 475),
    (1765, 2, 499),
    (1767, 2, 465),
    (1768, 2, 477),
    (1770, 2, 501),
    (1771, 2, 453),
    (1772, 2, 446),
    (1772, 2, 475),
    (1773, 2, 458),
    (1773, 2, 482),
    (1774, 2, 502),
    (1777, 2, 445),
    (1777, 2, 483),
    (1777, 2, 495),
    (1777, 2, 545),
    (1778, 2, 454),
    (1779, 2, 471),
    (1780, 2, 459),
    (1780, 2, 460),
    (1780, 2, 463),
    (1782, 2, 543),
    (1782, 2, 550),
    (1783, 2, 496),
    (1784, 2, 442),
    (1784, 2, 457),
    (1785, 2, 444),
    (1786, 2, 449),
    (1786, 2, 475),
    (1787, 2, 493),
    (1788, 2, 461),
    (1788, 2, 549),
    (1789, 2, 511),
    (1790, 2, 494),
    (1794, 2, 476),
    (1796, 2, 467),
    (1799, 2, 446),
    (1799, 2, 472),
    (1800, 2, 480),
    (1801, 2, 545),
    (1802, 2, 496),
    (1802, 2, 524),
    (1803, 2, 502),
    (1805, 2, 463),
    (1805, 2, 479),
    (1805, 2, 559),
    (1806, 2, 476),
    (1810, 2, 464),
    (1811, 2, 446),
    (1811, 2, 488),
    (1811, 2, 509),
    (1812, 2, 458),
    (1813, 2, 568),
    (1815, 2, 462),
    (1820, 2, 483),
    (1821, 2, 465),
    (1821, 2, 478),
    (1823, 2, 458),
    (1824, 2, 538),
    (1827, 2, 599),
    (1829, 2, 455),
    (1829, 2, 559),
    (1831, 2, 456),
    (1832, 2, 467),
    (1832, 2, 490),
    (1832, 2, 573),
    (1833, 2, 498),
    (1834, 2, 457),
    (1835, 2, 475),
    (1838, 2, 484),
    (1838, 2, 539),
    (1840, 2, 533),
    (1842, 2, 494),
    (1844, 2, 476),
    (1844, 2, 491),
    (1847, 2, 455),
    (1847, 2, 456),
    (1847, 2, 470),
    (1847, 2, 544),
    (1848, 2, 521),
    (1849, 2, 488),
    (1850, 2, 476),
    (1850, 2, 492),
    (1850, 2, 501),
    (1851, 2, 524),
    (1853, 2, 456),
    (1854, 2, 481),
    (1855, 2, 464),
    (1855, 2, 465),
    (1856, 2, 533),
    (1858, 2, 481),
    (1858, 2, 530),
    (1862, 2, 571),
    (1866, 2, 561),
    (1867, 2, 487),
    (1868, 2, 510),
    (1873, 2, 473),
    (1874, 2, 522),
    (1880, 2, 480),
    (1880, 2, 574),
    (1882, 2, 464),
    (1885, 2, 487),
    (1886, 2, 475),
    (1886, 2, 488),
    (1886, 2, 561),
    (1887, 2, 499),
    (1889, 2, 500),
    (1891, 2, 477),
    (1893, 2, 472),
    (1893, 2, 487),
    (1894, 2, 537),
    (1898, 2, 473),
    (1899, 2, 470),
    (1899, 2, 487),
    (1901, 2, 513),
    (1903, 2, 482),
    (1903, 2, 554),
    (1903, 2, 562),
    (1903, 2, 588),
    (1905, 2, 561),
    (1906, 2, 600),
    (1908, 2, 478),
    (1912, 2, 585),
    (1915, 2, 483),
    (1917, 2, 472),
    (1921, 2, 568),
    (1922, 2, 483),
    (1922, 2, 496),
    (1925, 2, 501),
    (1931, 2, 525),
    (1932, 2, 543),
    (1935, 2, 960),
    (1940, 2, 505),
    (1940, 2, 514),
    (1941, 2, 484),
    (1942, 2, 534),
    (1943, 2, 543),
    (1944, 2, 512),
    (1950, 2, 540),
    (1951, 2, 522),
    (1951, 2, 535),
    (1951, 2, 600),
    (1954, 2, 502),
    (1955, 2, 577),
    (1956, 2, 488),
    (1956, 2, 515),
    (1956, 2, 549),
    (1961, 2, 521),
    (1965, 2, 529),
    (1967, 2, 491),
    (1969, 2, 550),
    (1977, 2, 575),
    (1982, 2, 504),
    (1983, 2, 542),
    (1984, 2, 533),
    (1986, 2, 499),
    (1994, 2, 529),
    (1998, 2, 497),
    (1998, 2, 545),
    (1999, 2, 512),
    (1999, 2, 515),
    (2002, 2, 538),
    (2003, 2, 506),
    (2003, 2, 559),
    (2005, 2, 527),
    (2006, 2, 494),
    (2006, 2, 504),
    (2007, 2, 559),
    (2009, 2, 526),
    (2010, 2, 507),
    (2010, 2, 518),
    (2012, 2, 519),
    (2012, 2, 526),
    (2014, 2, 547),
    (2017, 2, 522),
    (2018, 2, 505),
    (2023, 2, 512),
    (2027, 2, 516),
    (2028, 2, 499),
    (2034, 2, 524),
    (2037, 2, 588),
    (2038, 2, 540),
    (2039, 2, 565),
    (2043, 2, 548),
    (2046, 2, 577),
    (2047, 2, 519),
    (2047, 2, 552),
    (2048, 1, 1024),
    (2049, 2, 547),
    (2049, 2, 562),
    (2050, 2, 659),
    (2055, 3, 405),
    (2056, 2, 559),
    (2064, 2, 520),
    (2075, 2, 540),
    (2075, 2, 546),
    (2081, 2, 528),
    (2082, 4, 344),
    (2084, 2, 549),
    (2102, 2, 585),
    (2103, 2, 540),
    (2107, 2, 533),
    (2112, 2, 556),
    (2114, 3, 365),
    (2122, 2, 599),
    (2124, 2, 583),
    (2135, 2, 564),
    (2137, 2, 529),
    (2147, 2, 554),
    (2147, 2, 570),
    (2151, 2, 541),
    (2165, 2, 750),
    (2170, 2, 564),
    (2176, 2, 618),
    (2184, 3, 457),
    (2190, 2, 569),
    (2193, 2, 545),
    (2194, 2, 592),
    (2195, 2, 636),
    (2209, 2, 568),
    (2242, 3, 450),
    (2258, 2, 569),
    (2261, 3, 417),
    (2290, 2, 572),
    (2342, 3, 455),
    (2362, 3, 418),
    (2379, 2, 618),
    (2396, 3, 509),
    (2398, 3, 445),
    (2430, 3, 465),
    (2440, 3, 462),
    (2443, 2, 649),
    (2448, 3, 423),
    (2462, 3, 468),
    (2486, 3, 448),
    (2494, 3, 461),
    (2495, 3, 448),
    (2509, 3, 468),
    (2518, 3, 441),
    (2527, 3, 442),
    (2535, 3, 511),
    (2540, 3, 459),
    (2552, 3, 488),
    (2570, 3, 468),
    (2585, 3, 436),
    (2634, 3, 469),
    (2655, 3, 532),
    (2658, 3, 438),
    (2672, 3, 547),
    (2673, 3, 500),
    (2683, 3, 481),
    (2716, 3, 499),
    (2722, 3, 472),
    (2727, 3, 507),
    (2742, 3, 499),
    (2744, 3, 471),
    (2750, 3, 570),
    (2757, 3, 522),
    (2778, 3, 588),
    (2798, 3, 540),
    (2820, 3, 488),
    (2842, 3, 585),
    (2848, 3, 494),
    (2880, 3, 540),
    (2893, 3, 522),
    (2895, 3, 542),
    (2939, 3, 524),
    (2997, 3, 630),
    (3141, 5, 351),
    (3189, 5, 350),
    (3234, 4, 557),
    (3278, 4, 455),
    (3305, 5, 349),
    (3317, 4, 433),
    (3365, 3, 618),
    (3629, 4, 528),
    (8087, 12, 416),
    (8104, 10, 463),
    (8112, 10, 542),
    (8118, 9, 548),
    (8126, 10, 471),
    (8142, 9, 581),
    (8142, 10, 475),
    (8143, 10, 472),
    (8150, 8, 660),
    (8172, 10, 659),
    (8173, 11, 417),
    (8177, 11, 421),
    (8178, 8, 570),
    (8178, 8, 582),
    (8179, 11, 411),
    (8192, 1, 4096),
]

# Every (T, num_contexts, max_ctx_compressed_tokens[2], max_seq_len) the runtime-value dumps of
# the envelope engines recorded for this op: four engines at max_batch_size 256, max_num_tokens
# 16384, L 32768 with DSpark (CUDA graph, eager, TP4, DEP4); two at max_batch_size 8, L 32768;
# one at L 131072 without DSpark; one at L 1,048,576 with DSpark. The DEP4 engine's ranks see
# different steps, so a value one rank recorded is kept.
_ENVELOPE_DUMP_PREFILL_CALLS = [
    (1, 1, 0, 32768),
    (1, 1, 0, 131072),
    (1, 1, 0, 1048576),
    (2, 1, 0, 131072),
    (2, 1, 1, 32768),
    (2, 1, 1, 131072),
    (2, 1, 1, 1048576),
    (5, 1, 2, 32768),
    (5, 1, 2, 131072),
    (5, 1, 2, 1048576),
    (7, 1, 0, 32768),
    (7, 1, 0, 1048576),
    (12, 1, 6, 32768),
    (16, 1, 8, 32768),
    (29, 2, 8, 131072),
    (34, 2, 8, 32768),
    (34, 2, 8, 1048576),
    (597, 1, 299, 32768),
    (610, 1, 299, 32768),
    (2252, 1, 1126, 32768),
    (4559, 1, 2276, 32768),
    (8147, 2, 2938, 32768),
    (8192, 1, 4089, 32768),
    (8192, 1, 4096, 32768),
    (8467, 25, 453, 32768),
    (9193, 27, 446, 32768),
    (9240, 28, 447, 32768),
    (9364, 15, 456, 32768),
    (9370, 15, 456, 32768),
    (9524, 28, 445, 32768),
    (10441, 1, 5220, 32768),
    (10444, 1, 5222, 32768),
    (10444, 1, 5222, 131072),
    (10444, 1, 5222, 1048576),
    (10447, 1, 5214, 32768),
    (10451, 1, 5218, 32768),
    (10451, 1, 5225, 32768),
    (16298, 24, 451, 32768),
    (16308, 34, 473, 32768),
    (16316, 29, 472, 32768),
    (16320, 29, 456, 32768),
    (16322, 29, 472, 32768),
    (16327, 23, 472, 32768),
    (16328, 23, 466, 32768),
    (16329, 25, 462, 32768),
    (16333, 31, 466, 32768),
    (16340, 39, 431, 32768),
    (16378, 24, 467, 32768),
    (16384, 1, 8192, 32768),
    (16384, 1, 8192, 131072),
    (16384, 1, 8192, 1048576),
]

_BMAX_DEVICE_TOTAL = 297_020_948_480  # torch.cuda.mem_get_info() total of a GB300 here
# The largest compressor-state pool over the engine envelope at 0.5 x total memory,
# PP2 stages included (measured maxima, max_batch_size sampled): the PP2 stage-0 manager built
# alone (stage 0 holds layers 0..19, so the
# compressor state of owners 2 / 8 / 14 gets a larger share of the quota) with native KV
# storage, max_num_tokens 16384, batch 2, DSpark off, L 16384 -- 247806 pages. Below it: the
# bf16 stage-0 largest 188796 (16384, batch 1, DSpark), the unsplit managers' largest
# (960 tuples per storage mode) 172416 (native,
# 16384, batch 2) and 120192 (bf16, 16384, batch 1, DSpark), the gates' old bound 133248 (native,
# 8192, batch 8), the native gates' largest 94464 and the bf16 gates' 78078.
_BMAX_PAGES = 247806
_BF16_STAGE0_BMAX_PAGES = 188796
_UNSPLIT_BMAX_PAGES = 172416
_GATES_BMAX_PAGES = 133248
_BF16_ENVELOPE_BMAX_PAGES = 120192
_BF16_BMAX_PAGES = 78078
_NATIVE_GATE_BMAX_PAGES = 94464


class _EnvStep:
    """One prefill step of the envelope: k context chunks (sp, n) and G generation rows, at
    max_seq_len ``seq`` with or without DSpark (the manager's length Lm bounds every context's
    kv length; the tables are ``W`` = max_blocks_per_seq columns wide)."""

    def __init__(self, ctx, gen_rows, prior, seed, ape_nonzero=False, seq=16384, dspark=False):
        self.ctx = list(ctx)
        self.gen_rows = gen_rows
        self.prior = prior  # True: parked tokens come from an earlier op call; False: planted
        self.seed = seed
        self.ape_nonzero = ape_nonzero
        self.seq, self.dspark = seq, dspark
        self.lm = _envelope_lm(seq, dspark)
        self.W = _ENVELOPE_WIDTH_OF_L[seq]
        assert all(sp + n <= self.lm for sp, n in self.ctx), (seq, dspark)

    @property
    def k(self):
        return len(self.ctx)

    @property
    def tc(self):
        return sum(n for _, n in self.ctx)

    @property
    def max_outputs(self):
        return max((s + n) // RATIO - s // RATIO for s, n in self.ctx)


def _compose(rng, total: int, k: int) -> List[int]:
    """Random composition of ``total`` into k positive lengths (uniform cuts, or many
    1..3-token chunks and one remainder), in random order."""
    if k == 1:
        return [total]
    if rng.random() < 0.5 or total - (k - 1) < 1:
        cuts = sorted(rng.sample(range(1, total), k - 1))
        return [b - a for a, b in zip([0] + cuts, cuts + [total])]
    small = [rng.choice([1, 1, 1, 2, 3]) for _ in range(k - 1)]
    if total - sum(small) < 1:
        small = [1] * (k - 1)
    parts = small + [total - sum(small)]
    rng.shuffle(parts)
    return parts


def _random_start(rng, n: int, lm: int = 16384) -> int:
    """Cached prefix of a request with n new tokens (kv_len <= lm): 0, odd or even, and in one
    request of ten the chunk ends exactly at lm (the envelope's longest kv length)."""
    room = lm - n
    u = rng.random()
    if room < 1 or u < 0.3:
        return 0
    if u > 0.9:
        return room
    sp = rng.randint(1, room)
    if u < 0.6:
        sp = sp if sp % 2 else (sp - 1 if sp > 1 else 1)
    else:
        sp = sp if sp % 2 == 0 else sp - 1
    return sp


# (T context tokens, k) edge cases that lead the seeded partitions (step 0 is T 1).
_ENV_EDGES = [
    (1, 1),
    (16384, 256),
    (256, 256),
    (16384, 1),
    (257, 256),
    (16384, 2),
    (16128, 256),
    (2, 2),
    (16383, 255),
    (8192, 32),
    (33, 32),
    (16384, 128),
]
_ENV_PARTITIONS = len(_ENV_EDGES) + _ENVELOPE_REQS  # the edges, then every k 1..256
_ENV_GROUPS = 8


def _random_step(i: int) -> _EnvStep:
    """Seeded step i: the edges first, then k = every value 1..256; context tokens log-uniform
    in [k, 16384] (or within 128 of 16384), random compositions and starts, a random max_seq_len
    L of the six envelope values with DSpark on or off (contexts up to Lm), generation rows at
    next_n 6 (DSpark) or 1 after the contexts, parked tokens from an earlier call in every
    other step, a nonzero ape in one of five."""
    import random

    rng = random.Random(23_000 + i)
    seq = rng.choice(_ENVELOPE_L)
    dspark = rng.random() < 0.5
    if i in (
        3,
        5,
    ):  # a 16384-token chunk ending at Lm = 1,048,590 (DSpark) / at 1,048,576 (no DSpark)
        seq, dspark = _ENVELOPE_L[-1], i == 3
    lm = _envelope_lm(seq, dspark)
    if i == 0:
        return _EnvStep([(0, 1)], 0, prior=False, seed=23_000)  # T 1, max_outputs 0
    if i < len(_ENV_EDGES):
        tc, k = _ENV_EDGES[i]
    else:
        k = 1 + (i - len(_ENV_EDGES)) % _ENV_REQS
        if rng.random() < 0.15:
            tc = _ENV_TOKENS - rng.randint(0, 128)
        else:
            tc = int(round(math.exp(rng.uniform(math.log(k), math.log(_ENV_TOKENS)))))
        tc = min(max(tc, k), _ENV_TOKENS)
    lens = _compose(rng, tc, k)
    ctx = [(_random_start(rng, n, lm), n) for n in lens]
    if i in (3, 5):
        ctx[0] = (lm - lens[0], lens[0])
    next_n = 6 if dspark else 1
    g = rng.randint(0, _ENV_REQS - k)
    g = min(g, (_ENV_TOKENS - tc) // next_n)
    return _EnvStep(
        ctx,
        g * next_n,
        prior=(i % 2 == 0),
        seed=23_000 + i,
        ape_nonzero=(i % 5 == 4),
        seq=seq,
        dspark=dspark,
    )


def _dump_step(j: int, T: int, nc: int, M: int, seq: int = 16384) -> _EnvStep:
    """A step with exactly the dumped (T, num_contexts, max_outputs) at max_seq_len ``seq``:
    nc contexts, their tokens plus one-token generation rows (at most 256 - nc) summing to
    T, one context with exactly M outputs and none with more; random starts."""
    import random

    rng = random.Random(24_000 + j)
    g_lo, g_hi = max(0, T - nc * (2 * M + 1)), min(_ENV_REQS - nc, T - nc)
    for _ in range(4000 if g_lo <= g_hi else 0):
        g = rng.randint(g_lo, g_hi)
        tc = T - g
        par = [rng.random() < 0.5 for _ in range(nc)] if M > 0 else [False] * nc  # True: odd start
        caps = [2 * M if odd else 2 * M + 1 for odd in par]
        n0 = (
            rng.choice([2 * M - 1, 2 * M])
            if par[0]
            else (rng.choice([2 * M, 2 * M + 1]) if M else 1)
        )
        rest = tc - n0
        if n0 < 1 or rest < nc - 1 or rest > sum(caps[1:]):
            continue
        lens = [n0] + [1] * (nc - 1)
        left = rest - (nc - 1)
        idx = list(range(1, nc))
        while left > 0:
            i = rng.choice(idx)
            add = min(caps[i] - lens[i], left, rng.randint(1, max(1, left)))
            if add <= 0:
                idx.remove(i)
                continue
            lens[i] += add
            left -= add
        ctx = []
        for n, odd in zip(lens, par):
            room = seq - n
            sp = rng.randint(0, room // 2) * 2
            if odd:
                sp = sp + 1 if sp + 1 <= room else sp - 1
            ctx.append((sp, n))
        order = list(range(nc))
        rng.shuffle(order)
        step = _EnvStep([ctx[o] for o in order], g, prior=(j % 2 == 0), seed=24_000 + j, seq=seq)
        assert (step.tc + g, step.k, step.max_outputs) == (T, nc, M), (T, nc, M, ctx, g)
        return step
    raise AssertionError(f"no partition found for the dumped step {(T, nc, M)}")


class _EnvPool:
    """One fp32 [pages, 128, 512] tensor serving as both the kv and the score pool (as the
    manager's compressor-state pool does), with per-request block tables whose pages are
    distinct random pages and whose unmapped columns point at a trap page."""

    def __init__(
        self, pool: torch.Tensor, spans, rng_seed: int, pages=None, trap: int = 0, width: int = 132
    ):
        self.pool = pool
        self.ps = pool.shape[1]
        cpu = torch.Generator().manual_seed(rng_seed)
        logical = [
            list(range(lo // self.ps, (hi - 1) // self.ps + 1)) if hi > lo else []
            for lo, hi in spans
        ]
        need = 2 * sum(len(q) for q in logical)
        if pages is None:
            pages = [p for p in range(pool.shape[0]) if p != trap]
        pages = torch.tensor(list(pages))
        assert need <= pages.numel(), (need, pages.numel())
        pick = pages[torch.randperm(pages.numel(), generator=cpu)[:need]].tolist()
        self.trap = trap
        self.bt_kv = torch.full((_ENV_REQS, width), trap, dtype=torch.int32)
        self.bt_sc = torch.full_like(self.bt_kv, trap)
        used = {trap}
        for b, qs in enumerate(logical):
            for q in qs:
                pk, pc = pick.pop(), pick.pop()
                self.bt_kv[b, q], self.bt_sc[b, q] = pk, pc
                used |= {pk, pc}
        self.bt_kv = self.bt_kv.to(DEV)
        self.bt_sc = self.bt_sc.to(DEV)
        self.mapped = torch.tensor(sorted(used), device=DEV)


def _env_ref(kv, sc, ape, lo: int, w0: int, w1: int) -> torch.Tensor:
    """model.py's ratio-2 pooling of windows [w0, w1) of per-position arrays that start at
    (even) position lo; fp32."""
    p = torch.arange(w0 * RATIO, w1 * RATIO, device=DEV)
    k = kv[p - lo].unflatten(0, (w1 - w0, RATIO))
    s = (sc[p - lo] + ape[p % RATIO]).unflatten(0, (w1 - w0, RATIO))
    return (k * s.softmax(dim=1)).sum(dim=1)


def _env_call(
    step,
    pool_t,
    bt_kv,
    bt_sc,
    ctx,
    rows,
    ape,
    out_dtype,
    reserved: int,
    max_outputs=None,
    op_starts=None,
):
    """The target's call for the contexts ``ctx`` [(sp, n)] whose kv_score rows are ``rows``
    (context rows first, then NaN generation rows): returns (output buffer incl. the
    reserved rows and one guard row, cu_out list). ``op_starts(ctx)`` (controls only)
    replaces the start_pos the op is handed; every other argument keeps the true ctx."""
    nc = len(ctx)
    n_out = [(s + n) // RATIO - s // RATIO for s, n in ctx]
    cu_seq, cu_out = [0], [0]
    for (_, n), o in zip(ctx, n_out):
        cu_seq.append(cu_seq[-1] + n)
        cu_out.append(cu_out[-1] + o)
    total = cu_out[-1] + reserved
    buf = torch.zeros(total + 1, HEAD_DIM, device=DEV, dtype=out_dtype)
    buf[total] = float("nan")
    mo = max(n_out) if max_outputs is None else max_outputs
    op(
        rows,
        ape,
        pool_t,
        pool_t,
        bt_kv[:nc],
        bt_sc[:nc],
        buf[:total],
        _i32_tensor([s + n for s, n in ctx]),
        _i32_tensor([s for s, _ in ctx] if op_starts is None else op_starts(ctx)),
        _i32_tensor(cu_seq),
        _i32_tensor(cu_out),
        nc,
        128,
        HEAD_DIM,
        RATIO,
        mo,
    )
    return buf, cu_out


def _run_env_step(
    step: _EnvStep,
    pool_t: torch.Tensor,
    env: "_EnvPool" = None,
    fill: bool = True,
    out_dtypes=(torch.bfloat16, torch.float32),
    max_outputs=None,
    op_tables=None,
    op_starts=None,
):
    """Run one envelope step on ``pool_t`` and check it. Returns a dict with the outputs,
    the checks' verdicts (raises on the first failure) and the pool snapshot inputs."""
    gen = _gen(step.seed)
    ape = (
        torch.randn(RATIO, HEAD_DIM, generator=gen, device=DEV) if step.ape_nonzero else _zero_ape()
    )
    # Per request: positions [lo, sp + n) of kv / score; lo is even. With ``prior`` a
    # request with sp > 0 gets an earlier chunk [p0, sp) (p0 even) run by an earlier op
    # call, which parks the odd token; otherwise the parked token is planted.
    import random

    rng = random.Random(step.seed)
    spans, data, prior_ctx = [], [], []
    for sp, n in step.ctx:
        if step.prior and sp > 0:
            p0 = sp - rng.randint(1, min(sp, 128))
            p0 -= p0 % RATIO
        else:
            p0 = sp - sp % RATIO
        spans.append((p0, sp + n))
        prior_ctx.append((p0, sp - p0))
        data.append(_make_sequence(sp + n - p0, gen))
    if env is None:
        env = _EnvPool(pool_t, spans, step.seed, width=step.W)
    # op_tables (controls only): the tables handed to this step's call, while planting and
    # checking keep the true ones.
    bt_kv, bt_sc = op_tables(env.bt_kv, env.bt_sc) if op_tables else (env.bt_kv, env.bt_sc)
    whole = (
        pool_t.shape[0] <= _ENV_PAGES
    )  # small pools are checked whole, large ones on their mapped pages
    if fill:
        if whole:
            pool_t.fill_(_ENV_SENTINEL)
        else:
            pool_t[env.mapped] = _ENV_SENTINEL
    ps = pool_t.shape[1]

    # Earlier chunks (prior mode) or planted parked tokens.
    planted = 0
    if step.prior and any(n for _, n in prior_ctx if n):
        idx = [b for b, (_, n) in enumerate(prior_ctx) if n]
        rows = torch.cat(
            [
                torch.cat([data[b][0][: prior_ctx[b][1]], data[b][1][: prior_ctx[b][1]]], 1)
                for b in idx
            ]
        )
        sub_kv = env.bt_kv[idx].contiguous()
        sub_sc = env.bt_sc[idx].contiguous()
        buf, cu = _env_call(
            step,
            pool_t,
            sub_kv,
            sub_sc,
            [prior_ctx[b] for b in idx],
            rows.contiguous(),
            ape,
            torch.float32,
            0,
        )
        torch.cuda.synchronize()
        for j, b in enumerate(idx):
            p0, m = prior_ctx[b]
            ref = _env_ref(data[b][0], data[b][1], ape, p0, p0 // RATIO, (p0 + m) // RATIO)
            torch.testing.assert_close(
                buf[cu[j] : cu[j + 1]], ref, msg=lambda s: f"earlier chunk b={b}: {s}"
            )
    else:
        for b, (sp, n) in enumerate(step.ctx):
            if sp % RATIO:
                p = sp - 1
                kv, sc = data[b]
                lo = spans[b][0]
                pool_t[int(env.bt_kv[b, p // ps]), p % ps] = kv[p - lo]
                pool_t[int(env.bt_sc[b, p // ps]), p % ps] = sc[p - lo] + ape[p % RATIO]
                planted += 1

    # This step's kv_score: context rows in order, then the generation rows (NaN: never read).
    rows = [
        torch.cat([kv[sp - lo :], sc[sp - lo :]], 1)
        for (kv, sc), (sp, _), (lo, _) in zip(data, step.ctx, spans)
    ]
    kv_score = torch.cat(
        rows + [torch.full((step.gen_rows, 2 * HEAD_DIM), float("nan"), device=DEV)]
    ).contiguous()
    reserved = (
        step.gen_rows
    )  # output rows past the contexts' (the decode sibling's), must stay zero
    outs = {}
    for dt in out_dtypes:
        buf, cu = _env_call(
            step,
            pool_t,
            bt_kv,
            bt_sc,
            step.ctx,
            kv_score,
            ape,
            dt,
            reserved,
            max_outputs,
            op_starts,
        )
        outs[dt] = (buf, cu)
    torch.cuda.synchronize()

    ref = torch.cat(
        [
            _env_ref(kv, sc, ape, lo, sp // RATIO, (sp + n) // RATIO)
            for (kv, sc), (sp, n), (lo, _) in zip(data, step.ctx, spans)
        ]
    )
    result = {
        "ref": ref,
        "outs": outs,
        "env": env,
        "planted": planted,
        "kv_score": kv_score,
        "ape": ape,
    }
    for dt, (buf, cu) in outs.items():
        N = cu[-1]
        got = buf[:N]
        assert torch.isnan(buf[-1]).all(), "op wrote past the reserved rows"
        assert (buf[N:-1] == 0).all(), "op wrote a reserved generation row"
        if dt == torch.float32:
            torch.testing.assert_close(got, ref)
        else:
            torch.testing.assert_close(got.float(), ref, atol=1e-5, rtol=2**-8)
    if torch.float32 in outs and torch.bfloat16 in outs:
        f32, b16 = outs[torch.float32][0], outs[torch.bfloat16][0]
        N = outs[torch.float32][1][-1]
        # The kernel computes one fp32 quotient and rounds it on store: the bf16 output is
        # exactly the RNE of the fp32 output.
        assert torch.equal(b16[:N], f32[:N].bfloat16()), "bf16 output != RNE(fp32 output)"

    # State: every position of [lo, sp + n) of every request -- this call's new positions, and
    # the earlier chunk or the planted parked token -- holds kv / score + ape bit for bit.
    for b, ((kv, sc), (lo, hi)) in enumerate(zip(data, spans)):
        pos = torch.arange(lo, hi, device=DEV)
        pk = env.bt_kv[b].long()[pos // ps]
        pc = env.bt_sc[b].long()[pos // ps]
        assert torch.equal(pool_t[pk, pos % ps], kv[pos - lo]), f"request {b}: kv state"
        assert torch.equal(pool_t[pc, pos % ps], sc[pos - lo] + ape[pos % RATIO]), (
            f"request {b}: score state"
        )
    expect = 2 * sum(hi - lo for lo, hi in spans)
    written = int(((pool_t if whole else pool_t[env.mapped]) != _ENV_SENTINEL).any(-1).sum())
    assert written == expect, f"{written} slots written, {expect} named"
    assert (pool_t[env.trap] == _ENV_SENTINEL).all(), "the trap page was written"
    return result


@pytest.mark.parametrize("group", range(_ENV_GROUPS))
def test_random_partitions_of_the_envelope(group: int):
    """Over the engine envelope: seeded random prefill steps -- k in 1..256
    context chunks (every value) of T in 1..16384 context tokens (edge cases T 1 / 2 / 33 /
    256 / 257 / 16128 / 16383 / 16384 at k 1 / 2 / 32 / 128 / 255 / 256 first), random lengths
    (uniform cuts, or 1..3-token chunks plus one remainder), at a random max_seq_len L of the
    six envelope values with DSpark on or off (tables 132 .. 8196 columns wide), cached
    prefixes in [0, Lm - n] (zero, odd, even, and chunks ending exactly at Lm, up to
    1,048,590), random order, random distinct pages for kv and score, generation rows (next_n 6
    with DSpark, else 1; at most 256 requests and 16384 tokens) after the contexts, max_outputs
    = max_b(kv_len // 2 - sp // 2) up to 8192. Half the steps get each odd start's parked
    token from an earlier multi-request op call (itself checked), half get it planted; one in
    five uses a nonzero ape. Each step runs with the target's zero-filled bf16 output and an
    fp32 output: fp32 = reference at the default tolerances, bf16 = RNE of the fp32 output bit
    for bit, reserved generation rows stay zero, every state slot exact, nothing else
    written."""
    steps = [_random_step(i) for i in range(group, _ENV_PARTITIONS, _ENV_GROUPS)]
    pool_t = torch.empty(_ENV_PAGES, 128, HEAD_DIM, device=DEV)
    exact, total = 0, 0
    for step in steps:
        r = _run_env_step(step, pool_t)
        b16 = r["outs"][torch.bfloat16][0][: r["ref"].shape[0]]
        exact += int((b16 == r["ref"].bfloat16()).sum())
        total += b16.numel()
    frac = exact / max(total, 1)
    ks = sorted({s.k for s in steps})
    print(
        f"[envelope random group {group}] {len(steps)} steps, k {ks[0]}..{ks[-1]} ({len(ks)} values), "
        f"T {min(s.tc for s in steps)}..{max(s.tc for s in steps)}, max_outputs "
        f"{min(s.max_outputs for s in steps)}..{max(s.max_outputs for s in steps)}, widths "
        f"{sorted({s.W for s in steps})}, kv lengths up to {max(sp + n for s in steps for sp, n in s.ctx)}; "
        f"bf16 vs reference bit-exact {frac:.6f}"
    )
    assert frac >= 0.999, frac  # the contract's bf16 gate (see test_bf16_output)


def test_random_partitions_cover_the_envelope():
    """The seeded partitions reach the envelope's corners: every k in 1..256, T 1 and 16384,
    odd, even and zero starts, 1-token chunks, max_outputs 0 and 8192, every max_seq_len
    (table width) with DSpark on and off, chunks ending at 1,048,590 (Lm at L 1M with DSpark)
    and at 1,048,576, and at least 200 steps."""
    steps = [_random_step(i) for i in range(_ENV_PARTITIONS)]
    ks = {s.k for s in steps}
    tcs = {s.tc for s in steps}
    assert len(steps) >= 200
    assert ks == set(range(1, _ENV_REQS + 1)), sorted(set(range(1, _ENV_REQS + 1)) - ks)
    assert {1, _ENV_TOKENS} <= tcs and all(1 <= t <= _ENV_TOKENS for t in tcs)
    starts = [sp for s in steps for sp, _ in s.ctx]
    assert (
        any(sp == 0 for sp in starts)
        and any(sp % 2 for sp in starts)
        and any(sp and sp % 2 == 0 for sp in starts)
    )
    assert any(n == 1 for s in steps for _, n in s.ctx)
    mos = {s.max_outputs for s in steps}
    assert 0 in mos and _ENV_TOKENS // 2 in mos
    assert {(s.W, s.dspark) for s in steps} == {
        (w, d) for w in _ENVELOPE_WIDTHS for d in (False, True)
    }
    ends = {sp + n for s in steps for sp, n in s.ctx}
    assert _ENVELOPE_LM_1M in ends and _ENVELOPE_L[-1] in ends
    assert all(
        s.tc + s.gen_rows <= _ENV_TOKENS and s.k + (s.gen_rows > 0) <= _ENV_REQS for s in steps
    )
    assert all(sp + n <= s.lm for s in steps for sp, n in s.ctx)
    print(
        f"[envelope coverage] {len(steps)} partitions: k 1..{_ENV_REQS} all, {len(tcs)} distinct T, max_outputs "
        f"{min(mos)}..{max(mos)}, {sum(1 for sp in starts if sp % 2)} odd starts of {len(starts)}, kv lengths up to "
        f"{max(ends)}"
    )


def test_every_dumped_prefill_step():
    """The dumps: each (T, num_contexts, max_outputs) of the gates' dumps (1014,
    max_seq_len 16384) and of the envelope engines' dumps (51: T up to 16384, nc up to 39,
    max_outputs up to 8192, at L 32768 / 131072 / 1,048,576) built as a step with exactly
    those values at that max_seq_len (random lengths / starts / order, one-token generation
    rows filling T) and run as in the random partitions."""
    pool_t = torch.empty(_ENV_PAGES, 128, HEAD_DIM, device=DEV)
    cells = [(T, nc, M, 16384) for T, nc, M in _DUMP_PREFILL_CALLS] + _ENVELOPE_DUMP_PREFILL_CALLS
    assert len(_ENVELOPE_DUMP_PREFILL_CALLS) == 51
    exact, total = 0, 0
    for j, (T, nc, M, seq) in enumerate(cells):
        step = _dump_step(j, T, nc, M, seq)
        r = _run_env_step(step, pool_t)
        b16 = r["outs"][torch.bfloat16][0][: r["ref"].shape[0]]
        exact += int((b16 == r["ref"].bfloat16()).sum())
        total += b16.numel()
    frac = exact / max(total, 1)
    print(
        f"[envelope dump] {len(cells)} dumped steps (nc up to {max(t[1] for t in cells)}, T up to "
        f"{max(t[0] for t in cells)}, max_outputs up to {max(t[2] for t in cells)}) == reference; "
        f"bf16 bit-exact {frac:.6f}"
    )
    assert frac >= 0.999, frac


def test_kernel_variant_map_over_the_envelope():
    """The profiler's kernel launches over steps spread across the envelope
    (k 1..256, T 1..16384, max_outputs 0 / 1 / up to 8192, tables 132 .. 8196 columns wide,
    bf16 and fp32 output): the op launches exactly one kernel per call, always the same
    instantiation
    (prefillReductionKernel<512, 4, 4, 2, 1>: head_dim 512, fp32 kv_score, fp32 state,
    ratio 2, one reduction warp), with grid (num_contexts, max(max_outputs, 1), 1) and
    128 threads; the output dtype is a runtime argument, not a variant. Every step run
    here is also checked against the reference (by _run_env_step, outside the profile)."""
    import json
    import os
    import tempfile

    steps = [_random_step(i) for i in range(0, _ENV_PARTITIONS, 6)] + [
        _random_step(i) for i in (1, 2, 3, 8)
    ]
    steps += [
        _EnvStep([(0, 1)], 0, False, 1),
        _EnvStep([(1, 2)], 0, False, 2),
        _EnvStep([(0, 8192)], 0, False, 3),
        _EnvStep([(0, 16384)], 0, False, 4, seq=32768),
    ]
    pool_t = torch.empty(_ENV_PAGES, 128, HEAD_DIM, device=DEV)
    calls = []
    for step in steps:
        r = _run_env_step(step, pool_t)  # checked against the reference
        calls.append((step, r))
    # Profile the bare op calls again (new rows; the state re-stores itself), every input
    # built before the profile starts, so every kernel in it is the op's.
    zero_ape = _zero_ape()
    launches, expected = [], []
    for step, r in calls:
        env = r["env"]
        rows = torch.randn(
            step.tc + step.gen_rows, 2 * HEAD_DIM, generator=_gen(step.seed), device=DEV
        )
        lens = [n for _, n in step.ctx]
        outs = [(s + n) // 2 - s // 2 for s, n in step.ctx]
        args = (
            _i32_tensor([s + n for s, n in step.ctx]),
            _i32_tensor([s for s, _ in step.ctx]),
            _i32_tensor([0] + torch.tensor(lens).cumsum(0).tolist()),
            _i32_tensor([0] + torch.tensor(outs).cumsum(0).tolist()),
        )
        for dt in (torch.bfloat16, torch.float32):
            out = torch.empty(sum(outs), HEAD_DIM, device=DEV, dtype=dt)
            launches.append(
                (
                    rows,
                    env.bt_kv[: step.k],
                    env.bt_sc[: step.k],
                    out,
                    args,
                    step.k,
                    step.max_outputs,
                )
            )
            expected.append([step.k, max(step.max_outputs, 1), 1])
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for rows, bkv, bsc, out, (kvl, spos, cus, cuo), k, mo in launches:
            op(
                rows,
                zero_ape,
                pool_t,
                pool_t,
                bkv,
                bsc,
                out,
                kvl,
                spos,
                cus,
                cuo,
                k,
                128,
                HEAD_DIM,
                RATIO,
                mo,
            )
        torch.cuda.synchronize()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "trace.json")
        prof.export_chrome_trace(path)
        trace = json.load(open(path))
    kernels = [e for e in trace["traceEvents"] if e.get("cat") == "kernel"]
    ops = [e for e in kernels if "prefillReduction" in e["name"]]
    names = {e["name"] for e in ops}
    grids = [e["args"]["grid"] for e in sorted(ops, key=lambda e: e["ts"])]
    blocks = {tuple(e["args"]["block"]) for e in ops}
    others = sorted({e["name"] for e in kernels if "prefillReduction" not in e["name"]})
    print(
        f"[envelope variant map] {len(ops)} op launches over {len(steps)} steps x 2 dtypes: kernels {sorted(names)}, "
        f"blocks {sorted(blocks)}; grids (k, max(max_outputs, 1), 1) from {grids[:3]} to {grids[-3:]}; "
        f"{len({g[0] for g in grids})} distinct k (max {max(g[0] for g in grids)}), {len({g[1] for g in grids})} "
        f"distinct grid.y (max {max(g[1] for g in grids)}); other kernels {others}"
    )
    assert others == [] and len(kernels) == len(launches), (
        "the op launched something besides one kernel per call"
    )
    assert len(names) == 1, names
    (name,) = names
    assert "prefillReductionKernel<512,4,4,2,1>" in name.replace(" ", ""), name
    assert blocks == {(128, 1, 1)}, blocks
    assert grids == expected, "grid differs from (num_contexts, max(max_outputs, 1), 1)"


def test_envelope_controls():
    """The envelope checks can see the call-site defects that would matter: an undersized
    max_outputs (one below the target's max_ctx_compressed_tokens[2]) and two requests'
    kv tables swapped both fail _run_env_step on the steps where they apply."""
    pool_t = torch.empty(_ENV_PAGES, 128, HEAD_DIM, device=DEV)
    step = _EnvStep([(0, 1000), (129, 301), (4095, 3)], 5, False, 77)
    _run_env_step(step, pool_t)
    with pytest.raises(AssertionError):
        _run_env_step(step, pool_t, max_outputs=step.max_outputs - 1)

    def swap(kv, sc):
        kv = kv.clone()
        kv[[0, 1]] = kv[[1, 0]]
        return kv, sc

    with pytest.raises(AssertionError):
        _run_env_step(step, pool_t, op_tables=swap)
    print("[envelope controls] max_outputs - 1 and swapped kv tables: both rejected")


def _bmax_pool():
    """The largest compressor-state pool the engine envelope gives at 0.5 x the device's
    total memory, PP2 stages included: the live PP2 stage-0 manager (built alone) with the
    native KV storage (its smaller SWA / main-KV entries leave the state pool more of the
    quota), max_num_tokens 16384, max_batch_size 2, DSpark off, max_seq_len 16384 -- the
    largest of the sampled configurations (kv_cache_fraction 0.5 applies to free memory after
    the weights, which is below the total, and every role's pages grow monotonically with the
    quota)."""
    from tensorrt_llm._torch.attention.backends.sparse.deepseek_v4.params import (
        DeepseekV4AttentionType as T,
    )

    total = torch.cuda.mem_get_info()[1]
    pool_t = _engine_manager(
        0, 2, quota=total // 2, storage="native", max_num_tokens=_ENVELOPE_TOKENS, pp_stage=0
    ).get_buffers(2, T.COMPRESSOR_KV)
    if total == _BMAX_DEVICE_TOTAL:
        assert tuple(pool_t.shape) == (_BMAX_PAGES, 128, HEAD_DIM), tuple(pool_t.shape)
    assert pool_t.is_contiguous() and pool_t.dtype == torch.float32
    return pool_t


def _pool_sum(pool_t: torch.Tensor) -> int:
    """int64 sum of a pool's int32 bit patterns, 8192 pages at a time: one reduction over a
    B_max pool would need a temporary as large as the pool itself."""
    return sum(
        int(pool_t[i : i + 8192].view(torch.int32).sum(dtype=torch.int64))
        for i in range(0, pool_t.shape[0], 8192)
    )


_INVARIANCE_STEP = _EnvStep(
    [
        (0, 1000),
        (129, 777),
        (4095, 3),
        (1, 1),
        (16000, 383),
        (2, 2000),
        (8191, 1),
        (0, 2),
        (_ENVELOPE_LM_1M - 700, 700),
        (_ENVELOPE_LM_1M - 3001, 3000),
    ],
    6,
    False,
    31_337,
    seq=_ENVELOPE_L[-1],
    dspark=True,
)


def test_pool_size_invariance():
    """One multi-request step (odd and even starts, parked tokens, generation
    rows, two contexts ending at 1,048,590 and 1,048,589 on 8196-column tables) on the same
    pages -- a random permutation of pages 1..P with trap page 0, so the smallest pool that
    holds them has B = P + 1 pages -- run on views pool[:B] of the largest pool of the
    engine envelope (the live PP2 stage-0 manager's, _bmax_pool) for B = P + 1, a mid
    size, the bf16 gates' largest (78078), the native gates' largest (94464), the bf16
    unsplit largest (120192), the gates' old bound (133248), the native unsplit largest
    (172416), the bf16 stage-0 largest (188796) and B_max (247806), and on a separate
    exact-size [P + 1, 128, 512] allocation: outputs (bf16 and fp32) and the P + 1 pages of
    state are bitwise identical across the ten, equal the reference, and the views' pages
    past P + 1 are untouched (checksum)."""
    big = _bmax_pool()
    B_max = big.shape[0]
    step = _INVARIANCE_STEP
    spans = [(sp - sp % RATIO, sp + n) for sp, n in step.ctx]
    P = 2 * sum((hi - 1) // 128 - lo // 128 + 1 for lo, hi in spans)
    small = P + 1
    results = []
    for label, pool_t in [
        ("exact", torch.empty(small, 128, HEAD_DIM, device=DEV)),
        ("B_small", big[:small]),
        ("B_mid", big[: B_max // 2]),
        ("B_bf16_max", big[:_BF16_BMAX_PAGES]),
        ("B_native_gate", big[:_NATIVE_GATE_BMAX_PAGES]),
        ("B_bf16_envelope_max", big[:_BF16_ENVELOPE_BMAX_PAGES]),
        ("B_gates_max", big[:_GATES_BMAX_PAGES]),
        ("B_unsplit_max", big[:_UNSPLIT_BMAX_PAGES]),
        ("B_bf16_stage0_max", big[:_BF16_STAGE0_BMAX_PAGES]),
        ("B_max", big),
    ]:
        env = _EnvPool(pool_t, spans, 31_337, pages=range(1, small), trap=0, width=step.W)
        pool_t[:small] = _ENV_SENTINEL
        tail = _pool_sum(pool_t[small:]) if pool_t.shape[0] > small else None
        r = _run_env_step(step, pool_t, env=env, fill=False)
        if tail is not None:
            assert _pool_sum(pool_t[small:]) == tail, f"{label}: tail written"
        results.append(
            (
                label,
                pool_t.shape[0],
                {dt: o[0].clone() for dt, o in r["outs"].items()},
                pool_t[:small].clone(),
            )
        )
    base = results[0]
    for label, B, outs, state in results[1:]:
        for dt in outs:
            assert torch.equal(
                outs[dt].view(torch.int16 if dt == torch.bfloat16 else torch.int32),
                base[2][dt].view(torch.int16 if dt == torch.bfloat16 else torch.int32),
            ), (label, dt)
        assert torch.equal(state.view(torch.int32), base[3].view(torch.int32)), (
            f"{label}: state differs"
        )
    print(
        f"[envelope pool invariance] pool B in {[r[1] for r in results]} (views of one [{B_max}, 128, 512] fp32 "
        f"allocation + an exact-size one): {step.k} contexts on pages 1..{P}, outputs and state bitwise identical"
    )


def test_far_end_multi_request_at_b_max():
    """Far end: a multi-request step with odd starts (one context ending at
    1,048,590 on 8196-column tables) whose kv / score pages are the pages on both sides of
    every 32-bit offset hazard of the largest pool, B_max = 247806 on the live PP2 stage-0
    manager (byte offsets 2^31, 2^32;
    element offsets 2^31, 2^32, 2^33 = pages 8191/8192, 16383/16384, 32767/32768,
    65535/65536, 131071/131072) and otherwise its last pages, the very last page included:
    outputs equal the reference, state exact, nothing else in the pool changed (checksum).
    Control: the same step with every page id >= 65536 wrapped mod 65536 (a 32-bit
    element offset) fails."""
    big = _bmax_pool()
    B_max = big.shape[0]
    step = _EnvStep(
        [
            (129, 4000),
            (1, 3000),
            (16000, 383),
            (4096, 700),
            (3, 1),
            (0, 107),
            (_ENVELOPE_LM_1M - 3001, 3001),
        ],
        3,
        False,
        4242,
        seq=_ENVELOPE_L[-1],
        dspark=True,
    )
    spans = [(sp - sp % RATIO, sp + n) for sp, n in step.ctx]
    need = 2 * sum((hi - 1) // 128 - lo // 128 + 1 for lo, hi in spans)
    hz = [
        p
        for p in (8191, 8192, 16383, 16384, 32767, 32768, 65535, 65536, 131071, 131072)
        if p < B_max
    ]
    tailp = [p for p in range(B_max - need - 1, B_max) if p not in hz]
    pages = hz + tailp[len(tailp) - (need - len(hz)) :]
    assert B_max - 1 in pages and len(set(pages)) == need
    trap = next(p for p in range(B_max - 1, 0, -1) if p not in set(pages))
    env = _EnvPool(big, spans, 4242, pages=pages, trap=trap, width=step.W)
    big[env.mapped] = _ENV_SENTINEL
    outside = _pool_sum(big) - int(big[env.mapped].view(torch.int32).sum(dtype=torch.int64))
    _run_env_step(step, big, env=env, fill=False)
    after = _pool_sum(big) - int(big[env.mapped].view(torch.int32).sum(dtype=torch.int64))
    assert after == outside, "pool written outside the mapped pages"
    print(
        f"[far end] pool [{B_max}, 128, 512] fp32 ({big.numel() / 2**32:.2f} x 2^32 elements): {step.k} contexts "
        f"on hazard pages {hz} and pages up to {max(pages)}: exact, nothing else written"
    )

    def wrap(kv, sc):
        return (torch.where(kv >= 65536, kv % 65536, kv), torch.where(sc >= 65536, sc % 65536, sc))

    big[env.mapped] = _ENV_SENTINEL
    with pytest.raises(AssertionError):
        _run_env_step(step, big, env=env, fill=False, op_tables=wrap)
    print("[envelope far-end control] page ids >= 65536 wrapped mod 65536: the check fails")


# ---------------------------------------------------------------------------
# Encoder replay: context requests with zero compute rows (kv_len == start_pos)
# ---------------------------------------------------------------------------
# Under encoder SWA bounded replay a prefix hit of `reused` tokens (a multiple of 2, the
# compress ratio) starts the request's first chunk R = min(128, reused) positions early, and
# the target's _compress calls this op over the step's COMPUTE rows only:
#   kv_score     fp32 [rows, 1024] of the compute rows (v41_compute_token_index_cuda), the
#                context compute rows first, then the generation rows
#   kv_lens      metadata.kv_lens_cuda_runtime[:nc]              (cached + chunk rows)
#   start_pos    metadata.v41_compute_start_pos_cuda[:nc]        (cached + R = reused)
#   cu_seq_lens  metadata.v41_compute_cu_seq_lens_cuda[:nc + 1]  (the compute rows' offsets)
#   cu_kv_comp, max_outputs from the compute start (cu_new_comp_kv_cuda[2],
#                max_ctx_compressed_tokens[2])
# A first chunk that holds exactly its R replay rows -- the runtime admits it:
# resize_context refuses only a chunk shorter than R, and scheduler_v2 rounds a non-final
# chunk down to 128 -- is a context request with kv_len == start_pos: no row, no output,
# at an even start_pos = reused >= 2 (2..16382; a later chunk follows it). Such requests
# sit anywhere in the batch, several per step, beside ordinary chunks and generation rows.
# A step whose every context request is zero-row still calls the op when it has generation
# rows (the target returns early only when no row of the step is a compute row), with
# max_outputs 0. The zero-row chunks' replay rows count against max_num_tokens.

_REPLAY_WINDOW = 128
_ZR_PARTITIONS = _ENVELOPE_REQS  # every context count 1..256
_ZR_GROUPS = 8


def _zr_step(
    ctx,
    gen_rows: int,
    next_n: int,
    prior: bool,
    seed: int,
    ape_nonzero: bool = False,
    seq: int = 16384,
    dspark: bool = False,
) -> _EnvStep:
    step = _EnvStep(ctx, gen_rows, prior, seed, ape_nonzero, seq=seq, dspark=dspark)
    step.next_n = next_n
    return step


def _zr_replay_rows(step: _EnvStep) -> int:
    """Replay rows of the step's zero-row chunks (R = min(128, reused) each): tokens of the
    step that the op does not see."""
    return sum(min(_REPLAY_WINDOW, sp) for sp, n in step.ctx if n == 0)


def _zr_sp(rng, lm: int = 16384) -> int:
    """start_pos of a zero-row replay chunk = reused, even: a hit of 2..128 tokens (R =
    reused, the chunk starts at 0) in about a tenth, a hit within 1024 tokens of the
    manager's length lm in a tenth, else 130..lm - 2 (R = 128)."""
    u = rng.random()
    if u < 0.1:
        return rng.randrange(2, _REPLAY_WINDOW + 1, 2)
    if u < 0.2:
        return rng.randrange(max(_REPLAY_WINDOW + 2, lm - 1024), lm - 1, 2)
    return rng.randrange(_REPLAY_WINDOW + 2, lm - 1, 2)


def _zero_row_corners() -> List[_EnvStep]:
    """The corners, each with parked tokens from an earlier op call and planted."""
    many = [(2 * (65 + 307 * j), 0) for j in range(26)]
    most = [(2 * (100 + 257 * j), 0) for j in range(31)]
    mixed = [((2 * j + 1) if j % 2 else 100 * j, 260) for j in range(31)]
    # 255 short hits (R = sp) beside one chunk filling the 16384-token step (256 contexts)
    short = [(2 + 2 * (j % 40), 0) for j in range(255)]
    short_chunk = [(1, _ENVELOPE_TOKENS - sum(sp for sp, _ in short))]
    # 127 hits of R = 128 up to the 1,048,576-token envelope's far end, beside a 128-token chunk
    far = [(2 * (65 + 4127 * j), 0) for j in range(126)] + [(_ENVELOPE_L[-1] - 2, 0)]
    shapes = [
        ([(4098, 0), (131, 7), (2, 2)], 5, 1, False, 16384, False),  # zero-row first
        ([(4098, 6), (136, 0), (2, 2)], 0, 1, False, 16384, False),  # zero-row in the middle
        (
            [(6, 4), (3, 7), (10, 0)],
            12,
            6,
            False,
            16384,
            False,
        ),  # zero-row last, beside two DSpark verify requests
        ([(128, 0), (256, 0), (5, 3)], 0, 1, False, 16384, False),  # two zero-row requests
        (
            [(4098, 0)],
            31,
            1,
            False,
            16384,
            False,
        ),  # the only context, beside 31 decode rows (32 requests)
        (
            [(16382, 0)],
            30,
            6,
            False,
            16384,
            False,
        ),  # the only context at the last start, beside 5 verify requests
        (
            many,
            36,
            6,
            False,
            16384,
            False,
        ),  # 26 contexts, every one zero-row, beside 6 verify requests
        (
            most[:15] + [(1, 4224)] + most[15:],
            0,
            1,
            False,
            16384,
            False,
        ),  # 31 zero-row + one chunk: 8192 tokens
        (
            mixed[:16] + [(8190, 0)] + mixed[16:],
            0,
            1,
            False,
            16384,
            False,
        ),  # one zero-row among 31 chunks
        (
            [(2, 0), (0, 300), (16382, 0), (64, 0), (777, 5)],
            6,
            1,
            False,
            16384,
            False,
        ),  # sp 2, 16382, 64 (< 128)
        ([(4098, 0), (129, 1000)], 0, 1, True, 16384, False),  # nonzero ape
        ([(7, 0), (4, 4)], 0, 1, False, 16384, False),  # odd start_pos: not an engine form
        (
            short[:100] + short_chunk + short[100:],
            0,
            1,
            False,
            32768,
            False,
        ),  # 256 contexts, 255 zero-row
        (
            far[:60] + [(1_000_001, 128)] + far[60:],
            0,
            1,
            False,
            _ENVELOPE_L[-1],
            False,
        ),  # 127 zero-row to 1,048,574
        (
            [(_ENVELOPE_LM_1M - 2, 0), (_ENVELOPE_LM_1M - 501, 501), (2, 0)],
            12,
            6,
            False,
            _ENVELOPE_L[-1],
            True,
        ),  # at Lm - 2 = 1,048,588
    ]
    out = []
    for c, (ctx, gen_rows, next_n, ape, seq, dspark) in enumerate(shapes):
        for prior in (True, False):
            out.append(
                _zr_step(
                    ctx, gen_rows, next_n, prior, 67_000 + 2 * c + int(prior), ape, seq, dspark
                )
            )
    return out


def _zero_row_step(i: int) -> _EnvStep:
    """A seeded random step of k = i % 256 + 1 context requests, z of them zero-row (1..k, or
    1..3), at random batch positions, at a random max_seq_len L of the six envelope values with
    DSpark on or off (zero-row starts up to Lm - 2; hits shortened to R < 128 where the replay
    rows would not fit the 16384-token step); the others a random composition of the tokens
    left after the zero-row chunks' replay rows (log-uniform total), at random cached starts;
    generation rows (next_n 6 with DSpark, else 1) after the contexts; parked tokens from an
    earlier op call in three steps of four; a nonzero ape in one of five."""
    import random

    rng = random.Random(66_000 + i)
    seq = rng.choice(_ENVELOPE_L)
    dspark = rng.random() < 0.5
    lm = _envelope_lm(seq, dspark)
    k = i % _ENV_REQS + 1
    z = rng.randint(1, k) if rng.random() < 0.5 else rng.randint(1, min(3, k))
    zero = [(_zr_sp(rng, lm), 0) for _ in range(z)]
    j = 0
    while sum(min(_REPLAY_WINDOW, sp) for sp, _ in zero) > _ENV_TOKENS - (k - z) - 16:
        zero[j] = (rng.randrange(2, 16, 2), 0)
        j += 1
    budget = _ENV_TOKENS - sum(min(_REPLAY_WINDOW, sp) for sp, _ in zero)
    nz = k - z
    ctx = []
    if nz:
        tc = (
            nz
            if budget <= nz
            else int(round(math.exp(rng.uniform(math.log(nz), math.log(budget)))))
        )
        ctx = [(_random_start(rng, n, lm), n) for n in _compose(rng, min(max(tc, nz), budget), nz)]
    tc = sum(n for _, n in ctx)
    next_n = 6 if dspark else 1
    g = min(rng.randint(0, _ENV_REQS - k), (budget - tc) // next_n)
    if nz == 0 and g == 0:
        # zero-row contexts alone call the op only beside generation rows
        if k < _ENV_REQS and budget - tc >= next_n:
            g = 1
        else:
            zero.pop()
            ctx = [(_random_start(rng, 1, lm), 1)]
    order = ctx + zero
    rng.shuffle(order)
    return _zr_step(order, g * next_n, next_n, i % 4 != 3, 66_000 + i, i % 5 == 4, seq, dspark)


def _zero_row_steps() -> List[_EnvStep]:
    return _zero_row_corners() + [_zero_row_step(i) for i in range(_ZR_PARTITIONS)]


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16 if t.dtype == torch.bfloat16 else torch.int32)


def _run_zero_row_pair(step: _EnvStep, pool_t: torch.Tensor) -> dict:
    """_run_env_step on the step (reference, state of every request exact, nothing else
    written, reserved rows zero), then the step's call again from the resulting pool with
    its zero-row requests and without them: the output buffers (bf16 and fp32, reserved and
    guard rows included) and the pools after the two calls bitwise equal, and equal to the
    first run's output and to the pool before the call. With no non-empty context the
    'without' side is no call at all: the target's zero-filled buffer and an unchanged pool."""
    r = _run_env_step(step, pool_t)
    env, kv_score, ape = r["env"], r["kv_score"], r["ape"]
    keep = [b for b, (_, n) in enumerate(step.ctx) if n]
    snap = pool_t.clone()
    same_out = same_pool = True
    for dt in (torch.bfloat16, torch.float32):
        pool_t.copy_(snap)
        with_buf, _ = _env_call(
            step, pool_t, env.bt_kv, env.bt_sc, step.ctx, kv_score, ape, dt, step.gen_rows
        )
        with_pool = pool_t.clone()
        pool_t.copy_(snap)
        if keep:
            idx = torch.tensor(keep, device=DEV)
            wo_buf, _ = _env_call(
                step,
                pool_t,
                env.bt_kv[idx],
                env.bt_sc[idx],
                [step.ctx[b] for b in keep],
                kv_score,
                ape,
                dt,
                step.gen_rows,
            )
        else:
            wo_buf = torch.zeros_like(with_buf)
            wo_buf[-1] = float("nan")
        same_out &= torch.equal(_bits(with_buf), _bits(wo_buf)) and torch.equal(
            _bits(with_buf), _bits(r["outs"][dt][0])
        )
        same_pool &= torch.equal(_bits(with_pool), _bits(pool_t)) and torch.equal(
            _bits(with_pool), _bits(snap)
        )
    pool_t.copy_(snap)
    return {"same_out": same_out, "same_pool": same_pool, "ref": r["ref"], "outs": r["outs"]}


def test_zero_row_requests_at_both_page_sizes():
    """The reviewer's probe plans, frozen, at page sizes 128 and 256 (chunk plans through
    _run_plan: a 0 is a zero-row request in that call): a zero-row request first (sp 4098),
    in the middle (sp 136), last (sp 10), two in one call, and an odd sp 7 (not an engine
    form); fp32 and bf16 outputs match the reference, every position's state exact, no
    other slot (trap page included) and no guard row written."""
    plans = {
        "zero-row first (sp 4098)": [[4098, 0, 9], [131, 5, 7], [2, 2, 2]],
        "zero-row in the middle (sp 136)": [[4098, 6, 9], [131, 5, 0], [2, 2, 2]],
        "zero-row last (sp 10)": [[6, 4], [3, 7], [10, 0]],
        "two zero-row in one call (sp 128, 256)": [[128, 0, 2], [256, 0, 4], [5, 3, 1]],
        "zero-row at odd sp 7 (not an engine form)": [[7, 0, 3], [4, 4, 4]],
    }
    ape = torch.randn(RATIO, HEAD_DIM, generator=_gen(19), device=DEV)
    ran = 0
    for page_size in (128, 256):
        for name, plan in plans.items():
            for out_dtype, a in (
                (torch.float32, _zero_ape()),
                (torch.bfloat16, _zero_ape()),
                (torch.float32, ape),
            ):
                pool, kvs, scores, outs = _run_plan(plan, page_size, a, out_dtype, seed=31)
                for b, (kv, sc, out) in enumerate(zip(kvs, scores, outs)):
                    ref = _reference_pool(kv, sc, a)
                    if out_dtype == torch.float32:
                        torch.testing.assert_close(
                            out, ref, msg=lambda s: f"{name} page {page_size} seq {b}: {s}"
                        )
                    else:
                        torch.testing.assert_close(out.float(), ref, atol=1e-5, rtol=2**-8)
                _check_state(pool, kvs, scores, a)
                ran += 1
    print(
        f"[zero-row plans] {len(plans)} plans x pages 128/256 x (fp32, bf16, fp32 + ape): {ran} runs exact"
    )


@pytest.mark.parametrize("group", range(_ZR_GROUPS))
def test_zero_row_requests_over_the_envelope(group: int):
    """Zero-row context requests (kv_len == start_pos, the encoder-replay chunk of only its
    replay rows) over the engine envelope at the engine's page size 128: the 15 corners
    (each with parked tokens from an earlier op call and planted; 255 zero-row requests beside
    one chunk, 127 hits of 128 rows up to 1,048,574, one at 1,048,588 beside a chunk ending at
    Lm 1,048,590) and 256 seeded random steps (_zero_row_step: every context count 1..256,
    1..k zero-row requests at random batch positions, even start_pos 2..Lm - 2 at a random
    max_seq_len, the other chunks random, next_n 1 or 6 generation rows). Each step:
    _run_env_step's checks (fp32 = reference, bf16 = RNE(fp32), reserved
    rows zero, every request's state exact -- a zero-row request's earlier positions
    included -- and nothing else written), then the call with and without its zero-row
    requests from the same pool: outputs and pools bitwise equal (_run_zero_row_pair)."""
    steps = _zero_row_steps()[group::_ZR_GROUPS]
    pool_t = torch.empty(_ENV_PAGES, 128, HEAD_DIM, device=DEV)
    diff_out = diff_pool = exact = total = zero = 0
    for step in steps:
        r = _run_zero_row_pair(step, pool_t)
        diff_out += int(not r["same_out"])
        diff_pool += int(not r["same_pool"])
        zero += sum(1 for _, n in step.ctx if n == 0)
        b16 = r["outs"][torch.bfloat16][0][: r["ref"].shape[0]]
        exact += int((b16 == r["ref"].bfloat16()).sum())
        total += b16.numel()
    frac = exact / max(total, 1)
    print(
        f"[zero-row group {group}] {len(steps)} steps, {zero} zero-row requests, k "
        f"{min(s.k for s in steps)}..{max(s.k for s in steps)}: reference, state and untouched slots exact; "
        f"with vs without the zero-row requests: {diff_out} steps' outputs and {diff_pool} steps' pools differ; "
        f"bf16 bit-exact {frac:.6f}"
    )
    assert diff_out == 0 and diff_pool == 0
    assert frac >= 0.999, frac


def test_zero_row_partitions_cover_the_envelope():
    """The zero-row steps reach: every context count 1..256; zero-row requests first, in the
    middle and last; several in one step; steps whose every context is zero-row (beside
    generation rows); 255 zero-row requests beside one chunk; start_pos 2, below 128 and up
    to 1,048,588 (Lm - 2 at L 1M with DSpark); next_n 1 and 6; every max_seq_len (table
    width); parked tokens from an earlier call and planted; a step of 16384 tokens with the
    replay rows; all inside the envelope."""
    steps = _zero_row_steps()
    ks, first, middle, last, multi, only, nn, prior = set(), 0, 0, 0, 0, 0, set(), set()
    sps, most = [], 0
    for s in steps:
        zs = [b for b, (_, n) in enumerate(s.ctx) if n == 0]
        assert zs, "a step without a zero-row request"
        ks.add(s.k)
        first += int(0 in zs)
        last += int(s.k - 1 in zs and s.k > 1)
        middle += int(any(0 < b < s.k - 1 for b in zs))
        multi += int(len(zs) >= 2)
        only += int(len(zs) == s.k)
        most = max(most, len(zs))
        if s.gen_rows:
            nn.add(s.next_n)
            assert s.gen_rows % s.next_n == 0
        if len(zs) == s.k:
            assert s.gen_rows > 0, "zero-row contexts alone make no call"
        prior.add(s.prior)
        sps += [s.ctx[b][0] for b in zs]
        assert s.tc + _zr_replay_rows(s) + s.gen_rows <= _ENV_TOKENS
        assert s.k + s.gen_rows // s.next_n <= _ENV_REQS
        assert all(sp + n <= s.lm for sp, n in s.ctx)
    t_max = max(s.tc + _zr_replay_rows(s) + s.gen_rows for s in steps)
    engine_sps = [sp for sp in sps if sp % 2 == 0]
    print(
        f"[zero-row coverage] {len(steps)} steps, {len(sps)} zero-row requests (first {first}, middle {middle}, "
        f"last {last}; >= 2 in {multi} steps, every context in {only}, max {most}); k {min(ks)}..{max(ks)}; "
        f"start_pos {min(engine_sps)}..{max(engine_sps)} even, {len(sps) - len(engine_sps)} odd (not an engine "
        f"form); next_n {sorted(nn)}; T with the replay rows up to {t_max}"
    )
    assert (
        ks == set(range(1, _ENV_REQS + 1))
        and min(first, middle, last, multi, only) > 0
        and most == _ENV_REQS - 1
    )
    assert nn == {1, 6} and prior == {True, False} and t_max == _ENV_TOKENS
    assert (
        min(engine_sps) == 2
        and max(engine_sps) == _ENVELOPE_LM_1M - 2
        and any(sp < _REPLAY_WINDOW for sp in engine_sps)
    )
    assert {s.W for s in steps} == set(_ENVELOPE_WIDTHS)
    assert len(steps) - len(_zero_row_corners()) >= 200


def _wrong_start_in_bounds(step: _EnvStep) -> bool:
    """Whether the control call with start_pos = reused - R for the zero-row requests stays
    inside kv_score and the output buffer (incl. its guard row): each such request then
    pools min(R / 2, grid rows) windows from rows cu_seq_lens[b].. into rows cu_kv_comp[b]..
    Steps where it would not are left out of that control rather than run out of bounds."""
    cu, cu_out = [0], [0]
    for sp, n in step.ctx:
        cu.append(cu[-1] + n)
        cu_out.append(cu_out[-1] + (sp + n) // RATIO - sp // RATIO)
    rows, out_rows, grid = (
        cu[-1] + step.gen_rows,
        cu_out[-1] + step.gen_rows + 1,
        max(step.max_outputs, 1),
    )
    for b, (sp, n) in enumerate(step.ctx):
        k = min(min(_REPLAY_WINDOW, sp) // RATIO, grid)
        if n == 0 and (sp % RATIO or cu[b] + RATIO * k > rows or cu_out[b] + k > out_rows):
            return False
    return True


def test_zero_row_controls():
    """The zero-row checks see the call-site defects that would matter: (a) start_pos taken
    from cached_token_lens (the replay start, reused - R) instead of the compute start for
    the zero-row requests -- the op then pools R rows it was not given -- fails every step
    it is run on (the engine-form steps where that wrong call stays in bounds); (b) the
    block tables packed without the zero-row requests' rows (moved to the end) fails exactly
    the steps where a zero-row request precedes a non-empty one, and passes the others."""
    pool_t = torch.empty(_ENV_PAGES, 128, HEAD_DIM, device=DEV)
    steps = _zero_row_steps()
    wrong_start = [s for s in steps if _wrong_start_in_bounds(s)]
    for step in wrong_start:
        with pytest.raises(AssertionError):
            _run_env_step(
                step,
                pool_t,
                op_starts=lambda ctx: [
                    sp - min(_REPLAY_WINDOW, sp) if n == 0 else sp for sp, n in ctx
                ],
            )
    assert len(wrong_start) >= 20, len(wrong_start)
    corners = _zero_row_corners()
    caught = passed = 0
    for step in corners + [_zero_row_step(i) for i in range(0, _ZR_PARTITIONS, 7)]:
        zs = [b for b, (_, n) in enumerate(step.ctx) if n == 0]
        keep = [b for b in range(step.k) if b not in zs]
        order = torch.tensor(keep + zs + list(range(step.k, _ENV_REQS)), device=DEV)
        precedes = any(z < max(keep) for z in zs) if keep else False
        try:
            _run_env_step(
                step,
                pool_t,
                op_tables=lambda kv, sc: (kv[order].contiguous(), sc[order].contiguous()),
            )
            ok = True
        except AssertionError:
            ok = False
        assert ok != precedes, (step.ctx, precedes)
        caught += int(not ok)
        passed += int(ok)
    print(
        f"[zero-row controls] start_pos = cached tokens: {len(wrong_start)} of {len(wrong_start)} steps rejected "
        f"({len(steps) - len(wrong_start)} steps where that call would leave its buffers not run); "
        f"tables packed without the zero-row rows: {caught} rejected (a zero-row request before a non-empty one), "
        f"{passed} unchanged (none before)"
    )
    assert caught > 0 and passed > 0


def test_zero_row_kernel_launches():
    """The zero-row corners launch what every envelope step launches: one
    prefillReductionKernel<512, 4, 4, 2, 1> per call, 128 threads, grid (num_contexts,
    max(max_outputs, 1), 1) -- a zero-row request only adds a grid row -- and nothing else."""
    import json
    import os
    import tempfile

    pool_t = torch.empty(_ENV_PAGES, 128, HEAD_DIM, device=DEV)
    zero_ape = _zero_ape()
    launches, expected = [], []
    for step in _zero_row_corners()[::2]:
        r = _run_env_step(step, pool_t)  # checked against the reference
        env = r["env"]
        rows = r["kv_score"]
        lens = [n for _, n in step.ctx]
        outs = [(s + n) // 2 - s // 2 for s, n in step.ctx]
        args = (
            _i32_tensor([s + n for s, n in step.ctx]),
            _i32_tensor([s for s, _ in step.ctx]),
            _i32_tensor([0] + torch.tensor(lens).cumsum(0).tolist()),
            _i32_tensor([0] + torch.tensor(outs).cumsum(0).tolist()),
        )
        for dt in (torch.bfloat16, torch.float32):
            out = torch.empty(sum(outs) + step.gen_rows, HEAD_DIM, device=DEV, dtype=dt)
            launches.append(
                (
                    rows,
                    env.bt_kv[: step.k],
                    env.bt_sc[: step.k],
                    out,
                    args,
                    step.k,
                    step.max_outputs,
                )
            )
            expected.append([step.k, max(step.max_outputs, 1), 1])
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for rows, bkv, bsc, out, (kvl, spos, cus, cuo), k, mo in launches:
            op(
                rows,
                zero_ape,
                pool_t,
                pool_t,
                bkv,
                bsc,
                out,
                kvl,
                spos,
                cus,
                cuo,
                k,
                128,
                HEAD_DIM,
                RATIO,
                mo,
            )
        torch.cuda.synchronize()
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "trace.json")
        prof.export_chrome_trace(path)
        trace = json.load(open(path))
    kernels = [e for e in trace["traceEvents"] if e.get("cat") == "kernel"]
    ops = sorted((e for e in kernels if "prefillReduction" in e["name"]), key=lambda e: e["ts"])
    names = {e["name"] for e in ops}
    grids = [e["args"]["grid"] for e in ops]
    blocks = {tuple(e["args"]["block"]) for e in ops}
    print(
        f"[zero-row launches] {len(ops)} launches over {len(launches) // 2} corner steps x 2 dtypes: {sorted(names)}, "
        f"blocks {sorted(blocks)}, grids {grids[::2]}"
    )
    assert len(kernels) == len(launches) == len(ops) and len(names) == 1
    assert "prefillReductionKernel<512,4,4,2,1>" in next(iter(names)).replace(" ", "")
    assert blocks == {(128, 1, 1)} and grids == expected
