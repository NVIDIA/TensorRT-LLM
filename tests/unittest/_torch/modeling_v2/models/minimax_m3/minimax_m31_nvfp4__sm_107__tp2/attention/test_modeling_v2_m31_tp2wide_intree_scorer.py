# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP2 WIDE in-tree scorer (``modeling.INDEX_SCORE_INTREE``) of the MiniMax-M3.1 sm_107 tp2 target: from
``INDEX_SCORE_INTREE_MIN_REQUESTS`` requests on, at steps of ``TP2_WIDE_MIN_ROWS`` rows or more, the sparse decode
scores with the in-tree CuTe DSL scorer (``cute_dsl_minimax_m3_index_decode_score``, its 256-CTA split) instead of the
split-512 ``m3_index_score``. The claim is the same scores bit for bit (measured on VR).

GPU (SM107): both scorers on the dispatch's argument layouts (the [heads, blocks, tokens] selector buffer passed
transposed, the index-K cache squeezed from [pages, 1, page, dim]) at 2 index heads, qlen 8, 5 / 8 / 16 / 24 requests of
mixed lengths (tile / page edges, past 256 pages, 16K), seeds 0..3, FP8 q / k; and the same at 2^+-12 scaled operands
that saturate at the FP8 max and underflow into the subnormals. The whole backing buffer (``-inf`` padding included) is
compared as int32.

CPU: the dispatch constants, the dispatch condition (evaluated from modeling.py's source: no step of <= 32 rows and no
step of < 5 requests takes the in-tree scorer) and the kill switch ``TRTLLM_M31_DISABLE_TP2WIDE_INTREE_SCORER=1``
(evaluated from source against a private copy of m31_switches, as the S4-logits default test does).
"""

import ast
import importlib.util
import os
from pathlib import Path

import pytest
import torch

import tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3 as _family

_DIR = Path(_family.__file__).resolve().parent / "minimax_m31_nvfp4__sm_107__tp2"
_SWITCHES = _DIR / "m31_switches.py"
_MODELING = _DIR / "modeling.py"

_sm107 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7),
    reason="the sm_107 TP2 target runs on SM107 only",
)

HEADS = 2  # index heads per rank at TP2 (4 index heads over 2 ranks)
HEAD_DIM = 128  # sparse_index_dim
QLEN = 8  # K7 verify rows
REQUESTS = (5, 8, 16, 24)
SEEDS = (0, 1, 2, 3)
FP8_MAX = 448.0
# Per-request lengths, rotated by the seed: the shortest K7 request (qlen), one past it, the page / tile edges, 255..257
# pages around the in-tree split (its CTAs loop past 256 pages), 16K, and a 4-block mid length.
LENGTHS = (8, 129, 16384, 127, 128, 9, 32767, 32768, 32769, 255, 256, 257, 512, 16383, 16385, 4097)


def _lengths(requests: int, seed: int):
    return tuple(LENGTHS[(seed * 5 + i) % len(LENGTHS)] for i in range(requests))


def _operands(seq_lens, seed: int, stress: bool):
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.ops.attention.m3_index_score import (  # noqa: E501
        PAGE,
    )

    g = torch.Generator(device="cuda").manual_seed(seed)
    batch = len(seq_lens)
    tokens = batch * QLEN
    max_blocks = (max(seq_lens) + PAGE - 1) // PAGE
    stride = ((max_blocks + 15) // 16) * 16
    pages = batch * max_blocks
    block_table = (
        torch.randperm(pages, device="cuda", generator=g).to(torch.int32).view(batch, max_blocks)
    )

    def fp8(shape):
        x = torch.randn(shape, device="cuda", generator=g)
        if stress:
            # 2^+-12 per element: past the FP8 max (saturated to +-448 before the cast) and below its min normal 2^-6.
            e = torch.randint(-12, 13, shape, device="cuda", generator=g).float()
            x = (x * torch.exp2(e)).clamp(-FP8_MAX, FP8_MAX)
        return x.to(torch.float8_e4m3fn)

    # The dispatch's layouts: idx_q viewed from the projection's [tokens, heads * dim], index-K [pages, 1, page, dim].
    idx_q = fp8((tokens, HEADS * HEAD_DIM))
    idx_k_cache = fp8((pages, 1, PAGE, HEAD_DIM))
    seq = torch.tensor(seq_lens, device="cuda", dtype=torch.int32)
    return idx_q, idx_k_cache, block_table, seq, stride


def _max_score(stride: int, tokens: int):
    """The selector buffer [heads, blocks, tokens]; the dispatch passes ``max_score.transpose(1, 2)``."""
    return torch.full((HEADS, stride, tokens), -float("inf"), device="cuda", dtype=torch.float32)


def _check(seq_lens, seed: int, stress: bool) -> None:
    import tensorrt_llm._torch.custom_ops  # noqa: F401 — registers torch.ops.trtllm.*
    from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2.ops.attention.m3_index_score import (  # noqa: E501
        SPLIT_K,
        m3_index_score,
    )
    from tensorrt_llm._torch.custom_ops.cute_dsl_custom_ops import (
        CuteDSLMiniMaxM3IndexDecodeScoreRunner as in_tree,
    )

    assert SPLIT_K == 512 and in_tree.SPLIT_K == 256
    assert in_tree.is_supported(
        q_dtype=torch.float8_e4m3fn,
        num_heads=HEADS,
        head_dim=HEAD_DIM,
        page_size=in_tree.SUPPORTED_PAGE_SIZE,
        max_decode_query_len=QLEN,
    )
    idx_q, idx_k_cache, block_table, seq_lens_t, stride = _operands(seq_lens, seed, stress)
    num_tokens = idx_q.shape[0]
    assert in_tree.SUPPORTED_PAGE_SIZE == idx_k_cache.shape[2]
    idx_q_view = idx_q.view(num_tokens, HEADS, HEAD_DIM)
    base, wide = _max_score(stride, num_tokens), _max_score(stride, num_tokens)
    # The split-512 branch (early = rows <= INDEX_SCORE_EARLY_MAX_REQUESTS = 2: False from 5 requests on).
    m3_index_score(
        idx_q_view,
        idx_k_cache.squeeze(1),
        block_table,
        seq_lens_t,
        base.transpose(1, 2),
        QLEN,
        SPLIT_K,
        False,
    )
    # The TP2 WIDE branch.
    torch.ops.trtllm.cute_dsl_minimax_m3_index_decode_score(
        idx_q_view,
        idx_k_cache.squeeze(1),
        block_table,
        seq_lens_t,
        wide.transpose(1, 2),
        QLEN,
    )
    torch.cuda.synchronize()
    if not stress:
        ref = m3_index_score.reference(
            idx_q_view,
            idx_k_cache.squeeze(1),
            block_table,
            seq_lens_t,
            base.transpose(1, 2),
            QLEN,
            SPLIT_K,
            False,
        )
        torch.testing.assert_close(wide.transpose(1, 2), ref, rtol=1e-5, atol=1e-4)
    diff = base.view(torch.int32) != wide.view(torch.int32)
    assert not bool(diff.any()), (
        f"{int(diff.sum())} score words differ from split 512 (seq_lens={seq_lens}, seed={seed}, stress={stress}); "
        f"first at [head, block, token] {diff.nonzero()[0].tolist()}"
    )
    assert bool(torch.isfinite(wide).any()), "nothing was scored"


@_sm107
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("requests", REQUESTS)
def test_in_tree_scorer_is_split_512_bitwise(requests, seed) -> None:
    _check(_lengths(requests, seed), seed, stress=False)


@_sm107
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("requests", REQUESTS)
def test_in_tree_scorer_is_split_512_bitwise_at_fp8_range_edges(requests, seed) -> None:
    _check(_lengths(requests, seed), 100 + seed, stress=True)


# --------------------------------------------------------------------------------------------- CPU: the dispatch


def _tree():
    return ast.parse(_MODELING.read_text(), filename=str(_MODELING))


def _module_assign(tree, name: str) -> ast.Assign:
    found = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == name
    ]
    assert len(found) == 1, name
    return found[0]


def _constant(name: str) -> int:
    return ast.literal_eval(_module_assign(_tree(), name).value)


def _dispatch_condition():
    """The ``elif`` test that sends a sparse decode step to the in-tree scorer."""
    tests = [
        node.test
        for node in ast.walk(_tree())
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.BoolOp)
        and isinstance(node.test.op, ast.And)
        and any(isinstance(v, ast.Name) and v.id == "INDEX_SCORE_INTREE" for v in node.test.values)
    ]
    assert len(tests) == 1
    return compile(ast.Expression(tests[0]), str(_MODELING), "eval")


@pytest.mark.cpu_only
def test_dispatch_constants():
    assert _constant("INDEX_SCORE_INTREE_MIN_REQUESTS") == 5
    assert _constant("TP2_WIDE_MIN_ROWS") == 33


@pytest.mark.cpu_only
@pytest.mark.parametrize("on", [True, False], ids=["on", "killed"])
def test_no_step_of_32_rows_or_fewer_takes_the_in_tree_scorer(on):
    condition = _dispatch_condition()
    names = {
        "INDEX_SCORE_INTREE": on,
        "INDEX_SCORE_INTREE_MIN_REQUESTS": _constant("INDEX_SCORE_INTREE_MIN_REQUESTS"),
        "TP2_WIDE_MIN_ROWS": _constant("TP2_WIDE_MIN_ROWS"),
    }
    taken = set()
    for rows in range(1, 65):
        for qlen in range(1, 9):
            num_tokens = rows * qlen
            if eval(condition, {}, dict(names, rows=rows, num_tokens=num_tokens, qlen=qlen)):
                assert on and num_tokens > 32 and rows >= 5, (rows, qlen)
                taken.add((rows, qlen))
            else:
                assert not on or num_tokens <= 32 or rows < 5, (rows, qlen)
    if on:  # the K7 cells of the MB: 5 / 8 / 16 / 24 requests at qlen 8; 4 requests (32 rows) keep split 512
        assert {(5, 8), (8, 8), (16, 8), (24, 8)} <= taken and (4, 8) not in taken


def _intree_on(monkeypatch, env) -> bool:
    for name in list(os.environ):
        if name.startswith(("TRTLLM_M31_DISABLE_", "TRTLLM_M31_ENABLE_")):
            monkeypatch.delenv(name)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    spec = importlib.util.spec_from_file_location(f"_m31_switches_intree_{id(env)}", _SWITCHES)
    switches = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(switches)
    statement = _module_assign(_tree(), "INDEX_SCORE_INTREE")
    namespace = {"m31_disabled": switches.m31_disabled, "m31_enabled": switches.m31_enabled}
    exec(compile(ast.Module([statement], []), str(_MODELING), "exec"), namespace)
    return namespace["INDEX_SCORE_INTREE"]


@pytest.mark.cpu_only
def test_on_by_default(monkeypatch):
    assert _intree_on(monkeypatch, {}) is True


@pytest.mark.cpu_only
def test_the_kill_switch_turns_it_off(monkeypatch):
    assert _intree_on(monkeypatch, {"TRTLLM_M31_DISABLE_TP2WIDE_INTREE_SCORER": "1"}) is False
