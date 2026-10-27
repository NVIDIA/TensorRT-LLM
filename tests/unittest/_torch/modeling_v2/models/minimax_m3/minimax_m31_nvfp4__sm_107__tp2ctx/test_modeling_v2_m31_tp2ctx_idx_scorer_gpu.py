# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""TP2CTX_IDX_SCORER on sm_107: the MSA indexer's top-16 prefill tables with the exact K=32 Q8KV8 scorer registered
(``prefill_scorer_hook``) equal the tables of the fmha_sm100 proxy pass, on single and multi-request prefill steps,
prefixes off the page grid and wide-range scales; the hook declines (the proxy runs) on BF16 operands and on rows not
aligned bottom-right. GPU; needs the scorer module (``TRTLLM_M31_IDX_SCORER_MODULE``), else skipped.
"""

import functools
import inspect

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m31_nvfp4__sm_107__tp2ctx import (
    idx_scorer,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.common import MiniMaxM3SparseConfig
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
    per_token_valid_blocks,
)
from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.msa_indexer import MsaIndexer


def _scorer_available() -> bool:
    """The exact scorer through the target's own loader (the MSA cutlass shim first, then the K=32 check)."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        return False
    return idx_scorer._module() is not None


pytestmark = [
    pytest.mark.skipif(
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() != (10, 7)
        or not _scorer_available(),
        reason="needs sm_107 and the vendored exact scorer module",
    ),
    pytest.mark.skipif(
        "prefill_scorer_hook" not in inspect.getsource(MsaIndexer),
        reason="this MSA indexer has no prefill_scorer_hook: the opt-in scorer never registers",
    ),
]

H, D, PAGE = 2, 128, 128


def _config(num_kv_heads: int = 1) -> MiniMaxM3SparseConfig:
    return MiniMaxM3SparseConfig(
        num_q_heads=16,
        num_kv_heads=num_kv_heads,
        head_dim=D,
        num_index_heads=H,
        sparse_index_dim=D,
        block_size=PAGE,
        topk=16,
    )


def _inputs(qo, kv, seed, scale_exp=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    total_q, pages = sum(qo), sum(-(-k // PAGE) for k in kv)
    q = torch.randn(total_q, H, D, device="cuda", generator=g) * 2
    k = torch.randn(pages, 1, PAGE, D, device="cuda", generator=g) * 2
    if scale_exp:
        q = q * torch.pow(
            2.0,
            torch.randint(
                -scale_exp, scale_exp + 1, (total_q, H, 1), device="cuda", generator=g
            ).float(),
        )
        k = k * torch.pow(
            2.0,
            torch.randint(
                -scale_exp, scale_exp + 1, (pages, 1, PAGE, 1), device="cuda", generator=g
            ).float(),
        )
    perm = torch.randperm(pages, device="cuda", generator=g).to(
        torch.int32
    )  # a shuffled page table
    lens = (torch.tensor(qo, dtype=torch.int32), torch.tensor(kv, dtype=torch.int32))
    offset = lens[1] - lens[0]
    return q.to(torch.float8_e4m3fn), k.to(torch.float8_e4m3fn), perm, lens[0], lens[1], offset


def _tables(qo, kv, seed, scale_exp=0):
    q, k, kv_indices, qo_cpu, kv_cpu, off_cpu = _inputs(qo, kv, seed, scale_exp)
    args = dict(
        idx_sm_scale=D**-0.5,
        kv_indices=kv_indices,
        qo_lens_cpu=qo_cpu,
        kv_lens_cpu=kv_cpu,
        qo_offset_cpu=off_cpu,
    )
    proxy = MsaIndexer(_config())
    hooked = MsaIndexer(_config())
    calls = []

    def hook(*a):
        out = idx_scorer.score(*a)
        calls.append(out is not None)
        return out

    hooked.prefill_scorer_hook = hook
    a = proxy.select_blocks(q, k, **args)
    b = hooked.select_blocks(q, k, **args)
    torch.cuda.synchronize()
    return a, b, calls


@pytest.mark.parametrize(
    "qo, kv, scale_exp",
    [
        ([512], [512 + 8192], 0),
        ([2048], [2048 + 40960], 0),
        ([300], [300 + 12345], 0),  # prefix off the page grid
        ([256, 1000, 64], [256 + 4096, 1000 + 20000, 64 + 777], 0),  # multi-request varlen
        ([512], [512 + 8192], 6),  # 2^+-6 per-row / per-page scales
    ],
)
def test_tables_equal_the_proxy(qo, kv, scale_exp) -> None:
    a, b, calls = _tables(qo, kv, seed=sum(qo) + sum(kv), scale_exp=scale_exp)
    assert calls and all(calls), "the scorer hook declined"
    assert torch.equal(a, b)


def test_declines_bf16_and_misaligned_rows() -> None:
    q, k, kv_indices, qo_cpu, kv_cpu, off_cpu = _inputs([256], [256 + 4096], seed=3)
    assert idx_scorer.score(q.to(torch.bfloat16), k, qo_cpu, kv_cpu, off_cpu, kv_indices) is None
    assert idx_scorer.score(q, k, qo_cpu, kv_cpu, off_cpu + 1, kv_indices) is None


def _written(ref, qo_cpu, kv_cpu, off_cpu):
    """The cells the scorer writes: each token's historical pages (block < n_valid_blocks - 1); the local page, which
    the select forces, and later pages keep the score buffer's old contents."""
    n_valid = per_token_valid_blocks(qo_cpu, kv_cpu, off_cpu, causal=True, block_size=PAGE).cuda()
    return torch.arange(ref.shape[1], device="cuda")[:, None] < n_valid[None, :] - 1


def test_plan_is_kept_and_an_in_place_page_table_rewrite_is_scored() -> None:
    q, k, kv_indices, qo_cpu, kv_cpu, off_cpu = _inputs([256], [256 + 4096], seed=4)
    idx_scorer.score(q, k, qo_cpu, kv_cpu, off_cpu, kv_indices)
    first = idx_scorer._STATE["wrapper"]
    idx_scorer.score(q, k, qo_cpu, kv_cpu, off_cpu, kv_indices)
    assert idx_scorer._STATE["wrapper"] is first
    kv_indices.copy_(kv_indices.flip(0))  # in place: same lengths, new pages
    kept = idx_scorer.score(q, k, qo_cpu, kv_cpu, off_cpu, kv_indices)
    assert idx_scorer._STATE["wrapper"] is first
    idx_scorer._STATE.pop("wrapper")
    fresh = idx_scorer.score(q, k, qo_cpu, kv_cpu, off_cpu, kv_indices)
    torch.cuda.synchronize()
    written = _written(fresh, qo_cpu, kv_cpu, off_cpu)
    assert torch.equal(kept[0][written], fresh[0][written])


def test_inference_mode_page_table_scores_equal() -> None:
    """Under torch.inference_mode (serving: the page table is an inference tensor, no version counter) the scorer runs,
    keeps one plan, and its written scores and top-16 tables equal the non-inference ones and the proxy's, also after an
    in-place rewrite of the inference page table."""
    q, k, kv_indices, qo_cpu, kv_cpu, off_cpu = _inputs(
        [256, 1000], [256 + 4096, 1000 + 20000], seed=5
    )
    ref = idx_scorer.score(q, k, qo_cpu, kv_cpu, off_cpu, kv_indices)
    args = dict(idx_sm_scale=D**-0.5, qo_lens_cpu=qo_cpu, kv_lens_cpu=kv_cpu, qo_offset_cpu=off_cpu)
    ref_table = MsaIndexer(_config())
    ref_table.prefill_scorer_hook = idx_scorer.score
    c = ref_table.select_blocks(q, k, kv_indices=kv_indices, **args)
    with torch.inference_mode():
        kv_inf = kv_indices.clone()
        assert kv_inf.is_inference()
        first = idx_scorer.score(q, k, qo_cpu, kv_cpu, off_cpu, kv_inf)
        w1 = idx_scorer._STATE["wrapper"]
        second = idx_scorer.score(q, k, qo_cpu, kv_cpu, off_cpu, kv_inf)
        assert idx_scorer._STATE["wrapper"] is w1
        proxy = MsaIndexer(_config())
        hooked = MsaIndexer(_config())
        hooked.prefill_scorer_hook = idx_scorer.score
        a = proxy.select_blocks(q, k, kv_indices=kv_inf, **args)
        b = hooked.select_blocks(q, k, kv_indices=kv_inf, **args)
        kv_inf.copy_(kv_inf.flip(0))
        a2 = proxy.select_blocks(q, k, kv_indices=kv_inf, **args)
        b2 = hooked.select_blocks(q, k, kv_indices=kv_inf, **args)
    torch.cuda.synchronize()
    written = _written(ref, qo_cpu, kv_cpu, off_cpu)
    for out in (first, second):
        assert out.shape == ref.shape
        print(
            f"IDXFIX_UNWRITTEN_CELLS_DIFFER: {int(((out != ref)[0] & ~written).sum())} of {int((~written).sum())}"
        )
        assert torch.equal(out[0][written], ref[0][written])
    assert torch.equal(a, b) and torch.equal(b, c)
    assert torch.equal(a2, b2) and not torch.equal(a2, a)


def test_per_kv_head_tables_equal_the_proxy_after_an_in_place_rewrite() -> None:
    """b22 (tp2ctx: 2 index heads and 2 KV heads per rank, gate 841031): the hook registered as ``register`` binds it
    returns one score row per KV head, and its top-16 tables equal the proxy's under torch.inference_mode, also after
    an in-place rewrite of the inference page table with the plan kept."""
    q, k, kv_indices, qo_cpu, kv_cpu, off_cpu = _inputs(
        [256, 1000], [256 + 4096, 1000 + 20000], seed=6
    )
    args = dict(idx_sm_scale=D**-0.5, qo_lens_cpu=qo_cpu, kv_lens_cpu=kv_cpu, qo_offset_cpu=off_cpu)
    hook = functools.partial(idx_scorer.score, num_kv_heads=2)
    with torch.inference_mode():
        kv_inf = kv_indices.clone()
        proxy = MsaIndexer(_config(num_kv_heads=2))
        hooked = MsaIndexer(_config(num_kv_heads=2))
        hooked.prefill_scorer_hook = hook
        scores = hook(q, k, qo_cpu, kv_cpu, off_cpu, kv_inf)
        a = proxy.select_blocks(q, k, kv_indices=kv_inf, **args)
        b = hooked.select_blocks(q, k, kv_indices=kv_inf, **args)
        w = idx_scorer._STATE["wrapper"]
        kv_inf.copy_(kv_inf.flip(0))
        a2 = proxy.select_blocks(q, k, kv_indices=kv_inf, **args)
        b2 = hooked.select_blocks(q, k, kv_indices=kv_inf, **args)
        assert idx_scorer._STATE["wrapper"] is w
    torch.cuda.synchronize()
    assert scores.shape[0] == 2 and scores.shape[2] == q.shape[0]
    assert b.shape[1] == 2
    assert torch.equal(a, b) and torch.equal(a2, b2) and not torch.equal(a2, a)
