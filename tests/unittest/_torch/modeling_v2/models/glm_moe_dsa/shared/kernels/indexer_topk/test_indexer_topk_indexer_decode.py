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
"""GPU correctness of the indexer_topk indexer decode package against native TRT-LLM.

Native oracle: the CuTe-DSL FP8 paged-MQA logits launch exactly as
``Indexer.sparse_attn_indexer`` issues it for the GEN DSL path (wave-aware
row-group split from ``_pick_dsl_expand`` with expanded lengths, page tables
and schedule), followed by the same ``TopK`` module. Most fixtures are
synthetic (random FP8 keys with UE8M0 scales, random queries and weights).
``test_real_weight_fixture`` uses a real GLM-5.2 layer-0 fixture when
``GLM_MOE_DSA_INDEXER_TOPK_REAL_FIXTURE`` points at it.
"""

import os
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE
from tensorrt_llm._utils import get_sm_version

if not torch.cuda.is_available() or not IS_CUTLASS_DSL_AVAILABLE:
    pytest.skip("CUDA and the CUTLASS DSL are required", allow_module_level=True)
if get_sm_version() not in (100, 103):
    pytest.skip("indexer_topk targets SM100/SM103", allow_module_level=True)

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.attention import (  # noqa: E501
    GlmIndexer,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.indexer_topk import (  # noqa: E501
    indexer_decode as indexer_topk,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.indexer_topk.logits_runner import (  # noqa: E501
    DEFAULT_CONFIG,
    LogitsConfig,
)
from tensorrt_llm._torch.attention.backends.sparse.dsa import indexer as native_indexer_mod
from tensorrt_llm._torch.attention.backends.sparse.dsa.indexer import _pick_dsl_expand
from tensorrt_llm._torch.modules.top_k import TopK, TopKImplementation
from tensorrt_llm.deep_gemm import get_num_sms, get_paged_mqa_logits_metadata

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import plan_testlib  # noqa: E402

HEADS = 32
HEAD_DIM = 128
BLOCK_KV = 64
TOP_K = 2048
NEXT_N = 6
SCHEDULE_BLOCK_KV = 64
FP8_ATOMS = (1, 2, 3, 4)
_DEV = torch.device("cuda")


def _ue8m0(x):
    return torch.pow(2.0, torch.ceil(torch.log2(x.clamp_min(1e-12))))


def make_pool(num_blocks, gen, k_values=None):
    """Fused FP8 indexer K pages ``[num_blocks, 64, 1, 132]`` plus their parts."""
    if k_values is None:
        k_values = torch.randn(num_blocks, BLOCK_KV, HEAD_DIM, device=_DEV, generator=gen)
    scale = _ue8m0(k_values.abs().amax(dim=-1) / 448.0)
    k_fp8 = (k_values / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    data = k_fp8.view(torch.uint8).reshape(num_blocks, BLOCK_KV * HEAD_DIM)
    scales = scale.float().contiguous().view(torch.uint8).reshape(num_blocks, BLOCK_KV * 4)
    fused = (
        torch.cat([data, scales], dim=1).contiguous().view(num_blocks, BLOCK_KV, 1, HEAD_DIM + 4)
    )
    return fused, k_fp8, scale.float()


def make_top_k():
    return TopK(
        TOP_K, decode_implementation=TopKImplementation.CUTE_DSL_GVR, gvr_self_sampling=True
    )


class Case:
    """B requests x NEXT_N verify rows over one layer's paged FP8 K cache."""

    def __init__(
        self,
        kv_lens,
        seed=0,
        max_seq_len=1 << 20,
        shuffle=True,
        q_scale=1.0,
        weight_sign=None,
        distinct_pages=None,
        block_table_cols=None,
        boost_from_token=None,
    ):
        """Build the case.

        Args:
            block_table_cols: Block-table width; defaults to
                ``ceil(max_seq_len / 64)``.
            boost_from_token: When set, the pages holding logical tokens at or
                beyond this position get 64x larger keys (so 64x larger
                per-token scales and identical FP8 values), which puts those
                tokens into the top-k when the weights are positive.
        """
        gen = torch.Generator(device=_DEV).manual_seed(seed)
        self.B = len(kv_lens)
        self.kv_lens_list = list(kv_lens)
        self.max_seq_len = max_seq_len
        blocks = [max(1, -(-n // BLOCK_KV)) for n in kv_lens]
        max_blocks = block_table_cols or -(-max_seq_len // BLOCK_KV)
        assert max(blocks) <= max_blocks
        num_blocks = (distinct_pages or sum(blocks)) + 4
        self.pool, self.k_fp8, self.k_scale = make_pool(num_blocks, gen)
        bt = torch.zeros(self.B, max_blocks, dtype=torch.int32, device=_DEV)
        base = 0
        for b, nb in enumerate(blocks):
            if distinct_pages:
                pages = torch.randint(0, distinct_pages, (nb,), device=_DEV, generator=gen)
            else:
                pages = torch.arange(base, base + nb, device=_DEV)
                if shuffle:
                    pages = pages[torch.randperm(nb, device=_DEV, generator=gen)]
                base += nb
            bt[b, :nb] = pages.to(torch.int32)
        self.block_table = bt
        if boost_from_token is not None:
            k_values = torch.randn(num_blocks, BLOCK_KV, HEAD_DIM, device=_DEV, generator=gen)
            first_page = boost_from_token // BLOCK_KV
            for b, nb in enumerate(blocks):
                if nb > first_page:
                    k_values[bt[b, first_page:nb].long()] *= 64.0
            self.pool, self.k_fp8, self.k_scale = make_pool(num_blocks, gen, k_values)
        self.kv_lens = torch.tensor(kv_lens, dtype=torch.int32, device=_DEV)
        q = torch.randn(self.B, NEXT_N, HEADS, HEAD_DIM, device=_DEV, generator=gen) * q_scale
        self.q = q.to(torch.float8_e4m3fn)
        w = torch.randn(self.B * NEXT_N, HEADS, device=_DEV, generator=gen)
        if weight_sign is not None:
            w = w.abs() * weight_sign
        self.weights = w.float()
        self.num_sms = get_num_sms()
        self.schedule = get_paged_mqa_logits_metadata(
            self.kv_lens.view(-1, 1), SCHEDULE_BLOCK_KV, self.num_sms
        )
        self.factor, self.atom = _pick_dsl_expand(
            NEXT_N,
            batch_size=self.B,
            max_ctx=max(kv_lens),
            num_sms=self.num_sms,
            kernel_atoms=FP8_ATOMS,
        )
        self.kv_lens_exp = self.kv_lens.repeat_interleave(self.factor)
        self.block_table_exp = self.block_table.repeat_interleave(self.factor, dim=0)
        self.schedule_exp = get_paged_mqa_logits_metadata(
            self.kv_lens_exp.view(-1, 1), SCHEDULE_BLOCK_KV, self.num_sms
        )
        rows = self.B * NEXT_N
        self.radix_idx = torch.empty(rows, 10, TOP_K, dtype=torch.int32, device=_DEV)
        self.radix_val = torch.empty(rows, 10, TOP_K, dtype=torch.float32, device=_DEV)

    @classmethod
    def from_fixture(cls, fixture, kv_len, shuffle, seed=0):
        """Real-weight case: the fixture's pages, queries and weights."""
        case = cls([kv_len], seed=seed)
        gen = torch.Generator(device=_DEV).manual_seed(seed)
        nb = -(-kv_len // BLOCK_KV)
        pages = fixture["pool"][:nb].to(_DEV)
        perm = torch.arange(nb, device=_DEV)
        if shuffle:
            perm = perm[torch.randperm(nb, device=_DEV, generator=gen)]
        pool = torch.zeros(nb + 4, BLOCK_KV, 1, HEAD_DIM + 4, dtype=torch.uint8, device=_DEV)
        pool[perm] = pages
        case.pool = pool
        flat = pool.view(nb + 4, -1)
        case.k_fp8 = flat[:, : BLOCK_KV * HEAD_DIM].contiguous().view(torch.float8_e4m3fn)
        case.k_fp8 = case.k_fp8.view(nb + 4, BLOCK_KV, HEAD_DIM)
        case.k_scale = flat[:, BLOCK_KV * HEAD_DIM :].contiguous().view(torch.float32)
        case.block_table.zero_()
        case.block_table[0, :nb] = perm.to(torch.int32)
        case.block_table_exp = case.block_table.repeat_interleave(case.factor, dim=0)
        case.q = fixture["q"][kv_len].to(_DEV).view(1, NEXT_N, HEADS, HEAD_DIM)
        case.weights = fixture["weights"][kv_len].to(_DEV).float()
        return case

    @property
    def row_lens(self):
        return [
            max(0, self.kv_lens_list[r // NEXT_N] - NEXT_N + r % NEXT_N + 1)
            for r in range(self.B * NEXT_N)
        ]

    def native_logits(self):
        """Native GEN DSL launch, including the wave-aware row-group split."""
        if self.factor > 1:
            q = self.q.reshape(self.B * self.factor, self.atom, HEADS, HEAD_DIM)
            return torch.ops.trtllm.cute_dsl_fp8_paged_mqa_logits(
                q,
                self.pool,
                self.weights,
                self.kv_lens_exp,
                self.block_table_exp,
                self.schedule_exp,
                self.max_seq_len,
            )
        return torch.ops.trtllm.cute_dsl_fp8_paged_mqa_logits(
            self.q,
            self.pool,
            self.weights,
            self.kv_lens,
            self.block_table,
            self.schedule,
            self.max_seq_len,
        )

    def indexer_topk_logits(self, config=None):
        return indexer_topk.paged_mqa_logits(
            self.q,
            self.pool,
            self.weights,
            self.kv_lens,
            self.block_table,
            self.schedule,
            self.max_seq_len,
            config=config,
        )

    def native_region(self, top_k, out):
        top_k(
            self.native_logits(),
            out,
            is_prefill=False,
            sequence_lengths=self.kv_lens,
            scan_lengths=self.kv_lens,
            row_kv_lens=None,
            next_n=NEXT_N,
            max_seq_len=self.max_seq_len,
            gvr_ext_kwargs=None,
            radix_aux_indices=self.radix_idx,
            radix_aux_logits=self.radix_val,
        )
        return out

    def indexer_topk_region(self, top_k, out, selector=None, config=None):
        return indexer_topk.decode_topk(
            q=self.q,
            weights=self.weights,
            k_cache=self.pool,
            block_table=self.block_table,
            context_lens=self.kv_lens,
            kv_lens=self.kv_lens,
            schedule_meta=self.schedule,
            max_seq_len=self.max_seq_len,
            top_k=top_k,
            out=out,
            radix_aux_indices=self.radix_idx,
            radix_aux_logits=self.radix_val,
            logits_config=config,
            selector=selector,
        )

    def new_out(self, fill=-7):
        return torch.full((self.B * NEXT_N, TOP_K), fill, dtype=torch.int32, device=_DEV)


def assert_logits_equal(case, native, cand):
    for r, n in enumerate(case.row_lens):
        if n > 0:
            assert torch.equal(native[r, :n], cand[r, :n]), f"row {r}: logits differ"


def assert_selection_contract(case, logits, sel):
    """Native the index_selection contract semantics, tie-aware (value multiset vs torch.topk)."""
    for r, n in enumerate(case.row_lens):
        idx = sel[r].long()
        if n <= 0:
            assert bool((idx == -1).all()), f"row {r}: empty row must be all -1"
        elif n <= TOP_K:
            assert torch.equal(torch.sort(idx[:n]).values, torch.arange(n, device=_DEV)), r
            assert bool((idx[n:] == -1).all()), f"row {r}: short row tail must be -1"
        else:
            assert int(idx.min()) >= 0 and int(idx.max()) < n, f"row {r}: index out of range"
            assert int(torch.unique(idx).numel()) == TOP_K, f"row {r}: duplicate index"
            ref = torch.topk(logits[r, :n], TOP_K).values
            got = torch.sort(logits[r].gather(0, idx) + 0.0, descending=True).values
            assert torch.equal(got, torch.sort(ref + 0.0, descending=True).values), (
                f"row {r}: selected value multiset is not the exact top-k"
            )


def reference_logits(case):
    """Independent fp32 oracle: scale_j * sum_h w_h * relu(q_h . k_j)."""
    k_all = case.k_fp8.float()
    out = torch.full(
        (case.B * NEXT_N, max(case.kv_lens_list)), float("nan"), device=_DEV, dtype=torch.float32
    )
    for b, n in enumerate(case.kv_lens_list):
        nb = -(-n // BLOCK_KV)
        pages = case.block_table[b, :nb].long()
        k = k_all[pages].reshape(-1, HEAD_DIM)[:n]
        s = case.k_scale[pages].reshape(-1)[:n]
        qk = torch.matmul(case.q[b].float().reshape(NEXT_N * HEADS, HEAD_DIM), k.t())
        qk = torch.relu(qk).view(NEXT_N, HEADS, n)
        w = case.weights[b * NEXT_N : (b + 1) * NEXT_N]
        out[b * NEXT_N : (b + 1) * NEXT_N, :n] = (w.unsqueeze(-1) * qk).sum(dim=1) * s
    return out


# --- logits ---------------------------------------------------------------

LENGTHS = [3, 6, 7, 2047, 2048, 2049, 2054, 32768, 131072, 491061, (1 << 20) - 1]


# None runs DEFAULT_CONFIG (rows, 8 register-cached weights), the kernel the
# adapter hook ships; the rowsplit variants are opt-in alternatives.
VARIANTS = {
    "default_rows_draft_loop": None,
    "rowsplit": LogitsConfig(variant="rowsplit"),
    "rowsplit_wg3": LogitsConfig(variant="rowsplit", num_math_wg=3),
}


def test_default_config_is_rows_draft_loop():
    assert DEFAULT_CONFIG == LogitsConfig(variant="rows", num_w_in_reg=8)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
@pytest.mark.parametrize("kv_len", LENGTHS)
def test_logits_bit_exact_vs_native(kv_len, variant):
    case = Case([kv_len], seed=kv_len)
    native = case.native_logits()[:, :kv_len].clone()
    cand = case.indexer_topk_logits(VARIANTS[variant])[:, :kv_len]
    assert_logits_equal(case, native, cand)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
@pytest.mark.parametrize(
    "kv_lens",
    [[4224, 64], [491061, 300], [7, 131072, 2049], [1, 2, 3, 4]],
    ids=lambda v: "-".join(map(str, v)),
)
def test_logits_bit_exact_multi_request(kv_lens, variant):
    # Odd and even tile counts per request and schedule splits that end at
    # request boundaries.
    case = Case([max(n, NEXT_N) for n in kv_lens], seed=sum(kv_lens))
    width = max(case.kv_lens_list)
    native = case.native_logits()[:, :width].clone()
    cand = case.indexer_topk_logits(VARIANTS[variant])[:, :width]
    assert_logits_equal(case, native, cand)


@pytest.mark.parametrize("kv_len", [2049, 32768, 131072])
def test_logits_vs_reference(kv_len):
    case = Case([kv_len], seed=7 * kv_len)
    cand = case.indexer_topk_logits()[:, :kv_len]
    ref = reference_logits(case)
    for r, n in enumerate(case.row_lens):
        # Same tolerances as the native CuTe-DSL FP8 paged-MQA logits test.
        torch.testing.assert_close(cand[r, :n], ref[r, :n], atol=5e-5, rtol=1e-5)


# --- selection --------------------------------------------------------------


SELECTORS = ["native", "radix"]


@pytest.mark.parametrize("selector", SELECTORS)
@pytest.mark.parametrize("kv_len", LENGTHS)
def test_selection_matches_native(kv_len, selector):
    case = Case([kv_len], seed=11 + kv_len)
    top_k = make_top_k()
    native = case.native_region(top_k, case.new_out()).clone()
    logits = case.native_logits()[:, :kv_len].clone()
    cand = case.indexer_topk_region(top_k, case.new_out(), selector=selector)
    assert_selection_contract(case, logits, cand)
    # Continuous random logits: the top-k set is unique, so any exact
    # selection engine must return the native set.
    assert torch.equal(torch.sort(native, dim=1).values, torch.sort(cand, dim=1).values)


@pytest.mark.parametrize("selector", SELECTORS)
@pytest.mark.parametrize(
    "kv_lens",
    [[30000, 64], [491061, 7, 2048, 2055], [131072, 131073]],
    ids=lambda v: "-".join(map(str, v)),
)
def test_selection_multi_request(kv_lens, selector):
    case = Case(kv_lens, seed=sum(kv_lens) + 1)
    top_k = make_top_k()
    width = max(kv_lens)
    logits = case.native_logits()[:, :width].clone()
    native = case.native_region(top_k, case.new_out()).clone()
    cand = case.indexer_topk_region(top_k, case.new_out(), selector=selector)
    assert_selection_contract(case, logits, cand)
    assert torch.equal(torch.sort(native, dim=1).values, torch.sort(cand, dim=1).values)


@pytest.mark.parametrize("selector", SELECTORS)
@pytest.mark.parametrize("kind", ["duplicate_pages", "all_zero", "negative_weights"])
def test_ties(kind, selector):
    kv_len = 40000
    if kind == "duplicate_pages":
        # 8 distinct pages repeated across the context: every logit value
        # occurs ~78 times, so the top-k boundary falls inside a tie class.
        case = Case([kv_len], seed=3, distinct_pages=8)
    elif kind == "all_zero":
        case = Case([kv_len], seed=4, q_scale=0.0)
    else:
        case = Case([kv_len], seed=5, weight_sign=-1.0)
    top_k = make_top_k()
    native = case.native_region(top_k, case.new_out()).clone()
    logits = case.native_logits()[:, :kv_len].clone()
    cand = case.indexer_topk_region(top_k, case.new_out(), selector=selector)
    assert_logits_equal(case, logits, case.indexer_topk_logits()[:, :kv_len])
    assert_selection_contract(case, logits, cand)
    # Characterize tie handling (all-zero logits overflow the radix
    # candidate buffer and exercise its full-row fallback).
    same_ids = torch.equal(torch.sort(native, dim=1).values, torch.sort(cand, dim=1).values)
    print(f"[ties:{kind}:{selector}] identical index sets vs native: {same_ids}")


def test_padded_batch_rows():
    # Request 1 stands in for a CUDA-graph padding request with a short cache.
    case = Case([30000, 64], seed=21)
    top_k = make_top_k()
    native = case.native_region(top_k, case.new_out()).clone()
    logits = case.native_logits()[:, :30000].clone()
    cand = case.indexer_topk_region(top_k, case.new_out())
    assert_logits_equal(case, logits, case.indexer_topk_logits()[:, :30000])
    assert_selection_contract(case, logits, cand)
    assert torch.equal(torch.sort(native, dim=1).values, torch.sort(cand, dim=1).values)


_FIXTURE_PATH = os.environ.get("GLM_MOE_DSA_INDEXER_TOPK_REAL_FIXTURE", "")


@pytest.mark.skipif(
    not os.path.isfile(_FIXTURE_PATH), reason="GLM_MOE_DSA_INDEXER_TOPK_REAL_FIXTURE not provided"
)
@pytest.mark.parametrize("shuffle", [False, True], ids=["contiguous", "shuffled"])
def test_real_weight_fixture(shuffle):
    fixture = torch.load(_FIXTURE_PATH, map_location="cpu", weights_only=False)
    print(f"[real] {fixture['label']} sha={fixture['token_ids_sha256'][:16]}")
    top_k = make_top_k()
    for kv_len in fixture["kv_lens"]:
        case = Case.from_fixture(fixture, kv_len, shuffle)
        logits = case.native_logits()[:, :kv_len].clone()
        assert_logits_equal(case, logits, case.indexer_topk_logits()[:, :kv_len])
        native = case.native_region(top_k, case.new_out()).clone()
        cand = case.indexer_topk_region(top_k, case.new_out())
        assert_selection_contract(case, logits, cand)
        assert torch.equal(torch.sort(native, dim=1).values, torch.sort(cand, dim=1).values)
        split = LogitsConfig(variant="rowsplit")
        assert_logits_equal(case, logits, case.indexer_topk_logits(split)[:, :kv_len])
        radix = case.indexer_topk_region(top_k, case.new_out(), selector="radix", config=split)
        assert_selection_contract(case, logits, radix)
        assert torch.equal(torch.sort(native, dim=1).values, torch.sort(radix, dim=1).values)
        if kv_len <= 131072:
            ref = reference_logits(case)
            for r, n in enumerate(case.row_lens):
                torch.testing.assert_close(logits[r, :n], ref[r, :n], atol=5e-5, rtol=1e-5)
        # Boundary sensitivity of the real distribution: gap between the k-th
        # and (k+1)-th largest logits, and overlap of adjacent verify rows.
        for r, n in enumerate(case.row_lens):
            vals = torch.topk(logits[r, :n], TOP_K + 1).values
            gap = (vals[TOP_K - 1] - vals[TOP_K]).item()
            overlap = (
                torch.isin(cand[r], cand[r - 1]).float().mean().item() if r > 0 else float("nan")
            )
            print(
                f"[real] kv={kv_len} row={r} kth={vals[TOP_K - 1].item():.6g} "
                f"gap={gap:.3g} overlap_prev_row={overlap:.4f}"
            )


# --- CUDA graph replay --------------------------------------------------------


GRAPH_CONFIGS = {"default_rows_draft_loop": None, "rowsplit": LogitsConfig(variant="rowsplit")}
GRAPH_STEPS = [20000, 20006, 20012, 35000, 9000, 65530]


def _graph_step_lens(kv_len, batch):
    # Request 0 follows the step length; later requests are shorter, down to
    # a 6-token request (verify rows of 1..6 keys, padding-like).
    return [max(NEXT_N, kv_len - 4099 * b) for b in range(batch)]


def _native_oracle(lens, max_seq_len, pool, k_fp8, k_scale, block_table, q, weights, kv_lens):
    """Eager native launch state for the given tensors (no new pool)."""
    batch = len(lens)
    ref = Case([NEXT_N] * batch, seed=0, max_seq_len=max_seq_len)
    ref.kv_lens_list = list(lens)
    ref.pool, ref.k_fp8, ref.k_scale = pool, k_fp8, k_scale
    ref.block_table, ref.q, ref.weights, ref.kv_lens = block_table, q, weights, kv_lens
    ref.schedule = get_paged_mqa_logits_metadata(kv_lens.view(-1, 1), 64, ref.num_sms)
    ref.factor, ref.atom = _pick_dsl_expand(
        NEXT_N, batch_size=batch, max_ctx=max(lens), num_sms=ref.num_sms, kernel_atoms=FP8_ATOMS
    )
    ref.kv_lens_exp = kv_lens.repeat_interleave(ref.factor)
    ref.block_table_exp = block_table.repeat_interleave(ref.factor, dim=0)
    ref.schedule_exp = get_paged_mqa_logits_metadata(ref.kv_lens_exp.view(-1, 1), 64, ref.num_sms)
    return ref


def _run_graph_replay(batch, logits_config, selector, max_seq_len, steps, block_table_cols=None):
    """Capture indexer_topk once, then replay with new lengths, pages, tables and queries.

    After every replay the indexer_topk logits (read from the graph-stable arena the
    captured launch wrote) must be bit-exact to an eager native launch on the
    same state, and the selection must match the native region.
    """
    case = Case(
        _graph_step_lens(steps[0], batch),
        seed=31,
        max_seq_len=max_seq_len,
        block_table_cols=block_table_cols,
    )
    # Persistent pool sized for the largest step; pages are rewritten between
    # replays to model page reuse by later requests.
    total_blocks = batch * -(-max_seq_len // BLOCK_KV) + 4
    gen = torch.Generator(device=_DEV).manual_seed(99 + batch)
    case.pool, case.k_fp8, case.k_scale = make_pool(total_blocks, gen)
    top_k = make_top_k()
    out = case.new_out()
    q, weights = case.q.clone(), case.weights.clone()
    kv_lens, block_table = case.kv_lens.clone(), case.block_table.clone()
    schedule, pool = case.schedule.clone(), case.pool
    arena = {}
    indexer_topk_logits_fn = indexer_topk.paged_mqa_logits

    def recording_logits(*args, **kwargs):
        arena["logits"] = indexer_topk_logits_fn(*args, **kwargs)
        return arena["logits"]

    def indexer_topk_call():
        return indexer_topk.decode_topk(
            q=q,
            weights=weights,
            k_cache=pool,
            block_table=block_table,
            context_lens=kv_lens,
            kv_lens=kv_lens,
            schedule_meta=schedule,
            max_seq_len=max_seq_len,
            top_k=top_k,
            out=out,
            radix_aux_indices=case.radix_idx,
            radix_aux_logits=case.radix_val,
            logits_config=logits_config,
            selector=selector,
        )

    with patch.object(indexer_topk, "paged_mqa_logits", side_effect=recording_logits):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            indexer_topk_call()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            indexer_topk_call()
    captured_logits = arena["logits"]
    assert captured_logits.shape == (batch * NEXT_N, max_seq_len)

    for step, kv_len in enumerate(steps):
        lens = _graph_step_lens(kv_len, batch)
        new_pool, new_k, new_s = make_pool(total_blocks, gen)
        pool.copy_(new_pool)
        del new_pool
        perm = torch.randperm(total_blocks, device=_DEV, generator=gen).to(torch.int32)
        block_table.zero_()
        base = 0
        for b, n in enumerate(lens):
            nb = -(-n // BLOCK_KV)
            block_table[b, :nb] = perm[base : base + nb]
            base += nb
        kv_lens.copy_(torch.tensor(lens, dtype=torch.int32))
        schedule.copy_(get_paged_mqa_logits_metadata(kv_lens.view(-1, 1), 64, case.num_sms))
        q.copy_((torch.randn(q.shape, device=_DEV, generator=gen)).to(torch.float8_e4m3fn))
        weights.copy_(torch.randn(weights.shape, device=_DEV, generator=gen))
        out.fill_(-7)
        graph.replay()
        torch.cuda.synchronize()
        ref = _native_oracle(
            lens, max_seq_len, pool, new_k, new_s, block_table, q, weights, kv_lens
        )
        width = max(lens)
        logits = ref.native_logits()[:, :width].clone()
        assert_logits_equal(ref, logits, captured_logits[:, :width])
        native = ref.native_region(make_top_k(), ref.new_out())
        assert_selection_contract(ref, logits, out)
        assert torch.equal(torch.sort(native, dim=1).values, torch.sort(out, dim=1).values), step


@pytest.mark.parametrize("selector", SELECTORS)
@pytest.mark.parametrize("config_name", sorted(GRAPH_CONFIGS))
@pytest.mark.parametrize("batch", [1, 4])
def test_cuda_graph_replay_growing_context_and_page_reuse(batch, config_name, selector):
    _run_graph_replay(batch, GRAPH_CONFIGS[config_name], selector, 65536, GRAPH_STEPS)


# --- served indexer max_seq_len (2**20 + 14) --------------------------------------

# Served GEN: net max_seq_len 2**20, MTP K = 5, overlap scheduler on. The
# indexer max_seq_len is 2**20 + 5 + 4 + 5 and KV cache manager V2 sizes the
# block table at round_up(ceil((2**20 + 14 + 10) / 64), 4) columns; manager V1
# would use ceil((2**20 + 14) / 64) columns.
SERVED_MAX_SEQ_LEN = (1 << 20) + 14
V2_BLOCK_TABLE_COLS = 16388
V1_BLOCK_TABLE_COLS = 16385
# Live indexer lengths around 2**20 up to the served bound (target layers
# reach 2**20 + 10 with the overlap scheduler; the bound itself is included).
BOUND_LENGTHS = [
    (1 << 20) - 6,
    (1 << 20) - 1,
    1 << 20,
    (1 << 20) + 1,
    (1 << 20) + 5,
    (1 << 20) + 10,
    (1 << 20) + 13,
    SERVED_MAX_SEQ_LEN,
]


def _bound_case(kv_lens, seed, **kwargs):
    return Case(
        kv_lens,
        seed=seed,
        max_seq_len=SERVED_MAX_SEQ_LEN,
        block_table_cols=V2_BLOCK_TABLE_COLS,
        **kwargs,
    )


def test_policy_envelope_is_the_served_bound():
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.indexer_topk import (  # noqa: E501
        policy,
    )

    assert policy.MAX_INDEXER_SEQ_LEN == SERVED_MAX_SEQ_LEN
    assert V2_BLOCK_TABLE_COLS % 2 == 0 and V1_BLOCK_TABLE_COLS % 2 == 1


@pytest.mark.parametrize("kv_len", BOUND_LENGTHS)
def test_served_bound_logits_and_selection(kv_len):
    case = _bound_case([kv_len], seed=kv_len % 9973)
    native = case.native_logits()[:, :kv_len].clone()
    cand = case.indexer_topk_logits()
    assert cand.shape == (NEXT_N, SERVED_MAX_SEQ_LEN)
    assert cand.stride(0) % 256 == 0 and cand.stride(0) >= SERVED_MAX_SEQ_LEN
    assert_logits_equal(case, native, cand[:, :kv_len])
    top_k = make_top_k()
    native_sel = case.native_region(top_k, case.new_out()).clone()
    sel = case.indexer_topk_region(top_k, case.new_out())
    assert_selection_contract(case, native, sel)
    assert torch.equal(torch.sort(native_sel, dim=1).values, torch.sort(sel, dim=1).values)


@pytest.mark.parametrize("kv_len", [(1 << 20) + 1, (1 << 20) + 10, SERVED_MAX_SEQ_LEN])
def test_served_bound_selects_ids_beyond_2pow20(kv_len):
    """Keys at the tail dominate, so IDs >= 2**20 must be selected exactly."""
    boost_from = (1 << 20) - 64
    case = _bound_case([kv_len], seed=kv_len % 7919, weight_sign=1.0, boost_from_token=boost_from)
    native = case.native_logits()[:, :kv_len].clone()
    assert_logits_equal(case, native, case.indexer_topk_logits()[:, :kv_len])
    top_k = make_top_k()
    native_sel = case.native_region(top_k, case.new_out()).clone()
    sel = case.indexer_topk_region(top_k, case.new_out())
    assert_selection_contract(case, native, sel)
    assert torch.equal(torch.sort(native_sel, dim=1).values, torch.sort(sel, dim=1).values)
    beyond = 0
    for r, n in enumerate(case.row_lens):
        tail = torch.arange(boost_from, n, device=_DEV, dtype=torch.int32)
        assert bool(torch.isin(tail, sel[r]).all()), f"row {r}: boosted tail not selected"
        beyond += int((sel[r] >= (1 << 20)).sum())
    assert beyond > 0, "no ID >= 2**20 was selected"


@pytest.mark.parametrize(
    "kv_lens",
    [
        [SERVED_MAX_SEQ_LEN, (1 << 20) + 1],
        [SERVED_MAX_SEQ_LEN, (1 << 20) + 10, 1 << 20, (1 << 20) - 1],
        [(1 << 20) + 7, 491061, NEXT_N, 2049],
    ],
    ids=lambda v: "-".join(map(str, v)),
)
def test_served_bound_multi_request(kv_lens):
    case = _bound_case(kv_lens, seed=sum(kv_lens) % 10007)
    width = max(kv_lens)
    native = case.native_logits()[:, :width].clone()
    assert_logits_equal(case, native, case.indexer_topk_logits()[:, :width])
    top_k = make_top_k()
    native_sel = case.native_region(top_k, case.new_out()).clone()
    sel = case.indexer_topk_region(top_k, case.new_out())
    assert_selection_contract(case, native, sel)
    assert torch.equal(torch.sort(native_sel, dim=1).values, torch.sort(sel, dim=1).values)


def test_served_bound_vs_reference():
    case = _bound_case([SERVED_MAX_SEQ_LEN], seed=17)
    cand = case.indexer_topk_logits()[:, :SERVED_MAX_SEQ_LEN]
    ref = reference_logits(case)
    for r, n in enumerate(case.row_lens):
        torch.testing.assert_close(cand[r, :n], ref[r, :n], atol=5e-5, rtol=1e-5)


@pytest.mark.parametrize("launch", ["indexer_topk", "native"])
@pytest.mark.parametrize(
    "cols", [V2_BLOCK_TABLE_COLS, V1_BLOCK_TABLE_COLS], ids=["v2_cols", "v1_cols"]
)
def test_served_bound_single_launch(launch, cols):
    """One logits launch at the bound, checked against the fp32 oracle.

    Each parameter runs exactly one kind of kernel launch so the test can run
    under compute-sanitizer memcheck (with PYTORCH_NO_CUDA_MEMORY_CACHING=1)
    one parameter at a time. With the odd V1 width the last 128-token tile's
    second page ID is one entry past the row; the policy declines that width.
    """
    case = Case(
        [SERVED_MAX_SEQ_LEN], seed=23, max_seq_len=SERVED_MAX_SEQ_LEN, block_table_cols=cols
    )
    logits = case.indexer_topk_logits() if launch == "indexer_topk" else case.native_logits()
    torch.cuda.synchronize()
    ref = reference_logits(case)
    for r, n in enumerate(case.row_lens):
        torch.testing.assert_close(logits[r, :n], ref[r, :n], atol=5e-5, rtol=1e-5)


SERVED_GRAPH_STEPS = [
    (1 << 20) - 600,
    1 << 20,
    (1 << 20) + 1,
    SERVED_MAX_SEQ_LEN,
    491061,
    (1 << 20) + 10,
]


@pytest.mark.parametrize("batch", [1, 4])
def test_cuda_graph_replay_across_2pow20(batch):
    """Shipped kernel + native top-k, captured at the served max_seq_len."""
    _run_graph_replay(
        batch,
        None,
        "native",
        SERVED_MAX_SEQ_LEN,
        SERVED_GRAPH_STEPS,
        block_table_cols=V2_BLOCK_TABLE_COLS,
    )


# --- adapter hook ---------------------------------------------------------------


def _hook_objects(case, **overrides):
    top_k = make_top_k()
    indexer = SimpleNamespace(
        use_cute_dsl_paged_mqa_logits=True,
        use_fp4=False,
        use_gvr_emission=False,
        top_k=top_k,
        n_heads=HEADS,
        head_dim=HEAD_DIM,
        layer_idx=0,
    )
    metadata = SimpleNamespace(
        num_contexts=0,
        num_generations=case.B,
        seq_lens=torch.full((case.B,), NEXT_N, dtype=torch.int32),
        is_ragged_verify=False,
        in_mtp_draft_loop=False,
        gen_indexer_kv_lens_cuda_runtime=case.kv_lens,
        kv_lens_cuda_runtime=case.kv_lens,
        use_expanded_buffers_for_mtp=False,
        mapping=SimpleNamespace(enable_attention_dp=False),
        kv_cache_manager=SimpleNamespace(get_indexer_k_cache_buffers=lambda idx: case.pool),
        indexer_k_cache_block_offsets=case.block_table,
        scheduler_metadata_buffer=case.schedule,
        get_indexer_max_seq_len=lambda: case.max_seq_len,
        radix_aux_indices=case.radix_idx,
        radix_aux_logits=case.radix_val,
    )
    for key, value in overrides.items():
        target = indexer if hasattr(indexer, key) else metadata
        setattr(target, key, value)
    return indexer, metadata


HOOK_LENS = [50000, 4224, 64, 131072]


@pytest.mark.parametrize("batch", [1, 2, 3, 4])
def test_adapter_hook_engages_and_matches_native(batch):
    case = Case(HOOK_LENS[:batch], seed=41 + batch)
    indexer, metadata = _hook_objects(case)
    rows = batch * NEXT_N
    q_tok = case.q.reshape(rows, HEADS, HEAD_DIM)
    out = case.new_out()
    assert indexer_topk.try_indexer_decode(indexer, metadata, q_tok, case.weights, 0, rows, out)
    native = case.native_region(make_top_k(), case.new_out())
    assert_selection_contract(case, case.native_logits()[:, : max(HOOK_LENS[:batch])], out)
    assert torch.equal(torch.sort(native, dim=1).values, torch.sort(out, dim=1).values)


@pytest.mark.parametrize("batch", [1, 4])
def test_adapter_hook_engages_at_served_bound(batch):
    """Served GEN values: indexer max_seq_len 2**20 + 14, V2 table width."""
    lens = [SERVED_MAX_SEQ_LEN, (1 << 20) + 1, 1 << 20, (1 << 20) - 1][:batch]
    case = _bound_case(lens, seed=61 + batch)
    indexer, metadata = _hook_objects(case)
    rows = batch * NEXT_N
    q_tok = case.q.reshape(rows, HEADS, HEAD_DIM)
    out = case.new_out()
    engaged_before = indexer_topk.DECISION_COUNTS.get("engaged", 0)
    assert indexer_topk.try_indexer_decode(indexer, metadata, q_tok, case.weights, 0, rows, out)
    assert indexer_topk.DECISION_COUNTS["engaged"] == engaged_before + 1
    logits = case.native_logits()[:, : max(lens)].clone()
    native = case.native_region(make_top_k(), case.new_out())
    assert_selection_contract(case, logits, out)
    assert torch.equal(torch.sort(native, dim=1).values, torch.sort(out, dim=1).values)


# Each case: indexer/metadata overrides plus the call shape. Every one must be
# declined by the policy before any device work.
_I32 = torch.int32
FALLBACK_CASES = {
    "fp4_keys": dict(overrides={"use_fp4": True}),
    "non_dsl_native_logits": dict(overrides={"use_cute_dsl_paged_mqa_logits": False}),
    "gvr_emission": dict(overrides={"use_gvr_emission": True}),
    "ragged_verify": dict(overrides={"is_ragged_verify": True}),
    "expanded_mtp_buffers": dict(overrides={"use_expanded_buffers_for_mtp": True}),
    "mtp_draft_step": dict(overrides={"in_mtp_draft_loop": True}),
    "attention_dp": dict(overrides={"mapping": SimpleNamespace(enable_attention_dp=True)}),
    "radix_topk_engine": dict(radix_topk=True),
    "sm100": dict(sm=100),
    "page16": dict(page=16),
    "page32": dict(page=32),
    "page128": dict(page=128),
    "non_uniform_seq_lens": dict(overrides={"seq_lens": torch.tensor([4], dtype=_I32)}),
    "next_n4": dict(overrides={"seq_lens": torch.tensor([4], dtype=_I32)}, next_n=4),
    # Constructor-invalid for the rows kernel (8 cached weights > 4 heads).
    "heads4_next_n4": dict(
        overrides={"n_heads": 4, "seq_lens": torch.tensor([4], dtype=_I32)}, heads=4, next_n=4
    ),
    "batch5": dict(
        overrides={"num_generations": 5, "seq_lens": torch.full((5,), NEXT_N, dtype=_I32)},
        batch=5,
    ),
    "mixed_context_batch": dict(
        overrides={"num_contexts": 1, "seq_lens": torch.tensor([100, NEXT_N], dtype=_I32)}
    ),
    "max_seq_len_2m": dict(overrides={"get_indexer_max_seq_len": lambda: 1 << 21}),
    "max_seq_len_above_served_bound": dict(
        overrides={"get_indexer_max_seq_len": lambda: SERVED_MAX_SEQ_LEN + 1}
    ),
    # KV cache manager V1 at the served max_seq_len: odd block-table width.
    "block_table_v1_odd_width": dict(
        overrides={"get_indexer_max_seq_len": lambda: SERVED_MAX_SEQ_LEN},
        block_table_cols=V1_BLOCK_TABLE_COLS,
    ),
    "bf16_weights": dict(weights_dtype=torch.bfloat16),
    "non_contiguous_query": dict(non_contiguous_q=True),
}


@pytest.mark.parametrize("name", sorted(FALLBACK_CASES))
def test_adapter_hook_falls_back_before_side_effects(name):
    spec = FALLBACK_CASES[name]
    case = Case([50000], seed=43)
    overrides = dict(spec.get("overrides", {}))
    if "page" in spec:
        view = case.pool.view(-1, spec["page"], 1, HEAD_DIM + 4)
        overrides["kv_cache_manager"] = SimpleNamespace(
            get_indexer_k_cache_buffers=lambda idx: view
        )
    if spec.get("radix_topk"):
        overrides["top_k"] = TopK(TOP_K, decode_implementation=TopKImplementation.CUDA_RADIX)
    if "block_table_cols" in spec:
        overrides["indexer_k_cache_block_offsets"] = torch.zeros(
            1, spec["block_table_cols"], dtype=torch.int32, device=_DEV
        )
    indexer, metadata = _hook_objects(case, **overrides)
    heads = spec.get("heads", HEADS)
    rows = spec.get("next_n", NEXT_N) * spec.get("batch", 1)
    q_wide = torch.zeros(rows, heads, 2 * HEAD_DIM, device=_DEV).to(torch.float8_e4m3fn)
    q_tok = q_wide[..., :HEAD_DIM]
    if not spec.get("non_contiguous_q"):
        q_tok = q_tok.contiguous()
    weights = torch.zeros(rows, heads, device=_DEV, dtype=spec.get("weights_dtype", torch.float32))
    out = torch.full((rows, TOP_K), -7, dtype=torch.int32, device=_DEV)
    torch.cuda.synchronize()
    decisions_before = sum(indexer_topk.DECISION_COUNTS.values())
    engaged_before = indexer_topk.DECISION_COUNTS.get("engaged", 0)
    with patch.object(
        indexer_topk, "get_sm_version", return_value=spec.get("sm", get_sm_version())
    ):
        with patch.object(indexer_topk, "decode_topk") as launched:
            assert not indexer_topk.try_indexer_decode(
                indexer, metadata, q_tok, weights, 0, rows, out
            )
    torch.cuda.synchronize()
    assert not launched.called, "indexer_topk must decline before building or launching a kernel"
    assert bool((out == -7).all()), "fallback must leave the output untouched"
    assert sum(indexer_topk.DECISION_COUNTS.values()) == decisions_before + 1
    assert indexer_topk.DECISION_COUNTS.get("engaged", 0) == engaged_before


def test_adapter_hook_temporal_gvr_prior_falls_back():
    case = Case([50000], seed=45)
    indexer, metadata = _hook_objects(case)
    indexer.top_k = TopK(
        TOP_K, decode_implementation=TopKImplementation.CUTE_DSL_GVR, gvr_self_sampling=False
    )
    out = case.new_out(fill=-7)
    q_tok = case.q.reshape(NEXT_N, HEADS, HEAD_DIM)
    assert not indexer_topk.try_indexer_decode(
        indexer, metadata, q_tok, case.weights, 0, NEXT_N, out
    )
    assert bool((out == -7).all())


# --- the target's indexer against the built-in Indexer ------------------------------


def _make_indexer(module):
    from tensorrt_llm._torch.attention.backends.interface import (
        PositionalEmbeddingParams,
        RopeParams,
    )
    from tensorrt_llm.functional import PositionEmbeddingType
    from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig

    sparse_params = DeepSeekSparseAttentionConfig(
        index_head_dim=HEAD_DIM,
        index_n_heads=HEADS,
        index_topk=TOP_K,
        use_cute_dsl_topk=True,
        use_cute_dsl_paged_mqa_logits=True,
        enable_heuristic_topk=True,
        index_share_for_mtp_iteration=True,
    ).to_sparse_params()
    pos = PositionalEmbeddingParams(
        type=PositionEmbeddingType.rope_gpt_neox,
        rope=RopeParams(dim=64, theta=10000.0, max_positions=4096),
        is_neox=True,
    )
    mla = SimpleNamespace(hidden_size=6144, q_lora_rank=2048, qk_rope_head_dim=64)
    with patch.object(module, "RotaryEmbedding", return_value=Mock()):
        return module.Indexer(
            quant_config=None,
            pos_embd_params=pos,
            mla_params=mla,
            skip_create_weights_in_init=True,
            sparse_params=sparse_params,
            dtype=torch.bfloat16,
            layer_idx=0,
        )


def _decode_metadata(case, in_mtp_draft_loop, num_accepted=None):
    rows = case.B * NEXT_N
    return SimpleNamespace(
        kv_cache_manager=SimpleNamespace(
            quant_block_size=128,
            get_indexer_k_cache_buffers=lambda idx: case.pool,
            layer_offsets={0: 0},
        ),
        num_contexts=0,
        num_generations=case.B,
        num_ctx_tokens=0,
        num_tokens=rows,
        get_empty=lambda buffers, shape, cache_name, dtype, capture_graph: torch.empty(
            shape, dtype=dtype, device=_DEV
        ),
        cuda_graph_buffers=None,
        is_cuda_graph=False,
        skip_indexer_for_ctx_reqs=False,
        skip_indexer_for_gen_reqs=False,
        gvr_prior_indices=None,
        in_mtp_draft_loop=in_mtp_draft_loop,
        indexer_skip_topk=False,
        shared_topk_indices=None,
        mtp_num_accepted=num_accepted,
        seq_lens=torch.full((case.B,), NEXT_N, dtype=torch.int32),
        is_ragged_verify=False,
        use_expanded_buffers_for_mtp=False,
        kv_lens_cuda_2d=case.kv_lens.view(-1, 1).expand(-1, NEXT_N).contiguous(),
        indexer_k_cache_block_offsets=case.block_table,
        scheduler_metadata_buffer=case.schedule,
        scheduler_metadata_buffer_full_next_n=case.schedule,
        gen_indexer_kv_lens_cuda_runtime=case.kv_lens,
        get_indexer_max_seq_len=lambda: case.max_seq_len,
        dsl_expand_factor=case.factor,
        dsl_atom=case.atom,
        kv_lens_expanded_cuda=case.kv_lens_exp,
        block_table_expanded=case.block_table_exp,
        scheduler_metadata_buffer_expanded=case.schedule_exp,
        kv_lens_row_reorder=None,
        kv_lens_cuda_runtime=case.kv_lens,
        ragged_row_kv_lens=lambda n: None,
        radix_aux_indices=case.radix_idx,
        radix_aux_logits=case.radix_val,
        _tokens_per_block=BLOCK_KV,
    )


@pytest.mark.parametrize("in_mtp_draft_loop", [False, True], ids=["target", "draft_loop"])
def test_adopted_indexer_preserves_native_indexer_semantics(monkeypatch, in_mtp_draft_loop):
    """Stock projections -> the target's ``GlmIndexer.sparse_attn_indexer`` in a decode step
    (indexer_topk on) == the built-in indexer.

    Covers the target forward (indexer_topk engaged) and an MTP draft step (indexer_topk declines;
    the native path computes the selection and stashes each request's last
    accepted row for the reuse steps).
    """
    case = Case([60000, 45000], seed=51)
    rows = case.B * NEXT_N
    q_tok = case.q.reshape(rows, HEADS, HEAD_DIM)
    k_dummy = torch.zeros(rows, HEAD_DIM, device=_DEV).to(torch.float8_e4m3fn)
    k_scale_dummy = torch.ones(rows, 1, device=_DEV)
    hidden = torch.zeros(rows, 6144, device=_DEV, dtype=torch.bfloat16)
    accepted = torch.tensor([3, 6], dtype=torch.int32, device=_DEV)

    native_indexer = _make_indexer(native_indexer_mod)
    meta_native = _decode_metadata(case, in_mtp_draft_loop, accepted)
    native = native_indexer.sparse_attn_indexer(
        meta_native, hidden, q_tok, k_dummy, k_scale_dummy, case.weights, is_generation=True
    ).clone()

    with monkeypatch.context() as mp:
        plan_testlib.select(dispatch, "indexer_topk", mp)
        indexer_topk_indexer = GlmIndexer.adopt(_make_indexer(native_indexer_mod))
        meta_indexer_topk = _decode_metadata(case, in_mtp_draft_loop, accepted)
        with patch.object(
            indexer_topk, "decode_topk", wraps=indexer_topk.decode_topk
        ) as indexer_topk_launch:
            with dispatch.step(dispatch.Phase.DECODE):
                got = indexer_topk_indexer.sparse_attn_indexer(
                    meta_indexer_topk,
                    hidden,
                    q_tok,
                    k_dummy,
                    k_scale_dummy,
                    case.weights,
                    is_generation=True,
                )

    # indexer_topk engages for the target forward and declines in the MTP draft step.
    assert indexer_topk_launch.call_count == (0 if in_mtp_draft_loop else 1)
    assert torch.equal(torch.sort(native, dim=1).values, torch.sort(got, dim=1).values)
    if in_mtp_draft_loop:
        assert meta_native.shared_topk_indices is not None
        # The native engine's output order within a row varies run to run.
        assert torch.equal(
            torch.sort(meta_native.shared_topk_indices, dim=1).values,
            torch.sort(meta_indexer_topk.shared_topk_indices, dim=1).values,
        )
    else:
        assert meta_indexer_topk.shared_topk_indices is None


def test_without_a_plan_indexer_topk_is_off(monkeypatch):
    monkeypatch.setattr(dispatch, "_INSTALLED", [None])
    assert not dispatch.is_enabled("indexer_topk")
