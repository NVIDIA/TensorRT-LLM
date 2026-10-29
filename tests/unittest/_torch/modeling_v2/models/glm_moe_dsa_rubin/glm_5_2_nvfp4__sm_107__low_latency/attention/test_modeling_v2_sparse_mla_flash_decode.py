# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 tp16 sparse_mla_flash_decode entry.

The cells build GLM-5.2's TP16 pool geometry directly: FP8 latent rows in 79 layers' 64-token
pages per pool slot, a block table that maps each sequence's blocks to shuffled slots (-1 past
the sequence), and top-k rows laid out as the DSA indexer writes them -- the valid entries
packed in front, this step's visible tokens among them, -1 behind. Every cell is driven through
the entry's guard and gated against the entry's reference.

Two further tests: a captured CUDA graph replays the shared-layer cell bit for bit, and on a
real DSACacheManagerV2 the KV append writes this step's rows where main's
`convert_req_index_to_global` says they live and the flash decode's in-kernel remap reads
exactly those rows.
"""

import math
import zlib
from typing import NamedTuple

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.catalog._op import true_fp32_matmul
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa_rubin.glm_5_2_nvfp4__sm_107__low_latency.ops.attention.mla_kv_half_append import (  # noqa: E501
    LATENT_OFFSET,
    mla_kv_half_append,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa_rubin.glm_5_2_nvfp4__sm_107__low_latency.ops.attention.sparse_mla_flash_decode import (  # noqa: E501
    GLM_SOFTMAX_SCALE,
    HEAD_DIM_QK,
    KV_LORA_RANK,
    LATENT_NUM_HEADS,
    MAX_TOP_K,
    NUM_HEADS,
    V_HEAD_DIM,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa_rubin.glm_5_2_nvfp4__sm_107__low_latency.ops.attention.sparse_mla_flash_decode import (  # noqa: E501
    sparse_mla_flash_decode as op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa_rubin.glm_5_2_nvfp4__sm_107__low_latency.ops.attention.sparse_mla_flash_decode import (  # noqa: E501
    sparse_mla_flash_decode_latent as latent_op,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa_rubin.glm_5_2_nvfp4__sm_107__low_latency.ops.gemm._low_m_splitk import (  # noqa: E501
    bmm_prewait_tactic,
    run_splitk_bmm,
)
from tensorrt_llm._torch.attention.backends.interface import RopeParams
from tensorrt_llm._torch.attention.backends.sparse.dsa.cache_manager import DSACacheManagerV2
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping

__extra_import_path__ = ["../../../.."]
from _validating import assert_within_ulp, certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107,
    reason="the GLM-5.2 tp16 target's ops are certified on sm_107",
)

#: log2(e) as an fp32 constant, as the KV append forms mla_bmm1_scale[1].
_LOG2E = 1.4426950408889634
#: Pool slots no sequence maps, and block-table columns past every sequence.
_SPARE_SLOTS = 5
_SPARE_COLUMNS = 3
#: Output rows past the query tokens, which the op must leave alone.
_SPARE_ROWS = 2
#: Magnitudes of the random operands: KV rows ~ N(0, 1) as kv_a_layernorm leaves them, the
#: absorbed query ~ N(0, 0.25), v_b ~ N(0, 1 / 512).
_Q_STD = 0.5
_VB_STD = 1.0 / math.sqrt(KV_LORA_RANK)
#: The relative L2 distance allowed between the reference and an attention whose
#: probabilities are not rounded to E4M3. E4M3 keeps 3 mantissa bits: a rounded probability is
#: off by up to 2**-4 relative (RMS 1.8-3.6 % across a binade), and on outputs that average
#: zero-mean rows the error stays at that relative size: the kernel measured 2.3 % against an
#: fp32 reference, and the bound is 4x that. A reference that mis-masks a slot,
#: drops a split or misses a scale is off by tens of percent.
_E4M3_P_REL_L2 = 0.1


def _compare(out: torch.Tensor, ref: torch.Tensor) -> None:
    """4 ulp of each token row's largest output, 1 ulp relative RMS, in BF16.

    The reference states the kernel's three roundings (P to E4M3 per 128-slot split, the merged
    attention output to BF16, the v_b product to BF16), so what is left is fp32 accumulation
    order: the QK and P.V sums inside mma.sync, the 16-way merge, exp2f, and the v_b sum. Those
    move a value by about 1e-6 relative; that flips an E4M3 probability with probability about
    1e-5 per element (one flip in a cell of 64k probabilities at most), and one flipped
    probability moves the outputs of its head by under 1 % of their magnitude (2 ulp of the
    element, under 1 ulp of the row's largest); the final BF16 roundings flip by one ulp at
    the same rate. A mirror with a wrong merge weight, a wrong split boundary or a dropped
    head is off by many percent.
    """
    assert_within_ulp(out, ref, element_ulp=4.0, rms_ulp=1.0)


def _fill_e4m3_(dst: torch.Tensor, gen: torch.Generator, std: float = 1.0) -> None:
    """Fill a contiguous uint8 view with E4M3 values of N(0, std^2) (finite: far below 448)."""
    flat = dst.reshape(-1)
    chunk = 1 << 26
    for start in range(0, flat.numel(), chunk):
        stop = min(start + chunk, flat.numel())
        values = torch.randn(stop - start, generator=gen, device=flat.device) * std
        flat[start:stop].copy_(values.to(torch.float8_e4m3fn).view(torch.uint8))


def _select(kv_lens, seq_len_q: int, top_k: int, holes: bool, gen: torch.Generator) -> torch.Tensor:
    """[T, top_k] int32 top-k positions as the DSA indexer lays them out.

    Row t of sequence s selects min(own + 1, top_k) positions out of 0 .. own (own = its
    position): every token of this step it can see, the rest drawn from the earlier steps',
    shuffled, then -1. With `holes`, about 1/64 of the selected slots become -1 in place.
    """
    rows = []
    for kv_len in kv_lens:
        first_new = kv_len - seq_len_q
        for t in range(seq_len_q):
            own = first_new + t
            count = min(own + 1, top_k)
            fresh = torch.arange(first_new, own + 1, dtype=torch.int64)
            num_old = count - fresh.numel()
            old = torch.randperm(first_new, generator=gen)[:num_old]
            chosen = torch.cat((fresh, old))
            chosen = chosen[torch.randperm(chosen.numel(), generator=gen)]
            if holes:
                drop = torch.randperm(count, generator=gen)[: max(1, count // 64)]
                chosen[drop] = -1
            row = torch.full((top_k,), -1, dtype=torch.int32)
            row[:count] = chosen.to(torch.int32)
            rows.append(row)
    return torch.stack(rows)


def _global_rows(
    positions: torch.Tensor,
    table: torch.Tensor,
    seq_len_q: int,
    tokens_per_block: int,
    page_index_scale: int,
    layer_offset: int,
) -> torch.Tensor:
    """The pool rows of `positions` with convert_req_index_to_global's arithmetic (-1 kept)."""
    num_tokens = positions.shape[0]
    seq = torch.arange(num_tokens, device=positions.device) // seq_len_q
    position = positions.long()
    block = position.clamp(min=0) // tokens_per_block
    page = table[seq[:, None], block.clamp(max=table.shape[1] - 1)].long()
    rows = (
        page * page_index_scale * tokens_per_block
        + layer_offset * tokens_per_block
        + position.clamp(min=0) % tokens_per_block
    )
    return torch.where((position >= 0) & (page >= 0), rows, -1).to(torch.int32)


class _Case(NamedTuple):
    kwargs: dict
    num_tokens: int
    global_rows: torch.Tensor


def _build(spec: dict, seed: int) -> _Case:
    """Turn a cell spec into the op's arguments, over a pool of the spec's geometry."""
    gen_cpu = torch.Generator().manual_seed(seed)
    gen = torch.Generator(device="cuda").manual_seed(seed)
    heads = spec.get("heads", NUM_HEADS)
    tokens_per_block = spec["tokens_per_block"]
    num_layers = spec["num_layers"]
    layer_offset = spec["layer_offset"]
    seq_len_q = spec["seq_len_q"]
    top_k = spec["top_k"]
    kv_lens = list(spec["kv_lens"])
    num_tokens = len(kv_lens) * seq_len_q

    blocks = [-(-kv_len // tokens_per_block) for kv_len in kv_lens]
    num_slots = sum(blocks) + _SPARE_SLOTS
    order = torch.randperm(num_slots, generator=gen_cpu).to(torch.int32)
    table = torch.full((len(kv_lens), max(blocks) + _SPARE_COLUMNS), -1, dtype=torch.int32)
    start = 0
    for s, num_blocks in enumerate(blocks):
        table[s, :num_blocks] = order[start : start + num_blocks]
        start += num_blocks

    pool = torch.empty(
        num_slots, num_layers, tokens_per_block * HEAD_DIM_QK, dtype=torch.uint8, device="cuda"
    )
    _fill_e4m3_(pool, gen)
    positions = _select(kv_lens, seq_len_q, top_k, spec["holes"], gen_cpu)
    global_rows = _global_rows(
        positions, table, seq_len_q, tokens_per_block, num_layers, layer_offset
    )

    scale = torch.tensor(GLM_SOFTMAX_SCALE, dtype=torch.float32)
    kwargs = dict(
        q=(torch.randn(num_tokens, heads, HEAD_DIM_QK, generator=gen, device="cuda") * _Q_STD).to(
            torch.float8_e4m3fn
        ),
        kv_pool=pool.view(torch.float8_e4m3fn),
        indices=(positions if spec["remap"] else global_rows).cuda(),
        seq_lens_kv=torch.tensor(kv_lens, dtype=torch.int32, device="cuda"),
        mla_bmm1_scale=torch.stack((scale, scale * torch.tensor(_LOG2E))).cuda(),
        mla_bmm2_scale=torch.ones(1, dtype=torch.float32, device="cuda"),
        v_b_proj=(
            torch.randn(heads, V_HEAD_DIM, KV_LORA_RANK, generator=gen, device="cuda") * _VB_STD
        ).to(torch.bfloat16),
        output=torch.full(
            (num_tokens + _SPARE_ROWS, heads * V_HEAD_DIM),
            float("nan"),
            dtype=torch.bfloat16,
            device="cuda",
        ),
        seq_len_q=seq_len_q,
        top_k=top_k,
    )
    if spec["remap"]:
        kwargs.update(
            block_table=table.cuda(),
            tokens_per_block=tokens_per_block,
            page_index_scale=num_layers,
            layer_offset=layer_offset,
            gather_before_wait=spec["gather_before_wait"],
        )
    return _Case(kwargs=kwargs, num_tokens=num_tokens, global_rows=global_rows.cuda())


def _unrounded_attention(case: _Case) -> torch.Tensor:
    """[T, 1024] fp32: softmax attention over the selected rows without any intermediate
    rounding, times v_b. An independent statement of what the op approximates."""
    kw = case.kwargs
    num_tokens, seq_len_q, top_k = case.num_tokens, kw["seq_len_q"], kw["top_k"]
    rows = case.global_rows.long()
    token = torch.arange(num_tokens, device="cuda")
    kv_len = kw["seq_lens_kv"][token // seq_len_q].long()
    num_valid = torch.clamp(kv_len - (seq_len_q - 1 - token % seq_len_q), max=top_k)
    valid = (rows >= 0) & (torch.arange(top_k, device="cuda")[None, :] < num_valid[:, None])
    pool_rows = kw["kv_pool"].view(torch.uint8).reshape(-1, HEAD_DIM_QK)
    keys = pool_rows[rows.clamp(min=0)].view(torch.float8_e4m3fn).float().double()
    queries = kw["q"].float().double()
    scores = torch.einsum("thd,tjd->thj", queries, keys) * GLM_SOFTMAX_SCALE
    scores = scores.masked_fill(~valid[:, None, :], float("-inf"))
    weights = torch.nan_to_num(torch.softmax(scores, dim=-1), nan=0.0)
    attn = torch.einsum("thj,tjd->thd", weights, keys[..., :KV_LORA_RANK])
    out = torch.einsum("thd,hvd->thv", attn, kw["v_b_proj"].double())
    return out.reshape(num_tokens, kw["q"].shape[1] * V_HEAD_DIM).float()


def _seed(why: str) -> int:
    return zlib.crc32(why.encode()) % 2**31


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by name
    certifying(op)
    case = _build(cell.spec, _seed(cell.why))
    output = case.kwargs["output"]
    op(**case.kwargs)
    torch.cuda.synchronize()
    ref = op.reference(**case.kwargs)

    # The entry's reference is attention: within the E4M3-probability band of an unrounded
    # one. Checked before the kernel gate, so a mirror that drifted from attention fails as
    # such rather than as a kernel mismatch.
    unrounded = _unrounded_attention(case)
    rel_l2 = ((ref.float() - unrounded).norm() / unrounded.norm()).item()
    assert rel_l2 <= _E4M3_P_REL_L2, f"reference is {rel_l2:.3%} from unrounded attention"

    _compare(output[: case.num_tokens], ref)
    untouched = output[case.num_tokens :].float()
    assert torch.isnan(untouched).all(), "rows past the tokens were written"


@pytest.mark.parametrize(
    "heads,seq_len_q,kv_lens",
    [(4, 8, (20011,)), (8, 4, (9000, 17003)), (8, 8, (20011,))],
    ids=("tp16-b1-q8", "tp8-b2-q4", "tp8-b1-q8"),
)
def test_capture_and_replay_match_the_eager_call(
    heads: int, seq_len_q: int, kv_lens: tuple[int, ...]
) -> None:
    """Replay changed Q, selected indices and KV into a strided output without touching canaries."""
    spec = dict(op.CELLS[1].spec, heads=heads, seq_len_q=seq_len_q, kv_lens=kv_lens)
    case = _build(spec, 7391 + heads)
    columns = heads * V_HEAD_DIM
    storage = torch.full(
        (case.num_tokens + _SPARE_ROWS, columns + 16),
        float("nan"),
        dtype=torch.bfloat16,
        device="cuda",
    )
    output = storage[:, :columns]
    case.kwargs["output"] = output
    op(**case.kwargs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        with torch.cuda.graph(graph, stream=stream):
            op(**case.kwargs)
    torch.cuda.current_stream().wait_stream(stream)
    gen = torch.Generator(device="cuda").manual_seed(9843 + heads)
    gen_cpu = torch.Generator().manual_seed(9843 + heads)
    for _ in range(3):
        _fill_e4m3_(case.kwargs["q"].view(torch.uint8), gen)
        _fill_e4m3_(case.kwargs["kv_pool"].view(torch.uint8), gen)
        case.kwargs["indices"].copy_(
            _select(kv_lens, seq_len_q, spec["top_k"], spec["holes"], gen_cpu)
        )
        storage.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        replay = output[: case.num_tokens].clone()
        assert torch.isnan(storage[case.num_tokens :]).all(), "spare rows were written"
        assert torch.isnan(storage[:, columns:]).all(), "spare columns were written"
        output.fill_(float("nan"))
        op(**case.kwargs)
        torch.cuda.synchronize()
        assert torch.equal(output[: case.num_tokens], replay)


def _guard_case(cell_index: int) -> _Case:
    cell = op.CELLS[cell_index]
    return _build(cell.spec, _seed(cell.why))


def test_guard_refuses_a_selection_past_the_valid_count() -> None:
    """A selected token behind the row's valid count is never read."""
    case = _guard_case(4)  # short context: rows hold fewer valid slots than the top-k
    seq_len_q = case.kwargs["seq_len_q"]
    first_unread = int(case.kwargs["seq_lens_kv"][0]) - (seq_len_q - 1)  # row 0's valid count
    indices = case.kwargs["indices"]
    assert first_unread < indices.shape[1]
    indices[0, first_unread] = 0
    with pytest.raises(AssertionError, match="past the row's valid count"):
        with validating(op):
            op(**case.kwargs)


def test_guard_refuses_a_position_past_the_token_itself() -> None:
    """A position after the query token's own reads a row it may not see."""
    case = _guard_case(0)
    kv_len = int(case.kwargs["seq_lens_kv"][0])
    case.kwargs["indices"][0, 0] = kv_len - 1  # row 0 sits at kv_len - 8
    with pytest.raises(AssertionError, match="past the query token's own"):
        with validating(op):
            op(**case.kwargs)


def test_guard_refuses_a_table_narrower_than_the_kv() -> None:
    """A table cut short masks live tokens instead of failing."""
    case = _guard_case(0)
    kv_len = int(case.kwargs["seq_lens_kv"][0])
    needed = -(-kv_len // case.kwargs["tokens_per_block"])
    case.kwargs["block_table"] = case.kwargs["block_table"][:, : needed - 1]
    with pytest.raises(AssertionError, match="columns; a sequence needs"):
        with validating(op):
            op(**case.kwargs)


def test_guard_refuses_a_hole_in_the_table() -> None:
    """A -1 inside a sequence's blocks masks its tokens instead of failing."""
    case = _guard_case(0)
    case.kwargs["block_table"][0, 3] = -1
    with pytest.raises(AssertionError, match="-1 inside"):
        with validating(op):
            op(**case.kwargs)


def test_guard_refuses_a_layer_offset_outside_the_slot() -> None:
    """An offset at the page scale reads the next slot's first layer."""
    case = _guard_case(0)
    case.kwargs["layer_offset"] = case.kwargs["page_index_scale"]
    with pytest.raises(AssertionError, match="below page_index_scale"):
        with validating(op):
            op(**case.kwargs)


def test_guard_refuses_a_global_row_past_the_pool() -> None:
    """Global rows are read as given."""
    case = _guard_case(3)  # draft steps 1-6: global rows
    assert "block_table" not in case.kwargs
    num_rows = case.kwargs["kv_pool"].numel() // HEAD_DIM_QK
    case.kwargs["indices"][0, 0] = num_rows
    with pytest.raises(AssertionError, match="past the pool"):
        with validating(op):
            op(**case.kwargs)


class _DsaEnv:
    """A DSACacheManagerV2 over GLM-5.2's latent rows (FP8, 576 per token, 64-token pages)."""

    TOKENS_PER_BLOCK = 64

    def __init__(self, num_layers: int, max_batch_size: int, max_seq_len: int) -> None:
        self.max_batch_size = max_batch_size
        self.manager = DSACacheManagerV2(
            kv_cache_config=KvCacheConfig(
                max_tokens=max_batch_size * (max_seq_len + self.TOKENS_PER_BLOCK),
                enable_block_reuse=False,
            ),
            kv_cache_type=CacheType.SELFKONLY,
            num_layers=num_layers,
            num_kv_heads=1,
            head_dim=HEAD_DIM_QK,
            tokens_per_block=self.TOKENS_PER_BLOCK,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            dtype=DataType.FP8,
            sparse_attention_config=DeepSeekSparseAttentionConfig(
                index_head_dim=128, index_n_heads=32, index_topk=MAX_TOP_K
            ),
        )
        rope = RopeParams(
            dim=HEAD_DIM_QK - KV_LORA_RANK,
            theta=8_000_000.0,
            max_positions=max_seq_len,
            duplicate_data=True,
        )
        _, self.rotary_cos_sin = rope.create_rope_const_params()

    def generation_step(self, cached: list, seq_len_q: int) -> TrtllmAttentionMetadata:
        """Requests 0 .. n-1 with `cached` tokens each, grown by one step of `seq_len_q`."""
        request_ids = list(range(len(cached)))
        self.manager.add_dummy_requests(request_ids, token_nums=cached)
        for rid in request_ids:
            cache = self.manager.kv_cache_map[rid]
            assert cache.resize(cache.capacity + seq_len_q)
        metadata = TrtllmAttentionMetadata(
            max_num_requests=self.max_batch_size,
            max_num_tokens=8192,
            kv_cache_manager=self.manager,
        )
        metadata.seq_lens = torch.tensor([seq_len_q] * len(cached), dtype=torch.int)
        metadata.num_contexts = 0
        metadata.request_ids = request_ids
        metadata.prompt_lens = cached
        metadata.kv_cache_params = KVCacheParams(use_cache=True, num_cached_tokens_per_seq=cached)
        metadata.prepare()
        return metadata

    def block_table(self, kv_lens: list) -> torch.Tensor:
        """The DSA metadata's generation block table: pool slots, -1 past each sequence."""
        slots = self.manager.get_pool_block_indices(
            len(kv_lens), request_ids=list(range(len(kv_lens))), num_contexts=0
        )
        blocks = torch.tensor([-(-n // self.TOKENS_PER_BLOCK) for n in kv_lens])
        width = int(blocks.max())
        table = slots[:, :width].to(torch.int32).clone()
        table[torch.arange(width)[None, :] >= blocks[:, None]] = -1
        return table.cuda()

    def shutdown(self) -> None:
        self.manager.shutdown()


def _convert(
    block_table: torch.Tensor, positions: torch.Tensor, seq_len_q: int, page_params, tpb: int
) -> torch.Tensor:
    """Main's pool rows of `positions` ([T, k] int32): convert_req_index_to_global."""
    page_index_scale, layer_offset = page_params
    num_tokens = positions.shape[0]
    req_idx = (torch.arange(num_tokens, device="cuda") // seq_len_q).to(torch.int32)
    return torch.ops.trtllm.convert_req_index_to_global(
        req_idx,
        block_table,
        positions,
        tpb,
        positions.shape[1],
        page_index_scale * tpb,
        layer_offset,
    )


@pytest.mark.parametrize("heads", (4, 8), ids=("tp16-b1-q8", "tp8-b1-q8"))
def test_reads_the_rows_the_kv_append_writes_in_the_dsa_pool(heads: int) -> None:
    """Both ops against main's DSA cache layout.

    On a DSACacheManagerV2 pool, glm_mla_kv_half_append appends this step's 8 rows of one layer
    through the attention metadata's block offsets; they must land where main's
    convert_req_index_to_global places those positions. The flash decode then remaps the step's
    top-k positions in the kernel, with and without the early gather, and must read exactly the
    rows convert_req_index_to_global names: its output is gated against the reference run on
    those global rows.
    """
    seq_len_q, layer = 8, 2
    cached = [5_000]
    kv_lens = [c + seq_len_q for c in cached]
    num_tokens = len(cached) * seq_len_q
    gen_cpu = torch.Generator().manual_seed(1)
    gen = torch.Generator(device="cuda").manual_seed(1)
    env = _DsaEnv(num_layers=4, max_batch_size=2, max_seq_len=8192)
    try:
        metadata = env.generation_step(cached, seq_len_q)
        tpb = env.TOKENS_PER_BLOCK
        pool = env.manager.get_unique_primary_pool()
        pool_rows = pool.view(torch.uint8).reshape(-1, HEAD_DIM_QK)
        _fill_e4m3_(pool_rows, gen)
        block_table = env.block_table(kv_lens)
        page_params = env.manager.get_primary_pool_page_index_params(layer)

        fused_a = torch.randn(num_tokens, 2624, generator=gen, device="cuda").to(torch.bfloat16)
        latent = fused_a[:, LATENT_OFFSET : LATENT_OFFSET + HEAD_DIM_QK]
        kv_norm_weight = (1.0 + 0.1 * torch.randn(KV_LORA_RANK, generator=gen, device="cuda")).to(
            torch.bfloat16
        )
        bmm1 = torch.full((2,), float("nan"), dtype=torch.float32, device="cuda")
        bmm2 = torch.full((1,), float("nan"), dtype=torch.float32, device="cuda")
        append_args = (
            latent,
            kv_norm_weight,
            1e-5,
            env.rotary_cos_sin,
            metadata.kv_lens_cuda_runtime,
            0,
            seq_len_q,
            metadata.kv_cache_block_offsets,
            env.manager.kv_cache_pool_pointers,
            env.manager.kv_cache_pool_mapping,
            layer,
            tpb,
            env.manager.max_seq_len,
            bmm1,
            bmm2,
            GLM_SOFTMAX_SCALE,
            True,
        )
        with validating(mla_kv_half_append):
            mla_kv_half_append(*append_args)
        torch.cuda.synchronize()
        want_rows, positions, want_bmm1, want_bmm2 = mla_kv_half_append.reference(*append_args)
        fresh = _convert(
            block_table, positions.to(torch.int32)[:, None], seq_len_q, page_params, tpb
        )
        got_rows = pool_rows[fresh[:, 0].long()].view(torch.float8_e4m3fn)
        torch.testing.assert_close(got_rows.float(), want_rows.float(), rtol=2**-3, atol=2**-9)
        assert torch.equal(bmm1, want_bmm1) and torch.equal(bmm2, want_bmm2)

        indices = _select(kv_lens, seq_len_q, MAX_TOP_K, False, gen_cpu).cuda()
        global_rows = _convert(block_table, indices, seq_len_q, page_params, tpb)
        q = (torch.randn(num_tokens, heads, HEAD_DIM_QK, generator=gen, device="cuda") * _Q_STD).to(
            torch.float8_e4m3fn
        )
        v_b_proj = (
            torch.randn(heads, V_HEAD_DIM, KV_LORA_RANK, generator=gen, device="cuda") * _VB_STD
        ).to(torch.bfloat16)
        output = torch.empty(num_tokens, heads * V_HEAD_DIM, dtype=torch.bfloat16, device="cuda")
        ref = op.reference(
            q,
            pool,
            global_rows,
            metadata.kv_lens_cuda_runtime,
            bmm1,
            bmm2,
            v_b_proj,
            output,
            seq_len_q,
            MAX_TOP_K,
        )
        for gather_before_wait in (False, True):
            output.fill_(float("nan"))
            with validating(op):
                op(
                    q,
                    pool,
                    indices,
                    metadata.kv_lens_cuda_runtime,
                    bmm1,
                    bmm2,
                    v_b_proj,
                    output,
                    seq_len_q,
                    MAX_TOP_K,
                    block_table,
                    tpb,
                    page_params[0],
                    page_params[1],
                    gather_before_wait,
                )
            torch.cuda.synchronize()
            _compare(output, ref)
    finally:
        env.shutdown()


def _latent_case(spec, seed):
    case = _build(spec, seed)
    gen = torch.Generator(device="cuda").manual_seed(seed + 1)
    case.kwargs["q"] = (
        torch.randn(case.num_tokens, LATENT_NUM_HEADS, HEAD_DIM_QK, generator=gen, device="cuda")
        * _Q_STD
    ).to(torch.float8_e4m3fn)
    del case.kwargs["v_b_proj"]
    case.kwargs["output"] = torch.full(
        (case.num_tokens + _SPARE_ROWS, LATENT_NUM_HEADS, KV_LORA_RANK),
        float("nan"),
        dtype=torch.bfloat16,
        device="cuda",
    )
    return case


@pytest.mark.parametrize("cell", latent_op.CELLS, ids=[c.why[:44] for c in latent_op.CELLS])
@pytest.mark.parametrize("early_trigger", [False, True])
def test_latent_certified_cells(cell, early_trigger, certifying) -> None:  # noqa: F811
    certifying(latent_op)
    case = _latent_case(cell.spec, _seed(cell.why))
    case.kwargs["trigger_after_wait"] = early_trigger
    output = latent_op(**case.kwargs)
    torch.cuda.synchronize()
    _compare(output[: case.num_tokens], latent_op.reference(**case.kwargs))
    assert torch.isnan(output[case.num_tokens :].float()).all()


@pytest.mark.parametrize("tokens", [6, 12, 18])
def test_latent_cuda_graph_replay(tokens) -> None:
    cell = next(
        c
        for c in latent_op.CELLS
        if c.spec["seq_len_q"] == 6 and 6 * len(c.spec["kv_lens"]) == tokens
    )
    case = _latent_case(cell.spec, 9100 + tokens)
    output = latent_op(**case.kwargs)
    torch.cuda.synchronize()
    eager = output[:tokens].clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        latent_op(**case.kwargs)
    for _ in range(3):
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(output[:tokens], eager)
        assert torch.isnan(output[tokens:].float()).all()


def test_latent_refuses_more_than_twenty_two_queries() -> None:
    spec = dict(latent_op.CELLS[-1].spec, seq_len_q=23)
    case = _latent_case(spec, 9201)
    with pytest.raises(RuntimeError, match="at most 22"):
        latent_op(**case.kwargs)


def test_latent_output_abi_is_explicit() -> None:
    case = _latent_case(latent_op.CELLS[0].spec, 9202)
    case.kwargs["output"] = torch.empty(
        case.num_tokens, LATENT_NUM_HEADS * V_HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    with pytest.raises(RuntimeError, match="16, 512"):
        latent_op(**case.kwargs)


def test_latent_padded_output_stride() -> None:
    case = _latent_case(latent_op.CELLS[0].spec, 9203)
    storage = torch.full(
        (case.num_tokens + _SPARE_ROWS, LATENT_NUM_HEADS * KV_LORA_RANK + 8),
        float("nan"),
        dtype=torch.bfloat16,
        device="cuda",
    )
    case.kwargs["output"] = storage[:, : LATENT_NUM_HEADS * KV_LORA_RANK].view(
        case.num_tokens + _SPARE_ROWS, LATENT_NUM_HEADS, KV_LORA_RANK
    )
    latent_op(**case.kwargs)
    torch.cuda.synchronize()
    _compare(case.kwargs["output"][: case.num_tokens], latent_op.reference(**case.kwargs))
    assert torch.isnan(storage[:, -8:].float()).all()
    assert torch.isnan(storage[case.num_tokens :].float()).all()


@pytest.mark.parametrize("tokens", [1, 3, 6, 12, 18])
def test_latent_early_trigger_with_prewait_v_b(tokens) -> None:
    """The consumer preloads weights, waits for latent output, and writes token-major v_b rows."""
    cell = next(
        c for c in latent_op.CELLS if c.spec["seq_len_q"] * len(c.spec["kv_lens"]) == tokens
    )
    case = _latent_case(cell.spec, 9300 + tokens)
    case.kwargs["trigger_after_wait"] = True
    gen = torch.Generator(device="cuda").manual_seed(9400 + tokens)
    weights = (
        torch.randn(LATENT_NUM_HEADS, V_HEAD_DIM, KV_LORA_RANK, generator=gen, device="cuda")
        * _VB_STD
    ).to(torch.bfloat16)
    projected = torch.empty(
        tokens, LATENT_NUM_HEADS, V_HEAD_DIM, dtype=torch.bfloat16, device="cuda"
    )
    tactic = bmm_prewait_tactic(tokens, V_HEAD_DIM, KV_LORA_RANK)
    assert tactic is not None

    def step():
        latent_op(**case.kwargs)
        run_splitk_bmm(
            case.kwargs["output"][:tokens].transpose(0, 1),
            weights,
            projected.transpose(0, 1),
            get_env_enable_pdl(),
            tactic,
            reserve_full_smem=False,
        )

    def reference():
        latent = latent_op.reference(**case.kwargs)
        with true_fp32_matmul():
            return torch.einsum("thd,hvd->thv", latent.float(), weights.float()).to(torch.bfloat16)

    step()  # Compile/load both kernels before capture.
    torch.cuda.synchronize()
    _compare(projected, reference())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    q = case.kwargs["q"]
    for seed in (9500 + tokens, 9600 + tokens):
        gen.manual_seed(seed)
        q.copy_((torch.randn(q.shape, generator=gen, device="cuda") * _Q_STD).to(q.dtype))
        case.kwargs["output"].fill_(float("nan"))
        projected.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        _compare(projected, reference())
        replay = projected.clone()
        step()
        torch.cuda.synchronize()
        assert torch.equal(projected, replay)
