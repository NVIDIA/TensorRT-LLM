# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for the GLM-5.2 tp16 mla_kv_half_append entry.

The paged state is real: a DSACacheManagerV2 over GLM-5.2's latent rows (FP8, 576 per token,
64-token pages) with several layers, the attention metadata of one generation step after a
context, and the MLA rope table at GLM-5.2's rope_theta. The rows the op appends are read back
where main's `convert_req_index_to_global` places their positions -- the rows main's built-in
sparse MLA decode reads -- and gated against the entry's reference; no other byte of the pool
may change.
"""

import zlib

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_107__tp16.ops.attention.mla_kv_half_append import (  # noqa: E501
    KV_LORA_RANK,
    LATENT_OFFSET,
    ROPE_DIM,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_107__tp16.ops.attention.mla_kv_half_append import (  # noqa: E501
    mla_kv_half_append as op,
)
from tensorrt_llm._torch.attention.backends.interface import RopeParams
from tensorrt_llm._torch.attention.backends.sparse.dsa.cache_manager import DSACacheManagerV2
from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttentionMetadata
from tensorrt_llm._torch.metadata import KVCacheParams
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import DeepSeekSparseAttentionConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping

__extra_import_path__ = ["../../../.."]
from _validating import certifying, validating  # noqa: E402,F401

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or get_sm_version() != 107,
    reason="the GLM-5.2 tp16 target's ops are certified on sm_107",
)

HEAD_DIM = KV_LORA_RANK + ROPE_DIM
MAX_SEQ_LEN = 8192
#: One E4M3 step: 2**-3 relative at the bottom of a binade, 2**-9 among the subnormals.
_E4M3_STEP_RTOL = 2**-3
_E4M3_SUBNORMAL_STEP = 2**-9
#: How much of an appended row may miss bit-exactness. See `_compare`.
_MAX_INEXACT_FRACTION = 1e-3


def _compare(got: torch.Tensor, want: torch.Tensor) -> None:
    """One E4M3 step, and at most 0.1 % of the bytes off bit-exactness.

    Kernel and reference evaluate the same fp32 expressions and round once; they differ in the
    order of the 512-term sum of squares, in rsqrtf against torch.rsqrt (2 ulp), and in FMA
    contraction of the rotation -- about 1e-7 relative, which crosses an E4M3 rounding boundary
    (3 mantissa bits) for about one value in a million, and then by one step. A reference that
    rounds the normalized latent and the rotated K to BF16 first (mla_rope_generation's KV
    half) misses bit-exactness on about 3 % of the bytes, so the floor tells the two apart; a
    wrong weight, epsilon, pair layout or position moves most bytes by several steps.
    """
    assert got.shape == want.shape, (got.shape, want.shape)
    torch.testing.assert_close(
        got.float(), want.float(), rtol=_E4M3_STEP_RTOL, atol=_E4M3_SUBNORMAL_STEP
    )
    inexact = int((got.view(torch.uint8) != want.view(torch.uint8)).sum())
    allowed = max(1, int(_MAX_INEXACT_FRACTION * got.numel()))
    assert inexact <= allowed, f"{inexact}/{got.numel()} bytes not bit-exact (allowed {allowed})"


def _fill_e4m3_(dst: torch.Tensor, gen: torch.Generator) -> None:
    """Fill a contiguous uint8 view with E4M3 values of N(0, 1)."""
    flat = dst.reshape(-1)
    chunk = 1 << 26
    for start in range(0, flat.numel(), chunk):
        stop = min(start + chunk, flat.numel())
        values = torch.randn(stop - start, generator=gen, device=flat.device)
        flat[start:stop].copy_(values.to(torch.float8_e4m3fn).view(torch.uint8))


class _DsaEnv:
    """A DSACacheManagerV2 over GLM-5.2's latent rows and one generation step's metadata."""

    def __init__(self, num_layers: int, tokens_per_block: int, rope_theta: float) -> None:
        self.tokens_per_block = tokens_per_block
        self.max_batch_size = 2
        self.manager = DSACacheManagerV2(
            kv_cache_config=KvCacheConfig(
                max_tokens=self.max_batch_size * (MAX_SEQ_LEN + tokens_per_block),
                enable_block_reuse=False,
            ),
            kv_cache_type=CacheType.SELFKONLY,
            num_layers=num_layers,
            num_kv_heads=1,
            head_dim=HEAD_DIM,
            tokens_per_block=tokens_per_block,
            max_seq_len=MAX_SEQ_LEN,
            max_batch_size=self.max_batch_size,
            mapping=Mapping(world_size=1, tp_size=1, rank=0),
            dtype=DataType.FP8,
            sparse_attention_config=DeepSeekSparseAttentionConfig(
                index_head_dim=128, index_n_heads=32, index_topk=2048
            ),
        )
        rope = RopeParams(
            dim=ROPE_DIM, theta=rope_theta, max_positions=MAX_SEQ_LEN, duplicate_data=True
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

    def pool_rows(self) -> torch.Tensor:
        """The pool as [rows, 576] bytes, the rows convert_req_index_to_global counts."""
        pool = self.manager.get_unique_primary_pool()
        return pool.view(torch.uint8).reshape(-1, HEAD_DIM)

    def global_rows(self, kv_lens: list, positions: torch.Tensor, seq_len_q: int, layer: int):
        """Main's pool rows of each generation token's position in `layer`."""
        tpb = self.tokens_per_block
        slots = self.manager.get_pool_block_indices(
            len(kv_lens), request_ids=list(range(len(kv_lens))), num_contexts=0
        )
        blocks = torch.tensor([-(-n // tpb) for n in kv_lens])
        width = int(blocks.max())
        table = slots[:, :width].to(torch.int32).clone()
        table[torch.arange(width)[None, :] >= blocks[:, None]] = -1
        page_index_scale, layer_offset = self.manager.get_primary_pool_page_index_params(layer)
        req_idx = (torch.arange(positions.shape[0], device="cuda") // seq_len_q).to(torch.int32)
        rows = torch.ops.trtllm.convert_req_index_to_global(
            req_idx,
            table.cuda(),
            positions.to(torch.int32)[:, None].contiguous(),
            tpb,
            1,
            page_index_scale * tpb,
            layer_offset,
        )
        return rows[:, 0].long()

    def shutdown(self) -> None:
        self.manager.shutdown()


def _step_args(env: _DsaEnv, spec: dict, gen: torch.Generator) -> tuple:
    """One generation step's call: the metadata, a fused-A output and the scale buffers."""
    cached = list(spec["cached"])
    seq_len_q = spec["seq_len_q"]
    metadata = env.generation_step(cached, seq_len_q)
    num_tokens = len(cached) * seq_len_q
    fused_a = torch.randn(num_tokens, spec["latent_width"], generator=gen, device="cuda").to(
        torch.bfloat16
    )
    kv_norm_weight = (1.0 + 0.1 * torch.randn(KV_LORA_RANK, generator=gen, device="cuda")).to(
        torch.bfloat16
    )
    return (
        fused_a[:, LATENT_OFFSET : LATENT_OFFSET + HEAD_DIM],
        kv_norm_weight,
        spec["kv_norm_eps"],
        env.rotary_cos_sin,
        metadata.kv_lens_cuda_runtime,
        0,
        seq_len_q,
        metadata.kv_cache_block_offsets,
        env.manager.kv_cache_pool_pointers,
        env.manager.kv_cache_pool_mapping,
        spec["layer_idx"],
        spec["tokens_per_block"],
        env.manager.max_seq_len,
        torch.full((2,), float("nan"), dtype=torch.float32, device="cuda"),
        torch.full((1,), float("nan"), dtype=torch.float32, device="cuda"),
        spec["softmax_scale"],
        spec["trigger_at_entry"],
    )


@pytest.mark.parametrize("cell", op.CELLS, ids=[c.why[:44] for c in op.CELLS])
def test_certified_cells(cell, certifying) -> None:  # noqa: F811 — pytest resolves the fixture by name
    certifying(op)
    spec = cell.spec
    seed = zlib.crc32(cell.why.encode()) % 2**31
    gen = torch.Generator(device="cuda").manual_seed(seed)
    env = _DsaEnv(spec["num_layers"], spec["tokens_per_block"], spec["rope_theta"])
    try:
        args = _step_args(env, spec, gen)
        latent, bmm1, bmm2 = args[0], args[13], args[14]
        pool_rows = env.pool_rows()
        _fill_e4m3_(pool_rows, gen)
        pool_before = pool_rows.clone()
        latent_before = latent.clone()

        op(*args)
        torch.cuda.synchronize()
        want_rows, positions, want_bmm1, want_bmm2 = op.reference(*args)

        # The position of each row: this step's tokens are the last seq_len_q of each sequence.
        seq_len_q = spec["seq_len_q"]
        kv_lens = [c + seq_len_q for c in spec["cached"]]
        assert positions.tolist() == [n - seq_len_q + t for n in kv_lens for t in range(seq_len_q)]

        # Each row lands where main's layout puts its position in this layer.
        rows = env.global_rows(kv_lens, positions, seq_len_q, spec["layer_idx"])
        assert rows.unique().numel() == rows.numel() and bool((rows >= 0).all())
        _compare(pool_rows[rows].view(torch.float8_e4m3fn), want_rows)

        # And nothing else moves: not another layer, slot or position, not the latent.
        changed = (pool_rows != pool_before).any(dim=-1).nonzero().flatten()
        assert set(changed.tolist()) <= set(rows.tolist()), "the op wrote outside its rows"
        assert torch.equal(latent, latent_before)

        # The attention scales, bit for bit: one fp32 product.
        assert torch.equal(bmm1, want_bmm1), (bmm1, want_bmm1)
        assert torch.equal(bmm2, want_bmm2), (bmm2, want_bmm2)

        # Rerunning the prepared step rewrites the same bytes.
        appended = pool_rows[rows].clone()
        op(*args)
        torch.cuda.synchronize()
        assert torch.equal(pool_rows[rows], appended)
    finally:
        env.shutdown()


def _guard_args(**overrides) -> tuple:
    """The first cell's call on a fresh environment, with arguments replaced by position."""
    spec = op.CELLS[0].spec
    gen = torch.Generator(device="cuda").manual_seed(0)
    env = _DsaEnv(spec["num_layers"], spec["tokens_per_block"], spec["rope_theta"])
    args = list(_step_args(env, spec, gen))
    names = ("sequence_length", "rotary_cos_sin", "kv_cache_block_offsets")
    positions = {"sequence_length": 4, "rotary_cos_sin": 3, "kv_cache_block_offsets": 7}
    for name, value in overrides.items():
        assert name in names
        args[positions[name]] = value(args[positions[name]])
    return env, tuple(args)


def test_guard_refuses_a_sequence_shorter_than_its_step() -> None:
    """Rows at negative positions are skipped: those tokens never reach the cache."""
    env, args = _guard_args(sequence_length=lambda t: torch.full_like(t, 5))
    try:
        with pytest.raises(AssertionError, match="unwritten"):
            with validating(op):
                op(*args)
    finally:
        env.shutdown()


def test_guard_refuses_a_rope_table_shorter_than_the_positions() -> None:
    """The table is read at the position without a bound."""
    env, args = _guard_args(rotary_cos_sin=lambda t: t.reshape(-1)[: 100 * 2 * ROPE_DIM])
    try:
        with pytest.raises(AssertionError, match="reads past it"):
            with validating(op):
                op(*args)
    finally:
        env.shutdown()


def test_guard_refuses_block_offsets_narrower_than_the_positions() -> None:
    """A block past the offsets' width is read from the next sequence's offsets."""
    env, args = _guard_args(kv_cache_block_offsets=lambda t: t[..., :8].contiguous())
    try:
        with pytest.raises(AssertionError, match="needs more"):
            with validating(op):
                op(*args)
    finally:
        env.shutdown()
