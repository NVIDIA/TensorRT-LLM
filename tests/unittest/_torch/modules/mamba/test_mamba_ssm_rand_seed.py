# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Mamba SSM stochastic-rounding Philox seed plumbing.

The Mamba SSM SR path previously generated `rand_seed` tensors via
`torch.randint(..., (1,))` on every decode forward.  The cache manager now
owns a persistent per-cache-slot int64 buffer that is deterministically
initialized and rewritten on fresh request assignment.  These tests pin the
contract: pure-function seed generation, deterministic allocation, and
per-slot reset without `torch.randint`.
"""

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import (
    PythonMambaCacheManager,
    _allocate_mamba_seed_buffer,
    _compute_deterministic_mamba_seed,
    _mamba_rank_offset,
)
from tensorrt_llm.mapping import Mapping


def test_deterministic_seed_is_pure_function():
    s1 = _compute_deterministic_mamba_seed(counter=7, slot=3, rank_offset=42)
    s2 = _compute_deterministic_mamba_seed(counter=7, slot=3, rank_offset=42)
    assert s1 == s2
    assert 0 < s1 < (1 << 62)


def test_deterministic_seed_distinct_inputs_distinct_outputs():
    s_a = _compute_deterministic_mamba_seed(1, 0, 0)
    s_b = _compute_deterministic_mamba_seed(1, 1, 0)
    s_c = _compute_deterministic_mamba_seed(1, 0, 1)
    s_d = _compute_deterministic_mamba_seed(2, 0, 0)
    # All four (counter, slot, rank_offset) keys yield distinct seeds.
    assert len({s_a, s_b, s_c, s_d}) == 4


def test_rank_offset_distinct_per_rank():
    o0 = _mamba_rank_offset(Mapping(world_size=4, tp_size=4, pp_size=1, rank=0))
    o1 = _mamba_rank_offset(Mapping(world_size=4, tp_size=4, pp_size=1, rank=1))
    o2 = _mamba_rank_offset(Mapping(world_size=4, tp_size=4, pp_size=1, rank=2))
    o3 = _mamba_rank_offset(Mapping(world_size=4, tp_size=4, pp_size=1, rank=3))
    assert len({o0, o1, o2, o3}) == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_allocated_buffer_is_reproducible_and_nonzero():
    device = torch.device("cuda")
    buf1 = _allocate_mamba_seed_buffer(8, rank_offset=5, device=device)
    buf2 = _allocate_mamba_seed_buffer(8, rank_offset=5, device=device)
    assert buf1.dtype == torch.int64
    assert buf1.shape == (8,)
    assert torch.equal(buf1, buf2)
    assert (buf1 > 0).all().item()
    # All eight slots should differ; collisions would defeat per-slot
    # variance in the replay kernel.
    assert buf1.unique().numel() == 8


def _make_python_manager(*, sr: bool, replay: bool, max_batch_size: int = 4):
    """Build a tiny PythonMambaCacheManager with MTP/spec enabled."""
    mapping = Mapping(world_size=1, tp_size=1, pp_size=1, rank=0)
    return PythonMambaCacheManager(
        d_state=4,
        d_conv=2,
        num_heads=2,
        n_groups=1,
        head_dim=2,
        num_layers=1,
        max_batch_size=max_batch_size,
        spec_state_size=max_batch_size,
        mapping=mapping,
        dtype=torch.float16,
        ssm_cache_dtype=torch.float16,
        layer_mask=[True],
        speculative_num_draft_tokens=1,
        model_type="nemotron_hybrid",
        use_replay_state_update=replay,
        mamba_ssm_stochastic_rounding=sr,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_buffer_allocated_when_sr_only_no_replay():
    # The regression: SR enabled but replay off must still produce a
    # persistent seed buffer so mamba2_mixer's non-replay branch can read it.
    mgr = _make_python_manager(sr=True, replay=False)
    seed_buf = mgr.get_mamba_ssm_rand_seed()
    assert seed_buf is not None
    assert seed_buf.dtype == torch.int64
    assert seed_buf.device.type == "cuda"
    # Replay-only buffers are not allocated.
    assert mgr.mamba_cache.old_x is None
    assert mgr.mamba_cache.intermediate_ssm is not None  # legacy SSM cache
    # SpeculativeState exposes the same buffer (same Python identity).
    assert mgr.mamba_cache.mamba_ssm_rand_seed is seed_buf


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_buffer_absent_when_neither_sr_nor_replay():
    mgr = _make_python_manager(sr=False, replay=False)
    assert mgr.get_mamba_ssm_rand_seed() is None
    assert mgr.mamba_cache.mamba_ssm_rand_seed is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_per_slot_reset_is_deterministic_and_uses_no_randint():
    mgr = _make_python_manager(sr=True, replay=False, max_batch_size=8)
    seed_buf = mgr.get_mamba_ssm_rand_seed()
    assert seed_buf is not None
    pre = seed_buf.clone()
    mgr._prepare_mamba_cache_blocks([1001])
    block = mgr.mamba_cache_index[1001]
    post = seed_buf.clone()
    # Exactly the freshly-assigned slot is rewritten.
    diff_mask = pre != post
    assert diff_mask.sum().item() == 1
    assert diff_mask[block].item()
    # Rewrite is a deterministic function of (counter, slot, rank_offset).
    expected = _compute_deterministic_mamba_seed(
        mgr._seed_request_counter, block, mgr._seed_rank_offset
    )
    assert post[block].item() == expected
    assert expected > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_padding_sentinel_does_not_churn_seeds():
    # CUDA-graph padding sentinels alias to the shared _padding_slot and
    # must not rotate the seed entry of live real requests.
    mgr = _make_python_manager(sr=True, replay=False, max_batch_size=8)
    seed_buf = mgr.get_mamba_ssm_rand_seed()
    assert seed_buf is not None
    from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import CUDA_GRAPH_DUMMY_REQUEST_ID

    sentinel_id = CUDA_GRAPH_DUMMY_REQUEST_ID
    pre = seed_buf.clone()
    counter_pre = mgr._seed_request_counter
    mgr.add_dummy_requests([sentinel_id])
    post = seed_buf.clone()
    # Sentinel assignment must leave the buffer (and the host counter)
    # untouched; otherwise the seed buffer would churn every graph capture.
    assert torch.equal(pre, post)
    assert mgr._seed_request_counter == counter_pre


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_replay_path_still_allocates_seed_buffer():
    # Backward-compatibility: the replay path used to allocate the seed
    # buffer; the new wiring must not regress that.
    mgr = _make_python_manager(sr=False, replay=True)
    seed_buf = mgr.get_mamba_ssm_rand_seed()
    assert seed_buf is not None
    assert mgr.mamba_cache.mamba_ssm_rand_seed is seed_buf
    assert mgr.mamba_cache.old_x is not None  # replay buffers also exist


def _build_cpp_hybrid(*, spec_config, use_replay: bool, sr: bool, max_batch_size: int = 4):
    """Construct a CppMambaHybridCacheManager with one mamba + one attention
    layer.  Mirrors test_mamba_cache_manager._build_hybrid_with_mamba_layer
    but parameterizes the replay / SR flags so we can exercise the
    non-replay MTP SR layer-cache hand-off path."""
    from tensorrt_llm._torch.pyexecutor.mamba_cache_manager import CppMambaHybridCacheManager
    from tensorrt_llm._torch.pyexecutor.resource_manager import CacheTypeCpp
    from tensorrt_llm.llmapi.llm_args import KvCacheConfig

    mamba_mask = [True, False]
    attn_mask = [False, True]
    mapping = Mapping(world_size=1, rank=0, tp_size=1, pp_size=1)
    kv_cache_config = KvCacheConfig(max_tokens=512, enable_block_reuse=False)
    return CppMambaHybridCacheManager(
        mamba_d_state=8,
        mamba_d_conv=4,
        mamba_num_heads=4,
        mamba_n_groups=1,
        mamba_head_dim=8,
        mamba_num_layers=1,
        mamba_layer_mask=mamba_mask,
        mamba_cache_dtype=torch.float16,
        mamba_ssm_cache_dtype=torch.float16,
        kv_cache_config=kv_cache_config,
        kv_cache_type=CacheTypeCpp.SELF,
        num_layers=1,
        num_kv_heads=4,
        head_dim=64,
        tokens_per_block=32,
        max_seq_len=128,
        max_batch_size=max_batch_size,
        mapping=mapping,
        spec_config=spec_config,
        layer_mask=attn_mask,
        use_replay_state_update=use_replay,
        mamba_ssm_stochastic_rounding=sr,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cpp_hybrid_non_replay_mtp_layer_cache_carries_rand_seed():
    """Regression: CppMambaHybridCacheManager.mamba_layer_cache() with
    spec_config != None AND _use_replay_state_update == False AND
    mamba_ssm_stochastic_rounding == True must still surface
    mamba_ssm_rand_seed on the returned SpeculativeState.

    The mixer's non-replay MTP SR branch (mamba2_mixer.py) reads
    `layer_cache.mamba_ssm_rand_seed` and asserts non-None.  Iter5 review
    caught the regression where the seed was only forwarded inside the
    replay branch of mamba_layer_cache; this test pins both paths."""
    from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig

    spec_config = MTPDecodingConfig(max_draft_len=2)
    mgr = _build_cpp_hybrid(spec_config=spec_config, use_replay=False, sr=True)
    # Manager-level buffer must exist (SR is on).
    seed_buf = mgr.get_mamba_ssm_rand_seed()
    assert seed_buf is not None
    assert seed_buf.dtype == torch.int64
    # Layer cache exposes the same buffer (Python identity equality).
    layer_cache = mgr.mamba_layer_cache(0)
    assert layer_cache is not None
    assert layer_cache.mamba_ssm_rand_seed is mgr.mamba_ssm_rand_seed
    # And the SpeculativeState is on the non-replay legacy branch.
    assert layer_cache.intermediate_ssm is not None
    assert layer_cache.old_x is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cpp_hybrid_replay_mtp_layer_cache_still_carries_rand_seed():
    """Backward-compat: the replay branch must keep forwarding the seed
    buffer through mamba_layer_cache."""
    from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig

    spec_config = MTPDecodingConfig(max_draft_len=2)
    mgr = _build_cpp_hybrid(spec_config=spec_config, use_replay=True, sr=True)
    seed_buf = mgr.get_mamba_ssm_rand_seed()
    assert seed_buf is not None
    layer_cache = mgr.mamba_layer_cache(0)
    assert layer_cache is not None
    assert layer_cache.mamba_ssm_rand_seed is mgr.mamba_ssm_rand_seed
    # Replay-specific compact buffers are populated.
    assert layer_cache.old_x is not None
    assert layer_cache.cache_buf_idx is not None
    assert layer_cache.prev_num_accepted_tokens is not None
