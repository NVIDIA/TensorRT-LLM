# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""KDA 2FP4 replay state in the V2 managers: layout, lifecycle, and suspension."""

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.glm_kpool.cache_manager import (
    Glm5NextCacheManager,
)
from tensorrt_llm._torch.modules.kimi_kda.cache_manager import (
    Kda2Fp4LayerCache,
    KDA2Fp4ReplayState,
    KimiK3HybridCacheManager,
)
from tensorrt_llm._torch.modules.kimi_kda.kda_2fp4 import HISTORY_SIZE, STATE_BYTES_PER_HEAD
from tensorrt_llm._utils import get_sm_version, is_sm_100f
from tensorrt_llm.bindings import DataType
from tensorrt_llm.bindings.internal.batch_manager import CacheType as CacheTypeCpp
from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping

HEADS = 4
LAYERS = 3


def _requires_2fp4_kernels():
    from tensorrt_llm._torch.cute_dsl_utils import IS_CUTLASS_DSL_AVAILABLE

    if not torch.cuda.is_available() or not is_sm_100f(get_sm_version()):
        pytest.skip("KDA 2FP4 replay requires an SM100/SM103 GPU")
    if not IS_CUTLASS_DSL_AVAILABLE:
        pytest.skip("KDA 2FP4 replay requires CUTLASS DSL")


def _manager(
    manager_cls=KimiK3HybridCacheManager, dspark=False, heads=HEADS, layers=LAYERS, **kwargs
):
    """A V2 hybrid manager with packed 2FP4 KDA checkpoints."""
    if manager_cls is Glm5NextCacheManager:
        kwargs.setdefault("index_state_dim", 16)
    return manager_cls(
        mamba_d_state=128,
        mamba_d_conv=4,
        mamba_num_heads=heads,
        mamba_n_groups=heads,
        mamba_head_dim=128,
        mamba_num_layers=layers,
        mamba_layer_mask=[True] * layers + [False],
        mamba_cache_dtype=torch.bfloat16,
        mamba_ssm_cache_dtype=torch.float32,
        kv_cache_config=KvCacheConfig(
            max_tokens=4096,
            enable_block_reuse=False,
            max_util_for_resume=1.0,
            mamba_state_config={"kda_2fp4_replay": True},
        ),
        kv_cache_type=CacheTypeCpp.SELF,
        num_layers=1,
        num_kv_heads=4,
        head_dim=64,
        tokens_per_block=32,
        max_seq_len=128,
        max_batch_size=4,
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        dtype=DataType.HALF,
        spec_config=DSparkDecodingConfig(max_draft_len=3) if dspark else None,
        layer_mask=[False] * layers + [True],
        vocab_size=1024,
        conv_state_layout="q_k_v",
        kda_2fp4_replay=True,
        **kwargs,
    )


@pytest.mark.parametrize("manager_cls", [KimiK3HybridCacheManager, Glm5NextCacheManager])
@pytest.mark.parametrize("dspark", [False, True], ids=["decode", "dspark"])
def test_2fp4_manager_keeps_packed_checkpoints_and_region_histories(manager_cls, dspark):
    _requires_2fp4_kernels()
    mgr = _manager(manager_cls, dspark)
    try:
        assert mgr.use_kda_2fp4_replay
        assert isinstance(mgr._kda_replay, KDA2Fp4ReplayState)
        assert mgr.commits_every_generation_step is dspark
        assert mgr.replay_layer_group_id is not None
        ssm = mgr.all_ssm_states[0]
        assert ssm.dtype == torch.uint8
        assert tuple(ssm.shape[1:]) == (HEADS, 1, STATE_BYTES_PER_HEAD)
        capacity = HISTORY_SIZE + (4 if dspark else 0)
        for layer in mgr.mamba_pp_layers:
            cache = mgr.mamba_layer_cache(layer)
            assert isinstance(cache, Kda2Fp4LayerCache)
            history = cache.history
            replay_slots = history.k.shape[0]
            for plane, dtype in ((history.k, torch.bfloat16), (history.u, torch.bfloat16)):
                assert plane.shape == (replay_slots, HEADS, capacity, 128)
                assert plane.dtype == dtype
                assert plane.stride()[1:] == (capacity * 128, 128, 1)
            assert history.g.dtype == torch.float32
            assert history.length.shape == (replay_slots,)
            assert (history.candidate_x is not None) is dspark
            assert (history.old_lengths is not None) is dspark
            assert (history.arrivals is not None) is not dspark
        roles = {str(role) for role in mgr.get_disagg_ignored_roles()}
        assert {"kda_2fp4_k", "kda_2fp4_u", "kda_2fp4_g", "kda_2fp4_len"} <= roles
    finally:
        mgr.shutdown()


def test_2fp4_context_worker_keeps_no_history(monkeypatch):
    _requires_2fp4_kernels()
    monkeypatch.setenv("TRTLLM_DISAGG_ROLE", "context")
    mgr = _manager(is_disagg=True)
    try:
        assert mgr.kda_context_only
        assert mgr._kda_replay is None
        assert mgr.all_ssm_states[0].dtype == torch.uint8
        cache = mgr.mamba_layer_cache(mgr.mamba_pp_layers[0])
        assert isinstance(cache, Kda2Fp4LayerCache) and cache.history is None
        assert not mgr.commits_every_generation_step
    finally:
        mgr.shutdown()


def test_2fp4_new_and_transferred_requests_start_with_empty_histories():
    _requires_2fp4_kernels()
    mgr = _manager()
    requests = []
    try:
        requests = mgr.add_dummy_requests([1, 2], token_nums=[8, 8], is_gen=True)
        histories = [mgr.mamba_layer_cache(layer).history for layer in mgr.mamba_pp_layers]
        replay_slots = mgr.get_replay_state_indices([1, 2])
        for history in histories:
            assert history.length[replay_slots].tolist() == [0, 0]
            history.length[replay_slots] = torch.tensor([5, 9], dtype=torch.int32, device="cuda")
        mgr.on_state_transfer_complete([2])
        for history in histories:
            assert history.length[replay_slots].tolist() == [5, 0]
    finally:
        for request in requests:
            mgr.free_resources(request)
        mgr.shutdown()


def _take_ssm_slot(mgr, slot, others):
    """Create caches until one of them takes ``slot``, moving the suspended owner's SSM
    page to the host tier; keep that one, close the rest."""
    holder = None
    attempts = min(
        mgr._state_layout.slot_capacity + 1, mgr._index_mapper_capacity - len(mgr.kv_cache_map)
    )
    for _ in range(attempts):
        cache = mgr.impl.create_kv_cache()
        assert cache.resume(mgr._stream.cuda_stream)
        others.append(cache)
        assert cache.resize(1)
        if cache.get_ssm_block_base_index(mgr.ssm_layer_group_id) == slot:
            holder = cache
            break
    assert holder is not None, "no cache took the suspended request's SSM slot"
    for cache in others:
        if cache is not holder:
            cache.close()
    others[:] = [holder]


# Packed slots of 32 heads on 4 layers (~2.4 MB) exceed the 2 MiB allocation
# grain, so the SSM pool holds few slots and another cache can take a
# suspended request's slot.
SUSPEND_HEADS = 32
SUSPEND_LAYERS = 4


def _decode_step(mgr, weights, step):
    """One ordinary 2FP4 decode of request 1 on every KDA layer; inputs depend only on the step."""
    from tensorrt_llm._torch.modules.kimi_kda.kda_2fp4 import replay

    heads = SUSPEND_HEADS
    width = heads * 128
    slots = torch.tensor([mgr._request_id_to_state_index[1]], dtype=torch.int32, device="cuda")
    replay_slots = torch.tensor(mgr.get_replay_state_indices([1]), dtype=torch.int32, device="cuda")
    dummy = torch.zeros(1, dtype=torch.bool, device="cuda")
    outputs = []
    for offset, layer in enumerate(mgr.mamba_pp_layers):
        cache = mgr.mamba_layer_cache(layer)
        generator = torch.Generator(device="cuda").manual_seed(1000 * step + offset)

        def randn(*shape, scale=0.5, shift=0.0):
            return (
                torch.randn(shape, generator=generator, device="cuda") * scale + shift
            ).bfloat16()

        outputs.append(
            replay(
                randn(1, 3 * width),
                cache.conv,
                weights[offset]["conv"],
                randn(1, width, shift=-0.5),
                randn(1, heads, scale=1.0),
                randn(1, heads, 128),
                weights[offset]["norm"],
                cache.temporal,
                slots,
                replay_slots,
                dummy,
                cache.history,
                weights[offset]["A_log"],
                weights[offset]["dt_bias"],
                -5.0,
                1e-6,
            ).clone()
        )
    return torch.stack(outputs)


def _run_decode(steps, suspend_after=None, slot_taken=False):
    mgr = _manager(heads=SUSPEND_HEADS, layers=SUSPEND_LAYERS)
    others = []
    request = None
    try:
        (request,) = mgr.add_dummy_requests([1], token_nums=[8], is_gen=True)
        generator = torch.Generator(device="cuda").manual_seed(7)
        weights = [
            {
                "conv": (
                    torch.randn(3 * SUSPEND_HEADS * 128, 4, generator=generator, device="cuda")
                    * 0.3
                ).bfloat16(),
                "norm": torch.ones(128, dtype=torch.bfloat16, device="cuda"),
                "A_log": torch.randn(SUSPEND_HEADS, generator=generator, device="cuda") * 0.3,
                "dt_bias": torch.randn(SUSPEND_HEADS * 128, generator=generator, device="cuda")
                * 0.1,
            }
            for _ in mgr.mamba_pp_layers
        ]
        slot = mgr._request_id_to_state_index[1]
        for ssm, conv in zip(mgr.all_ssm_states, mgr.all_conv_states):
            ssm[slot].zero_()
            conv[slot] = torch.randn(conv.shape[1:], generator=generator, device="cuda") * 0.1
        mgr.on_state_transfer_complete([1])
        outputs = []
        for step in range(steps):
            outputs.append(_decode_step(mgr, weights, step))
            if step == suspend_after:
                slot = mgr._request_id_to_state_index[1]
                lengths = [
                    h.length[mgr.get_replay_state_indices([1])].item()
                    for h in (mgr.mamba_layer_cache(layer).history for layer in mgr.mamba_pp_layers)
                ]
                assert all(0 < length < HISTORY_SIZE for length in lengths)
                mgr.suspend_request(request)
                if slot_taken:
                    _take_ssm_slot(mgr, slot, others)
                    for ssm, conv in zip(mgr.all_ssm_states, mgr.all_conv_states):
                        ssm[slot].random_(0, 256)
                        conv[slot].normal_()
                assert mgr.resume_request(request)
                mgr._setup_state_indices([request], num_contexts=0)
                if slot_taken:
                    assert mgr._request_id_to_state_index[1] != slot
        torch.cuda.synchronize()
        return outputs
    finally:
        for cache in others:
            cache.close()
        if request is not None:
            mgr.free_resources(request)
        mgr.shutdown()


@pytest.mark.parametrize("slot_taken", [True, False], ids=["slot_taken", "slot_kept"])
def test_2fp4_history_follows_a_suspended_request(slot_taken):
    """Suspended with a partial history and resumed, possibly in another SSM slot, a
    request decodes bit for bit like one never suspended, through the next window
    commits. No flush is needed: the history pages stay with the request."""
    _requires_2fp4_kernels()
    steps = 2 * HISTORY_SIZE + 5
    expected = _run_decode(steps)
    actual = _run_decode(steps, suspend_after=HISTORY_SIZE + 6, slot_taken=slot_taken)
    for step, (got, want) in enumerate(zip(actual, expected, strict=True)):
        assert torch.equal(got, want), f"decode step {step}"


def _standalone_state(heads, slots, max_batch_size, tokens_per_gen_step):
    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import MambaStateLayout

    layout = MambaStateLayout(
        layer_mask=(True,),
        pp_layers=(0,),
        mamba_pp_layers=(0,),
        mapping=Mapping(world_size=1, rank=0, tp_size=1, pp_size=1),
        conv_state_shape=(3 * heads * 128, 3),
        ssm_state_shape=(heads, 1, STATE_BYTES_PER_HEAD),
        conv_state_dtype=torch.bfloat16,
        ssm_state_dtype=torch.uint8,
        n_groups_per_rank=heads,
        max_batch_size=max_batch_size,
        state_index_capacity=max_batch_size,
        slot_capacity=slots,
        spec_config=(
            None
            if tokens_per_gen_step is None
            else DSparkDecodingConfig(max_draft_len=tokens_per_gen_step - 1)
        ),
        conv_section_dims=(heads * 128,) * 3,
        conv_state_layout="q_k_v",
    )
    conv = torch.randn(slots, 3 * heads * 128, 3, device="cuda").bfloat16() * 0.01
    pool = torch.zeros(slots, heads, 1, STATE_BYTES_PER_HEAD, dtype=torch.uint8, device="cuda")
    state = KDA2Fp4ReplayState(tokens_per_gen_step)
    state.validate(layout)
    state.bind(layout, [pool], [conv])
    return state, state.make_layer_cache(0, conv, pool)


@torch.no_grad()
def test_mixer_prefills_packed_checkpoints_and_replays_them():
    """The mixer's packed-checkpoint paths end to end: prefill matches a dense
    reference exactly, ordinary decode crosses a window commit, a mixed batch
    routes its generation row through the replay, and DSpark verification
    (including a draft-free step) leaves both checkpoints for the commit."""
    _requires_2fp4_kernels()
    from types import SimpleNamespace

    from tensorrt_llm._torch.modules.kimi_kda import KimiKDALinearAttention
    from tensorrt_llm._torch.modules.kimi_kda.kda_2fp4 import materialize_checkpoint
    from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import (
        MambaAcceptanceBatch,
    )

    torch.manual_seed(123)
    heads, batch, slots_total = 3, 2, 11
    cfg = SimpleNamespace(
        hidden_size=512,
        rms_norm_eps=1e-6,
        linear_attn_config={
            "num_heads": heads,
            "head_dim": 128,
            "short_conv_kernel_size": 4,
            "use_full_rank_gate": True,
            "gate_lower_bound": -5.0,
        },
    )
    layer = KimiKDALinearAttention(cfg, layer_idx=0).cuda()
    torch.nn.init.normal_(layer.dt_bias, std=0.1)
    layer.finalize_decode_weights()
    state, cache = _standalone_state(heads, slots_total, 4, None)
    conv, pool, history = cache.conv, cache.temporal, cache.history
    slots = torch.tensor([7, 2], dtype=torch.int32, device="cuda")
    replay_slots = torch.tensor([4, 9], dtype=torch.int32, device="cuda")

    # Prefill: one sequence with an initial state, one without.
    flags = torch.tensor([True, False], device="cuda")
    pool[7].random_(0, 64)
    dense = torch.zeros((slots_total, heads, 128, 128), device="cuda")
    dense[slots.long()] = materialize_checkpoint(pool, slots, flags)
    metadata = SimpleNamespace(
        use_initial_states=True,
        has_initial_states=flags,
        kda_chunk_indices=torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], device="cuda"),
        kda_varlen_is_aligned=True,
        kda_single_sequence_length=None,
        query_start_loc=torch.tensor([0, 128, 256], dtype=torch.int32, device="cuda"),
        state_indices=slots,
        _arange_buffer=torch.arange(3, dtype=torch.int32, device="cuda"),
    )
    reference_conv = conv.clone()
    x = torch.randn((256, 512), dtype=torch.bfloat16, device="cuda") * 0.1
    reference = layer.forward_prefill(
        x, metadata.query_start_loc.long(), metadata, batch, reference_conv, dense, slots
    )
    actual = layer.forward_prefill(
        x, metadata.query_start_loc.long(), metadata, batch, conv, pool, slots, cache
    )
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)
    torch.testing.assert_close(conv, reference_conv, atol=0, rtol=0)
    assert torch.isfinite(materialize_checkpoint(pool, slots, torch.ones_like(flags))).all()

    def generation_metadata(state_indices, replay_indices, num_contexts=0, num_ctx_tokens=0):
        dummy = torch.zeros(state_indices.numel(), dtype=torch.bool, device="cuda")
        return SimpleNamespace(
            mamba_metadata=SimpleNamespace(
                state_indices=state_indices,
                replay_state_indices=replay_indices,
                query_start_loc_long=torch.zeros(1, dtype=torch.long, device="cuda"),
            ),
            num_contexts=num_contexts,
            num_ctx_tokens=num_ctx_tokens,
            num_tokens=0,
            seq_lens=torch.ones(state_indices.numel(), dtype=torch.int32),
            kv_cache_manager=SimpleNamespace(
                mamba_layer_cache=lambda _: cache,
                get_replay_dummy_request_mask=lambda: dummy,
            ),
        )

    # Ordinary decode through the controller: 17 steps commit one window.
    state.reset_replay_slots(replay_slots.long())
    attention = generation_metadata(slots, replay_slots)
    attention.num_tokens = batch
    for _ in range(HISTORY_SIZE + 1):
        hidden = torch.randn((batch, 512), dtype=torch.bfloat16, device="cuda") * 0.1
        output = layer._forward_impl(hidden, attention)
        assert output.shape == (batch, heads, 128) and torch.isfinite(output).all()
    assert history.length[replay_slots.long()].tolist() == [1, 1]

    # DSpark verification, including a draft-free step, then the commit.
    state, cache = _standalone_state(heads, slots_total, 4, 4)
    conv, pool, history = cache.conv, cache.temporal, cache.history
    state.reset_replay_slots(replay_slots.long())
    attention = generation_metadata(slots, replay_slots)
    for tokens in (4, 1):
        attention.num_tokens = batch * tokens
        hidden = torch.randn((batch * tokens, 512), dtype=torch.bfloat16, device="cuda") * 0.1
        before = (pool.clone(), conv.clone(), history.length.clone())
        output = layer._forward_impl(hidden, attention)
        assert output.shape == (batch * tokens, heads, 128) and torch.isfinite(output).all()
        for after, prior in zip((pool, conv, history.length), before):
            torch.testing.assert_close(after, prior, rtol=0, atol=0)
        state.update(
            MambaAcceptanceBatch(
                attention_metadata=attention,
                num_contexts=0,
                num_generations=batch,
                num_accepted_tokens=torch.ones(batch, dtype=torch.int32, device="cuda"),
                accepted_positions=torch.zeros(batch, dtype=torch.int32, device="cuda"),
                source_state_indices=slots,
                destination_state_indices=slots,
                is_dummy_request=torch.zeros(batch, dtype=torch.bool, device="cuda"),
                replay_indices=replay_slots,
            ),
            replay_slots,
        )
    assert history.length[replay_slots.long()].tolist() == [2, 2]
