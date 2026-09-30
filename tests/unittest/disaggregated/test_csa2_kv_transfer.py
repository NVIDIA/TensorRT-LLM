# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Transfer packed CSA2 owners and private SWA/compressor state through V2."""

import functools
import uuid
from collections.abc import Sequence

import pytest
import torch
from kv_transfer_harness import (
    MAX_BATCH_SIZE,
    MAX_SEQ_LEN,
    TOKENS_PER_BLOCK,
    VOCAB_SIZE,
    create_instance_transceivers,
    get_ctx_info_endpoint,
    run_concurrent,
    run_kv_transfer_test,
)
from test_deepseek_v4_kv_transfer import _init_pool_data

from tensorrt_llm import DisaggregatedParams
from tensorrt_llm._torch.attention.backends.sparse.csa2.cache_manager import (
    CSA2CacheManager,
    CSA2CacheRole,
)
from tensorrt_llm._torch.attention.backends.sparse.csa2.params import CSA2Layout
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, LlmRequestType
from tensorrt_llm.bindings import DataType, SamplingConfig
from tensorrt_llm.bindings.internal.batch_manager import CacheType
from tensorrt_llm.llmapi.llm_args import CacheTransceiverConfig, KvCacheConfig
from tensorrt_llm.mapping import Mapping
from tensorrt_llm.runtime.kv_cache_manager_v2 import _introspection


def _managers(
    tp: int, pp: int, enable_dp: bool, layout: CSA2Layout, *, reuse: bool = False
) -> list[CSA2CacheManager]:
    assert pp == 1  # CSA2 PP remains explicitly unsupported.
    return [
        CSA2CacheManager(
            KvCacheConfig(max_gpu_total_bytes=128 << 20, enable_block_reuse=reuse, dtype="fp8"),
            CacheType.SELFKONLY,
            num_layers=len(layout.compress_ratios),
            tokens_per_block=TOKENS_PER_BLOCK,
            max_seq_len=MAX_SEQ_LEN,
            max_batch_size=MAX_BATCH_SIZE,
            max_num_tokens=MAX_SEQ_LEN * MAX_BATCH_SIZE,
            vocab_size=VOCAB_SIZE,
            dtype=DataType.BF16,
            mapping=Mapping(world_size=tp, rank=rank, tp_size=tp, enable_attention_dp=enable_dp),
            layout=layout,
            is_disagg=True,
        )
        for rank in range(tp)
    ]


def _init_csa2_pool_data(
    managers: Sequence[CSA2CacheManager],
    tp: int,
    *,
    seed_base: int = 0,
    fill_random: bool = True,
) -> None:
    # Attention-DP ranks own unrelated requests; distinct data makes a wrong
    # source rank observable even when every rank allocates the same page IDs.
    for rank, manager in enumerate(managers):
        seed = seed_base + (rank if manager.mapping.enable_attention_dp else rank // tp)
        _init_pool_data([manager], 1, seed_base=seed, fill_random=fill_random)


def _verify(
    request_lengths,
    ctx_managers,
    gen_managers,
    ctx_tp,
    ctx_pp,
    gen_tp,
    gen_pp,
    ctx_enable_dp,
    gen_enable_dp,
    ctx_request_ids,
    gen_request_ids,
) -> None:
    assert ctx_pp == gen_pp == 1
    for request, length in enumerate(request_lengths):
        source = ctx_managers[request % ctx_tp if ctx_enable_dp else 0]
        for rank, target in enumerate(gen_managers):
            if gen_enable_dp and request % gen_tp != rank:
                continue
            for layer in target.pp_layers:
                spec = target.layout.layer(layer)
                roles = [CSA2CacheRole.SWA]
                if spec.kv_source is not None:
                    roles.extend((CSA2CacheRole.GLOBAL, CSA2CacheRole.INDEX))
                    assert target.get_buffers(layer, CSA2CacheRole.GLOBAL).data_ptr() == (
                        target.get_buffers(spec.kv_source, CSA2CacheRole.GLOBAL).data_ptr()
                    )
                if spec.kv_source == layer and spec.compress_ratio == 2:
                    roles.extend((CSA2CacheRole.COMPRESSOR_KV, CSA2CacheRole.COMPRESSOR_SCORE))
                for role in roles:
                    window = None
                    if role == CSA2CacheRole.SWA:
                        window = target._swa_retention(layer)
                    elif role not in (CSA2CacheRole.GLOBAL, CSA2CacheRole.INDEX):
                        window = target._window(2)
                    first = (
                        0 if window is None else max(0, (length + 1 - window) // TOKENS_PER_BLOCK)
                    )
                    end = (length + TOKENS_PER_BLOCK - 1) // TOKENS_PER_BLOCK
                    source_pages = source.get_cache_indices(ctx_request_ids[request], layer, role)
                    target_pages = target.get_cache_indices(gen_request_ids[request], layer, role)
                    src = source.get_buffers(layer, role)
                    dst = target.get_buffers(layer, role)
                    assert src.shape[1:] == dst.shape[1:]
                    for logical in range(first, end):
                        assert source_pages[logical] >= 0 and target_pages[logical] >= 0
                        # Compare all bytes, including packed main/index scales
                        # and both FP32 compressor state buffers.
                        torch.testing.assert_close(
                            dst[target_pages[logical]].view(torch.uint8),
                            src[source_pages[logical]].view(torch.uint8),
                            rtol=0,
                            atol=0,
                            msg=f"request={request}, layer={layer}, role={role.value}, page={logical}",
                        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "ctx_tp,gen_tp,ctx_enable_dp,gen_enable_dp",
    [
        (1, 1, False, False),
        (2, 1, False, False),
        (1, 2, False, False),
        (2, 2, False, False),
        pytest.param(4, 4, True, True, id="dp4-to-dp4"),
        pytest.param(4, 1, True, False, id="dp4-to-tp1"),
        pytest.param(1, 4, False, True, id="tp1-to-dp4"),
    ],
)
@pytest.mark.parametrize("update_before_transfer", [False, True])
def test_csa2_shared_owner_kv_transfer(
    ctx_tp: int,
    gen_tp: int,
    ctx_enable_dp: bool,
    gen_enable_dp: bool,
    update_before_transfer: bool,
) -> None:
    layout = CSA2Layout((0, 2, 2, 1, 1, 1), (1, 3), (1, 3, 4), candidate_source_layer_id=3)
    run_kv_transfer_test(
        ctx_tp=ctx_tp,
        ctx_pp=1,
        gen_tp=gen_tp,
        gen_pp=1,
        ctx_enable_dp=ctx_enable_dp,
        gen_enable_dp=gen_enable_dp,
        update_before_transfer=update_before_transfer,
        manager_factory=functools.partial(_managers, layout=layout),
        init_fn=_init_csa2_pool_data,
        verify_fn=_verify,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("ctx_tp,gen_tp", [(1, 1), (2, 1), (1, 2)])
@pytest.mark.parametrize("policy", ["required", "hit", "partial-eviction", "evicted"])
def test_csa2_ced_transfer_with_generation_prefix_reuse(monkeypatch, ctx_tp, gen_tp, policy):
    """A Global hit cannot suppress missing Encoder or private Decoder transfers."""
    monkeypatch.setenv("TRTLLM_V41_DECODER_BOUNDED_REPLAY", "1")
    monkeypatch.setenv("TRTLLM_V41_ENCODER_REPLAY", "0" if policy == "required" else "1")
    layout = CSA2Layout((0, 2, 2, 1, 1, 1), (1, 3), (1, 3, 4), candidate_source_layer_id=3)
    ctx_managers = _managers(ctx_tp, 1, False, layout, reuse=True)
    gen_managers = _managers(gen_tp, 1, False, layout, reuse=True)
    ctx_tcs, gen_tcs = [], []

    def request(req_id, kind):
        return LlmRequest(
            request_id=req_id,
            max_new_tokens=2,
            input_tokens=list(range(257)),
            sampling_config=SamplingConfig(),
            is_streaming=False,
            llm_request_type=kind,
        )

    def fill_pages(manager, req_id, end):
        # Distinct logical page/layer/role bytes catch shifted or omitted
        # transfers, while matching reused data across independent allocators.
        for (layer, role), virtual_layer in manager._layer_roles.items():
            pages = manager.get_cache_indices(req_id, layer, role)
            data = manager.get_buffers(layer, role)
            for logical, page in enumerate(pages[:end]):
                if page >= 0:
                    data[page].view(torch.uint8).fill_((virtual_layer * 17 + logical + 1) % 251)

    try:
        ctx_req = request(0, LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY)
        gen_req = request(1, LlmRequestType.LLMREQUEST_TYPE_GENERATION_ONLY)
        for manager in ctx_managers:
            assert manager._decoder_replay_window == layout.window_size
            assert manager.prepare_context(ctx_req)
            assert manager.resize_context(ctx_req, 257)
            fill_pages(manager, 0, 3)
        for manager in gen_managers:
            _init_pool_data([manager], 1, fill_random=False)
            warm = request(2, LlmRequestType.LLMREQUEST_TYPE_CONTEXT_ONLY)
            assert manager.prepare_context(warm)
            assert manager.resize_context(warm, 256)
            fill_pages(manager, 2, 2)
            cache = manager.kv_cache_map[2]
            cache.commit(list(range(256)))
            missing = (
                list(manager._encoder_optional_groups[:1])
                if policy == "partial-eviction"
                else list(manager._encoder_optional_groups)
                if policy == "evicted"
                else []
            )
            if missing:
                _introspection.drop_optional_pages(cache, 256, missing)
            manager.free_resources(warm)
            assert manager.prepare_disagg_gen_init(gen_req)
            cache = manager.kv_cache_map[1]
            assert cache.num_committed_tokens == 256
            assert gen_req.py_ced_replay is None
            for group in manager._encoder_optional_groups:
                assert cache.reuse_status[group].complete == (group not in missing)
            for (layer, role), virtual_layer in manager._layer_roles.items():
                group = manager.impl.get_layer_group_id(virtual_layer)
                restored = cache.reuse_status[group].complete
                data = manager.get_buffers(layer, role)
                for logical, page in enumerate(manager.get_cache_indices(1, layer, role)):
                    if page >= 0 and (not restored or logical >= 2):
                        data[page].zero_()

        config = CacheTransceiverConfig(
            backend="NIXL", transceiver_runtime="PYTHON", max_tokens_in_buffer=512
        )
        ctx_tcs = create_instance_transceivers(ctx_tp, 1, False, ctx_managers, config)
        gen_tcs = create_instance_transceivers(gen_tp, 1, False, gen_managers, config)
        transfer_id = uuid.uuid4().int & 0x7FFFFFFFFFFFFFFF
        ctx_req.py_disaggregated_params = DisaggregatedParams(disagg_request_id=transfer_id)
        gen_req.py_disaggregated_params = DisaggregatedParams(
            ctx_request_id=0,
            ctx_dp_rank=0,
            ctx_info_endpoint=get_ctx_info_endpoint(ctx_tcs[0]),
            disagg_request_id=transfer_id,
        )
        for req in (ctx_req, gen_req):
            req.context_current_position = req.prompt_len
            req.add_new_token(req.prompt_len, 0)
        for tc in gen_tcs:
            tc.request_and_receive_async(gen_req)
        for tc in ctx_tcs:
            tc.respond_and_send_async(ctx_req)
        run_concurrent(
            ctx_tcs, lambda tc: tc.check_context_transfer_status(None, mark_complete=True)
        )
        run_concurrent(gen_tcs, lambda tc: tc.check_gen_transfer_status(None))
        _verify([257], ctx_managers, gen_managers, ctx_tp, 1, gen_tp, 1, False, False, [0], [1])
    finally:
        for tc in ctx_tcs + gen_tcs:
            tc.shutdown()
        for manager in ctx_managers + gen_managers:
            manager.shutdown()
