# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.modeling_v2.models.nemotron_shortcut.gdp import gdp_mixer
from tensorrt_llm._torch.modeling_v2.models.nemotron_shortcut.indexer import IndexState
from tensorrt_llm._torch.modules.mamba.cache_manager import Mamba2State
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager.common import (
    MambaAcceptanceBatch,
    MambaStateLayout,
)
from tensorrt_llm.llmapi.llm_args import MTPDecodingConfig
from tensorrt_llm.mapping import Mapping


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires GDP and cache promotion kernels"
)
def test_gdp_verify_promotes_each_requests_accepted_prefix():
    torch.manual_seed(178)
    device = "cuda"
    dtype = torch.bfloat16
    batch, width, heads, dk, dv, substeps = 3, 3, 2, 32, 16, 3
    value_size = heads * dv
    channels = heads * (substeps * (dk + dv) + dk)
    core = SimpleNamespace(
        gdp_heads=heads,
        gdp_v=value_size,
        gdp_conv=channels,
        m=substeps,
        gdp_dv=dv,
        gdp_dk=dk,
        gdp_eps=1e-5,
        gdp_g_clamp_value=20.0,
        mm=F.linear,
    )

    def rand(*shape):
        return torch.randn(shape, device=device, dtype=dtype)

    projection_dim = value_size + channels + (substeps + 1) * heads
    layer = dict(
        gdp_in=torch.eye(projection_dim, device=device, dtype=dtype),
        gdp_conv=rand(channels, 4),
        gdp_a_log=torch.tensor([5.0, -1.0], device=device),
        gdp_dt_bias=rand(heads),
        gdp_norm=rand(value_size),
    )
    slots = torch.tensor([4, 1, 3], dtype=torch.int32, device=device)
    accepted = torch.tensor([1, 2, 3], dtype=torch.int32, device=device)
    tokens = rand(batch, width, projection_dim)
    conv = rand(5, 2, channels, 3)[:, 1]
    temporal = torch.randn(5, 2, heads, dv, dk, device=device)[:, 0]
    original_conv, original_temporal = conv.clone(), temporal.clone()
    manager = Mamba2State()
    geometry = MambaStateLayout(
        layer_mask=(True,),
        pp_layers=(0,),
        mamba_pp_layers=(0,),
        mapping=Mapping(),
        conv_state_shape=(channels, 3),
        ssm_state_shape=(heads, dv, dk),
        conv_state_dtype=dtype,
        ssm_state_dtype=torch.float32,
        n_groups_per_rank=heads,
        max_batch_size=batch,
        state_index_capacity=5,
        slot_capacity=5,
        spec_config=MTPDecodingConfig(max_draft_len=width - 1),
    )
    manager.bind(geometry, [temporal], [conv])
    cache = manager.make_layer_cache(0, conv, temporal)

    def mix(inputs, layer_cache, step_slots):
        md = SimpleNamespace(
            num_contexts=0,
            seq_lens=step_slots,
            mamba_metadata=SimpleNamespace(state_indices=step_slots),
            kv_cache_manager=SimpleNamespace(
                intermediate_state_indices=manager.intermediate_indices,
                mamba_layer_cache=lambda _: layer_cache,
            ),
        )
        return gdp_mixer(core, layer, inputs, md, 0)

    verified = mix(tokens.flatten(0, 1), cache, slots).reshape(batch, width, -1)
    torch.testing.assert_close(conv, original_conv, rtol=0, atol=0)
    torch.testing.assert_close(temporal, original_temporal, rtol=0, atol=0)
    serial = SimpleNamespace(conv=conv.clone(), temporal=temporal.clone())
    for step in range(width):
        expected = mix(tokens[:, step].contiguous(), serial, slots)
        torch.testing.assert_close(verified[:, step], expected, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(
            cache.intermediate_conv_window[:, step], serial.conv[slots], rtol=0, atol=0
        )
        torch.testing.assert_close(
            cache.intermediate_ssm[:, step], serial.temporal[slots], rtol=1e-5, atol=1e-6
        )

    manager.update(
        MambaAcceptanceBatch(
            attention_metadata=None,
            num_contexts=0,
            num_generations=batch,
            num_accepted_tokens=accepted,
            accepted_positions=accepted - 1,
            source_state_indices=manager.intermediate_indices,
            destination_state_indices=slots,
            is_dummy_request=None,
        )
    )
    # An independent serialized run commits only the accepted prefix for each slot.
    serial = SimpleNamespace(conv=original_conv.clone(), temporal=original_temporal.clone())
    for step in range(width):
        live = slice(step, batch)
        mix(tokens[live, step].contiguous(), serial, slots[live])
    torch.testing.assert_close(conv, serial.conv, rtol=0, atol=0)
    torch.testing.assert_close(temporal, serial.temporal, rtol=1e-5, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires token-sparse index kernels")
@pytest.mark.parametrize("case", ["over_topk", "short", "short_mixed"])
def test_verify_sparse_queries_see_only_their_own_causal_prefix(case):
    batch, width, page_size, topk, dim = 2, 3, 32, 2048, 128
    long = case == "over_topk"
    blocks = 65 if long else 2
    num_contexts = int(case == "short_mixed")
    seq_lens = torch.tensor([5 if num_contexts else width, width], dtype=torch.int32)
    kv_lens = torch.tensor([topk + 1, topk + 15] if long else [37, 48], dtype=torch.int32)
    pages = torch.arange(batch * blocks, dtype=torch.int32, device="cuda").reshape(batch, blocks)
    offsets = torch.stack((2 * pages, 2 * pages + 1), dim=1)
    cache = torch.zeros(batch * blocks, page_size, 1, dim, device="cuda", dtype=torch.bfloat16)
    positions = torch.arange(blocks * page_size, device="cuda")
    # Exactly represented increasing scores, including keys beyond the current candidate.
    keys = cache.view(batch, -1, dim)
    keys[:, :, 0] = positions // 64
    keys[:, :, 1] = positions % 64

    class Pages:
        layer_offsets = {0: 0}
        index_scales = [2]

        def get_index_k_buffer(self, layer, head_dim):
            return cache

    md = SimpleNamespace(
        kv_cache_manager=Pages(),
        host_kv_cache_pool_mapping=torch.tensor([[0, 0]]),
        kv_cache_block_offsets=offsets.unsqueeze(0),
        seq_lens=seq_lens,
        seq_lens_cuda=seq_lens.cuda(),
        kv_lens_runtime=kv_lens,
        kv_lens_cuda_runtime=kv_lens.cuda(),
        num_seqs=batch,
        max_num_sequences=batch,
        max_seq_len=blocks * page_size,
        num_contexts=num_contexts,
        num_ctx_tokens=int(seq_lens[0]) if num_contexts else 0,
        max_total_draft_tokens=width - 1,
    )
    state = IndexState(
        topk, [0], torch.cuda.get_device_properties(0).multi_processor_count, index_dim=dim
    )
    state.bind(md, cache.device)
    n_rows = int(seq_lens.sum())
    plan = state.plan(md, n_rows)
    assert plan.sparse
    query = cache.new_zeros(n_rows, dim)
    query[:, 0] = 64
    query[:, 1] = 1
    selection = plan.select(0, query)[0]
    start = 0
    for b, (length, count) in enumerate(zip(kv_lens.tolist(), seq_lens.tolist())):
        for t in range(count):
            end = length - count + t + 1
            expected = torch.arange(max(0, end - topk), end, device="cuda")
            k, v = selection[start + t].long()
            valid = k >= 0
            assert torch.equal(k < 0, v < 0)
            ordered = k[valid].argsort()
            base = 2 * pages[b, expected // page_size].long() * page_size
            torch.testing.assert_close(k[valid][ordered], base + expected % page_size)
            torch.testing.assert_close(v[valid][ordered], base + page_size + expected % page_size)
        start += count
