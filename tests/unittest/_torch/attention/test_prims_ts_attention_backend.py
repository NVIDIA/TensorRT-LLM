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

import pytest
import torch
from backend_case import (
    BackendCase,
    generate_inputs,
    generate_mla_gen_inputs,
    run_backend,
    run_case,
)
from utils.util import isSM100Family

pytestmark = pytest.mark.skipif(
    not isSM100Family(),
    reason="PrimsTS attention kernels require SM100 or SM103",
)


_QWEN2_7B = {
    "num_heads": 28,
    "num_kv_heads": 4,
    "head_dim": 128,
    "dtype": "bfloat16",
    "kv_layout": "HND",
    "page_size": 32,
    "rope": {
        "dim": 128,
        "theta": 1_000_000.0,
        "max_positions": 8192,
        "is_neox": True,
    },
    "fused_rope": True,
}

_DEEPSEEK_V3_LITE_MLA = {
    "num_heads": 32,
    "num_kv_heads": 1,
    "head_dim": 192,
    "dtype": "bfloat16",
    "kv_layout": "HND",
    "page_size": 32,
    "is_mla": True,
    "kv_lora_rank": 512,
    "q_lora_rank": 1536,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
}


@pytest.mark.parametrize("use_kv_cache_manager_v2", [False, True], ids=["v1", "v2"])
@pytest.mark.parametrize(
    "phase_args",
    [
        pytest.param(
            {
                "seq_lens": [65, 37],
                "num_cached_tokens": [0, 0],
                "num_contexts": 2,
            },
            id="context",
        ),
        pytest.param(
            {
                "seq_lens": [1, 1],
                "num_cached_tokens": [64, 96],
                "num_contexts": 0,
            },
            id="generation",
        ),
        pytest.param(
            {
                "seq_lens": [41, 1],
                "num_cached_tokens": [0, 63],
                "num_contexts": 1,
            },
            id="mixed",
        ),
    ],
)
def test_prims_ts_qwen2_gqa(
    monkeypatch: pytest.MonkeyPatch,
    use_kv_cache_manager_v2: bool,
    phase_args: dict,
) -> None:
    monkeypatch.setenv("TLLM_FMHA_LIBS", "prims_ts")
    case = BackendCase(
        **_QWEN2_7B,
        **phase_args,
        use_kv_cache_manager_v2=use_kv_cache_manager_v2,
    )

    run_case(case)


def test_prims_ts_fp16_dense_context_with_alternate_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TLLM_FMHA_LIBS", "prims_ts")
    case = BackendCase(
        num_heads=8,
        num_kv_heads=2,
        head_dim=256,
        seq_lens=[67, 33],
        num_cached_tokens=[0, 0],
        num_contexts=2,
        dtype="float16",
        causal=False,
        kv_layout="HND",
        page_size=64,
        use_kv_cache_manager_v2=True,
    )

    results = run_case(case)

    assert "TRTLLM" in results


def test_prims_ts_oversized_decode_workspace_replays_a_b_a(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensorrt_llm._torch.attention_backend.fmha.prims_ts as prims_ts_module
    from tensorrt_llm._torch.attention_backend.fmha.prims_ts import PrimsTSFmha

    monkeypatch.setenv("TLLM_FMHA_LIBS", "prims_ts")
    case = BackendCase(
        **_QWEN2_7B,
        seq_lens=[1, 1],
        num_cached_tokens=[64, 96],
        num_contexts=0,
        use_kv_cache_manager_v2=True,
    )
    inputs = generate_inputs(case, seed=0)
    golden = run_backend(
        case,
        "VANILLA",
        inputs,
        kv_dtype=case.compute_dtype,
        kv_layout="NHD",
    )

    max_kv_len = 128
    prims_workspace_bytes = prims_ts_module._get_prims_decode_workspace_size(
        case.num_seqs,
        case.num_heads,
        case.num_kv_heads,
        case.head_dim,
        case.page_size,
        max_kv_len,
        seq_len_q=1,
        q_dtype=case.compute_dtype,
        kv_dtype=case.compute_dtype,
        out_dtype=case.compute_dtype,
        mask_type="causal",
        window_left=-1,
        device=torch.device("cuda"),
    )
    assert prims_workspace_bytes > 0

    original_generation_layout = prims_ts_module.thop.get_trtllm_gen_generation_workspace_layout
    generation_layout_records = []

    def force_oversized_tail_threshold(*args, **kwargs):
        real_layout = original_generation_layout(*args, **kwargs)
        generation_layout_records.append(dict(real_layout))
        forced_layout = dict(real_layout)
        # Supported single-token PrimTS plans currently fit in the fixed TRT-LLM
        # slab. Lower only its advertised capacity to exercise the tail while
        # preserving the real THOP total layout, preprocessing, and kernel launch.
        forced_layout["trtllm_gen_workspace_size"] = prims_workspace_bytes - 1
        return forced_layout

    monkeypatch.setattr(
        prims_ts_module.thop,
        "get_trtllm_gen_generation_workspace_layout",
        force_oversized_tail_threshold,
    )

    original_get_decode_workspace = PrimsTSFmha._get_decode_workspace
    workspace_records = []

    def record_decode_workspace(
        self: PrimsTSFmha,
        root_workspace: torch.Tensor,
        thop_fmha_workspace: torch.Tensor,
    ) -> torch.Tensor:
        decode_workspace = original_get_decode_workspace(
            self,
            root_workspace,
            thop_fmha_workspace,
        )
        workspace_records.append(
            (
                self,
                root_workspace.data_ptr(),
                root_workspace.numel() * root_workspace.element_size(),
                thop_fmha_workspace.data_ptr(),
                thop_fmha_workspace.numel() * thop_fmha_workspace.element_size(),
                decode_workspace,
            )
        )
        return decode_workspace

    monkeypatch.setattr(PrimsTSFmha, "_get_decode_workspace", record_decode_workspace)

    eager = run_backend(
        case,
        "TRTLLM",
        inputs,
        kv_dtype=case.compute_dtype,
        fuse_rope=True,
        kv_layout="HND",
    )

    replay_index = 0

    def select_replay_state(metadata, static_inputs) -> None:
        nonlocal replay_index
        adapter, _, _, _, _, decode_workspace = workspace_records[-1]
        wrapper = adapter._decode_wrappers[case.num_seqs]
        control_offset = wrapper._workspace_layout.split_kv_counter.byte_offset
        control_end = wrapper._workspace_layout.total_bytes
        control_span = decode_workspace[control_offset:control_end]
        if replay_index > 0:
            assert torch.count_nonzero(control_span) == 0
        control_span.fill_(0xFF)

        if replay_index in (1, 2):
            qkv = static_inputs["q"]
            qkv.copy_(qkv.flip(0).clone())

            block_offsets = metadata.kv_cache_block_offsets
            block_offsets.copy_(block_offsets.flip(1).clone())

            kv_lens = metadata.kv_lens_cuda_runtime
            kv_lens.copy_(kv_lens.flip(0).clone())
        replay_index += 1

    replay_outputs = []

    actual = run_backend(
        case,
        "TRTLLM",
        inputs,
        kv_dtype=case.compute_dtype,
        fuse_rope=True,
        cuda_graph=True,
        kv_layout="HND",
        before_cuda_graph_replay=select_replay_state,
        cuda_graph_replay_count=3,
        cuda_graph_replay_outputs=replay_outputs,
    )

    reference_b = golden.flip(0)
    assert not torch.allclose(golden, reference_b, atol=3e-2, rtol=3e-3)
    torch.testing.assert_close(eager, golden, atol=3e-2, rtol=3e-3)
    assert len(replay_outputs) == 3
    for replay_output, reference in zip(
        replay_outputs,
        (golden, reference_b, golden),
        strict=True,
    ):
        torch.testing.assert_close(replay_output, reference, atol=3e-2, rtol=3e-3)
    torch.testing.assert_close(actual, golden, atol=3e-2, rtol=3e-3)

    assert replay_index == 3
    assert generation_layout_records
    real_total_bytes = int(generation_layout_records[0]["total_size"])
    assert all(
        int(layout["trtllm_gen_workspace_size"]) >= prims_workspace_bytes
        for layout in generation_layout_records
    )

    records_by_adapter = {}
    for adapter, root_ptr, root_bytes, thop_ptr, thop_bytes, decode_workspace in workspace_records:
        byte_offset = adapter._decode_workspace_offset_bytes
        required_bytes = adapter._decode_workspace_required_bytes
        assert byte_offset is not None
        assert byte_offset % 256 == 0
        assert byte_offset >= real_total_bytes
        assert required_bytes == prims_workspace_bytes
        assert root_bytes >= byte_offset + required_bytes
        assert decode_workspace.data_ptr() == root_ptr + byte_offset
        assert decode_workspace.numel() * decode_workspace.element_size() == required_bytes
        decode_end = decode_workspace.data_ptr() + required_bytes
        thop_end = thop_ptr + thop_bytes
        assert decode_workspace.data_ptr() >= thop_end or decode_end <= thop_ptr
        wrapper = adapter._decode_wrappers[case.num_seqs]
        assert wrapper._workspace_buffer.data_ptr() == decode_workspace.data_ptr()
        records_by_adapter.setdefault(adapter, []).append((root_ptr, decode_workspace.data_ptr()))

    assert len(records_by_adapter) == 2
    for records in records_by_adapter.values():
        assert len({root_ptr for root_ptr, _ in records}) == 1
        assert len({decode_ptr for _, decode_ptr in records}) == 1

    captured_adapter, _, _, _, _, captured_workspace = workspace_records[-1]
    captured_wrapper = captured_adapter._decode_wrappers[case.num_seqs]
    control_offset = captured_wrapper._workspace_layout.split_kv_counter.byte_offset
    control_end = captured_wrapper._workspace_layout.total_bytes
    assert torch.count_nonzero(captured_workspace[control_offset:control_end]) == 0


@pytest.mark.parametrize("use_kv_cache_manager_v2", [False, True], ids=["v1", "v2"])
def test_prims_ts_deepseek_v3_lite_mla_generation(
    monkeypatch: pytest.MonkeyPatch,
    use_kv_cache_manager_v2: bool,
) -> None:
    monkeypatch.setenv("TLLM_FMHA_LIBS", "prims_ts")
    case = BackendCase(
        **_DEEPSEEK_V3_LITE_MLA,
        seq_lens=[1, 1],
        num_cached_tokens=[64, 96],
        num_contexts=0,
        use_kv_cache_manager_v2=use_kv_cache_manager_v2,
    )

    run_case(case)


def test_prims_ts_mla_cuda_graph_replay_refreshes_query_latent_and_page_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TLLM_FMHA_LIBS", "prims_ts")
    case = BackendCase(
        **_DEEPSEEK_V3_LITE_MLA,
        seq_lens=[1, 1],
        num_cached_tokens=[64, 96],
        num_contexts=0,
        use_kv_cache_manager_v2=True,
    )
    inputs = generate_mla_gen_inputs(case, seed=0)
    replay_inputs = generate_mla_gen_inputs(case, seed=1)
    replay_inputs["cached_latent"] = inputs["cached_latent"]
    golden = run_backend(
        case,
        "VANILLA",
        replay_inputs,
        kv_dtype=case.compute_dtype,
        kv_layout="NHD",
    )

    def swap_requests_before_replay(metadata, static_inputs) -> None:
        query = static_inputs["q"]
        query.copy_(replay_inputs["fused_q"].flip(0))
        static_inputs["q_pe"].copy_(replay_inputs["q_pe"].flip(0))

        # MLA cache appends use capture-time physical destinations. Keep the
        # latent rows in place while swapping the live page-table lookup.
        replay_latent = replay_inputs["latent_cache"]
        static_inputs["latent_cache"].copy_(replay_latent)
        # The harness precomputes cache expectations as views of this tensor.
        inputs["latent_cache"].copy_(replay_latent)

        block_offsets = metadata.kv_cache_block_offsets
        block_offsets.copy_(block_offsets.flip(1).clone())

        kv_lens = metadata.kv_lens_cuda_runtime
        kv_lens.copy_(kv_lens.flip(0).clone())

    actual = run_backend(
        case,
        "TRTLLM",
        inputs,
        kv_dtype=case.compute_dtype,
        cuda_graph=True,
        kv_layout="HND",
        before_cuda_graph_replay=swap_requests_before_replay,
    )

    torch.testing.assert_close(
        actual,
        golden.flip(0),
        atol=3e-2,
        rtol=3e-3,
    )


def test_prims_ts_context_live_wrapper_cuda_graph_replay() -> None:
    from tensorrt_llm._torch.attention_backend.prims_ts import (
        BatchPrefillPagedTSWrapper,
        batch_prefill_with_paged_kv_cache,
    )

    batch_size = 2
    num_qo_heads = 8
    num_kv_heads = 2
    head_dim = 128
    page_size = 32
    max_seq_len = 64
    max_pages_per_request = 4
    dtype = torch.bfloat16
    device = torch.device("cuda")

    query = torch.randn(5, num_qo_heads, head_dim, device=device, dtype=dtype)
    k_cache = torch.randn(
        4,
        num_kv_heads,
        page_size,
        head_dim,
        device=device,
        dtype=dtype,
    )
    v_cache = torch.randn_like(k_cache)
    qo_indptr = torch.tensor([0, 3, 5], device=device, dtype=torch.int32)
    logical_kv_indptr = torch.tensor([0, 33, 97], device=device, dtype=torch.int32)
    seq_lens = torch.tensor([33, 64], device=device, dtype=torch.int32)
    dense_page_table = torch.tensor(
        [
            [[0, 1, 1, 1], [0, 1, 1, 1]],
            [[2, 3, 3, 3], [2, 3, 3, 3]],
        ],
        device=device,
        dtype=torch.int32,
    )
    output = torch.empty_like(query)
    external_workspace = torch.empty(4096, device=device, dtype=torch.uint8)
    wrapper = BatchPrefillPagedTSWrapper(
        kv_layout="HND",
        workspace_buffer=external_workspace,
    )
    wrapper.plan_live(
        query,
        k_cache,
        v_cache,
        batch_size=batch_size,
        max_seq_len_q=3,
        max_seq_len_k=max_seq_len,
        max_num_pages_per_seq_kv=max_pages_per_request,
        page_size=page_size,
        mask_type="causal",
        out_dtype=dtype,
    )
    compiled = wrapper._compiled

    wrapper.run(
        query,
        k_cache,
        v_cache,
        out=output,
        qo_indptr=qo_indptr,
        logical_kv_indptr=logical_kv_indptr,
        dense_page_idx_kv=dense_page_table,
        seq_lens_kv=seq_lens,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = wrapper.run(
            query,
            k_cache,
            v_cache,
            out=output,
            qo_indptr=qo_indptr,
            logical_kv_indptr=logical_kv_indptr,
            dense_page_idx_kv=dense_page_table,
            seq_lens_kv=seq_lens,
        )

    query.copy_(query.flip(0).clone())
    qo_indptr.copy_(torch.tensor([0, 2, 5], device=device, dtype=torch.int32))
    logical_kv_indptr.copy_(torch.tensor([0, 64, 97], device=device, dtype=torch.int32))
    seq_lens.copy_(torch.tensor([64, 33], device=device, dtype=torch.int32))
    dense_page_table.copy_(dense_page_table.flip(0).clone())

    reference = batch_prefill_with_paged_kv_cache(
        query,
        k_cache,
        v_cache,
        qo_indptr,
        torch.tensor([0, 2, 4], device=device, dtype=torch.int32),
        torch.tensor([2, 3, 0, 1], device=device, dtype=torch.int32),
        torch.tensor([32, 1], device=device, dtype=torch.int32),
        page_size=page_size,
        mask_type="causal",
        out_dtype=dtype,
    )
    graph.replay()
    torch.cuda.synchronize()

    assert graph_output.data_ptr() == output.data_ptr()
    assert wrapper._workspace_buffer is external_workspace
    assert wrapper._compiled is compiled
    torch.testing.assert_close(graph_output, reference, atol=3e-2, rtol=3e-3)


def test_prims_ts_decode_live_wrapper_cuda_graph_replay() -> None:
    from tensorrt_llm._torch.attention_backend.prims_ts import (
        BatchDecodePagedTSWrapper,
        get_prims_ts_batch_decode_workspace_size,
        prims_ts_batch_decode_with_kv_cache,
    )

    batch_size = 2
    num_qo_heads = 8
    num_kv_heads = 2
    head_dim = 128
    page_size = 32
    max_seq_len = 64
    dtype = torch.bfloat16
    device = torch.device("cuda")

    query = torch.randn(
        batch_size,
        num_qo_heads,
        head_dim,
        device=device,
        dtype=dtype,
    )
    kv_cache = torch.randn(
        4,
        2,
        num_kv_heads,
        page_size,
        head_dim,
        device=device,
        dtype=dtype,
    )
    paged_kv_indptr = torch.tensor([0, 2, 4], device=device, dtype=torch.int32)
    paged_kv_indices = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.int32)
    plan_paged_kv_indptr = paged_kv_indptr.clone()
    plan_paged_kv_indices = paged_kv_indices.clone()
    seq_lens = torch.tensor([33, 64], device=device, dtype=torch.int32)
    output = torch.empty_like(query)
    workspace_bytes = get_prims_ts_batch_decode_workspace_size(
        batch_size,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        max_seq_len,
        q_dtype=dtype,
        kv_dtype=dtype,
        out_dtype=dtype,
        mask_type="causal",
        device=device,
    )
    external_workspace = torch.zeros(
        max(workspace_bytes, query.numel() * query.element_size()),
        device=device,
        dtype=torch.uint8,
    )
    wrapper = BatchDecodePagedTSWrapper(
        kv_layout="HND",
        workspace_buffer=external_workspace,
    )
    wrapper.plan(
        plan_paged_kv_indptr,
        plan_paged_kv_indices,
        None,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        seq_len_q=1,
        q_data_type=dtype,
        kv_data_type=dtype,
        o_data_type=dtype,
        mask_type="causal",
        max_kv_len=max_seq_len,
        live_metadata=True,
    )
    compiled_main = wrapper._compiled_main
    assert plan_paged_kv_indptr.data_ptr() != paged_kv_indptr.data_ptr()
    assert plan_paged_kv_indices.data_ptr() != paged_kv_indices.data_ptr()

    aliased_query = (
        external_workspace[: query.numel() * query.element_size()].view(dtype).view_as(query)
    )
    with pytest.raises(ValueError, match="workspace_buffer must not overlap query storage"):
        wrapper.run(
            aliased_query,
            kv_cache,
            seq_lens,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            out=output,
        )

    external_workspace.zero_()
    wrapper.run(
        query,
        kv_cache,
        seq_lens,
        paged_kv_indptr=paged_kv_indptr,
        paged_kv_indices=paged_kv_indices,
        out=output,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        external_workspace.zero_()
        graph_output = wrapper.run(
            query,
            kv_cache,
            seq_lens,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            out=output,
        )

    query.copy_(query.flip(0).clone())
    paged_kv_indptr.copy_(torch.tensor([0, 1, 4], device=device, dtype=torch.int32))
    paged_kv_indices.copy_(torch.tensor([2, 0, 1, 3], device=device, dtype=torch.int32))
    seq_lens.copy_(torch.tensor([32, 64], device=device, dtype=torch.int32))
    reference_workspace = torch.zeros_like(external_workspace)
    reference = prims_ts_batch_decode_with_kv_cache(
        query,
        kv_cache,
        reference_workspace,
        paged_kv_indptr,
        paged_kv_indices,
        seq_lens,
        max_seq_len,
        out_dtype=dtype,
        out=torch.empty_like(query),
        mask_type="causal",
        kv_layout="HND",
    )
    graph.replay()
    torch.cuda.synchronize()

    assert graph_output.data_ptr() == output.data_ptr()
    assert wrapper._workspace_buffer is external_workspace
    assert wrapper._compiled_main is compiled_main
    assert wrapper._kv_prefix_mode == "dynamic"
    assert wrapper._kv_lengths_mode == "dynamic"
    torch.testing.assert_close(graph_output, reference, atol=3e-2, rtol=3e-3)


def test_prims_ts_mla_live_wrapper_cuda_graph_replay() -> None:
    from tensorrt_llm._torch.attention_backend.prims_ts import (
        BatchMLADecodePagedTSWrapper,
        get_prims_ts_batch_decode_mla_workspace_size,
        prims_ts_batch_decode_with_kv_cache_mla,
    )

    batch_size = 2
    num_heads = 32
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    page_size = 32
    max_seq_len = 1024
    dtype = torch.bfloat16
    device = torch.device("cuda")

    query = torch.randn(
        batch_size,
        1,
        num_heads,
        kv_lora_rank + qk_rope_head_dim,
        device=device,
        dtype=dtype,
    )
    kv_cache = torch.randn(
        4,
        page_size,
        kv_lora_rank + qk_rope_head_dim,
        device=device,
        dtype=dtype,
    )
    block_tables = torch.zeros(
        batch_size,
        max_seq_len // page_size,
        device=device,
        dtype=torch.int32,
    )
    block_tables[0, :2] = torch.tensor([0, 1], device=device, dtype=torch.int32)
    block_tables[1, :2] = torch.tensor([2, 3], device=device, dtype=torch.int32)
    seq_lens = torch.tensor([33, 64], device=device, dtype=torch.int32)
    plan_block_tables = block_tables.clone()
    plan_seq_lens = seq_lens.clone()
    output = torch.empty(
        batch_size,
        1,
        num_heads,
        kv_lora_rank,
        device=device,
        dtype=dtype,
    )
    workspace_bytes = get_prims_ts_batch_decode_mla_workspace_size(
        batch_size,
        num_heads,
        kv_lora_rank,
        qk_rope_head_dim,
        page_size,
        max_seq_len,
        max_seq_len_q=1,
        q_dtype=dtype,
        kv_dtype=dtype,
        out_dtype=dtype,
        mask_type="causal",
        device=device,
    )
    external_workspace = torch.empty(
        max(workspace_bytes, query.numel() * query.element_size()),
        device=device,
        dtype=torch.uint8,
    )
    wrapper = BatchMLADecodePagedTSWrapper(external_workspace)
    with pytest.raises(ValueError, match="max_kv_len is required"):
        wrapper.plan(
            plan_block_tables,
            plan_seq_lens,
            num_heads,
            kv_lora_rank,
            qk_rope_head_dim,
            page_size,
            max_seq_len_q=1,
            q_data_type=dtype,
            kv_data_type=dtype,
            o_data_type=dtype,
            mask_type="causal",
            live_metadata=True,
        )
    with pytest.raises(ValueError, match="max_seq_len_q is required"):
        wrapper.plan(
            plan_block_tables,
            plan_seq_lens,
            num_heads,
            kv_lora_rank,
            qk_rope_head_dim,
            page_size,
            qo_indptr=torch.tensor([0, 1, 2], device=device, dtype=torch.int32),
            q_data_type=dtype,
            kv_data_type=dtype,
            o_data_type=dtype,
            mask_type="causal",
            max_kv_len=max_seq_len,
            live_metadata=True,
        )
    wrapper.plan(
        plan_block_tables,
        plan_seq_lens,
        num_heads,
        kv_lora_rank,
        qk_rope_head_dim,
        page_size,
        max_seq_len_q=1,
        q_data_type=dtype,
        kv_data_type=dtype,
        o_data_type=dtype,
        mask_type="causal",
        max_kv_len=max_seq_len,
        live_metadata=True,
    )
    compiled = wrapper._compiled
    assert plan_block_tables.data_ptr() != block_tables.data_ptr()
    assert plan_seq_lens.data_ptr() != seq_lens.data_ptr()
    bmm1_scale = (128 + qk_rope_head_dim) ** -0.5

    aliased_query = (
        external_workspace[: query.numel() * query.element_size()].view(dtype).view_as(query)
    )
    with pytest.raises(ValueError, match="workspace_buffer must not overlap query storage"):
        wrapper.run(
            aliased_query,
            kv_cache,
            block_tables=block_tables,
            seq_lens=seq_lens,
            bmm1_scale=bmm1_scale,
            out=output,
        )

    wrapper.run(
        query,
        kv_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
        bmm1_scale=bmm1_scale,
        out=output,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = wrapper.run(
            query,
            kv_cache,
            block_tables=block_tables,
            seq_lens=seq_lens,
            bmm1_scale=bmm1_scale,
            out=output,
        )

    query.copy_(query.flip(0).clone())
    block_tables.copy_(block_tables.flip(0).clone())
    seq_lens.copy_(seq_lens.flip(0).clone())
    reference_workspace = torch.empty_like(external_workspace)
    reference = prims_ts_batch_decode_with_kv_cache_mla(
        query,
        kv_cache,
        reference_workspace,
        kv_lora_rank,
        qk_rope_head_dim,
        block_tables,
        seq_lens,
        max_seq_len,
        max_seq_len_q=1,
        bmm1_scale=bmm1_scale,
        out_dtype=dtype,
        out=torch.empty_like(output),
        mask_type="causal",
    )
    graph.replay()
    torch.cuda.synchronize()

    assert graph_output.data_ptr() == output.data_ptr()
    assert wrapper._workspace_buffer is external_workspace
    assert wrapper._compiled is compiled
    torch.testing.assert_close(graph_output, reference, atol=3e-2, rtol=3e-3)


def test_prims_ts_decode_graph_profiles_reset_shared_workspace_a_b_a() -> None:
    from tensorrt_llm._torch.attention_backend.prims_ts import (
        BatchDecodePagedTSWrapper,
        get_prims_ts_batch_decode_workspace_size,
        prims_ts_batch_decode_with_kv_cache,
    )

    num_qo_heads = 8
    num_kv_heads = 2
    head_dim = 128
    page_size = 32
    max_seq_len_a = 64
    max_seq_len_b = 4096
    dtype = torch.bfloat16
    device = torch.device("cuda")
    kv_cache = torch.randn(
        256,
        2,
        num_kv_heads,
        page_size,
        head_dim,
        device=device,
        dtype=dtype,
    )
    workspace_bytes = max(
        get_prims_ts_batch_decode_workspace_size(
            batch_size,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            max_seq_len,
            q_dtype=dtype,
            kv_dtype=dtype,
            out_dtype=dtype,
            mask_type="causal",
            device=device,
        )
        for batch_size, max_seq_len in (
            (1, max_seq_len_a),
            (2, max_seq_len_b),
        )
    )
    shared_workspace = torch.zeros(
        workspace_bytes,
        device=device,
        dtype=torch.uint8,
    )
    wrapper = BatchDecodePagedTSWrapper(kv_layout="HND")
    query_a = torch.randn(1, num_qo_heads, head_dim, device=device, dtype=dtype)
    query_b = torch.randn(2, num_qo_heads, head_dim, device=device, dtype=dtype)
    indptr_a = torch.tensor([0, 2], device=device, dtype=torch.int32)
    indptr_b = torch.tensor([0, 128, 256], device=device, dtype=torch.int32)
    indices_a = torch.tensor([0, 1], device=device, dtype=torch.int32)
    indices_b = torch.arange(256, device=device, dtype=torch.int32)
    seq_lens_a = torch.tensor([33], device=device, dtype=torch.int32)
    seq_lens_b = torch.tensor([2049, 4096], device=device, dtype=torch.int32)
    output_a = torch.empty_like(query_a)
    output_b = torch.empty_like(query_b)

    def plan_live(
        indptr: torch.Tensor,
        indices: torch.Tensor,
        max_seq_len: int,
    ) -> None:
        wrapper.plan(
            indptr,
            indices,
            None,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            seq_len_q=1,
            q_data_type=dtype,
            kv_data_type=dtype,
            o_data_type=dtype,
            mask_type="causal",
            max_kv_len=max_seq_len,
            live_metadata=True,
            workspace_buffer=shared_workspace,
        )

    def capture(
        query: torch.Tensor,
        indptr: torch.Tensor,
        indices: torch.Tensor,
        seq_lens: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.cuda.CUDAGraph:
        control_offset = wrapper._workspace_layout.split_kv_counter.byte_offset
        control_end = wrapper._workspace_layout.total_bytes
        shared_workspace[control_offset:control_end].zero_()
        wrapper.run(
            query,
            kv_cache,
            seq_lens,
            paged_kv_indptr=indptr,
            paged_kv_indices=indices,
            out=output,
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            shared_workspace[control_offset:control_end].zero_()
            wrapper.run(
                query,
                kv_cache,
                seq_lens,
                paged_kv_indptr=indptr,
                paged_kv_indices=indices,
                out=output,
            )
        return graph

    plan_live(indptr_a, indices_a, max_seq_len_a)
    compiled_a = wrapper._compiled_main
    layout_a = wrapper._workspace_layout
    control_span_a = slice(layout_a.split_kv_counter.byte_offset, layout_a.total_bytes)
    graph_a = capture(query_a, indptr_a, indices_a, seq_lens_a, output_a)
    plan_live(indptr_b, indices_b, max_seq_len_b)
    layout_b = wrapper._workspace_layout
    assert layout_a.split_kv_counter.byte_offset != layout_b.split_kv_counter.byte_offset
    control_span_b = slice(layout_b.split_kv_counter.byte_offset, layout_b.total_bytes)
    graph_b = capture(query_b, indptr_b, indices_b, seq_lens_b, output_b)
    plan_live(indptr_a, indices_a, max_seq_len_a)

    reference_workspace = torch.zeros_like(shared_workspace)
    reference_a = prims_ts_batch_decode_with_kv_cache(
        query_a,
        kv_cache,
        reference_workspace,
        indptr_a,
        indices_a,
        seq_lens_a,
        max_seq_len_a,
        out_dtype=dtype,
        out=torch.empty_like(output_a),
        mask_type="causal",
        kv_layout="HND",
    ).clone()
    reference_workspace.zero_()
    reference_b = prims_ts_batch_decode_with_kv_cache(
        query_b,
        kv_cache,
        reference_workspace,
        indptr_b,
        indices_b,
        seq_lens_b,
        max_seq_len_b,
        out_dtype=dtype,
        out=torch.empty_like(output_b),
        mask_type="causal",
        kv_layout="HND",
    ).clone()

    actual_a = []
    shared_workspace[control_span_a].fill_(0xFF)
    graph_a.replay()
    torch.cuda.synchronize()
    assert torch.count_nonzero(shared_workspace[control_span_a]) == 0
    actual_a.append(output_a.clone())
    shared_workspace[control_span_b].fill_(0xFF)
    graph_b.replay()
    torch.cuda.synchronize()
    assert torch.count_nonzero(shared_workspace[control_span_b]) == 0
    actual_b = output_b.clone()
    shared_workspace[control_span_a].fill_(0xFF)
    graph_a.replay()
    torch.cuda.synchronize()
    assert torch.count_nonzero(shared_workspace[control_span_a]) == 0
    actual_a.append(output_a.clone())

    assert wrapper._workspace_buffer is shared_workspace
    assert wrapper._compiled_main is compiled_a
    for actual in actual_a:
        torch.testing.assert_close(actual, reference_a, atol=3e-2, rtol=3e-3)
    torch.testing.assert_close(actual_b, reference_b, atol=3e-2, rtol=3e-3)


def test_prims_ts_decode_wrappers_share_workspace_across_serialized_layers() -> None:
    from tensorrt_llm._torch.attention_backend.prims_ts import (
        BatchDecodePagedTSWrapper,
        get_prims_ts_batch_decode_workspace_size,
        prims_ts_batch_decode_with_kv_cache,
    )

    batch_size = 2
    num_qo_heads = 8
    num_kv_heads = 2
    head_dim = 128
    page_size = 32
    max_seq_len = 64
    dtype = torch.bfloat16
    device = torch.device("cuda")
    workspace_bytes = get_prims_ts_batch_decode_workspace_size(
        batch_size,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        max_seq_len,
        q_dtype=dtype,
        kv_dtype=dtype,
        out_dtype=dtype,
        mask_type="causal",
        device=device,
    )
    shared_workspace = torch.zeros(
        workspace_bytes,
        device=device,
        dtype=torch.uint8,
    )
    paged_kv_indptr = torch.tensor([0, 2, 4], device=device, dtype=torch.int32)
    paged_kv_indices = torch.tensor([0, 1, 2, 3], device=device, dtype=torch.int32)
    seq_lens = torch.tensor([33, 64], device=device, dtype=torch.int32)
    wrappers = [
        BatchDecodePagedTSWrapper(
            kv_layout="HND",
            workspace_buffer=shared_workspace,
        )
        for _ in range(2)
    ]
    for wrapper in wrappers:
        wrapper.plan(
            paged_kv_indptr,
            paged_kv_indices,
            None,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            seq_len_q=1,
            q_data_type=dtype,
            kv_data_type=dtype,
            o_data_type=dtype,
            mask_type="causal",
            max_kv_len=max_seq_len,
            live_metadata=True,
        )

    for layer_index, wrapper in enumerate(wrappers):
        query = torch.randn(
            batch_size,
            num_qo_heads,
            head_dim,
            device=device,
            dtype=dtype,
        )
        kv_cache = torch.randn(
            4,
            2,
            num_kv_heads,
            page_size,
            head_dim,
            device=device,
            dtype=dtype,
        )
        output = torch.empty_like(query)
        control_offset = wrapper._workspace_layout.split_kv_counter.byte_offset
        shared_workspace[control_offset : wrapper._workspace_layout.total_bytes].zero_()
        actual = wrapper.run(
            query,
            kv_cache,
            seq_lens,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            out=output,
        )
        reference_workspace = torch.zeros_like(shared_workspace)
        reference = prims_ts_batch_decode_with_kv_cache(
            query,
            kv_cache,
            reference_workspace,
            paged_kv_indptr,
            paged_kv_indices,
            seq_lens,
            max_seq_len,
            out_dtype=dtype,
            out=torch.empty_like(query),
            mask_type="causal",
            kv_layout="HND",
        )

        assert wrapper._workspace_buffer.data_ptr() == shared_workspace.data_ptr(), layer_index
        torch.testing.assert_close(actual, reference, atol=3e-2, rtol=3e-3)


def test_prims_ts_unsupported_context_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    from tensorrt_llm._torch.attention_backend.fmha.fallback import FallbackFmha
    from tensorrt_llm._torch.attention_backend.fmha.prims_ts import PrimsTSFmha

    calls = {"fallback": 0, "prims_context": 0}
    fallback_forward = FallbackFmha.forward
    prims_context = PrimsTSFmha.run_context

    def counted_fallback(self, *args, **kwargs):
        calls["fallback"] += 1
        return fallback_forward(self, *args, **kwargs)

    def counted_prims_context(self, *args, **kwargs):
        calls["prims_context"] += 1
        return prims_context(self, *args, **kwargs)

    monkeypatch.setattr(FallbackFmha, "forward", counted_fallback)
    monkeypatch.setattr(PrimsTSFmha, "run_context", counted_prims_context)
    monkeypatch.setenv("TLLM_FMHA_LIBS", "prims_ts,fallback")
    case = BackendCase(
        num_heads=14,
        num_kv_heads=2,
        head_dim=64,
        seq_lens=[65, 37],
        num_cached_tokens=[0, 0],
        num_contexts=2,
        dtype="bfloat16",
        kv_layout="HND",
        page_size=32,
    )

    run_case(case)

    assert calls["fallback"] > 0
    assert calls["prims_context"] == 0
