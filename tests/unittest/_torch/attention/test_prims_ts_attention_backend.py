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

import functools
import inspect

import pytest
import torch
from backend_case import BackendCase, generate_inputs, run_backend, run_case
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


@pytest.mark.parametrize("use_kv_cache_manager_v2", [False, True], ids=["v1", "v2"])
def test_prims_ts_context_zero_fills_nan_v_tail(
    monkeypatch: pytest.MonkeyPatch,
    use_kv_cache_manager_v2: bool,
) -> None:
    from tensorrt_llm._torch.attention_backend.prims_ts.context import BatchPrefillPagedTSWrapper

    original_run = BatchPrefillPagedTSWrapper.run
    poisoned_tails: list[tuple[int, int]] = []

    def run_with_nan_v_tail(
        self: BatchPrefillPagedTSWrapper,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        qo_indptr: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens_kv: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        # Poison after TRT-LLM writes valid KV and immediately before PrimTS
        # launches, so only the kernel's handling of unused V rows is tested.
        page_size = v_cache.shape[2]
        seq_lens = seq_lens_kv.cpu().tolist()
        page_indices = block_tables.cpu()
        for batch_idx, seq_len in enumerate(seq_lens):
            logical_last_page = (seq_len - 1) // page_size
            tail_start = (seq_len - 1) % page_size + 1
            if tail_start == page_size:
                continue
            physical_page = int(page_indices[batch_idx, logical_last_page].item())
            v_cache[physical_page, :, tail_start:, :].fill_(float("nan"))
            poisoned_tails.append((batch_idx, tail_start))

        return original_run(
            self,
            q,
            k_cache,
            v_cache,
            qo_indptr,
            block_tables,
            seq_lens_kv,
            **kwargs,
        )

    monkeypatch.setattr(BatchPrefillPagedTSWrapper, "run", run_with_nan_v_tail)
    monkeypatch.setenv("TLLM_FMHA_LIBS", "prims_ts")
    case = BackendCase(
        **_QWEN2_7B,
        seq_lens=[65, 37],
        num_cached_tokens=[0, 0],
        num_contexts=2,
        use_kv_cache_manager_v2=use_kv_cache_manager_v2,
    )

    results = run_case(case)

    assert "TRTLLM" in results
    assert poisoned_tails == [(0, 1), (1, 5)]


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


def test_prims_ts_uses_compact_preprocessing_and_separate_decode_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensorrt_llm._torch.attention_backend.fmha.prims_ts as prims_ts_module
    from tensorrt_llm._torch.attention_backend.fmha.prims_ts import PrimsTSFmha
    from tensorrt_llm._torch.attention_backend.prims_ts import (
        get_prims_ts_batch_decode_workspace_size,
    )

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
    prims_workspace_bytes = get_prims_ts_batch_decode_workspace_size(
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

    def record_compact_layout(*args, **kwargs):
        layout = original_generation_layout(*args, **kwargs)
        generation_layout_records.append((args, kwargs, dict(layout)))
        return layout

    monkeypatch.setattr(
        prims_ts_module.thop,
        "get_trtllm_gen_generation_workspace_layout",
        record_compact_layout,
    )

    original_get_decode_workspace = PrimsTSFmha._get_decode_workspace
    workspace_records = []

    def record_decode_workspace(
        self: PrimsTSFmha,
        root_workspace: torch.Tensor,
    ) -> torch.Tensor:
        decode_workspace = original_get_decode_workspace(self, root_workspace)
        workspace_records.append(
            (
                self,
                root_workspace.data_ptr(),
                root_workspace.numel() * root_workspace.element_size(),
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
    actual = run_backend(
        case,
        "TRTLLM",
        inputs,
        kv_dtype=case.compute_dtype,
        fuse_rope=True,
        cuda_graph=True,
        kv_layout="HND",
    )

    torch.testing.assert_close(eager, golden, atol=3e-2, rtol=3e-3)
    torch.testing.assert_close(actual, golden, atol=3e-2, rtol=3e-3)

    assert generation_layout_records
    compact_preprocess_bytes = int(generation_layout_records[0][2]["total_size"])
    assert all(
        kwargs["skip_fmha_workspace"] is True
        for _args, kwargs, _layout in generation_layout_records
    )
    assert all(
        int(layout["trtllm_gen_workspace_size"]) == 0
        for _args, _kwargs, layout in generation_layout_records
    )

    records_by_adapter = {}
    for adapter, root_ptr, root_bytes, decode_workspace in workspace_records:
        byte_offset = adapter._decode_workspace_offset_bytes
        required_bytes = adapter._decode_workspace_required_bytes
        assert byte_offset is not None
        assert byte_offset % 32 == 0
        assert byte_offset >= compact_preprocess_bytes
        assert required_bytes == prims_workspace_bytes
        assert root_bytes >= byte_offset + required_bytes
        assert decode_workspace.data_ptr() == root_ptr + byte_offset
        assert decode_workspace.numel() * decode_workspace.element_size() == required_bytes
        wrapper = adapter._decode_wrappers[case.num_seqs]
        plan_state = wrapper._plan_state
        assert plan_state is not None
        assert plan_state.workspace_buffer.data_ptr() == decode_workspace.data_ptr()
        records_by_adapter.setdefault(adapter, []).append((root_ptr, decode_workspace.data_ptr()))

    assert len(records_by_adapter) == 2
    for records in records_by_adapter.values():
        assert len({root_ptr for root_ptr, _ in records}) == 1
        assert len({decode_ptr for _, decode_ptr in records}) == 1

    captured_adapter, _, _, captured_workspace = workspace_records[-1]
    captured_wrapper = captured_adapter._decode_wrappers[case.num_seqs]
    captured_plan_state = captured_wrapper._plan_state
    assert captured_plan_state is not None
    assert torch.count_nonzero(captured_plan_state.workspace.split_kv_counter) == 0


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


def test_prims_ts_context_wrapper_cuda_graph_replay_with_updated_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensorrt_llm._torch.attention_backend.prims_ts.context as context_module
    from tensorrt_llm._torch.attention_backend.prims_ts import (
        BatchPrefillPagedTSWrapper,
        batch_prefill_with_paged_kv_cache,
    )

    # Isolate this regression from compile-cache entries created by earlier
    # tests while retaining each compiled module through the full A/E/F
    # sequence. The former ragged V tensor map aborted on SM100 only after a
    # CLC D128 -> nonpersistent D256 -> distinct CLC D128 compile/run order.
    uncached_compile = inspect.unwrap(context_module._get_compiled_paged_context)
    compile_records = []

    @functools.cache
    def compile_with_record(*args):
        result = uncached_compile(*args)
        compiled, policy = result
        compile_records.append((args, dict(policy), compiled))
        return result

    monkeypatch.setattr(context_module, "_get_compiled_paged_context", compile_with_record)
    monkeypatch.setenv("TLLM_FMHA_LIBS", "prims_ts")

    a_results = run_case(
        BackendCase(
            **_QWEN2_7B,
            seq_lens=[65, 37],
            num_cached_tokens=[0, 0],
            num_contexts=2,
            use_kv_cache_manager_v2=False,
        )
    )
    assert "TRTLLM" in a_results

    e_results = run_case(
        BackendCase(
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
    )
    assert "TRTLLM" in e_results

    batch_size = 2
    num_qo_heads = 8
    num_kv_heads = 2
    head_dim = 128
    page_size = 32
    max_seq_len = 64
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
    trt_block_tables = torch.tensor(
        [[[0, 1], [2, 3]], [[2, 3], [0, 1]]],
        device=device,
        dtype=torch.int32,
    )
    block_tables = trt_block_tables[:, 0, :]
    assert block_tables.stride() == (4, 1)
    seq_lens = torch.tensor([33, 64], device=device, dtype=torch.int32)
    output = torch.empty_like(query)
    wrapper = BatchPrefillPagedTSWrapper(kv_layout="HND")
    wrapper.plan(
        device=device,
        batch_size=batch_size,
        max_seq_len_q=3,
        max_kv_len=max_seq_len,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_dtype=dtype,
        kv_dtype=dtype,
        out_dtype=dtype,
        page_size=page_size,
        mask_type="causal",
    )
    plan_state = wrapper._plan_state
    assert plan_state is not None
    compiled = plan_state.compiled

    assert len(compile_records) == 3
    assert [
        (args[0].max_seq_len_q, args[0].max_kv_len, policy["scheduler"])
        for args, policy, _compiled in compile_records
    ] == [
        (96, 96, "clc_dynamic_persistent"),
        (128, 128, "nonpersistent"),
        (3, 64, "clc_dynamic_persistent"),
    ]
    assert compile_records[0][0] != compile_records[2][0]
    assert len({id(recorded) for _args, _policy, recorded in compile_records}) == 3

    wrapper.run(
        query,
        k_cache,
        v_cache,
        qo_indptr,
        block_tables,
        seq_lens,
        out=output,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = wrapper.run(
            query,
            k_cache,
            v_cache,
            qo_indptr,
            block_tables,
            seq_lens,
            out=output,
            validate=False,
        )

    query.copy_(query.flip(0).clone())
    qo_indptr.copy_(torch.tensor([0, 2, 5], device=device, dtype=torch.int32))
    seq_lens.copy_(torch.tensor([64, 33], device=device, dtype=torch.int32))
    block_tables.copy_(torch.tensor([[2, 3], [0, 1]], device=device, dtype=torch.int32))

    reference = batch_prefill_with_paged_kv_cache(
        query,
        k_cache,
        v_cache,
        qo_indptr,
        block_tables,
        seq_lens,
        page_size=page_size,
        mask_type="causal",
        out_dtype=dtype,
    )
    graph.replay()
    torch.cuda.synchronize()

    assert graph_output.data_ptr() == output.data_ptr()
    assert wrapper._plan_state is plan_state
    assert wrapper._plan_state.compiled is compiled
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
    trt_block_tables = torch.tensor(
        [[[0, 1], [2, 3]], [[2, 3], [0, 1]]],
        device=device,
        dtype=torch.int32,
    )
    block_tables = trt_block_tables[:, 0, :]
    assert block_tables.stride() == (4, 1)
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
    wrapper = BatchDecodePagedTSWrapper(kv_layout="HND")
    wrapper.plan(
        device,
        batch_size,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        max_seq_len,
        max_seq_len_q=1,
        packed_query=False,
        q_data_type=dtype,
        kv_data_type=dtype,
        o_data_type=dtype,
        mask_type="causal",
        workspace_buffer=external_workspace,
    )
    plan_state = wrapper._plan_state
    assert plan_state is not None
    compiled_main = plan_state.compiled_main

    aliased_query = (
        external_workspace[: query.numel() * query.element_size()].view(dtype).view_as(query)
    )
    with pytest.raises(ValueError, match="workspace_buffer must not overlap query storage"):
        wrapper.run(
            aliased_query,
            kv_cache,
            seq_lens,
            block_tables,
            out=output,
        )

    external_workspace.zero_()
    wrapper.run(
        query,
        kv_cache,
        seq_lens,
        block_tables,
        out=output,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        external_workspace.zero_()
        graph_output = wrapper.run(
            query,
            kv_cache,
            seq_lens,
            block_tables,
            out=output,
            validate=False,
        )

    query.copy_(query.flip(0).clone())
    block_tables.copy_(torch.tensor([[2, 3], [0, 1]], device=device, dtype=torch.int32))
    seq_lens.copy_(torch.tensor([32, 64], device=device, dtype=torch.int32))
    reference_workspace = torch.zeros_like(external_workspace)
    reference = prims_ts_batch_decode_with_kv_cache(
        query,
        kv_cache,
        reference_workspace,
        block_tables,
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
    assert wrapper._plan_state is plan_state
    assert plan_state.workspace_buffer is external_workspace
    assert plan_state.compiled_main is compiled_main
    assert plan_state.kv_prefix_mode == "dynamic"
    assert plan_state.kv_lengths_mode == "dynamic"
    torch.testing.assert_close(graph_output, reference, atol=3e-2, rtol=3e-3)


def test_prims_ts_mla_live_wrapper_cuda_graph_replay() -> None:
    from tensorrt_llm._torch.attention_backend.prims_ts import (
        BatchMLADecodePagedTSWrapper,
        get_prims_ts_batch_mla_decode_workspace_size,
        prims_ts_batch_mla_decode_with_kv_cache,
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
    output = torch.empty(
        batch_size,
        1,
        num_heads,
        kv_lora_rank,
        device=device,
        dtype=dtype,
    )
    workspace_bytes = get_prims_ts_batch_mla_decode_workspace_size(
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
    wrapper = BatchMLADecodePagedTSWrapper()
    wrapper.plan(
        device,
        batch_size,
        num_heads,
        kv_lora_rank,
        qk_rope_head_dim,
        page_size,
        max_seq_len,
        max_seq_len_q=1,
        packed_query=False,
        q_data_type=dtype,
        kv_data_type=dtype,
        o_data_type=dtype,
        mask_type="causal",
        workspace_buffer=external_workspace,
    )
    plan_state = wrapper._plan_state
    assert plan_state is not None
    compiled = plan_state.compiled
    bmm1_scale = (128 + qk_rope_head_dim) ** -0.5

    aliased_query = (
        external_workspace[: query.numel() * query.element_size()].view(dtype).view_as(query)
    )
    with pytest.raises(ValueError, match="workspace_buffer must not overlap query storage"):
        wrapper.run(
            aliased_query,
            kv_cache,
            block_tables,
            seq_lens,
            bmm1_scale=bmm1_scale,
            out=output,
        )

    wrapper.run(
        query,
        kv_cache,
        block_tables,
        seq_lens,
        bmm1_scale=bmm1_scale,
        out=output,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = wrapper.run(
            query,
            kv_cache,
            block_tables,
            seq_lens,
            bmm1_scale=bmm1_scale,
            out=output,
            validate=False,
        )

    query.copy_(query.flip(0).clone())
    block_tables.copy_(block_tables.flip(0).clone())
    seq_lens.copy_(seq_lens.flip(0).clone())
    reference_workspace = torch.empty_like(external_workspace)
    reference = prims_ts_batch_mla_decode_with_kv_cache(
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
    assert wrapper._plan_state is plan_state
    assert plan_state.workspace_buffer is external_workspace
    assert plan_state.compiled is compiled
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
    block_tables_a = torch.tensor([[0, 1]], device=device, dtype=torch.int32)
    block_tables_b = torch.arange(256, device=device, dtype=torch.int32).view(2, 128)
    seq_lens_a = torch.tensor([33], device=device, dtype=torch.int32)
    seq_lens_b = torch.tensor([2049, 4096], device=device, dtype=torch.int32)
    output_a = torch.empty_like(query_a)
    output_b = torch.empty_like(query_b)

    def plan(batch_size: int, max_seq_len: int) -> None:
        wrapper.plan(
            device,
            batch_size,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            max_seq_len,
            max_seq_len_q=1,
            packed_query=False,
            q_data_type=dtype,
            kv_data_type=dtype,
            o_data_type=dtype,
            mask_type="causal",
            workspace_buffer=shared_workspace,
        )

    def capture(
        query: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.cuda.CUDAGraph:
        plan_state = wrapper._plan_state
        assert plan_state is not None
        control_offset = plan_state.workspace_layout.split_kv_counter.byte_offset
        control_end = plan_state.workspace_layout.total_bytes
        shared_workspace[control_offset:control_end].zero_()
        wrapper.run(
            query,
            kv_cache,
            seq_lens,
            block_tables,
            out=output,
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            shared_workspace[control_offset:control_end].zero_()
            wrapper.run(
                query,
                kv_cache,
                seq_lens,
                block_tables,
                out=output,
                validate=False,
            )
        return graph

    plan(1, max_seq_len_a)
    plan_state_a = wrapper._plan_state
    assert plan_state_a is not None
    compiled_a = plan_state_a.compiled_main
    layout_a = plan_state_a.workspace_layout
    control_span_a = slice(layout_a.split_kv_counter.byte_offset, layout_a.total_bytes)
    graph_a = capture(query_a, block_tables_a, seq_lens_a, output_a)
    plan(2, max_seq_len_b)
    plan_state_b = wrapper._plan_state
    assert plan_state_b is not None
    layout_b = plan_state_b.workspace_layout
    assert layout_a.split_kv_counter.byte_offset != layout_b.split_kv_counter.byte_offset
    control_span_b = slice(layout_b.split_kv_counter.byte_offset, layout_b.total_bytes)
    graph_b = capture(query_b, block_tables_b, seq_lens_b, output_b)
    plan(1, max_seq_len_a)

    reference_workspace = torch.zeros_like(shared_workspace)
    reference_a = prims_ts_batch_decode_with_kv_cache(
        query_a,
        kv_cache,
        reference_workspace,
        block_tables_a,
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
        block_tables_b,
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

    final_plan_state = wrapper._plan_state
    assert final_plan_state is not None
    assert final_plan_state.workspace_buffer is shared_workspace
    assert final_plan_state.compiled_main is compiled_a
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
    trt_block_tables = torch.tensor(
        [[[0, 1], [2, 3]], [[2, 3], [0, 1]]],
        device=device,
        dtype=torch.int32,
    )
    block_tables = trt_block_tables[:, 0, :]
    seq_lens = torch.tensor([33, 64], device=device, dtype=torch.int32)
    wrappers = [BatchDecodePagedTSWrapper(kv_layout="HND") for _ in range(2)]
    for wrapper in wrappers:
        wrapper.plan(
            device,
            batch_size,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            max_seq_len,
            max_seq_len_q=1,
            packed_query=False,
            q_data_type=dtype,
            kv_data_type=dtype,
            o_data_type=dtype,
            mask_type="causal",
            workspace_buffer=shared_workspace,
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
        plan_state = wrapper._plan_state
        assert plan_state is not None
        control_offset = plan_state.workspace_layout.split_kv_counter.byte_offset
        shared_workspace[control_offset : plan_state.workspace_layout.total_bytes].zero_()
        actual = wrapper.run(
            query,
            kv_cache,
            seq_lens,
            block_tables,
            out=output,
        )
        reference_workspace = torch.zeros_like(shared_workspace)
        reference = prims_ts_batch_decode_with_kv_cache(
            query,
            kv_cache,
            reference_workspace,
            block_tables,
            seq_lens,
            max_seq_len,
            out_dtype=dtype,
            out=torch.empty_like(query),
            mask_type="causal",
            kv_layout="HND",
        )

        assert plan_state.workspace_buffer.data_ptr() == shared_workspace.data_ptr(), layer_index
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
