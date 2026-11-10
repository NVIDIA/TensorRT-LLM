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
import math
from types import SimpleNamespace
from unittest.mock import Mock

import backend_case
import pytest
import torch
from backend_case import (
    BF16_ATOL,
    BF16_RTOL,
    FP8_ATOL,
    BackendCase,
    generate_inputs,
    run_backend,
    run_case,
)
from utils.util import getSMVersion

pytestmark = pytest.mark.skipif(
    getSMVersion() not in (100, 103, 107),
    reason="PrimsTS attention kernels require SM100, SM103, or SM107",
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


@pytest.mark.parametrize(
    (
        "seq_lens",
        "num_cached_tokens",
        "num_contexts",
        "dtype",
        "kv_dtype",
        "use_kv_cache_manager_v2",
    ),
    [
        pytest.param([65, 37], [0, 0], 2, "bfloat16", None, False, id="native-context-v1"),
        pytest.param([65, 37], [0, 0], 2, "bfloat16", None, True, id="native-context-v2"),
        pytest.param([1, 1], [64, 96], 0, "bfloat16", None, False, id="native-generation-v1"),
        pytest.param([1, 1], [64, 96], 0, "bfloat16", None, True, id="native-generation-v2"),
        pytest.param([41, 1], [0, 63], 1, "bfloat16", None, False, id="native-mixed-v1"),
        pytest.param([41, 1], [0, 63], 1, "bfloat16", None, True, id="native-mixed-v2"),
        pytest.param([65, 37], [0, 0], 2, "bfloat16", "fp8", False, id="fp8-context-bf16-v1"),
        pytest.param([65, 37], [0, 0], 2, "float16", "fp8", True, id="fp8-context-fp16-v2"),
        pytest.param([1, 1], [64, 96], 0, "float16", "fp8", False, id="fp8-generation-fp16-v1"),
        pytest.param(
            [1, 1], [8191, 4096], 0, "bfloat16", "fp8", True, id="fp8-long-generation-bf16-v2"
        ),
        pytest.param([41, 1], [0, 63], 1, "bfloat16", "fp8", False, id="fp8-mixed-bf16-v1"),
        pytest.param([41, 1], [0, 63], 1, "float16", "fp8", True, id="fp8-mixed-fp16-v2"),
    ],
)
def test_prims_ts_qwen2_gqa(
    monkeypatch: pytest.MonkeyPatch,
    use_kv_cache_manager_v2: bool,
    seq_lens: list[int],
    num_cached_tokens: list[int],
    num_contexts: int,
    kv_dtype: str | None,
    dtype: str,
) -> None:
    monkeypatch.setenv("TLLM_FMHA_LIBS", "prims_ts")
    if kv_dtype == "fp8":
        monkeypatch.setattr(backend_case, "BACKENDS_UNDER_TEST", ("TRTLLM",))
    case = BackendCase(
        **{**_QWEN2_7B, "dtype": dtype},
        seq_lens=seq_lens,
        num_cached_tokens=num_cached_tokens,
        num_contexts=num_contexts,
        kv_dtype=kv_dtype,
        use_kv_cache_manager_v2=use_kv_cache_manager_v2,
    )

    run_case(case)


@pytest.mark.parametrize(
    (
        "dtype",
        "preprocessing",
        "head_dim",
        "use_kv_cache_manager_v2",
        "num_cached_tokens",
        "kv_dtype",
    ),
    [
        pytest.param("bfloat16", "cached-rope", 128, False, [0, 0], None, id="native-v1"),
        pytest.param("bfloat16", "cached-rope", 128, True, [0, 0], None, id="native-v2"),
        pytest.param(
            "bfloat16",
            "cached-rope",
            128,
            False,
            [0, 0],
            "fp8",
            id="fp8-bf16-cached-rope-d128-v1-context",
        ),
        pytest.param(
            "bfloat16",
            "cached-rope",
            256,
            True,
            [32, 64],
            "fp8",
            id="fp8-bf16-cached-rope-d256-v2-prefix",
        ),
        pytest.param(
            "bfloat16",
            "computed-rope",
            128,
            True,
            [0, 0],
            "fp8",
            id="fp8-bf16-computed-rope-d128-v2-context",
        ),
        pytest.param(
            "bfloat16",
            "computed-rope",
            256,
            False,
            [32, 64],
            "fp8",
            id="fp8-bf16-computed-rope-d256-v1-prefix",
        ),
        pytest.param(
            "float16",
            "cached-rope",
            128,
            True,
            [32, 64],
            "fp8",
            id="fp8-fp16-cached-rope-d128-v2-prefix",
        ),
        pytest.param(
            "float16",
            "cached-rope",
            256,
            False,
            [0, 0],
            "fp8",
            id="fp8-fp16-cached-rope-d256-v1-context",
        ),
        pytest.param(
            "float16",
            "computed-rope",
            128,
            False,
            [32, 64],
            "fp8",
            id="fp8-fp16-computed-rope-d128-v1-prefix",
        ),
        pytest.param(
            "float16",
            "computed-rope",
            256,
            True,
            [0, 0],
            "fp8",
            id="fp8-fp16-computed-rope-d256-v2-context",
        ),
    ],
)
def test_prims_ts_context_receives_zeroed_v_tail(
    monkeypatch: pytest.MonkeyPatch,
    use_kv_cache_manager_v2: bool,
    kv_dtype: str | None,
    num_cached_tokens: list[int],
    dtype: str,
    preprocessing: str,
    head_dim: int,
) -> None:
    from tensorrt_llm._torch.attention.backends.fmha.prims_ts import PrimsTSFmha
    from tensorrt_llm._torch.attention.backends.prims_ts.context import BatchPrefillPagedTSWrapper

    original_run_context = PrimsTSFmha.run_context

    def run_context_with_selected_preprocessing(self, params):
        # An absent cosine/sine table selects the preprocessing kernel that
        # computes RoPE coefficients, with the same RoPE semantics.
        with monkeypatch.context() as context_patch:
            if preprocessing == "computed-rope":
                context_patch.setattr(self.attn, "rotary_cos_sin", None)
            return original_run_context(self, params)

    original_build_kv_cache_manager = backend_case._build_kv_cache_manager
    original_fill_kv_cache = backend_case.fill_kv_cache_logical
    original_generate_inputs = backend_case.generate_inputs
    trt_managers: set[int] = set()
    inputs_for_check: dict = {}

    def generate_inputs(*args, **kwargs):
        inputs = original_generate_inputs(*args, **kwargs)
        # Bound RoPE operands so FP8 rounding at a bin boundary stays within
        # the cache oracle's tolerance. Valid V values remain unmodified.
        for tensor in (inputs["q"], inputs["new_k"], *inputs["cached_k"]):
            tensor.clamp_(-1, 1)
        inputs_for_check.update(inputs)
        return inputs

    def fill_cache_and_poison_tail(manager, layer_idx, request_ids, *args, **kwargs):
        original_fill_kv_cache(manager, layer_idx, request_ids, *args, **kwargs)
        if id(manager) not in trt_managers:
            return
        pool = manager.get_buffers(layer_idx, kv_layout="HND")
        page_size = pool.shape[3]
        block_indices = manager.get_batch_cache_indices(list(request_ids), layer_idx)
        for batch_idx, seq_len in enumerate(case.token_nums):
            tail_start = (seq_len - 1) % page_size + 1
            physical_page = int(block_indices[batch_idx][(seq_len - 1) // page_size])
            pool[physical_page, 1, :, tail_start:, :].fill_(float("nan"))

    original_run = BatchPrefillPagedTSWrapper.run
    checked_tails: list[tuple[int, int]] = []
    plan_flags: list[bool] = []
    original_plan = BatchPrefillPagedTSWrapper.plan

    def record_plan(self: BatchPrefillPagedTSWrapper, *args, **kwargs) -> None:
        plan_flags.append(kwargs["paged_v_tail_is_zero"])
        original_plan(self, *args, **kwargs)

    def build_kv_cache_manager_with_nan_v(
        case: BackendCase,
        backend: str,
        kv_dtype: torch.dtype,
    ) -> object:
        manager = original_build_kv_cache_manager(case, backend, kv_dtype)
        if backend == "TRTLLM":
            trt_managers.add(id(manager))
        return manager

    def run_with_zeroed_v_tail_check(
        self: BatchPrefillPagedTSWrapper,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        qo_indptr: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens_kv: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor:
        # Check the producer's zero-tail contract before the consumer runs,
        # including preservation of every valid V element in the last page.
        page_size = v_cache.shape[2]
        seq_lens = seq_lens_kv.cpu().tolist()
        page_indices = block_tables.cpu()
        for batch_idx, seq_len in enumerate(seq_lens):
            logical_last_page = (seq_len - 1) // page_size
            tail_start = (seq_len - 1) % page_size + 1
            if tail_start == page_size:
                continue
            physical_page = int(page_indices[batch_idx, logical_last_page].item())
            tail = v_cache[physical_page, :, tail_start:, :]
            assert torch.count_nonzero(tail.float()).item() == 0
            new_v_start = sum(case.seq_lens[:batch_idx])
            new_v = inputs_for_check["new_v"][new_v_start : new_v_start + case.seq_lens[batch_idx]]
            logical_v = torch.cat(
                (
                    inputs_for_check["cached_v"][batch_idx],
                    new_v.view(-1, case.num_kv_heads, case.head_dim),
                )
            )
            expected = logical_v[logical_last_page * page_size :].to(v_cache.dtype)
            actual = v_cache[physical_page, :, :tail_start, :].transpose(0, 1)
            torch.testing.assert_close(actual.float(), expected.float(), rtol=0, atol=0)
            checked_tails.append((batch_idx, tail_start))

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

    monkeypatch.setattr(backend_case, "BACKENDS_UNDER_TEST", ("TRTLLM",))
    monkeypatch.setattr(PrimsTSFmha, "run_context", run_context_with_selected_preprocessing)
    monkeypatch.setattr(backend_case, "generate_inputs", generate_inputs)
    monkeypatch.setattr(backend_case, "fill_kv_cache_logical", fill_cache_and_poison_tail)
    monkeypatch.setattr(
        backend_case,
        "_build_kv_cache_manager",
        build_kv_cache_manager_with_nan_v,
    )
    monkeypatch.setattr(
        BatchPrefillPagedTSWrapper,
        "run",
        run_with_zeroed_v_tail_check,
    )
    monkeypatch.setattr(BatchPrefillPagedTSWrapper, "plan", record_plan)
    monkeypatch.setenv("TLLM_FMHA_LIBS", "prims_ts")
    rope = dict(_QWEN2_7B["rope"])
    rope["dim"] = head_dim
    case = BackendCase(
        **{**_QWEN2_7B, "head_dim": head_dim, "dtype": dtype, "rope": rope},
        seq_lens=[65, 37],
        num_cached_tokens=num_cached_tokens,
        kv_dtype=kv_dtype,
        num_contexts=2,
        use_kv_cache_manager_v2=use_kv_cache_manager_v2,
    )

    results = run_case(case)

    assert "TRTLLM" in results
    assert torch.isfinite(results["TRTLLM"]).all()
    assert checked_tails == [(0, 1), (1, 5)]
    assert plan_flags == [True]


@pytest.mark.parametrize(
    ("dtype", "kv_dtype"),
    [("bfloat16", None), ("float16", "fp8")],
    ids=["native-bf16", "fp8-fp16"],
)
def test_prims_ts_decode_with_computed_rope(
    monkeypatch: pytest.MonkeyPatch, dtype: str, kv_dtype: str | None
) -> None:
    from tensorrt_llm._torch.attention.backends.fmha.prims_ts import PrimsTSFmha

    original_run_generation = PrimsTSFmha.run_generation

    def run_generation_without_rope_table(self, params):
        with monkeypatch.context() as context_patch:
            context_patch.setattr(self.attn, "rotary_cos_sin", None)
            return original_run_generation(self, params)

    monkeypatch.setattr(PrimsTSFmha, "run_generation", run_generation_without_rope_table)
    monkeypatch.setenv("TLLM_FMHA_LIBS", "prims_ts")
    monkeypatch.setattr(backend_case, "BACKENDS_UNDER_TEST", ("TRTLLM",))
    run_case(
        BackendCase(
            **{**_QWEN2_7B, "dtype": dtype},
            seq_lens=[1, 1],
            num_cached_tokens=[65, 37],
            num_contexts=0,
            kv_dtype=kv_dtype,
            use_kv_cache_manager_v2=True,
        )
    )


@pytest.mark.parametrize("kv_dtype", [None, "fp8"], ids=["fp16-kv", "fp8-kv"])
def test_prims_ts_fp16_dense_context_with_alternate_shape(
    monkeypatch: pytest.MonkeyPatch,
    kv_dtype: str | None,
) -> None:
    monkeypatch.setenv("TLLM_FMHA_LIBS", "prims_ts")
    if kv_dtype == "fp8":
        monkeypatch.setattr(backend_case, "BACKENDS_UNDER_TEST", ("TRTLLM",))
    case = BackendCase(
        num_heads=8,
        num_kv_heads=2,
        head_dim=256,
        seq_lens=[67, 33],
        num_cached_tokens=[0, 0],
        num_contexts=2,
        dtype="float16",
        kv_dtype=kv_dtype,
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
    import tensorrt_llm._torch.attention.backends.fmha.prims_ts as prims_ts_module
    from tensorrt_llm._torch.attention.backends.fmha.prims_ts import PrimsTSFmha
    from tensorrt_llm._torch.attention.backends.prims_ts import (
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


@pytest.mark.parametrize(
    "head_dim,v_head_dim,strided_v,kv_length",
    [
        pytest.param(128, 128, True, 5, id="qk128-strided"),
        pytest.param(192, 128, True, 5, id="qk192-strided"),
        pytest.param(256, 256, True, 5, id="qk256-strided"),
        pytest.param(192, 128, False, 5, id="compact-control"),
        pytest.param(192, 128, True, 0, id="empty-kv"),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("fp8", [False, True])
def test_prims_ts_mla_context_preprocess_layout(
    head_dim: int, v_head_dim: int, dtype: torch.dtype, strided_v: bool, kv_length: int, fp8: bool
) -> None:
    """Preserve non-FP8 views; quantize FP8 from the actual V row stride."""
    from tensorrt_llm.quantization.mode import QuantMode

    torch.manual_seed(1729)
    heads, query_length, page_size = 6, 3, 32
    q = torch.randn(query_length, heads * head_dim, device="cuda", dtype=dtype)
    k = torch.randn(kv_length, heads * head_dim, device="cuda", dtype=dtype)
    padding = heads * (head_dim - 64) if strided_v else 0
    v_storage = torch.full(
        (kv_length, padding + heads * v_head_dim), float("nan"), device="cuda", dtype=dtype
    )
    v = v_storage[:, padding:]
    v.copy_(torch.randn_like(v))
    originals = tuple(tensor.clone() for tensor in (q, k, v))
    processed = (
        tuple(
            torch.empty(tensor.shape, dtype=torch.float8_e4m3fn, device="cuda")
            for tensor in (q, k, v)
        )
        if fp8
        else (q, k, v)
    )
    # No RoPE/cache append is performed for an already-prepared cached partition.
    # Supply a real pool so the native page-table carrier is valid nonetheless.
    pool = torch.zeros(
        1, page_size, 576, device="cuda", dtype=torch.float8_e4m3fn if fp8 else dtype
    )
    block_offsets = torch.zeros(1, 1, 2, 1, device="cuda", dtype=torch.int32)
    pool_pointers = torch.tensor([[pool.data_ptr(), 0]], dtype=torch.int64)
    pool_mapping = torch.zeros(1, 2, dtype=torch.int32)
    q_offsets = torch.tensor([0, query_length], device="cuda", dtype=torch.int32)
    kv_lengths = torch.tensor([kv_length], device="cuda", dtype=torch.int32)
    rope = torch.zeros(page_size, 128, device="cuda", dtype=torch.float32)
    quant_scale = torch.tensor([2.0], device="cuda")
    dequant_scale = torch.tensor([0.5], device="cuda")
    bmm1_scale = torch.empty(2, device="cuda")
    bmm2_scale = torch.empty(1, device="cuda")
    torch.ops.trtllm.mla_context_preprocess(
        q,
        k,
        v,
        None,
        q_offsets,
        kv_lengths,
        rope,
        block_offsets,
        pool_pointers,
        pool_mapping,
        quant_scale,
        dequant_scale,
        heads,
        query_length,
        0,
        page_size,
        page_size,
        int(QuantMode.from_description(use_fp8_kv_cache=fp8)),
        0.125,
        *processed,
        bmm1_scale,
        bmm2_scale,
    )
    for original, actual, result in zip(originals, (q, k, v), processed, strict=True):
        torch.testing.assert_close(actual, original, atol=0, rtol=0)
        expected = (
            (original.float() * 2.0).to(torch.float8_e4m3fn).float() if fp8 else original.float()
        )
        torch.testing.assert_close(result.float(), expected, atol=0, rtol=0)
        if not fp8:
            assert result.data_ptr() == actual.data_ptr()
            assert result.stride() == actual.stride()
    if fp8:
        torch.testing.assert_close(
            bmm1_scale, torch.tensor([0.03125, 0.03125 * math.log2(math.e)], device="cuda")
        )
        torch.testing.assert_close(bmm2_scale, dequant_scale)


@pytest.mark.parametrize("kv_cache_dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize(
    "qk_head_dim,v_head_dim,num_heads,use_kv_cache_manager_v2",
    [
        pytest.param(128, 128, 6, False, id="qk128-h6-v1"),
        pytest.param(128, 128, 12, True, id="qk128-h12-v2"),
        pytest.param(128, 128, 96, False, id="qk128-h96-v1"),
        pytest.param(192, 128, 6, True, id="qk192-h6-v2"),
        pytest.param(192, 128, 12, False, id="qk192-h12-v1"),
        pytest.param(192, 128, 96, True, id="qk192-h96-v2"),
        pytest.param(256, 256, 6, False, id="qk256-h6-v1"),
        pytest.param(256, 256, 12, True, id="qk256-h12-v2"),
        pytest.param(256, 256, 96, False, id="qk256-h96-v1"),
    ],
)
def test_prims_ts_mla_preprocessing(
    monkeypatch: pytest.MonkeyPatch,
    num_heads: int,
    use_kv_cache_manager_v2: bool,
    kv_cache_dtype: torch.dtype,
    qk_head_dim: int,
    v_head_dim: int,
) -> None:
    from test_attention_mla import RopeConfig, _run_test_for_backend

    from tensorrt_llm._torch.attention.backends.fmha.phased import FmhaParams
    from tensorrt_llm._torch.attention.backends.fmha.prims_ts import PrimsTSFmha

    # No fallback: both context and every decode must select PrimTS.
    monkeypatch.setenv("TLLM_FMHA_LIBS", "prims_ts")
    original_run = PrimsTSFmha.run_mla_generation
    original_context = PrimsTSFmha.run_mla_context
    calls = []
    context_calls = []

    def run_context(fmha: PrimsTSFmha, params: FmhaParams) -> None:
        original_context(fmha, params)
        context_calls.append(params.attn.local_layer_idx)

    monkeypatch.setattr(PrimsTSFmha, "run_mla_context", run_context)

    def run_and_capture(fmha: PrimsTSFmha, params: FmhaParams) -> None:
        assert params.query_input.dtype == torch.bfloat16
        if kv_cache_dtype == torch.float8_e4m3fn:
            assert params.fwd.quant_q_buffer.dtype == torch.uint8
        original_run(fmha, params)
        eager = params.output.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            original_run(fmha, params)
        params.output.zero_()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(params.output, eager, atol=0, rtol=0)
        calls.append(params.attn.local_layer_idx)

    monkeypatch.setattr(PrimsTSFmha, "run_mla_generation", run_and_capture)
    _run_test_for_backend(
        backend_name="TRTLLM",
        num_heads=num_heads,
        num_kv_heads=num_heads,
        num_layers=2,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=qk_head_dim - 64,
        qk_rope_head_dim=64,
        v_head_dim=v_head_dim,
        rope_config=RopeConfig(
            num_attention_heads=num_heads,
            max_position_embeddings=4096,
            rope_scaling={
                "beta_fast": 32,
                "beta_slow": 1,
                "factor": 40.0,
                "mscale": 1.0,
                "mscale_all_dim": 1.0,
                "original_max_position_embeddings": 4096,
                "type": "yarn",
            },
        ),
        kv_cache_tokens_per_block=32,
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        kv_cache_dtype=kv_cache_dtype,
        context_sequence_lengths=[129, 191],
        generation_seq_len_q=1,
        num_generation_steps=2,
        v2_kv_cache=use_kv_cache_manager_v2,
    )
    assert calls == [0, 1, 0, 1]
    assert context_calls == [0, 1]


@pytest.mark.parametrize("num_heads", [6, 96], ids=["tp16-local-heads", "dp-heads"])
@pytest.mark.parametrize("fp8_kv", [False, True], ids=["bf16-kv", "fp8-kv"])
@pytest.mark.parametrize("chunked", [False, True], ids=["cached", "chunked"])
def test_prims_ts_kimi_k3_full_attention(
    monkeypatch: pytest.MonkeyPatch, num_heads: int, fp8_kv: bool, chunked: bool
) -> None:
    """Exercise K3's real NoPE/gated MLA module, without full-model weights.

    Head counts cover the TP16-local and attention-DP geometries, not distributed
    collectives. The full-model GB300 test covers those separately.
    """
    _run_prims_ts_kimi_k3_full_attention(monkeypatch, num_heads, fp8_kv, chunked)


@pytest.mark.parametrize("query_width", [6, 8], ids=["mtp5", "mtp7"])
def test_prims_ts_kimi_k3_fp8_mla_linear_mtp(
    monkeypatch: pytest.MonkeyPatch, query_width: int
) -> None:
    """Verify uniform MTP with the real K3 module, V2 FP8/page128 cache and graph."""
    _run_prims_ts_kimi_k3_full_attention(
        monkeypatch, 96, True, False, generation_query_width=query_width
    )


def _run_prims_ts_kimi_k3_full_attention(
    monkeypatch: pytest.MonkeyPatch,
    num_heads: int,
    fp8_kv: bool,
    chunked: bool,
    *,
    generation_query_width: int = 1,
) -> None:
    import tensorrt_llm
    from tensorrt_llm._torch.attention.backends import TrtllmAttention
    from tensorrt_llm._torch.attention.backends.fmha.prims_ts import PrimsTSFmha
    from tensorrt_llm._torch.attention.backends.interface import AttentionRuntimeFeatures
    from tensorrt_llm._torch.metadata import KVCacheParams
    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.modules.kimi_k3_mla.kimi_k3_mla_attention import KimiK3MLAAttention
    from tensorrt_llm._torch.pyexecutor.kv_cache.kv_cache_manager_v2 import KVCacheManagerV2
    from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequest, SamplingConfig
    from tensorrt_llm._torch.pyexecutor.resource_manager import KVCacheManager
    from tensorrt_llm._utils import torch_dtype_to_binding
    from tensorrt_llm.llmapi import KvCacheConfig, SamplingParams
    from tensorrt_llm.llmapi.llm_args import DSparkDecodingConfig
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization import QuantAlgo

    context_calls = []
    generation_calls = []
    prefill_reference = None
    run_context = PrimsTSFmha.run_mla_context
    run_generation = PrimsTSFmha.run_mla_generation

    def record_context(fmha, params):
        nonlocal prefill_reference
        context_calls.append(
            (params.fwd.softmax_stats_tensor is not None, params.key_input.shape[0])
        )
        run_context(fmha, params)
        if len(context_calls) == 1:
            # Preprocessing has now populated K's initially uninitialized RoPE
            # slice. Compare both FP8 kernels to an independent reference, not
            # to each other: their softmax rounding errors can add rather than
            # cancel even when each meets the same accuracy bound.
            operands = (params.query_input, params.key_input, params.value_input)
            scale = fmha._get_bmm1_scale(params.attn)
            output_scale = 1.0
            if fp8_kv:
                operands = tuple(
                    (tensor.float() * params.fwd.kv_scale_orig_quant).to(torch.float8_e4m3fn)
                    for tensor in operands
                )
                output_scale = params.fwd.kv_scale_quant_orig.item()
                scale *= output_scale**2
            q_ref, k_ref, v_ref = (
                tensor.reshape(tensor.shape[0], num_heads, -1).double() for tensor in operands
            )
            references = []
            offset = 0
            for length in lengths[0]:
                q_seq, k_seq, v_seq = (
                    tensor[offset : offset + length].transpose(0, 1)
                    for tensor in (q_ref, k_ref, v_ref)
                )
                scores = (q_seq @ k_seq.transpose(-1, -2)) * scale
                causal_mask = torch.ones(
                    (length, length), device=scores.device, dtype=torch.bool
                ).tril()
                scores.masked_fill_(~causal_mask, -torch.inf)
                references.append(((scores.softmax(-1) @ v_seq) * output_scale).transpose(0, 1))
                offset += length
            prefill_reference = torch.cat(references).flatten(1).to(params.output.dtype)

    def record_generation(fmha, params):
        assert params.input_seq_length == generation_query_width
        assert params.attn.predicted_tokens_per_seq == 1
        if generation_query_width > 1:
            assert params.tokens_per_block == 128
            assert params.fwd.quant_q_buffer.dtype == torch.uint8
            assert isinstance(params.meta.kv_cache_manager, KVCacheManagerV2)
        generation_calls.append(params.batch_size)
        run_generation(fmha, params)

    monkeypatch.setattr(PrimsTSFmha, "run_mla_context", record_context)
    monkeypatch.setattr(PrimsTSFmha, "run_mla_generation", record_generation)
    kv_dtype = torch.float8_e4m3fn if fp8_kv else torch.bfloat16
    use_mtp = generation_query_width > 1
    # The MTP verification group crosses a page boundary in both requests.
    lengths = (
        [[65, 129], [62, 126], [generation_query_width] * 2]
        if use_mtp
        else [[129, 65], [17, 193], [1, 1]]
    )
    spec_config = (
        DSparkDecodingConfig(
            max_draft_len=generation_query_width - 1, block_size=generation_query_width - 1
        )
        if use_mtp
        else None
    )
    torch.manual_seed(2718)
    hidden_steps = [
        torch.randn(sum(q_lengths), 256, dtype=torch.bfloat16, device="cuda")
        for q_lengths in lengths
    ]
    results = {}
    for library in ("fallback", "prims_ts"):
        monkeypatch.setenv("TLLM_FMHA_LIBS", library)
        torch.manual_seed(1729)
        mapping = Mapping(world_size=1, tp_size=1, rank=0)
        module = (
            KimiK3MLAAttention(
                hidden_size=256,
                num_heads=num_heads,
                q_lora_rank=128,
                kv_lora_rank=512,
                qk_nope_head_dim=128,
                qk_rope_head_dim=64,
                v_head_dim=128,
                dtype=torch.bfloat16,
                model_config=ModelConfig(
                    mapping=mapping,
                    quant_config=QuantConfig(kv_cache_quant_algo=QuantAlgo.FP8 if fp8_kv else None),
                    spec_config=spec_config,
                ),
                aux_stream_dict={},
            )
            .cuda()
            .eval()
        )
        # Use the module's eager entry, retaining the real projection, cache,
        # context/chunk-merge, absorption, gate and output-projection paths.
        module.register_to_config = False
        with torch.no_grad():
            for name, parameter in module.named_parameters():
                if "layernorm" in name:
                    parameter.fill_(1.0)
                else:
                    parameter.normal_(std=0.02)
            k_weight, v_weight = module.kv_b_proj.weight.split(num_heads * 128)
            module.k_b_proj_trans.copy_(k_weight.view(num_heads, 128, 512).transpose(1, 2))
            module.v_b_proj.copy_(v_weight.view(num_heads, 128, 512))
        manager_cls = KVCacheManagerV2 if use_mtp else KVCacheManager
        manager = manager_cls(
            KvCacheConfig(max_tokens=1024, enable_block_reuse=False),
            tensorrt_llm.bindings.internal.batch_manager.CacheType.SELFKONLY,
            num_layers=1,
            num_kv_heads=1,
            head_dim=576,
            tokens_per_block=128 if use_mtp else 64,
            max_seq_len=512,
            max_batch_size=2,
            mapping=mapping,
            dtype=torch_dtype_to_binding(kv_dtype),
            **(
                {"spec_config": spec_config, "layer_mask": [True], "vocab_size": 128}
                if use_mtp
                else {}
            ),
        )
        # Both decode implementations require finite values in unused cache
        # slots. Reused BF16 allocations can contain FP8 NaN byte patterns;
        # keep padding deterministic while real context writes every live slot.
        manager.get_buffers(0).fill_(1.0)
        requests = []
        for request_id, length in enumerate(lengths[0]):
            request = LlmRequest(
                request_id=request_id,
                input_tokens=[1] * length,
                max_new_tokens=256,
                sampling_config=SamplingConfig(SamplingParams()._get_sampling_config()),
                is_streaming=False,
            )
            request.paged_kv_block_ids = []
            if use_mtp:
                cache = manager._create_kv_cache(request_id, None, [1] * length)
                assert cache is not None
                assert cache.resume(torch.cuda.current_stream().cuda_stream)
                assert cache.resize(length + manager.num_extra_kv_tokens)
            else:
                manager.impl.add_sequence_batch([(request_id, length, 1)], [request])
            requests.append(request)
        cached = [0, 0]
        results[library] = []
        try:
            with torch.inference_mode():
                for step, q_lengths in enumerate(lengths):
                    if step:
                        for request_id, length in enumerate(q_lengths):
                            if use_mtp:
                                cache = manager.kv_cache_map[request_id]
                                assert cache.resize(
                                    cached[request_id] + length + manager.num_extra_kv_tokens
                                )
                            else:
                                for _ in range(length):
                                    manager.impl.add_token(request_id)
                    metadata = TrtllmAttention.Metadata(
                        seq_lens=torch.tensor(q_lengths, dtype=torch.int32),
                        request_ids=[0, 1],
                        max_num_requests=2,
                        num_contexts=2 if step < 2 else 0,
                        prompt_lens=q_lengths,
                        max_num_tokens=512,
                        kv_cache_manager=manager,
                        kv_cache_params=KVCacheParams(
                            use_cache=True,
                            num_cached_tokens_per_seq=cached,
                            num_extra_kv_tokens=manager.num_extra_kv_tokens,
                        ),
                        mapping=mapping,
                        enable_context_mla_with_cached_kv=True,
                        runtime_features=AttentionRuntimeFeatures(
                            cache_reuse=True,
                            chunked_prefill=chunked,
                            chunk_size=128,
                            chunked_prefill_buffer_batch_size=1,
                            has_speculative_draft_tokens=use_mtp,
                        ),
                    )
                    metadata.prepare()
                    if use_mtp:
                        metadata.update_spec_dec_param(
                            batch_size=2,
                            is_spec_decoding_enabled=True,
                            is_spec_dec_tree=False,
                            is_spec_dec_dynamic_tree=False,
                            max_draft_len=generation_query_width - 1,
                            max_total_draft_tokens=generation_query_width - 1,
                        )
                    hidden = hidden_steps[step]
                    positions = torch.cat(
                        [
                            torch.arange(start, start + length, device="cuda")
                            for start, length in zip(cached, q_lengths)
                        ]
                    )
                    result = module(positions, hidden, metadata)
                    assert torch.isfinite(result).all(), (library, step, "eager")
                    results[library].append(result.clone())
                    if step == 0 and library == "prims_ts":
                        assert prefill_reference is not None
                        expected = module.o_proj(
                            module._apply_output_gate(hidden, prefill_reference)
                        )
                        for candidate in (results["fallback"][0], result):
                            torch.testing.assert_close(
                                candidate, expected, atol=0.02 if fp8_kv else 0.003, rtol=0.03
                            )
                    if step == 2 and library == "prims_ts":
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            captured = module(positions, hidden, metadata)
                        graph.replay()
                        torch.cuda.synchronize()
                        torch.testing.assert_close(captured, result, atol=0, rtol=0)
                    cached = [start + length for start, length in zip(cached, q_lengths)]
        finally:
            manager.shutdown()
    # Fresh prefill was checked against FP64 math above; retain the direct
    # backend comparison for cached/chunk-merged prefill and decode.
    for actual, expected in zip(results["prims_ts"][1:], results["fallback"][1:]):
        torch.testing.assert_close(actual, expected, atol=0.02 if fp8_kv else 0.003, rtol=0.03)
    assert len(context_calls) == (4 if chunked else 2)
    assert any(has_stats for has_stats, _ in context_calls) == chunked
    assert generation_calls == [2, 2]  # Eager and graph capture; replay is device-only.


@pytest.mark.parametrize("num_heads", [6, 12, 96])
@pytest.mark.parametrize("batch_size", [2, 65])
@pytest.mark.parametrize("max_seq_len", [64, 128], ids=["short-kv", "regular-kv"])
def test_prims_ts_fp8_mla_scales_and_graph_replay(
    monkeypatch: pytest.MonkeyPatch, num_heads: int, batch_size: int, max_seq_len: int
) -> None:
    import tensorrt_llm._torch.attention.backends.fmha.prims_ts as prims_ts_module
    from tensorrt_llm._torch.attention.backends.fmha.phased import FmhaParams
    from tensorrt_llm._torch.attention.backends.fmha.prims_ts import PrimsTSFmha
    from tensorrt_llm._torch.attention.backends.interface import (
        AttentionForwardArgs,
        AttentionInputType,
    )
    from tensorrt_llm.quantization.mode import QuantMode

    torch.manual_seed(0)
    device = torch.device("cuda")
    attn = Mock(
        is_mla_enable=True,
        num_heads=num_heads,
        num_kv_heads=1,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        head_dim=576,
        v_head_dim=128,
        local_layer_idx=0,
        quant_mode=QuantMode.FP8_KV_CACHE,
        q_scaling=1.0,
    )
    fmha = PrimsTSFmha(attn)
    query = torch.randn(batch_size, num_heads, 576, device=device).to(torch.float8_e4m3fn)
    kv_cache = torch.randn(batch_size * 2, 1, 32, 576, device=device).to(torch.float8_e4m3fn)
    block_tables = torch.zeros((batch_size, 2, max_seq_len // 32), dtype=torch.int32, device=device)
    block_tables[:, 0, :2] = torch.arange(batch_size * 2, device=device).view(batch_size, 2)
    seq_lens = torch.full((batch_size,), 33, dtype=torch.int32, device=device)
    output = torch.empty((batch_size, num_heads * 512), dtype=torch.bfloat16, device=device)
    # Deliberately use different, non-unit scales. BMM1[1] is log2-scaled and
    # must not be passed to PrimTS; BMM2 must scale V independently of QK.
    bmm1_scale, bmm2_scale = 0.025, 1.75
    fwd = AttentionForwardArgs(
        attention_input_type=AttentionInputType.generation_only,
        attention_window_size=max_seq_len,
        output=output,
        quant_q_buffer=query.view(torch.uint8),
        mla_bmm1_scale=torch.tensor([bmm1_scale, bmm1_scale * math.log2(math.e)], device=device),
        mla_bmm2_scale=torch.tensor([bmm2_scale], device=device),
    )
    metadata = SimpleNamespace(
        host_kv_cache_pool_pointers=None,
        host_kv_cache_pool_mapping=None,
        kv_cache_block_offsets=block_tables,
        num_contexts=0,
        num_ctx_tokens=0,
        num_generations=batch_size,
        tokens_per_block=32,
        helix_position_offsets=None,
    )
    # Pool ownership/indexing is covered by the real v1/v2 preprocessing test.
    monkeypatch.setattr(
        prims_ts_module.thop,
        "build_trtllm_gen_kv_cache_metadata",
        lambda *args: (kv_cache, block_tables, None),
    )
    # Poison the BF16 input: the adapter must consume the preprocessing bytes.
    q = torch.full((batch_size, num_heads * 576), float("nan"), dtype=torch.bfloat16, device=device)
    workspace = torch.empty(0, dtype=torch.uint8, device=device)
    fmha.prepare_workspace(q, None, None, metadata, fwd, workspace)
    params = FmhaParams(
        attn=attn,
        meta=metadata,
        fwd=fwd,
        workspace=workspace,
        query_input=q,
        output=output,
        sequence_lengths=seq_lens,
        input_seq_length=1,
        num_tokens=batch_size,
        batch_size=batch_size,
        num_requests=batch_size,
        tokens_per_block=32,
        kv_factor=1,
        total_num_blocks=batch_size * 2,
    )

    def check_output() -> None:
        pages = kv_cache[:, 0].float()[block_tables[:, 0, :2].long()].reshape(batch_size, 64, 576)
        scores = torch.einsum("bhd,bkd->bhk", query.float(), pages) * bmm1_scale
        invalid = torch.arange(64, device=device)[None, :] >= seq_lens[:, None]
        scores.masked_fill_(invalid[:, None, :], float("-inf"))
        ideal = torch.einsum("bhk,bkd->bhd", scores.softmax(-1), pages[..., :512]) * bmm2_scale
        # This case fits one KV tile. FP8 MLA rounds the unnormalized P tile
        # (scaled to E4M3's maximum 448) before PV, but keeps its sum in FP32.
        # Model that intermediate rounding for the elementwise comparison.
        probabilities = (scores - scores.amax(dim=-1, keepdim=True)).exp() * 448.0
        rounded_p = probabilities.to(torch.float8_e4m3fn).float()
        expected = (
            torch.einsum("bhk,bkd->bhd", rounded_p, pages[..., :512])
            / probabilities.sum(dim=-1, keepdim=True)
            * bmm2_scale
        )
        torch.testing.assert_close(
            output, expected.reshape_as(output).to(output.dtype), atol=2e-2, rtol=2e-2
        )
        # Also bound aggregate error against full-precision attention so the
        # low-precision oracle does not hide a large numerical regression.
        relative_error = (output.float().view_as(ideal) - ideal).norm() / ideal.norm()
        assert relative_error.item() < 0.04

    fmha.run_mla_generation(params)
    if max_seq_len < 128:
        # Auto policy must fall back when the 1CTA profile cannot cover this capacity.
        wrapper = fmha._mla_decode_wrappers[(batch_size, 1)]
        assert dict(wrapper._plan_state.policy)["kernel"] == "throughput_2cta"
    check_output()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fmha.run_mla_generation(params)
    # New query bytes and page-table/length contents must be read on replay.
    query.view(torch.uint8).copy_(query.view(torch.uint8).flip(0))
    block_tables[:, 0, :2].copy_(block_tables[:, 0, :2].flip(0))
    seq_lens.add_(7)
    output.zero_()
    graph.replay()
    torch.cuda.synchronize()
    check_output()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("log2_jump", [7.0, 9.0], ids=["freeze-row-max", "rescale-row-max"])
@torch.no_grad()
def test_prims_ts_context_lazy_rescale_with_late_key_block(
    dtype: torch.dtype, log2_jump: float
) -> None:
    from tensorrt_llm._torch.attention.backends.prims_ts import BatchPrefillPagedTSWrapper

    device = torch.device("cuda")
    q_len, kv_len, page_size, head_dim = 64, 512, 64, 128
    sm_scale = head_dim**-0.5
    query = torch.zeros((q_len, 1, head_dim), device=device, dtype=dtype)
    query[..., 0] = 1
    k_cache = torch.zeros((kv_len // page_size, 1, page_size, head_dim), device=device, dtype=dtype)
    v_cache = torch.ones_like(k_cache)
    # Only the last page raises the row maximum, after earlier KV tiles have
    # contributed to the accumulator. The skip threshold is 8 in log2 logits.
    k_cache[-1, :, :, 0] = log2_jump / (sm_scale * math.log2(math.e))
    v_cache[-1] = -1
    qo_indptr = torch.tensor([0, q_len], device=device, dtype=torch.int32)
    block_tables = torch.arange(kv_len // page_size, device=device, dtype=torch.int32)[None]
    seq_lens = torch.tensor([kv_len], device=device, dtype=torch.int32)
    output = torch.empty_like(query)

    logical_k = k_cache[:, 0].flatten(0, 1).float()
    logical_v = v_cache[:, 0].flatten(0, 1).float()
    scores = (query[:, 0].float() @ logical_k.T) * sm_scale
    actual_log2_jump = (scores[0, -1] - scores[0, 0]).item() * math.log2(math.e)
    assert (actual_log2_jump > 8.0) == (log2_jump > 8.0)
    expected = (torch.softmax(scores, dim=-1) @ logical_v).to(dtype)[:, None]

    wrapper = BatchPrefillPagedTSWrapper(kv_layout="HND")
    wrapper.plan(
        device=device,
        batch_size=1,
        max_q_len=q_len,
        max_kv_len=kv_len,
        num_qo_heads=1,
        num_kv_heads=1,
        head_dim=head_dim,
        q_dtype=dtype,
        k_dtype=dtype,
        v_dtype=dtype,
        out_dtype=dtype,
        page_size=page_size,
        mask_type="dense",
        softmax_scale=sm_scale,
    )
    wrapper.run(query, k_cache, v_cache, qo_indptr, block_tables, seq_lens, out=output)
    torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(
            query, k_cache, v_cache, qo_indptr, block_tables, seq_lens, out=output, validate=False
        )
    output.zero_()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)


def test_prims_ts_context_wrapper_cuda_graph_replay_with_updated_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tensorrt_llm._torch.attention.backends.prims_ts.context as context_module
    from tensorrt_llm._torch.attention.backends.prims_ts import (
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
            use_kv_cache_manager_v2=True,
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
        max_q_len=3,
        max_kv_len=max_seq_len,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        q_dtype=dtype,
        k_dtype=dtype,
        v_dtype=dtype,
        out_dtype=dtype,
        page_size=page_size,
        mask_type="causal",
    )
    plan_state = wrapper._plan_state
    assert plan_state is not None
    compiled = plan_state.compiled

    assert len(compile_records) == 3
    assert [
        (args[0].max_q_len, args[0].max_kv_len, args[0].runtime_lengths, policy["scheduler"])
        for args, policy, _compiled in compile_records
    ] == [
        (None, None, True, "clc_dynamic_persistent"),
        (None, None, True, "nonpersistent"),
        (None, None, True, "clc_dynamic_persistent"),
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
    from tensorrt_llm._torch.attention.backends.prims_ts import (
        BatchDecodePagedTSWrapper,
        batch_decode_with_paged_kv_cache,
        get_prims_ts_batch_decode_workspace_size,
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
        workspace_bytes,
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
        max_q_len=1,
        packed_query=False,
        q_dtype=dtype,
        kv_dtype=dtype,
        out_dtype=dtype,
        mask_type="causal",
        workspace_buffer=external_workspace,
    )
    plan_state = wrapper._plan_state
    assert plan_state is not None
    compiled_main = plan_state.compiled_main

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
    reference = batch_decode_with_paged_kv_cache(
        query,
        kv_cache,
        block_tables,
        seq_lens,
        workspace_buffer=reference_workspace,
        max_kv_len=max_seq_len,
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


@pytest.mark.parametrize("max_seq_len", [64, 1024], ids=["short-kv", "regular-kv"])
def test_prims_ts_mla_live_wrapper_cuda_graph_replay(max_seq_len: int) -> None:
    from tensorrt_llm._torch.attention.backends.prims_ts import (
        BatchMLADecodePagedTSWrapper,
        batch_mla_decode_with_paged_kv_cache,
        get_prims_ts_batch_mla_decode_workspace_size,
    )

    batch_size = 2
    num_heads = 32
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    page_size = 32
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
        max_q_len=1,
        q_dtype=dtype,
        kv_dtype=dtype,
        out_dtype=dtype,
        mask_type="causal",
        device=device,
    )
    external_workspace = torch.empty(
        workspace_bytes,
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
        max_q_len=1,
        packed_query=False,
        q_dtype=dtype,
        kv_dtype=dtype,
        out_dtype=dtype,
        mask_type="causal",
        workspace_buffer=external_workspace,
    )
    plan_state = wrapper._plan_state
    assert plan_state is not None
    compiled = plan_state.compiled
    if max_seq_len == 64:
        assert dict(plan_state.policy)["kernel"] == "throughput_2cta"
    bmm1_scale = (128 + qk_rope_head_dim) ** -0.5

    wrapper.run(
        query,
        kv_cache,
        block_tables,
        seq_lens,
        softmax_scale=bmm1_scale,
        out=output,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = wrapper.run(
            query,
            kv_cache,
            block_tables,
            seq_lens,
            softmax_scale=bmm1_scale,
            out=output,
            validate=False,
        )

    query.copy_(query.flip(0).clone())
    block_tables.copy_(block_tables.flip(0).clone())
    seq_lens.copy_(seq_lens.flip(0).clone())
    reference_workspace = torch.empty_like(external_workspace)
    reference = batch_mla_decode_with_paged_kv_cache(
        query,
        kv_cache,
        block_tables,
        seq_lens,
        workspace_buffer=reference_workspace,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        max_kv_len=max_seq_len,
        max_q_len=1,
        softmax_scale=bmm1_scale,
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
    from tensorrt_llm._torch.attention.backends.prims_ts import (
        BatchDecodePagedTSWrapper,
        batch_decode_with_paged_kv_cache,
        get_prims_ts_batch_decode_workspace_size,
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
            max_q_len=1,
            packed_query=False,
            q_dtype=dtype,
            kv_dtype=dtype,
            out_dtype=dtype,
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
    reference_a = batch_decode_with_paged_kv_cache(
        query_a,
        kv_cache,
        block_tables_a,
        seq_lens_a,
        workspace_buffer=reference_workspace,
        max_kv_len=max_seq_len_a,
        out_dtype=dtype,
        out=torch.empty_like(output_a),
        mask_type="causal",
        kv_layout="HND",
    ).clone()
    reference_workspace.zero_()
    reference_b = batch_decode_with_paged_kv_cache(
        query_b,
        kv_cache,
        block_tables_b,
        seq_lens_b,
        workspace_buffer=reference_workspace,
        max_kv_len=max_seq_len_b,
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
    from tensorrt_llm._torch.attention.backends.prims_ts import (
        BatchDecodePagedTSWrapper,
        batch_decode_with_paged_kv_cache,
        get_prims_ts_batch_decode_workspace_size,
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
            max_q_len=1,
            packed_query=False,
            q_dtype=dtype,
            kv_dtype=dtype,
            out_dtype=dtype,
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
        reference = batch_decode_with_paged_kv_cache(
            query,
            kv_cache,
            block_tables,
            seq_lens,
            workspace_buffer=reference_workspace,
            max_kv_len=max_seq_len,
            out_dtype=dtype,
            out=torch.empty_like(query),
            mask_type="causal",
            kv_layout="HND",
        )

        assert plan_state.workspace_buffer.data_ptr() == shared_workspace.data_ptr(), layer_index
        torch.testing.assert_close(actual, reference, atol=3e-2, rtol=3e-3)


def _forward_with_recipe(monkeypatch: pytest.MonkeyPatch, recipe, vc_operands=None) -> None:
    """Make the shared harness send separate Q/K/V with the recipe flags.

    The VisualGen attention layer prepares the VC-Attention operands before the
    attention op, so a VC recipe passes them in here.
    """
    import backend_case

    from tensorrt_llm._torch.attention.backends.interface import AttentionForwardArgs
    from tensorrt_llm._torch.attention.backends.trtllm import TrtllmAttention

    monkeypatch.setattr(TrtllmAttention, "support_fused_qkv", classmethod(lambda cls: False))
    monkeypatch.setattr(
        backend_case,
        "AttentionForwardArgs",
        functools.partial(
            AttentionForwardArgs,
            qk_fp8_per_tensor=not recipe.vc_attention and recipe.qk_dtype == "fp8",
            v_fp8_per_tensor=not recipe.vc_attention and recipe.v_dtype == "fp8",
            vc_attention=recipe.vc_attention,
            vc_operands=vc_operands,
        ),
    )


def _vc_operands(case: BackendCase, inputs: dict, recipe, denoise_step: tuple[int, int]):
    """Prepare the VC operands the way the VisualGen TRTLLM attention layer does."""
    from tensorrt_llm._torch.attention.backends.prims_ts.vc_attention.quantization import (
        VCAttentionPreprocessor,
    )
    from tensorrt_llm._torch.visual_gen.attention.backends.trtllm import prepare_vc_operands

    def bshd(x: torch.Tensor) -> torch.Tensor:
        return x.view(case.num_seqs, -1, case.num_heads, case.head_dim)

    return prepare_vc_operands(
        bshd(inputs["q"]),
        bshd(inputs["new_k"]),
        bshd(inputs["new_v"]),
        qk_fp8=recipe.qk_dtype == "fp8",
        repair_budget=recipe.vc_repair_budget,
        preprocessor=None if recipe.vc_repair_budget else VCAttentionPreprocessor(),
        denoise_step=denoise_step,
    )


_CACHE_FREE_CASE = dict(
    num_heads=8,
    num_kv_heads=8,
    head_dim=128,
    seq_lens=[256, 256],
    num_cached_tokens=[0, 0],
    num_contexts=2,
    dtype="bfloat16",
    cache="none",
    causal=False,
)


@pytest.mark.parametrize(
    (
        "qk_dtype",
        "v_dtype",
        "vc_attention",
        "vc_repair_budget",
        "vc_smooth",
        "seq_len",
        "outlier",
        "cuda_graph",
    ),
    [
        pytest.param(
            "bf16",
            "fp8",
            False,
            0.0,
            True,
            200,
            1.0,
            True,
            id="bf16-qk-fp8-v-partial-cuda-graph",
        ),
        pytest.param(
            "fp8",
            "fp8",
            False,
            0.0,
            True,
            256,
            1.0,
            True,
            id="fp8-qkv-cuda-graph",
        ),
        ("bf16", "fp8", False, 0.0, True, 200, 10.0, False),
        ("fp8", "fp8", False, 0.0, True, 256, 10.0, False),
        ("bf16", "fp8", True, 0.0, True, 256, 1.0, False),
        ("bf16", "fp8", True, 0.0, True, 256, 10.0, False),
        ("bf16", "fp8", True, 0.0, False, 200, 1.0, False),
        ("bf16", "fp8", True, 0.0, False, 200, 10.0, False),
        ("bf16", "fp8", True, 0.01, True, 200, 1.0, False),
        ("bf16", "fp8", True, 0.01, True, 200, 10.0, False),
        ("fp8", "fp8", True, 0.0, True, 200, 1.0, False),
        ("fp8", "fp8", True, 0.0, True, 200, 10.0, False),
        ("fp8", "fp8", True, 0.0, False, 256, 1.0, False),
        ("fp8", "fp8", True, 0.0, False, 256, 10.0, False),
        ("fp8", "fp8", True, 0.01, True, 256, 1.0, False),
        ("fp8", "fp8", True, 0.01, True, 256, 10.0, False),
    ],
)
def test_prims_ts_cache_free_fp8_recipes(
    monkeypatch: pytest.MonkeyPatch,
    qk_dtype: str,
    v_dtype: str,
    vc_attention: bool,
    vc_repair_budget: float,
    vc_smooth: bool,
    seq_len: int,
    outlier: float,
    cuda_graph: bool,
) -> None:
    """Cache-free recipes plan FP8 operands, fold the per-tensor scales, and match SDPA.

    ``outlier`` multiplies the second sample's Q and V and divides its K, so the two
    samples differ in magnitude and set different per-tensor scales while their QK
    logits are unchanged. Each sample is checked against SDPA. The VC rows cover both
    V-Smooth phases, inside the window with tile means and after it on the plain
    demean-off plan, and a length that is not a multiple of the 128-token tile.
    The CUDA graph rows replay quantization and attention with changed Q/K/V,
    covering both float/tensor and tensor/tensor global-scale combinations.
    """
    from tensorrt_llm._torch.attention.backends.prims_ts.context import BatchPrefillTSWrapper
    from tensorrt_llm.visual_gen.args import QuantAttentionConfig

    recipe = QuantAttentionConfig(
        qk_dtype=qk_dtype,
        v_dtype=v_dtype,
        vc_attention=vc_attention,
        vc_repair_budget=vc_repair_budget,
    )
    plans: list[dict] = []
    runs: list[dict] = []
    forward_calls: list[tuple] = []
    forward = None
    if cuda_graph:
        from tensorrt_llm._torch.attention.backends.fmha.prims_ts import PrimsTSFmha

        assert not vc_attention
        forward = PrimsTSFmha._forward_without_kv_cache

        def recording_forward(self: PrimsTSFmha, *args: object, **kwargs: object) -> None:
            # The library keeps only a weak reference to its owning attention.
            forward_calls.append((self.attn, self, args, kwargs))
            forward(self, *args, **kwargs)

        monkeypatch.setattr(PrimsTSFmha, "_forward_without_kv_cache", recording_forward)

    plan, run = BatchPrefillTSWrapper.plan, BatchPrefillTSWrapper.run

    def recording_plan(self, *args, **kwargs):
        plans.append(kwargs)
        return plan(self, *args, **kwargs)

    def recording_run(self, *args, **kwargs):
        runs.append(kwargs)
        return run(self, *args, **kwargs)

    monkeypatch.setattr(BatchPrefillTSWrapper, "plan", recording_plan)
    monkeypatch.setattr(BatchPrefillTSWrapper, "run", recording_run)
    monkeypatch.setenv("TLLM_FMHA_LIBS", "+prims_ts")
    case = BackendCase(**{**_CACHE_FREE_CASE, "seq_lens": [seq_len, seq_len]})
    inputs = generate_inputs(case, seed=0)
    second = slice(case.seq_lens[0], None)
    inputs["q"][second] *= outlier
    inputs["new_k"][second] /= outlier
    inputs["new_v"][second] *= outlier
    # Step 0 of 50 lies inside the V-Smooth window, step 40 after it.
    denoise_step = (0, 50) if vc_smooth else (40, 50)
    _forward_with_recipe(
        monkeypatch,
        recipe,
        _vc_operands(case, inputs, recipe, denoise_step) if vc_attention else None,
    )

    out = run_backend(case, "TRTLLM", inputs, kv_dtype=torch.bfloat16, kv_layout="HND")

    def bhsd(x: torch.Tensor) -> torch.Tensor:
        return x.view(case.num_seqs, -1, case.num_heads, case.head_dim).transpose(1, 2).float()

    golden = torch.nn.functional.scaled_dot_product_attention(
        bhsd(inputs["q"]), bhsd(inputs["new_k"]), bhsd(inputs["new_v"])
    )
    golden = golden.transpose(1, 2).reshape_as(out).to(out.dtype)

    fp8, bf16 = torch.float8_e4m3fn, torch.bfloat16
    (plan_kwargs,) = plans
    (run_kwargs,) = runs
    expected_dtypes = (fp8, fp8, fp8) if qk_dtype == "fp8" else (bf16, bf16, fp8)
    assert (
        plan_kwargs["q_dtype"],
        plan_kwargs["k_dtype"],
        plan_kwargs["v_dtype"],
    ) == expected_dtypes

    def dequant_scale(x: torch.Tensor) -> torch.Tensor:
        return (x.float().abs().amax() / 448.0).reshape(1)

    if vc_attention:
        # The VC operands carry the V scale and tile means (and the Q/K block scales under
        # QK8); no per-tensor scale is folded.
        assert run_kwargs["vc"] is not None
        assert (run_kwargs["vc"].q_scale is not None) == (qk_dtype == "fp8")
        assert (plan_kwargs["vc_config"].repair_tiles > 0) == (vc_repair_budget > 0)
        assert plan_kwargs["vc_config"].demean == (vc_smooth and not vc_repair_budget)
        assert run_kwargs["output_scale"] == 1.0
        assert run_kwargs["softmax_scale"] == 1.0 / math.sqrt(case.head_dim)
    else:
        # The quantize op returns BF16-rounded dequant scales.
        torch.testing.assert_close(
            run_kwargs["output_scale"], dequant_scale(inputs["new_v"]), rtol=1e-2, atol=0.0
        )
        if qk_dtype == "fp8":
            expected = (
                dequant_scale(inputs["q"])
                * dequant_scale(inputs["new_k"])
                / math.sqrt(case.head_dim)
            )
            torch.testing.assert_close(run_kwargs["softmax_scale"], expected, rtol=1e-2, atol=0.0)
        else:
            assert run_kwargs["softmax_scale"] == 1.0 / math.sqrt(case.head_dim)
    # The second sample's output is ``outlier`` times larger. Normalize it before comparing.
    sample_gain = torch.tensor([1.0, outlier], device=out.device).view(-1, 1)
    torch.testing.assert_close(
        out.view(case.num_seqs, -1) / sample_gain,
        golden.view(case.num_seqs, -1) / sample_gain,
        atol=FP8_ATOL + BF16_ATOL,
        rtol=BF16_RTOL,
    )

    if cuda_graph:
        ((_owner, library, args, kwargs),) = forward_calls
        q_input, k_input, v_input = args[:3]
        captured_output = args[4].output
        assert captured_output is not None
        original_q, original_k, original_v = q_input.clone(), k_input.clone(), v_input.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            forward(library, *args, **kwargs)
        # Replay the warmed adapter, including quantization and its live scales.
        for q_gain, k_gain, v_gain in ((3.0, 0.7, 0.3), (0.3, 1.7, 3.0)):
            q_input.copy_(original_q * q_gain)
            k_input.copy_(original_k * k_gain)
            v_input.copy_(original_v * v_gain)
            captured_output.zero_()
            graph.replay()
            expected = (
                torch.nn.functional.scaled_dot_product_attention(
                    bhsd(q_input), bhsd(k_input), bhsd(v_input)
                )
                .transpose(1, 2)
                .reshape_as(captured_output)
            )
            relative_error = (captured_output.float() - expected).norm() / expected.norm()
            assert relative_error.item() < 0.15
            torch.testing.assert_close(
                captured_output.float() / v_gain,
                expected / v_gain,
                atol=FP8_ATOL + BF16_ATOL,
                rtol=BF16_RTOL,
            )
        assert len(plans) == 1


def test_prims_ts_recipe_requires_prims_ts_library(monkeypatch: pytest.MonkeyPatch) -> None:
    """A per-tensor recipe fails instead of running unquantized on another FMHA library."""
    from tensorrt_llm.visual_gen.args import QuantAttentionConfig

    recipe = QuantAttentionConfig(qk_dtype="bf16", v_dtype="fp8")
    _forward_with_recipe(monkeypatch, recipe)
    monkeypatch.delenv("TLLM_FMHA_LIBS", raising=False)
    case = BackendCase(**_CACHE_FREE_CASE)

    with pytest.raises(RuntimeError, match="needs the PrimTS FMHA library"):
        run_backend(
            case, "TRTLLM", generate_inputs(case, seed=0), kv_dtype=torch.bfloat16, kv_layout="HND"
        )


def test_prims_ts_rejects_vc_attention_with_kv_cache() -> None:
    """The VC recipe is prepared on the cache-free path only, so paged requests must not claim it."""
    from tensorrt_llm._torch.attention.backends.fmha.prims_ts import PrimsTSFmha

    fmha = object.__new__(PrimsTSFmha)
    q = torch.empty(4, 128, device="cuda", dtype=torch.bfloat16)
    supported, reason = fmha._is_supported_with_reason(
        q,
        None,
        None,
        SimpleNamespace(is_mla_enable=False),
        SimpleNamespace(kv_cache_manager=object()),
        SimpleNamespace(is_fused_qkv=True, vc_attention=True, sparse_runtime_params=None),
    )

    assert not supported
    assert reason == "VC-Attention runs on cache-free context attention only."


def test_prims_ts_unsupported_context_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    from tensorrt_llm._torch.attention.backends.fmha.fallback import FallbackFmha
    from tensorrt_llm._torch.attention.backends.fmha.prims_ts import PrimsTSFmha

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


def test_prims_ts_fp8_decode_bf16_device_scales(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Check FP8 decode with live device scales across graph replay."""
    import cutlass

    from tensorrt_llm._torch.attention.backends.prims_ts import decode
    from tensorrt_llm._torch.attention.backends.prims_ts.kernels.fmha_decode.fmha_decode_config import (
        make_decode_config,
    )

    torch.manual_seed(123)
    head_dim, page_size, head_ratio = 128, 64, 8
    batch_size, kv_heads = 2, 2
    qo_heads = kv_heads * head_ratio
    max_kv_len = 2048
    pages_per_seq = max_kv_len // page_size
    num_pages = batch_size * pages_per_seq
    q = torch.randn(batch_size, qo_heads, head_dim, device="cuda").to(torch.float8_e4m3fn)
    k = torch.randn(num_pages, kv_heads, page_size, head_dim, device="cuda").to(torch.float8_e4m3fn)
    v = torch.randn_like(k, dtype=torch.float32).to(torch.float8_e4m3fn)
    # Permuted pages and padded table rows catch assumptions about pool layout.
    table = torch.full((batch_size, pages_per_seq + 3), -1, dtype=torch.int32, device="cuda")
    table[:, :pages_per_seq] = torch.randperm(num_pages, device="cuda").reshape(batch_size, -1)
    lengths_host = [max_kv_len - 13, max_kv_len // 2 + 7]
    lengths = torch.tensor(lengths_host, dtype=torch.int32, device="cuda")
    bmm1 = torch.tensor([0.25 / math.sqrt(head_dim)], device="cuda")
    bmm2 = torch.tensor([1.75], device="cuda")
    output = torch.empty_like(q, dtype=torch.bfloat16)
    splits = 4
    cfg = make_decode_config(
        headdim=head_dim,
        args={"tile_size_q": 16, "groups_tokens_heads_q": True, "use_keeps_mma_ab": False},
        seq_len_q=1,
        seq_len_kv=max_kv_len,
        batch_size=batch_size,
        num_heads_q=qo_heads,
        num_heads_kv=kv_heads,
        qkv_dtype=cutlass.Float8E4M3FN,
        o_dtype=cutlass.BFloat16,
        qkv_layout="pagedKv",
        num_tokens_per_page=page_size,
        split_kv_mode="gmem_reduction_with_separate_kernel",
        splits_kv=splits,
        max_splits_kv=splits,
        mask_type="causal",
        auto_tuner=False,
    )

    spec = decode._decode_launch_spec_from_config(
        cfg,
        batch_size=batch_size,
        num_qo_heads=qo_heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        seq_len_q=1,
        max_active_clusters=torch.cuda.get_device_properties(0).multi_processor_count,
    )
    monkeypatch.setattr(decode, "_resolve_decode_launch_spec", lambda *args, **kwargs: spec)
    wrapper = decode.BatchDecodePagedTSWrapper()
    wrapper.plan(
        q.device,
        batch_size,
        qo_heads,
        kv_heads,
        head_dim,
        page_size,
        max_kv_len,
        q_dtype=q.dtype,
        out_dtype=output.dtype,
        mask_type="causal",
        softmax_scale=bmm1,
        output_scale=bmm2,
    )

    def run() -> torch.Tensor:
        return wrapper.run(
            q,
            (k, v),
            lengths,
            table,
            out=output,
            softmax_scale=bmm1,
            output_scale=bmm2,
            validate=False,
        )

    def reference() -> torch.Tensor:
        results = []
        for batch_idx, length in enumerate(lengths_host):
            ids = table[batch_idx, :pages_per_seq].long()
            keys = k.float()[ids].permute(1, 0, 2, 3).reshape(kv_heads, -1, head_dim)[:, :length]
            values = v.float()[ids].permute(1, 0, 2, 3).reshape(kv_heads, -1, head_dim)[:, :length]
            keys = keys.repeat_interleave(head_ratio, dim=0)
            values = values.repeat_interleave(head_ratio, dim=0)
            scores = torch.einsum("hd,htd->ht", q[batch_idx].float(), keys) * bmm1
            results.append(torch.einsum("ht,htd->hd", scores.softmax(-1), values) * bmm2)
        return torch.stack(results)

    def check_output() -> None:
        # The FP8 kernel rounds the intermediate P operand to E4M3.
        torch.testing.assert_close(output.float(), reference(), atol=0.01, rtol=0.05)

    wrapper.run(
        q,
        (k, v),
        lengths,
        table,
        out=output,
        softmax_scale=bmm1,
        output_scale=bmm2,
        validate=True,
    )
    torch.cuda.synchronize()
    check_output()

    def no_host_read(*args, **kwargs):
        raise AssertionError("decode must not read scales back to the host")

    with monkeypatch.context() as guard:
        guard.setattr(torch.Tensor, "item", no_host_read)
        guard.setattr(torch.Tensor, "__float__", no_host_read)
        sync_debug_mode = torch.cuda.get_sync_debug_mode()
        try:
            torch.cuda.set_sync_debug_mode("error")
            run()
        finally:
            torch.cuda.set_sync_debug_mode(sync_debug_mode)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        # Both attention scores and final output must consume updated scales.
        bmm1.mul_(2)
        bmm2.mul_(0.5)
        graph.replay()
    torch.cuda.synchronize()
    check_output()
