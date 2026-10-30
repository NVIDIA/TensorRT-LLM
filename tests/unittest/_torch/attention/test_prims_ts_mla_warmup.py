# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Startup planning contracts without importing the TensorRT-LLM extension."""

import ast
import importlib.util
import sys
from collections.abc import Callable
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[4]
_FMHA = _ROOT / "tensorrt_llm/_torch/attention/backends/fmha"
_MODULE_NAME = "tensorrt_llm._torch.attention.backends.fmha.prims_ts_mla_warmup"
_SPEC = importlib.util.spec_from_file_location(_MODULE_NAME, _FMHA / "prims_ts_mla_warmup.py")
_warmup = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_warmup)


def _capacity_function() -> Callable[..., tuple[int, int]]:
    source = _FMHA / "prims_ts.py"
    function = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "_mla_context_capacities"
    )
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[function.name]


def _geometry(**overrides: object) -> SimpleNamespace:
    fields = dict(
        device=torch.device("cuda:0"),
        batch_size=4,
        max_seq_len_q=2048,
        max_seq_len_k=2048,
        num_qo_heads=96,
        num_kv_heads=96,
        head_dim=192,
        head_dim_vo=128,
        qk_dtype=torch.float8_e4m3fn,
        pv_dtype=torch.float8_e4m3fn,
        output_dtype=torch.bfloat16,
        packed=True,
        uniform_packed_lengths=False,
        mask_type="causal",
        window_left=-1,
        store_softmax_stats=False,
    )
    return SimpleNamespace(**(fields | overrides))


def _capacities() -> Callable[[int, int], tuple[int, int]]:
    geometry = vars(_geometry())
    fields = (
        "head_dim",
        "head_dim_vo",
        "qk_dtype",
        "pv_dtype",
        "output_dtype",
        "packed",
        "uniform_packed_lengths",
        "mask_type",
        "window_left",
    )
    return partial(_capacity_function(), **{name: geometry[name] for name in fields})


def _profiles(**overrides: object) -> list[tuple[int, int, str, bool]]:
    fields = dict(
        max_num_tokens=2048,
        max_seq_len=4608,
        cached_context=True,
        chunked_prefill=True,
        chunk_size=2048,
        chunk_batch_size=4,
        capacities=_capacities(),
    )
    return _warmup._context_profiles(**(fields | overrides))


def test_normalized_context_envelope_includes_tails_and_three_families() -> None:
    profiles = _profiles()
    assert len(profiles) == 38
    assert (256, 128, "causal", False) in profiles
    assert (256, 128, "causal", True) in profiles
    assert (256, 4096, "dense", True) in profiles
    assert (512, 8192, "dense", True) in profiles
    assert (1024, 8192, "dense", True) in profiles
    assert len(set(profiles)) == len(profiles)


def test_context_envelope_covers_independent_mixed_batch_maxima() -> None:
    max_num_tokens, max_seq_len = 2048, 4608
    chunk_size, chunk_batch_size = 2048, 4
    query_lengths = torch.tensor([1024, 128], device="cpu")
    cached_kv_lengths = torch.tensor([0, 4224], device="cpu")
    assert int(query_lengths.sum()) <= max_num_tokens
    assert bool(torch.all(query_lengths + cached_kv_lengths <= max_seq_len))
    assert int(cached_kv_lengths.sum()) <= chunk_size * chunk_batch_size

    max_query = int(query_lengths.max())
    max_cached_kv = int(cached_kv_lengths.max())
    assert max_query + max_cached_kv > max_seq_len
    profile = (*_capacities()(max_query, max_cached_kv), "dense", True)
    assert profile == (1024, 8192, "dense", True)
    assert profile in _profiles(
        max_num_tokens=max_num_tokens,
        max_seq_len=max_seq_len,
        chunk_size=chunk_size,
        chunk_batch_size=chunk_batch_size,
    )


def test_context_envelope_covers_every_small_reachable_length() -> None:
    capacities = _capacities()
    profiles = set(
        _profiles(max_num_tokens=257, max_seq_len=513, chunk_size=129, chunk_batch_size=1)
    )
    for query in range(1, 258):
        assert (*capacities(query, query), "causal", False) in profiles
        assert (*capacities(query, query), "causal", True) in profiles
        for kv_length in range(130):
            assert (*capacities(query, kv_length), "dense", True) in profiles


def test_context_profiles_respect_staging_and_cache_modes() -> None:
    assert len(_profiles(cached_context=False)) == 5
    limited = _profiles(chunk_size=128, chunk_batch_size=1)
    assert all(kv == 128 for _, kv, mask, _ in limited if mask == "dense")
    unchunked = _profiles(chunked_prefill=False)
    assert (256, 8192, "causal", False) in unchunked
    assert all(mask == "causal" and not stats for _, _, mask, stats in unchunked)
    assert _profiles(max_seq_len=1) == [(256, 128, "causal", False)]
    with pytest.raises(ValueError, match="positive staging capacity"):
        _profiles(chunk_size=0)


@pytest.fixture
def planners(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    for name in (
        "tensorrt_llm",
        "tensorrt_llm._torch",
        "tensorrt_llm._torch.attention",
        "tensorrt_llm._torch.attention.backends",
        "tensorrt_llm._torch.attention.backends.fmha",
        "tensorrt_llm._torch.attention.backends.prims_ts",
    ):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    context = SimpleNamespace()
    context._resolve_context_plan_geometry = Mock(side_effect=lambda **kwargs: kwargs)
    context._context_compile_spec = Mock(
        side_effect=lambda geometry: ("context", tuple(sorted(geometry.items())))
    )
    context.plan = Mock()
    context.BatchPrefillTSWrapper = Mock(side_effect=lambda: SimpleNamespace(plan=context.plan))
    decode = SimpleNamespace()
    decode._dtype_key = str
    decode._resolve_mla_decode_launch_spec = Mock(side_effect=lambda *args: (args[1] % 3, args[0]))
    decode._make_mla_decode_compile_spec = Mock(
        side_effect=lambda launch, **kwargs: ("decode", launch, tuple(sorted(kwargs.items())))
    )
    decode.plan = Mock()
    decode.BatchMLADecodePagedTSWrapper = Mock(
        side_effect=lambda: SimpleNamespace(plan=decode.plan)
    )
    package = sys.modules["tensorrt_llm._torch.attention.backends.prims_ts"]
    package.context, package.mla_decode = context, decode

    class MLA:
        sparse_attn_hooks = None

    class TrtllmAttention:
        pass

    class Metadata:
        enable_context_mla_with_cached_kv = True
        runtime_features = SimpleNamespace(
            chunked_prefill=True, chunk_size=2048, chunked_prefill_buffer_batch_size=4
        )

    class PrimsTSFmha:
        pass

    prims = ModuleType("tensorrt_llm._torch.attention.backends.fmha.prims_ts")
    prims._mla_context_capacities = _capacity_function()
    prims.PrimsTSFmha = PrimsTSFmha
    monkeypatch.setitem(sys.modules, prims.__name__, prims)
    trtllm = ModuleType("tensorrt_llm._torch.attention.backends.trtllm")
    trtllm.TrtllmAttention, trtllm.TrtllmAttentionMetadata = TrtllmAttention, Metadata
    monkeypatch.setitem(sys.modules, trtllm.__name__, trtllm)
    mla = ModuleType("tensorrt_llm._torch.attention.mla")
    mla.MLA = MLA
    monkeypatch.setitem(sys.modules, mla.__name__, mla)
    log = ModuleType("tensorrt_llm.logger")
    log.logger = SimpleNamespace(info=Mock())
    monkeypatch.setitem(sys.modules, log.__name__, log)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return SimpleNamespace(
        context=context,
        decode=decode,
        MLA=MLA,
        backend=TrtllmAttention,
        metadata=Metadata(),
        provider=PrimsTSFmha,
    )


def _decode_state(**overrides: object) -> SimpleNamespace:
    fields = dict(
        device=torch.device("cuda:0"),
        batch_size=1,
        num_heads=96,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        page_size=128,
        max_kv_len=4608,
        max_seq_len_q=6,
        packed_query=False,
        q_dtype=torch.float8_e4m3fn,
        kv_dtype=torch.float8_e4m3fn,
        output_dtype=torch.bfloat16,
        mask_type="causal",
        store_softmax_stats=False,
        use_kv_bounds=False,
        workspace_buffer=object(),
    )
    return SimpleNamespace(**(fields | overrides))


def test_context_normal_planning_and_full_spec_dedup(planners: SimpleNamespace) -> None:
    warmer = _warmup.PrimsTSDenseMlaWarmup()
    args = dict(max_num_tokens=2048, max_seq_len=4608)
    assert warmer._prewarm_context(_geometry(), planners.metadata, **args) == 38
    assert warmer._prewarm_context(_geometry(batch_size=16), planners.metadata, **args) == 0
    assert planners.context.plan.call_count == 38
    for call in planners.context.plan.call_args_list:
        assert call.kwargs["batch_size"] == 1
        assert call.kwargs["q_dtype"] == torch.float8_e4m3fn
    assert (
        warmer._prewarm_context(_geometry(device=torch.device("cuda:1")), planners.metadata, **args)
        == 38
    )
    assert warmer._prewarm_context(_geometry(packed=False), planners.metadata, **args) == 0


def test_failed_plan_is_retried(planners: SimpleNamespace) -> None:
    warmer = _warmup.PrimsTSDenseMlaWarmup()
    planners.context.plan.side_effect = RuntimeError("compile failure")
    args = dict(max_num_tokens=1, max_seq_len=1)
    with pytest.raises(RuntimeError, match="compile failure"):
        warmer._prewarm_context(_geometry(), planners.metadata, **args)
    assert not warmer._completed
    planners.context.plan.side_effect = None
    assert warmer._prewarm_context(_geometry(), planners.metadata, **args) == 1


def test_decode_scans_batches_not_unobserved_widths(planners: SimpleNamespace) -> None:
    warmer = _warmup.PrimsTSDenseMlaWarmup()
    state = _decode_state()
    original_workspace = state.workspace_buffer
    assert warmer._prewarm_decode(state, max_batch_size=16, max_num_tokens=2048) == 3
    resolve_calls = planners.decode._resolve_mla_decode_launch_spec.call_args_list
    assert [call.args[1] for call in resolve_calls] == list(range(1, 17))
    assert {call.args[11] for call in resolve_calls} == {6}
    assert {call.args[6] for call in resolve_calls} == {4608}
    assert {call.args[1] for call in planners.decode.plan.call_args_list} == {14, 15, 16}
    assert all(
        "workspace_buffer" not in call.kwargs for call in planners.decode.plan.call_args_list
    )
    assert state.workspace_buffer is original_workspace
    assert warmer._prewarm_decode(state, max_batch_size=16, max_num_tokens=2048) == 0
    assert (
        warmer._prewarm_decode(
            _decode_state(store_softmax_stats=True, use_kv_bounds=True),
            max_batch_size=16,
            max_num_tokens=2048,
        )
        == 3
    )
    assert (
        warmer._prewarm_decode(
            _decode_state(device=torch.device("cuda:1")), max_batch_size=16, max_num_tokens=2048
        )
        == 3
    )


def test_decode_respects_token_budget_and_failure_retry(planners: SimpleNamespace) -> None:
    warmer = _warmup.PrimsTSDenseMlaWarmup()
    planners.decode.plan.side_effect = RuntimeError("compile failure")
    with pytest.raises(RuntimeError, match="compile failure"):
        warmer._prewarm_decode(_decode_state(), max_batch_size=16, max_num_tokens=13)
    assert not warmer._completed
    assert [
        call.args[1] for call in planners.decode._resolve_mla_decode_launch_spec.call_args_list
    ] == [1, 2]
    planners.decode.plan.side_effect = None
    assert warmer._prewarm_decode(_decode_state(), max_batch_size=16, max_num_tokens=13) == 2


def _model(
    planners: SimpleNamespace, *, selected: bool = True, device: str = "cuda:0"
) -> tuple[SimpleNamespace, object, SimpleNamespace]:
    library = planners.provider()
    wrapper = SimpleNamespace(_plan_state=_decode_state(device=torch.device(device)))
    library._mla_context_wrappers = {}
    library._mla_decode_wrappers = {(1, 6): wrapper} if selected else {}
    backend = planners.backend()
    backend._fmha_manager = SimpleNamespace(fmha_libs=[library])
    module = planners.MLA()
    module.mha, module.mqa = None, backend
    return SimpleNamespace(modules=lambda: [module]), library, wrapper


def test_selected_provider_and_capture_gates(
    planners: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    warmer = _warmup.PrimsTSDenseMlaWarmup()
    args = dict(max_batch_size=16, max_num_tokens=2048, max_seq_len=4608)
    model, library, wrapper = _model(planners, selected=False)
    assert warmer.run(model, planners.metadata, **args) == (0, 0)
    assert not planners.decode.plan.called
    model, library, wrapper = _model(planners)
    assert warmer.run(model, None, **args) == (0, 0)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="during capture"):
        warmer.run(model, planners.metadata, **args)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert warmer.run(model, planners.metadata, **args) == (0, 3)
    assert library._mla_decode_wrappers == {(1, 6): wrapper}
    assert warmer.run(model, planners.metadata, **args) == (0, 0)
    cpu_model, _, _ = _model(planners, device="cpu")
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", Mock(side_effect=AssertionError("CUDA touched"))
    )
    assert warmer.run(cpu_model, planners.metadata, **args) == (0, 0)


def test_startup_hook_is_after_graphs_and_not_gated_on_graph_enablement() -> None:
    source = _ROOT / "tensorrt_llm/_torch/pyexecutor/engine/runners/decoder/warmup.py"
    tree = ast.parse(source.read_text())
    scheduled = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_warmup_scheduled"
    )
    phases = [
        (index, ast.unparse(node))
        for index, node in enumerate(scheduled.body)
        if isinstance(node, ast.With)
    ]
    graph_index = next(index for index, text in phases if "self._graphs.warmup_and_capture" in text)
    prewarm_index = next(
        index for index, text in phases if "self._prims_ts_dense_mla_warmup.run" in text
    )
    assert prewarm_index > graph_index
