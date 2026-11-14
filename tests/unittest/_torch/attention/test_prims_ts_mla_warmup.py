# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Startup planning contracts without importing the TensorRT-LLM extension."""

import ast
import gc
import importlib
import importlib.util
import sys
import weakref
from collections.abc import Callable, Iterator
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, create_autospec

import pytest
import torch

pytestmark = pytest.mark.cpu_only

_ROOT = Path(__file__).resolve().parents[4]
_FMHA = _ROOT / "tensorrt_llm/_torch/attention/backends/fmha"
_MODULE_NAME = "tensorrt_llm._torch.attention.backends.fmha._isolated_mla_warmup"
_SPEC = importlib.util.spec_from_file_location(_MODULE_NAME, _FMHA / "prims_ts_mla_warmup.py")
_warmup = importlib.util.module_from_spec(_SPEC)
sys.modules[_MODULE_NAME] = _warmup
_SPEC.loader.exec_module(_warmup)
del sys.modules[_MODULE_NAME]


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


def _geometry(**overrides: object) -> _warmup._MlaContextPlan:
    fields = dict(
        device=torch.device("cuda:0"),
        batch_size=4,
        max_q_len=2048,
        max_kv_len=2048,
        num_qo_heads=96,
        num_kv_heads=96,
        head_dim=192,
        head_dim_vo=128,
        q_dtype=torch.float8_e4m3fn,
        k_dtype=torch.float8_e4m3fn,
        v_dtype=torch.float8_e4m3fn,
        out_dtype=torch.bfloat16,
        packed_query=True,
        mask_type="causal",
        window_left=-1,
        store_softmax_stats=False,
        softmax_scale=1.0,
        output_scale=1.0,
    )
    return _warmup._MlaContextPlan(**(fields | overrides))


def _capacities() -> Callable[[int, int], tuple[int, int]]:
    geometry = vars(_geometry())
    geometry["output_dtype"] = geometry.pop("out_dtype")
    geometry["packed"] = geometry.pop("packed_query")
    geometry["qk_dtype"] = geometry.pop("q_dtype")
    geometry["pv_dtype"] = geometry.pop("v_dtype")
    geometry["uniform_packed_lengths"] = False
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
def planners(monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
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
    package = sys.modules["tensorrt_llm._torch.attention.backends.prims_ts"]
    package.__path__ = [str(_FMHA.parent / "prims_ts")]
    context_api = importlib.import_module(f"{package.__name__}.context")
    decode_api = importlib.import_module(f"{package.__name__}.mla_decode")
    context = ModuleType(context_api.__name__)
    context.plan = create_autospec(context_api.BatchPrefillTSWrapper().plan)
    context.BatchPrefillTSWrapper = Mock(side_effect=lambda: SimpleNamespace(plan=context.plan))
    decode = ModuleType(decode_api.__name__)
    decode.plan = create_autospec(decode_api.BatchMLADecodePagedTSWrapper().plan)
    decode.BatchMLADecodePagedTSWrapper = Mock(
        side_effect=lambda: SimpleNamespace(plan=decode.plan)
    )
    # Only public symbols exist on these substitutes: private accesses fail.
    monkeypatch.setitem(sys.modules, context.__name__, context)
    monkeypatch.setitem(sys.modules, decode.__name__, decode)
    package.context, package.mla_decode = context, decode
    workspace_size = create_autospec(
        decode_api.get_prims_ts_batch_mla_decode_workspace_size,
        side_effect=lambda **kwargs: (128 - kwargs["batch_size"]) * kwargs["max_q_len"]
        + 1024 * kwargs["store_softmax_stats"],
    )
    package.get_prims_ts_batch_mla_decode_workspace_size = workspace_size
    allocations = []
    empty, ones = torch.empty, torch.ones

    def allocate(size: int, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        buffer = empty(size, dtype=dtype, device="cpu")
        allocations.append((device, size, weakref.ref(buffer)))
        return buffer

    monkeypatch.setattr(torch, "empty", allocate)
    monkeypatch.setattr(torch, "ones", lambda *args, **kwargs: ones(*args, dtype=kwargs["dtype"]))

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
    yield SimpleNamespace(
        context=context,
        decode=decode,
        workspace_size=workspace_size,
        allocations=allocations,
        MLA=MLA,
        backend=TrtllmAttention,
        metadata=Metadata(),
        provider=PrimsTSFmha,
    )


def _decode_profile(**overrides: object) -> _warmup._MlaDecodePlan:
    fields = dict(
        device=torch.device("cuda:0"),
        batch_size=1,
        num_qo_heads=96,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        page_size=128,
        max_kv_len=4608,
        max_q_len=6,
        packed_query=False,
        q_dtype=torch.float8_e4m3fn,
        kv_dtype=torch.float8_e4m3fn,
        out_dtype=torch.bfloat16,
        mask_type="causal",
        store_softmax_stats=False,
        use_kv_bounds=False,
    )
    return _warmup._MlaDecodePlan(**(fields | overrides))


def _model(
    planners: SimpleNamespace,
    *,
    contexts: tuple[_warmup._MlaContextPlan, ...] = (),
    decodes: tuple[_warmup._MlaDecodePlan, ...] = (),
) -> tuple[SimpleNamespace, object, object]:
    library = planners.provider()
    # Production wrappers intentionally expose no library implementation state.
    wrapper = object()
    library._mla_context_wrappers = {0: wrapper}
    library._mla_decode_wrappers = {(1, 6): wrapper}
    library._mla_context_plan_profiles = set(contexts)
    library._mla_decode_plan_profiles = set(decodes)
    backend = planners.backend()
    backend._fmha_manager = SimpleNamespace(fmha_libs=[library])
    module = planners.MLA()
    module.mha, module.mqa = None, backend
    return SimpleNamespace(modules=lambda: [module]), library, wrapper


def _run(
    warmer: object, planners: SimpleNamespace, model: object, **overrides: int
) -> tuple[int, int]:
    capacities = dict(max_batch_size=16, max_num_tokens=2048, max_seq_len=4608)
    return warmer.run(model, planners.metadata, **(capacities | overrides))


def test_context_public_planning_and_model_profile_dedup(planners: SimpleNamespace) -> None:
    warmer = _warmup.PrimsTSDenseMlaWarmup()
    observed = _geometry()
    model, library, _ = _model(planners, contexts=(observed, replace(observed, batch_size=16)))
    assert _run(warmer, planners, model) == (38, 0)
    assert planners.context.plan.call_count == 38
    assert {
        (
            c.kwargs["max_q_len"],
            c.kwargs["max_kv_len"],
            c.kwargs["mask_type"],
            c.kwargs["store_softmax_stats"],
        )
        for c in planners.context.plan.call_args_list
    } == set(_profiles())
    assert all(c.kwargs["batch_size"] == 1 for c in planners.context.plan.call_args_list)
    assert _run(warmer, planners, model) == (0, 0)
    library._mla_context_plan_profiles.add(replace(observed, device=torch.device("cuda:1")))
    assert _run(warmer, planners, model) == (38, 0)
    library._mla_context_plan_profiles.add(replace(observed, packed_query=False))
    assert _run(warmer, planners, model) == (0, 0)
    assert not planners.allocations


def test_context_scale_kinds_do_not_retain_or_read_device_operands(
    planners: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scale = torch.tensor([0.25])
    ref = weakref.ref(scale)
    monkeypatch.setattr(torch.Tensor, "item", Mock(side_effect=AssertionError("device read")))
    monkeypatch.setattr(torch.Tensor, "cpu", Mock(side_effect=AssertionError("device read")))
    profile = _geometry(softmax_scale=_warmup._scale_profile(scale), output_scale=0.5)
    assert profile == _geometry(
        softmax_scale=_warmup._scale_profile(scale.clone()), output_scale=0.5
    )
    del scale
    assert ref() is None
    model, library, _ = _model(planners, contexts=(profile,))
    warmer = _warmup.PrimsTSDenseMlaWarmup()
    assert _run(warmer, planners, model, max_num_tokens=1, max_seq_len=1) == (1, 0)
    kwargs = planners.context.plan.call_args.kwargs
    assert isinstance(kwargs["softmax_scale"], torch.Tensor)
    assert kwargs["output_scale"] == 0.5
    assert "workspace_buffer" not in kwargs
    library._mla_context_plan_profiles.add(replace(profile, softmax_scale=1.0))
    assert _run(warmer, planners, model, max_num_tokens=1, max_seq_len=1) == (1, 0)
    assert all(
        not isinstance(value, torch.Tensor)
        for p in warmer._completed
        for value in asdict(p).values()
    )


def test_decode_all_batches_modes_and_devices_share_maximum_workspace(
    planners: SimpleNamespace,
) -> None:
    warmer = _warmup.PrimsTSDenseMlaWarmup()
    base = _decode_profile()
    modes = (
        base,
        replace(base, store_softmax_stats=True, use_kv_bounds=True),
        replace(base, max_q_len=8),
        replace(base, device=torch.device("cuda:1")),
    )
    model, library, wrapper = _model(planners, decodes=modes + (replace(base, batch_size=3),))
    assert _run(warmer, planners, model, max_batch_size=4, max_num_tokens=19) == (0, 11)
    calls = planners.decode.plan.call_args_list
    assert {(c.kwargs["max_q_len"], c.kwargs["batch_size"]) for c in calls} == {
        (6, 1),
        (6, 2),
        (6, 3),
        (8, 1),
        (8, 2),
    }
    assert planners.workspace_size.call_count == 11
    assert len(planners.allocations) == 2
    expected_sizes = {torch.device("cuda:0"): 127 * 6 + 1024, torch.device("cuda:1"): 127 * 6}
    workspaces = {}
    for call in calls:
        device, workspace = call.kwargs["device"], call.kwargs["workspace_buffer"]
        assert workspace.numel() == expected_sizes[device]
        assert workspaces.setdefault(device, workspace) is workspace
        assert call.kwargs["max_kv_len"] == 4608
    assert library._mla_decode_wrappers == {(1, 6): wrapper}
    assert _run(warmer, planners, model, max_batch_size=4, max_num_tokens=19) == (0, 0)
    assert len(planners.allocations) == 2
    del calls, call, workspace, workspaces
    planners.decode.plan.reset_mock()
    gc.collect()
    assert all(ref() is None for _, _, ref in planners.allocations)


@pytest.mark.parametrize("phase", ["context", "decode"])
def test_failed_public_plan_retries_and_releases_temporary_workspace(
    planners: SimpleNamespace,
    phase: str,
) -> None:
    warmer = _warmup.PrimsTSDenseMlaWarmup()
    model, _, _ = _model(
        planners,
        contexts=(_geometry(),) if phase == "context" else (),
        decodes=(_decode_profile(),),
    )
    plan = planners.context.plan if phase == "context" else planners.decode.plan
    plan.side_effect = RuntimeError("compile failure")
    with pytest.raises(RuntimeError, match="compile failure"):
        _run(warmer, planners, model, max_num_tokens=13, max_seq_len=1)
    assert not warmer._completed
    plan.side_effect = None
    plan.reset_mock()
    gc.collect()
    assert all(ref() is None for _, _, ref in planners.allocations)
    assert _run(warmer, planners, model, max_num_tokens=13, max_seq_len=1) == (
        1 if phase == "context" else 0,
        2,
    )


def test_selected_provider_and_capture_gates(
    planners: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    warmer = _warmup.PrimsTSDenseMlaWarmup()
    model, _, _ = _model(planners)
    assert _run(warmer, planners, model) == (0, 0)
    assert not planners.decode.plan.called
    model, _, _ = _model(planners, decodes=(_decode_profile(),))
    assert warmer.run(model, None, max_batch_size=16, max_num_tokens=2048, max_seq_len=4608) == (
        0,
        0,
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="during capture"):
        _run(warmer, planners, model)
    cpu_model, _, _ = _model(planners, decodes=(_decode_profile(device=torch.device("cpu")),))
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", Mock(side_effect=AssertionError("CUDA touched"))
    )
    assert _run(warmer, planners, cpu_model) == (0, 0)
