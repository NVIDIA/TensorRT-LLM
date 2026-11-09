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
"""The fmha_sm100 MiniMax-M3 performance paths engage on SM107 only (3rdparty/patches/msa_strided_paged_kv.patch).

SM107 runs the sm_100 builds, so every gate is a runtime check of the device capability, resolved once on the host:
the plan side stream and the deferred split-decision read (api.py), the indexer R2 store
(cute/src/sm100/fwd/atten_fwd_nvfp4_kv.py + the AOT fingerprint in cute/src/common/aot_cache.py), the proxy's skipped
correction chain / S double buffer (jit.py defines) and the k2q CSR build's scratch arena (cute/src/sm100/build_k2q_csr,
host C++). SM100 / SM103 keep the stock fmha_sm100. The jit.py / aot_cache.py cells load the installed files with a
stand-in torch, and the k2q cell reads the installed source, so they run without a GPU.
"""

import ast
import contextlib
import importlib.util
import inspect
import os
import sys
import types
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from tensorrt_llm._torch.attention.backends.sparse.minimax_m3.kernels.msa_utils import (
    msa_package_available,
)


@pytest.fixture
def api():
    # Installs the CuTe aliases the packaged MSA sources still reference, then reports importability.
    if not msa_package_available():
        pytest.skip("fmha_sm100 (MSA) is not installed")
    return pytest.importorskip("fmha_sm100.api")


def _fmha_sm100_dir() -> str:
    spec = importlib.util.find_spec("fmha_sm100")
    if spec is None or not spec.submodule_search_locations:
        pytest.skip("fmha_sm100 (MSA) is not installed")
    return list(spec.submodule_search_locations)[0]


class _Bound:
    def __init__(self, **arguments):
        self.arguments = arguments


def _side_stream_eligible_args():
    lens = torch.tensor([16, 32], dtype=torch.int32)  # host tensors: the side stream may read them
    return _Bound(
        device=0, qo_segment_lens=lens, kv_segment_lens=lens, qo_offset=None, kv_block_num=-1
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "cc, expected", [((10, 0), False), ((10, 3), False), ((10, 7), True), ((12, 0), False)]
)
def test_plan_side_stream_engages_on_sm107_only(api, monkeypatch, cc, expected):
    monkeypatch.setattr(api, "_SS_CC", {0: cc})
    # The eligibility check also asks whether a CUDA graph is being captured; no GPU is needed for that here.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert api._ss_eligible(_side_stream_eligible_args()) is expected


@pytest.mark.cpu_only
def test_plan_side_stream_resolves_the_capability_once(api, monkeypatch):
    calls = []

    def fake_capability(idx):
        calls.append(idx)
        return (10, 7)

    monkeypatch.setattr(api, "_SS_CC", {})
    monkeypatch.setattr(torch.cuda, "get_device_capability", fake_capability)
    for _ in range(5):
        assert api._ss_is_sm107(0)
    assert calls == [0]


@pytest.mark.cpu_only
def test_plan_side_stream_kill_env_name_unchanged(api):
    assert '"TLLM_FMHA_PLAN_SIDE_STREAM", "1"' in inspect.getsource(api)


def _load_with_capability(monkeypatch, root: str, cap: tuple, calls: list):
    """jit.py and aot_cache.py from the installed package, with a stand-in torch reporting ``cap``."""

    def capability(*_):
        calls.append(cap)
        return cap

    cuda = types.SimpleNamespace(is_available=lambda: True, get_device_capability=capability)
    monkeypatch.setitem(sys.modules, "torch", types.SimpleNamespace(cuda=cuda))
    monkeypatch.setitem(sys.modules, "jinja2", types.ModuleType("jinja2"))
    cutlass = types.ModuleType("cutlass")
    cutlass.cute = types.ModuleType("cutlass.cute")
    monkeypatch.setitem(sys.modules, "cutlass", cutlass)
    monkeypatch.setitem(sys.modules, "cutlass.cute", cutlass.cute)
    modules = []
    for rel, name in (("jit.py", "_gate_jit"), ("cute/src/common/aot_cache.py", "_gate_aot_cache")):
        spec = importlib.util.spec_from_file_location(name, os.path.join(root, rel))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


@pytest.mark.cpu_only
@pytest.mark.parametrize("cap", [(10, 7), (10, 0), (10, 3)], ids=["sm107", "sm100", "sm103"])
@pytest.mark.parametrize("chain, dbuf, r2kill", [("0", "1", "0"), ("1", "1", "1"), ("0", "0", "0")])
def test_proxy_modes_and_r2_store_engage_on_sm107_only(monkeypatch, cap, chain, dbuf, r2kill):
    root = _fmha_sm100_dir()
    monkeypatch.setenv("TLLM_FMHA_PROXY_CORRECTION_CHAIN", chain)
    monkeypatch.setenv("TLLM_FMHA_PROXY_S_DOUBLE_BUFFER", dbuf)
    monkeypatch.setenv("TRTLLM_MSA_DISABLE_R2STORE", r2kill)
    calls = []
    jit, aot = _load_with_capability(monkeypatch, root, cap, calls)
    # Importing either module queries no device (no CUDA initialization before a worker fork).
    assert calls == []
    sm107 = cap == (10, 7)
    want_chain = chain == "1" or not sm107
    want_single = dbuf == "0" or not sm107
    jit._get_tvm_ffi_include = lambda: "/stub/tvm_ffi/include"  # flags only; nothing is compiled
    flags = jit._get_nvcc_flags(jit.CACHE_BASE)
    assert tuple(jit._proxy_modes()) == (want_chain, want_single)
    assert ("-DFMHA_ONLYSCORE_PER_TILE_CORRECTION" in flags) is want_chain
    assert ("-DFMHA_ONLYSCORE_SINGLE_S" in flags) is want_single
    assert aot.r2_store_enabled() is (sm107 and r2kill != "1")


def _k2q_csr_source() -> str:
    path = os.path.join(_fmha_sm100_dir(), "cute/src/sm100/build_k2q_csr/build_k2q_csr.cu")
    if not os.path.isfile(path):
        pytest.skip("build_k2q_csr.cu is not installed")
    with open(path) as f:
        return f.read()


@pytest.mark.cpu_only
def test_k2q_scratch_arena_engages_on_sm107_only():
    # The k2q CSR build is host C++ compiled at first use, so its gate is checked in the installed source: the arena
    # takes the runtime SM107 check, and the other devices keep the stock allocations.
    src = _k2q_csr_source()
    sm107 = src.index("sm107_state = (cc_major == 10 && cc_minor == 7) ? 1 : 2;")
    arena = src.index("if (is_sm107 && k2q_arena_enabled()) {")
    assert sm107 < arena
    assert "row_counts = torch::zeros({H, total_rows}, opts);" in src
    assert 'std::getenv("FMHA_K2Q_WS_ARENA")' in src


@pytest.mark.cpu_only
def test_deferred_split_read_engages_on_sm107_only(api):
    # The adaptive plan defers its split-decision read only where _ss_is_sm107 holds (the capability check the side
    # stream uses, covered above); SM100 / SM103 read plan_cost immediately.
    src = inspect.getsource(api)
    assert (
        "defer = _PLAN_DEFER_READ and _ss_is_sm107(device) and not torch.cuda.is_current_stream_capturing()"
        in src
    )
    assert '"TLLM_FMHA_PLAN_DEFER_READ", "1"' in src


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "case,error",
    [
        ("cpu_scales", "destinations must use"),
        ("cpu_values", "destinations must use"),
        ("strided_scales", "contiguous one-dimensional"),
        ("matrix_scales", "contiguous one-dimensional"),
        ("short_scales", "padded 128x4"),
        ("wrong_value_dtype", "uint8 or float8_e4m3fn"),
    ],
)
def test_combine_rejects_invalid_mxfp8_destinations(case: str, error: str) -> None:
    # Follow the installed-source tests above: run the real host entry point
    # with meta buffers so bad destinations must fail before any CuTe/CUDA work.
    source = Path(_fmha_sm100_dir()) / "cute/src/sm100/fwd/combine.py"
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "combine")
    module = ast.Module(body=[function], type_ignores=[])
    namespace = {"torch": torch}
    exec(compile(module, str(source), "exec"), namespace)
    out = torch.empty((4, 2, 128), dtype=torch.bfloat16, device="meta")
    values = torch.empty((4, 256), dtype=torch.uint8, device="meta")
    scales = torch.empty(1024, dtype=torch.uint8, device="meta")
    if case == "cpu_scales":
        scales = torch.empty(1024, dtype=torch.uint8)
    elif case == "cpu_values":
        values = torch.empty((4, 256), dtype=torch.uint8)
    elif case == "strided_scales":
        scales = torch.empty(2048, dtype=torch.uint8, device="meta")[::2]
    elif case == "matrix_scales":
        scales = scales.view(128, 8)
    elif case == "short_scales":
        scales = scales[:1]
    else:
        values = values.float()
    with pytest.raises(ValueError, match=error):
        namespace["combine"](
            torch.empty((1, 4, 2, 128), device="meta"),
            None,
            out,
            None,
            cu_seqlens=torch.empty(2, dtype=torch.int32, device="meta"),
            use_pdl=True,
            o_fp8=values,
            sf_out=scales,
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize("failure", ["capture", "synchronize", "allocate"])
def test_deferred_plan_retries_after_resolution_failure(monkeypatch, failure: str) -> None:
    # Match the installed-source validation tests above, without importing CUDA kernels.
    source = Path(_fmha_sm100_dir()) / "api.py"
    tree = ast.parse(source.read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_DeferredSplitPlanInfo"
    )
    event = Mock(spec_set=["synchronize"])
    allocator = Mock(return_value=torch.empty(1))
    namespace = {
        "torch": torch,
        "PlanInfo": dict,
        "_SPLIT_KEYS": frozenset(("num_kv_splits", "predicted_speedup")),
        "_log_plan_decision": Mock(),
        "_alloc_workspace_buf": allocator,
        "_BuffTag": types.SimpleNamespace(workspace_o=0),
    }
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(source), "exec"), namespace)
    plan = namespace["_DeferredSplitPlanInfo"](num_kv_splits=7, predicted_speedup=-1)
    pending = dict(
        event=event,
        host=torch.tensor([1.0, 1.5]),
        max_kv_splits=3,
        total_qo_len=2,
        num_qo_heads=1,
        device="cuda",
    )
    plan.__dict__["_lazy"] = pending
    capturing = {"active": failure == "capture"}
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing["active"])
    if failure == "synchronize":
        event.synchronize.side_effect = RuntimeError("controlled synchronize failure")
    if failure == "allocate":
        allocator.side_effect = RuntimeError("controlled allocation failure")
    with pytest.raises(RuntimeError, match="capture|controlled"):
        plan.get("num_kv_splits")
    assert plan.__dict__["_lazy"] is pending
    assert dict.__getitem__(plan, "num_kv_splits") == 7
    if failure == "capture":
        event.synchronize.assert_not_called()
    capturing["active"] = False
    event.synchronize.side_effect = None
    allocator.side_effect = None
    assert plan["num_kv_splits"] == 3
    assert plan["predicted_speedup"] == 1.5
    assert "_lazy" not in plan.__dict__
    calls = event.synchronize.call_count
    assert plan.get("num_kv_splits") == 3
    assert event.synchronize.call_count == calls


@pytest.mark.cpu_only
@pytest.mark.parametrize("caller", [None, 0, 123])
def test_plan_consumer_uses_requested_device(
    api: types.ModuleType, monkeypatch: pytest.MonkeyPatch, caller: int | None
) -> None:
    side = Mock(spec_set=["cuda_stream"])
    side.cuda_stream = 456
    consumer = Mock(spec_set=["wait_stream"])
    result = object()
    buffer = Mock(spec_set=["is_cuda", "record_stream"])
    buffer.is_cuda = True

    def plan_impl(device: int, stream: int | None = None) -> object:
        assert device == 1 and stream == side.cuda_stream
        return result

    external = Mock(return_value=consumer)
    current = Mock(return_value=consumer)
    side_for_device = Mock(return_value=side)
    monkeypatch.setattr(api, "_SS_ON", True)
    monkeypatch.setattr(api, "_SS_VERIFY", False)
    monkeypatch.setattr(api, "_SS_SIG", inspect.signature(plan_impl))
    monkeypatch.setattr(api, "_SS_STATS", {"n": 0, "waits": [], "logged": True})
    monkeypatch.setattr(api, "_ss_eligible", lambda bound: True)
    monkeypatch.setattr(api, "_ss_stream", side_for_device)
    monkeypatch.setattr(api, "_ss_tensors", lambda output: (buffer,))
    monkeypatch.setattr(api, "_fmha_sm100_plan_impl", plan_impl)
    monkeypatch.setattr(torch.cuda, "ExternalStream", external)
    monkeypatch.setattr(torch.cuda, "current_stream", current)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: contextlib.nullcontext())
    assert api._fmha_sm100_plan(device=1, stream=caller) is result
    side_for_device.assert_called_once_with(1)
    consumer.wait_stream.assert_called_once_with(side)
    buffer.record_stream.assert_called_once_with(consumer)
    if caller:
        external.assert_called_once_with(caller, device=1)
        current.assert_not_called()
    else:
        current.assert_called_once_with(1)
        external.assert_not_called()


@pytest.mark.cpu_only
@pytest.mark.parametrize("kind", ["paged", "nonpaged", "legacy"])
def test_split_boundaries_handle_nonpaged_metadata(api: types.ModuleType, kind: str) -> None:
    plan = {"qo_segment_offsets": torch.tensor([0, 12]), "pack_factor": 2, "total_rows": 9}
    if kind != "legacy":
        plan["kv_page_indptr"] = torch.tensor([0, 9]) if kind == "paged" else None
    api._cache_split_boundaries(plan)
    assert plan["_split_nnz"] == 6
    if kind == "nonpaged":
        assert "_split_kv_pages" not in plan
    else:
        assert plan["_split_kv_pages"] == 9


@pytest.mark.cpu_only
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("storage_offset", [0, 1])
def test_mixed_split_output_preserves_copy_conversion(
    api: types.ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    dtype: torch.dtype,
    contiguous: bool,
    storage_offset: int,
) -> None:
    q = torch.arange(512, dtype=torch.bfloat16).view(4, 1, 128)
    width = 128 if contiguous else 256
    storage = torch.empty(4 * width + storage_offset, dtype=dtype)
    destination = storage[storage_offset:].view(4, 1, width)
    out = destination if contiguous else destination[..., ::2]
    calls = []

    def run(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        plan: dict,
        *,
        out: torch.Tensor | None = None,
        **kwargs: object,
    ) -> tuple[torch.Tensor, None]:
        calls.append(out)
        expected = q + 1
        if out is not None:
            assert out.dtype == torch.bfloat16 and out.device == q.device
            assert out.data_ptr() % 16 == 0
            out.copy_(expected)
            return out, None
        return expected, None

    monkeypatch.setattr(api, "_fmha_sm100", run)
    decode = {"_split_nnz": 2, "kv_segment_offsets": torch.tensor([0, 2])}
    result, scores = api.fmha_sm100(q, q, q, (True, 1, 2, decode, {}), out=out)
    assert result is out and scores is None
    torch.testing.assert_close(out, (q + 1).to(dtype), rtol=0, atol=0)
    assert all(value is not None for value in calls) is (
        dtype == torch.bfloat16 and contiguous and storage_offset == 0
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "target,offset_device", [("cpu", "cpu"), ("meta", "cpu"), ("meta", "meta"), ("meta", None)]
)
def test_sparse_plan_accepts_host_lengths_and_device_offset(
    target: str, offset_device: str | None
) -> None:
    # Execute the installed planner with CUDA allocations mapped to CPU/meta.
    source = Path(_fmha_sm100_dir()) / "sparse_fmha_adapter.py"
    tree = ast.parse(source.read_text())
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "sparse_fmha_plan"
    )
    torch_api = Mock(spec_set=torch)
    torch_api.device = Mock(return_value=torch.device(target))
    torch_api.zeros = torch.zeros
    torch_api.cumsum = torch.cumsum
    torch_api.int32 = torch.int32
    torch_api.bfloat16 = torch.bfloat16
    schedule = Mock(spec_set=["balanced_target_q_per_cta", "flat_schedule_capacity"])
    schedule.balanced_target_q_per_cta.return_value = 64
    schedule.flat_schedule_capacity.return_value = 128
    namespace = {
        "torch": torch_api,
        "SPARSE_SCHEDULE_MODEL": schedule,
        "_compute_aot_kernel_paths": Mock(return_value={}),
    }
    unit = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            function,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(unit)
    exec(compile(unit, str(source), "exec"), namespace)
    lengths = torch.tensor([16, 32], dtype=torch.int32)
    kv_lengths = torch.tensor([128, 256], dtype=torch.int32)
    offset = (
        torch.tensor([112, 224], dtype=torch.int32, device=offset_device)
        if offset_device is not None
        else None
    )
    plan = namespace["sparse_fmha_plan"](
        lengths,
        kv_lengths,
        num_qo_heads=2,
        num_kv_heads=1,
        qo_offset=offset,
        page_size=128,
        kv_block_num=4,
    )
    assert plan["seqused_k"].device == torch.device(target)
    assert plan["qo_segment_lens"].device == torch.device(target)
    assert plan["kv_segment_lens"].device.type == "cpu"
    assert plan["total_rows"] == 3 and plan["total_k"] == 384
    assert plan["max_seqlen_q"] == 32 and plan["max_seqlen_k"] == 256
    assert schedule.balanced_target_q_per_cta.call_args.kwargs["total_q"] == 48
    if target == "cpu":
        torch.testing.assert_close(plan["seqused_k"], kv_lengths, rtol=0, atol=0)


@pytest.fixture
def page_table_helpers() -> dict:
    # Load the installed adapter's CPU-capable helpers without initializing CUDA.
    source = Path(_fmha_sm100_dir()) / "sparse_fmha_adapter.py"
    names = {"_build_page_table", "_version_token", "_page_table_for_plan"}
    functions = [
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert len(functions) == len(names)
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "inference_indices,inference_lengths",
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_page_table_cache_tracks_input_mutations(
    page_table_helpers: dict, inference_indices: bool, inference_lengths: bool
) -> None:
    with torch.inference_mode(inference_indices):
        indices = torch.tensor([11, 12, 21, 22], dtype=torch.int32)
    with torch.inference_mode(inference_lengths):
        lengths = torch.tensor([256, 256], dtype=torch.int32)
    table_for_plan = page_table_helpers["_page_table_for_plan"]
    build_table = page_table_helpers["_build_page_table"]
    plan = {}
    first = table_for_plan(plan, indices, lengths, 128, 2)
    second = table_for_plan(plan, indices, lengths, 128, 2)
    if not inference_indices and not inference_lengths:
        assert second is first
    with torch.inference_mode():
        indices.add_(10)
    updated = table_for_plan(plan, indices, lengths, 128, 2)
    assert torch.equal(updated, build_table(indices, lengths, 128, 2))
    assert not torch.equal(updated, first)
    with torch.inference_mode():
        lengths.copy_(torch.tensor([128, 384], dtype=torch.int32))
    resized = table_for_plan(plan, indices, lengths, 128, 2)
    assert torch.equal(resized, build_table(indices, lengths, 128, 2))
    assert resized.shape == (2, 3)


@pytest.mark.cpu_only
def test_page_table_cache_distinguishes_strided_views(page_table_helpers: dict) -> None:
    storage = torch.arange(8, dtype=torch.int32)
    lengths = torch.tensor([256, 256], dtype=torch.int32)
    indices = storage[:4]
    strided = storage[::2]
    assert indices.data_ptr() == strided.data_ptr() and indices.numel() == strided.numel()
    plan = {}
    table_for_plan = page_table_helpers["_page_table_for_plan"]
    first = table_for_plan(plan, indices, lengths, 128, 2)
    # An equivalent tracked alias can reuse the table.
    assert table_for_plan(plan, storage[:4], lengths, 128, 2) is first
    updated = table_for_plan(plan, strided, lengths, 128, 2)
    assert torch.equal(updated, page_table_helpers["_build_page_table"](strided, lengths, 128, 2))
    assert not torch.equal(updated, first)
