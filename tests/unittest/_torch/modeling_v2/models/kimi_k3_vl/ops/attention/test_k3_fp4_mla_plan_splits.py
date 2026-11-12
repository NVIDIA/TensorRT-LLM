# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""K3's generation KV update plans the FP4 MLA backend's packed split plan with the decode's 8 KV splits.

The first FP4 MLA kernel of a generation step (``kernels/k3r_fp4_mla/fp4_mla_kv_append``) writes the
backend's packed split plan fused with the generation lengths. K3's packed decode always runs 8 task slots
per request (``ops/attention/_fp4_mla_decode._SPLITS``), and the reference recipes' generation servers
planned with 8 as well (``TRTLLM_FP4_MLA_SPLIT_KV=8``). The KV update takes the plan geometry from the
family (``_fp4_mla_decode.backend_plan_geometry``): the backend's geometry with the count fixed, so no
environment variable decides it and the backend's ``packed_plan`` stays as it is. Host side, with the kernel
modules stubbed.
"""

import importlib
import inspect
import sys
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.cpu_only

_FAMILY = "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl"
_REDUCE = "tensorrt_llm._torch.attention.backends.fp4_mla.fp4_mla_packed_reduce"
_FAMILY_REDUCE = f"{_FAMILY}.kernels.k3r_fp4_mla.fp4_mla_packed_reduce"
_SPLIT_KV_ENV = "TRTLLM_FP4_MLA_SPLIT_KV"
_SM_COUNT = 212


def _rows_per_seq(heads, query_len):
    return 0 if heads * query_len > 1024 else -(-heads * query_len // 128) * 128


def _modules(monkeypatch):
    """The backend's packed_plan and the family decode, with the kernel modules and the device stubbed."""
    from tensorrt_llm._torch.attention.backends.fp4_mla import packed_plan

    decode = importlib.import_module(f"{_FAMILY}.ops.attention._fp4_mla_decode")
    kernel = SimpleNamespace(_select_runtime_kv_profile=lambda kv: 1 << (kv - 1).bit_length())
    reduce = SimpleNamespace(packed_plan_length=lambda requests, tasks: 3 + requests + 3 * tasks)
    for module in (packed_plan, decode):
        monkeypatch.setattr(module, "_fp4_mla_cutedsl_kernel_module", lambda: kernel)
        monkeypatch.setattr(module, "_get_sm_count", lambda device: _SM_COUNT)
        monkeypatch.setattr(module, "_fp4_mla_packed_rows_per_seq", _rows_per_seq)
    monkeypatch.setattr(decode, "_reduce_module", lambda: reduce)
    monkeypatch.setitem(sys.modules, _REDUCE, reduce)
    return packed_plan, decode


def _metadata(num_requests, is_cuda_graph=True):
    state = SimpleNamespace(
        device_page_table=True,
        page_table_stride=3904,
        generation_kv_lens=torch.zeros(num_requests, dtype=torch.int32),
        workspaces={"_fp4_mla_packed_plan": torch.empty(1 << 16, dtype=torch.int32)},
    )
    return SimpleNamespace(page_size=128, is_cuda_graph=is_cuda_graph, fp4_mla_state=state)


def _geometry(fn, metadata, num_requests, num_heads, query_len):
    return fn(
        metadata,
        num_heads=num_heads,
        query_len=query_len,
        num_requests=num_requests,
        device=torch.device("cpu"),
        helix_kv_bounds=None,
    )


# (requests, heads, query length): helix8 MTP7 (96 heads x 8), dep16 MTP5 (6 heads x 6) at its batch sizes.
_SHAPES = [(1, 96, 8), (2, 96, 8), (4, 96, 8), (16, 96, 8), (1, 6, 6), (8, 6, 6), (16, 6, 6)]


@pytest.mark.parametrize("is_cuda_graph", [True, False])
@pytest.mark.parametrize("env", [None, "auto", "3"])
def test_family_geometry_is_the_backend_geometry_planned_with_8(monkeypatch, env, is_cuda_graph):
    """Under any environment the family's geometry equals the backend's with the reference recipes'
    ``TRTLLM_FP4_MLA_SPLIT_KV=8``: same splits, task-length cap, KV profile, capacity, plan buffer and key."""
    packed_plan, decode = _modules(monkeypatch)
    for shape in _SHAPES:
        metadata = _metadata(shape[0], is_cuda_graph)
        if env is None:
            monkeypatch.delenv(_SPLIT_KV_ENV, raising=False)
        else:
            monkeypatch.setenv(_SPLIT_KV_ENV, env)
        family = _geometry(decode.backend_plan_geometry, metadata, *shape)
        monkeypatch.setenv(_SPLIT_KV_ENV, "8")
        backend = _geometry(packed_plan._fp4_mla_packed_plan_geometry, metadata, *shape)
        assert isinstance(family, packed_plan._PackedPlanGeometry), shape
        assert family.splits == decode._SPLITS == 8, shape
        assert family._replace(plan=None) == backend._replace(plan=None), shape
        # The same workspace buffer (the workspace hands out a fresh view per call).
        assert family.plan.data_ptr() == backend.plan.data_ptr(), shape
        assert family.plan.shape == backend.plan.shape, shape


def test_without_the_fixed_count_the_selection_is_not_always_8(monkeypatch):
    """What the K3 generation servers would plan with the backend's own selection: fixing the count matters."""
    packed_plan, _ = _modules(monkeypatch)
    monkeypatch.delenv(_SPLIT_KV_ENV, raising=False)
    selected = {
        shape: _geometry(
            packed_plan._fp4_mla_packed_plan_geometry, _metadata(shape[0]), *shape
        ).splits
        for shape in _SHAPES
    }
    assert any(n != 8 for n in selected.values()), selected


def test_no_packed_geometry_without_packed_rows(monkeypatch):
    _, decode = _modules(monkeypatch)
    assert (
        _geometry(decode.backend_plan_geometry, _metadata(4), 4, 96, 16) is None
    )  # 1536 rows > 1024


def test_kv_update_plans_with_the_family_geometry():
    decode = importlib.import_module(f"{_FAMILY}.ops.attention._fp4_mla_decode")
    append = importlib.import_module(f"{_FAMILY}.kernels.k3r_fp4_mla.fp4_mla_kv_append")
    assert decode._SPLITS == 8
    assert append.backend_plan_geometry is decode.backend_plan_geometry
    source = inspect.getsource(append._scatter_generation_kv_only_2d)
    assert source.count("backend_plan_geometry(") == 1
    assert "_fp4_mla_packed_plan_geometry" not in inspect.getsource(append)
