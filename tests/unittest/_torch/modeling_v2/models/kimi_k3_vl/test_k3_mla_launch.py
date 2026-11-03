# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
from pathlib import Path

import cuda.bindings.driver as cuda_driver
import pytest

source = (
    Path(__file__).resolve().parents[6]
    / "tensorrt_llm/_torch/_experimental/modeling_v2/models/kimi_k3_vl/kernels/k3_mla/_launch.py"
)
spec = importlib.util.spec_from_file_location("_k3_mla_launch_test", source)
launch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launch)
pytestmark = pytest.mark.cpu_only


def _resource(sms, resource_type=None):
    resource = cuda_driver.CUdevResource()
    resource.type = (
        cuda_driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM
        if resource_type is None
        else resource_type
    )
    resource.sm.smCount = sms
    return resource


@pytest.fixture
def stream_queries(monkeypatch):
    states = {11: (148, 0), 22: (64, 7)}
    monkeypatch.setattr(
        cuda_driver,
        "cuStreamGetCtx",
        lambda stream: (
            cuda_driver.CUresult.CUDA_SUCCESS,
            cuda_driver.CUcontext(int(stream) + 100),
        ),
    )
    monkeypatch.setattr(
        cuda_driver,
        "cuCtxGetDevResource",
        lambda context, resource_type: (
            cuda_driver.CUresult.CUDA_SUCCESS,
            _resource(states[int(context) - 100][0]),
        ),
    )
    monkeypatch.setattr(
        cuda_driver,
        "cuStreamGetGreenCtx",
        lambda stream: (
            cuda_driver.CUresult.CUDA_SUCCESS,
            cuda_driver.CUgreenCtx(states[int(stream)][1]),
        ),
    )
    return states


@pytest.mark.parametrize("sms,expected", [(127, False), (128, True), (129, True)])
def test_stream_capacity_boundary(stream_queries, sms, expected):
    stream_queries[11] = (sms, 0)
    assert launch._attn_no_cluster(cuda_driver.CUstream(11), 8, 16, 7) is expected


def test_full_green_full_does_not_reuse_device_capacity(stream_queries):
    assert launch._attn_no_cluster(cuda_driver.CUstream(11), 8, 16, 7) is True
    with pytest.raises(RuntimeError, match="16-CTA cluster fallback"):
        launch._attn_no_cluster(cuda_driver.CUstream(22), 8, 16, 7)
    assert launch._attn_no_cluster(cuda_driver.CUstream(11), 8, 16, 7) is True
    stream_queries[22] = (128, 7)
    assert launch._attn_no_cluster(cuda_driver.CUstream(22), 8, 16, 7) is True


def test_small_grid_preserves_normal_cluster_without_sm_query(monkeypatch, stream_queries):
    def unexpected_sm_query(*args):
        raise AssertionError("small clustered calls must not query SM capacity")

    monkeypatch.setattr(cuda_driver, "cuCtxGetDevResource", unexpected_sm_query)
    assert launch._attn_no_cluster(cuda_driver.CUstream(11), 7, 16, 7) is False
    with pytest.raises(RuntimeError, match="16-CTA cluster fallback"):
        launch._attn_no_cluster(cuda_driver.CUstream(22), 1, 16, 7)


@pytest.mark.parametrize(
    "sms,resource_type",
    [
        (0, cuda_driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_SM),
        (128, cuda_driver.CUdevResourceType.CU_DEV_RESOURCE_TYPE_INVALID),
    ],
)
def test_unknown_resources_cannot_enable_spin_or_cluster(
    monkeypatch, stream_queries, sms, resource_type
):
    monkeypatch.setattr(
        cuda_driver,
        "cuCtxGetDevResource",
        lambda *args: (cuda_driver.CUresult.CUDA_SUCCESS, _resource(sms, resource_type)),
    )
    with pytest.raises(RuntimeError, match="no valid SM resource count"):
        launch._attn_no_cluster(cuda_driver.CUstream(11), 8, 16, 7)


@pytest.mark.parametrize(
    "status",
    [cuda_driver.CUresult.CUDA_ERROR_NOT_SUPPORTED, cuda_driver.CUresult.CUDA_ERROR_LAUNCH_FAILED],
)
def test_driver_failure_is_not_capacity_or_missing_api(monkeypatch, stream_queries, status):
    monkeypatch.setattr(
        cuda_driver,
        "cuCtxGetDevResource",
        lambda *args: (status, _resource(148)),
    )
    with pytest.raises(RuntimeError, match=status.name):
        launch._attn_no_cluster(cuda_driver.CUstream(11), 8, 16, 7)


def test_failed_context_compatibility_query_is_closed(monkeypatch, stream_queries):
    monkeypatch.setattr(
        cuda_driver,
        "cuStreamGetCtx",
        lambda *args: (cuda_driver.CUresult.CUDA_ERROR_INVALID_CONTEXT, cuda_driver.CUcontext(0)),
    )
    with pytest.raises(RuntimeError, match="cuStreamGetCtx failed: CUDA_ERROR_INVALID_CONTEXT"):
        launch._attn_no_cluster(cuda_driver.CUstream(11), 8, 16, 7)


def test_failed_context_resource_query_is_closed(monkeypatch, stream_queries):
    monkeypatch.setattr(
        cuda_driver,
        "cuStreamGetCtx",
        lambda *args: (cuda_driver.CUresult.CUDA_SUCCESS, cuda_driver.CUcontext(111)),
    )
    monkeypatch.setattr(
        cuda_driver,
        "cuCtxGetDevResource",
        lambda *args: (cuda_driver.CUresult.CUDA_ERROR_NOT_SUPPORTED, _resource(148)),
    )
    with pytest.raises(
        RuntimeError, match="stream SM resource query failed: CUDA_ERROR_NOT_SUPPORTED"
    ):
        launch._attn_no_cluster(cuda_driver.CUstream(11), 8, 16, 7)


@pytest.mark.parametrize("clusters", [1, 8])
def test_unknown_cluster_stream_owner_is_closed(monkeypatch, stream_queries, clusters):
    stream_queries[11] = (64, 0)
    monkeypatch.setattr(
        cuda_driver,
        "cuStreamGetGreenCtx",
        lambda *args: (cuda_driver.CUresult.CUDA_ERROR_INVALID_CONTEXT, cuda_driver.CUgreenCtx(0)),
    )
    with pytest.raises(
        RuntimeError, match="cuStreamGetGreenCtx failed: CUDA_ERROR_INVALID_CONTEXT"
    ):
        launch._attn_no_cluster(cuda_driver.CUstream(11), clusters, 16, 7)
