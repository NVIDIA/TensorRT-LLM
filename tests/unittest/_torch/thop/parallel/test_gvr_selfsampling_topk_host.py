# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only tests for self-sampling GVR host routing and locality caches."""

import importlib.util
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

_HOST_PATH = (
    Path(__file__).parents[5]
    / "tensorrt_llm/_torch/cute_dsl_kernels/blackwell/top_k/gvr_topk_decode_self_sampling_host.py"
)
_SPEC = importlib.util.spec_from_file_location("gvr_self_sampling_host_cpu_test", _HOST_PATH)
assert _SPEC is not None and _SPEC.loader is not None
ss_host = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ss_host)


def test_default_route_preserves_b200_plans():
    """The new topology argument must preserve every default B200 decision."""
    for batch_size in (1, 8, 16, 64, 148, 296, 1024):
        for n_valid in (4096, 65536, 131072, 262144):
            for top_k in (512, 1024, 2048):
                default = ss_host.route(batch_size, n_valid, n_valid, top_k)
                explicit = ss_host.route(
                    batch_size,
                    n_valid,
                    n_valid,
                    top_k,
                    num_sms=148,
                )
                assert default == explicit
                assert (
                    ss_host.route_split(
                        batch_size,
                        n_valid,
                        n_valid,
                        top_k,
                        num_sms=148,
                    )
                    == explicit
                )


def test_rubin_full_device_route_uses_available_sms():
    """Single-row splits may use all SMs; MAXC limits rows, not per-row CTAs."""
    b200 = ss_host.route(1, 1 << 20, 1 << 20, 1024)
    rubin_212 = ss_host.route(1, 1 << 20, 1 << 20, 1024, num_sms=212)
    rubin_224 = ss_host.route(1, 1 << 20, 1 << 20, 1024, num_sms=224)
    assert b200["rt"]["R"] == 148
    assert rubin_212["kernel"] == rubin_224["kernel"] == "main"
    assert rubin_212["rt"]["R"] == rubin_212["grid"][0] == 212
    assert rubin_224["rt"]["R"] == rubin_224["grid"][0] == 224

    # The 74-row B200 half-wave threshold scales to 106 rows on R200.
    assert ss_host.route(80, 262144, 262144, 1024, num_sms=148)["kernel"] == "main"
    assert ss_host.route(80, 262144, 262144, 1024, num_sms=212)["kernel"] == "clus"

    # A synthetic topology above 320 SMs must not select SPLIT for more
    # than the workspace's 160 row slabs.
    assert ss_host.route(160, 262144, 262144, 1024, num_sms=400)["grid"][0] == 2
    assert ss_host.route(161, 262144, 262144, 1024, num_sms=400)["grid"][0] == 1


def test_route_dynamic_requires_matching_execution_topology():
    """A non-default static plan must not silently use the B200 SM count."""
    args = (80, 8192, 8192, 512)
    static = ss_host.route_static(*args, num_sms=64)
    dynamic, smem = ss_host.route_dynamic(static, args[1], num_sms=64)
    recombined = {
        key: (dict(value) if isinstance(value, dict) else value) for key, value in static.items()
    }
    recombined["rt"].update(dynamic)
    recombined["smem"] = smem
    assert recombined == ss_host.route(*args, num_sms=64)
    with pytest.raises(TypeError, match="num_sms"):
        ss_host.route_dynamic(static, args[1])


def test_execution_domain_uses_full_and_partition_sm_counts(monkeypatch):
    """Full-device properties are cached; a current domain uses its partition."""
    calls = []
    device_contexts = []

    @contextmanager
    def fake_device(device):
        device_contexts.append(device)
        yield

    monkeypatch.setattr(ss_host, "_current_locality_domain", lambda: None)
    monkeypatch.setattr(ss_host.torch.cuda, "device", fake_device)
    monkeypatch.setattr(
        ss_host.torch.cuda,
        "get_device_properties",
        lambda device: calls.append(device)
        or SimpleNamespace(major=10, minor=7, multi_processor_count=212),
    )
    ss_host._DEVICE_COMPUTE_INFO.clear()
    assert ss_host._execution_domain(0) == (212, None)
    assert ss_host._execution_domain(ss_host.torch.device("cuda:0")) == (212, None)
    assert calls == [0]

    monkeypatch.setattr(ss_host, "_current_locality_domain", lambda: 1)
    monkeypatch.setattr(ss_host, "_locality_domain_topology", lambda: ((104, 212), (108, 212)))
    assert ss_host._execution_domain(0) == (108, 1)
    assert device_contexts == [0]

    monkeypatch.setattr(ss_host, "_locality_domain_topology", lambda: ((104, 224), (108, 224)))
    with pytest.raises(RuntimeError, match="topology.*target device"):
        ss_host._execution_domain(ss_host.torch.device("cuda:0"))


def test_b200_execution_domain_does_not_query_locality_runtime(monkeypatch):
    """B200 keeps the pre-Rubin dependency and hot-path behavior."""
    monkeypatch.setattr(
        ss_host.torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(major=10, minor=0, multi_processor_count=148),
    )
    monkeypatch.setattr(
        ss_host,
        "_current_locality_domain",
        lambda: (_ for _ in ()).throw(AssertionError("locality runtime queried on B200")),
    )
    ss_host._DEVICE_COMPUTE_INFO.clear()
    assert ss_host._execution_domain(0) == (148, None)


class _FakeDeviceModule:
    STATIC_BYTES = 0

    def __init__(self):
        self.calls = []

    def _compiled(self, family, *args, **kwargs):
        marker = object()
        self.calls.append((family, args, kwargs, marker))
        return marker

    def get_compiled(self, *args, **kwargs):
        return self._compiled("main", *args, **kwargs)

    def get_compiled__reg(self, *args, **kwargs):
        return self._compiled("reg", *args, **kwargs)

    def get_compiled__regclus(self, *args, **kwargs):
        return self._compiled("reg_clus", *args, **kwargs)

    def get_compiled__clus(self, *args, **kwargs):
        return self._compiled("clus", *args, **kwargs)


def test_varlen_cache_separates_full_device_and_locality_domain(monkeypatch):
    """Equal SM counts in different execution domains must not alias launchers."""
    fake_device = _FakeDeviceModule()
    monkeypatch.setattr(ss_host, "_device", lambda: fake_device)
    ss_host._VARLEN_CACHE.clear()
    args = (64, 262144, 1024, 262144, 1, 4)
    full = ss_host._varlen_launcher(*args, num_sms=212, locality_domain_id=None)
    local = ss_host._varlen_launcher(*args, num_sms=212, locality_domain_id=0)
    assert full is not local
    assert (*args, 212, None) in ss_host._VARLEN_CACHE
    assert (*args, 212, 0) in ss_host._VARLEN_CACHE
    clus_calls = [call for call in fake_device.calls if call[0] == "clus"]
    assert clus_calls
    assert all(call[2]["num_sms"] == 212 for call in clus_calls)


def test_default_workspace_is_locality_domain_scoped(monkeypatch):
    """Concurrent locality streams receive distinct, domain-local slabs."""
    allocations = []
    pool_entries = []
    device_contexts = []

    class FakeBuffer:
        def __init__(self, allocation_id):
            self.allocation_id = allocation_id

        def view(self, dtype):
            return self, dtype

    @contextmanager
    def fake_pool():
        pool_entries.append("enter")
        yield

    @contextmanager
    def fake_device(device):
        device_contexts.append(device)
        yield

    ref = SimpleNamespace(get_device=lambda: 0, device="cuda:0")
    monkeypatch.setattr(ss_host, "_optional_locality_domain_mem_pool", fake_pool)
    monkeypatch.setattr(ss_host.torch.cuda, "device", fake_device)
    monkeypatch.setattr(
        ss_host.torch,
        "zeros",
        lambda *args, **kwargs: allocations.append((args, kwargs)) or FakeBuffer(len(allocations)),
    )
    ss_host._ws_keep.clear()
    full = ss_host._default_workspace(ref, None)
    domain0 = ss_host._default_workspace(ref, 0)
    domain1 = ss_host._default_workspace(ref, 1)
    assert ss_host._default_workspace(ref, 0) is domain0
    assert len({id(full), id(domain0), id(domain1)}) == 3
    assert set(ss_host._ws_keep) == {0, (0, 0), (0, 1)}
    assert len(allocations) == 3
    assert len(pool_entries) == 2
    assert device_contexts == [0, 0]


def test_varlen_capture_workspace_miss_does_not_allocate(monkeypatch):
    """A cold domain slab is never allocated or published inside capture."""
    allocations = []
    ref = SimpleNamespace(get_device=lambda: 0)
    monkeypatch.setattr(ss_host, "_is_capturing", lambda: True)
    monkeypatch.setattr(
        ss_host,
        "_default_workspace",
        lambda *args: allocations.append(args),
    )
    ss_host._ws_keep.clear()

    with pytest.raises(RuntimeError, match="workspace.*warm up"):
        ss_host._workspace_for_varlen_launch(ref, None, 1)
    assert allocations == []
    assert ss_host._ws_keep == {}


def test_varlen_capture_checks_launcher_before_workspace(monkeypatch):
    """A launcher miss aborts capture before workspace resolution begins."""

    class FakeTensor:
        is_cuda = True

        def __init__(self, shape, dtype):
            self.shape = shape
            self.dtype = dtype

        def dim(self):
            return len(self.shape)

        def get_device(self):
            return 0

        def stride(self, dim):
            return self.shape[1] if dim == 0 else 1

        def is_contiguous(self):
            return True

        def data_ptr(self):
            return 16

    logits = FakeTensor((1, 4096), ss_host.torch.float32)
    kv_lens = FakeTensor((1,), ss_host.torch.int32)
    indices = FakeTensor((1, 512), ss_host.torch.int32)
    workspace_calls = []
    monkeypatch.setattr(ss_host, "_TENSOR", FakeTensor)
    monkeypatch.setattr(ss_host, "_execution_domain", lambda device: (108, 1))
    monkeypatch.setattr(ss_host, "_is_capturing", lambda: True)
    monkeypatch.setattr(
        ss_host,
        "_workspace_for_varlen_launch",
        lambda *args: workspace_calls.append(args),
    )
    ss_host._VARLEN_CACHE.clear()

    with pytest.raises(RuntimeError, match="launcher.*warm up"):
        ss_host.run_varlen(logits, kv_lens, indices, max_seq_len=4096)
    assert workspace_calls == []


def test_warmup_done_key_includes_exact_requested_rows(monkeypatch):
    """Equal band representatives must not hide a new capture row count."""

    class FakeTensor:
        def __getitem__(self, key):
            return self

    exact_launchers = []
    monkeypatch.setattr(ss_host.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(ss_host.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(ss_host, "_execution_domain", lambda device: (212, 0))
    monkeypatch.setattr(ss_host.torch, "zeros", lambda *args, **kwargs: FakeTensor())
    monkeypatch.setattr(ss_host.torch, "full", lambda *args, **kwargs: FakeTensor())
    monkeypatch.setattr(ss_host.torch, "empty", lambda *args, **kwargs: FakeTensor())
    monkeypatch.setattr(ss_host, "run_varlen", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        ss_host,
        "_varlen_launcher",
        lambda rows, *args, **kwargs: exact_launchers.append(rows),
    )
    ss_host._VARLEN_WARMUP_DONE.clear()

    common = dict(
        top_k=1024,
        max_seq_len=262144,
        compress_ratio=4,
        row_stride=65536,
    )
    ss_host.warmup_varlen(**common, num_rows_list=(32, 128))
    first_call_count = len(exact_launchers)
    ss_host.warmup_varlen(**common, num_rows_list=(64, 128))

    assert first_call_count > 0
    assert len(exact_launchers) > first_call_count
    assert exact_launchers[-2:] == [64, 128]


def test_route_rejects_invalid_sm_count():
    with pytest.raises(RuntimeError, match="num_sms >= 1"):
        ss_host.route(1, 4096, 4096, 512, num_sms=0)
