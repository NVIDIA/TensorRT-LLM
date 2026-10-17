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
"""Package cross_layer_expert_prefetch's speculative expert prefetch in hop_routing's routed_moe fork.

* Every routed-MoE runner class of the package tree (routed_moe's runner and every
  subclass, e.g. hop_routing's fork) compiles the prefetch when its key carries a
  cross_layer_expert_prefetch spec (``key[6]``): ``spec_k`` / ``spec_ctas`` / ``spec_self`` of the
  kernel equal the hook's values, ``spec_k == 0`` without a hook, and the
  capability line names the runner's package. A runner that dropped the spec
  would compile the prefetch out silently (the call still passes the spec
  row's address). Kernel compilation is stubbed: no GPU needed.
* On an SM100-family GPU with real GLM-5.2 weights (``GLM_MOE_DSA_REAL_MOE_INPUTS_DIR``), the
  fork's outputs with the prefetch equal the fork's outputs without it bit
  for bit (raw tuple, finalized output; the FP32 accumulator within the FP32
  reorder bound of the exact sum), for valid and out-of-range expert IDs, R
  = 1 / 6 / 8 and K = 8 / 12 with all or a subset of issuing CTAs; the
  self-prediction store writes the call's distinct experts ranked by row
  count; a CUDA graph with the prefetch replays bit-identically.
"""

import importlib
import os
import pkgutil
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_ROOT, "routed_moe"))
sys.path.insert(0, os.path.join(_ROOT, "hop_routing"))

KMAX = 32
REORDER_REL = 2.0**-20
_PKG = "tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels"

sm100 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10,
    reason="needs an SM100-family GPU",
)


# ---------------------------------------------------------------------------
# Structural guard: the spec reaches every runner's kernel (no compilation)
# ---------------------------------------------------------------------------
def _runner_classes():
    """routed_moe's routed-MoE runner and every subclass in the package tree."""
    pkg = importlib.import_module(_PKG)
    for mod in pkgutil.iter_modules(pkg.__path__):
        if mod.ispkg:
            try:
                importlib.import_module(f"{_PKG}.{mod.name}.routed_moe")
            except ModuleNotFoundError:
                pass
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.routed_moe.routed_moe import (  # noqa: E501
        RoutedMoe,
    )

    out, stack = {RoutedMoe}, [RoutedMoe]
    while stack:
        for sub in stack.pop().__subclasses__():
            if sub not in out:
                out.add(sub)
                stack.append(sub)
    return sorted(out, key=lambda c: f"{c.__module__}.{c.__qualname__}")


class _Owner:
    """Stands in for the NVFP4 expert leaf: only what the compile path reads."""

    def __init__(self):
        for name in ("w3_w1_weight", "w3_w1_weight_scale", "w2_weight", "w2_weight_scale"):
            setattr(self, name, torch.zeros(64, dtype=torch.uint8))


class _Stream:
    cuda_stream = 0


def _compile_kernels(cls, monkeypatch, hook):
    """The kernel objects ``cls._compiled`` constructs for an FP32-accumulator call (stubbed compile)."""
    import cutlass.cute as cute

    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.routed_moe import (  # noqa: E501
        routed_moe as routed_moe_mod,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch

    built = []

    def fake_compile(fn, *args, **kwargs):
        built.append(getattr(fn, "__self__", fn))
        return object()

    logged = []
    monkeypatch.setattr(cute, "compile", fake_compile)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a, **k: _Stream())
    monkeypatch.setattr(
        dispatch, "log_capability", lambda f, k, e, r="": logged.append((f, k, e)) or e
    )
    # 24 co-resident four-CTA clusters: a 96-CTA expert-cluster grid, as on the served GPU.
    monkeypatch.setattr(routed_moe_mod, "_max_active_clusters", lambda device, cluster: 24)
    # Each class keeps its own compile cache: start from empty ones.
    monkeypatch.setattr(routed_moe_mod, "_COMPILED", {})
    mod = sys.modules[cls.__module__]
    if hasattr(mod, "_COMPILED"):
        monkeypatch.setattr(mod, "_COMPILED", {})
    owner = _Owner()
    if hook is not None:
        owner._expert_prefetch_spec = hook
    runner = cls.__new__(cls)
    runner.module = owner
    runner.bias = torch.zeros(256, dtype=torch.bfloat16)
    runner.device = torch.device("cpu")
    runner.num_ctas = 152
    key = runner._key(False, 0, 6, True)
    runner._compiled(False, 0, 6, True)
    kernels = [k for k in built if hasattr(k, "spec_k")]
    assert len(kernels) == 1, f"{cls.__qualname__}: expected one expert-cluster kernel, got {built}"
    return key, kernels[0], logged


@pytest.mark.parametrize("spec", [(12, 0, True), (8, 32, True), (8, 40, False), (0, 0, True)])
def test_every_runner_compiles_the_expert_prefetch_spec(monkeypatch, spec):
    ids = torch.zeros(KMAX, dtype=torch.int32)
    classes = _runner_classes()
    assert any(c.__module__.endswith(".hop_routing.routed_moe") for c in classes), classes
    for cls in classes:
        key, kern, logged = _compile_kernels(cls, monkeypatch, lambda: (ids,) + spec)
        assert key[6] == spec, (cls.__qualname__, key)
        got = (kern.spec_k, kern.spec_ctas, kern.spec_self)
        assert got == spec, (
            f"{cls.__qualname__} dropped the cross_layer_expert_prefetch spec: kernel {got}, key {spec}"
        )
        if spec[0]:
            pkg = cls.__module__.split(".")[-2]
            line = f"expert_prefetch[k={spec[0]},ctas={spec[1]},self={int(spec[2])}]"
            assert (pkg, line, True) in logged, (cls.__qualname__, logged)


def test_runner_without_expert_prefetch_has_no_prefetch(monkeypatch):
    for cls in _runner_classes():
        key, kern, logged = _compile_kernels(cls, monkeypatch, None)
        assert key[6] is None
        assert (kern.spec_k, kern.spec_ctas, kern.spec_self) == (0, 0, False), cls.__qualname__
        assert not [entry for entry in logged if entry[1].startswith("expert_prefetch")], logged


def test_expert_prefetch_requires_routed_moe():
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared import dispatch

    assert dispatch.REQUIRED_FEATURES["cross_layer_expert_prefetch"] == frozenset({"routed_moe"})
    auto = dispatch.ModelingV2Mode.AUTO
    assert dispatch.resolve(auto, {}).enabled("cross_layer_expert_prefetch")
    plan = dispatch.resolve(auto, {dispatch.DISABLE_ENV: "routed_moe"})
    assert not plan.enabled("cross_layer_expert_prefetch")


# ---------------------------------------------------------------------------
# Real weights: the fork with and without the prefetch
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def layer():
    import routed_moe_fixtures as fx

    torch.manual_seed(0)
    return fx.build_native_layer(
        layer=int(os.environ.get("GLM_MOE_DSA_ROUTED_MOE_LAYER", "10")),
        tp_rank=int(os.environ.get("GLM_MOE_DSA_ROUTED_MOE_RANK", "0")),
    )


@pytest.fixture(scope="module")
def fork(layer):
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.glm_5_2_nvfp4__sm_103__tp8.kernels.hop_routing.routed_moe import (  # noqa: E501
        HopRoutedMoe,
    )

    r = HopRoutedMoe(layer.module, layer.bias)
    if r.schedule(8, False, True) != "xc":
        pytest.skip("the speculative prefetch is part of the expert-cluster schedule")
    return r


class _Spec:
    """Install a cross_layer_expert_prefetch hook on the fork's expert leaf."""

    def __init__(self, runner, ids, k, ctas=0, self_store=False):
        self.owner = runner.module
        self.spec = (ids, k, ctas, self_store)

    def __enter__(self):
        self.owner._expert_prefetch_spec = lambda: self.spec
        return self

    def __exit__(self, *exc):
        del self.owner._expert_prefetch_spec


def _pack(logits, bias):
    import hop_routing_oracle as orc

    ids, w = orc.native_routing(logits, bias)
    routing = torch.empty(2, ids.shape[0], 8, dtype=torch.int32, device="cuda")
    routing[0].copy_(ids)
    routing[1].copy_(w.float().view(torch.int32))
    return routing


def _windows(layer, rows, count, offset=0):
    import routed_moe_fixtures as fx

    win = [
        (x, lg)
        for x, lg in fx.real_windows(fx.load_real_capture(layer.layer), rows)
        if x is not None
    ]
    if not win:
        pytest.skip(
            f"no captured hidden states of layer {layer.layer} (GLM_MOE_DSA_REAL_MOE_INPUTS_DIR)"
        )
    step = max(1, len(win) // count)
    out = []
    for i in range(count):
        x, lg = win[(offset + i * step) % len(win)]
        x = x.cuda().contiguous()
        lg = lg.float().cuda().contiguous()
        out.append((x, _pack(lg, layer.bias)))
    return out


def _ids(kind, seed):
    g = torch.Generator().manual_seed(seed)
    if kind == "valid":
        ids = torch.randperm(256, generator=g)[:KMAX]
    else:  # the kernel masks each ID into [0, 256)
        ids = torch.randint(-(2**31), 2**31 - 1, (KMAX,), generator=g)
    return ids.to(torch.int32).cuda()


def _outputs(fork, x, routing):
    fin = fork(x, routing)
    fc2, scale, _ = fork(x, routing, do_finalize=False)
    acc = torch.zeros(x.shape[0], 6144, dtype=torch.float32, device="cuda")
    fork(x, routing, red_out=acc)
    torch.cuda.synchronize()
    return fin.clone(), fc2.clone(), scale.clone(), acc


def _check_same(ref, got, what):
    fin, fc2, scale, _ = ref
    gfin, gfc2, gscale, gacc = got
    assert torch.equal(gfin, fin), f"{what}: finalized output differs"
    assert torch.equal(gfc2, fc2), f"{what}: raw FC2 rows differ"
    assert torch.equal(gscale, scale), f"{what}: routing weights differ"
    rows = gacc.shape[0]
    terms = scale.double()[:, :, None] * fc2.view(rows, 8, -1).double()
    exact = terms.sum(1)
    bound = REORDER_REL * terms.abs().sum(1)
    assert ((gacc.double() - exact).abs() <= bound).all(), f"{what}: accumulator"


@sm100
@pytest.mark.parametrize("rows", [1, 6, 8])
@pytest.mark.parametrize("k,ctas", [(12, 0), (8, 32), (8, 0), (12, 40)])
@pytest.mark.parametrize("ids_kind", ["valid", "out_of_range"])
def test_fork_prefetch_outputs_bit_identical(layer, fork, rows, k, ctas, ids_kind):
    for i, (x, routing) in enumerate(_windows(layer, rows, 2, offset=rows)):
        ref = _outputs(fork, x, routing)
        ids = _ids(ids_kind, seed=rows * 100 + i)
        with _Spec(fork, ids, k, ctas):
            got = _outputs(fork, x, routing)
        _check_same(ref, got, f"rows={rows} k={k} ctas={ctas} {ids_kind} window {i}")


def _ranked_experts(routing):
    ids = routing[0].flatten().long().cpu()
    counts = torch.bincount(ids, minlength=256)
    distinct = [e for e in range(256) if counts[e] > 0]
    return sorted(distinct, key=lambda e: (-int(counts[e]), e))


@sm100
@pytest.mark.parametrize("rows", [1, 3, 6, 8])
@pytest.mark.parametrize("k,ctas", [(12, 0), (8, 32)])
def test_fork_self_store_ranks_by_row_count(layer, fork, rows, k, ctas):
    for i, (x, routing) in enumerate(_windows(layer, rows, 3, offset=5 * rows)):
        ref = _outputs(fork, x, routing)
        ids = torch.full((KMAX,), 999, dtype=torch.int32, device="cuda")
        with _Spec(fork, ids, k, ctas, self_store=True):
            acc = torch.zeros(rows, 6144, dtype=torch.float32, device="cuda")
            fork(x, routing, red_out=acc)
            torch.cuda.synchronize()
            got = ids.tolist()
            # the store is idempotent for the same call (the prefetch reads the
            # previous call's row first)
            fc2, scale, _ = fork(x, routing, do_finalize=False)
            gfin = fork(x, routing)
            torch.cuda.synchronize()
        assert torch.equal(fc2, ref[1]) and torch.equal(scale, ref[2]), (rows, i)
        assert torch.equal(gfin, ref[0]), (rows, i)
        want = _ranked_experts(routing)[:KMAX]
        assert got[: len(want)] == want, (rows, i, got, want)
        assert all(v == 999 for v in got[len(want) :]), (rows, i, got)


@sm100
def test_fork_prefetch_cuda_graph_replay(layer, fork):
    rows = 6
    wins = _windows(layer, rows, 6, offset=3)
    # References: the fork without the prefetch, eager.
    refs = []
    for x, routing in wins:
        fc2, scale, _ = fork(x, routing, do_finalize=False)
        torch.cuda.synchronize()
        refs.append((fc2.clone(), scale.clone()))
    x_static = wins[0][0].clone()
    r_static = wins[0][1].clone()
    ids = torch.zeros(KMAX, dtype=torch.int32, device="cuda")
    with _Spec(fork, ids, 8, 32, self_store=True):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            fork(x_static, r_static, do_finalize=False)
        torch.cuda.current_stream().wait_stream(stream)
        # the eager call compiled the prefetch variant: the capture records it
        assert fork.ready_for_capture(False, rows=rows)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            fc2_s, scale_s, _ = fork(x_static, r_static, do_finalize=False)
    for it, (x, routing) in enumerate(wins):
        x_static.copy_(x)
        r_static.copy_(routing)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(fc2_s, refs[it][0]) and torch.equal(scale_s, refs[it][1]), it
        # the replay stored this call's ranked experts for the next one
        want = _ranked_experts(routing)[:KMAX]
        assert ids.tolist()[: len(want)] == want, it
    del graph
