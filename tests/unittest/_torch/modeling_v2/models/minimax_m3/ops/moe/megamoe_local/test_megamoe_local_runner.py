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
"""CPU tests for the MiniMax-M3.1 MegaMoE local runner: the default tactics by problem shape and their switches, the
gather-index padding fill, and FASTLAUNCH's record / replay / invalidation."""

import types
import weakref

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.moe.megamoe_local import (
    local_impl as local_mod,
)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.moe.megamoe_local import (
    local_plan as plan,
)
from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.moe.megamoe_local import local_runner
from tensorrt_llm._torch.moe.custom_ops import cute_dsl_megamoe_custom_op as megamoe_op

pytestmark = pytest.mark.cpu_only


def _problem_key(world_size, experts_per_rank, max_tokens, helper_expert_count=0):
    """The runner's problem key of a MiniMax-M3.1 NVFP4 MegaMoE launch."""
    return megamoe_op._megamoe_problem_key(
        sm_version=107,
        world_size=world_size,
        num_topk=4,
        num_experts_per_rank=experts_per_rank,
        hidden_size=6144,
        intermediate_size_per_partition=3072,
        expand_intermediate_size_per_partition=6144,
        max_tokens_per_rank=max_tokens,
        output_dtype=torch.bfloat16,
        apply_topk_in_fc1=False,
        gate_up_clamp=7.0,
        situ_beta=None,
        situ_linear_beta=None,
        in_kernel_fc2_reduce=False,
        combine_format="bf16",
        swiglu_alpha=1.702,
        swiglu_beta=1.0,
        helper_expert_count=helper_expert_count,
    )


DEP8_B128 = _problem_key(8, 16, 128)
DEP2_B16K = _problem_key(2, 64, 16384)


@pytest.mark.parametrize(
    "key, switch",
    [(DEP8_B128, plan.HINT8_DISABLE_ENV), (DEP2_B16K, plan.CTX_T16K_DISABLE_ENV)],
    ids=["dep8_b128", "dep2_b16384"],
)
def test_default_tactic_by_shape_and_switch(key, switch):
    entry = plan.default_tactic(key, {})
    assert entry is not None and entry[0] == switch
    megamoe_op.validate_megamoe_tactic(tuple(entry[1]), sm_version=107)
    for value in ("1", "true", " ON "):
        assert plan.default_tactic(key, {switch: value}) is None
    assert plan.default_tactic(key, {switch: "0"}) == entry
    with pytest.raises(ValueError, match=switch):
        plan.default_tactic(key, {switch: "maybe"})


@pytest.mark.parametrize(
    "key",
    [
        _problem_key(8, 16, 64),
        _problem_key(8, 16, 256),
        _problem_key(4, 32, 128),
        _problem_key(2, 64, 8192),
        _problem_key(8, 16, 128, helper_expert_count=1),
    ],
    ids=["dep8_b64", "dep8_b256", "dep4_b128", "dep2_b8192", "helpers"],
)
def test_other_shapes_keep_shared_tables(key):
    assert plan.default_tactic(key, {}) is None


def test_hint8_changes_only_the_phase_interleave_hint():
    shared = megamoe_op._unpack_tactic(
        megamoe_op._default_megamoe_tactic_for_key(DEP8_B128, num_tokens=128)
    )
    ours = megamoe_op._unpack_tactic(tuple(plan.default_tactic(DEP8_B128, {})[1]))
    assert shared[3] == ("phase_interleave", 3)
    assert ours[3] == ("phase_interleave", 8)
    assert ours[:3] + ours[4:] == shared[:3] + shared[4:]


def test_t16k_keeps_the_tile_and_reduce_form():
    shared = megamoe_op._unpack_tactic(
        megamoe_op._default_megamoe_tactic_for_key(DEP2_B16K, num_tokens=16384)
    )
    ours = megamoe_op._unpack_tactic(tuple(plan.default_tactic(DEP2_B16K, {})[1]))
    assert ours[0] == shared[0] and ours[1] == shared[1]
    assert ours[11] is False and shared[11] is False
    assert ours[3:7] == (("grouped", 512), "grid_stride", "standalone_warps", True)


class _Workspace:
    def __init__(self, regions):
        self._regions = regions

    def offset(self, name):
        return self._regions[name][0]

    def nbytes(self, name):
        return self._regions[name][1]


def test_gather_fill_writes_out_of_range_ids_into_its_region_only():
    local_workspace = torch.zeros(256, dtype=torch.uint8)
    kernel = types.SimpleNamespace(
        _device_workspace=_Workspace({local_runner.GATHER_INDEX_REGION: (64, 128)})
    )
    assert local_runner.fill_gather_index_padding(kernel, local_workspace) == 32
    ids = local_workspace[64:192].view(torch.int32)
    base = local_runner.GATHER_INDEX_OOB_BASE
    assert torch.equal(ids, torch.arange(base, base + 32, dtype=torch.int32))
    assert not local_workspace[:64].any() and not local_workspace[192:].any()


def test_gather_fill_skips_kernels_without_the_region():
    local_workspace = torch.zeros(64, dtype=torch.uint8)
    assert local_runner.fill_gather_index_padding(object(), local_workspace) == 0
    no_region = types.SimpleNamespace(_device_workspace=_Workspace({}))
    assert local_runner.fill_gather_index_padding(no_region, local_workspace) == 0
    assert not local_workspace.any()


def test_recording_is_armed_per_call():
    local_runner.record_next_launch()
    assert local_runner.take_recorded_launch() is None
    assert local_runner.take_recorded_launch() is None


# ---------------------------------------------------------------------------
# FASTLAUNCH
# ---------------------------------------------------------------------------
class _Fences:
    """The op's fence registries for one workspace pairing (``fence_key``): ``fence`` fences a fresh allocation of
    both workspaces at the pairing's addresses, as the op does before a launch on an unfenced pairing."""

    def __init__(self):
        self.key = ("local", "shared", 2, "proto")
        self.storages = None

    def fence(self):
        self.storages = (torch.zeros(1).untyped_storage(), torch.zeros(1).untyped_storage())
        megamoe_op._MEGAMOE_FENCED_KEYS.add(self.key)
        megamoe_op._MEGAMOE_FENCED_KEY_WS_REFS[self.key] = tuple(
            weakref.ref(storage) for storage in self.storages
        )

    def refence_same_allocations(self):
        megamoe_op._MEGAMOE_FENCED_KEY_WS_REFS[self.key] = tuple(
            weakref.ref(storage) for storage in self.storages
        )

    def free(self):
        self.storages = None


class _Launch:
    """A stand-in for the stock launch path: counts calls, fences an unfenced pairing and records a launch when
    armed, as the op and the runner do."""

    def __init__(self, fences):
        self.calls = 0
        self.fences = fences

    def __call__(self, backend, **kwargs):
        self.calls += 1
        if not megamoe_op._megamoe_fence_key_live(self.fences.key):
            self.fences.fence()
        if getattr(local_runner._LAST_LAUNCH, "armed", False):
            local_runner._LAST_LAUNCH.armed = False
            local_runner._LAST_LAUNCH.launch = local_runner.RecordedLaunch(
                compiled=None,
                launch_kwargs={"output_activation": None},
                fence_key=self.fences.key,
                fence_refs=megamoe_op._MEGAMOE_FENCED_KEY_WS_REFS.get(self.fences.key),
            )
        return "op"


@pytest.fixture
def fast_backend(monkeypatch):
    backend = local_mod.TrtllmCutedslMegaMoeLocalNvfp4Impl.__new__(
        local_mod.TrtllmCutedslMegaMoeLocalNvfp4Impl
    )
    backend._fast_launch_enabled = True
    backend._fast_launches = {}
    backend._in_kernel_reduce_buckets = {}
    backend._shared_expert = None
    backend.tactic_autotune = False
    keys = {"key": ("bucket", 128)}
    monkeypatch.setattr(backend, "_fast_launch_key", lambda **kw: keys["key"], raising=False)
    relaunches = []
    monkeypatch.setattr(
        backend,
        "_fast_relaunch",
        lambda launch, **kw: relaunches.append(launch) or "replayed",
        raising=False,
    )
    monkeypatch.setattr(megamoe_op, "_MEGAMOE_FENCED_KEYS", set())
    monkeypatch.setattr(megamoe_op, "_MEGAMOE_FENCED_KEY_WS_REFS", {})
    fences = _Fences()
    stock = _Launch(fences)
    monkeypatch.setattr(
        local_mod.TrtllmCutedslMegaMoeNvfp4Impl,
        "_launch_megamoe_kernel",
        lambda self, **kw: stock(self, **kw),
    )
    monkeypatch.setattr(local_runner, "local_runner_class", lambda: None)
    return types.SimpleNamespace(
        backend=backend, stock=stock, relaunches=relaunches, fences=fences, keys=keys
    )


_BUFFERS = {
    name: torch.zeros(4, dtype=torch.uint8)
    for name in (
        "activation",
        "activation_sf",
        "topk_idx",
        "topk_weights",
        "combine_output",
        "shared_workspace",
    )
}


def _launch(backend):
    return backend._launch_megamoe_kernel(
        num_tokens=3, hidden=8, output_dtype=torch.bfloat16, launch_max_T=128, **_BUFFERS
    )


def test_fastlaunch_records_then_replays(fast_backend):
    assert _launch(fast_backend.backend) == "op"
    assert fast_backend.stock.calls == 1 and len(fast_backend.backend._fast_launches) == 1
    recorded = fast_backend.backend._fast_launches[("bucket", 128)]
    assert all(a is b for a, b in zip(recorded.keepalive, _BUFFERS.values()))
    assert _launch(fast_backend.backend) == "replayed"
    assert _launch(fast_backend.backend) == "replayed"
    assert fast_backend.stock.calls == 1 and len(fast_backend.relaunches) == 2


def test_fastlaunch_new_key_goes_through_the_op(fast_backend):
    _launch(fast_backend.backend)
    fast_backend.keys["key"] = ("bucket", 256)
    assert _launch(fast_backend.backend) == "op"
    assert fast_backend.stock.calls == 2 and len(fast_backend.backend._fast_launches) == 2


def test_fastlaunch_drops_a_launch_whose_workspace_died(fast_backend):
    _launch(fast_backend.backend)
    fast_backend.fences.free()
    assert _launch(fast_backend.backend) == "op"
    assert fast_backend.stock.calls == 2 and not fast_backend.relaunches
    # The op path fenced the new pairing and recorded the launch again.
    assert _launch(fast_backend.backend) == "replayed"


def test_fastlaunch_rejects_a_recycled_workspace_address(fast_backend):
    _launch(fast_backend.backend)
    # The recorded workspaces are freed and new ones are allocated and fenced at the same addresses: the fence key
    # is live again, but for other allocations.
    fast_backend.fences.free()
    fast_backend.fences.fence()
    assert megamoe_op._megamoe_fence_key_live(fast_backend.fences.key)
    assert _launch(fast_backend.backend) == "op"
    assert fast_backend.stock.calls == 2 and not fast_backend.relaunches


def test_fastlaunch_rejects_a_refenced_pairing(fast_backend):
    _launch(fast_backend.backend)
    fast_backend.fences.refence_same_allocations()
    assert _launch(fast_backend.backend) == "op"
    assert fast_backend.stock.calls == 2 and not fast_backend.relaunches
    assert _launch(fast_backend.backend) == "replayed"


def test_fastlaunch_ineligible_launch_records_nothing(fast_backend):
    fast_backend.keys["key"] = None
    assert _launch(fast_backend.backend) == "op"
    assert _launch(fast_backend.backend) == "op"
    assert fast_backend.stock.calls == 2 and not fast_backend.backend._fast_launches
    assert local_runner.take_recorded_launch() is None


def test_fastlaunch_cache_is_bounded(fast_backend, monkeypatch):
    monkeypatch.setattr(local_mod, "FASTLAUNCH_MAX_ENTRIES", 2)
    for bucket in (1, 2, 3):
        fast_backend.keys["key"] = ("bucket", bucket)
        _launch(fast_backend.backend)
    assert list(fast_backend.backend._fast_launches) == [("bucket", 3)]


def _real_key_kwargs():
    tensor = torch.zeros(4, dtype=torch.uint8)
    weight_view = types.SimpleNamespace(
        fc1_weight=tensor,
        fc1_weight_sf=tensor,
        fc2_weight=tensor,
        fc2_weight_sf=tensor,
        fc31_alpha=tensor,
        fc2_alpha=tensor,
        fc1_norm_const=tensor,
    )
    return dict(
        activation=tensor,
        activation_sf=tensor,
        topk_idx=tensor,
        topk_weights=tensor,
        weight_view=weight_view,
        combine_output=tensor,
        shared_workspace=tensor,
        world_size=2,
        local_rank=0,
        top_k=4,
        hidden=8,
        peer_offsets=[0, 0],
        num_tokens=3,
        output_dtype=torch.bfloat16,
        launch_max_T=128,
        replica_plan=None,
        staged_inputs=True,
    )


@pytest.fixture
def key_backend(monkeypatch):
    backend = local_mod.TrtllmCutedslMegaMoeLocalNvfp4Impl.__new__(
        local_mod.TrtllmCutedslMegaMoeLocalNvfp4Impl
    )
    backend._fast_launch_enabled = True
    # Every bucket the tests key is cached, so no key computes the in-kernel reduce choice (which reads the
    # device's SM version and attributes this bare backend does not have).
    backend._in_kernel_reduce_buckets = {128: False, 256: False, 32: True}
    backend._shared_expert = None
    backend.tactic_autotune = False
    monkeypatch.setattr(type(backend), "compute_slot_count", 16, raising=False)
    tuning = types.SimpleNamespace(is_tuning_mode=False)
    monkeypatch.setattr(local_mod, "AutoTuner", types.SimpleNamespace(get=lambda: tuning))
    capturing = {"on": False}
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing["on"])
    monkeypatch.setattr(torch._C, "_cuda_getDevice", lambda: 0, raising=False)
    monkeypatch.setattr(torch._C, "_cuda_getCurrentRawStream", lambda device: 7, raising=False)
    return types.SimpleNamespace(backend=backend, tuning=tuning, capturing=capturing)


def test_fastlaunch_key_tracks_buffers_and_stream(key_backend, monkeypatch):
    backend = key_backend.backend
    kwargs = _real_key_kwargs()
    key = backend._fast_launch_key(**kwargs)
    assert key is not None and key == backend._fast_launch_key(**kwargs)
    other_workspace = torch.zeros(4, dtype=torch.uint8)
    assert key != backend._fast_launch_key(**dict(kwargs, shared_workspace=other_workspace))
    other_activation = torch.zeros(4, dtype=torch.uint8)
    assert key != backend._fast_launch_key(**dict(kwargs, activation=other_activation))
    assert key != backend._fast_launch_key(**dict(kwargs, peer_offsets=[0, 64]))
    assert key != backend._fast_launch_key(**dict(kwargs, launch_max_T=256))
    monkeypatch.setattr(torch._C, "_cuda_getCurrentRawStream", lambda device: 9, raising=False)
    assert key != backend._fast_launch_key(**kwargs)


@pytest.mark.parametrize(
    "case",
    ["switch_off", "unstaged", "replica_plan", "in_kernel_reduce", "capture", "tuning", "autotune"],
)
def test_fastlaunch_key_declines(key_backend, case):
    backend = key_backend.backend
    kwargs = _real_key_kwargs()
    if case == "switch_off":
        backend._fast_launch_enabled = False
    elif case == "unstaged":
        kwargs["staged_inputs"] = False
    elif case == "replica_plan":
        kwargs["replica_plan"] = object()
    elif case == "in_kernel_reduce":
        kwargs["launch_max_T"] = 32
    elif case == "capture":
        key_backend.capturing["on"] = True
    elif case == "tuning":
        key_backend.tuning.is_tuning_mode = True
    else:
        backend.tactic_autotune = True
    assert backend._fast_launch_key(**kwargs) is None


def test_runner_selection_nests_and_restores(monkeypatch):
    selected = {"runner_class": "outer"}

    def set_active(runner_class):
        previous, selected["runner_class"] = selected["runner_class"], runner_class
        return previous

    monkeypatch.setattr(megamoe_op, "set_active_megamoe_runner_class", set_active, raising=False)
    monkeypatch.setattr(local_runner, "local_runner_class", lambda: "leaf")
    backend = local_mod.TrtllmCutedslMegaMoeLocalNvfp4Impl.__new__(
        local_mod.TrtllmCutedslMegaMoeLocalNvfp4Impl
    )
    with backend._local_runner():
        assert selected["runner_class"] == "leaf"
        with backend._local_runner():
            assert selected["runner_class"] == "leaf"
        assert selected["runner_class"] == "leaf"
    assert selected["runner_class"] == "outer"
    with pytest.raises(RuntimeError):
        with backend._local_runner():
            raise RuntimeError("launch failed")
    assert selected["runner_class"] == "outer"


@pytest.mark.skipif(
    not megamoe_op.IS_MEGAMOE_OP_AVAILABLE, reason="the MegaMoE op is unavailable on this host"
)
def test_op_runner_selection_is_per_thread():
    import threading

    seen = []
    previous = megamoe_op.set_active_megamoe_runner_class(dict)
    try:
        thread = threading.Thread(
            target=lambda: seen.append(megamoe_op._active_megamoe_runner_class())
        )
        thread.start()
        thread.join()
        assert seen == [None]
        assert megamoe_op._active_megamoe_runner_class() is dict
    finally:
        assert megamoe_op.set_active_megamoe_runner_class(previous) is dict
    assert megamoe_op._active_megamoe_runner_class() is previous


def test_switches_are_registered_in_the_targets():
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (
        m31_switches as dep,
    )
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep2ctx import (
        m31_switches as dep2ctx,
    )

    prefix = "TRTLLM_M31_DISABLE_"
    for name in (plan.HINT8_DISABLE_ENV, plan.GATHER_FILL_DISABLE_ENV, plan.FASTLAUNCH_DISABLE_ENV):
        assert name[len(prefix) :] in dep.SWITCHES
    for name in (
        plan.CTX_T16K_DISABLE_ENV,
        plan.GATHER_FILL_DISABLE_ENV,
        plan.FASTLAUNCH_DISABLE_ENV,
    ):
        assert name[len(prefix) :] in dep2ctx.SWITCHES
