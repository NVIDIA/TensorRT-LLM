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
"""CPU contracts for the MegaMoE tactic autotune opt-in (MEGAMOE_TACTIC_AUTOTUNE=1).

The runner and backend are the real classes; kernel launches, the symmetric
profiling scratch and torch.distributed are mocks. The autotuner's collective
scoring itself is covered in tests/unittest/_torch/misc/test_autotuner.py.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from _torch.moe.moe_test_utils import MoeBackendType

from tensorrt_llm._torch.autotuner import (
    AutoTuner,
    DistributedTuningStrategy,
    OptimizationProfile,
    TuningConfig,
)
from tensorrt_llm._torch.model_config import ModelConfig
from tensorrt_llm._torch.moe.custom_ops import cute_dsl_megamoe_custom_op as op_module
from tensorrt_llm._torch.moe.fused_moe import RenormalizeMoeRoutingMethod
from tensorrt_llm._torch.moe.fused_moe.mega_moe import MegaMoECuteDsl
from tensorrt_llm._torch.moe.fused_moe.mega_moe import mega_moe_cute_dsl as backend_module
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.cpu_only

_HIDDEN = 7168


def _runner_inputs(rows: int) -> list:
    """The op's twelve inputs, with ``rows`` activation rows."""
    return [
        torch.zeros(rows, _HIDDEN // 2, dtype=torch.uint8),
        torch.zeros(rows, _HIDDEN // 16, dtype=torch.uint8),
        torch.zeros(rows, 6, dtype=torch.int32),
        torch.zeros(rows, 6),
        *[torch.zeros(1)] * 7,
        torch.zeros(rows, 1, _HIDDEN, dtype=torch.bfloat16),
    ]


def _sm107_occupancy(monkeypatch) -> None:
    """The measured SM107 cluster occupancy, which kernel construction otherwise queries the GPU for."""
    monkeypatch.setattr(
        op_module,
        "_max_active_clusters",
        lambda cluster_size, sm_version=None: {1: 212, 2: 106, 4: 46, 8: 22}[cluster_size],
    )


@pytest.mark.parametrize("capacity", [32, 192, 4096, 8448])
def test_runner_tunes_one_profile_per_bucket_collectively(monkeypatch, capacity) -> None:
    """Every launch stages ``capacity`` rows, so that is the one profile tuned
    and the shape every lookup maps to (power of two or not); the default
    tactic is the fallback candidate collective tuning breaks ties toward."""
    monkeypatch.setattr(op_module, "get_sm_version", lambda: 107)
    runner = op_module.Sm100MegaMoENvfp4Runner(
        world_size=4,
        local_rank=0,
        num_topk=6,
        num_experts_per_rank=96,
        hidden_size=_HIDDEN,
        intermediate_size_per_partition=3072,
        expand_intermediate_size_per_partition=6144,
        max_tokens_per_rank=capacity,
        output_dtype=torch.bfloat16,
        tactic_autotune=True,
    )
    config = runner.get_tuning_config()
    assert config.distributed_tuning_strategy == DistributedTuningStrategy.COLLECTIVE
    inputs = _runner_inputs(capacity)
    tuner = AutoTuner.__new__(AutoTuner)
    tuner.skip_dynamic_tuning_buckets = False
    tuner._debug_logger = lambda *args: None
    profiles = tuner._optimization_profiles(config, inputs)
    assert [p.get_opt_shapes()[0][0] for p in profiles] == [capacity]
    key = AutoTuner._find_nearest_profile(
        tuple(t.shape for t in inputs),
        config.dynamic_tensor_specs,
        config.constraint_specs,
        config.tune_max_num_tokens,
    )
    assert key[0][0] == capacity

    # get_valid_tactics may build each candidate's kernel and fallback_tactic resolves the default's,
    # which queries the GPU's cluster occupancy.
    _sm107_occupancy(monkeypatch)
    monkeypatch.setattr(
        op_module.Sm100MegaMoENvfp4Runner, "_build_kernel_once", lambda self, tactic: None
    )
    candidates = runner.get_valid_tactics(inputs, OptimizationProfile())
    fallback = runner.fallback_tactic(inputs, OptimizationProfile())
    assert fallback in candidates
    assert op_module._comparable_megamoe_tactic(fallback) == op_module._comparable_megamoe_tactic(
        runner._default_tactic(capacity)
    )


def test_tuning_group_must_span_the_launch_ep_ranks(monkeypatch) -> None:
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: group.size)
    ep_group = SimpleNamespace(size=4)
    try:
        op_module.set_active_megamoe_tuning_group(None)
        with pytest.raises(RuntimeError, match="EP process group"):
            op_module._megamoe_tuning_group(4)
        op_module.set_active_megamoe_tuning_group(SimpleNamespace(size=2))
        with pytest.raises(RuntimeError, match="has 2 ranks"):
            op_module._megamoe_tuning_group(4)
        op_module.set_active_megamoe_tuning_group(ep_group)
        assert op_module._megamoe_tuning_group(4) is ep_group
    finally:
        op_module.set_active_megamoe_tuning_group(None)


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> MegaMoECuteDsl:
    """A Rubin EP4 backend whose kernel launches and buffers are mocks."""
    monkeypatch.delenv("MEGAMOE_UGPU_LAYOUT", raising=False)
    monkeypatch.delenv("MEGAMOE_COMBINE_FORMAT", raising=False)
    monkeypatch.setenv("MEGAMOE_TACTIC_AUTOTUNE", "1")
    monkeypatch.setattr(backend_module, "get_sm_version", lambda: 107)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *args: Mock())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    model_config = ModelConfig(
        mapping=Mapping(world_size=1, rank=0, tp_size=1, moe_tp_size=1, moe_ep_size=1),
        moe_backend=MoeBackendType.MEGAMOE_CUTEDSL.value,
        skip_create_weights_in_init=True,
    )
    moe = MegaMoECuteDsl(
        routing_method=RenormalizeMoeRoutingMethod(top_k=2),
        num_experts=8,
        hidden_size=512,
        intermediate_size=512,
        dtype=torch.bfloat16,
        model_config=model_config,
        init_load_balancer=False,
    )
    assert moe.tactic_autotune
    moe.ep_size = 4
    moe._maxt_buckets = [32, 64, 1024, 4096]
    return moe


def _buffers(launch_max_T: int, **kwargs) -> SimpleNamespace:
    return SimpleNamespace(
        activation=torch.zeros(launch_max_T, 256, dtype=torch.uint8),
        activation_sf=torch.zeros(launch_max_T, 32, dtype=torch.uint8),
        topk_idx_local=torch.zeros(launch_max_T, 2, dtype=torch.int32),
        topk_weights=torch.zeros(launch_max_T, 2),
        combine_output=torch.zeros(launch_max_T, 1, 512, dtype=torch.bfloat16),
        shared_workspace=torch.zeros(16, dtype=torch.uint8),
        peer_offsets=[0] * 4,
    )


@pytest.mark.parametrize("tactic_autotune", [True, False])
def test_priming_tunes_every_other_bucket_in_ladder_order(
    backend, monkeypatch, tactic_autotune
) -> None:
    """With the opt-in, priming launches (and so tunes) every bucket but the live
    one in ladder order, freeing each bucket's profiling scratch before the next."""
    backend.tactic_autotune = tactic_autotune
    events = []
    monkeypatch.setattr(backend, "_acquire_buffers", lambda **kw: _buffers(kw["launch_max_T"]))
    monkeypatch.setattr(backend_module, "stage_megamoe_inputs", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        backend,
        "_launch_megamoe_kernel",
        lambda **kw: events.append(("launch", kw["launch_max_T"], kw["num_tokens"])),
    )
    monkeypatch.setattr(
        op_module, "release_megamoe_profiling_scratch", lambda: events.append(("release",))
    )
    primed = []
    backend._prime_ladder_buckets(
        live_T=1024,
        top_k=2,
        hidden=512,
        device=torch.device("cpu"),
        output_dtype=torch.bfloat16,
        weight_view=None,
        primed=primed,
    )
    assert primed == [32, 64, 4096]
    launches = [("launch", bucket, 0) for bucket in primed]
    if tactic_autotune:
        assert events == [e for launch in launches for e in (launch, ("release",))]
    else:
        assert events == launches


def test_tuning_mode_primes_the_ladder_with_the_opt_in(backend, monkeypatch) -> None:
    """The warmup forward in tuning mode primes (tunes) the other buckets, then
    launches the bucket its chunk selects."""
    monkeypatch.setattr(
        backend_module,
        "AutoTuner",
        SimpleNamespace(get=lambda: SimpleNamespace(is_tuning_mode=True)),
    )
    prime = Mock()
    launch = Mock(return_value="out")
    monkeypatch.setattr(backend, "_prime_ladder", prime)
    monkeypatch.setattr(backend, "_acquire_buffers", lambda **kw: _buffers(kw["launch_max_T"]))
    monkeypatch.setattr(backend_module, "stage_megamoe_inputs", lambda *args, **kwargs: None)
    monkeypatch.setattr(backend, "_launch_megamoe_kernel", launch)
    backend.set_adaptive_launch_tokens(40)
    out = backend._run_moe(
        x=torch.zeros(40, 256, dtype=torch.uint8),
        x_sf=torch.zeros(40, 32, dtype=torch.uint8),
        topk_idx=torch.zeros(40, 2, dtype=torch.int32),
        topk_weights=torch.zeros(40, 2),
        weight_view=None,
        num_tokens=40,
        top_k=2,
        hidden=512,
        device=torch.device("cpu"),
        output_dtype=torch.bfloat16,
    )
    assert out == "out"
    prime.assert_called_once()
    assert prime.call_args.kwargs["live_T"] == 64
    assert launch.call_args.kwargs["launch_max_T"] == 64


class _FakeProvider:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs
        self.max_tokens_per_rank = kwargs["max_tokens_per_rank"]
        self.regions = SimpleNamespace(bucket=self.max_tokens_per_rank)

    def get_regions(self):
        return self.regions


def test_launch_hands_the_op_its_bucket_scratch_and_the_ep_group(backend, monkeypatch) -> None:
    """Each bucket profiles on scratch sized for it, and the opt-in tunes across
    the backend's EP process group."""
    monkeypatch.setattr(
        backend_module,
        "AutoTuner",
        SimpleNamespace(get=lambda: SimpleNamespace(is_tuning_mode=True)),
    )
    providers = []  # The op module's scratch cache is the strong owner in production.

    def get_scratch(**kwargs):
        providers.append(_FakeProvider(**kwargs))
        return providers[-1]

    seen = []

    def run_op(**kwargs):
        factory = op_module._ACTIVE_MEGAMOE_PROFILING_SCRATCH_FACTORY
        scratch = op_module._ACTIVE_MEGAMOE_PROFILING_SCRATCH
        seen.append(
            (
                kwargs["max_tokens_per_rank"],
                None if scratch is None else scratch.bucket,
                None if factory is None else factory().bucket,
                op_module._ACTIVE_MEGAMOE_TUNING_GROUP,
            )
        )

    monkeypatch.setattr(op_module, "get_megamoe_profiling_scratch", get_scratch)
    monkeypatch.setattr(op_module, "run_cute_dsl_megamoe_nvfp4_blackwell", run_op)
    backend._ep_pg = ep_group = object()
    backend._profiling_scratch_kwargs = {
        bucket: dict(max_tokens_per_rank=bucket) for bucket in backend._maxt_buckets
    }
    weight_view = SimpleNamespace(
        fc1_weight=torch.zeros(4, dtype=torch.uint8),
        fc1_weight_sf=torch.zeros(4, dtype=torch.uint8),
        fc2_weight=torch.zeros(4, dtype=torch.uint8),
        fc2_weight_sf=torch.zeros(4, dtype=torch.uint8),
        fc31_alpha=torch.ones(1),
        fc2_alpha=torch.ones(1),
        fc1_norm_const=torch.ones(1),
    )

    def launch(bucket: int) -> None:
        bufs = _buffers(bucket)
        backend._launch_megamoe_kernel(
            activation=bufs.activation,
            activation_sf=bufs.activation_sf,
            topk_idx=bufs.topk_idx_local,
            topk_weights=bufs.topk_weights,
            weight_view=weight_view,
            combine_output=bufs.combine_output,
            shared_workspace=bufs.shared_workspace,
            world_size=4,
            local_rank=0,
            top_k=2,
            hidden=512,
            peer_offsets=bufs.peer_offsets,
            num_tokens=0,
            output_dtype=torch.bfloat16,
            launch_max_T=bucket,
        )

    # First launch of a bucket: a deferred factory for that bucket's scratch. Once
    # it is live the same bucket gets it directly; another bucket never does.
    for bucket in (64, 64, 1024):
        launch(bucket)
    assert seen == [
        (64, None, 64, ep_group),
        (64, 64, None, ep_group),
        (1024, None, 1024, ep_group),
    ]
    assert op_module._ACTIVE_MEGAMOE_TUNING_GROUP is None
    assert op_module._ACTIVE_MEGAMOE_PROFILING_SCRATCH_FACTORY is None

    # Without the opt-in the op gets neither.
    backend.tactic_autotune = False
    backend._profiling_scratch_kwargs = {}
    backend._profiling_scratch_provider_ref = None
    launch(32)
    assert seen[-1] == (32, None, None, None)


def _tuning_mode(monkeypatch, tuning: bool) -> None:
    monkeypatch.setattr(
        backend_module,
        "AutoTuner",
        SimpleNamespace(get=lambda: SimpleNamespace(is_tuning_mode=tuning)),
    )


@pytest.mark.parametrize("tuning", [False, True])
def test_opt_in_fuses_route_quant_outside_tuning_mode(backend, monkeypatch, tuning) -> None:
    """The fused route+quant launch writes the bucket's staged buffers: the opt-in
    skips it only while tuning."""
    _tuning_mode(monkeypatch, tuning)
    bufs = _buffers(64)
    monkeypatch.setattr(backend, "_acquire_buffers", lambda **kw: bufs)
    backend.set_adaptive_launch_tokens(40)
    staged = backend._route_quant_bucket(40, 2, torch.device("cpu"), torch.bfloat16)
    if tuning:
        assert staged is None
    else:
        assert staged == (
            bufs.topk_idx_local,
            bufs.topk_weights,
            bufs.activation,
            bufs.activation_sf,
        )


@pytest.mark.parametrize("tuning", [False, True])
def test_opt_in_takes_direct_inputs_outside_tuning_mode(backend, monkeypatch, tuning) -> None:
    """On a direct-input bucket the opt-in passes the caller's tensors (with a
    staging fallback for the op) unless tuning, which stages every input."""
    _tuning_mode(monkeypatch, tuning)
    bufs = _buffers(64)
    staged = []
    launch = Mock(return_value="out")
    monkeypatch.setattr(backend, "_prime_ladder", Mock())
    monkeypatch.setattr(backend, "_acquire_buffers", lambda **kw: bufs)
    monkeypatch.setattr(
        backend_module, "stage_megamoe_inputs", lambda *args, **kwargs: staged.append(kwargs)
    )
    monkeypatch.setattr(backend, "_launch_megamoe_kernel", launch)
    backend._genphase_direct_input_buckets = {64}
    backend.set_adaptive_launch_tokens(64)
    x = torch.zeros(64, 256, dtype=torch.uint8)
    x_sf = torch.zeros(64, 32, dtype=torch.uint8)
    topk_idx = torch.zeros(64, 2, dtype=torch.int32)
    topk_weights = torch.zeros(64, 2)
    backend._run_moe(
        x=x,
        x_sf=x_sf,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        weight_view=None,
        num_tokens=64,
        top_k=2,
        hidden=512,
        device=torch.device("cpu"),
        output_dtype=torch.bfloat16,
    )
    call = launch.call_args.kwargs
    if tuning:
        assert call["activation"] is bufs.activation and call["staged_inputs"]
        assert call["restage"] is None
        assert staged and staged[-1]["x"] is x
    else:
        assert call["activation"] is x and call["topk_idx"] is topk_idx
        assert not call["staged_inputs"] and not staged
        # The fallback stages exactly what the direct path skipped.
        restaged = call["restage"]()
        assert restaged[0] is bufs.activation and restaged[1] is bufs.activation_sf
        assert restaged[2] is bufs.topk_idx_local
        assert restaged[3].data_ptr() == bufs.topk_weights.data_ptr()
        assert restaged[3].shape == (64, 2)
        assert staged[-1]["x"] is x and staged[-1]["act_buffer"] is bufs.activation


def _active_ep_group(monkeypatch, size: int) -> SimpleNamespace:
    """An active EP group of ``size`` ranks whose collectives fail the test."""
    group = SimpleNamespace(size=size)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: group.size)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda group=None: 0)

    def no_collective(*args, **kwargs):
        raise AssertionError("a serving lookup communicated")

    monkeypatch.setattr(torch.distributed, "all_gather_object", no_collective)
    monkeypatch.setattr(torch.distributed, "barrier", no_collective)
    op_module.set_active_megamoe_tuning_group(group)
    return group


def _serve_op(runner, capacity: int, num_tokens: int) -> None:
    """One serving launch of the op on EP4 rank 0 with ``num_tokens`` live tokens."""
    op_module.run_cute_dsl_megamoe_nvfp4_blackwell(
        *_runner_inputs(capacity)[:11],
        torch.zeros(capacity, 1, _HIDDEN, dtype=torch.bfloat16),
        torch.zeros(16, dtype=torch.uint8),
        world_size=4,
        local_rank=0,
        num_topk=6,
        num_experts_per_rank=96,
        hidden_size=_HIDDEN,
        intermediate_size_per_partition=3072,
        expand_intermediate_size_per_partition=6144,
        max_tokens_per_rank=capacity,
        peer_offsets=[0] * 4,
        tactic_autotune=True,
        num_tokens=num_tokens,
    )


def test_serving_replays_the_agreed_tactic_whatever_the_live_tokens(monkeypatch) -> None:
    """Ranks serve with different live tokens (none, some or all of the bucket), but
    the op hands the tuner the bucket's staged rows: every rank replays the entry the
    EP ranks agreed on while tuning, without communicating; a bucket with no agreed
    entry takes the default on every rank."""
    capacity = 64
    monkeypatch.setattr(op_module, "get_sm_version", lambda: 107)
    runner = op_module.Sm100MegaMoENvfp4Runner(
        world_size=4,
        local_rank=0,
        num_topk=6,
        num_experts_per_rank=96,
        hidden_size=_HIDDEN,
        intermediate_size_per_partition=3072,
        expand_intermediate_size_per_partition=6144,
        max_tokens_per_rank=capacity,
        output_dtype=torch.bfloat16,
        tactic_autotune=True,
    )
    launched = []
    monkeypatch.setattr(
        runner, "forward", lambda inputs, *, tactic=-1, **kwargs: launched.append(tactic)
    )
    monkeypatch.setattr(op_module, "_megamoe_get_runner", lambda *args, **kwargs: runner)
    tuner = AutoTuner()
    monkeypatch.setattr(op_module, "AutoTuner", SimpleNamespace(get=lambda: tuner))
    _active_ep_group(monkeypatch, 4)
    keys = []
    lookup = tuner._agreed_entry_key
    monkeypatch.setattr(
        tuner, "_agreed_entry_key", lambda *args: keys.append(lookup(*args)) or keys[-1]
    )
    try:
        _serve_op(runner, capacity, capacity)
        assert launched == [-1]  # nothing agreed yet: the default
        tuner._collective_agreed[keys[0]] = (0, ("agreed",))
        for num_tokens in (0, 1, 37, capacity):
            _serve_op(runner, capacity, num_tokens)
    finally:
        op_module.set_active_megamoe_tuning_group(None)
    assert launched[1:] == [("agreed",)] * 4
    assert len(set(keys)) == 1


def test_runner_prepare_tactic_builds_without_launching(monkeypatch) -> None:
    """prepare_tactic resolves the kernel a tactic runs (the default here) and raises
    on one the op refuses, so collective tuning can agree on refusals before a launch."""
    monkeypatch.setattr(op_module, "get_sm_version", lambda: 107)
    monkeypatch.setattr(
        op_module,
        "_max_active_clusters",
        lambda cluster_size, sm_version=None: {1: 212, 2: 106, 4: 46, 8: 22}[cluster_size],
    )
    runner = op_module.Sm100MegaMoENvfp4Runner(
        world_size=4,
        local_rank=0,
        num_topk=6,
        num_experts_per_rank=96,
        hidden_size=_HIDDEN,
        intermediate_size_per_partition=3072,
        expand_intermediate_size_per_partition=6144,
        max_tokens_per_rank=64,
        output_dtype=torch.bfloat16,
        tactic_autotune=True,
    )
    monkeypatch.setattr(
        runner, "forward", Mock(side_effect=AssertionError("prepare_tactic launched"))
    )
    runner.prepare_tactic(_runner_inputs(64), -1)
    default = runner._default_tactic(64)
    runner.prepare_tactic(_runner_inputs(64), list(default))
    refused = list(op_module._unpack_tactic(default))
    refused[0] = [7, 7, 7]  # no such MMA tile
    with pytest.raises((ValueError, NotImplementedError)):
        runner.prepare_tactic(_runner_inputs(64), tuple(refused))


def test_fallback_tactic_is_the_default_as_the_launch_resolves_it(monkeypatch) -> None:
    """tactic=-1 runs the default as the launch resolves it, so a default the op refuses makes its
    first accepted fallback the candidate collective tuning breaks ties toward; with nothing
    accepted there is no such candidate."""
    monkeypatch.setattr(op_module, "get_sm_version", lambda: 107)
    _sm107_occupancy(monkeypatch)
    runner = op_module.Sm100MegaMoENvfp4Runner(
        world_size=4,
        local_rank=0,
        num_topk=6,
        num_experts_per_rank=96,
        hidden_size=_HIDDEN,
        intermediate_size_per_partition=3072,
        expand_intermediate_size_per_partition=6144,
        max_tokens_per_rank=128,
        output_dtype=torch.bfloat16,
        tactic_autotune=True,
    )
    default = op_module._unpack_tactic(runner._default_tactic(128))
    resolved = op_module._megamoe_default_tactic_candidates(default)[1]
    other = (*resolved[:7], 2, *resolved[8:])
    monkeypatch.setattr(runner, "get_valid_tactics", lambda *args, **kwargs: [other, resolved])
    inputs = _runner_inputs(128)

    def refuse(*refused):
        def build(self, tactic):
            if op_module._unpack_tactic(tactic) in refused:
                raise ValueError("refused here")

        monkeypatch.setattr(op_module.Sm100MegaMoENvfp4Runner, "_build_kernel_once", build)
        monkeypatch.setattr(op_module.Sm100MegaMoENvfp4Runner, "_kernel_descriptor_cache", {})

    refuse()
    assert runner.fallback_tactic(inputs, OptimizationProfile()) == -1  # the default is not offered
    refuse(default)
    assert runner.fallback_tactic(inputs, OptimizationProfile()) == resolved
    refuse(*op_module._megamoe_default_tactic_candidates(default))
    assert runner.fallback_tactic(inputs, OptimizationProfile()) == -1


@pytest.mark.parametrize("reads_locally", [True, False])
def test_op_stages_direct_inputs_for_a_kernel_reading_peers(monkeypatch, reads_locally) -> None:
    """Direct inputs reach the kernel only when the tuned tactic's kernel reads them locally."""
    staged_inputs = [torch.ones(1), torch.ones(1), torch.ones(1), torch.ones(1)]
    seen = {}

    class _Runner:
        helper_expert_count = 0

        def get_tuning_config(self):
            return TuningConfig()

        def reads_inputs_locally(self, tactic):
            seen["tactic"] = tactic
            return reads_locally

        def __call__(self, inputs, **kwargs):
            seen["inputs"] = list(inputs)

    runner = _Runner()
    tuner = SimpleNamespace(
        is_tuning_mode=False, choose_one=lambda *args, **kwargs: (runner, ("tuned",))
    )
    monkeypatch.setattr(op_module, "get_sm_version", lambda: 107)
    monkeypatch.setattr(op_module, "_megamoe_get_runner", lambda *args, **kwargs: runner)
    monkeypatch.setattr(op_module, "AutoTuner", SimpleNamespace(get=lambda: tuner))
    direct = [torch.zeros(1) for _ in range(4)]
    _active_ep_group(monkeypatch, 4)
    op_module.set_active_megamoe_direct_input_restage(lambda: tuple(staged_inputs))
    try:
        op_module.run_cute_dsl_megamoe_nvfp4_blackwell(
            *direct,
            *[torch.zeros(1)] * 7,
            torch.zeros(4, 1, 8),
            torch.zeros(1),
            world_size=4,
            local_rank=0,
            num_topk=2,
            num_experts_per_rank=2,
            hidden_size=8,
            intermediate_size_per_partition=8,
            expand_intermediate_size_per_partition=16,
            max_tokens_per_rank=4,
            peer_offsets=[0] * 4,
            tactic_autotune=True,
        )
    finally:
        op_module.set_active_megamoe_direct_input_restage(None)
        op_module.set_active_megamoe_tuning_group(None)
    assert seen["tactic"] == ("tuned",)
    expected = direct if reads_locally else staged_inputs
    assert all(a is b for a, b in zip(seen["inputs"][:4], expected))


@pytest.mark.parametrize("tuning", [False, True])
def test_minimax_fused_route_quant_declines_only_while_tuning(monkeypatch, tuning) -> None:
    """With the opt-in, MiniMax's fused route+quant seat declines while tuning only."""
    from tensorrt_llm._torch.modeling_v2.models.minimax_m3.ops.moe.megamoe_local import local_impl
    from tensorrt_llm._torch.moe.fused_moe.routing import (
        DeepSeekV3MoeRoutingMethod,
        Deepseekv3RoutingImpl,
    )

    def tensor(shape, dtype=torch.float32):
        t = Mock(spec=torch.Tensor)
        t.dtype, t.ndim, t.shape, t.is_cuda = dtype, len(shape), shape, True
        t.device = torch.device("cuda", 0)
        t.numel.return_value = int(torch.Size(shape).numel())
        t.is_contiguous.return_value = True
        return t

    impl = object.__new__(Deepseekv3RoutingImpl)
    impl.__dict__.update(is_fused=True, n_group=1, topk_group=1, top_k=8)
    routing = object.__new__(DeepSeekV3MoeRoutingMethod)
    torch.nn.Module.__init__(routing)
    for name, value in (
        ("routing_impl", impl),
        ("output_dtype", torch.float32),
        ("e_score_correction_bias", tensor((256,))),
    ):
        if isinstance(getattr(DeepSeekV3MoeRoutingMethod, name, None), property):
            monkeypatch.setattr(DeepSeekV3MoeRoutingMethod, name, property(lambda self, v=value: v))
        else:
            object.__setattr__(routing, name, value)
    moe = SimpleNamespace(
        _route_quant_enabled=True,
        routing_method=routing,
        hidden_size=1024,
        fc31_input_scale=tensor((1,)),
        tactic_autotune=True,
        # Read by the checks after the autotuning one.
        _topk_idx_dtype=torch.int32,
        _weights_created=True,
        quant_method=object(),
        _active_launch_max_tokens=None,
        _maxt_buckets=[64],
        _select_launch_max_tokens=lambda chunk: 64,
        ep_size=1,
        _symm_providers={},
    )
    monkeypatch.setattr(
        local_impl,
        "AutoTuner",
        SimpleNamespace(get=lambda: SimpleNamespace(is_tuning_mode=tuning)),
    )
    reason = local_impl.TrtllmCutedslMegaMoeLocalNvfp4Impl._route_quant_decline_reason(
        moe, tensor((16, 1024), torch.bfloat16), tensor((16, 256))
    )
    assert (reason == "autotuning") == tuning, reason
