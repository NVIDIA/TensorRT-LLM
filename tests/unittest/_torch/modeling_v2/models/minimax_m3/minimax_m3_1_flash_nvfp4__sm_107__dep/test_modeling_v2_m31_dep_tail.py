# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""DEP_TAIL on stub MoE blocks and layers, CPU only.

The install puts form (a) on every attention-DP MoE block with a shared expert on the MegaMoE local backend, and form
(b) on the layers whose boundary norm is FlashInfer's fused add + RMSNorm, unless the config could change a bucket's
kernel class from step to step (torch.compile; chunked steps under CUDA graphs). Each forward requests the deferred
reduce and takes whichever output the launch returns: a deferred one runs the tail kernel, a reduced tensor the stock
add and norm. The tail kernels are replaced by recorders; their bitwise checks are the GPU tests.
"""

import os
import types

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_1_flash_nvfp4__sm_107__dep import (  # noqa: E501
    m31_switches,
    tail,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.moe.megamoe_local import (
    local_plan as megamoe_local_plan,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.ops.moe.megamoe_local.local_impl import (
    MegaMoEDeferredTopkReduce,
)
from tensorrt_llm._torch.models import modeling_minimaxm3
from tensorrt_llm._torch.models.modeling_minimaxm3 import MiniMaxM3DecoderLayer, MiniMaxM3MoE
from tensorrt_llm._torch.modules import linear as _linear
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.utils import EventType

pytestmark = pytest.mark.cpu_only

# The linear layer's MXFP8 dispatch scopes the pinned shared expert runs in; without them the pin is a no-op.
_PIN_SCOPES = hasattr(_linear, "mxfp8_cute_dsl_dispatch_active") and hasattr(
    _linear, "mxfp8_native_dispatch_active"
)
_NEEDS_PIN_SCOPES = pytest.mark.skipif(
    not _PIN_SCOPES, reason="the linear layer has no MXFP8 dispatch scopes on this runtime"
)


def mxfp8_cute_dsl_dispatch_active() -> bool:
    return getattr(_linear, "mxfp8_cute_dsl_dispatch_active", lambda: False)()


def mxfp8_native_dispatch_active() -> bool:
    return getattr(_linear, "mxfp8_native_dispatch_active", lambda: False)()


ROWS, HIDDEN = 3, 64
ALL_RANK = [3, 7]


class _Backend:
    def __init__(self, reason=None, static_reason=None, bucket=192):
        self.reason, self.static_reason, self.bucket = reason, static_reason, bucket
        self.requests, self.cancels, self.bucket_asks = [], 0, []
        # Like the real backend, the knobs are read once, when it is built.
        self.environ = megamoe_local_plan.lever_environ(os.environ)

    def megamoe_launch_bucket(self, max_tokens_per_rank):
        self.bucket_asks.append(max_tokens_per_rank)
        return self.bucket

    def shared_expert_serial(self, max_tokens_per_rank):
        return megamoe_local_plan.shared_expert_serial(
            self.megamoe_launch_bucket(max_tokens_per_rank), 8, self.environ
        )

    def shared_corun_pin(self, max_tokens_per_rank):
        return megamoe_local_plan.shared_corun_pin(
            self.megamoe_launch_bucket(max_tokens_per_rank), self.environ
        )

    def shared_corun_pin_kernel(self):
        return megamoe_local_plan.shared_corun_pin_kernel(self.environ)

    def tail_static_decline_reason(self):
        return self.static_reason

    def request_deferred_topk_reduce(self, all_rank_num_tokens, num_tokens):
        self.requests.append((all_rank_num_tokens, num_tokens))
        return self.reason

    def cancel_deferred_topk_reduce(self):
        self.cancels += 1


class _PrestagingBackend(_Backend):
    """A backend that routes beside a serial shared expert, its prestaged seat recorded instead of launched."""

    def __init__(self, beside=True, **kwargs):
        super().__init__(**kwargs)
        self.beside, self.prestaged = beside, []

    def route_beside_shared(self, max_tokens_per_rank):
        return self.beside and self.shared_expert_serial(max_tokens_per_rank)

    def prestage_route_quant(self, x, router_logits, chunk_max_tokens):
        self.prestaged.append((x, router_logits, chunk_max_tokens))


class _Experts:
    def __init__(self, backend, output):
        self.backend, self.output, self.calls = backend, output, []

    def __call__(self, hidden_states, router_logits, **kwargs):
        self.calls.append(kwargs)
        return self.output


class _Deferred(MegaMoEDeferredTopkReduce):
    """Records which tail form ran; returns what the kernels would write."""

    def add_into(self, shared):
        object.__setattr__(self, "ran", ("a", shared))
        return shared.fill_(1.0)

    def add_rmsnorm_into(self, shared, residual, weight, eps):
        object.__setattr__(self, "ran", ("b", shared, residual, weight, eps))
        return shared.fill_(2.0), residual.fill_(3.0)


def _deferred():
    return _Deferred(
        plane=torch.zeros(ROWS, 4, HIDDEN, dtype=torch.bfloat16), scores=torch.zeros(ROWS, 4)
    )


def _moe(
    backend=None,
    routed=None,
    shared_experts=True,
    allreduce=None,
    routed_is_global=True,
    serial=False,
    pinned=False,
    beside=False,
):
    moe = object.__new__(MiniMaxM3MoE)
    torch.nn.Module.__init__(moe)
    backend = backend or _Backend()
    routed = routed if routed is not None else torch.full((ROWS, HIDDEN), 0.5, dtype=torch.bfloat16)
    moe.gate = lambda h: torch.zeros(h.shape[0], 8)
    moe.experts = _Experts(backend, routed)
    moe.shared_experts = (
        (lambda h, lora_params=None: torch.full((ROWS, HIDDEN), 0.25, dtype=torch.bfloat16))
        if shared_experts
        else None
    )
    moe.allreduce = allreduce
    moe.routed_output_is_global = routed_is_global
    moe.event_dict = {EventType.Main: None, EventType.MoeShared: None}
    moe.aux_stream = None
    moe.shared_expert_serial = serial
    moe.shared_corun_pinnable = pinned
    moe.route_beside_shared = beside
    moe.mapping = types.SimpleNamespace(moe_ep_size=8)
    return moe


def _metadata(all_rank=ALL_RANK):
    return types.SimpleNamespace(all_rank_num_tokens=all_rank)


def _hidden():
    return torch.zeros(ROWS, HIDDEN, dtype=torch.bfloat16)


def _layer(moe, norm=True):
    layer = object.__new__(MiniMaxM3DecoderLayer)
    torch.nn.Module.__init__(layer)
    layer.layer_idx = 5
    layer.block_sparse_moe = moe
    layer.post_feed_forward_fusion = False
    layer.pre_feed_forward_fusion = False
    layer.next_layer_layernorm = (
        RMSNorm(hidden_size=HIDDEN, eps=1e-6, dtype=torch.bfloat16) if norm else None
    )
    layer.norms = []
    layer._apply_pre_feed_forward_norm = lambda h, r: (h, r)

    def boundary_norm(h, r):
        layer.norms.append(h.clone())
        return h, r

    layer._apply_next_layer_layernorm = boundary_norm
    return layer


@pytest.fixture
def eligible_norm(monkeypatch):
    monkeypatch.setattr(tail, "DEP_TAIL", True)
    monkeypatch.setattr(tail, "norm_decline_reason", lambda layer: None)


# --- the MoE block ---------------------------------------------------------------------------------------------------


def test_a_deferred_launch_runs_form_a_into_the_shared_output():
    deferred = _deferred()
    backend = _Backend()
    moe = _moe(backend, routed=deferred)
    out = tail._moe_forward(moe, _hidden(), _metadata())
    assert deferred.ran[0] == "a" and deferred.ran[1] is out
    assert torch.equal(out, torch.ones(ROWS, HIDDEN, dtype=torch.bfloat16))
    assert backend.requests == [(ALL_RANK, ROWS)] and backend.cancels == 0


def test_the_experts_get_the_stock_call():
    moe = _moe()
    tail._moe_forward(moe, _hidden(), _metadata(), lora_params={"x": 1})
    assert moe.experts.calls == [
        dict(all_rank_num_tokens=ALL_RANK, use_dp_padding=False, lora_params={"x": 1})
    ]


def test_a_reduced_launch_runs_the_stock_add():
    out = tail._moe_forward(_moe(), _hidden(), _metadata())
    assert torch.equal(out, torch.full((ROWS, HIDDEN), 0.75, dtype=torch.bfloat16))


def test_a_declined_request_disarms_the_backend():
    backend = _Backend(reason="weights are not created")
    tail._moe_forward(_moe(backend), _hidden(), _metadata())
    assert backend.cancels == 1


def test_no_request_for_a_non_bf16_activation():
    backend = _Backend()
    tail._moe_forward(_moe(backend), _hidden().float(), _metadata())
    assert backend.requests == [] and backend.cancels == 1


def test_a_non_contiguous_shared_output_is_copied_before_form_a():
    deferred = _deferred()
    moe = _moe(routed=deferred)
    moe.shared_experts = lambda h, lora_params=None: torch.zeros(
        HIDDEN, ROWS, dtype=torch.bfloat16
    ).t()
    out = tail._moe_forward(moe, _hidden(), _metadata())
    assert out.is_contiguous()


@pytest.mark.parametrize(
    "moe, reason",
    [
        pytest.param(lambda: torch.nn.Linear(1, 1), "MoE block Linear", id="other-block"),
        pytest.param(lambda: _moe(shared_experts=False), "no shared expert", id="no-shared"),
        pytest.param(
            lambda: _moe(allreduce=object()),
            "not attention DP on a fused-communication MoE",
            id="allreduce",
        ),
        pytest.param(
            lambda: _moe(routed_is_global=False),
            "not attention DP on a fused-communication MoE",
            id="not-fused-comm",
        ),
        pytest.param(
            lambda: _moe(backend=types.SimpleNamespace()),
            "backend SimpleNamespace reduces its own top-k",
            id="stock-backend",
        ),
        pytest.param(
            lambda: _moe(backend=_Backend(static_reason="the TAIL lever is off")),
            "the TAIL lever is off",
            id="lever-off",
        ),
    ],
)
def test_moe_declines(moe, reason):
    assert tail.moe_decline_reason(moe()) == reason


def test_an_eligible_moe_is_admitted():
    assert tail.moe_decline_reason(_moe()) is None


# --- shared-expert ordering (the backend's shared_expert_serial on the launch bucket) ---


class _OrderedExperts(_Experts):
    def __init__(self, backend, output, order):
        super().__init__(backend, output)
        self.order = order

    def __call__(self, hidden_states, router_logits, **kwargs):
        self.order.append("routed")
        return super().__call__(hidden_states, router_logits, **kwargs)


def _ordered_moe(
    monkeypatch, serial, bucket, routed=None, pinned=False, native=None, backend=None, beside=False
):
    """A MoE block whose branches record their order, and whether the aux-stream helper ran; ``native`` collects
    the pinned dispatch each shared-expert call ran inside: ``cute_dsl``, ``native`` or None."""
    order = []
    backend = backend or _Backend(bucket=bucket)
    moe = _moe(backend, routed=routed, serial=serial, pinned=pinned, beside=beside)
    moe.experts = _OrderedExperts(backend, moe.experts.output, order)

    def shared_call(h, lora_params=None):
        order.append("shared")
        if native is not None:
            native.append(
                "cute_dsl"
                if mxfp8_cute_dsl_dispatch_active()
                else "native"
                if mxfp8_native_dispatch_active()
                else None
            )
        return torch.full((ROWS, HIDDEN), 0.25, dtype=torch.bfloat16)

    moe.shared_experts = shared_call
    parallel = []
    real = tail.maybe_execute_in_parallel

    def recording_parallel(*args, **kwargs):
        parallel.append(True)
        return real(*args, **kwargs)

    monkeypatch.setattr(tail, "maybe_execute_in_parallel", recording_parallel)
    # The route beside the shared expert forks in the model module's helper.
    monkeypatch.setattr(modeling_minimaxm3, "maybe_execute_in_parallel", recording_parallel)
    return moe, backend, order, parallel


@pytest.mark.parametrize("bucket", [32, 64, 128, 1024])
def test_a_genphase_bucket_runs_the_shared_expert_after_the_routed_launch(monkeypatch, bucket):
    monkeypatch.delenv("TRTLLM_M31_DISABLE_MEGAMOE_SHARED_SERIAL", raising=False)
    monkeypatch.delenv("TRTLLM_M31_MEGAMOE_SHARED_SERIAL_MAX_ROWS", raising=False)
    monkeypatch.delenv("TRTLLM_M31_MEGAMOE_SHARED_SERIAL_MIN_EP_SIZE", raising=False)
    moe, backend, order, parallel = _ordered_moe(monkeypatch, serial=True, bucket=bucket)
    tail._moe_forward(moe, _hidden(), _metadata())
    assert order == ["routed", "shared"] and parallel == []
    assert backend.bucket_asks == [max(ALL_RANK)]


@pytest.mark.parametrize("bucket", [64, 128])
def test_a_bucket_above_the_knob_keeps_the_shared_expert_on_the_aux_path(monkeypatch, bucket):
    monkeypatch.delenv("TRTLLM_M31_DISABLE_MEGAMOE_SHARED_SERIAL", raising=False)
    monkeypatch.setenv("TRTLLM_M31_MEGAMOE_SHARED_SERIAL_MAX_ROWS", "32")
    moe, _, _, parallel = _ordered_moe(monkeypatch, serial=True, bucket=bucket)
    tail._moe_forward(moe, _hidden(), _metadata())
    assert parallel == [True]


def test_the_ordering_keeps_the_knobs_the_backend_was_built_with(monkeypatch):
    monkeypatch.delenv("TRTLLM_M31_DISABLE_MEGAMOE_SHARED_SERIAL", raising=False)
    monkeypatch.delenv("TRTLLM_M31_MEGAMOE_SHARED_SERIAL_MAX_ROWS", raising=False)
    monkeypatch.delenv("TRTLLM_M31_MEGAMOE_SHARED_SERIAL_MIN_EP_SIZE", raising=False)
    moe, _, order, parallel = _ordered_moe(monkeypatch, serial=True, bucket=32)
    monkeypatch.setenv("TRTLLM_M31_MEGAMOE_SHARED_SERIAL_MAX_ROWS", "1")
    tail._moe_forward(moe, _hidden(), _metadata())
    assert order == ["routed", "shared"] and parallel == []


def test_a_layer_without_the_serial_flag_keeps_the_aux_path(monkeypatch):
    moe, backend, _, parallel = _ordered_moe(monkeypatch, serial=False, bucket=32)
    tail._moe_forward(moe, _hidden(), _metadata())
    assert parallel == [True] and backend.bucket_asks == []


def test_the_shared_serial_kill_switch_keeps_the_aux_path(monkeypatch):
    monkeypatch.setenv("TRTLLM_M31_DISABLE_MEGAMOE_SHARED_SERIAL", "1")
    moe, _, _, parallel = _ordered_moe(monkeypatch, serial=True, bucket=32)
    tail._moe_forward(moe, _hidden(), _metadata())
    assert parallel == [True]


def test_without_metadata_the_bucket_is_asked_for_the_full_launch(monkeypatch):
    monkeypatch.delenv("TRTLLM_M31_DISABLE_MEGAMOE_SHARED_SERIAL", raising=False)
    moe, backend, _, _ = _ordered_moe(monkeypatch, serial=True, bucket=32)
    tail._moe_forward(moe, _hidden(), _metadata(all_rank=None))
    assert backend.bucket_asks == [None]


_SCOPE_KNOBS = (
    "TRTLLM_M31_DISABLE_MEGAMOE_SHARED_SERIAL",
    "TRTLLM_M31_MEGAMOE_SHARED_SERIAL_MAX_ROWS",
    "TRTLLM_M31_MEGAMOE_SHARED_SERIAL_MIN_EP_SIZE",
    "TRTLLM_M31_MEGAMOE_SHARED_CORUN_BAND",
    "TRTLLM_M31_MEGAMOE_SHARED_CORUN_PIN_ABOVE_ROWS",
    "TRTLLM_M31_DISABLE_MEGAMOE_SHARED_CORUN_PIN",
    "TRTLLM_M31_MEGAMOE_SHARED_CORUN_PIN_KERNEL",
)


@pytest.mark.parametrize(
    "bucket, environ, pinned, aux, native",
    [
        (128, (("TRTLLM_M31_MEGAMOE_SHARED_SERIAL_MAX_ROWS", "32"),), True, True, "cute_dsl"),
        (128, (("TRTLLM_M31_MEGAMOE_SHARED_SERIAL_MAX_ROWS", "32"),), False, True, None),
        (32, (), True, False, None),
        (128, (), True, False, None),
        (64, (("TRTLLM_M31_MEGAMOE_SHARED_SERIAL_MAX_ROWS", "32"),), True, True, "cute_dsl"),
        (
            64,
            (
                ("TRTLLM_M31_MEGAMOE_SHARED_SERIAL_MAX_ROWS", "32"),
                ("TRTLLM_M31_MEGAMOE_SHARED_CORUN_PIN_ABOVE_ROWS", "32"),
            ),
            True,
            True,
            "cute_dsl",
        ),
        (
            128,
            (
                ("TRTLLM_M31_MEGAMOE_SHARED_SERIAL_MAX_ROWS", "32"),
                ("TRTLLM_M31_DISABLE_MEGAMOE_SHARED_CORUN_PIN", "1"),
            ),
            True,
            True,
            None,
        ),
        (
            128,
            (
                ("TRTLLM_M31_MEGAMOE_SHARED_SERIAL_MAX_ROWS", "32"),
                ("TRTLLM_M31_MEGAMOE_SHARED_CORUN_PIN_KERNEL", "cutlass"),
            ),
            True,
            True,
            "native",
        ),
        (
            64,
            (
                ("TRTLLM_M31_MEGAMOE_SHARED_SERIAL_MAX_ROWS", "32"),
                ("TRTLLM_M31_MEGAMOE_SHARED_CORUN_PIN_KERNEL", "cutlass"),
            ),
            True,
            True,
            None,
        ),
        (64, (("TRTLLM_M31_MEGAMOE_SHARED_CORUN_BAND", "32:64"),), True, True, "cute_dsl"),
        (128, (("TRTLLM_M31_MEGAMOE_SHARED_CORUN_BAND", "32:64"),), True, False, None),
    ],
    ids=[
        "aux-pinned-cute-dsl",
        "unpinnable-aux-automatic",
        "serial-automatic",
        "serial-bucket-128-automatic",
        "serial-32-bucket-64-aux-cute-dsl",
        "pin-above-32-bucket-64-aux-cute-dsl",
        "pin-disabled-aux-automatic",
        "cutlass-aux-pinned-native",
        "cutlass-serial-32-bucket-64-aux-automatic",
        "band-32-64-bucket-64-aux-cute-dsl",
        "band-32-64-bucket-128-serial-automatic",
    ],
)
@_NEEDS_PIN_SCOPES
def test_only_a_pinned_shared_expert_on_the_aux_path_runs_the_pinned_mxfp8(
    monkeypatch, bucket, environ, pinned, aux, native
):
    for name in _SCOPE_KNOBS:
        monkeypatch.delenv(name, raising=False)
    for name, value in environ:
        monkeypatch.setenv(name, value)
    seen = []
    moe, _, order, parallel = _ordered_moe(
        monkeypatch, serial=True, bucket=bucket, pinned=pinned, native=seen
    )
    tail._moe_forward(moe, _hidden(), _metadata())
    assert parallel == ([True] if aux else []) and seen == [native]


@_NEEDS_PIN_SCOPES
def test_the_pin_keeps_the_knobs_the_backend_was_built_with(monkeypatch):
    for name in _SCOPE_KNOBS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("TRTLLM_M31_MEGAMOE_SHARED_SERIAL_MAX_ROWS", "32")
    seen = []
    moe, _, _, parallel = _ordered_moe(
        monkeypatch, serial=True, bucket=128, pinned=True, native=seen
    )
    monkeypatch.setenv("TRTLLM_M31_DISABLE_MEGAMOE_SHARED_CORUN_PIN", "1")
    tail._moe_forward(moe, _hidden(), _metadata())
    assert parallel == [True] and seen == ["cute_dsl"]


def test_a_deferred_launch_runs_form_a_after_a_serial_shared_expert(monkeypatch):
    monkeypatch.delenv("TRTLLM_M31_DISABLE_MEGAMOE_SHARED_SERIAL", raising=False)
    monkeypatch.delenv("TRTLLM_M31_MEGAMOE_SHARED_SERIAL_MAX_ROWS", raising=False)
    deferred = _deferred()
    moe, _, order, parallel = _ordered_moe(monkeypatch, serial=True, bucket=32, routed=deferred)
    out = tail._moe_forward(moe, _hidden(), _metadata())
    assert order == ["routed", "shared"] and parallel == []
    assert deferred.ran[0] == "a" and torch.equal(
        out, torch.ones(ROWS, HIDDEN, dtype=torch.bfloat16)
    )


def _routing_moe(monkeypatch, bucket, routed=None, beside=True):
    """An ordered MoE block on a prestaging backend whose gate records its call and returns ``logits``."""
    for name in _SCOPE_KNOBS:
        monkeypatch.delenv(name, raising=False)
    backend = _PrestagingBackend(beside=beside, bucket=bucket)
    moe, _, order, parallel = _ordered_moe(
        monkeypatch, serial=True, bucket=bucket, routed=routed, backend=backend, beside=True
    )
    logits = torch.zeros(ROWS, 8)

    def gate(hidden_states):
        order.append("gate")
        return logits

    moe.gate = gate
    return moe, backend, order, parallel, logits


@pytest.mark.parametrize("bucket", [32, 64, 128, 1024])
def test_a_serial_step_routes_and_stages_beside_the_shared_expert(monkeypatch, bucket):
    moe, backend, order, parallel, logits = _routing_moe(monkeypatch, bucket)
    hidden = _hidden()
    tail._moe_forward(moe, hidden, _metadata())
    assert order == ["shared", "gate", "routed"] and parallel == [True]
    ((x, staged_logits, chunk_max),) = backend.prestaged
    assert x is hidden and staged_logits is logits and chunk_max == max(ALL_RANK)
    assert backend.requests == [(ALL_RANK, ROWS)]


def test_a_backend_that_does_not_route_beside_keeps_the_serial_order(monkeypatch):
    moe, backend, order, parallel, _ = _routing_moe(monkeypatch, 32, beside=False)
    tail._moe_forward(moe, _hidden(), _metadata())
    assert order == ["gate", "routed", "shared"] and parallel == [] and backend.prestaged == []


def test_a_step_beside_the_launch_does_not_route_beside_the_shared_expert(monkeypatch):
    moe, backend, _, parallel, _ = _routing_moe(monkeypatch, 128)
    monkeypatch.setenv("TRTLLM_M31_MEGAMOE_SHARED_SERIAL_MAX_ROWS", "32")
    backend.environ = megamoe_local_plan.lever_environ(os.environ)
    tail._moe_forward(moe, _hidden(), _metadata())
    assert parallel == [True] and backend.prestaged == []


def test_a_deferred_launch_runs_form_a_after_the_route_beside_the_shared_expert(monkeypatch):
    deferred = _deferred()
    moe, _, order, _, _ = _routing_moe(monkeypatch, 32, routed=deferred)
    out = tail._moe_forward(moe, _hidden(), _metadata())
    assert order == ["shared", "gate", "routed"]
    assert deferred.ran[0] == "a" and torch.equal(
        out, torch.ones(ROWS, HIDDEN, dtype=torch.bfloat16)
    )


# --- form (b) --------------------------------------------------------------------------------------------------------


def _residual():
    return torch.zeros(ROWS, HIDDEN, dtype=torch.bfloat16)


def test_a_deferred_launch_runs_form_b_with_the_next_norm():
    deferred = _deferred()
    layer = _layer(_moe(routed=deferred))
    residual = _residual()
    hidden, res = tail._forward_moe_norm(layer, _hidden(), _metadata(), residual)
    norm = layer.next_layer_layernorm
    assert deferred.ran[0] == "b"
    assert deferred.ran[2] is residual and res is residual
    assert deferred.ran[3] is norm.weight and deferred.ran[4] == norm.variance_epsilon
    assert torch.equal(hidden, torch.full((ROWS, HIDDEN), 2.0, dtype=torch.bfloat16))
    assert layer.norms == []


def test_a_reduced_launch_runs_the_stock_add_then_the_stock_norm():
    layer = _layer(_moe())
    tail._forward_moe_norm(layer, _hidden(), _metadata(), _residual())
    assert len(layer.norms) == 1
    assert torch.equal(layer.norms[0], torch.full((ROWS, HIDDEN), 0.75, dtype=torch.bfloat16))


@pytest.mark.parametrize(
    "residual",
    [
        pytest.param(lambda: torch.zeros(ROWS, HIDDEN, dtype=torch.float32), id="fp32"),
        pytest.param(lambda: torch.zeros(ROWS + 1, HIDDEN, dtype=torch.bfloat16), id="rows"),
        pytest.param(lambda: torch.zeros(HIDDEN, ROWS, dtype=torch.bfloat16).t(), id="strided"),
    ],
)
def test_a_residual_form_b_cannot_take_runs_form_a_then_the_stock_norm(residual):
    deferred = _deferred()
    layer = _layer(_moe(routed=deferred))
    tail._forward_moe_norm(layer, _hidden(), _metadata(), residual())
    assert deferred.ran[0] == "a" and len(layer.norms) == 1


def test_a_norm_that_gained_an_nvfp4_scale_runs_form_a_then_the_stock_norm():
    deferred = _deferred()
    layer = _layer(_moe(routed=deferred))
    layer.next_layer_layernorm.nvfp4_scale = torch.ones(1)
    tail._forward_moe_norm(layer, _hidden(), _metadata(), _residual())
    assert deferred.ran[0] == "a" and len(layer.norms) == 1


def test_a_capture_layer_runs_the_built_in_forward_moe(monkeypatch):
    calls = []
    monkeypatch.setattr(
        MiniMaxM3DecoderLayer,
        "forward_MoE",
        lambda self, *args, **kwargs: calls.append((self, args, kwargs)) or ("h", "r"),
    )
    backend = _Backend()
    layer = _layer(_moe(backend))
    spec = types.SimpleNamespace(is_layer_capture=lambda idx: idx == 5)
    assert tail._forward_moe_norm(
        layer, _hidden(), _metadata(), _residual(), spec_metadata=spec
    ) == ("h", "r")
    assert len(calls) == 1 and calls[0][0] is layer
    assert backend.requests == []


def _norm_layer(**norm_attrs):
    layer = _layer(_moe())
    for name, value in norm_attrs.items():
        setattr(layer.next_layer_layernorm, name, value)
    return layer


@pytest.mark.parametrize(
    "build, reason",
    [
        pytest.param(lambda: _layer(_moe(), norm=False), "no next-layer norm", id="no-norm"),
        pytest.param(
            lambda: types.SimpleNamespace(
                next_layer_layernorm=RMSNorm(hidden_size=HIDDEN, eps=1e-6),
                post_feed_forward_fusion=True,
            ),
            "the boundary norm is fused with an AllReduce",
            id="post-ff-fusion",
        ),
        pytest.param(
            lambda: types.SimpleNamespace(
                next_layer_layernorm=torch.nn.LayerNorm(HIDDEN), post_feed_forward_fusion=False
            ),
            "norm LayerNorm",
            id="other-norm",
        ),
        pytest.param(
            lambda: _norm_layer(is_nvfp4=True, nvfp4_scale=torch.ones(1)),
            "the norm is fused with an NVFP4 quantize",
            id="nvfp4",
        ),
        pytest.param(
            lambda: _norm_layer(use_cuda_tile=True),
            "the norm is not FlashInfer's fused add + RMSNorm",
            id="cuda-tile",
        ),
        pytest.param(
            lambda: _norm_layer(use_gemma=True),
            "the norm is not FlashInfer's fused add + RMSNorm",
            id="gemma",
        ),
        pytest.param(
            lambda: _norm_layer(return_hp_output=True),
            "the norm is not FlashInfer's fused add + RMSNorm",
            id="hp-output",
        ),
    ],
)
def test_norm_declines(monkeypatch, build, reason):
    monkeypatch.setattr(tail, "DEP_TAIL_NORM", True)
    assert tail.norm_decline_reason(build()) == reason


def test_norm_declines_without_flashinfer(monkeypatch):
    monkeypatch.setattr(tail, "DEP_TAIL_NORM", True)
    monkeypatch.setattr(tail, "IS_FLASHINFER_AVAILABLE", False)
    assert tail.norm_decline_reason(_layer(_moe())) == "FlashInfer is not available"


def test_norm_declines_a_cpu_weight(monkeypatch):
    monkeypatch.setattr(tail, "DEP_TAIL_NORM", True)
    monkeypatch.setattr(tail, "IS_FLASHINFER_AVAILABLE", True)
    assert (
        tail.norm_decline_reason(_layer(_moe()))
        == "the norm weight is not a contiguous 16-byte-aligned BF16 CUDA tensor"
    )


def test_norm_declines_when_its_switch_is_off(monkeypatch):
    monkeypatch.setattr(tail, "DEP_TAIL_NORM", False)
    assert tail.norm_decline_reason(_layer(_moe())) == "TRTLLM_M31_DISABLE_DEP_TAIL_NORM=1"


# --- install ---------------------------------------------------------------------------------------------------------


def _config(**overrides):
    """The served DEP GEN config's fields the install reads: CUDA graphs on, the MoE cap at max_num_tokens."""
    fields = dict(use_cuda_graph=True, max_num_tokens=384, moe_max_num_tokens=384, extra_attrs={})
    fields.update(overrides)
    return types.SimpleNamespace(**fields)


@pytest.mark.parametrize(
    "overrides, reason",
    [
        pytest.param(
            {"extra_attrs": {"torch_compile_enabled": True}},
            "torch.compile is on",
            id="torch-compile",
        ),
        pytest.param(
            {"moe_max_num_tokens": 192},
            "moe_max_num_tokens 192 is below max_num_tokens 384 with CUDA graphs on, so a step can run in chunks",
            id="chunks-under-cuda-graphs",
        ),
    ],
)
def test_install_declines_a_config_where_a_bucket_could_change_kernel_class(
    eligible_norm, overrides, reason
):
    layer = _layer(_moe())
    assert tail.config_decline_reason(_config(**overrides)) == reason
    assert tail.install([layer], _config(**overrides)) == (0, 0)
    assert "forward" not in layer.block_sparse_moe.__dict__ and "forward_MoE" not in layer.__dict__
    assert ("not installed", reason) in tail._LOGGED


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({}, id="cap-at-max"),
        pytest.param({"moe_max_num_tokens": 1536}, id="cap-above-max"),
        pytest.param(
            {"moe_max_num_tokens": 192, "use_cuda_graph": False}, id="chunks-without-cuda-graphs"
        ),
        pytest.param({"extra_attrs": {"torch_compile_enabled": False}}, id="torch-compile-off"),
    ],
)
def test_install_admits_a_config_where_every_bucket_keeps_one_kernel_class(
    eligible_norm, overrides
):
    assert tail.config_decline_reason(_config(**overrides)) is None
    assert tail.install([_layer(_moe())], _config(**overrides)) == (0, 1)


def test_install_puts_form_b_where_the_norm_admits_it_and_form_a_elsewhere(monkeypatch):
    monkeypatch.setattr(tail, "DEP_TAIL", True)
    admitted = _layer(_moe())
    form_a_only = _layer(_moe(), norm=False)
    declined = _layer(_moe(shared_experts=False))
    dense = types.SimpleNamespace(block_sparse_moe=None)
    monkeypatch.setattr(
        tail,
        "norm_decline_reason",
        lambda layer: None if layer is admitted else "no next-layer norm",
    )
    layers = [admitted, form_a_only, declined, dense]
    assert tail.install(layers, _config()) == (1, 1)
    assert admitted.forward_MoE.__func__ is tail._forward_moe_norm
    assert admitted.block_sparse_moe.forward.__func__ is tail._moe_forward
    assert form_a_only.block_sparse_moe.forward.__func__ is tail._moe_forward
    assert "forward_MoE" not in form_a_only.__dict__
    assert "forward" not in declined.block_sparse_moe.__dict__
    assert tail.install(layers, _config()) == (1, 1)


def test_install_leaves_a_forward_something_else_replaced(eligible_norm):
    layer = _layer(_moe())
    other = types.MethodType(lambda self, *a, **k: None, layer.block_sparse_moe)
    layer.block_sparse_moe.forward = other
    assert tail.install([layer], _config()) == (0, 0)
    assert layer.block_sparse_moe.forward is other


def test_install_sets_nothing_when_off(monkeypatch):
    monkeypatch.setattr(tail, "DEP_TAIL", False)
    layer = _layer(_moe())
    assert tail.install([layer], _config()) == (0, 0)
    assert "forward" not in layer.block_sparse_moe.__dict__ and "forward_MoE" not in layer.__dict__


def test_an_installed_moe_block_calls_the_tail_forward(eligible_norm):
    deferred = _deferred()
    layer = _layer(_moe(routed=deferred))
    tail.install([layer], _config())
    out = layer.block_sparse_moe(_hidden(), _metadata())
    assert deferred.ran[0] == "a" and torch.equal(
        out, torch.ones(ROWS, HIDDEN, dtype=torch.bfloat16)
    )


def test_the_switches_are_the_dep_tail_items():
    assert tail.DEP_TAIL == (not m31_switches.m31_disabled("DEP_TAIL"))
    assert tail.DEP_TAIL_NORM == (not m31_switches.m31_disabled("DEP_TAIL_NORM"))


def test_the_shared_placement_installs_where_the_tail_does_not():
    backend = _PrestagingBackend(static_reason="the tail is off")
    moe = _moe(backend)
    layer = _layer(moe)
    assert tail.install_shared_plan([layer, types.SimpleNamespace()]) == 1
    assert tail.serves(moe) and moe._dep_tail_defer is False
    assert moe.shared_corun_pinnable and moe.route_beside_shared
    tail._moe_forward(moe, _hidden(), _metadata())
    assert backend.requests == [] and backend.cancels == 0
    # A second install keeps the block as it is.
    assert tail.install_shared_plan([layer]) == 1


def test_the_shared_serial_switch_is_read_at_install(monkeypatch):
    monkeypatch.setenv("TRTLLM_M31_DISABLE_MEGAMOE_SHARED_SERIAL", "1")
    moe = _moe(_Backend())
    assert tail.install_shared_plan([_layer(moe)]) == 1
    assert moe.shared_expert_serial is False and moe.route_beside_shared is False


def test_a_backend_without_the_placement_keeps_the_built_in_forward():
    class _Plain:
        pass

    moe = _moe()
    moe.experts.backend = _Plain()
    assert tail.install_shared_plan([_layer(moe)]) == 0
    assert not tail.serves(moe)
