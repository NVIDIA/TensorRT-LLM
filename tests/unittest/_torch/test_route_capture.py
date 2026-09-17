# Copyright 2026 NVIDIA CORPORATION & AFFILIATES
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
#
# SPDX-License-Identifier: Apache-2.0
"""CPU-only unit tests for Router Replay (R3) capture.

These cover the pure-logic core (no GPU / no engine): the output-assembly
contract, prefix-cache position keying + store/read-back round trip, fail-closed
backend gating, and the opt-in config flags.
"""

import pytest
import torch

from tensorrt_llm._torch.route_capture import (
    ROUTE_CAPTURE_ATTR,
    RouteCapture,
    assert_capturable,
    get_active_route_capture,
)

pytestmark = pytest.mark.cpu_only
_L, _K = 4, 2  # small MoE-layer count / top-k for tests


def _row(v: int) -> torch.Tensor:
    """A distinct [L, K] int16 route row."""
    return torch.full((_L, _K), v, dtype=torch.int16)


def test_assemble_contract_keeps_every_captured_position():
    rc = RouteCapture(rank=0)
    # positions {0, 1, 2} are all complete, real captures -> assemble keeps
    # all of them: [0, max_pos] inclusive == 3 rows.
    rc._store[7] = {0: _row(10), 1: _row(11), 2: _row(12)}
    out = rc.assemble(7)
    assert out.shape == (3, _L, _K)
    assert out.dtype == torch.int16
    assert torch.equal(out[0], _row(10))
    assert torch.equal(out[1], _row(11))
    assert torch.equal(out[2], _row(12))


def test_assemble_none_when_empty():
    rc = RouteCapture(rank=0)
    assert rc.assemble(123) is None
    rc._store[1] = {0: _row(1)}  # only one position -> keep == 0 -> None
    assert rc.assemble(1) is None


def test_assemble_fail_closed_on_internal_gap():
    rc = RouteCapture(rank=0)
    # position 1 missing but position 2 present -> a genuine internal gap.
    rc._store[9] = {0: _row(1), 2: _row(3)}
    with pytest.raises(ValueError):
        rc.assemble(9)


def test_attach_routes_propagates_errors_and_attaches_once():
    class _Result:
        def __init__(self):
            self._additional_generation_outputs = None
            self.appended = []

        def append_additional_generation_outputs(self, name, value):
            self.appended.append((name, value))

    class _Req:
        def __init__(self, rid):
            self.py_request_id = rid
            self.py_result = _Result()

    rc = RouteCapture(rank=0)
    # Incomplete store (a single position): nothing to attach yet -> keep the
    # store for a later call, no error.
    rc._store[5] = {0: _row(1)}
    rc.attach_routes(_Req(5))
    assert 5 in rc._store and 5 not in rc._attached
    # Internal gap: fail closed -- the error surfaces instead of the request
    # silently completing without routed_experts.
    rc._store[6] = {0: _row(1), 2: _row(3)}
    with pytest.raises(ValueError):
        rc.attach_routes(_Req(6))
    # Complete store: attached exactly once, then every per-request record
    # (store and attach marker) is released; a repeat call is a no-op.
    rc._store[7] = {0: _row(1), 1: _row(2)}
    req = _Req(7)
    rc.attach_routes(req)
    assert [name for name, _ in req.py_result.appended] == ["routed_experts"]
    assert req.py_result.appended[0][1].shape == (2, _L, _K)
    assert 7 not in rc._store and 7 not in rc._attached
    rc.attach_routes(req)
    assert len(req.py_result.appended) == 1
    # A later request reusing the id is not suppressed by the stale marker.
    rc._store[7] = {0: _row(5), 1: _row(6)}
    req2 = _Req(7)
    rc.attach_routes(req2)
    assert len(req2.py_result.appended) == 1


def test_abort_forward_disarms_without_staging():
    rc = RouteCapture(rank=0)
    rc._layout = [(1, 0), (2, 0)]  # armed by prepare() for a step that then failed
    rc.abort_forward()
    assert rc._layout is None
    rc.finish_forward()  # nothing armed -> no-op, no error
    assert rc._layout is None


def test_prefix_hashes_deterministic_across_requests():
    rc = RouteCapture(rank=0)
    toks = list(range(100, 164))  # 64-token shared prefix
    h1 = rc._hashes_for(1, toks, len(toks))
    h2 = rc._hashes_for(2, list(toks), len(toks))
    assert h1 == h2  # same prefix content -> same cumulative keys
    assert len(h1) == len(toks)
    assert len(set(h1)) == len(h1)  # cumulative hashes are position-distinct here


def test_prefix_store_and_readback_roundtrip():
    rc = RouteCapture(rank=0)
    toks = list(range(200, 232))  # 32-token prompt
    # Owner (rid=1) captured every prompt position.
    rc._store[1] = {p: _row(p) for p in range(len(toks))}
    rc._req_plen[1] = len(toks)
    hashes_owner = rc._hashes_for(1, toks, len(toks))
    rc._store_positions(1, hashes_owner)

    # A sibling (rid=2) sharing the same prefix computes the same keys, so every
    # position resolves to the owner's stored row -> read-back is exact.
    hashes_sib = rc._hashes_for(2, list(toks), len(toks))
    assert hashes_sib == hashes_owner
    for p in range(len(toks)):
        assert hashes_sib[p] in rc._shared
        assert torch.equal(rc._shared[hashes_sib[p]], _row(p))


def test_create_gating_and_fail_closed():
    common = dict(rank=0, model_engine=None)
    # Feature off -> no capturer at all.
    assert (
        RouteCapture.create(
            **common, enabled=False, pp_size=1, is_spec_decode=False, is_draft_model=False
        )
        is None
    )
    # Draft engines never capture, even with the feature on.
    assert (
        RouteCapture.create(
            **common, enabled=True, pp_size=1, is_spec_decode=False, is_draft_model=True
        )
        is None
    )
    # Supported path -> a capturer instance.
    rc = RouteCapture.create(
        **common, enabled=True, pp_size=1, is_spec_decode=False, is_draft_model=False
    )
    assert isinstance(rc, RouteCapture)
    # Unsupported paths fail closed instead of returning wrong routes.
    with pytest.raises(RuntimeError):
        RouteCapture.create(
            **common, enabled=True, pp_size=2, is_spec_decode=False, is_draft_model=False
        )
    with pytest.raises(RuntimeError):
        RouteCapture.create(
            **common, enabled=True, pp_size=1, is_spec_decode=True, is_draft_model=False
        )


def test_active_capturer_is_looked_up_per_engine_forward():
    from tensorrt_llm._torch.utils import model_extra_attrs

    rc = RouteCapture(rank=0)
    assert get_active_route_capture() is None  # outside any forward
    with model_extra_attrs({ROUTE_CAPTURE_ATTR: rc}):  # this engine's forward
        assert get_active_route_capture() is rc
    with model_extra_attrs({}):  # an engine without Router Replay
        assert get_active_route_capture() is None
    assert get_active_route_capture() is None


def test_assert_capturable_gates_on_separated_routing():
    class _Sep:
        def _supports_load_balancer(self):
            return True

    class _Fused:
        def _supports_load_balancer(self):
            return False

    assert_capturable(_Sep())  # separated routing -> OK
    with pytest.raises(RuntimeError):
        assert_capturable(_Fused())  # fused -> fail closed


def test_sampling_params_return_routed_experts_flag():
    from tensorrt_llm.sampling_params import SamplingParams

    assert SamplingParams().return_routed_experts is False
    assert SamplingParams(return_routed_experts=True).return_routed_experts is True


def test_llm_args_enable_flag_present_and_defaults_false():
    from tensorrt_llm.llmapi.llm_args import TorchLlmArgs

    fields = TorchLlmArgs.model_fields
    assert "enable_return_routed_experts" in fields
    assert fields["enable_return_routed_experts"].default is False


def test_completion_output_routed_experts_property():
    from tensorrt_llm.executor.result import CompletionOutput

    # None when not present.
    assert CompletionOutput(index=0).routed_experts is None
    # Surfaces the single whole-sequence tensor from the transport list.
    routes = torch.zeros((5, _L, _K), dtype=torch.int16)
    out = CompletionOutput(index=0, additional_generation_outputs={"routed_experts": [routes]})
    assert out.routed_experts is not None
    assert out.routed_experts.shape == (5, _L, _K)
    assert torch.equal(out.routed_experts, routes)
