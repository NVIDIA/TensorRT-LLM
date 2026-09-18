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
            self.py_return_routed_experts = True
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


# ---- SharedRouteCache bound + per-request gating -------------------------------


class _FakeResult:
    def __init__(self):
        self._additional_generation_outputs = None
        self.appended = []

    def append_additional_generation_outputs(self, name, value):
        self.appended.append((name, value))


class _FakeReq:
    """A finished request as attach_routes sees it."""

    def __init__(self, rid, flag=True):
        self.py_request_id = rid
        self.py_return_routed_experts = flag
        self.py_result = _FakeResult()


class _CtxReq(_FakeReq):
    """A scheduled context request whose first ``prepop`` prompt tokens hit the
    KV prefix cache (so only [prepop, len) run through the model)."""

    def __init__(self, rid, toks, prepop=0, flag=True):
        super().__init__(rid, flag)
        self.is_dummy = False
        self._toks = list(toks)
        self.py_prompt_len = len(toks)
        self.prepopulated_prompt_len = prepop
        self.context_current_position = prepop
        self.context_chunk_size = len(toks) - prepop

    def get_tokens(self, beam):
        return list(self._toks)


class _Batch:
    def __init__(self, ctx=(), gen=()):
        self.context_requests = list(ctx)
        self.generation_requests = list(gen)


def _publish(rc, rid, toks):
    """Owner ``rid`` captured every prompt position and publishes them."""
    rc._store[rid] = {p: _row(p) for p in range(len(toks))}
    rc._req_toks[rid] = list(toks)
    rc._req_plen[rid] = len(toks)
    hashes = rc._hashes_for(rid, list(toks), len(toks))
    rc._store_positions(rid, hashes)
    return hashes


def test_shared_cache_bounded_with_lru_eviction():
    from tensorrt_llm._torch.route_capture import _DEFAULT_SHARED_CAPACITY

    rc = RouteCapture(rank=0)
    assert rc._shared_cap == _DEFAULT_SHARED_CAPACITY
    rc.prepare(_Batch(), 32, shared_capacity=4)
    toks = list(range(100, 108))
    hashes = _publish(rc, 1, toks)
    assert len(rc._shared) == 4 and list(rc._shared) == hashes[4:]
    assert rc._pfx_evicted == 4
    # A read-back hit refreshes the entry: the next insert evicts the oldest
    # untouched key (hashes[5]), not the touched one (hashes[4]).
    assert rc._shared_get(hashes[4]) is not None
    _publish(rc, 9, [1])
    assert hashes[4] in rc._shared and hashes[5] not in rc._shared
    assert rc._shared_get(12345) is None
    # prepare() without a capacity keeps the current one; shrinking evicts at
    # once; a negative capacity is a programming error.
    rc.prepare(_Batch(), 32)
    assert rc._shared_cap == 4
    rc.prepare(_Batch(), 32, shared_capacity=2)
    assert len(rc._shared) == 2
    with pytest.raises(ValueError):
        rc.prepare(_Batch(), 32, shared_capacity=-1)


def test_shared_cache_zero_capacity_publishes_nothing():
    rc = RouteCapture(rank=0)
    rc.prepare(_Batch(), 32, shared_capacity=0)  # block reuse off
    _publish(rc, 1, list(range(8)))
    assert len(rc._shared) == 0


def test_prefix_hit_after_eviction_fills_missing_with_sentinel_not_gap():
    rc = RouteCapture(rank=0)
    rc.prepare(_Batch(), 32, shared_capacity=4)
    toks = list(range(100, 108))
    _publish(rc, 1, toks)  # positions 0..3 evicted, 4..7 still cached
    sib = _CtxReq(2, toks, prepop=6)  # KV prefix hit on the first 6 tokens
    rc.prepare(_Batch(ctx=[sib]), 32)
    assert rc._req_reused[2] == 6
    assert set(rc._store[2]) == {4, 5} and 2 not in rc._readback_done
    rc._store[2][6] = _row(6)  # this request's own captured rows
    rc._store[2][7] = _row(7)
    rc.attach_routes(sib)
    ((_, routes),) = sib.py_result.appended
    assert routes.shape == (8, _L, _K)
    assert bool((routes[:4] == -1).all())  # evicted reused positions -> sentinel
    assert torch.equal(routes[4], _row(4)) and torch.equal(routes[7], _row(7))
    assert rc._pfx_misses == 4
    # The sentinel is never published under the evicted keys.
    assert all(v is not rc._missing_row for v in rc._shared.values())
    assert 2 not in rc._store and 2 not in rc._req_reused


def test_attach_final_readback_fills_late_owner_without_sentinel():
    rc = RouteCapture(rank=0)
    rc.prepare(_Batch(), 32, shared_capacity=64)
    toks = list(range(200, 206))
    sib = _CtxReq(2, toks, prepop=4)
    rc.prepare(_Batch(ctx=[sib]), 32)  # owner has not published yet -> misses
    assert 2 not in rc._readback_done and not rc._store.get(2)
    _publish(rc, 1, toks)  # owner publishes later
    rc._store[2][4] = _row(4)
    rc._store[2][5] = _row(5)
    rc.attach_routes(sib)
    ((_, routes),) = sib.py_result.appended
    assert routes.shape == (6, _L, _K) and int(routes.min()) >= 0
    assert rc._pfx_misses == 0


def test_internal_gap_at_or_above_reused_prefix_still_raises():
    rc = RouteCapture(rank=0)
    rc._store[3] = {0: _row(1), 1: _row(2), 3: _row(4)}
    rc._req_reused[3] = 2  # positions >= 2 were scheduled for capture
    with pytest.raises(ValueError):
        rc.attach_routes(_FakeReq(3))


def test_evicted_entry_survives_for_in_flight_reader():
    rc = RouteCapture(rank=0)
    rc.prepare(_Batch(), 32, shared_capacity=4)
    toks = list(range(100, 108))
    _publish(rc, 1, toks)
    sib = _CtxReq(2, toks, prepop=6)
    rc.prepare(_Batch(ctx=[sib]), 32)  # reads back positions 4, 5 by reference
    _publish(rc, 5, list(range(500, 504)))  # evicts the owner's remaining keys
    assert torch.equal(rc._store[2][4], _row(4)) and torch.equal(rc._store[2][5], _row(5))


def test_clear_shared_resets_bounded_cache_but_keeps_capacity():
    rc = RouteCapture(rank=0)
    rc.prepare(_Batch(), 32, shared_capacity=4)
    _publish(rc, 1, list(range(8)))
    rc.clear_shared()
    assert len(rc._shared) == 0 and rc._shared_cap == 4 and rc._pfx_evicted == 4
    _publish(rc, 2, list(range(3)))
    assert len(rc._shared) == 3


def test_attach_routes_opt_out_frees_store_and_feeds_shared_cache():
    rc = RouteCapture(rank=0)
    rc.prepare(_Batch(), 32, shared_capacity=64)
    toks = list(range(300, 304))
    rc._store[8] = {p: _row(p) for p in range(4)}
    rc._req_toks[8] = list(toks)
    rc._req_plen[8] = 4
    rc._gen_count[8] = 0
    req = _FakeReq(8, flag=False)
    rc.attach_routes(req)
    assert req.py_result.appended == []
    assert 8 not in rc._store and 8 not in rc._req_toks
    assert 8 not in rc._gen_count and 8 not in rc._attached
    hashes = rc._hashes_for(99, toks, 4)
    assert all(h in rc._shared for h in hashes)  # prompt rows published for reuse
    rc.attach_routes(req)  # revisit after the release is a no-op
    assert req.py_result.appended == []


def test_opt_out_owner_feeds_opt_in_prefix_hit():
    rc = RouteCapture(rank=0)
    rc.prepare(_Batch(), 32, shared_capacity=64)
    toks = list(range(300, 304))
    rc._store[8] = {p: _row(p) for p in range(4)}
    rc._req_toks[8] = list(toks)
    rc._req_plen[8] = 4
    rc.attach_routes(_FakeReq(8, flag=False))
    # An opted-in sibling reuses the whole prompt (+1 new token): every reused
    # position comes from the cache the opted-out owner filled.
    sib = _CtxReq(2, toks + [999], prepop=4)
    rc.prepare(_Batch(ctx=[sib]), 32)
    assert set(rc._store[2]) == {0, 1, 2, 3} and 2 in rc._readback_done
    rc._store[2][4] = _row(4)
    rc.attach_routes(sib)
    ((_, routes),) = sib.py_result.appended
    assert routes.shape == (5, _L, _K) and int(routes.min()) >= 0
    assert rc._pfx_misses == 0
