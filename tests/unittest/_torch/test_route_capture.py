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

from tensorrt_llm._torch.route_capture import RouteCapture, assert_capturable

_L, _K = 4, 2  # small MoE-layer count / top-k for tests


def _row(v: int) -> torch.Tensor:
    """A distinct [L, K] int16 route row."""
    return torch.full((_L, _K), v, dtype=torch.int16)


def test_assemble_contract_drops_final_position():
    rc = RouteCapture(rank=0)
    # positions [0, 3): assemble keeps [0, max_pos) == [0, 2) (drops the final).
    rc._store[7] = {0: _row(10), 1: _row(11), 2: _row(12)}
    out = rc.assemble(7)
    assert out.shape == (2, _L, _K)
    assert out.dtype == torch.int16
    assert torch.equal(out[0], _row(10))
    assert torch.equal(out[1], _row(11))


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
    out = CompletionOutput(
        index=0, additional_generation_outputs={"routed_experts": [routes]})
    assert out.routed_experts is not None
    assert out.routed_experts.shape == (5, _L, _K)
    assert torch.equal(out.routed_experts, routes)
