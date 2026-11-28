# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The chain's layer plan (``chain.plan_chain``) and its KDA state contract (CPU)."""

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k3_vl.ops.chain import k3_ull_chain as chain

pytestmark = pytest.mark.cpu_only

MLA = (*range(3, 91, 4), 91, 92)


def test_model_layers():
    layers = chain.model_layers()
    assert len(layers) == 93 and layers[0] == (0, chain.CHAIN_OURS)
    assert [i for i, k in layers if k == chain.CHAIN_MOE] == list(MLA)
    assert len([i for i, k in layers if k == chain.CHAIN_CHAINED]) == 68


def test_plan_full_model():
    plan = chain.plan_chain(chain.model_layers())
    assert plan.indices == tuple(range(1, 93))  # MoE tail ordinal = position: 0 .. 91
    assert len(plan.indices) % 3 != 1
    assert plan.first == {1}
    # Every chain layer before an MLA layer ends a run; the model's last layer too.
    assert plan.last == {i - 1 for i in MLA} | {92}
    assert plan.next_of[92] is None and plan.next_of[90] == 91 and plan.next_of[1] == 2
    assert plan.moe_only == set(MLA)


def test_plan_subset_with_layers_off_the_chain():
    keep = {1, 2, 3, 45, 89, 90, 91, 92}
    layers = tuple((i, k if i in keep else chain.CHAIN_OURS) for i, k in chain.model_layers())
    plan = chain.plan_chain(layers)
    assert plan.indices == tuple(sorted(keep))
    assert plan.first == {1, 45, 89}
    assert plan.last == {2, 3, 45, 90, 91, 92}
    assert plan.next_of[3] == 4 and plan.next_of[45] == 46


def test_plan_rejects():
    layers = list(chain.model_layers())
    with pytest.raises(ValueError, match="1 mod 3"):
        chain.plan_chain([(0, chain.CHAIN_OURS), (1, chain.CHAIN_CHAINED), (2, chain.CHAIN_MOE), (3, chain.CHAIN_MOE),
                          (4, chain.CHAIN_CHAINED)])  # fmt: skip
    with pytest.raises(ValueError, match="last layer"):
        chain.plan_chain(
            [(0, chain.CHAIN_OURS), (1, chain.CHAIN_CHAINED), (2, chain.CHAIN_CHAINED)]
        )
    # A layer off the chain at a block boundary right after a chain layer (its consumer leaves
    # no block snapshot).
    off = [(i, chain.CHAIN_OURS if i in (12, 13) else k) for i, k in layers]
    with pytest.raises(ValueError, match="block boundary"):
        chain.plan_chain(off)
    with pytest.raises(ValueError, match="kinds"):
        chain.plan_chain([(0, 3)])


class _State:
    def __init__(self, dtype):
        self.dtype = dtype

    def caches(self, layer_idx):
        return [torch.empty(4, 2304, 10), torch.empty(4, 6, 128, 128, dtype=self.dtype)]

    def indices(self, state_indices, slots, accepted):
        slots.copy_(state_indices[: slots.shape[0], None] * 8 + torch.arange(slots.shape[1]))
        accepted.fill_(1)


def test_kda_state_protocol():
    state = _State(torch.bfloat16)
    assert isinstance(state, chain.KdaCheckpointState)
    assert chain.K3UllChain._state_dtype(state, 1) == torch.bfloat16
    with pytest.raises(TypeError):
        chain.K3UllChain._state_dtype(object(), 1)


def test_bind_slots_uses_the_step_requests():
    rt = chain.K3UllChain.__new__(chain.K3UllChain)
    rt.kda_state = _State(torch.float32)
    md = type("Md", (), {})()
    md.num_spec_decodes = 2
    md.spec_state_indices_tensor = torch.zeros(2, 8, dtype=torch.int32)
    md.num_accepted_tokens = torch.zeros(2, dtype=torch.int32)
    rt.metadata = {16: md}
    rt.bind_slots(torch.tensor([5, 9, 11], dtype=torch.int32), 16)
    assert md.spec_state_indices_tensor[1].tolist() == list(range(72, 80))
    assert md.num_accepted_tokens.tolist() == [1, 1]
    with pytest.raises(RuntimeError):
        rt.bind_slots(torch.tensor([5], dtype=torch.int32), 16)
