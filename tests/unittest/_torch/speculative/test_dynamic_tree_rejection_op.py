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
"""Per-request RNG of the dynamic-tree rejection kernel.

Every test uses one tree: a root with two children, so ``max_draft_len`` is 1 and the
first emitted token is either an accepted child or the correction drawn at the root.
"""

import pytest
import torch

from tensorrt_llm._torch.speculative.dynamic_tree_ops import DynamicTreeOpsConverter

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")

DEV = "cuda"
NUM_NODES = 3
NUM_SPEC_STEP = 2


def _tree(rows: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Root 0 with children 1 and 2: first-child and next-sibling links, all valid."""
    next_token = torch.tensor([1, -1, -1], dtype=torch.int32, device=DEV).repeat(rows, 1)
    next_sibling = torch.tensor([-1, 2, -1], dtype=torch.int32, device=DEV).repeat(rows, 1)
    return next_token, next_sibling, torch.ones(rows, dtype=torch.bool, device=DEV)


def _target_probs(vocab: int, *, seed: int) -> torch.Tensor:
    gen = torch.Generator(device=DEV).manual_seed(seed)
    logits = torch.randn(NUM_NODES, vocab, device=DEV, generator=gen) * 1.5
    return torch.softmax(logits, dim=-1)


def _verify(
    draft_tokens: torch.Tensor, target_probs: torch.Tensor, seed: torch.Tensor, offset: torch.Tensor
) -> torch.Tensor:
    """Accepted/correction tokens, ``[rows, NUM_SPEC_STEP]``."""
    rows = draft_tokens.shape[0]
    next_token, next_sibling, valid = _tree(rows)
    accept_index = torch.empty(rows, NUM_SPEC_STEP, dtype=torch.int64, device=DEV)
    accept_num = torch.empty(rows, dtype=torch.int64, device=DEV)
    accept_token = torch.empty(rows, NUM_SPEC_STEP, dtype=torch.int64, device=DEV)
    torch.ops.trtllm.verify_dynamic_tree_rejection_out_op(
        draft_tokens,
        target_probs,
        next_token,
        next_sibling,
        valid,
        accept_index,
        accept_num,
        accept_token,
        NUM_SPEC_STEP,
        seed,
        offset,
    )
    return accept_token


def test_rows_with_the_same_rng_state_agree_wherever_they_sit() -> None:
    rows, vocab = 8, 1000
    request_probs = _target_probs(vocab, seed=1)
    request_draft = torch.tensor([3, 5], dtype=torch.int64, device=DEV)

    def run(position: int) -> torch.Tensor:
        probs = torch.stack([_target_probs(vocab, seed=10 + i) for i in range(rows)])
        draft = torch.randint(0, vocab, (rows, NUM_NODES - 1), dtype=torch.int64, device=DEV)
        probs[position], draft[position] = request_probs, request_draft
        seed = torch.arange(100, 100 + rows, dtype=torch.int64, device=DEV)
        offset = torch.arange(rows, dtype=torch.int64, device=DEV) * 64
        seed[position], offset[position] = 7, 32
        return _verify(draft, probs, seed, offset)[position]

    first = run(0)
    for position in range(1, rows):
        assert torch.equal(run(position), first)


def test_first_token_follows_the_target() -> None:
    """Target-only tree acceptance is lossless at the root."""
    rows, vocab = 32768, 16
    probs = _target_probs(vocab, seed=2)
    draft = torch.tensor([[0, 1]], dtype=torch.int64, device=DEV).expand(rows, -1).contiguous()
    tokens = _verify(
        draft,
        probs.unsqueeze(0).expand(rows, -1, -1).contiguous(),
        torch.arange(rows, dtype=torch.int64, device=DEV),
        torch.zeros(rows, dtype=torch.int64, device=DEV),
    )
    counts = torch.bincount(tokens[:, 0], minlength=vocab).float() / rows
    tv = 0.5 * (counts - probs[0]).abs().sum().item()
    assert tv < 0.02, f"total-variation distance {tv:.4f} from the root target"


def test_op_rejects_a_shared_rng_entry() -> None:
    rows, vocab = 4, 64
    draft = torch.zeros(rows, NUM_NODES - 1, dtype=torch.int64, device=DEV)
    probs = _target_probs(vocab, seed=3).unsqueeze(0).expand(rows, -1, -1).contiguous()
    one = torch.zeros(1, dtype=torch.int64, device=DEV)
    with pytest.raises(RuntimeError, match="one element per request"):
        _verify(draft, probs, one, one)


@pytest.mark.parametrize("per_request", [True, False], ids=["tensor", "int"])
def test_converter_accepts_one_rng_entry_per_request(per_request: bool) -> None:
    """The converter takes the per-request seed/offset the worker passes, or one int."""
    num_gens, vocab = 4, 64
    converter = DynamicTreeOpsConverter(
        dynamic_tree_max_topK=2,
        max_draft_len=1,
        max_total_draft_tokens=NUM_NODES - 1,
        max_batch_size=8,
        device=torch.device(DEV),
    )
    next_token, next_sibling, valid = _tree(num_gens)
    if per_request:
        seed = torch.arange(num_gens, dtype=torch.int64, device=DEV)
        offset = torch.zeros(num_gens, dtype=torch.int64, device=DEV)
    else:
        seed, offset = 5, 0
    _, accept_num, accept_token = converter.verify_dynamic_tree_rejection_out(
        torch.zeros(num_gens, NUM_NODES - 1, dtype=torch.int64, device=DEV),
        _target_probs(vocab, seed=4).unsqueeze(0).expand(num_gens, -1, -1).contiguous(),
        next_token,
        next_sibling,
        valid,
        num_gens,
        NUM_SPEC_STEP,
        seed=seed,
        offset=offset,
    )
    assert accept_num.shape == (num_gens,)
    assert ((accept_token[:, 0] >= 0) & (accept_token[:, 0] < vocab)).all()
