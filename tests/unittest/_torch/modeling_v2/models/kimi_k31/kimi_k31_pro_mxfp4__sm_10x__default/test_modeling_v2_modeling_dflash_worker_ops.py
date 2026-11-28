# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 worker's target-token selection and acceptance through the family spec operations.

The target's DFlashWorker overrides SpecWorkerBase's selection / acceptance methods only to call the
engine kernels through operations. Each override must return what the base method returns, bit for
bit, on the inputs the target hands it: a greedy batch and a sampled (single-rank) vocab-sharded
batch.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_10x__default.modeling import (
    K31DFlashWorker,
)
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase
from tensorrt_llm._torch.speculative.vocab_parallel import (
    VocabParallelHead,
    VocabShardedLogits,
    _PairExchange,
)
from tensorrt_llm.llmapi.llm_args import AdvancedSamplingMode

__extra_import_path__ = [".."]
from _k31_utils import on_sm

pytestmark = on_sm(103, 107)

VOCAB = 163840


class _SingleRankHead:
    """A vocab-parallel head over one rank (no exchange), running the engine's token selection."""

    def __init__(self, vocab: int):
        self.col_base = 0
        self.vocab_size = vocab
        self._exchange = _PairExchange(None, 1, 0)

    argmax = VocabParallelHead.argmax
    sample = VocabParallelHead.sample


def _worker() -> K31DFlashWorker:
    worker = object.__new__(K31DFlashWorker)
    worker.mapping = None
    # The base __init__ sets it: no forced acceptance rate.
    worker.force_num_accepted_tokens = 0.0
    # Identity: the forced-acceptance hook is the same call on both sides.
    worker._apply_force_accepted_tokens = lambda num, num_contexts, k, spec_metadata=None: num
    return worker


def _metadata(rows: int, greedy: bool, mode: AdvancedSamplingMode, seed: int = 0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return SimpleNamespace(
        is_all_greedy_sample=greedy,
        is_ragged_verify=False,
        sharded_target_logits=False,
        is_cuda_graph=False,
        advanced_sampling_mode=mode,
        temperatures=torch.rand(rows, generator=gen, device="cuda") + 0.5,
        top_ks=torch.full((rows,), 64, dtype=torch.int32, device="cuda"),
        top_ps=torch.full((rows,), 0.95, dtype=torch.float32, device="cuda"),
        min_ps=torch.zeros(rows, dtype=torch.float32, device="cuda"),
        seeds=torch.full((rows,), 1234 + seed, dtype=torch.int64, device="cuda"),
        offsets=torch.arange(rows, dtype=torch.int64, device="cuda") * 17,
    )


def _logits(rows: int, seed: int, dtype=torch.float32) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return (torch.randn(rows, VOCAB, generator=gen, device="cuda") * 3.0).to(dtype)


@pytest.mark.parametrize(
    "greedy, sharded", [(True, False), (False, True)], ids=["greedy", "sampled-sharded"]
)
def test_select_and_accept_match_base(greedy, sharded):
    # Contexts and generations mixed in one batch.
    num_contexts, num_gens, k = 2, 3, 7
    batch_size = num_contexts + num_gens
    rows = num_contexts + num_gens * (k + 1)
    worker = _worker()
    md = _metadata(rows, greedy=greedy, mode=AdvancedSamplingMode.FULL, seed=rows)
    local = _logits(rows, 7 * rows + num_contexts, torch.bfloat16 if sharded else torch.float32)
    logits = VocabShardedLogits(_SingleRankHead(VOCAB), local) if sharded else local
    if sharded and not greedy:
        # The vocab-parallel draw is temperature-only.
        md.top_ks.fill_(0)
        md.top_ps.fill_(1.0)

    ours = K31DFlashWorker._sample_tokens_for_batch(worker, logits, md, num_contexts, batch_size)
    base = SpecWorkerBase._sample_tokens_for_batch(worker, logits, md, num_contexts, batch_size)
    assert torch.equal(ours, base)

    gen = torch.Generator(device="cuda").manual_seed(batch_size)
    draft_tokens = torch.randint(0, VOCAB, (num_gens, k), generator=gen, device="cuda").int()
    # Make some drafts agree with the target so acceptance counts vary.
    target = ours.view(-1)[num_contexts:].view(num_gens, k + 1)[:, :k].int()
    draft_tokens[: num_gens // 2 + 1, : k // 2] = target[: num_gens // 2 + 1, : k // 2]
    ours_accept = K31DFlashWorker._sample_and_accept_draft_tokens_base(
        worker, logits, draft_tokens, num_contexts, batch_size, md
    )
    base_accept = SpecWorkerBase._sample_and_accept_draft_tokens_base(
        worker, logits, draft_tokens, num_contexts, batch_size, md
    )
    (ours_tokens, ours_num), (base_tokens, base_num) = ours_accept, base_accept
    assert torch.equal(ours_num, base_num)
    # Only the accepted prefix of each row is defined (the rest of the buffer is never written).
    valid = torch.arange(k + 1, device="cuda").unsqueeze(0) < base_num.unsqueeze(1)
    assert torch.equal(ours_tokens[valid], base_tokens[valid])
