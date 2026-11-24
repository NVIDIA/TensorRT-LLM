# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The DFlash2 worker's target-token selection and acceptance through the family spec operations.

The target's DFlashWorker overrides SpecWorkerBase's selection / acceptance methods only to call the
engine kernels through operations. Each override must return what the base method returns, bit for
bit, on the inputs the target hands it: greedy and sampled batches, both advanced-sampling kernel
families, a (single-rank) vocab-sharded batch, and the DFlash2 candidate top-k.
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k31.kimi_k31_pro_mxfp4__sm_10x__default.modeling import (
    K31DFlashWorker,
)
from tensorrt_llm._torch.pyexecutor.sampler.ops import flashinfer as flashinfer_sampling
from tensorrt_llm._torch.speculative.dflash import DFlashWorker as EngineDFlashWorker
from tensorrt_llm._torch.speculative.interface import SpecWorkerBase
from tensorrt_llm._torch.speculative.vocab_parallel import (
    VocabParallelHead,
    VocabShardedLogits,
    _PairExchange,
)
from tensorrt_llm.llmapi.llm_args import AdvancedSamplingMode

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() not in ((10, 3), (10, 7)),
    reason="certified on sm_103 / sm_107",
)

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
    # Identity: the forced-acceptance hook is the same call on both sides.
    worker._apply_force_accepted_tokens = lambda num, num_contexts, k, spec_metadata=None: num
    return worker


def _metadata(rows: int, greedy: bool, mode: AdvancedSamplingMode, seed: int = 0):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return SimpleNamespace(
        is_all_greedy_sample=greedy,
        is_ragged_verify=False,
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


@pytest.mark.parametrize("mode", list(AdvancedSamplingMode))
@pytest.mark.parametrize("rows", [6, 40])
def test_sample_and_compute_probs_match_base(mode, rows, monkeypatch):
    logits = _logits(rows, rows)
    md = _metadata(rows, greedy=False, mode=mode)
    args = (mode, logits, md.temperatures, md.top_ks, md.top_ps, md.min_ps)
    ours = K31DFlashWorker._sample_from_logits(*args, seed=md.seeds, offset=md.offsets)
    base = SpecWorkerBase._sample_from_logits(*args, seed=md.seeds, offset=md.offsets)
    assert torch.equal(ours, base)
    if mode.is_fused:
        assert torch.equal(
            K31DFlashWorker._compute_probs_from_logits(*args),
            SpecWorkerBase._compute_probs_from_logits(*args),
        )
        return
    # The flashinfer top-p renormalization is not bitwise reproducible from call to call, so the
    # flashinfer modes check that the override makes the base method's call, argument for argument
    # (the kernel's numerics are its operation's test).
    calls = []

    def record(*call_args, **call_kwargs):
        calls.append((call_args, call_kwargs))
        return torch.zeros(1)

    monkeypatch.setattr(flashinfer_sampling, "compute_probs_from_logits", record)
    K31DFlashWorker._compute_probs_from_logits(*args)
    SpecWorkerBase._compute_probs_from_logits(*args)
    (ours_args, ours_kwargs), (base_args, base_kwargs) = calls
    assert len(ours_args) == len(base_args) and ours_kwargs == base_kwargs
    assert all(a is b for a, b in zip(ours_args, base_args))


@pytest.mark.parametrize("greedy", [True, False])
@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("batch", [(0, 4, 4), (2, 3, 7)])
def test_select_and_accept_match_base(greedy, sharded, batch):
    num_contexts, num_gens, k = batch
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


@pytest.mark.parametrize("shape", [(3, 5), (8, 7)])
def test_dflash2_global_top_k_full_vocab_matches_engine(shape):
    worker = _worker()
    gen_logits = _logits(shape[0] * shape[1], 11).view(*shape, VOCAB)
    ours = K31DFlashWorker._dflash2_global_top_k(worker, gen_logits.clone(), None, 32, VOCAB)
    base = EngineDFlashWorker._dflash2_global_top_k(worker, gen_logits.clone(), None, 32, VOCAB)
    for a, b in zip(ours, base):
        assert torch.equal(a, b)
