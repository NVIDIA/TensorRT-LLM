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

"""Draft sampling kernels and ADP transitions; CI requires at most two GPUs."""

from types import MethodType, SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
from tensorrt_llm._torch.speculative import interface as spec
from tensorrt_llm._torch.speculative.mtp import MTPWorker
from tensorrt_llm.llmapi.llm_args import AdvancedSamplingMode

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA/FlashInfer")


def _worker() -> SimpleNamespace:
    worker = SimpleNamespace(force_num_accepted_tokens=0.0, _d2t=None)
    for name in ("_rng_state_per_request", "_rng_state_per_token", "_apply_force_accepted_tokens"):
        setattr(worker, name, MethodType(getattr(spec.SpecWorkerBase, name), worker))
    return worker


def _metadata(batch=2, vocab=8, steps=2, device="cpu", skip=True, mode="full") -> SimpleNamespace:
    # Non-contiguous and permuted slots catch accidental writes by batch index.
    slots = torch.arange(batch, device=device).flip(0) * 2
    return SimpleNamespace(
        use_rejection_sampling=True,
        draft_skip_top_k_top_p=skip,
        advanced_sampling_mode=AdvancedSamplingMode(mode),
        request_temperatures=torch.full((batch,), 0.8, device=device),
        request_top_ks=torch.full((batch,), min(32, vocab), dtype=torch.int32, device=device),
        request_top_ps=torch.full((batch,), 0.7, device=device),
        request_seeds=torch.arange(batch, dtype=torch.int64, device=device) + 1321,
        request_offsets=torch.zeros(batch, dtype=torch.int64, device=device),
        batch_slot_ids=slots,
        draft_probs=torch.full((2 * batch + 1, steps, vocab), -1.0, device=device),
        vocab_size=vocab,
    )


def _real_metadata(skip=True, batch=8) -> spec.SpecMetadata:
    return spec.SpecMetadata(
        max_num_requests=batch,
        max_draft_len=2,
        max_total_draft_tokens=2,
        runtime_draft_len=2,
        vocab_size=128,
        spec_dec_mode=spec.SpeculativeDecodingMode.MTP_EAGLE_ONE_MODEL,
        use_rejection_sampling=True,
        draft_skip_top_k_top_p=skip,
    )


@CUDA
@pytest.mark.parametrize("skip", [False, True])
def test_cuda_actual_sampler_saved_q_and_graph_replay(skip) -> None:
    meta = _metadata(batch=8, vocab=128, steps=8, device="cuda", skip=skip)
    logits = torch.linspace(-3, 3, 128, device="cuda").repeat(8, 1)
    top_ps = meta.request_top_ps.clone()
    worker = _worker()

    def draft():
        return spec.SpecWorkerBase.advanced_sample_draft(worker, logits, meta, 8, draft_step=3)

    # Warm the actual compiled sampler before capturing CUDA work.
    with torch.cuda.stream(torch.cuda.Stream()):
        for _ in range(3):
            draft()
    torch.cuda.synchronize()
    expected = spec.compute_probs_from_logits(
        logits,
        meta.request_temperatures,
        None if skip else meta.request_top_ks,
        None if skip else meta.request_top_ps,
    )
    tokens = draft()
    torch.testing.assert_close(meta.draft_probs[meta.batch_slot_ids, 3], expected, rtol=0, atol=0)
    assert (expected.gather(1, tokens.long()[:, None]) > 0).all()
    if skip:
        torch.testing.assert_close(
            expected,
            torch.softmax(logits / meta.request_temperatures[:, None], dim=-1),
            rtol=1e-5,
            atol=1e-7,
        )
        assert (expected > 0).all()
    else:
        assert (expected == 0).any()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = draft()
    for offset in (19, 37):
        meta.request_offsets.fill_(offset)
        graph.replay()
        torch.cuda.synchronize()
        eager = draft()
        assert torch.equal(captured, eager)
        torch.testing.assert_close(
            meta.draft_probs[meta.batch_slot_ids, 3], expected, rtol=0, atol=0
        )
    assert torch.equal(meta.request_top_ps, top_ps)
    assert (meta.draft_probs[:, :3] == -1).all()
    assert (meta.draft_probs[1::2] == -1).all()


@CUDA
@pytest.mark.parametrize("skip", [False, True])
def test_cuda_actual_rs_preserves_target_distribution(skip) -> None:
    batch, vocab = 32768, 128
    meta = _metadata(batch=batch, vocab=vocab, steps=1, device="cuda", skip=skip)
    worker = _worker()
    draft_logits = torch.zeros(batch, vocab, device="cuda")
    tokens = spec.SpecWorkerBase.advanced_sample_draft(
        worker, draft_logits, meta, batch, draft_step=0
    )
    q = meta.draft_probs[meta.batch_slot_ids, :1].contiguous()
    # Target has a genuinely truncated support, unlike the unfiltered draft.
    target_logits = torch.linspace(-4, 4, vocab, device="cuda").repeat(batch * 2, 1)
    meta.temperatures = meta.request_temperatures.repeat_interleave(2)
    meta.top_ks = meta.request_top_ks.repeat_interleave(2)
    meta.top_ps = meta.request_top_ps.repeat_interleave(2)
    p = spec.compute_probs_from_logits(target_logits, meta.temperatures, meta.top_ks, meta.top_ps)
    accepted, counts = spec.SpecWorkerBase._sample_and_accept_draft_tokens_rejection(
        worker, target_logits, tokens[:, None], q, 0, batch, meta
    )
    tail = p[::2].gather(1, tokens.long()[:, None]).squeeze(1) == 0
    assert tail.any()
    assert (counts[tail] == 1).all()
    first = accepted[:, 0].long()
    frequencies = torch.bincount(first, minlength=vocab).double() / batch
    expected = p[0].double()
    assert (frequencies[expected == 0] == 0).all()
    tolerance = 6 * torch.sqrt(expected * (1 - expected) / batch) + 2 / batch
    assert (torch.abs(frequencies - expected) <= tolerance).all()
    assert torch.isfinite(q).all() and torch.isfinite(p).all()


@CUDA
@pytest.mark.parametrize("skip", [False, True])
def test_cuda_mixed_greedy_proposals_preserve_temperature(skip: bool) -> None:
    from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import GREEDY_TEMPERATURE_THRESHOLD

    meta = _metadata(batch=2, vocab=128, steps=1, device="cuda", skip=skip)
    # Admission normalizes greedy rows (temperature=0 or top_k=1) to this sentinel.
    meta.request_temperatures[0] = GREEDY_TEMPERATURE_THRESHOLD / 10
    meta.request_top_ks[0] = 1
    temperatures = meta.request_temperatures.clone()
    logits = torch.linspace(-3, 3, 128, device="cuda").repeat(2, 1)
    tokens = spec.SpecWorkerBase.advanced_sample_draft(_worker(), logits, meta, 2, draft_step=0)
    proposal = meta.draft_probs[meta.batch_slot_ids, 0]
    assert tokens[0].item() == logits[0].argmax().item()
    assert proposal[0, tokens[0]].item() == 1.0
    assert torch.count_nonzero(proposal[0]).item() == 1
    torch.testing.assert_close(meta.request_temperatures, temperatures, rtol=0, atol=0)
    if skip:
        torch.testing.assert_close(
            proposal[1], torch.softmax(logits[1] / temperatures[1], dim=-1), rtol=1e-5, atol=1e-7
        )


@CUDA
@pytest.mark.parametrize("skip", [False, True])
def test_cuda_token_only_sampler_graph_without_rejection(skip: bool) -> None:
    meta = _metadata(batch=8, vocab=128, device="cuda", skip=skip)
    meta.use_rejection_sampling = False
    meta.draft_probs = None
    meta.batch_slot_ids = None
    logits = torch.linspace(-3, 3, 128, device="cuda").repeat(8, 1)
    worker = _worker()

    def draft():
        return spec.SpecWorkerBase.advanced_sample_draft(worker, logits, meta, 8, draft_step=0)

    with torch.cuda.stream(torch.cuda.Stream()):
        for _ in range(3):
            draft()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = draft()
    for offset in (19, 37):
        meta.request_offsets.fill_(offset)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, draft())
    assert meta.draft_probs is None


@torch.inference_mode()
def _run_adp_draft_transitions(skip: bool, rejection: bool) -> int:
    from mpi4py import MPI

    from tensorrt_llm._torch.distributed.communicator import MPIDist
    from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3MTPHead
    from tensorrt_llm._torch.modules.embedding import LMHead
    from tensorrt_llm._torch.pyexecutor.sampler.ops.vanilla import GREEDY_TEMPERATURE_THRESHOLD
    from tensorrt_llm.mapping import Mapping

    rank = MPI.COMM_WORLD.Get_rank()
    assert MPI.COMM_WORLD.Get_size() == 2 and rank in (0, 1)
    torch.cuda.set_device(rank)
    mapping = Mapping(
        world_size=2,
        tp_size=2,
        rank=rank,
        enable_attention_dp=True,
        enable_lm_head_tp_in_adp=True,
    )
    batch, vocab = 2, 128
    meta = _real_metadata(skip, batch=batch)
    meta.spec_dec_mode = spec.SpeculativeDecodingMode.MTP
    buffers = _metadata(batch=batch, vocab=vocab, steps=2, device="cuda", skip=skip)
    for name, value in vars(buffers).items():
        setattr(meta, name, value)
    meta.use_rejection_sampling = rejection
    if not rejection:
        meta.draft_probs = None
    worker = _worker()
    worker.mapping = mapping
    for name in (
        "advanced_sample_draft",
        "greedy_sample_draft_with_tp_gather",
        "_draft_logits_are_sharded",
        "maybe_gather_sharded_draft_logits",
    ):
        setattr(worker, name, MethodType(getattr(spec.SpecWorkerBase, name), worker))
    head = LMHead(vocab, vocab, dtype=torch.float32, mapping=mapping).cuda()
    head.weight.copy_(torch.eye(vocab, device="cuda"))
    config = SimpleNamespace(
        mapping=mapping,
        pretrained_config=SimpleNamespace(
            hidden_size=vocab, rms_norm_eps=1e-6, torch_dtype=torch.float32
        ),
    )
    layer = SimpleNamespace(shared_head=DeepseekV3MTPHead(config).cuda())
    model = SimpleNamespace(lm_head=head)
    indices = torch.tensor([1, 3], device="cuda")
    hidden = torch.zeros(4, vocab, device="cuda")
    expected_ids = torch.tensor([80 + rank, 100 + rank], device="cuda", dtype=torch.int32)
    hidden[indices, expected_ids.long()] = 8
    attn = SimpleNamespace(seq_lens_cuda=torch.tensor([2, 2], device="cuda", dtype=torch.int32))
    engine = SimpleNamespace(mapping=mapping, dist=MPIDist(mapping))
    for flags in ((True, True), (True, False), (False, True), (False, False), (True, True)):
        local_greedy = flags[rank]
        meta.is_all_greedy_sample = local_greedy
        meta.group_all_greedy_sample = None
        meta.request_temperatures.fill_(GREEDY_TEMPERATURE_THRESHOLD / 10 if local_greedy else 0.8)
        meta.request_top_ks.fill_(1 if local_greedy else 32)
        PyTorchModelEngine._sync_group_all_greedy_sample(engine, meta)
        assert meta.is_all_greedy_sample == all(flags)
        logits = MTPWorker._compute_draft_logits(worker, layer, hidden, model, attn, meta, indices)
        torch.testing.assert_close(logits, hidden[indices], rtol=0, atol=0)
        tokens = spec.SpecWorkerBase.sample_draft_tokens(worker, logits, meta, batch, draft_step=0)
        input_ids = torch.zeros(4, dtype=torch.int32, device="cuda")
        input_ids[indices] = tokens
        assert tokens.shape == (batch,) and ((tokens >= 0) & (tokens < vocab)).all()
        if local_greedy:
            torch.testing.assert_close(tokens, expected_ids, rtol=0, atol=0)
        if rejection and not all(flags):
            expected_q = spec.compute_probs_from_logits(
                logits,
                meta.request_temperatures,
                None if skip else meta.request_top_ks,
                None if skip else meta.request_top_ps,
            )
            torch.testing.assert_close(meta.draft_probs[meta.batch_slot_ids, 0], expected_q)
        torch.cuda.synchronize()
    return rank


@pytest.mark.threadleak(enabled=False)
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
@pytest.mark.parametrize("skip,rejection", [(True, False), (True, True), (False, True)])
def test_vanilla_mtp_adp_draft_transitions_tp2(skip: bool, rejection: bool) -> None:
    import sys

    from mpi4py.futures import MPIPoolExecutor

    # Propagate pytest's module path instead of changing MPI's global serializer.
    with MPIPoolExecutor(max_workers=2, path=sys.path) as pool:
        futures = [pool.submit(_run_adp_draft_transitions, skip, rejection) for _ in range(2)]
        assert sorted(future.result(timeout=300) for future in futures) == [0, 1]
