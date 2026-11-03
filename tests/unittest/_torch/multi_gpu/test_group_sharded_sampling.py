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
"""Attention-DP verify rows sampled from a node-local group's shards of the replicated BF16 LM head
(``enable_lm_head_tp_in_adp``, groups of ``min(gpus_per_node, tp_size)`` ranks): greedy steps accept
exactly what the replicated head accepts, and a temperature-only row takes the inverse-CDF draw of
``softmax(logits / T)`` at its Philox uniform, so the draws follow the softmax.
"""

import pickle
import sys
import traceback
import types

import cloudpickle
import numpy as np
import pytest
import torch
from mpi4py import MPI

import tensorrt_llm
from tensorrt_llm._torch.models.modeling_deepseekv4 import DeepseekV4LogitsProcessor
from tensorrt_llm._torch.models.modeling_speculative import SpecDecOneEngineForCausalLM
from tensorrt_llm._torch.modules.embedding import LMHead
from tensorrt_llm._torch.modules.linear import TensorParallelMode
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.pyexecutor.llm_request import LlmRequestState
from tensorrt_llm._torch.speculative.dspark import DSv4DSparkWorker
from tensorrt_llm._torch.speculative.interface import SpecMetadata, SpeculativeDecodingMode
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

pytestmark = pytest.mark.threadleak(enabled=False)

HIDDEN = 1024
VOCAB = 129280
DRAFT_LEN = 5
STEPS = 60
GPUS_PER_NODE = 4


def _hidden(rows: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(rows, HIDDEN, generator=gen, device="cuda").bfloat16()


def _philox_first_word(key: int, counters) -> int:
    """Philox-4x32-10's first output word for a 64-bit key and four uint32 counters."""
    mask = 0xFFFFFFFF
    k0, k1 = key & mask, (key >> 32) & mask
    c0, c1, c2, c3 = counters
    for _ in range(10):
        p0, p2 = c0 * 0xD2511F53, c2 * 0xCD9E8D57
        c0, c1, c2, c3 = (
            ((p2 >> 32) ^ c1 ^ k0) & mask,
            p2 & mask,
            ((p0 >> 32) ^ c3 ^ k1) & mask,
            p0 & mask,
        )
        k0, k1 = (k0 + 0x9E3779B9) & mask, (k1 + 0xBB67AE85) & mask
    return c0


def _inverse_cdf(logits, temperatures, seeds, offsets):
    """Per row the token whose softmax(logits / T) interval holds u = Philox(seed, offset), in
    float64, and the distance from u to the nearer edge of that interval."""
    logits = logits.float().cpu().numpy().astype(np.float64)
    tokens, margins = [], []
    for row, (temperature, seed, offset) in enumerate(zip(temperatures, seeds, offsets)):
        bits = _philox_first_word(
            seed & 0xFFFFFFFFFFFFFFFF, (offset & 0xFFFFFFFF, offset >> 32, 0, 0)
        )
        u = (2 * (bits >> 9) + 1) / 2.0**24
        z = logits[row] / temperature
        p = np.exp(z - z.max())
        cdf = np.cumsum(p / p.sum())
        token = min(int(np.searchsorted(cdf, u, side="right")), cdf.size - 1)
        low = cdf[token - 1] if token else 0.0
        tokens.append(token)
        margins.append(min(u - low, cdf[token] - u))
    return tokens, margins


def _request(slot, temperature, seed):
    return types.SimpleNamespace(
        sampling_config=types.SimpleNamespace(
            temperature=temperature, top_k=0, top_p=1.0, min_p=0.0, seed=seed
        ),
        state=LlmRequestState.GENERATION_IN_PROGRESS,
        py_seq_slot=slot,
        py_request_id=100 + slot,
        py_spec_sampling_params=None,
    )


@torch.inference_mode()
def _run(world: int, group_size: int) -> bool:
    rank = tensorrt_llm.mpi_rank()
    torch.cuda.set_device(rank % torch.cuda.device_count())
    try:

        def mapping(lm_head_tp: bool) -> Mapping:
            return Mapping(
                world_size=world,
                rank=rank,
                gpus_per_node=GPUS_PER_NODE,
                tp_size=world,
                enable_attention_dp=True,
                enable_lm_head_tp_in_adp=lm_head_tp,
            )

        # The same head on every rank, as attention DP replicates it.
        gen = torch.Generator(device="cuda").manual_seed(1234)
        weight = (torch.randn(VOCAB, HIDDEN, generator=gen, device="cuda") * 0.05).bfloat16()
        # Exact ties across shards: the smaller index must win.
        weight[VOCAB - 3] = weight[7]

        def lm_head(lm_head_tp: bool) -> LMHead:
            head = LMHead(
                VOCAB,
                HIDDEN,
                dtype=torch.bfloat16,
                mapping=mapping(lm_head_tp),
                tensor_parallel_mode=TensorParallelMode.COLUMN,
                gather_output=True,
            )
            head.load_weights([dict(weight=weight)])
            return head.cuda()

        norm = RMSNorm(hidden_size=HIDDEN, eps=1e-6, dtype=torch.bfloat16).cuda()
        norm.weight.copy_(torch.rand(HIDDEN, generator=gen, device="cuda") * 4)
        processor = DeepseekV4LogitsProcessor(
            types.SimpleNamespace(
                pretrained_config=types.SimpleNamespace(hc_mult=4, hidden_size=HIDDEN),
                mapping=mapping(False),
            ),
            None,
            norm,
        )

        def model(grouped: bool):
            # enable_lm_head_tp_in_adp turns the group head on; the mapping sets its size.
            head = lm_head(grouped)
            worker = DSv4DSparkWorker(
                types.SimpleNamespace(
                    max_draft_len=DRAFT_LEN,
                    spec_dec_mode=SpeculativeDecodingMode.DSPARK,
                    use_rejection_sampling=False,
                    enable_penalty=False,
                    acceptance_rate_window_size=None,
                    acceptance_rate_threshold=None,
                ),
                head.mapping,
            )
            worker.prepare_vocab_sharded_sampling(head)
            group = worker.group_verify
            assert (group.group_size if group else None) == (group_size if grouped else None)
            return types.SimpleNamespace(
                logits_processor=processor, lm_head=head, spec_worker=worker
            )

        grouped, plain = model(True), model(False)
        head = plain.lm_head

        def accept(model, hidden, draft, spec_metadata=None):
            num_gens, rows = draft.shape[0], hidden.shape[0]
            attn_metadata = types.SimpleNamespace(
                num_seqs=num_gens, num_contexts=0, all_rank_num_tokens=[rows] * world
            )
            if spec_metadata is None:
                spec_metadata = types.SimpleNamespace(
                    runtime_draft_len=DRAFT_LEN,
                    is_all_greedy_sample=True,
                    is_all_temperature_only=False,
                    use_rejection_sampling=False,
                    is_ragged_verify=False,
                    enable_penalty=False,
                )
            spec_metadata.draft_tokens = draft.reshape(-1)
            spec_metadata.gather_ids = torch.arange(rows, device="cuda")
            spec_metadata.is_cuda_graph = True
            logits, vocab_sharded = SpecDecOneEngineForCausalLM._verify_logits(
                model, hidden, attn_metadata, spec_metadata
            )
            assert vocab_sharded == (model is grouped)
            return model.spec_worker._fused_accept(
                logits, attn_metadata, spec_metadata, vocab_sharded=vocab_sharded
            )

        def inputs(num_gens, seed):
            hidden = _hidden(num_gens * (DRAFT_LEN + 1), seed + 7919 * rank)
            target = torch.argmax(head(norm(hidden)), -1).view(num_gens, DRAFT_LEN + 1)
            draft = target[:, :DRAFT_LEN].to(torch.int32)
            # The last request rejects a seed-dependent draft position.
            draft[-1, seed % (DRAFT_LEN + 1) :] += 1
            return hidden, draft

        # Greedy: every member runs every call with the same row count before any assertion.
        results = []
        for step in range(STEPS):
            num_gens = (1, 4, 25)[step % 3]
            hidden, draft = inputs(num_gens, 1000 + step)
            if step == 1:
                hidden[0] = weight[7]
                hidden[2] = float("nan")
            got, want = accept(grouped, hidden, draft), accept(plain, hidden, draft)
            results.extend(torch.equal(g, w) for g, w in zip(got, want))

        # Replayed from captured graphs with new inputs.
        graphs = {}
        for num_gens in (1, 25):
            hidden, draft = inputs(num_gens, 3000 + num_gens)
            accept(grouped, hidden, draft)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = accept(grouped, hidden, draft)
            graphs[num_gens] = (graph, hidden, draft, out)
        for step in range(STEPS):
            graph, hidden, draft, got = graphs[(1, 25)[step % 2]]
            new_hidden, new_draft = inputs(draft.shape[0], 4000 + step)
            hidden.copy_(new_hidden)
            draft.copy_(new_draft)
            graph.replay()
            want = accept(plain, new_hidden, new_draft)
            results.extend(torch.equal(g, w) for g, w in zip(got, want))
        torch.cuda.synchronize()
        # Every member runs every check before any assertion: a member that stopped would leave the
        # others waiting on it.
        failures = (
            [] if all(results) else [f"greedy: {results.count(False)} of {len(results)} differ"]
        )
        failures += _check_temperature_draws(grouped, head, norm, accept)
        assert not failures, f"rank {rank}: {failures}"
    except Exception:
        traceback.print_exc()
        raise
    return True


def _check_temperature_draws(grouped, head, norm, accept):
    """Temperature-only steps through the model path (eager and graph-replayed, unseeded rows at
    temperatures 1.0 and 0.7 and offsets above 2**62): each row's token is the inverse-CDF draw at
    its Philox uniform."""
    rank = tensorrt_llm.mpi_rank()

    def metadata():
        spec_metadata = SpecMetadata(
            max_num_requests=2,
            max_draft_len=DRAFT_LEN,
            max_total_draft_tokens=DRAFT_LEN,
            spec_dec_mode=SpeculativeDecodingMode.DSPARK,
        )
        spec_metadata.runtime_draft_len = DRAFT_LEN
        spec_metadata.samples_from_vocab_shards = True
        spec_metadata.populate_sampling_params_for_one_model(
            [_request(0, 1.0, None), _request(1, 0.7, None)]
        )
        assert spec_metadata.is_all_temperature_only
        return spec_metadata

    checked, total, results = 0, 0, []

    def compare(accepted, hidden, spec_metadata):
        nonlocal checked, total
        rows = hidden.shape[0]
        want, margins = _inverse_cdf(
            head(norm(hidden)),
            spec_metadata.temperatures[:rows].tolist(),
            spec_metadata.seeds[:rows].tolist(),
            spec_metadata.offsets[:rows].tolist(),
        )
        # The draft proposes token 0 everywhere, so the first target token is accepted[:, 0].
        got = accepted[0][:, 0].tolist()
        for request, token in enumerate(got):
            row = request * (DRAFT_LEN + 1)
            total += 1
            # The device sums are FP32: a u within their rounding of an interval edge is undecided.
            if margins[row] > 2e-5:
                results.append(token == want[row])
                checked += 1

    draft = torch.zeros(2, DRAFT_LEN, dtype=torch.int32, device="cuda")
    for step in range(STEPS):
        spec_metadata = metadata()
        hidden = _hidden(2 * (DRAFT_LEN + 1), 5000 + step + 7919 * rank)
        compare(accept(grouped, hidden, draft, spec_metadata), hidden, spec_metadata)

    spec_metadata = metadata()
    hidden = _hidden(2 * (DRAFT_LEN + 1), 6000 + 7919 * rank)
    accept(grouped, hidden, draft, spec_metadata)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = accept(grouped, hidden, draft, spec_metadata)
    for step in range(STEPS):
        hidden.copy_(_hidden(hidden.shape[0], 7000 + step + 7919 * rank))
        spec_metadata.offsets.add_(1 << 20)
        graph.replay()
        compare(out, hidden, spec_metadata)
    torch.cuda.synchronize()
    failures = [] if all(results) else [f"draws: {results.count(False)} of {len(results)} differ"]
    return failures + (
        [] if checked > 0.8 * total else [f"draws: only {checked} of {total} decided"]
    )


@pytest.mark.skipif(torch.cuda.device_count() < GPUS_PER_NODE, reason="needs a 4-GPU node")
@pytest.mark.parametrize(
    "mpi_pool_executor, group_size",
    [(4, 4)],
    indirect=["mpi_pool_executor"],
    ids=["tp4"],
)
def test_group_sharded_verify_matches_replicated_head(mpi_pool_executor, group_size):
    world = mpi_pool_executor.num_workers
    results = mpi_pool_executor.map(_run, *zip(*[(world, group_size)] * world))
    for result in results:
        assert result is True


if __name__ == "__main__":
    world = tensorrt_llm.mpi_world_size()
    assert _run(world, min(GPUS_PER_NODE, world))
    tensorrt_llm.mpi_barrier()
    if tensorrt_llm.mpi_rank() == 0:
        print(f"GROUP_SHARDED_OK ranks={tensorrt_llm.mpi_world_size()}", flush=True)
