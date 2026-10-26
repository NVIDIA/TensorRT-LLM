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
"""Target sampling from a column-parallel LM head's shards equals sampling its gather: the
argmax for greedy rows, the same Gumbel-max draw for temperature-only rows. Steps whose requests
post-process the logits keep the gathered full-vocabulary logits.

Also an MPI program for groups across nodes (``srun -n 8 python3 <this file>``), where the
exchange crosses trays over MNNVL fabric memory.
"""

import math
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
from tensorrt_llm._torch.speculative.sharded_sampling import VocabShardedSampler
from tensorrt_llm._utils import mpi_allgather
from tensorrt_llm.mapping import Mapping

cloudpickle.register_pickle_by_value(sys.modules[__name__])
MPI.pickle.__init__(cloudpickle.dumps, cloudpickle.loads, pickle.HIGHEST_PROTOCOL)

pytestmark = pytest.mark.threadleak(enabled=False)

HIDDEN = 1024
VOCAB = 129280
DRAFT_LEN = 5
STEPS = 200


def _hidden(rows: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randn(rows, HIDDEN, generator=gen, device="cuda").bfloat16()


def _head(mapping: Mapping, vocab: int, gen: torch.Generator) -> tuple[LMHead, torch.Tensor]:
    weight = (torch.randn(vocab, HIDDEN, generator=gen, device="cuda") * 0.05).bfloat16()
    # Exact ties across shards: the smaller global index must win.
    weight[vocab - 3] = weight[7]
    head = LMHead(
        vocab,
        HIDDEN,
        dtype=torch.bfloat16,
        mapping=mapping,
        tensor_parallel_mode=TensorParallelMode.COLUMN,
        gather_output=True,
    )
    head.load_weights([dict(weight=weight)])
    return head.cuda(), weight


def _philox_first_word(key, counters):
    """Philox-4x32-10's first output word for a 64-bit key and four uint32 counter arrays."""
    mask = np.uint64(0xFFFFFFFF)
    k0, k1 = np.uint64(key & 0xFFFFFFFF), np.uint64(key >> 32)
    c0, c1, c2, c3 = (np.asarray(c, dtype=np.uint64) for c in counters)
    for _ in range(10):
        p0, p2 = c0 * np.uint64(0xD2511F53), c2 * np.uint64(0xCD9E8D57)
        c0, c1, c2, c3 = (
            ((p2 >> np.uint64(32)) ^ c1 ^ k0) & mask,
            p2 & mask,
            ((p0 >> np.uint64(32)) ^ c3 ^ k1) & mask,
            p0 & mask,
        )
        k0, k1 = (k0 + np.uint64(0x9E3779B9)) & mask, (k1 + np.uint64(0xBB67AE85)) & mask
    return c0


def _gumbel_max(logits, temperatures, seeds, offsets):
    """Full-vocabulary Gumbel-max reference in float64 with the sampler's keyed noise: per row the
    winning token and the margin to the runner-up (rows within FP32 rounding are not decided)."""
    logits = logits.float().cpu().numpy().astype(np.float64)
    vocab = logits.shape[1]
    tokens, margins = [], []
    for row, (temperature, seed, offset) in enumerate(zip(temperatures, seeds, offsets)):
        bits = _philox_first_word(
            int(seed) & 0xFFFFFFFFFFFFFFFF,
            (
                np.arange(vocab),
                np.full(vocab, offset & 0xFFFFFFFF),
                np.full(vocab, offset >> 32),
                0,
            ),
        )
        u = (2.0 * (bits >> np.uint64(9)).astype(np.float64) + 1.0) / 2.0**24
        z = logits[row] / temperature - np.log(-np.log(u))
        top2 = np.partition(z, -2)[-2:]
        tokens.append(int(np.argmax(z)))
        margins.append(float(top2[1] - top2[0]))
    return tokens, margins


@torch.inference_mode()
def _run(tp_size: int) -> bool:
    rank = tensorrt_llm.mpi_rank()
    gpus_per_node = min(torch.cuda.device_count(), tp_size)
    torch.cuda.set_device(rank % gpus_per_node)
    try:
        mapping = Mapping(
            world_size=tp_size, tp_size=tp_size, rank=rank, gpus_per_node=gpus_per_node
        )
        gen = torch.Generator(device="cuda").manual_seed(1234)
        # Every rank runs every launch before any assertion.
        results = []

        # The kernel: row counts change between launches; ties and NaN follow torch.argmax. Not a
        # multiple of the group size, so the last shard is padded.
        head, weight = _head(mapping, VOCAB - 37, gen)
        sampler = VocabShardedSampler.create(mapping, head.out_features, head.num_embeddings)
        assert sampler is not None

        def both(hidden):
            return sampler.sample(head.local_logits(hidden)), torch.argmax(head(hidden), -1)

        for step, rows in enumerate((6, 12, 1, 6, 12, 256, 5, 6)):
            hidden = _hidden(rows, step)
            hidden[0] = weight[7]
            if rows > 2:
                hidden[2] = float("nan")
            got, want = both(hidden)
            results.append(torch.equal(got, want))

        # The model contract: DeepSeek-V4.1's target logits (final norm, then the head) through
        # the worker's fused acceptance, on this rank's columns vs on the gathered logits.
        head, _ = _head(mapping, VOCAB, gen)
        norm = RMSNorm(hidden_size=HIDDEN, eps=1e-6, dtype=torch.bfloat16).cuda()
        # Not uniform: a uniform norm weight only rescales a row and keeps its argmax.
        norm.weight.copy_(torch.rand(HIDDEN, generator=gen, device="cuda") * 4)
        processor = DeepseekV4LogitsProcessor(
            types.SimpleNamespace(
                pretrained_config=types.SimpleNamespace(hc_mult=4, hidden_size=HIDDEN),
                mapping=mapping,
            ),
            None,
            norm,
        )

        def model(sharded: bool):
            worker = DSv4DSparkWorker(
                types.SimpleNamespace(
                    max_draft_len=DRAFT_LEN,
                    spec_dec_mode=SpeculativeDecodingMode.DSPARK,
                    use_rejection_sampling=False,
                    enable_penalty=False,
                ),
                mapping,
            )
            if sharded:
                worker.prepare_vocab_sharded_sampling(head)
            return types.SimpleNamespace(
                logits_processor=processor, lm_head=head, spec_worker=worker
            )

        sharded, plain = model(True), model(False)

        def accept(model, hidden, draft):
            num_gens = draft.shape[0]
            attn_metadata = types.SimpleNamespace(num_seqs=num_gens, num_contexts=0)
            spec_metadata = types.SimpleNamespace(
                runtime_draft_len=DRAFT_LEN,
                draft_tokens=draft.reshape(-1),
                gather_ids=torch.arange(hidden.shape[0], device="cuda"),
                is_all_greedy_sample=True,
                is_all_temperature_only=False,
                use_rejection_sampling=False,
                is_ragged_verify=False,
                enable_penalty=False,
                is_cuda_graph=False,
            )
            logits, vocab_sharded = SpecDecOneEngineForCausalLM._verify_logits(
                model, hidden, attn_metadata, spec_metadata
            )
            assert vocab_sharded == (model is sharded)
            return model.spec_worker._fused_accept(
                logits, attn_metadata, spec_metadata, vocab_sharded=vocab_sharded
            )

        def inputs(num_gens, seed):
            hidden = _hidden(num_gens * (DRAFT_LEN + 1), seed)
            target = torch.argmax(head(norm(hidden)), -1).view(num_gens, DRAFT_LEN + 1)
            draft = target[:, :DRAFT_LEN].to(torch.int32)
            # The last request rejects a seed-dependent draft position.
            draft[-1, seed % (DRAFT_LEN + 1) :] += 1
            return hidden, draft

        for step in range(STEPS):
            hidden, draft = inputs(1 + step % 2, 1000 + step)
            got, want = accept(sharded, hidden, draft), accept(plain, hidden, draft)
            results.extend(torch.equal(g, w) for g, w in zip(got, want))

        # Replayed from captured graphs with new inputs, 6 and 12 rows.
        graphs = {}
        for num_gens in (1, 2):
            hidden, draft = inputs(num_gens, 3000 + num_gens)
            accept(sharded, hidden, draft)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = accept(sharded, hidden, draft)
            graphs[num_gens] = (graph, hidden, draft, out)
        for step in range(STEPS):
            graph, hidden, draft, got = graphs[1 + step % 2]
            new_hidden, new_draft = inputs(draft.shape[0], 4000 + step)
            hidden.copy_(new_hidden)
            draft.copy_(new_draft)
            graph.replay()
            want = accept(plain, new_hidden, new_draft)
            results.extend(torch.equal(g, w) for g, w in zip(got, want))
        torch.cuda.synchronize()
        assert all(results), f"rank {rank}: {results.count(False)} of {len(results)} checks failed"
        _check_temperature_only(mapping, head, norm, processor, sharded.spec_worker)
        _check_temperature_distribution(mapping)
    except Exception:
        traceback.print_exc()
        raise
    return True


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


def _check_temperature_only(mapping, head, norm, processor, worker):
    """Temperature-only steps: the worker's sharded Gumbel-max draw equals a full-vocabulary
    Gumbel-max reference with the same keyed noise, and every rank takes the same tokens."""
    assert worker.samples_from_vocab_shards
    model = types.SimpleNamespace(logits_processor=processor, lm_head=head, spec_worker=worker)

    def metadata(num_gens):
        spec_metadata = SpecMetadata(
            max_num_requests=2,
            max_draft_len=DRAFT_LEN,
            max_total_draft_tokens=DRAFT_LEN,
            spec_dec_mode=SpeculativeDecodingMode.DSPARK,
        )
        spec_metadata.runtime_draft_len = DRAFT_LEN
        spec_metadata.samples_from_vocab_shards = True
        # AgentX requests send no sampling fields (temperature 1); one seeded row at 0.7.
        requests = [_request(0, 1.0, None), _request(1, 0.7, 1234)][:num_gens]
        spec_metadata.populate_sampling_params_for_one_model(requests)
        assert spec_metadata.is_all_temperature_only
        return spec_metadata

    def accept(spec_metadata, hidden, draft):
        num_gens = draft.shape[0]
        attn_metadata = types.SimpleNamespace(num_seqs=num_gens, num_contexts=0)
        spec_metadata.draft_tokens = draft.reshape(-1)
        spec_metadata.gather_ids = torch.arange(hidden.shape[0], device="cuda")
        logits, vocab_sharded = SpecDecOneEngineForCausalLM._verify_logits(
            model, hidden, attn_metadata, spec_metadata
        )
        assert vocab_sharded
        return worker._fused_accept(logits, attn_metadata, spec_metadata, vocab_sharded=True)

    def reference(spec_metadata, hidden):
        rows = hidden.shape[0]
        return _gumbel_max(
            head(norm(hidden)),
            spec_metadata.temperatures[:rows].tolist(),
            spec_metadata.seeds[:rows].tolist(),
            spec_metadata.offsets[:rows].tolist(),
        )

    checked, results, tokens = 0, [], []

    def compare(accepted, want, margins):
        nonlocal checked
        got = accepted[0].reshape(-1).tolist()
        tokens.append(got)
        for g, w, margin in zip(got, want, margins):
            # The device draw is FP32: rows whose top two are within its rounding are undecided.
            if margin > 1e-3:
                results.append(g == w)
                checked += 1

    for step in range(STEPS // 4):
        num_gens = 1 + step % 2
        spec_metadata = metadata(num_gens)
        hidden = _hidden(num_gens * (DRAFT_LEN + 1), 5000 + step)
        draft = torch.zeros(num_gens, DRAFT_LEN, dtype=torch.int32, device="cuda")
        want, margins = reference(spec_metadata, hidden)
        compare(accept(spec_metadata, hidden, draft), want, margins)

    # Replayed from a captured graph with new inputs and RNG offsets, 6 and 12 rows.
    for num_gens in (1, 2):
        spec_metadata = metadata(num_gens)
        hidden = _hidden(num_gens * (DRAFT_LEN + 1), 6000 + num_gens)
        draft = torch.zeros(num_gens, DRAFT_LEN, dtype=torch.int32, device="cuda")
        accept(spec_metadata, hidden, draft)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = accept(spec_metadata, hidden, draft)
        for step in range(STEPS // 8):
            hidden.copy_(_hidden(hidden.shape[0], 7000 + 10 * num_gens + step))
            spec_metadata.offsets.add_(1 << 20)
            graph.replay()
            want, margins = reference(spec_metadata, hidden)
            compare(out, want, margins)
    torch.cuda.synchronize()
    rank = mapping.tp_rank
    assert all(results), f"rank {rank}: {results.count(False)} of {len(results)} draws differ"
    assert checked > 0.9 * sum(len(t) for t in tokens), f"rank {rank}: only {checked} decided"
    # Every rank of the group takes the same token for every row.
    assert all(peer == tokens for peer in mpi_allgather(tokens)), "ranks disagree"


def _chi_square_p(observed, expected):
    """Upper tail of Pearson's chi-square, bins with expectation below 5 pooled."""
    big = expected >= 5
    obs = np.append(observed[big], observed[~big].sum())
    exp = np.append(expected[big], expected[~big].sum())
    keep = exp > 0
    obs, exp = obs[keep], exp[keep]
    chi2, dof = float(((obs - exp) ** 2 / exp).sum()), obs.size - 1
    # Wilson-Hilferty normal approximation.
    z = ((chi2 / dof) ** (1 / 3) - (1 - 2 / (9 * dof))) / math.sqrt(2 / (9 * dof))
    return 0.5 * math.erfc(z / math.sqrt(2))


def _check_temperature_distribution(mapping):
    """Small vocabulary split over the group: 2**18 rows per temperature drawn through the
    exchange follow softmax(logits / T); -inf entries never win and greedy rows take the argmax."""
    vocab, rows, draws = 96, 256, 1 << 18
    local = vocab // mapping.tp_size
    sampler = VocabShardedSampler.create(mapping, local, vocab)
    gen = torch.Generator().manual_seed(17)
    base = torch.randn(vocab, generator=gen) * 2
    base[torch.randperm(vocab, generator=gen)[:6]] = float("-inf")
    columns = slice(mapping.tp_rank * local, (mapping.tp_rank + 1) * local)
    logits = base[columns].cuda().expand(rows, local).contiguous()
    seeds = torch.full((rows,), 77, dtype=torch.int64, device="cuda")
    for temperature in (0.5, 1.0, 2.0):
        probs = torch.softmax(base.double() / temperature, dim=0).numpy()
        temperatures = torch.full((rows,), temperature, device="cuda")
        counts = np.zeros(vocab)
        for chunk in range(draws // rows):
            offsets = torch.arange(rows, dtype=torch.int64, device="cuda") + chunk * rows
            tokens = sampler.sample(logits, temperatures, seeds, offsets)
            counts += np.bincount(tokens.cpu().numpy(), minlength=vocab)
        assert counts[probs == 0].sum() == 0, temperature
        assert _chi_square_p(counts, probs * draws) > 1e-4, temperature
    greedy = sampler.sample(logits, torch.zeros(rows, device="cuda"), seeds, offsets)
    assert torch.equal(greedy, torch.full_like(greedy, int(torch.argmax(base))))


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="needs 4 GPUs")
@pytest.mark.parametrize("mpi_pool_executor", [4], indirect=True)
def test_sharded_sampling_matches_sampling_the_gathered_logits(mpi_pool_executor):
    tp_size = mpi_pool_executor.num_workers
    results = mpi_pool_executor.map(_run, *zip(*[(tp_size,)] * tp_size))
    for result in results:
        assert result is True


if __name__ == "__main__":
    assert _run(tensorrt_llm.mpi_world_size())
    tensorrt_llm.mpi_barrier()
    if tensorrt_llm.mpi_rank() == 0:
        print(f"SHARDED_SAMPLING_OK ranks={tensorrt_llm.mpi_world_size()}", flush=True)
