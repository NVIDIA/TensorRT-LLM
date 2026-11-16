# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The steady glue: a steady decode step's staging launch also doing the graph's Engram pre-replay refresh
(for joining requests, with their history seeds) writes what the staging launch plus the replay's own hook
write, bit for bit."""

import types

import pytest
import torch
from _torch.helpers import create_mock_cuda_graph_runner

from tensorrt_llm._torch.modules.engram import EngramConfig, EngramHashProvider
from tensorrt_llm._torch.modules.engram import engram as engram_module
from tensorrt_llm._torch.pyexecutor.cuda_graph_runner import KeyType
from tensorrt_llm._torch.pyexecutor.steady_spec_prepare import SteadyMtpPrepare
from tensorrt_llm._utils import is_blackwell_datacenter

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_blackwell_datacenter(),
    reason="The steady glue's one-launch Engram refresh runs on SM100 / SM103",
)

SLOTS, MAX_SEQ = 8, 4096


def _provider(monkeypatch, ngram, heads, layers):
    bound = torch.iinfo(torch.int64).max // 256
    mapping = types.SimpleNamespace(
        compressed_tokenizer=types.SimpleNamespace(lookup_table=torch.arange(256) // 2),
        pad_id=-7,
        layer_multipliers={
            layer: torch.tensor([bound - 2 * (layer + i) for i in range(ngram)]) for layer in layers
        },
        vocab_size_across_layers={
            layer: [
                [101 + 2 * (order * heads + head) for head in range(heads)]
                for order in range(ngram - 1)
            ]
            for layer in layers
        },
    )
    monkeypatch.setattr(engram_module, "NgramHashMapping", lambda **kwargs: mapping)
    config = EngramConfig(layer_ids=list(layers), max_ngram_size=ngram, n_head_per_ngram=heads)
    return EngramHashProvider(config)


class _Side:
    """One engine's steady decode path: graph runner with the Engram hook, provider, steady prepare."""

    def __init__(self, monkeypatch, engram, glue, b, t, slots, bases):
        n = b * t
        self.provider = _provider(monkeypatch, *engram)
        self.runner = create_mock_cuda_graph_runner(n, max_num_tokens=n)
        self.key = KeyType(batch_size=b, draft_len=t - 1)
        self.hook_calls = 0
        i32 = dict(dtype=torch.int32, device="cuda")
        self.buffers = types.SimpleNamespace(
            input_ids_cuda=torch.zeros((n,), **i32),
            draft_tokens_cuda=torch.zeros((n,), **i32),
            previous_batch_indices_cuda=torch.zeros((n,), **i32),
            previous_pos_indices_cuda=torch.zeros((n,), **i32),
            previous_pos_id_offsets_cuda=torch.zeros((n,), **i32),
            previous_kv_lens_offsets_cuda=torch.zeros((b,), **i32),
            position_ids_cuda=torch.zeros((n,), **i32),
            mrope_position_ids_cuda=None,
            gather_ids_cuda=torch.zeros((n,), **i32),
        )
        self.requests = [
            types.SimpleNamespace(
                py_request_id=100 + r,
                py_seq_slot=slot,
                py_batch_idx=slot,
                is_dummy=False,
                py_multimodal_data=None,
                py_needs_onehot_draft_probs=False,
                max_beam_num_tokens=base + 1,
                py_num_compressed_tokens=0,
            )
            for r, (slot, base) in enumerate(zip(slots, bases))
        ]
        self.attn = types.SimpleNamespace(
            is_cuda_graph=True,
            num_ctx_tokens=0,
            num_tokens=n,
            request_ids=[request.py_request_id for request in self.requests],
            seq_lens=torch.full((b,), t, dtype=torch.int32),
            seq_lens_kv=torch.full((b,), t, dtype=torch.int32),
            kv_cache_manager=types.SimpleNamespace(max_seq_len=MAX_SEQ),
        )

        def refresh(key, num_tokens, static_tensors, current_inputs):
            # DeepseekV41ForCausalLM's hook (_compute_engram_hashes with refresh=True).
            self.hook_calls += 1
            attn = current_inputs["attn_metadata"]
            self.provider.refresh_captured_hashes(
                static_tensors["input_ids"][:num_tokens].view(-1)[: attn.num_tokens],
                position_ids=static_tensors["position_ids"][..., :num_tokens][
                    ..., : attn.num_tokens
                ],
                request_ids=attn.request_ids,
                seq_lens_host=attn.seq_lens,
                max_seq_len=attn.kv_cache_manager.max_seq_len,
                padded_num_tokens=num_tokens,
            )

        refresh.engram_provider = self.provider
        self.runner.register_pre_replay_hook(refresh)
        self.steady = SteadyMtpPrepare(
            max_batch_size=b,
            max_num_tokens=n,
            max_tokens_per_step=t,
            use_mrope=False,
            has_static_inputs=True,
            graph_runner=self.runner if glue else None,
            steady_decode_prepare=True,
        )
        assert (self.steady.glue_provider is not None) == glue

    def full_pass(self, ids, positions, seeds):
        """The recorded full pass: history seeds, the provider's own refresh, graph capture, record."""
        for request, seed in zip(self.requests, seeds):
            self.provider.queue_history_seed(request.py_request_id, *seed)
        n = ids.numel()
        hashes = self.provider.compute_hashes(
            ids,
            position_ids=positions,
            request_ids=self.attn.request_ids,
            seq_lens_host=self.attn.seq_lens,
            max_seq_len=MAX_SEQ,
            padded_num_tokens=n,
        )
        static = self.runner.shared_static_tensors
        inputs = dict(
            attn_metadata=self.attn,
            input_ids=static["input_ids"][:n],
            position_ids=static["position_ids"][:, :n],
        )
        layers = sorted(hashes)
        self.runner.capture(
            self.key,
            lambda inputs: torch.cat(
                [inputs["input_ids"].long()] + [hashes[layer].view(-1) for layer in layers]
            ),
            inputs,
        )
        self.steady.record_batch(
            requests=self.requests,
            prompt_lengths=[1] * len(self.requests),
            sequence_lengths=[n // len(self.requests)] * len(self.requests),
            tokens_per_step=n // len(self.requests),
            runtime_draft_len=n // len(self.requests) - 1,
            num_extra_kv_tokens=0,
            use_full_generation_page_table=False,
            attn_metadata=self.attn,
            spec_metadata=object(),
            static_tensors=static,
        )

    def admit(self, slots, bases, seeds, tokens):
        """New requests take the batch's rows at their first step, their history seeds queued as at a
        disaggregated generation request's arrival."""
        for request in self.requests:
            self.provider.release_request_state(request.py_request_id)
        self.requests = [
            types.SimpleNamespace(
                py_request_id=200 + r,
                py_seq_slot=slot,
                py_batch_idx=None,
                is_dummy=False,
                py_multimodal_data=None,
                py_needs_onehot_draft_probs=False,
                max_beam_num_tokens=base + 1,
                py_num_compressed_tokens=0,
                py_prompt_len=base,
                py_draft_tokens=list(ids[1:]),
                get_last_tokens=lambda beam, last=ids[0]: last,
            )
            for r, (slot, base, ids) in enumerate(zip(slots, bases, tokens))
        ]
        for request, seed in zip(self.requests, seeds):
            self.provider.queue_history_seed(request.py_request_id, *seed)
        self.steady.admit(self.requests)
        self.attn.request_ids = [request.py_request_id for request in self.requests]

    def step(self, device_tensors, from_host=False):
        n = self.attn.num_tokens
        _, _, staged, _, glue = self.steady.stage(
            self.requests, device_tensors, self.buffers, self.attn, from_host=from_host
        )
        for request in self.requests:
            request.py_batch_idx = request.py_seq_slot
        inputs = dict(
            attn_metadata=self.attn,
            input_ids=self.buffers.input_ids_cuda[:n],
            position_ids=self.buffers.position_ids_cuda[:n].view(1, n),
        )
        output = self.runner.replay(
            self.key,
            inputs,
            position_id_offsets=self.buffers.previous_pos_id_offsets_cuda,
            static_inputs_staged=staged,
            pre_replay_staged=glue,
        )
        return glue, output.clone()


def test_steady_glue_matches_the_staging_launch_and_replay_hook(monkeypatch):
    engram, b, t = (3, 8, (1, 15)), 2, 6
    n = b * t
    gen = torch.Generator().manual_seed(11)
    slots, bases = [5, 2], [1000, 2500]
    candidate = _Side(monkeypatch, engram, True, b, t, slots, bases)
    reference = _Side(monkeypatch, engram, False, b, t, slots, bases)
    ids = torch.randint(-5, 300, (n,), generator=gen, dtype=torch.int32).cuda()
    positions = torch.tensor(
        [base + j for base in bases for j in range(t)], dtype=torch.int32
    ).cuda()
    seeds = [(base - 2, [7 + base % 5, 300]) for base in bases]
    for side in (candidate, reference):
        side.full_pass(ids.clone(), positions.clone(), seeds)
    for step in range(9):
        device_tensors = types.SimpleNamespace(
            new_tokens=torch.randint(
                -5, 300, (t, SLOTS, 1), generator=gen, dtype=torch.int32
            ).cuda(),
            new_tokens_lens=torch.randint(
                1, t + 1, (SLOTS,), generator=gen, dtype=torch.int32
            ).cuda(),
            next_draft_tokens=torch.randint(
                0, 300, (SLOTS, t), generator=gen, dtype=torch.int32
            ).cuda(),
        )
        if step == 4:
            # A pending history seed in the batch: the step takes the replay's hook.
            for side in (candidate, reference):
                side.provider.queue_history_seed(100, 2400 + step, [17, 23])
        if step == 6:
            # New requests join on recycled history rows, seeded: the launch writes the seeds too.
            tokens = torch.randint(-5, 300, (b, t), generator=gen, dtype=torch.int32).tolist()
            for side in (candidate, reference):
                side.admit([3, 6], [3000, 700], [(2998, [11, 302]), (698, [4, 9])], tokens)
        results = [
            side.step(device_tensors, from_host=step == 6) for side in (candidate, reference)
        ]
        torch.cuda.synchronize()
        assert results[0][0] == (step < 4 or step >= 6) and not results[1][0], step
        assert torch.equal(results[0][1], results[1][1]), step
        assert torch.equal(candidate.provider._history, reference.provider._history), step
        assert torch.equal(
            candidate.runner._pre_replay_position_ids, reference.runner._pre_replay_position_ids
        ), step
        for name, value in vars(candidate.buffers).items():
            if value is not None:
                assert torch.equal(value, getattr(reference.buffers, name)), (step, name)
        for side in (candidate, reference):
            for request in side.requests:
                request.max_beam_num_tokens += 3
    assert candidate.hook_calls == 2 and reference.hook_calls == 9
