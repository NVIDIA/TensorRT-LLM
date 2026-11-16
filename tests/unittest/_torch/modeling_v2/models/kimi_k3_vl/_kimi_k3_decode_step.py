# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Kimi K3 targets' step checks, host-side, each run by the target's test directory on its own modules: how a
target's ``modeling`` classifies a step (``decode_step``, the attention-residual epilogue ceiling, which steps the MoE
layers push on), the one path its text model runs for a step through every layer, and the DSpark worker's spec accept
and drafter inputs reading the current batch-to-slot buffer.

A step in the case tables is a ``DecodeStep``'s fields, built with the target's own class."""

import types
from importlib import import_module

import torch

from tensorrt_llm._device_copy import DeviceCopyBuffer

spec_accept_op = import_module(
    "tensorrt_llm._torch._experimental.modeling_v2.models.kimi_k3_vl.kernels.k3_spec_accept.op"
)


def step_metadata(seq_lens, num_contexts=0):
    return types.SimpleNamespace(
        num_contexts=num_contexts,
        num_generations=len(seq_lens) - num_contexts,
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
    )


#: (requests, tokens): R <= 8 generation requests of the same T <= 8 tokens, a pure decode step.
PURE_DECODE_STEPS = [
    (1, 1),
    (1, 8),
    (8, 1),
    (2, 4),
    (4, 2),
    (3, 1),
    (5, 1),
    (2, 8),
    (4, 8),
    (8, 8),
    (8, 2),
]


def check_pure_decode_step(modeling, requests, tokens):
    """A pure decode step: small up to one token tile, wide above."""
    step = modeling.decode_step(step_metadata([tokens] * requests), requests * tokens)
    assert step == modeling.DecodeStep(requests * tokens, requests, tokens)
    assert step.decode
    assert step.small == (requests * tokens <= 8)
    assert step.wide == (requests * tokens > 8)


#: (seq_lens, num_contexts, rows): other steps of at most 8 tokens.
SHORT_STEPS = [
    ([5], 1, 5),  # a short prefill
    ([3, 1], 1, 4),  # a short mixed step
    ([4, 4], 0, 4),  # rows that are not the step's tokens
    ([3, 2], 0, 5),  # a ragged decode step
]


def check_short_step_is_small_only(modeling, seq_lens, num_contexts, rows):
    """Other steps of at most 8 tokens take the token-count kernels only."""
    step = modeling.decode_step(step_metadata(seq_lens, num_contexts), rows)
    assert step == modeling.DecodeStep(rows)
    assert step.small and not step.decode and not step.wide


#: (seq_lens, num_contexts, rows): steps no decode kernel takes.
GENERIC_STEPS = [
    ([1024], 1, 1024),  # prefill
    ([100, 8], 1, 108),  # mixed step
    ([1] * 9, 0, 9),  # more requests than the kernels take
    ([16], 0, 16),  # more tokens per request than the kernels take
    ([8, 4], 0, 12),  # ragged decode step
    ([4, 4], 0, 16),  # rows that are not the step's tokens
    ([], 0, 0),  # empty step
]


def check_step_takes_the_generic_path(modeling, seq_lens, num_contexts, rows):
    assert modeling.decode_step(step_metadata(seq_lens, num_contexts), rows) is None


#: (step, ceiling): the most tokens of the step the fused attn_res kernels take.
ATTN_RES_CEILINGS = [
    ((5,), 8),  # a short prefill or mixed step
    ((8, 8, 1), 8),  # a decode step of one token tile
    ((6, 1, 6), 8),
    ((16, 2, 8), 32),  # a wide decode step
    ((64, 8, 8), 32),
]


def check_attn_res_epilogue_ceiling(modeling, step, ceiling):
    """The fused attn_res kernels take a classified step up to one token tile and a wide decode step up to 32
    tokens."""
    assert modeling._attn_res_max_tokens(modeling.DecodeStep(*step)) == ceiling


#: (step, given, pushes): the flags ``latent_push`` is given beyond the defaults of ``check_latent_push``, and whether
#: the MoE layers push.
LATENT_PUSH_STEPS = [
    ((1, 1, 1), {}, True),  # one token per request on the KDA decode kernels
    ((8, 8, 1), {}, True),
    ((8, 1, 8), {"kda_draft_records": True}, True),  # a DSpark verify with per-token states
    ((4, 2, 2), {"kda_draft_records": True}, True),
    ((8, 1, 8), {}, False),  # the built-in KDA verify
    ((5,), {}, False),  # a context request, or rows that are not the step's tokens
    ((16, 2, 8), {"kda_draft_records": True}, False),  # wide: the routed experts' all-reduce
    ((1, 1, 1), {"capturing": False}, False),  # an eager step
    ((1, 1, 1), {"breakable": True}, False),
    ((1, 1, 1), {"kda_decode_kernels": False}, False),  # a KDA layer on the built-in kernels
    ((1, 1, 1), {"mla_decode_branch": False}, False),  # MLA on the built-in path
]


def check_latent_push(modeling, step, given, pushes):
    """Which steps the MoE layers push on (``latent_push``): a pure decode step of at most 8 tokens, captured into a
    CUDA graph, whose attention layers all run the decode kernels; every other step keeps the routed experts'
    all-reduce."""
    flags = dict(
        capturing=True,
        breakable=False,
        kda_draft_records=False,
        kda_decode_kernels=True,
        mla_decode_branch=True,
    )
    flags.update(given)
    assert modeling.latent_push(modeling.DecodeStep(*step), **flags) == pushes


class _Layer:
    is_moe = False

    def __init__(self, calls):
        self.calls = calls

    def __call__(
        self, hidden_states, block_residual, num_snapshots, attn_metadata, *, step, **kwargs
    ):
        self.calls.append(step)
        return hidden_states, num_snapshots

    def forward_generic(
        self, hidden_states, block_residual, num_snapshots, attn_metadata, capture=None
    ):
        self.calls.append("generic")
        return hidden_states, num_snapshots


#: (seq_lens, num_contexts, rows, path): a step and the path it takes, a ``DecodeStep``'s fields or "generic".
STEP_PATHS = [
    ([1], 0, 1, (1, 1, 1)),
    ([1] * 8, 0, 8, (8, 8, 1)),
    ([8], 0, 8, (8, 1, 8)),
    ([8, 8], 0, 16, (16, 2, 8)),
    ([5], 1, 5, (5,)),
    ([3, 1], 1, 4, (4,)),
    ([3, 2], 0, 5, (5,)),
    ([1024], 1, 1024, "generic"),
    ([100, 8], 1, 108, "generic"),
    ([1] * 9, 0, 9, "generic"),
    ([1] * 16, 0, 16, "generic"),
    ([8, 4], 0, 12, "generic"),
]


def check_each_step_takes_one_path_through_every_layer(
    modeling, seq_lens, num_contexts, rows, path
):
    """A target's text model (``modeling.KimiLinearModel``) runs one path for the step through every layer."""
    model = object.__new__(modeling.KimiLinearModel)
    torch.nn.Module.__init__(model)
    calls = []
    model.layers = [_Layer(calls), _Layer(calls), _Layer(calls)]
    model.embed_tokens = torch.nn.Embedding(4, 8)
    model.num_attn_res_snapshots = 2
    model.decode_gemvs = None
    model._latent_push = lambda attn_metadata, step: False
    model._final_norm = lambda hidden_states, *args: hidden_states
    model.forward(
        step_metadata(seq_lens, num_contexts), input_ids=torch.zeros(rows, dtype=torch.long)
    )
    assert calls == [path if path == "generic" else modeling.DecodeStep(*path)] * 3


def slot_worker(spec_worker):
    """A target's DSpark worker (``spec_worker.KimiK3DSparkWorker``) with three requests in slots 2, 0 and 1."""
    worker = object.__new__(spec_worker.KimiK3DSparkWorker)
    torch.nn.Module.__init__(worker)
    worker._batch_to_slot_buffer = DeviceCopyBuffer(torch.zeros(3, dtype=torch.long))
    worker._req_to_slot = {100: 2, 101: 0, 102: 1}
    worker._dummy_slot = 3
    worker._ctx_len = torch.tensor([10, 20, 30, 0])
    worker._max_ctx = 64
    worker._compute_block_size = 8
    worker.spec_config = types.SimpleNamespace(max_draft_len=7)
    worker._resolved_mask_token_id = 0
    worker._ctx_pool_idx = 0
    worker._ctx_block_divisor = 1
    worker._ctx_block_counts = torch.full((3,), 4, dtype=torch.int32)
    worker._ctx_block_tables = None
    worker._ctx_page_table = None
    worker._ctx_page_size = 32
    worker._ctx_k_buf = torch.empty(4, 1, 64, 1, 4)
    worker._ctx_v_buf = torch.empty_like(worker._ctx_k_buf)
    worker._ctx_kv_buf = None
    return worker


#: (request_ids, expected): the batch order and the slots the spec accept reads.
SLOT_ORDERS = [([100, 101, 102], [2, 0, 1]), ([102, 100, 101], [1, 2, 0])]


def check_spec_accept_uses_current_slot_buffer(monkeypatch, spec_worker, request_ids, expected):
    """A target's DSpark worker accepts on the current batch-to-slot buffer."""
    worker = slot_worker(spec_worker)
    worker._rebuild_batch_to_slot([100, 101, 102])
    worker._rebuild_batch_to_slot(request_ids)
    worker.force_num_accepted_tokens = 0.0
    worker._k3_dummy_rng = (torch.zeros(4), torch.zeros(2, dtype=torch.int64))
    monkeypatch.setattr(spec_accept_op, "force_mode", lambda *args: (0, 0, 0))
    monkeypatch.setattr(
        spec_accept_op, "_kernel_module", lambda: types.SimpleNamespace(FORCE_FRAC=1)
    )
    monkeypatch.setattr(worker, "_k3_mask_row", lambda *args: torch.zeros(4))
    received = []

    def accept_kernel(*args):
        received.append(args[11])
        return tuple(torch.zeros(3) for _ in range(7))

    monkeypatch.setattr(torch.ops.trtllm, "k3_spec_accept", accept_kernel, raising=False)
    manager = types.SimpleNamespace(
        _kda_replay=types.SimpleNamespace(prev_num_accepted_tokens=torch.zeros(3)),
        _dummy_request_mask=torch.zeros(3),
    )
    metadata = types.SimpleNamespace(
        num_seqs=3,
        num_contexts=0,
        kv_cache_manager=manager,
        draft_kv_cache_block_offsets=torch.zeros(3, 1),
        mamba_metadata=types.SimpleNamespace(
            state_indices=torch.arange(3), replay_state_indices=torch.arange(3)
        ),
        kv_lens_cuda=torch.ones(3, dtype=torch.int32),
    )
    spec = types.SimpleNamespace(
        runtime_draft_len=7, is_cuda_graph=False, draft_tokens=torch.zeros(3, 7)
    )
    draft = types.SimpleNamespace(
        draft_model_full=types.SimpleNamespace(
            model=types.SimpleNamespace(embed_tokens=torch.nn.Embedding(8, 4))
        )
    )
    worker._k3_accept(torch.zeros(3, 8), metadata, spec, draft)
    assert received[0] is worker._batch_to_slot_buffer.tensor
    torch.testing.assert_close(received[0], torch.tensor(expected), rtol=0, atol=0)


def check_drafter_inputs_use_current_slot_buffer(monkeypatch, spec_worker, num_contexts, paged):
    """A target's DSpark worker builds the drafter inputs from the current
    batch-to-slot buffer."""
    worker = slot_worker(spec_worker)
    worker._rebuild_batch_to_slot([102, 100, 101])
    if paged:
        worker._ctx_block_tables = torch.zeros(3, 4, dtype=torch.int32)
    monkeypatch.setattr(worker, "_k3_static_arange", lambda start, end: torch.arange(start, end))
    metadata = types.SimpleNamespace(
        num_contexts=num_contexts,
        num_seqs=3,
        _seq_lens_cuda=torch.ones(3, dtype=torch.int32),
        _seq_lens=torch.ones(3, dtype=torch.int32),
    )
    spec = types.SimpleNamespace(
        hidden_size=4, runtime_draft_len=7, get_hidden_states=lambda count: None
    )
    draft = types.SimpleNamespace(
        draft_model_full=types.SimpleNamespace(
            model=types.SimpleNamespace(embed_tokens=torch.nn.Embedding(8, 4))
        )
    )
    num_gens = 3 - num_contexts
    prep = dict(
        bonus=torch.zeros(num_gens, dtype=torch.long),
        qpos=torch.zeros(num_gens, 8, dtype=torch.long),
        cpos=torch.zeros(num_gens, 8, dtype=torch.long),
        noise=torch.zeros(num_gens, 8, 4),
    )
    result = worker.prepare_1st_drafter_inputs(
        torch.zeros(24, dtype=torch.long),
        torch.arange(24),
        torch.zeros(24, 4),
        torch.zeros(3, 8, dtype=torch.long),
        torch.full((3,), 2, dtype=torch.int32),
        metadata,
        spec,
        draft,
        total_target_tokens=24,
        prep=prep,
    )
    expected = torch.arange(num_contexts, 3) if paged else torch.tensor([1, 2, 0])[num_contexts:]
    torch.testing.assert_close(result["ctx_cache_batch_idx"], expected, rtol=0, atol=0)
    torch.testing.assert_close(
        result["num_ctx_per_req"], torch.tensor([20, 30, 10])[num_contexts:], rtol=0, atol=0
    )
