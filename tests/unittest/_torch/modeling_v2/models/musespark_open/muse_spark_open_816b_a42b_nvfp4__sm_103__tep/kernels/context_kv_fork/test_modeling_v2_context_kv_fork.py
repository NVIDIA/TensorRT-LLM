# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GPU test for ``context_kv_fork``: the DFlash context-K/V store on a side stream, on the published drafter.

Two DFlash workers hold the same arena and lengths. Every step one runs the
built-in ``prepare_1st_drafter_inputs``; the other is re-classed by the
fork, forks its side chain (armed and fired as ``DecodeTarget`` and the LM
head do), runs unrelated work on the main stream meanwhile, and prepares. The
whole arenas (as int16), the lengths, every drafter input the prepare returns,
and then the drafter's block decode (``drafter_block_decode``, output and arena) must be
bitwise equal -- over steps with every accepted count from 1 to K + 1, up to
and past the arena's last column (there with a deterministic ``index_put``,
since the built-in stores several rows to that one column at once), eagerly
and under CUDA-graph replay. The side stream is a stream of its own: never one
that PyTorch's stream pool hands out, never the capturing stream.

Needs ``MUSESPARK_REFERENCE_DRAFTER`` (the published drafter checkpoint).
"""

import contextlib
import os
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.context_kv_fork import (  # noqa: E501
    fork as context_kv_fork,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.drafter_block_decode import (  # noqa: E501
    block_decode as draft,
)
from tensorrt_llm._torch.speculative.dflash import DFlashWorker

assert torch.cuda.is_available(), "the context-K/V fork requires a CUDA device"

_DRAFTER_DIR = os.environ.get("MUSESPARK_REFERENCE_DRAFTER")
pytestmark = pytest.mark.skipif(not _DRAFTER_DIR, reason="needs MUSESPARK_REFERENCE_DRAFTER")

K = 7
MAX_CTX = 4096
SLOT = 0
TAPS = (1, 16, 30, 44, 59)


@pytest.fixture(scope="module")
def drafter():
    """The published drafter (TP1) with the fused block decode; a random embedding stands in for the target's."""
    from safetensors.torch import load_file
    from transformers import AutoConfig

    from tensorrt_llm._torch.model_config import ModelConfig
    from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM

    pretrained = AutoConfig.from_pretrained(_DRAFTER_DIR)
    with torch.device("cuda"):
        model = DFlashForCausalLM(ModelConfig(pretrained_config=pretrained, attn_backend="TRTLLM"))
    model.load_weights(load_file(os.path.join(_DRAFTER_DIR, "model.safetensors")))
    assert model.dflash_attention_backend == "VANILLA"
    draft.install(model, False, True)
    g = torch.Generator(device="cuda").manual_seed(3)
    vocab = int(model.mask_token_id) + 1
    embed = nn.Embedding(vocab, pretrained.hidden_size, device="cuda", dtype=torch.bfloat16)
    embed.weight.data.copy_(torch.randn(embed.weight.shape, generator=g, device="cuda"))
    model.draft_model_full.model.embed_tokens = embed
    model._build_fused_kv_buffers()
    return model


class _SpecMetadata:
    """What the prepare reads of a ``DFlashSpecMetadata`` (the capture buffer and its view)."""

    def __init__(self, hidden: int):
        self.hidden_size = hidden
        self.num_capture_layers = len(TAPS)
        self.layers_to_capture = list(TAPS)
        self.runtime_draft_len = K
        self.captured_hidden_states = torch.zeros(
            16, hidden * len(TAPS), dtype=torch.bfloat16, device="cuda"
        )

    def get_hidden_states(self, num_tokens):
        return self.captured_hidden_states[
            :num_tokens, : self.hidden_size * self.num_capture_layers
        ]


def _attn_metadata(num_contexts=0):
    return SimpleNamespace(
        num_contexts=num_contexts,
        num_seqs=1,
        num_tokens=K + 1,
        num_ctx_tokens=0,
        _seq_lens_cuda=torch.zeros(1, dtype=torch.int32, device="cuda"),
        _seq_lens=torch.zeros(1, dtype=torch.int32),
    )


def _worker(drafter, start_len: int):
    worker = DFlashWorker.__new__(DFlashWorker)
    # The DFlash context arena's windowing state, as DFlashWorker.__init__ sets it.
    worker._ctx_base = None
    worker._ctx_base_host = None
    worker._ctx_keep = 0
    worker.windowed_ctx_arena = False
    worker._saved_ctx_base = None
    worker._saved_ctx_base_host = None
    nn.Module.__init__(worker)
    worker.spec_config = SimpleNamespace(mask_token_id=None, max_draft_len=K)
    worker._resolved_mask_token_id = None
    worker._compute_block_size = K + 1
    worker._dflash_attention_backend = "VANILLA"
    worker._ctx_buf_inited = True
    worker._ctx_paged = False
    worker._ctx_block_tables = None
    worker._ctx_page_table = None
    worker._ctx_kv_buf = None
    worker._max_ctx = MAX_CTX
    L, nkv, hd = drafter._num_attn_layers, drafter._num_kv_heads, drafter._head_dim
    shape = (2, L, MAX_CTX + K + 1, nkv, hd)
    g = torch.Generator(device="cuda").manual_seed(start_len)
    worker._ctx_k_buf = torch.randn(shape, generator=g, device="cuda").to(torch.bfloat16)
    worker._ctx_v_buf = torch.randn(shape, generator=g, device="cuda").to(torch.bfloat16)
    worker._ctx_len = torch.tensor([start_len, 5], dtype=torch.int64, device="cuda")
    worker._batch_to_slot = torch.tensor([SLOT], dtype=torch.long, device="cuda")
    return worker


def _copy_state(dst, src):
    for name in ("_ctx_k_buf", "_ctx_v_buf", "_ctx_len"):
        getattr(dst, name).copy_(getattr(src, name))


def _fork_worker(drafter, start_len: int):
    worker = _worker(drafter, start_len)
    context_kv_fork._reclass(worker)
    assert type(worker) is context_kv_fork.ContextKvForkDFlashWorker
    worker._kv_fork = context_kv_fork.ContextKvFork(worker, drafter, "head")
    return worker


class _Inputs:
    """Static step inputs: the captured rows, the target tokens, the accepted count."""

    def __init__(self, hidden):
        self.spec = _SpecMetadata(hidden)
        self.attn = _attn_metadata()
        self.accepted = torch.zeros(1, K + 1, dtype=torch.int32, device="cuda")
        self.num_accepted = torch.ones(1, dtype=torch.int32, device="cuda")
        self.hidden = torch.zeros(K + 1, hidden, dtype=torch.bfloat16, device="cuda")

    def load(self, step: int, vocab: int):
        g = torch.Generator(device="cuda").manual_seed(1000 + step)
        rows = self.spec.captured_hidden_states
        rows.copy_((2.0 * torch.randn(rows.shape, generator=g, device="cuda")).to(rows.dtype))
        self.accepted.copy_(
            torch.randint(0, vocab - 1, self.accepted.shape, generator=g, device="cuda")
        )
        self.num_accepted.fill_(1 + step % (K + 1))

    def prepare(self, worker, drafter):
        return worker.prepare_1st_drafter_inputs(
            input_ids=None,
            position_ids=None,
            hidden_states=self.hidden,
            accepted_tokens=self.accepted,
            num_accepted_tokens=self.num_accepted,
            attn_metadata=self.attn,
            spec_metadata=self.spec,
            draft_model=drafter,
            total_target_tokens=K + 1,
        )


def _decode(drafter, inputs: dict, worker):
    return drafter.dflash_forward(
        noise_embedding=inputs["noise_embedding"],
        query_positions=inputs["query_positions"],
        num_ctx_per_req=inputs["num_ctx_per_req"],
        ctx_k_cache=worker._ctx_k_buf,
        ctx_v_cache=worker._ctx_v_buf,
        ctx_cache_batch_idx=inputs["ctx_cache_batch_idx"],
    )


def _busy_main_stream():
    """Unrelated work on the main stream while the side chain runs (the head and sampler's place)."""
    a = torch.randn(2048, 8192, device="cuda", dtype=torch.bfloat16)
    for _ in range(4):
        a = torch.nn.functional.linear(a, a[:2048])
    return a


def _forked_prepare(worker, drafter, inp: _Inputs, busy=True):
    fork = worker._kv_fork
    assert fork.arm(inp.spec, inp.attn) is None
    fork.fire("head")
    assert fork.launched
    if busy:
        _busy_main_stream()
    return inp.prepare(worker, drafter)


def _bits(t):
    return t.view(torch.int16) if t.dtype in (torch.bfloat16, torch.float16) else t


def _assert_same(a, b, what):
    assert a.shape == b.shape and a.dtype == b.dtype, (what, a.shape, b.shape, a.dtype, b.dtype)
    assert torch.equal(_bits(a), _bits(b)), f"{what}: {(a != b).sum().item()} elements differ"


def _assert_states(fork_worker, base_worker, got, ref, what):
    for key in ("noise_embedding", "query_positions", "num_ctx_per_req", "first_prev_tokens",
                "ctx_cache_batch_idx"):  # fmt: skip
        _assert_same(got[key], ref[key], f"{what} {key}")
    assert (
        got["ctx_k_cache"] is fork_worker._ctx_k_buf
        and got["ctx_v_cache"] is fork_worker._ctx_v_buf
    )
    _assert_same(fork_worker._ctx_len, base_worker._ctx_len, f"{what} ctx_len")
    _assert_same(fork_worker._ctx_k_buf, base_worker._ctx_k_buf, f"{what} arena K")
    _assert_same(fork_worker._ctx_v_buf, base_worker._ctx_v_buf, f"{what} arena V")


STARTS = (1_000, MAX_CTX - 3 * (K + 1))


@contextlib.contextmanager
def _last_duplicate_wins(enabled: bool):
    """Deterministic ``index_put`` (of duplicate indices, the last wins) while ``enabled``.

    At the arena's end several rows clamp to column ``MAX_CTX - 1`` and the
    built-in stores them with one ``index_put``, whose winner among duplicate
    indices is otherwise unspecified -- for the built-in run against itself too.
    """
    if not enabled:
        yield
        return
    was = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(was, warn_only=warn_only)


@pytest.mark.parametrize("start", STARTS, ids=["mid", "to_the_end"])
@torch.inference_mode()
def test_forked_store_is_bitwise_the_builtin(drafter, start):
    base = _worker(drafter, start)
    forked = _fork_worker(drafter, start)
    vocab = drafter.draft_model_full.model.embed_tokens.num_embeddings
    inp = _Inputs(drafter.config.hidden_size)
    with _last_duplicate_wins(start + 3 * (K + 1) * (K + 1) > MAX_CTX):
        for step in range(3 * (K + 1)):
            inp.load(step, vocab)
            ref = inp.prepare(base, drafter)
            # The drafter's block append writes the arena too: compare the
            # prepared states before either decode, then the decodes.
            prepared = _worker(drafter, 0)
            _copy_state(prepared, base)
            ref_out = _decode(drafter, ref, base)
            got = _forked_prepare(forked, drafter, inp)
            assert not forked._kv_fork.launched
            _assert_states(forked, prepared, got, ref, f"step {step}")
            got_out = _decode(drafter, got, forked)
            _assert_same(got_out, ref_out, f"step {step} drafter output")
            _assert_same(forked._ctx_k_buf, base._ctx_k_buf, f"step {step} arena K after decode")
            _assert_same(forked._ctx_v_buf, base._ctx_v_buf, f"step {step} arena V after decode")
    assert forked._kv_fork.launches["head"] == 3 * (K + 1)
    assert int(forked._ctx_len[SLOT]) == min(
        start + sum(1 + s % (K + 1) for s in range(3 * (K + 1))), MAX_CTX
    )


@torch.inference_mode()
def test_cuda_graph_replay_is_bitwise_the_eager_builtin(drafter):
    start = 2_000
    base = _worker(drafter, start)
    forked = _fork_worker(drafter, start)
    vocab = drafter.draft_model_full.model.embed_tokens.num_embeddings
    inp = _Inputs(drafter.config.hidden_size)
    snapshot = _worker(drafter, start)
    _copy_state(snapshot, forked)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for step in range(2):
            inp.load(step, vocab)
            _decode(drafter, _forked_prepare(forked, drafter, inp, busy=False), forked)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    forked._kv_fork.arm(inp.spec, inp.attn)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fork = forked._kv_fork
        fork.fire("head")
        static_inputs = inp.prepare(forked, drafter)
        static_out = _decode(drafter, static_inputs, forked)
    _copy_state(forked, snapshot)
    _copy_state(base, snapshot)
    for step in range(2 * (K + 1)):
        inp.load(100 + step, vocab)
        ref = inp.prepare(base, drafter)
        ref_out = _decode(drafter, ref, base)
        graph.replay()
        torch.cuda.synchronize()
        _assert_states(forked, base, static_inputs, ref, f"replay {step}")
        _assert_same(static_out, ref_out, f"replay {step} drafter output")


@torch.inference_mode()
def test_steps_the_fork_does_not_reproduce_run_the_builtin(drafter, caplog):
    forked = _fork_worker(drafter, 500)
    base = _worker(drafter, 500)
    vocab = drafter.draft_model_full.model.embed_tokens.num_embeddings
    inp = _Inputs(drafter.config.hidden_size)
    inp.load(0, vocab)
    fork = forked._kv_fork
    # A step with context rows is never forked (DecodeTarget does not arm it;
    # the plan refuses it too).
    inp.attn = _attn_metadata(num_contexts=1)
    assert fork.plan(inp.spec, inp.attn) is None
    inp.attn = _attn_metadata()
    # A fork point other than the configured one does nothing ...
    fork.arm(inp.spec, inp.attn)
    fork.fire("layer")
    assert not fork.launched
    # ... and the verify is the fallback, logged once.
    fork.fire("verify")
    assert fork.launched and fork.launches["verify"] == 1
    ref = inp.prepare(base, drafter)
    got = inp.prepare(forked, drafter)
    _assert_states(forked, base, got, ref, "verify fallback")
    # Nothing armed: the built-in method runs.
    ref = inp.prepare(base, drafter)
    got = inp.prepare(forked, drafter)
    _assert_states(forked, base, got, ref, "not armed")
    # A failed step leaves a launched chain: the next arm joins and forgets it.
    fork.arm(inp.spec, inp.attn)
    fork.fire("head")
    assert fork.launched
    fork.arm(inp.spec, inp.attn)
    assert not fork.launched


def test_install_refusals(drafter, monkeypatch):
    worker = _worker(drafter, 10)
    assert context_kv_fork.worker_unsupported_reason(worker, drafter) is None
    assert "not a DFlashWorker" in context_kv_fork.worker_unsupported_reason(object(), drafter)

    class Overriding(DFlashWorker):
        def prepare_1st_drafter_inputs(self, *args, **kwargs):
            return None

    other = _worker(drafter, 10)
    other.__class__ = Overriding
    assert "overrides prepare_1st_drafter_inputs" in context_kv_fork.worker_unsupported_reason(
        other, drafter
    )
    with monkeypatch.context() as patch:
        patch.setattr(drafter, "dflash_attention_backend", "TRTLLM")
        assert "dense arena" in context_kv_fork.worker_unsupported_reason(worker, drafter)

    from tensorrt_llm._torch.modules.linear import Linear

    head = Linear(64, 128, dtype=torch.bfloat16)
    assert context_kv_fork.head_unsupported_reason(head) is None
    head.apply_linear = lambda *a, **k: None
    assert "already replaced" in context_kv_fork.head_unsupported_reason(head)
    assert "not a Linear" in context_kv_fork.head_unsupported_reason(nn.Linear(4, 4))
    assert context_kv_fork.POINT == "head"


def test_composes_with_another_feature_worker_class(drafter):
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
        spec_worker as drafter_worker,
    )

    worker = _worker(drafter, 10)
    worker.force_num_accepted_tokens = 0.0
    assert drafter_worker.install_fused_accept(worker) is None
    accept_cls = type(worker)
    context_kv_fork._reclass(worker)
    assert isinstance(worker, drafter_worker.FusedAcceptDFlashWorker)
    assert isinstance(worker, context_kv_fork.ContextKvForkDFlashWorker)
    assert type(worker).__mro__[1] is context_kv_fork.ContextKvForkDFlashWorker
    assert type(worker).__mro__[2] is accept_cls
    # A repeated fused_accept install sees its class and changes nothing.
    assert drafter_worker.install_fused_accept(worker) is None
    assert context_kv_fork.worker_unsupported_reason(worker, drafter) is None


@torch.inference_mode()
def test_side_stream_is_its_own_stream(drafter):
    """Not among the pooled streams (more than the pool holds), not the capture stream; a fork from it raises."""
    forked = _fork_worker(drafter, 10)
    fork = forked._kv_fork
    pooled = [torch.cuda.Stream() for _ in range(80)]
    assert all(stream != fork.stream for stream in pooled)
    assert fork.stream.priority == torch.cuda.Stream().priority
    x = torch.zeros(4, device="cuda")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        assert torch.cuda.current_stream() != fork.stream
        x += 1
    graph.replay()
    torch.cuda.synchronize()
    assert x.tolist() == [1.0] * 4
    vocab = drafter.draft_model_full.model.embed_tokens.num_embeddings
    inp = _Inputs(drafter.config.hidden_size)
    inp.load(0, vocab)
    fork.arm(inp.spec, inp.attn)
    with torch.cuda.stream(fork.stream), pytest.raises(RuntimeError, match="side stream"):
        fork.fire("head")
    assert not fork.launched
