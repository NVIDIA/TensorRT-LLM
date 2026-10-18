# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host tests of the head fast paths' worker routing and composition (``target_head.py``).

The worker is a bare ``DFlashWorker`` re-classed the way the fast paths do it;
the head's device ops are replaced by recorders, so every case runs on the
CPU and checks which path a step takes:

* the fast path (non-greedy fused sampling) samples the pending row itself;
* greedy steps, a non-fused sampling mode, occurrence penalties and
  rejection sampling materialize the built-in FP32 capped row first;
* a copied or re-wrapped row reaching the sampler raises;
* a row nothing consumed is drained by the forward hook, also when the
  forward raises; a second head call with a row still pending raises;
* ``takes`` is rank-uniform and logs a guided-decoder bypass once;
* the composed class overrides only ``WORKER_METHODS``, so it passes the
  refusal checks of the fast paths that re-class the worker for other
  methods; with ``context_kv_fork`` / ``verify_fold`` present in the tree, their
  real installs compose with the head in both orders, repeated calls included.
"""

from __future__ import annotations

import importlib
import importlib.util
import types

import pytest
import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.split_sampler import (  # noqa: E501
    target_head,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.ops.spec import (  # noqa: E501
    head_sampler,
)
from tensorrt_llm._torch.speculative.dflash import DFlashWorker

PACKAGE = target_head.__name__.rsplit(".", 1)[0]
MULTIPLIER = 0.1767766952966369
SOFT_CAP = 20.0
ROWS = 8
VOCAB = 64

#: The worker methods the context-K/V fork refuses an override of (``context_kv_fork.FORK_METHODS``)
#: and the verify fold (``verify_fold.fold.OVERRIDDEN``).
CONTEXT_KV_FORK_FORK_METHODS = (
    "prepare_1st_drafter_inputs",
    "sample_and_accept_draft_tokens",
    "_lazy_init_ctx_buffers",
    "_ensure_spec_dec_state_restored",
    "_forward_impl",
)
VERIFY_FOLD_OVERRIDDEN = (
    "_forward_impl",
    "sample_and_accept_draft_tokens",
    "_sample_and_accept_draft_tokens_base",
    "_prepare_attn_metadata_for_dflash",
    "_prepare_kv_for_draft_forward",
    "prepare_1st_drafter_inputs",
    "_apply_dflash2_selector",
    "_apply_kv_rewind_after_draft",
    "_prepare_next_new_tokens",
)


class Recorder:
    """Stands in for the head's device ops and records each call."""

    def __init__(self):
        self.calls = []

    def split_sample(self, logits, *args):
        self.calls.append(("split_sample", logits))
        return torch.full((logits.shape[0],), 7, dtype=torch.int32)

    def gather_split_sample(self, inbox, gathered, *args):
        sample = args[-2]
        self.calls.append(("gather", gathered, sample))
        if not sample:
            return gathered
        return torch.full((gathered.shape[0],), 9, dtype=torch.int32)


@pytest.fixture
def ops(monkeypatch):
    rec = Recorder()
    monkeypatch.setattr(head_sampler, "split_sample", rec.split_sample)
    monkeypatch.setattr(head_sampler, "gather_split_sample", rec.gather_split_sample)
    return rec


class Below(DFlashWorker):
    """What sits below the head in the composed class: the fused acceptance entry, recording.

    Like ``FusedAcceptDFlashWorker`` (and the verify fold) it hands the logits it
    gets to ``self._sample_tokens_for_batch``.
    """

    def _sample_and_accept_draft_tokens_base(
        self, logits, draft_tokens, num_contexts, batch_size, spec_metadata
    ):
        self.seen.append(("base", logits))
        tokens = self._sample_tokens_for_batch(logits, spec_metadata, num_contexts, batch_size)
        return tokens, tokens


class Copies(Below):
    def _sample_and_accept_draft_tokens_base(
        self, logits, draft_tokens, num_contexts, batch_size, spec_metadata
    ):
        return super()._sample_and_accept_draft_tokens_base(
            logits.clone(), draft_tokens, num_contexts, batch_size, spec_metadata
        )


class Rewraps(Below):
    def sample_and_accept_draft_tokens(self, logits, attn_metadata, spec_metadata):
        return super().sample_and_accept_draft_tokens(
            logits.view(logits.shape), attn_metadata, spec_metadata
        )


class Raises(Below):
    def _forward_impl(self, *args, **kwargs):
        raise ValueError("a failed step")


class PassesThrough(Below):
    """Overrides the methods context_kv_fork / verify_fold re-class the worker for, passing the logits on."""

    def _forward_impl(self, *args, **kwargs):
        return super()._forward_impl(*args, **kwargs)

    def sample_and_accept_draft_tokens(self, logits, attn_metadata, spec_metadata):
        return super().sample_and_accept_draft_tokens(logits, attn_metadata, spec_metadata)

    def _ensure_spec_dec_state_restored(self, attn_metadata, spec_metadata):
        return super()._ensure_spec_dec_state_restored(attn_metadata, spec_metadata)

    def prepare_1st_drafter_inputs(self, *args, **kwargs):
        return super().prepare_1st_drafter_inputs(*args, **kwargs)


def _bare(cls):
    worker = object.__new__(cls)
    nn.Module.__init__(worker)
    worker.guided_decoder = None
    return worker


def _head(worker, inbox: bool = False) -> target_head.TargetHead:
    head = object.__new__(target_head.TargetHead)
    head.multiplier, head.soft_cap = MULTIPLIER, SOFT_CAP
    head.worker, head.gather, head.pdl = worker, inbox, False
    head.device = torch.device("cpu")
    head.library, head.load_seconds = "", 0.0
    head.workspace = (
        types.SimpleNamespace(
            inbox=torch.empty(1), states=torch.empty(1), status=torch.zeros(1), shard=VOCAB // 8
        )
        if inbox
        else None
    )
    head.states = torch.empty(1)
    head.pending, head.pending_inbox = None, False
    head._bypass_logged = False
    head.hook_handle = None
    return head


def _worker(below=Below, inbox: bool = False):
    worker = _bare(below)
    worker.seen = []
    head = _head(worker, inbox)
    target_head.attach_worker(worker, head)
    # The built-in RNG bookkeeping reads request state this stand-in has none of.
    worker._rng_state_per_token = lambda spec_metadata, n: (None, None)
    return worker, head


def _pending(head, inbox: bool = False) -> torch.Tensor:
    g = torch.Generator().manual_seed(3)
    row = (torch.randn(ROWS, VOCAB, generator=g) * 40).to(torch.bfloat16)
    head.pending, head.pending_inbox = row, inbox
    return row


def _spec(**kw):
    defaults = dict(
        is_all_greedy_sample=False,
        advanced_sampling_mode=types.SimpleNamespace(is_fused=True),
        enable_penalty=False,
        use_rejection_sampling=False,
        temperatures=torch.ones(ROWS),
        top_ks=torch.zeros(ROWS, dtype=torch.int32),
        top_ps=torch.ones(ROWS),
        min_ps=torch.zeros(ROWS),
        draft_probs=None,
        batch_slot_ids=None,
        draft_probs_last_dim=0,
        draft_probs_vocab_size=VOCAB,
        is_ragged_verify=False,
    )
    defaults.update(kw)
    return types.SimpleNamespace(**defaults)


def _accept(worker, logits, spec):
    draft = torch.zeros(1, ROWS - 1, dtype=torch.int32)
    return worker._accept_draft_tokens(logits, draft, 0, 1, spec)


def test_the_head_worker_overrides_exactly_its_methods():
    own = {
        name
        for name, value in vars(target_head.TargetHeadDFlashWorker).items()
        if callable(value) and not name.startswith("__")
    }
    assert own == set(target_head.WORKER_METHODS)


def test_the_head_methods_are_not_ones_context_kv_fork_or_verify_fold_refuse():
    overridden = set(target_head.WORKER_METHODS)
    assert not overridden & set(CONTEXT_KV_FORK_FORK_METHODS)
    assert not overridden & set(VERIFY_FOLD_OVERRIDDEN)


def test_a_class_overriding_the_other_features_methods_is_admitted():
    assert target_head.worker_unsupported_reason(_bare(PassesThrough)) is None


@pytest.mark.parametrize("name", target_head.WORKER_METHODS + target_head.PASS_THROUGH_METHODS)
def test_a_class_overriding_a_head_method_is_refused(name):
    cls = type("Overrides", (DFlashWorker,), {name: lambda self, *a, **k: None})
    reason = target_head.worker_unsupported_reason(_bare(cls))
    assert reason is not None and name in reason


def test_fast_path_samples_the_pending_row(ops):
    worker, head = _worker()
    row = _pending(head)
    tokens, _ = _accept(worker, row, _spec())
    assert [c[0] for c in ops.calls] == ["split_sample"]
    assert ops.calls[0][1] is row
    assert worker.seen[0][1] is row
    assert head.pending is None
    assert torch.equal(tokens, torch.full((ROWS,), 7, dtype=torch.int32))


def test_fast_path_through_the_inbox_samples_with_the_gather(ops):
    worker, head = _worker(inbox=True)
    row = _pending(head, inbox=True)
    _accept(worker, row, _spec())
    assert [(c[0], c[2]) for c in ops.calls] == [("gather", True)]
    assert head.pending is None and not head.pending_inbox


def _capped(row):
    return head_sampler.reference_capped(row, MULTIPLIER, SOFT_CAP)


def test_greedy_steps_take_argmax_of_the_capped_row(ops):
    worker, head = _worker(inbox=True)
    row = _pending(head, inbox=True)
    tokens, _ = _accept(worker, row, _spec(is_all_greedy_sample=True))
    assert [(c[0], c[2]) for c in ops.calls] == [("gather", False)]
    assert torch.equal(tokens, torch.argmax(_capped(row), dim=-1))
    assert head.pending is None


def test_a_non_fused_mode_samples_the_capped_row_with_the_builtin(ops):
    worker, head = _worker()
    row = _pending(head)
    seen = []

    def builtin(mode, logits, *args, **kwargs):
        seen.append(logits)
        return torch.zeros(logits.shape[0], dtype=torch.int32)

    worker._sample_from_logits = builtin
    _accept(worker, row, _spec(advanced_sampling_mode=types.SimpleNamespace(is_fused=False)))
    assert ops.calls == []
    assert seen[0].dtype == torch.float32 and torch.equal(seen[0], _capped(row))


def test_penalties_read_the_capped_row(ops):
    class Penalized(Below):
        def _apply_occurrence_penalties(self, logits, *args):
            self.seen.append(("penalties", logits))
            return logits

        def _commit_occurrence_counts(self, accepted, batch_size, spec_metadata):
            return accepted

    worker, head = _worker(Penalized, inbox=True)
    row = _pending(head, inbox=True)
    sampled = []

    def builtin(mode, logits, *args, **kwargs):
        sampled.append(logits)
        return torch.zeros(logits.shape[0], dtype=torch.int32)

    worker._sample_from_logits = builtin
    _accept(worker, row, _spec(enable_penalty=True))
    assert worker.seen[0][0] == "penalties"
    assert torch.equal(worker.seen[0][1], _capped(row))
    assert [(c[0], c[2]) for c in ops.calls] == [("gather", False)]
    # The built-in sampler then samples the capped row (fast path not taken).
    assert worker.seen[1][1] is worker.seen[0][1]
    assert sampled[0] is worker.seen[0][1]


def test_rejection_sampling_reads_the_capped_row(ops):
    worker, head = _worker()
    row = _pending(head)
    seen = []

    def builtin(mode, logits, *args, **kwargs):
        seen.append(logits)
        return torch.zeros(logits.shape[0], dtype=torch.int32)

    worker._sample_from_logits = builtin
    # No draft-probability buffers: the built-in falls back to strict acceptance.
    _accept(worker, row, _spec(use_rejection_sampling=True))
    assert ops.calls == []
    assert torch.equal(worker.seen[0][1], _capped(row))
    assert seen[0] is worker.seen[0][1]


def test_a_copied_row_reaching_the_sampler_raises(ops):
    worker, head = _worker(Copies)
    row = _pending(head)
    with pytest.raises(RuntimeError, match="_sample_tokens_for_batch received logits other"):
        _accept(worker, row, _spec())
    assert ops.calls == []


def test_a_rewrapped_row_reaching_acceptance_raises(ops):
    worker, head = _worker(Rewraps)
    row = _pending(head)
    attn = types.SimpleNamespace(num_seqs=1, num_contexts=0)
    spec = _spec(draft_tokens=torch.zeros(ROWS - 1, dtype=torch.int32), runtime_draft_len=ROWS - 1)
    with pytest.raises(RuntimeError, match="_accept_draft_tokens received logits other"):
        worker.sample_and_accept_draft_tokens(row, attn, spec)


def test_other_logits_pass_when_nothing_is_pending(ops):
    worker, head = _worker()
    logits = torch.randn(ROWS, VOCAB)
    tokens, _ = _accept(worker, logits, _spec(is_all_greedy_sample=True))
    assert torch.equal(tokens, torch.argmax(logits, dim=-1))
    assert ops.calls == []


def test_a_failed_forward_drains_the_pending_row(ops):
    worker, head = _worker(Raises, inbox=True)
    _pending(head, inbox=True)
    with pytest.raises(ValueError, match="a failed step"):
        worker(attn_metadata=None, spec_metadata=None)
    assert [(c[0], c[2]) for c in ops.calls] == [("gather", False)]
    assert head.pending is None and not head.pending_inbox


def test_a_forward_that_consumed_its_row_drains_nothing(ops):
    class Consumes(Below):
        def _forward_impl(self, *args, **kwargs):
            return self._split_sampler_head.materialize(self._split_sampler_head.pending)

    worker, head = _worker(Consumes, inbox=True)
    _pending(head, inbox=True)
    worker(attn_metadata=None, spec_metadata=None)
    assert [(c[0], c[2]) for c in ops.calls] == [("gather", False)]


def test_a_head_call_with_a_row_still_pending_raises():
    worker, head = _worker()
    _pending(head)
    with pytest.raises(RuntimeError, match="neither sampled nor drained"):
        head.produce(torch.zeros(ROWS, 16, dtype=torch.bfloat16), lm_head=None)


def test_the_inbox_path_refuses_a_replaced_apply_linear():
    worker, head = _worker(inbox=True)
    head.workspace.slot = lambda rows: None
    lm_head = types.SimpleNamespace(apply_linear=lambda *a: None)
    with pytest.raises(RuntimeError, match="apply_linear was replaced"):
        head.produce(torch.zeros(ROWS, 16, dtype=torch.bfloat16), lm_head)
    assert head.pending is None


def _hidden(rows=ROWS, dtype=torch.bfloat16):
    return torch.zeros(rows, 16, dtype=dtype)


@pytest.mark.parametrize(
    "hidden, contexts, flag, takes",
    [
        (_hidden(), 0, True, True),
        (_hidden(1), 0, True, True),
        (_hidden(9), 0, True, True),
        (_hidden(64), 0, True, True),
        (_hidden(128), 0, True, True),
        (_hidden(129), 0, True, False),
        (_hidden(dtype=torch.float32), 0, True, False),
        (_hidden(), 1, True, False),
        (_hidden(), 0, False, False),
    ],
)
def test_takes_decode_rows_only(hidden, contexts, flag, takes):
    worker, head = _worker()
    attn = types.SimpleNamespace(num_contexts=contexts)
    assert head.takes(hidden, attn, flag) is takes


def test_a_guided_decoder_bypasses_the_head_and_logs_once(monkeypatch):
    worker, head = _worker()
    worker.guided_decoder = object()
    logged = []
    monkeypatch.setattr(target_head.logger, "warning", lambda msg: logged.append(msg))
    attn = types.SimpleNamespace(num_contexts=0)
    assert not head.takes(_hidden(), attn, True)
    assert not head.takes(_hidden(), attn, True)
    assert len(logged) == 1 and "guided decoder" in logged[0]


def test_the_poll_timeout_is_a_constant():
    assert head_sampler.POLL_TIMEOUT_S == 600


def test_the_split_sampler_takes_up_to_the_shipped_wide_block_and_the_inbox_eight_rows():
    assert head_sampler.MAX_ROWS == 128
    assert head_sampler.INBOX_ROWS == target_head.GATHER_ROWS == 8


@pytest.mark.parametrize("load_fails", [False, True], ids=["load-ok", "load-fails"])
def test_the_inbox_workspace_agrees_on_the_library_load_before_its_allocation(
    monkeypatch, load_fails
):
    """The fabric allocation is a collective: a rank whose library load failed must reach
    the load's agreement, not skip the allocation its peers are in. Here every rank
    reports this rank's outcome, so a failed load raises before any allocation."""
    from tensorrt_llm._torch.distributed import ops as dist_ops

    calls = []

    def agree(comm, local_success):
        calls.append(("agree", local_success))
        return local_success

    def allocate(comm, size, mapping, multi_node):
        calls.append(("allocate", size))
        raise RuntimeError("stop after the allocation")

    def load():
        calls.append(("load",))
        if load_fails:
            raise OSError("library load failed")
        return "lib"

    monkeypatch.setattr(dist_ops, "_get_mnnvl_workspace_comm", lambda mapping: "comm")
    monkeypatch.setattr(dist_ops, "_mnnvl_device_index", lambda mapping: 0)
    monkeypatch.setattr(dist_ops, "_mnnvl_workspace_all_succeeded", agree)
    monkeypatch.setattr(dist_ops, "_make_mnnvl_mcast_buffer", allocate)
    monkeypatch.setattr(head_sampler, "load_library", load)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (10, 7))
    mapping = types.SimpleNamespace(tp_size=8, tp_rank=4, is_multi_node=lambda: True)
    with pytest.raises(RuntimeError) as raised:
        head_sampler.HeadGatherWorkspace(mapping, VOCAB // 8)
    if load_fails:
        assert calls == [("load",), ("agree", False)]
        assert "failed to load" in str(raised.value)
    else:
        assert calls == [
            ("load",),
            ("agree", True),
            ("allocate", 8 * head_sampler.INBOX_ROWS * (VOCAB // 8) * 2),
            ("agree", False),
        ]
        assert "workspace failed" in str(raised.value)


@pytest.mark.parametrize("inbox", [False, True], ids=["split-sampler", "lm-head-inbox"])
def test_the_head_owns_states_for_every_row_count_it_takes(monkeypatch, inbox):
    """``split_sample`` runs at up to ``MAX_ROWS`` rows also under ``lm_head_inbox`` (every
    step but the 8-row ones), so the head's scratch is sized to ``MAX_ROWS``, never the inbox's."""
    sized = []
    monkeypatch.setattr(
        head_sampler, "new_states", lambda rows, device: sized.append(rows) or torch.empty(rows)
    )
    workspace = types.SimpleNamespace(library="lib", states=torch.empty(head_sampler.INBOX_ROWS))
    monkeypatch.setattr(head_sampler, "shared_gather_workspace", lambda mapping, shard: workspace)
    monkeypatch.setattr(head_sampler, "load_library_agreed", lambda mapping: "lib")
    processor = types.SimpleNamespace(logit_multiplier=MULTIPLIER, soft_cap=SOFT_CAP)
    lm_head = types.SimpleNamespace(weight=torch.empty(VOCAB // 8, 16))
    head = target_head.TargetHead(processor, lm_head, _bare(Below), None, inbox)
    assert sized == [head_sampler.MAX_ROWS]
    assert head.states.numel() == head_sampler.MAX_ROWS
    assert head.states is not workspace.states


def test_an_inbox_head_samples_other_row_counts_from_the_builtin_gather(monkeypatch):
    """Under ``lm_head_inbox`` a 64-row step runs the built-in LM head (its all-gather) and the
    split sampler on the head's own scratch; only 8-row steps use the inbox."""
    worker, head = _worker(inbox=True)
    head.states = torch.empty(head_sampler.MAX_ROWS)
    rows = 64
    row = (torch.randn(rows, VOCAB) * 40).to(torch.bfloat16)
    lm_head_calls = []

    def lm_head(hidden):
        lm_head_calls.append(hidden.shape[0])
        return row

    sampled = []
    monkeypatch.setattr(
        head_sampler,
        "split_sample",
        lambda logits, *args: sampled.append((logits, args[-2])) or torch.zeros(rows),
    )
    monkeypatch.setattr(
        head_sampler, "gather_split_sample", lambda *a: pytest.fail("the inbox gather ran")
    )
    logits = head.produce(torch.zeros(rows, 16, dtype=torch.bfloat16), lm_head)
    assert lm_head_calls == [rows] and logits is row and not head.pending_inbox
    ones = torch.ones(rows)
    head.sample(logits, ones, ones.int(), ones, ones, None, None)
    assert len(sampled) == 1 and sampled[0][0] is row and sampled[0][1] is head.states
    assert head.pending is None


def test_the_library_name_carries_the_arch_target():
    names = {head_sampler.library_name(t) for t in ("103a", "107a")}
    assert names == {f"{head_sampler.LIBRARY_NAME}_sm103a", f"{head_sampler.LIBRARY_NAME}_sm107a"}


# --------------------------------------------------------------------------
# With context_kv_fork / verify_fold in the tree: their real installs
# --------------------------------------------------------------------------
def _lane(module: str):
    if importlib.util.find_spec(f"{PACKAGE}.{module}") is None:
        pytest.skip(f"{module}.py is not in this tree")
    return importlib.import_module(f"{PACKAGE}.{module}")


def _served_worker():
    """A bare worker of the composed class below the head: the fused acceptance over the draft head."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep import (  # noqa: E501
        spec_worker as drafter_worker,
    )
    from tensorrt_llm._torch._experimental.modeling_v2.models.musespark_open.muse_spark_open_816b_a42b_nvfp4__sm_103__tep.kernels.draft_head import (  # noqa: E501
        head as draft_head,
    )

    worker = _bare(DFlashWorker)
    worker.__class__ = draft_head.ShardedDraftHeadDFlashWorker
    assert drafter_worker.install_fused_accept(worker) is None
    return worker


def _attach_head(worker):
    assert target_head.worker_unsupported_reason(worker) is None
    target_head.attach_worker(worker, _head(worker))


@pytest.mark.parametrize("head_first", [False, True])
def test_the_head_composes_with_verify_fold(head_first):
    fold = _lane("spec_loop_fold")
    worker = _served_worker()
    if head_first:
        _attach_head(worker)
        fold.install(worker, True, True)
    else:
        fold.install(worker, True, True)
        _attach_head(worker)
    # The shell installs from every load hook: repeated calls must not refuse.
    fold.install(worker, True, True)
    assert fold.unsupported_reason(worker) is None
    cls = type(worker)
    assert isinstance(worker, target_head.TargetHeadDFlashWorker)
    assert isinstance(worker, fold.SpecLoopFoldDFlashWorker)
    assert (
        cls._sample_tokens_for_batch is target_head.TargetHeadDFlashWorker._sample_tokens_for_batch
    )
    assert cls._accept_draft_tokens is target_head.TargetHeadDFlashWorker._accept_draft_tokens
    assert (
        cls._sample_and_accept_draft_tokens_base
        is fold.SpecLoopFoldDFlashWorker._sample_and_accept_draft_tokens_base
    )
    assert (
        cls.sample_and_accept_draft_tokens
        is fold.SpecLoopFoldDFlashWorker.sample_and_accept_draft_tokens
    )


@pytest.mark.parametrize("head_first", [False, True])
def test_the_head_composes_with_context_kv_fork(head_first):
    fork = _lane("context_kv_fork")
    from tensorrt_llm._torch.models.modeling_dflash import DFlashForCausalLM

    drafter = object.__new__(DFlashForCausalLM)
    nn.Module.__init__(drafter)
    drafter.dflash_attention_backend = "VANILLA"
    worker = _served_worker()
    if head_first:
        _attach_head(worker)
        assert fork.worker_unsupported_reason(worker, drafter) is None
        fork._reclass(worker)
    else:
        assert fork.worker_unsupported_reason(worker, drafter) is None
        fork._reclass(worker)
        _attach_head(worker)
    assert fork.worker_unsupported_reason(worker, drafter) is None
    cls = type(worker)
    assert (
        cls._sample_tokens_for_batch is target_head.TargetHeadDFlashWorker._sample_tokens_for_batch
    )
    assert (
        cls.sample_and_accept_draft_tokens
        is fork.ContextKvForkDFlashWorker.sample_and_accept_draft_tokens
    )


def _llm(replaced_apply_linear: bool):
    from tensorrt_llm._torch.models.modeling_musespark_open import MuseSparkOpenLogitsProcessor
    from tensorrt_llm._torch.modules.linear import Linear, TensorParallelMode

    processor = object.__new__(MuseSparkOpenLogitsProcessor)
    nn.Module.__init__(processor)
    lm_head = object.__new__(Linear)
    nn.Module.__init__(lm_head)
    lm_head.tp_mode = TensorParallelMode.COLUMN
    lm_head.tp_size = 8
    lm_head.gather_output = True
    lm_head.gather_output_sizes = None
    lm_head.bias = None
    lm_head.use_custom_cublas_mm = False
    lm_head.weight = torch.empty(25256, 8192, dtype=torch.bfloat16, device="meta")
    if replaced_apply_linear:
        # What the context-K/V fork's head fork point does (context_kv_fork._head_apply_linear).
        lm_head.apply_linear = types.MethodType(lambda self, *a, **k: None, lm_head)
    return types.SimpleNamespace(
        logits_processor=processor, spec_worker=_served_worker(), lm_head=lm_head, draft_model=None
    )


def _context_kv_fork_head(fork):
    """An LM head carrying the context-K/V fork's real head fork point (``context_kv_fork``'s binding)."""
    fork_module = _lane("context_kv_fork")
    llm = _llm(False)
    lm_head = llm.lm_head
    lm_head._context_kv_fork = fork
    lm_head.apply_linear = types.MethodType(fork_module._head_apply_linear, lm_head)
    return llm, lm_head


class _Fork:
    """Records ``fire`` calls in the order they happen relative to the head's GEMM."""

    def __init__(self, order):
        self.order = order

    def fire(self, point):
        self.order.append(("fire", point))


def test_context_kv_fork_head_fork_point_is_admitted_and_fired_after_the_gemm(monkeypatch):
    """Under ``lm_head_inbox`` the inbox path fires the context-K/V fork's head fork right after
    its own GEMM, as the fork's apply_linear does after ``Linear.apply_linear``; any other
    replacement is refused."""
    monkeypatch.setattr(target_head._low_m, "LOW_M_GEMM_ACTIVE", True)
    mapping = types.SimpleNamespace(enable_attention_dp=False, tp_size=8)
    order = []
    llm, lm_head = _context_kv_fork_head(_Fork(order))
    assert target_head.context_kv_head_fork(lm_head) == (True, lm_head._context_kv_fork)
    assert target_head.unsupported_reason(llm, mapping, gather=True) is None
    # Another function under the fork point's name, or the fork point bound to another module, is refused.
    impostor = _llm(False).lm_head
    impostor.apply_linear = types.MethodType(lambda self, *a: None, impostor)
    impostor.apply_linear.__func__.__name__ = "_head_apply_linear"
    assert target_head.context_kv_head_fork(impostor) == (False, None)
    other = _llm(False).lm_head
    other.apply_linear = types.MethodType(
        _lane("context_kv_fork")._head_apply_linear, _llm(False).lm_head
    )
    assert target_head.context_kv_head_fork(other) == (False, None)
    assert target_head.context_kv_head_fork(_llm(False).lm_head) == (True, None)

    worker, head = _worker(inbox=True)
    slot = torch.empty(ROWS, VOCAB // 8, dtype=torch.bfloat16)
    head.workspace.slot = lambda rows: slot
    head.workspace.gathered = torch.empty(ROWS, VOCAB, dtype=torch.bfloat16)

    def gemm(lm, x, out):
        assert lm is lm_head and out is slot
        order.append(("gemm", x.shape[0]))

    monkeypatch.setattr(target_head, "lm_head_gemm_into", gemm)
    logits = head.produce(torch.zeros(ROWS, 16, dtype=torch.bfloat16), lm_head)
    assert order == [("gemm", ROWS), ("fire", "head")]
    assert logits is head.pending and head.pending_inbox
    head.pending = None
    # Before the context-K/V fork attached its fork (no _context_kv_fork yet): admitted, nothing to fire.
    del lm_head._context_kv_fork
    order.clear()
    head.produce(torch.zeros(ROWS, 16, dtype=torch.bfloat16), lm_head)
    assert order == [("gemm", ROWS)]


def test_the_install_check_on_the_served_layout(monkeypatch):
    monkeypatch.setattr(target_head._low_m, "LOW_M_GEMM_ACTIVE", True)
    mapping = types.SimpleNamespace(enable_attention_dp=False, tp_size=8)
    assert target_head.unsupported_reason(_llm(False), mapping, gather=True) is None
    reason = target_head.unsupported_reason(_llm(True), mapping, gather=True)
    assert reason is not None and "apply_linear is replaced" in reason
    # split_sampler keeps the built-in LM-head call, which runs a replaced apply_linear.
    assert target_head.unsupported_reason(_llm(True), mapping, gather=False) is None
