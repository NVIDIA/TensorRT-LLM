# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks of MTP row ordering, metadata transitions and failure cleanup.

The native sample/glue is replaced by its existing mathematical reference;
these tests exercise the real bridge around it, not GPU kernel numerics.
"""

import contextlib
import inspect
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa_rubin.glm_5_2_nvfp4__sm_107__low_latency import (
    _mtp_worker as worker_module,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa_rubin.glm_5_2_nvfp4__sm_107__low_latency import (
    _small_batch_draft as small,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa_rubin.glm_5_2_nvfp4__sm_107__low_latency._mtp_worker import (  # noqa: E501
    MTPDraftWorker,
)

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def reference_ops(monkeypatch):
    gather = small.glm_gather_last_accepted_rows
    sample = small.glm_draft_sample_glue
    signature = inspect.signature(sample.reference)
    table = small.glm_draft_ehproj_table
    table_signature = inspect.signature(table.reference)
    calls = []

    def sample_reference(logits, *args):
        # Rank zero wins; other shards still participate in the global argmax.
        bound = signature.bind(logits, *args)
        bound.arguments["logits"] = [logits] + [
            torch.full_like(logits, -float("inf")) for _ in range(bound.arguments["group_size"] - 1)
        ]
        outputs = sample.reference(*bound.args, **bound.kwargs)
        for name, value in outputs.items():
            bound.arguments[name].copy_(value)
        calls.append(("sample", bound.arguments["draft_step"], outputs))

    def table_reference(logits, *args):
        bound = table_signature.bind(logits, *args)
        world = bound.arguments["group_size"]
        bound.arguments["logits"] = [logits] + [
            torch.full_like(logits, -float("inf")) for _ in range(world - 1)
        ]
        bound.arguments["hidden_half"] = [bound.arguments["hidden_half"]] * world
        bound.arguments["token_table"] = [bound.arguments["token_table"]] * world
        outputs = table.reference(*bound.args, **bound.kwargs)
        for name, value in outputs.items():
            bound.arguments[name].copy_(value)
        calls.append(("table", bound.arguments["draft_step"], outputs))

    monkeypatch.setattr(small, "glm_gather_last_accepted_rows", gather.reference)
    monkeypatch.setattr(small, "glm_draft_sample_glue", sample_reference)
    monkeypatch.setattr(small, "glm_draft_ehproj_table", table_reference)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda: SimpleNamespace(wait_event=lambda event: None)
    )
    return calls


class _Metadata:
    def __init__(self, batch, k=5):
        self._seq_lens = torch.full((batch,), k + 1, dtype=torch.int32)
        self.seq_lens_cuda = self._seq_lens.clone()
        self.kv_lens_cuda = torch.arange(batch, dtype=torch.int32) + 100
        self.num_contexts = 0
        self.num_tokens = batch * (k + 1)
        self.kv_cache_manager = "target"
        self.use_spec_decoding = True
        self.in_mtp_draft_loop = False
        self.indexer_skip_topk = False
        self.mtp_num_accepted = None
        self.updates = []

    def set_in_mtp_draft_loop(self, value):
        self.in_mtp_draft_loop = value

    def set_skip_topk(self, value):
        self.indexer_skip_topk = value

    def set_mtp_num_accepted(self, value):
        self.mtp_num_accepted = value

    def on_update(self):
        self.num_tokens = int(self._seq_lens.sum())

    def update_for_spec_dec(self):
        self.updates.append((self.seq_lens_cuda.clone(), self.kv_lens_cuda.clone()))


def _loop_fixture(batch, fail_step=None, eligible=True, tp_size=4, k=None, accepted=None):
    k = (3 if tp_size == 8 else 5) if k is None else k
    hidden = 8 if tp_size == 8 else 4
    rows = batch * (k + 1)
    metadata = _Metadata(batch, k)
    # Permutation and heterogeneous accepted counts expose double selection and
    # selection of the last verify row instead of the last accepted row.
    order = torch.tensor(
        [2, 0, 1][:batch] if batch == 3 else list(reversed(range(batch))), dtype=torch.int32
    )
    counts = torch.tensor(
        [1, k + 1, 3][:batch] if accepted is None else accepted, dtype=torch.int32
    )
    spec = SimpleNamespace(runtime_draft_len=k, batch_indices_cuda=order)
    ids = torch.arange(rows, dtype=torch.int32)
    positions = torch.arange(rows, dtype=torch.int64) + 1000
    states = torch.arange(rows, dtype=torch.bfloat16).unsqueeze(1).expand(rows, hidden).clone()
    inputs = dict(
        input_ids=ids,
        position_ids=positions,
        hidden_states=states,
        attn_metadata=metadata,
        spec_metadata=spec,
    )
    helper = object.__new__(small.SmallBatchDraftKernels)
    helper.use_table = tp_size == 8
    helper.DRAFT_LEN = k
    helper.hidden_size = hidden
    helper.window = (0, 2)
    helper.decode = SimpleNamespace(mapping=SimpleNamespace(tp_size=tp_size))
    helper.sample_exchange = SimpleNamespace(buffer=torch.empty(0), flags=torch.empty(0))
    helper.embed_table = torch.ones(32, hidden, dtype=torch.bfloat16)
    helper.enorm_weight = torch.ones(hidden, dtype=torch.bfloat16)
    helper.hnorm_weight = torch.ones(hidden, dtype=torch.bfloat16)
    helper.layer = SimpleNamespace(
        enorm=SimpleNamespace(variance_epsilon=1e-5),
        input_layernorm=SimpleNamespace(weight=torch.ones(hidden), variance_epsilon=1e-5),
    )
    helper.table_exchange = SimpleNamespace(buffer=torch.empty(0), flags=torch.empty(0))
    helper.token_table = torch.arange(8 * hidden, dtype=torch.bfloat16).reshape(8, hidden) + 1
    helper.half_done = object()
    helper.head_inputs = []
    helper.projected_inputs = []
    calls = []
    preflights = []

    def supports(meta, first_rows, selected_rows):
        preflights.append((meta.kv_cache_manager, first_rows, selected_rows))
        return eligible

    def head(value, *, next_step=False):
        helper.head_inputs.append(value.clone())
        logits = torch.zeros(batch, 8, dtype=torch.bfloat16)
        logits.scatter_(1, value[:, :1].long() % 8, 1)
        return logits, value[:, :1].float() if helper.use_table and next_step else None

    helper.supports = supports
    helper.head = head

    @contextlib.contextmanager
    def draft_context(meta, manager):
        saved = meta.kv_cache_manager
        meta.kv_cache_manager = manager
        try:
            yield meta
        finally:
            meta.kv_cache_manager = saved

    def run_forward(draft_model, step_inputs, spec_metadata, step):
        calls.append(
            (
                step,
                step_inputs["hidden_states"].shape[0],
                step_inputs["position_ids"].clone(),
                metadata.kv_lens_cuda.clone(),
                metadata.indexer_skip_topk,
            )
        )
        if step == fail_step:
            raise RuntimeError("injected draft failure")
        projected = step_inputs.get("eh_proj_output")
        helper.projected_inputs.append(projected)
        value = step_inputs["hidden_states"] if projected is None else projected[0]
        select = step_inputs.get("row_select")
        if select is not None:
            value, _ = select(value, value + 100)
        return value, None

    worker = SimpleNamespace(
        _drafted_rows=None, draft_kv_cache_context=draft_context, _run_draft_forward=run_forward
    )
    return worker, helper, inputs, metadata, spec, counts, calls, preflights


@pytest.mark.parametrize("batch", [1, 2, 3])
def test_loop_selects_once_and_advances_metadata_after_every_step(reference_ops, batch):
    worker, helper, inputs, meta, spec, counts, calls, preflights = _loop_fixture(batch)
    original_positions = inputs["position_ids"].clone()
    original_kv = meta.kv_lens_cuda.clone()
    got = MTPDraftWorker._forward_small_batch_draft_loop(
        worker, helper, inputs, meta, spec, object(), "draft", batch, counts
    )
    selected = spec.batch_indices_cuda.long() * 6 + counts.long() - 1
    expected_tokens = (selected % 8).to(torch.int32).unsqueeze(1).expand(batch, 5)
    torch.testing.assert_close(got, expected_tokens)
    assert preflights == [("draft", batch * 6, batch)]
    assert [call[1] for call in calls] == [batch * 6, batch, batch, batch, batch]
    assert [call[4] for call in calls] == [False, True, True, True, True]
    for step in range(1, 5):
        torch.testing.assert_close(calls[step][2], original_positions[selected] + step)
        torch.testing.assert_close(calls[step][3], original_kv - (5 - counts) + step - 1)
    assert len(meta.updates) == 5
    for step, (seq_lens, kv_lens) in enumerate(meta.updates):
        torch.testing.assert_close(seq_lens, torch.ones(batch, dtype=torch.int32))
        torch.testing.assert_close(kv_lens, original_kv - (5 - counts) + step)
    assert meta.num_tokens == batch and not meta.use_spec_decoding
    assert meta.kv_cache_manager == "target"
    assert not meta.in_mtp_draft_loop and not meta.indexer_skip_topk
    assert meta.mtp_num_accepted is None
    assert worker._drafted_rows[0] is got
    torch.testing.assert_close(worker._drafted_rows[1][:, 0], selected.to(torch.int32))
    torch.testing.assert_close(worker._drafted_rows[1][:, 1:], expected_tokens)
    # The original inputs remain available to outer bookkeeping.
    assert "owned_draft" not in inputs and "row_select" not in inputs


@pytest.mark.parametrize(
    "batch,k,accepted", [(1, 3, [1]), (2, 3, [1, 4]), (1, 7, [1]), (1, 7, [8])]
)
def test_tp8_table_feedback_and_final_glue(reference_ops, batch, k, accepted):
    worker, helper, inputs, meta, spec, counts, calls, preflights = _loop_fixture(
        batch, tp_size=8, k=k, accepted=accepted
    )
    original_kv = meta.kv_lens_cuda.clone()
    got = MTPDraftWorker._forward_small_batch_draft_loop(
        worker, helper, inputs, meta, spec, object(), "draft", batch, counts
    )
    selected = spec.batch_indices_cuda.long() * (k + 1) + counts.long() - 1
    assert preflights == [("draft", batch * (k + 1), batch)]
    assert [call[1] for call in calls] == [batch * (k + 1)] + [batch] * (k - 1)
    assert [call[4] for call in calls] == [False] + [True] * (k - 1)
    assert [(kind, step) for kind, step, _ in reference_ops] == [
        ("table", step) for step in range(k - 1)
    ] + [("sample", k - 1)]
    assert helper.projected_inputs[0] is None
    torch.testing.assert_close(helper.head_inputs[0], inputs["hidden_states"][selected])
    for step in range(1, k):
        prior = reference_ops[step - 1][2]
        projected = helper.projected_inputs[step]
        torch.testing.assert_close(projected[0], prior["normed_out"])
        torch.testing.assert_close(projected[1], prior["residual_out"])
        torch.testing.assert_close(helper.head_inputs[step], prior["normed_out"])
        torch.testing.assert_close(calls[step][2], inputs["position_ids"][selected] + step)
    assert len(meta.updates) == k
    for step, (seq_lens, kv_lens) in enumerate(meta.updates):
        torch.testing.assert_close(seq_lens, torch.ones(batch, dtype=torch.int32))
        torch.testing.assert_close(kv_lens, original_kv - (k - counts) + step)
    assert tuple(worker._drafted_rows[1].shape) == (batch, k + 1)
    torch.testing.assert_close(worker._drafted_rows[1][:, 0], selected.to(torch.int32))
    torch.testing.assert_close(worker._drafted_rows[1][:, 1:], got)
    assert meta.kv_cache_manager == "target" and not meta.use_spec_decoding
    assert not meta.in_mtp_draft_loop and not meta.indexer_skip_topk
    assert meta.mtp_num_accepted is None


def test_projected_output_bypasses_glue_gemm_and_input_hop(monkeypatch):
    helper = object.__new__(small.SmallBatchDraftKernels)
    helper.layer = object()
    seen = []

    def forbidden(*args, **kwargs):
        raise AssertionError("projected rows must not execute the initial projection again")

    def attention(layer, positions, hidden, metadata, sharded, *, draft_state):
        seen.append(hidden)
        return hidden + 1

    helper.decode = SimpleNamespace(
        _hop=forbidden,
        _attention=attention,
        _post_attention_hop=lambda layer, hidden, residual: (hidden, residual),
    )
    helper._moe_postnorm = lambda hidden, residual: hidden + residual
    monkeypatch.setattr(small, "glm_draft_glue_rmsnorm_cat", forbidden)
    monkeypatch.setattr(small, "low_m_gemm", forbidden)
    normed, residual = torch.ones(2, 8), torch.full((2, 8), 3.0)
    got = helper.forward_step(
        None, None, torch.full_like(normed, -10), object(), eh_proj_output=(normed, residual)
    )
    assert seen == [normed]
    torch.testing.assert_close(got, normed + 1 + residual)
    with pytest.raises(ValueError, match="mutually exclusive"):
        helper.forward_step(
            None, None, normed, object(), eh_proj_input=normed, eh_proj_output=(normed, residual)
        )
    assert len(seen) == 1


@pytest.mark.parametrize("rows", [1, 2])
def test_tp8_head_launch_order_and_weighted_norm(monkeypatch, rows):
    calls = []
    helper = object.__new__(small.SmallBatchDraftKernels)
    helper.use_table = True
    helper.hidden_rows = torch.ones(1, 8, dtype=torch.bfloat16)
    helper.hnorm_weight = torch.full((8,), 2.0, dtype=torch.bfloat16)
    helper.half_start = SimpleNamespace(record=lambda: calls.append("fork"))
    helper.half_done = SimpleNamespace(record=lambda: calls.append("ready"))
    aux = SimpleNamespace(wait_event=lambda event: calls.append("aux_wait"))

    class Norm:
        variance_epsilon = 1e-5

        def __call__(self, hidden):
            calls.append("weighted_hnorm")
            return hidden * helper.hnorm_weight

    helper.layer = SimpleNamespace(mlp=SimpleNamespace(aux_stream=aux), hnorm=Norm())
    helper.lm_head = object()

    def lm_head(lm, value):
        calls.append("lm_head")
        return value

    def half(value, weight, out, norm_weight, eps):
        calls.append("hidden_half")
        assert out.dtype == torch.float32
        assert (norm_weight is helper.hnorm_weight) == (rows == 1)
        torch.testing.assert_close(value, torch.full_like(value, 1 if rows == 1 else 2))
        return out.fill_(7)

    helper.decode = SimpleNamespace(_lm_head_shard=lm_head)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: contextlib.nullcontext())
    monkeypatch.setattr(small, "draft_hidden_half", half)
    hidden = torch.ones(rows, 8, dtype=torch.bfloat16)
    logits, projected = helper.head(hidden, next_step=True)
    assert logits is hidden and tuple(projected.shape) == (rows, 1)
    assert calls == (
        ["fork", "aux_wait", "hidden_half", "ready", "lm_head"]
        if rows == 1
        else ["fork", "lm_head", "aux_wait", "weighted_hnorm", "hidden_half", "ready"]
    )


@pytest.mark.parametrize("tp_size,batch,k,fail_step", [(4, 3, 5, 2), (8, 2, 3, 2), (8, 1, 7, 6)])
def test_exception_restores_draft_manager_and_index_share_without_replay(
    reference_ops, tp_size, batch, k, fail_step
):
    worker, helper, inputs, meta, spec, counts, calls, _ = _loop_fixture(
        batch, fail_step=fail_step, tp_size=tp_size, k=k
    )
    with pytest.raises(RuntimeError, match="injected draft failure"):
        MTPDraftWorker._forward_small_batch_draft_loop(
            worker, helper, inputs, meta, spec, object(), "draft", batch, counts
        )
    assert [call[0] for call in calls] == list(range(fail_step + 1))
    assert len(meta.updates) == fail_step
    assert meta.kv_cache_manager == "target"
    assert not meta.in_mtp_draft_loop and not meta.indexer_skip_topk
    assert meta.mtp_num_accepted is None and worker._drafted_rows is None


@pytest.mark.parametrize("tp_size,batch,k", [(4, 3, 5), (8, 2, 3), (8, 1, 7)])
def test_rebound_manager_preflight_declines_before_any_forward(reference_ops, tp_size, batch, k):
    worker, helper, inputs, meta, spec, counts, calls, preflights = _loop_fixture(
        batch, eligible=False, tp_size=tp_size, k=k
    )
    original_kv = meta.kv_lens_cuda.clone()
    got = MTPDraftWorker._forward_small_batch_draft_loop(
        worker, helper, inputs, meta, spec, object(), "draft", batch, counts
    )
    assert got is None and calls == [] and meta.updates == []
    assert preflights == [("draft", batch * (spec.runtime_draft_len + 1), batch)]
    torch.testing.assert_close(meta.kv_lens_cuda, original_kv)
    assert meta.kv_cache_manager == "target" and not meta.in_mtp_draft_loop
    assert meta.mtp_num_accepted is None


@pytest.mark.parametrize("batch,k", [(3, 5), (2, 3), (1, 3), (1, 7)])
def test_first_attention_precedes_selection_of_hidden_and_residual(
    reference_ops, monkeypatch, batch, k
):
    helper = object.__new__(small.SmallBatchDraftKernels)
    helper.hidden_size = 4
    helper.zero_residual = torch.zeros(24, 4)
    helper.layer = SimpleNamespace(
        eh_proj=SimpleNamespace(weight=torch.eye(4)), input_layernorm=object()
    )
    seen = []

    def attention(layer, positions, value, metadata, sharded, *, draft_state):
        assert not sharded and draft_state is helper
        seen.append(("attention", value.clone()))
        return value + 10

    helper.decode = SimpleNamespace(
        _hop=lambda norm, partial, residual: (partial, partial + 100),
        _attention=attention,
        _post_attention_hop=lambda layer, output, residual: (output * 2, residual + 3),
    )

    def moe(value, residual):
        seen.append(("moe", value.clone(), residual.clone()))
        return value

    helper._moe_postnorm = moe
    monkeypatch.setattr(small, "low_m_gemm", lambda x, weight, out: x @ weight.T)
    full = torch.arange(batch * (k + 1) * 4, dtype=torch.float32).reshape(-1, 4)
    order = torch.tensor([2, 0, 1] if batch == 3 else list(reversed(range(batch))))
    counts = torch.tensor([1, k + 1, 3][:batch])
    select = small.FirstDraftRowSelect(order, counts, k + 1)
    got = helper.forward_step(None, None, full, object(), eh_proj_input=full, row_select=select)
    expected_ids = order * (k + 1) + counts - 1
    torch.testing.assert_close(seen[0][1], full)
    torch.testing.assert_close(seen[1][1], (full + 10)[expected_ids] * 2)
    torch.testing.assert_close(seen[1][2], (full + 103)[expected_ids])
    torch.testing.assert_close(got, seen[1][1])
    with pytest.raises(RuntimeError, match="already selected"):
        select(full, full)


def test_tp16_inline_dispatch_precedes_small_batch_branch():
    sentinel = object()

    def no_small_batch(*args):
        raise AssertionError("the old inline path must win")

    worker = SimpleNamespace(
        _drafting_target=lambda *args: sentinel,
        _draft_on_target=lambda *args: "inline",
        _small_batch_target=no_small_batch,
    )
    result = MTPDraftWorker._forward_linear_draft_loop(
        worker, {}, object(), object(), object(), None, 0, 1, object(), None
    )
    assert result == "inline"


@pytest.mark.parametrize("error_type", [MemoryError, OSError, RuntimeError])
@pytest.mark.parametrize("tp_size", [4, 8])
def test_local_draft_state_failure_reaches_group_vote(monkeypatch, error_type, tp_size):
    comm = object()
    votes = []
    decode = SimpleNamespace(
        mapping=SimpleNamespace(tp_size=tp_size),
        core=SimpleNamespace(
            model_config=SimpleNamespace(
                spec_config=SimpleNamespace(max_draft_len=3 if tp_size == 8 else 5)
            )
        ),
    )
    monkeypatch.setattr(small, "_get_mnnvl_workspace_comm", lambda mapping: comm)
    monkeypatch.setattr(small, "_static_layout_supported", lambda *args: True)

    def vote(actual_comm, ok):
        assert actual_comm is comm
        votes.append(ok)
        return ok

    def fail(*args, **kwargs):
        raise error_type("local allocation failed")

    monkeypatch.setattr(small, "_all_succeeded", vote)
    monkeypatch.setattr(small.DraftExchangeWorkspace, "for_sample_glue", lambda mapping: object())
    monkeypatch.setattr(small.DraftExchangeWorkspace, "for_ehproj_table", lambda mapping: object())
    monkeypatch.setattr(small, "SmallBatchDraftKernels", fail)
    with pytest.raises(RuntimeError, match="failed on at least one rank") as caught:
        small.build_small_batch_draft(decode, object(), object(), object())
    assert isinstance(caught.value.__cause__, error_type)
    assert votes == [True, False]


def test_tp8_token_table_build_failure_reaches_group_vote(monkeypatch):
    votes = []
    mapping = SimpleNamespace(tp_size=8, tp_rank=0)
    norm = SimpleNamespace(weight=torch.ones(8, dtype=torch.bfloat16), variance_epsilon=1e-5)
    decode = SimpleNamespace(
        mapping=mapping,
        core=SimpleNamespace(
            embed_tokens=SimpleNamespace(weight=torch.ones(32, 8, dtype=torch.bfloat16)),
            model_config=SimpleNamespace(
                spec_config=SimpleNamespace(max_draft_len=3),
                pretrained_config=SimpleNamespace(
                    hidden_size=8, num_experts_per_tok=8, routed_scaling_factor=2.5, index_topk=2
                ),
            ),
        ),
    )
    layer = SimpleNamespace(enorm=norm, hnorm=norm)
    weights = SimpleNamespace(w3_w1=object(), w2=object(), eh_proj=object())
    lm_head = SimpleNamespace(weight=torch.ones(4, 8, dtype=torch.bfloat16))
    monkeypatch.setattr(small, "_get_mnnvl_workspace_comm", lambda mapping: object())
    monkeypatch.setattr(small, "_static_layout_supported", lambda *args: True)
    monkeypatch.setattr(small, "_all_succeeded", lambda comm, ok: votes.append(ok) or ok)
    monkeypatch.setattr(small, "get_env_enable_pdl", lambda: True)
    monkeypatch.setattr(small.DraftExchangeWorkspace, "for_sample_glue", lambda mapping: object())
    monkeypatch.setattr(small.DraftExchangeWorkspace, "for_ehproj_table", lambda mapping: object())

    def fail_table(*args):
        raise MemoryError("injected token-table allocation failure")

    monkeypatch.setattr(small, "build_draft_token_table", fail_table)
    with pytest.raises(RuntimeError, match="failed on at least one rank") as caught:
        small.build_small_batch_draft(decode, layer, lm_head, weights)
    assert isinstance(caught.value.__cause__, MemoryError)
    assert votes == [True, False]


@pytest.mark.parametrize(
    "tp,batch,configured_k,runtime_k,helper_k,eligible",
    [
        (4, 3, 5, 5, 5, True),
        (8, 2, 3, 3, 3, True),
        (8, 1, 7, 7, 7, True),
        (8, 2, 7, 7, 7, False),
        (8, 1, 5, 5, 5, False),
        (4, 1, 7, 7, 7, False),
        (8, 1, 7, 3, 7, False),
        (8, 1, 7, 7, 3, False),
    ],
)
def test_worker_checks_configured_runtime_and_allocated_draft_shape(
    monkeypatch, tp, batch, configured_k, runtime_k, helper_k, eligible
):
    # Tensor metadata stands in for device tensors; the real worker eligibility
    # predicate runs before any kernel or collective can use these operands.
    def operand(shape, dtype):
        return SimpleNamespace(
            shape=shape,
            dtype=dtype,
            device="cuda",
            is_cuda=True,
            is_contiguous=lambda: True,
            data_ptr=lambda: 32,
            dim=lambda: len(shape),
            numel=lambda: shape[0],
        )

    rows = batch * (runtime_k + 1)
    meta = _Metadata(batch, runtime_k)
    monkeypatch.setattr(worker_module, "MTPIndexShareMetadata", _Metadata)
    layer, lm_head = object(), object()
    helper = SimpleNamespace(
        layer=layer,
        lm_head=lm_head,
        DRAFT_LEN=helper_k,
        MAX_BATCH=3 if tp == 4 else (1 if helper_k == 7 else 2),
    )
    worker = SimpleNamespace(
        mapping=SimpleNamespace(tp_size=tp),
        max_draft_len=configured_k,
        is_mtp_eagle=True,
        spec_config=SimpleNamespace(draft_len_schedule=None),
        guided_decoder=None,
        sa_enhancer=None,
        _d2t=None,
        _decode_target=lambda metadata: SimpleNamespace(_small_batch_draft=helper),
    )
    spec = SimpleNamespace(
        runtime_draft_len=runtime_k,
        batch_indices_cuda=operand((batch,), torch.int32),
        wants_advanced_draft_sampling=False,
        is_ragged_verify=False,
    )
    inputs = dict(
        input_ids=operand((rows,), torch.int32),
        position_ids=operand((rows,), torch.int64),
        hidden_states=operand((rows, 6144), torch.bfloat16),
    )
    got = MTPDraftWorker._small_batch_target(
        worker,
        inputs,
        meta,
        spec,
        SimpleNamespace(mtp_layers=[layer], lm_head=lm_head),
        0,
        batch,
        operand((batch,), torch.int32),
    )
    assert got is (helper if eligible else None)


@pytest.mark.parametrize("configured_k", [None, 5, 8, [7]])
def test_unsupported_tp8_draft_config_votes_before_workspace_allocation(monkeypatch, configured_k):
    votes = []
    decode = SimpleNamespace(
        mapping=SimpleNamespace(tp_size=8),
        core=SimpleNamespace(
            model_config=SimpleNamespace(
                spec_config=None
                if configured_k is None
                else SimpleNamespace(max_draft_len=configured_k)
            )
        ),
    )
    monkeypatch.setattr(small, "_get_mnnvl_workspace_comm", lambda mapping: object())
    monkeypatch.setattr(small, "_all_succeeded", lambda comm, ok: votes.append(ok) or ok)
    monkeypatch.setattr(small, "_static_layout_supported", lambda *args: True)

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported draft configuration allocated a workspace")

    monkeypatch.setattr(small.DraftExchangeWorkspace, "for_sample_glue", forbidden)
    monkeypatch.setattr(small.DraftExchangeWorkspace, "for_ehproj_table", forbidden)
    assert small.build_small_batch_draft(decode, object(), object(), object()) is None
    assert votes == [False]
