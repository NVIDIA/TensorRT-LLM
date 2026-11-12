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
"""Draft step 0 runs the draft layer's post-attention part only on the rows
the draft loop reads (``pick_rows``).

The draft tokens and the hidden states carried to step 1 must match the
unpicked run, the attention (draft KV) must still see every row, and a draft
layer without ``pick_rows`` support must run exactly as before.
"""

import contextlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch.speculative.eagle3 import Eagle3OneModelWorker
from tensorrt_llm._torch.speculative.interface import DraftRowPick, draft_layer_picks_rows
from tensorrt_llm._torch.speculative.mtp import MTPWorker

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

H = 64
VOCAB = 97


class _Meta:
    """The attention-metadata surface the linear draft loop touches."""

    def __init__(self, seq_lens, num_contexts, kv_lens):
        self._seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
        self._seq_lens_cuda = self._seq_lens.cuda()
        self.kv_lens_cuda = torch.tensor(kv_lens, dtype=torch.int32, device="cuda")
        self.num_contexts = num_contexts
        self.num_seqs = len(seq_lens)
        self.num_ctx_tokens = int(sum(seq_lens[:num_contexts]))
        self.kv_cache_manager = None
        self.enable_flash_mla = False
        self.use_spec_decoding = True
        self.all_rank_num_tokens = None

    @property
    def seq_lens_cuda(self):
        return self._seq_lens_cuda

    def on_update(self):
        pass

    def update_for_spec_dec(self):
        pass


class _DraftAttnLayer(nn.Module):
    """A toy draft layer: "attention" mixes every input row (a running mean
    over the flattened rows) and is logged as the draft KV it writes; the rest
    is per-token (residual, MLP, norm)."""

    def __init__(self, seed):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.embed = (torch.randn(VOCAB, H, generator=g) * 0.5).cuda()
        self.w = (torch.randn(H, H, generator=g) / H**0.5).cuda()
        self.head = torch.randn(H, VOCAB, generator=g).cuda()
        self.kv_writes = []  # every row the attention saw, per call
        self.post_rows = []  # rows the post-attention part ran on, per call
        self.inputs = []  # hidden input rows, per call
        self.all_rank_num_tokens = []  # the all_rank_num_tokens argument, per call

    def _attention(self, input_ids, position_ids, hidden_states):
        x = (
            hidden_states
            + self.embed[input_ids.long()]
            + 0.01 * (position_ids.reshape(-1, 1).float())
        )
        self.kv_writes.append(x.clone())
        n = torch.arange(1, x.shape[0] + 1, device=x.device).unsqueeze(1)
        return x.cumsum(0) / n, x

    def _post(self, attn_out, residual):
        self.post_rows.append(attn_out.shape[0])
        y = torch.tanh(attn_out @ self.w) + residual
        return y / y.norm(dim=-1, keepdim=True)

    def shared_head(self, hidden_states, lm_head, attn_metadata, return_context_logits=False):
        if not return_context_logits:  # the request's last row, as DeepseekV3MTPHead
            last = torch.cumsum(attn_metadata.seq_lens_cuda, dim=0, dtype=torch.long) - 1
            hidden_states = hidden_states[last]
        return hidden_states @ self.head


class _PickingMtpLayer(_DraftAttnLayer):
    supports_pick_rows = True

    def forward(
        self,
        input_ids,
        position_ids,
        hidden_states,
        embed_tokens,
        attn_metadata,
        all_rank_num_tokens=None,
        spec_metadata=None,
        pick_rows=None,
    ):
        self.inputs.append(hidden_states.clone())
        self.all_rank_num_tokens.append(all_rank_num_tokens)
        attn_out, residual = self._attention(input_ids, position_ids, hidden_states)
        if pick_rows is not None:
            attn_out, residual = attn_out[pick_rows], residual[pick_rows]
        return self._post(attn_out, residual)


class _PlainMtpLayer(_DraftAttnLayer):
    """No ``pick_rows`` support: must never be given one."""

    def forward(
        self,
        input_ids,
        position_ids,
        hidden_states,
        embed_tokens,
        attn_metadata,
        all_rank_num_tokens=None,
        spec_metadata=None,
        **kwargs,
    ):
        assert "pick_rows" not in kwargs
        self.inputs.append(hidden_states.clone())
        self.all_rank_num_tokens.append(all_rank_num_tokens)
        attn_out, residual = self._attention(input_ids, position_ids, hidden_states)
        return self._post(attn_out, residual)


class _OverridingMtpLayer(_PickingMtpLayer):
    """Inherits the capability attribute but overrides ``forward`` without
    ``pick_rows``: must be treated as unsupported."""

    def forward(
        self, input_ids, position_ids, hidden_states, embed_tokens, attn_metadata, **kwargs
    ):
        assert "pick_rows" not in kwargs
        return super().forward(input_ids, position_ids, hidden_states, embed_tokens, attn_metadata)


class _Eagle3Model(_DraftAttnLayer):
    """Eagle3DraftModel-style: returns (hidden, hidden_to_save)."""

    def __init__(self, seed, supports):
        super().__init__(seed)
        self.supports_pick_rows = supports

    def forward(
        self,
        attn_metadata,
        input_ids,
        position_ids,
        hidden_states,
        spec_metadata,
        all_rank_num_tokens=None,
        pick_rows=None,
    ):
        assert pick_rows is None or self.supports_pick_rows
        self.inputs.append(hidden_states.clone())
        self.all_rank_num_tokens.append(all_rank_num_tokens)
        attn_out, residual = self._attention(input_ids, position_ids, hidden_states)
        if pick_rows is not None:
            attn_out, residual = attn_out[pick_rows], residual[pick_rows]
        y = self._post(attn_out, residual)
        return y, y + residual


def _worker(is_mtp_eagle, fast_paths):
    w = object.__new__(Eagle3OneModelWorker)
    w.is_mtp_eagle = is_mtp_eagle
    w.spec_decode_fast_paths = fast_paths
    w.guided_decoder = None
    w.sa_enhancer = None
    w.model_config = None
    w.mapping = None
    w.draft_head = None
    w._d2t = None
    w._draft_folds_kv_lens_increment = False
    w._step0_row_pick = DraftRowPick("test draft layer")
    w.draft_kv_cache_context = lambda md, mgr: contextlib.nullcontext(md)
    w.sample_draft_tokens = (
        lambda logits,
        spec_metadata,
        batch_size,
        draft_step,
        mapping_lm_head_tp=None: logits.argmax(-1).to(torch.int32)
    )
    return w


def _run_loop(
    layer, is_mtp_eagle, fast_paths, num_contexts, num_gens, k, seed, all_rank=None, subseq=None
):
    g = torch.Generator().manual_seed(seed)
    ctx_lens = torch.randint(2, 9, (num_contexts,), generator=g).tolist()
    seq_lens = ctx_lens + [k + 1] * num_gens
    b = num_contexts + num_gens
    n = sum(seq_lens)
    num_acc = torch.cat(
        [
            torch.ones(num_contexts, dtype=torch.int32),
            torch.randint(1, k + 2, (num_gens,), generator=g, dtype=torch.int32),
        ]
    ).cuda()
    accepted = torch.randint(0, VOCAB, (b, k + 1), generator=g, dtype=torch.int32).cuda()
    md = _Meta(seq_lens, num_contexts, torch.randint(50, 90, (b,), generator=g).tolist())
    inputs = {
        "input_ids": torch.randint(0, VOCAB, (n,), generator=g, dtype=torch.int32).cuda(),
        "position_ids": torch.randint(0, 1000, (n,), generator=g, dtype=torch.int32).cuda(),
        "hidden_states": torch.randn(n, H, generator=g).cuda(),
        "attn_metadata": md,
        "spec_metadata": None,
    }
    spec_metadata = SimpleNamespace(
        runtime_draft_len=k,
        batch_indices_cuda=torch.arange(b, dtype=torch.int32, device="cuda"),
        wants_advanced_draft_sampling=False,
        all_rank_num_tokens=all_rank,
        subseq_all_rank_num_tokens=subseq,
        max_num_requests=b,
    )
    inputs["spec_metadata"] = spec_metadata
    if is_mtp_eagle:
        draft_model = SimpleNamespace(mtp_layers=[layer], embed_tokens=None, lm_head=None)
    else:
        draft_model = SimpleNamespace(
            model=layer,
            lm_head=None,
            logits_processor=lambda h, lm_head, md, ctx: layer.shared_head(h, lm_head, md, ctx),
        )
    worker = _worker(is_mtp_eagle, fast_paths)
    tokens, _ = worker._forward_linear_draft_loop(
        inputs,
        md,
        spec_metadata,
        draft_model,
        None,
        num_contexts,
        b,
        num_acc,
        None,
        accepted_tokens=accepted,
        pre_draft_seq_lens=md._seq_lens_cuda.clone(),
    )
    torch.cuda.synchronize()
    return tokens, md.kv_lens_cuda.clone(), worker._step0_row_pick._picks_rows


def _compare(ref_layer, ref, layer, out, b, n_rows):
    ref_tokens, ref_kv, ref_picked = ref
    tokens, kv, picked = out
    assert not ref_picked
    assert torch.equal(tokens, ref_tokens)
    assert torch.equal(kv, ref_kv)
    # The draft KV of every row is written on every step.
    assert len(layer.kv_writes) == len(ref_layer.kv_writes)
    for got, want in zip(layer.kv_writes, ref_layer.kv_writes):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
    assert layer.kv_writes[0].shape[0] == n_rows
    # Step 1's carried hidden states.
    for got, want in zip(layer.inputs[1:], ref_layer.inputs[1:]):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)
    assert ref_layer.post_rows[0] == n_rows
    assert all(r == b for r in ref_layer.post_rows[1:])
    return picked


@pytest.mark.parametrize("fast_paths", [False, True])
# Mixed contexts and generations; contexts only with a single draft step.
@pytest.mark.parametrize("num_contexts,num_gens,k", [(2, 3, 3), (3, 0, 1)])
def test_mtp_eagle_step0_pick_matches_full(fast_paths, num_contexts, num_gens, k):
    b = num_contexts + num_gens
    ref_layer = _PlainMtpLayer(seed=7)
    ref = _run_loop(ref_layer, True, fast_paths, num_contexts, num_gens, k, 3)
    layer = _PickingMtpLayer(seed=7)
    out = _run_loop(layer, True, fast_paths, num_contexts, num_gens, k, 3)
    n_rows = ref_layer.kv_writes[0].shape[0]
    assert _compare(ref_layer, ref, layer, out, b, n_rows)
    # The post-attention part ran on one row per request on every step.
    assert layer.post_rows == [b] * k


def test_eagle3_step0_pick_matches_full():
    num_contexts, num_gens, k = 0, 4, 2
    b = num_contexts + num_gens
    ref_layer = _Eagle3Model(seed=11, supports=False)
    ref = _run_loop(ref_layer, False, False, num_contexts, num_gens, k, 5)
    layer = _Eagle3Model(seed=11, supports=True)
    out = _run_loop(layer, False, False, num_contexts, num_gens, k, 5)
    n_rows = ref_layer.kv_writes[0].shape[0]
    assert _compare(ref_layer, ref, layer, out, b, n_rows)
    assert layer.post_rows == [b] * k


@pytest.mark.parametrize("layer_cls", [_PlainMtpLayer, _OverridingMtpLayer])
def test_layer_without_pick_support_is_unaffected(layer_cls):
    num_contexts, num_gens, k = 2, 3, 3
    ref_layer = _PlainMtpLayer(seed=7)
    ref = _run_loop(ref_layer, True, True, num_contexts, num_gens, k, 3)
    layer = layer_cls(seed=7)
    out = _run_loop(layer, True, True, num_contexts, num_gens, k, 3)
    n_rows = ref_layer.kv_writes[0].shape[0]
    assert not _compare(ref_layer, ref, layer, out, num_contexts + num_gens, n_rows)
    assert layer.post_rows == ref_layer.post_rows


def _mtp_worker(mapping=None):
    """An ``MTPWorker`` whose steps around the draft loop are stubbed out."""
    w = object.__new__(MTPWorker)
    w.guided_decoder = None
    w.sa_enhancer = None
    w.mapping = mapping
    w._auxiliary_state_handlers = []
    w._row_pick = DraftRowPick("test MTP layers")
    w._execute_guided_decoder_if_present = lambda logits: None
    w._commit_target_mamba_states = lambda *args: None
    w.update_mtp_hidden_states = lambda **kwargs: None
    w.change_attn_metadata = lambda *args: None
    w.get_draft_kv_cache_manager = lambda resource_manager: None
    w.draft_kv_cache_context = lambda md, mgr: contextlib.nullcontext(md)
    w._restore_attn_metadata_from_spec_dec = lambda md: None
    w._prepare_next_new_tokens = lambda *args: None
    w.sample_draft_tokens = lambda logits, spec_metadata, batch_size, draft_step: logits.argmax(
        -1
    ).to(torch.int32)
    return w


def _run_mtp_worker(layer_cls, mapping=None, num_contexts=2, num_gens=3, k=3, seed=9):
    """``MTPWorker.forward``'s draft loop over ``k`` MTP layers, in the
    layout after ``change_attn_metadata`` (``k`` rows per generation)."""
    g = torch.Generator().manual_seed(seed)
    seq_lens = torch.randint(2, 9, (num_contexts,), generator=g).tolist() + [k] * num_gens
    b, n = num_contexts + num_gens, sum(seq_lens)
    md = _Meta(seq_lens, num_contexts, [60] * b)
    layers = [layer_cls(seed=100 + i) for i in range(k)]
    draft_inputs = {
        "input_ids": torch.randint(0, VOCAB, (n,), generator=g, dtype=torch.int32).cuda(),
        "position_ids": torch.randint(0, 1000, (n,), generator=g, dtype=torch.int32).cuda(),
        "hidden_states": torch.randn(n, H, generator=g).cuda(),
        "attn_metadata": md,
    }
    accepted = torch.randint(0, VOCAB, (b, k + 1), generator=g, dtype=torch.int32).cuda()
    num_acc = torch.ones(b, dtype=torch.int32, device="cuda")
    worker = _mtp_worker(mapping)
    worker.sample_and_accept_draft_tokens = lambda *args: (accepted, num_acc)
    worker.prepare_drafter_inputs = lambda **kwargs: draft_inputs
    out = worker._forward_impl(
        draft_inputs["input_ids"],
        draft_inputs["position_ids"],
        draft_inputs["hidden_states"],
        torch.zeros(b, VOCAB, device="cuda"),
        md,
        SimpleNamespace(runtime_draft_len=k, batch_indices_cuda=None),
        SimpleNamespace(mtp_layers=layers, embed_tokens=None, lm_head=None),
    )
    torch.cuda.synchronize()
    return out["next_draft_tokens"], layers, worker._row_pick._picks_rows, b, n


def test_mtp_worker_layers_pick_matches_full():
    """Vanilla MTP: every MTP layer runs after attention only on each
    request's last row; draft tokens and every layer's carried hidden input
    match, and each layer's attention still sees (and writes) every row."""
    ref_tokens, ref_layers, ref_picked, b, n = _run_mtp_worker(_PlainMtpLayer)
    tokens, layers, picked, _, _ = _run_mtp_worker(_PickingMtpLayer)
    assert picked and not ref_picked
    assert torch.equal(tokens, ref_tokens)
    for layer, ref_layer in zip(layers, ref_layers):
        assert layer.post_rows == [b] and ref_layer.post_rows == [n]
        assert [x.shape[0] for x in layer.kv_writes] == [n]
        torch.testing.assert_close(layer.inputs[0], ref_layer.inputs[0], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(layer.kv_writes[0], ref_layer.kv_writes[0], rtol=1e-5, atol=1e-5)


def test_step0_pick_under_attention_dp():
    """Attention DP. The draft loop: a picked step 0 hands the layer the
    per-request counts (``subseq_all_rank_num_tokens``, the counts of its
    post-attention rows), an unpicked one the full step-0 counts, later steps
    always the per-request counts; picking without the per-request counts
    fails. Vanilla MTP declines the pick and runs as before."""
    shape = num_contexts, num_gens, k = 2, 3, 3
    b = num_contexts + num_gens
    full_counts, per_request_counts = [101, 202], [5, 6]
    for layer, is_mtp_eagle, picking in [
        (_PickingMtpLayer(seed=7), True, True),
        (_Eagle3Model(seed=11, supports=False), False, False),
    ]:
        _, _, picked = _run_loop(
            layer, is_mtp_eagle, False, *shape, 3, full_counts, per_request_counts
        )
        assert picked == picking
        step0 = per_request_counts if picking else full_counts
        assert layer.all_rank_num_tokens == [step0] + [per_request_counts] * (k - 1)
        assert layer.post_rows[0] == (b if picking else layer.kv_writes[0].shape[0])
    with pytest.raises(AssertionError):
        _run_loop(_PickingMtpLayer(seed=7), True, False, *shape, 3, full_counts)

    adp = SimpleNamespace(enable_attention_dp=True, cp_size=1)
    tokens, layers, picked, _, n = _run_mtp_worker(_PickingMtpLayer, mapping=adp)
    ref_tokens, _, _, _, _ = _run_mtp_worker(_PlainMtpLayer)
    assert not picked
    assert all(layer.post_rows == [n] for layer in layers)
    assert torch.equal(tokens, ref_tokens)


def test_draft_row_pick_gates(monkeypatch):
    """``draft_layer_picks_rows`` and the ``DraftRowPick`` decision, which is
    resolved again whenever the draft layer classes change."""
    monkeypatch.delenv("TLLM_SPEC_DISABLE_DRAFT_ROW_PICK", raising=False)
    assert draft_layer_picks_rows(_PickingMtpLayer(0))
    assert not draft_layer_picks_rows(None)
    assert not draft_layer_picks_rows(_PlainMtpLayer(0))
    assert not draft_layer_picks_rows(_OverridingMtpLayer(0))
    assert not draft_layer_picks_rows(_Eagle3Model(0, supports=False))
    assert draft_layer_picks_rows(_Eagle3Model(0, supports=True))
    assert not draft_layer_picks_rows(_PickingMtpLayer(0), SimpleNamespace(cp_size=2))
    assert draft_layer_picks_rows(_PickingMtpLayer(0), SimpleNamespace(cp_size=1))

    resolver = DraftRowPick("test")
    assert resolver.resolve([_PickingMtpLayer(0)])
    assert not resolver.resolve([_OverridingMtpLayer(0)])
    assert resolver.resolve([_PickingMtpLayer(0)])
    assert not resolver.resolve([_PickingMtpLayer(0), _PlainMtpLayer(0)])
    assert not DraftRowPick("test").resolve([_PickingMtpLayer(0)], decline="attention DP")

    monkeypatch.setenv("TLLM_SPEC_DISABLE_DRAFT_ROW_PICK", "1")
    assert not draft_layer_picks_rows(_PickingMtpLayer(0))


class _StubNorm:
    """RMSNorm (with the optional residual add) without weights."""

    weight = None
    variance_epsilon = 1e-6

    def __call__(self, x, residual=None):
        if residual is not None:
            x = x + residual
            residual = x
        y = x / x.float().pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
        return y if residual is None else (y, residual)


_stub_norm = _StubNorm()


def _mtp_layer_cls(name):
    if name == "DeepseekV3MTP":
        from tensorrt_llm._torch.models.modeling_deepseekv3 import DeepseekV3MTP

        return DeepseekV3MTP
    from tensorrt_llm._torch.models.modeling_glm import Glm4MTP

    return Glm4MTP


@pytest.mark.parametrize(
    "layer_name,fused_allreduce", [("DeepseekV3MTP", False), ("Glm4MTP", True)]
)
def test_mtp_layer_forward_pick_rows(layer_name, fused_allreduce):
    """The MTP layer's ``forward`` with ``pick_rows`` equals the picked rows of
    the full forward; attention sees every row, the MoE only the picked."""
    mtp_cls = _mtp_layer_cls(layer_name)
    assert draft_layer_picks_rows(object.__new__(mtp_cls))
    g = torch.Generator().manual_seed(0)
    w_eh = (torch.randn(2 * H, H, generator=g) / H**0.5).cuda()
    w_moe = (torch.randn(H, H, generator=g) / H**0.5).cuda()
    emb = torch.randn(VOCAB, H, generator=g).cuda()
    seen = {"attn": [], "moe": []}

    def self_attn(position_ids, hidden_states, attn_metadata, all_reduce_params, **kwargs):
        seen["attn"].append(hidden_states.shape[0])
        n = torch.arange(1, hidden_states.shape[0] + 1, device="cuda").unsqueeze(1)
        return hidden_states.cumsum(0) / n

    def mlp(hidden_states, all_rank_num_tokens, final_all_reduce_params, **kwargs):
        seen["moe"].append(hidden_states.shape[0])
        return torch.relu(hidden_states @ w_moe)

    def allreduce(hidden_states, all_reduce_params):
        p = all_reduce_params
        return _stub_norm(hidden_states, p.residual)

    from tensorrt_llm._torch.utils import EventType

    shared_head = SimpleNamespace(norm=_stub_norm)
    layer = SimpleNamespace(
        enorm=_stub_norm,
        hnorm=_stub_norm,
        event_dict={EventType.Main: torch.cuda.Event(), EventType.MoeShared: torch.cuda.Event()},
        aux_stream=None,
        model_config=SimpleNamespace(
            mapping=SimpleNamespace(tp_size=1, tp_rank=0, enable_attention_dp=False)
        ),
        mapping=SimpleNamespace(tp_size=1),
        eh_proj=lambda x: x @ w_eh,
        input_layernorm=_stub_norm,
        self_attn=self_attn,
        disable_attn_allreduce=False,
        fusion_config=SimpleNamespace(
            PRE_MOE_FUSION=fused_allreduce, POST_MOE_FUSION=fused_allreduce
        ),
        allreduce=allreduce,
        post_attention_layernorm=_stub_norm,
        mlp=mlp,
        shared_head=shared_head,
    )
    n = 23
    input_ids = torch.randint(0, VOCAB, (n,), generator=g).cuda()
    hidden = torch.randn(n, H, generator=g).cuda()
    pick = torch.tensor([4, 9, 15, 22], device="cuda")
    kwargs = dict(
        input_ids=input_ids,
        position_ids=None,
        hidden_states=hidden,
        embed_tokens=lambda ids: emb[ids],
        attn_metadata=None,
    )
    full = mtp_cls.forward(layer, **kwargs)
    picked = mtp_cls.forward(layer, pick_rows=pick, **kwargs)
    torch.testing.assert_close(picked, full[pick], rtol=1e-5, atol=1e-5)
    assert seen == {"attn": [n, n], "moe": [n, pick.numel()]}


def test_eagle3_decoder_layer_pick_rows():
    from tensorrt_llm._torch.models.modeling_speculative import Eagle3DecoderLayer, Eagle3DraftModel

    g = torch.Generator().manual_seed(1)
    w_mlp = (torch.randn(H, H, generator=g) / H**0.5).cuda()
    seen = {"attn": [], "mlp": []}

    def self_attn(position_ids, hidden_states, attn_metadata):
        seen["attn"].append(hidden_states.shape[0])
        n = torch.arange(1, hidden_states.shape[0] + 1, device="cuda").unsqueeze(1)
        return (hidden_states.cumsum(0) / n)[:, :H]

    def mlp(hidden_states):
        seen["mlp"].append(hidden_states.shape[0])
        return torch.relu(hidden_states @ w_mlp)

    decoder = SimpleNamespace(
        layer_idx=40,
        hidden_norm=_stub_norm,
        _next_layer_regular=False,
        input_layernorm=_stub_norm,
        self_attn=self_attn,
        post_attention_layernorm=_stub_norm,
        mlp=mlp,
    )
    model = SimpleNamespace(
        supports_pick_rows=True,
        embed_tokens=None,
        dtype=torch.float32,
        _eh_proj_before_attn=False,
        num_layers=1,
        midlayer=lambda **kw: Eagle3DecoderLayer.forward(decoder, **kw),
        norm=_stub_norm,
        _return_hidden_post_norm=False,
    )
    spec_metadata = SimpleNamespace(maybe_capture_hidden_states=lambda *a, **k: None)
    n = 17
    pick = torch.tensor([0, 5, 16], device="cuda")
    kwargs = dict(
        attn_metadata=SimpleNamespace(all_rank_num_tokens=None),
        inputs_embeds=torch.randn(n, H, generator=g).cuda(),
        position_ids=None,
        spec_metadata=spec_metadata,
        hidden_states=torch.randn(n, H, generator=g).cuda(),
    )
    full, full_save = Eagle3DraftModel.forward(model, **kwargs)
    picked, picked_save = Eagle3DraftModel.forward(model, pick_rows=pick, **kwargs)
    torch.testing.assert_close(picked, full[pick], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(picked_save, full_save[pick], rtol=1e-5, atol=1e-5)
    assert seen == {"attn": [n, n], "mlp": [n, pick.numel()]}
    model.supports_pick_rows = False
    with pytest.raises(AssertionError):
        Eagle3DraftModel.forward(model, pick_rows=pick, **kwargs)
