# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The GLM-5.3-Flash target's in-model MTP step against the stock MTP-Eagle worker it replaces.

Both run on the same synthetic batch -- context requests and generation requests whose drafts are accepted to
different depths -- with a toy MTP layer that records the attention-metadata lengths of every draft step. The step
must return what the worker returns, draft on the metadata the worker drafts on, commit the same accepted counts to
the KDA state, and leave the metadata as the worker leaves it.
"""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tensorrt_llm._torch._experimental.modeling_v2.models.glm5_next.glm_5_3_flash_nvfp4__sm_100__default.modeling import (  # noqa: E501
    Glm5NextMtpStep,
    _check_speculation,
)
from tensorrt_llm._torch.pyexecutor.kv_cache.mamba_cache_manager import MambaHybridCacheManager
from tensorrt_llm._torch.speculative.utils import get_spec_worker
from tensorrt_llm.llmapi.llm_args import AdvancedSamplingMode, MTPDecodingConfig
from tensorrt_llm.mapping import Mapping

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="needs a GPU (Triton acceptance kernel)"
)

_HIDDEN = 64
_VOCAB = 97


class _AcceptanceRecorder(MambaHybridCacheManager):
    """Stands in for the hybrid cache manager: records the accepted counts committed to the KDA state."""

    def __init__(self):  # noqa: D107 -- deliberately skips the manager's construction
        self.commits = []

    def update_mamba_states(self, attn_metadata, num_accepted_tokens, state_indices):
        self.commits.append((num_accepted_tokens.clone(), state_indices.clone()))


# Only update_mamba_states is called; the manager's resource API is never reached.
_AcceptanceRecorder.__abstractmethods__ = frozenset()


class _Metadata:
    """The attention-metadata surface the draft chain reads and rewrites."""

    def __init__(self, ctx_lens, num_gens, draft_len, kv_lens, device):
        seq_lens = list(ctx_lens) + [draft_len + 1] * num_gens
        self.num_seqs = len(seq_lens)
        self.num_contexts = len(ctx_lens)
        self._seq_lens = torch.tensor(seq_lens, dtype=torch.int32)
        self._seq_lens_cuda = self._seq_lens.to(device)
        self.kv_lens_cuda = torch.tensor(kv_lens, dtype=torch.int32, device=device)
        self.host_request_types = torch.tensor(
            [0] * len(ctx_lens) + [1] * num_gens, dtype=torch.int32
        )
        self.kv_cache_manager = _AcceptanceRecorder()
        self.mamba_metadata = SimpleNamespace(
            state_indices=torch.arange(self.num_seqs, device=device) + 10
        )
        self.all_rank_num_tokens = None
        self.padded_num_tokens = None
        self.spec_decoding_packed_mask = None
        self.spec_decoding_position_offsets = None
        self.spec_decoding_position_offsets_cpp = None
        self.spec_decoding_generation_lengths = None
        self.use_spec_decoding = True
        self.enable_flash_mla = False
        self._saved_tensors = {}
        self.on_update()

    @property
    def seq_lens_cuda(self):
        return self._seq_lens_cuda

    @property
    def num_ctx_tokens(self):
        return self._num_ctx_tokens

    def on_update(self):
        self._num_ctx_tokens = int(self._seq_lens[: self.num_contexts].sum())

    def on_update_kv_lens(self):
        pass

    def update_for_spec_dec(self):
        pass

    def prepare_for_spec_dec(self, *fields):
        assert not self._saved_tensors
        for f in fields:
            self._saved_tensors[f] = getattr(self, f)
            setattr(self, f, getattr(self, f).clone())

    def restore_from_spec_dec(self):
        for f, v in self._saved_tensors.items():
            setattr(self, f, v)
        self._saved_tensors.clear()

    @property
    def has_spec_dec_saved_state(self):
        return bool(self._saved_tensors)

    def snapshot(self):
        return dict(
            seq_lens=self._seq_lens.tolist(),
            kv_lens=self.kv_lens_cuda.tolist(),
            num_contexts=self.num_contexts,
            use_spec_decoding=self.use_spec_decoding,
            host_request_types=self.host_request_types.tolist(),
        )


class _ToyMtpLayer(nn.Module):
    """A deterministic MTP layer: mixes the token embedding, the incoming hidden row and the position, and records
    the metadata it was called on."""

    def __init__(self, mix):
        super().__init__()
        self.mix = mix
        self.calls = []

    def forward(self, embed_tokens, all_rank_num_tokens, input_ids, position_ids, hidden_states, attn_metadata,
                spec_metadata):  # fmt: skip
        self.calls.append(dict(attn_metadata.snapshot(), input_ids=input_ids.tolist(),
                               position_ids=position_ids.tolist()))  # fmt: skip
        mixed = (
            embed_tokens(input_ids)
            + hidden_states @ self.mix
            + 0.01 * position_ids.float().unsqueeze(-1)
        )
        return torch.tanh(mixed)

    def shared_head(self, hidden_states, lm_head, attn_metadata, return_context_logits):
        return lm_head(hidden_states).float()


def _batch(seed, ctx_lens, num_gens, draft_len, accept_depths, greedy=True):
    """One step's inputs: a decoder output, the verify logits, and drafts accepted to ``accept_depths``."""
    device = torch.device("cuda")
    g = torch.Generator(device="cpu").manual_seed(seed)
    embed = nn.Embedding(_VOCAB, _HIDDEN).to(device)
    lm_head = nn.Linear(_HIDDEN, _VOCAB, bias=False).to(device)
    with torch.no_grad():
        embed.weight.copy_(torch.randn(_VOCAB, _HIDDEN, generator=g))
        lm_head.weight.copy_(torch.randn(_VOCAB, _HIDDEN, generator=g))
    mix = torch.randn(_HIDDEN, _HIDDEN, generator=g).to(device) / _HIDDEN**0.5
    num_ctx_tokens = sum(ctx_lens)
    num_tokens = num_ctx_tokens + num_gens * (draft_len + 1)
    hidden = torch.randn(num_tokens, _HIDDEN, generator=g).to(device)
    input_ids = torch.randint(0, _VOCAB, (num_tokens,), generator=g).to(device, torch.int32)
    kv_lens = [L for L in ctx_lens] + [20 + 7 * i for i in range(num_gens)]
    positions = []
    for L in ctx_lens:
        positions += list(range(L))
    for i in range(num_gens):
        positions += list(
            range(kv_lens[len(ctx_lens) + i] - draft_len - 1, kv_lens[len(ctx_lens) + i])
        )
    position_ids = torch.tensor([positions], dtype=torch.int32, device=device)
    last_ctx = torch.tensor(ctx_lens, dtype=torch.long).cumsum(0) - 1
    gen_rows = num_ctx_tokens + torch.arange(num_gens * (draft_len + 1))
    gather_ids = torch.cat([last_ctx, gen_rows]).to(device)
    logits = lm_head(hidden[gather_ids]).float()
    # Drafts that match the target argmax to the requested depth, then miss.
    target = logits.argmax(-1)
    drafts = torch.empty((num_gens, draft_len), dtype=torch.int32)
    for i, depth in enumerate(accept_depths):
        rows = target[
            len(ctx_lens) + i * (draft_len + 1) : len(ctx_lens) + (i + 1) * (draft_len + 1)
        ].cpu()
        for k in range(draft_len):
            drafts[i, k] = rows[k] if k < depth else (rows[k] + 1) % _VOCAB
    rows = gather_ids.numel()
    spec_metadata = SimpleNamespace(
        runtime_draft_len=draft_len,
        draft_tokens=drafts.flatten().to(device),
        gather_ids=gather_ids,
        batch_indices_cuda=torch.arange(len(ctx_lens) + num_gens, dtype=torch.int32, device=device),
        is_all_greedy_sample=greedy,
        is_ragged_verify=False,
        wants_advanced_draft_sampling=False,
        use_rejection_sampling=False,
        enable_penalty=False,
        is_cuda_graph=False,
        context_prompt_lookahead_tokens=torch.full(
            (len(ctx_lens),), -1, dtype=torch.int32, device=device
        ),
        all_rank_num_tokens=None,
        subseq_all_rank_num_tokens=None,
        spec_resource_manager=None,
        request_ids=list(range(len(ctx_lens) + num_gens)),
        max_num_requests=64,
        temperatures=torch.full((rows,), 0.7, device=device),
        top_ks=torch.full((rows,), 20, dtype=torch.int32, device=device),
        top_ps=torch.full((rows,), 0.9, device=device),
        min_ps=torch.zeros(rows, device=device),
        seeds=torch.arange(rows, dtype=torch.int64, device=device) + 1234,
        offsets=torch.zeros(rows, dtype=torch.int64, device=device),
    )
    spec_metadata.advanced_sampling_mode = AdvancedSamplingMode.FULL
    md = _Metadata(ctx_lens, num_gens, draft_len, kv_lens, device)
    draft_model = SimpleNamespace(
        mtp_layers=[_ToyMtpLayer(mix)], lm_head=lm_head, embed_tokens=embed
    )
    return dict(input_ids=input_ids, position_ids=position_ids, hidden_states=hidden, logits=logits,
                attn_metadata=md, spec_metadata=spec_metadata, draft_model=draft_model)  # fmt: skip


def _run_stock(batch, draft_len):
    spec_config = MTPDecodingConfig(max_draft_len=draft_len)
    worker = get_spec_worker(spec_config, SimpleNamespace(mapping=Mapping()), Mapping())
    return worker(resource_manager=None, **batch)


def _run_fused(batch):
    return Glm5NextMtpStep(Mapping(), _VOCAB)(**batch)


def _copy(batch):
    out = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
    out["attn_metadata"] = copy.deepcopy(batch["attn_metadata"])
    out["spec_metadata"] = copy.copy(batch["spec_metadata"])
    out["draft_model"] = SimpleNamespace(
        mtp_layers=[_ToyMtpLayer(batch["draft_model"].mtp_layers[0].mix)],
        lm_head=batch["draft_model"].lm_head,
        embed_tokens=batch["draft_model"].embed_tokens,
    )
    return out


@pytest.mark.parametrize("greedy", [True, False], ids=["greedy", "sampled"])
@pytest.mark.parametrize(
    "ctx_lens, accept_depths",
    [((), (0, 1, 3)), ((5, 3), ()), ((4,), (3, 0, 2, 1))],
    ids=["generation-only", "context-only", "mixed"],
)
@pytest.mark.parametrize("draft_len", [3, 1])
def test_step_matches_stock_worker(ctx_lens, accept_depths, draft_len, greedy):
    accept_depths = tuple(min(d, draft_len) for d in accept_depths)
    batch = _batch(7, ctx_lens, len(accept_depths), draft_len, accept_depths, greedy=greedy)
    stock_in, fused_in = _copy(batch), _copy(batch)
    stock = _run_stock(stock_in, draft_len)
    fused = _run_fused(fused_in)

    for key in ("new_tokens_lens", "next_draft_tokens", "next_new_tokens"):
        assert torch.equal(stock[key], fused[key]), key
    # A context request defines only column 0 of new_tokens; generation rows up to their accepted count.
    lens = stock["new_tokens_lens"].tolist()
    for b, n in enumerate(lens):
        assert torch.equal(stock["new_tokens"][b, :n], fused["new_tokens"][b, :n]), b
    assert stock["logits"] is stock_in["logits"] and fused["logits"] is fused_in["logits"]
    if greedy and accept_depths:
        assert lens[len(ctx_lens) :] == [d + 1 for d in accept_depths]

    # The same metadata on every draft step, and the same metadata left behind.
    assert (
        stock_in["draft_model"].mtp_layers[0].calls == fused_in["draft_model"].mtp_layers[0].calls
    )
    assert stock_in["attn_metadata"].snapshot() == fused_in["attn_metadata"].snapshot()
    assert not fused_in["attn_metadata"].has_spec_dec_saved_state
    # The same accepted counts committed to the KDA state.
    s_commits = stock_in["attn_metadata"].kv_cache_manager.commits
    f_commits = fused_in["attn_metadata"].kv_cache_manager.commits
    assert len(s_commits) == len(f_commits) == (1 if accept_depths else 0)
    for (sa, si), (fa, fi) in zip(s_commits, f_commits):
        assert torch.equal(sa, fa) and torch.equal(si, fi)


def test_no_drafts_step_matches_stock_worker():
    batch = _batch(3, (4,), 3, 1, (1, 0, 1))
    stock_in, fused_in = _copy(batch), _copy(batch)
    stock_in["spec_metadata"].runtime_draft_len = fused_in["spec_metadata"].runtime_draft_len = 0
    # With no drafts every generation request verifies one row.
    for b in (stock_in, fused_in):
        b["logits"] = b["logits"][:4]
    stock = _run_stock(stock_in, 1)
    fused = _run_fused(fused_in)
    for key in ("new_tokens", "new_tokens_lens", "next_draft_tokens", "next_new_tokens"):
        assert torch.equal(stock[key].to(torch.int32), fused[key].to(torch.int32)), key


@pytest.mark.parametrize(
    "overrides, refused",
    [
        (dict(use_rejection_sampling=True), "rejection sampling"),
        (dict(use_relaxed_acceptance_for_thinking=True), "relaxed acceptance"),
        (dict(use_mtp_vanilla=True), "other than one-model MTP-Eagle"),
    ],
)
def test_unsupported_speculation_is_refused(overrides, refused):
    with pytest.raises(ValueError, match=refused):
        _check_speculation(MTPDecodingConfig(max_draft_len=3, **overrides))


def test_supported_speculation_is_admitted():
    _check_speculation(None)
    _check_speculation(MTPDecodingConfig(max_draft_len=3))
