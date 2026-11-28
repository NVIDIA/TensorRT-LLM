# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The attention-DP heads over vocabulary slices (`DP_HEAD_SHARD_ENV`, opt-in, CPU stand-ins).

* **Values**: which heads, which group.
* **Build** (`_dp_head_shard_build`): where it declines, the rows each head serves (the measured
  request ranges at the CUDA-graph batch sizes), the group's agreement on the counts whose slice
  logits are not the full head's bits, and the worker's `draft_head` hook.
* **Decisions** (`_DpHeadShard`): a head engages only in a CUDA-graph step in which every member
  of the group holds one served count; a rank holding other rows raises.
* **The verify head in the shell's step**: the worker gets the drawn tokens as
  `_ShardVerifyLogits` and the returned `logits` is a placeholder.

The group's kernels and the slice GEMM are upstream's `GroupShardedVerify`
(`tests/unittest/_torch/multi_gpu/test_group_sharded_sampling.py`).
"""

from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch.modeling_v2.models.qwen4_exp.qwen3_8_flash_next_nvfp4__sm_10x__default import (  # noqa: E501
    modeling as M,
)

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize(
    "value, heads",
    [
        (None, set()),
        ("0", set()),
        ("1", {"draft", "verify"}),
        ("draft", {"draft"}),
        ("verify, draft", {"draft", "verify"}),
    ],
)
def test_heads(monkeypatch, value, heads):
    if value is None:
        monkeypatch.delenv(M.DP_HEAD_SHARD_ENV, raising=False)
    else:
        monkeypatch.setenv(M.DP_HEAD_SHARD_ENV, value)
    assert M._dp_head_shard_heads() == heads


def test_values_refused(monkeypatch):
    monkeypatch.setenv(M.DP_HEAD_SHARD_ENV, "target")
    with pytest.raises(ValueError, match=M.DP_HEAD_SHARD_ENV):
        M._dp_head_shard_heads()
    monkeypatch.setenv(M.DP_HEAD_SHARD_GROUP_ENV, "rack")
    with pytest.raises(ValueError, match=M.DP_HEAD_SHARD_GROUP_ENV):
        M._dp_head_shard_group()
    monkeypatch.delenv(M.DP_HEAD_SHARD_GROUP_ENV)
    assert M._dp_head_shard_group() == "node"


class _Group:
    """`GroupShardedVerify` stand-in: records the calls; `sample` returns the rows' first column."""

    def __init__(self, size=4, rank=1, first=0, calls=None):
        self.group_size, self.rank, self.first_dp_rank = size, rank, first
        self.vocab_offset, self.local_vocab, self.hidden = 0, 8, 4
        self.weight = torch.zeros(8, 4, dtype=torch.bfloat16)
        self.slice_logits = torch.zeros(size * 64, 8, dtype=torch.bfloat16)
        self.buffers = SimpleNamespace(transport=SimpleNamespace(value="ipc"))
        self.calls = [] if calls is None else calls

    def logits(self, rows, temperatures=None, seeds=None, offsets=None):
        self.calls.append(("logits", rows.shape[0], temperatures is not None))
        return rows

    def sample(self, slice_logits):
        self.calls.append(("sample", slice_logits.shape[0]))
        return torch.arange(slice_logits.shape[0], dtype=torch.int64)


def _shard(verify=(8, 16), draft=(2, 4), **kw):
    core = SimpleNamespace(dp_rank=1)
    return M._DpHeadShard(_Group(**kw), core, frozenset(verify), frozenset(draft))


def _spec_md(counts=(2, 2, 2, 2), graph=True, greedy=True, temp=False):
    return SimpleNamespace(
        subseq_all_rank_num_tokens=list(counts),
        is_cuda_graph=graph,
        is_all_greedy_sample=greedy,
        is_all_temperature_only=temp,
        is_ragged_verify=False,
        temperatures=torch.ones(64),
        seeds=torch.zeros(64, dtype=torch.int64),
        offsets=torch.zeros(64, dtype=torch.int64),
    )


def test_draft_engages_only_on_a_served_uniform_graph_step():
    shard = _shard()
    shard.begin_step(_spec_md((2, 2, 2, 2)))
    assert shard.draft_engages(2)
    shard.begin_step(_spec_md((2, 2, 3, 2)))  # unequal counts in the group
    assert not shard.draft_engages(2)
    shard.begin_step(_spec_md((3, 3, 3, 3)))  # not a served count
    assert not shard.draft_engages(3)
    shard.begin_step(_spec_md((2, 2, 2, 2), graph=False))  # an eager step
    assert not shard.draft_engages(2)
    shard.begin_step(_spec_md((4, 4, 4, 4)))
    with pytest.raises(RuntimeError, match="holds 2 draft rows"):
        shard.draft_engages(2)


def test_node_group_reads_its_own_members():
    """Ranks 4..7 form the second node group at dep8: only their counts decide."""
    shard = _shard(size=4, rank=1, first=4)
    shard.begin_step(_spec_md((1, 1, 3, 1, 2, 2, 2, 2)))
    assert shard.draft_engages(2)


@pytest.mark.parametrize(
    "greedy, temp, served",
    [(True, False, True), (False, True, True), (False, False, False)],
    ids=["greedy", "temperature-only", "other sampling"],
)
def test_verify_tokens(greedy, temp, served):
    shard = _shard()
    md = _spec_md(greedy=greedy, temp=temp)
    shard.begin_step(md)
    attn = SimpleNamespace(all_rank_num_tokens=[8, 8, 8, 8])
    worker = SimpleNamespace(guided_decoder=None)
    tokens = shard.verify_tokens(worker, md, attn, torch.zeros(8, 4))
    if not served:
        assert tokens is None and not shard.group.calls
        return
    assert tokens.shape == (8,)
    assert shard.group.calls == [("logits", 8, temp), ("sample", 8)]


def test_verify_tokens_declines():
    shard = _shard()
    md = _spec_md()
    shard.begin_step(md)
    attn = SimpleNamespace(all_rank_num_tokens=[8, 8, 8, 8])
    rows = torch.zeros(8, 4)
    assert shard.verify_tokens(SimpleNamespace(guided_decoder=object()), md, attn, rows) is None
    md.is_ragged_verify = True
    assert shard.verify_tokens(SimpleNamespace(guided_decoder=None), md, attn, rows) is None
    md.is_ragged_verify = False
    attn.all_rank_num_tokens = [8, 8, 12, 8]
    assert shard.verify_tokens(SimpleNamespace(guided_decoder=None), md, attn, rows) is None
    assert not shard.group.calls


def test_draft_hook_samples_through_the_group(monkeypatch):
    monkeypatch.setattr(M, "_mtp_head_final", lambda core, h: h[:, :4])
    shard = _shard()
    hook = M._DpShardDraftHead(shard.core, shard)
    out = torch.full((2,), -1, dtype=torch.int32)
    assert hook.sample(torch.zeros(2, 16), out=out) is out and out.tolist() == [0, 1]
    tok = hook.sample(torch.zeros(2, 16))
    assert tok.dtype == torch.int32 and tok.tolist() == [0, 1]


def test_shard_verify_logits():
    lg = M._ShardVerifyLogits(torch.tensor([3, 4, 5]), vocab=100)
    assert isinstance(lg, M.VocabShardedLogits)
    assert lg.dim() == 2 and lg.shape == (3, 100) and lg.num_rows == 3
    assert lg.select_tokens(None).tolist() == [3, 4, 5]
    with pytest.raises(RuntimeError, match=M.DP_HEAD_SHARD_ENV):
        lg.gather()


class _Head:
    """A replicated bf16 `lm_head` stand-in that the build's type check accepts."""

    def __init__(self, vocab=8, hidden=4):
        self.weight = torch.zeros(vocab, hidden, dtype=torch.bfloat16)
        self.bias = None

    def __call__(self, x):
        return torch.zeros(x.shape[0], self.weight.shape[0], dtype=torch.bfloat16)


def _build_shell(sizes=(1, 2, 4, 8, 16, 32, 40, 64, 80)):
    core = SimpleNamespace(
        mtp_enabled=True,
        mapping=SimpleNamespace(enable_lm_head_tp_in_adp=False, gpus_per_node=4),
        dp_size=8,
        dp_rank=1,
        vocab=8,
        hidden=4,
        max_verify_rows=4,
        model_config=SimpleNamespace(cuda_graph_batch_sizes=list(sizes)),
    )
    worker = SimpleNamespace(is_mtp_eagle=True, draft_head=None)
    spec = SimpleNamespace(use_rejection_sampling=False, enable_penalty=False)
    return SimpleNamespace(model=core, spec_worker=worker, spec_config=spec, lm_head=_Head())


def _patch_build(monkeypatch, reports_bad=(), dp=True, no_group=False, peer_error="", sent=None):
    created = []

    def create(mapping, weight, size, max_rows, transport):
        created.append((size, max_rows, transport))
        return None if no_group else _Group(size=size)

    import tensorrt_llm._torch.distributed.communicator as comm
    import tensorrt_llm._torch.distributed.peer_buffers as pb
    import tensorrt_llm._torch.speculative.group_sharded_sampling as gss

    monkeypatch.setattr(gss, "GroupShardedVerify", SimpleNamespace(create=create), raising=False)
    monkeypatch.setattr(pb, "PeerTransport", SimpleNamespace(IPC="ipc"), raising=False)

    def allgather(obj):
        if sent is not None:
            sent.append(obj)
        return [obj, (peer_error, list(reports_bad))]

    gathered = SimpleNamespace(tp_allgather=allgather)
    monkeypatch.setattr(
        comm, "Distributed", SimpleNamespace(get=lambda mapping: gathered), raising=False
    )
    monkeypatch.setattr(M, "_dp_decode_rank", lambda core, device: dp)
    monkeypatch.setattr(M, "LMHead", _Head)
    monkeypatch.setattr(M, "_dp_head_shard_check", lambda group, head, counts: [])
    return created


@pytest.mark.parametrize("group, size, transport", [("node", 4, "ipc"), ("domain", 8, None)])
def test_build(monkeypatch, group, size, transport):
    monkeypatch.setenv(M.DP_HEAD_SHARD_ENV, "1")
    monkeypatch.setenv(M.DP_HEAD_SHARD_GROUP_ENV, group)
    created = _patch_build(monkeypatch, reports_bad=[8])
    shell = _build_shell()
    shard = M._dp_head_shard_build(shell, torch.device("cpu"))
    # verify: 4 rows a request at 1..32 requests; draft: 1..64 requests; 8 failed on a peer.
    assert shard.verify_rows == {4, 16, 32, 64, 128}
    assert shard.draft_rows == {1, 2, 4, 16, 32, 40, 64}
    assert created == [(size, 128, transport)]
    assert isinstance(shell.spec_worker.draft_head, M._DpShardDraftHead)


def test_build_verify_only_keeps_the_hook_free(monkeypatch):
    monkeypatch.setenv(M.DP_HEAD_SHARD_ENV, "verify")
    _patch_build(monkeypatch)
    shell = _build_shell()
    shard = M._dp_head_shard_build(shell, torch.device("cpu"))
    assert shard.verify_rows and not shard.draft_rows
    assert shell.spec_worker.draft_head is None


@pytest.mark.parametrize(
    "case",
    [
        "not-dp",
        "rejection",
        "acceptance-gate",
        "lm-head-tp",
        "hook-taken",
        "fp32-logits",
        "no-graph-rows",
    ],
)
def test_build_declines(monkeypatch, case):
    """Declines decided from the layout, the deploy configuration or the env: on every rank
    alike, before any collective (no group is created, nothing is exchanged)."""
    monkeypatch.setenv(M.DP_HEAD_SHARD_ENV, "1")
    sent = []
    created = _patch_build(monkeypatch, dp=case != "not-dp", sent=sent)
    shell = _build_shell(sizes=(80, 96) if case == "no-graph-rows" else (1, 2, 4))
    if case == "rejection":
        shell.spec_config.use_rejection_sampling = True
    if case == "acceptance-gate":
        shell.spec_config.acceptance_rate_window_size = 16
        shell.spec_config.acceptance_rate_threshold = 0.5
    if case == "lm-head-tp":
        shell.model.mapping.enable_lm_head_tp_in_adp = True
    if case == "hook-taken":
        shell.spec_worker.draft_head = object()
    if case == "fp32-logits":
        shell.lm_head.output_dtype = torch.float32
    warned = []
    monkeypatch.setattr(M.logger, "warning", warned.append)
    assert M._dp_head_shard_build(shell, torch.device("cpu")) is None
    assert len(warned) == 1 and M.DP_HEAD_SHARD_ENV in warned[0]
    assert not created and not sent


@pytest.mark.parametrize("where", ["this rank", "a peer"])
def test_build_agrees_when_a_group_is_missing(monkeypatch, where):
    """A rank whose group could not be built still joins the attention-DP exchange, and every
    rank then declines: under `node` the other group may have built its own, and a rank that
    returned early would leave that group's members waiting in the exchange."""
    monkeypatch.setenv(M.DP_HEAD_SHARD_ENV, "1")
    sent = []
    _patch_build(
        monkeypatch,
        no_group=where == "this rank",
        peer_error="no group of 4 ranks" if where == "a peer" else "",
        sent=sent,
    )
    shell = _build_shell()
    warned = []
    monkeypatch.setattr(M.logger, "warning", warned.append)
    assert M._dp_head_shard_build(shell, torch.device("cpu")) is None
    assert len(sent) == 1
    assert len(warned) == 1 and "no group of 4 ranks" in warned[0]
    assert shell.spec_worker.draft_head is None


def test_check_runs_with_grad_on_and_a_weight_that_requires_grad():
    """The build check runs in `post_load_weights`, with grad mode on, over the head's weight,
    which still requires grad there: its `torch.mm(..., out=)` must not see autograd (a dep8 GSM8K
    run: `mm(): functions with out=... arguments don't support automatic
    differentiation`, so the heads were declined on every rank)."""
    vocab, hidden, g, lo, n = 16, 8, 2, 8, 8
    full = torch.nn.Parameter(torch.randn(vocab, hidden).to(torch.bfloat16))
    assert full.requires_grad and torch.is_grad_enabled()
    group = SimpleNamespace(
        group_size=g,
        vocab_offset=lo,
        local_vocab=n,
        hidden=hidden,
        rank=1,
        weight=full[lo : lo + n],
        slice_logits=torch.empty(g * 4, n, dtype=torch.bfloat16),
    )
    bad = M._dp_head_shard_check(group, lambda x: x @ full.t(), [1, 4])
    assert set(bad) <= {1, 4}


def test_build_off(monkeypatch):
    monkeypatch.delenv(M.DP_HEAD_SHARD_ENV, raising=False)
    assert M._dp_head_shard_build(_build_shell(), torch.device("cpu")) is None


class _Core:
    _lamport_steps = None
    vocab = 100

    def __init__(self, final):
        self.final = final

    def begin_step(self, *args, **kwargs) -> None:
        pass

    def __call__(self, **kwargs):
        return None

    def split_spec_output(self, out):
        return self.final, self.final


class _Worker(torch.nn.Module):
    """The spec worker: returns the `logits` it got, as the MTP-Eagle worker's step does."""

    def forward(self, **kwargs):
        return {"logits": kwargs["logits"]}


def test_shell_hands_the_worker_the_drawn_tokens(monkeypatch):
    cls = M.ModelingV2Qwen38FlashNextNvfp4Sm10xDefault
    shell = cls.__new__(cls)
    torch.nn.Module.__init__(shell)
    shell.model = _Core(torch.zeros(8, 4))
    shell.spec_worker = _Worker()
    shell.one_rank = False
    shell.lm_head = "head"
    shell.logits_processor = SimpleNamespace(forward=lambda *a: pytest.fail("full head ran"))
    shell.draft_model = SimpleNamespace(mtp_layers=[SimpleNamespace(spec_metadata=None)])
    shell.vision_tower = None
    shell._dp_head_shard = _shard()
    monkeypatch.setattr(M._vision, "multimodal_embeds", lambda tower, kwargs: None)
    monkeypatch.setattr(M, "_head_rows", lambda core, ids, final: final)
    spec_md = _spec_md(counts=(2, 2, 2, 2))
    spec_md.gather_ids = torch.arange(8)
    attn = SimpleNamespace(padded_num_tokens=None, all_rank_num_tokens=[8, 8, 8, 8])
    out = shell.forward(attn, torch.zeros(8, dtype=torch.int32), None, spec_metadata=spec_md)
    assert out["logits"].shape == (8, 1) and torch.isnan(out["logits"]).all()
    assert shell.samples_from_vocab_shards
    shell._dp_head_shard = None
    assert not shell.samples_from_vocab_shards
