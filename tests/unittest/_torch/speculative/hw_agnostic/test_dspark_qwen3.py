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
"""Golden tests for the Qwen3 DSpark drafter (modeling_dspark_qwen3.py).

The reference implementation below is a line-for-line torch-only port of the
DeepSpec ``Qwen3DSparkModel`` inference path
(``deepspec/modeling/dspark/qwen3/modeling.py`` `_forward_backbone` +
``deepspec/eval/dspark/draft_ops.py`` `forward_dspark_draft_block` /
`build_dspark_proposal`): full-context draft attention over
``[ctx_kv_cache, block]`` with bidirectional block attention, HF-style RoPE,
per-head q/k RMSNorm, and greedy Markov-chained block sampling.

The tests drive ``Qwen3DSparkDraftModel`` exactly the way ``DSparkWorker``
does — seeding via ``write_context_windows`` (frames = position + 1),
back-filling interims via ``write_context_windows_batched``, drafting via
``forward_batched`` — across multiple decode steps, and assert token-exact /
logit-close agreement with the reference.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from tensorrt_llm._torch.models.modeling_dspark_qwen3 import Qwen3DSparkDraftModel, _apply_rope

VOCAB = 97
HID = 32
INTER = 48
N_HEADS = 4
N_KV_HEADS = 2
HEAD_DIM = 8
N_LAYERS = 2
N_CAP = 3
BLOCK = 3
MASK_ID = 90
RANK = 16
THETA = 10000.0
MAX_POS = 512
EPS = 1e-6

DTYPE = torch.bfloat16


def _make_config():
    pretrained = SimpleNamespace(
        architectures=["Qwen3DSparkModel"],
        hidden_size=HID,
        intermediate_size=INTER,
        num_attention_heads=N_HEADS,
        num_key_value_heads=N_KV_HEADS,
        head_dim=HEAD_DIM,
        num_hidden_layers=N_LAYERS,
        vocab_size=VOCAB,
        rms_norm_eps=EPS,
        rope_parameters={"rope_theta": THETA},
        max_position_embeddings=MAX_POS,
        block_size=BLOCK,
        mask_token_id=MASK_ID,
        target_layer_ids=[0, 2, 4],
        markov_rank=RANK,
        markov_head_type="vanilla",
        enable_confidence_head=True,
        confidence_head_with_markov=True,
        num_anchors=8,
    )
    return SimpleNamespace(pretrained_config=pretrained, spec_config=None)


def _rand_weights(gen):
    def r(*shape, scale=0.05):
        return (torch.randn(*shape, generator=gen) * scale).to(DTYPE)

    w = {
        "fc.weight": r(HID, N_CAP * HID),
        "hidden_norm.weight": 1.0 + r(HID),
        "norm.weight": 1.0 + r(HID),
        "markov_head.markov_w1.weight": r(VOCAB, RANK),
        "markov_head.markov_w2.weight": r(VOCAB, RANK),
        "confidence_head.proj.weight": r(1, HID + RANK),
        "confidence_head.proj.bias": r(1),
        # frozen copies (skipped by the model; shared modules used instead)
        "embed_tokens.weight": r(VOCAB, HID, scale=0.5),
        "lm_head.weight": r(VOCAB, HID, scale=0.5),
    }
    for i in range(N_LAYERS):
        p = f"layers.{i}."
        w[p + "self_attn.q_proj.weight"] = r(N_HEADS * HEAD_DIM, HID)
        w[p + "self_attn.k_proj.weight"] = r(N_KV_HEADS * HEAD_DIM, HID)
        w[p + "self_attn.v_proj.weight"] = r(N_KV_HEADS * HEAD_DIM, HID)
        w[p + "self_attn.o_proj.weight"] = r(HID, N_HEADS * HEAD_DIM)
        w[p + "self_attn.q_norm.weight"] = 1.0 + r(HEAD_DIM)
        w[p + "self_attn.k_norm.weight"] = 1.0 + r(HEAD_DIM)
        w[p + "mlp.gate_proj.weight"] = r(INTER, HID)
        w[p + "mlp.up_proj.weight"] = r(INTER, HID)
        w[p + "mlp.down_proj.weight"] = r(HID, INTER)
        w[p + "input_layernorm.weight"] = 1.0 + r(HID)
        w[p + "post_attention_layernorm.weight"] = 1.0 + r(HID)
    return w


def _build_model(weights):
    model = Qwen3DSparkDraftModel(_make_config())
    model.load_weights(weights)
    device = model.fc.weight.device
    embed = torch.nn.Embedding(VOCAB, HID)
    embed.weight.data = weights["embed_tokens.weight"].to(device)
    lm_head = torch.nn.Linear(HID, VOCAB, bias=False)
    lm_head.weight.data = weights["lm_head.weight"].to(device)
    model.embed_tokens = embed.to(device)
    model.lm_head = lm_head.to(device)
    return model, device


# --------------------------------------------------------------------------
# Reference: DeepSpec Qwen3DSparkModel inference path (torch-only port)
# --------------------------------------------------------------------------


class _Ref:
    """Full-context reference drafter operating on one request."""

    def __init__(self, w, device):
        self.w = {k: v.to(device) for k, v in w.items()}
        self.device = device
        inv = 1.0 / (
            THETA ** (torch.arange(0, HEAD_DIM, 2, device=device, dtype=torch.float32) / HEAD_DIM)
        )
        t = torch.arange(MAX_POS, device=device, dtype=torch.float32)
        emb = torch.cat([torch.outer(t, inv)] * 2, dim=-1)
        self.cos, self.sin = emb.cos(), emb.sin()
        # per-request context stream: [T, HID] projected hiddens, in order
        self.ctx_x = torch.zeros(0, HID, dtype=DTYPE, device=device)

    def _norm(self, x, wname):
        wt = self.w[wname]
        dt = x.dtype
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
        return wt * xf.to(dt)

    def append_ctx(self, captured):
        """captured: [M, N_CAP*HID] raw target hiddens (positions in order)."""
        x = F.linear(captured.to(self.device), self.w["fc.weight"])
        x = self._norm(x, "hidden_norm.weight")
        self.ctx_x = torch.cat([self.ctx_x, x.to(DTYPE)], dim=0)

    def draft(self, bonus_id, start_pos):
        assert self.ctx_x.shape[0] == start_pos
        w = self.w
        ids = torch.full((BLOCK,), MASK_ID, dtype=torch.long, device=self.device)
        ids[0] = bonus_id
        h = F.embedding(ids, w["embed_tokens.weight"])  # [B, HID]
        pos_q = start_pos + torch.arange(BLOCK, device=self.device)
        pos_c = torch.arange(start_pos, device=self.device)
        for i in range(N_LAYERS):
            p = f"layers.{i}."
            x = self._norm(h, p + "input_layernorm.weight")
            q = F.linear(x, w[p + "self_attn.q_proj.weight"]).view(BLOCK, N_HEADS, HEAD_DIM)
            q = self._norm(q, p + "self_attn.q_norm.weight")
            q = _apply_rope(q, self.cos[pos_q].to(DTYPE), self.sin[pos_q].to(DTYPE))
            src = torch.cat([self.ctx_x, x], dim=0)  # [T+B, HID]
            pos_k = torch.cat([pos_c, pos_q])
            k = F.linear(src, w[p + "self_attn.k_proj.weight"]).view(-1, N_KV_HEADS, HEAD_DIM)
            k = self._norm(k, p + "self_attn.k_norm.weight")
            k = _apply_rope(k, self.cos[pos_k].to(DTYPE), self.sin[pos_k].to(DTYPE))
            v = F.linear(src, w[p + "self_attn.v_proj.weight"]).view(-1, N_KV_HEADS, HEAD_DIM)
            rep = N_HEADS // N_KV_HEADS
            kk = k.transpose(0, 1).repeat_interleave(rep, dim=0)
            vv = v.transpose(0, 1).repeat_interleave(rep, dim=0)
            o = F.scaled_dot_product_attention(q.transpose(0, 1), kk, vv, scale=HEAD_DIM**-0.5)
            o = o.transpose(0, 1).reshape(BLOCK, N_HEADS * HEAD_DIM)
            h = h + F.linear(o, w[p + "self_attn.o_proj.weight"])
            x = self._norm(h, p + "post_attention_layernorm.weight")
            mlp = F.linear(
                F.silu(F.linear(x, w[p + "mlp.gate_proj.weight"]))
                * F.linear(x, w[p + "mlp.up_proj.weight"]),
                w[p + "mlp.down_proj.weight"],
            )
            h = h + mlp
        h = self._norm(h, "norm.weight")
        base = F.linear(h, w["lm_head.weight"])  # [B, VOCAB]
        # Greedy Markov-chained sampling (VanillaMarkov).
        toks, logits = [], []
        prev = bonus_id.view(1).long()
        for kstep in range(BLOCK):
            bias = F.linear(
                F.embedding(prev, w["markov_head.markov_w1.weight"]),
                w["markov_head.markov_w2.weight"],
            )
            step = base[kstep : kstep + 1] + bias
            logits.append(step)
            prev = step.argmax(dim=-1)
            toks.append(prev)
        return torch.cat(toks), torch.cat(logits)


@pytest.fixture(scope="module")
def setup():
    gen = torch.Generator().manual_seed(1234)
    weights = _rand_weights(gen)
    model, device = _build_model(weights)
    return weights, model, device


def _rand_hidden(gen, n):
    return (torch.randn(n, N_CAP * HID, generator=gen) * 0.3).to(DTYPE)


def test_worker_protocol_golden(setup):
    """Drive the model exactly like DSparkWorker across 3 decode steps."""
    weights, model, device = setup
    gen = torch.Generator().manual_seed(7)

    prompt_len = 6
    max_batch = 4
    win = model._attn_params["window_size"]
    kv_windows = torch.zeros(
        max_batch, model.num_stages, win, model._attn_params["head_dim"], dtype=DTYPE, device=device
    )
    slot = 1
    ref = _Ref(weights, device)

    # ---- prefill: the worker seeds ALL prompt positions (frames = pos+1).
    # The target processed positions 0..L-1, so the ring holds 0..L-1 and
    # start_pos == L for the first draft; each gen step's main_hidden is the
    # newest committed token's captured hidden (position start_pos-1).
    prompt_hidden = _rand_hidden(gen, prompt_len).to(device)
    positions = torch.arange(prompt_len, device=device) + 1
    model.write_context_windows(prompt_hidden, positions, kv_windows[slot])
    ref.append_ctx(prompt_hidden)

    start_pos = prompt_len
    bonus_val = 13

    for step in range(3):
        # The worker's gen step: target verified some tokens; captured hiddens
        # for the (nacc) newly committed context tokens exist. nacc-1 interim
        # rows are back-filled batched; the newest one rides in main_hidden.
        nacc = [1, 3, 2][step]  # 1 = no interim (only the bonus position)
        new_hidden = _rand_hidden(gen, nacc).to(device)

        if nacc > 1:
            interim = new_hidden[: nacc - 1].unsqueeze(0)  # [1, nacc-1, ...]
            # frames old+1+j for j in 0..nacc-2  (old = pre-step start_pos)
            interim_pos = (start_pos + 1 + torch.arange(nacc - 1, device=device)).unsqueeze(0)
            mask = torch.ones(1, nacc - 1, dtype=torch.bool, device=device)
            model.write_context_windows_batched(
                interim, interim_pos, torch.tensor([slot], device=device), mask, kv_windows
            )

        ref.append_ctx(new_hidden)
        start_pos += nacc
        bonus = torch.tensor([bonus_val], device=device)

        got_toks, got_n, got_logits = model.forward_batched(
            new_hidden[-1:].reshape(1, -1),
            bonus,
            torch.tensor([start_pos], device=device),
            kv_windows=kv_windows,
            slots=torch.tensor([slot], device=device),
            return_logits=True,
        )
        exp_toks, exp_logits = ref.draft(bonus[0], start_pos)

        assert got_n.item() == BLOCK
        torch.testing.assert_close(got_logits[0].float(), exp_logits.float(), atol=0.05, rtol=0.05)
        assert torch.equal(got_toks[0].long().cpu(), exp_toks.long().cpu()), (
            f"step {step}: draft tokens diverge from reference"
        )
        bonus_val = (bonus_val * 7 + 3) % VOCAB


def test_batched_matches_eager_singletons(setup):
    """One batched call over G requests == G independent eager calls."""
    weights, model, device = setup
    gen = torch.Generator().manual_seed(21)
    G = 3
    win = model._attn_params["window_size"]
    kv_windows = torch.zeros(
        G, model.num_stages, win, model._attn_params["head_dim"], dtype=DTYPE, device=device
    )
    ctx_lens = [4, 7, 5]
    for g in range(G):
        h = _rand_hidden(gen, ctx_lens[g]).to(device)
        pos = torch.arange(ctx_lens[g], device=device) + 1
        model.write_context_windows(h, pos, kv_windows[g])

    main = _rand_hidden(gen, G).to(device)
    bonus = torch.tensor([11, 22, 33], device=device)
    start = torch.tensor([c + 1 for c in ctx_lens], device=device)
    # per-request singleton calls on cloned windows
    exp_toks, exp_logits = [], []
    for g in range(G):
        wins = kv_windows[g : g + 1].clone()
        t, _, lg = model.forward_batched(
            main[g : g + 1],
            bonus[g : g + 1],
            start[g : g + 1],
            kv_windows=wins,
            slots=torch.tensor([0], device=device),
            return_logits=True,
        )
        exp_toks.append(t)
        exp_logits.append(lg)
    got_toks, _, got_logits = model.forward_batched(
        main,
        bonus,
        start,
        kv_windows=kv_windows,
        slots=torch.arange(G, device=device),
        return_logits=True,
    )
    assert torch.equal(got_toks, torch.cat(exp_toks, dim=0))
    torch.testing.assert_close(got_logits, torch.cat(exp_logits, dim=0))


def test_ring_window_wraparound(setup, monkeypatch):
    """start_pos > window: the ring holds the last `win` positions and the
    draft attends to all of them (mask all-valid)."""
    weights, _, device = setup
    monkeypatch.setenv("TRTLLM_DSPARK_QWEN3_CTX_WINDOW", "32")
    model, device = _build_model(weights)
    gen = torch.Generator().manual_seed(5)
    win = model._attn_params["window_size"]
    assert win == 32
    total = win + 9  # forces wraparound
    kv_windows = torch.zeros(
        1, model.num_stages, win, model._attn_params["head_dim"], dtype=DTYPE, device=device
    )
    h = _rand_hidden(gen, total).to(device)
    pos = torch.arange(total, device=device) + 1
    # Seed in two chunks like chunked prefill (worker keeps last min(win, len)).
    model.write_context_windows(h[:win], pos[:win], kv_windows[0])
    model.write_context_windows(h[win:], pos[win:], kv_windows[0])

    # Reference limited to the last `win` context positions.
    ref = _Ref(weights, device)
    ref.append_ctx(h)
    ref.ctx_x = ref.ctx_x[-win:]

    bonus = torch.tensor([42], device=device)
    start = torch.tensor([total + 1], device=device)
    main = _rand_hidden(gen, 1).to(device)
    # keep ref in sync: main_hidden row is position `total` (= start-1)
    ref.append_ctx(main)
    ref.ctx_x = ref.ctx_x[-win:]

    got_toks, _, got_logits = model.forward_batched(
        main,
        bonus,
        start,
        kv_windows=kv_windows,
        slots=torch.tensor([0], device=device),
        return_logits=True,
    )

    # Reference drafts with positions: ctx = last `win` absolute positions.
    class _WrapRef(_Ref):
        pass

    wref = _WrapRef(weights, device)
    wref.ctx_x = ref.ctx_x
    # override position bookkeeping: ctx positions are total+1-win .. total
    w = wref.w
    ids = torch.full((BLOCK,), MASK_ID, dtype=torch.long, device=device)
    ids[0] = bonus[0]
    hh = F.embedding(ids, w["embed_tokens.weight"])
    pos_q = start[0] + torch.arange(BLOCK, device=device)
    pos_c = torch.arange(start[0] - win, start[0], device=device)
    for i in range(N_LAYERS):
        p = f"layers.{i}."
        x = wref._norm(hh, p + "input_layernorm.weight")
        q = F.linear(x, w[p + "self_attn.q_proj.weight"]).view(BLOCK, N_HEADS, HEAD_DIM)
        q = wref._norm(q, p + "self_attn.q_norm.weight")
        q = _apply_rope(q, wref.cos[pos_q].to(DTYPE), wref.sin[pos_q].to(DTYPE))
        src = torch.cat([wref.ctx_x, x], dim=0)
        pos_k = torch.cat([pos_c, pos_q])
        k = F.linear(src, w[p + "self_attn.k_proj.weight"]).view(-1, N_KV_HEADS, HEAD_DIM)
        k = wref._norm(k, p + "self_attn.k_norm.weight")
        k = _apply_rope(k, wref.cos[pos_k].to(DTYPE), wref.sin[pos_k].to(DTYPE))
        v = F.linear(src, w[p + "self_attn.v_proj.weight"]).view(-1, N_KV_HEADS, HEAD_DIM)
        rep = N_HEADS // N_KV_HEADS
        kk = k.transpose(0, 1).repeat_interleave(rep, dim=0)
        vv = v.transpose(0, 1).repeat_interleave(rep, dim=0)
        o = F.scaled_dot_product_attention(q.transpose(0, 1), kk, vv, scale=HEAD_DIM**-0.5)
        o = o.transpose(0, 1).reshape(BLOCK, N_HEADS * HEAD_DIM)
        hh = hh + F.linear(o, w[p + "self_attn.o_proj.weight"])
        x = wref._norm(hh, p + "post_attention_layernorm.weight")
        mlp = F.linear(
            F.silu(F.linear(x, w[p + "mlp.gate_proj.weight"]))
            * F.linear(x, w[p + "mlp.up_proj.weight"]),
            w[p + "mlp.down_proj.weight"],
        )
        hh = hh + mlp
    hh = wref._norm(hh, "norm.weight")
    base = F.linear(hh, w["lm_head.weight"])
    toks, logits = [], []
    prev = bonus.long()
    for kstep in range(BLOCK):
        bias = F.linear(
            F.embedding(prev, w["markov_head.markov_w1.weight"]), w["markov_head.markov_w2.weight"]
        )
        step = base[kstep : kstep + 1] + bias
        logits.append(step)
        prev = step.argmax(dim=-1)
        toks.append(prev)
    torch.testing.assert_close(
        got_logits[0].float(), torch.cat(logits).float(), atol=0.05, rtol=0.05
    )
    assert torch.equal(got_toks[0].long(), torch.cat(toks).long())
