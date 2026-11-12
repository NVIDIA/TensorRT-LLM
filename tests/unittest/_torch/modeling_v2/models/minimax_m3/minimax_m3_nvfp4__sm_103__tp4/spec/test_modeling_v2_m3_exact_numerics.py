# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The tp4 target's speculative step against the in-tree (stock) Eagle3 numerics, at the real TP4 shapes.

* Verify: the target logits are the in-tree BF16 LM head's own shard GEMM, and the fused shard argmax + acceptance
  gives the in-tree greedy acceptance of the gathered logits, bit for bit, on every rank (peers' keys are the real
  per-shard maxima, computed by the same kernel).
* Drafter: the worker runs the in-tree drafter's modules and weights; its own drafter arithmetic is the first pass's
  fc_norm (chunk_norm, or capture_norm folded into the capture), held here to the in-tree ``apply_eagle3_fc`` RMSNorm
  modules within one bf16 ulp (its effect on the fc output and on the full-vocabulary BF16 head's draft tokens is
  measured too), and the fused draft loop's draft input and BF16 head, held to the in-tree modules and greedy draft
  sampling by their own tests (test_modeling_v2_m3_draft_input.py, test_modeling_v2_m3_draft_step.py).

Synthetic weights: hidden 6144, vocabulary 200064 (4 x 50016 shards), 3 captured layers, draft length 3.
"""

import importlib.util
from types import SimpleNamespace

import pytest
import torch

from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4 import (  # noqa: E501
    modeling,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops import (  # noqa: E501
    m3_spec_tail,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.spec.m3_shard_accept import (  # noqa: E501
    SHARD_ROWS,
    make_keys,
)
from tensorrt_llm._torch._experimental.modeling_v2.models.minimax_m3.minimax_m3_nvfp4__sm_103__tp4.ops.spec.m3_shard_accept import (  # noqa: E501
    m3_shard_accept as shard_accept_op,
)
from tensorrt_llm._torch.models.modeling_speculative import Eagle3ForCausalLM
from tensorrt_llm._torch.modules.embedding import LMHead
from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.modules.rms_norm import RMSNorm
from tensorrt_llm._torch.speculative.eagle3 import Eagle3OneModelWorker

__extra_import_path__ = ["../../../..", "."]
from _m3_spec_inputs import (  # noqa: E402
    EPS,
    HIDDEN,
    TARGET_SHARD,
    VOCAB,
    WORLD,
    assert_bf16_close,
    craft_drafts,
    gen,
    head_workspace,
    num_sms,
    randn,
    seed_of,
)

assert torch.cuda.is_available(), "the exact-numerics checks require a CUDA device"

DRAFT_LEN = 3
CAPTURED_LAYERS = 3
#: Largest share of fc_norm outputs one bf16 ulp off the in-tree RMSNorm (fp32 statistic summed in another order).
NORM_ULP_FRAC = 1e-3


def _stock_accept(logits: torch.Tensor, draft: torch.Tensor):
    """The in-tree greedy strict acceptance (``SpecWorkerBase._sample_and_accept_draft_tokens_base``) of the gathered
    float logits of a decode-only batch, no forcing."""
    md = SimpleNamespace(
        is_all_greedy_sample=True,
        is_ragged_verify=False,
        is_cuda_graph=False,
        sharded_target_logits=False,
    )
    worker = SimpleNamespace(force_num_accepted_tokens=0.0)
    worker._sample_tokens_for_batch = lambda *a: Eagle3OneModelWorker._sample_tokens_for_batch(
        worker, *a
    )
    worker._apply_force_accepted_tokens = lambda *a, **k: (
        Eagle3OneModelWorker._apply_force_accepted_tokens(worker, *a, **k)
    )
    worker._fused_force_params = lambda *a: Eagle3OneModelWorker._fused_force_params(worker, *a)
    nreq = draft.shape[0]
    return Eagle3OneModelWorker._sample_and_accept_draft_tokens_base(
        worker, logits, draft, 0, nreq, md
    )


def _lm_head_shard(weight: torch.Tensor, rank: int) -> LMHead:
    """Rank ``rank``'s local part of the target's vocab-sharded BF16 LM head at TP4: an LM head over that rank's
    50016 rows (the TP4 module's local GEMM is this module's; its all-gather is the caller's)."""
    head = LMHead(num_embeddings=TARGET_SHARD, embedding_dim=HIDDEN, dtype=torch.bfloat16)
    head.weight.data = weight[rank * TARGET_SHARD : (rank + 1) * TARGET_SHARD].contiguous()
    return head


def _run_shard_accept(logits, rank, world, draft, keys_of_peers=None, call=0):
    """The fused verify of one rank: its shard logits, the peers' keys planted in its slots."""
    nreq = draft.shape[0]
    rows = logits.shape[0]
    uc, keys, counters = head_workspace(world, call=call, slot_rows=SHARD_ROWS)
    half = call & 1
    for peer, peer_keys in (keys_of_peers or {}).items():
        for t in range(rows):
            uc[(half * SHARD_ROWS + t) * world + peer] = peer_keys[t]
    acc = torch.full((nreq, DRAFT_LEN + 1), -7, dtype=torch.int32, device="cuda")
    num = torch.full((nreq,), -7, dtype=torch.int32, device="cuda")
    gather = torch.full((nreq,), -7, dtype=torch.int64, device="cuda")
    next_new = torch.full((nreq, DRAFT_LEN + 1), -7, dtype=torch.int32, device="cuda")
    batch_idx = torch.arange(nreq, dtype=torch.int32, device="cuda")
    shard_accept_op(
        logits, rank * TARGET_SHARD if world > 1 else 0, keys, counters, uc, uc.data_ptr(), rank, world,
        draft, batch_idx, acc, num, gather, 0, next_new, num_sms(), None,
    )  # fmt: skip
    torch.cuda.synchronize()
    return acc, num, gather, next_new


def _local_keys(logits: torch.Tensor, rank: int) -> torch.Tensor:
    """What rank ``rank`` pushes: the key of its own shard maximum per row, found by the same kernel at world 1."""
    rows = logits.shape[0]
    nreq = rows // (DRAFT_LEN + 1)
    dummy = torch.full((nreq, DRAFT_LEN), -1, dtype=torch.int32, device="cuda")
    acc, _, _, _ = _run_shard_accept(logits, 0, 1, dummy)
    local = acc.view(-1).to(torch.int64)
    value = logits.float().gather(1, local[:, None]).squeeze(1)
    return make_keys(value, local + rank * TARGET_SHARD)


@pytest.mark.parametrize(
    "nreq,ties", [(1, False), (2, False), (4, False), (8, True), (12, False)], ids=str
)
def test_fused_verify_is_the_stock_greedy_acceptance(nreq, ties) -> None:
    """Target logits (in-tree BF16 head) and verify outputs bit-identical to the in-tree path, on all four ranks."""
    g = gen(seed_of(f"verify {nreq} {ties}"))
    rows = nreq * (DRAFT_LEN + 1)
    x = randn((rows, HIDDEN), g, 1.0)
    weight = randn((VOCAB, HIDDEN), g, 0.02)
    if ties:
        # Two equal head rows on ranks 1 and 3 that win every row: the lower global id must be the token.
        win = (x.float().sum(0) / x.float().sum(0).norm() * 4.0).to(torch.bfloat16)
        weight[TARGET_SHARD + 77] = win
        weight[3 * TARGET_SHARD + 5] = win
    heads = [_lm_head_shard(weight, r) for r in range(WORLD)]
    # Stock: each rank's LMHead forward, all-gathered, then LogitsProcessor's float cast.
    shard_logits = [head(x) for head in heads]
    for head, stock in zip(heads, shard_logits):
        assert torch.equal(head.apply_linear(x, None), stock), (
            "the verify's shard GEMM is not the head's"
        )
    gathered = torch.cat(shard_logits, dim=1).float()
    target_tokens = gathered.argmax(dim=-1).to(torch.int32)
    if ties:
        assert bool((target_tokens == TARGET_SHARD + 77).all())
    draft = craft_drafts(target_tokens, nreq, DRAFT_LEN)
    stock_acc, stock_num = _stock_accept(gathered, draft)
    peer_keys = {r: _local_keys(shard_logits[r], r) for r in range(WORLD)}
    for rank in range(WORLD):
        peers = {r: k for r, k in peer_keys.items() if r != rank}
        acc, num, gather, next_new = _run_shard_accept(
            shard_logits[rank], rank, WORLD, draft, peers
        )
        assert torch.equal(acc, stock_acc.to(torch.int32)), f"rank {rank}: accepted tokens"
        assert torch.equal(num, stock_num.to(torch.int32)), f"rank {rank}: accepted counts"
        expect_gather = torch.arange(nreq, device="cuda") * (DRAFT_LEN + 1) + num.long() - 1
        assert torch.equal(gather, expect_gather), f"rank {rank}: first-pass gather rows"
        assert torch.equal(next_new[:, 0], target_tokens[expect_gather]), f"rank {rank}: next_new"


class _Drafter:
    """The in-tree Eagle3 first-pass pieces at the target's shapes: per-layer fc_norm RMSNorms and the fc."""

    def __init__(self, g: torch.Generator):
        self.norms = []
        for _ in range(CAPTURED_LAYERS):
            norm = RMSNorm(hidden_size=HIDDEN, eps=EPS, dtype=torch.bfloat16).cuda()
            norm.weight.data = (1.0 + 0.1 * torch.randn(HIDDEN, generator=g, device="cuda")).to(
                torch.bfloat16
            )
            self.norms.append(norm)
        self.fc = Linear(HIDDEN * CAPTURED_LAYERS, HIDDEN, bias=False, dtype=torch.bfloat16).cuda()
        self.fc.weight.data = randn((HIDDEN, HIDDEN * CAPTURED_LAYERS), g, 0.02)
        self.model = SimpleNamespace(
            dtype=torch.bfloat16,
            hidden_size=HIDDEN,
            fc_norm=torch.nn.ModuleList(self.norms),
            _norm_before_fc=False,
            fc=self.fc,
        )
        self.stacked = torch.stack([n.weight.detach() for n in self.norms]).contiguous()

    def stock_normed(self, hs: torch.Tensor) -> torch.Tensor:
        """``apply_eagle3_fc``'s fc input: each captured layer through its own in-tree RMSNorm."""
        chunks = hs.chunk(CAPTURED_LAYERS, dim=-1)
        return torch.cat([norm(c) for norm, c in zip(self.norms, chunks)], dim=-1)

    def stock_fc(self, hs: torch.Tensor) -> torch.Tensor:
        return Eagle3ForCausalLM.apply_eagle3_fc(self, hs)


def _captures(rows: int, g: torch.Generator):
    """Per captured layer (hidden, residual) and the in-tree capture buffer of bf16(hidden + residual)."""
    layers = [(randn((rows, HIDDEN), g, 1.5), randn((rows, HIDDEN), g, 3.0)) for _ in range(3)]
    buf = torch.cat([h + r for h, r in layers], dim=-1)
    return layers, buf


def _row_identical(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a == b).all(dim=-1)


@pytest.mark.parametrize("rows", (4, 8, 32, 48), ids=lambda r: f"rows{r}")
def test_first_drafter_inputs_are_the_stock_fc_norm(rows) -> None:
    """chunk_norm and capture_norm against ``apply_eagle3_fc``'s in-tree RMSNorms: within one bf16 ulp on at most
    NORM_ULP_FRAC of the elements; the fc (the in-tree module in both paths) then differs only on rows whose input
    did."""
    g = gen(seed_of(f"fc_norm {rows}"))
    drafter = _Drafter(g)
    layers, buf = _captures(rows, g)
    stock_in = drafter.stock_normed(buf)
    stock_out = drafter.stock_fc(buf)
    assert torch.equal(drafter.fc(stock_in), stock_out)

    chunk_in = m3_spec_tail.chunk_norm(buf, drafter.stacked, EPS)
    capture_in = torch.empty_like(buf)
    for i, (h, r) in enumerate(layers):
        m3_spec_tail.capture_norm(
            h, r, drafter.stacked[i], EPS, capture_in[:, i * HIDDEN : (i + 1) * HIDDEN]
        )
    torch.cuda.synchronize()
    for name, ours in (("chunk_norm", chunk_in), ("capture_norm", capture_in)):
        assert_bf16_close(ours, stock_in, max_frac=NORM_ULP_FRAC)
        out = drafter.fc(ours)
        same_rows = _row_identical(ours, stock_in)
        assert torch.equal(out[same_rows], stock_out[same_rows]), (
            f"{name}: fc moved an unchanged row"
        )
        moved = (ours != stock_in).float().mean().item()
        print(
            f"[exact-numerics] rows={rows} {name}: {moved:.2e} of fc inputs one ulp off, "
            f"{int((~same_rows).sum())}/{rows} rows differ, fc max |diff| "
            f"{(out.float() - stock_out.float()).abs().max().item():.3e}"
        )
    assert torch.equal(chunk_in, capture_in), "capture_norm and chunk_norm disagree"


@pytest.mark.parametrize("rows", (4, 32), ids=lambda r: f"rows{r}")
def test_draft_tokens_through_the_full_vocab_head(rows) -> None:
    """The fc_norm's ulp-level differences carried through the full-vocabulary BF16 head (a proxy of the draft head
    input): the greedy draft token equals the stock one on every row whose stock top-2 margin exceeds the logit
    perturbation."""
    g = gen(seed_of(f"draft head {rows}"))
    drafter = _Drafter(g)
    _, buf = _captures(rows, g)
    weight = randn((VOCAB, HIDDEN), g, 0.02)
    head = LMHead(num_embeddings=VOCAB, embedding_dim=HIDDEN, dtype=torch.bfloat16)
    head.weight.data = weight
    stock_logits = head(drafter.stock_fc(buf)).float()
    ours_logits = head(drafter.fc(m3_spec_tail.chunk_norm(buf, drafter.stacked, EPS))).float()
    perturb = (ours_logits - stock_logits).abs().max(dim=-1).values
    top2 = stock_logits.topk(2, dim=-1).values
    decidable = (top2[:, 0] - top2[:, 1]) > 2 * perturb
    stock_tok, ours_tok = stock_logits.argmax(-1), ours_logits.argmax(-1)
    assert torch.equal(ours_tok[decidable], stock_tok[decidable])
    print(
        f"[exact-numerics] rows={rows} draft head: {int((ours_tok == stock_tok).sum())}/{rows} tokens equal, "
        f"{int(decidable.sum())} decidable, max logit |diff| {perturb.max().item():.3e}"
    )


def test_the_drafter_is_the_in_tree_drafter() -> None:
    """Outside the fused draft loop's steps the worker drafts with the in-tree loop, forward and sampling, and nothing
    re-quantizes the drafter's linears or the LM heads: no hook of a quantized / pruned head or drafter projection is
    left in the target, and the fused head streams the BF16 head weight itself."""
    assert (
        modeling.M3Eagle3Worker._forward_draft_loop is not Eagle3OneModelWorker._forward_draft_loop
    )
    for name in (
        "_forward_linear_draft_loop",
        "_run_draft_forward",
        "sample_draft_tokens",
        "sample_and_accept_draft_tokens",
    ):
        assert getattr(modeling.M3Eagle3Worker, name) is getattr(Eagle3OneModelWorker, name), name
    for module, name in (
        (modeling, "M3DraftGemvLinearMethod"),
        (modeling, "adopt_draft_linears"),
        (modeling, "greedy_head_outputs"),
        (modeling, "M3FP8LMHeadMethod"),
        (modeling, "adopt_fp8_lm_heads"),
    ):
        assert not hasattr(module, name), f"{module.__name__}.{name}"
    pkg = modeling.__name__.rsplit(".", 1)[0]
    for gone in ("ops.m3_draft_head", "ops.spec.m3_draft_head", "ops.gemm.m3_draft_gemv"):
        assert importlib.util.find_spec(f"{pkg}.{gone}") is None, gone
