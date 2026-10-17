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
"""Single-GPU correctness of the draft_tail against the native oracle.

The eight TP ranks of GLM-5.2 are emulated shard by shard on one GPU: each
shard runs the draft_tail kernel without exchange (``tp_size=1`` runner, explicit
vocab offset) and returns its packed key; the global decision is the unsigned
max of the eight keys, exactly what the exchange computes on every rank. The
cross-rank exchange itself needs 8 GPUs and is not tested here.
"""

import json
import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import draft_tail_nearties as nt  # noqa: E402
import draft_tail_oracle as oracle  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")

NUM_VARIANTS = 15
FIRST_MMA_VARIANT = 7
NATIVE_ORDER_VARIANTS = (11, 12, 13, 14)


def _family(variant):
    """Kernels of one family accumulate a row in the same order: CUDA-core
    (0-6), tensor-core permuted K (7-10), tensor-core native K order
    (11-14)."""
    if variant < FIRST_MMA_VARIANT:
        return 0
    return 11 if variant in NATIVE_ORDER_VARIANTS else FIRST_MMA_VARIANT


def _draft_tail():
    import tensorrt_llm  # noqa: F401  (loads the trtllm torch ops)
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.draft_tail import (  # noqa: E501
        draft_tail,
    )

    return draft_tail


@pytest.fixture(scope="module")
def runner():
    dt = _draft_tail()
    r = dt.DraftTailRunner(0, 1)
    assert r.prepare()
    return r


@pytest.fixture(scope="module")
def shards():
    if not oracle.checkpoint_available():
        pytest.skip("GLM-5.2 checkpoint not mounted")
    return oracle.load_lm_head_shards()


@pytest.fixture(scope="module")
def gamma():
    if not oracle.checkpoint_available():
        pytest.skip("GLM-5.2 checkpoint not mounted")
    return oracle.mtp_norm_weight()


class _Head:
    """Minimal stand-in exposing ``weight`` like LMHead."""

    def __init__(self, weight):
        self.weight = weight


@pytest.fixture(scope="module")
def full_head(shards):
    """The whole vocabulary as one shard: with a tp_size 1 runner the draft_tail
    decision over it is the global decision of the 8 shards (the packed-key
    max is associative and every row's dot product is computed alike)."""
    return _Head(torch.cat(shards, dim=0))


def _full_keys(runner, hidden, head, variant):
    keys = torch.zeros(hidden.shape[0], dtype=torch.int64, device="cuda")
    runner.run_fused(hidden, head.weight, 0, keys_out=keys, variant=variant)
    return keys


def _e2e_near_tie_rows():
    """Draft LM-head input rows of the real BF16 near-ties found end to end.

    ``GLM_MOE_DSA_DRAFT_TAIL_NEAR_TIE_DIR`` is a directory with the native run's captured draft LM-head
    inputs (with each draft-loop call's first row) and ``tie_analysis.json``.
    """
    d = os.environ.get("GLM_MOE_DSA_DRAFT_TAIL_NEAR_TIE_DIR")
    if not d:
        return []
    ties = [t for t in json.load(open(os.path.join(d, "tie_analysis.json"))) if t.get("near_tie")]
    starts = torch.load(os.path.join(d, "native_eager_draft.pt"))["start_rows"]
    rows = torch.load(os.path.join(d, "native_eager_hidden.pt"))["rows"]
    return [
        (f"e2e call {t['call']} step {t['step']}", rows[starts[t["call"]] + t["step"]].to("cuda"))
        for t in ties
    ]


def _family_split_rows(runner, head, gamma, want=4):
    """Constructed BF16 near-ties: seeded flat MTP-normed rows whose winner
    the CUDA-core family (variant 4) and the tensor-core family (variant 10)
    round differently. ``GLM_MOE_DSA_DRAFT_TAIL_FAMILY_SPLIT_BATCHES`` bounds the search."""
    batches = int(os.environ.get("GLM_MOE_DSA_DRAFT_TAIL_FAMILY_SPLIT_BATCHES", "512"))
    gen = torch.Generator(device="cuda").manual_seed(2026)
    found = []
    for b in range(batches):
        h = oracle.synthetic_hidden(8, gamma, gen)
        a = oracle.keys_to_tokens(_full_keys(runner, h, head, 4))
        c = oracle.keys_to_tokens(_full_keys(runner, h, head, 10))
        for i in torch.nonzero(a != c).flatten().tolist():
            found.append((f"family split, search batch {b} row {i}", h[i]))
        if len(found) >= want:
            break
    return found[:want]


@pytest.fixture(scope="module")
def live_rows(runner, full_head, gamma):
    """(label, hidden row) pairs used as live rows: real and constructed
    BF16 near-ties on which the two kernel families disagree, plus random
    flat and peaked rows."""
    gen = torch.Generator(device="cuda").manual_seed(31)
    rows = _e2e_near_tie_rows()
    rows += _family_split_rows(runner, full_head, gamma)
    rows.append(("flat", oracle.synthetic_hidden(1, gamma, gen)[0]))
    peak = full_head.weight[123456:123457]
    rows.append(("peaked", oracle.synthetic_hidden(1, gamma, gen, peak, 0.3)[0]))
    return rows


def _draft_tail_keys_fused(runner, hidden, shards, **kw):
    keys = []
    for r, w in enumerate(shards):
        k = torch.zeros(hidden.shape[0], dtype=torch.int64, device="cuda")
        runner.run_fused(hidden, w, r * w.shape[0], keys_out=k, **kw)
        keys.append(k)
    return keys


def _draft_tail_keys_logits(runner, shard_logits):
    keys = []
    for r, lg in enumerate(shard_logits):
        k = torch.zeros(lg.shape[0], dtype=torch.int64, device="cuda")
        runner.run_from_logits(lg, r * lg.shape[1], keys_out=k)
        keys.append(k)
    return keys


# ---------------------------------------------------------------------------
# Selection semantics on given BF16 logits (logits kernel): exact equality.
# ---------------------------------------------------------------------------


def _random_logits(rows, gen, tp=oracle.TP, shard=oracle.SHARD):
    return [
        (torch.randn(rows, shard, generator=gen, device="cuda") * 3).to(torch.bfloat16)
        for _ in range(tp)
    ]


@pytest.mark.parametrize("rows", [1, 2, 6, 8])
def test_logits_selection_matches_native_random(runner, rows):
    gen = torch.Generator(device="cuda").manual_seed(rows)
    for _ in range(20):
        lg = _random_logits(rows, gen)
        ref = oracle.native_tokens_from_logits(lg)
        got = oracle.keys_to_tokens(oracle.global_key(_draft_tail_keys_logits(runner, lg)))
        torch.testing.assert_close(got, ref, rtol=0, atol=0)


def test_logits_selection_ties_and_specials(runner):
    """Cross-shard and in-shard ties, -0/+0, infinities, NaN."""
    gen = torch.Generator(device="cuda").manual_seed(7)
    rows = 8
    cases = []
    lg = _random_logits(rows, gen)
    # row 0: same max value in shards 5 and 2 -> lower global id (shard 2)
    for r, c in ((5, 100), (2, 19000)):
        lg[r][0, c] = 1000.0
    # row 1: tie inside one shard at two columns -> lower column
    lg[3][1, 777] = 1000.0
    lg[3][1, 12] = 1000.0
    # row 2: every logit equal -> global id 0
    for r in range(oracle.TP):
        lg[r][2].fill_(-1.5)
    # row 3: max is +0 vs -0 at different ids -> treated equal -> lower id
    for r in range(oracle.TP):
        lg[r][3].fill_(-5.0)
    lg[6][3, 3] = 0.0
    lg[1][3, 9] = -0.0
    # row 4: +inf in shard 4, large finite elsewhere
    lg[4][4, 55] = float("inf")
    lg[0][4, 1] = 3e38
    # row 5: all -inf except one -inf-adjacent value
    for r in range(oracle.TP):
        lg[r][5].fill_(float("-inf"))
    lg[7][5, 19359] = -3e38
    # row 6: tie across the last column of shard 0 and first of shard 1
    lg[0][6, oracle.SHARD - 1] = 500.0
    lg[1][6, 0] = 500.0
    # row 7: NaN in shard 3 (native: NaN ranks above every number)
    lg[3][7, 4242] = float("nan")
    cases.append(lg)
    for lg in cases:
        ref = oracle.native_tokens_from_logits(lg)
        got = oracle.keys_to_tokens(oracle.global_key(_draft_tail_keys_logits(runner, lg)))
        torch.testing.assert_close(got, ref, rtol=0, atol=0)
    assert got[0].item() == 2 * oracle.SHARD + 19000
    assert got[1].item() == 3 * oracle.SHARD + 12
    assert got[2].item() == 0


def test_logits_selection_padded_vocab(runner):
    """Vocab not divisible by TP: the last shard carries zero padding rows.

    The native path includes the padding columns of the last shard in its
    local max (they produce logit 0); draft_tail must make the same decision.
    """
    gen = torch.Generator(device="cuda").manual_seed(11)
    tp, shard = 8, 1000
    vocab = tp * shard - 37
    for rows in (1, 3):
        lg = [
            (torch.randn(rows, shard, generator=gen, device="cuda") - 4).to(torch.bfloat16)
            for _ in range(tp)
        ]
        lg[-1][:, vocab - (tp - 1) * shard :] = 0.0  # zero weight rows
        ref = oracle.native_tokens_from_logits(lg)
        got = oracle.keys_to_tokens(oracle.global_key(_draft_tail_keys_logits(runner, lg)))
        torch.testing.assert_close(got, ref, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Fused GEMV path on the real GLM-5.2 LM head.
# ---------------------------------------------------------------------------


def _assert_near_ties(reports):
    """A token mismatch is acceptable only when the two winners' exact logits
    are within one BF16 ulp of the maximum: both kernels then round FP32 sums
    of different accumulation order across the same BF16 boundary."""
    for rep in reports:
        top = max(abs(v) for v in rep["fp64"])
        ulp = 2.0 ** (int(torch.tensor(top).log2().floor().item()) - 7)
        assert abs(rep["fp64_gap"]) <= ulp, rep


def _decision_report(hidden, shards, got, ref):
    """For each mismatch, the FP64 logit gap between the two winners."""
    full = torch.cat(shards, dim=0)
    out = []
    for i in torch.nonzero(got != ref).flatten().tolist():
        ids = torch.tensor([ref[i].item(), got[i].item()], device="cuda")
        exact = hidden[i : i + 1].double() @ full[ids].double().t()
        native_bf16 = F.linear(hidden[i : i + 1], full[ids])
        out.append(
            {
                "row": i,
                "native": ref[i].item(),
                "draft_tail": got[i].item(),
                "fp64_gap": (exact[0, 0] - exact[0, 1]).item(),
                "fp64": exact[0].tolist(),
                "native_bf16": native_bf16[0].float().tolist(),
            }
        )
    return out


@pytest.mark.parametrize("peak", [0.0, 0.3, 1.0])
def test_fused_matches_native_real_lm_head(runner, shards, gamma, peak):
    """Exact token equality vs native on the real head; any mismatch must be
    a BF16 rounding near-tie (FP64 gap below one BF16 ulp of the max)."""
    gen = torch.Generator(device="cuda").manual_seed(int(peak * 10) + 1)
    full_rows = torch.cat(shards, dim=0)
    rows_total, mismatches, reports = 0, 0, []
    for it in range(16):
        rows = 8
        target = torch.randint(0, oracle.VOCAB, (rows,), generator=gen, device="cuda")
        hidden = oracle.synthetic_hidden(
            rows, gamma, gen, peaked_from=full_rows[target] if peak > 0 else None, peak_scale=peak
        )
        ref = oracle.native_tokens(hidden, shards)
        got = oracle.keys_to_tokens(
            oracle.global_key(_draft_tail_keys_fused(runner, hidden, shards))
        )
        rows_total += rows
        bad = got != ref
        if bool(bad.any()):
            mismatches += int(bad.sum())
            reports += _decision_report(hidden, shards, got, ref)
    print(f"[draft_tail] peak={peak}: {mismatches}/{rows_total} token mismatches; {reports}")
    _assert_near_ties(reports)


@pytest.mark.parametrize("variant", list(range(NUM_VARIANTS)))
def test_fused_variants_bitwise_identical(runner, shards, gamma, variant):
    """Within a kernel family (CUDA-core 0-6, tensor-core 7-10, tensor-core
    native order 11-14) every shape, CTA count and L2 keep hint accumulates
    a row in the same order, so the keys are identical."""
    base_variant = _family(variant)
    gen = torch.Generator(device="cuda").manual_seed(3)
    for rows in (1, 2, 5, 8):
        hidden = oracle.synthetic_hidden(rows, gamma, gen)
        base = _draft_tail_keys_fused(runner, hidden, shards[:2], variant=base_variant)
        for num_ctas, keep in ((0, 0), (37, 0), (300, 0), (0, 500), (304, 1000)):
            got = _draft_tail_keys_fused(
                runner, hidden, shards[:2], variant=variant, num_ctas=num_ctas, keep_permille=keep
            )
            for a, b in zip(base, got):
                torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("variant", [0, FIRST_MMA_VARIANT, 11, 13])
@pytest.mark.parametrize("rows", [1, 3, 6, 8])
def test_fused_families_match_native(runner, shards, gamma, variant, rows):
    """Every kernel family makes the native decision on the real head for
    every row count (mismatches only at BF16 near-ties; none at all for the
    native-order family)."""
    gen = torch.Generator(device="cuda").manual_seed(100 + rows)
    reports = []
    for it in range(8):
        peak = (0.0, 0.3, 1.0)[it % 3]
        hidden = oracle.synthetic_hidden(
            rows, gamma, gen, torch.cat(shards[:1])[it * 8 : it * 8 + rows] if peak else None, peak
        )
        ref = oracle.native_tokens(hidden, shards)
        got = oracle.keys_to_tokens(
            oracle.global_key(_draft_tail_keys_fused(runner, hidden, shards, variant=variant))
        )
        reports += _decision_report(hidden, shards, got, ref)
    print(f"[draft_tail] variant={variant} rows={rows}: mismatches {reports}")
    _assert_near_ties(reports)
    if variant in NATIVE_ORDER_VARIANTS:
        assert not reports, reports


@pytest.mark.parametrize("variant", [4, 10, 11, 13])
def test_fused_family_rows_independent(runner, shards, gamma, variant):
    """Within one kernel, row r's key does not depend on the other rows of
    the call or on the row count."""
    gen = torch.Generator(device="cuda").manual_seed(5)
    live = oracle.synthetic_hidden(3, gamma, gen)
    w = shards[4]
    single = [
        _draft_tail_keys_fused(runner, live[i : i + 1], [w], variant=variant)[0] for i in range(3)
    ]
    for rows in range(3, 9):
        padded = torch.zeros(rows, oracle.HIDDEN, dtype=torch.bfloat16, device="cuda")
        padded[:3] = live
        if rows > 3:
            padded[3:] = oracle.synthetic_hidden(rows - 3, gamma, gen) * 7
        keys = _draft_tail_keys_fused(runner, padded, [w], variant=variant)[0]
        for i in range(3):
            assert keys[i].item() == single[i].item()


def test_select_live_row_key_independent_of_row_count(runner, full_head, gamma, live_rows):
    """The shipped dispatch (``runner.select``, as the model adapter calls
    it) gives a live row the same packed key, hence the same draft token,
    for every row count R = 1..8 and position, so CUDA-graph padding and
    batch composition cannot change a live row's decision. The live rows
    include real and constructed BF16 near-ties on which the two kernel
    families round differently: a dispatch that picked the family by row
    count would fail here."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.draft_tail.draft_tail import (  # noqa: E501
        DraftTailOptions,
    )

    assert runner.options == DraftTailOptions()  # the shipped defaults
    gen = torch.Generator(device="cuda").manual_seed(17)
    pad = torch.cat(
        [
            oracle.synthetic_hidden(4, gamma, gen) * 7,
            torch.zeros(4, oracle.HIDDEN, dtype=torch.bfloat16, device="cuda"),
        ]
    )
    split = 0
    for label, row in live_rows:
        v4 = int(_full_keys(runner, row[None], full_head, 4)[0])
        v10 = int(_full_keys(runner, row[None], full_head, 10)[0])
        vdef = int(_full_keys(runner, row[None], full_head, DraftTailOptions().variant)[0])
        logits = F.linear(row[None], full_head.weight)
        native_key = int(
            nt.bf16_key(logits, torch.arange(logits.shape[1], device="cuda")[None]).max()
        )
        split += int(v4 != v10)
        seen = set()
        for rows in range(1, 9):
            for pos in sorted({0, rows - 1}):
                batch = pad[:rows].clone()
                batch[pos] = row
                keys = torch.zeros(rows, dtype=torch.int64, device="cuda")
                tokens = runner.select(batch, full_head, keys_out=keys)
                assert int(tokens[pos]) == int(oracle.keys_to_tokens(keys[pos : pos + 1])[0])
                seen.add(int(keys[pos]))
        print(
            f"[draft_tail-rowcount] {label}: select keys over R=1..8 {seen}; "
            f"v4 token {int(oracle.keys_to_tokens(torch.tensor([v4])))} "
            f"v10 token {int(oracle.keys_to_tokens(torch.tensor([v10])))}"
        )
        assert len(seen) == 1, (label, seen)
        # The shipped kernel for every R is the default variant, and its key
        # (winning BF16 value and id) is native's.
        assert seen == {vdef}, (label, seen, vdef)
        assert vdef == native_key, (label, vdef, native_key)
    print(
        f"[draft_tail-rowcount] live rows {len(live_rows)}, rows where the two "
        f"families disagree: {split}"
    )


def test_row_count_characterization(runner, full_head, shards, gamma, live_rows):
    """For the record (no assertion on native): the native decision on each
    live row when it shares the LM-head GEMM with M - 1 other rows (8-shard
    native oracle; cuBLAS picks its kernel by M), next to both draft_tail kernel
    families and the FP64 logits of every candidate."""
    gen = torch.Generator(device="cuda").manual_seed(23)
    pad = oracle.synthetic_hidden(64, gamma, gen)
    full = full_head.weight
    for label, row in live_rows:
        native = {}
        bf16 = {}
        for m in (1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64):
            batch = pad[:m].clone()
            batch[0] = row
            logits = [F.linear(batch, w) for w in shards]
            tok = int(oracle.native_tokens_from_logits(logits)[0])
            native[m] = tok
            bf16[m] = logits[tok // oracle.SHARD][0, tok % oracle.SHARD].item()
        fam = {
            v: int(oracle.keys_to_tokens(_full_keys(runner, row[None], full_head, v))[0])
            for v in (4, 10)
        }
        cands = sorted(set(native.values()) | set(fam.values()))
        exact = (row[None].double() @ full[cands].double().t())[0].tolist()
        print(
            "[draft_tail-rowcount-native] "
            + json.dumps(
                {
                    "row": label,
                    "native_token_by_M": native,
                    "native_bf16_winner_logit_by_M": bf16,
                    "draft_tail_v4": fam[4],
                    "draft_tail_v10": fam[10],
                    "fp64_logits": dict(zip(cands, exact)),
                }
            )
        )


def test_fused_cuda_graph_replay(runner, shards, gamma):
    """Scratch keys and the arrival counter are reset by every launch:
    repeated replays with changing inputs keep matching native."""
    gen = torch.Generator(device="cuda").manual_seed(9)
    rows = 2
    static_h = torch.zeros(rows, oracle.HIDDEN, dtype=torch.bfloat16, device="cuda")
    static_keys = [torch.zeros(rows, dtype=torch.int64, device="cuda") for _ in shards]

    def step():
        for r, w in enumerate(shards):
            runner.run_fused(static_h, w, r * w.shape[0], keys_out=static_keys[r])

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        step()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    for it in range(50):
        static_h.copy_(oracle.synthetic_hidden(rows, gamma, gen))
        g.replay()
        torch.cuda.synchronize()
        got = oracle.keys_to_tokens(oracle.global_key(static_keys))
        eager = oracle.keys_to_tokens(
            oracle.global_key(_draft_tail_keys_fused(runner, static_h, shards))
        )
        torch.testing.assert_close(got, eager, rtol=0, atol=0)
        ref = oracle.native_tokens(static_h, shards)
        _assert_near_ties(_decision_report(static_h, shards, got, ref))
    assert runner.scratch[:8].abs().sum().item() == 0
    assert runner.scratch[8].item() & 0xFFFFFFFF == 0


def test_runner_tp1_tokens_match_native_full_vocab(runner, shards, gamma):
    """tp_size == 1 (no exchange): one shard is the whole local vocab; the
    native path is a plain argmax over the logits."""
    gen = torch.Generator(device="cuda").manual_seed(13)
    w = shards[1]
    for rows in (1, 4, 8):
        h = oracle.synthetic_hidden(rows, gamma, gen, w[:rows], 1.0)
        tokens = runner.run_fused(h, w, 0)
        ref = torch.argmax(F.linear(h, w), dim=-1).to(torch.int32)
        torch.testing.assert_close(tokens, ref, rtol=0, atol=0)


@pytest.mark.skipif(
    not os.environ.get("GLM_MOE_DSA_DRAFT_TAIL_HIDDEN_FIXTURE"),
    reason="GLM_MOE_DSA_DRAFT_TAIL_HIDDEN_FIXTURE (captured draft LM-head inputs) not set",
)
@pytest.mark.parametrize("variant", [0, FIRST_MMA_VARIANT, 11, 13])
def test_real_hidden_states_exact_tokens(runner, shards, variant):
    """Real MTP draft LM-head inputs captured from a native GLM-5.2 run:
    token-exact vs native for every row, through
    the fused kernel (both families) and the logits kernel; any mismatch must
    be a BF16 near-tie."""
    rows = torch.load(os.environ["GLM_MOE_DSA_DRAFT_TAIL_HIDDEN_FIXTURE"])["rows"].to("cuda")
    assert rows.dtype == torch.bfloat16 and rows.shape[1] == oracle.HIDDEN
    total = fused_bad = logits_bad = 0
    reports = []
    for i in range(0, rows.shape[0], 8):
        h = rows[i : i + 8].contiguous()
        logits = [F.linear(h, w) for w in shards]
        ref = oracle.native_tokens_from_logits(logits)
        got_f = oracle.keys_to_tokens(
            oracle.global_key(_draft_tail_keys_fused(runner, h, shards, variant=variant))
        )
        got_l = oracle.keys_to_tokens(oracle.global_key(_draft_tail_keys_logits(runner, logits)))
        # The logits kernel consumes the native BF16 logits: always exact.
        torch.testing.assert_close(got_l, ref, rtol=0, atol=0)
        total += h.shape[0]
        fused_bad += int((got_f != ref).sum())
        logits_bad += int((got_l != ref).sum())
        reports += _decision_report(h, shards, got_f, ref)
    print(
        f"[draft_tail] real hidden rows={total} variant={variant}: fused "
        f"mismatches={fused_bad}, logits mismatches={logits_bad}; "
        f"{reports}"
    )
    _assert_near_ties(reports)
    if variant in NATIVE_ORDER_VARIANTS:
        assert fused_bad == 0


def _exactness_rows(full_head, gamma):
    """Rows for the bit-level comparison: real draft LM-head inputs
    (``GLM_MOE_DSA_DRAFT_TAIL_HIDDEN_FIXTURE``, first ``GLM_MOE_DSA_DRAFT_TAIL_EXACT_ROWS``),
    knife-edge rows built from the real rows with the smallest top-2 gaps (their two best logits
    straddle a BF16 rounding midpoint within a fraction of an FP32 ulp, so
    any difference in their FP32 sums changes the BF16 values), and
    synthetic flat / peaked MTP-normed rows."""
    gen = torch.Generator(device="cuda").manual_seed(41)
    rows = [oracle.synthetic_hidden(16, gamma, gen)]
    rows.append(oracle.synthetic_hidden(16, gamma, gen, full_head.weight[1000:1016], 0.5))
    path = os.environ.get("GLM_MOE_DSA_DRAFT_TAIL_HIDDEN_FIXTURE")
    if path:
        n = int(os.environ.get("GLM_MOE_DSA_DRAFT_TAIL_EXACT_ROWS", "256"))
        real = torch.load(path)["rows"][:n].to("cuda")
        rows.append(real)
        shards = [
            full_head.weight[r * oracle.SHARD : (r + 1) * oracle.SHARD] for r in range(oracle.TP)
        ]
        logits = nt.native_bf16_logits(real, shards, 8)
        ids, vals = nt.topk_fp64(real, full_head.weight, logits)
        gaps = (vals[:, 0] - vals[:, 1]) / nt.bf16_ulp(vals[:, 0])
        base = gaps.argsort()[:16]
        lo = torch.minimum(ids[base, 0], ids[base, 1])
        hi = torch.maximum(ids[base, 0], ids[base, 1])
        for up, down in ((lo, hi), (hi, lo)):
            res = nt.construct_straddle(
                real[base], full_head.weight, up, down, torch.full((16,), 0.5, device="cuda")
            )
            rows.append(res["hidden"])
    return torch.cat(rows)


@pytest.mark.parametrize("variant", list(NATIVE_ORDER_VARIANTS))
def test_native_order_logits_bitwise_equal_native(runner, shards, full_head, gamma, variant):
    """The native-order fused variants compute every BF16 logit bit for bit
    like the native LM-head GEMM (``F.linear``), for every row count, so
    their draft tokens are native ones. Variant 10 (permuted K order) is
    measured on the same rows for comparison."""
    rows = _exactness_rows(full_head, gamma)
    diffs = {variant: 0, 10: 0}
    total = 0
    for i in range(0, rows.shape[0], 8):
        for r in (1, 3, 8):
            h = rows[i : i + r].contiguous()
            for s, w in enumerate(shards):
                ref = F.linear(h, w)
                for v in diffs:
                    out = torch.empty_like(ref)
                    runner.run_fused(h, w, s * w.shape[0], variant=v, logits_out=out)
                    diffs[v] += int((out.view(torch.int16) != ref.view(torch.int16)).sum())
                total += ref.numel()
    print(
        f"[draft_tail-exact] rows={rows.shape[0]} logits compared={total} "
        f"BF16 logits != native: {diffs}"
    )
    assert diffs[variant] == 0


def _loaded_cublas_versions():
    """cuBLAS / cuBLASLt libraries mapped into this process and their
    versions (the native LM-head GEMM's kernel is chosen by these)."""
    import ctypes

    paths = set()
    with open("/proc/self/maps") as f:
        for line in f:
            p = line.split()[-1]
            if "libcublas" in os.path.basename(p) and ".so" in p:
                paths.add(p)
    out = {}
    for p in sorted(paths):
        lib = ctypes.CDLL(p)
        fn = "cublasLtGetProperty" if "cublasLt" in p else "cublasGetProperty"
        try:
            ver = []
            for prop in range(3):  # MAJOR_VERSION, MINOR_VERSION, PATCH_LEVEL
                v = ctypes.c_int(-1)
                getattr(lib, fn)(prop, ctypes.byref(v))
                ver.append(v.value)
            out[p] = ".".join(str(v) for v in ver)
        except AttributeError:
            out[p] = None
    return out


def test_native_lm_head_kernel_record(shards):
    """For the record (no assertion): the GEMM kernel this image's cuBLAS
    picks for the native draft LM head (``F.linear``, one vocab shard) at
    M = 1, 3, 8, and the library versions. The native-order bitwise claim
    is tied to this kernel; a different name on another image is the cue to
    re-run ``test_native_order_logits_bitwise_equal_native``."""
    from torch.profiler import ProfilerActivity, profile

    h = torch.randn(8, oracle.HIDDEN, device="cuda").to(torch.bfloat16)
    w = shards[0]
    kernels = {}
    for m in (1, 3, 8):
        F.linear(h[:m], w)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            F.linear(h[:m], w)
            torch.cuda.synchronize()
        kernels[m] = sorted(
            {e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA}
        )
    try:
        blas = str(torch.backends.cuda.preferred_blas_library())
    except (AttributeError, RuntimeError):
        blas = None
    print(
        "[draft_tail-nvjet-record] "
        + json.dumps(
            {
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(0),
                "capability": list(torch.cuda.get_device_capability(0)),
                "preferred_blas": blas,
                "cublas_libs": _loaded_cublas_versions(),
                "env": {
                    k: v
                    for k, v in os.environ.items()
                    if k.startswith(("CUBLAS", "TORCH_BLAS", "NVIDIA_TF32"))
                },
                "kernels_by_M": kernels,
            }
        )
    )


@pytest.mark.parametrize("m", [1, 8])
def test_native_order_logits_bitwise_equal_native_in_graph(runner, shards, full_head, gamma, m):
    """The served draft head runs inside a CUDA graph. Capture the native
    LM-head GEMM and the shipped default draft_tail kernel (``DraftTailOptions().variant``)
    per shard in graphs with M rows, replay them over the exactness rows
    (real, knife-edge, synthetic) and require the draft_tail BF16 logits to equal
    the graph-captured native logits bit for bit; the graph-captured native
    logits are also compared with eager ``F.linear`` (printed)."""
    from tensorrt_llm._torch._experimental.modeling_v2.models.glm_moe_dsa.shared.kernels.draft_tail.draft_tail import (  # noqa: E501
        DraftTailOptions,
    )

    variant = DraftTailOptions().variant
    rows = _exactness_rows(full_head, gamma)
    rows = rows[: rows.shape[0] // m * m]
    static_h = torch.zeros(m, oracle.HIDDEN, dtype=torch.bfloat16, device="cuda")
    nat = [None] * len(shards)
    w3 = [torch.empty(m, w.shape[0], dtype=torch.bfloat16, device="cuda") for w in shards]

    def step():
        for s, w in enumerate(shards):
            nat[s] = F.linear(static_h, w)
            runner.run_fused(static_h, w, s * w.shape[0], variant=variant, logits_out=w3[s])

    st = torch.cuda.Stream()
    st.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(st):
        step()
    torch.cuda.current_stream().wait_stream(st)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        step()
    draft_tail_vs_graph = graph_vs_eager = total = 0
    for i in range(0, rows.shape[0], m):
        static_h.copy_(rows[i : i + m])
        g.replay()
        torch.cuda.synchronize()
        for s, w in enumerate(shards):
            eager = F.linear(rows[i : i + m].contiguous(), w)
            draft_tail_vs_graph += int((w3[s].view(torch.int16) != nat[s].view(torch.int16)).sum())
            graph_vs_eager += int((nat[s].view(torch.int16) != eager.view(torch.int16)).sum())
            total += eager.numel()
    print(
        f"[draft_tail-exact-graph] M={m} variant={variant} rows={rows.shape[0]} "
        f"logits compared={total} draft_tail != graph native: {draft_tail_vs_graph}; "
        f"graph native != eager native: {graph_vs_eager}"
    )
    assert draft_tail_vs_graph == 0
