# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chunked prefill against whole-prompt prefill on Mistral-Large-4-1T-A52B-NVFP4's sm_103 dep4 target.

At the checkpoint's own context the target serves prompts longer than one step: the engine
prefills them in chunks of up to max_num_tokens, and every chunk after a request's first is a
context request over a cached prefix -- its rope/append offsets past the prefix, its indexer
selecting over the cached and the new keys (the engine's chunk plan and its full-KV gather
mappings), sparse MLA attending over the pool rows the earlier chunks wrote, layer 35's latent
read by the NoPE layers 39..71 and every shared-indexer layer reusing the selection of the full
one before it. A step may carry another request's decode rows behind its context rows.

So the same prompts run through two engines, one after the other, and greedy decoding must give
the same tokens:

* the delivered one, `test_modeling_v2_long_context.py`'s four-rank engine: per rank max_seq_len =
  max_num_tokens = 32768, chunked prefill off, so every prompt is prefilled in one step;
* the same engine with max_num_tokens 8192 and chunked prefill on.

Both hold max_seq_len 32768, so the target runs the same paths in both: every step's context
KV fits its held workspace twice (65536 rows unchunked, twice max_num_tokens; 2**19 chunked,
where a step may attend its requests' cached prefixes), so layer 35's gather is kept for its
sharers; every indexer chunk holds at most 32768 keys and is scored whole, in tiles under the
logits budget of 2**24 both engines' max_seq_len gives; and every MoE layer makes one expert
call (`_EXPERT_CALL_TOKENS`, 131072 rows unchunked; `_EXPERT_CALL_TOKENS_CHUNKED`, 32768 = 4 x
8192 chunked). The paths past these (the decode path past the held workspace, packed indexer
chunks cut at their requests, the larger logits budget and the cut expert calls) are
test_modeling_v2_retrieval_1m.py's on the 1M engine, and test_modeling_v2_ctx_bound.py's,
test_modeling_v2_indexer_packed_chunks.py's and test_modeling_v2_routed_experts.py's on their
pieces.

Each rank carries its own requests, pinned by `SchedulingParams.attention_dp_rank` and read back
from that rank's own iteration statistics, in two batches:

* "mixed": rank 0 a ~32000-token prompt (four chunks, two of them 8192 tokens over a cached
  prefix of 8192 and 16384), rank 1 ~20000, rank 2 ~9000 (a short chunk over an 8192-token
  prefix), rank 3 ~3000 and ~12000 together (the second's last chunk shares its step with the
  first's decode row);
* "far": one 26000- to 30000-token prompt per rank, so every rank prefills three full chunks
  and its last over a prefix of 24576.

Every prompt is over `index_topk` (2048), so the indexer selects, and each is a passkey prompt
(the long-context witness's) whose key lies in an earlier chunk than its question (all but rank
3's two in "mixed"), so the answer is read across chunk boundaries.

The comparison is gated by the engine's own noise. The target's outputs are not bitwise
reproducible from run to run: the same batch run twice in the same whole-prompt engine gives
different log-probabilities at the first generated token and different greedy tokens after the
key, wherever the top two are close. So each prompt's tokens are compared over its leading steps
whose top-1 / top-2 log-probability margin is at least `NOISE` in both engines, and those steps
must hold the key: the key's own tokens carry margins of 3.4 to 12.9 in every run measured, the
token after it 0.0 to 1.1. Every step's margins in both engines are printed, and for a prompt
whose tokens differ later, both engines' top two at the first differing step.

Runs on four GB300s with the ranks already in place: pytest under `trtllm-llmapi-launch`, with
the checkpoint under `LLM_MODELS_ROOT` as `Mistral-Large-4-1T-A52B-NVFP4` (or
`mistral-large-4-nvfp4`) in its Mistral-native format, which both engines load with
checkpoint_format "mistral_large_4". Skipped where the checkpoint is absent, below four GPUs, or
off sm_103.
"""

from __future__ import annotations

import collections
from types import SimpleNamespace

import pytest
from test_common.llm_data import llm_models_root
from utils.util import skip_num_gpus_less_than

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch._experimental.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.inputs import TokensPrompt
from tensorrt_llm.scheduling_params import SchedulingParams

from . import test_modeling_v2_long_context as long_context

pytestmark = [
    skip_num_gpus_less_than(4),
    long_context._launched_as(4),
    pytest.mark.skipif(
        get_sm_version() != 103,
        reason="the mistral_large4 modeling_v2 targets are certified on sm_103 only",
    ),
    pytest.mark.threadleak(enabled=False),
]

#: dep4's four attention-DP ranks, each pinned its own prompts below.
RANKS = 4
#: The delivered engine: whole prompts of up to 32768 tokens in one step.
UNCHUNKED = long_context._engine(RANKS)
CHUNK = 8192
#: The same engine prefilling in steps of at most CHUNK tokens per rank.
CHUNKED = dict(UNCHUNKED, max_num_tokens=CHUNK, enable_chunked_prefill=True)
INDEX_TOPK = long_context.INDEX_TOPK

#: Per batch, per rank, its prompts as (tokens, depth of the key).
BATCHES = {
    "mixed": {
        0: ((32000, 0.05),),
        1: ((20000, 0.5),),
        2: ((9000, 0.3),),
        3: ((3000, 0.5), (12000, 0.9)),
    },
    "far": {0: ((26000, 0.3),), 1: ((30000, 0.6),), 2: ((28000, 0.8),), 3: ((26500, 0.45),)},
}
#: Greedy, with the top two log-probabilities at each step.
ANSWER = SamplingParams(max_tokens=12, temperature=0.0, logprobs=2)
#: The top-1 / top-2 log-probability margin from which a step's greedy token is compared: reruns
#: of one whole-prompt engine on the same batch flip tokens at steps whose margin is below it.
NOISE = 1.5


def _context_iterations(llm: LLM) -> dict[int, list[tuple[int, int, int, int]]]:
    """Since the last call, per attention-DP rank, every iteration that ran context rows there:
    `(iteration, context requests, context tokens, generation requests)`, in order."""
    by_rank: dict[int, list[tuple[int, int, int, int]]] = collections.defaultdict(list)
    for record in llm.get_stats(timeout=2):
        ifb = record.get("inflightBatchingStats") or {}
        if ifb.get("numContextRequests", 0):
            rank = int(record.get("attentionDpRank", 0))
            by_rank[rank].append(
                (
                    int(record["iter"]),
                    ifb["numContextRequests"],
                    ifb["numCtxTokens"],
                    ifb.get("numGenRequests", 0),
                )
            )
    return {rank: sorted(steps) for rank, steps in by_rank.items()}


def _engine(engine: dict) -> LLM:
    return LLM(
        str(_checkpoint()),
        checkpoint_format="mistral_large_4",
        modeling_v2="require",
        env_overrides={MODELING_V2_VALIDATE_ENV: "1"},
        enable_iter_perf_stats=True,
        **engine,
    )


def _requests(tokenizer, prompts: dict, seed: int) -> list[SimpleNamespace]:
    """The prompts as token ids, BOS included, each pinned to its rank, each key distinct."""
    specs = [
        (rank, tokens, depth) for rank, prompts in prompts.items() for tokens, depth in prompts
    ]
    keys = long_context._keys(len(specs), seed=seed)
    requests = []
    for (rank, tokens, depth), key in zip(specs, keys):
        ids = long_context._passkey_ids(tokenizer, tokens, depth, key)
        assert INDEX_TOPK < len(ids) <= tokens, (tokens, len(ids))
        requests.append(SimpleNamespace(rank=rank, key=key, ids=ids))
    return requests


def _generate(llm: LLM, requests: list) -> tuple[list, dict]:
    """Every request through `llm` at once: per request (token ids, top-2 log-probs per step,
    text), and each rank's context iterations."""
    _context_iterations(llm)
    outputs = llm.generate(
        [TokensPrompt(prompt_token_ids=r.ids) for r in requests],
        ANSWER,
        scheduling_params=[
            SchedulingParams(attention_dp_rank=r.rank, attention_dp_relax=False) for r in requests
        ],
    )
    steps = _context_iterations(llm)
    results = []
    for out in outputs:
        o = out.outputs[0]
        top2 = [
            sorted(((lp.logprob, tok) for tok, lp in step.items()), reverse=True)[:2]
            for step in o.logprobs
        ]
        results.append(SimpleNamespace(ids=list(o.token_ids), top2=top2, text=o.text))
    return results, steps


def _checkpoint():
    path = long_context._checkpoint_dir()
    if path is None:
        pytest.skip(
            f"checkpoint {long_context.CHECKPOINTS} is not under LLM_MODELS_ROOT "
            f"({llm_models_root()})"
        )
    return path


def _margins(top2: list) -> list[float]:
    """Each step's top-1 / top-2 log-probability gap."""
    return [s[0][0] - s[1][0] if len(s) > 1 else float("inf") for s in top2]


def _check_steps(name: str, steps: dict, per_rank: dict, chunked: bool) -> None:
    """Each rank prefilled exactly its own prompts' tokens; whole prompts in one step each, or
    chunks of at most CHUNK with a full one over a cached prefix."""
    assert sorted(steps) == list(range(RANKS)), (name, steps)
    for rank in range(RANKS):
        assert sum(t for _, _, t, _ in steps[rank]) == sum(per_rank[rank]), (
            name,
            rank,
            steps[rank],
            per_rank[rank],
        )
    if not chunked:
        for rank, lengths in per_rank.items():
            if len(lengths) == 1:
                assert [t for _, _, t, _ in steps[rank]] == lengths, (name, steps[rank])
        return
    assert all(t <= CHUNK for s in steps.values() for _, _, t, _ in s), (name, steps)
    # A rank holding one request: any of its context steps but the first is over a prefix.
    over_prefix = [
        (rank, it)
        for rank, lengths in per_rank.items()
        if len(lengths) == 1
        for it, n, t, _ in steps[rank][1:]
        if n == 1 and t == CHUNK
    ]
    assert over_prefix, (name, steps)


def test_chunked_prefill_gives_the_whole_prompt_tokens():
    _checkpoint()
    with _engine(UNCHUNKED) as llm:
        tokenizer = llm.tokenizer
        batches = {
            name: _requests(tokenizer, prompts, seed=CHUNK + i)
            for i, (name, prompts) in enumerate(BATCHES.items())
        }
        whole = {name: _generate(llm, reqs) for name, reqs in batches.items()}
    with _engine(CHUNKED) as llm:
        chunked = {name: _generate(llm, reqs) for name, reqs in batches.items()}

    problems = []
    for name, requests in batches.items():
        (w_out, w_steps), (c_out, c_steps) = whole[name], chunked[name]
        per_rank = collections.defaultdict(list)
        for r in requests:
            per_rank[r.rank].append(len(r.ids))
        for label, steps in (("whole", w_steps), ("chunked", c_steps)):
            for rank in range(RANKS):
                print(f"CHUNKEQ {name} {label} rank {rank} prompts {per_rank[rank]}: "
                      f"(iteration, ctx requests, ctx tokens, gen requests) {steps.get(rank)}")  # fmt: skip
        _check_steps(f"{name} whole", w_steps, per_rank, chunked=False)
        _check_steps(f"{name} chunked", c_steps, per_rank, chunked=True)

        for r, a, b in zip(requests, w_out, c_out):
            ma, mb = _margins(a.top2), _margins(b.top2)
            n = min(len(a.ids), len(b.ids))
            gated = next((i for i in range(n) if min(ma[i], mb[i]) < NOISE), n)
            shown = tokenizer.decode(a.ids[:gated])
            print(
                f"CHUNKEQ {name} rank {r.rank} tokens={len(r.ids)} key={r.key}: whole {a.text!r}, "
                f"chunked {b.text!r}; {gated} steps at margin >= {NOISE} in both ({shown!r}); "
                f"margins whole {[round(m, 2) for m in ma]} chunked {[round(m, 2) for m in mb]}"
            )
            for label, out in (("whole", a), ("chunked", b)):
                if str(r.key) not in out.text:
                    problems.append(f"{name} rank {r.rank}: {label} lost the key: {out.text!r}")
            if a.ids[:gated] != b.ids[:gated]:
                problems.append(
                    f"{name} rank {r.rank} tokens={len(r.ids)}: differ within the {gated} steps "
                    f"above the noise: {a.ids[:gated]} vs {b.ids[:gated]}"
                )
            elif str(r.key) not in shown:
                problems.append(
                    f"{name} rank {r.rank}: the key {r.key} is not within the {gated} steps "
                    f"above the noise ({shown!r})"
                )
            k = next((i for i in range(n) if a.ids[i] != b.ids[i]), None)
            if k is not None:
                print(
                    f"CHUNKEQ {name} rank {r.rank}: first differing step {k}, whole top-2 "
                    f"{a.top2[k]}, chunked top-2 {b.top2[k]}"
                )
    assert not problems, problems
