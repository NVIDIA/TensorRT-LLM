# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chunked prefill against whole-prompt prefill on Mistral-Large-4-1T-A52B-NVFP4's sm_103 tp4
target, at tp4 (four ranks) and at tp8 (eight ranks over two nodes of four).

At the checkpoint's own context the target serves prompts longer than one step: the engine
prefills them in chunks of up to max_num_tokens, and every chunk after a request's first is a
context request over a cached prefix -- its rope/append offsets past the prefix, its indexer
selecting over the cached and the new keys (the engine's chunk plan and its full-KV gather
mappings), sparse MLA attending over the pool rows the earlier chunks wrote, layer 35's latent
read by the NoPE layers 39..71 and every shared-indexer layer reusing the selection of the full
one before it. A chunk of 8192 or more context tokens is split over the TP ranks: each rank
selects for its share of the queries (2048 rows of a full 8192-token chunk at tp4, 1024 at tp8)
and every full indexer layer gathers every rank's share. A step may carry another request's
decode rows behind its context rows.

So the same prompts run through two engines of one world, one after the other, and greedy
decoding must give the same tokens:

* the delivered one, `test_modeling_v2_long_context.py`'s `ENGINE`: max_seq_len =
  max_num_tokens = 32768, chunked prefill off, so every prompt is prefilled in one step;
* the same engine with max_num_tokens 8192 and chunked prefill on: every full chunk is 8192
  context tokens, so the query split runs on each.

Both are at max_seq_len 32768, so the target tiles its indexer logits under the same budget in
both, 2**24 (`_prefill_logits_budget`: 512 rows at 32768 keys). It sizes its held context
workspace from each engine's quantities (`_ctx_workspace_rows`): 65536 rows, twice
max_num_tokens, for the unchunked engine; 2**19 under chunked prefill, where a step attends
cached prefixes. Every step of either fits it twice, so in both layer 35's rows are kept beside
the step's for the NoPE layers; an indexer chunk is scored as planned unless it packs past
32768 keys, when it is cut at its requests. What differs is the chunking above.

The prompts are passkey prompts (the long-context witness's) of 4096, 12288, 20480 and 32752
tokens, every one past `index_topk` (2048), each key in an earlier chunk than its question (but
the 4096-token prompt's, one chunk), so the answer is read across chunk boundaries. Each runs
alone, twice, on each engine; then the four run together as one batch, so that a chunk carries
two requests and later steps carry chunks behind decode rows. Each step's composition is read
back from the engine's iteration statistics.

The comparison is gated by the engine's own noise. The target's outputs are not bitwise
reproducible from run to run: the indexer's top-k breaks ties at its 2048th score arbitrarily, so
two runs of one engine on one input may attend over different keys of equal score, and greedy
tokens flip wherever the top two are close. So each prompt's tokens are compared over its leading
steps whose top-1 / top-2 log-probability margin is at least `NOISE` in both engines, and those
steps must hold the key. Every step's margins are printed, and where the tokens differ later, both
engines' top two at the first differing step in the prompt's alone runs, and whether either
engine's repeat differs from itself there.

Both engines run in this one process, the second once the first has shut down, and the second
sizes its KV pool from the memory the first left free. Under the launcher the ranks' processes
outlive each engine, so before the first engine boots and after each shuts down every rank
reports its device's free memory through the launcher's MPI session (the engines' own channel,
idle between them; `_rank_probe` runs there, imported from this module by its path): the first
engine's shutdown must give back what it took.

Runs with the ranks already in place, on four GB300s at tp4 or on eight over two nodes of four
at tp8: pytest under `trtllm-llmapi-launch`, as the whole-model accuracy class runs, with the
checkpoint under `LLM_MODELS_ROOT` as `Mistral-Large-4-1T-A52B-NVFP4` (or
`mistral-large-4-nvfp4`) in its Mistral-native format, which both engines load with
checkpoint_format "mistral_large_4". Skipped where the checkpoint is absent or off sm_103; each
world size is skipped below its own ranks (four GPUs at tp4, eight ranks at tp8) and inside a
launch of another world size; the release check is skipped outside the launcher.
"""

from __future__ import annotations

import gc
import random
import site
import socket
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_common.llm_data import llm_models_root

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch._experimental.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._utils import get_sm_version, global_mpi_rank
from tensorrt_llm.executor.utils import create_mpi_comm_session, get_spawn_proxy_process_env
from tensorrt_llm.inputs import TokensPrompt

from . import test_modeling_v2_long_context as long_context

#: The engines' world sizes, gated as the long-context witness gates its engine: tp4 on four
#: GPUs, tp8 on eight ranks over two nodes of four, each skipped inside a launch of the other.
WORLDS = long_context.WORLDS

pytestmark = [
    pytest.mark.skipif(
        get_sm_version() != 103,
        reason="the mistral_large4 modeling_v2 targets are certified on sm_103 only",
    ),
    pytest.mark.threadleak(enabled=False),
    # Two engine loads, one after the other, about 10 minutes each.
    pytest.mark.timeout(3600),
]

#: The delivered engine: whole prompts of up to 32768 tokens in one step. Its TP sizes are the
#: world's, set per engine.
UNCHUNKED = long_context.ENGINE
CHUNK = 8192
#: The same engine prefilling in steps of at most CHUNK tokens.
CHUNKED = dict(UNCHUNKED, max_num_tokens=CHUNK, enable_chunked_prefill=True)
#: The engine's default: a step of this many context rows or more splits its queries over the
#: TP ranks.
Q_SPLIT_THRESHOLD = 8192
INDEX_TOPK = long_context.INDEX_TOPK
#: How long the request queue holds an iteration for more requests while fewer than
#: max_batch_size are waiting, so the batch's prompts, submitted together as token ids, are
#: admitted together; every iteration pays it.
BATCH_WAIT_MS = 200

#: (tokens, depth of the key): the key in an earlier chunk of 8192 than the question, but for
#: the one-chunk prompt; the longest leaves room for the answer within max_seq_len 32768.
PROMPTS = ((4096, 0.5), (12288, 0.3), (20480, 0.6), (32752, 0.1))
#: Greedy, with the top two log-probabilities at each step.
ANSWER = SamplingParams(max_tokens=12, temperature=0.0, logprobs=2)
#: The top-1 / top-2 log-probability margin from which a step's greedy token is compared: below
#: it a token may flip between two runs of one engine on one input.
NOISE = 1.5
#: What the first engine may keep of a rank's device once it is shut down: its free memory must
#: come back to within this of what the rank had before that engine booted, or the second engine
#: would size its pool from a device the first still holds. The engine itself is far above it
#: (tens of GiB of weights a rank, and a KV pool of most of the rest); what a process kept of an
#: engine it had shut down was 1.82 to 2.56 GiB a rank on two GB300 nodes, measured by the tep8
#: target's chunk-equivalence witness on the same engine machinery.
RELEASE_SLACK = 4 << 30

HEAD = long_context.HEAD
FILLER = long_context.FILLER
QUESTION = long_context.QUESTION


def _checkpoint():
    path = long_context._checkpoint_dir()
    if path is None:
        pytest.skip(
            f"checkpoint {long_context.CHECKPOINTS} is not under LLM_MODELS_ROOT "
            f"({llm_models_root()})"
        )
    return path


def _engine(engine: dict, world: int) -> LLM:
    return LLM(
        str(_checkpoint()),
        checkpoint_format="mistral_large_4",
        modeling_v2="require",
        env_overrides={MODELING_V2_VALIDATE_ENV: "1"},
        enable_iter_perf_stats=True,
        batch_wait_timeout_ms=BATCH_WAIT_MS,
        tensor_parallel_size=world,
        moe_tensor_parallel_size=world,
        **engine,
    )


# ------------------------------------------------------------------- every rank, between engines


def _rank_probe() -> dict:
    """This rank's device free memory after a garbage collection and a cache release, and what
    of the device this process's torch allocator still holds. Runs on every rank of the
    launcher's MPI world through the engines' session while no engine is up (the session runs
    one task at a time)."""
    rank = global_mpi_rank()
    # The engine's device for this rank: its local rank among the node's devices.
    device = rank % torch.cuda.device_count()
    torch.cuda.set_device(device)
    gc.collect()
    torch.cuda.empty_cache()
    free, total = torch.cuda.mem_get_info(device)
    return dict(
        rank=rank,
        host=socket.gethostname(),
        device=device,
        free=free,
        total=total,
        allocated=torch.cuda.memory_allocated(device),
        reserved=torch.cuda.memory_reserved(device),
    )


def _reach_ranks(session) -> None:
    """Make this module importable on every rank: a task's function is unpickled by module path
    before the session hands the rank this process's `sys.path`, so first add the directory that
    holds this module's top-level package."""
    base = Path(__file__).resolve().parents[__name__.count(".")]
    session.submit_sync(site.addsitedir, str(base))


def _probe(session, world: int, label: str, baseline: dict[int, dict] | None) -> dict[int, dict]:
    """`_rank_probe` on every rank, by rank; printed beside `baseline`."""
    reports = {r["rank"]: r for r in session.submit_sync(_rank_probe)}
    assert sorted(reports) == list(range(world)), sorted(reports)
    for rank, report in sorted(reports.items()):
        delta = ""
        if baseline is not None:
            delta = f" ({report['free'] - baseline[rank]['free']:+d} B against the first probe)"
        print(
            f"CHUNKEQ probe {label} rank {rank} {report['host']}:{report['device']} free "
            f"{report['free'] / 2**30:.3f} of {report['total'] / 2**30:.3f} GiB{delta}, torch "
            f"allocated {report['allocated']} B reserved {report['reserved']} B",
            flush=True,
        )
    return reports


# ------------------------------------------------------------------------- the prompts and the runs


def _passkey_ids(tokenizer, tokens: int, depth: float, key: int) -> list[int]:
    """Exactly `tokens` ids: BOS, then the head, the filler with the needle at `depth` of it, and
    the question, each part tokenized on its own and the filler cut at the token that gives the
    length."""

    def encode(text: str) -> list[int]:
        ids = tokenizer.encode(text)
        return ids[1:] if ids and ids[0] == tokenizer.bos_token_id else ids

    head, question = encode(HEAD), encode(QUESTION)
    needle = encode(f"\nThe pass key is {key}. Remember it. {key} is the pass key.\n")
    filler = encode(FILLER * (tokens // 8))
    room = tokens - 1 - len(head) - len(needle) - len(question)
    assert 0 < room <= len(filler), (tokens, room, len(filler))
    before = int(room * depth)
    ids = [tokenizer.bos_token_id, *head, *filler[:before], *needle]
    ids += [*filler[: room - before], *question]
    assert len(ids) == tokens, (len(ids), tokens)
    return ids


def _requests(tokenizer) -> list[SimpleNamespace]:
    keys = random.Random(CHUNK).sample(range(10000, 100000), len(PROMPTS))
    requests = []
    for (tokens, depth), key in zip(PROMPTS, keys):
        ids = _passkey_ids(tokenizer, tokens, depth, key)
        assert INDEX_TOPK < len(ids), (tokens, len(ids))
        requests.append(SimpleNamespace(key=key, ids=ids, depth=depth))
    return requests


def _steps(llm: LLM) -> list[tuple[int, int, int]]:
    """(context requests, context tokens, generation requests) of each iteration since the last
    call that ran any context rows."""
    steps = []
    for record in llm.get_stats(timeout=2):
        ifb = record.get("inflightBatchingStats") or {}
        if ifb.get("numContextRequests", 0):
            steps.append(
                (ifb["numContextRequests"], ifb["numCtxTokens"], ifb.get("numGenRequests", 0))
            )
    return steps


def _generate(llm: LLM, requests: list) -> tuple[list[SimpleNamespace], list]:
    """`requests` through `llm` in one generate call: per request its token ids, its top two
    log-probabilities at each step and its text; and the context steps they ran in."""
    _steps(llm)
    outputs = llm.generate([TokensPrompt(prompt_token_ids=r.ids) for r in requests], ANSWER)
    steps = _steps(llm)
    results = []
    for out in outputs:
        o = out.outputs[0]
        top2 = [
            sorted(((lp.logprob, tok) for tok, lp in step.items()), reverse=True)[:2]
            for step in o.logprobs
        ]
        results.append(SimpleNamespace(ids=list(o.token_ids), top2=top2, text=o.text))
        print(f"CHUNKEQ run tokens={len(out.prompt_token_ids)}: {o.text!r} {list(o.token_ids)}")
    return results, steps


def _run(llm: LLM, requests: list) -> dict:
    """Each prompt alone, twice, then all of them as one batch."""
    alone = [[], []]
    alone_steps = []
    for r in requests:
        for repeat in range(2):
            (result,), steps = _generate(llm, [r])
            alone[repeat].append(result)
            if repeat == 0:
                alone_steps.append(steps)
    batch, batch_steps = _generate(llm, requests)
    return dict(alone=alone[0], again=alone[1], alone_steps=alone_steps, batch=batch,
                batch_steps=batch_steps)  # fmt: skip


def _margins(top2: list) -> list[float]:
    """Each step's top-1 / top-2 log-probability gap."""
    return [s[0][0] - s[1][0] if len(s) > 1 else float("inf") for s in top2]


def _first_difference(a: list[int], b: list[int]) -> int | None:
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)


@pytest.fixture(scope="module", params=WORLDS)
def ran(request):
    """Both engines at the world, one after the other, each over the same prompts; under the
    launcher, every rank's device memory before the first boots and after each shuts down."""
    world = request.param
    _checkpoint()
    session, probes = None, {}
    if get_spawn_proxy_process_env():
        # The engines' own session: a singleton every LLM in this process reuses.
        session = create_mpi_comm_session(world)
        _reach_ranks(session)
        probes["before"] = _probe(session, world, f"tp{world} before the whole engine", None)
    with _engine(UNCHUNKED, world) as llm:
        tokenizer = llm.tokenizer
        requests = _requests(tokenizer)
        runs = {"whole": _run(llm, requests)}
    if session is not None:
        label = f"tp{world} after the whole engine"
        probes["after_whole"] = _probe(session, world, label, probes["before"])
    with _engine(CHUNKED, world) as llm:
        runs["chunked"] = _run(llm, requests)
    if session is not None:
        label = f"tp{world} after the chunked engine"
        probes["after_chunked"] = _probe(session, world, label, probes["before"])
    return SimpleNamespace(tokenizer=tokenizer, requests=requests, runs=runs, probes=probes)


def test_chunked_prefill_gives_the_whole_prompt_tokens(ran):
    tokenizer, requests, runs = ran.tokenizer, ran.requests, ran.runs

    problems = []
    lengths = [len(r.ids) for r in requests]
    for engine, run in runs.items():
        for n, steps in zip(lengths, run["alone_steps"]):
            print(f"CHUNKEQ {engine} alone tokens={n}: (ctx requests, ctx tokens, gen requests) "
                  f"{steps}")  # fmt: skip
        print(f"CHUNKEQ {engine} batch tokens={lengths}: {run['batch_steps']}")
        for n, steps in zip(lengths, run["alone_steps"]):
            if sum(t for _, t, _ in steps) != n or any(c != 1 for c, _, _ in steps):
                problems.append(f"{engine} alone tokens={n}: context steps {steps}")
        if sum(t for _, t, _ in run["batch_steps"]) != sum(lengths):
            problems.append(f"{engine} batch: context steps {run['batch_steps']}")

    # The whole-prompt engine prefills each prompt in one step; the chunked one in full chunks of
    # CHUNK context rows (the query split) and a last one of the rest.
    for n, steps in zip(lengths, runs["whole"]["alone_steps"]):
        if [t for _, t, _ in steps] != [n]:
            problems.append(f"whole alone tokens={n}: not one step: {steps}")
    for n, steps in zip(lengths, runs["chunked"]["alone_steps"]):
        want = [CHUNK] * (n // CHUNK) + ([n % CHUNK] if n % CHUNK else [])
        if [t for _, t, _ in steps] != want:
            problems.append(f"chunked alone tokens={n}: chunks {steps}, not {want}")
    split = sum(
        1 for s in runs["chunked"]["alone_steps"] for _, t, _ in s if t >= Q_SPLIT_THRESHOLD
    )
    print(f"CHUNKEQ chunked alone: {split} context steps of >= {Q_SPLIT_THRESHOLD} rows (split)")
    batch_steps = runs["chunked"]["batch_steps"]
    if not any(c > 1 for c, _, _ in batch_steps):
        problems.append(f"chunked batch: no step carries two context requests: {batch_steps}")
    if not any(g > 0 for _, _, g in batch_steps):
        problems.append(f"chunked batch: no step mixes context and decode rows: {batch_steps}")

    whole, chunked = runs["whole"], runs["chunked"]
    for kind in ("alone", "batch"):
        for i, r in enumerate(requests):
            a, b = whole[kind][i], chunked[kind][i]
            ma, mb = _margins(a.top2), _margins(b.top2)
            n = min(len(a.ids), len(b.ids))
            gated = next((s for s in range(n) if min(ma[s], mb[s]) < NOISE), n)
            shown = tokenizer.decode(a.ids[:gated])
            print(
                f"CHUNKEQ {kind} tokens={len(r.ids)} key={r.key}: whole {a.text!r}, chunked "
                f"{b.text!r}; {gated} steps at margin >= {NOISE} in both ({shown!r}); margins "
                f"whole {[round(m, 2) for m in ma]} chunked {[round(m, 2) for m in mb]}"
            )
            for label, out in (("whole", a), ("chunked", b)):
                if str(r.key) not in out.text:
                    problems.append(
                        f"{kind} tokens={len(r.ids)}: {label} lost the key: {out.text!r}"
                    )
            if a.ids[:gated] != b.ids[:gated]:
                problems.append(
                    f"{kind} tokens={len(r.ids)}: differ within the {gated} steps above the "
                    f"noise: {a.ids[:gated]} vs {b.ids[:gated]}"
                )
            elif str(r.key) not in shown:
                problems.append(
                    f"{kind} tokens={len(r.ids)}: the key {r.key} is not within the {gated} steps "
                    f"above the noise ({shown!r})"
                )
            k = _first_difference(a.ids, b.ids)
            if k is None:
                continue
            # The first differing step measured on each engine alone, and each engine's repeat.
            for label, run in (("whole", whole), ("chunked", chunked)):
                alone, again = run["alone"][i], run["again"][i]
                self_k = _first_difference(alone.ids, again.ids)
                top = alone.top2[k] if k < len(alone.top2) else None
                print(
                    f"CHUNKEQ {kind} tokens={len(r.ids)}: first differing step {k}; {label} alone "
                    f"top-2 {top}, its repeat first differs from it at step {self_k}"
                )
    assert not problems, problems


def test_the_first_engine_gives_its_devices_back(ran):
    """After the whole-prompt engine shuts down, every rank's device is back within RELEASE_SLACK
    of its free memory before that engine booted, so the chunked engine sizes its pool from the
    same devices."""
    if not ran.probes:
        pytest.skip("the per-rank probe runs through trtllm-llmapi-launch's MPI session")
    for rank, before in ran.probes["before"].items():
        after = ran.probes["after_whole"][rank]
        assert (after["host"], after["device"]) == (before["host"], before["device"])
        assert before["free"] - after["free"] <= RELEASE_SLACK, (
            rank,
            before["free"],
            after["free"],
        )
