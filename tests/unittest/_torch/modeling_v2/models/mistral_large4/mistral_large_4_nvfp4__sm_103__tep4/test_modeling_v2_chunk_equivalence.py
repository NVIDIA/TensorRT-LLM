# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Chunk equivalence of Mistral-Large-4-1T-A52B-NVFP4 on its sm_103 tep4 target, at tep4 (four
ranks) and at tep8 (eight ranks over two nodes of four).

Under chunked prefill the engine prefills a prompt in steps of up to `max_num_tokens`.
Every chunk after a request's first is a context request over a cached prefix -- the rope
append at the prefix's positions, the indexer's chunk plan and key gather over the whole
prefix, sparse MLA over selections that reach into it -- and from `q_split_threshold` (8192)
context rows on, the engine splits a step's queries over every TP rank (the four, or the eight
of both nodes) and the target all-gathers their selections. The same text prompts of up to
32768 tokens go through two engines, one at a time, and greedy decoding must give the same
tokens:

* chunked: the 1M config (max_seq_len 1048576, chunked prefill, sixteen sequences, the
  fp8_ds_mla latent KV cache at its default dtype, CUDA graphs over batches 1, 2, 4, 8 and 16,
  padded) at max_num_tokens 8192, so each prompt is prefilled in steps of 8192 tokens and a
  last shorter one, and the engine splits each full step 2048 rows per rank at tep4, 1024 at
  tep8;
* delivered: the engine the target was delivered at (max_seq_len = max_num_tokens = 32768,
  chunked prefill off), where each prompt is one context step with no cached prefix and no
  split.

The target runs the same code in both: each step picks its paths from its own sizes, and only
what the engines' quantities size differs -- the indexer's logits tiles (`_prefill_logits_budget`:
2**28 logits in the chunked engine, 2**24 delivered) and the held context workspace's rows
(`_ctx_workspace_rows`: 2**19 chunked, 65536 delivered). Every step of either engine attends its
context KV, at most 32768 rows, through the held workspace with layer 35's gather kept for the
NoPE sharers.

Every prompt passes `index_topk` (2048), so the indexer scores and selects in both engines and
every layer runs, layer 35's NoPE sharers and the shared-indexer layers included. The step
composition is read back from the engines' iteration statistics, and what each rank's context
steps received from an observe-only record on every rank: the query split, the indexer chunk
plan, the cached prefixes and how many selection gathers the step made.

Same tokens, not the same bits: the row-bound GEMMs run at another M (8192 against up to 32768
rows), cuBLAS may pick another algorithm for them, and row-count rounding through the FP8
latent moves top-1/top-2 margins. Neither engine repeats its own near ties either: what the
model writes after a passkey changes between two generations of one prompt in one engine. So
the compared continuations are ones that win by a wide margin -- a passkey's digits, a counting
sequence, a cycle of month names -- `max_tokens` stays at 32 or fewer, and a mismatch reports
the first divergence and every step's top-1/top-2 logprob margin from both engines.

Text only: no image, no chat template. Runs with the ranks already in place, on four GB300s at
tep4 or on eight over two nodes of four at tep8: pytest under `trtllm-llmapi-launch`, as the
whole-model accuracy class runs, with the checkpoint under `LLM_MODELS_ROOT` as
`Mistral-Large-4-1T-A52B-NVFP4` (or `mistral-large-4-nvfp4`) in its Mistral-native format,
which both engines load with checkpoint_format "mistral_large_4". The ranks' devices are not
there for two engines at once, so one process holds them in turn. The second engine sizes its
KV pool from the memory the first left free, so before the first boots and after each shuts
down every rank reports its device's free memory, through the launcher's MPI session (the
engines' own channel, idle between them): the first engine's shutdown must give back what it
took. The same probe installs and collects the per-step record (`_chunk_equivalence_ranks.py`);
outside the launcher there is no such session, and the checks on what the probe reports are
skipped. Skipped where the checkpoint is absent or off sm_103; each world size is skipped below
its own ranks (four GPUs at tep4, eight ranks at tep8) and inside a launch of another world size.
"""

from __future__ import annotations

import random
import site
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_common.llm_data import llm_models_root
from utils.util import skip_num_gpus_less_than

from tensorrt_llm import LLM, SamplingParams
from tensorrt_llm._torch._experimental.modeling_v2 import MODELING_V2_VALIDATE_ENV
from tensorrt_llm._torch.models.checkpoints.mistral.config_loader import MistralConfigLoader
from tensorrt_llm._utils import get_sm_version
from tensorrt_llm.executor.utils import create_mpi_comm_session, get_spawn_proxy_process_env
from tensorrt_llm.inputs import TokensPrompt
from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig
from tensorrt_llm.llmapi.mpi_session import get_mpi_world_size

from . import _chunk_equivalence_ranks as rank_side


def _ranks() -> int:
    """The ranks this process can serve: the MPI world a launcher started (eight over two nodes
    of four, where each node sees four devices), or, unlaunched, the devices the LLM API spawns
    its workers on -- as the integration conftest's `skip_less_mpi_world_size` counts."""
    world = get_mpi_world_size()
    return torch.cuda.device_count() if world == 1 else world


def _launched_elsewhere(world: int) -> bool:
    """A launcher started an MPI world of another size: an engine of `world` ranks does not
    start inside it."""
    return get_mpi_world_size() not in (1, world)


#: The engines' world sizes, each gated on its own ranks: tep4 on four GPUs, tep8 on eight
#: ranks over two nodes of four.
WORLDS = [
    pytest.param(
        4,
        marks=[
            skip_num_gpus_less_than(4),
            pytest.mark.skipif(
                _launched_elsewhere(4), reason="tep4 needs a launch of four ranks or none"
            ),
        ],
        id="tep4",
    ),
    pytest.param(
        8,
        marks=[
            pytest.mark.skipif(_ranks() < 8, reason="tep8 needs eight ranks"),
            pytest.mark.skipif(
                _launched_elsewhere(8), reason="tep8 needs a launch of eight ranks or none"
            ),
        ],
        id="tep8",
    ),
]

pytestmark = [
    pytest.mark.skipif(
        get_sm_version() != 103,
        reason="the mistral_large4 modeling_v2 targets are certified on sm_103 only",
    ),
    # The module's engines start the LLM API's result-dispatch thread on their first request
    # and stop it when the fixture shuts them down.
    pytest.mark.threadleak(enabled=False),
]

CHECKPOINTS = ("Mistral-Large-4-1T-A52B-NVFP4", "mistral-large-4-nvfp4")

INDEX_TOPK = 2048
# sparse_attention_config.q_split_threshold's default (llmapi/llm_args.py): a step of at least
# this many context rows is split over the TP ranks.
Q_SPLIT_THRESHOLD = 8192
CHUNK = 8192
DELIVERED_LEN = 32768
# What one engine may keep of a rank's device once it is shut down: its free memory must come
# back to within this of what the rank had before the first engine booted, or the second
# engine would size its pool from a device the first still holds. The engine itself is far
# above it (about 75 GiB of weights and a KV pool of about 150 GiB per rank at tep8, twice the
# weights at tep4); what a process keeps of an engine it has shut down was 1.82 to 2.56 GiB per
# rank on two GB300 nodes at tep8, a few hundred MiB of it still allocated in torch's caching
# allocator.
RELEASE_SLACK = 4 << 30

COMMON = dict(
    enable_attention_dp=False,
    max_batch_size=16,
    cuda_graph_config=CudaGraphConfig(max_batch_size=16, enable_padding=True),
    kv_cache_config=KvCacheConfig(enable_block_reuse=False, free_gpu_memory_fraction=0.85),
)
ENGINES = {
    "chunked": dict(max_seq_len=1048576, max_num_tokens=CHUNK, enable_chunked_prefill=True),
    "delivered": dict(
        max_seq_len=DELIVERED_LEN, max_num_tokens=DELIVERED_LEN, enable_chunked_prefill=False
    ),
}

HEAD = (
    "There is an important piece of information hidden inside a lot of irrelevant text. "
    "Find it and memorize it. I will quiz you about it.\n\n"
)
FILLER = (
    "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
)
QUESTION = "\nWhat is the pass key? The pass key is"
MONTHS = (
    "January, February, March, April, May, June, July, August, September, October, "
    "November, December, "
)

# (name, kind, prompt tokens at most, passkey depth): one prompt within one chunk, the others
# two to four chunks with the passkey in their first chunk or a middle one, so the question in
# the last chunk reads it through the cached prefix.
PROMPTS = (
    ("passkey-6k-mid", "passkey", 6000, 0.5),
    ("passkey-12k-early", "passkey", 12000, 0.25),
    ("passkey-20k-late", "passkey", 20000, 0.75),
    ("passkey-31k-early", "passkey", 31000, 0.1),
    ("counting-16k", "counting", 16000, None),
    ("months-28k", "months", 28000, None),
)
# A passkey's answer is the key's three tokens (the space, then its five digits in two
# pieces). What the model writes after the key is a near tie (top-1/top-2 margins well under a
# nat) that differs between two boots of one and the same engine, so it is not generated. A
# sequence: its next 32 tokens.
MAX_TOKENS = {"passkey": 3, "counting": 32, "months": 32}


# ------------------------------------------------------------------- every rank, between engines


def _reach_ranks(session) -> None:
    """Make the rank side importable on every rank: a task's function is unpickled by module
    path before the session hands the rank this process's `sys.path`, so first add the
    directory that holds the rank side's top-level package."""
    base = Path(rank_side.__file__).resolve().parents[rank_side.__name__.count(".")]
    session.submit_sync(site.addsitedir, str(base))


def _probe(session, action: str, world: int) -> dict[int, dict] | None:
    """`rank_probe(action)` on every rank, by rank; None without the launcher's session."""
    if session is None:
        return None
    reports = {r["rank"]: r for r in session.submit_sync(rank_side.rank_probe, action)}
    assert sorted(reports) == list(range(world)), sorted(reports)
    return reports


# ------------------------------------------------------------------------------- the prompts


def _checkpoint_dir() -> Path | None:
    """The checkpoint's directory, found by its Mistral-native config, params.json."""
    root = llm_models_root()
    if root is None:
        return None
    for name in CHECKPOINTS:
        if (Path(root) / name / "params.json").is_file():
            return Path(root) / name
    return None


def _fit(tokenizer, text_of, budget: int) -> tuple[list[int], int]:
    """The longest `text_of(n)` whose ids (BOS included, as the LLM API's own tokenization adds
    it) fit in `budget`, by bisection over n."""
    lo, hi = 0, 1
    while len(tokenizer.encode(text_of(hi))) <= budget:
        lo, hi = hi, hi * 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if len(tokenizer.encode(text_of(mid))) <= budget:
            lo = mid
        else:
            hi = mid
    return tokenizer.encode(text_of(lo)), lo


def _build(tokenizer, name: str, kind: str, budget: int, depth: float | None) -> SimpleNamespace:
    if kind == "passkey":
        key = random.Random(budget).randrange(10000, 100000)

        def text_of(reps: int) -> str:
            before = round(reps * depth)
            needle = f"\nThe pass key is {key}. Remember it. {key} is the pass key.\n"
            return HEAD + FILLER * before + needle + FILLER * (reps - before) + QUESTION

        ids, _ = _fit(tokenizer, text_of, budget)
        expect = str(key)
    elif kind == "counting":

        def text_of(n: int) -> str:
            return "Count up by one:\n" + "".join(f"{i}, " for i in range(1, n + 1))

        _, n = _fit(tokenizer, text_of, budget)
        expect = f"{n + 1}, {n + 2}"
        # The prompt ends on the comma: the continuation is " n+1, n+2, ...".
        ids = tokenizer.encode(text_of(n).rstrip())
    else:

        def text_of(n: int) -> str:
            return "The months, over and over:\n" + MONTHS * n

        _, n = _fit(tokenizer, text_of, budget)
        expect = "January, February"
        ids = tokenizer.encode(text_of(n).rstrip())
    # Every key count stays within the delivered 32768, the widths the prefill rows' catalog
    # cells hold.
    assert INDEX_TOPK < len(ids) <= min(budget, DELIVERED_LEN), (name, len(ids))
    return SimpleNamespace(
        name=name,
        kind=kind,
        ids=ids,
        expect=expect,
        params=SamplingParams(max_tokens=MAX_TOKENS[kind], temperature=0.0, logprobs=2),
    )


def _context_steps(llm: LLM) -> list[tuple[int, int]]:
    """(context requests, context tokens) of each iteration since the last call that ran any
    context rows."""
    steps = []
    for record in llm.get_stats(timeout=2):
        ifb = record.get("inflightBatchingStats") or {}
        if ifb.get("numContextRequests", 0):
            steps.append((ifb["numContextRequests"], ifb["numCtxTokens"]))
    return steps


def _margins(output) -> list[tuple[int, float, int | None, float]]:
    """Each generated step's (top-1 token, its logprob, top-2 token, its logprob)."""
    rows = []
    for step in output.logprobs:
        ranked = [(t, lp.logprob) for t, lp in sorted(step.items(), key=lambda kv: -kv[1].logprob)]
        ranked.append((None, float("-inf")))
        (t1, l1), (t2, l2) = ranked[:2]
        rows.append((t1, l1, t2, l2))
    return rows


def _run(path: Path, engine: str, prompts, world: int) -> tuple[list, dict]:
    """Every prompt through one engine of `world` ranks, one at a time. Without `prompts`, they
    are built with this engine's tokenizer (the checkpoint's tekken.json), so both engines read
    the same ids."""
    results = {}
    with LLM(
        str(path),
        checkpoint_format="mistral_large_4",
        modeling_v2="require",
        env_overrides={MODELING_V2_VALIDATE_ENV: "1"},
        enable_iter_perf_stats=True,
        tensor_parallel_size=world,
        moe_expert_parallel_size=world,
        **COMMON,
        **ENGINES[engine],
    ) as llm:
        print(f"CHUNKEQ {engine} kv_cache_capacity {llm.get_kv_cache_capacity()}", flush=True)
        if prompts is None:
            prompts = [_build(llm.tokenizer, *spec) for spec in PROMPTS]
        _context_steps(llm)
        for p in prompts:
            out = llm.generate(TokensPrompt(prompt_token_ids=p.ids), p.params).outputs[0]
            steps = _context_steps(llm)
            margins = _margins(out)
            results[p.name] = SimpleNamespace(
                tokens=list(out.token_ids), text=out.text, steps=steps, margins=margins
            )
            print(
                f"CHUNKEQ {engine} {p.name} tokens={len(p.ids)} context_steps={steps} "
                f"text={out.text!r} ids={list(out.token_ids)} "
                f"margins={[round(m[1] - m[3], 3) for m in margins]}",
                flush=True,
            )
    return prompts, results


def _print_probe(
    label: str, reports: dict[int, dict] | None, baseline: dict[int, dict] | None
) -> None:
    if reports is None:
        return
    for rank, report in sorted(reports.items()):
        delta = ""
        if baseline is not None:
            delta = f" vs baseline {report['free'] - baseline[rank]['free']:+d} B"
        print(
            f"CHUNKEQ probe {label} rank {rank} {report['host']}:{report['device']} free "
            f"{report['free'] / 2**30:.3f} of {report['total'] / 2**30:.3f} GiB{delta}, torch "
            f"allocated {report['allocated']} B reserved {report['reserved']} B, "
            f"{len(report['steps'])} context steps",
            flush=True,
        )
    for step in reports[0]["steps"]:
        print(f"CHUNKEQ step {label} {step}", flush=True)


@pytest.fixture(scope="module", params=WORLDS)
def runs(request):
    world = request.param
    path = _checkpoint_dir()
    if path is None:
        pytest.skip(f"checkpoint {CHECKPOINTS} is not under LLM_MODELS_ROOT ({llm_models_root()})")
    # The checkpoint's config as the engines read it from params.json.
    config = MistralConfigLoader().load_pretrained_config(str(path))
    # The engines' own session under trtllm-llmapi-launch: a singleton every LLM in this
    # process reuses. Unlaunched, each engine spawns its own workers and no probe runs.
    session = None
    if get_spawn_proxy_process_env():
        session = create_mpi_comm_session(world)
        _reach_ranks(session)
    baseline = _probe(session, "install", world)
    _print_probe("before-chunked", baseline, None)
    prompts, chunked = _run(path, "chunked", None, world)
    after_chunked = _probe(session, "collect", world)
    _print_probe("after-chunked", after_chunked, baseline)
    _, delivered = _run(path, "delivered", prompts, world)
    after_delivered = _probe(session, "remove", world)
    _print_probe("after-delivered", after_delivered, baseline)
    return SimpleNamespace(
        world=world,
        prompts={p.name: p for p in prompts},
        chunked=chunked,
        delivered=delivered,
        selecting_layers=sum(role == "full" for role in config.index_sharing),
        baseline=baseline,
        after_chunked=after_chunked,
        after_delivered=after_delivered,
    )


# ------------------------------------------------------------------------------------ tests


def _chunks(n: int) -> list[tuple[int, int]]:
    return [(1, min(CHUNK, n - lo)) for lo in range(0, n, CHUNK)]


def _probed(runs) -> None:
    if runs.baseline is None:
        pytest.skip("the per-rank probe runs through trtllm-llmapi-launch's MPI session")


def test_the_first_engine_gives_its_devices_back(runs):
    """After the chunked engine shuts down, every rank's device is back within RELEASE_SLACK of
    its free memory before that engine booted, so the delivered engine sizes its pool from the
    same devices."""
    _probed(runs)
    for rank, before in runs.baseline.items():
        after = runs.after_chunked[rank]
        assert (after["host"], after["device"]) == (before["host"], before["device"])
        assert before["free"] - after["free"] <= RELEASE_SLACK, (
            rank,
            before["free"],
            after["free"],
        )


def test_the_chunked_engine_splits_its_chunks(runs):
    """Each prompt is prefilled in steps of 8192 tokens and a last shorter one, one request per
    step. A step of q_split_threshold context rows or more is one the engine splits over every
    rank (`_split_prefill_queries` with attention DP off at tp 4 or tp 8, pinned on CPU by the
    target's engine-configuration test); the prompt set must hold at least one."""
    split_steps = 0
    for name, p in runs.prompts.items():
        steps = runs.chunked[name].steps
        assert steps == _chunks(len(p.ids)), (name, len(p.ids), steps)
        split_steps += sum(tokens >= Q_SPLIT_THRESHOLD for _, tokens in steps)
    assert split_steps >= 1


def _shares(tc: int, rank: int, world: int) -> tuple[int, int, list[int]]:
    """`_split_prefill_queries` at tp `world`, attention DP off: rank r's
    [tc*r//world, tc*(r+1)//world)."""
    bounds = [tc * r // world for r in range(world + 1)]
    return bounds[rank], bounds[rank + 1], [bounds[r + 1] - bounds[r] for r in range(world)]


def test_the_chunked_engine_splits_its_full_chunks(runs):
    """On every rank, each context step of q_split_threshold rows or more carries the engine's
    split, this rank's share of it, and one selection gather per selecting indexer layer (none
    where the engine skipped the context indexer); a shorter step carries no split and gathers
    nothing. Every rank sees the same steps. Among them are full 8192-row chunks over a cached
    prefix."""
    _probed(runs)
    per_rank = {r: report["steps"] for r, report in runs.after_chunked.items()}
    shape = [(s["tc"], s["q_split"] and s["q_split"][2], s["gathers"]) for s in per_rank[0]]
    over_prefix = 0
    for rank, steps in per_rank.items():
        assert [(s["tc"], s["q_split"] and s["q_split"][2], s["gathers"]) for s in steps] == shape
        for s in steps:
            assert s["rank"] == rank and s["cached_kv"], s
            if s["tc"] >= Q_SPLIT_THRESHOLD:
                assert s["q_split"] == _shares(s["tc"], rank, runs.world), (rank, s)
                assert s["gathers"] == (0 if s["skip_ctx"] else runs.selecting_layers), (rank, s)
                over_prefix += rank == 0 and not s["skip_ctx"] and sum(s["cached"]) > 0
            else:
                assert s["q_split"] is None and s["gathers"] == 0, (rank, s)
    assert over_prefix >= 1, shape


def test_the_delivered_engine_prefills_each_prompt_whole(runs):
    """Each prompt is one context step of one request: no cached prefix and no split; on every
    rank the probe reached, no context step over cached KV, no split and no gather."""
    for name, p in runs.prompts.items():
        assert runs.delivered[name].steps == [(1, len(p.ids))], (name, runs.delivered[name].steps)
    for rank, report in (runs.after_delivered or {}).items():
        for s in report["steps"]:
            assert not s["cached_kv"] and s["q_split"] is None and s["gathers"] == 0, (rank, s)


def _report(name: str, chunked, delivered) -> str:
    first = next(
        (i for i, (a, b) in enumerate(zip(chunked.tokens, delivered.tokens)) if a != b),
        min(len(chunked.tokens), len(delivered.tokens)),
    )
    lines = [f"{name}: first divergence at generated step {first}"]
    for label, run in (("chunked", chunked), ("delivered", delivered)):
        lines.append(f"  {label}: text={run.text!r} ids={run.tokens}")
        lines.append(
            f"  {label} (top-1, top-2, margin): "
            + ", ".join(f"({t1}, {t2}, {l1 - l2:.3f})" for t1, l1, t2, l2 in run.margins)
        )
    return "\n".join(lines)


@pytest.mark.parametrize("name", [spec[0] for spec in PROMPTS])
def test_same_greedy_tokens(runs, name):
    """Both engines decode the same greedy tokens from the same prompt."""
    chunked, delivered = runs.chunked[name], runs.delivered[name]
    assert chunked.tokens == delivered.tokens, _report(name, chunked, delivered)


@pytest.mark.parametrize("name", [spec[0] for spec in PROMPTS])
def test_the_answer_is_in_both(runs, name):
    """The passkey, or the sequence's next items, is in both engines' text: the prompts ask
    something both engines answer."""
    expect = runs.prompts[name].expect
    assert expect in runs.chunked[name].text, (name, expect, runs.chunked[name].text)
    assert expect in runs.delivered[name].text, (name, expect, runs.delivered[name].text)
