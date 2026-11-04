# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Retrieval at length: Mistral-Large-4-1T-A52B-NVFP4 on its sm_103 tep4 target at the 1M config,
at tep4 (four ranks) and at tep8 (eight ranks over two nodes of four).

The 1M config: max_seq_len 1048576 (the checkpoint's max_position_embeddings), chunked prefill in
steps of max_num_tokens 32768, sixteen sequences, the fp8_ds_mla latent KV cache at its
default dtype, CUDA graphs over batches 1, 2, 4, 8 and 16, padded. A passkey hidden in plain
filler text at early, middle and late depths (0.1, 0.5, 0.9) of prompts of exactly 32768,
131072, 524288 and 1040000 tokens at tep4 (1040384 at tep8), one request at a time, must be
recalled by greedy decoding through the LLM API. Every chunk after a prompt's first is a
context step over a cached prefix: the rope append at the prefix's positions, the indexer's key
gather and scoring over the whole prefix, sparse MLA over selections that reach back to
position 0. Each full chunk is 32768 rows, so the engine splits its indexer queries over every
TP rank (q_split_threshold 8192; 8192 rows a rank at tep4, 4096 at tep8) and the target
all-gathers the selections, across both nodes at tep8; the longest prompts run fourteen such
chunks over cached prefixes past 524288.

Sparse MLA takes one of two paths per step. While the step's context KV (the request's cached and
new rows) fits the held context workspace, 2**19 rows at this config (`_ctx_workspace_rows`: chunked
prefill lets a step attend cached prefixes), those rows are dequantized out of the fp8_ds_mla pool
into it for the sparse prefill kernel; the 524288-token prompt's last chunk reaches exactly 2**19
and stays there. Past it -- the longest prompt's last 16 chunks -- every context row attends
straight off the pool as a one-row sequence of the sparse decode kernel, in sub-batches of 8192
rows. At tep8 the longest prompt is 1040384 tokens, 31 full chunks and a last one of three full
sub-batches, the sub-batch size the decode kernel's tep8 1M cells certify; at tep4 it is 1040000,
whose last chunk ends on a sub-batch of 7808 rows.

Per prompt, reported: the time to the first token and the decode time per token (sixteen
tokens, streamed), the context steps (read back from the engine's iteration statistics), and
the peak device memory of every rank while it runs, KV pool included. Torch's allocator
statistics miss the pool, so device-wide memory is read: this process polls
`torch.cuda.mem_get_info` on the four devices of its node (ranks 0-3), and every rank's own
process reads its device through NVML (`_retrieval_at_length_ranks.py`; at tep8 ranks 4-7 are
on the other node). The decode scorer's column count is frozen at max_seq_len, so a decode step
may cost more at this config than at the delivered one whatever the sequence's length: two short
prompts (one within the indexer's `index_topk`, one past it) are decoded for 128 tokens at this
config and at the delivered configuration (max_seq_len = max_num_tokens = 32768, chunked prefill
off), one engine after the other. The engines are named "envelope" (the 1M config) and
"delivered" in the printed lines.

What each rank's context steps received is read through an observe-only record on every rank,
installed and collected through the launcher's MPI session (the engines' own channel, idle
between them): the query split, the indexer chunk plan, the cached prefixes, the selection
gathers, and the most keys any indexer key gather, scorer or prefill top-k call received.
Outside the launcher there is no such session: the engines still run, and the checks on the
records are skipped.

Text only: no image, no chat template. Runs with the ranks already in place, on four GB300s at
tep4 or on eight over two nodes of four at tep8: pytest under `trtllm-llmapi-launch`, as the
whole-model accuracy class runs, with the checkpoint under `LLM_MODELS_ROOT` as
`Mistral-Large-4-1T-A52B-NVFP4` (or `mistral-large-4-nvfp4`) in its Mistral-native format,
which both engines load with checkpoint_format "mistral_large_4". Skipped where the checkpoint
is absent or off sm_103; each world size is skipped below its own ranks (four GPUs at tep4, eight
ranks at tep8) and inside a launch of another world size. About 25 minutes, two engine loads
included.
"""

from __future__ import annotations

import random
import site
import threading
import time
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

from . import _retrieval_at_length_ranks as rank_side


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
    # and stop it when the fixture shuts them down; the memory poller is a daemon thread.
    pytest.mark.threadleak(enabled=False),
]

CHECKPOINTS = ("Mistral-Large-4-1T-A52B-NVFP4", "mistral-large-4-nvfp4")

MAX_SEQ_LEN = 1048576
CHUNK = 32768
DELIVERED_LEN = 32768
INDEX_TOPK = 2048
# sparse_attention_config.q_split_threshold's default (llmapi/llm_args.py): a step of at least
# this many context rows is split over the TP ranks.
Q_SPLIT_THRESHOLD = 8192
# A cached prefix past this is "at length" for the split's check.
FAR_PREFIX = 524288

COMMON = dict(
    enable_attention_dp=False,
    max_batch_size=16,
    cuda_graph_config=CudaGraphConfig(max_batch_size=16, enable_padding=True),
    kv_cache_config=KvCacheConfig(enable_block_reuse=False, free_gpu_memory_fraction=0.85),
)
ENGINES = {
    "envelope": dict(max_seq_len=MAX_SEQ_LEN, max_num_tokens=CHUNK, enable_chunked_prefill=True),
    "delivered": dict(
        max_seq_len=DELIVERED_LEN, max_num_tokens=DELIVERED_LEN, enable_chunked_prefill=False
    ),
}

# Prompt lengths in tokens, BOS included, by world; the longest leaves room for the answer
# within max_seq_len. At tep8 the longest one's last chunk is 3 x 8192 rows, so every
# decode-path sub-batch is full.
LENGTHS = {4: (32768, 131072, 524288, 1040000), 8: (32768, 131072, 524288, 1040384)}
# The lengths' test ids, one per place in LENGTHS at either world.
LENGTH_IDS = ("32k", "128k", "512k", "longest")
DEPTHS = (0.1, 0.5, 0.9)
# Short prompts decoded at both engines: one within index_topk (the engine skips the indexer
# and its decode graph has no scorer), one past it (the decode scorer runs over max_seq_len
# columns).
SHORT = (1000, 8000)

HEAD = (
    "There is an important piece of information hidden inside a lot of irrelevant text. "
    "Find it and memorize it. I will quiz you about it.\n\n"
)
FILLER = (
    "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again. "
)
QUESTION = "\nWhat is the pass key? The pass key is"
# The key's tokens come first; past them the model's text is a near tie no engine repeats, so
# only the key is checked. A fixed count of tokens times the decode.
NEEDLE = SamplingParams(max_tokens=16, temperature=0.0, ignore_eos=True)
SHORT_DECODE = SamplingParams(max_tokens=128, temperature=0.0, ignore_eos=True)

GIB = 2**30


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


def _peak_used(changes: list[tuple[float, int]], start: float, end: float) -> int | None:
    """The most bytes a device held over [start, end]: the value in effect at `start` and every
    change up to `end`."""
    before = [used for t, used in changes if t <= start]
    inside = [used for t, used in changes if start < t <= end]
    values = before[-1:] + inside
    return max(values) if values else None


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


def _prompt(reps: int, depth: float, key: int) -> str:
    before = round(reps * depth)
    needle = f"\nThe pass key is {key}. Remember it. {key} is the pass key.\n"
    return HEAD + FILLER * before + needle + FILLER * (reps - before) + QUESTION


def _passkey_ids(tokenizer, tokens: int, depth: float, key: int) -> list[int]:
    """Exactly `tokens` ids, BOS included as the LLM API's own tokenization adds it: the prompt
    with the fewest whole filler sentences that reach `tokens`, tokenized as one text, then the
    few ids past `tokens` taken off the filler right after the head. A long run of filler
    tokenizes a little shorter per sentence than a short one, so the count is topped up from
    what the whole text gives."""
    head = tokenizer.encode(HEAD)
    unit = len(tokenizer.encode(FILLER * 65)) - len(tokenizer.encode(FILLER * 64))
    bare = len(tokenizer.encode(_prompt(0, depth, key)))
    reps = max(0, -(-(tokens - bare) // unit))
    ids = tokenizer.encode(_prompt(reps, depth, key))
    while len(ids) < tokens:
        reps += max(1, (tokens - len(ids)) // unit)
        ids = tokenizer.encode(_prompt(reps, depth, key))
    excess = len(ids) - tokens
    assert ids[: len(head)] == head and excess < 4 * unit <= round(reps * depth) * unit, (
        tokens,
        excess,
        unit,
    )
    return ids[: len(head)] + ids[len(head) + excess :]


def _keys(count: int, seed: int) -> list[int]:
    return random.Random(seed).sample(range(10000, 100000), count)


def _chunks(n: int) -> list[tuple[int, int]]:
    """The context steps of one `n`-token prompt alone at the 1M config."""
    return [(1, min(CHUNK, n - lo)) for lo in range(0, n, CHUNK)]


def _shares(tc: int, rank: int, world: int) -> tuple[int, int, list[int]]:
    """`_split_prefill_queries` at tp `world`, attention DP off: rank r's
    [tc*r//world, tc*(r+1)//world)."""
    bounds = [tc * r // world for r in range(world + 1)]
    return bounds[rank], bounds[rank + 1], [bounds[r + 1] - bounds[r] for r in range(world)]


def _context_steps(llm: LLM) -> list[tuple[int, int]]:
    """(context requests, context tokens) of each iteration since the last call that ran any
    context rows."""
    steps = []
    for record in llm.get_stats(timeout=2):
        ifb = record.get("inflightBatchingStats") or {}
        if ifb.get("numContextRequests", 0):
            steps.append((ifb["numContextRequests"], ifb["numCtxTokens"]))
    return steps


class _DeviceMemory:
    """The least free memory of every device of this node since the last `reset`, polled with
    `torch.cuda.mem_get_info` from a thread of this process (not an engine's): device-wide, so
    it counts what every rank's process holds, the KV pool and the workspaces outside torch's
    allocator included. Device d of this node runs rank d."""

    def __init__(self, period_s: float = 0.02) -> None:
        self.devices = list(range(torch.cuda.device_count()))
        self.total = [torch.cuda.mem_get_info(d)[1] for d in self.devices]
        self._period = period_s
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._least = self._free()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _free(self) -> list[int]:
        return [torch.cuda.mem_get_info(d)[0] for d in self.devices]

    def _run(self) -> None:
        while not self._stop.is_set():
            free = self._free()
            with self._lock:
                self._least = [min(a, b) for a, b in zip(self._least, free)]
            self._stop.wait(self._period)

    def reset(self) -> list[float]:
        """Start a new window; the used GiB per device now."""
        free = self._free()
        with self._lock:
            self._least = list(free)
        return [round((t - f) / GIB, 2) for t, f in zip(self.total, free)]

    def peak(self) -> list[float]:
        """The most GiB used per device in this window."""
        free = self._free()
        with self._lock:
            least = [min(a, b) for a, b in zip(self._least, free)]
        return [round((t - f) / GIB, 2) for t, f in zip(self.total, least)]

    def stop(self) -> None:
        self._stop.set()
        self._thread.join()


def _generate(llm: LLM, ids: list[int], params: SamplingParams, memory: _DeviceMemory):
    """One request alone, streamed: its text, its tokens, the client-side time to the first
    token and decode time per token, the engine's own first-token and last-token times, its
    context steps, this node's devices' peak memory while it ran and its wall-clock window."""
    before = memory.reset()
    start, wall_start = time.perf_counter(), time.time()
    first = None
    result = llm.generate_async(TokensPrompt(prompt_token_ids=ids), params, streaming=True)
    for partial in result:
        if first is None and partial.outputs[0].token_ids:
            first = time.perf_counter()
    end, wall_end = time.perf_counter(), time.time()
    peak = memory.peak()
    out = result.outputs[0]
    n = len(out.token_ids)
    engine_ttft = engine_tpot = None
    metrics = out.request_perf_metrics
    if metrics is not None and metrics.timing_metrics is not None:
        timing = metrics.timing_metrics
        arrival = timing.arrival_time.total_seconds()
        first_token = timing.first_token_time.total_seconds()
        last_token = timing.last_token_time.total_seconds()
        if first_token > 0 and arrival > 0:
            engine_ttft = round(first_token - arrival, 3)
        if n > 1 and last_token > first_token > 0:
            engine_tpot = round((last_token - first_token) / (n - 1), 5)
    return SimpleNamespace(
        tokens=len(ids),
        text=out.text,
        generated=n,
        ttft_s=round(first - start, 3),
        tpot_s=round((end - first) / max(n - 1, 1), 5),
        engine_ttft_s=engine_ttft,
        engine_tpot_s=engine_tpot,
        steps=_context_steps(llm),
        used_before_gib=before,
        peak_gib=peak,
        window=(wall_start, wall_end),
        rank_peak_gib=None,
    )


def _engine(path: Path, name: str, world: int) -> LLM:
    return LLM(
        str(path),
        checkpoint_format="mistral_large_4",
        modeling_v2="require",
        env_overrides={MODELING_V2_VALIDATE_ENV: "1"},
        enable_iter_perf_stats=True,
        return_perf_metrics=True,
        tensor_parallel_size=world,
        moe_expert_parallel_size=world,
        **COMMON,
        **ENGINES[name],
    )


def _rank_peaks(
    reports: dict[int, dict] | None, window: tuple[float, float]
) -> list[float | None] | None:
    """Every rank's most device memory over `window`, GiB, from its own NVML record; None
    without the records."""
    if reports is None:
        return None
    peaks = []
    for rank in sorted(reports):
        used = _peak_used(reports[rank]["memory"], *window)
        peaks.append(None if used is None else round(used / GIB, 2))
    return peaks


def _report(label: str, r) -> None:
    print(
        f"RETRIEVAL {label} tokens={r.tokens} ttft_s={r.ttft_s} tpot_s={r.tpot_s} "
        f"engine_ttft_s={r.engine_ttft_s} engine_tpot_s={r.engine_tpot_s} "
        f"generated={r.generated} context_steps={len(r.steps)} "
        f"steps={r.steps if len(r.steps) <= 4 else (r.steps[:2], '...', r.steps[-2:])} "
        f"used_before_gib(this node, mem_get_info)={r.used_before_gib} "
        f"peak_gib(this node, mem_get_info)={r.peak_gib} "
        f"peak_gib(every rank, nvml)={r.rank_peak_gib} text={r.text!r}",
        flush=True,
    )


def _short_decodes(llm: LLM, memory: _DeviceMemory) -> dict:
    results = {}
    for tokens, key in zip(SHORT, _keys(len(SHORT), seed=7)):
        r = _generate(llm, _passkey_ids(llm.tokenizer, tokens, 0.5, key), SHORT_DECODE, memory)
        r.key = key
        results[tokens] = r
    return results


def _print_steps(label: str, reports: dict[int, dict] | None) -> None:
    """Each rank's context steps over a cached prefix past FAR_PREFIX, the first and last of
    them in full, and the most keys each indexer prefill entry received in any step."""
    if reports is None:
        return
    for rank, report in sorted(reports.items()):
        steps = report["steps"]
        far = [s for s in steps if s["cached"] and s["cached"][0] > FAR_PREFIX]
        most = {
            name: max((s["keys"][name] for s in steps), default=0)
            for name in (
                "indexer_k_cache_gather_op",
                "fp8_mqa_logits",
                "cute_dsl_gvr_topk_prefill",
            )
        }
        print(
            f"RETRIEVAL steps {label} rank {rank} {report['host']}:{report['device']}: "
            f"{len(steps)} context steps, {len(far)} over a cached prefix past {FAR_PREFIX}; "
            f"most keys per call {most}; nvml_error {report['nvml_error']}",
            flush=True,
        )
        for s in far[:1] + far[-1:]:
            print(f"RETRIEVAL step {label} rank {rank} {s}", flush=True)


@pytest.fixture(scope="module", params=WORLDS)
def runs(request):
    world = request.param
    path = _checkpoint_dir()
    if path is None:
        pytest.skip(f"checkpoint {CHECKPOINTS} is not under LLM_MODELS_ROOT ({llm_models_root()})")
    # The checkpoint's config as the engines read it from params.json.
    config = MistralConfigLoader().load_pretrained_config(str(path))
    # The engines' own session under trtllm-llmapi-launch: a singleton every LLM in this
    # process reuses. Unlaunched, each engine spawns its own workers and no record runs.
    session = None
    if get_spawn_proxy_process_env():
        session = create_mpi_comm_session(world)
        _reach_ranks(session)
    _probe(session, "install", world)
    memory = _DeviceMemory()
    needles, short = {}, {}
    with _engine(path, "envelope", world) as llm:
        print(f"RETRIEVAL envelope kv_cache_capacity {llm.get_kv_cache_capacity()}", flush=True)
        print(f"RETRIEVAL envelope used_gib_when_up(this node) {memory.reset()}", flush=True)
        _context_steps(llm)
        for tokens in LENGTHS[world]:
            for depth, key in zip(DEPTHS, _keys(len(DEPTHS), seed=tokens)):
                ids = _passkey_ids(llm.tokenizer, tokens, depth, key)
                r = _generate(llm, ids, NEEDLE, memory)
                r.key, r.depth = key, depth
                needles[(tokens, depth)] = r
                print(f"RETRIEVAL needle tokens={tokens} depth={depth} key={key} "
                      f"ttft_s={r.ttft_s} text={r.text!r}", flush=True)  # fmt: skip
        short["envelope"] = _short_decodes(llm, memory)
    envelope_ranks = _probe(session, "collect", world)
    with _engine(path, "delivered", world) as llm:
        print(f"RETRIEVAL delivered kv_cache_capacity {llm.get_kv_cache_capacity()}", flush=True)
        _context_steps(llm)
        short["delivered"] = _short_decodes(llm, memory)
    delivered_ranks = _probe(session, "remove", world)
    memory.stop()

    for (tokens, depth), r in needles.items():
        r.rank_peak_gib = _rank_peaks(envelope_ranks, r.window)
        _report(f"needle depth={depth} key={r.key}", r)
    for name, reports in (("envelope", envelope_ranks), ("delivered", delivered_ranks)):
        for tokens, r in short[name].items():
            r.rank_peak_gib = _rank_peaks(reports, r.window)
            _report(f"{name} short key={r.key}", r)
    for tokens in SHORT:
        env, dlv = short["envelope"][tokens], short["delivered"][tokens]
        print(
            f"RETRIEVAL short decode tokens={env.tokens} tpot_s envelope {env.tpot_s} "
            f"delivered {dlv.tpot_s} (engine {env.engine_tpot_s} / {dlv.engine_tpot_s})",
            flush=True,
        )
    _print_steps("envelope", envelope_ranks)
    return SimpleNamespace(
        world=world,
        needles=needles,
        short=short,
        steps=None
        if envelope_ranks is None
        else {rank: report["steps"] for rank, report in envelope_ranks.items()},
        ranks=envelope_ranks,
        selecting_layers=sum(role == "full" for role in config.index_sharing),
    )


# ------------------------------------------------------------------------------------ tests


def _recorded(runs) -> None:
    if runs.ranks is None:
        pytest.skip("the per-rank record runs through trtllm-llmapi-launch's MPI session")


@pytest.mark.parametrize("length", range(len(LENGTH_IDS)), ids=LENGTH_IDS)
def test_passkey_recalled_at_every_depth(runs, length):
    """The key at depths 0.1, 0.5 and 0.9 of an exactly `tokens`-token prompt (the world's
    `length`-th in LENGTHS), sent alone, is in the greedy answer; the prompt ran in chunks of
    32768 tokens, one context step each."""
    tokens = LENGTHS[runs.world][length]
    misses = []
    for depth in DEPTHS:
        r = runs.needles[(tokens, depth)]
        assert r.tokens == tokens, (r.tokens, tokens)
        assert r.steps == _chunks(tokens), (tokens, depth, r.steps)
        assert r.generated == NEEDLE.max_tokens, r.generated
        if str(r.key) not in r.text:
            misses.append((tokens, depth, r.key, r.text))
    assert not misses, misses


def test_split_chunks_ran_over_long_prefixes(runs):
    """The longest prompts (1040000 tokens at tep4, 1040384 at tep8) ran 32768-token context
    steps over cached prefixes past 524288 tokens: steps of at least q_split_threshold rows,
    which the engine splits over every rank."""
    for depth in DEPTHS:
        steps = runs.needles[(LENGTHS[runs.world][-1], depth)].steps
        cached = [CHUNK * k for k in range(len(steps))]
        far = [c for c, (_, rows) in zip(cached, steps) if c > 524288 and rows == CHUNK]
        assert CHUNK >= Q_SPLIT_THRESHOLD and len(far) == 14, (depth, far)


def test_split_chunks_ran_over_long_prefixes_on_every_rank(runs):
    """On every rank, each longest prompt ran fourteen 32768-token context steps over a cached
    prefix past 524288 tokens. Each carried the engine's query split with this rank's share
    (8192 rows at tep4, 4096 at tep8); the rank gathered keys and scored them over the prefix and
    its share's rows (cached + the share's end keys) and made one selection gather per selecting
    indexer layer."""
    _recorded(runs)
    for rank, steps in runs.steps.items():
        far = [
            s
            for s in steps
            if s["contexts"] == 1 and s["new"] == [CHUNK] and s["cached"][0] > FAR_PREFIX
        ]
        assert len(far) == 14 * len(DEPTHS), (rank, [(s["cached"], s["new"]) for s in far])
        for s in far:
            lo, hi, sizes = _shares(CHUNK, rank, runs.world)
            keys = s["cached"][0] + hi
            assert s["q_split"] == (lo, hi, sizes), (rank, s)
            assert s["gathers"] == runs.selecting_layers, (rank, s)
            assert s["keys"]["indexer_k_cache_gather_op"] == keys, (rank, s)
            assert s["keys"]["fp8_mqa_logits"] == keys, (rank, s)
            assert keys <= s["keys"]["cute_dsl_gvr_topk_prefill"] <= MAX_SEQ_LEN, (rank, s)


def test_no_indexer_prefill_call_past_max_seq_len_keys(runs):
    """No key gather, scorer or prefill top-k call on any rank received more than max_seq_len
    keys, and the longest prompt's last chunk took its scorer to all of its keys."""
    _recorded(runs)
    most = {}
    for steps in runs.steps.values():
        for s in steps:
            for name, keys in s["keys"].items():
                most[name] = max(most.get(name, 0), keys)
    assert all(keys <= MAX_SEQ_LEN for keys in most.values()), most
    longest = LENGTHS[runs.world][-1]
    assert most["indexer_k_cache_gather_op"] == most["fp8_mqa_logits"] == longest, most


def test_every_rank_reported_its_device_memory(runs):
    """Every rank's NVML record covers each needle's window."""
    _recorded(runs)
    for rank, report in runs.ranks.items():
        assert report["nvml_error"] is None and report["memory"], (rank, report["nvml_error"])
    for key, r in runs.needles.items():
        assert None not in r.rank_peak_gib, (key, r.rank_peak_gib)


@pytest.mark.parametrize("tokens", SHORT)
def test_short_prompt_decodes_at_both_envelopes(runs, tokens):
    """A short prompt's key is recalled and 128 tokens are decoded at the 1M config and at the
    delivered 32768 one; their decode times per token are reported side by side."""
    for name in ("envelope", "delivered"):
        r = runs.short[name][tokens]
        assert r.generated == SHORT_DECODE.max_tokens, (name, r.generated)
        assert r.steps == [(1, r.tokens)], (name, r.steps)
        assert str(r.key) in r.text, (name, r.key, r.text)
