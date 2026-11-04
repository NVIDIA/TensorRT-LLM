# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Request-linked evidence for long speculative decoding, collected only by the test."""

import contextlib
import json
from collections import defaultdict
from pathlib import Path

_original_collect = None


def request_snapshot(req, phase: str) -> dict:
    """Results expose client IDs; executor IDs are a separate rank-local namespace."""
    return dict(
        id=req.py_client_id,
        executor_id=req.request_id,
        phase=phase,
        dummy=bool(req.is_dummy),
        prompt_tokens=req.orig_prompt_len,
        tokens=req.get_num_tokens(0),
        draft_tokens=len(req.py_draft_tokens),
    )


def _install_trace(directory: str | None) -> None:
    from tensorrt_llm._torch.pyexecutor.py_executor import PyExecutor
    from tensorrt_llm._utils import mpi_barrier, mpi_rank

    global _original_collect
    # A barrier makes each submitted task occupy a different MPI worker.
    mpi_barrier()
    if directory is None:
        PyExecutor._collect_scheduled_batch_stats = _original_collect
        _original_collect = None
        return
    assert _original_collect is None
    _original_collect = PyExecutor._collect_scheduled_batch_stats
    path = Path(directory) / f"rank-{mpi_rank()}.jsonl"

    def collect(executor, batch):
        stats = _original_collect(executor, batch)
        requests = []
        for phase, items in (
            ("context", batch.context_requests),
            ("generation", batch.generation_requests),
        ):
            for req in items:
                requests.append(request_snapshot(req, phase))
        with path.open("a") as stream:
            stream.write(
                json.dumps(
                    dict(
                        iteration=executor.iter_counter,
                        rank=executor.dist.rank,
                        context_tokens=stats.num_ctx_tokens,
                        requests=requests,
                    )
                )
                + "\n"
            )
        return stats

    PyExecutor._collect_scheduled_batch_stats = collect


@contextlib.contextmanager
def traced_llm(model: str, trace: Path, **kwargs):
    """Use the normal MPI substrate, with a scoped observer on every worker."""
    from tensorrt_llm import LLM
    from tensorrt_llm.executor.utils import create_mpi_comm_session, get_spawn_proxy_process_env
    from tensorrt_llm.llmapi.mpi_session import MpiPoolSession

    trace.mkdir()
    world = kwargs["tensor_parallel_size"]
    session = (
        create_mpi_comm_session(world)
        if get_spawn_proxy_process_env()
        else MpiPoolSession(n_workers=world)
    )
    installed = False
    try:
        session.submit_sync(_install_trace, str(trace))
        installed = True
        with LLM(model, _mpi_session=session, **kwargs) as llm:
            yield llm
    finally:
        try:
            if installed:
                session.submit_sync(_install_trace, None)
        finally:
            session.shutdown()


def request_metrics(result) -> dict:
    """Keep cumulative counts and positional counts; zero/missing counts are not proof."""
    totals = result.spec_dec_totals
    accepted, drafted = totals if totals is not None else (0, 0)
    return dict(
        accepted=accepted,
        drafted=drafted,
        rejected=drafted - accepted,
        drafted_per_position=result.per_pos_drafted,
        accepted_per_position=result.per_pos_accepted,
    )


def summarize(
    predictions: list[dict],
    steps: list[dict],
    need: int,
    world: int,
    attention_dp: bool,
    max_context_tokens: int = 32768,
) -> dict:
    """Require actual long verification, reuse, rejection, mixed steps and DEP padding."""
    by_request = {row["request_id"]: row for row in predictions}
    failures = []
    grouped = defaultdict(lambda: dict(accepted=0, drafted=0, iterations=0, requests=0))
    long_requests, rejected = set(), set()
    reused = defaultdict(set)
    for row in predictions:
        metrics = row["spec_decoding"]
        accepted, drafted = metrics["accepted"], metrics["drafted"]
        assert 0 <= accepted <= drafted, metrics
        positions = metrics["drafted_per_position"] or []
        group = grouped[f"{row['id'][1]}/{row['phase']}"]
        group["accepted"] += accepted
        group["drafted"] += drafted
        group["iterations"] += positions[0] if positions else 0
        group["requests"] += 1
        if (
            row["prompt_tokens"] >= need
            and len(positions) >= 3
            and min(positions[:3]) > 0
            and not any(positions[3:])
        ):
            long_requests.add(row["request_id"])
            if (
                row["phase"] in ("first_b", "warm_a")
                and row["cached_tokens"] > 0
                and row["score"] == 1
                and accepted > 0
            ):
                reused[row["phase"]].add(row["request_id"])
            if drafted > accepted and row["score"] == 1:
                rejected.add(row["request_id"])
    for group in grouped.values():
        group["acceptance_length"] = (
            1 + group["accepted"] / group["iterations"] if group["iterations"] else None
        )

    verified, mixed, iterations = set(), [], defaultdict(dict)
    for step in steps:
        iterations[step["iteration"]][step["rank"]] = step
        assert 0 <= step["context_tokens"] <= max_context_tokens, step
        live_long = [
            req
            for req in step["requests"]
            if not req["dummy"]
            and req["phase"] == "generation"
            and req["id"] in long_requests
            and req["draft_tokens"] == 3
            and req["prompt_tokens"] == by_request[req["id"]]["prompt_tokens"]
        ]
        verified.update(req["id"] for req in live_long)
        cold_chunks = [
            req["id"]
            for req in step["requests"]
            if not req["dummy"]
            and req["phase"] == "context"
            and req["id"] in by_request
            and req["prompt_tokens"] == by_request[req["id"]]["prompt_tokens"]
            and req["prompt_tokens"] > max_context_tokens
            and by_request[req["id"]]["cached_tokens"] == 0
        ]
        if live_long and cold_chunks and step["context_tokens"] > 0:
            mixed.append(
                dict(iteration=step["iteration"], rank=step["rank"], cold_requests=cold_chunks)
            )
    padding = []
    if attention_dp:
        for iteration, ranks in iterations.items():
            if set(ranks) != set(range(world)):
                continue
            live = [
                rank
                for rank, step in ranks.items()
                if any(
                    not req["dummy"]
                    and req["id"] in verified
                    and req["phase"] == "generation"
                    and req["draft_tokens"] == 3
                    for req in step["requests"]
                )
            ]
            dummy = [
                rank
                for rank, step in ranks.items()
                if any(
                    req["dummy"] and req["phase"] == "generation" and req["draft_tokens"] == 3
                    for req in step["requests"]
                )
            ]
            if live and set(dummy) - set(live):
                padding.append(dict(iteration=iteration, live_ranks=live, dummy_ranks=dummy))
        if not padding:
            failures.append("INCONCLUSIVE: no rank-local long speculative DEP dummy witness")
    if not verified:
        failures.append(f"INCONCLUSIVE: no completed {need}-token four-token verification")
    if any(not reused[phase] & verified for phase in ("first_b", "warm_a")):
        failures.append("INCONCLUSIVE: no long speculative warm/cached-prefix retrieval")
    if not rejected & verified:
        failures.append("INCONCLUSIVE: no successful long retrieval following draft rejection")
    if not mixed:
        failures.append("INCONCLUSIVE: no same-step long verification and chunked prefill")
    return dict(
        required_prompt_tokens=need,
        verified_requests=sorted(verified),
        rejected_requests=sorted(rejected & verified),
        by_length_phase=dict(grouped),
        mixed_steps=mixed,
        dep_padding=padding,
        failures=failures,
    )


def evaluate(root: Path, need: int, world: int, attention_dp: bool) -> dict:
    predictions = [
        json.loads(line) for line in (root / "ruler_predictions.jsonl").read_text().splitlines()
    ]
    steps = [
        json.loads(line)
        for path in sorted((root / "spec-trace").glob("rank-*.jsonl"))
        for line in path.read_text().splitlines()
    ]
    result = summarize(predictions, steps, need, world, attention_dp)
    (root / "spec_evidence.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
