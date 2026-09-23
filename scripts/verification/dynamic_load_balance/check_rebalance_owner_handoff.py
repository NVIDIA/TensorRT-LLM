#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise actual handoff methods with CPU threads and a fake CUDA boundary.

No torch/TRT-LLM import is required. This verifies ownership/state ordering,
not actual GPU completion; the device drain is covered by the target E2E run.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import threading
from pathlib import Path
from types import SimpleNamespace


def extract_class(path, name, methods, torch):
    tree = ast.parse(path.read_text())
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    selected = [n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in methods]
    assert {n.name for n in selected} == set(methods)
    cls = ast.ClassDef(name=name, bases=[], keywords=[], body=selected, decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    namespace = {"torch": torch, "threading": threading}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[name], original


def must_reject(call):
    try:
        call()
    except RuntimeError:
        return
    raise AssertionError("unsafe ownership transition was accepted")


def in_worker(call):
    outcome = []

    def run():
        try:
            call()
        except BaseException as error:
            outcome.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive(), "CPU ownership check blocked"
    if outcome:
        raise outcome[0]


def check(repo):
    group_path = (
        repo / "tensorrt_llm/_torch/moe/fused_moe/mega_moe/rebalance_slot_scheduler_v2.py"
    )
    executor_path = repo / "tensorrt_llm/_torch/pyexecutor/py_executor.py"
    events = []
    main = SimpleNamespace(cuda_stream=11, priority=0, device=0)
    current = SimpleNamespace(stream=main, device=0)
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            current_device=lambda: current.device,
            current_stream=lambda device: current.stream,
            synchronize=lambda device: events.append(("synchronize", device)),
        )
    )
    Group, _ = extract_class(
        group_path,
        "RebalanceSlotSchedulerGroupV2",
        {"_check_owner", "release_warmup_owner_after_sync"},
        torch,
    )
    Executor, executor_source = extract_class(
        executor_path, "PyExecutor", {"_handoff_rebalance_warmup_owners"}, torch
    )
    passed = []

    def make_group(bound=True):
        group = Group()
        group.device = 0
        group._owner_thread_id = None
        group._execution_stream_handle = None
        group._stream_handle = 99
        group.copy_stream = SimpleNamespace(priority=-1)
        group._plan_part = None
        group._generation = group._finished_generation = group.plan_calls = 3
        group._pending_release = (3, object())
        group._route_wait_generation = 3
        if bound:
            group._check_owner()
        return group

    group = make_group()
    assert group._owner_thread_id == threading.get_ident()
    assert group._execution_stream_handle == main.cuda_stream
    in_worker(lambda: must_reject(group._check_owner))
    in_worker(lambda: must_reject(lambda: group.release_warmup_owner_after_sync(11)))
    passed.append("worker cannot steal a live warmup owner")

    for attr, value in (
        ("_plan_part", object()),
        ("_generation", 4),
        ("_finished_generation", 2),
        ("plan_calls", 4),
    ):
        old = getattr(group, attr)
        setattr(group, attr, value)
        must_reject(lambda: group.release_warmup_owner_after_sync(11))
        assert group._owner_thread_id == threading.get_ident()
        setattr(group, attr, old)
    must_reject(lambda: group.release_warmup_owner_after_sync(12))
    assert group._owner_thread_id == threading.get_ident()
    passed.append("inflight plans, counter mismatches, and wrong stream reject handoff")

    before = group.__dict__.copy()
    group.release_warmup_owner_after_sync(11)
    assert group._owner_thread_id is None
    assert {k: v for k, v in group.__dict__.items() if k != "_owner_thread_id"} == {
        k: v for k, v in before.items() if k != "_owner_thread_id"
    }
    assert group._pending_release is before["_pending_release"]
    current.stream = SimpleNamespace(cuda_stream=12, priority=0)
    in_worker(lambda: must_reject(group._check_owner))
    assert group._owner_thread_id is None
    current.stream = main
    passed.append("handoff preserves leases/counters and rejects a new MAIN stream")

    worker_id = []

    def bind_worker():
        group._check_owner()
        worker_id.append(threading.get_ident())
        current.stream = SimpleNamespace(cuda_stream=12, priority=0)
        must_reject(group._check_owner)
        current.stream = main
        group._check_owner()

    in_worker(bind_worker)
    assert group._owner_thread_id == worker_id[0]
    must_reject(group._check_owner)
    passed.append("serving thread rebinds only the original MAIN stream")

    for stream, device in (
        (SimpleNamespace(cuda_stream=99, priority=-1), 0),
        (SimpleNamespace(cuda_stream=12, priority=-1), 0),
        (main, 1),
    ):
        unbound = make_group(bound=False)
        current.stream, current.device = stream, device
        must_reject(unbound._check_owner)
        assert unbound._owner_thread_id is None
    current.stream, current.device = main, 0
    passed.append("original device, distinct-stream, and priority guards remain")

    target, draft, unbound = make_group(), make_group(), make_group(False)
    target_pending, draft_pending = target._pending_release, draft._pending_release
    for label, item in (("target", target), ("draft", draft)):
        original = item.release_warmup_owner_after_sync

        def logged_release(handle, original=original, label=label):
            events.append(("release", label))
            original(handle)

        item.release_warmup_owner_after_sync = logged_release

    def engine(*groups):
        modules = [SimpleNamespace(_rebalance_scheduler_group=item) for item in groups]
        modules.append(SimpleNamespace())
        return SimpleNamespace(model=SimpleNamespace(modules=lambda: modules))

    executor = Executor()
    executor.execution_stream = main
    executor.model_engine = engine(target, target, unbound)
    executor.draft_model_engine = engine(draft, target)
    executor._handoff_rebalance_warmup_owners()
    assert events == [("synchronize", 0), ("release", "target"), ("release", "draft")]
    assert target._owner_thread_id is draft._owner_thread_id is None
    assert target._pending_release is target_pending and draft._pending_release is draft_pending
    assert unbound._execution_stream_handle is None
    passed.append("one device drain precedes deduplicated target/draft owner release")

    events.clear()
    for model, draft_model in (
        (None, None),
        (SimpleNamespace(), None),
        (engine(), engine()),
        (engine(unbound), None),
    ):
        executor.model_engine, executor.draft_model_engine = model, draft_model
        executor._handoff_rebalance_warmup_owners()
    assert not events
    passed.append("OFF, missing-model, draftless, no-MoE, and unbound paths have no CUDA work")

    constructor = next(
        n for n in executor_source.body if isinstance(n, ast.FunctionDef) and n.name == "__init__"
    )
    calls = [n for n in ast.walk(constructor) if isinstance(n, ast.Call)]
    handoff = [
        n.lineno
        for n in calls
        if isinstance(n.func, ast.Attribute) and n.func.attr == "_handoff_rebalance_warmup_owners"
    ]
    starts = [
        n.lineno
        for n in calls
        if isinstance(n.func, ast.Attribute) and n.func.attr == "start_worker"
    ]
    warms = [
        n.lineno for n in calls if isinstance(n.func, ast.Attribute) and n.func.attr == "warmup"
    ]
    encoder_wait = [
        n.lineno
        for n in calls
        if isinstance(n.func, ast.Attribute)
        and n.func.attr == "result"
        and "_warmup_encoder_cuda_graphs_enc_dec" in ast.unparse(n)
    ]
    warmup_false = [
        n.lineno
        for n in ast.walk(constructor)
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Constant)
        and n.value.value is False
        and any(isinstance(t, ast.Attribute) and t.attr == "is_warmup" for t in n.targets)
    ]
    assert len(handoff) == 1 and warms and starts and encoder_wait and warmup_false
    assert max(warms + encoder_wait + warmup_false) < handoff[0] < min(starts)
    passed.append("constructor handoff follows all warmups and precedes worker startup")
    return {
        "status": "passed",
        "scope": "source-derived CPU contracts; CUDA completion requires GPU E2E",
        "checks": passed,
        "check_count": len(passed),
        "source_sha256": {
            str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (group_path, executor_path)
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = check(args.repo)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
