# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The config ladder on CPU: a fake runner writes the label dirs a bench job would."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from agent_flow.workflows.staircase_onboard.tune.ladder import (
    BenchSpec,
    Ladder,
    Rung,
    deep_merge,
    default_rungs,
)
from agent_flow.workflows.staircase_onboard.tune.records import Measurement, NoiseFloor, compare


class FakeRunner:
    """Finishes every job on the next poll with metrics chosen by the merged config."""

    def __init__(self, spools, data: Path, effect) -> None:
        self.spools = list(spools)
        self.data = data
        self.effect = effect
        self.inflight: dict[str, tuple[str, str]] = {}
        self.submitted: list[str] = []
        self.configs: dict[str, dict] = {}

    def free_spools(self):
        busy = {s for s, _ in self.inflight.values()}
        return [s for s in self.spools if s not in busy]

    def adopt(self, label):
        return False

    def submit(self, spool, label, command, *, wall="2h"):
        merged: dict = {}
        tokens = command.split()
        for i, tok in enumerate(tokens):
            if tok == "--config":
                merged = deep_merge(merged, yaml.safe_load(Path(tokens[i + 1]).read_text()) or {})
        self.configs[label] = merged
        self.inflight[label] = (spool, label)
        self.submitted.append(label)
        return label

    def poll(self):
        done = []
        for name, (spool, label) in list(self.inflight.items()):
            lat, thr = self.effect(self.configs[label], label)
            d = self.data / label
            d.mkdir(parents=True, exist_ok=True)
            rows = [
                {
                    "concurrency": 1,
                    "tok_s_user": lat,
                    "output_tok_s_gpu": lat,
                    "mean_tpot_ms": 1000 / lat,
                    "mean_ttft_ms": 200,
                },
                {
                    "concurrency": 128,
                    "tok_s_user": thr / 128,
                    "output_tok_s_gpu": thr,
                    "mean_tpot_ms": 50,
                    "mean_ttft_ms": 4000,
                },
            ]
            (d / "summary.json").write_text(json.dumps(rows))
            (d / "meta.json").write_text(
                json.dumps({"label": label, "modeling_v2": "require", "tree": {}})
            )
            del self.inflight[name]
            done.append((label, spool, 0))
        return done


def _effect(config: dict, label: str):
    lat, thr = 100.0, 2000.0
    if "control" in label:  # identical configs wander by +-0.5%
        lat *= 1.0 + 0.005 * (hash(label) % 3 - 1)
    cg = config.get("cuda_graph_config", {})
    if cg.get("enable_padding"):
        thr *= 1.10
    if cg.get("batch_sizes"):
        thr *= 1.06
    if config.get("enable_in_graph_sampling"):
        lat *= 1.05
    if config.get("max_num_tokens") == 16384:
        thr *= 0.95
    return lat, thr


def _ladder(tmp_path: Path, rungs):
    base = tmp_path / "identity.yaml"
    base.write_text(yaml.safe_dump({"modeling_v2": "require", "max_batch_size": 128}))
    bench = BenchSpec(
        bench_py="bench.py",
        model="m",
        env_script="env.sh",
        isl=8192,
        osl=1024,
        concurrencies=[1, 128],
        tree=".",
    )
    runner = FakeRunner(["s1", "s2", "s3"], tmp_path / "work" / "data", _effect)
    ladder = Ladder(
        workdir=tmp_path / "work",
        base_configs=[base],
        rungs=rungs,
        bench=bench,
        runner=runner,
        log=lambda m: None,
    )
    return ladder, runner


def test_ladder_keeps_winners_and_stops(tmp_path):
    rungs = [
        r
        for r in default_rungs()
        if r.name in ("cg-padding", "cg-dense", "in-graph-sampling", "mnt-2x", "gc-gen0-100k")
    ]
    ladder, runner = _ladder(tmp_path, rungs)
    for _ in range(50):
        if not ladder.step():
            break
    st = ladder.state
    assert st.phase == "done"
    # cg-padding beats cg-dense within the cuda-graph group; in-graph-sampling stacks.
    assert set(st.kept) == {"cg-padding", "in-graph-sampling"}
    assert st.rungs["cg-dense"].status == "rejected"
    assert st.rungs["mnt-2x"].status == "rejected"
    final = yaml.safe_load((tmp_path / "work" / "ladder_final.yaml").read_text())
    assert final["cuda_graph_config"] == {"enable_padding": True, "max_batch_size": 128}
    assert final["enable_in_graph_sampling"] is True
    kinds = [
        json.loads(line)["kind"]
        for line in (tmp_path / "work" / "perf_ledger.jsonl").read_text().splitlines()
    ]
    assert "noise_floor" in kinds and "ladder_combine" in kinds and kinds[-1] == "ladder_done"


def test_module_knobs_are_skipped_for_targets(tmp_path):
    rungs = [Rung("moe-cutlass", {"moe_config": {"backend": "CUTLASS"}}, "module knob")]
    ladder, _ = _ladder(tmp_path, rungs)
    assert "moe-cutlass" not in ladder.rungs


def test_templated_rungs_resolve_against_the_workload(tmp_path):
    ladder, _ = _ladder(tmp_path, default_rungs())
    dense = ladder.rungs["cg-dense"].resolve(ladder.fill)
    assert dense["cuda_graph_config"]["batch_sizes"] == list(range(1, 129))
    assert ladder.rungs["mnt-2x"].resolve(ladder.fill) == {"max_num_tokens": 16384}


def test_ladder_resumes_from_its_state_file(tmp_path):
    rungs = [r for r in default_rungs() if r.name in ("cg-padding", "in-graph-sampling")]
    ladder, runner = _ladder(tmp_path, rungs)
    ladder.step()  # controls queued
    ladder.step()  # controls measured, probes queued
    again, runner2 = _ladder(tmp_path, rungs)
    # the probes the first process queued have no job in the new runner: re-queued
    for _ in range(50):
        if not again.step():
            break
    assert set(again.state.kept) == {"cg-padding", "in-graph-sampling"}


def test_keep_rule_scales_with_noise(tmp_path):
    def m(label, lat, thr):
        d = tmp_path / label
        d.mkdir()
        (d / "summary.json").write_text(
            json.dumps(
                [
                    {
                        "concurrency": 1,
                        "tok_s_user": lat,
                        "output_tok_s_gpu": lat,
                        "mean_tpot_ms": 1,
                        "mean_ttft_ms": 1,
                    },
                    {
                        "concurrency": 128,
                        "tok_s_user": thr / 128,
                        "output_tok_s_gpu": thr,
                        "mean_tpot_ms": 1,
                        "mean_ttft_ms": 1,
                    },
                ]
            )
        )
        (d / "meta.json").write_text(json.dumps({"label": label}))
        return Measurement.load(d)

    ctrl, cand = m("ctrl", 100, 2000), m("cand", 104, 2000)
    assert compare(cand, ctrl, NoiseFloor()).verdict == "WIN"
    noisy = NoiseFloor(spread={"con1.tok_s_user": 0.025})
    assert compare(cand, ctrl, noisy).verdict == "NEUTRAL"
    worse = m("worse", 110, 1900)
    assert compare(worse, ctrl, NoiseFloor()).verdict == "LOSS"


def test_probes_are_judged_against_their_own_host(tmp_path):
    """GB300 nodes differed by 4% at the latency end for identical configs.

    A no-op rung measured on the slow node was judged a LOSS against the fast node's control.
    """
    spools = []
    for name, host in (("s1", "fast"), ("s2", "slow")):
        d = tmp_path / name
        d.mkdir()
        (d / "status.json").write_text(json.dumps({"host": host}))
        spools.append(str(d))

    class HostRunner(FakeRunner):
        def poll(self):
            done = super().poll()
            for label, spool, rc in done:
                meta = self.data / label / "meta.json"
                data = json.loads(meta.read_text())
                data["hostname"] = json.loads((Path(spool) / "status.json").read_text())["host"]
                meta.write_text(json.dumps(data))
                if data["hostname"] == "slow":
                    summ = self.data / label / "summary.json"
                    rows = json.loads(summ.read_text())
                    for r in rows:
                        r["tok_s_user"] *= 0.96
                        r["output_tok_s_gpu"] *= 0.96
                    summ.write_text(json.dumps(rows))
            return done

    base = tmp_path / "identity.yaml"
    base.write_text(yaml.safe_dump({"modeling_v2": "require"}))
    bench = BenchSpec(
        bench_py="b",
        model="m",
        env_script="e",
        isl=8192,
        osl=1024,
        concurrencies=[1, 128],
        tree=".",
    )
    runner = HostRunner(spools, tmp_path / "work" / "data", lambda config, label: (100.0, 2000.0))
    rungs = [
        Rung("noop-a", {"garbage_collection_gen0_threshold": 1}, "no effect"),
        Rung("noop-b", {"stream_interval": 1}, "no effect"),
    ]
    ladder = Ladder(
        workdir=tmp_path / "work",
        base_configs=[base],
        rungs=rungs,
        bench=bench,
        runner=runner,
        log=lambda m: None,
    )
    for _ in range(50):
        if not ladder.step():
            break
    verdicts = {rs.name: rs.verdict for rs in ladder.state.rungs.values()}
    assert verdicts == {"noop-a": "NEUTRAL", "noop-b": "NEUTRAL"}, ladder.summary()


def test_a_job_on_a_session_that_died_is_reported_lost(tmp_path):
    """The Slurm job behind a spool ended (its watcher died).

    The measurement in flight there is reported lost (rc 212) and set aside, not waited
    on forever.
    """
    import time as _time

    from agent_flow.workflows.staircase_onboard.substrate.scheduler import SpoolPaths
    from agent_flow.workflows.staircase_onboard.tune.ladder import SpoolRunner

    spool = tmp_path / "s1"
    paths = SpoolPaths(spool)
    paths.ensure()
    (paths.watchers / "w.json").write_text(json.dumps({"alive_epoch": _time.time() - 3600}))
    runner = SpoolRunner([str(spool)], prefix="camp")
    (spool / "running" / "camp-e2e-x.sh").write_text("#!/bin/bash\n")
    runner.inflight["camp-e2e-x"] = (str(spool), "e2e-x")
    assert runner.poll() == [("e2e-x", str(spool), 212)]
    assert not (spool / "running" / "camp-e2e-x.sh").exists() and not runner.inflight


def test_only_near_miss_neutral_rungs_are_probed_again(tmp_path):
    """After a keep, a neutral rung that came within reach (+0.7%) is measured against the new control.

    A flat one (+0.0%) is not -- re-probing every neutral rung costs a wave.
    """

    def effect(config, label):
        lat = 100.0 * (1.05 if config.get("enable_in_graph_sampling") else 1.0)
        return lat * (1.007 if config.get("stream_interval") == 4 else 1.0), 2000.0

    rungs = [
        r
        for r in default_rungs()
        if r.name in ("in-graph-sampling", "stream-interval-4", "gc-gen0-100k")
    ]
    base = tmp_path / "identity.yaml"
    base.write_text(yaml.safe_dump({"modeling_v2": "require", "max_batch_size": 128}))
    bench = BenchSpec(
        bench_py="bench.py",
        model="m",
        env_script="env.sh",
        isl=8192,
        osl=1024,
        concurrencies=[1, 128],
        tree=".",
    )
    runner = FakeRunner(["s1", "s2", "s3"], tmp_path / "work" / "data", effect)
    ladder = Ladder(
        workdir=tmp_path / "work",
        base_configs=[base],
        rungs=rungs,
        bench=bench,
        runner=runner,
        min_gain=0.01,
        noise_factor=1.0,
        log=lambda m: None,
    )
    for _ in range(80):
        if not ladder.step():
            break
    probes = [
        label for label in runner.submitted if "control" not in label and "combine" not in label
    ]
    assert sum("stream-interval-4" in label for label in probes) == 2, probes
    assert sum("gc-gen0-100k" in label for label in probes) == 1, probes


def test_a_probe_that_finished_while_no_process_watched_is_judged_on_resume(tmp_path):
    """The orchestrator restarts after a probe finished but before it polled it.

    The rung's measurement is on disk, so the resumed ladder judges it instead of waiting.
    """
    rungs = [r for r in default_rungs() if r.name in ("cg-padding", "in-graph-sampling")]
    ladder, runner = _ladder(tmp_path, rungs)
    ladder.step()  # controls queued
    ladder.step()  # controls measured, probes queued
    runner.poll()  # the probes finish; nobody calls ladder.step() before the restart
    assert {rs.status for rs in ladder.state.rungs.values()} == {"running"}
    again, runner2 = _ladder(tmp_path, rungs)
    for _ in range(50):
        if not again.step():
            break
    assert set(again.state.kept) == {"cg-padding", "in-graph-sampling"}
    assert not any(
        label.startswith("w00-cg-padding") for label in runner2.submitted
    )  # not measured twice


def test_a_dead_session_gets_no_control(tmp_path):
    """A session preempted mid-ladder refuses jobs (stale heartbeat).

    Queuing a control there crashed the stage. Controls go to live sessions only.
    """

    class OneDead(FakeRunner):
        def free_spools(self):
            return [s for s in super().free_spools() if s != "s3"]

    rungs = [r for r in default_rungs() if r.name in ("cg-padding", "in-graph-sampling")]
    base = tmp_path / "identity.yaml"
    base.write_text(yaml.safe_dump({"modeling_v2": "require", "max_batch_size": 128}))
    bench = BenchSpec(
        bench_py="bench.py",
        model="m",
        env_script="env.sh",
        isl=8192,
        osl=1024,
        concurrencies=[1, 128],
        tree=".",
    )
    runner = OneDead(["s1", "s2", "s3"], tmp_path / "work" / "data", _effect)
    ladder = Ladder(
        workdir=tmp_path / "work",
        base_configs=[base],
        rungs=rungs,
        bench=bench,
        runner=runner,
        log=lambda m: None,
    )
    for _ in range(50):
        if not ladder.step():
            break
    assert set(ladder.state.kept) == {"cg-padding", "in-graph-sampling"}
    assert not any(label.endswith("control-s3") for label in runner.submitted)
