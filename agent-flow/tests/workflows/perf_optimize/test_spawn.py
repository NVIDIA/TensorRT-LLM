"""The package's second subprocess door, and how narrow it is."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from agent_flow.workflows.perf_optimize import disagg_sol, spawn

POINT = {"shape": "tep_4_eplb0_mtp3", "concurrency": 1, "prefer": "interactive"}


def _launch(tmp_path, track="gen") -> disagg_sol.CampaignLaunch:
    return disagg_sol.CampaignLaunch(
        track=track,
        spec={"checkpoint_path": "/ckpt", "sol_track": {"track": track, "sweep": "/s.yaml"}},
        workspace=tmp_path / f"ws-{track}",
        task_path=tmp_path / f"ws-{track}" / "task.yaml",
    )


def test_what_a_campaign_runs_under_is_on_disk_before_it_runs(tmp_path):
    """Reviewable before any process exists -- and a dry run is the same path."""
    launch = _launch(tmp_path)
    path = spawn.materialize(launch)
    assert path.is_file()
    assert yaml.safe_load(path.read_text())["sol_track"]["track"] == "gen"


def test_starting_without_materializing_is_refused(tmp_path):
    """Otherwise the child reads a task.yaml nobody wrote."""
    with pytest.raises(disagg_sol.DisaggSolError, match="materialize"):
        spawn.start(_launch(tmp_path))


def test_only_perf_optimize_is_ever_started(tmp_path):
    """The door admits one kind of process, and argv is built not interpolated."""
    launch = _launch(tmp_path)
    assert launch.argv[0] == "perf-optimize"
    assert all(isinstance(a, str) for a in launch.argv)
    assert not any(";" in a or "|" in a or "&" in a for a in launch.argv)


def test_a_spec_that_cannot_be_written_stops_the_run_before_anything_starts(tmp_path, monkeypatch):
    """Half-started is the one state a supervisor must not produce.

    The campaign that did start would claim a checkout the other needed, and
    the failure would surface as a repo-claim refusal minutes later, pointing
    at the wrong cause.
    """
    good, bad = _launch(tmp_path, "gen"), _launch(tmp_path, "ctx")
    started: list[str] = []
    monkeypatch.setattr(spawn, "start", lambda launch, env=None: started.append(launch.track))

    real_materialize = spawn.materialize

    def explode(launch):
        if launch.track == "ctx":
            raise OSError("read-only filesystem")
        return real_materialize(launch)

    monkeypatch.setattr(spawn, "materialize", explode)
    with pytest.raises(OSError):
        spawn.start_all([good, bad])
    assert started == []


def test_every_campaign_is_waited_on_even_after_one_fails():
    """Every campaign is joined even after one has already failed.

    The other is still running on a cluster; abandoning it strands an
    allocation and a claimed checkout with nothing tracking either.
    """

    class Fake:
        def __init__(self, code):
            self.code = code
            self.waited = False

        def wait(self):
            self.waited = True
            return self.code

    a, b = Fake(1), Fake(0)
    launches = [
        disagg_sol.CampaignLaunch("ctx", {}, Path("/w1"), Path("/w1/t.yaml")),
        disagg_sol.CampaignLaunch("gen", {}, Path("/w2"), Path("/w2/t.yaml")),
    ]
    assert spawn.wait_all([(launches[0], a), (launches[1], b)]) == {"ctx": 1, "gen": 0}
    assert a.waited and b.waited
