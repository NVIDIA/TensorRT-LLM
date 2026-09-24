# tests/jobs/test_registry.py
from pathlib import Path

from agent_flow.jobs.registry import JobEntry, JobRegistry


def _entry(**kw):
    base = dict(
        handle_key="mmmu-full",
        kind="slurm",
        handle={"job_id": "561654", "name": "s6-mmmu-full"},
        state="RUNNING",
    )
    base.update(kw)
    return JobEntry(**base)


def test_upsert_then_get_roundtrips(tmp_path: Path):
    reg = JobRegistry(tmp_path / "jobs.json")
    reg.upsert(_entry())
    got = reg.get("mmmu-full")
    assert got is not None
    assert got.kind == "slurm"
    assert got.handle["job_id"] == "561654"
    assert got.state == "RUNNING"


def test_upsert_same_key_replaces(tmp_path: Path):
    reg = JobRegistry(tmp_path / "jobs.json")
    reg.upsert(_entry(state="EXPECTED", handle={"name": "s6-mmmu-full"}))
    reg.upsert(_entry(state="RUNNING", handle={"job_id": "561654", "name": "s6-mmmu-full"}))
    assert reg.get("mmmu-full").state == "RUNNING"
    assert reg.get("mmmu-full").handle["job_id"] == "561654"
    assert len(reg.all()) == 1


def test_reload_from_disk(tmp_path: Path):
    p = tmp_path / "jobs.json"
    JobRegistry(p).upsert(_entry())
    assert JobRegistry(p).get("mmmu-full").handle["job_id"] == "561654"


def test_missing_file_is_empty(tmp_path: Path):
    assert JobRegistry(tmp_path / "nope.json").all() == []
