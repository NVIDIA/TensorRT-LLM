from pathlib import Path

from agent_flow.jobs.registry import JobEntry, JobRegistry
from agent_flow.workflows.agent_team.node_workspace import NODES_DIRNAME, node_dir_slug
from agent_flow.workflows.agent_team.workflow import AgentTeamWorkflow


class _RecordingKind:
    def __init__(self):
        self.cancelled = []

    def status(self, handle): ...

    def cancel(self, handle):
        self.cancelled.append(handle)


def _seed_job(workspace: Path, node_id: str, kind: str) -> None:
    jobs = workspace / NODES_DIRNAME / node_dir_slug(node_id) / "jobs.json"
    reg = JobRegistry(jobs)
    reg.upsert(
        JobEntry(
            handle_key="k",
            kind=kind,
            handle={"job_id": "42", "name": "n"},
            state="RUNNING",
        )
    )


def test_cancel_node_jobs_cancels_each_live_entry(tmp_path):
    wf = AgentTeamWorkflow.__new__(AgentTeamWorkflow)  # bypass __init__; only workspace is needed
    wf.workspace = tmp_path
    _seed_job(tmp_path, "s1.g7", "slurm")
    rec = _RecordingKind()
    wf._cancel_node_jobs("s1.g7", kinds={"slurm": rec})
    assert rec.cancelled == [{"job_id": "42", "name": "n"}]


def test_cancel_node_jobs_missing_file_is_noop(tmp_path):
    wf = AgentTeamWorkflow.__new__(AgentTeamWorkflow)
    wf.workspace = tmp_path
    wf._cancel_node_jobs("nope", kinds={})  # no jobs.json, no crash
