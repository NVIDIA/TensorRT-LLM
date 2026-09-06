import os
import signal

from agent_flow.jobs.kinds import KINDS, LocalKind, SlurmKind


def _recording_run():
    calls = []

    def run(argv):
        calls.append(argv)
        return (0, "")

    return run, calls


def test_slurm_cancel_calls_scancel_with_job_id():
    run, calls = _recording_run()
    SlurmKind(run=run).cancel({"job_id": "561654", "name": "n"})
    assert calls == [["scancel", "561654"]]


def test_kinds_registry_has_slurm_and_local():
    assert set(KINDS) == {"slurm", "local"}


def test_local_cancel_terminates_a_live_process():
    proc = __import__("subprocess").Popen(["sleep", "30"])
    try:
        LocalKind().cancel({"pid": proc.pid})
        # SIGTERM was delivered; the process should exit promptly.
        assert proc.wait(timeout=10) != 0 or proc.returncode is not None
    finally:
        if proc.poll() is None:
            os.kill(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)


def test_local_cancel_on_dead_pid_is_swallowed():
    # A pid that is not a live process must not raise (best-effort cancel).
    LocalKind().cancel({"pid": 2**31 - 1})
