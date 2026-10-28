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


def test_local_cancel_stops_the_commands_a_detached_job_started():
    import subprocess
    import time

    proc = subprocess.Popen(["bash", "-c", "sleep 30; true"], start_new_session=True)
    try:
        for _ in range(50):
            children = subprocess.run(
                ["pgrep", "-P", str(proc.pid)], capture_output=True, text=True
            ).stdout.split()
            if children:
                break
            time.sleep(0.1)
        (child,) = map(int, children)

        LocalKind().cancel({"pid": proc.pid})

        proc.wait(timeout=10)
        for _ in range(50):
            try:
                os.kill(child, 0)
            except ProcessLookupError:
                break
            time.sleep(0.1)
        else:
            raise AssertionError("the job's command outlived the cancel")
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)
