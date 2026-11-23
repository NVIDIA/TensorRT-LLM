# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Supervise one launcher payload without importing TensorRT-LLM."""

from __future__ import annotations

import argparse
import ctypes
import errno
import os
import signal
import subprocess  # nosec B404
import sys
import time
from pathlib import Path
from types import FrameType

_POLL_INTERVAL = 0.05


def _signal_group(pgid: int, signum: int) -> None:
    try:
        os.killpg(pgid, signum)
    except ProcessLookupError:
        pass
    except PermissionError:
        # Darwin reports EPERM when the retained group contains only zombies.
        if sys.platform != "darwin" or _group_has_live_members(pgid):
            raise


def _group_has_live_members(pgid: int) -> bool:
    """Exclude zombies while the unreaped payload pins its process-group ID."""
    if sys.platform.startswith("linux"):
        try:
            leader_seen = False
            with os.scandir("/proc") as entries:
                for entry in entries:
                    if not entry.name.isdecimal():
                        continue
                    try:
                        if os.getpgid(int(entry.name)) != pgid:
                            continue
                        stat = Path(entry.path, "stat").read_text()
                    except (FileNotFoundError, ProcessLookupError):
                        continue
                    leader_seen |= int(entry.name) == pgid
                    fields = stat.rsplit(")", 1)[1].split()
                    if fields[0] not in ("Z", "X"):
                        return True
            return not leader_seen
        except (OSError, IndexError):
            # Unknown membership must not shorten the cleanup grace.
            return True

    try:
        # Use the launcher's PATH with fixed arguments and no shell.
        result = subprocess.run(  # nosec B607
            ["ps", "-axo", "pid=,pgid=,stat="],
            capture_output=True,
            text=True,
            check=True,
            timeout=1,
        )
        leader_seen = False
        for line in result.stdout.splitlines():
            pid, group, state = line.split()
            leader_seen |= int(pid) == pgid
            if int(group) == pgid and not state.startswith(("Z", "X")):
                return True
        return not leader_seen
    except (OSError, ValueError, subprocess.SubprocessError):
        return True


def _cleanup_group(pgid: int, grace: int, signum: int) -> None:
    _signal_group(pgid, signum)
    deadline = time.monotonic() + grace
    while _group_has_live_members(pgid):
        if time.monotonic() >= deadline:
            break
        time.sleep(_POLL_INTERVAL)
    _signal_group(pgid, signal.SIGKILL)


def _arm_parent_death_signal() -> None:
    if not sys.platform.startswith("linux"):
        return
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    prctl.restype = ctypes.c_int
    # getppid() polling remains active if a container denies this notification.
    if prctl(1, signal.SIGTERM, 0, 0, 0) != 0:  # PR_SET_PDEATHSIG
        error = ctypes.get_errno()
        print(
            f"MPI process guard: parent-death notification unavailable: {os.strerror(error)}",
            file=sys.stderr,
            flush=True,
        )


def _exec_payload(command: list[str], gate_read: int, gate_write: int, role: str) -> None:
    os.close(gate_write)
    try:
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGPIPE):
            signal.signal(signum, signal.SIG_DFL)
        os.setpgid(0, 0)
        if os.read(gate_read, 1) != b"1":
            os._exit(125)
        os.close(gate_read)
        # The payload is the launcher's explicit argv, not a shell command string.
        os.execvp(command[0], command)  # nosec B606
    except OSError as error:
        print(f"MPI {role} payload could not start: {error}", file=sys.stderr, flush=True)
        os._exit(127 if error.errno == errno.ENOENT else 126)


def _child_finished(pid: int) -> bool:
    # Keep the zombie until group cleanup completes, preventing PGID reuse.
    result = os.waitid(os.P_PID, pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    return result is not None and result.si_pid != 0


def _publish(path: Path, text: str) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _remove_ownerless_files(ready_file: Path, go_file: Path) -> None:
    for path in (ready_file, go_file, Path(f"{ready_file}.done"), Path(f"{ready_file}.reap")):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    try:
        ready_file.parent.rmdir()
    except OSError:
        pass


def _run(
    parent_pid: int,
    grace: int,
    ready_file: Path | None,
    go_file: Path | None,
    role: str,
    command: list[str],
) -> int:
    stop_signal = 0

    def request_stop(signum: int, _frame: FrameType | None) -> None:
        nonlocal stop_signal
        if not stop_signal:
            stop_signal = signum

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    if os.getppid() != parent_pid:
        return 143
    _arm_parent_death_signal()
    if stop_signal or os.getppid() != parent_pid:
        return 128 + (stop_signal or signal.SIGTERM)

    gate_read, gate_write = os.pipe()
    try:
        child_pid = os.fork()
    except OSError:
        os.close(gate_read)
        os.close(gate_write)
        raise
    if child_pid == 0:
        _exec_payload(command, gate_read, gate_write, role)
        os._exit(126)

    os.close(gate_read)
    group_ready = False
    ready_published = False
    completion_failed = False
    try:
        # Parent and child both establish the group before either can signal it.
        os.setpgid(child_pid, child_pid)
        group_ready = True
        if ready_file is not None and not stop_signal and os.getppid() == parent_pid:
            _publish(ready_file, f"{child_pid}\n")
            ready_published = True

        while True:
            if _child_finished(child_pid):
                break
            if stop_signal or os.getppid() != parent_pid:
                break
            if gate_write != -1 and (go_file is None or go_file.is_file()):
                # Bash registers the group before release. An autonomous owner
                # tracks only this guard and leaves all group cleanup to it.
                if os.getppid() != parent_pid:
                    break
                os.write(gate_write, b"1")
                os.close(gate_write)
                gate_write = -1
            time.sleep(_POLL_INTERVAL)
    finally:
        if gate_write != -1:
            os.close(gate_write)
        try:
            if not group_ready:
                try:
                    os.kill(child_pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            # The payload gate stays closed if parent group setup fails.
            _cleanup_group(child_pid, grace if group_ready else 0, stop_signal or signal.SIGTERM)
        finally:
            _signal_group(child_pid, signal.SIGKILL)
            try:
                if ready_published and os.getppid() == parent_pid:
                    # Bash drops its saved PGID before acknowledging completion.
                    # Until then, the unreaped leader prevents stale group kills.
                    try:
                        _publish(Path(f"{ready_file}.done"), "done\n")
                    except OSError as error:
                        completion_failed = True
                        try:
                            print(
                                f"MPI {role} guard could not publish completion: {error}",
                                file=sys.stderr,
                                flush=True,
                            )
                        except OSError:
                            pass
                        # Wake Bash's EXIT cleanup instead of leaving it waiting
                        # for a completion marker that could not be published.
                        if os.getppid() == parent_pid:
                            try:
                                os.kill(parent_pid, signal.SIGTERM)
                            except ProcessLookupError:
                                pass
                        # Keep the registered group pinned even if metadata or
                        # stderr is unavailable. EXIT cleanup can still retire
                        # it and acknowledge, or owner death releases this wait.
                    reap_file = Path(f"{ready_file}.reap")
                    while os.getppid() == parent_pid and not reap_file.is_file():
                        time.sleep(_POLL_INTERVAL)
            finally:
                _, status = os.waitpid(child_pid, 0)
                if ready_file is not None and go_file is not None and os.getppid() != parent_pid:
                    _remove_ownerless_files(ready_file, go_file)

    code = os.waitstatus_to_exitcode(status)
    if completion_failed and code == 0:
        return 1
    return code if code >= 0 else 128 - code


def _positive_integer(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-pid", type=_positive_integer, required=True)
    parser.add_argument("--term-grace", type=_positive_integer, required=True)
    parser.add_argument("--ready-file", type=Path)
    parser.add_argument("--go-file", type=Path)
    parser.add_argument(
        "--autonomous",
        action="store_true",
        help="Own cleanup without publishing process-group registration files",
    )
    parser.add_argument("--role", choices=("server", "task", "stop", "proxy"), required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.autonomous:
        if args.ready_file is not None or args.go_file is not None:
            parser.error("--autonomous cannot use --ready-file or --go-file")
    elif args.ready_file is None or args.go_file is None:
        parser.error("--ready-file and --go-file are required without --autonomous")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a payload command is required after --")
    try:
        return _run(
            args.parent_pid, args.term_grace, args.ready_file, args.go_file, args.role, command
        )
    except (OSError, RuntimeError) as error:
        print(f"MPI {args.role} process guard failed: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(_main())
