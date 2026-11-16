# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Who runs a command: a job in the run's spool, or (outside a run) a subprocess here.

``environment.py`` builds the command line; :func:`select_command_executor` maps the resolved
environment onto who runs it. Inside a run that is always the run's one spool
(``spoolctl.py``'s protocol): the watchers of its holders are the only ledger of devices, on one
host as on a cluster. :class:`SpoolCommandExecutor` is the gates' policy over the spool:

- **A job states its size and its wall**: ``gpus`` or ``nodes``, and the command's deadline.
  Nothing is sized by a default, so a caller that names no size is refused here.
- **The deadline counts from the job's start**: time queued behind other work says nothing about
  the command. A queued wait is bounded by the pool's liveness instead, so a dead session
  cannot hang its caller; a holder still in the queue is not a dead session.
- **A command given up on is withdrawn** (``cancel/<name>``), so a watcher that comes back does
  not run work nobody waits for. A spool whose heartbeat is stale and that no controller is
  getting a holder for gets nothing queued.
- **A lost job is not a failed command**: the scheduler re-runs it once; after that the result
  says so in ``infrastructure`` and is no verdict. Nothing above the scheduler runs it again.
- **Nor is a cancelled job, or one whose stated ranks reported no origin**: its
  ``infrastructure`` begins with :data:`CANCELLED`, or says which rank was not heard from.
- **A long wait is said**: a job waited on for ``spoolctl.WAIT_NOTE_S`` is passed to ``note``
  (the run's journal), whether queued or running and since when, and why the watcher holds it.
"""

from __future__ import annotations

import contextlib
import hashlib
import itertools
import json
import os
import shlex
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping

from . import pintree, spoolctl
from .commands import (
    TIMEOUT_RETURNCODE,
    CommandExecutor,
    CommandResult,
    CommandSpec,
    LocalCommandExecutor,
)
from .environment import RunEnvironment
from .scheduler import (
    CANCELLED_REASON,
    NO_ORIGIN_REASON,
    SESSION_LOSS_REASONS,
    STALLED_REASON,
    WATCHER_STALE_S,
    SpoolPaths,
    holder_coming,
    watcher_age,
)

#: How a command's ``infrastructure`` begins when its job was withdrawn through the spool
#: (by the operator, or by a caller that gave up on it): no verdict, and not a lost session.
CANCELLED = "the job was cancelled through the spool"

#: The classes a job queues at; the spool scheduler starts a waiting job of a
#: higher one first. Checks on the way to the first unit's accuracy: its boot and benchmark.
ACCURACY = 2
#: Every other check: the feature units, the variants, the proofs, delivery.
FEATURE = 1

_local = threading.local()


@contextlib.contextmanager
def job_priority(priority: int) -> Iterator[None]:
    """Jobs submitted in this thread, inside the block, queue at ``priority``.

    Set where the job's purpose is known, read where it is submitted (:func:`current_job_priority`),
    so the code between needs no argument for it.
    """
    previous = getattr(_local, "priority", None)
    _local.priority = priority
    try:
        yield
    finally:
        _local.priority = previous


def current_job_priority() -> int:
    """The class this thread's jobs queue at; the scheduler's 0 by default (a worker's own jobs)."""
    priority = getattr(_local, "priority", None)
    return 0 if priority is None else int(priority)


#: Under the spool: for each request a process has queued a job for and not yet read, the
#: job's name (``<request>.json``).
REQUESTS_DIRNAME = "requests"
#: A file a command names is part of its request when it is at most this large, and at most
#: this many of them are read.
REQUEST_FILE_BYTES = 1 << 20
REQUEST_FILES = 32


def left_in_spool(spool: str | Path) -> list[str]:
    """The jobs earlier processes queued on ``spool`` and never read, by name.

    Each is still queued, running, or finished with its receipt unread. The request that
    made one takes it when it is made again (:meth:`SpoolCommandExecutor.execute`).
    """
    root, paths, found = Path(spool), SpoolPaths(spool), []
    for path in sorted((root / REQUESTS_DIRNAME).glob("*.json")):
        try:
            name = str(json.loads(path.read_text(encoding="utf-8"))["name"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if (
            (paths.pending / f"{name}.sh").exists()
            or (paths.running / f"{name}.sh").exists()
            or paths.receipt(name).exists()
        ):
            found.append(name)
    return found


class SpoolCommandExecutor(CommandExecutor):
    """Submit through a spool directory and block on the receipt."""

    def __init__(
        self,
        spool: str | Path,
        *,
        gpus: str | int | None = None,
        nodes: int = 0,
        poll_s: float = 3.0,
        name_prefix: str = "gate",
        priority: int = 0,
        tree: str | Path | None = None,
        pin: str | Path | None = None,
        ranks: str = "",
        on_each_node: Callable[[str, int], str] | None = None,
        once: bool = False,
        note: Callable[[str], None] | None = None,
        python: str = "python3",
    ) -> None:
        self.root = Path(spool)
        #: The interpreter on the nodes: what a step of a job of several nodes runs its pin with.
        self.python = python
        self.poll_s = poll_s
        self.name_prefix = name_prefix
        if nodes < 0:
            raise ValueError("nodes must not be negative")
        if gpus is None and not nodes:
            raise ValueError(
                "a spool job states its size: give gpus (a count, 0 or 'all') or nodes"
            )
        #: Whole nodes the job holds; 0 when it states devices instead (on one node).
        self.nodes = nodes if nodes > 1 or gpus is None else 0
        #: What each job asks for when it states devices; whole nodes carry no count.
        self.gpus = None if self.nodes else str(gpus)
        #: The class each job queues at (:data:`ACCURACY`, :data:`FEATURE`, or 0).
        self.priority = priority
        #: The tree a job runs (its working directory and first ``PYTHONPATH`` entry). A job
        #: runs a pin: a pinned tree (``Integration.at``) names its pin, any other tree is
        #: pinned as each job is queued (``spoolctl.tree_header``). The scheduler refuses a
        #: result that imported the package from anywhere else, or found other libraries.
        self.tree = str(Path(tree).resolve()) if tree else None
        #: The pin every job runs, given as its file: for a caller that holds a pin and no
        #: tree of it here (a check of one commit, whose tree is made on the job's node).
        self.pin = str(pin) if pin else None
        #: The rank worlds an engine job must hear an origin from (``<role>:<size>,...``).
        self.ranks = ranks
        #: For a job of several nodes: the script that runs a command once on each of them
        #: (``RunEnvironment.on_each_node``). The lead runs a multi-node job's script once.
        self.on_each_node = on_each_node
        #: The command is itself the script the lead runs once: it starts its own steps
        #: across the reserved nodes (a deployment's session).
        self.once = once
        if self.nodes > 1 and on_each_node is None and not once:
            raise ValueError("a job of several nodes needs the environment's step line")
        #: Allowance past a job's own deadline for the session to stop it.
        self.grace_s = 180.0
        #: Where a long wait on a job is said (``spoolctl.wait_for_job``); nowhere when ``None``.
        self.note = note
        self._seq = itertools.count(1)

    def _name(self) -> str:
        """Unique across processes and threads: two gates in one millisecond must not collide.

        The pid alone does not separate processes: each sandboxed command has its own pid
        namespace, where two gates started together share a small pid.
        """
        return (
            f"{self.name_prefix}-{int(time.time() * 1000)}-{os.getpid()}-"
            f"{uuid.uuid4().hex[:6]}-{next(self._seq)}"
        )

    def _withdraw(self, root: Path, name: str) -> str:
        try:
            flag = spoolctl.cancel_job(root, name)
        except OSError as exc:
            return f"; withdrawing it failed ({exc}), so a session draining {root} may still run it"
        return f"; withdrawn through {flag}"

    def _log(self, root: Path, name: str) -> str:
        log = SpoolPaths(root).log(name)
        return log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""

    def _tree_fields(self, name: str, cwd: str | None) -> tuple[dict[str, str], str | None]:
        """The header field for the tree a job runs, and the directory its script still enters.

        A job runs a pin: the one this executor was given, else the tree's, taken now
        (``spoolctl.tree_header``). The pin's tree is then the job's working directory, so a
        command that names the tree itself as its directory enters nothing, and one that
        names a directory inside the tree enters that directory of the pin's tree, by its
        path from the tree's top: never the tree that moves on. A command that names a
        directory outside the tree, or a tree nothing can pin, states no tree, as before.
        """
        inside: str | None = None  # the command's directory, from the tree's top
        if self.tree and not cwd:
            inside = ""
        elif self.tree and cwd:
            real = os.path.realpath(cwd)
            if real == self.tree:
                inside = ""
            elif real.startswith(self.tree + os.sep):
                inside = os.path.relpath(real, self.tree)
        if self.pin:
            fields = {"pin": self.pin}
        elif inside is not None:
            fields = spoolctl.tree_header(self.tree, name)
            if cwd and "pin" not in fields:
                fields = {}
        else:
            fields = {}
        if fields.get("pin") and inside is not None:
            return fields, (inside or None)
        return fields, cwd

    # ----------------------------------------------- one request, one job
    def _request(self, header: Mapping[str, Any], spec: CommandSpec, cwd: str | None) -> str:
        """What a job would do, as one digest: the identity a job in the spool is adopted by.

        Its size, owner, pin, ranks, directory, command and environment, and the content
        of each small file its command names (a session's spec, a test selection): a job
        that would read other inputs is another request. Not its name, nor its token.
        """
        files: dict[str, str] = {}
        try:
            words = shlex.split(spec.rendered)
        except ValueError:
            words = spec.rendered.split()
        for word in words:
            # As a shell writes a path: after an option's `=`, before a `;` or a redirect.
            word = (word.rpartition("=")[2] if word.startswith("-") else word).strip(";&|()<>")
            if len(files) >= REQUEST_FILES or not os.path.isabs(word):
                continue
            try:
                if os.path.isfile(word) and os.path.getsize(word) <= REQUEST_FILE_BYTES:
                    files[word] = hashlib.sha256(Path(word).read_bytes()).hexdigest()
            except OSError:
                continue
        said = {k: v for k, v in header.items() if k != "token" and v is not None}
        payload = [said, spec.rendered, cwd, dict(spec.env or {}), files]
        text = json.dumps(payload, sort_keys=True, default=str)
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]

    def _adoptable(self, root: Path, request: str) -> tuple[str, str] | None:
        """The job an earlier process queued for ``request`` and never read: its name and token.

        A process that queues a job leaves its name under the request until it has read the
        job's result. One that dies meanwhile leaves the job in the spool with nobody
        waiting: queued, running, or finished with its receipt unread. The next process to
        make the same request takes that job, and queues no second one. A name whose job
        is nowhere in the spool any more is forgotten.
        """
        path = root / REQUESTS_DIRNAME / f"{request}.json"
        try:
            held = json.loads(path.read_text(encoding="utf-8"))
            name, token = str(held["name"]), str(held["token"])
        except (OSError, ValueError, KeyError, TypeError):
            return None
        paths = SpoolPaths(root)
        there = (
            (paths.pending / f"{name}.sh").exists()
            or (paths.running / f"{name}.sh").exists()
            or paths.receipt(name).exists()
        )
        if there:
            return name, token
        path.unlink(missing_ok=True)
        return None

    def _remember(self, root: Path, request: str, name: str, token: str) -> None:
        """Leave ``name`` under ``request`` until its result is read (:meth:`_adoptable`)."""
        path = root / REQUESTS_DIRNAME / f"{request}.json"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"name": name, "token": token, "at": time.time()}))
        except OSError:
            pass  # a job nobody can adopt is queued again by a later process, as before

    def execute(self, spec: CommandSpec) -> CommandResult:
        name = self._name()
        token = uuid.uuid4().hex
        submitted = time.time()
        root = self.root
        wall = int(spec.timeout_s) if spec.timeout_s and spec.timeout_s > 0 else 0
        if not wall and self.gpus != "0":
            raise ValueError(
                f"a job that holds devices states a wall: give the command of {name} a timeout"
            )
        tree, cwd = self._tree_fields(name, spec.cwd)
        header = {
            "gpus": self.gpus,
            "nodes": self.nodes or None,
            "wall": wall,
            "owner": self.name_prefix,
            "token": token,
            **tree,
            "priority": self.priority or None,
            "ranks": self.ranks or None,
        }
        request = self._request(header, spec, cwd)
        adopted = self._adoptable(root, request)
        if adopted is not None:
            # A job of this very request is in the spool, left by a process that is gone
            # or still waiting for it: it is waited for here, and none is queued beside it.
            name, token = adopted
            if self.note is not None:
                self.note(
                    f"{name} on {root} was queued for this request before: it is adopted, "
                    "not queued a second time"
                )
            return self._result(root, name, token, spec, submitted, request)
        command = spec.rendered
        if self.nodes > 1 and not self.once:
            # The lead runs this script once; the command itself runs once on each node.
            inner = SpoolPaths(root).jobs / name / "node.sh"
            inner.parent.mkdir(parents=True, exist_ok=True)
            on_node = f"bash {shlex.quote(str(inner))}"
            if tree.get("pin"):
                # Each node makes (or finds) the pin's tree and checks its libraries itself.
                record = SpoolPaths(root).tree_log(name)
                step = pintree.step_argv(tree["pin"], name, record, self.python)
                on_node = f"{shlex.join(step)} {on_node}"
            else:
                cwd = cwd or self.tree
            inner.write_text(spoolctl.render_job(command, cwd=cwd, env=spec.env), encoding="utf-8")
            command = self.on_each_node(on_node, self.nodes)
            script = spoolctl.render_job(command, header=header)
        else:
            script = spoolctl.render_job(command, header=header, cwd=cwd, env=spec.env)
        age = watcher_age(root, submitted)
        dead = age is not None and age > WATCHER_STALE_S and not holder_coming(root, submitted)
        if not dead:
            root.mkdir(parents=True, exist_ok=True)
            spoolctl.queue_job(root, name, script, owner=self.name_prefix, require_watcher=False)
            self._remember(root, request, name, token)
        else:
            reason = (
                f"the session watching {root} stopped heart-beating {age:.0f}s ago; "
                f"nothing was queued"
            )
            return CommandResult(
                returncode=TIMEOUT_RETURNCODE,
                stdout=f"=== spool executor: {reason} ===\n",
                timed_out=True,
                infrastructure=reason,
            )
        return self._result(root, name, token, spec, submitted, request)

    def _result(
        self, root: Path, name: str, token: str, spec: CommandSpec, submitted: float, request: str
    ) -> CommandResult:
        """Wait for job ``name`` and read its result; the request then has no job to adopt.

        A wait that is cut (the process is stopped or dies) leaves the request as it is:
        the job is still in the spool, and the next process to ask takes it.
        """
        result = self._wait(root, name, token, spec, submitted)
        # Read, or given up and withdrawn: the next process to ask queues its own.
        (root / REQUESTS_DIRNAME / f"{request}.json").unlink(missing_ok=True)
        return result

    def _wait(
        self, root: Path, name: str, token: str, spec: CommandSpec, submitted: float
    ) -> CommandResult:
        try:
            record = spoolctl.wait_for_job(
                root,
                name,
                token=token,
                poll_s=self.poll_s,
                timeout_s=spec.timeout_s,
                grace_s=self.grace_s,
                unwatched_s=spec.timeout_s,
                submitted_at=submitted,
                note=self.note,
            )
        except spoolctl.SpoolError as exc:
            reason = str(exc) + self._withdraw(root, name)
            lost = exc.exit_code == spoolctl.EXIT_NO_SESSION
            return CommandResult(
                returncode=TIMEOUT_RETURNCODE,
                stdout=f"{self._log(root, name)}\n=== spool executor: {reason} ===\n",
                timed_out=True,
                infrastructure=reason if lost else "",
            )
        lost = cancelled = stalled = ""
        if record.get("reason") == STALLED_REASON:
            stalled = str(record.get("detail") or "no progress after a failure line")
        if record.get("reason") in SESSION_LOSS_REASONS:
            lost = (
                f"the session that ran it stopped heart-beating and the job was closed "
                f"out as {record['reason']}: {record.get('detail') or 'no detail recorded'}"
            )
        elif record.get("reason") == NO_ORIGIN_REASON:
            lost = f"no result: {record.get('detail') or 'a stated rank reported no origin'}"
        elif record.get("reason") == CANCELLED_REASON:
            cancelled = f"{CANCELLED}: {record.get('detail') or 'no detail recorded'}"
        return CommandResult(
            returncode=int(record["rc"]),
            stdout=self._log(root, name),
            # The scheduler stops a job at its wall limit or when it stops making
            # progress, and a job whose session died under it measured nothing.
            timed_out=bool(record.get("timed_out")) or bool(lost),
            # A cancelled job measured nothing either, whatever its exit code says.
            infrastructure=lost or cancelled,
            stalled=stalled,
        )


def select_command_executor(
    environment: RunEnvironment,
    *,
    gpus: str | int | None = None,
    nodes: int = 0,
    name_prefix: str = "gate",
    tree: Path | str | None = None,
    pin: Path | str | None = None,
    priority: int = 0,
    ranks: str = "",
    note: Callable[[str], None] | None = None,
    once: bool = False,
) -> CommandExecutor:
    """The executor this run's commands go through: the run's one spool.

    The caller states the size of each job: ``gpus`` (a count, 0 for a job that holds no device,
    or ``all``) or ``nodes`` (whole nodes; more than one is run once by a holder's lead, the
    command once on each node through the environment's step line; with ``once`` the command
    is itself the script the lead runs, and starts its own steps). On the spool, ``tree`` (the
    tree under test, pinned) is every job's working directory and first ``PYTHONPATH`` entry;
    a caller that holds the pin itself gives its file as ``pin``. A long wait on a job is said
    to ``note``.

    Only an environment that names no spool runs the command in this process: ``kind: local``
    resolved outside a run (a gate started by hand), where no watcher holds the devices.
    """
    if environment.spool:
        return SpoolCommandExecutor(
            environment.spool,
            name_prefix=name_prefix,
            tree=tree,
            pin=pin,
            gpus=gpus,
            nodes=nodes,
            priority=priority,
            ranks=ranks,
            on_each_node=environment.on_each_node if nodes > 1 and not once else None,
            once=once,
            note=note,
            python=environment.python,
        )
    if environment.kind != "local":
        raise ValueError(f"environment.kind {environment.kind!r} names no spool to submit to")
    if nodes > 1:
        raise ValueError(
            "a command on several nodes needs a holder of several nodes, not this host"
        )
    return LocalCommandExecutor()


def shell_runner_for(executor: CommandExecutor, *, default_timeout: int = 1800) -> Any:
    """``executor`` as the ``(command, env, timeout) -> CommandResult`` callable gates take."""

    def run(
        command: str,
        env: Mapping[str, str] | None = None,
        timeout: int = default_timeout,
    ) -> CommandResult:
        return executor.execute(CommandSpec(argv=command, env=env, timeout_s=timeout))

    return run


__all__ = [
    "ACCURACY",
    "CANCELLED",
    "FEATURE",
    "SESSION_LOSS_REASONS",
    "SpoolCommandExecutor",
    "current_job_priority",
    "job_priority",
    "select_command_executor",
    "shell_runner_for",
]
