#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Local, restartable vendor-promotion monitor. Remote writes require --publish."""

from __future__ import annotations

import argparse
import contextlib
import datetime
import fcntl
import json
import logging
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path

if __package__:
    from . import manage, metadata, promote
else:
    import manage
    import metadata
    import promote

_LOG = logging.getLogger("vendor-bot")
_STATE_FILE = "state.sqlite"
_STOP = threading.Event()


class Store:
    """Transactional runtime state; no authentication material is persisted."""

    def __init__(self, path: Path) -> None:
        self.connection = sqlite3.connect(path)
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )

    def get(self, key: str, default: object = None) -> object:
        row = self.connection.execute("SELECT value FROM state WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def put(self, key: str, value: object) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO state VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, json.dumps(value, sort_keys=True)),
            )

    def close(self) -> None:
        """Close SQLite before releasing the process lock."""
        self.connection.close()


def _pages(gh: promote.GitHub, endpoint: str) -> Iterator[dict]:
    separator = "&" if "?" in endpoint else "?"
    page = 1
    while True:
        values = gh.api(f"{endpoint}{separator}per_page=100&page={page}")
        if not isinstance(values, list):
            raise ValueError(f"Expected a list from {endpoint}.")
        yield from values
        if len(values) < 100:
            return
        page += 1


class Monitor:
    """One operator-configured vendor; run other vendors in separate workdirs."""

    def __init__(self, args: argparse.Namespace, gh: promote.GitHub, store: Store) -> None:
        self.args, self.gh, self.store = args, gh, store
        self.actor = gh.get("user")["login"]
        self.prefix = f"repos/{gh.consumer_repo}"
        self.marker = f"<!-- vendor-bot:{gh.vendor_name} -->"
        settings = {
            "consumer_repo": gh.consumer_repo,
            "vendor": gh.vendor_name,
            "upstream_repo": gh.upstream_repo,
            "base_branch": gh.base_branch,
            "upstream_branch": gh.upstream_branch,
            "canonical_repo": args.canonical_repo,
            "canonical_branch": args.canonical_branch,
            "fork": args.fork,
            "actor": self.actor,
            "repo": str(args.repo),
        }
        saved = store.get("settings")
        if saved is not None and saved != settings:
            raise ValueError(
                "Workdir belongs to a different operator/vendor configuration; choose another."
            )
        store.put("settings", settings)
        if store.get("since") is None:
            store.put(
                "since",
                args.since
                or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            )

    def _discover(self) -> list[int]:
        # Advance only after a complete scan, with overlap for edits arriving
        # during pagination. Tracked operations are reconciled independently.
        cutoff = (
            datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=5)
        ).strftime("%Y-%m-%dT%H:%M:%SZ")
        tracked = set(self.store.get("tracked", []))
        endpoints = [f"{self.prefix}/pulls?state=all&sort=updated&direction=desc"]
        if not self.store.get("bootstrapped", False):
            endpoints.insert(0, f"{self.prefix}/pulls?state=open")
        since = self.store.get("cursor", self.store.get("since"))
        for endpoint in endpoints:
            for pr in _pages(self.gh, endpoint):
                if "state=all" in endpoint and pr["updated_at"] < since:
                    break
                if pr["base"]["ref"] != self.gh.base_branch:
                    continue
                number = pr["number"]
                if pr["state"] == "closed" and not pr.get("merged_at"):
                    continue
                head = pr["head"]["sha"]
                key = f"candidate:{number}"
                cached = self.store.get(key, {})
                if cached.get("head") != head:
                    relevant = any(
                        item["filename"] == promote._LOCK
                        for item in _pages(self.gh, f"{self.prefix}/pulls/{number}/files")
                    )
                    cached = {"head": head, "relevant": relevant}
                    self.store.put(key, cached)
                if cached["relevant"]:
                    tracked.add(number)
        self.store.put("tracked", sorted(tracked))
        self.store.put("bootstrapped", True)
        self.store.put("cursor", cutoff)
        return sorted(tracked)

    def _feedback(self, pr: dict, problem: str | None) -> None:
        number = pr["number"]
        message = (
            f"@{pr['user']['login']} — vendor promotion needs author action:\n\n{problem}\n\n"
            "Add or correct this block in the **PR description** (replace the example PR URL):\n\n"
            f"````markdown\n{metadata.template(self.gh.vendor_name, self.gh.upstream_repo)}\n````\n\n"
            "List every upstream PR; a full commit_map is not required. For an unresolved commit, "
            "add a missing PR, update the source pin, or add under this vendor:\n\n"
            "```yaml\nresolutions:\n  FULL_SOURCE_SHA:\n    upstream_pr: https://github.com/OWNER/REPO/pull/123\n"
            "    reason: Explain the adaptation or why this PR owns the change.\n```\n\n"
            "If there is no paired upstream PR, use only `unpaired_reason: ...` for this vendor. "
            "Author assertions are retained during future refresh until separately verified."
            if problem
            else "Vendor promotion metadata and source attribution are valid. "
            "This is **not** a code-review approval. Promotion still requires the source PR to merge."
        )
        body = f"{self.marker}\n{message}"
        dismissal_notice = (
            "\n\nAn authorized reviewer must dismiss the automated metadata "
            "review; this account lacks dismissal permission."
        )
        if not self.args.publish:
            _LOG.info("PR #%s: %s", number, problem or "metadata valid (dry run)")
            return
        # Check immutable head and description immediately before feedback. A
        # concurrent author edit will be retried on the next poll.
        current = self.gh.get(f"{self.prefix}/pulls/{number}")
        if current["head"]["sha"] != pr["head"]["sha"] or current.get("body") != pr.get("body"):
            return
        comments = list(_pages(self.gh, f"{self.prefix}/issues/{number}/comments"))
        existing = [
            item
            for item in comments
            if item["user"]["login"] == self.actor and item.get("body", "").startswith(self.marker)
        ]
        if len(existing) > 1:
            raise ValueError("Multiple bot feedback comments; inspect before continuing.")
        if existing:
            if existing[0]["body"] != body and not (
                not problem and existing[0]["body"] == body + dismissal_notice
            ):
                self.gh.api(
                    f"{self.prefix}/issues/comments/{existing[0]['id']}",
                    method="PATCH",
                    payload={"body": body},
                )
        else:
            self.gh.api(
                f"{self.prefix}/issues/{number}/comments", method="POST", payload={"body": body}
            )
        if current["state"] != "open" or current["user"]["login"] == self.actor:
            return
        reviews = [
            item
            for item in _pages(self.gh, f"{self.prefix}/pulls/{number}/reviews")
            if item["user"]["login"] == self.actor
        ]
        automated = [
            item
            for item in reviews
            if item.get("body", "").startswith(self.marker) and item["state"] == "CHANGES_REQUESTED"
        ]
        manual = [
            item
            for item in reviews
            if not item.get("body", "").startswith(self.marker)
            and item["state"] in ("APPROVED", "CHANGES_REQUESTED")
        ]
        if problem and not automated and not manual:
            self.gh.api(
                f"{self.prefix}/pulls/{number}/reviews",
                method="POST",
                payload={
                    "event": "REQUEST_CHANGES",
                    "commit_id": pr["head"]["sha"],
                    "body": (
                        f"{self.marker}\n{problem}\n\nSee the vendor-bot comment for the required "
                        "description format and author resolution instructions."
                    ),
                },
            )
        elif not problem:
            for review in automated:
                # Never replace a metadata objection with APPROVE or dismiss a
                # human review. GitHub may require an authorized human dismissal.
                try:
                    self.gh.api(
                        f"{self.prefix}/pulls/{number}/reviews/{review['id']}/dismissals",
                        method="PUT",
                        payload={
                            "message": "Automated vendor metadata issue resolved; code review remains required.",
                            "event": "DISMISS",
                        },
                    )
                except RuntimeError as error:
                    if "HTTP 403" not in str(error):
                        raise
                    _LOG.warning(
                        "PR #%s: metadata is valid, but an authorized human must dismiss automated review %s.",
                        number,
                        review["id"],
                    )
                    if existing and existing[0]["body"] != body + dismissal_notice:
                        self.gh.api(
                            f"{self.prefix}/issues/comments/{existing[0]['id']}",
                            method="PATCH",
                            payload={"body": body + dismissal_notice},
                        )
                    return
            if existing and existing[0]["body"] == body + dismissal_notice:
                self.gh.api(
                    f"{self.prefix}/issues/comments/{existing[0]['id']}",
                    method="PATCH",
                    payload={"body": body},
                )

    def _inputs(self, pr: dict) -> tuple[manage.Vendor, manage.Vendor] | None:
        if pr.get("merged"):
            revision = promote._sha(pr["merge_commit_sha"])
            commit = self.gh.get(f"{self.prefix}/commits/{revision}")
            base = commit["parents"][0]["sha"]
        else:
            base, revision = promote._main_sha(self.gh), pr["head"]["sha"]
        before = promote._lock_document_at(self.gh, self.gh.consumer_repo, base)["vendors"]
        after = promote._lock_document_at(self.gh, self.gh.consumer_repo, revision)["vendors"]
        name = self.gh.vendor_name
        if name not in before or name not in after:
            return None  # Vendor creation/removal is not promotion.
        previous = manage._validate_vendor(name, before[name])
        reviewed = manage._validate_vendor(name, after[name])
        if previous.commit == reviewed.commit or not reviewed.branch:
            return None  # Includes lock-only promotions and tag-only updates.
        canonical = (
            promote._repo_name(previous.url).lower() == self.args.canonical_repo.lower()
            and previous.branch == self.args.canonical_branch
        )
        target_is_canonical = (
            promote._repo_name(reviewed.url).lower() == self.args.canonical_repo.lower()
            and reviewed.branch == self.args.canonical_branch
        )
        if target_is_canonical:
            return None  # Periodic refresh/direct canonical update: separate workflow.
        if not canonical:
            raise ValueError(
                "Finish the preceding vendor promotion, then rebase this source-update PR onto the canonical pin."
            )
        return previous, reviewed

    def inspect(self, number: int) -> None:
        """Reconcile one source-update PR and its deterministic promotion operation."""
        if self.store.get(f"complete:{number}"):
            return
        pr = self.gh.get(f"{self.prefix}/pulls/{number}")
        if pr["state"] == "closed" and not pr.get("merged"):
            return
        try:
            inputs = self._inputs(pr)
            if inputs is None:
                return
            previous, reviewed = inputs
            entry = metadata.parse(pr.get("body") or "", self.gh.vendor_name, self.gh.upstream_repo)
            signature = metadata.fingerprint(
                {
                    "previous": previous.to_mapping(),
                    "reviewed": reviewed.to_mapping(),
                    "metadata": entry,
                }
            )
            key = f"matching:{number}:{signature}"
            saved = self.store.get(key)
            cache = self.args.workdir / "sources.git"
            if saved is None:
                assignments, evidence = metadata.resolve(self.gh, previous, reviewed, entry, cache)
                saved = {"assignments": assignments, "evidence": evidence}
                self.store.put(key, saved)
        except (ValueError, manage.VendorError) as error:
            self._feedback(pr, str(error))
            self.store.put(f"problem:{number}", str(error))
            return
        self._feedback(pr, None)
        if not pr.get("merged"):
            return
        # Re-read mutable description/head before any promotion. Immutable lock
        # checks inside promote also run before each remote write.
        current = self.gh.get(f"{self.prefix}/pulls/{number}")
        if current.get("body") != pr.get("body") or current["head"]["sha"] != pr["head"]["sha"]:
            return
        arguments = argparse.Namespace(
            source_pr=str(number),
            canonical_repo=self.args.canonical_repo,
            canonical_branch=self.args.canonical_branch,
            upstream_pr=[str(item) for item in entry["upstream_prs"]],
            unpaired_reason=entry["unpaired_reason"],
            map_upstream=[f"{sha}={target}" for sha, target in saved["assignments"].items()],
            repo=self.args.repo,
            fork=self.args.fork,
            source_repo=cache if not entry["unpaired_reason"] and cache.exists() else None,
            worktree=self.args.workdir / f"promotion-{number}-{reviewed.commit[:12]}",
            publish=self.args.publish,
            auto_merge=True,
            wait=False,
            timeout=3600,
            legacy_prims_ts=False,
        )
        plan = promote._make_plan(arguments, self.gh)
        # Attribution evidence is versioned with the durable promotion record.
        # Reuse the original published record on retries rather than replacing it.
        if plan.completed_pr is None:
            plan.record["attribution"] = saved["evidence"]
            plan.record["contributor_metadata"] = {
                "source_head": pr["head"]["sha"],
                "metadata_digest": metadata.fingerprint(entry),
                "pr_author": pr["user"]["login"],
            }
        if not self.args.publish:
            _LOG.info("PR #%s: promotion ready (dry run): %s", number, plan.branch)
            return
        followup = promote._publish(arguments, self.gh, plan)
        self.store.put(f"promotion:{number}", followup["number"])
        latest = self.gh.get(f"{self.prefix}/pulls/{followup['number']}")
        if latest.get("merged"):
            promote._verify_remote_commit(
                self.gh, plan, self.gh.consumer_repo, latest["merge_commit_sha"]
            )
            self.store.put(f"complete:{number}", latest["html_url"])
        _LOG.info("PR #%s: promotion %s", number, latest["html_url"])

    def poll(self) -> None:
        """Process candidates serially; preserve failures for safe later retries."""
        for number in self._discover():
            if _STOP.is_set() or (self.args.workdir / "stop").exists():
                _STOP.set()
                break
            try:
                self.inspect(number)
            except (
                RuntimeError,
                ValueError,
                OSError,
                KeyError,
                subprocess.TimeoutExpired,
            ) as error:
                self.store.put(f"problem:{number}", str(error))
                _LOG.error("PR #%s paused: %s", number, error)


def _initialize_workdir(path: Path) -> None:
    if path.exists() and any(path.iterdir()) and not (path / _STATE_FILE).exists():
        raise ValueError("Choose an empty workdir or an existing vendor-bot workdir.")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.stat().st_uid != os.getuid():
        raise ValueError("Workdir must be owned by the current user.")
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--is-inside-work-tree"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        raise ValueError(
            "Bot workdir must be outside a Git worktree to keep runtime state private."
        )
    path.chmod(0o700)


@contextlib.contextmanager
def _locked(workdir: Path) -> Iterator[None]:
    with (workdir / "bot.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("A bot already owns this workdir.") from error
        yield


def _running(workdir: Path) -> bool:
    if not (workdir / "bot.lock").exists():
        return False
    try:
        with _locked(workdir):
            return False
    except ValueError:
        return True


def _serve(args: argparse.Namespace) -> None:
    with _locked(args.workdir):
        (args.workdir / "stop").unlink(missing_ok=True)
        with contextlib.closing(Store(args.workdir / _STATE_FILE)) as store:
            store.put("process", {"pid": None, "publish": args.publish})
            gh = promote.GitHub(
                args.consumer_repo,
                args.vendor,
                args.upstream_repo,
                args.base_branch,
                args.upstream_branch,
            )
            monitor = Monitor(args, gh, store)
            store.put("process", {"pid": os.getpid(), "publish": args.publish})
            _LOG.info(
                "Started vendor %s as %s; remote writes %s",
                args.vendor,
                monitor.actor,
                args.publish,
            )
            failures = 0
            while not _STOP.is_set():
                try:
                    monitor.poll()
                    failures = 0
                    store.put("last_poll", datetime.datetime.now(datetime.timezone.utc).isoformat())
                except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as error:
                    failures += 1
                    _LOG.error("Poll paused: %s", error)
                    if args.once:
                        raise
                if args.once:
                    break
                deadline = time.monotonic() + min(args.interval * 2 ** min(failures, 5), 3600)
                while time.monotonic() < deadline and not _STOP.wait(1):
                    if (args.workdir / "stop").exists():
                        _STOP.set()
                        break
            store.put("process", {"pid": None, "publish": args.publish})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Monitor in foreground, or detach with --daemon.")
    run.add_argument("--workdir", type=Path, required=True)
    run.add_argument("--vendor", required=True)
    run.add_argument("--upstream-repo", required=True)
    run.add_argument("--canonical-repo", required=True)
    run.add_argument("--canonical-branch", required=True)
    run.add_argument("--fork", required=True, help="Consumer publishing fork OWNER/REPO.")
    run.add_argument("--consumer-repo", default="NVIDIA/TensorRT-LLM")
    run.add_argument("--base-branch", default="main")
    run.add_argument("--upstream-branch", default="main")
    run.add_argument(
        "--repo",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Trusted consumer checkout used for isolated promotion worktrees.",
    )
    run.add_argument(
        "--interval", type=int, default=120, help="Polling interval, seconds (minimum 30)."
    )
    run.add_argument(
        "--since",
        help="First-run merged-PR cutoff, UTC YYYY-MM-DDTHH:MM:SSZ; default startup time.",
    )
    run.add_argument(
        "--publish",
        action="store_true",
        help="Allow feedback/reviews, pushes, PR creation, CI skip, and squash auto-merge.",
    )
    mode = run.add_mutually_exclusive_group()
    mode.add_argument("--daemon", action="store_true")
    mode.add_argument(
        "--once",
        action="store_true",
        help="One reconciliation pass, useful for testing or a scheduler.",
    )
    for name in ("status", "stop"):
        command = commands.add_parser(name)
        command.add_argument("--workdir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Launch/control a POSIX daemon without storing credentials or requiring systemd."""
    args = _parser().parse_args(argv)
    args.workdir = args.workdir.expanduser().resolve()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        if args.command != "run":
            active = _running(args.workdir)
            if args.command == "stop" and active:
                (args.workdir / "stop").touch(mode=0o600)
            print(
                json.dumps({"running": active, "stop_requested": args.command == "stop" and active})
            )
            return 0
        if args.interval < 30:
            raise ValueError("--interval must be at least 30 seconds.")
        if not manage._NAME_PATTERN.fullmatch(args.vendor):
            raise ValueError("Invalid vendor key.")
        for name in (args.consumer_repo, args.upstream_repo, args.canonical_repo, args.fork):
            promote._repo_name(f"https://github.com/{name}")
        for branch in (args.canonical_branch, args.base_branch, args.upstream_branch):
            promote._run(["git", "check-ref-format", f"refs/heads/{branch}"])
        if args.since:
            datetime.datetime.strptime(args.since, "%Y-%m-%dT%H:%M:%SZ")
        args.repo = args.repo.expanduser().resolve()
        _initialize_workdir(args.workdir)
        if args.daemon:
            # Initialize the state marker before launching so two competing
            # starts can both reach the authoritative child-held flock.
            with contextlib.closing(Store(args.workdir / _STATE_FILE)):
                pass
            if _running(args.workdir):
                raise ValueError("A bot already owns this workdir.")
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                *list(argv if argv is not None else sys.argv[1:]),
            ]
            command.remove("--daemon")
            with (args.workdir / "bot.log").open("ab", buffering=0) as log:
                child = subprocess.Popen(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=log,
                    start_new_session=True,
                    close_fds=True,
                    umask=0o077,
                )
            for _ in range(100):
                if child.poll() is not None:
                    raise ValueError(
                        f"Daemon exited during startup; inspect {args.workdir / 'bot.log'}."
                    )
                with contextlib.closing(Store(args.workdir / _STATE_FILE)) as state:
                    process = state.get("process", {})
                if _running(args.workdir) and process.get("pid") == child.pid:
                    print(f"Started daemon PID {child.pid}; log: {args.workdir / 'bot.log'}")
                    return 0
                time.sleep(0.1)
            raise ValueError(
                "Daemon startup not confirmed; inspect log and status before retrying."
            )
        signal.signal(signal.SIGTERM, lambda *_: _STOP.set())
        signal.signal(signal.SIGINT, lambda *_: _STOP.set())
        _STOP.clear()
        _serve(args)
        return 0
    except (ValueError, RuntimeError, OSError, sqlite3.Error, subprocess.TimeoutExpired) as error:
        _LOG.error("Bot stopped: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
