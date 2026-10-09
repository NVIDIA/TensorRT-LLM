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
import re
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
_HEARTBEAT_SECONDS = 30


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _summary(error: object) -> str:
    """Keep diagnostics single-line; exclude YAML excerpts and authentication secrets."""
    text = str(error).splitlines()[0] if str(error) else type(error).__name__
    for name in ("GH_TOKEN", "GITHUB_TOKEN", "GH_ENTERPRISE_TOKEN", "GITHUB_ENTERPRISE_TOKEN"):
        if os.environ.get(name):
            text = text.replace(os.environ[name], "[REDACTED]")
    text = re.sub(r"https?://[^\s/]+@", "https://[REDACTED]@", text)
    text = re.sub(r"\b(?:gh[pousr]_[A-Za-z0-9_]+|github_pat_[A-Za-z0-9_]+)", "[REDACTED]", text)
    text = re.sub(r"(?i)(authorization\s*[:=]\s*).+", r"\1[REDACTED]", text)
    return text[:500]


class Store:
    """Disposable reconciliation cache; Git/GitHub own promotion provenance and state."""

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


class Progress:
    """Main-thread status checkpoints with an independent, log-only heartbeat."""

    def __init__(self, store: Store) -> None:
        self.store = store
        previous = store.get("health", {})
        self.state = {
            "operation": None,
            "progress_at": None,
            "poll_started_at": None,
            "last_completed_poll": previous.get("last_completed_poll"),
            "last_successful_poll": previous.get("last_successful_poll"),
            "last_poll_summary": previous.get("last_poll_summary"),
            "last_error": previous.get("last_error"),
            "next_poll_at": None,
            "counts": {},
        }
        self.lock = threading.Lock()
        self.last_checkpoint = time.monotonic()

    def update(self, **fields: object) -> None:
        with self.lock:
            self.state.update(fields)
            self.store.put("health", self.state)
        self.last_checkpoint = time.monotonic()

    def operation(self, phase: str, number: int | None = None) -> None:
        self.update(
            operation={"phase": phase, "pr": number, "started_at": _now()}, progress_at=_now()
        )

    def count(self, name: str, amount: int = 1) -> None:
        with self.lock:
            self.state["counts"][name] = self.state["counts"].get(name, 0) + amount
        if time.monotonic() - self.last_checkpoint >= _HEARTBEAT_SECONDS:
            self.update(progress_at=_now())

    def error(self, error: object, number: int | None = None) -> None:
        self.update(last_error={"at": _now(), "pr": number, "message": _summary(error)})

    def heartbeat(self) -> None:
        with self.lock:
            operation = self.state["operation"]
            if operation is None or operation["phase"] in ("waiting", "stopped"):
                return
            age = (
                datetime.datetime.now(datetime.timezone.utc)
                - datetime.datetime.fromisoformat(operation["started_at"])
            ).total_seconds()
            _LOG.info(
                "Heartbeat: phase=%s pr=%s operation_age=%.1fs counts=%s; "
                "worker alive, operation completion not yet confirmed",
                operation["phase"],
                operation["pr"],
                age,
                self.state["counts"],
            )

    @contextlib.contextmanager
    def heartbeats(self) -> Iterator[None]:
        finished = threading.Event()

        def report() -> None:
            while not finished.wait(_HEARTBEAT_SECONDS):
                self.heartbeat()

        thread = threading.Thread(target=report, name="vendor-bot-heartbeat", daemon=True)
        thread.start()
        try:
            yield
        finally:
            finished.set()
            thread.join()


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

    def __init__(
        self,
        args: argparse.Namespace,
        gh: promote.GitHub,
        store: Store,
        progress: Progress | None = None,
    ) -> None:
        self.args, self.gh, self.store = args, gh, store
        self.progress = progress or Progress(store)
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
                args.since or "1970-01-01T00:00:00Z",
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
        _LOG.info(
            "Discovery started: since=%s cold_start=%s",
            since,
            not self.store.get("bootstrapped", False),
        )
        for endpoint in endpoints:
            self.progress.operation(
                "discover_open" if "state=open" in endpoint else "discover_history"
            )
            for pr in _pages(self.gh, endpoint):
                if _STOP.is_set() or (self.args.workdir / "stop").exists():
                    _STOP.set()
                    return sorted(tracked)
                if "state=all" in endpoint and pr["updated_at"] < since:
                    break
                self.progress.count("scanned")
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
        self.progress.update(progress_at=_now())
        _LOG.info(
            "Discovery completed: scanned=%s tracked_lock_prs=%s",
            self.progress.state["counts"].get("scanned", 0),
            len(tracked),
        )
        return sorted(tracked)

    def _transition(self, number: int, state: str, reason: str = "") -> None:
        observation = {"state": state, "reason": _summary(reason) if reason else ""}
        if self.store.get(f"observation:{number}") != observation:
            _LOG.info("PR #%s: %s%s", number, state, f": {observation['reason']}" if reason else "")
            self.store.put(f"observation:{number}", observation)
        self.store.put(f"problem:{number}", observation["reason"] if state == "blocked" else None)

    def _feedback(
        self, pr: dict, problem: str | None, *, attribution: dict[str, dict] | None = None
    ) -> None:
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
        if attribution:
            details = metadata.similarity_feedback(attribution, self.gh.upstream_repo)
            if details:
                message += "\n\n" + details
        body = f"{self.marker}\n{message}"
        dismissal_notice = (
            "\n\nAn authorized reviewer must dismiss the automated metadata "
            "review; this account lacks dismissal permission."
        )
        if not self.args.publish:
            self._transition(
                number,
                "blocked" if problem else "metadata_valid",
                problem or "dry run; no feedback published",
            )
            return
        # Check immutable head and description immediately before feedback. A
        # concurrent author edit will be retried on the next poll.
        current = self.gh.get(f"{self.prefix}/pulls/{number}")
        if current["head"]["sha"] != pr["head"]["sha"] or current.get("body") != pr.get("body"):
            self._transition(
                number, "deferred", "source head or description changed; retry next poll"
            )
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
                _LOG.info("PR #%s: updated metadata feedback comment", number)
        else:
            self.gh.api(
                f"{self.prefix}/issues/{number}/comments", method="POST", payload={"body": body}
            )
            _LOG.info("PR #%s: posted metadata feedback comment", number)
        if current["state"] != "open" or current["user"]["login"] == self.actor:
            self._transition(
                number,
                "blocked" if problem else "metadata_valid",
                problem or "feedback reconciled; not a code-review approval",
            )
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
            _LOG.info("PR #%s: requested metadata changes", number)
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
                    _LOG.info("PR #%s: dismissed resolved metadata review %s", number, review["id"])
                except RuntimeError as error:
                    if "HTTP 403" not in str(error):
                        raise
                    _LOG.warning(
                        "PR #%s: metadata is valid, but an authorized human must dismiss automated review %s.",
                        number,
                        review["id"],
                    )
                    self._transition(
                        number, "blocked", "Authorized human must dismiss metadata review"
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
        self._transition(
            number,
            "blocked" if problem else "metadata_valid",
            problem or "feedback reconciled; not a code-review approval",
        )

    def _inputs(self, pr: dict) -> tuple[manage.Vendor, manage.Vendor] | None:
        for change in _pages(self.gh, f"{self.prefix}/pulls/{pr['number']}/files"):
            if change["filename"] == promote._LOCK and change.get("status") in ("added", "removed"):
                return None  # Creating/removing the entire lock is not a source update.
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
            if pr.get("merged"):
                return None  # Historical update belonging to another canonical branch.
            raise ValueError(
                "Finish the preceding vendor promotion, then rebase this source-update PR onto the canonical pin."
            )
        return previous, reviewed

    def _arguments(self, number: int, reviewed: manage.Vendor) -> argparse.Namespace:
        return argparse.Namespace(
            source_pr=str(number),
            canonical_repo=self.args.canonical_repo,
            canonical_branch=self.args.canonical_branch,
            upstream_pr=[],
            unpaired_reason=None,
            map_upstream=[],
            repo=self.args.repo,
            fork=self.args.fork,
            source_repo=None,
            worktree=self.args.workdir / f"promotion-{number}-{reviewed.commit[:12]}",
            publish=self.args.publish,
            auto_merge=True,
            wait=False,
            timeout=3600,
        )

    def _publish(self, arguments: argparse.Namespace, plan: promote.Promotion) -> None:
        number = plan.number
        if plan.completed_pr is not None:
            self.store.put(f"complete:{number}", plan.completed_pr["html_url"])
            self._transition(number, "completed", plan.completed_pr["html_url"])
            return
        if not self.args.publish:
            self._transition(number, "promotion_ready", f"dry run: {plan.branch}")
            return
        self.progress.operation("publishing_promotion", number)
        self._transition(number, "promoting", "create or resume verified lock-only promotion")
        followup = promote._publish(arguments, self.gh, plan)
        self.store.put(f"promotion:{number}", followup["number"])
        latest = self.gh.get(f"{self.prefix}/pulls/{followup['number']}")
        if latest.get("merged"):
            promote._verify_remote_commit(
                self.gh, plan, self.gh.consumer_repo, latest["merge_commit_sha"]
            )
            self.store.put(f"complete:{number}", latest["html_url"])
        self._transition(
            number, "completed" if latest.get("merged") else "promotion_pending", latest["html_url"]
        )

    def inspect(self, number: int) -> None:
        """Reconcile one source-update PR and its deterministic promotion operation."""
        completed = self.store.get(f"complete:{number}")
        if completed:
            self._transition(number, "completed", completed)
            return
        self.progress.operation("inspecting", number)
        pr = self.gh.get(f"{self.prefix}/pulls/{number}")
        if pr["state"] == "closed" and not pr.get("merged"):
            self._transition(number, "ignored", "closed without merging")
            return
        try:
            inputs = self._inputs(pr)
            if inputs is None:
                self._transition(
                    number, "ignored", "not a promotable update for this canonical vendor"
                )
                return
            previous, reviewed = inputs
            self.progress.count("relevant")
        except (ValueError, manage.VendorError) as error:
            self._feedback(pr, str(error))
            return
        arguments = self._arguments(number, reviewed)
        if pr.get("merged"):
            self.progress.operation("recovering_promotion", number)
            plan = promote._plan_identity(arguments, self.gh)
            if promote._recover_plan(arguments, self.gh, plan):
                self._transition(number, "recovered", "verified durable promotion provenance")
                self._publish(arguments, plan)
                return
            current = promote._lock_at(self.gh, self.gh.consumer_repo, plan.main_sha)
            if current.to_mapping() != reviewed.to_mapping():
                self._transition(number, "ignored", "historical pin is no longer pending promotion")
                return
            promote._check_current(self.gh, plan)
        try:
            self.progress.operation("validating_metadata", number)
            entry = metadata.parse(pr.get("body") or "", self.gh.vendor_name, self.gh.upstream_repo)
            cache = self.args.workdir / "sources.git"
            # Until provenance is committed, decisions follow current remote
            # inputs, not whichever upstream heads a previous process cached.
            _, evidence = metadata.resolve(self.gh, previous, reviewed, entry, cache)
        except (ValueError, manage.VendorError) as error:
            self._feedback(pr, str(error))
            return
        self.progress.operation("reconciling_feedback", number)
        self._feedback(pr, None, attribution=evidence)
        if not pr.get("merged"):
            return
        # Re-read mutable description/head before any promotion. Immutable lock
        # checks inside promote also run before each remote write.
        current = self.gh.get(f"{self.prefix}/pulls/{number}")
        if current.get("body") != pr.get("body") or current["head"]["sha"] != pr["head"]["sha"]:
            self._transition(
                number, "deferred", "source head or description changed; retry next poll"
            )
            return
        arguments.upstream_pr = [str(item) for item in entry["upstream_prs"]]
        arguments.unpaired_reason = entry["unpaired_reason"]
        arguments.source_repo = cache if cache.exists() else None
        self.progress.operation("planning_promotion", number)
        plan = promote._make_plan(
            arguments,
            self.gh,
            auto_match=bool(entry["upstream_prs"]),
            resolutions=entry["resolutions"],
        )
        # Attribution evidence is versioned with the durable promotion record.
        # Reuse the original published record on retries rather than replacing it.
        if plan.completed_pr is None:
            plan.record["contributor_metadata"] = {
                "source_head": pr["head"]["sha"],
                "metadata_digest": metadata.fingerprint(entry),
                "pr_author": pr["user"]["login"],
            }
        self._publish(arguments, plan)

    def poll(self) -> None:
        """Process candidates serially; preserve failures for safe later retries."""
        started = time.monotonic()
        self.progress.update(
            poll_started_at=_now(),
            next_poll_at=None,
            counts=dict(scanned=0, tracked=0, relevant=0, processed=0, blocked=0, errors=0),
        )
        _LOG.info("Poll started")
        completed = False
        try:
            candidates = self._discover()
            self.progress.count("tracked", len(candidates))
            for number in candidates:
                if _STOP.is_set() or (self.args.workdir / "stop").exists():
                    _STOP.set()
                    break
                try:
                    self.inspect(number)
                    if self.store.get(f"problem:{number}"):
                        self.progress.count("blocked")
                except (
                    RuntimeError,
                    ValueError,
                    OSError,
                    KeyError,
                    subprocess.TimeoutExpired,
                ) as error:
                    self.progress.count("errors")
                    self.progress.error(error, number)
                    self._transition(number, "blocked", _summary(error))
                    _LOG.error("PR #%s paused: %s", number, _summary(error))
                self.progress.count("processed")
            completed = not _STOP.is_set()
        except (RuntimeError, OSError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
            self.progress.count("errors")
            self.progress.error(error)
            raise
        finally:
            counts = self.progress.state["counts"].copy()
            outcome = (
                "failed"
                if not completed and not _STOP.is_set()
                else "interrupted"
                if _STOP.is_set()
                else "needs_attention"
                if counts["errors"] or counts["blocked"]
                else "ok"
            )
            summary = {
                "outcome": outcome,
                "finished_at": _now(),
                "duration_seconds": round(time.monotonic() - started, 3),
                **counts,
            }
            fields = {"last_poll_summary": summary, "progress_at": _now()}
            if completed:
                fields["last_completed_poll"] = summary["finished_at"]
                self.store.put("last_poll", summary["finished_at"])
            if outcome == "ok":
                fields["last_successful_poll"] = summary["finished_at"]
            self.progress.update(**fields)
            _LOG.info("Poll finished: %s", summary)


def _initialize_workdir(path: Path) -> None:
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
    files = {
        _STATE_FILE,
        "state.sqlite-wal",
        "state.sqlite-shm",
        "state.sqlite-journal",
        "bot.log",
        "bot.lock",
        "stop",
    }
    for entry in path.iterdir():
        directory = entry.name == "sources.git" or re.fullmatch(
            r"promotion-[0-9]+-[0-9a-f]{12}", entry.name
        )
        if (
            entry.is_symlink()
            or entry.stat().st_uid != os.getuid()
            or not (entry.is_dir() if directory else entry.name in files and entry.is_file())
        ):
            raise ValueError(
                "Choose an empty workdir or an existing vendor-bot workdir; unexpected entry: "
                + entry.name
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
            progress = Progress(store)
            progress.operation("starting")
            with progress.heartbeats():
                try:
                    gh = promote.GitHub(
                        args.consumer_repo,
                        args.vendor,
                        args.upstream_repo,
                        args.base_branch,
                        args.upstream_branch,
                    )
                    monitor = Monitor(args, gh, store, progress)
                    store.put("process", {"pid": os.getpid(), "publish": args.publish})
                    _LOG.info(
                        "Started vendor %s as %s; remote writes %s; interval=%ss",
                        args.vendor,
                        monitor.actor,
                        args.publish,
                        args.interval,
                    )
                    failures = 0
                    while not _STOP.is_set():
                        try:
                            monitor.poll()
                            failures = 0
                        except (
                            RuntimeError,
                            OSError,
                            ValueError,
                            KeyError,
                            subprocess.TimeoutExpired,
                        ) as error:
                            failures += 1
                            _LOG.error("Poll paused: %s", _summary(error))
                            if args.once:
                                raise
                        if args.once or _STOP.is_set():
                            break
                        delay = min(args.interval * 2 ** min(failures, 5), 3600)
                        deadline = time.monotonic() + delay
                        progress.operation("waiting")
                        progress.update(
                            next_poll_at=(
                                datetime.datetime.now(datetime.timezone.utc)
                                + datetime.timedelta(seconds=delay)
                            ).isoformat()
                        )
                        _LOG.info(
                            "Next poll at %s (in %ss; consecutive_poll_failures=%s)",
                            progress.state["next_poll_at"],
                            delay,
                            failures,
                        )
                        while time.monotonic() < deadline and not _STOP.wait(1):
                            if (args.workdir / "stop").exists():
                                _STOP.set()
                                break
                except (
                    RuntimeError,
                    OSError,
                    ValueError,
                    KeyError,
                    subprocess.TimeoutExpired,
                ) as error:
                    progress.error(error)
                    raise
                finally:
                    progress.operation("stopped")
                    progress.update(next_poll_at=None)
                    store.put("process", {"pid": None, "publish": args.publish})
                    _LOG.info("Stopped vendor %s", args.vendor)


def _status(workdir: Path) -> dict:
    """Read cached telemetry without creating or repairing the state database."""
    active = _running(workdir)
    result = {
        "running": active,
        "stop_requested": active and (workdir / "stop").exists(),
        "pid": None,
        "publish": None,
        "operation": None,
    }
    database = workdir / _STATE_FILE
    if not database.exists():
        result["state_available"] = False
        return result
    try:
        with contextlib.closing(
            sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True, timeout=1)
        ) as connection:
            rows = connection.execute(
                "SELECT key, value FROM state WHERE key IN ('health', 'process', 'settings')"
            ).fetchall()
        state = {key: json.loads(value) for key, value in rows}
        if any(not isinstance(value, dict) for value in state.values()):
            raise ValueError("Malformed status cache; expected JSON objects.")
        health, process = state.get("health", {}), state.get("process", {})
        for key in (
            "operation",
            "progress_at",
            "poll_started_at",
            "last_completed_poll",
            "last_successful_poll",
            "last_poll_summary",
            "last_error",
            "next_poll_at",
            "counts",
        ):
            result[key] = health.get(key)
        result.update(
            state_available=True,
            pid=process.get("pid") if active else None,
            publish=process.get("publish"),
            vendor=state.get("settings", {}).get("vendor"),
        )
        if not active:
            result["operation"] = None
            result["next_poll_at"] = None
    except (sqlite3.Error, ValueError, OSError) as error:
        result.update(state_available=False, state_error=_summary(error))
    return result


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
        "--log-level",
        type=str.upper,
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Daemon/foreground log threshold (default: INFO).",
    )
    run.add_argument(
        "--since",
        help=(
            "Optional historical discovery cutoff, UTC YYYY-MM-DDTHH:MM:SSZ; "
            "default all history. Reuse the same cutoff after cache loss."
        ),
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
    logging.basicConfig(
        level=getattr(args, "log_level", "INFO"), format="%(asctime)sZ %(levelname)s %(message)s"
    )
    logging.Formatter.converter = time.gmtime
    try:
        if args.command != "run":
            status = _status(args.workdir)
            if args.command == "stop" and status["running"]:
                (args.workdir / "stop").touch(mode=0o600)
                status["stop_requested"] = True
            print(json.dumps(status))
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
        _LOG.error("Bot stopped: %s", _summary(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
