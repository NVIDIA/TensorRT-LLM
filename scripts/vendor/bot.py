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
    from .diagnostics import Diagnostic, field
else:
    import manage
    import metadata
    import promote
    from diagnostics import Diagnostic, field

_LOG = logging.getLogger("vendor-bot")
_STATE_FILE = "state.sqlite"
_STOP = threading.Event()
_HEARTBEAT_SECONDS = 30
_FEEDBACK_KEY = re.compile(r"<!-- vendor-bot-feedback:v1 ([0-9a-f]{64}) -->")
_FEEDBACK_BASE = re.compile(r"<!-- vendor-bot-base:v1 ([^\s<>]+) -->")


def _supported_base(branch: str) -> bool:
    return branch == "main" or (branch.startswith("release/") and len(branch) > len("release/"))


def _feedback_base(item: dict) -> str | None:
    for line in item.get("body", "").splitlines():
        match = _FEEDBACK_BASE.fullmatch(line)
        if match:
            return match[1]
    return None


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


def _upstream_feedback(proof: dict, vendor_name: str) -> str:
    return (
        "Upstream-main source ancestry is valid: pinned commit "
        f"`{proof['promoted']}` contains canonical "
        f"dev tip `{proof['checked_canonical_tip']}` and is reachable from "
        f"the configured upstream branch at `{proof['upstream_observed_main']}`. "
        f"No vendor-promotion metadata or upstream PR list is required for "
        f"{field(vendor_name)} in this mode. Promotion will fast-forward dev to "
        "the exact reviewed pin, not the moving upstream tip, and restore the canonical "
        "lock URL/branch. This is **not** a code-review approval. Normal review and CI "
        "still apply; promotion requires the source PR to merge and ancestry to remain valid."
    )


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

    def migrate_branch(self, settings: dict, record: dict) -> None:
        """Atomically archive derived results and start reconciliation for a new branch."""
        keys = ("cursor", "bootstrapped", "health", "last_poll")
        prefixes = ("complete:", "promotion:", "observation:", "problem:")
        rows = self.connection.execute("SELECT key, value FROM state").fetchall()
        archived = {
            key: json.loads(value) for key, value in rows if key in keys or key.startswith(prefixes)
        }
        history = self.get("branch_migrations", [])
        history.append({**record, "previous_results": archived})
        with self.connection:
            self.connection.executemany(
                "DELETE FROM state WHERE key = ?", [(key,) for key in archived]
            )
            self.connection.executemany(
                "INSERT INTO state VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                [
                    (key, json.dumps(value, sort_keys=True))
                    for key, value in (
                        ("settings", settings),
                        ("branch_migrations", history),
                        ("migration", record),
                    )
                ],
            )


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
    """One vendor/base-branch pair; run other pairs in separate workdirs."""

    def __init__(
        self,
        args: argparse.Namespace,
        gh: promote.GitHub,
        store: Store,
        progress: Progress | None = None,
    ) -> None:
        if not _supported_base(gh.base_branch):
            raise ValueError("--base-branch must be main or release/<name>.")
        self.args, self.gh, self.store = args, gh, store
        self.progress = progress or Progress(store)
        self.actor = gh.get("user")["login"]
        self.prefix = f"repos/{gh.consumer_repo}"
        self.marker = f"<!-- vendor-bot:{gh.vendor_name} -->"
        self.refresh_branches = frozenset(args.refresh_branch)
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
        migration_from = args.migrate_canonical_branch_from
        if saved is not None and saved != settings:
            changed = {
                key for key in saved.keys() | settings.keys() if saved.get(key) != settings.get(key)
            }
            if changed != {"canonical_branch"}:
                raise ValueError(
                    "Workdir belongs to a different operator/vendor configuration; choose another."
                )
            if migration_from != saved["canonical_branch"]:
                raise ValueError(
                    "Canonical branch changed; stop the old process and restart with "
                    f"--migrate-canonical-branch-from {saved['canonical_branch']} after the refresh merges."
                )
            record = self._verify_branch_migration(migration_from)
            store.migrate_branch(settings, record)
            self.progress.update(
                last_completed_poll=None,
                last_successful_poll=None,
                last_poll_summary=None,
                last_error=None,
                next_poll_at=None,
            )
            _LOG.info(
                "Canonical branch migrated: %s -> %s; rediscovery required",
                migration_from,
                args.canonical_branch,
            )
        elif migration_from:
            record = store.get("migration", {})
            if (
                saved == settings
                and record.get("from") == migration_from
                and record.get("to") == args.canonical_branch
            ):
                _LOG.info(
                    "Canonical branch migration already applied: %s -> %s",
                    migration_from,
                    args.canonical_branch,
                )
            else:
                raise ValueError(
                    "No matching saved branch migration. For a fresh/cache-loss start, omit "
                    "--migrate-canonical-branch-from and use the current canonical configuration."
                )
        store.put("settings", settings)
        # Refresh targets are an operator policy, not publication destinations.
        # Changing them does not migrate the canonical branch or its provenance.
        store.put("refresh_branches", sorted(self.refresh_branches))
        if store.get("since") is None:
            store.put(
                "since",
                args.since or "1970-01-01T00:00:00Z",
            )

    def _verify_branch_migration(self, previous_branch: str) -> dict:
        """Read authoritative remote state before changing the local configuration."""
        self.progress.operation("verifying_branch_migration")
        revision = promote._main_sha(self.gh)
        current = promote._lock_at(self.gh, self.gh.consumer_repo, revision)
        if (
            promote._repo_name(current.url).lower() != self.args.canonical_repo.lower()
            or current.branch != self.args.canonical_branch
        ):
            raise ValueError(
                "Branch migration requires the merged base-branch lock to point to the new "
                "canonical repository/branch. Finish the refresh before restarting."
            )
        ref = self.gh.get(
            f"repos/{self.args.canonical_repo}/git/ref/heads/{self.args.canonical_branch}"
        )
        tip = promote._sha(ref["object"]["sha"])
        if self.gh.compare(self.args.canonical_repo, current.commit, tip)["status"] not in (
            "identical",
            "ahead",
        ):
            raise ValueError(
                "New canonical branch does not contain the merged lock's pinned commit."
            )
        self._check_pending_promotions(previous_branch)
        latest = promote._lock_at(self.gh, self.gh.consumer_repo, promote._main_sha(self.gh))
        if latest.to_mapping() != current.to_mapping():
            raise ValueError("Base vendor lock changed during migration; inspect it and retry.")
        return {
            "from": previous_branch,
            "to": self.args.canonical_branch,
            "at": _now(),
            "consumer_revision": revision,
            "canonical_tip": tip,
            "pin": current.commit,
        }

    def _check_pending_promotions(self, previous_branch: str) -> None:
        """Do not strand open follow-ups or unpublished promotion branches."""
        prefix = f"chore/{self.gh.vendor_name}-promote-"
        branches = {
            item["name"]: item["commit"]["sha"]
            for item in _pages(self.gh, f"repos/{self.args.fork}/branches")
            if item["name"].startswith(prefix)
        }
        local = promote._git(
            self.args.repo,
            "for-each-ref",
            "--format=%(refname:short) %(objectname)",
            f"refs/heads/{prefix}*",
        )
        for line in local.splitlines():
            branch, sha = line.split()
            if branch not in branches:
                branches[branch] = sha
        # Open follow-ups can survive deletion of their publishing branch.
        for pr in _pages(self.gh, f"{self.prefix}/pulls?state=open"):
            branch = pr["head"]["ref"]
            if pr["base"]["ref"] == self.gh.base_branch and branch.startswith(prefix):
                branches[branch] = pr["head"]["sha"]
        for branch, sha in branches.items():
            remote = self.gh.optional(f"{self.prefix}/commits/{sha}")
            if remote is None:
                remote = self.gh.optional(f"repos/{self.args.fork}/commits/{sha}")
            message = (
                remote["commit"]["message"]
                if remote
                else promote._git(self.args.repo, "log", "-1", "--format=%B", sha)
            )
            record = promote._record_from_message(message)
            if record.get("canonical_branch") != previous_branch:
                continue
            consumer_pr = record.get("consumer_pr")
            consumer_base = record.get("consumer_base")
            consumer = (
                re.fullmatch(r"https://github\.com/([\w.-]+/[\w.-]+)/pull/[1-9][0-9]*", consumer_pr)
                if isinstance(consumer_pr, str)
                else None
            )
            if consumer is None or not isinstance(consumer_base, str) or not consumer_base:
                raise ValueError(
                    f"Missing or invalid consumer provenance on promotion branch {branch}; "
                    "inspect its promotion commit before migrating."
                )
            if (
                consumer.group(1).lower() != self.gh.consumer_repo.lower()
                or consumer_base != self.gh.base_branch
            ):
                continue
            if record.get("vendor") != self.gh.vendor_name:
                raise ValueError(f"Unexpected vendor provenance on promotion branch {branch}.")
            prs = list(
                _pages(
                    self.gh,
                    f"{self.prefix}/pulls?state=all&head={self.args.fork.split('/')[0]}:{branch}"
                    f"&base={self.gh.base_branch}",
                )
            )
            if len(prs) != 1 or prs[0]["state"] != "closed":
                location = prs[0]["html_url"] if len(prs) == 1 else branch
                raise ValueError(
                    f"Unresolved old-branch promotion: {location}. Finish it, or explicitly "
                    "close its superseded PR before migrating. Unpublished branches need "
                    "maintainer inspection; migration never deletes or redirects them."
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
                    if not _supported_base(pr["base"]["ref"]) and pr["state"] == "open":
                        snapshot = self._out_of_scope_snapshot(pr)
                        if self.store.get(f"out_of_scope:{pr['number']}") != snapshot:
                            comments, reviews = self._owned_feedback(pr)
                            latest = max(comments, key=lambda item: int(item["id"]), default=None)
                            marked = [
                                item
                                for item in reviews
                                if item.get("body", "").startswith(self.marker)
                            ]
                            if (latest or marked) and self._cleanup_owner(
                                latest, marked
                            ) == self.gh.base_branch:
                                tracked.add(pr["number"])
                            else:
                                self.store.put(f"out_of_scope:{pr['number']}", snapshot)
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

    def _owned_feedback(self, pr: dict) -> tuple[list[dict], list[dict]]:
        number = pr["number"]
        comments = [
            item
            for item in _pages(self.gh, f"{self.prefix}/issues/{number}/comments")
            if item["user"]["login"] == self.actor and item.get("body", "").startswith(self.marker)
        ]
        reviews = (
            [
                item
                for item in _pages(self.gh, f"{self.prefix}/pulls/{number}/reviews")
                if item["user"]["login"] == self.actor
            ]
            if pr["state"] == "open" and pr["user"]["login"] != self.actor
            else []
        )
        return comments, reviews

    @staticmethod
    def _cleanup_owner(latest: dict | None, reviews: list[dict]) -> str:
        if latest and (base := _feedback_base(latest)):
            return base
        for review in sorted(reviews, key=lambda item: int(item["id"]), reverse=True):
            if base := _feedback_base(review):
                return base
        # Unscoped legacy feedback has one cleanup owner across monitor instances.
        return "main"

    @staticmethod
    def _out_of_scope_snapshot(pr: dict) -> dict:
        return {
            "base": pr["base"]["ref"],
            "head": pr["head"]["sha"],
            "updated_at": pr["updated_at"],
        }

    def _retargeted(self, pr: dict) -> None:
        number = pr["number"]
        if not _supported_base(pr["base"]["ref"]) and pr["state"] == "open":
            if not self._feedback(pr, None, out_of_scope=True):
                return
            self.store.put(f"out_of_scope:{number}", self._out_of_scope_snapshot(pr))
        else:
            self._transition(number, "ignored", "PR targets a different consumer base branch")
        self.store.put(
            "tracked", [item for item in self.store.get("tracked", []) if item != number]
        )

    def _transition(self, number: int, state: str, reason: str = "") -> None:
        observation = {"state": state, "reason": _summary(reason) if reason else ""}
        if self.store.get(f"observation:{number}") != observation:
            _LOG.info("PR #%s: %s%s", number, state, f": {observation['reason']}" if reason else "")
            self.store.put(f"observation:{number}", observation)
        self.store.put(f"problem:{number}", observation["reason"] if state == "blocked" else None)

    def _feedback(
        self,
        pr: dict,
        problem: Diagnostic | None,
        *,
        attribution: dict[str, dict] | None = None,
        upstream_proof: dict | None = None,
        no_promotion: bool = False,
        out_of_scope: bool = False,
        source: manage.Vendor | None = None,
    ) -> bool:
        number = pr["number"]
        no_promotion = no_promotion or out_of_scope
        state = "blocked" if problem else "ignored" if no_promotion else "metadata_valid"
        message = (
            f"Vendor promotion needs {problem.owner} action:\n\n{problem}"
            "\n\nThe bot will recheck automatically on a later poll."
            if problem
            else "Vendor promotion metadata and source attribution are valid. "
            "This is **not** a code-review approval. Promotion still requires the source PR to merge."
        )
        notification_message = message
        if problem is not None and problem.notification_text is not None:
            notification_message = (
                f"Vendor promotion needs {problem.owner} action:\n\n{problem.notification_text}"
                "\n\nThe bot will recheck automatically on a later poll."
            )
        if upstream_proof is not None and problem is None:
            message = _upstream_feedback(upstream_proof, self.gh.vendor_name)
            notification_message = _upstream_feedback(
                upstream_proof
                | {
                    "checked_canonical_tip": "LATEST_DEV_TIP",
                    "upstream_observed_main": "UPSTREAM_TIP",
                },
                self.gh.vendor_name,
            )
        if no_promotion:
            message = (
                f"This PR does not require vendor promotion for {field(self.gh.vendor_name)} "
                "under this monitor's configuration. The previous promotion-metadata request "
                "no longer applies; no upstream-PR metadata is required by this monitor. "
                "This is **not** a code-review approval. Normal review and CI still apply."
            )
            notification_message = message
        if out_of_scope:
            message = (
                f"This PR now targets {field(pr['base']['ref'])}, outside the vendor bot's "
                "supported targets (`main` and `release/<name>`). Vendor validation and "
                "promotion are not required by this bot on development branches. The previous "
                "vendor-metadata objection no longer applies. This PR will leave active "
                "tracking after cleanup; retargeting it to an explicitly monitored branch "
                "will trigger fresh validation. This is **not** a code-review approval. "
                "Normal review and CI still apply."
            )
            notification_message = message
        if attribution:
            details = metadata.similarity_feedback(attribution, self.gh.upstream_repo)
            if details:
                message += "\n\n" + details
                notification_message += "\n\n" + details
        head = promote._sha(pr["head"]["sha"])
        dismissal_notice = (
            "\n\nAn authorized reviewer must dismiss the automated metadata "
            "review; this account lacks dismissal permission."
        )
        if not self.args.publish:
            self._transition(
                number,
                state,
                str(problem) if problem else "dry run; no feedback published",
            )
            return False
        # Check immutable head and description immediately before feedback. A
        # concurrent author edit will be retried on the next poll.
        current = self.gh.get(f"{self.prefix}/pulls/{number}")
        if out_of_scope and (
            _supported_base(current["base"]["ref"]) or current["base"]["ref"] != pr["base"]["ref"]
        ):
            self._transition(number, "deferred", "PR target changed; retry next poll")
            return False
        if not out_of_scope and current["base"]["ref"] != self.gh.base_branch:
            self._transition(number, "ignored", "PR targets a different consumer base branch")
            return False
        if no_promotion and current["state"] != "open":
            self._transition(number, "ignored", "PR is no longer open")
            return False
        if current["head"]["sha"] != pr["head"]["sha"] or current.get("body") != pr.get("body"):
            self._transition(
                number, "deferred", "source head or description changed; retry next poll"
            )
            return False
        existing, reviews = self._owned_feedback(current)
        latest = max(existing, key=lambda item: int(item["id"]), default=None)
        marked_reviews = [
            item
            for item in reviews
            if item.get("body", "").startswith(self.marker)
            and item["state"] in ("CHANGES_REQUESTED", "DISMISSED")
        ]
        if out_of_scope and self._cleanup_owner(latest, marked_reviews) != self.gh.base_branch:
            self._transition(number, "ignored", "Retargeting cleanup belongs to another monitor")
            return True
        automated = [item for item in marked_reviews if item["state"] == "CHANGES_REQUESTED"]
        manual = [
            item
            for item in reviews
            if not item.get("body", "").startswith(self.marker)
            and item["state"] in ("APPROVED", "CHANGES_REQUESTED")
        ]
        if no_promotion and latest is None and not marked_reviews:
            self._transition(number, "ignored", "not a promotable update for this canonical vendor")
            return True
        dismissal_blocked = False
        if not problem:
            for review in automated:
                # Never replace a metadata objection with APPROVE or dismiss a
                # human review. GitHub may require an authorized human dismissal.
                try:
                    self.gh.api(
                        f"{self.prefix}/pulls/{number}/reviews/{review['id']}/dismissals",
                        method="PUT",
                        payload={
                            "message": (
                                "PR targets a development branch; automated vendor objection "
                                "no longer applies. Code review remains required."
                            )
                            if out_of_scope
                            else "Automated vendor metadata issue resolved; code review remains required.",
                            "event": "DISMISS",
                        },
                    )
                    _LOG.info("PR #%s: dismissed resolved metadata review %s", number, review["id"])
                except RuntimeError as error:
                    if "HTTP 403" not in str(error):
                        raise
                    _LOG.warning(
                        "PR #%s: metadata review is resolved, but an authorized human "
                        "must dismiss automated review %s.",
                        number,
                        review["id"],
                    )
                    dismissal_blocked = True
        if dismissal_blocked:
            message += dismissal_notice
            notification_message += dismissal_notice
        fingerprint = metadata.fingerprint(
            {
                "vendor": self.gh.vendor_name,
                "base_branch": self.gh.base_branch,
                "source": {"url": source.url, "branch": source.branch, "commit": source.commit}
                if source is not None
                else None,
                "state": state,
                "message": notification_message,
            }
        )
        # Only the newest bot-owned result counts: A -> B -> A must notify again.
        # The fingerprint is durable on GitHub even if the POST response or SQLite is lost.
        lines = latest["body"].splitlines() if latest else []
        recorded = _FEEDBACK_KEY.fullmatch(lines[1]) if len(lines) > 1 else None
        if recorded is None or recorded[1] != fingerprint:
            body = (
                f"{self.marker}\n<!-- vendor-bot-feedback:v1 {fingerprint} -->\n"
                f"<!-- vendor-bot-base:v1 {self.gh.base_branch} -->\n"
                f"### Vendor validation: {field(self.gh.vendor_name)}\n\n"
                f"@{pr['user']['login']} — {message}\n\n"
                f"Checked PR revision: [`{head[:12]}`]"
                f"(https://github.com/{self.gh.consumer_repo}/pull/{number}/commits/{head}). "
                "This is a snapshot; newer results are posted as new comments. "
                "Observed dev/upstream tip SHAs may be stale: tip movement alone does not "
                "post a new comment when the result is unchanged. Fetch the live branch "
                "before rebasing or choosing a new pin."
            )
            if source is not None:
                body += (
                    f"\n\nChecked source: {field(source.url)} / {field(source.branch)}, "
                    f"pin `{source.commit}`."
                )
            if latest is not None:
                body += (
                    f"\n\nSupersedes [previous vendor validation]"
                    f"(https://github.com/{self.gh.consumer_repo}/pull/{number}"
                    f"#issuecomment-{latest['id']})."
                )
            latest = self.gh.api(
                f"{self.prefix}/issues/{number}/comments", method="POST", payload={"body": body}
            )
            _LOG.info("PR #%s: posted metadata feedback comment %s", number, latest["id"])
        feedback_url = (
            f"https://github.com/{self.gh.consumer_repo}/pull/{number}#issuecomment-{latest['id']}"
        )
        review_status = (
            "not applicable on this target branch"
            if out_of_scope
            else f"needs {problem.owner} action"
            if problem
            else "no source-validation objection"
        )
        review_body = (
            f"{self.marker}\n<!-- vendor-bot-base:v1 {self.gh.base_branch} -->\n"
            f"Vendor validation result: {review_status}."
            f"\n\n[Latest vendor validation and step-by-step fix]({feedback_url})."
        )
        for review in marked_reviews:
            if review["body"] != review_body:
                self.gh.api(
                    f"{self.prefix}/pulls/{number}/reviews/{review['id']}",
                    method="PUT",
                    payload={"body": review_body},
                )
                _LOG.info("PR #%s: updated metadata review %s", number, review["id"])
        if (
            problem
            and problem.owner == "author"
            and not automated
            and not manual
            and current["state"] == "open"
            and current["user"]["login"] != self.actor
        ):
            self.gh.api(
                f"{self.prefix}/pulls/{number}/reviews",
                method="POST",
                payload={"event": "REQUEST_CHANGES", "commit_id": head, "body": review_body},
            )
            _LOG.info("PR #%s: requested metadata changes", number)
        self._transition(
            number,
            "blocked" if dismissal_blocked else state,
            "Authorized human must dismiss metadata review"
            if dismissal_blocked
            else str(problem)
            if problem
            else "feedback reconciled; not a code-review approval",
        )
        return not dismissal_blocked

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
        try:
            before = promote._lock_document_at(self.gh, self.gh.consumer_repo, base)["vendors"]
        except Diagnostic as error:
            raise RuntimeError(f"Canonical/base vendor lock is invalid: {error}") from error
        after = promote._lock_document_at(self.gh, self.gh.consumer_repo, revision)["vendors"]
        name = self.gh.vendor_name
        if name not in before or name not in after:
            return None  # Vendor creation/removal is not promotion.
        previous = manage._validate_vendor(name, before[name])
        try:
            reviewed = manage._validate_vendor(name, after[name])
            promote._repo_name(reviewed.url)
        except (manage.VendorError, ValueError) as error:
            raise Diagnostic(
                f"Invalid lock entry at `vendors.{name}`: {field(_summary(error))}.",
                f"Correct that entry in `{promote._LOCK}` with `scripts/vendor/manage.py`, "
                "regenerate the vendored snapshot if its source changed, and push the PR. "
                "Editing the PR description will not fix a lock-file error.",
            ) from error
        canonical = (
            promote._repo_name(previous.url).lower() == self.args.canonical_repo.lower()
            and previous.branch == self.args.canonical_branch
        )
        target_is_canonical = promote._repo_name(
            reviewed.url
        ).lower() == self.args.canonical_repo.lower() and (
            reviewed.branch == self.args.canonical_branch
            or reviewed.branch in self.refresh_branches
        )
        if target_is_canonical:
            return None  # Periodic refresh/direct canonical update: separate workflow.
        if not reviewed.branch or (pr.get("merged") and previous.commit == reviewed.commit):
            return None  # Includes lock-only promotions and tag-only updates.
        if not pr.get("merged"):
            comparison = self.gh.compare(self.gh.consumer_repo, base, revision)
            merge_base = promote._sha(comparison["merge_base_commit"]["sha"])
            try:
                original = (
                    before
                    if merge_base == base
                    else promote._lock_document_at(self.gh, self.gh.consumer_repo, merge_base)[
                        "vendors"
                    ]
                )
            except Diagnostic as error:
                raise RuntimeError(f"PR merge-base vendor lock is invalid: {error}") from error
            # Classify the PR's own changes, but retain the current canonical
            # pin as the baseline for source-history checks and promotion.
            if name not in original:
                return None
            original_vendor = manage._validate_vendor(name, original[name])
            reference_changed = (
                promote._repo_name(original_vendor.url).lower(),
                original_vendor.branch,
            ) != (promote._repo_name(reviewed.url).lower(), reviewed.branch)
            if canonical and previous.commit == reviewed.commit and reference_changed:
                raise Diagnostic(
                    f"Temporary source reference adds no new vendor commit: "
                    f"{field(reviewed.commit)} is already pinned on {field(self.gh.base_branch)}.",
                    f"In `{promote._LOCK}` at `vendors.{name}`, restore `url` to "
                    f"{field(previous.url)} and `branch` to {field(previous.branch)}. "
                    "Change only these source-reference fields; keep the commit and other lock "
                    "fields, vendored files, and patch unchanged. No upstream-PR metadata or "
                    "vendor promotion is needed for this same-commit reference change.",
                )
            if original_vendor.commit == reviewed.commit:
                return None
        if previous.commit == reviewed.commit:
            return None
        if not canonical:
            if (
                promote._repo_name(previous.url).lower() == self.args.canonical_repo.lower()
                and previous.branch in self.refresh_branches
            ):
                raise Diagnostic(
                    f"The base lock now uses refresh branch {field(previous.branch)}, but this "
                    f"monitor still publishes to {field(self.args.canonical_branch)}.",
                    "A maintainer must stop this monitor and migrate its workdir with "
                    f"`--canonical-branch {previous.branch} "
                    f"--migrate-canonical-branch-from {self.args.canonical_branch}`. "
                    "Verify a dry run before resuming publication. Do not promote back to "
                    "the old branch or change contributor metadata to bypass this cutover.",
                    owner="maintainer",
                )
            if pr.get("merged"):
                return None  # Historical update belonging to another canonical branch.
            raise Diagnostic(
                "Waiting for the preceding vendor promotion: the base lock still points "
                f"to repository {field(previous.url)}, branch {field(previous.branch)}.",
                f"A maintainer must finish the preceding lock-only promotion back to "
                f"{field(self.args.canonical_repo)} / {field(self.args.canonical_branch)}. "
                "After it merges, the author should rebase this consumer PR onto the latest "
                "base branch and rebase its temporary source branch onto the new canonical "
                "pin, then update the lock/vendor snapshot. No metadata edit can unblock "
                "the preceding promotion.",
                owner="maintainer",
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
        if pr["base"]["ref"] != self.gh.base_branch:
            self._retargeted(pr)
            return
        if pr["state"] == "closed" and not pr.get("merged"):
            self._transition(number, "ignored", "closed without merging")
            return
        try:
            inputs = self._inputs(pr)
            if inputs is None:
                if pr["state"] == "open":
                    self._feedback(pr, None, no_promotion=True)
                else:
                    self._transition(
                        number, "ignored", "not a promotable update for this canonical vendor"
                    )
                return
            previous, reviewed = inputs
            self.progress.count("relevant")
        except Diagnostic as error:
            self._feedback(pr, error)
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
        try:
            self.progress.operation("validating_metadata", number)
            if pr.get("merged"):
                upstream_proof = promote._check_current(self.gh, plan)
            else:
                upstream_proof = promote._check_source(self.gh, previous, reviewed)
            entry, evidence = None, None
            cache = self.args.workdir / "sources.git"
            if upstream_proof is None:
                entry = metadata.parse(
                    pr.get("body") or "", self.gh.vendor_name, self.gh.upstream_repo
                )
                # Until provenance is committed, decisions follow current remote
                # inputs, not whichever upstream heads a previous process cached.
                _, evidence = metadata.resolve(self.gh, previous, reviewed, entry, cache)
            # Attribution may take long enough for another consumer's promotion
            # to advance the shared canonical branch. Do not report stale inputs as valid.
            upstream_proof = promote._check_source(self.gh, previous, reviewed)
        except Diagnostic as error:
            self._feedback(pr, error, source=reviewed)
            return
        self.progress.operation("reconciling_feedback", number)
        self._feedback(
            pr, None, attribution=evidence, upstream_proof=upstream_proof, source=reviewed
        )
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
        if entry is not None:
            arguments.upstream_pr = [str(item) for item in entry["upstream_prs"]]
            arguments.unpaired_reason = entry["unpaired_reason"]
            arguments.source_repo = cache if cache.exists() else None
        self.progress.operation("planning_promotion", number)
        plan = promote._make_plan(
            arguments,
            self.gh,
            auto_match=entry is not None and bool(entry["upstream_prs"]),
            resolutions=entry["resolutions"] if entry is not None else None,
        )
        # Attribution evidence is versioned with the durable promotion record.
        # Reuse the original published record on retries rather than replacing it.
        if plan.completed_pr is None and entry is not None:
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
                "SELECT key, value FROM state WHERE key IN "
                "('health', 'process', 'settings', 'migration', 'refresh_branches')"
            ).fetchall()
        state = {key: json.loads(value) for key, value in rows}
        refresh_branches = state.pop("refresh_branches", [])
        if not isinstance(refresh_branches, list) or any(
            not isinstance(branch, str) for branch in refresh_branches
        ):
            raise ValueError("Malformed refresh policy; expected a JSON list of branch names.")
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
            canonical_repo=state.get("settings", {}).get("canonical_repo"),
            canonical_branch=state.get("settings", {}).get("canonical_branch"),
            base_branch=state.get("settings", {}).get("base_branch"),
            refresh_branches=refresh_branches,
            migration=state.get("migration"),
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
    run.add_argument(
        "--refresh-branch",
        action="append",
        default=[],
        help=(
            "Recognize this branch in --canonical-repo as a refresh target, not a temporary "
            "source branch. Repeat for multiple targets; does not enable publication to them."
        ),
    )
    run.add_argument(
        "--migrate-canonical-branch-from",
        help="Explicitly migrate this stopped workdir from OLD_BRANCH after a merged refresh.",
    )
    run.add_argument("--fork", required=True, help="Consumer publishing fork OWNER/REPO.")
    run.add_argument("--consumer-repo", default="NVIDIA/TensorRT-LLM")
    run.add_argument(
        "--base-branch",
        default="main",
        help="Exact consumer target: main or release/<name>. Other branches are not monitored.",
    )
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
        if not _supported_base(args.base_branch):
            raise ValueError("--base-branch must be main or release/<name>.")
        if not manage._NAME_PATTERN.fullmatch(args.vendor):
            raise ValueError("Invalid vendor key.")
        for name in (args.consumer_repo, args.upstream_repo, args.canonical_repo, args.fork):
            promote._repo_name(f"https://github.com/{name}")
        for branch in (args.canonical_branch, args.base_branch, args.upstream_branch):
            promote._run(["git", "check-ref-format", f"refs/heads/{branch}"])
        for branch in args.refresh_branch:
            promote._run(["git", "check-ref-format", f"refs/heads/{branch}"])
        if args.migrate_canonical_branch_from:
            promote._run(
                ["git", "check-ref-format", f"refs/heads/{args.migrate_canonical_branch_from}"]
            )
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
