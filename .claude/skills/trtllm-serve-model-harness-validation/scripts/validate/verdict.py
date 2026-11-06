#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Fail-closed verdicts for one cell over one population (references/validate.md).

Reads request-trace shaped input of any of the three protocols (trace windows,
replay output, recorder captures), runs the turn-record checks, and writes:

    <out-dir>/report.json    counts, denominators, rule breakdowns, locators
    <out-dir>/verdict.json   one verdict per requested claim, per-check records

Exit code: 0 every requested claim PASS or NOT_APPLICABLE; 1 any FAIL;
2 otherwise any BLOCKED; 3 usage or internal error, or outputs that cannot be
written (no verdict).

Thresholds for rate claims come from the run's matrix file (--matrix). An
unset threshold makes its claim BLOCKED, never PASS. Nothing in the outputs
carries record content: locators, counts, tool names and hashes only.

usage: verdict.py --harness H.yaml [--model M.yaml] [--matrix matrix.yaml]
                  [--population replay|smoke|window|controls] [--claims a,b,...]
                  [--build-id ID] [--tier fast|full] [--controls-receipt controls.json]
                  [--requests PATH ...] [--responses PATH ...] [--raw PATH ...] [--workers N]
                  --out-dir DIR PATH [PATH ...]
"""

import argparse
import hashlib
import json
import math
import os
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline  # noqa: E402
import profiles as prof  # noqa: E402
import sdk_diff  # noqa: E402
from tracelib import expand_shards  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SKILL = os.path.dirname(os.path.dirname(HERE))
VALIDATOR_REFERENCES = ("validate.md", "protocols.md", "turn-ir.md", "acceptance-cases.md")
CLAIMS = ("inventory", "conformance", "runtime", "model-behavior", "harness-outcome")
DEFAULT_CLAIMS = {
    "replay": ("inventory", "conformance", "model-behavior"),
    "smoke": ("inventory", "conformance", "runtime", "model-behavior", "harness-outcome"),
    "window": ("inventory", "conformance", "model-behavior", "harness-outcome"),
    "controls": ("inventory", "conformance", "runtime"),
}
# check id -> (claim, kind, unit, threshold key). kind: hard | rate | flag.
CHECKS = {
    "C-01": ("conformance", "hard", "reply", None),
    "C-02": ("conformance", "hard", "reply", None),
    "C-03": ("conformance", "hard", "call", None),
    "C-04": ("conformance", "hard", "reply", None),
    "C-05": ("conformance", "hard", "reply", None),
    "C-06": ("conformance", "hard", "call", None),
    "C-07": ("conformance", "hard", "reply", None),
    "C-08": ("conformance", "hard", "reply", None),
    "C-09": ("conformance", "flag", "reply", None),
    "C-10": ("conformance", "hard", "reply", None),
    "C-11": ("conformance", "coverage", "reply", None),
    "C-12": ("conformance", "hard", "reply", None),
    "C-13": ("conformance", "hard", "reply", None),
    "C-14": ("conformance", "hard", "reply", None),
    "C-16": ("conformance", "hard", "reply", None),
    "R-01": ("runtime", "hard", "reply", None),
    "R-02": ("runtime", "hard", "reply", None),
    "R-04": ("runtime", "hard", "reply", None),
    "R-06": ("runtime", "hard", "reply", None),
    "M-01:undeclared": ("model-behavior", "rate", "call", "undeclared_call_rate"),
    "M-01:payload_invalid": ("model-behavior", "rate", "call", "payload_invalid_rate"),
    "M-01:server_tool_called": ("model-behavior", "flag", "call", None),
    "M-02": ("model-behavior", "rate", "reply", "degenerate_rate"),
    "M-03": ("model-behavior", "rate", "reply", "markup_leak_rate"),
    "M-05": ("model-behavior", "flag", "reply", None),
    "M-06": ("model-behavior", "rate", "reply", "empty_turn_rate"),
    "M-07": ("model-behavior", "flag", "reply", None),
    "H-01:aborted": ("harness-outcome", "rate", "call", "aborted_rate"),
    "H-01:arguments_rejected": ("harness-outcome", "rate", "call", "arguments_rejected_rate"),
    "H-01:unsupported": ("harness-outcome", "flag", "call", None),
    "H-01:unavailable_in_mode": ("harness-outcome", "flag", "call", None),
    "H-02": ("harness-outcome", "hard", "request", None),
    "H-03": ("harness-outcome", "flag", "reply", None),
    "H-04": ("harness-outcome", "hard", "reply", None),
}
Z_ONE_SIDED_95 = 1.6448536269514722


# The matrix's unit for rates. `session`: sessions with at least one event, out of all sessions.
# `turn`: each check's own unit, a reply (one model output) or, for call-level checks, a call.
RATE_UNITS = ("session", "turn")


def wilson(k: int, n: int):
    """One-sided 95% Wilson bounds (lower, upper) for k events in n units."""
    if n <= 0:
        return None
    z = Z_ONE_SIDED_95
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (centre - margin) / denom), min(1.0, (centre + margin) / denom)


def tree_sha256() -> str:
    """Identity of the validator: every file under scripts/validate plus its references."""
    entries = []
    for name in sorted(os.listdir(HERE)):
        path = os.path.join(HERE, name)
        if os.path.isfile(path) and not name.endswith(".pyc"):
            with open(path, "rb") as fh:
                entries.append(f"scripts/validate/{name}:{hashlib.sha256(fh.read()).hexdigest()}")
    for name in VALIDATOR_REFERENCES:
        path = os.path.join(SKILL, "references", name)
        if os.path.isfile(path):
            with open(path, "rb") as fh:
                entries.append(f"references/{name}:{hashlib.sha256(fh.read()).hexdigest()}")
    return hashlib.sha256("\n".join(entries).encode()).hexdigest()


# What each role of a verdict's input is read for: the population's own request and response
# shards, the boundary hours' requests (--requests) and responses (--responses), and the raw
# output (--raw).
EVIDENCE_ROLES = (("population", ("requests", "responses")), ("requests", ("requests",)),
                  ("responses", ("responses",)), ("raw", ("raw",)))  # fmt: skip
MANIFEST = "MANIFEST.sha256"


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def evidence(inputs: dict, workers: int = 1) -> dict:
    """The evidence a verdict read, by content (V-02).

    ``inputs`` maps each role of ``EVIDENCE_ROLES`` to the paths given for it. Every shard the
    passes read is listed with its role, size and SHA-256 (as stored, compressed or not).
    ``sha256`` is the SHA-256 of one line ``<role> <file sha256>`` per shard, sorted, each ending
    in a newline: it names the bytes judged in their roles, wherever they lie. A
    ``MANIFEST.sha256`` in a given directory or the one above it (a frozen snapshot's root) is
    recorded with its own SHA-256; it is not checked against the shards (I-03).
    """
    files = []
    for role, kinds in EVIDENCE_ROLES:
        paths = list(inputs.get(role) or [])
        files += [(role, path) for path in
                  sorted({s for kind in kinds for s in expand_shards(paths, kind)})]  # fmt: skip
    with ThreadPoolExecutor(max_workers=max(1, min(workers, 8))) as pool:
        digests = list(pool.map(file_sha256, [path for _, path in files]))
    shards = [{"role": role, "path": path, "bytes": os.path.getsize(path), "sha256": digest}
              for (role, path), digest in zip(files, digests)]  # fmt: skip
    lines = sorted(f"{s['role']} {s['sha256']}\n" for s in shards)
    manifests = {}
    for path in {p for paths in inputs.values() for p in paths or ()}:
        directory = os.path.abspath(path if os.path.isdir(path) else os.path.dirname(path) or ".")
        for where in (directory, os.path.dirname(directory)):
            candidate = os.path.join(where, MANIFEST)
            if os.path.isfile(candidate) and candidate not in manifests:
                manifests[candidate] = file_sha256(candidate)
    return {"sha256": hashlib.sha256("".join(lines).encode()).hexdigest(), "shards": shards,
            "bytes": sum(s["bytes"] for s in shards),
            "manifests": [{"path": p, "sha256": d} for p, d in sorted(manifests.items())]}  # fmt: skip


def revision_record(counts: dict, build_id) -> dict:
    """What the replies say served them, and the revision they were checked against (I-05).

    ``build_id`` is ``--build-id`` as given: null when it was not given, "" when it was empty.
    """
    seen = {k.split(":", 2)[2]: v for k, v in counts.items() if k.startswith("I-05:build_id:")}
    return {"declared_build_id": build_id, "checked": declared_revision(build_id) is not None,
            "replies_with_build_id": counts.get("I-05:with_build_id", 0),
            "replies_without_build_id": counts.get("I-05:missing_build_id", 0),
            "replies_with_other_build_id": counts.get("I-05:other_build_id", 0),
            "build_ids_seen": dict(sorted(seen.items()))}  # fmt: skip


def check_records(report: dict, thresholds: dict, harness: prof.Profile, population: str) -> list:
    counts = report["counts"]
    replies = counts.get("den:replies", 0)
    calls = counts.get("den:calls", 0)
    streams = counts.get("den:streams", 0)
    sessions = report.get("sessions", 0)
    unit_pref = (thresholds or {}).get("unit_for_rates") or "session"
    outcome = report.get("outcome_classes", {})
    answered = sum(
        v for k, v in outcome.items() if k not in ("pending", "unanswered", "unknown_profile")
    )
    records = []
    checks = dict(CHECKS)
    # Any other outcome class of the harness profile (tool_error, error, ...) is reported as a flag.
    for cls in outcome:
        if (
            cls not in ("result", "pending", "unanswered", "unknown_profile")
            and f"H-01:{cls}" not in checks
        ):
            checks[f"H-01:{cls}"] = ("harness-outcome", "flag", "call", None)
    for check, (claim, kind, unit, key) in checks.items():
        count = counts.get(check, 0)
        if check.startswith("H-01:"):
            count = outcome.get(check.split(":", 1)[1], 0)
        denominator = {
            "reply": replies,
            "call": calls,
            "request": counts.get("inventory:requests", 0),
        }[unit]
        if check.startswith("H-01:"):
            denominator = answered
        if check == "M-02" and counts.get("den:raw_shards", 0):
            # With raw output, only replies that could show the declared tokens count.
            denominator = counts.get("den:m02_observable", 0)
        record = {"id": check, "claim": claim, "kind": kind, "count": count, "unit": unit,
                  "denominator": denominator, "status": None, "reason": None,
                  "examples": report["examples"].get(check, [])}  # fmt: skip
        if kind == "coverage":
            elements = sorted(
                k.split(":", 2)[2] for k in counts if k.startswith(check + ":element:")
            )
            record["status"] = "BLOCKED" if count else "PASS"
            if count:
                record["reason"] = (
                    "protocol elements this validator has no rule for: " + ", ".join(elements)[:300]
                )
        elif kind == "flag":
            record["status"] = "FLAGGED" if count else "PASS"
        elif kind == "hard":
            record["status"] = "FAIL" if count else "PASS"
        else:
            threshold = None
            group = "harness_outcome" if claim == "harness-outcome" else "model_behavior"
            if isinstance(thresholds.get(group), dict):
                threshold = thresholds[group].get(key)
            k, n = count, denominator
            # Sessions are reported whatever the unit, so a turn-level verdict shows how many
            # sessions its turns came from.
            record["sessions"] = sessions
            record["sessions_with_events"] = report.get("sessions_with", {}).get(check, 0)
            if unit_pref not in RATE_UNITS:
                record.update(status="BLOCKED", threshold=None, interval=None,
                              reason=f"unknown unit_for_rates {unit_pref!r}: use one of "
                              + ", ".join(RATE_UNITS))  # fmt: skip
                records.append(record)
                continue
            if unit_pref == "session" and not sessions:
                record.update(status="BLOCKED", reason="session-level rate but no session identity",
                              unit="session", threshold=threshold, interval=None)  # fmt: skip
                records.append(record)
                continue
            if unit_pref == "session":
                k, n = report.get("sessions_with", {}).get(check, 0), sessions
                record["unit"] = "session"
                if check.startswith("H-01:"):
                    k = report.get("sessions_with", {}).get(check, 0)
            record.update(count=k, denominator=n)
            bounds = wilson(k, n)
            record["interval"] = [round(b, 6) for b in bounds] if bounds else None
            record["threshold"] = threshold
            if not isinstance(threshold, (int, float)):
                record["status"], record["reason"] = "BLOCKED", "threshold not set in the matrix"
            elif bounds is None:
                record["status"], record["reason"] = "BLOCKED", "no units observed"
            elif bounds[1] <= threshold:
                record["status"] = "PASS"
            elif bounds[0] > threshold:
                record["status"] = "FAIL"
            else:
                record["status"], record["reason"] = "BLOCKED", "insufficient n for the threshold"
        records.append(record)

    by_id = {r["id"]: r for r in records}
    # Evidence the checks need (fail closed).
    if counts.get("I-01:response_without_request", 0):
        for cid in ("C-03", "C-06", "C-07", "C-08"):
            if by_id[cid]["status"] == "PASS":
                by_id[cid].update(status="BLOCKED", reason="replies without their request: pass the "
                                  "boundary hour's request shards with --requests")  # fmt: skip
    no_next = counts.get("C-03:no_next_turn", 0)
    echoed = (counts.get("C-03:echo_equal", 0) + counts.get("C-03:echo_reserialized", 0)
              + counts.get("C-03:echo_rewritten", 0)
              + counts.get("C-03:text_echo_equal", 0) + counts.get("C-03:reasoning_echo_equal", 0)
              + by_id["C-03"]["count"])  # fmt: skip
    if echoed == 0:
        reason = ("replay has no next turn" if population == "replay"
                  else "no delivered call was echoed by a next turn")  # fmt: skip
        if by_id["C-03"]["status"] == "PASS":
            by_id["C-03"].update(status="NOT_APPLICABLE" if population in ("replay", "controls")
                                 or calls == 0 else "BLOCKED", reason=reason)  # fmt: skip
    by_id["C-03"]["no_next_turn"] = no_next
    by_id["C-03"]["echo_missing"] = counts.get("C-03:echo_missing", 0)
    # Calls the harness sent back: the same bytes (a Messages echo is an object: the same value),
    # the same value in other bytes, or changed only as its profile declares (`echo.tool_input`),
    # by rewrite.
    by_id["C-03"]["echo_equal"] = counts.get("C-03:echo_equal", 0)
    by_id["C-03"]["echo_reserialized"] = counts.get("C-03:echo_reserialized", 0)
    by_id["C-03"]["echo_rewritten"] = counts.get("C-03:echo_rewritten", 0)
    by_id["C-03"]["echo_rewritten_by"] = {
        k.split(":", 2)[2]: v
        for k, v in sorted(counts.items())
        if k.startswith("C-03:echo_rewritten:")
    }
    # Answers to a request the client sent again and was answered again: not what it went on with.
    by_id["C-03"]["superseded_answers"] = counts.get("C-03:superseded_answers", 0)
    # Of those, answers the harness dropped and asked again for with more appended (an empty turn).
    by_id["C-03"]["reasked_answers"] = counts.get("C-03:reasked_answers", 0)
    # Echoed texts and reasoning, and those not echoed: missing, by design (a side request's reply
    # never enters the conversation; a compacted one is replaced by a summary), or no next turn.
    for key in (
        "text_echo_equal",
        "text_echo_missing",
        "text_echo_unchecked",
        "text_not_echoed_side_request",
        "text_not_echoed_compacted",
        "reasoning_echo_equal",
        "reasoning_echo_missing",
        "reasoning_echo_unchecked",
        "reasoning_not_echoed_side_request",
        "reasoning_not_echoed_compacted",
        "text_no_next_turn",
        "reasoning_no_next_turn",
    ):
        by_id["C-03"][key] = counts.get("C-03:" + key, 0)
    if "M-01:payload_invalid" in by_id:
        # Payloads invalid only in ways the harness tolerates (profile tools.tolerated_payload).
        by_id["M-01:payload_invalid"]["tolerated"] = counts.get("M-01:payload_tolerated", 0)
    unchecked = [f"{counts[k]} echoed {what} were not compared"
                 for k, what in (("C-03:text_echo_unchecked", "texts"),
                                 ("C-03:reasoning_echo_unchecked", "reasoning items"))
                 if counts.get(k, 0)]  # fmt: skip
    if unchecked and by_id["C-03"]["status"] == "PASS":
        by_id["C-03"].update(
            status="BLOCKED", reason="harness profile `echo` is TODO: " + "; ".join(unchecked)
        )
    unread = sorted(k.rsplit(":", 1)[1] for k in counts if k.startswith("C-12:sdk_unavailable:"))
    if unread and by_id["C-12"]["status"] == "PASS":
        by_id["C-12"].update(status="BLOCKED", reason="no SDK reading for " + ", ".join(unread)
                             + ": install scripts/validate/requirements.txt")  # fmt: skip
    elif not counts.get("den:sdk_read", 0) and by_id["C-12"]["status"] == "PASS":
        by_id["C-12"].update(status="NOT_APPLICABLE", reason="no reply with a success status")
    for key in ("usage_vs_trace_prompt", "usage_vs_engine_output", "count_tokens_pairs",
                "next_input_pairs"):  # fmt: skip
        by_id["C-14"][key] = counts.get("den:" + key, 0)
    by_id["C-14"]["count_tokens_requests"] = counts.get("inventory:count_tokens_requests", 0)
    joined, unjoined = counts.get("den:raw_joined", 0), counts.get("C-13:unjoined", 0)
    # The denominator is the replies compared. A joined reply whose raw reading has a model
    # format anomaly (M-07) is not compared (`not_judged`).
    compared, not_judged = counts.get("den:raw_compared", 0), counts.get("C-13:not_judged", 0)
    # Replies whose engine id names several raw records (one per generation worker): joined to the
    # one that fits the reply's window and output count, or, with none or several fitting, unjoined.
    resolved = counts.get("C-13:resolved_ambiguous_id", 0)
    ambiguous = counts.get("C-13:unjoined_ambiguous_id", 0)
    # Joined raw records of generations the server stopped (partial or aborted), compared as such.
    stopped_raw = sum(v for k, v in counts.items() if k.startswith("C-13:stopped_raw:"))
    by_id["C-13"].update(denominator=compared, joined=joined, not_judged=not_judged,
                         unjoined=unjoined, resolved_ambiguous_id=resolved,
                         unjoined_ambiguous_id=ambiguous, stopped_raw=stopped_raw)  # fmt: skip
    # Replies never held against a raw record: cut by the client before their end (incomplete by
    # construction; the hook may write no record for the aborted generation), answered with an
    # HTTP error, lost by the recorder (R-06), or pending at the capture's end. With raw output,
    # joined + unjoined + these = replies.
    client_cut = counts.get("den:client_cut_replies", 0)
    by_id["C-13"].update(replies=replies, client_cut=client_cut,
                         http_error=counts.get("C-13:http_error", 0),
                         recorder_lost=counts.get("den:upstream_cut_replies", 0),
                         pending=counts.get("inventory:pending_at_capture_end", 0))  # fmt: skip
    shared = (f" ({ambiguous} by an id that names several raw records, not exactly one of which "
              "fits the reply)" if ambiguous else "")  # fmt: skip
    if by_id["C-13"]["status"] == "PASS":
        if not counts.get("den:raw_shards", 0):
            by_id["C-13"].update(status="NOT_APPLICABLE", reason="no raw output given (--raw): "
                                 "what the model wrote is not known")  # fmt: skip
        elif counts.get("C-13:no_native_reader", 0) or not counts.get("den:raw_records", 0):
            by_id["C-13"].update(status="BLOCKED", reason="raw output given, but the model profile "
                                 "has no native reader or no raw record was read")  # fmt: skip
        elif not joined:
            by_id["C-13"].update(status="BLOCKED", reason="no reply joined its raw record" + shared)
        elif not compared:
            by_id["C-13"].update(status="BLOCKED", reason=f"every joined reply ({joined}) had a model "
                                 "format anomaly (M-07): none was compared")  # fmt: skip
        else:
            uncovered = []
            if not_judged:
                uncovered.append(f"{not_judged} of {joined} joined replies had a model format "
                                 "anomaly (M-07) and were not compared")  # fmt: skip
            if unjoined:
                uncovered.append(f"{unjoined} replies joined no raw record{shared}; they are not "
                                 "covered")  # fmt: skip
            if client_cut:
                uncovered.append(f"{client_cut} replies the client cut before their end were not "
                                 "compared")  # fmt: skip
            if uncovered:
                by_id["C-13"]["reason"] = "; ".join(uncovered)
    # C-16: rejections the harness profile declares as capability retries are flagged, not failed.
    declared = counts.get("C-16:declared", 0)
    by_id["C-16"]["declared_rejections"] = declared
    by_id["C-16"]["declared_fields"] = {
        k.split(":", 2)[2]: v for k, v in counts.items() if k.startswith("C-16:declared_field:")
    }
    if declared and by_id["C-16"]["status"] == "PASS":
        by_id["C-16"].update(status="FLAGGED", reason=f"{declared} rejections the harness profile "
                             "declares as capability retries: each costs a round trip")  # fmt: skip
    # M-02 found nothing because it was told nothing: the model's own runaway tokens are unknown.
    if not report.get("model_degenerate_declared") and by_id["M-02"]["status"] == "PASS":
        by_id["M-02"].update(status="BLOCKED", reason="the model profile declares no degenerate "
                             "tokens (S1): only the default pattern, 20 or more [PAD] in a row, "
                             "was looked for")  # fmt: skip
    # Tokens named only by id show only on raw token ids: without them M-02 saw nothing.
    elif (report.get("model_degenerate_declared") and not report.get("model_degenerate_text_declared")
          and not counts.get("den:raw_shards", 0) and by_id["M-02"]["status"] == "PASS"):  # fmt: skip
        by_id["M-02"].update(status="BLOCKED", reason="the model profile names its degenerate tokens "
                             "by id only: pass --raw records that carry token_ids")  # fmt: skip
    if counts.get("C-10:profile_unknown", 0):
        by_id["C-10"].update(status="BLOCKED", reason="harness profile errors.overlength is TODO")
    elif counts.get("den:overlength", 0) == 0 and by_id["C-10"]["status"] == "PASS":
        by_id["C-10"].update(status="NOT_APPLICABLE", reason="no over-length reply observed")
    # A check whose situation never arose tested nothing: NOT_APPLICABLE, never PASS.
    for cid, key, reason in (
        ("C-07", "single_call_replies", "no request asked for at most one call"),
        ("C-08", "tool_choice_replies", "no request set tool_choice to none, required or a name"),
        ("H-04", "overlength_next", "no over-length reply was followed by a next turn"),
    ):
        by_id[cid][key] = counts.get("den:" + key, 0)
        if not by_id[cid][key] and by_id[cid]["status"] == "PASS":
            by_id[cid].update(status="NOT_APPLICABLE", reason=reason)
    # C-14 passes on the rules that had data; the ones that had none are named, not hidden.
    idle = [rule for rule, key in (("USG-05", "count_tokens_pairs"),
                                   ("USG-06", "usage_vs_engine_output"),
                                   ("USG-07", "usage_vs_trace_prompt"),
                                   ("USG-08", "next_input_pairs")) if not by_id["C-14"].get(key)]  # fmt: skip
    by_id["C-14"]["not_exercised"] = idle
    if idle and by_id["C-14"]["status"] == "PASS" and not by_id["C-14"].get("reason"):
        by_id["C-14"]["reason"] = "not exercised (no data): " + ", ".join(idle)
    if replies == 0:
        for record in records:
            if record["claim"] == "conformance" and record["status"] == "PASS":
                record.update(status="BLOCKED", reason="no replies")
    for rid, cut_key in (("R-01", "timeouts.first_event_s"), ("R-02", "timeouts.idle_s")):
        no_timing = counts.get(rid + ":no_timing", 0)
        cut, exact = prof.client_cut(harness, cut_key)
        beyond = counts.get(rid + ":beyond_known_bound", 0)
        by_id[rid]["client_cut_s"] = cut
        by_id[rid]["client_cut_exact"] = exact
        if by_id[rid]["status"] == "PASS":
            if streams == 0:
                by_id[rid].update(status="NOT_APPLICABLE", reason="no streamed replies")
            elif no_timing:
                by_id[rid].update(status="BLOCKED", reason=f"{no_timing} streamed replies without "
                                  "timing (record per-chunk times)")  # fmt: skip
            elif cut is None:
                by_id[rid].update(status="BLOCKED", reason=f"harness profile {cut_key} is TODO")
            elif beyond:
                by_id[rid].update(status="BLOCKED", reason=f"{beyond} streams waited at least "
                                  f"{cut:g} s, the longest the client is known to tolerate; "
                                  "its cut itself is unknown")  # fmt: skip
    # C-02 compares a stream with its final snapshot; Anthropic and Chat streams carry none.
    snapshots = streams - counts.get("den:no_streamed_snapshot", 0)
    by_id["C-02"]["snapshot_streams"] = snapshots
    if snapshots <= 0 and by_id["C-02"]["status"] == "PASS":
        by_id["C-02"].update(status="NOT_APPLICABLE", reason="no stream carries a final snapshot "
                             "to compare (the protocol has none)")  # fmt: skip
    # M-02 under raw output: say how many replies it could not see into.
    if counts.get("den:raw_shards", 0):
        by_id["M-02"]["raw_unobservable"] = counts.get("M-02:raw_unobservable", 0)
    # Degenerate runs the client never saw: only the raw record shows them, and their replies
    # ended without an answer. M-02 counts them; R-04 does not.
    by_id["R-04"]["stopped"] = counts.get("R-04:stopped", 0)
    # A rated outcome the harness profile does not name cannot be counted: a call of that kind
    # lands in another class, so the rate reads 0 whatever happens. Declared absent with
    # `{none: reason}`, the check is NOT_APPLICABLE.
    rules = prof.known((harness or {}).get("outcomes")) if harness else None
    rules = rules if isinstance(rules, dict) else {}
    for cid, (claim, kind, _, _) in CHECKS.items():
        if not cid.startswith("H-01:") or kind != "rate" or by_id[cid]["status"] == "FAIL":
            continue
        cls = cid.split(":", 1)[1]
        rule = rules.get(cls)
        if isinstance(rule, dict) and "none" in rule:
            by_id[cid].update(status="NOT_APPLICABLE", reason=str(rule["none"]))
        elif rule is None and rules:
            by_id[cid].update(status="BLOCKED", reason=f"the harness profile declares no `{cls}` "
                              "outcome: such a call would be counted in another class")  # fmt: skip
    # A flagged outcome the harness profile does not name cannot fire either. A measured profile
    # names every outcome its harness reports, so the flag is NOT_APPLICABLE; under a profile that
    # is not measured, whether the harness reports one is unknown, so it is BLOCKED.
    measured = (harness or {}).get("status") == "measured"
    for cid, (claim, kind, _, _) in CHECKS.items():
        if not cid.startswith("H-01:") or kind != "flag" or by_id[cid]["status"] != "PASS" \
                or not rules:  # fmt: skip
            continue
        cls = cid.split(":", 1)[1]
        rule = rules.get(cls)
        if isinstance(rule, dict) and "none" in rule:
            by_id[cid].update(status="NOT_APPLICABLE", reason=str(rule["none"]))
        elif rule is None and measured:
            by_id[cid].update(status="NOT_APPLICABLE", reason=f"the harness profile declares no "
                              f"`{cls}` outcome: this harness reports none to flag")  # fmt: skip
        elif rule is None:
            by_id[cid].update(status="BLOCKED", reason=f"the harness profile declares no `{cls}` "
                              f"outcome and its status is {harness.get('status')!r}, not measured: "
                              "whether the harness reports one is unknown")  # fmt: skip
    return records


def declared_revision(build_id) -> Optional[str]:
    """The revision ``--build-id`` declares, or None when it was not given or is blank."""
    return (build_id.strip() or None) if isinstance(build_id, str) else None


def claim_verdicts(records: list, report: dict, requested: list, harness: prof.Profile,
                   model: prof.Profile, build_id, tier: str, controls_ok: bool) -> dict:  # fmt: skip
    """``build_id`` is ``--build-id`` as given: None when it was not given."""
    counts = report["counts"]
    revision = declared_revision(build_id)
    verdicts, reasons = {}, {}
    for claim in requested:
        mine = [r for r in records if r["claim"] == claim]
        statuses = [r["status"] for r in mine]
        blocked = []
        if claim == "inventory":
            if not counts.get("inventory:requests") or not counts.get("inventory:responses"):
                blocked.append("no requests or no responses")
            if not harness:
                blocked.append("no harness profile")
            if revision and (
                counts.get("I-05:other_build_id") or counts.get("I-05:missing_build_id")
            ):
                blocked.append("records missing or not matching --build-id")
            elif not revision and counts.get("I-05:with_build_id"):
                # The records name the build that served them, and nothing was checked against
                # it: which code served is unproven (I-05), not complete.
                given = "not given" if build_id is None else "empty"
                blocked.append(f"no revision declared (--build-id {given}): "
                               f"{counts['I-05:with_build_id']} of {counts.get('den:replies', 0)} "
                               "replies carry a build_id that nothing was checked against "
                               "(I-05)")  # fmt: skip
            if counts.get("I-01:request_without_response"):
                blocked.append(f"partial: {counts['I-01:request_without_response']} requests "
                               "without a response record (I-01); for a window, pass the next "
                               "hour's response shards with --responses")  # fmt: skip
            verdicts[claim] = "BLOCKED" if blocked else "PASS"
        else:
            if claim == "harness-outcome":
                if harness.get("status") != "measured":
                    blocked.append(
                        f"harness profile status is {harness.get('status')!r}, not measured"
                    )
                if prof.outcome_rules(harness) is None:
                    blocked.append("harness profile outcomes are TODO")
                if counts.get("I-09:turn_without_thread"):
                    blocked.append("turns without thread identity")
            if claim == "runtime" and counts.get("I-05:missing_build_id"):
                blocked.append("records without build_id: deployed identity unproven (R-05)")
            if claim == "model-behavior" and not model:
                blocked.append("no model profile (markers and degenerate ids unknown)")
            # A claim BLOCKED by its checks names them, as one blocked at claim level does.
            blocked += [f"{r['id']}: {r.get('reason') or 'BLOCKED'}" for r in mine
                        if r["status"] == "BLOCKED"]  # fmt: skip
            if "FAIL" in statuses:
                verdicts[claim] = "FAIL"
            elif blocked or "BLOCKED" in statuses:
                verdicts[claim] = "BLOCKED"
            elif mine and all(s == "NOT_APPLICABLE" for s in statuses):
                verdicts[claim] = "NOT_APPLICABLE"
            else:
                verdicts[claim] = "PASS"
        if tier == "full" and not controls_ok and verdicts[claim] != "FAIL":
            verdicts[claim] = "BLOCKED"
            blocked.append("no passing control receipt for this validator tree (V-01)")
        reasons[claim] = blocked
    return {"claims": verdicts, "claim_blockers": reasons}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)  # fmt: skip
    ap.add_argument("--harness", required=True)
    ap.add_argument("--model")
    ap.add_argument("--matrix")
    ap.add_argument("--population", default="replay", choices=sorted(DEFAULT_CLAIMS))
    ap.add_argument("--claims", help="comma-separated; default depends on --population")
    ap.add_argument("--build-id")
    ap.add_argument("--tier", default="fast", choices=("fast", "full"))
    ap.add_argument("--controls-receipt")
    ap.add_argument("--requests", nargs="*", default=[])
    ap.add_argument("--responses", nargs="*", default=[],
                    help="next hours' response shards, read only to answer this population")  # fmt: skip
    ap.add_argument("--raw", nargs="*", default=[],
                    help="raw-*.jsonl shards or directories: what the model wrote (C-13)")  # fmt: skip
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1),
                    help="processes for the request and reply passes (1 = serial)")  # fmt: skip
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args(argv)
    try:
        harness = prof.load_profile(args.harness)
        model = prof.load_profile(args.model)
        thresholds, thresholds_sha256 = prof.load_thresholds(args.matrix)
        requested = args.claims.split(",") if args.claims else list(DEFAULT_CLAIMS[args.population])
        unknown = [c for c in requested if c not in CLAIMS]
        if unknown:
            raise ValueError(f"unknown claims {unknown}; choose from {CLAIMS}")
        report = pipeline.run(
            args.paths,
            harness,
            model,
            args.requests,
            declared_revision(args.build_id),
            args.workers,
            args.raw,
            args.responses,
        )
        read = evidence({"population": args.paths, "requests": args.requests,
                         "responses": args.responses, "raw": args.raw}, args.workers)  # fmt: skip
        tree = tree_sha256()
        sdks = sdk_diff.versions()
        controls_ok = False
        if args.controls_receipt:
            with open(args.controls_receipt) as fh:
                receipt = json.load(fh)
            # A receipt qualifies one validator tree read by one set of SDK versions.
            controls_ok = (
                bool(receipt.get("passed"))
                and receipt.get("validator_tree_sha256") == tree
                and receipt.get("sdk_versions") == sdks
            )
        records = check_records(report, thresholds, harness, args.population)
        verdict = claim_verdicts(records, report, requested, harness, model, args.build_id,
                                 args.tier, controls_ok)  # fmt: skip
    except (OSError, ValueError) as error:
        print(f"verdict.py: {type(error).__name__}: {error}", file=sys.stderr)
        return 3
    except Exception:  # noqa: BLE001 - an internal error must never look like a verdict
        traceback.print_exc()
        return 3

    # Every check that counted at least one event, plus every wire rule broken: what the
    # control matrix compares with each fault mode's expected set.
    tripped = sorted({r["id"] for r in records if r["count"] or r["status"] == "FLAGGED"}
                     | {rule for rules in report["by_rule"].values() for rule in rules})  # fmt: skip
    out = {
        "schema": "trtllm-serve-model-harness-validation/verdict@1",
        "cell": {
            "model": model.get("id"),
            "harness": harness.get("id"),
            "protocol": harness.get("protocol"),
        },  # fmt: skip
        "tier": args.tier,
        "population": {
            "kind": args.population,
            "paths": args.paths,
            # What was read (V-02): the shards by role and SHA-256, and the digest over them.
            "evidence_sha256": read["sha256"],
            "evidence_bytes": read["bytes"],
            "evidence_shards": read["shards"],
            "manifests": read["manifests"],
        },
        "revision": revision_record(report["counts"], args.build_id),
        "claims": verdict["claims"],
        "claim_blockers": verdict["claim_blockers"],
        "checks": [{k: v for k, v in r.items() if k != "examples"} for r in records],
        "tripped": tripped,
        "validator": {
            "tree_sha256": tree,
            "sdk_versions": sdks,
            "sdk_pinned": sdk_diff.pinned(),
            "controls_receipt": args.controls_receipt,
            "controls_passed": controls_ok,
        },  # fmt: skip
        "profiles": {
            "model": model.path,
            "model_sha256": model.sha256,
            "harness": harness.path,
            "harness_sha256": harness.sha256,
        },  # fmt: skip
        "thresholds_from": f"{args.matrix}@{thresholds_sha256}" if args.matrix else None,
    }
    report_out = {"schema": "trtllm-serve-model-harness-validation/report@1", **report,
                  "checks": records}  # fmt: skip
    try:
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "report.json"), "w") as fh:
            json.dump(report_out, fh, indent=1, sort_keys=True)
        with open(os.path.join(args.out_dir, "verdict.json"), "w") as fh:
            json.dump(out, fh, indent=1, sort_keys=True)
    except OSError as error:  # no verdict was written, which is not a FAIL
        print(f"verdict.py: {type(error).__name__}: {error}", file=sys.stderr)
        return 3
    except (TypeError, ValueError):  # a value JSON cannot hold: an internal error
        traceback.print_exc()
        return 3
    print(json.dumps({"claims": verdict["claims"], "tripped": tripped}))
    statuses = [verdict["claims"][c] for c in requested]
    if "FAIL" in statuses:
        return 1
    if "BLOCKED" in statuses:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
