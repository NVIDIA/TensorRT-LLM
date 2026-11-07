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

import os
import re
import sys

# Keep in sync with .github/workflows/pr-check.yml.
PR_TITLE_REGEX = re.compile(
    r"^(\[(None|[A-Z0-9]+-[0-9]+|#[0-9]+|https:\/\/nvbugs\/[0-9]+)\])"
    r" *"
    r"(\[[a-z0-9]+\])"
    r" "
    r"(([^ ].*)?[^ ])$"
)

JIRA_TICKET_REGEX = re.compile(r"^[A-Z0-9]+-[0-9]+$")
GITHUB_ISSUE_REGEX = re.compile(r"^#[0-9]+$")
NVBUGS_URL_REGEX = re.compile(r"^https://nvbugs/[0-9]+$")

FORMAT_GUIDE = """\
Expected PR title format:
  [JIRA ticket/NVBugs ID/GitHub issue/None][type] Summary

Valid ticket formats:
  - JIRA ticket: [TRTLLM-1234] or [FOOBAR-123] for other FOOBAR project
  - NVBugs ID: [https://nvbugs/1234567]
  - GitHub issue: [#1234]
  - No ticket: [None]

Valid types (lowercase): [fix], [feat], [doc], [infra], [chore], etc.

Examples:
  - [TRTLLM-1234][feat] Add new feature
  - [https://nvbugs/1234567][fix] Fix some bugs
  - [#1234][doc] Update documentation
  - [None][chore] Minor clean-up"""


def _diagnose_ticket(ticket_inner: str) -> list[str]:
    errors: list[str] = []
    if not ticket_inner:
        errors.append("Ticket segment cannot be empty. Use [None] when no ticket applies.")
        return errors

    if ticket_inner == "None":
        return errors

    if JIRA_TICKET_REGEX.fullmatch(ticket_inner):
        return errors

    if re.fullmatch(r"[A-Za-z]+", ticket_inner):
        errors.append(
            f"Found '[{ticket_inner}]' where a ticket is expected. "
            "Titles start with a ticket ([TRTLLM-1234], [#1234], "
            "[https://nvbugs/1234567], or [None]) followed by a type "
            f"(e.g. [None][{ticket_inner.lower()}] Summary)."
        )
        return errors

    if re.fullmatch(r"[A-Za-z0-9]+-[0-9]+", ticket_inner):
        if ticket_inner != ticket_inner.upper():
            errors.append(
                f"JIRA ticket '[{ticket_inner}]' must use an uppercase project key "
                f"(e.g. [{ticket_inner.split('-', 1)[0].upper()}-{ticket_inner.split('-', 1)[1]}])."
            )
        else:
            errors.append(
                f"Ticket '[{ticket_inner}]' is not a recognized format. "
                "Use [PROJECT-123] for JIRA tickets."
            )
        return errors

    if re.fullmatch(r"[A-Z0-9]+[0-9]+", ticket_inner):
        errors.append(
            f"Ticket '[{ticket_inner}]' looks like a JIRA ID but is missing a hyphen. "
            "Use [PROJECT-123], e.g. [TRTLLM-1234]."
        )
        return errors

    if ticket_inner.startswith("#"):
        if re.fullmatch(r"#[0-9]*", ticket_inner):
            errors.append("GitHub issue ticket must be [#1234] with a numeric issue ID.")
        else:
            errors.append(f"GitHub issue ticket '[{ticket_inner}]' must match [#1234].")
        return errors

    if "nvbugs" in ticket_inner:
        errors.append(
            "NVBugs ticket must be the full URL in brackets, e.g. [https://nvbugs/1234567]."
        )
        return errors

    errors.append(
        f"Ticket '[{ticket_inner}]' is not valid. Use [TRTLLM-1234], [#1234], "
        "[https://nvbugs/1234567], or [None]."
    )
    return errors


def validate_pr_title(title: str) -> list[str]:
    """Return a list of human-readable validation errors (empty if valid)."""
    if not title or title.isspace():
        return ["PR title is empty."]

    if title != title.lstrip():
        return ["PR title must not contain leading whitespace."]

    if PR_TITLE_REGEX.fullmatch(title):
        return []

    errors: list[str] = []

    ticket_match = re.match(r"^\[([^\]]*)\]", title)
    if not ticket_match:
        if title.startswith("["):
            errors.append("Missing closing ']' after the ticket segment.")
        else:
            errors.append(
                "Title must start with a ticket in square brackets "
                "(e.g. [TRTLLM-1234], [#1234], [https://nvbugs/1234567], or [None])."
            )
        return errors

    ticket_inner = ticket_match.group(1)
    if ticket_inner != "None" and not (
        JIRA_TICKET_REGEX.fullmatch(ticket_inner)
        or GITHUB_ISSUE_REGEX.fullmatch(ticket_inner)
        or NVBUGS_URL_REGEX.fullmatch(ticket_inner)
    ):
        errors.extend(_diagnose_ticket(ticket_inner))

    rest = title[ticket_match.end() :]
    rest_after_spaces = rest.lstrip(" ")

    type_match = re.match(r"^\[([^\]]*)\]", rest_after_spaces)
    if not type_match:
        if rest_after_spaces.startswith("["):
            errors.append("Missing closing ']' after the type segment.")
        elif not rest_after_spaces:
            errors.append("Missing type segment after the ticket (e.g. [fix], [feat], [doc]).")
        else:
            preview = rest_after_spaces.split(" ", 1)[0]
            errors.append(
                f"Expected a type in square brackets after the ticket, but found "
                f"'{preview}'. Example: [TRTLLM-1234][fix] Summary."
            )
        return errors

    type_inner = type_match.group(1)
    if not re.fullmatch(r"[a-z0-9]+", type_inner):
        if not type_inner:
            errors.append("Type segment cannot be empty (e.g. [fix]).")
        elif type_inner != type_inner.lower():
            errors.append(f"Type '[{type_inner}]' must be all lowercase (e.g. [fix], not [Fix]).")
        else:
            errors.append(
                f"Type '[{type_inner}]' contains invalid characters; "
                "use only lowercase letters and digits (e.g. [fix], [feat])."
            )

    summary_part = rest_after_spaces[type_match.end() :]
    if not summary_part:
        errors.append(
            "Missing summary after the type segment. Add a space followed by a short description."
        )
    elif not summary_part.startswith(" "):
        errors.append(
            "Expected a space between the type segment and the summary (e.g. [fix] Fix the bug)."
        )
    else:
        summary = summary_part[1:]
        if not summary:
            errors.append("Summary cannot be empty.")
        elif summary != summary.rstrip():
            errors.append("Summary must not end with trailing whitespace.")
        elif summary.startswith(" "):
            errors.append(
                "The last square bracket must be followed by a single space, but there are multiple."
            )

    return errors


def main() -> None:
    title = os.environ.get("PR_TITLE", "")
    errors = validate_pr_title(title)

    if not errors:
        print(f"PR title is valid: {title!r}")
        return

    print(f"PR title validation failed for: {title!r}")
    print()
    for error in errors:
        print(f"::error::{error}")
    print()
    print(FORMAT_GUIDE)
    sys.exit(1)


if __name__ == "__main__":
    main()
