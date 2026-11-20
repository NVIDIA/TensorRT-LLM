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
"""Build the pytest commands for isolated runs and reruns from the stage's base command."""

import re
import shlex

# Leading NAME=value shell assignments, kept verbatim when rebuilding a command
# so values such as ${PYTHONPATH:-} still expand and stay assignments.
_ENV_ASSIGNMENT_RE = re.compile(
    r"""\s*[A-Za-z_][A-Za-z0-9_]*=(?:"[^"]*"|'[^']*'|[^\s"'])*(?=\s|$)"""
)


def rebuild_pytest_command(base_cmd, drop_patterns, append_args):
    """Drop args matching any prefix in drop_patterns, then append append_args.

    For tokens of the form ``--flag=value`` the single token is dropped; for
    ``--flag value`` the following value token is also dropped (when it does
    not look like another flag).

    Shell pipeline prefixes (e.g. ``unset VAR && ``) are preserved verbatim
    so that shlex.join does not quote ``&&`` into a literal argument.

    Args:
        base_cmd: The original pytest command string.
        drop_patterns: Argument prefixes to strip from base_cmd.
        append_args: Tokens to append after stripping (already individually
            shell-safe; quoting is handled by ``shlex.join``).

    Returns:
        Rebuilt pytest command string.
    """
    # Preserve shell prefix (everything before the last " && ") verbatim so
    # shlex.join never quotes && into '&&' and breaks the pipeline.
    shell_prefix = ""
    cmd_part = base_cmd
    if " && " in base_cmd:
        and_parts = base_cmd.split(" && ")
        shell_prefix = " && ".join(and_parts[:-1]) + " && "
        cmd_part = and_parts[-1]

    # Keep leading env assignments verbatim: shlex.join would quote a value like
    # PYTHONPATH=/x:${PYTHONPATH:-}, and bash then runs it as a command name.
    env_end = 0
    while match := _ENV_ASSIGNMENT_RE.match(cmd_part, env_end):
        env_end = match.end()
    shell_prefix += cmd_part[:env_end].strip() + " " if env_end else ""
    cmd_part = cmd_part[env_end:]

    parts = []
    tokens = shlex.split(cmd_part)
    skip_next = False
    for i, token in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue
        if any(token.startswith(p) for p in drop_patterns):
            if "=" not in token and i + 1 < len(tokens) and not tokens[i + 1].startswith("-"):
                skip_next = True
            continue
        parts.append(token)
    parts.extend(append_args)
    return shell_prefix + shlex.join(parts)


def build_rerun_command(base_cmd, test_list, xml_path, csv_path, reruns):
    """Build a rerun pytest command by stripping split/cov args and replacing test-list/output args.

    Args:
        base_cmd: The original pytest command string.
        test_list: Path to the rerun test list file.
        xml_path: Path for the rerun results XML.
        csv_path: Path for the rerun results CSV.
        reruns: Number of reruns (passed to --reruns flag).

    Returns:
        Modified pytest command string.
    """
    drop_patterns = [
        "--splitting-algorithm",
        "--splits",
        "--group",
        "--cov",
        "--test-list",
        "--csv",
        "--periodic-junit-xmlpath",
    ]
    append_args = [
        f"--test-list={test_list}",
        f"--csv={csv_path}",
        f"--periodic-junit-xmlpath={xml_path}",
    ]
    if reruns > 0:
        append_args.extend(["--reruns", str(reruns)])
    return rebuild_pytest_command(base_cmd, drop_patterns, append_args)


def build_isolated_command(base_cmd, test_list, xml_path, csv_path, stage_name):
    """Build an isolated test command by replacing test-list/output/prefix args.

    Args:
        base_cmd: The original pytest command string.
        test_list: Path to the single-test list file.
        xml_path: Path for the isolated results XML.
        csv_path: Path for the isolated results CSV.
        stage_name: Stage name passed as --test-prefix, so isolated results and
            unfinished_test.txt entries carry the same "<stage>/" prefix as regular runs.

    Returns:
        Modified pytest command string.
    """
    drop_patterns = ["--test-list", "--test-prefix", "--csv", "--periodic-junit-xmlpath"]
    append_args = [
        f"--test-list={test_list}",
        f"--test-prefix={stage_name}",
        f"--csv={csv_path}",
        f"--periodic-junit-xmlpath={xml_path}",
    ]
    return rebuild_pytest_command(base_cmd, drop_patterns, append_args)
