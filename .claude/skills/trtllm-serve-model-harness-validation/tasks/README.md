<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Local tasks for real-harness sessions

Eight small tasks for an S4 run (`references/smoke.md` section 2). With them, a first run of
Codex, Claude Code or pi against a trtllm-serve endpoint needs no tasks of its own. Each task is
a directory:
- `prompt.md`: the one paragraph the agent gets, sent as is;
- `workspace/`: the files it works on, copied fresh into every session.

**Nothing grades them.** They exist for realistic multi-turn tool use. The validator judges the
traffic (`scripts/validate/verdict.py --population smoke`), not the agent's answer or the files it
leaves behind.

## The tasks

| Task | The agent | What it forces | Tools |
|---|---|---|---|
| `t01-region-totals` | Totals four regional CSV files. A fifth file the README lists does not exist | Several reads in one turn; a missing file; an answer computed from what it read | read |
| `t02-settings-layers` | Learns a loader's merge rules from its code, then works out the effective settings of three layered files. The fourth layer does not exist | Several reads in one turn; a missing file; a computed answer; a sub-agent where the harness has them (optional) | read |
| `t03-wordstats-bug` | Fixes the bug behind 2 failing tests of 6 | A command that exits non-zero; an edit; tests | shell, edit |
| `t04-extract-helper` | Moves a helper copied into three modules to a new module. One test module fails to import until it exists | Edits to several files and a new file; a command that exits non-zero; tests | shell, edit |
| `t05-log-audit` | Generates a 3,000-line log and reads it 200 lines at a time, 15 reads, noting what it finds in a file after each one; answers from the notes | Long tool outputs read in parts; repeated edits; a computed answer. **Reaches compaction with the context window set to 32k** (below) | shell, edit |
| `t06-orders-report` | Generates a 2,500-row CSV whose malformed rows crash the report, then makes the report skip and count them, as the tests (3 of 5 failing at first) expect | A crash; a large file; edits; tests; an answer read off the report | shell, edit |
| `t07-precision-option` | Adds a command-line option across the code, the usage text and the tests | Edits to several files; a usage error; tests | shell, edit |
| `t08-rotate-script` | Fixes a shell script that fails 2 checks of 5 | A command that exits non-zero; an edit; shell tests | shell, edit |

- **Tools.** `read`: read-only tools are enough. `shell, edit`: the agent runs commands and changes
  files, so run it only in a disposable container or VM (`references/smoke.md` section 0).
- **Requirements.** Bash, coreutils and Python 3 with its standard library. No network, no GPU,
  nothing to install. A session writes only inside its workspace.
- **Tests** run from the workspace root: `python3 -m unittest -v` (t03, t04, t06, t07) or
  `bash test_rotate.sh` (t08). The failures at the start of t03, t04, t06 and t08 are part of the
  task.
- **Generated inputs.** t05 and t06 start with `python3 make_data.py`, which writes the same file
  on every run:
  - t05: `station.log`, 3,000 lines, 141 KB, about 52k to 69k tokens (DeepSeek-V3 and Qwen3
    tokenizers);
  - t06: `orders.csv`, 2,500 rows, 53 KB, about 31k to 42k tokens.

## Running them

**The pi kit** reads `tasks.tsv` (`id<TAB>kind<TAB>size<TAB>dir`, each dir relative to the file),
and gives pi `read`, `bash`, `edit` and `write` in its sandbox:

```bash
<skill>/scripts/pi_kit/bin/pikit run --run-dir <run> --tasks <skill>/tasks/tasks.tsv \
  --sessions 24 <the options of "Running a cell" in scripts/pi_kit/README.md>
```

- `--sessions 24` gives each task three sessions: sessions take the tasks in turn.
- The image (`PIKIT_IMAGE`) needs bash, coreutils and Python 3.

**The session runners** (`scripts/run_codex_sessions.sh`, `run_claude_sessions.sh`,
`run_pi_sessions.sh`) read `runner_tasks.tsv`: one `<workspace dir><TAB><prompt>` line per task.

- **The dirs are relative to this directory.** A runner takes relative paths from where it is
  started, so start it from here, or make the dirs absolute first, from here:
  `sed "/^#/!s|^|$PWD/|" runner_tasks.tsv > <run>/runner_tasks.tsv`. The runners skip blank
  lines and `#` lines.
- **Point `BASE_URL` at the recorder** in front of the server, or at a server that writes its own
  request trace (`references/smoke.md` section 2).
- **Allow the tools the tasks need.** With the Claude Code and pi runners' default tools, which
  are read-only, only t01 and t02 can finish:

  | Runner | Settings for the whole set |
  |---|---|
  | `run_codex_sessions.sh` | The default, `SANDBOX=workspace-write`. In a container that forbids Linux namespaces, `danger-full-access` (the runner's header) |
  | `run_claude_sessions.sh` | `ALLOWED="Read Glob Grep Bash Edit Write Agent"` and `DISALLOWED="NotebookEdit WebFetch WebSearch"`. `Agent`, declared by Claude Code 2.1.289 (`contracts/claude-code.md` section 9), starts sub-agents, for t02 |
  | `run_pi_sessions.sh` | `TOOLS=read,bash,edit,write` |

  ```bash
  cd <skill>/tasks
  BASE_URL=http://<recorder host>:<port>/v1 MODEL=<served name> CONTEXT_WINDOW=<window> \
    ../scripts/run_codex_sessions.sh runner_tasks.tsv /abs/path/run/codex
  ```

- **More sessions per task:** repeat the lines, e.g.
  `for i in 1 2 3; do cat runner_tasks.tsv; done > <run>/runner_tasks_x3.tsv`, and run it from
  here as above.

**Compaction (t05).** Set the harness's context window to 32,768: `--context-window 32768` in the
kit, `CONTEXT_WINDOW=32768` for the Codex and pi runners, and Claude Code's setting from
`references/smoke.md` section 2.
- pi compacts 16,384 tokens below its window, and Codex when its count reaches the window. Both
  happen well before the log has been read, so the session compacts at least once.
- Keep the window at or below the served `max_seq_len`. On main, pi compacts after a request past
  it; an over-length first request, or a refusal pi does not read, ends the session instead
  (`scripts/pi_kit/README.md`).
- `pikit status` counts compactions per session.

**Judge** a run like any S4 run: a verdict over its trace (`references/smoke.md` section 3).

## Changing the set

- **`prompt.md` is one line without tabs and without a license header.** The kit sends the file
  as is, and `runner_tasks.tsv` carries it on one line.
- **`runner_tasks.tsv` starts with its license header as `#` lines,** which the runners skip.
  After changing a prompt or adding a task, regenerate it from here:

  ```bash
  { printf '%s\n' "# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved." \
        "# SPDX-License-Identifier: Apache-2.0"
    for d in t*/; do printf '%s\t%s\n' "${d}workspace" "$(cat "${d}prompt.md")"; done; } > runner_tasks.tsv
  ```

  A new task also needs its line in `tasks.tsv`.
- **Content.** Synthetic only:
  - no credentials and no token-like strings;
  - no personal or company names;
  - no code from other projects.

  Standard library only, and no network.
- **Size.** No file over 50 KB. Generate larger inputs with a `make_data.py` the prompt runs
  first, from a fixed seed.
- **Checked in like any other file.** The Python passes the repository's lint. Code files carry
  the license header; data files do not.
