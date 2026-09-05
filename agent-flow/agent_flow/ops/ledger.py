"""Parse the roles' hand-written verdict ledger into a per-gate scoreboard.

A long autonomous run records every gate attempt as a markdown row in
``PASS-LEDGER.md`` under the workspace, and every red gate's reason as a line
in ``GATE-REASONS.md``. The dashboard reads them so the scoreboard reflects the
roles' own verdicts rather than a guess from log files.

The rows are written by hand, by several roles, over days. They are therefore
irregular: the column set varies, gate ids carry trailing words, and a verdict
cell can hold two words at once. The rules below are the ones the running
system uses, and each exists because of a specific way a row was misread:

* The gate id and the commit are matched by PATTERN, not by column position.
* A row whose verdict cell contains neither a pass nor a fail word (RUNNING,
  VOID, half-written) is skipped.
* A cell holding both ("baseline PASS / enabled UNRUN", "FAIL then FIXED")
  counts as a FAIL. A false red costs one look; a false green never gets
  looked at, which is the asymmetry the whole rule set is built around.
* A gate id may be followed by variant words ("AC-05 enabled", "AC-13 all
  three legs"). A FAIL on any such row is a red for the gate. A PASS counts
  only when the trailing words name the gate itself or a configuration
  variant of it — never a supporting artefact ("assets", "analysis",
  "preflight", ...), because passing the scaffolding is not passing the gate.

The vocabulary (gate id pattern, pass/fail words, the not-a-gate word list) is
configurable; the defaults are the ones in use.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

GATE_ID_RE = r"AC-\d+"
VERDICT_RE = r"\b(PASS|FAIL|DOES NOT HOLD|FALSE GREEN|UNRUN)\b"
BAD_WORDS = ("fail", "unrun", "does not hold", "false green")
# Trailing words that make a row describe supporting work, not the gate.
NOT_GATE_WORDS = (
    "supporting",
    "assets",
    "analysis",
    "diagnostic",
    "harness",
    "logic",
    "launch",
    "control",
    "stack",
    "corroboration",
    "preflight",
    "defect",
    "pairing",
    "review",
    "matrix",
    "leg",
    "replay",
    "experiment",
    "attribution",
    "candidate",
    "audit",
    "prep",
    "reference",
)
# Trailing words that keep a row a gate row despite matching NOT_GATE_WORDS.
GATE_ANYWAY_WORDS = ("qualification",)
# Verdict-cell words that demote a PASS to "not a gate verdict".
PASS_DISQUALIFIERS = (
    "preflight",
    "supporting",
    "not a gate",
    "reference stage",
    "partial",
    "no reproduction",
    "negative",
)

REASON_LINE = re.compile(
    r"^-\s*(" + GATE_ID_RE + r")\s*\|\s*([^|]*?)\s*\|\s*([^|]*?)\s*\|\s*(.+?)\s*$"
)


@dataclass
class LedgerRules:
    """Vocabulary the parser matches against (see the module docstring)."""

    gate_id: str = GATE_ID_RE
    verdict: str = VERDICT_RE
    bad_words: tuple[str, ...] = BAD_WORDS
    not_gate_words: tuple[str, ...] = NOT_GATE_WORDS
    gate_anyway_words: tuple[str, ...] = GATE_ANYWAY_WORDS
    pass_disqualifiers: tuple[str, ...] = PASS_DISQUALIFIERS
    year: int = field(default_factory=lambda: datetime.now().year)


def gate_reasons(workspace: Path, filename: str = "GATE-REASONS.md") -> dict:
    """Role-maintained one-liners: why a gate is not passing yet.

    One line per gate id::

        - AC-03 | 05-04 05:10 | reviewer | <why it is red now; next action>

    Last line per id wins; the text is consumed verbatim by the dashboard.
    """
    path = Path(workspace) / filename
    out: dict = {}
    if not path.exists():
        return out
    for line in path.read_text(errors="replace").splitlines():
        m = REASON_LINE.match(line.strip())
        if m:
            out[m.group(1)] = {"time": m.group(2), "role": m.group(3), "text": m.group(4)}
    return out


def _is_gate_pass(gate_tail: str, verdict_lower: str, rules: LedgerRules) -> bool:
    variant_ok = any(w in gate_tail for w in rules.gate_anyway_words) or not any(
        w in gate_tail for w in rules.not_gate_words
    )
    if not (variant_ok and "pass" in verdict_lower):
        return False
    if any(k in verdict_lower for k in rules.pass_disqualifiers):
        return False
    # "not a pass", "does not pass": a negation right before the word.
    return "not" not in verdict_lower.split("pass")[0][-8:]


def parse_rows(text: str, rules: LedgerRules | None = None) -> dict:
    """``{gate: [{epoch, time, run, commit, rc, pass}]}``, oldest first.

    Rows look like ``| <mm-dd HH:MM> | <gate> | ... | <commit> | ... |
    **PASS** | <log> | <note> |`` with a variable column set.
    """
    rules = rules or LedgerRules()
    out: dict = {}
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 4:
            continue
        gcell = next((c for c in cells[1:3] if re.match(rf"^{rules.gate_id}\b", c)), None)
        if not gcell:
            continue
        gate = re.match(rules.gate_id, gcell).group(0)
        gtail = gcell[len(gate) :].strip().lower()
        verdict = next((c for c in cells[3:] if re.search(rules.verdict, c)), "")
        if not verdict:
            continue
        vl = verdict.lower()
        if any(k in vl for k in rules.bad_words):
            ok = False
        elif _is_gate_pass(gtail, vl, rules):
            ok = True
        else:
            continue  # supporting / half / analysis PASS rows are not gate verdicts
        m = re.match(r"(\d{2})-(\d{2})\s+~?(\d{2}):(\d{2})", cells[0])
        if not m:
            continue
        try:
            when = datetime(
                rules.year, int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
            ).timestamp()
        except ValueError:
            continue
        commit = next((c for c in cells if re.match(r"^[0-9a-f]{7,12}$", c)), "")
        log = next((c for c in cells if re.search(r"\.(log|out)\b", c)), "")
        # Ledger rows carry minute precision and are written after the run's
        # exit-code file, so they are placed at the END of their minute: the
        # roles' verdict outranks a log record stamped in the same minute.
        out.setdefault(gate, []).append(
            {
                "epoch": when + 59,
                "time": cells[0].lstrip("~"),
                "rc": "0" if ok else "fail",
                "pass": ok,
                "run": "ledger:"
                + (log.split("/")[-1].rsplit(".", 1)[0] or verdict.strip("*")[:24]),
                "commit": commit,
            }
        )
    for rows in out.values():
        rows.sort(key=lambda r: r["epoch"])
    return out


def ledger_rows(
    workspace: Path, filename: str = "PASS-LEDGER.md", rules: LedgerRules | None = None
) -> dict:
    """:func:`parse_rows` over the workspace's ledger file."""
    path = Path(workspace) / filename
    if not path.exists():
        return {}
    return parse_rows(path.read_text(errors="replace"), rules)


def latest_passes(rows_by_gate: dict) -> dict:
    """Newest PASS row per gate, dropping gates whose only rows are failures."""
    out = {}
    for gate, rows in rows_by_gate.items():
        ok = [r for r in rows if r["pass"]]
        if ok:
            out[gate] = ok[-1]
    return out


def scoreboard(rows_by_gate: dict) -> dict:
    """``{gate: 'pass' | 'fail'}`` from the NEWEST row of each gate.

    Newest-wins, not sticky: a gate that passed and then failed reads red, and
    a re-run that passes turns it green again.
    """
    return {
        gate: ("pass" if rows[-1]["pass"] else "fail")
        for gate, rows in rows_by_gate.items()
        if rows
    }
