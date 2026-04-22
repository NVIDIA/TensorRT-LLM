"""Promptfoo prompt builder.

Loads perf-test-sync.md, strips Claude Code specific sections, and appends the
user request.

Stripped sections:
  - YAML frontmatter between leading `---` markers
  - `# Persistent Agent Memory` section and everything after it (Claude Code
    memory infrastructure, not relevant to prompt-quality evaluation)
"""

import os
import re

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_AGENT_MD = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "..", "agents", "perf-test-sync.md"))


def _load_agent_body() -> str:
    with open(_AGENT_MD, "r", encoding="utf-8") as f:
        text = f.read()
    text = re.sub(r"\A---\n.*?\n---\n", "", text, count=1, flags=re.DOTALL)
    text = text.split("# Persistent Agent Memory", 1)[0].rstrip()
    return text


def build(context: dict) -> str:
    user_prompt = context["vars"]["prompt"]
    agent_body = _load_agent_body()
    return f"{agent_body}\n\n## User request\n\n{user_prompt}\n"
