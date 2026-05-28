"""Demo of the optional human-in-the-loop knob on ``AgentLayer``.

Setting ``human_input_enabled=True`` registers an ``ask_human`` MCP tool
the agent can invoke mid-turn whenever it needs information from the
human, and disables Claude Code's built-in ``AskUserQuestion`` so the
agent's questions actually reach the human via stdin instead of being
silently auto-defaulted by the CLI.

The framework never injects a post-turn "did I get it right?" prompt of
its own — once the agent finishes its turn, the layer returns. The
agent decides when to ask the human; the framework does not add a
required human-input step on top.

Run it:

    python examples/human_in_the_loop.py

Replies are read from stdin. When the agent supplies an ``options``
list, the panel renders a numbered choice table — the human can pick a
number, type a label, or write free-form text.
"""
from __future__ import annotations

from agent_flow import (CLAUDE_CODE_DEFAULT_MODEL, AgentLayer, AgentLayerConfig,
                        BackendConfig, SessionConfig)

if __name__ == "__main__":
    agent = AgentLayer(
        AgentLayerConfig(
            name="Planner",
            backend=BackendConfig(
                kind="claude-code",
                model=CLAUDE_CODE_DEFAULT_MODEL,
            ),
            session=SessionConfig(mode="persistent"),
            system_prompt=(
                "You are a planning assistant.\n\n"
                "If anything about the user's request is ambiguous, call "
                "the `ask_human` tool BEFORE producing a plan. Treat it "
                "as a drop-in for AskUserQuestion: supply a clear "
                "`question`, an optional short `header` chip "
                "(≤12 chars), and — when the answer is one of a small "
                "set of choices — an `options` list of "
                "`{label, description}` entries. The human's reply "
                "(the chosen label, or free text) comes back as the "
                "tool result; treat it as authoritative and proceed.\n\n"
                "Do not use the AskUserQuestion tool — it is intentionally "
                "disabled in this app."),
            human_input_enabled=True,
        ))

    with agent:
        # Intentionally vague: the agent should call ``ask_human`` mid-turn
        # before producing a plan. Once the plan prints, the layer returns
        # — there's no auto-injected "refine?" prompt.
        agent("Draft a 3-day onboarding plan for a new hire on our team.")
