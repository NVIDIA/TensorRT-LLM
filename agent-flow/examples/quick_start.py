from __future__ import annotations

from agent_flow import (CLAUDE_CODE_DEFAULT_MODEL, AgentLayer, AgentLayerConfig,
                        BackendConfig, SessionConfig)

if __name__ == "__main__":
    agent = AgentLayer(
        AgentLayerConfig(
            name="Agent",
            backend=BackendConfig(kind="claude-code",
                                  model=CLAUDE_CODE_DEFAULT_MODEL),
            session=SessionConfig(mode="stateless"),
        ))
    agent("introduce the project")
