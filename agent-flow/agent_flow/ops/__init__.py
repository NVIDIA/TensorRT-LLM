"""Operations tooling for long autonomous agent-flow runs.

These are coordinator-side utilities, not part of a workflow graph: a
reservation table for shared machine allocations, a role-addressed notice
queue, a persistent in-container command dispatcher, a background-job runner
and a run dashboard. Every module is driven by one config file — see
``agent_flow/ops/config.py`` and ``agent-flow-ops.example.toml``.
"""

from agent_flow.ops.config import OpsConfig, OpsConfigError, load_config

__all__ = ["OpsConfig", "OpsConfigError", "load_config"]
