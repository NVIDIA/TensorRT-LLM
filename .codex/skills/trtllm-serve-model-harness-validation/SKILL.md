---
name: trtllm-serve-model-harness-validation
description: >-
  Support a model x agent-harness pair on trtllm-serve and prove it, as an
  agent loop: capture the harness's wire contract, adapt the model's
  rendering, parsers and stop rules, replay recorded traffic, smoke-test the
  real harness, and let an independent validator issue fail-closed verdicts
  per cell (inventory, conformance, runtime, model behavior, harness outcome)
  over Responses, Anthropic Messages and Chat Completions traffic. Works for
  any model and any agent harness: each is a profile, and the shipped
  profiles (GLM-5.x, Kimi-K3, DeepSeek-V4.1, Qwen3; Codex, Claude Code, pi)
  are examples. Triggers on: new model for agents, new harness, an agent
  harness against trtllm-serve, harness contract, tool parser for agents,
  reasoning parser, replay traces, smoke test a harness, validate a serving
  change, stream/final mismatch, parallel_tool_calls, model x harness
  matrix, agent support loop.
license: Apache-2.0
metadata:
  author: NVIDIA Corporation
---

# trtllm-serve-model-harness-validation: where the skill lives

Codex finds skills under `.codex/skills`. This skill lives under `.claude/skills`, so this file only
points there.

Read `.claude/skills/trtllm-serve-model-harness-validation/SKILL.md`, from the root of this
repository, and follow it. Every path it names is relative to that directory.

This file's `name` and `description` are the skill's own; the skill's tests check that they stay
equal.
