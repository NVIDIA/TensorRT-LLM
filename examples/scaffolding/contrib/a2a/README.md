# Scaffolding A2A (Agent2Agent) Example

This example shows how a Scaffolding controller can delegate work to remote
agents that speak the [A2A (Agent2Agent) protocol](https://a2a-protocol.org/),
the agent-to-agent counterpart to the MCP tool-calling contrib. The generation
model decides which remote agent to call; the `A2AWorker` forwards the message
over A2A and feeds the reply back to the model for a final answer.

This is a **client-side** integration: Scaffolding acts as an A2A client that
consumes other agents. (Exposing a Scaffolding pipeline *as* an A2A server is a
possible follow-up.)

## Install

```bash
pip install a2a-sdk httpx uvicorn
```

`a2a-sdk` is only needed when actually talking to a remote agent; the contrib
imports it lazily, and the unit tests do not require it.

## Step 1: Start a remote A2A agent

A minimal reference agent server is included:

```bash
python weather_agent_server.py --port 9999
```

This exposes a `weather_agent` whose agent card is discoverable at
`http://0.0.0.0:9999/.well-known/agent-card.json`. You can also point the
example at any other A2A-compatible agent server.

## Step 2: Run the orchestrator

```bash
python a2a_run.py \
    --API_KEY YOUR_API_KEY \
    --base_url https://your-openai-compatible-endpoint/v1 \
    --model your-model \
    --agent_urls http://0.0.0.0:9999 \
    --prompt "What is the weather in LA?"
```

The generation model receives the remote agents as callable tools, delegates to
`weather_agent`, and summarizes its reply.

## Files

| File | Role |
|------|------|
| `a2a_run.py` | Scaffolding A2A orchestrator runner (client) |
| `weather_agent_server.py` | Minimal reference A2A agent server for local testing |

> `a2a-sdk` server/client APIs evolve across versions. The scripts target the
> SDK's published "helloworld" pattern; adjust imports if your installed version
> differs.
