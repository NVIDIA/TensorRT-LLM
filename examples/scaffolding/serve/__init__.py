"""Agent Client-Server for remote agent execution over WebSocket.

This package provides a client-server architecture for running scaffolding
agents (Coder, Deep Research) backed by TensorRT-LLM. The server
orchestrates the LLM pipeline while tool calls are relayed to the client
for local execution.

Modules:
    client: WebSocket client with local tool execution (no GPU required).
    server: WebSocket server with scaffolding + TRT-LLM generation.
    tools:  Local tool implementations (filesystem, shell, patch, planning).
"""
