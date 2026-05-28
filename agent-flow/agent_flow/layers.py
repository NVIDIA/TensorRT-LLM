from __future__ import annotations

from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, AsyncIterator, Callable

import anyio
from claude_agent_sdk import tool

from .backends import create_backend
from .backends.base import Backend, BackendClient, ResultEvent
from .config import AgentLayerConfig, HumanRequest, HumanRequestOption
from .console import (print_agent_completed, print_agent_failed,
                      print_agent_started, print_agent_text,
                      print_compact_boundary, print_human_input_request,
                      print_human_reply, print_rate_limit,
                      print_server_tool_call, print_session_init,
                      print_thinking, print_tool_call, print_user_prompt)
from .logger import Logger, get_logger
from .module import Module
from .runtime import PortalRunner, build_request
from .types import (AgentRequest, AgentResponse, AgentTextEvent,
                    CompactBoundaryEvent, RateLimitWarningEvent,
                    ServerToolCallEvent, SessionInitEvent, ThinkingEvent,
                    ToolCallEvent, UsageInfo)

PromptBuilder = Callable[[str], AgentRequest]


def _read_stdin(prompt: str = "> ") -> str:
    """Indirection so tests can monkeypatch the blocking stdin read."""
    return input(prompt)


def _read_human_reply(request: HumanRequest) -> str:
    """Read one ``ask_human`` reply from stdin, resolving choice options.

    When the request includes options the reply is interpreted in this
    order, falling through on each miss:

    1. A 1-based numeric index — returns the matching option's canonical
       label (so the agent never sees "2" as a tool result).
    2. The label text itself, matched case-insensitively after stripping
       whitespace — returns the canonical label.
    3. Anything else — returned verbatim as free-form text. This is the
       escape hatch for when none of the offered options captures what
       the human actually wants to say.

    Empty / whitespace-only input is returned as-is and the dispatcher
    normalizes it to ``None`` ("no answer").
    """
    raw = _read_stdin("> ")
    stripped = raw.strip()
    if request.options and stripped:
        try:
            idx = int(stripped)
        except ValueError:
            pass
        else:
            if 1 <= idx <= len(request.options):
                return request.options[idx - 1].label
        lowered = stripped.lower()
        for opt in request.options:
            if opt.label.strip().lower() == lowered:
                return opt.label
    return raw


class AgentLayer(Module):

    def __init__(
        self,
        config: AgentLayerConfig,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.prompt_builder = prompt_builder
        self._runner = PortalRunner()
        self._backend: Backend | None = None
        self._persistent_client: BackendClient | None = None
        self._persistent_client_stack: AsyncExitStack | None = None
        self._persistent_system_prompt: str | None = None

    @property
    def layer_name(self) -> str:
        return self.config.name or self.__class__.__name__

    def forward(self, content: str) -> str:
        if self.config.session.mode == "persistent":
            return self._runner.call(self._invoke_persistent, content)
        return anyio.run(self._invoke_once, content)

    async def aforward(self, content: str) -> str:
        if self.config.session.mode == "persistent":
            return await self._invoke_persistent(content)
        return await self._invoke_once(content)

    def _build_request(self, content: str) -> AgentRequest:
        if self.prompt_builder is None:
            return build_request(content,
                                 system_prompt=self.config.system_prompt)

        request = self.prompt_builder(content)
        return AgentRequest(
            content=request.content,
            system_prompt=(request.system_prompt if request.system_prompt
                           is not None else self.config.system_prompt),
            metadata=dict(request.metadata),
        )

    async def _ensure_backend(self) -> Backend:
        if self._backend is None:
            self._backend = create_backend(self.config.backend)
            await self._backend.__aenter__()
        return self._backend

    def _resolve_tools(self) -> list[Any] | None:
        """Return the tool list for ``backend.create_client``.

        Prepends the in-process ``ask_human`` MCP tool when human-input is
        enabled. Returns ``None`` when no tools are configured anywhere —
        preserving the existing behavior of the underlying backends,
        which treat ``None`` and ``[]`` slightly differently in some code
        paths.
        """
        base = list(self.config.backend.tools or [])
        if self.config.human_input_enabled:
            base.insert(0, self._build_ask_human_tool())
        return base or None

    def _resolve_disallowed_tools(self) -> list[str] | None:
        """Tools the agent must not call for this layer.

        When HITL is enabled we ban the built-in ``AskUserQuestion`` tool.
        It is handled inside the Claude Code CLI process, which has no UI
        on the SDK side and silently auto-defaults to "No answers
        selected" — so leaving it available means the agent's questions
        never reach the human. With it disallowed, the agent reaches for
        our ``ask_human`` MCP tool instead, whose handler reads the
        reply from stdin.
        """
        if self.config.human_input_enabled:
            return ["AskUserQuestion"]
        return None

    def _build_ask_human_tool(self):
        """Construct the ``ask_human`` MCP tool bound to this layer.

        The schema is a deliberate partial-alignment with Claude Code's
        built-in ``AskUserQuestion``: same single-question shape with
        ``question``, optional ``header`` (short chip), and optional
        ``options`` of ``{label, description}``. We omit ``questions[]``
        (multi-question), ``multiSelect``, and ``preview`` because they
        either don't translate to stdin or would force a structured
        reply where a single string suffices. The goal is to lean on
        the model's training around ``AskUserQuestion`` without paying
        the complexity tax of full alignment.
        """
        layer = self

        schema = {
            "type": "object",
            "properties": {
                "question": {
                    "type":
                    "string",
                    "description":
                    ("The complete question to ask the human. Should "
                     "be clear, specific, and end with a question "
                     "mark."),
                },
                "header": {
                    "type":
                    "string",
                    "description":
                    ("Optional very short label (≤12 chars) shown as a "
                     "chip/tag next to the question. Examples: \"Auth "
                     "method\", \"Library\", \"Approach\"."),
                },
                "options": {
                    "type":
                    "array",
                    "description":
                    ("Optional list of mutually exclusive choices. When "
                     "provided, the human is shown a numbered selection "
                     "panel; their reply is the matching option "
                     "``label`` (or free-form text if they type "
                     "something that doesn't match — the framework "
                     "always allows that escape hatch). Omit ``options`` "
                     "for free-form input."),
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {
                                "type":
                                "string",
                                "description":
                                ("Display text for this option. Should "
                                 "be concise (1-5 words)."),
                            },
                            "description": {
                                "type":
                                "string",
                                "description":
                                ("Explanation of what this option means "
                                 "or what will happen if chosen."),
                            },
                        },
                        "required": ["label", "description"],
                    },
                },
            },
            "required": ["question"],
        }

        @tool(
            "ask_human",
            ("Ask the human operator a question and wait for their "
             "reply. This is the drop-in replacement for "
             "``AskUserQuestion`` in this app: supply a clear "
             "``question``, optionally a short ``header`` chip, and — "
             "when the answer is one of a small set of choices — an "
             "``options`` list of ``{label, description}`` entries. The "
             "reply is returned as free text (the chosen ``label``, or "
             "whatever the human typed)."),
            schema,
        )
        async def ask_human(args):
            raw_options = args.get("options") or []
            options = tuple(
                HumanRequestOption(
                    label=str(opt.get("label", "")),
                    description=str(opt.get("description", "")),
                ) for opt in raw_options if isinstance(opt, dict))
            reply = await layer._dispatch_human_request(
                HumanRequest(
                    layer_name=layer.layer_name,
                    prompt=str(args.get("question", "")),
                    options=options,
                    header=str(args.get("header", "")),
                ))
            text = reply if reply else "(no response from human)"
            return {"content": [{"type": "text", "text": text}]}

        return ask_human

    @asynccontextmanager
    async def _temporary_client(
        self, system_prompt: str | None
    ) -> AsyncIterator[tuple[Backend, BackendClient]]:
        backend = create_backend(self.config.backend)
        async with backend:
            async with backend.create_client(
                    system_prompt=system_prompt or "",
                    model=self.config.backend.model,
                    tools=self._resolve_tools(),
                    hooks=self.config.backend.hooks,
                    disallowed_tools=self._resolve_disallowed_tools(),
                    extra_mcp_servers=self.config.backend.extra_mcp_servers,
            ) as client:
                yield backend, client

    async def _ensure_persistent_client(
            self, system_prompt: str | None) -> BackendClient:
        if self._persistent_system_prompt is None:
            self._persistent_system_prompt = system_prompt
        elif self._persistent_system_prompt != system_prompt:
            raise ValueError(
                "Persistent AgentLayer sessions require a stable system prompt."
            )

        if self._persistent_client is None:
            backend = await self._ensure_backend()
            stack = AsyncExitStack()
            await stack.__aenter__()
            client = await stack.enter_async_context(
                backend.create_client(
                    system_prompt=system_prompt or "",
                    model=self.config.backend.model,
                    tools=self._resolve_tools(),
                    hooks=self.config.backend.hooks,
                    disallowed_tools=self._resolve_disallowed_tools(),
                    extra_mcp_servers=self.config.backend.extra_mcp_servers,
                ))
            self._persistent_client = client
            self._persistent_client_stack = stack
        return self._persistent_client

    async def _drop_persistent_client(self) -> None:
        if self._persistent_client_stack is not None:
            await self._persistent_client_stack.aclose()
        self._persistent_client = None
        self._persistent_client_stack = None
        self._persistent_system_prompt = None

    async def _execute(
        self,
        client: BackendClient,
        request: AgentRequest,
        logger: Logger,
    ) -> AgentResponse:
        sink = logger.console
        result_text = ""
        usage: UsageInfo | None = None
        async for event in client.send_message(request.content):
            if isinstance(event, ResultEvent):
                result_text = event.text
                usage = event.usage
            elif isinstance(event, ToolCallEvent):
                if self.config.print_activity:
                    print_tool_call(self.layer_name, event, sink)
            elif isinstance(event, ServerToolCallEvent):
                if self.config.print_activity:
                    print_server_tool_call(self.layer_name, event, sink)
            elif isinstance(event, ThinkingEvent):
                if self.config.print_activity:
                    print_thinking(self.layer_name, event, sink)
            elif isinstance(event, AgentTextEvent):
                if self.config.print_activity:
                    print_agent_text(self.layer_name, event, sink)
            elif isinstance(event, SessionInitEvent):
                if self.config.print_activity:
                    print_session_init(self.layer_name, event, sink)
            elif isinstance(event, RateLimitWarningEvent):
                if self.config.print_activity:
                    print_rate_limit(self.layer_name, event, sink)
            elif isinstance(event, CompactBoundaryEvent):
                if self.config.print_activity:
                    print_compact_boundary(self.layer_name, event, sink)
            else:
                raise TypeError(
                    f"Unexpected backend event: {type(event).__name__}")

        return AgentResponse(
            content=result_text,
            metadata=dict(request.metadata),
            usage=usage,
        )

    async def _run_with_client(
        self,
        request: AgentRequest,
        client: BackendClient,
        backend: Backend,
    ) -> str:
        logger = get_logger()
        sink = logger.console

        if self.config.print_activity:
            print_agent_started(
                self.layer_name,
                self.config.backend.kind,
                self.config.backend.model,
                sink,
                version=backend.version() or None,
                reasoning_effort=backend.reasoning_effort() or None,
            )
            print_user_prompt(self.layer_name, request.content, sink)

        try:
            response = await self._execute(client, request, logger)
        except Exception as exc:
            if self.config.print_activity:
                print_agent_failed(self.layer_name, exc, sink)
            raise
        else:
            if self.config.print_activity:
                print_agent_completed(self.layer_name, sink, response.usage)
            return response.content

    async def _invoke_once(self, content: str) -> str:
        request = self._build_request(content)
        if request.system_prompt is None:
            request.system_prompt = self.config.system_prompt

        async with self._temporary_client(request.system_prompt) as (backend,
                                                                     client):
            return await self._run_with_client(request, client, backend)

    async def _invoke_persistent(self, content: str) -> str:
        request = self._build_request(content)
        if request.system_prompt is None:
            request.system_prompt = self.config.system_prompt

        try:
            client = await self._ensure_persistent_client(request.system_prompt)
            assert self._backend is not None
            return await self._run_with_client(request, client, self._backend)
        except Exception:
            await self._drop_persistent_client()
            raise

    async def _dispatch_human_request(self,
                                      request: HumanRequest) -> str | None:
        """Surface an ``ask_human`` request to the human via stdin.

        Prints a panel so the user sees the agent's question and any
        options, then reads one reply from stdin, prints another panel
        reflecting the reply (or ``(no reply)`` when the input was
        empty), and returns the normalized reply text — ``None`` when
        the input was empty. The stdin read runs synchronously inside
        the awaiting coroutine: ``ask_human`` is an exclusive
        interaction (the agent is waiting for the human), so blocking
        the event loop here is not a contention concern, and avoiding
        the worker-thread hop sidesteps environments where
        ``anyio.to_thread.run_sync`` cannot reliably spawn a thread.
        """
        sink = get_logger().console if self.config.print_activity else None

        if self.config.print_activity:
            print_human_input_request(self.layer_name, request, sink)

        raw = _read_human_reply(request)
        stripped = raw.strip() if isinstance(raw, str) else ""
        normalized: str | None = stripped or None

        if self.config.print_activity:
            print_human_reply(self.layer_name, request, normalized or "", sink)
        return normalized

    async def _aclose_owned_resources(self) -> None:
        await self._drop_persistent_client()
        if self._backend is not None:
            await self._backend.__aexit__(None, None, None)
            self._backend = None

    async def _aclose_self(self) -> None:
        if self._runner.started:
            try:
                self._runner.call(self._aclose_owned_resources)
            finally:
                self._runner.close()
            return
        await self._aclose_owned_resources()
