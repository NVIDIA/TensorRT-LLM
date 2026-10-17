# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The turn this process runs itself.

"In process" names where the *call* happens, not where the model does. The
backend still spawns a CLI; what makes this path different from the mailbox
is that the orchestrator calls a layer and gets the answer back on the same
stack, with no file queue and no second party in between.

It is the shorter path and the default. It stops working when the host
cannot authenticate the CLI it spawns — a read-only credential store, or a
refresh token rotated out from under a child — which is a host failure that
presents as a model that returned nothing.
"""

from __future__ import annotations

import time
from typing import Any, Callable, ClassVar

from .base import TurnExecutionError, TurnExecutor, TurnOutcome, TurnRequest

#: Resolves the callable that runs a turn for one request. Given a request,
#: return something invocable as ``layer(prompt) -> str`` — in practice an
#: :class:`agent_flow.AgentLayer`, but nothing here requires that.
LayerProvider = Callable[[TurnRequest], Any]


class InProcessTurnExecutor(TurnExecutor):
    """Run a turn by calling a layer directly.

    The provider indirection is what keeps this package free of workflow
    vocabulary: role-to-layer construction (which backend, which model,
    which tools, stateless or persistent) is a property of the workflow, so
    the workflow supplies it and this executor only decides *when* to call.
    """

    name: ClassVar[str] = "inprocess"

    def __init__(self, layer_provider: LayerProvider) -> None:
        self._provider = layer_provider

    def bound(self, session: Any) -> "InProcessTurnExecutor":
        """Pin every turn to ``session``, whatever the request asks for.

        Used where the caller already owns the layer — a batch that builds
        one agent per item and closes it afterwards. Without this the caller
        would have to ask whether its transport has a session concept, which
        is exactly the branch this package exists to remove.
        """
        return InProcessTurnExecutor(lambda request: session)

    def run(self, request: TurnRequest) -> TurnOutcome:
        layer = self._provider(request)
        if layer is None:
            raise TurnExecutionError(
                f"no layer available for role {request.role!r}; the provider returned None"
            )
        started = time.monotonic()
        text = layer(request.prompt)
        return TurnOutcome(
            role=request.role,
            text=text if isinstance(text, str) else str(text),
            transport=self.name,
            decision_ref=request.decision_file,
            session_ref=getattr(layer, "layer_name", None) or None,
            duration_s=time.monotonic() - started,
        )


__all__ = ["InProcessTurnExecutor", "LayerProvider"]
