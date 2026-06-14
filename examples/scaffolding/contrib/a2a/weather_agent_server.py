# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A minimal reference A2A agent server used to exercise the A2A contrib.

This follows the ``a2a-sdk`` "helloworld" server pattern and exposes a single
``weather_agent`` that returns a canned reply. Run it with::

    pip install a2a-sdk uvicorn
    python weather_agent_server.py --port 9999

then point ``a2a_run.py --agent_urls http://0.0.0.0:9999`` at it.

Note: ``a2a-sdk`` server APIs evolve; this script targets the published
helloworld example. Adjust imports if your installed SDK version differs.
"""

import argparse

import uvicorn
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill
from a2a.utils import new_agent_text_message


class WeatherAgentExecutor(AgentExecutor):
    """Returns a canned weather reply regardless of the incoming message."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        await event_queue.enqueue_event(new_agent_text_message("It is sunny in LA, around 75F."))

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("cancel is not supported by this agent")


def build_app(host: str, port: int) -> A2AStarletteApplication:
    skill = AgentSkill(
        id="weather",
        name="weather",
        description="Returns the current weather for a location.",
        tags=["weather"],
        examples=["What is the weather in LA?"],
    )
    agent_card = AgentCard(
        name="weather_agent",
        description="A demo agent that reports the weather.",
        url=f"http://{host}:{port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=False),
        skills=[skill],
    )
    request_handler = DefaultRequestHandler(
        agent_executor=WeatherAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    return A2AStarletteApplication(agent_card=agent_card, http_handler=request_handler)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9999)
    args = parser.parse_args()

    app = build_app(args.host, args.port)
    uvicorn.run(app.build(), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
