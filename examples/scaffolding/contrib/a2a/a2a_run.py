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
"""Run a Scaffolding A2A orchestrator against one or more remote A2A agents.

The generation side uses an OpenAI-compatible endpoint (any vendor, or a local
``trtllm-serve``); the orchestration side talks the Agent2Agent protocol via
``A2AWorker``. See README.md for how to start a sample remote agent server.
"""

import argparse
import asyncio

from openai import AsyncOpenAI

from tensorrt_llm.scaffolding import OpenaiWorker, ScaffoldingLlm
from tensorrt_llm.scaffolding.contrib.a2a import A2AController, A2AWorker
from tensorrt_llm.scaffolding.contrib.mcp.chat_handler import chat_handler
from tensorrt_llm.scaffolding.contrib.mcp.chat_task import ChatTask


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base_url",
        type=str,
        default="https://dashscope.aliyuncs.com/compatible-mode/v1",
        help="OpenAI-compatible base URL for the generation model.",
    )
    parser.add_argument("--model", type=str, default="qwen-plus-latest")
    parser.add_argument("--API_KEY", type=str)
    parser.add_argument(
        "--agent_urls",
        type=str,
        nargs="+",
        default=["http://0.0.0.0:9999"],
        help="Base URLs of the remote A2A agents to orchestrate.",
    )
    parser.add_argument("--prompt", type=str, default="What is the weather like today in LA?")
    return parser.parse_args()


async def main():
    args = parse_arguments()

    client = AsyncOpenAI(api_key=args.API_KEY, base_url=args.base_url)
    generation_worker = OpenaiWorker(client, args.model)
    generation_worker.register_task_handler(ChatTask, chat_handler)

    a2a_worker = await A2AWorker.init_with_urls(args.agent_urls)

    controller = A2AController()
    llm = ScaffoldingLlm(
        controller,
        {
            A2AController.WorkerTag.GENERATION: generation_worker,
            A2AController.WorkerTag.A2A: a2a_worker,
        },
    )

    future = llm.generate_async(args.prompt)
    result = await future.aresult()
    print(f"\nresult is {result.outputs[0].text}\n")

    print("shutting down...")
    llm.shutdown()
    generation_worker.shutdown()
    await a2a_worker.async_shutdown()
    print("shut down done")


if __name__ == "__main__":
    asyncio.run(main())
