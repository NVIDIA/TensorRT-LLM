# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import asyncio

from openai import AsyncOpenAI

from tensorrt_llm.scaffolding import execution_trace
from tensorrt_llm.scaffolding import replay as scaffolding_replay
from tensorrt_llm.scaffolding import worker as scaffolding_worker
from tensorrt_llm.scaffolding.contrib.open_deep_research import (
    supervisor as open_deep_research_supervisor,
)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace_path", type=str, default="execution_trace.json")
    parser.add_argument("--openai_api_key", type=str, default="tensorrt_llm")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--model", type=str, default="Qwen3/Qwen3-30B-A3B")
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--max_parallel_requests", type=int, default=1024)
    parser.add_argument("--latency_scale", type=float, default=1.0)
    return parser.parse_args()


async def main():
    args = parse_arguments()
    trace = execution_trace.ExecutionTrace.load(args.trace_path)
    client = AsyncOpenAI(api_key=args.openai_api_key, base_url=args.base_url)

    generation_worker = scaffolding_worker.TRTOpenaiWorker(client, args.model)
    prototype_controller = open_deep_research_supervisor.create_open_deep_research_controller(
        max_tokens=args.max_tokens,
    )
    llm = scaffolding_replay.create_replay_scaffolding_llm(
        trace=trace,
        generation_worker=generation_worker,
        prototype_controller=prototype_controller,
        latency_scale=args.latency_scale,
        max_parallel_requests=args.max_parallel_requests,
    )

    try:
        future = llm.generate_async(trace.prompt)
        result = await future.aresult()
        assert result.outputs[0].text is not None
    finally:
        llm.shutdown()
        generation_worker.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
