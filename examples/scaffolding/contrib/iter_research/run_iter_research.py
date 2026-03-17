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

"""Runner script for IterResearch with TensorRT-LLM scaffolding.

Usage:
    # Terminal 1: Start MCP server
    cd examples/scaffolding/contrib/iter_research/IterResearchMCP
    uv run iter_research_tools.py --config ../config.yaml

    # Terminal 2: Start LLM server
    trtllm-serve <hf_model> --port 8000

    # Terminal 3: Run (reads the same config.yaml)
    python examples/scaffolding/contrib/iter_research/run_iter_research.py \\
        --config examples/scaffolding/contrib/iter_research/config.yaml
"""

import argparse
import asyncio
from pathlib import Path

import yaml
from openai import AsyncOpenAI

from tensorrt_llm.scaffolding import (
    ChatTokenCounter,
    MCPWorker,
    TaskMetricsCollector,
    TaskTimer,
    TRTOpenaiWorker,
)
from tensorrt_llm.scaffolding.contrib.iter_research import (
    create_iter_research_scaffolding_llm,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="IterResearch with TensorRT-LLM scaffolding")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.yaml")
    parser.add_argument("--openai_api_key", type=str, default=None)
    parser.add_argument("--base_url", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--mcp_url", type=str, default=None)
    parser.add_argument("--max_turn", type=int, default=None)
    parser.add_argument("--max_tokens", type=int, default=None)
    parser.add_argument("--question", type=str, default=None,
                        help="Research question to ask")
    parser.add_argument("--enable_statistics", action="store_true")
    parser.add_argument("--enable_tracing", action="store_true")
    return parser.parse_args()


def _load_config(config_path: str | None) -> dict:
    """Load configuration from YAML file."""
    if config_path and Path(config_path).is_file():
        with open(config_path) as f:
            return yaml.safe_load(f) or {}
    return {}


async def main():
    args = parse_arguments()
    cfg = _load_config(args.config)

    # CLI flags override config.yaml values; config.yaml overrides defaults.
    openai_api_key = args.openai_api_key or cfg.get("openai_api_key",
                                                     "tensorrt_llm")
    base_url = args.base_url or cfg.get("base_url",
                                        "http://localhost:8000/v1")
    model = args.model or cfg.get("model", "Qwen/Qwen2.5-72B-Instruct")
    mcp_port = cfg.get("mcp_port", 8083)
    mcp_host = cfg.get("mcp_host", "0.0.0.0")
    mcp_url = args.mcp_url or cfg.get(
        "mcp_url", f"http://{mcp_host}:{mcp_port}/sse")
    max_turn = args.max_turn or cfg.get("max_turn", 25)
    max_tokens = args.max_tokens or cfg.get("max_tokens", 16384)

    client = AsyncOpenAI(api_key=openai_api_key, base_url=base_url)
    generation_worker = TRTOpenaiWorker(client, model)

    mcp_worker = MCPWorker.init_with_urls([mcp_url])
    await mcp_worker.init_in_asyncio_event_loop()

    llm = create_iter_research_scaffolding_llm(
        generation_worker,
        mcp_worker,
        max_tokens=max_tokens,
        max_turn=max_turn,
        enable_statistics=args.enable_statistics,
        enable_tracing=args.enable_tracing,
    )

    question = args.question or (
        "What are the latest advances in large language model inference "
        "optimization? Compare the approaches of vLLM, TensorRT-LLM, and "
        "SGLang in terms of performance, features, and architecture.")

    print(f"Question: {question}")
    print(f"Max turns: {max_turn}")
    print("Starting IterResearch...")

    future = llm.generate_async(question)
    result = await future.aresult()

    assert result.outputs[0].text is not None
    print("\n" + "=" * 80)
    print("FINAL ANSWER:")
    print("=" * 80)
    print(result.outputs[0].text)

    if args.enable_statistics:
        token_info = ChatTokenCounter.get_global_info()
        print(f"\nToken counting info: {token_info}")
        timer_info = TaskTimer.get_global_info()
        print(f"Timer info: {timer_info}")
        TaskMetricsCollector.export_to_json("iter_research_metrics.json")

    if args.enable_tracing and result.task_collections:
        tracer = result.task_collections.get("execution_tracer")
        if tracer:
            trace = tracer.export_trace(question)
            trace.save("iter_research_trace.json")
            print("Execution trace saved to iter_research_trace.json")

    llm.shutdown()
    generation_worker.shutdown()
    mcp_worker.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
