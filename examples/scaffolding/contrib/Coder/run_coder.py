import argparse
import asyncio
import io
import logging
import os
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

from apiary_client import AsyncApiary
from openai import AsyncOpenAI

from tensorrt_llm.scaffolding import (
    ApiaryMCPWorker,
    QueryCollector,
    TaskMetricsCollector,
    TRTOpenaiWorker,
)
from tensorrt_llm.scaffolding.contrib.Coder import create_coder_scaffolding_llm

logger = logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run the Coder agent for code generation and modification tasks."
    )
    parser.add_argument("--openai_api_key", type=str, default="tensorrt_llm")
    parser.add_argument("--base_url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--model", type=str, default="Qwen3/Qwen3-30B-A3B")
    parser.add_argument(
        "--mcp_url",
        type=str,
        default="http://0.0.0.0:8083/sse",
        help="URL for the Coder Apiary MCP server (coder_mcp.py)",
    )
    parser.add_argument(
        "--max_mcp_connections",
        type=int,
        default=200,
        help="Maximum concurrent sandbox connections",
    )
    parser.add_argument(
        "--image",
        type=str,
        default="ubuntu:22.04",
        help="Docker image registered with the Apiary daemon and used for the sandbox session",
    )
    parser.add_argument(
        "--apiary_url",
        type=str,
        default=os.getenv("APIARY_URL", "http://127.0.0.1:8080"),
        help="Apiary daemon URL used to register the sandbox image",
    )
    parser.add_argument(
        "--apiary_token",
        type=str,
        default=os.getenv("APIARY_API_TOKEN"),
        help="Bearer token for the Apiary daemon",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt for the Coder agent. If not provided, a default example is used.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=16384,
        help="Maximum tokens for generation",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=50,
        help="Maximum tool-calling iterations",
    )
    parser.add_argument("--enable_query_collector", action="store_true")
    parser.add_argument(
        "--enable_statistics",
        action="store_true",
        help="Print live task metrics and save the final summary as text. Does not write metrics JSON.",
    )
    parser.add_argument("--enable_tracing", action="store_true")
    parser.add_argument(
        "--trace_output_dir",
        type=str,
        default=None,
        help="Directory for trace outputs when --enable_tracing is set",
    )
    return parser.parse_args()


def _dump_task_metrics_summary(output_dir: Path) -> None:
    buf = io.StringIO()
    with redirect_stdout(buf):
        TaskMetricsCollector.print_summary()
    summary = buf.getvalue()
    print(summary, end="")

    summary_path = output_dir / "coder.metrics.txt"
    summary_path.write_text(summary, encoding="utf-8")
    print(f"Task metrics summary saved to {summary_path}")


async def register_image(image: str, apiary_url: str, apiary_token: str | None) -> None:
    """Ensure ``image`` is registered with the Apiary daemon before dispatching work.

    Apiary returns 404 for any session targeting an unregistered image. This
    helper performs a single-image load via :class:`AsyncApiary` so the MCP
    server can create sandboxes immediately.
    """
    apiary = AsyncApiary(
        apiary_url=apiary_url,
        apiary_token=apiary_token,
        images=[image],
    )
    try:
        if not await apiary.health_check(retries=10, interval=1.0):
            raise RuntimeError(
                f"Apiary daemon at {apiary_url} is not reachable. "
                "Start it with `apiary init && apiary daemon --bind ...`."
            )
        status = await apiary.load()
        if status is not None and image in status.failed:
            reason = next(
                (
                    entry.get("reason")
                    for entry in status.failed_images
                    if entry.get("name") == image
                ),
                "unknown",
            )
            raise RuntimeError(f"Failed to register image {image!r} with Apiary: {reason}")
        logger.info("Apiary image %s registered", image)
    finally:
        await apiary.close()


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_arguments()
    trace_output_dir: Path | None = None
    if args.enable_tracing or args.enable_statistics:
        if args.trace_output_dir:
            trace_output_dir = Path(args.trace_output_dir)
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            trace_output_dir = Path(f"coder_trace_{timestamp}")
        trace_output_dir.mkdir(parents=True, exist_ok=True)

    await register_image(args.image, args.apiary_url, args.apiary_token)

    client = AsyncOpenAI(api_key=args.openai_api_key, base_url=args.base_url)

    generation_worker = TRTOpenaiWorker(client, args.model)

    mcp_worker = ApiaryMCPWorker(
        args.mcp_url,
        max_connections=args.max_mcp_connections,
    )

    if args.enable_statistics:
        TaskMetricsCollector.reset()
        print(
            "Statistics enabled: "
            f"model={args.model}, base_url={args.base_url}, max_iterations={args.max_iterations}, "
            f"max_tokens={args.max_tokens}"
        )

    llm = create_coder_scaffolding_llm(
        generation_worker,
        mcp_worker,
        max_tokens=args.max_tokens,
        max_iterations=args.max_iterations,
        enable_statistics=args.enable_statistics,
        enable_tracing=args.enable_tracing,
    )

    if args.prompt:
        prompt = args.prompt
    else:
        prompt = "Implement a collective communication library in C++"

    print(f"Running Coder agent with prompt:\n{prompt}\n")
    if trace_output_dir is not None:
        print(f"Trace output directory: {trace_output_dir}")
    print("-" * 50)

    try:
        future = llm.generate_async(prompt)
        mcp_worker.set_scope_params(future.id, image=args.image)
        result = await future.aresult()

        assert result.outputs[0].text is not None
        print("\nFinal output:\n" + result.outputs[0].text)

        if args.enable_query_collector:
            QueryCollector.get_global_info()
            print("Query info dumped to query_result.json!")

        if args.enable_tracing and result.task_collections:
            tracer = result.task_collections.get("execution_tracer")
            if tracer:
                trace = tracer.export_trace()
                assert trace_output_dir is not None
                trace_path = trace_output_dir / "coder.trace.json"
                full_trace_path = trace_output_dir / "coder.full.trace.json"
                trace.save(str(trace_path))
                trace.save(str(full_trace_path), full=True)
                print(f"Execution trace saved to {trace_path}")
                print(f"Full execution trace saved to {full_trace_path}")
    finally:
        if args.enable_statistics:
            assert trace_output_dir is not None
            _dump_task_metrics_summary(trace_output_dir)

        await mcp_worker.async_shutdown()
        llm.shutdown()
        generation_worker.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
