import argparse
import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from apiary_client import AsyncApiary, ImageJobStatus
from apiary_client.swebench import get_docker_image, load_instances
from openai import AsyncOpenAI

from tensorrt_llm.scaffolding import ApiaryMCPWorker, TRTOpenaiWorker
from tensorrt_llm.scaffolding.contrib.Coder import create_swebench_coder_scaffolding_llm

logger = logging.getLogger(__name__)


def build_prompt(instance: dict[str, Any]) -> str:
    """Build the SWE-bench prompt for one instance."""
    parts = [
        "<pr_description>",
        instance["problem_statement"],
        "</pr_description>",
    ]
    hints = instance.get("hints_text", "")
    if hints and hints.strip():
        parts.extend(["", "<hints>", hints.strip(), "</hints>"])
    return "\n".join(parts)


def update_preds(
    preds_path: Path,
    instance_id: str,
    model_name: str,
    patch: str,
) -> None:
    """Update the SWE-bench predictions file."""
    data: dict[str, dict[str, str]] = {}
    if preds_path.exists():
        data = json.loads(preds_path.read_text())
    data[instance_id] = {
        "model_name_or_path": model_name,
        "instance_id": instance_id,
        "model_patch": patch,
    }
    preds_path.write_text(json.dumps(data, indent=2))


def save_trajectory(
    output_dir: Path,
    instance_id: str,
    prompt: str,
    output_text: str,
    elapsed_s: float,
) -> None:
    """Persist the trajectory for one SWE-bench instance."""
    instance_dir = output_dir / instance_id
    instance_dir.mkdir(parents=True, exist_ok=True)
    trajectory = {
        "instance_id": instance_id,
        "prompt": prompt,
        "output": output_text,
        "elapsed_seconds": elapsed_s,
    }
    (instance_dir / f"{instance_id}.traj.json").write_text(json.dumps(trajectory, indent=2))


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run SWE-bench evaluation using the Coder agent with Apiary sandboxes."
    )

    parser.add_argument("--openai_api_key", default="tensorrt_llm")
    parser.add_argument("--base_url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen3/Qwen3-30B-A3B")
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--max_iterations", type=int, default=100)

    parser.add_argument("--mcp_url", default="http://0.0.0.0:8083/sse")
    parser.add_argument("--max_mcp_connections", type=int, default=200)

    parser.add_argument(
        "--apiary_url",
        default=os.getenv("APIARY_URL", "http://127.0.0.1:8080"),
        help="Apiary daemon URL for runtime image registration",
    )
    parser.add_argument(
        "--apiary_token",
        default=os.getenv("APIARY_API_TOKEN"),
        help="Bearer token for the Apiary daemon",
    )
    parser.add_argument(
        "--apiary_load_timeout",
        type=float,
        default=None,
        help="Maximum seconds to wait for Apiary image registration (default: no timeout)",
    )

    parser.add_argument(
        "--dataset",
        required=True,
        help="Local JSON/JSONL path or HuggingFace dataset name/alias",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="HuggingFace split to load when --dataset is not a local file",
    )

    parser.add_argument("--output_dir", default="./swebench_output")
    parser.add_argument("--max_parallel_requests", type=int, default=16)

    parser.add_argument("--enable_statistics", action="store_true")
    parser.add_argument("--enable_tracing", action="store_true")

    return parser.parse_args()


def _log_image_progress(status: ImageJobStatus) -> None:
    """Progress callback for the AsyncApiary load loop."""
    counts = {
        "queued": 0,
        "pulling": 0,
        "extracting": 0,
        "done": 0,
        "alreadypresent": 0,
        "failed": 0,
    }
    for prog in status.per_image.values():
        counts[prog.state] = counts.get(prog.state, 0) + 1
    total = sum(counts.values())
    finished = counts["done"] + counts["alreadypresent"] + counts["failed"]
    logger.info(
        "Apiary image-load job=%s state=%s progress=%d/%d "
        "(done=%d, present=%d, pulling=%d, extracting=%d, queued=%d, failed=%d)",
        status.job_id[:8],
        status.state,
        finished,
        total,
        counts["done"],
        counts["alreadypresent"],
        counts["pulling"],
        counts["extracting"],
        counts["queued"],
        counts["failed"],
    )


async def register_swebench_images(
    images: list[str],
    apiary_url: str,
    apiary_token: str | None,
    load_timeout: float | None,
) -> dict[str, Any]:
    """Register the SWE-bench image set with the Apiary daemon and report status.

    Apiary returns 404 for any session targeting an unregistered image. This
    helper waits for the daemon to be healthy, submits the image set via
    :class:`AsyncApiary`, polls the load job to completion (logging per-image
    progress), and surfaces per-image failures so callers can decide how to
    react. Returns the daemon's :code:`/api/v1/status` snapshot for logging.
    """
    apiary = AsyncApiary(
        apiary_url=apiary_url,
        apiary_token=apiary_token,
        images=images,
        load_timeout=load_timeout,
        on_progress=_log_image_progress,
    )
    try:
        if not await apiary.health_check(retries=30, interval=1.0):
            raise RuntimeError(
                f"Apiary daemon at {apiary_url} is not reachable. "
                "Start it with `apiary init && apiary daemon --bind ...`."
            )
        status = await apiary.load()
        if status is None:
            raise RuntimeError("register_swebench_images: empty image set")
        if status.state == "failed":
            raise RuntimeError(
                f"Apiary failed to load any of the {len(images)} SWE-bench images; "
                f"failures: {status.failed_images}"
            )
        if status.failed:
            for entry in status.failed_images:
                logger.error(
                    "Apiary image %s failed: %s",
                    entry.get("name"),
                    entry.get("reason", "unknown"),
                )
            logger.warning(
                "%d of %d SWE-bench images failed to load; instances using those "
                "images will fail at session-creation time",
                len(status.failed),
                len(images),
            )
        logger.info(
            "Apiary registered %d SWE-bench images (succeeded=%d, failed=%d)",
            len(images),
            len(status.succeeded),
            len(status.failed),
        )
        return await apiary.status()
    finally:
        await apiary.close()


async def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_arguments()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    preds_path = output_dir / "preds.json"

    logger.info("[Phase 1] Loading dataset ...")
    instances = load_instances(args.dataset, args.split)
    if not instances:
        logger.info("No instances to process.")
        return

    unique_images = sorted({get_docker_image(instance) for instance in instances})
    logger.info(
        "%d instances, %d unique Docker images",
        len(instances),
        len(unique_images),
    )

    logger.info("[Phase 1] Registering %d images with Apiary ...", len(unique_images))
    status = await register_swebench_images(
        unique_images,
        args.apiary_url,
        args.apiary_token,
        args.apiary_load_timeout,
    )
    logger.info(
        "Apiary ready: total=%s busy=%s error=%s max_sandboxes=%s registered_images=%s",
        status.get("total", "n/a"),
        status.get("busy", "n/a"),
        status.get("error", "n/a"),
        status.get("max_sandboxes", "n/a"),
        status.get("registered_images", "n/a"),
    )

    logger.info("[Phase 2] Running agents ...")
    client = AsyncOpenAI(api_key=args.openai_api_key, base_url=args.base_url)
    generation_worker = TRTOpenaiWorker(client, args.model)
    mcp_worker = ApiaryMCPWorker(
        args.mcp_url,
        max_connections=args.max_mcp_connections,
    )

    llm = create_swebench_coder_scaffolding_llm(
        generation_worker,
        mcp_worker,
        max_tokens=args.max_tokens,
        max_iterations=args.max_iterations,
        max_parallel_requests=args.max_parallel_requests,
        enable_statistics=args.enable_statistics,
        enable_tracing=args.enable_tracing,
    )

    pending: list[tuple[dict[str, Any], str, Any, float]] = []
    for instance in instances:
        prompt = build_prompt(instance)
        result = llm.generate_async(prompt)
        mcp_worker.set_scope_params(
            result.id,
            image=get_docker_image(instance),
        )
        pending.append((instance, prompt, result, time.monotonic()))

    logger.info("Dispatched %d instances, awaiting results ...", len(pending))

    completed = 0
    errored = 0
    for instance, prompt, result, start_time in pending:
        instance_id = instance["instance_id"]
        try:
            output = await result.aresult()
            elapsed = time.monotonic() - start_time
            text = output.outputs[0].text or ""

            save_trajectory(output_dir, instance_id, prompt, text, elapsed)
            update_preds(preds_path, instance_id, args.model, text)

            if args.enable_tracing and result.task_collections:
                tracer = result.task_collections.get("execution_tracer")
                if tracer:
                    trace = tracer.export_trace()
                    trace_path = output_dir / instance_id / f"{instance_id}.trace.json"
                    trace.save(str(trace_path))
                    logger.info("Execution trace saved to %s", trace_path)

            completed += 1
            logger.info(
                "[%d/%d] %s completed (%.1fs)",
                completed + errored,
                len(pending),
                instance_id,
                elapsed,
            )
        except Exception:
            errored += 1
            logger.error("Instance %s failed", instance_id, exc_info=True)

    logger.info(
        "Completed: %d, Errors: %d, Total: %d",
        completed,
        errored,
        len(pending),
    )
    logger.info("Predictions: %s", preds_path)

    await mcp_worker.async_shutdown()
    llm.shutdown()
    generation_worker.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
