import argparse
import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from apiary_client import AsyncApiary, ImageJobStatus
from apiary_client.swebench import get_docker_image, load_instances
from openai import AsyncOpenAI

from tensorrt_llm.scaffolding import ApiaryMCPWorker, TRTOpenaiWorker
from tensorrt_llm.scaffolding.contrib.Coder import create_swebench_coder_scaffolding_llm

logger = logging.getLogger(__name__)

DEFAULT_SWEBENCH_OUTPUT_DIR = Path("./swebench_output")


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
    summary: str | None = None,
) -> None:
    """Update the SWE-bench predictions file."""
    data: dict[str, dict[str, str]] = {}
    if preds_path.exists():
        data = json.loads(preds_path.read_text())
    record = {
        "model_name_or_path": model_name,
        "instance_id": instance_id,
        "model_patch": patch,
    }
    if summary is not None:
        record["summary"] = summary
    data[instance_id] = record
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


def safe_output_component(value: str) -> str:
    """Return a filesystem-safe label for SWE-bench run directory names."""
    label = Path(value).stem if Path(value).suffix else value
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_") or "unknown"


def default_swebench_output_dir(dataset: str, split: str, model: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = "-".join(
        [
            safe_output_component(dataset),
            safe_output_component(split),
            safe_output_component(model),
            timestamp,
        ]
    )
    return DEFAULT_SWEBENCH_OUTPUT_DIR / run_name


def _duration_ms(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _tool_call_time_summary(trace_path: Path) -> dict[str, float | int]:
    data = json.loads(trace_path.read_text(encoding="utf-8"))
    events = data.get("events", [])
    if not isinstance(events, list):
        raise ValueError(f"{trace_path} must contain an events list.")

    tool_call_count = 0
    llm_turn_count = 0
    tool_call_duration_ms = 0.0
    llm_duration_ms = 0.0
    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("event_type") == "tool_call":
            tool_call_count += 1
            tool_call_duration_ms += _duration_ms(event.get("duration_ms"))
        elif event.get("event_type") == "message" and event.get("role") == "assistant":
            duration_ms = _duration_ms(event.get("llm_duration_ms"))
            if duration_ms > 0:
                llm_turn_count += 1
                llm_duration_ms += duration_ms

    recorded_duration_ms = tool_call_duration_ms + llm_duration_ms
    tool_call_time_ratio = (
        tool_call_duration_ms / recorded_duration_ms if recorded_duration_ms > 0 else 0.0
    )
    return {
        "event_count": len(events),
        "tool_call_count": tool_call_count,
        "llm_turn_count": llm_turn_count,
        "tool_call_duration_ms": tool_call_duration_ms,
        "llm_duration_ms": llm_duration_ms,
        "recorded_duration_ms": recorded_duration_ms,
        "tool_call_time_ratio": tool_call_time_ratio,
        "tool_call_time_percent": tool_call_time_ratio * 100,
    }


def write_tool_call_time_summary(
    trace_path: Path,
    output_path: Path,
    elapsed_s: float,
) -> None:
    """Write per-trace tool-call time share computed from the full trace."""
    summary = _tool_call_time_summary(trace_path)
    elapsed_ms = elapsed_s * 1000
    elapsed_ratio = summary["tool_call_duration_ms"] / elapsed_ms if elapsed_ms > 0 else 0.0
    lines = [
        f"trace_path: {trace_path}",
        f"event_count: {summary['event_count']}",
        f"tool_call_count: {summary['tool_call_count']}",
        f"llm_turn_count: {summary['llm_turn_count']}",
        f"tool_call_duration_ms: {summary['tool_call_duration_ms']:.3f}",
        f"llm_duration_ms: {summary['llm_duration_ms']:.3f}",
        f"recorded_duration_ms: {summary['recorded_duration_ms']:.3f}",
        f"tool_call_time_ratio: {summary['tool_call_time_ratio']:.6f}",
        f"tool_call_time_percent: {summary['tool_call_time_percent']:.2f}%",
        f"elapsed_seconds: {elapsed_s:.3f}",
        f"tool_call_elapsed_time_ratio: {elapsed_ratio:.6f}",
        f"tool_call_elapsed_time_percent: {elapsed_ratio * 100:.2f}%",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _split_instance_ids(instance_ids: list[str] | None) -> list[str]:
    """Parse instance IDs from space-separated and comma-separated CLI values."""
    if not instance_ids:
        return []

    parsed_ids: list[str] = []
    for value in instance_ids:
        parsed_ids.extend(
            instance_id.strip() for instance_id in value.split(",") if instance_id.strip()
        )
    return parsed_ids


def _load_instance_ids_file(instance_ids_file: str | None) -> list[str]:
    """Load instance IDs from a text file, ignoring blank lines and comments."""
    if not instance_ids_file:
        return []

    path = Path(instance_ids_file)
    instance_ids: list[str] = []
    for line in path.read_text().splitlines():
        value = line.split("#", maxsplit=1)[0].strip()
        if value:
            instance_ids.append(value)
    return instance_ids


def resolve_instance_ids(args: argparse.Namespace) -> list[str]:
    """Resolve the requested SWE-bench instance IDs from CLI arguments."""
    instance_ids = _split_instance_ids(args.instance_ids)
    instance_ids.extend(_load_instance_ids_file(args.instance_ids_file))
    return list(dict.fromkeys(instance_ids))


def write_selected_instances(
    dataset: str,
    split: str,
    instance_ids: list[str],
    output_path: Path,
) -> Path:
    """Write a filtered local JSON dataset for the selected SWE-bench instances."""
    instances = load_instances(dataset, split)
    requested_ids = set(instance_ids)
    selected_instances = [
        instance for instance in instances if instance.get("instance_id") in requested_ids
    ]
    selected_ids = {instance["instance_id"] for instance in selected_instances}
    missing_ids = sorted(requested_ids - selected_ids)
    if missing_ids:
        raise ValueError(
            f"Requested instance IDs were not found in dataset={dataset!r} "
            f"split={split!r}: "
            + ", ".join(missing_ids)
            + ". Check that --dataset and --split match the selected IDs."
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(selected_instances, indent=2))
    return output_path


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run SWE-bench evaluation using the Coder agent with Apiary sandboxes."
    )

    parser.add_argument("--openai_api_key", default="tensorrt_llm")
    parser.add_argument("--base_url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen3/Qwen3-30B-A3B")
    parser.add_argument("--max_tokens", type=int, default=16384)
    parser.add_argument("--max_iterations", type=int, default=100)

    parser.add_argument("--mcp_url", default="http://0.0.0.0:8086/sse")
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
    parser.add_argument(
        "--instance_ids",
        nargs="+",
        default=None,
        help=(
            "Optional SWE-bench instance IDs to run. Accepts space-separated "
            "values, comma-separated values, or both."
        ),
    )
    parser.add_argument(
        "--instance_ids_file",
        default=None,
        help=(
            "Optional text file containing SWE-bench instance IDs to run, "
            "one per line. Blank lines and comments are ignored."
        ),
    )
    parser.add_argument(
        "--selected_dataset_path",
        default=None,
        help=(
            "Path to write the filtered JSON dataset when instance IDs are "
            "specified. Defaults to <output_dir>/selected_instances.json."
        ),
    )

    parser.add_argument(
        "--output_dir",
        default=None,
        help=(
            "Directory for preds.json, selected_instances.json, trajectories, and traces. "
            "Default: ./swebench_output/<dataset>-<split>-<model>-<time>."
        ),
    )
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
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else default_swebench_output_dir(args.dataset, args.split, args.model)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    preds_path = output_dir / "preds.json"

    dataset = args.dataset
    instance_ids = resolve_instance_ids(args)
    if instance_ids:
        selected_dataset_path = Path(
            args.selected_dataset_path or output_dir / "selected_instances.json"
        )
        logger.info(
            "Selecting %d instances from %s and writing %s ...",
            len(instance_ids),
            args.dataset,
            selected_dataset_path,
        )
        dataset = str(
            write_selected_instances(
                args.dataset,
                args.split,
                instance_ids,
                selected_dataset_path,
            )
        )

    logger.info("[Phase 1] Loading dataset ...")
    instances = load_instances(dataset, args.split)
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
            metadata = output.outputs[0].metadata or {}
            model_patch = metadata.get("swebench_model_patch", text)
            summary = metadata.get("swebench_summary")
            if not isinstance(model_patch, str):
                model_patch = str(model_patch)
            if summary is not None and not isinstance(summary, str):
                summary = str(summary)

            save_trajectory(output_dir, instance_id, prompt, text, elapsed)
            update_preds(preds_path, instance_id, args.model, model_patch, summary)

            if args.enable_tracing and result.task_collections:
                tracer = result.task_collections.get("execution_tracer")
                if tracer:
                    trace = tracer.export_trace()
                    instance_dir = output_dir / instance_id
                    trace_path = instance_dir / f"{instance_id}.trace.json"
                    full_trace_path = instance_dir / f"{instance_id}.full.trace.json"
                    tool_call_time_path = instance_dir / "tool_call_time_summary.txt"
                    trace.save(str(trace_path))
                    trace.save(str(full_trace_path), full=True)
                    write_tool_call_time_summary(
                        full_trace_path,
                        tool_call_time_path,
                        elapsed,
                    )
                    logger.info("Execution trace saved to %s", trace_path)
                    logger.info("Full execution trace saved to %s", full_trace_path)
                    logger.info("Tool-call time summary saved to %s", tool_call_time_path)

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
