import argparse
import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import AsyncOpenAI

from tensorrt_llm.scaffolding import ApiaryMCPWorker, TRTOpenaiWorker
from tensorrt_llm.scaffolding.contrib.Coder import create_swebench_coder_scaffolding_llm

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------
# Dataset helpers
# -----------------------------------------------------------------------

DATASET_MAPPING = {
    "full": "princeton-nlp/SWE-Bench",
    "verified": "princeton-nlp/SWE-Bench_Verified",
    "lite": "princeton-nlp/SWE-Bench_Lite",
    "multimodal": "princeton-nlp/SWE-Bench_Multimodal",
    "multilingual": "swe-bench/SWE-Bench_Multilingual",
}


def load_instances(dataset: str) -> list[dict]:
    """Load SWE-bench instances from a local JSON/JSONL file or HuggingFace."""
    p = Path(dataset)
    if p.exists():
        text = p.read_text()
        if p.suffix == ".jsonl":
            return [json.loads(line) for line in text.strip().splitlines()]
        return json.loads(text) if text.strip().startswith("[") else list(json.loads(text).values())

    try:
        from datasets import load_dataset as hf_load
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required for loading HuggingFace datasets.  "
            "Install it with:  pip install datasets"
        ) from exc

    hf_path = DATASET_MAPPING.get(dataset, dataset)
    logger.info("Loading HuggingFace dataset %s ...", hf_path)
    return list(hf_load(hf_path, split="dev"))


def get_docker_image(instance: dict) -> str:
    """Derive the Docker image name for an instance."""
    image = instance.get("image_name") or instance.get("docker_image")
    if image:
        return image
    iid = instance["instance_id"]
    id_compat = iid.replace("__", "_1776_")
    return f"docker.io/swebench/sweb.eval.x86_64.{id_compat}:latest".lower()


def build_prompt(instance: dict) -> str:
    """Build the user prompt from an instance's problem statement."""
    parts = [
        "<pr_description>",
        instance["problem_statement"],
        "</pr_description>",
    ]
    hints = instance.get("hints_text", "")
    if hints and hints.strip():
        parts += ["", "<hints>", hints.strip(), "</hints>"]
    return "\n".join(parts)


# -----------------------------------------------------------------------
# Results helpers
# -----------------------------------------------------------------------


def update_preds(
    preds_path: Path,
    instance_id: str,
    model_name: str,
    patch: str,
) -> None:
    data: dict = {}
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
    inst_dir = output_dir / instance_id
    inst_dir.mkdir(parents=True, exist_ok=True)
    traj = {
        "instance_id": instance_id,
        "prompt": prompt,
        "output": output_text,
        "elapsed_seconds": elapsed_s,
    }
    (inst_dir / f"{instance_id}.traj.json").write_text(json.dumps(traj, indent=2))


# -----------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------


def parse_arguments():
    p = argparse.ArgumentParser(
        description="Run SWE-bench evaluation using the Coder agent with Apiary sandboxes."
    )

    # LLM
    p.add_argument("--openai_api_key", default="tensorrt_llm")
    p.add_argument("--base_url", default="http://localhost:8000/v1")
    p.add_argument("--model", default="Qwen3/Qwen3-30B-A3B")
    p.add_argument("--max_tokens", type=int, default=16384)
    p.add_argument("--max_iterations", type=int, default=100)

    # MCP
    p.add_argument("--mcp_url", default="http://0.0.0.0:8083/sse")
    p.add_argument("--max_mcp_connections", type=int, default=200)

    # Dataset
    p.add_argument(
        "--dataset",
        required=True,
        help="Local JSON/JSONL path or HuggingFace dataset name/alias",
    )

    # Rootfs
    p.add_argument("--rootfs_cache_dir", default="/tmp/apiary_rootfs")
    p.add_argument("--rootfs_workers", type=int, default=4)

    # Output
    p.add_argument("--output_dir", default="./swebench_output")
    p.add_argument("--max_parallel_requests", type=int, default=16)

    # Misc
    p.add_argument("--enable_statistics", action="store_true")
    p.add_argument("--enable_tracing", action="store_true")

    return p.parse_args()


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------


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
    instances = load_instances(args.dataset)

    if not instances:
        logger.info("No instances to process.")
        return

    image_map = {get_docker_image(i): None for i in instances}
    unique_images = list(image_map.keys())
    logger.info("%d instances, %d unique Docker images", len(instances), len(unique_images))

    logger.info("[Phase 1] Extracting image layers ...")
    try:
        from apiary_swebench.rootfs import RootfsManager
    except ImportError as exc:
        raise ImportError(
            "apiary_swebench is required for SWE-bench evaluation.  "
            "Install it with:\n"
            "  pip install -e /path/to/apiary-integration/apiary/swebench\n"
            "See examples/scaffolding/contrib/Coder/README.md for details."
        ) from exc

    rootfs_mgr = RootfsManager(cache_dir=args.rootfs_cache_dir)
    rootfs_map: dict[str, list[str]] = {}
    with ThreadPoolExecutor(max_workers=args.rootfs_workers) as pool:
        futures = {pool.submit(rootfs_mgr.ensure_layers, img): img for img in unique_images}
        for fut in as_completed(futures):
            img = futures[fut]
            rootfs_map[img] = fut.result()
    logger.info("Layers ready: %d images", len(rootfs_map))

    logger.info("[Phase 2] Running agents ...")
    client = AsyncOpenAI(api_key=args.openai_api_key, base_url=args.base_url)
    generation_worker = TRTOpenaiWorker(client, args.model)
    mcp_worker = ApiaryMCPWorker(args.mcp_url, max_connections=args.max_mcp_connections)

    llm = create_swebench_coder_scaffolding_llm(
        generation_worker,
        mcp_worker,
        max_tokens=args.max_tokens,
        max_iterations=args.max_iterations,
        max_parallel_requests=args.max_parallel_requests,
        enable_statistics=args.enable_statistics,
        enable_tracing=args.enable_tracing,
    )

    pending: list[tuple[dict, str, object, float]] = []

    for inst in instances:
        prompt = build_prompt(inst)
        result = llm.generate_async(prompt)
        image = get_docker_image(inst)
        mcp_worker.set_scope_params(result.id, base_image=rootfs_map[image])
        pending.append((inst, prompt, result, time.monotonic()))

    logger.info("Dispatched %d instances, awaiting results ...", len(pending))

    completed = 0
    errored = 0
    for inst, prompt, result, t0 in pending:
        iid = inst["instance_id"]
        try:
            output = await result.aresult()
            elapsed = time.monotonic() - t0
            text = output.outputs[0].text or ""

            save_trajectory(output_dir, iid, prompt, text, elapsed)
            update_preds(preds_path, iid, args.model, text)

            if args.enable_tracing and result.task_collections:
                tracer = result.task_collections.get("execution_tracer")
                if tracer:
                    trace = tracer.export_trace()
                    trace_path = output_dir / iid / f"{iid}.trace.json"
                    trace.save(str(trace_path))
                    logger.info("Execution trace saved to %s", trace_path)

            completed += 1
            logger.info(
                "[%d/%d] %s completed (%.1fs)",
                completed + errored,
                len(pending),
                iid,
                elapsed,
            )
        except Exception:
            errored += 1
            logger.error("Instance %s failed", iid, exc_info=True)

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
