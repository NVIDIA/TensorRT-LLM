# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build the four real-prompt trtllm-bench datasets of CONFIDENCE_REPRODUCTION.md.

Each dataset is a JSONL of 20,480 requests in trtllm-bench format. The unique prompts come from
public datasets at pinned revisions and are selected and ordered by datasets_manifest.json, which
holds record IDs and text hashes only. Every prompt is rendered as one user turn with the
checkpoint's DeepSeek-V4 chat encoder (chat mode, no thinking). The pool is then shuffled with a
fixed seed and repeated until it fills 20,480 rows, so a run with N requests uses the first N.
output_tokens is a cap: requests stop at EOS.

SPEED-Bench publishes some throughput_1k rows as placeholders whose text is fetched from the
original sources by Model-Optimizer's specdec_bench. Pass its directory with --specdec-bench (see
CONFIDENCE_REPRODUCTION.md); the HLE rows need Hugging Face access to the gated cais/hle dataset.

Each output is checked against the SHA-256 of the files used for the published results.
"""

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

ROWS = 20480
MAX_OUTPUT_TOKENS = 1024
PLACEHOLDER = "FULL BENCHMARK DATA SHOULD BE FETCHED"
DATASETS = ["speed", "arena", "sharegpt", "gsm8k"]
SOURCES = {
    "speed": (
        "nvidia/SPEED-Bench",
        "throughput_1k/test-00000-of-00001.parquet",
        "487aa718444e816458d1a0a52bfce7a454285cf4",
    ),
    "arena": (
        "lmarena-ai/arena-hard-auto-v0.1",
        "data/train-00000-of-00001.parquet",
        "2a69efe86cff85c593cfd4c2b4491c128e52c6f1",
    ),
    "sharegpt": (
        "anon8231489123/ShareGPT_Vicuna_unfiltered",
        "ShareGPT_V3_unfiltered_cleaned_split.json",
        "192ab2185289094fc556ec8ce5ce1e8e587154ca",
    ),
    "gsm8k_test": (
        "openai/gsm8k",
        "main/test-00000-of-00001.parquet",
        "740312add88f781978c0658806c59bc2815b9866",
    ),
    "gsm8k_train": (
        "openai/gsm8k",
        "main/train-00000-of-00001.parquet",
        "740312add88f781978c0658806c59bc2815b9866",
    ),
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(key: str, manifest: dict) -> Path:
    repo, filename, revision = SOURCES[key]
    path = Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision))
    if sha256(path.read_bytes()) != manifest["source_sha256"][key]:
        sys.exit(f"{repo}/{filename}@{revision} does not match the expected SHA-256")
    return path


def resolve_placeholders(path: Path, specdec_bench: Path) -> list[str]:
    """Fill SPEED-Bench placeholder rows with specdec_bench's resolver, on the pinned parquet."""
    sys.path.insert(0, str(specdec_bench))
    from datasets import Dataset
    from specdec_bench.datasets.speed import SPEEDBench

    rows = Dataset.from_parquet(str(path))
    rows = rows.select([i for i, r in enumerate(rows) if r["turns"][0].startswith(PLACEHOLDER)])
    bench = SPEEDBench.__new__(SPEEDBench)  # skip __init__: it would load the latest SPEED-Bench
    bench.external_datasets, bench.num_samples, bench._config_name = {}, None, "throughput_1k"
    try:
        resolved = bench._resolve_external_data(rows, "throughput_1k")
    except Exception as error:
        sys.exit(
            f"specdec_bench could not fetch the SPEED-Bench sources ({error!r}). The HLE rows "
            "need access to the gated cais/hle dataset: accept its terms on Hugging Face and "
            "log in with `hf auth login`."
        )
    return [r["turns"][0] for r in resolved]


def speed_prompts(manifest: dict, specdec_bench: Path | None) -> list[str]:
    path = fetch("speed", manifest)
    texts = [r["turns"][0] for r in pq.read_table(path).to_pylist()]
    if specdec_bench is not None:
        texts += resolve_placeholders(path, specdec_bench)
    by_hash = {sha256(t.encode())[:16]: t for t in texts}
    wanted = manifest["speed"]["text_sha256_16"]
    missing = [h for h in wanted if h not in by_hash]
    if missing:
        hint = (
            "Pass --specdec-bench to fill the placeholder rows."
            if specdec_bench is None
            else "Check out Model-Optimizer at the commit given in CONFIDENCE_REPRODUCTION.md."
        )
        sys.exit(f"{len(missing)} of {len(wanted)} SPEED-Bench prompts are missing. {hint}")
    return [by_hash[h] for h in wanted]


def arena_prompts(manifest: dict) -> list[str]:
    rows = pq.read_table(fetch("arena", manifest)).to_pylist()
    by_id = {r["question_id"]: r["turns"][0]["content"].strip()[:512] for r in rows}
    return [by_id[q] for q in manifest["arena"]["question_ids"]]


def sharegpt_prompts(manifest: dict) -> list[str]:
    rows = json.loads(fetch("sharegpt", manifest).read_text())
    return [rows[i]["conversations"][0]["value"] for i in manifest["sharegpt"]["rows"]]


def gsm8k_prompts(manifest: dict) -> list[str]:
    test = pq.read_table(fetch("gsm8k_test", manifest)).to_pylist()
    train = pq.read_table(fetch("gsm8k_train", manifest)).to_pylist()
    prompts = []
    for question, shots in manifest["gsm8k"]["items"]:
        text = "".join(
            f"Question: {train[j]['question']}\nAnswer: {train[j]['answer']}\n\n" for j in shots
        )
        prompts.append(text + f"Question: {test[question]['question']}\nAnswer:")
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="DeepSeek-V4-Pro checkpoint (tokenizer.json and encoding/)",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--specdec-bench",
        type=Path,
        help="Model-Optimizer/examples/specdec_bench, to fill SPEED-Bench placeholders",
    )
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    args = parser.parse_args()

    sys.path.insert(0, str(args.model / "encoding"))
    from encoding_dsv4 import encode_messages
    from transformers import PreTrainedTokenizerFast

    tokenizer = PreTrainedTokenizerFast(tokenizer_file=str(args.model / "tokenizer.json"))
    manifest = json.loads((Path(__file__).parent / "datasets_manifest.json").read_text())
    builders = {
        "speed": lambda: speed_prompts(manifest, args.specdec_bench),
        "arena": lambda: arena_prompts(manifest),
        "sharegpt": lambda: sharegpt_prompts(manifest),
        "gsm8k": lambda: gsm8k_prompts(manifest),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    differs = []
    for name in args.datasets:
        pool = []
        for text in builders[name]():
            prompt = encode_messages([{"role": "user", "content": text}], thinking_mode="chat")
            pool.append(tokenizer.encode(prompt, add_special_tokens=False))
        rng = random.Random(20261102 + DATASETS.index(name))
        order = []
        while len(order) < ROWS:
            epoch = list(range(len(pool)))
            rng.shuffle(epoch)
            order.extend(epoch)
        lines = "".join(
            json.dumps({"task_id": i, "input_ids": pool[j], "output_tokens": MAX_OUTPUT_TOKENS})
            + "\n"
            for i, j in enumerate(order[:ROWS])
        )
        path = args.output_dir / f"{name}.jsonl"
        path.write_text(lines)
        same = sha256(lines.encode()) == manifest["jsonl_sha256"][name]
        if not same:
            differs.append(name)
        lengths = sorted(len(p) for p in pool)
        print(
            f"{path}: {len(pool)} unique prompts, input tokens {lengths[0]}-{lengths[-1]} "
            f"(median {lengths[len(lengths) // 2]}), "
            f"{'matches the published dataset' if same else 'DIFFERS from the published dataset'}"
        )
    if differs:
        sys.exit(f"{', '.join(differs)} differ from the published datasets (see the guide)")


if __name__ == "__main__":
    main()
