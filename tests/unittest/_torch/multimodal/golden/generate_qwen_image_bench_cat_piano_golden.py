# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import argparse
import importlib.util
import json
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[5]


def _load_eval_module():
    repo_root = _repo_root()
    script_path = (
        repo_root / "examples" / "models" / "core" / "multimodal" / "qwen_image_bench_eval.py"
    )
    spec = importlib.util.spec_from_file_location("qwen_image_bench_eval", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _stable_result(result: dict) -> dict:
    return {
        "prompt": result["prompt"],
        "image_name": Path(result["image_path"]).name,
        "dimensions": result["dimensions"],
        "level1_scores": result["level1_scores"],
        "level2_scores": result["level2_scores"],
        "level3_scores": result["level3_scores"],
        "total_score": result["total_score"],
        "parse_failures": result["parse_failures"],
        "parsed_scores": result["parsed_scores"],
        "raw_outputs": result["raw_outputs"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument(
        "--output_path",
        default=str(Path(__file__).with_name("qwen_image_bench_cat_piano_alignment.json")),
    )
    args = parser.parse_args()

    repo_root = _repo_root()
    eval_args = argparse.Namespace(
        model_path=args.model_path,
        prompt="A cute cat playing piano",
        image_path=str(repo_root / "examples" / "visual_gen" / "cat_piano.png"),
        output_path=None,
        dimensions=["Alignment"],
        backend="pytorch",
        image_data_format="pt",
        max_tokens=4096,
        max_num_tokens=8192,
        max_seq_len=8192,
        kv_cache_max_tokens=8192,
        temperature=0.0,
        top_k=1,
        top_p=1.0,
        repetition_penalty=1.05,
        use_fast_processor=False,
        enable_block_reuse=False,
        include_raw_outputs=True,
    )

    stable = _stable_result(_load_eval_module().evaluate(eval_args))
    output_path = Path(args.output_path)
    output_path.write_text(json.dumps(stable, ensure_ascii=False, indent=2) + "\n")
    print(f"Wrote {output_path}")
    print(
        json.dumps(
            {
                "level1_scores": stable["level1_scores"],
                "total_score": stable["total_score"],
                "parse_failures": stable["parse_failures"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
