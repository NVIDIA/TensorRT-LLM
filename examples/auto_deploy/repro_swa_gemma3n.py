# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""Minimal repro for the Gemma 3n E2B AutoDeploy SWA front-eviction regression.

Reproduces the GSM8K-style failure mode (long prefill + long decode that fires
SWA front-eviction dozens of times per sequence) in <1 minute, without pulling
in the full integration-test harness.

Usage (inside the TRT-LLM dev container, on a GPU node):
    AUTO_DEPLOY_LOG_LEVEL=debug \
        python examples/auto_deploy/repro_swa_gemma3n.py \
        --model-path /path/to/google/gemma-3n-E2B-it \
        --prompt-tokens 1024 --max-new-tokens 256 \
        2>&1 | tee /tmp/swa-trace.log

Then grep the SWA traces:
    grep '\\[swa-trace\\]' /tmp/swa-trace.log | head -40

The defaults (1024-token prompt, 256-token output, sliding_window=512) fire
SWA eviction on the first decode token and again every 32 generated tokens.
"""

import argparse
from pathlib import Path

from tensorrt_llm import SamplingParams
from tensorrt_llm._torch.auto_deploy import LLM as AutoDeployLLM

_DEFAULT_CONFIG = (
    Path(__file__).resolve().parent / "model_registry" / "configs" / "gemma3n_e2b_it.yaml"
)


def _build_long_prompt(target_token_count: int) -> str:
    """Build a deterministic synthetic prompt that tokenizes to roughly ``target_token_count``.

    Uses GSM8K-style math content because that is what triggers the failing path
    in CI; the actual numbers don't matter for the bug, only the length.
    """
    # ~22 tokens per repetition on Gemma 3n's tokenizer (rough rule of thumb).
    base = (
        "Alice has 47 apples. She gives 13 to Bob, then buys 28 more, then "
        "loses 9 to a thief. Carl has twice as many apples as Alice now. "
    )
    n_repeats = max(1, target_token_count // 22)
    prefix = base * n_repeats
    suffix = "Show your full step-by-step reasoning, then end with '#### <number>'."
    return prefix + "\n\n" + suffix


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-path",
        required=True,
        help="HF model ID or local path to google/gemma-3n-E2B-it",
    )
    p.add_argument(
        "--config",
        default=str(_DEFAULT_CONFIG),
        help="AutoDeploy yaml_extra config (defaults to gemma3n_e2b_it.yaml)",
    )
    p.add_argument(
        "--prompt-tokens",
        type=int,
        default=1024,
        help="Target prefill length in tokens. Must exceed the SWA window (512) "
        "to fire front-eviction during decode.",
    )
    p.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Decode budget. >=32 to cross at least one SWA page boundary.",
    )
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=4352,
        help="Mirrors the test's MAX_SEQ_LEN = max(MMLU 4096, GSM8K 4352).",
    )
    args = p.parse_args()

    prompt = _build_long_prompt(args.prompt_tokens)

    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        end_id=None,
        pad_id=None,
        n=1,
        use_beam_search=False,  # greedy, matches the failing test
    )

    print(f"[repro] model           = {args.model_path}")
    print(f"[repro] config          = {args.config}")
    print(f"[repro] prompt chars    = {len(prompt)}")
    print(f"[repro] target prompt tk= {args.prompt_tokens}")
    print(f"[repro] max_new_tokens  = {args.max_new_tokens}")
    print(f"[repro] max_seq_len     = {args.max_seq_len}")

    with AutoDeployLLM(
        model=args.model_path,
        tokenizer=args.model_path,
        world_size=1,
        max_seq_len=args.max_seq_len,
        max_num_tokens=args.max_seq_len,
        yaml_extra=[args.config],
    ) as llm:
        outputs = llm.generate([prompt], sampling_params=sampling_params)

    out = outputs[0].outputs[0]
    print()
    print("=" * 80)
    print(f"[repro] generated {len(out.token_ids)} tokens")
    print("=" * 80)
    print(out.text)
    print("=" * 80)


if __name__ == "__main__":
    main()
