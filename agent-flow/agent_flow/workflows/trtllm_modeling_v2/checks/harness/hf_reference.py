#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Write a reference of first-token log-probs with HuggingFace ``transformers``.

Usage, with an interpreter that has ``torch`` and ``transformers`` and on devices that hold
the checkpoint (it imports nothing of this package, so it runs by path in any job)::

    <python> hf_reference.py --model CKPT --out reference.json [--prompts FILE] [--top-k 20]
                            [--trust-remote-code] [--dtype bfloat16]

It loads the checkpoint natively (``AutoModelForCausalLM``, ``device_map="auto"``), runs each
prompt once and writes the top-k log-probs of the next token, in the file
``accuracy.reference_logits.path`` names (``checks/divergence.py`` reads it). A checkpoint
``transformers`` cannot load natively has no such reference: the boot gate then records the
divergence as UNKNOWN, and the benchmark score stays the golden rule.
"""

from __future__ import annotations

import argparse
import json
import sys

#: Short prompts of different kinds, when ``--prompts`` names none.
DEFAULT_PROMPTS = (
    "The capital of France is",
    "Natalia sold 48 clips in April and half as many in May. How many clips did she sell in all?",
    "def fibonacci(n):",
    "Water boils at a temperature of",
    "The three primary colors are",
    "Q: What is 17 times 23?\nA:",
    "The largest planet in our solar system is",
    "In 1969, the first humans to walk on the Moon were",
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--prompts", help="a file of prompts, one per line")
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--trust-remote-code", action="store_true")
    args = parser.parse_args(argv)

    import torch
    import transformers

    prompts = DEFAULT_PROMPTS
    if args.prompts:
        with open(args.prompts, encoding="utf-8") as f:
            prompts = tuple(line.rstrip("\n") for line in f if line.strip())
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=getattr(torch, args.dtype),
        device_map="auto",
        trust_remote_code=args.trust_remote_code,
    )
    model.eval()
    first_tokens = {}
    with torch.no_grad():
        for i, prompt in enumerate(prompts):
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            logits = model(**inputs).logits[0, -1].float()
            top = torch.log_softmax(logits, dim=-1).topk(args.top_k)
            first_tokens[f"p{i}"] = {
                str(int(t)): float(v) for t, v in zip(top.indices.tolist(), top.values.tolist())
            }
    record = {
        "source": f"transformers {transformers.__version__} {args.dtype} {args.model}",
        "top_k": args.top_k,
        "prompts": {f"p{i}": p for i, p in enumerate(prompts)},
        "first_tokens": first_tokens,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=1)
    print(f"wrote {len(first_tokens)} prompt(s) to {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
