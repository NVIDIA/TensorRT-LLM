# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Measure native PLE K/V projection reuse, including D2H, planning and restore.

Run in a built TensorRT-LLM environment with one CUDA GPU:
    python tests/microbenchmarks/ple_projection_reuse.py --tokens 8192 16384 32768

This uses synthetic complete token contexts, postlookup inputs and random
weights. It excludes embedding lookup, TP communication, query gating and
convolution, so its results are not whole-PLE or application speedups.
"""

import argparse
import json
import statistics
import time
from types import SimpleNamespace

import torch
from torch import nn

from tensorrt_llm._torch.modules.linear import Linear
from tensorrt_llm._torch.modules.qwen4_exp.hyper_connection import GroupedRMSNorm
from tensorrt_llm._torch.modules.qwen4_exp.ple import PLEMetadata, Qwen4ExpPLE


def _make_projection_module(hidden: int, streams: int, embedding: int) -> Qwen4ExpPLE:
    # Instantiate only the measured subgraph, avoiding the large n-gram table.
    module = Qwen4ExpPLE.__new__(Qwen4ExpPLE)
    nn.Module.__init__(module)
    module.hidden_size, module.hc_count, module.ngram_size = hidden, streams, 3
    module._experimental_projection_reuse = False
    module.ple_embedding = SimpleNamespace(use_attention_dp_sharding=False)
    with torch.device("cuda"):
        module.key_proj = Linear(embedding, streams * hidden, bias=False, dtype=torch.bfloat16)
        module.value_proj = Linear(embedding, hidden, bias=False, dtype=torch.bfloat16)
        module.norm_key = GroupedRMSNorm(
            streams * hidden, eps=1e-6, group_size=hidden, dtype=torch.bfloat16
        )
    for parameter in module.parameters():
        parameter.normal_(0.0, 0.015)
    return module.eval()


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", nargs="+", type=int, default=[8192, 16384, 32768])
    parser.add_argument("--repetitions", type=int, default=40)
    parser.add_argument("--hidden-size", type=int, default=2560)
    parser.add_argument("--embedding-size", type=int, default=2560)
    parser.add_argument("--streams", type=int, default=4)
    parser.add_argument("--run-length", type=int, default=1024)
    args = parser.parse_args()
    if min(*args.tokens, args.repetitions, args.run_length) <= 0:
        parser.error("token counts, repetitions and run length must be positive")
    torch.manual_seed(1811)
    module = _make_projection_module(args.hidden_size, args.streams, args.embedding_size)
    results = []
    for count in args.tokens:
        for label, run_length in (("repeated_contexts", args.run_length), ("all_unique", 1)):
            tokens_cpu = torch.arange(count, dtype=torch.long) // run_length + 100
            contexts = torch.cat((torch.full((2,), 1), tokens_cpu)).unfold(0, 3, 1)
            unique_contexts, inverse = torch.unique(contexts, dim=0, return_inverse=True)
            embedding_rows = torch.randn(
                len(unique_contexts), args.embedding_size, device="cuda", dtype=torch.bfloat16
            )
            embeddings = embedding_rows.index_select(0, inverse.cuda())
            metadata = PLEMetadata.build(
                tokens_cpu.cuda(),
                torch.tensor([count], device="cuda"),
                torch.tensor([0], device="cuda"),
                is_decode=False,
                eos_token_id=1,
                num_contexts=1,
                host_seq_lens=[count],
            )
            module._experimental_projection_reuse = False
            dense = module._project_key_value(embeddings, metadata)
            module._experimental_projection_reuse = True
            reused = module._project_key_value(embeddings, metadata)
            errors = []
            for expected, actual in zip(dense, reused):
                torch.testing.assert_close(expected, actual, atol=0.04, rtol=0.04)
                errors.append(
                    {
                        "max_abs_error": (expected.float() - actual.float()).abs().max().item(),
                        "raw_equal": torch.equal(
                            expected.view(torch.uint8), actual.view(torch.uint8)
                        ),
                    }
                )
            for _ in range(5):
                for enabled in (False, True):
                    module._experimental_projection_reuse = enabled
                    module._project_key_value(embeddings, metadata)
            observations: dict[bool, list[float]] = {False: [], True: []}
            for repetition in range(args.repetitions):
                # Alternate paired order, with inclusive synchronized wall time.
                for enabled in (False, True) if repetition % 2 == 0 else (True, False):
                    module._experimental_projection_reuse = enabled
                    torch.cuda.synchronize()
                    started = time.perf_counter()
                    output = module._project_key_value(embeddings, metadata)
                    torch.cuda.synchronize()
                    observations[enabled].append((time.perf_counter() - started) * 1000)
                    del output
            baseline_ms = statistics.median(observations[False])
            candidate_ms = statistics.median(observations[True])
            results.append(
                {
                    "input": label,
                    "tokens": count,
                    "baseline_ms": baseline_ms,
                    "candidate_ms": candidate_ms,
                    "change_percent": (candidate_ms / baseline_ms - 1) * 100,
                    "key_value_errors": errors,
                }
            )
    print(
        json.dumps(
            {
                "scope": "synthetic native K/V projection subgraph; D2H/planning/restore included",
                "gpu": torch.cuda.get_device_name(),
                "torch": torch.__version__,
                "parameters": vars(args),
                "results": results,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
