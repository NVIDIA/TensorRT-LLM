# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
#
# Minimal script to test FSD (Fuzzy Speculative Decoding) via the TensorRT-LLM
# Python API. Run from the TensorRT-LLM repo root (or set PYTHONPATH).
#
# Models:
#   Without --draft_engine_dir: only TARGET (--engine_dir) is loaded; draft tokens
#   are placeholders (pad_id repeated).
#   With --draft_engine_dir: DRAFT (1B) and TARGET (3B) are both loaded; draft model
#   generates candidate tokens each step, target verifies with FSD.
#
# Prerequisites:
#   1. Target engine (and if using --draft_engine_dir, draft engine too) built with:
#        --speculative_decoding_mode=draft_tokens_external --max_draft_len=<N>
#   2. C++ session; KV cache block reuse: --kv_cache_enable_block_reuse
#   For draft+target: build both engines with the same max_draft_len (e.g. 5).
#
# Usage (target only; placeholder draft tokens):
#   python examples/draft_target_model/run_dtm_fsd_test.py \
#       --engine_dir /path/to/target_engine \
#       --fsd_threshold 0.05 --fsd_divergence_type kl
#
# Usage (draft 1B + target 3B; real speculative decoding):
#   python examples/draft_target_model/run_dtm_fsd_test.py \
#       --draft_engine_dir /path/to/llama32_1b_engine \
#       --engine_dir /path/to/llama32_3b_engine \
#       --fsd_threshold 0.05 --fsd_divergence_type kl
#
# Benchmark (KL thresholds 0,0.1,...,1, report tokens/sec):
#   python ... --engine_dir /path/to/target_engine [--draft_engine_dir /path/to/draft_engine] \
#       --benchmark --benchmark_num_tokens 64
#
# Bindings-only test (no engine):
#   python -c "
#   import tensorrt_llm.bindings as trtllm
#   c = trtllm.ExternalDraftTokensConfig([1,2,3], fsd_threshold=0.5, fsd_divergence_type=1)
#   print('FSD config:', c.fsd_threshold, c.fsd_divergence_type)
#   "

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils import add_common_args, load_tokenizer, read_model_name

import tensorrt_llm
from tensorrt_llm.logger import logger
from tensorrt_llm.runtime import PYTHON_BINDINGS

if PYTHON_BINDINGS:
    from tensorrt_llm.runtime import ModelRunnerCpp
else:
    ModelRunnerCpp = None

# FSD thresholds to sweep when --benchmark is used (KL divergence)
BENCHMARK_FSD_THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def parse_args():
    parser = argparse.ArgumentParser(description="Test FSD via TensorRT-LLM Python API")
    parser = add_common_args(parser)
    parser.add_argument("--input_text", type=str, nargs="+", default=["Hello, world"])
    parser.add_argument("--max_output_len", type=int, default=64)
    parser.add_argument(
        "--max_draft_len", type=int, default=5, help="Must match engine --max_draft_len"
    )
    parser.add_argument("--max_input_length", type=int, default=512, help="Max input token length")
    parser.add_argument(
        "--fsd_threshold",
        type=float,
        default=0.05,
        help="FSD divergence threshold (>=0). 0 = strict SD.",
    )
    parser.add_argument(
        "--fsd_divergence_type", type=str, default="kl", choices=["js", "kl", "tv", "reverse_kl"]
    )
    parser.add_argument(
        "--use_logits",
        action="store_true",
        help="Use draft logits (need gather_generation_logits when building)",
    )
    parser.add_argument(
        "--hf_tokenizer_model",
        type=str,
        default=None,
        help="HuggingFace model ID for tokenizer when engine_dir has no HF config "
        "(e.g. meta-llama/Llama-3.2-1B-Instruct)",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run FSD with thresholds 0,0.1,...,1 and report tokens/sec for each",
    )
    parser.add_argument(
        "--benchmark_num_tokens",
        type=int,
        default=64,
        help="Number of new tokens to generate per run when --benchmark (fixed target for tokens/sec)",
    )
    parser.add_argument(
        "--draft_engine_dir",
        type=str,
        default=None,
        help="Path to draft model engine (e.g. Llama 3.2 1B). "
        "If set, use real draft+target; else use placeholder draft tokens.",
    )
    return parser.parse_args()


def _get_tokenizer_dir(args, engine_dir):
    """Resolve tokenizer source.

    Prefers explicit tokenizer_dir or hf_tokenizer_model; fallback to engine_dir or HF default.
    """
    if args.tokenizer_dir:
        return args.tokenizer_dir
    if getattr(args, "hf_tokenizer_model", None):
        return args.hf_tokenizer_model
    config_path = os.path.join(engine_dir, "config.json")
    if os.path.isfile(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        if "model_type" in cfg:
            return engine_dir
        if "pretrained_config" in cfg:
            arch = cfg["pretrained_config"].get("architecture", "")
            if "Llama" in arch:
                return "meta-llama/Llama-3.2-1B-Instruct"
    return engine_dir


def run_fsd_test(args):
    if not PYTHON_BINDINGS:
        logger.error("C++ Python bindings required for FSD. Rebuild with Python bindings.")
        return 1

    runtime_rank = tensorrt_llm.mpi_rank()
    engine_dir = args.engine_dir
    model_name, model_version = read_model_name(engine_dir)
    tokenizer_dir = _get_tokenizer_dir(args, engine_dir)
    tokenizer, pad_id, end_id = load_tokenizer(
        tokenizer_dir=tokenizer_dir,
        vocab_file=args.vocab_file,
        model_name=model_name,
        model_version=model_version,
        tokenizer_type=args.tokenizer_type,
    )

    batch_input_ids = []
    for text in args.input_text:
        ids = tokenizer.encode(
            text, add_special_tokens=True, truncation=True, max_length=args.max_input_length
        )
        batch_input_ids.append(torch.tensor(ids, dtype=torch.int32))
    input_lengths = [x.size(0) for x in batch_input_ids]
    batch_size = len(batch_input_ids)

    max_output_tokens_for_runner = (
        args.benchmark_num_tokens if args.benchmark else args.max_output_len
    )
    max_output_tokens = max_output_tokens_for_runner
    max_draft = min(args.max_draft_len, max_output_tokens - 1)
    common_runner_kwargs = dict(
        rank=runtime_rank,
        debug_mode=args.debug_mode,
        max_output_len=max_output_tokens_for_runner,
        is_enc_dec=False,
        max_batch_size=batch_size,
        max_input_len=max(input_lengths) + max_output_tokens_for_runner,
        max_beam_width=1,
        max_attention_window_size=args.max_attention_window_size,
        sink_token_length=args.sink_token_length,
        max_tokens_in_paged_kv_cache=args.max_tokens_in_paged_kv_cache,
        kv_cache_enable_block_reuse=True,
        kv_cache_free_gpu_memory_fraction=args.kv_cache_free_gpu_memory_fraction,
        gather_generation_logits=args.use_logits,
    )

    use_draft_model = args.draft_engine_dir is not None
    if use_draft_model:
        draft_runner = ModelRunnerCpp.from_dir(
            engine_dir=args.draft_engine_dir, **common_runner_kwargs
        )
        target_runner = ModelRunnerCpp.from_dir(engine_dir=engine_dir, **common_runner_kwargs)
        if args.use_logits:
            assert getattr(draft_runner, "gather_generation_logits", False) and getattr(
                target_runner, "gather_generation_logits", False
            ), "Build draft and target with --gather_generation_logits for --use_logits"
        runner = target_runner  # for any single-run path that still expects runner
    else:
        runner_kwargs = dict(engine_dir=engine_dir, **common_runner_kwargs)
        runner = ModelRunnerCpp.from_dir(**runner_kwargs)

    draft_tokens_list = [[pad_id] * max_draft for _ in range(batch_size)]
    draft_logits_list = None
    if (
        not use_draft_model
        and args.use_logits
        and getattr(runner, "gather_generation_logits", False)
    ):
        vocab_size = runner.model_config.vocab_size_padded
        draft_logits_list = [
            torch.zeros(max_draft, vocab_size, dtype=torch.float16) for _ in range(batch_size)
        ]

    def _run_draft_target_loop(fsd_threshold):
        """Iterative draft->target FSD generation. Returns (outputs, num_generated, elapsed_sec)."""
        max_seq_len = [input_lengths[i] + max_output_tokens for i in range(batch_size)]
        prefix = [batch_input_ids[i].clone() for i in range(batch_size)]
        batch_slot = list(range(batch_size))
        output_ids = torch.full(
            (batch_size, 1, max(input_lengths) + max_output_tokens),
            end_id,
            dtype=torch.int32,
        )
        for bi in range(batch_size):
            output_ids[bi, 0, : input_lengths[bi]] = batch_input_ids[bi]
        sequence_lengths = torch.zeros(batch_size, 1, dtype=torch.int32)

        t0 = time.perf_counter()
        n_steps = 0
        while True:
            n_steps += 1
            batch_size_cur = len(prefix)
            prefix_len = [prefix[i].size(0) for i in range(batch_size_cur)]

            # Draft model: generate up to max_draft new tokens
            draft_kwargs = dict(
                batch_input_ids=prefix,
                max_new_tokens=max_draft,
                end_id=end_id,
                pad_id=pad_id,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                return_dict=True,
                output_sequence_lengths=True,
                output_generation_logits=args.use_logits,
            )
            with torch.no_grad():
                draft_out = draft_runner.generate(**draft_kwargs)
            d_seq_len = draft_out["sequence_lengths"][:, 0].tolist()
            d_ids = []
            d_logits = [None] * batch_size_cur
            for bi in range(batch_size_cur):
                lo, r = prefix_len[bi], d_seq_len[bi]
                if lo >= r:
                    d_ids.append([end_id])
                    continue
                d_ids.append(draft_out["output_ids"][bi, 0, lo:r].tolist())
                if args.use_logits and draft_out.get("generation_logits") is not None:
                    d_logits[bi] = draft_out["generation_logits"][bi, 0, -len(d_ids[bi]) :, :]

            # Target model: verify with FSD
            target_kwargs = dict(
                batch_input_ids=prefix,
                max_new_tokens=max_draft + 1,
                end_id=end_id,
                pad_id=pad_id,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                return_dict=True,
                output_sequence_lengths=True,
                draft_tokens_list=d_ids,
                fsd_threshold=fsd_threshold,
                fsd_divergence_type=args.fsd_divergence_type,
            )
            if args.use_logits and all(lg is not None for lg in d_logits):
                target_kwargs["draft_logits_list"] = d_logits
            with torch.no_grad():
                target_out = target_runner.generate(**target_kwargs)
            t_seq_len = target_out["sequence_lengths"][:, 0].tolist()

            # Update outputs and prefix for next iteration
            prefix_next = []
            batch_slot_next = []
            for bi in range(batch_size_cur):
                gbi = batch_slot[bi]
                lo = prefix_len[bi]
                r = min(t_seq_len[bi], max_seq_len[gbi])
                t_seq = target_out["output_ids"][bi, 0, :r]
                output_ids[gbi, 0, lo:r] = t_seq[lo:r]
                sequence_lengths[gbi, 0] = r
                if r >= max_seq_len[gbi]:
                    continue
                if end_id in t_seq[lo:].tolist():
                    continue
                if r == lo:
                    continue
                prefix_next.append(t_seq)
                batch_slot_next.append(gbi)
            prefix = prefix_next
            batch_slot = batch_slot_next
            if len(prefix) == 0:
                break
        elapsed_sec = time.perf_counter() - t0
        num_generated = sum(
            int(sequence_lengths[i, 0].item()) - input_lengths[i] for i in range(batch_size)
        )
        outputs = {"output_ids": output_ids, "sequence_lengths": sequence_lengths}
        return outputs, num_generated, elapsed_sec

    def run_one_generate(fsd_threshold):
        if use_draft_model:
            return _run_draft_target_loop(fsd_threshold)
        generate_kwargs = dict(
            batch_input_ids=batch_input_ids,
            max_new_tokens=max_output_tokens,
            end_id=end_id,
            pad_id=pad_id,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            return_dict=True,
            output_sequence_lengths=True,
            draft_tokens_list=draft_tokens_list,
            fsd_threshold=fsd_threshold,
            fsd_divergence_type=args.fsd_divergence_type,
        )
        if draft_logits_list is not None:
            generate_kwargs["draft_logits_list"] = draft_logits_list
        with torch.no_grad():
            t0 = time.perf_counter()
            outputs = runner.generate(**generate_kwargs)
            elapsed_sec = time.perf_counter() - t0
        seq_lengths = outputs["sequence_lengths"]
        num_generated = sum(
            int(seq_lengths[i, 0].item()) - input_lengths[i] for i in range(batch_size)
        )
        return outputs, num_generated, elapsed_sec

    if args.benchmark:
        if runtime_rank == 0:
            print(
                f"Benchmark: FSD divergence type={args.fsd_divergence_type}, "
                f"generating up to {max_output_tokens} tokens per run"
            )
            print("Thresholds:", BENCHMARK_FSD_THRESHOLDS)
            if use_draft_model:
                print(f"Draft model:  {args.draft_engine_dir}")
                print(f"Target model: {engine_dir} (from config: {model_name})")
            else:
                print(f"Target model: {engine_dir} (from config: {model_name})")
                print("Draft tokens: placeholder only (no draft model)")
            print(f"Input prompt: {args.input_text!r}")
            print("")
        results = []
        for thresh in BENCHMARK_FSD_THRESHOLDS:
            if runtime_rank == 0:
                logger.info("Running FSD threshold=%.1f", thresh)
            outputs, num_generated, elapsed_sec = run_one_generate(thresh)
            tokens_per_sec = num_generated / elapsed_sec if elapsed_sec > 0 else 0.0
            results.append((thresh, num_generated, elapsed_sec, tokens_per_sec))
            if runtime_rank == 0:
                output_ids = outputs["output_ids"]
                seq_lengths = outputs["sequence_lengths"]
                for i in range(batch_size):
                    out_len = seq_lengths[i, 0].item()
                    gen_ids = output_ids[i, 0, input_lengths[i] : out_len].tolist()
                    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                    print(
                        f"  threshold={thresh:.1f} Output [{i}]: {text!r} "
                        f"(tokens={len(gen_ids)}, time={elapsed_sec:.3f}s, "
                        f"{tokens_per_sec:.2f} tok/s)"
                    )
        if runtime_rank == 0:
            print("\n--- Summary: FSD threshold -> tokens/second ---")
            for thresh, num_gen, elapsed, tps in results:
                print(f"  {thresh:.1f} -> {tps:.2f}")
            print("---")
        return 0

    if runtime_rank == 0:
        logger.info(
            "Running generate with FSD: fsd_threshold=%s, fsd_divergence_type=%s",
            args.fsd_threshold,
            args.fsd_divergence_type,
        )
    outputs, num_generated, elapsed_sec = run_one_generate(args.fsd_threshold)

    if runtime_rank == 0:
        output_ids = outputs["output_ids"]
        seq_lengths = outputs["sequence_lengths"]
        for i in range(batch_size):
            out_len = seq_lengths[i, 0].item()
            gen_ids = output_ids[i, 0, input_lengths[i] : out_len].tolist()
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            print(f"Output [{i}]: {text!r} (len={len(gen_ids)})")
        print("FSD test finished successfully.")
    return 0


def main():
    args = parse_args()
    tensorrt_llm.logger.set_level(args.log_level)
    # Default prompt for benchmark often stops after 1 token; use a prompt that tends to generate more
    if args.benchmark and args.input_text == ["Hello, world"]:
        args.input_text = ["Once upon a time there was a little model that loved to generate text."]
    return run_fsd_test(args)


if __name__ == "__main__":
    sys.exit(main())
