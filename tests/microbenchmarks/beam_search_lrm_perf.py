# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
r"""End-to-end beam-search benchmark shaped like a generative-recommendation workload.

Each request is a long prompt (a user's history as semantic-ID (SID) tokens) decoded for a few
tokens with a wide beam, the case the opt-in ``beam_shared_prefix`` library targets. The model is
a Qwen3 checkpoint with its embedding table extended by the SID tokens; prompts are token ids.
Run each arm (library on / off) in its own process: the library is chosen when the engine is built.

Usage::

    # 1. extend a Qwen3 checkpoint with the SID tokens (real weights kept)
    python beam_search_lrm_perf.py build-model --src $MODELS/Qwen3/Qwen3-0.6B --dst /tmp/lrm_model

    # 2. baseline and candidate, same prompts (same --seed)
    TLLM_WORKER_USE_SINGLE_PROCESS=1 python beam_search_lrm_perf.py run \
        --model-dir /tmp/lrm_model --beam-width 96 --min-history 500 --max-history 700 \
        --out base.json
    TLLM_WORKER_USE_SINGLE_PROCESS=1 TLLM_FMHA_LIBS=+beam_shared_prefix \
        python beam_search_lrm_perf.py run --model-dir /tmp/lrm_model --beam-width 96 \
        --min-history 500 --max-history 700 --out cascade.json

    # 3. compare beams and throughput
    python beam_search_lrm_perf.py compare base.json cascade.json

``TLLM_WORKER_USE_SINGLE_PROCESS=1`` is required for ``--count-iterations``. For an Nsight
Systems trace of the measured batches, pass ``--profile`` under
``nsys profile -c cudaProfilerApi --cuda-graph-trace node``.

Reported per arm (``run``):

* ``prompts_per_s``: prompts in a batch divided by the wall time of that
  ``generate`` call, which includes prefill; median over the measured batches.
* iteration counts (``--count-iterations``): executor iterations after warmup
  that were pure decode, mixed prefill and decode, or prefill only, and how
  many of each the library planned (the others ran the regular decode kernel),
  plus the host seconds spent in its per-step planning (``plan_seconds``).
"""

import argparse
import json
import os
import random
import statistics
import time

# One semantic ID is one code per codebook level, optionally wrapped in begin/end markers.
DEFAULT_CODEBOOK_SIZES = (1024, 512, 256)
SYSTEM_PROMPT = (
    "<|im_start|>system\nYou are a recommender. Given the user's engagement "
    "history, predict the next item.<|im_end|>\n<|im_start|>user\nHistory: "
)
ASSISTANT_PROMPT = "<|im_end|>\n<|im_start|>assistant\n"


def _dtype_kwarg(transformers) -> str:
    return "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"


def build_model(args) -> None:
    """Write ``--dst``: ``--src`` with the embedding table grown by the SID tokens."""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(
        args.src, **{_dtype_kwarg(transformers): torch.bfloat16}
    )
    base_vocab = model.config.vocab_size
    sizes = [int(size) for size in args.codebook_sizes.split(",")]
    extra_tokens = 2 + sum(sizes)
    model.resize_token_embeddings(base_vocab + extra_tokens)
    # resize_token_embeddings starts every new row at the mean embedding, which makes the SID
    # tokens identical. Draw distinct rows at the scale of the original ones instead.
    embed = model.get_input_embeddings().weight
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        std = embed[:base_vocab].float().std()
        embed[base_vocab:] = (torch.randn(embed[base_vocab:].shape, generator=generator) * std).to(
            embed.dtype
        )
        output = model.get_output_embeddings()
        if output is not None and output.weight.data_ptr() != embed.data_ptr():
            output.weight[base_vocab:] = embed[base_vocab:]
    os.makedirs(args.dst, exist_ok=True)
    model.save_pretrained(args.dst, safe_serialization=True)
    AutoTokenizer.from_pretrained(args.src).save_pretrained(args.dst)
    with open(os.path.join(args.dst, "lrm_perf_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "sid_base_id": base_vocab,
                "codebook_sizes": sizes,
                "src": os.path.abspath(args.src),
            },
            f,
        )
    print(f"wrote {args.dst}: vocab {base_vocab} -> {base_vocab + extra_tokens}")


def synthetic_prompts(model_dir, num_prompts, min_history, max_history, seed, item_markers=True):
    """Token-id prompts: ChatML text around a random SID history.

    With ``item_markers`` each history item is begin, one code per level, end,
    and the prompt ends on a begin marker; without, an item is only its codes.
    """
    from transformers import AutoTokenizer

    with open(os.path.join(model_dir, "lrm_perf_metadata.json"), encoding="utf-8") as f:
        metadata = json.load(f)
    sid_base = metadata["sid_base_id"]
    codebook_sizes = metadata.get("codebook_sizes", list(DEFAULT_CODEBOOK_SIZES))
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    head = tokenizer(SYSTEM_PROMPT, add_special_tokens=False)["input_ids"]
    tail = tokenizer(ASSISTANT_PROMPT, add_special_tokens=False)["input_ids"]
    sid_begin, sid_end = sid_base, sid_base + 1
    level_base = [sid_base + 2]
    for size in codebook_sizes[:-1]:
        level_base.append(level_base[-1] + size)
    rng = random.Random(seed)
    prompts = []
    for _ in range(num_prompts):
        history = []
        for _ in range(rng.randint(min_history, max_history)):
            if item_markers:
                history.append(sid_begin)
            history.extend(
                level_base[level] + rng.randrange(size) for level, size in enumerate(codebook_sizes)
            )
            if item_markers:
                history.append(sid_end)
        prompts.append(head + history + tail + ([sid_begin] if item_markers else []))
    return prompts


WORDS = (
    "the of and to in is that for it as was with be by on not he this are or his from at which "
    "but have an had they you were their one all we can her has there been if more when will would "
    "who so no she other its may these two like him into time has look more write go see number way "
    "could people my than first water been call who oil now find long down day did get come made "
    "part over new sound take only little work know place year live me back give most very after "
    "thing our just name good sentence man think say great where help through much before line "
    "right too mean old any same tell boy follow came want show also around form three small set "
    "put end does another well large must big even such because turn here why ask went men read "
    "need land different home us move try kind hand picture again change off play spell air away "
    "animal house point page letter mother answer found study still learn should world high every "
    "near add food between own below country plant last school father keep tree never start city "
    "earth eye light thought head under story saw left don't few while along might close something "
    "seem next hard open example begin life always those both paper together got group often run "
    "important until children side feet car mile night walk white sea began grow took river four "
    "carry state once book hear stop without second later miss idea enough eat face watch far "
).split()


def text_prompts(model_dir, num_prompts, min_tokens, max_tokens, seed):
    """Natural-looking English prompts of ``min_tokens``..``max_tokens`` tokens."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    rng = random.Random(seed)
    prompts = []
    for _ in range(num_prompts):
        target = rng.randint(min_tokens, max_tokens)
        words = []
        ids = []
        while len(ids) < target:
            sentence = " ".join(rng.choice(WORDS) for _ in range(rng.randint(6, 16))) + ". "
            words.append(sentence)
            ids = tokenizer("".join(words), add_special_tokens=False)["input_ids"]
        prompts.append(ids[:target])
    return prompts


class IterationCounter:
    """Counts executor iterations by batch composition at the library's plan hook.

    The hook runs on the host every iteration, even when attention is replayed from a CUDA graph.
    """

    def __init__(self):
        from tensorrt_llm._torch.attention.backends.fmha import beam_shared_prefix as lib

        self._lib = lib
        self._plan_step = lib.plan_step
        self.reset()
        counter = self

        def counted_plan_step(metadata):
            start = time.perf_counter()
            result = counter._plan_step(metadata)
            counter.plan_seconds += time.perf_counter() - start
            counter._record(metadata)
            return result

        lib.plan_step = counted_plan_step

    def reset(self):
        self.plan_seconds = 0.0
        self.counts = {
            "decode_only": 0,
            "mixed": 0,
            "prefill_only": 0,
            "decode_planned": 0,
            "mixed_planned": 0,
        }

    def _record(self, metadata):
        contexts, generations = metadata.num_contexts, metadata.num_generations
        state = metadata.fmha_plan_caches.get(self._lib.LIB_NAME)
        planned = state is not None and state.plan is not None
        if generations and not contexts:
            self.counts["decode_only"] += 1
            self.counts["decode_planned"] += int(planned)
        elif generations:
            self.counts["mixed"] += 1
            self.counts["mixed_planned"] += int(planned)
        else:
            self.counts["prefill_only"] += 1

    def snapshot(self):
        return dict(self.counts, plan_seconds=self.plan_seconds)


def run(args) -> None:
    import torch

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig

    num_prompts = args.batch_size * (args.warmup_batches + args.measure_batches)
    if args.prompt_kind == "text":
        prompts = text_prompts(
            args.model_dir, num_prompts, args.min_text_tokens, args.max_text_tokens, args.seed
        )
    else:
        prompts = synthetic_prompts(
            args.model_dir,
            num_prompts,
            args.min_history,
            args.max_history,
            args.seed,
            item_markers=not args.no_item_markers,
        )
    longest = max(len(p) for p in prompts)
    cuda_graph_config = (
        None
        if args.no_cuda_graph
        else CudaGraphConfig(batch_sizes=sorted({1, 2, 4, 8, 16, 32, 64, 128, args.batch_size}))
    )
    counter = IterationCounter() if args.count_iterations else None
    llm = LLM(
        model=args.model_dir,
        max_beam_width=args.beam_width,
        max_batch_size=args.batch_size,
        max_num_tokens=args.max_num_tokens,
        max_seq_len=args.max_seq_len or longest + args.max_new_tokens + 1,
        kv_cache_config=KvCacheConfig(
            enable_block_reuse=False, free_gpu_memory_fraction=args.kv_cache_fraction
        ),
        cuda_graph_config=cuda_graph_config,
        skip_tokenizer_init=True,
    )
    sampling = SamplingParams(
        use_beam_search=True,
        n=args.beam_width,
        best_of=args.beam_width,
        max_tokens=args.max_new_tokens,
        ignore_eos=True,
        end_id=-1,
        # Beam search returns only the sampled token's logprob.
        logprobs=0,
    )

    batch_seconds, beams = [], []
    iterations = None
    with llm:
        for index in range(args.warmup_batches + args.measure_batches):
            batch = prompts[index * args.batch_size : (index + 1) * args.batch_size]
            measured = index >= args.warmup_batches
            if index == args.warmup_batches:
                if counter is not None:
                    counter.reset()
                if args.profile:
                    torch.cuda.cudart().cudaProfilerStart()
            torch.cuda.nvtx.range_push(f"{'batch' if measured else 'warmup_batch'}_{index}")
            start = time.perf_counter()
            outputs = llm.generate(batch, sampling_params=sampling)
            elapsed = time.perf_counter() - start
            torch.cuda.nvtx.range_pop()
            if measured:
                batch_seconds.append(elapsed)
                beams.extend(
                    [(tuple(b.token_ids), b.cumulative_logprob) for b in out.outputs]
                    for out in outputs
                )
        if args.profile:
            torch.cuda.cudart().cudaProfilerStop()
        if counter is not None:
            iterations = counter.snapshot()

    per_batch = [args.batch_size / s for s in batch_seconds]
    result = {
        "config": {k: v for k, v in vars(args).items() if k != "func"},
        "env": {
            k: os.environ.get(k) for k in ("TLLM_FMHA_LIBS", "TLLM_BEAM_SHARED_PREFIX_MAX_TAIL")
        },
        "prompt_tokens_mean": statistics.mean(len(p) for p in prompts),
        "batch_seconds": batch_seconds,
        "prompts_per_s_median": statistics.median(per_batch),
        "prompts_per_s_per_batch": per_batch,
        "iterations": iterations,
        "beams": beams,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f)
    print(
        f"prompts/s median {result['prompts_per_s_median']:.2f} over {len(per_batch)} batches "
        f"(prompt tokens mean {result['prompt_tokens_mean']:.0f}); iterations {iterations}"
    )


def compare(args) -> None:
    """Throughput ratio and beam agreement between two ``run`` outputs."""
    with open(args.baseline, encoding="utf-8") as f:
        base = json.load(f)
    with open(args.candidate, encoding="utf-8") as f:
        cand = json.load(f)
    ratio = cand["prompts_per_s_median"] / base["prompts_per_s_median"]
    print(
        f"prompts/s median: baseline {base['prompts_per_s_median']:.2f}, "
        f"candidate {cand['prompts_per_s_median']:.2f}, ratio {ratio:.3f}"
    )
    overlaps, top_match, lp_diffs = [], 0, []
    for ref_beams, cand_beams in zip(base["beams"], cand["beams"]):
        ref_tokens = [tuple(t) for t, _ in ref_beams]
        cand_tokens = [tuple(t) for t, _ in cand_beams]
        overlaps.append(len(set(ref_tokens) & set(cand_tokens)) / len(ref_tokens))
        top_match += ref_tokens[0] == cand_tokens[0]
        ref_lp = dict(zip(ref_tokens, (lp for _, lp in ref_beams)))
        for tokens, lp in zip(cand_tokens, (lp for _, lp in cand_beams)):
            if tokens in ref_lp and lp is not None and ref_lp[tokens] is not None:
                lp_diffs.append(abs(lp - ref_lp[tokens]))
    print(
        f"beams: mean overlap {statistics.mean(overlaps):.3f}, min {min(overlaps):.3f}; "
        f"top beam equal in {top_match}/{len(overlaps)} prompts"
    )
    if lp_diffs:
        print(
            f"cumulative logprob |diff| on shared beams: median {statistics.median(lp_diffs):.4f}, "
            f"max {max(lp_diffs):.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(required=True)

    build = sub.add_parser("build-model", help="Extend a Qwen3 checkpoint with SID tokens.")
    build.add_argument("--src", required=True, help="Source HF checkpoint directory.")
    build.add_argument("--dst", required=True, help="Output directory.")
    build.add_argument(
        "--codebook-sizes",
        default=",".join(str(size) for size in DEFAULT_CODEBOOK_SIZES),
        help="Comma-separated SID codebook size per level.",
    )
    build.set_defaults(func=build_model)

    run_p = sub.add_parser("run", help="Time beam-search generation.")
    run_p.add_argument("--model-dir", required=True)
    run_p.add_argument("--beam-width", type=int, default=96)
    run_p.add_argument("--min-history", type=int, default=500, help="SID items per prompt.")
    run_p.add_argument("--max-history", type=int, default=700, help="SID items per prompt.")
    run_p.add_argument(
        "--prompt-kind",
        choices=("sid", "text"),
        default="sid",
        help="sid: semantic-ID history (needs a build-model directory); text: English prompts.",
    )
    run_p.add_argument(
        "--no-item-markers",
        action="store_true",
        help="History items are only their codes (no begin/end marker tokens).",
    )
    run_p.add_argument("--min-text-tokens", type=int, default=2500)
    run_p.add_argument("--max-text-tokens", type=int, default=3500)
    run_p.add_argument("--max-new-tokens", type=int, default=3)
    run_p.add_argument("--batch-size", type=int, default=128, help="Prompts per generate().")
    run_p.add_argument("--warmup-batches", type=int, default=1)
    run_p.add_argument("--measure-batches", type=int, default=3)
    run_p.add_argument("--max-seq-len", type=int, default=None)
    run_p.add_argument("--max-num-tokens", type=int, default=16384)
    run_p.add_argument("--kv-cache-fraction", type=float, default=0.8)
    run_p.add_argument("--no-cuda-graph", action="store_true")
    run_p.add_argument("--count-iterations", action="store_true")
    run_p.add_argument("--profile", action="store_true", help="Bracket measured batches.")
    run_p.add_argument("--seed", type=int, default=0)
    run_p.add_argument("--out", required=True)
    run_p.set_defaults(func=run)

    cmp_p = sub.add_parser("compare", help="Compare two run outputs.")
    cmp_p.add_argument("baseline")
    cmp_p.add_argument("candidate")
    cmp_p.set_defaults(func=compare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
