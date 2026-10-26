# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""DS-R1 end-to-end functional check of the KV-cache dtype (FP8 vs NVFP4 "FP4 MLA").

Runs greedy generation for N MLPerf DS-R1 prompts on one node (TP4/EP4 with
attention DP) and stores the generated token ids, so that two runs that only
differ in ``kv_cache_config.dtype`` (and the FP4 MLA decode backend) can be
compared with ``--compare``.

Launch (4 GPUs, one task per GPU)::

    srun --ntasks-per-node=4 --mpi=pmix trtllm-llmapi-launch \
        python3 e2e_dsr1_kv_dtype_check.py --kv nvfp4 --out nvfp4.json
    python3 e2e_dsr1_kv_dtype_check.py --compare fp8.json nvfp4.json
"""

import argparse
import json
import os
import time

# A DeepSeek-R1 NVFP4 checkpoint and the MLPerf DS-R1 dataset pickle (columns tok_input,
# tok_input_len, dataset); set the paths for your environment.
MODEL = os.environ.get("DSR1_FP4_MODEL", "nvidia/DeepSeek-R1-FP4")
DATA = os.environ.get("DSR1_MLPERF_DATASET_PKL", "mlperf_deepseek_r1_dataset_4388_fp8_eval.pkl")


def load_prompts(n: int, max_isl: int):
    import pandas as pd

    df = pd.read_pickle(DATA)
    df = df[df["tok_input_len"] <= max_isl]
    # Spread over the dataset subsets: take every k-th eligible sample.
    step = max(1, len(df) // n)
    df = df.iloc[::step].head(n)
    return [list(map(int, ids)) for ids in df["tok_input"]], df["dataset"].tolist()


def run(args):
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import CudaGraphConfig, KvCacheConfig, MoeConfig

    prompts, subsets = load_prompts(args.n, args.max_isl)
    order = list(range(len(prompts)))
    if args.reverse:
        # Same requests, different batch placement (attention-DP rank and slot):
        # a control for batch-composition numerics.
        order.reverse()
    kv = KvCacheConfig(
        dtype=args.kv,
        enable_block_reuse=False,
        free_gpu_memory_fraction=args.kv_fraction,
        tokens_per_block=128 if args.kv == "nvfp4" else 32,
    )
    t0 = time.time()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        moe_expert_parallel_size=args.tp,
        enable_attention_dp=True,
        kv_cache_config=kv,
        max_batch_size=32,
        max_num_tokens=8192,
        max_seq_len=args.max_isl + args.max_tokens + 64,
        cuda_graph_config=CudaGraphConfig(batch_sizes=[1, 2, 4, 8, 16, 32], enable_padding=True),
        moe_config=MoeConfig(backend=args.moe_backend),
    )
    t1 = time.time()
    outputs = llm.generate(
        [prompts[i] for i in order],
        SamplingParams(max_tokens=args.max_tokens, logprobs=args.logprobs),
    )
    outputs = [out for _, out in sorted(zip(order, outputs), key=lambda item: item[0])]
    t2 = time.time()
    records = []
    for i, out in enumerate(outputs):
        comp = out.outputs[0]
        top = []
        for pos in comp.logprobs or []:
            top.append({str(tok): float(getattr(lp, "logprob", lp)) for tok, lp in pos.items()})
        records.append(
            {
                "idx": i,
                "subset": subsets[i],
                "isl": len(prompts[i]),
                "token_ids": list(comp.token_ids),
                "text": comp.text,
                "top": top,
            }
        )
    meta = {
        "kv": args.kv,
        "env_backend": os.environ.get("TRTLLM_FP4_MLA_ATTENTION_BACKEND"),
        "reverse": args.reverse,
        "load_s": t1 - t0,
        "gen_s": t2 - t1,
        "n": len(records),
    }
    with open(args.out, "w") as f:
        json.dump({"meta": meta, "records": records}, f, indent=1)
    print(f"E2E_DONE {meta}", flush=True)
    for r in records[:5]:
        print(f"--- [{r['idx']} {r['subset']} isl={r['isl']}] {r['text'][:300]!r}", flush=True)
    llm.shutdown()


def compare(a_path, b_path):
    a = json.load(open(a_path))
    b = json.load(open(b_path))
    prefix, exact, n, margins = [], 0, 0, []
    for ra, rb in zip(a["records"], b["records"]):
        ta, tb = ra["token_ids"], rb["token_ids"]
        k = 0
        while k < min(len(ta), len(tb)) and ta[k] == tb[k]:
            k += 1
        prefix.append(k)
        exact += int(ta == tb)
        n += 1
        margin = ""
        if k < min(len(ta), len(tb)) and ra.get("top") and k < len(ra["top"]):
            # Run A's log-prob gap between its own token and run B's token at the
            # first divergence (nan when B's token is outside A's top-N).
            pos = ra["top"][k]
            la = pos.get(str(ta[k]), float("nan"))
            lb = pos.get(str(tb[k]), float("nan"))
            margins.append(la - lb)
            margin = f" first_div_logprob_gap={la - lb:.3f}"
        print(
            f"idx={ra['idx']:2d} {ra['subset']:>10s} common_prefix={k:3d}/{len(ta)}{margin}",
            flush=True,
        )
    import math
    import statistics

    finite = [m for m in margins if not math.isnan(m)]
    if margins:
        print(
            f"first-divergence logprob gap (run A): n={len(margins)} in_topN={len(finite)} "
            f"median={statistics.median(finite) if finite else float('nan'):.3f} "
            f"max={max(finite) if finite else float('nan'):.3f}",
            flush=True,
        )

    print(
        f"COMPARE {a['meta']['kv']}/{a['meta']['env_backend']}/rev={a['meta'].get('reverse')} vs "
        f"{b['meta']['kv']}/{b['meta']['env_backend']}/rev={b['meta'].get('reverse')}: "
        f"n={n} exact={exact} "
        f"mean_prefix={statistics.mean(prefix):.1f} median_prefix={statistics.median(prefix)} "
        f">=50:{sum(p >= 50 for p in prefix)} >=20:{sum(p >= 20 for p in prefix)}",
        flush=True,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--kv", choices=["fp8", "nvfp4"], default="nvfp4")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--max-isl", type=int, default=2000)
    p.add_argument("--max-tokens", type=int, default=100)
    p.add_argument("--tp", type=int, default=4)
    p.add_argument("--kv-fraction", type=float, default=0.5)
    p.add_argument("--moe-backend", default="CUTEDSL")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--out", default="e2e.json")
    p.add_argument("--compare", nargs=2, default=None)
    p.add_argument("--reverse", action="store_true", help="submit the prompts in reverse order")
    p.add_argument("--logprobs", type=int, default=2)
    args = p.parse_args()
    if args.compare:
        compare(*args.compare)
    else:
        run(args)


if __name__ == "__main__":
    main()
