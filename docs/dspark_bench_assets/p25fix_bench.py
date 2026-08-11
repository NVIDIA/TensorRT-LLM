#!/usr/bin/env python3
"""Round-style throughput bench: bs concurrent completions, JSON-line summary.

Usage: p25fix_bench.py <host> <port> <tag> <summary_file>
Single-threaded asyncio client (no thread-pool: 1024 threads exhausts login
nodes and starves its own concurrency). Run node-local against the serve.
Mirrors the p25 rounds methodology: poetry (single prompt fan-out) and arena
(question cycle), bs in {512, 1024} global, 5 reps, max_tokens 768, temp 0.7;
metric is aggregate completion tokens / wall seconds. A leading warm rep per
(dataset, bs) is tagged rep=-1 so steady-state reps are comparable.
"""
import asyncio
import json
import os
import sys
import time

import aiohttp

HOST, PORT, TAG, SUMMARY = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
RUNS = "/lustre/fsw/coreai_comparch_trtllm/laliao/dspark-runs"
MODEL = (sys.argv[5] if len(sys.argv) > 5 else
         "/lustre/fsw/coreai_comparch_trtllm/laliao/llm-models/DeepSeek-V4-Pro-DSpark")
URL = f"http://{HOST}:{PORT}/v1/completions"

poetry = open(f"{RUNS}/poetry_prompt.txt").read().strip()
arena = [json.loads(l)["prompt"] for l in open(f"{RUNS}/arena_questions.jsonl")]


async def one(session, prompt):
    body = {"model": MODEL, "prompt": prompt,
            "max_tokens": 768, "temperature": 0.7}
    try:
        async with session.post(URL, json=body) as r:
            d = await r.json()
        return d["usage"]["completion_tokens"], len(d["choices"][0]["text"])
    except Exception:
        return 0, 0


async def bench(session, dataset, prompts, bs, rep):
    t0 = time.time()
    results = await asyncio.gather(
        *[one(session, prompts[i % len(prompts)]) for i in range(bs)])
    wall = time.time() - t0
    toks = sum(r[0] for r in results)
    chars = sum(r[1] for r in results)
    fails = sum(1 for r in results if r[0] == 0)
    line = json.dumps({
        "tag": f"{TAG}_{dataset}", "bs": bs, "rep": rep,
        "completion_tokens": toks, "wall_s": round(wall, 2),
        "out_tok_per_s": round(toks / wall, 1),
        "chars_per_tok": round(chars / max(toks, 1), 2), "fails": fails,
    })
    with open(SUMMARY, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


async def main():
    timeout = aiohttp.ClientTimeout(total=1800)
    conn = aiohttp.TCPConnector(limit=1100)
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as s:
        bs_list = tuple(
            int(x) for x in os.environ.get("BENCH_BS_LIST", "512,1024").split(","))
        for dataset, prompts in (("poetry", [poetry]), ("arena", arena)):
            for bs in bs_list:
                await bench(s, dataset, prompts, bs, -1)  # warm rep
                for rep in range(5):
                    await bench(s, dataset, prompts, bs, rep)
    print("BENCH-DONE", TAG, flush=True)


asyncio.run(main())
