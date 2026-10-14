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
"""JIT prefetch coverage check: which kernels still compile during serving.

Builds the model with dummy weights (optionally fewer layers, keeping every
layer kind), serves a workload chosen to reach many kernel variants, and
reports every JIT compile and autotune benchmark left on the executor thread
after warmup, from the ``[JIT stats]`` events. Each one is a variant no
provider covers.

    TLLM_JIT_PREFETCH=1 python3 jit_prefetch_coverage.py --model <hf dir> \
        [--tp N] [--layers N] [--config extra.yaml] [--report out.json]

Exits 1 if anything compiled or autotuned on the executor after warmup.
"""

import argparse
import json
import os
import random
import re
import sys
import time

_EVENT_RE = re.compile(
    r"\[JIT stats\] rank (\d+) t=[0-9.]+ "
    r"(COMPILE on executor|AUTOTUNE on executor|CUTE_DSL compile on executor|"
    r"DeepGEMM compile on executor|DEEP_GEMM compile in process)\s+(\S+)"
)


def shrink_checkpoint(src: str, dst: str, layers: int) -> str:
    """A copy of ``src`` (config and tokenizer files only, no weights) whose
    config keeps the first ``layers`` decoder layers.

    Every per-layer list in the config is cut to the same length, so models
    that check those lists against ``num_hidden_layers`` still load. Layer
    index lists (Kimi K3's ``kda_layers`` / ``full_attn_layers``, 1-based) keep
    only the indices that remain. Pick ``layers`` so every layer kind the
    model has appears at least once; the run log names the kinds built.
    """
    import shutil

    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        p = os.path.join(src, name)
        if os.path.isfile(p) and not name.endswith((".safetensors", ".bin", ".pt", ".pth")):
            if name not in ("model.safetensors.index.json",):
                shutil.copy2(p, os.path.join(dst, name))
    with open(os.path.join(src, "config.json")) as f:
        cfg = json.load(f)

    def cut(c):
        n = c.get("num_hidden_layers")
        if not isinstance(n, int):
            return
        for k, v in list(c.items()):
            if isinstance(v, list) and len(v) == n:
                c[k] = v[:layers]
            elif isinstance(v, dict):
                for kk, vv in list(v.items()):
                    if isinstance(vv, list) and len(vv) == n:
                        v[kk] = vv[:layers]
        lac = c.get("linear_attn_config")
        if isinstance(lac, dict):
            for k in ("kda_layers", "full_attn_layers"):
                if isinstance(lac.get(k), list):
                    lac[k] = [i for i in lac[k] if i <= layers]
        c["num_hidden_layers"] = layers

    for key in ("text_config", "llm_config", "language_config"):
        if isinstance(cfg.get(key), dict):
            cut(cfg[key])
    cut(cfg)
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    return dst


def workload(max_len: int, seed: int = 0):
    """Prompt lengths and decode lengths chosen to reach many variants.

    Prefill: ragged lengths, lengths on and off the 16 / 64-token boundaries,
    single-token prompts. Decode: batches of every size from 1 up, as the
    requests finish at different times. A second pass reuses prefixes
    (cached-prefix prefill, initial states for the linear-attention layers).
    """
    rng = random.Random(seed)
    lens = [1, 2, 15, 16, 17, 63, 64, 65, 127, 128, 129, 255, 511, 1000, 2047]
    lens += [rng.randint(1, max_len) for _ in range(17)]
    lens = [min(n, max_len) for n in lens]
    out_lens = [1 + (i * 7) % 48 for i in range(len(lens))]
    return lens, out_lens


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--ep", type=int, default=None)
    ap.add_argument("--layers", type=int, default=None, help="keep only the first N layers")
    ap.add_argument("--workdir", default="/tmp/jitp_coverage", help="where a shrunk config goes")
    ap.add_argument("--config", default=None, help="extra LLM args YAML")
    ap.add_argument("--max-num-tokens", type=int, default=4096)
    ap.add_argument("--max-batch-size", type=int, default=32)
    ap.add_argument("--max-seq-len", type=int, default=4096)
    ap.add_argument("--report", default=None)
    ap.add_argument("--real-weights", action="store_true")
    args = ap.parse_args()

    import yaml

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import KvCacheConfig

    extra = {}
    if args.config:
        with open(args.config) as f:
            extra = yaml.safe_load(f) or {}
    kv = extra.pop("kv_cache_config", {}) or {}
    kv.setdefault("free_gpu_memory_fraction", 0.5)
    kv.setdefault("enable_block_reuse", True)
    model = args.model
    if args.layers:
        if args.real_weights:
            raise SystemExit("--layers needs dummy weights")
        model = shrink_checkpoint(
            args.model, os.path.join(args.workdir, os.path.basename(args.model.rstrip("/"))), args.layers
        )
    kwargs = dict(
        model=model,
        tensor_parallel_size=args.tp,
        max_num_tokens=args.max_num_tokens,
        max_batch_size=args.max_batch_size,
        max_seq_len=args.max_seq_len,
        trust_remote_code=True,
        kv_cache_config=KvCacheConfig(**kv),
    )
    if args.ep is not None:
        kwargs["moe_expert_parallel_size"] = args.ep
    if not args.real_weights:
        kwargs["load_format"] = "dummy"
    kwargs.update(extra)

    t0 = time.time()
    llm = LLM(**kwargs)
    init_s = time.time() - t0
    print(f"COVERAGE init {init_s:.1f} s", flush=True)

    tok = llm.tokenizer
    vocab = getattr(tok, "vocab_size", None) or 32000
    in_lens, out_lens = workload(args.max_seq_len - 64)
    rng = random.Random(1)
    prompts = [[rng.randint(100, vocab - 1) for _ in range(n)] for n in in_lens]
    params = [SamplingParams(max_tokens=m, ignore_eos=True) for m in out_lens]
    t1 = time.time()
    llm.generate(prompts, params)
    # Cached prefixes: the same prompts extended, so prefill starts mid-sequence.
    ext = [p + [rng.randint(100, vocab - 1) for _ in range(1 + i % 37)] for i, p in enumerate(prompts)]
    llm.generate(ext, [SamplingParams(max_tokens=4, ignore_eos=True) for _ in ext])
    print(f"COVERAGE generate {time.time() - t1:.1f} s", flush=True)
    llm.shutdown()
    return 0


def summarize(log_path: str, report_path: str = None) -> int:
    """Group the executor-side JIT events in a run log by kind and kernel."""
    found = {}
    with open(log_path, errors="replace") as f:
        for line in f:
            m = _EVENT_RE.search(line)
            if m:
                key = (m.group(2), m.group(3))
                found[key] = found.get(key, 0) + 1
    rows = sorted(found.items(), key=lambda kv: (-kv[1], kv[0]))
    for (kind, name), n in rows:
        print(f"UNCOVERED {n:5d}  {kind:32s} {name}")
    if report_path:
        with open(report_path, "w") as f:
            json.dump([{"kind": k, "kernel": n, "count": c} for (k, n), c in rows], f, indent=1)
    print(f"COVERAGE {'FAIL' if rows else 'PASS'}: {len(rows)} uncovered kernel(s)")
    return 1 if rows else 0


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "--summarize":
        sys.exit(summarize(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None))
    os.environ.setdefault("TLLM_JIT_STATS", "1")
    sys.exit(main())
