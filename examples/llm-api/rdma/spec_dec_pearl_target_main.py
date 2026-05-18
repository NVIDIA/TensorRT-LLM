# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# All rights reserved.
# SPDX-License-Identifier: Apache-2.0
r"""PEARL target driver script.

Sibling of ``spec_dec_target_main.py`` that selects
``PEARLDecodingConfig`` instead of ``DraftTargetDecodingConfig``. Pairs
with ``pearl_draft_server.py``, which adds post-verify pipelining on the
draft side (pre-drafts the next round while the target verifies the
current round). Greedy verification is paper-faithful and yields
byte-identical output to ``--baseline``.

Usage:

    GPU_ID=6 python3 examples/llm-api/rdma/spec_dec_pearl_target_main.py \\
        --target-model /scratch.trt_llm_data/llm-models/Qwen3/Qwen3-4B \\
        --draft-model  /scratch.trt_llm_data/llm-models/Qwen3/Qwen3-0.6B \\
        --draft-host 127.0.0.1 --draft-control-port 47331 \\
        --transport ibverbs --nic mlx5_0 \\
        --prompt "Explain GPUDirect RDMA in one short sentence." \\
        --max-tokens 16 --max-draft-len 4

If ibverbs is misbehaving (cross-NIC routing, missing nvidia_peermem),
fall back with ``--transport tcp``. The same draft server image speaks
both transports — only the target side switches.
"""

import argparse
import os


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-model", required=True)
    ap.add_argument(
        "--draft-model", required=True, help="Path the target tells the draft server to lazy-load"
    )
    ap.add_argument("--draft-host", default="127.0.0.1")
    ap.add_argument(
        "--draft-port",
        type=int,
        default=0,
        help="Draft data-plane port. 0 lets the draft server allocate and return one via control-plane ack.",
    )
    ap.add_argument(
        "--draft-control-port",
        type=int,
        default=47331,
        help="Draft control-plane port (TcpModelInit / TcpPromptInit)",
    )
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--max-draft-len", type=int, default=4)
    ap.add_argument(
        "--transport",
        choices=["tcp", "ibverbs", "doca"],
        default="ibverbs",
        help="Draft offload transport.",
    )
    ap.add_argument(
        "--nic",
        default="mlx5_0",
        help="NIC device for --transport ibverbs.",
    )
    ap.add_argument("--prompt", default="Explain GPUDirect RDMA in one short sentence.")
    ap.add_argument(
        "--baseline",
        action="store_true",
        help="Skip spec-dec; just run fresh LLM(target).generate for byte-match check.",
    )
    ap.add_argument(
        "--max-num-requests",
        type=int,
        default=4096,
        help="Channel slot count sent to the draft server during TcpModelInit.",
    )
    ap.add_argument(
        "--kv-cache-free-fraction",
        type=float,
        default=0.4,
        help="Target-side KV cache free GPU memory fraction.",
    )
    ap.add_argument(
        "--adaptive-gamma",
        action="store_true",
        default=True,
        help="Enable PEARL profile-based adaptive gamma selection.",
    )
    ap.add_argument(
        "--no-adaptive-gamma",
        dest="adaptive_gamma",
        action="store_false",
        help="Disable adaptive gamma; use max_draft_len for all batch sizes.",
    )
    ap.add_argument(
        "--trace-log",
        default="",
        help="Write target-side PEARL communication trace as JSONL to this file.",
    )
    args = ap.parse_args()
    if args.trace_log:
        os.environ["PEARL_TARGET_TRACE_PATH"] = args.trace_log

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.speculative.pearl_trace import log as pearl_trace_log
    from tensorrt_llm.llmapi import KvCacheConfig, PEARLDecodingConfig

    prompt_tokens = []
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
        prompt_tokens = [int(t) for t in tokenizer.encode(args.prompt, add_special_tokens=True)]
    except Exception as exc:
        pearl_trace_log(
            "target",
            "prompt_tokenize_failed",
            prompt_text=args.prompt,
            error=repr(exc),
        )
    pearl_trace_log(
        "target",
        "run_start",
        prompt_text=args.prompt,
        prompt_tokens=prompt_tokens,
        prompt_token_count=len(prompt_tokens),
        target_model=args.target_model,
        draft_model=args.draft_model,
        transport=args.transport,
        nic=args.nic,
        max_tokens=int(args.max_tokens),
        max_draft_len=int(args.max_draft_len),
        baseline=bool(args.baseline),
    )

    common_kv = KvCacheConfig(
        enable_block_reuse=True,
        free_gpu_memory_fraction=args.kv_cache_free_fraction,
    )

    if args.baseline:
        llm = LLM(
            model=args.target_model,
            kv_cache_config=common_kv,
            cuda_graph_config=None,
        )
    else:
        spec_kwargs = dict(
            max_draft_len=args.max_draft_len,
            speculative_model=args.draft_model,
            draft_offload_enabled=True,
            draft_offload_v2=True,
            draft_offload_server_host=args.draft_host,
            draft_offload_server_port=args.draft_port or 1,
            draft_offload_v2_max_num_requests=args.max_num_requests,
            draft_offload_v2_transport=args.transport,
            draft_offload_v2_kv_cache_free_fraction=args.kv_cache_free_fraction,
            draft_offload_v2_tcp_prompt_port=args.draft_control_port,
            pearl_adaptive_gamma=args.adaptive_gamma,
        )
        if args.transport in ("tcp", "ibverbs", "doca"):
            spec_kwargs["draft_offload_v2_model_path"] = args.draft_model
        if args.transport in ("ibverbs", "doca"):
            spec_kwargs["draft_offload_nic_name"] = args.nic
        spec = PEARLDecodingConfig(**spec_kwargs)
        llm = LLM(
            model=args.target_model,
            speculative_config=spec,
            kv_cache_config=common_kv,
            cuda_graph_config=None,
        )

    sampling = SamplingParams(
        max_tokens=int(args.max_tokens),
        temperature=0.0,
        top_p=1.0,
    )
    outputs = llm.generate([args.prompt], sampling_params=sampling)
    out = outputs[0].outputs[0]
    pearl_trace_log(
        "target",
        "run_finish",
        generated_token_ids=[int(t) for t in out.token_ids],
        decoded_text=out.text,
    )
    print("=== generated token_ids ===")
    print(list(out.token_ids))
    print("=== decoded text ===")
    print(out.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
