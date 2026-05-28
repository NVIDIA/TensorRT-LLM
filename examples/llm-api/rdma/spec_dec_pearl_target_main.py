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
import json
import os
import socket
import threading
import time


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
    ap.add_argument(
        "--warmup-tokens",
        type=int,
        default=0,
        help=(
            "Run one unmeasured PEARL request with this many max tokens after both "
            "target and draft models are loaded. 0 disables application-level warmup."
        ),
    )
    ap.add_argument("--max-draft-len", type=int, default=4)
    ap.add_argument(
        "--tp-size",
        type=int,
        default=1,
        help="Target model tensor parallel size.",
    )
    ap.add_argument(
        "--transport",
        choices=["tcp", "ibverbs", "doca", "shm", "cudaipc"],
        default="ibverbs",
        help="Draft offload transport.",
    )
    ap.add_argument(
        "--nic",
        default="mlx5_0",
        help="NIC device for --transport ibverbs.",
    )
    ap.add_argument(
        "--shm-name",
        dest="shm_name",
        default="pearl_shm_default",
        help=(
            "Shared-memory region prefix for --transport shm. Must match "
            "between target and draft when multiple PEARL pairs share the host."
        ),
    )
    ap.add_argument(
        "--cudaipc-name",
        dest="cudaipc_name",
        default="pearl_ipc_default",
        help=(
            "CPU meta-region prefix for --transport cudaipc (the GPU rings "
            "are exchanged through cudaIpc handles stored inside)."
        ),
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
    ap.add_argument(
        "--no-early-draft-init",
        dest="early_draft_init",
        action="store_false",
        default=True,
        help="Disable sending TcpModelInit before target LLM construction.",
    )
    ap.add_argument(
        "--no-chat-template",
        dest="chat_template",
        action="store_false",
        default=True,
        help="Use the raw --prompt string instead of tokenizer.apply_chat_template.",
    )
    ap.add_argument(
        "--cuda-graph",
        dest="cuda_graph",
        action="store_true",
        default=False,
        help=(
            "Use LLM's default CudaGraphConfig instead of disabling CUDA graphs. "
            "PEARL is compatible with CUDA graphs and this typically closes a "
            "large per-token Python/launch overhead gap on multi-layer models."
        ),
    )
    ap.add_argument(
        "--nsys-profile",
        dest="nsys_profile",
        action="store_true",
        default=False,
        help=(
            "Wrap the measured llm.generate(...) call with "
            "torch.cuda.profiler.start()/stop(). Pair with "
            "`nsys profile --capture-range=cudaProfilerApi ...` to record only "
            "the steady-state inference (skips model load, CUDA graph capture, "
            "and warmup)."
        ),
    )
    ap.add_argument(
        "--attn-backend",
        dest="attn_backend",
        default=None,
        help=(
            "Override LLM's default attention backend (e.g. TRTLLM, FLASHINFER, "
            "FLASHATTENTION). Left empty means LLM picks its default."
        ),
    )
    args = ap.parse_args()
    if args.trace_log:
        os.environ["PEARL_TARGET_TRACE_PATH"] = args.trace_log

    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm._torch.speculative.pearl_trace import log as pearl_trace_log
    from tensorrt_llm.llmapi import KvCacheConfig, PEARLDecodingConfig

    def _send_early_model_init():
        """Start draft model loading before constructing the target LLM.

        The regular offload layer still sends TcpModelInit later when it
        starts the data-plane channel. That second request is cheap because
        the draft server reuses the already-loaded model and returns the
        selected data port. This early request exists only to overlap draft
        model load with target model load.
        """
        msg = {
            "msg_type": "model_init",
            "model_path": str(args.draft_model),
            "dtype": "bfloat16",
            "max_draft_len": int(args.max_draft_len),
            "kv_cache_free_fraction": float(args.kv_cache_free_fraction),
            "extra_kwargs_json": json.dumps(
                {
                    "transport": str(args.transport),
                    "data_port": int(args.draft_port or 1),
                    "nic_name": str(args.nic),
                    "max_num_requests": int(args.max_num_requests),
                    "early_init": True,
                }
            ),
        }
        pearl_trace_log("target", "early_control_model_init_send", message=msg)
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(900.0)
        try:
            sock.connect((str(args.draft_host), int(args.draft_control_port)))
            sock.sendall((json.dumps(msg) + "\n").encode("utf-8"))
            chunks = []
            while True:
                ch = sock.recv(4096)
                if not ch:
                    raise RuntimeError("draft server closed during early model-init ack")
                chunks.append(ch)
                if b"\n" in ch:
                    break
            ack = json.loads(b"".join(chunks).split(b"\n", 1)[0].decode("utf-8"))
        finally:
            sock.close()
        if ack.get("status") != "ok":
            raise RuntimeError(
                "early TcpModelInit failed: " + str(ack.get("error", "<no error message>"))
            )
        pearl_trace_log("target", "early_control_model_init_ack", ack=ack)
        return ack

    generation_prompt = args.prompt
    prompt_tokens = []
    used_chat_template = False
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.target_model, trust_remote_code=True)
        if args.chat_template and getattr(tokenizer, "chat_template", None):
            messages = [{"role": "user", "content": args.prompt}]
            generation_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_tokens = [
                int(t)
                for t in tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                )
            ]
            used_chat_template = True
        else:
            prompt_tokens = [
                int(t) for t in tokenizer.encode(generation_prompt, add_special_tokens=True)
            ]
    except Exception as exc:
        pearl_trace_log(
            "target",
            "prompt_tokenize_failed",
            prompt_text=args.prompt,
            error=repr(exc),
        )
    run_metadata = dict(
        prompt_text=args.prompt,
        generation_prompt=generation_prompt,
        prompt_tokens=prompt_tokens,
        prompt_token_count=len(prompt_tokens),
        used_chat_template=bool(used_chat_template),
        target_model=args.target_model,
        draft_model=args.draft_model,
        transport=args.transport,
        nic=args.nic,
        max_tokens=int(args.max_tokens),
        max_draft_len=int(args.max_draft_len),
        baseline=bool(args.baseline),
    )
    pearl_trace_log("target", "setup_start", **run_metadata)

    early_init_result = {"ack": None, "error": None}
    early_init_thread = None
    if not args.baseline and args.early_draft_init:

        def _early_init_main():
            try:
                early_init_result["ack"] = _send_early_model_init()
            except Exception as exc:
                early_init_result["error"] = exc
                pearl_trace_log(
                    "target",
                    "early_control_model_init_failed",
                    error=repr(exc),
                )

        early_init_thread = threading.Thread(
            target=_early_init_main,
            daemon=True,
            name="pearl-early-draft-init",
        )
        early_init_thread.start()

    common_kv = KvCacheConfig(
        enable_block_reuse=True,
        free_gpu_memory_fraction=args.kv_cache_free_fraction,
    )

    if args.baseline:
        llm = LLM(
            model=args.target_model,
            kv_cache_config=common_kv,
            tensor_parallel_size=args.tp_size,
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
        if args.transport in ("tcp", "ibverbs", "doca", "shm", "cudaipc"):
            spec_kwargs["draft_offload_v2_model_path"] = args.draft_model
        if args.transport in ("ibverbs", "doca"):
            spec_kwargs["draft_offload_nic_name"] = args.nic
        if args.transport == "shm":
            spec_kwargs["draft_offload_v2_shm_name"] = args.shm_name
        if args.transport == "cudaipc":
            spec_kwargs["draft_offload_v2_cudaipc_name"] = args.cudaipc_name
        spec = PEARLDecodingConfig(**spec_kwargs)
        llm_kwargs = dict(
            model=args.target_model,
            speculative_config=spec,
            kv_cache_config=common_kv,
            tensor_parallel_size=args.tp_size,
        )
        if not args.cuda_graph:
            llm_kwargs["cuda_graph_config"] = None
        if args.attn_backend:
            llm_kwargs["attn_backend"] = args.attn_backend
        llm = LLM(**llm_kwargs)

    if early_init_thread is not None:
        early_init_thread.join()
        if early_init_result["error"] is not None:
            raise RuntimeError("early draft model init failed") from early_init_result["error"]

    generation_input = {"prompt_token_ids": prompt_tokens} if prompt_tokens else generation_prompt
    if int(args.warmup_tokens) > 0:
        warmup_sampling = SamplingParams(
            max_tokens=int(args.warmup_tokens),
            temperature=0.0,
            top_p=1.0,
            ignore_eos=True,
        )
        pearl_trace_log(
            "target",
            "warmup_start",
            warmup_tokens=int(args.warmup_tokens),
            **run_metadata,
        )
        warmup_start = time.perf_counter()
        warmup_outputs = llm.generate([generation_input], sampling_params=warmup_sampling)
        warmup_elapsed = max(time.perf_counter() - warmup_start, 1e-9)
        warmup_out = warmup_outputs[0].outputs[0]
        warmup_generated_tokens = len(warmup_out.token_ids)
        pearl_trace_log(
            "target",
            "warmup_finish",
            generated_tokens=int(warmup_generated_tokens),
            elapsed_sec=float(warmup_elapsed),
            tokens_per_sec=float(warmup_generated_tokens / warmup_elapsed),
        )
        print("=== warmup ===")
        print(f"warmup_tokens: {int(args.warmup_tokens)}")
        print(f"generated_tokens: {warmup_generated_tokens}")
        print(f"elapsed_sec: {warmup_elapsed:.3f}")
        print(f"tokens_per_sec: {warmup_generated_tokens / warmup_elapsed:.2f}")

    sampling = SamplingParams(
        max_tokens=int(args.max_tokens),
        temperature=0.0,
        top_p=1.0,
    )
    pearl_trace_log("target", "run_start", **run_metadata)
    if args.nsys_profile:
        import torch

        torch.cuda.profiler.start()
    start_time = time.perf_counter()
    try:
        outputs = llm.generate([generation_input], sampling_params=sampling)
    finally:
        if args.nsys_profile:
            torch.cuda.profiler.stop()
    elapsed = max(time.perf_counter() - start_time, 1e-9)
    out = outputs[0].outputs[0]
    generated_tokens = len(out.token_ids)
    tokens_per_sec = generated_tokens / elapsed
    pearl_trace_log(
        "target",
        "run_finish",
        generated_token_ids=[int(t) for t in out.token_ids],
        decoded_text=out.text,
        generated_tokens=int(generated_tokens),
        elapsed_sec=float(elapsed),
        tokens_per_sec=float(tokens_per_sec),
    )
    print("=== performance ===")
    print(f"generated_tokens: {generated_tokens}")
    print(f"elapsed_sec: {elapsed:.3f}")
    print(f"tokens_per_sec: {tokens_per_sec:.2f}")
    print("=== generated token_ids ===")
    print(list(out.token_ids))
    print("=== decoded text ===")
    print(out.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
