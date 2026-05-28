#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Product-style PEARL draft server entry point.

This wraps ``pearl_draft_server.py`` behind a stable service-oriented CLI:

    trtllm_pearl_draft_server.py
      --backend trtllm
      --transport ibverbs
      --nic mlx5_0
      --control-port 47331
      --data-port 0

By default the server lazy-loads the model requested by the target's
``TcpModelInit``.  Passing ``--model`` pins the server to one model path;
passing ``--strict-model-match`` additionally rejects mismatched target
requests instead of overriding them.

The target still connects through ``PEARLDecodingConfig`` /
``draft_offload_*``.  The draft process owns its runtime, networking, logs,
and GPU visibility independently from the target process.
"""

import argparse
import dataclasses
import os
import signal
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import draft_rdma_server as _drs  # noqa: E402
from pearl_draft_server import PEARLDraftServer  # noqa: E402


class ProductPEARLDraftServer(PEARLDraftServer):
    """PEARL draft server with a product-facing model/backend contract."""

    def _make_backend_from_model_init(self, msg):
        configured_model = str(getattr(self._args, "model", "") or "")
        if configured_model:
            if (
                bool(getattr(self._args, "strict_model_match", False))
                and str(msg.model_path) != configured_model
            ):
                raise RuntimeError(
                    "target requested draft model %r, but server was started with %r"
                    % (msg.model_path, configured_model)
                )
            msg = dataclasses.replace(msg, model_path=configured_model)

        backend = str(getattr(self._args, "backend", "trtllm"))

        if backend == "mock":
            return _drs._MockBackend(max_draft_len=int(msg.max_draft_len))
        if backend == "transformers":
            return _drs._TransformersBackend(
                model_path=msg.model_path,
                prompt=self._args.prompt,
                max_draft_len=int(msg.max_draft_len),
                device=self._args.device,
            )
        if backend == "trtllm":
            return _drs._TrtLlmBackend(
                model_path=msg.model_path,
                prompt=self._args.prompt,
                max_draft_len=int(msg.max_draft_len),
                device=self._args.device,
                stream_max_tokens=int(self._args.trtllm_stream_max_tokens),
            )
        raise RuntimeError("unknown backend: %s" % backend)


def _parse_args():
    ap = argparse.ArgumentParser(
        description="TensorRT-LLM PEARL draft server",
    )
    ap.add_argument(
        "--model",
        default="",
        help=(
            "Optional draft model path owned by this server. If omitted, "
            "the target's TcpModelInit.model_path is used for lazy-load."
        ),
    )
    ap.add_argument(
        "--backend",
        choices=["trtllm", "transformers", "mock"],
        default="trtllm",
        help=(
            "Draft runtime backend. trtllm uses the PyExecutor/manual path "
            "with resident KV state and rollback support."
        ),
    )
    ap.add_argument(
        "--transport",
        choices=["ibverbs", "tcp", "doca", "shm"],
        default="ibverbs",
        help="Data-plane transport expected from the target.",
    )
    ap.add_argument("--nic", default="mlx5_0", help="RDMA NIC for ibverbs/doca.")
    ap.add_argument(
        "--control-port",
        type=int,
        required=True,
        help="TCP control-plane port for model-init / prompt-init / cancel.",
    )
    ap.add_argument(
        "--data-port",
        "--port",
        dest="port",
        type=int,
        default=0,
        help="Data-plane port. 0 means allocate a free port and return it in TcpModelInitAck.",
    )
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--prompt", default=None)
    ap.add_argument(
        "--strict-model-match",
        action="store_true",
        help="Fail TcpModelInit if target requests a different draft model path.",
    )
    ap.add_argument(
        "--allow-proxy-worker",
        action="store_true",
        help=(
            "Do not force TLLM_WORKER_USE_SINGLE_PROCESS=1 for --backend trtllm. "
            "Mostly useful for debugging; direct PyExecutor handles may be unavailable."
        ),
    )
    ap.add_argument(
        "--trtllm-stream-max-tokens",
        type=int,
        default=int(os.environ.get("PEARL_TRTLLM_STREAM_MAX_TOKENS", "2048")),
        help="Max tokens for each resident TRT-LLM streaming decode request before restart.",
    )
    ap.add_argument(
        "--prefetch-wait-timeout-s",
        type=float,
        default=float(os.environ.get("PEARL_PREFETCH_WAIT_TIMEOUT_S", "0.05")),
        help="Seconds to wait for an in-flight PEARL prefetch.",
    )
    ap.add_argument("--backend-init-timeout-s", type=float, default=3600.0)
    ap.add_argument("--handshake-timeout-s", type=float, default=300.0)
    ap.add_argument("--max-num-requests", type=int, default=4096)
    ap.add_argument(
        "--trace-log",
        default="",
        help="Write draft-side PEARL communication trace as JSONL to this file.",
    )
    return ap.parse_args()


def main():
    args = _parse_args()

    if args.trace_log:
        os.environ["PEARL_DRAFT_TRACE_PATH"] = args.trace_log

    if args.backend == "trtllm" and not args.allow_proxy_worker:
        # Product TRT-LLM mode needs the local worker path so the server can
        # inspect/own PyExecutor handles rather than seeing only an IPC proxy.
        os.environ.setdefault("TLLM_WORKER_USE_SINGLE_PROCESS", "1")

    # ``pearl_draft_server`` currently learns transport/max_num_requests from
    # TcpModelInit.extra_kwargs_json sent by the target.  Keep these attributes
    # on args for compatibility with the parent class and future direct-start
    # modes.
    args.data_transport = args.transport
    args.max_num_requests = int(args.max_num_requests)

    server = ProductPEARLDraftServer(args)

    def _sig(*_):
        print("[trtllm-pearl-draft-server] received signal, shutting down...", flush=True)
        server.stop()

    signal.signal(signal.SIGINT, _sig)
    signal.signal(signal.SIGTERM, _sig)

    server.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
