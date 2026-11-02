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
"""Standalone CPU-only renderer: ``trtllm-render`` / ``python -m tensorrt_llm.serve.render``.

Serves ``POST /v1/chat/completions/render`` and ``POST /v1/completions/render``
from a process that loads only the tokenizer, configuration and processor files
of a checkpoint. No model weights are read, no GPU is used and no executor is
built, so it runs on a CPU-only host.

Start it with the same model, tokenizer, chat template and parser options as the
workers it fronts; the rendering fingerprint (``GET /server_info``) lets a caller
check that they agree.
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="trtllm-render",
        description="CPU-only renderer: chat/completions request in, prompt token ids out.",
    )
    # Options that change the rendered prompt use the names `trtllm-serve` uses.
    parser.add_argument("--model", required=True, help="Checkpoint directory or Hugging Face id.")
    parser.add_argument("--tokenizer", default=None, help="Tokenizer path (default: --model).")
    parser.add_argument("--tokenizer_mode", choices=["auto", "slow"], default="auto")
    parser.add_argument("--custom_tokenizer", default=None)
    parser.add_argument("--trust_remote_code", action="store_true", default=False)
    parser.add_argument("--chat_template", default=None, help="Chat template file or literal.")
    parser.add_argument("--checkpoint_format", default="HF")
    parser.add_argument("--enable_tokenization_cache", action="store_true", default=False)
    parser.add_argument("--tool_parser", default=None)
    parser.add_argument("--reasoning_parser", default=None)
    parser.add_argument(
        "--allow_request_chat_template",
        action="store_true",
        default=False,
        help="Allow a per-request chat template, as on the workers.",
    )
    parser.add_argument("--served_model_name", default=None)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log_level", default="info")
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)

    # Deferred so `--help` and argument errors do not pay for the imports.
    import uvicorn

    from ._http import build_render_app
    from .resources import RenderResources

    try:
        resources = RenderResources.load(
            args.model,
            tokenizer=args.tokenizer,
            trust_remote_code=args.trust_remote_code,
            tokenizer_mode=args.tokenizer_mode,
            custom_tokenizer=args.custom_tokenizer,
            chat_template=args.chat_template,
            checkpoint_format=args.checkpoint_format,
            enable_tokenization_cache=args.enable_tokenization_cache,
            tool_parser=args.tool_parser,
            reasoning_parser=args.reasoning_parser,
            allow_request_chat_template=args.allow_request_chat_template,
        )
    except Exception as error:
        print(f"trtllm-render: could not load {args.model!r}: {error}", file=sys.stderr)
        raise SystemExit(2) from error

    app = build_render_app(resources, served_model_name=args.served_model_name or args.model)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)


if __name__ == "__main__":
    main()
