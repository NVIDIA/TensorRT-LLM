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
"""An ``OpenAIServer`` for in-process route tests: real constructor, stub engine."""

from types import SimpleNamespace
from typing import Any, Callable, Optional
from unittest import mock

from tensorrt_llm.inputs.media_io import BaseMediaIO
from tensorrt_llm.serve.openai_server import OpenAIServer


def make_stub_openai_server(
    tmp_path,
    model: str,
    model_config: Any,
    generate_async: Callable,
    tokenizer: Any = None,
    *,
    chat_template: Optional[str] = None,
    allow_request_chat_template: bool = False,
) -> OpenAIServer:
    """Build the server through ``OpenAIServer.__init__`` on an engine stub.

    The generator stands in for ``LLM``: it carries the ``args`` the server
    reads, a tokenizer (a vocab-size stub unless one is given), and
    ``generate_async`` recording what reaches the engine. ``tmp_path`` is the
    empty model directory, so the constructor loads no HF processor; the model
    config loader answers the given stub, so the constructor sees it while it
    builds the serving objects and the render resources (the chat handlers run
    on ``server._chat_serving``). ``chat_template`` and
    ``allow_request_chat_template`` are the constructor's own arguments.
    """
    if tokenizer is None:
        tokenizer = SimpleNamespace(tokenizer=SimpleNamespace(vocab_size=model_config.vocab_size))
    generator = SimpleNamespace(
        args=SimpleNamespace(
            backend="pytorch",
            reasoning_parser=None,
            gather_generation_logits=False,
            guided_decoding_backend=None,
            num_postprocess_workers=0,
            cache_transceiver_config=None,
            trust_remote_code=False,
            checkpoint_format=None,
            return_perf_metrics=False,
            perf_metrics_output_dir=None,
            speculative_config=None,
            post_processor_hook=None,
            sleep_config=None,
            num_serve_frontends=1,
            enable_iter_perf_stats=False,
        ),
        tokenizer=tokenizer,
        input_processor=None,
        llm_id="stub",
        hf_model_dir=str(tmp_path),
        resource_governor_queue=None,
        generate_async=generate_async,
    )
    previous_media_executor = BaseMediaIO._executor
    with mock.patch(
        "tensorrt_llm._torch.pyexecutor.config_utils.load_pretrained_config",
        return_value=model_config,
    ):
        server = OpenAIServer(
            generator=generator,
            model=model,
            tool_parser=None,
            server_role=None,
            metadata_server_cfg=None,
            chat_template=chat_template,
            allow_request_chat_template=allow_request_chat_template,
        )
    assert server.model_config is model_config
    # The constructor publishes a media-decode pool process-wide. In-process
    # tests decode on the event loop's default executor instead, so put the
    # previous pool back and retire the server's unused pools before they
    # start any worker thread.
    BaseMediaIO.set_executor(previous_media_executor)
    server._media_load_executor.shutdown(wait=False)
    server._input_proc_executor.shutdown(wait=False)
    return server
