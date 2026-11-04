# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serving objects with placeholder dependencies, for tests that call a handler off the app.

The composition root builds a serving object from its own attributes. A test
that drives one handler with stubs builds the object here instead, giving only
the dependencies that handler reads; every other constructor argument is a
placeholder that is never used.
"""

from types import SimpleNamespace
from typing import Any, Dict

from tensorrt_llm.serve.api.openai.chat.serving import ChatServing
from tensorrt_llm.serve.postprocess_handlers import select_postproc_strategy
from tensorrt_llm.serve.render import RenderResources

_CHAT_DEFAULTS: Dict[str, Any] = {
    "tool_parser": None,
    "tool_call_id_type": "random",
    "multimodal_server_config": None,
    "allow_request_chat_template": False,
    "use_harmony": False,
    "harmony_adapter": None,
}


def generate_serving_kwargs(**overrides: Any) -> Dict[str, Any]:
    """Every ``GenerateBaseServing`` constructor argument, placeholders unless overridden.

    ``postproc`` defaults to the strategy the composition root would choose
    from the generator's ``num_postprocess_workers`` (inline when there is no
    generator).
    """
    kwargs: Dict[str, Any] = {
        "model": "test-model",
        "generator": None,
        "tokenizer": None,
        "model_config": None,
        "collect_perf_metrics": False,
        "metrics_collector": None,
        "adjusted_steady_clock": None,
        "iteration_stats_wakeup_event": None,
        "iteration_stats_collector_task_getter": lambda: None,
        "internal_disagg_auth_key": None,
        "server_role": None,
        "per_request_spec_decode_stats": False,
        "spec_decode_num_spec_tokens": None,
        "input_proc_executor": None,
        "postproc": None,
    }
    kwargs.update(overrides)
    if kwargs["postproc"] is None:
        args = getattr(kwargs["generator"], "args", None)
        workers = getattr(args, "num_postprocess_workers", 0)
        kwargs["postproc"] = select_postproc_strategy(postproc_worker_enabled=workers > 0)
    return kwargs


def make_chat_serving(**overrides: Any) -> ChatServing:
    """A ``ChatServing`` built from stubs, with render resources built from the same stubs.

    Chat-specific arguments (``tool_parser``, ``tool_call_id_type``,
    ``multimodal_server_config``, ``allow_request_chat_template``,
    ``use_harmony``, ``harmony_adapter``) default to a plain text server; the
    rest are ``generate_serving_kwargs()`` placeholders. Unless
    ``render_resources`` is given, they are built the way the composition root
    builds the server's: from a stand-in that carries the same stubs, so the
    renderer sees the model config, tokenizer and template the handler sees.
    """
    chat = {name: overrides.pop(name, default) for name, default in _CHAT_DEFAULTS.items()}
    render_resources = overrides.pop("render_resources", None)
    chat_template = overrides.pop("chat_template", None)
    base = generate_serving_kwargs(**overrides)
    if render_resources is None:
        render_resources = RenderResources.from_server(
            SimpleNamespace(
                model_config=base["model_config"],
                generator=base["generator"],
                tokenizer=base["tokenizer"],
                processor=None,
                chat_template=chat_template,
                **chat,
            )
        )
    return ChatServing(render_resources=render_resources, **chat, **base)
