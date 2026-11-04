# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``OpenAIServer`` borrows the helpers that moved to the serving bases; it keeps no copies.

A helper that moved into a serving base while handlers still on ``OpenAIServer``
call it is reachable on the class as the moved object itself. A copy would
drift from the moved one, so identity is what this file pins. Each move that
takes the last ``OpenAIServer`` caller of a helper removes its row here.
"""

from types import SimpleNamespace

import pytest

from tensorrt_llm.serve._base.serving import BaseServing, GenerateBaseServing
from tensorrt_llm.serve.openai_server import OpenAIServer

pytestmark = pytest.mark.cpu_only

# Helper name -> the base class it moved to.
BORROWED = {
    "create_error_response": BaseServing,
    "await_disconnected": GenerateBaseServing,
    "postproc_worker_enabled": GenerateBaseServing,
    "_extract_metrics": GenerateBaseServing,
    # Still called by the Responses handler on OpenAIServer (the second through the first).
    "_validate_internal_disagg_request": GenerateBaseServing,
    "_has_cache_transceiver_config": GenerateBaseServing,
}


@pytest.mark.parametrize("name", sorted(BORROWED))
def test_openai_server_borrows_the_moved_helper(name):
    base = BORROWED[name]
    assert getattr(OpenAIServer, name) is getattr(base, name), (
        f"OpenAIServer.{name} must be the helper moved to {base.__name__} "
        f"(staticmethod({base.__name__}.{name}) for a static method, {base.__name__}.{name} "
        "otherwise); never copy a moved helper back: a copy drifts from the moved one"
    )


def test_static_helpers_survived_aliasing():
    # A bare `Base.helper` alias would bind the server as the first argument.
    assert OpenAIServer.create_error_response("x").status_code == 400


# Moved with their last OpenAIServer caller: the chat handlers to ChatServing,
# the helpers below them to GenerateBaseServing, where the serving classes
# inherit them. No alias stays, so nothing on the class can drift from them.
MOVED_OUT = (
    "openai_chat",
    "create_chat_completion",
    "_create_chat_response",
    "chat_harmony",
    "_get_scheduling_params",
    "_logit_bias_vocab_size",
    "_apply_spec_decode_stats_opt_in",
    "_attach_multimodal_data",
)


@pytest.mark.parametrize("name", MOVED_OUT)
def test_the_chat_path_left_openai_server(name):
    assert name not in vars(OpenAIServer), (
        f"OpenAIServer.{name} left with the chat handlers (ChatServing, GenerateBaseServing); "
        "do not add it or an alias of it back: the serving classes inherit the moved helper"
    )


def test_postproc_worker_enabled_is_still_a_property():
    # The composition root evaluates it while building the server (_init_llm),
    # so the alias must be the property object, not a plain function.
    assert isinstance(OpenAIServer.__dict__["postproc_worker_enabled"], property)
    server = SimpleNamespace(
        _is_visual_gen=False,
        generator=SimpleNamespace(args=SimpleNamespace(num_postprocess_workers=2)),
    )
    assert OpenAIServer.__dict__["postproc_worker_enabled"].fget(server) is True
