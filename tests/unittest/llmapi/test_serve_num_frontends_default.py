# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit coverage for trtllm-serve's multi-frontend default.

``--num_serve_frontends`` defaults to several HTTP frontend processes while
the LlmArgs field keeps default 1. The pieces that make the default reach the
executor -- surviving the CLI-vs-LlmArgs default filter, yielding to the YAML,
and falling back to a single frontend where more cannot run -- are covered
here; spawning the frontends needs a GPU and lives in the integration tests.
"""

from typing import Optional

import pytest

from tensorrt_llm.commands import serve as serve_cmd
from tensorrt_llm.commands.serve import (
    DEFAULT_NUM_SERVE_FRONTENDS,
    _init_multi_frontend_mode,
    _resolve_default_num_serve_frontends,
    get_llm_args,
)
from tensorrt_llm.executor.utils import MAX_NUM_FRONTENDS
from tensorrt_llm.llmapi.llm_args import BaseLlmArgs
from tensorrt_llm.llmapi.llm_utils import update_llm_args_with_extra_dict

pytestmark = pytest.mark.cpu_only


def _resolve(
    llm_args: dict,
    *,
    requested: bool = False,
    port: int = 8000,
    grpc: bool = False,
    report_addr: Optional[str] = None,
) -> int:
    _resolve_default_num_serve_frontends(
        llm_args, requested=requested, port=port, grpc=grpc, report_addr=report_addr
    )
    return llm_args.get("num_serve_frontends", 1)


def test_cli_default_is_multi_frontend() -> None:
    option = next(p for p in serve_cmd.serve.params if p.name == "num_serve_frontends")
    assert option.default == DEFAULT_NUM_SERVE_FRONTENDS
    assert 1 < DEFAULT_NUM_SERVE_FRONTENDS <= MAX_NUM_FRONTENDS


def test_llm_args_default_stays_single_frontend() -> None:
    # A bare LLM() has no HTTP frontends to fan out to; only trtllm-serve
    # spawns them, so the executor-side default must stay at one lane.
    assert BaseLlmArgs.model_fields["num_serve_frontends"].default == 1


def test_cli_default_survives_the_llm_args_default_filter() -> None:
    # get_llm_args drops CLI values equal to the LlmArgs default unless the
    # flag was typed. The serve default differs from the LlmArgs default, so
    # it must reach the LLM constructor without being typed on the CLI ...
    llm_args, _ = get_llm_args(
        model="m",
        backend="pytorch",
        gpus_per_node=1,
        num_serve_frontends=DEFAULT_NUM_SERVE_FRONTENDS,
    )
    assert llm_args["num_serve_frontends"] == DEFAULT_NUM_SERVE_FRONTENDS
    # ... while an untyped 1 collapses onto the LlmArgs default.
    llm_args, _ = get_llm_args(model="m", backend="pytorch", gpus_per_node=1, num_serve_frontends=1)
    assert "num_serve_frontends" not in llm_args


def test_yaml_value_overrides_the_cli_default() -> None:
    llm_args = update_llm_args_with_extra_dict(
        {"num_serve_frontends": DEFAULT_NUM_SERVE_FRONTENDS},
        {"num_serve_frontends": 1},
        explicit_cli_keys=set(),
    )
    assert llm_args["num_serve_frontends"] == 1


def test_explicit_cli_value_wins_over_yaml() -> None:
    llm_args = update_llm_args_with_extra_dict(
        {"num_serve_frontends": 4},
        {"num_serve_frontends": 1},
        explicit_cli_keys={"num_serve_frontends"},
    )
    assert llm_args["num_serve_frontends"] == 4


def test_plain_config_keeps_the_default() -> None:
    llm_args = {"num_serve_frontends": DEFAULT_NUM_SERVE_FRONTENDS}
    assert _resolve(llm_args) == DEFAULT_NUM_SERVE_FRONTENDS
    assert _init_multi_frontend_mode(llm_args, enabled=True).is_launcher


@pytest.mark.parametrize(
    "extra_llm_args,kwargs",
    [
        ({"orchestrator_type": "rpc"}, {}),
        ({"orchestrator_type": "ray"}, {}),
        ({"enable_resource_governor": True}, {}),
        ({}, {"grpc": True}),
        ({}, {"port": 0}),
        ({}, {"report_addr": "/tmp/bound.addr"}),
    ],
)
def test_default_falls_back_to_one_frontend(extra_llm_args: dict, kwargs: dict) -> None:
    llm_args = {"num_serve_frontends": DEFAULT_NUM_SERVE_FRONTENDS, **extra_llm_args}
    assert _resolve(llm_args, **kwargs) == 1
    # The fallback leaves a consistent single-frontend configuration behind:
    # the mode resolver no longer sees a launcher and does not raise.
    if "orchestrator_type" in extra_llm_args:
        assert not _init_multi_frontend_mode(llm_args, enabled=True).is_launcher


def test_explicit_request_is_not_silently_downgraded() -> None:
    llm_args = {"num_serve_frontends": DEFAULT_NUM_SERVE_FRONTENDS, "orchestrator_type": "rpc"}
    assert _resolve(llm_args, requested=True) == DEFAULT_NUM_SERVE_FRONTENDS
    with pytest.raises(ValueError, match="orchestrator_type"):
        _init_multi_frontend_mode(llm_args, enabled=True)


def test_single_frontend_is_left_alone() -> None:
    llm_args = {"num_serve_frontends": 1, "orchestrator_type": "rpc"}
    assert _resolve(llm_args) == 1
    assert _resolve({"orchestrator_type": "rpc"}) == 1


def test_fallback_logs_the_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    messages = []
    monkeypatch.setattr(serve_cmd.logger, "info", lambda msg, *a, **k: messages.append(msg))
    _resolve({"num_serve_frontends": DEFAULT_NUM_SERVE_FRONTENDS, "orchestrator_type": "rpc"})
    assert len(messages) == 1
    assert "orchestrator_type='rpc'" in messages[0]
    assert "--num_serve_frontends" in messages[0]
