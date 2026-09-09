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

import signal
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tensorrt_llm import visual_gen
from tensorrt_llm._torch.visual_gen import executor as executor_module
from tensorrt_llm.commands import serve

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def _disable_external_launch(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(executor_module, "_detect_external_launch", lambda: None)


def test_sigterm_during_visual_gen_startup_exits_with_signal_status(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_handler = signal.getsignal(signal.SIGTERM)

    def interrupt_startup(*args, **kwargs) -> None:
        del args, kwargs
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        handler(signal.SIGTERM, None)

    monkeypatch.setattr(visual_gen, "VisualGen", interrupt_startup)

    with pytest.raises(SystemExit) as exc_info:
        serve.launch_visual_gen_server("127.0.0.1", 0, "test-model")

    assert exc_info.value.code == 128 + signal.SIGTERM
    assert signal.getsignal(signal.SIGTERM) is original_handler


def test_sigterm_after_visual_gen_startup_shuts_down_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_handler = signal.getsignal(signal.SIGTERM)
    model = MagicMock()
    model.args.parallel_config = SimpleNamespace(
        n_workers=1,
        cfg_size=1,
        ulysses_size=1,
    )

    def interrupt_server_setup(*args, **kwargs) -> None:
        del args, kwargs
        handler = signal.getsignal(signal.SIGTERM)
        assert callable(handler)
        handler(signal.SIGTERM, None)

    monkeypatch.setattr(visual_gen, "VisualGen", lambda **_: model)
    monkeypatch.setattr(serve, "OpenAIServer", interrupt_server_setup)

    with pytest.raises(SystemExit) as exc_info:
        serve.launch_visual_gen_server("127.0.0.1", 0, "test-model")

    assert exc_info.value.code == 128 + signal.SIGTERM
    model.shutdown.assert_called_once_with()
    assert signal.getsignal(signal.SIGTERM) is original_handler


def test_missing_previous_sigterm_handler_restores_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signal_calls = []

    def set_signal_handler(signum, handler):
        signal_calls.append((signum, handler))
        return None

    fake_signal = SimpleNamespace(
        SIGTERM=signal.SIGTERM,
        SIG_DFL=signal.SIG_DFL,
        signal=set_signal_handler,
    )
    monkeypatch.setattr(serve, "signal", fake_signal)

    def fail_startup(*args, **kwargs) -> None:
        del args, kwargs
        raise RuntimeError("startup failed")

    monkeypatch.setattr(visual_gen, "VisualGen", fail_startup)

    with pytest.raises(RuntimeError, match="startup failed"):
        serve.launch_visual_gen_server("127.0.0.1", 0, "test-model")

    assert signal_calls == [
        (signal.SIGTERM, serve._terminate_visual_gen_startup),
        (signal.SIGTERM, signal.SIG_DFL),
    ]
