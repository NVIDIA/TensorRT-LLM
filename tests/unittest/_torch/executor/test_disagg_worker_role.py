# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""``disagg_worker_role`` and the generation-graph capture it skips.

A batch may replay a generation CUDA graph only when it carries no context
request (``ScheduledRequests.can_run_cuda_graph``), and every serving step on a
context worker carries one -- so those graphs can never be replayed there.

These tests pin both halves: the launcher deriving the role from the deployment
topology, and the engine acting on it. Nothing here touches a GPU.
"""

import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from tensorrt_llm._torch.pyexecutor.model_engine import PyTorchModelEngine
from tensorrt_llm.llmapi.disagg_utils import ServerRole
from tensorrt_llm.llmapi.llm_args import DisaggWorkerRole, TorchLlmArgs


class _FakeGraphRunner:
    """Only the attributes the gate reads before it returns."""

    def __init__(self):
        self.enabled = True
        self.is_warmup_only = False


def _engine(role):
    """A stub carrying only what the gate reads before it returns.

    A real engine needs a model and a device, so these cases bind the unbound
    method instead. They cover which branch the gate takes, not what capture
    then does: work added ahead of the gate needs its attributes added here.
    """
    return SimpleNamespace(
        _disagg_worker_role=role,
        cuda_graph_runner=_FakeGraphRunner(),
        _cuda_graph_batch_sizes=[1, 2, 4],
    )


class TestCaptureGate:
    """The engine skips generation capture only for a context worker."""

    def test_context_worker_skips_capture(self):
        engine = _engine(DisaggWorkerRole.CONTEXT)
        # _get_graphs_to_capture is the first real work after the gate; if the
        # gate lets the call through, this raises rather than silently passing.
        engine._get_graphs_to_capture = _unreachable
        PyTorchModelEngine._capture_generation_cuda_graphs(engine, None)

    @pytest.mark.parametrize(
        "role",
        [None, DisaggWorkerRole.GENERATION],
        ids=["aggregated", "generation_worker"],
    )
    def test_other_roles_still_capture(self, role):
        """A generation worker replays these graphs every step; aggregated
        serving may too. Neither may lose them."""
        engine = _engine(role)

        # Reaching _get_graphs_to_capture is the assertion: it is the first
        # real work past the gate. Raise a private marker from it so that an
        # unrelated failure before the gate cannot be mistaken for success.
        def reached(*args, **kwargs):
            raise _ReachedCapture

        engine._get_graphs_to_capture = reached
        with pytest.raises(_ReachedCapture):
            PyTorchModelEngine._capture_generation_cuda_graphs(engine, None)

    def test_disabled_graph_runner_short_circuits_first(self):
        """The pre-existing 'graphs are off' return must keep precedence, so
        the role never turns capture *on*."""
        engine = _engine(DisaggWorkerRole.GENERATION)
        engine.cuda_graph_runner.enabled = False
        engine._get_graphs_to_capture = _unreachable
        PyTorchModelEngine._capture_generation_cuda_graphs(engine, None)


class _ReachedCapture(Exception):
    """Raised from _get_graphs_to_capture to prove the gate let the call past."""


def _unreachable(*args, **kwargs):
    raise AssertionError("generation CUDA graph capture ran when it should have been skipped")


class TestArgsField:
    """The role is launcher-injected, not part of the public surface."""

    def test_defaults_to_none(self):
        assert TorchLlmArgs(model="dummy").disagg_worker_role is None

    def test_accepts_the_enum_and_its_string(self):
        for value, expected in (
            (DisaggWorkerRole.CONTEXT, DisaggWorkerRole.CONTEXT),
            ("context", DisaggWorkerRole.CONTEXT),
            (DisaggWorkerRole.GENERATION, DisaggWorkerRole.GENERATION),
            ("generation", DisaggWorkerRole.GENERATION),
        ):
            args = TorchLlmArgs(model="dummy", _disagg_worker_role=value)
            assert args.disagg_worker_role is expected

    def test_rejects_an_unknown_role(self):
        with pytest.raises(ValidationError, match="_disagg_worker_role"):
            TorchLlmArgs(model="dummy", _disagg_worker_role="bogus")

    def test_public_name_is_not_accepted(self):
        """Only the alias works -- what keeps this out of references/llm.yaml,
        for the same reason `mpi_session` is absent."""
        with pytest.raises(ValidationError, match="disagg_worker_role"):
            TorchLlmArgs(model="dummy", disagg_worker_role=DisaggWorkerRole.CONTEXT)

    def test_is_excluded_from_serialization(self):
        """A deployment-topology detail does not belong in a dumped config."""
        args = TorchLlmArgs(model="dummy", _disagg_worker_role=DisaggWorkerRole.CONTEXT)
        assert "disagg_worker_role" not in args.model_dump()
        # A dump keyed by the alias is a separate spelling, and the JSON form
        # is what actually gets persisted, so cover all three.
        assert "_disagg_worker_role" not in args.model_dump(by_alias=True)
        assert "disagg_worker_role" not in args.model_dump_json()


class TestLauncherMapping:
    """`launch_server` turns the deployment's topology into the field."""

    @staticmethod
    def _llm_args_seen(llm_args=None, **kwargs):
        """Run launch_server's role injection and report the resulting args.

        launch_server binds its listening socket before building the LLM, so
        this takes port 0 and a throwaway report path: a fixed port would race
        anything else on the CI host.
        """
        from tensorrt_llm.commands import serve

        captured = {}

        def fake_llm(**seen):
            captured.update(seen)
            raise _StopBeforeEngine

        llm_args = dict(llm_args or {"backend": "pytorch", "model": "dummy"})
        with tempfile.TemporaryDirectory() as tmp:
            with (
                patch.object(serve, "PyTorchLLM", fake_llm),
                patch.object(serve.logger, "warning") as warn,
                pytest.raises(_StopBeforeEngine),
            ):
                serve.launch_server(
                    host="127.0.0.1",
                    port=0,
                    llm_args=llm_args,
                    report_addr=str(Path(tmp) / "addr"),
                    **kwargs,
                )
        captured["__warnings__"] = [str(c.args[0]) for c in warn.call_args_list]
        return captured

    def test_context_server_role_becomes_the_worker_role(self):
        seen = self._llm_args_seen(server_role=ServerRole.CONTEXT)
        assert seen["_disagg_worker_role"] is DisaggWorkerRole.CONTEXT

    def test_generation_server_role_becomes_the_worker_role(self):
        seen = self._llm_args_seen(server_role=ServerRole.GENERATION)
        assert seen["_disagg_worker_role"] is DisaggWorkerRole.GENERATION

    def test_aggregated_serving_sets_nothing(self):
        seen = self._llm_args_seen(server_role=None)
        assert seen.get("_disagg_worker_role") is None

    @pytest.mark.parametrize(
        "server_role",
        [ServerRole.MM_ENCODER, ServerRole.VISUAL_GEN, ServerRole.EMBEDDING],
    )
    def test_non_disagg_roles_set_nothing(self, server_role):
        """These are separate model types, not halves of a split model. Mapping
        them onto a phase would skip capture for a worker that does generate."""
        seen = self._llm_args_seen(server_role=server_role)
        assert seen.get("_disagg_worker_role") is None

    def test_launch_topology_overrides_a_contradicting_config(self):
        """Trusting a YAML block that names the opposite role would skip the
        capture the worker needs, surfacing only as eager generation."""
        seen = self._llm_args_seen(
            llm_args={
                "backend": "pytorch",
                "model": "dummy",
                "_disagg_worker_role": DisaggWorkerRole.GENERATION,
            },
            server_role=ServerRole.CONTEXT,
        )
        assert seen["_disagg_worker_role"] is DisaggWorkerRole.CONTEXT
        # Overriding the user's value without saying so would be its own bug.
        assert any("Ignoring _disagg_worker_role" in w for w in seen["__warnings__"])

    def test_matching_config_is_left_alone(self):
        """Agreement is not a conflict; it must not warn or flip the value."""
        seen = self._llm_args_seen(
            llm_args={
                "backend": "pytorch",
                "model": "dummy",
                "_disagg_worker_role": DisaggWorkerRole.CONTEXT,
            },
            server_role=ServerRole.CONTEXT,
        )
        assert seen["_disagg_worker_role"] is DisaggWorkerRole.CONTEXT
        assert not any("Ignoring _disagg_worker_role" in w for w in seen["__warnings__"])

    def test_config_value_survives_aggregated_launch(self):
        """With no launch topology to contradict it, the user's value stands."""
        seen = self._llm_args_seen(
            llm_args={
                "backend": "pytorch",
                "model": "dummy",
                "_disagg_worker_role": DisaggWorkerRole.CONTEXT,
            },
            server_role=None,
        )
        assert seen["_disagg_worker_role"] is DisaggWorkerRole.CONTEXT

    def test_invalid_configured_role_is_rejected_with_guidance(self):
        """User input, not an internal error: name the value and what is valid."""
        from tensorrt_llm.commands.serve import _as_worker_role

        with pytest.raises(ValueError) as excinfo:
            _as_worker_role("bogus")
        message = str(excinfo.value)
        assert "bogus" in message
        assert "context" in message and "generation" in message

    def test_server_type_spelling_is_called_out(self):
        """'ctx' is what the same config file uses for `CtxGenServerConfig.type`,
        so it is the likely mistake; the error should say so."""
        from tensorrt_llm.commands.serve import _as_worker_role

        with pytest.raises(ValueError) as excinfo:
            _as_worker_role("ctx")
        assert "Did you mean 'context'" in str(excinfo.value)

    @pytest.mark.parametrize("value", [["ctx"], {"role": "ctx"}], ids=["list", "mapping"])
    def test_unhashable_values_are_rejected_cleanly(self, value):
        """YAML can produce a list or mapping here. The suggestion lookup is a
        dict membership test, so an unhashable value must not reach it and
        raise TypeError out of the handler that exists to give a clear error."""
        from tensorrt_llm.commands.serve import _as_worker_role

        with pytest.raises(ValueError, match="Invalid _disagg_worker_role"):
            _as_worker_role(value)

    def test_valid_roles_round_trip(self):
        from tensorrt_llm.commands.serve import _as_worker_role

        for role in DisaggWorkerRole:
            assert _as_worker_role(role.value) is role
            assert _as_worker_role(role) is role

    def test_explicit_parameter_wins_over_server_role(self):
        """The MPI launcher passes the role directly and leaves server_role
        unset, so that it does not also move the HTTP routes."""
        seen = self._llm_args_seen(
            disagg_worker_role=DisaggWorkerRole.CONTEXT,
            server_role=None,
        )
        assert seen["_disagg_worker_role"] is DisaggWorkerRole.CONTEXT


class TestMpiLauncherPropagation:
    """`_launch_disaggregated_server` derives the role from `server_cfg.type`.

    This is the path a real disaggregated deployment takes, and it is the one
    place the telemetry env var and the engine role are derived together, so a
    regression here would let the two disagree.
    """

    @pytest.mark.parametrize(
        "server_type, expected",
        [("ctx", DisaggWorkerRole.CONTEXT), ("gen", DisaggWorkerRole.GENERATION)],
    )
    def test_role_reaches_both_consumers(self, server_type, expected, monkeypatch):
        from tensorrt_llm.commands import serve

        monkeypatch.setenv(serve.DisaggLauncherEnvs.TLLM_DISAGG_INSTANCE_IDX, "0")
        # _launch_disaggregated_server assigns this one directly, so it has to
        # be registered with monkeypatch or it outlives the test and the next
        # reader sees whichever case ran last. setenv (not delenv) is what
        # records a restore entry when the variable is not already present.
        monkeypatch.setenv(serve.DisaggLauncherEnvs.TLLM_DISAGG_ROLE, "")
        server_cfg = SimpleNamespace(type=server_type, hostname="127.0.0.1", port=8000)
        config = SimpleNamespace(server_configs=[server_cfg], allow_request_chat_template=False)

        seen = {}
        with (
            patch.object(serve, "parse_disagg_config_file", lambda _: config),
            patch.object(serve, "launch_server", lambda **kw: seen.update(kw)),
        ):
            serve._launch_disaggregated_server("unused.yaml", {})

        assert seen["disagg_worker_role"] is expected
        # Telemetry reads the same value, so the two cannot drift.
        assert os.environ[serve.DisaggLauncherEnvs.TLLM_DISAGG_ROLE] == expected.value

    def test_unknown_server_type_sets_no_role(self, monkeypatch):
        """Roles other than ctx/gen leave both consumers untouched, as before."""
        from tensorrt_llm.commands import serve

        monkeypatch.setenv(serve.DisaggLauncherEnvs.TLLM_DISAGG_INSTANCE_IDX, "0")
        # _launch_disaggregated_server assigns this one directly, so it has to
        # be registered with monkeypatch or it outlives the test and the next
        # reader sees whichever case ran last. setenv (not delenv) is what
        # records a restore entry when the variable is not already present.
        monkeypatch.setenv(serve.DisaggLauncherEnvs.TLLM_DISAGG_ROLE, "")
        server_cfg = SimpleNamespace(type="mm_encoder", hostname="127.0.0.1", port=8000)
        config = SimpleNamespace(server_configs=[server_cfg], allow_request_chat_template=False)

        seen = {}
        with (
            patch.object(serve, "parse_disagg_config_file", lambda _: config),
            patch.object(serve, "launch_server", lambda **kw: seen.update(kw)),
        ):
            serve._launch_disaggregated_server("unused.yaml", {})

        assert seen["disagg_worker_role"] is None
        assert os.environ[serve.DisaggLauncherEnvs.TLLM_DISAGG_ROLE] == ""


class _StopBeforeEngine(Exception):
    """Unwinds launch_server once the LLM args are known."""
