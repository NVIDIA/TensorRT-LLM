# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Startup error classification, stage marking and hint logging (no GPU)."""

import errno
import pickle
import subprocess
from unittest import mock

import click
import pydantic
import pytest
import torch
import yaml

from tensorrt_llm import _startup_errors as se
from tensorrt_llm._startup import _StartupTimer
from tensorrt_llm._torch.pyexecutor.warmup_timer import _WarmupTimer

pytestmark = pytest.mark.cpu_only


def _fake_exception(name: str, module: str, base: type = Exception, **attrs) -> BaseException:
    """An instance of a stand-in for a third-party exception class, matched by name/module."""
    cls = type(name, (base,), {"__module__": module})
    exc = cls(f"{name} message")
    for key, value in attrs.items():
        setattr(exc, key, value)
    return exc


def _raise_from_module(module: str, exc: BaseException) -> BaseException:
    """Raise ``exc`` from a frame whose module is ``module`` and return it with its traceback."""
    namespace = {"__name__": module}
    exec(compile("def fail(exc):\n    raise exc\n", f"<{module}>", "exec"), namespace)
    try:
        namespace["fail"](exc)
    except BaseException as raised:  # noqa: BLE001 - returning the raised exception
        return raised
    raise AssertionError("unreachable")


def _chained(outer: BaseException, inner: BaseException) -> BaseException:
    try:
        try:
            raise inner
        except BaseException as e:
            raise outer from e
    except BaseException as raised:  # noqa: BLE001 - returning the raised exception
        return raised


def _context(outer: BaseException, inner: BaseException) -> BaseException:
    try:
        try:
            raise inner
        except BaseException:
            raise outer
    except BaseException as raised:  # noqa: BLE001 - returning the raised exception
        return raised


class _StrictModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    size: int = 1

    @pydantic.field_validator("size")
    @classmethod
    def _check_size(cls, value: int) -> int:
        if value == 7:
            raise ValueError("size=7 is not supported with paged context")
        if value < 0:
            raise ValueError("size must be non-negative")
        return value


def _validation_error(**kwargs) -> pydantic.ValidationError:
    with pytest.raises(pydantic.ValidationError) as excinfo:
        _StrictModel(**kwargs)
    return excinfo.value


@pytest.mark.parametrize(
    "make_exc, category",
    [
        # config_input
        (lambda: _validation_error(unknown_key=1), "config_input"),
        (lambda: _validation_error(size=-1), "config_input"),
        (lambda: yaml.YAMLError("bad yaml"), "config_input"),
        (lambda: click.BadParameter("bad flag"), "config_input"),
        (lambda: ModuleNotFoundError("No module named 'grpc'"), "config_input"),
        (lambda: _fake_exception("GatedRepoError", "huggingface_hub.errors"), "config_input"),
        (
            lambda: _fake_exception("RepositoryNotFoundError", "huggingface_hub.errors"),
            "config_input",
        ),
        (
            lambda: _fake_exception("LocalEntryNotFoundError", "huggingface_hub.errors"),
            "config_input",
        ),
        (lambda: _fake_exception("ConnectionError", "requests.exceptions"), "config_input"),
        # resource
        (lambda: torch.OutOfMemoryError("CUDA out of memory"), "resource"),
        (lambda: MemoryError(), "resource"),
        (lambda: OSError(errno.EADDRINUSE, "Address already in use"), "resource"),
        (lambda: OSError(errno.ENOSPC, "No space left on device"), "resource"),
        (lambda: _fake_exception("AcceleratorError", "torch", error_code=2), "resource"),
        (lambda: _fake_exception("AcceleratorError", "torch", error_code=35), "resource"),
        (lambda: RuntimeError("[TensorRT-LLM][ERROR] CUDA_ERROR_OUT_OF_MEMORY"), "resource"),
        (lambda: _fake_exception("AcceleratorError", "torch", error_code=46), "resource"),
        (
            lambda: RuntimeError("CUDA error: CUDA-capable device(s) is/are busy or unavailable"),
            "resource",
        ),
        # unsupported
        (lambda: NotImplementedError("fp8 lm_head"), "unsupported"),
        (lambda: ValueError("MTP is not supported with beam search"), "unsupported"),
        (lambda: RuntimeError("Unknown model architecture: FooForCausalLM"), "unsupported"),
        (lambda: _fake_exception("AcceleratorError", "torch", error_code=209), "unsupported"),
        (lambda: _validation_error(size=7), "unsupported"),
        # model_load
        (lambda: _fake_exception("SafetensorError", "builtins"), "model_load"),
        (lambda: _fake_exception("EntryNotFoundError", "huggingface_hub.errors"), "model_load"),
        (
            lambda: _raise_from_module(
                "tensorrt_llm._torch.models.modeling_fake", KeyError("model.layers.0.weight")
            ),
            "model_load",
        ),
        (
            lambda: _raise_from_module("transformers.configuration_utils", ValueError("bad")),
            "model_load",
        ),
        # runtime_init
        (lambda: _fake_exception("Exception", "mpi4py.MPI"), "runtime_init"),
        (
            lambda: RuntimeError("NCCL error in: comm.cpp:42, unhandled system error"),
            "runtime_init",
        ),
        (lambda: subprocess.CalledProcessError(1, ["nvcc"]), "runtime_init"),
        (lambda: _fake_exception("CompilationError", "triton.compiler.errors"), "runtime_init"),
        (
            lambda: _raise_from_module(
                "tensorrt_llm._torch.pyexecutor.fake", AttributeError("'NoneType' has no 'x'")
            ),
            "runtime_init",
        ),
        (
            lambda: RuntimeError(
                "Executor worker died during initialization: MPI worker exited unexpectedly"
            ),
            "runtime_init",
        ),
        # other
        (lambda: RuntimeError("something odd"), "other"),
        (lambda: _raise_from_module("my_plugin", ZeroDivisionError("division by zero")), "other"),
    ],
)
def test_category(make_exc, category):
    info = se.classify_startup_error(make_exc())
    assert info.category == category
    assert info.category in se.STARTUP_ERROR_CATEGORIES
    if category != "other":
        assert info.hint


@pytest.mark.parametrize(
    "message, hint_key",
    [
        ("CUDA error: CUDA-capable device(s) is/are busy or unavailable", "gpu_busy"),
        (
            "Failed, NCCL error opUtils.cpp:179 'invalid usage (run with NCCL_DEBUG=WARN)'",
            "distributed",
        ),
    ],
)
def test_gpu_contention_hints(message, hint_key):
    assert se.classify_startup_error(RuntimeError(message)).hint == se._HINTS[hint_key]


def test_too_few_mpi_ranks_is_a_launch_input_error():
    from tensorrt_llm.executor import GenerationExecutor

    with pytest.raises(RuntimeError, match="Cannot instantiate an executor requiring 4 ranks") as e:
        GenerationExecutor.create(model_world_size=4, world_size=2)

    info = se.classify_startup_error(e.value)
    assert info.category == "config_input"
    assert "srun -n / mpirun -np" in info.hint


def test_python_operand_type_error_is_not_unsupported():
    exc = _raise_from_module(
        "tensorrt_llm._torch.pyexecutor.fake", TypeError("unsupported operand type(s) for +")
    )
    assert se.classify_startup_error(exc).category == "runtime_init"


def test_deepest_typed_cause_wins_and_is_summarized():
    gated = _fake_exception("GatedRepoError", "huggingface_hub.errors")
    exc = _chained(ValueError("Could not find model locally or on the Hub"), gated)

    info = se.classify_startup_error(exc)

    assert info.category == "config_input"
    assert "HF_TOKEN" in info.hint
    assert info.summary == "GatedRepoError: GatedRepoError message"


@pytest.mark.parametrize(
    "status, hint_key",
    [
        (401, "hf_denied"),
        (403, "hf_denied"),
        (404, "hf_not_found"),
        (503, "hf_network"),
        (None, "hf_network"),
    ],
)
def test_http_status_selects_hub_hint(status, hint_key):
    response = None if status is None else mock.Mock(status_code=status)
    exc = _fake_exception("HTTPStatusError", "httpx", response=response)
    info = se.classify_startup_error(_chained(ValueError("Could not find model"), exc))
    assert (info.category, info.hint) == ("config_input", se._HINTS[hint_key])


def test_summary_drops_text_repeated_by_the_hint():
    exc = se.tag_startup_error(
        RuntimeError("No GPU is visible.\nPossible causes: a, b."),
        "resource",
        hint="Possible causes: a, b.",
    )
    assert se.classify_startup_error(exc).summary == "RuntimeError: No GPU is visible."


def test_implicit_context_is_followed():
    # serve.py raises "Failed to bind socket" while handling the OSError.
    exc = _context(
        RuntimeError("Failed to bind socket"), OSError(errno.EADDRINUSE, "Address in use")
    )
    info = se.classify_startup_error(exc)
    assert info.category == "resource"
    assert "--port" in info.hint


def test_raise_site_tag_beats_typed_rules():
    exc = se.tag_startup_error(
        ValueError("The following arguments are not supported by the PyTorch backend: ['x']"),
        "config_input",
    )
    info = se.classify_startup_error(exc)
    assert info.category == "config_input"
    assert info.hint


def test_explicit_hint_is_kept():
    exc = se.tag_startup_error(RuntimeError("no gpu"), "resource", hint="Mount the GPUs.")
    assert se.classify_startup_error(exc).hint == "Mount the GPUs."


def test_summary_is_one_bounded_line():
    exc = ValueError("line one\n   line two\t" + "x" * 1000)
    summary = se.classify_startup_error(exc).summary
    assert "\n" not in summary
    assert summary.startswith("ValueError: line one line two x")
    assert summary.endswith("...")
    assert len(summary) <= len("ValueError: ") + 300


def test_classification_never_raises():
    class Unprintable(Exception):
        def __str__(self):
            raise RuntimeError("broken __str__")

    info = se.classify_startup_error(Unprintable())
    assert info.category == "other"
    assert info.summary == "Unprintable"


def test_rank_is_recorded():
    assert se.classify_startup_error(RuntimeError("x"), rank=3).rank == 3


def test_startup_timer_marks_innermost_phase():
    exc = RuntimeError("weights")
    with pytest.raises(RuntimeError):
        with _StartupTimer("executor_creation") as timer:
            with timer.phase("model"):
                with timer.phase("inner"):
                    raise exc
    assert se.classify_startup_error(exc).stage == "executor_creation/inner"


def test_startup_timer_marks_timer_name_outside_phases():
    exc = RuntimeError("config")
    with pytest.raises(RuntimeError):
        with _StartupTimer("executor_creation"):
            raise exc
    assert se.classify_startup_error(exc).stage == "executor_creation"


def test_warmup_phase_beats_enclosing_startup_phase():
    exc = torch.OutOfMemoryError("CUDA out of memory")
    with pytest.raises(torch.OutOfMemoryError):
        with _StartupTimer("executor_creation") as timer:
            with timer.phase("final_executor_creation"):
                with _WarmupTimer(rank=0) as warmup:
                    with warmup.phase("cuda_graph_capture"):
                        raise exc
    assert se.classify_startup_error(exc).stage == "warmup/cuda_graph_capture"


def test_metadata_survives_pickling():
    exc = se.tag_startup_error(ValueError("bad"), "unsupported", hint="Use X.")
    se.mark_startup_stage(exc, "executor_creation/model")

    restored = pickle.loads(pickle.dumps(exc))
    info = se.classify_startup_error(restored)

    assert (info.category, info.hint, info.stage) == (
        "unsupported",
        "Use X.",
        "executor_creation/model",
    )
    assert pickle.loads(pickle.dumps(info)) == info


def test_worker_error_wrapper_carries_worker_classification():
    # What the proxy holds: the unpickled worker exception (cause chain and
    # traceback gone) plus the classification the worker computed.
    worker_exc = _raise_from_module(
        "tensorrt_llm._torch.models.checkpoints.hf.weight_loader", KeyError("lm_head.weight")
    )
    se.mark_startup_stage(worker_exc, "executor_creation/model")
    info = se.classify_startup_error(worker_exc, rank=1)
    unpickled = pickle.loads(pickle.dumps(worker_exc))

    wrapper = _chained(
        se.wrap_worker_startup_error("Executor worker returned error", info), unpickled
    )

    assert str(wrapper) == (
        "Executor worker returned error on rank 1 during executor_creation/model: "
        "KeyError: 'lm_head.weight'"
    )
    assert wrapper.__cause__ is unpickled
    proxy_info = se.classify_startup_error(wrapper)
    assert proxy_info == info
    assert proxy_info.category == "model_load"


def test_wrapper_without_worker_info_keeps_plain_message():
    assert str(se.wrap_worker_startup_error("Executor worker returned error", None)) == (
        "Executor worker returned error"
    )


def test_log_startup_error_logs_once_per_chain():
    exc = _chained(RuntimeError("outer"), torch.OutOfMemoryError("CUDA out of memory"))
    with mock.patch.object(se.logger, "error") as log_error:
        info = se.log_startup_error(exc, "LLM initialization failed")
        assert se.log_startup_error(exc, "Command failed") is None

    assert info is not None and info.category == "resource"
    log_error.assert_called_once()
    line = log_error.call_args.args[0]
    assert line.startswith("LLM initialization failed [resource]: OutOfMemoryError: ")
    assert "Hint: GPU memory ran out." in line


def test_format_includes_rank_and_stage():
    info = se.StartupErrorInfo("resource", "OutOfMemoryError: oom", "Lower it.", "warmup/x", 2)
    assert se.format_startup_error(info, "Command failed") == (
        "Command failed [resource] on rank 2 during warmup/x: OutOfMemoryError: oom Hint: Lower it."
    )


def test_log_startup_error_never_raises():
    with mock.patch.object(se.logger, "error", side_effect=RuntimeError("logger broken")):
        assert se.log_startup_error(ValueError("x"), "Command failed") is None
