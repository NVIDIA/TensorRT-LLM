# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Error categories, startup error classification, stage marking and hint logging (no GPU)."""

import errno
import pickle
import subprocess
from unittest import mock

import click
import pydantic
import pytest
import torch
import yaml

from tensorrt_llm import _errors as errors
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
        (lambda: ValueError("FP4 MLA does not support context parallelism."), "unsupported"),
        (
            lambda: RuntimeError("DeepSeek-V4 requires Hopper or newer GPUs, got SM89"),
            "unsupported",
        ),
        (lambda: RuntimeError("TriAttention requires an SM100-family device."), "unsupported"),
        (lambda: ValueError("This kernel is supported on SM90 only."), "unsupported"),
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
            lambda: RuntimeError(
                "Executor worker died during initialization: MPI worker exited unexpectedly"
            ),
            "runtime_init",
        ),
        # other
        (lambda: RuntimeError("something odd"), "other"),
        (
            lambda: _raise_from_module(
                "tensorrt_llm._torch.pyexecutor.fake", AttributeError("'NoneType' has no 'x'")
            ),
            "other",
        ),
        (lambda: _fake_exception("AcceleratorError", "torch", error_code=700), "other"),
        (lambda: _raise_from_module("my_plugin", ZeroDivisionError("division by zero")), "other"),
    ],
)
def test_category(make_exc, category):
    info = se.classify_startup_error(make_exc())
    assert info.category == category
    assert info.category in errors.ERROR_CATEGORIES
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
    assert se.classify_startup_error(RuntimeError(message)).hint == errors.ERROR_HINTS[hint_key]


@pytest.mark.parametrize(
    "message",
    [
        "[TensorRT-LLM][ERROR] NVRTC compilation failed",
        "[TensorRT-LLM][ERROR] Assertion failed: Cannot parse NVCC version",
    ],
)
def test_cpp_kernel_compilation_failure_is_jit(message):
    info = se.classify_startup_error(RuntimeError(message))
    assert (info.category, info.reason) == ("runtime_init", "jit")


def test_only_supports_phrase_is_unsupported():
    # DeepGEMM's C++ SM gate, as raised before the custom op re-raises it as UnsupportedError.
    exc = RuntimeError(
        "DeepGEMM only supports Hopper (SM90) architectures, but current device compute "
        "capability is 100."
    )
    info = se.classify_startup_error(exc)
    assert (info.category, info.reason) == ("unsupported", "unsupported")


def test_gpu_architecture_missing_from_build_is_unsupported():
    # The worker's BuildInfo.check_cuda_architecture_supported for a GPU the build left out.
    exc = RuntimeError(
        "GPU 0 (NVIDIA RTX PRO 6000 Blackwell Server Edition, compute capability 12.0) is not "
        'included in this TensorRT-LLM build, which was built for CUDA architectures "100". '
        'Rebuild TensorRT-LLM with 120 included, e.g. with --cuda_architectures "100;120".'
    )
    info = se.classify_startup_error(exc)
    assert (info.category, info.reason) == ("unsupported", "unsupported")
    assert info.hint == errors.ERROR_HINTS["unsupported"]


def test_too_few_mpi_ranks_is_a_launch_input_error():
    from tensorrt_llm.executor import GenerationExecutor

    with pytest.raises(ValueError, match="Cannot instantiate an executor requiring 4 ranks") as e:
        GenerationExecutor.create(model_world_size=4, world_size=2)

    assert type(e.value) is errors.ConfigInputError
    info = se.classify_startup_error(e.value)
    assert info.category == "config_input"
    assert info.summary.startswith("ConfigInputError: Cannot instantiate an executor")
    assert "srun -n / mpirun -np" in info.summary


@pytest.mark.parametrize(
    "module, category",
    [
        ("tensorrt_llm.commands.serve", "config_input"),
        ("tensorrt_llm.commands.eval", "config_input"),
        ("tensorrt_llm.bench.benchmark.utils.general", "config_input"),
        ("tensorrt_llm.llmapi.disagg_utils", "config_input"),
        # bench tuning opens the checkpoint's weight shards.
        ("tensorrt_llm.bench.tuning.dataclasses", "other"),
        ("tensorrt_llm.llmapi.llm_utils", "other"),
        ("tensorrt_llm._torch.models.checkpoints.hf.weight_loader", "model_load"),
    ],
)
def test_missing_file_is_config_input_only_where_cli_code_opens_it(module, category):
    missing = FileNotFoundError(errno.ENOENT, "No such file or directory", "/x/config.yaml")
    info = se.classify_startup_error(_raise_from_module(module, missing))
    assert info.category == category
    if category == "config_input":
        assert info.hint == errors.ERROR_HINTS["config"]


def test_unknown_quant_algo_is_unsupported_but_an_invalid_config_value_is_config_input():
    from tensorrt_llm.models.modeling_utils import QuantConfig
    from tensorrt_llm.quantization.mode import QuantAlgo

    # E.g. the quant_algo of a checkpoint quantized with a newer format.
    with pytest.raises(ValueError, match="^'FP8_NEW' is not a valid QuantAlgo$") as e:
        QuantAlgo("FP8_NEW")
    assert type(e.value) is errors.UnsupportedError
    assert se.classify_startup_error(e.value).category == "unsupported"
    assert "FP8_NEW" not in QuantAlgo
    assert QuantAlgo("FP8") is QuantAlgo.FP8
    # pydantic reports any error of _missing_ as its own enum validation error.
    with pytest.raises(pydantic.ValidationError) as e:
        QuantConfig(quant_algo="FP8_NEW")
    assert se.classify_startup_error(e.value).category == "config_input"


def test_python_operand_type_error_is_not_unsupported():
    exc = _raise_from_module(
        "tensorrt_llm._torch.pyexecutor.fake", TypeError("unsupported operand type(s) for +")
    )
    info = se.classify_startup_error(exc)
    assert (info.category, info.reason) == ("other", "internal")


@pytest.mark.parametrize(
    "make_exc",
    [
        lambda: _raise_from_module("tensorrt_llm._torch.models.fake", AssertionError("tp")),
        lambda: _fake_exception("AcceleratorError", "torch", error_code=700),
    ],
)
def test_internal_looking_errors_are_other_with_the_internal_hint(make_exc):
    info = se.classify_startup_error(make_exc())
    assert (info.category, info.reason) == ("other", "internal")
    assert info.hint == errors.ERROR_HINTS["internal"]


@pytest.mark.parametrize(
    "make_exc, reason",
    [
        (lambda: torch.OutOfMemoryError("CUDA out of memory"), "gpu_oom"),
        (lambda: OSError(errno.EADDRINUSE, "Address already in use"), "port"),
        (lambda: RuntimeError("NCCL error in: comm.cpp:42, unhandled system error"), "distributed"),
        (lambda: errors.ConfigInputError("bad"), "config"),
        (lambda: errors.ResourceError("no gpu"), ""),
        (lambda: errors.RuntimeInitError("x", reason="worker_died"), "worker_died"),
        (lambda: RuntimeError("[unsupported] x"), "unsupported"),
        (lambda: RuntimeError("something odd"), ""),
    ],
)
def test_reason_is_the_standard_hint_key(make_exc, reason):
    info = se.classify_startup_error(make_exc())
    assert info.reason == reason
    assert info.hint == errors.ERROR_HINTS.get(reason, "")


def test_runtime_init_default_hint_is_not_the_internal_hint():
    for exc in (
        errors.RuntimeInitError("x"),
        RuntimeError("[runtime_init] x"),
    ):
        info = se.classify_startup_error(exc)
        assert (info.category, info.reason) == ("runtime_init", "runtime_init")
        assert info.hint == errors.ERROR_HINTS["runtime_init"] != errors.ERROR_HINTS["internal"]


def test_config_hint_uses_python_api_terms():
    assert "--" not in errors.ERROR_HINTS["config"]
    assert "LLM arguments" in errors.ERROR_HINTS["config"]


def test_deepest_typed_cause_wins_and_is_summarized():
    gated = _fake_exception("GatedRepoError", "huggingface_hub.errors")
    exc = _chained(ValueError("Could not find model locally or on the Hub"), gated)

    info = se.classify_startup_error(exc)

    assert info.category == "config_input"
    assert "HF_TOKEN" in info.hint
    assert info.summary == "GatedRepoError: GatedRepoError message"


@pytest.mark.parametrize(
    "name, module",
    [
        ("GatedRepoError", "huggingface_hub.errors"),
        ("RepositoryNotFoundError", "huggingface_hub.errors"),
        ("HfHubHTTPError", "huggingface_hub.errors"),
        ("OfflineModeIsEnabled", "huggingface_hub.errors"),
        ("HTTPStatusError", "httpx"),
    ],
)
def test_hub_errors_share_one_hint(name, module):
    exc = _chained(ValueError("Could not find model"), _fake_exception(name, module))
    info = se.classify_startup_error(exc)
    assert (info.category, info.hint) == ("config_input", errors.ERROR_HINTS["hf"])


def test_missing_file_in_found_repo_is_model_load_but_offline_miss_is_not():
    entry_cls = type("EntryNotFoundError", (Exception,), {"__module__": "huggingface_hub.errors"})
    local = _fake_exception("LocalEntryNotFoundError", "huggingface_hub.errors", base=entry_cls)
    assert se.classify_startup_error(entry_cls("x")).category == "model_load"
    assert se.classify_startup_error(local).category == "config_input"


def test_summary_drops_text_repeated_by_the_hint():
    exc = errors.ResourceError(
        "No GPU is visible.\nPossible causes: a, b.", hint="Possible causes: a, b."
    )
    assert se.classify_startup_error(exc).summary == "ResourceError: No GPU is visible."


def test_implicit_context_is_followed():
    # serve.py raises "Failed to bind socket" while handling the OSError.
    exc = _context(
        RuntimeError("Failed to bind socket"), OSError(errno.EADDRINUSE, "Address in use")
    )
    info = se.classify_startup_error(exc)
    assert info.category == "resource"
    assert "--port" in info.hint


@pytest.mark.parametrize(
    "cls, base",
    [
        (errors.ConfigInputError, ValueError),
        (errors.UnsupportedError, ValueError),
        (errors.ResourceError, RuntimeError),
        (errors.ModelLoadError, RuntimeError),
        (errors.RuntimeInitError, RuntimeError),
    ],
)
def test_typed_error_declares_category_and_keeps_builtin_base(cls, base):
    exc = cls("went wrong")
    assert isinstance(exc, base)
    # The only exception base is the builtin; the shared mixin is not an exception type.
    assert [b for b in cls.__mro__ if issubclass(b, BaseException)][1] is base
    assert not issubclass(errors._CategorizedError, BaseException)
    assert str(exc) == "went wrong"
    info = se.classify_startup_error(exc)
    assert info.category == cls.category
    assert info.summary == f"{cls.__name__}: went wrong"


def test_any_exception_class_declares_a_category_with_an_attribute():
    class OptedIn(KeyError):
        category = "resource"

    class InvalidCategory(RuntimeError):
        category = "gpu"

    info = se.classify_startup_error(OptedIn("x"))
    assert (info.category, info.reason, info.hint) == ("resource", "", "")
    assert se.classify_startup_error(InvalidCategory("x")).category == "other"


def test_declared_category_beats_message_phrases():
    exc = errors.ConfigInputError(
        "The following arguments are not supported by the PyTorch backend: ['x']"
    )
    info = se.classify_startup_error(exc)
    assert info.category == "config_input"
    assert info.hint == errors.ERROR_HINTS["config"]


def test_declared_category_beats_typed_rules():
    # A raise site that wraps an error decides over the wrapped error's type.
    inner = _chained(errors.ModelLoadError("shard unreadable"), OSError(errno.ENOSPC, "No space"))
    exc = _chained(RuntimeError("outer"), inner)
    assert se.classify_startup_error(exc).category == "model_load"


def test_explicit_hint_is_kept():
    exc = errors.ResourceError("no gpu", hint="Mount the GPUs.")
    assert se.classify_startup_error(exc).hint == "Mount the GPUs."


def test_marker_declares_category_of_builtin_error():
    exc = RuntimeError("[model_load] Checkpoint has no quant_algo.")
    info = se.classify_startup_error(exc)
    assert info.category == "model_load"
    assert info.hint == errors.ERROR_HINTS["model_load"]
    assert info.summary == "RuntimeError: Checkpoint has no quant_algo."


def test_marker_beats_message_phrases():
    exc = RuntimeError("[config_input] ValueError: Option foo is not supported.")
    assert se.classify_startup_error(exc).category == "config_input"


def test_marker_keeps_the_specific_hint_of_a_matching_phrase():
    # What the proxy sees from trtllm-llmapi-launch: only the forwarded text.
    exc = RuntimeError(
        "Remote MPI worker died: RuntimeError: Executor worker returned error on rank 1: "
        "[resource] RuntimeError: CUDA Runtime API error: cudaErrorMemoryAllocation"
    )
    info = se.classify_startup_error(exc)
    assert (info.category, info.hint) == ("resource", errors.ERROR_HINTS["gpu_oom"])


@pytest.mark.parametrize("message", ["[other] x", "[gpu] x", "[resource]x"])
def test_unknown_or_malformed_marker_is_ignored(message):
    assert se.classify_startup_error(RuntimeError(message)).category == "other"


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
    exc = errors.UnsupportedError("bad", hint="Use X.")
    se.mark_startup_stage(exc, "executor_creation/model")

    restored = pickle.loads(pickle.dumps(exc))
    info = se.classify_startup_error(restored)

    assert type(restored) is errors.UnsupportedError
    assert str(restored) == "bad"
    assert (info.category, info.reason, info.hint, info.stage) == (
        "unsupported",
        "unsupported",
        "Use X.",
        "executor_creation/model",
    )
    worker_died = pickle.loads(pickle.dumps(errors.RuntimeInitError("x", reason="worker_died")))
    assert (worker_died.reason, worker_died.hint) == (
        "worker_died",
        errors.ERROR_HINTS["worker_died"],
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
        "[model_load] KeyError: 'lm_head.weight'"
    )
    assert wrapper.__cause__ is unpickled
    proxy_info = se.classify_startup_error(wrapper)
    assert proxy_info == info
    assert proxy_info.category == "model_load"
    # Forwarded as text only (trtllm-llmapi-launch), the marker keeps the category.
    assert se.classify_startup_error(RuntimeError(str(wrapper))).category == "model_load"


@pytest.mark.parametrize(
    "cls",
    [
        errors.ConfigInputError,
        errors.UnsupportedError,
        errors.ResourceError,
        errors.ModelLoadError,
        errors.RuntimeInitError,
    ],
)
def test_wrapper_marks_typed_worker_error_for_text_only_channels(cls):
    # A typed error's message has no marker. The worker classifies it in-process and
    # the wrapper writes the marker, so trtllm-llmapi-launch's forwarded text keeps it.
    info = se.classify_startup_error(cls("went wrong"), rank=1)
    wrapper = se.wrap_worker_startup_error("Executor worker returned error", info)

    assert str(wrapper) == (
        f"Executor worker returned error on rank 1: [{cls.category}] {cls.__name__}: went wrong"
    )
    forwarded = RuntimeError(f"Remote MPI worker died: RuntimeError: {wrapper}")
    assert se.classify_startup_error(forwarded).category == cls.category


def test_wrapper_without_worker_info_keeps_plain_message():
    assert str(se.wrap_worker_startup_error("Executor worker returned error", None)) == (
        "Executor worker returned error"
    )


def test_wrapper_of_unclassified_error_has_no_marker():
    info = se.StartupErrorInfo("other", "ZeroDivisionError: division by zero", rank=0)
    assert str(se.wrap_worker_startup_error("Executor worker returned error", info)) == (
        "Executor worker returned error on rank 0: ZeroDivisionError: division by zero"
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


@pytest.fixture
def fresh_process(monkeypatch):
    """No LLM has initialized in this process yet."""
    monkeypatch.setattr(se, "_llm_initialized", False)


def _cli_error_lines(exc: BaseException) -> list:
    """Run a failing command through the CLI boundary and return the ERROR lines it logged."""
    from tensorrt_llm.commands import _telemetry

    def fail():
        raise exc

    with (
        mock.patch.object(_telemetry.usage, "report_exit"),
        mock.patch.object(_telemetry.usage, "get_observed_signal", return_value=0),
        mock.patch.object(se.logger, "error") as log_error,
        pytest.raises(type(exc)),
    ):
        _telemetry.run_with_terminal_reporting(fail)
    return [call.args[0] for call in log_error.call_args_list]


def test_cli_boundary_classifies_startup_failures(fresh_process):
    lines = _cli_error_lines(OSError(errno.EADDRINUSE, "Address already in use"))
    assert len(lines) == 1
    assert lines[0].startswith("Command failed [resource]: OSError: ")


def test_cli_boundary_skips_failures_after_an_llm_initialized(fresh_process):
    se.mark_llm_initialized()
    assert se.llm_initialized()

    assert _cli_error_lines(torch.OutOfMemoryError("CUDA out of memory")) == []
    # Another LLM that fails to initialize is still classified at its own boundary.
    with mock.patch.object(se.logger, "error") as log_error:
        info = se.log_startup_error(ValueError("x"), "LLM initialization failed")
    assert info is not None
    log_error.assert_called_once()
