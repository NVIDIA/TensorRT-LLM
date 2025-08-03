import multiprocessing
import os
import sys
import time
import unittest
from contextlib import contextmanager
from difflib import SequenceMatcher
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, Callable, Generator, Mapping, Tuple

import pynvml
import pytest
import tensorrt as trt
import torch
from cuda import cuda, nvrtc
from parameterized import parameterized

import tensorrt_llm
from tensorrt_llm._utils import torch_dtype_to_trt, trt_dtype_to_torch
from tensorrt_llm.llmapi.utils import get_total_gpu_memory
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.runtime import Session, TensorInfo


def ASSERT_DRV(err):
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            raise RuntimeError('Cuda Error: {}'.format(err))
    elif isinstance(err, nvrtc.nvrtcResult):
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise RuntimeError('Nvrtc Error: {}'.format(err))
    else:
        raise RuntimeError('Unknown error type: {}'.format(err))


# ref: https://github.com/NVIDIA/cuda-python/blob/main/examples/extra/jit_program_test.py
def getSMVersion():
    # Init
    err, = cuda.cuInit(0)
    ASSERT_DRV(err)

    # Device
    err, cuDevice = cuda.cuDeviceGet(0)
    ASSERT_DRV(err)

    # Get target architecture
    err, sm_major = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        cuDevice)
    ASSERT_DRV(err)
    err, sm_minor = cuda.cuDeviceGetAttribute(
        cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        cuDevice)
    ASSERT_DRV(err)

    return sm_major * 10 + sm_minor


def getCUDAVersion():
    import subprocess

    try:
        cuda_version = subprocess.run(['nvcc', '--version'],
                                      stdout=subprocess.PIPE,
                                      universal_newlines=True)
        output = cuda_version.stdout.split()
        release_version = output[-4].replace(',', '.').split('.')
        return int(release_version[0]) * 100 + int(release_version[1])
    except Exception as e:
        print(f"Error getting CUDA version: {e}")


skip_pre_ada = pytest.mark.skipif(
    getSMVersion() < 89,
    reason="This test is not supported in pre-Ada architecture")
skip_pre_hopper = pytest.mark.skipif(
    getSMVersion() < 90,
    reason="This test is not supported in pre-Hopper architecture")
skip_pre_blackwell = pytest.mark.skipif(
    getSMVersion() < 100,
    reason="This test is not supported in pre-Blackwell architecture")
skip_blackwell = pytest.mark.skipif(
    getSMVersion() == 100,
    reason="This test is not supported in Blackwell architecture")
skip_blackwell_geforce = pytest.mark.skipif(
    getSMVersion() == 120, reason="This test is not supported on SM 120")

# If used together with @parameterized, we have to use unittest.skipIf instead of pytest.mark.skipif
skip_pre_ada_unittest = unittest.skipIf(
    getSMVersion() < 89 or (getSMVersion() == 89 and getCUDAVersion() < 1204),
    reason=
    "This test is not supported in pre-Ada architecture, and for Ada we require cuda version >= 12.4"
)
skip_pre_hopper_unittest = unittest.skipIf(
    getSMVersion() < 90,
    reason="This test is not supported in pre-Hopper architecture")
skip_pre_blackwell_unittest = unittest.skipIf(
    getSMVersion() < 100,
    reason="This test is not supported in pre-Blackwell architecture")
skip_non_ada_unittest = unittest.skipIf(
    getSMVersion() != 89,
    reason="This test is only supported in Ada architecture")
skip_non_hopper_unittest = unittest.skipIf(
    getSMVersion() != 90,
    reason="This test is only supported in Hopper architecture")
skip_neither_ada_nor_hopper_unittest = unittest.skipIf(
    getSMVersion() != 90 and getSMVersion() != 89,
    reason="This test is only supported in Ada or Hopper architecture")

IGNORE_ARCH = os.environ.get('TLLM_TEST_IGNORE_ARCH', False)
force_ampere = pytest.mark.skipif(
    (not IGNORE_ARCH) and (getSMVersion() < 80 or getSMVersion() > 89),
    reason="This test is only enabled in Ampere architecture")


def is_bf16(dtype):
    return dtype == 'bfloat16' or dtype == 'bf16' or dtype == torch.bfloat16


def skip_fp8_pre_ada(use_fp8):
    if use_fp8 and getSMVersion() < 89:
        pytest.skip("FP8 is not supported on pre-Ada architectures")


def skip_blackwell_for_fmha_tests(context_fmha_type, head_size):
    if getSMVersion() == 100 and (head_size not in [32, 64, 128]
                                  and context_fmha_type
                                  != ContextFMHAType.disabled):
        pytest.skip(
            "Context FMHA only supports head sizes [32, 64, 128] currently on blackwell."
        )


def skip_fp4_pre_blackwell(use_fp4):
    if use_fp4 and getSMVersion() < 100:
        pytest.skip("FP4 is not supported on pre-Blackwell architectures")


def skip_bf16_fp32_accum(dtype, context_fmha_type):
    if is_bf16(dtype
               ) and context_fmha_type == ContextFMHAType.enabled_with_fp32_acc:
        pytest.skip(
            "bfloat16 Context FMHA will always accumulate on FP32, so it has been tested with ContextFMHAType.enabled"
        )


def skip_num_gpus_less_than(num_gpus: int):
    return pytest.mark.skipif(
        torch.cuda.device_count() < num_gpus,
        reason=f"The test needs at least {num_gpus} GPUs, skipping")


skip_single_gpu = skip_num_gpus_less_than(2)


def compose_decorator(*decorators):

    def composed_decorator(f):
        for dec in reversed(decorators):
            f = dec(f)
        return f

    return composed_decorator


pytest.mark.gpu2 = compose_decorator(skip_single_gpu, pytest.mark.gpu2)
pytest.mark.gpu4 = compose_decorator(skip_num_gpus_less_than(4),
                                     pytest.mark.gpu4)


def skip_gpu_memory_less_than(required_memory: int):
    memory = get_total_gpu_memory(0)
    return pytest.mark.skipif(
        required_memory > memory,
        reason=
        f'Not enough GPU memory for this test (wanted {required_memory}, have {memory})'
    )


skip_gpu_memory_less_than_40gb = skip_gpu_memory_less_than(40 * 1024 * 1024 *
                                                           1024)

skip_gpu_memory_less_than_80gb = skip_gpu_memory_less_than(80 * 1024 * 1024 *
                                                           1024)

skip_gpu_memory_less_than_138gb = skip_gpu_memory_less_than(138 * 1024 * 1024 *
                                                            1024)


def modelopt_installed():
    try:
        # isort: off
        import modelopt.torch.quantization as atq  # NOQA
        from modelopt.torch.export import export_tensorrt_llm_checkpoint  # NOQA
        # isort: on
        return True
    except Exception:
        return False


skip_no_modelopt = unittest.skipIf(not modelopt_installed(),
                                   reason="Modelopt is not installed")


def check_nvlink():
    "check nvlink is active"
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count > 0:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            try:
                link_state = pynvml.nvmlDeviceGetNvLinkState(handle, 0)
                print(f"Link state is {link_state}")

            except pynvml.NVMLError as error:
                print(
                    f"Device does not seem to support NVLink or there's an issue: {error}"
                )
        pynvml.nvmlShutdown()
        return True

    except pynvml.NVMLError as error:
        print(f"Error initializing NVML or other NVML error: {error}")
        return False


skip_nvlink_inactive = unittest.skipIf(not check_nvlink(),
                                       reason="nvlink is inactive.")


# This function names will make all unit tests names to show the values of all parameters in @parameterized.expand
def unittest_name_func(testcase_func, param_num, param):
    expand_params = lambda params: '_'.join([
        expand_params(x) if isinstance(x, (list, tuple)) else str(x)
        for x in params
    ])
    name = expand_params(param.args)
    return "%s_%s" % (
        testcase_func.__name__,
        parameterized.to_safe_name(name),
    )


def set_input_shape(profile,
                    inp: tensorrt_llm.Tensor,
                    shape: tuple,
                    data: torch.Tensor = None):
    set_input_shapes(profile, inp, shape, shape, shape, data)
    return


def set_input_shapes(profile,
                     inp: tensorrt_llm.Tensor,
                     min_shape: tuple,
                     opt_shape: tuple,
                     max_shape: tuple,
                     data: torch.Tensor = None):
    if inp.trt_tensor.is_shape_tensor:
        # For shape tensors, TensorRT expects the full tensor (on CPU), not just shape
        assert data is not None, f"For shape tensor {inp.name}, TensorRT needs the tensor value."
        assert str(data.device) == "cpu", f"Shape tensor's data needs to be on CPU " \
            f"(device found={data.device}) for both updating the profile and for execution."
        np_data = data.flatten().numpy()
        profile.set_shape_input(inp.name, np_data, np_data, np_data)
        return
    profile.set_shape(inp.name, min_shape, opt_shape, max_shape)
    return


def create_session(builder,
                   network,
                   precision="float32",
                   int8=False,
                   fp8=False,
                   memory_pool_limit=None,
                   optimization_profiles=[],
                   quant_mode=QuantMode(0)):
    """
    This function creates an engine and a tensorrt_llm.runtime.Session for the engine.
    Args:
        network: a tensorrt_llm.Network object
        precision: the precision of the network, choose from ["float32", "float16", "bfloat16"]
        **kwargs: builder flags such as int8, fp8, etc.
    Returns:
        session: a tensorrt_llm.runtime.Session
    """
    builder_config = builder.create_builder_config(precision=precision,
                                                   int8=int8,
                                                   fp8=fp8,
                                                   quant_mode=quant_mode)
    # Some tests require to set mem pool limit to avoid OOM
    if memory_pool_limit is not None:
        builder_config.trt_builder_config.set_memory_pool_limit(
            trt.MemoryPoolType.WORKSPACE, memory_pool_limit)
    # Some tests include shape tensors, so the optimization profile needs to be feed in explicitly
    if len(optimization_profiles) > 0:
        for profile in optimization_profiles:
            builder_config.trt_builder_config.add_optimization_profile(profile)
    # Disable TF32 for accuracy in testing.
    builder_config.trt_builder_config.clear_flag(trt.BuilderFlag.TF32)
    engine = builder.build_engine(network, builder_config)
    assert engine is not None, "Failed to build engine"
    session = Session.from_serialized_engine(engine)
    return session


def run_session(session: Session,
                inputs,
                outputs={},
                override_shapes={},
                override_types={}):
    """
    The current session object needs to pass in both inputs and outputs bindings.
    For test convenience, create a function that infers output shapes automatically,
    This function is similar to tensorrt_llm.runtime.Session._debug_run, and Polygraphy runner.infer,
    where only input shape is required.
    NOTES:
        1. The outputs dictionary is required for outputs for which the shapes cannot be inferred.
           This function will prioritize to use the tensor in this dictionary.
        2. `override_shapes` can be used to force some input tensors' shape to be different than the passed tensor.
           Required for zero-volume tensors since torch.Tensor.data_ptr() is nullptr for such tensors.
        3. `override_types` can be used to force some input tensors' type to be different than the passed tensor.
           Required for zero-volume tensors since torch.Tensor.data_ptr() is nullptr for such tensors.
    """

    # Prepare output tensors.
    output_info = session.infer_shapes([
        TensorInfo(
            name,
            torch_dtype_to_trt(tensor.dtype if name not in
                               override_types else override_types[name]),
            tensor.shape
            if name not in override_shapes else override_shapes[name])
        for name, tensor in inputs.items()
    ])

    def create_torch(t):
        if t.dtype == trt.fp4:
            shape = list(t.shape)
            shape[-1] = shape[-1] // 2
            return torch.empty(tuple(shape), dtype=torch.uint8, device='cuda')
        else:
            return torch.empty(tuple(t.shape),
                               dtype=trt_dtype_to_torch(t.dtype),
                               device='cuda')

    outputs = {
        t.name: create_torch(t) if t.name not in outputs else outputs[t.name]
        for t in output_info
    }

    # Execute model inference
    stream = torch.cuda.current_stream()
    ok = session.run(inputs=inputs, outputs=outputs, stream=stream.cuda_stream)
    assert ok, 'Engine execution failed'
    stream.synchronize()

    return outputs


def similarity_score(a, b):
    "similar compare a and b "
    return SequenceMatcher(None, a, b).ratio()


def similar(a, b, threshold=0.8):
    "similar compare a and b "
    return similarity_score(a, b) >= threshold


def get_project_root(test_file: str) -> Path:
    return next(p for p in Path(test_file).resolve().parents
                if (p / 'tests').is_dir() and (p / "tensorrt_llm").is_dir())


@contextmanager
def default_dtype(dtype: torch.dtype):
    cur_default = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(cur_default)


def woq_assert_near_eq(ref, act, wTypeId):
    # match the scale in cpp/tensorrt_llm/kernels/cutlass_kernels/cutlass_preprocessors.cpp
    if wTypeId == 1:
        bits_in_type = 8
    else:
        bits_in_type = 4
    quant_range_scale = 1.0 / float(1 << (bits_in_type - 1))

    max_val = torch.max(abs(ref)).item()
    atol = (max_val * quant_range_scale) * 1.5  # allow for rounding
    torch.testing.assert_close(ref, act, atol=atol, rtol=1e-7)


def woq_groupwise_gt_matmul(mat1, ref_torch_weights, bias=None):
    ref = torch.matmul(mat1, ref_torch_weights)
    if bias is not None:
        ref += bias
    return ref


def flatten_list_generator(
        nested_list: list[Any]) -> Generator[Any, None, None]:
    if not isinstance(nested_list, list):
        yield nested_list
    else:
        for item in nested_list:
            yield from flatten_list_generator(item)


def flatten_list(nested_list: list[Any]) -> list[Any]:
    return list(flatten_list_generator(nested_list))


def duplicate_list_to_length(list: list[Any], target_length: int) -> list[Any]:
    if target_length < len(list):
        return list[:target_length]
    duplicated_list = list * (target_length // len(list))
    remain = target_length % len(list)
    if remain != 0:
        duplicated_list += list[:remain]
    return duplicated_list


def _target_wrapper(target: Callable, stdout_pipe: Connection,
                    stderr_pipe: Connection, *args, **kwargs) -> None:

    class PipeWriter:

        def __init__(self, conn: Connection):
            self.conn = conn

        def write(self, s: str):
            self.conn.send_bytes(s.encode("UTF8"))

        def flush(self):
            pass

    sys.stdout = PipeWriter(stdout_pipe)
    sys.stderr = PipeWriter(stderr_pipe)
    target(*args, **kwargs)


def run_function_in_sub_process(target: Callable,
                                args: tuple,
                                kwargs: Mapping[str, Any],
                                stop_waiting_criteria: Callable,
                                poll_interval_seconds: int = 5,
                                timeout_seconds: int = 240) -> Tuple[str, str]:
    multiprocessing.set_start_method("spawn", force=True)
    parent_stdout_pipe, child_stdout_pipe = multiprocessing.Pipe()
    parent_stderr_pipe, child_stderr_pipe = multiprocessing.Pipe()
    child_process = multiprocessing.Process(
        target=_target_wrapper,
        args=[target, child_stdout_pipe, child_stderr_pipe] + list(args),
        kwargs=kwargs)
    child_process.start()
    child_stdout_pipe.close()
    child_stderr_pipe.close()

    def _read_from_pipe(pipe: Connection):
        out = ""
        while pipe.poll(timeout=0.1):
            try:
                out += pipe.recv_bytes().decode("UTF8")
            except Exception:
                break
        return out

    child_stdout = ""
    child_stderr = ""
    try:
        total_waiting_seconds = 0
        while child_process.is_alive(
        ) and total_waiting_seconds < timeout_seconds:
            child_stdout += _read_from_pipe(parent_stdout_pipe)
            child_stderr += _read_from_pipe(parent_stderr_pipe)
            if stop_waiting_criteria(child_stdout, child_stderr):
                break
            time.sleep(poll_interval_seconds)
            total_waiting_seconds += poll_interval_seconds
    finally:
        parent_stdout_pipe.close()
        parent_stderr_pipe.close()
        if child_process.is_alive():
            child_process.terminate()

    assert total_waiting_seconds < timeout_seconds, "Reached timeout while waiting for target"
    return child_stdout, child_stderr


class EnvVarsContextManager:

    def __init__(self, new_env_vars: dict[str, str]):
        self._env_vars = new_env_vars
        self._original_value = None

    def __enter__(self):
        self._original_vars = {
            var_name: os.environ[var_name]
            for var_name in self._env_vars.keys() if var_name in os.environ
        }
        os.environ.update(self._env_vars)

    def __exit__(self, type, value, traceback):
        os.environ.update(self._original_vars)
        for var_name in self._env_vars.keys():
            if var_name not in self._original_vars:
                os.environ.pop(var_name)


# Check a certain percentage of elements in two tensors are within a tolerance
def check_accuracy(a, b, atol, rtol, percent):
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    a = a.to(torch.float32)
    b = b.to(torch.float32)
    left = torch.abs(a - b)
    right = atol + rtol * torch.abs(b)
    count = torch.sum(left > right)
    mismatch_percent = count / a.numel()
    if not (mismatch_percent < 1 - percent):
        raise Exception("Mismatch percentage is %f for rtol %f" %
                        (mismatch_percent, rtol))
