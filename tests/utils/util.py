import os
import unittest
from difflib import SequenceMatcher

import pytest
import tensorrt as trt
import torch
from cuda import cuda, nvrtc
from parameterized import parameterized

import tensorrt_llm
from tensorrt_llm._utils import torch_dtype_to_trt, trt_dtype_to_torch
from tensorrt_llm.plugin.plugin import ContextFMHAType
from tensorrt_llm.quantization import QuantMode
from tensorrt_llm.runtime import TensorInfo


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


skip_pre_ampere = pytest.mark.skipif(
    getSMVersion() < 80,
    reason="This test is not supported in pre-Ampere architecture")
skip_pre_ada = pytest.mark.skipif(
    getSMVersion() < 89,
    reason="This test is not supported in pre-Ada architecture")
skip_pre_hopper = pytest.mark.skipif(
    getSMVersion() < 90,
    reason="This test is not supported in pre-Hopper architecture")

# If used together with @parameterized, we have to use unittest.skipIf instead of pytest.mark.skipif
skip_pre_ampere_unittest = unittest.skipIf(
    getSMVersion() < 80,
    reason="This test is not supported in pre-Ampere architecture")
skip_pre_ada_unittest = unittest.skipIf(
    getSMVersion() < 89 or (getSMVersion() == 89 and getCUDAVersion() < 1204),
    reason=
    "This test is not supported in pre-Ada architecture, and for Ada we require cuda version >= 12.4"
)
skip_pre_hopper_unittest = unittest.skipIf(
    getSMVersion() < 90,
    reason="This test is not supported in pre-Hopper architecture")

IGNORE_ARCH = os.environ.get('TLLM_TEST_IGNORE_ARCH', False)
force_ampere = pytest.mark.skipif(
    (not IGNORE_ARCH) and (getSMVersion() < 80 or getSMVersion() > 89),
    reason="This test is only enabled in Ampere architecture")


def is_bf16(dtype):
    return dtype == 'bfloat16' or dtype == 'bf16' or dtype == torch.bfloat16


def skip_fp32_accum_pre_ampere(context_fmha_type):
    if context_fmha_type == ContextFMHAType.enabled_with_fp32_acc and getSMVersion(
    ) < 80:
        pytest.skip(
            "ContextFMHAType with fp32 acc is not supported in pre-Ampere architecture"
        )


def skip_bf16_pre_ampere(dtype):
    if is_bf16(dtype) and getSMVersion() < 80:
        pytest.skip("bfloat16 is not supported in pre-Ampere architecture")


def skip_fp8_pre_ada(use_fp8):
    if use_fp8 and getSMVersion() < 89:
        pytest.skip("FP8 is not supported on pre-Ada architectures")


def skip_bf16_fp32_accum(dtype, context_fmha_type):
    if is_bf16(dtype
               ) and context_fmha_type == ContextFMHAType.enabled_with_fp32_acc:
        pytest.skip(
            "bfloat16 Context FMHA will always accumulate on FP32, so it has been tested with ContextFMHAType.enabled"
        )


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


def create_session(builder,
                   network,
                   precision="float32",
                   int8=False,
                   opt_level=None,
                   memory_pool_limit=None,
                   optimization_profiles=[],
                   quant_mode=QuantMode(0)):
    """
    This function creates an engine and a tensorrt_llm.runtime.Session for the engine.
    Args:
        network: a tensorrt_llm.Network object
        precision: the precision of the network, choose from ["float32", "float16", "bfloat16"]
        **kwargs: builder flags such as int8, fp8, builder_opt, etc.
    Returns:
        session: a tensorrt_llm.runtime.Session
    """
    builder_config = builder.create_builder_config(precision=precision,
                                                   int8=int8,
                                                   opt_level=opt_level,
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
    session = tensorrt_llm.runtime.Session.from_serialized_engine(engine)
    return session


def run_session(session, inputs):
    """
    The current session object needs to pass in both inputs and outputs bindings.
    For test convenience, create a function that infers output shapes automatically,
    This function is similar to tensorrt_llm.runtime.Session._debug_run, and Polygraphy runner.infer,
    where only input shape is required.
    """

    # Prepare output tensors.
    output_info = session.infer_shapes([
        TensorInfo(name, torch_dtype_to_trt(tensor.dtype), tensor.shape)
        for name, tensor in inputs.items()
    ])

    outputs = {
        t.name: torch.empty(tuple(t.shape),
                            dtype=trt_dtype_to_torch(t.dtype),
                            device='cuda')
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
