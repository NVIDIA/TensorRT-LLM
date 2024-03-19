import pytest
import torch
from cuda import cuda, nvrtc
from parameterized import parameterized

from tensorrt_llm.plugin.plugin import ContextFMHAType


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


skip_pre_ampere = pytest.mark.skipif(
    getSMVersion() < 80,
    reason="This test is not supported in pre-Ampere architecture")
skip_pre_ada = pytest.mark.skipif(
    getSMVersion() < 89,
    reason="This test is not supported in pre-Ada architecture")
skip_pre_hopper = pytest.mark.skipif(
    getSMVersion() < 90,
    reason="This test is not supported in pre-Hopper architecture")

force_ampere = pytest.mark.skipif(
    getSMVersion() < 80 or getSMVersion() > 89,
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


def ammo_installed():
    try:
        # isort: off
        import ammo.torch.quantization as atq
        from ammo.torch.export import export_model_config
        print(type(atq))
        print(type(export_model_config))
        # isort: on
        return True
    except Exception:
        return False
    return False


skip_no_ammo = pytest.mark.skipif(not ammo_installed(),
                                  reason="AMMO is not installed")


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
