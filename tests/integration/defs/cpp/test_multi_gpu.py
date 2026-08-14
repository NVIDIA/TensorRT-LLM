import os as _os
import pathlib as _pl
import platform
from enum import Enum, auto

import defs.cpp.cpp_common as _cpp
import pytest
from defs.conftest import skip_no_nvls


# Helper filter for disagg google tests
def get_model_test_filter_prefix(model: str) -> str:
    if model == "llama":
        return "Llama"
    elif model == "gpt":
        return "Gpt"
    else:
        raise ValueError(f"Unsupported model: {model}")


class KVCacheType(Enum):
    NONE = auto()
    MPI = auto()
    UCX = auto()
    NIXL = auto()
    MOONCAKE = auto()


def get_multi_gpu_env(kv_cache_type=KVCacheType.NONE, llama_multi_gpu=False):
    env = {**_os.environ}

    match kv_cache_type:
        case KVCacheType.MPI:
            env["TRTLLM_USE_MPI_KVCACHE"] = "1"
        case KVCacheType.UCX:
            env["TRTLLM_USE_UCX_KVCACHE"] = "1"
        case KVCacheType.NIXL:
            env["TRTLLM_USE_NIXL_KVCACHE"] = "1"
        case KVCacheType.MOONCAKE:
            env["TRTLLM_USE_MOONCAKE_KVCACHE"] = "1"
            env["MC_FORCE_TCP"] = "1"
        case KVCacheType.NONE:
            pass
        case _:
            raise ValueError(f"Unsupported KVCacheType: {kv_cache_type}")

    if llama_multi_gpu:
        env["RUN_LLAMA_MULTI_GPU"] = "true"

    return env


def run_mpi_utils_tests(build_dir, timeout=300):

    tests_dir = build_dir / "tests" / "unit_tests" / "multi_gpu"
    mgpu_env = get_multi_gpu_env()

    mpi_utils_test = [
        "mpirun",
        "-n",
        "4",
        "--allow-run-as-root",
        "mpiUtilsTest",
    ]
    _cpp.run_command(mpi_utils_test,
                     cwd=tests_dir,
                     env=mgpu_env,
                     timeout=timeout)


def run_gemm_allreduce_tests(build_dir, nprocs, timeout=300):

    tests_dir = build_dir / "tests" / "unit_tests" / "multi_gpu"
    mgpu_env = get_multi_gpu_env()

    gemm_allreduce_test = [
        "mpirun",
        "-n",
        f"{nprocs}",
        "--allow-run-as-root",
        "kernels/gemmAllReduceTest",
        "--m=2032",
        "--n=8200",
        "--k=1024",
        "--iterations=1",
    ]
    _cpp.run_command(gemm_allreduce_test,
                     cwd=tests_dir,
                     env=mgpu_env,
                     timeout=timeout)


def run_cache_transceiver_tests(build_dir: _pl.Path,
                                nprocs=2,
                                kv_cache_type=KVCacheType.MPI,
                                timeout=600):

    tests_dir = build_dir / "tests" / "unit_tests" / "multi_gpu"
    mgpu_env = get_multi_gpu_env(kv_cache_type=kv_cache_type)

    cache_trans_test = [
        "mpirun",
        "-n",
        f"{nprocs}",
        "--allow-run-as-root",
        "cacheTransceiverTest",
    ]
    _cpp.run_command(cache_trans_test,
                     cwd=tests_dir,
                     env=mgpu_env,
                     timeout=timeout)


def run_user_buffer_tests(build_dir: _pl.Path, nprocs=2, timeout=300):
    tests_dir = build_dir / "tests" / "unit_tests" / "multi_gpu"
    mgpu_env = get_multi_gpu_env()

    user_buffer_test = [
        "mpirun",
        "-n",
        f"{nprocs}",
        "--allow-run-as-root",
        "userBufferTest",
    ]

    _cpp.run_command(user_buffer_test,
                     cwd=tests_dir,
                     env=mgpu_env,
                     timeout=timeout)


def run_nccl_utils_tests(build_dir: _pl.Path, nprocs=2, timeout=300):
    tests_dir = build_dir / "tests" / "unit_tests" / "multi_gpu"
    mgpu_env = get_multi_gpu_env()

    nccl_utils_test = [
        "mpirun",
        "-n",
        f"{nprocs}",
        "--allow-run-as-root",
        "ncclUtilsTest",
    ]

    _cpp.run_command(nccl_utils_test,
                     cwd=tests_dir,
                     env=mgpu_env,
                     timeout=timeout)


@pytest.mark.parametrize("build_google_tests", ["80", "86", "89", "90"],
                         indirect=True)
def test_mpi_utils(build_google_tests, build_dir):

    if platform.system() != "Windows":
        run_mpi_utils_tests(build_dir, timeout=300)


@skip_no_nvls
@pytest.mark.parametrize("build_google_tests", ["90", "100"], indirect=True)
@pytest.mark.parametrize("nprocs", [2, 4], ids=["2proc", "4proc"])
def test_fused_gemm_allreduce(build_google_tests, nprocs, build_dir):

    if platform.system() != "Windows":
        run_gemm_allreduce_tests(build_dir, nprocs, timeout=300)


@pytest.mark.parametrize("build_google_tests", ["80", "86", "89", "90"],
                         indirect=True)
@pytest.mark.parametrize(
    "kvcache_type", [KVCacheType.NIXL, KVCacheType.UCX, KVCacheType.MOONCAKE],
    ids=["nixl_kvcache", "ucx_kvcache", "mooncake_kvcache"])
@pytest.mark.parametrize("nprocs", [2, 8], ids=["2proc", "8proc"])
def test_cache_transceiver(build_google_tests, nprocs, kvcache_type, build_dir):

    if platform.system() != "Windows":
        run_cache_transceiver_tests(build_dir=build_dir,
                                    nprocs=nprocs,
                                    kv_cache_type=kvcache_type,
                                    timeout=600)


@pytest.mark.parametrize("build_google_tests", ["80", "86", "89", "90"],
                         indirect=True)
@pytest.mark.parametrize("nprocs", [2, 8], ids=["2proc", "8proc"])
def test_user_buffer(build_google_tests, nprocs, build_dir):

    if platform.system() != "Windows":
        run_user_buffer_tests(build_dir=build_dir, nprocs=nprocs, timeout=300)


@pytest.mark.parametrize("build_google_tests", ["80", "86", "89", "90"],
                         indirect=True)
@pytest.mark.parametrize("nprocs", [2, 8], ids=["2proc", "8proc"])
def test_nccl_utils(build_google_tests, nprocs, build_dir):

    if platform.system() != "Windows":
        run_nccl_utils_tests(build_dir=build_dir, nprocs=nprocs, timeout=300)
