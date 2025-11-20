import os as _os
import pathlib as _pl
import platform
import time
from enum import Enum, auto
from typing import List, Optional

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


def get_multi_gpu_env(kv_cache_type=KVCacheType.NONE, llama_multi_gpu=False):
    env = {**_os.environ}

    match kv_cache_type:
        case KVCacheType.MPI:
            env["TRTLLM_USE_MPI_KVCACHE"] = "1"
        case KVCacheType.UCX:
            env["TRTLLM_USE_UCX_KVCACHE"] = "1"
        case KVCacheType.NIXL:
            env["TRTLLM_USE_NIXL_KVCACHE"] = "1"
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


def run_llama_executor_leader_tests(build_dir: _pl.Path, timeout=1500):
    tests_dir = build_dir / "tests" / "e2e_tests"

    mgpu_env = get_multi_gpu_env(llama_multi_gpu=True)

    #Executor test in leader mode
    xml_output_file = build_dir / "results-multi-gpu-llama-exec-leader-mode.xml"
    trt_model_test = _cpp.produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=4,
        local_commands=[
            "executor/executorTest",
            "--gtest_filter=*LlamaExecutorTest*LeaderMode*:*LlamaMultiExecutorTest*LeaderMode*"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"])

    _cpp.run_command(trt_model_test, cwd=tests_dir, env=mgpu_env, timeout=1500)


def run_llama_executor_orchestrator_tests(build_dir: _pl.Path, timeout=1500):
    tests_dir = build_dir / "tests" / "e2e_tests"

    mgpu_env = get_multi_gpu_env(llama_multi_gpu=True)

    #Executor test in orchestrator mode
    xml_output_file = build_dir / "results-multi-gpu-llama-exec-orch-mode.xml"
    trt_model_test = [
        "mpirun", "-n", "1", "--allow-run-as-root", "executor/executorTest",
        "--gtest_filter=*LlamaExecutorTest*OrchMode*",
        f"--gtest_output=xml:{xml_output_file}"
    ]
    _cpp.run_command(trt_model_test, cwd=tests_dir, env=mgpu_env, timeout=1500)


def run_llama_executor_logits_proc_tests(build_dir: _pl.Path, timeout=1500):
    tests_dir = build_dir / "tests" / "e2e_tests"

    mgpu_env = get_multi_gpu_env(llama_multi_gpu=True)

    #Logits processor test in leader mode
    xml_output_file = build_dir / "results-multi-gpu-logits-proc.xml"

    tp_pp_sizes = [(4, 1), (2, 2), (1, 4)]
    gtest_filter = [
        f"LlamaExecutorTest/LogitsProcParamsTest*tp{tp}_pp{pp}*"
        for tp, pp in tp_pp_sizes
    ]

    gtest_filter = ":".join(gtest_filter)

    trt_model_test = _cpp.produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=4,
        local_commands=[
            "executor/executorTest", f"--gtest_filter={gtest_filter}"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"])

    _cpp.run_command(trt_model_test, cwd=tests_dir, env=mgpu_env, timeout=1500)


def run_llama_executor_guided_decoding_tests(build_dir: _pl.Path, timeout=1500):
    tests_dir = build_dir / "tests" / "e2e_tests"

    mgpu_env = get_multi_gpu_env(llama_multi_gpu=True)

    #Guided decoding test in leader mode
    xml_output_file = build_dir / "results-multi-gpu-guided-decoding.xml"

    tp_pp_sizes = [(4, 1), (2, 2), (1, 4)]
    gtest_filter = [
        f"LlamaExecutorGuidedDecodingTest/GuidedDecodingParamsTest*tp{tp}_pp{pp}*"
        for tp, pp in tp_pp_sizes
    ]

    gtest_filter = ":".join(gtest_filter)

    trt_model_test = _cpp.produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=4,
        local_commands=[
            "executor/executorTest", f"--gtest_filter={gtest_filter}"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"])

    _cpp.run_command(trt_model_test, cwd=tests_dir, env=mgpu_env, timeout=1500)


def run_enc_dec_multi_gpu_tests(build_dir: _pl.Path, timeout=1500):
    tests_dir = build_dir / "tests" / "e2e_tests"
    cpp_env = {**_os.environ}

    #EncDec test in leader mode
    xml_output_file = build_dir / "results-multi-gpu-t5-exec-leader-mode.xml"
    trt_model_test = _cpp.produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=4,
        local_commands=[
            "executor/encDecTest",
            "--gtest_filter=T5MultiGPUTest/EncDecParamsTest.Forward*"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"],
    )
    _cpp.run_command(trt_model_test, cwd=tests_dir, env=cpp_env, timeout=1500)


def run_trt_gpt_model_real_decoder_multi_gpu_tests(build_dir: _pl.Path,
                                                   timeout=1500):
    tests_dir = build_dir / "tests" / "e2e_tests"
    cpp_env = {**_os.environ}

    xml_output_file = build_dir / "results-multi-gpu-real-decoder.xml"
    trt_model_test = _cpp.produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=4,
        local_commands=[
            "batch_manager/trtGptModelRealDecoderTest",
            "--gtest_filter=*TP*:*PP*"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"])
    _cpp.run_command(trt_model_test,
                     cwd=tests_dir,
                     env=cpp_env,
                     timeout=timeout)  # expecting ~ 1200s


def run_disagg_symmetric_executor_tests(build_dir: _pl.Path,
                                        model: str,
                                        nprocs=2,
                                        kvcache_type=KVCacheType.MPI,
                                        timeout=1500):
    tests_dir = build_dir / "tests" / "e2e_tests"

    prefix = get_model_test_filter_prefix(model)

    mgpu_env = get_multi_gpu_env(kv_cache_type=kvcache_type,
                                 llama_multi_gpu=True)

    xml_output_file = build_dir / f"results-multi-gpu-disagg-executor-{nprocs}-process.xml"
    trt_model_test = _cpp.produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=nprocs,
        local_commands=[
            "executor/disaggExecutorTest",
            f"--gtest_filter=*{prefix}*DisaggSymmetricExecutorTest*"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"])

    _cpp.run_command(trt_model_test,
                     cwd=tests_dir,
                     env=mgpu_env,
                     timeout=timeout)


def run_disagg_asymmetric_executor_tests(build_dir: _pl.Path,
                                         model: str,
                                         nprocs=4,
                                         kvcache_type=KVCacheType.MPI,
                                         timeout=1500):

    tests_dir = build_dir / "tests" / "e2e_tests"

    prefix = get_model_test_filter_prefix(model)

    mgpu_env = get_multi_gpu_env(kv_cache_type=kvcache_type,
                                 llama_multi_gpu=True)

    xml_output_file = build_dir / f"results-multi-gpu-disagg-asymmetric-executor-{nprocs}-process.xml"

    trt_model_test = _cpp.produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=nprocs,
        local_commands=[
            "executor/disaggExecutorTest",
            f"--gtest_filter=*{prefix}*DisaggAsymmetricExecutorTest*"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"])

    _cpp.run_command(trt_model_test,
                     cwd=tests_dir,
                     env=mgpu_env,
                     timeout=timeout)


def run_disagg_orchestrator_params_tests(build_dir: _pl.Path,
                                         model: str,
                                         kvcache_type=KVCacheType.MPI,
                                         timeout=1500):

    tests_dir = build_dir / "tests" / "e2e_tests"

    prefix = get_model_test_filter_prefix(model)

    mgpu_env = get_multi_gpu_env(kv_cache_type=kvcache_type,
                                 llama_multi_gpu=True)

    xml_output_file = build_dir / "results-multi-gpu-disagg-asymmetric-orchestrator-executor-7-process.xml"
    trt_model_test = _cpp.produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=7,
        local_commands=[
            "executor/disaggExecutorTest",
            f"--gtest_filter=*{prefix}*DisaggOrchestratorParamsTest*"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"])
    _cpp.run_command(trt_model_test,
                     cwd=tests_dir,
                     env=mgpu_env,
                     timeout=timeout)


def run_disagg_spawn_orchestrator_tests(build_dir: _pl.Path,
                                        model: str,
                                        kvcache_type=False,
                                        timeout=1500):

    tests_dir = build_dir / "tests" / "e2e_tests"

    prefix = get_model_test_filter_prefix(model)

    mgpu_env = get_multi_gpu_env(kv_cache_type=kvcache_type,
                                 llama_multi_gpu=True)

    xml_output_file = build_dir / "results-multi-gpu-disagg-spawn-asymmetric-orchestrator-executor-1-process.xml"

    comms = [
        "executor/disaggExecutorTest",
        f"--gtest_filter=*{prefix}*DisaggSpawnOrchestrator*",
        f"--gtest_output=xml:{xml_output_file}"
    ]
    _cpp.run_command(comms, cwd=tests_dir, env=mgpu_env, timeout=timeout)


def prepare_multi_gpu_model_tests(test_list: List[str],
                                  python_exe: str,
                                  root_dir: _pl.Path,
                                  resources_dir: _pl.Path,
                                  model_cache: Optional[str] = None):

    model_cache_arg = ["--model_cache", model_cache] if model_cache else []

    if "llama" in test_list:
        _cpp.prepare_model_tests(model_name="llama",
                                 python_exe=python_exe,
                                 root_dir=root_dir,
                                 resources_dir=resources_dir,
                                 model_cache_arg=model_cache_arg,
                                 only_multi_gpu_arg=["--only_multi_gpu"])

    if "t5" in test_list:
        _cpp.prepare_model_tests(model_name="t5",
                                 python_exe=python_exe,
                                 root_dir=root_dir,
                                 resources_dir=resources_dir,
                                 model_cache_arg=model_cache_arg,
                                 only_multi_gpu_arg=['--tp', '4', '--pp', '1'])


@pytest.fixture(scope="session")
def prepare_model_multi_gpu(python_exe, root_dir, cpp_resources_dir,
                            model_cache):

    def _prepare(model_name: str):
        if platform.system() != "Windows":

            start_time = time.time()

            prepare_multi_gpu_model_tests(
                test_list=[model_name],
                python_exe=python_exe,
                root_dir=root_dir,
                resources_dir=cpp_resources_dir,
                model_cache=model_cache,
            )

            duration = time.time() - start_time
            print(f"Built multi-GPU model: {model_name}")
            print(f"Duration: {duration} seconds")

    return _prepare


@pytest.fixture(scope="session")
def gpt_single_gpu_model(prepare_model):
    prepare_model("gpt")
    return "gpt"


@pytest.fixture(scope="session")
def llama_single_gpu_model(prepare_model):
    prepare_model("llama")
    return "llama"


@pytest.fixture(scope="session")
def llama_multi_gpu_model(prepare_model_multi_gpu):
    prepare_model_multi_gpu("llama")
    return "llama"


# Allow us to dynamically choose a fixture at runtime
# Combined with session scope fixtures above to ensure
# that the model is built only once per pytest session
@pytest.fixture
def prepare_models_disagg(request):

    def _prepare(model_name: str):
        if model_name == "llama":
            fixture_names = [
                "llama_single_gpu_model",
                "llama_multi_gpu_model",
            ]
        elif model_name == "gpt":
            fixture_names = [
                "gpt_single_gpu_model",
            ]
        else:
            raise ValueError(f"Disagg tests don't support model: {model_name}")

        print(f"Preparing models for disagg tests: {fixture_names}")
        # Run the fixtures
        for fixture_name in fixture_names:
            request.getfixturevalue(fixture_name)

    return _prepare


# Use indirect parameterization to ensure that the model is built
# only once per pytest session
@pytest.fixture(scope="session")
def multi_gpu_model(request, prepare_model_multi_gpu):

    model_name = request.param
    prepare_model_multi_gpu(model_name)

    return model_name


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
@pytest.mark.parametrize("kvcache_type", [KVCacheType.NIXL, KVCacheType.UCX],
                         ids=["nixl_kvcache", "ucx_kvcache"])
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
@pytest.mark.parametrize("multi_gpu_model", ["t5"], indirect=True)
def test_enc_dec(build_google_tests, multi_gpu_model, build_dir):

    if platform.system() != "Windows":
        run_enc_dec_multi_gpu_tests(build_dir=build_dir,
                                    timeout=_cpp.default_test_timeout)


@pytest.mark.parametrize("build_google_tests", ["80", "86", "89", "90"],
                         indirect=True)
@pytest.mark.parametrize("mode", ["orchestrator", "leader"])
@pytest.mark.parametrize("multi_gpu_model", ["llama"], indirect=True)
def test_llama_executor(build_google_tests, multi_gpu_model, mode, lora_setup,
                        build_dir):

    if platform.system() == "Windows":
        return

    if mode == "orchestrator":
        run_llama_executor_orchestrator_tests(build_dir=build_dir,
                                              timeout=_cpp.default_test_timeout)
    elif mode == "leader":
        run_llama_executor_leader_tests(build_dir=build_dir,
                                        timeout=_cpp.default_test_timeout)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


@pytest.mark.parametrize("build_google_tests", ["80", "86", "89", "90"],
                         indirect=True)
@pytest.mark.parametrize("multi_gpu_model", ["llama"], indirect=True)
def test_llama_executor_logits_proc(build_google_tests, multi_gpu_model,
                                    lora_setup, build_dir):

    if platform.system() != "Windows":
        run_llama_executor_logits_proc_tests(build_dir=build_dir,
                                             timeout=_cpp.default_test_timeout)


@pytest.mark.parametrize("build_google_tests", ["80", "86", "89", "90"],
                         indirect=True)
@pytest.mark.parametrize("multi_gpu_model", ["llama"], indirect=True)
def test_llama_executor_guided_decoding(build_google_tests, multi_gpu_model,
                                        lora_setup, build_dir):

    if platform.system() != "Windows":
        run_llama_executor_guided_decoding_tests(
            build_dir=build_dir, timeout=_cpp.default_test_timeout)


@pytest.mark.parametrize("build_google_tests", ["80", "86", "89", "90"],
                         indirect=True)
@pytest.mark.parametrize("multi_gpu_model", ["llama"], indirect=True)
def test_trt_gpt_real_decoder(build_google_tests, multi_gpu_model, lora_setup,
                              build_dir):

    if platform.system() != "Windows":
        run_trt_gpt_model_real_decoder_multi_gpu_tests(
            build_dir=build_dir, timeout=_cpp.default_test_timeout)


@pytest.mark.parametrize("build_google_tests", ["80", "86", "89", "90"],
                         indirect=True)
class TestDisagg:

    @pytest.mark.parametrize(
        "kvcache_type", [KVCacheType.MPI, KVCacheType.UCX, KVCacheType.NIXL],
        ids=["mpi_kvcache", "ucx_kvcache", "nixl_kvcache"])
    @pytest.mark.parametrize("nprocs", [2, 4, 8],
                             ids=["2proc", "4proc", "8proc"])
    @pytest.mark.parametrize("model", ["gpt", "llama"])
    def test_symmetric_executor(self, build_google_tests, model, nprocs,
                                kvcache_type, prepare_models_disagg, build_dir):

        if model == "gpt" and nprocs > 2:
            pytest.skip(
                "test_symmetric_executor only supports 2 processes for gpt")

        if platform.system() != "Windows":
            prepare_models_disagg(model)

            run_disagg_symmetric_executor_tests(build_dir=build_dir,
                                                model=model,
                                                nprocs=nprocs,
                                                kvcache_type=kvcache_type)

    @pytest.mark.parametrize(
        "kvcache_type", [KVCacheType.MPI, KVCacheType.UCX, KVCacheType.NIXL],
        ids=["mpi_kvcache", "ucx_kvcache", "nixl_kvcache"])
    @pytest.mark.parametrize("nprocs", [4, 6, 8],
                             ids=["4proc", "6proc", "8proc"])
    @pytest.mark.parametrize("model", ["llama"])
    def test_asymmetric_executor(self, build_google_tests, model, nprocs,
                                 kvcache_type, prepare_models_disagg,
                                 build_dir):

        if platform.system() != "Windows":
            prepare_models_disagg(model_name=model)

            run_disagg_asymmetric_executor_tests(build_dir=build_dir,
                                                 model=model,
                                                 nprocs=nprocs,
                                                 kvcache_type=kvcache_type)

    @pytest.mark.parametrize(
        "kvcache_type", [KVCacheType.MPI, KVCacheType.UCX, KVCacheType.NIXL],
        ids=["mpi_kvcache", "ucx_kvcache", "nixl_kvcache"])
    @pytest.mark.parametrize("model", ["llama"])
    def test_orchestrator_params(self, build_google_tests, model, kvcache_type,
                                 prepare_models_disagg, build_dir):

        if platform.system() != "Windows":
            prepare_models_disagg(model)

            run_disagg_orchestrator_params_tests(build_dir=build_dir,
                                                 model=model,
                                                 kvcache_type=kvcache_type)

    @pytest.mark.parametrize("kvcache_type",
                             [KVCacheType.UCX, KVCacheType.NIXL],
                             ids=["ucx_kvcache", "nixl_kvcache"])
    @pytest.mark.parametrize("model", ["llama"])
    def test_spawn_orchestrator(self, build_google_tests, model, kvcache_type,
                                prepare_models_disagg, build_dir):

        if platform.system() != "Windows":
            prepare_models_disagg(model)

            run_disagg_spawn_orchestrator_tests(build_dir=build_dir,
                                                model=model,
                                                kvcache_type=kvcache_type)
