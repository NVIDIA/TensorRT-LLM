import copy
import os as _os
import pathlib as _pl
import platform
import time
from typing import List, Optional

import defs.cpp.cpp_common as _cpp
import pytest


def run_simple_multi_gpu_tests(build_dir: _pl.Path, timeout=1500):
    tests_dir = build_dir / "tests"
    cpp_env = {**_os.environ}
    # Utils tests
    mpi_utils_test = [
        "mpirun",
        "-n",
        "4",
        "--allow-run-as-root",
        "mpiUtilsTest",
    ]
    _cpp.run_command(mpi_utils_test, cwd=tests_dir, env=cpp_env, timeout=300)

    # Cache transceiver tests
    new_env = copy.copy(cpp_env)
    new_env["TRTLLM_USE_MPI_KVCACHE"] = "1"
    cache_trans_test = [
        "mpirun",
        "-n",
        "2",
        "--allow-run-as-root",
        "batch_manager/cacheTransceiverTest",
    ]
    _cpp.run_command(cache_trans_test, cwd=tests_dir, env=new_env, timeout=300)

    new_env = copy.copy(cpp_env)
    new_env["TRTLLM_USE_MPI_KVCACHE"] = "1"
    # Cache transceiver tests
    cache_trans_test_8_proc = [
        "mpirun",
        "-n",
        "8",
        "--allow-run-as-root",
        "batch_manager/cacheTransceiverTest",
    ]
    _cpp.run_command(cache_trans_test_8_proc,
                     cwd=tests_dir,
                     env=new_env,
                     timeout=600)

    # Cache transceiver tests with UCX
    new_env = copy.copy(cpp_env)
    new_env["TRTLLM_USE_UCX_KVCACHE"] = "1"
    cache_trans_test = [
        "mpirun",
        "-n",
        "2",
        "--allow-run-as-root",
        "batch_manager/cacheTransceiverTest",
    ]
    _cpp.run_command(cache_trans_test, cwd=tests_dir, env=new_env, timeout=300)

    new_env = copy.copy(cpp_env)
    new_env["TRTLLM_USE_UCX_KVCACHE"] = "1"
    # Cache transceiver tests
    cache_trans_test_8_proc = [
        "mpirun",
        "-n",
        "8",
        "--allow-run-as-root",
        "batch_manager/cacheTransceiverTest",
    ]
    _cpp.run_command(cache_trans_test_8_proc,
                     cwd=tests_dir,
                     env=new_env,
                     timeout=600)

    new_env = copy.copy(cpp_env)
    # Transfer agent tests
    transfer_agent_test_2_proc = [
        "mpirun",
        "-n",
        "2",
        "--allow-run-as-root",
        "executor/transferAgentTest",
    ]
    _cpp.run_command(transfer_agent_test_2_proc,
                     cwd=tests_dir,
                     env=new_env,
                     timeout=600)


def run_llama_executor_multi_gpu_tests(build_dir: _pl.Path, timeout=1500):
    tests_dir = build_dir / "tests"
    cpp_env = {**_os.environ}

    mgpu_env = copy.copy(cpp_env)
    mgpu_env["RUN_LLAMA_MULTI_GPU"] = "true"

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
    # https://nvbugspro.nvidia.com/bug/5026255 disable below tests for now.
    if False:
        _cpp.run_command(trt_model_test,
                         cwd=tests_dir,
                         env=mgpu_env,
                         timeout=1500)

    #Executor test in orchestrator mode
    xml_output_file = build_dir / "results-multi-gpu-llama-exec-orch-mode.xml"
    trt_model_test = [
        "mpirun", "-n", "1", "--allow-run-as-root", "executor/executorTest",
        "--gtest_filter=*LlamaExecutorTest*OrchMode*",
        f"--gtest_output=xml:{xml_output_file}"
    ]
    _cpp.run_command(trt_model_test, cwd=tests_dir, env=mgpu_env, timeout=1500)

    #Logits processor and guided decoding test in leader mode
    xml_output_file = build_dir / "results-multi-gpu-logits-proc.xml"
    tp_pp_sizes = [(4, 1), (2, 2), (1, 4)]
    gtest_filter = [
        f"LlamaExecutorTest/LogitsProcParamsTest*tp{tp}_pp{pp}*"
        for tp, pp in tp_pp_sizes
    ]
    gtest_filter.extend([
        f"LlamaExecutorGuidedDecodingTest/GuidedDecodingParamsTest*tp{tp}_pp{pp}*"
        for tp, pp in tp_pp_sizes
    ])
    gtest_filter = ":".join(gtest_filter)
    trt_model_test = _cpp.produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=4,
        local_commands=[
            "executor/executorTest", f"--gtest_filter={gtest_filter}"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"])
    _cpp.run_command(trt_model_test, cwd=tests_dir, env=mgpu_env, timeout=1500)


def run_t5_multi_gpu_tests(build_dir: _pl.Path, timeout=1500):
    tests_dir = build_dir / "tests"
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
    tests_dir = build_dir / "tests"
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


def run_disagg_multi_gpu_tests(build_dir: _pl.Path):

    tests_dir = build_dir / "tests"
    cpp_env = {**_os.environ}

    new_env = copy.copy(cpp_env)
    new_env["TRTLLM_USE_MPI_KVCACHE"] = "1"
    xml_output_file = build_dir / "results-multi-gpu-disagg-executor-2-process.xml"
    trt_model_test = _cpp.produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=2,
        local_commands=[
            "executor/disaggExecutorTest",
            "--gtest_filter=*DisaggSymmetricExecutorTest*"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"])
    _cpp.run_command(trt_model_test, cwd=tests_dir, env=new_env, timeout=1500)

    mgpu_env = copy.copy(cpp_env)
    mgpu_env["RUN_LLAMA_MULTI_GPU"] = "true"
    mgpu_env["TRTLLM_USE_MPI_KVCACHE"] = "1"
    xml_output_file = build_dir / "results-multi-gpu-disagg-executor-4-process.xml"
    trt_model_test = _cpp.produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=4,
        local_commands=[
            "executor/disaggExecutorTest",
            "--gtest_filter=*DisaggSymmetricExecutorTest*"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"])
    # https://nvbugspro.nvidia.com/bug/5026255 disable below tests for now.
    _cpp.run_command(trt_model_test, cwd=tests_dir, env=mgpu_env, timeout=1500)

    xml_output_file = build_dir / "results-multi-gpu-disagg-executor-8-process.xml"
    trt_model_test = _cpp.produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=8,
        local_commands=[
            "executor/disaggExecutorTest",
            "--gtest_filter=*LlamaTP2PP2DisaggSymmetricExecutorTest*"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"])
    _cpp.run_command(trt_model_test, cwd=tests_dir, env=mgpu_env, timeout=1500)

    xml_output_file = build_dir / "results-multi-gpu-disagg-asymmetric-executor-4-process.xml"
    trt_model_test = _cpp.produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=4,
        local_commands=[
            "executor/disaggExecutorTest",
            "--gtest_filter=*DisaggAsymmetricExecutorTest*"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"])
    _cpp.run_command(trt_model_test, cwd=tests_dir, env=mgpu_env, timeout=1500)

    xml_output_file = build_dir / "results-multi-gpu-disagg-asymmetric-executor-6-process.xml"
    trt_model_test = _cpp.produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=6,
        local_commands=[
            "executor/disaggExecutorTest",
            "--gtest_filter=*DisaggAsymmetricExecutorTest*"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"])
    _cpp.run_command(trt_model_test, cwd=tests_dir, env=mgpu_env, timeout=1500)

    xml_output_file = build_dir / "results-multi-gpu-disagg-asymmetric-executor-8-process.xml"
    trt_model_test = _cpp.produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=8,
        local_commands=[
            "executor/disaggExecutorTest",
            "--gtest_filter=*DisaggAsymmetricExecutorTest*"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"])
    _cpp.run_command(trt_model_test, cwd=tests_dir, env=mgpu_env, timeout=1500)

    xml_output_file = build_dir / "results-multi-gpu-disagg-asymmetric-orchestrator-executor-7-process.xml"
    trt_model_test = _cpp.produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=7,
        local_commands=[
            "executor/disaggExecutorTest",
            "--gtest_filter=*DisaggOrchestratorParamsTest*"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"])
    _cpp.run_command(trt_model_test, cwd=tests_dir, env=mgpu_env, timeout=1500)

    # UCX transceiver tests, the test may not be built if ENABLE_UCX is 0
    new_env = copy.copy(cpp_env)
    new_env["TRTLLM_USE_UCX_KVCACHE"] = "1"
    xml_output_file = build_dir / "results-multi-gpu-disagg-executor-2-process.xml"
    trt_model_test = _cpp.produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=2,
        local_commands=[
            "executor/disaggExecutorTest",
            "--gtest_filter=*DisaggSymmetricExecutorTest*"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"])
    _cpp.run_command(trt_model_test, cwd=tests_dir, env=new_env, timeout=1500)

    mgpu_env = copy.copy(cpp_env)
    mgpu_env["RUN_LLAMA_MULTI_GPU"] = "true"
    mgpu_env["TRTLLM_USE_UCX_KVCACHE"] = "1"
    xml_output_file = build_dir / "results-multi-gpu-disagg-executor-4-process.xml"
    trt_model_test = _cpp.produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=4,
        local_commands=[
            "executor/disaggExecutorTest",
            "--gtest_filter=*DisaggSymmetricExecutorTest*"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"])
    # https://nvbugspro.nvidia.com/bug/5026255 disable below tests for now.
    _cpp.run_command(trt_model_test, cwd=tests_dir, env=mgpu_env, timeout=1500)

    xml_output_file = build_dir / "results-multi-gpu-disagg-executor-8-process.xml"
    trt_model_test = _cpp.produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=8,
        local_commands=[
            "executor/disaggExecutorTest",
            "--gtest_filter=*LlamaTP2PP2DisaggSymmetricExecutorTest*"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"])
    _cpp.run_command(trt_model_test, cwd=tests_dir, env=mgpu_env, timeout=1500)

    xml_output_file = build_dir / "results-multi-gpu-disagg-asymmetric-executor-4-process.xml"
    trt_model_test = _cpp.produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=4,
        local_commands=[
            "executor/disaggExecutorTest",
            "--gtest_filter=*DisaggAsymmetricExecutorTest*"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"])
    _cpp.run_command(trt_model_test, cwd=tests_dir, env=mgpu_env, timeout=1500)

    xml_output_file = build_dir / "results-multi-gpu-disagg-asymmetric-executor-6-process.xml"
    trt_model_test = _cpp.produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=6,
        local_commands=[
            "executor/disaggExecutorTest",
            "--gtest_filter=*DisaggAsymmetricExecutorTest*"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"])
    _cpp.run_command(trt_model_test, cwd=tests_dir, env=mgpu_env, timeout=1500)

    xml_output_file = build_dir / "results-multi-gpu-disagg-asymmetric-executor-8-process.xml"
    trt_model_test = _cpp.produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=8,
        local_commands=[
            "executor/disaggExecutorTest",
            "--gtest_filter=*DisaggAsymmetricExecutorTest*"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"])
    _cpp.run_command(trt_model_test, cwd=tests_dir, env=mgpu_env, timeout=1500)

    xml_output_file = build_dir / "results-multi-gpu-disagg-asymmetric-orchestrator-executor-7-process.xml"
    trt_model_test = _cpp.produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=7,
        local_commands=[
            "executor/disaggExecutorTest",
            "--gtest_filter=*DisaggOrchestratorParamsTest*"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"])
    _cpp.run_command(trt_model_test, cwd=tests_dir, env=mgpu_env, timeout=1500)

    xml_output_file = build_dir / "results-multi-gpu-disagg-spawn-asymmetric-orchestrator-executor-1-process.xml"
    comms = [
        "executor/disaggExecutorTest",
        "--gtest_filter=*DisaaggSpawnOrchestrator*"
    ]
    _cpp.run_command(comms, cwd=tests_dir, env=mgpu_env, timeout=1500)


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


@pytest.mark.parametrize("build_google_tests", ["80", "86", "89", "90"],
                         indirect=True)
def test_simple(build_google_tests, build_dir):

    if platform.system() != "Windows":

        run_simple_multi_gpu_tests(build_dir=build_dir,
                                   timeout=_cpp.default_test_timeout)


@pytest.mark.parametrize("build_google_tests", ["80", "86", "89", "90"],
                         indirect=True)
def test_t5(build_google_tests, prepare_model_multi_gpu, build_dir):

    if platform.system() != "Windows":
        prepare_model_multi_gpu("t5")
        run_t5_multi_gpu_tests(build_dir=build_dir,
                               timeout=_cpp.default_test_timeout)


@pytest.mark.parametrize("build_google_tests", ["80", "86", "89", "90"],
                         indirect=True)
def test_llama_executor(build_google_tests, prepare_model_multi_gpu, lora_setup,
                        build_dir):

    if platform.system() != "Windows":
        prepare_model_multi_gpu("llama")
        run_llama_executor_multi_gpu_tests(build_dir=build_dir,
                                           timeout=_cpp.default_test_timeout)


@pytest.mark.parametrize("build_google_tests", ["80", "86", "89", "90"],
                         indirect=True)
def test_trt_gpt_real_decoder(build_google_tests, prepare_model_multi_gpu,
                              lora_setup, build_dir):

    if platform.system() != "Windows":
        prepare_model_multi_gpu("llama")
        run_trt_gpt_model_real_decoder_multi_gpu_tests(
            build_dir=build_dir, timeout=_cpp.default_test_timeout)


@pytest.mark.parametrize("build_google_tests", ["80", "86", "89", "90"],
                         indirect=True)
def test_disagg(prepare_model, prepare_model_multi_gpu, build_google_tests,
                build_dir):

    if platform.system() != "Windows":
        # Disagg tests need single + multi GPU llama models.
        prepare_model("llama")
        prepare_model_multi_gpu("llama")

        prepare_model("gpt")

        run_disagg_multi_gpu_tests(build_dir=build_dir)
