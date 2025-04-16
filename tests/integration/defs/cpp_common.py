import copy
import datetime
import glob
import logging as _logger
import os as _os
import pathlib as _pl
import subprocess as _sp
import sys as _sys
from typing import Generator, List, Optional, Sequence

build_script_dir = _pl.Path(
    __file__).parent.resolve().parent.parent.parent / "scripts"
assert build_script_dir.is_dir()
_sys.path.append(str(build_script_dir))

from build_wheel import get_build_dir as get_trt_llm_build_dir

default_test_parallel = 2
default_test_timeout = 3600

include_test_map = {
    "gpt": ("Gpt[^j]", ),
    "gpt_executor": ("GptExecutor", ),
    "gpt_session": ("GptSession", ),
    "gpt_tests": ("GptTests", ),
    "gptj": ("Gptj", ),
    "llama": ("Llama", ),
    "chatglm": ("ChatGlm", ),
    "medusa": ("Medusa", ),
    "eagle": ("Eagle", ),
    "mamba": ("Mamba", ),
    "recurrentgemma": ("RecurrentGemma", ),
    "encoder": ("EncoderModelTestSingleGPU", ),
    "bart": ("BartBasicTest", ),
    "t5": (
        "T5BasicTest",
        "T5Beam2Test",
    ),
    "enc_dec_language_adapter": ("LanguageAdapterBasicTest", ),
    "redrafter": ("ExplicitDraftTokens", )
}


def generate_excluded_model_tests() -> Generator[str, None, None]:
    yield "Gpt[^j]"
    yield "GptExecutor"
    yield "Gptj"
    yield "Llama"
    yield "ChatGlm"
    yield "Medusa"
    yield "Eagle"
    yield "ExplicitDraftTokensDecoding"
    yield "Mamba"
    yield "RecurrentGemma"
    yield "Encoder"
    yield "EncDec"
    yield "SpeculativeDecoding"


def generate_included_model_tests(
        test_list: List[str]) -> Generator[str, None, None]:

    yield from (item for model in test_list for item in include_test_map[model])


def generate_result_file_name(test_list: List[str],
                              run_fp8=False) -> Generator[str, None, None]:
    yield "results-single-gpu"
    yield from test_list

    if run_fp8:
        yield "fp8"


def generate_excluded_test_list(test_list):
    if "gpt" in test_list:
        if "gpt_session" not in test_list:
            yield "GptSession"
        if "gpt_executor" not in test_list:
            yield "GptExecutor"
        if "gpt_tests" not in test_list:
            yield "GptTests"


def find_dir_containing(files: Sequence[str],
                        start_dir: Optional[_pl.Path] = None) -> _pl.Path:
    if start_dir is None:
        start_dir = _pl.Path.cwd().absolute()

    assert isinstance(start_dir, _pl.Path)
    assert start_dir.is_dir()

    if set(files).issubset({f.name for f in start_dir.iterdir()}):
        return start_dir
    elif start_dir.parent is not start_dir:
        return find_dir_containing(files, start_dir.parent)
    else:
        raise FileNotFoundError(files)


def find_root_dir(start_dir: Optional[_pl.Path] = None) -> _pl.Path:
    return find_dir_containing(("scripts", "examples", "cpp"), start_dir)


def find_build_dir():
    root_dir = find_root_dir()
    dir = get_trt_llm_build_dir(None, "Release")

    return dir if dir.is_absolute() else root_dir / dir


def run_command(command: Sequence[str],
                cwd: _pl.Path,
                *,
                shell=False,
                env=None,
                timeout=None) -> None:
    _logger.info("Running: cd %s && %s", str(cwd), " ".join(command))
    override_timeout = int(_os.environ.get("CPP_TEST_TIMEOUT_OVERRIDDEN", "-1"))
    if override_timeout > 0 and (timeout is None or override_timeout > timeout):
        _logger.info(
            "Overriding the command timeout: %s (before) and %s (after)",
            timeout, override_timeout)
        timeout = override_timeout
    _sp.check_call(command, cwd=cwd, shell=shell, env=env, timeout=timeout)


def merge_report(parallel, retry, output):
    import xml.etree.ElementTree as ElementTree
    base = ElementTree.parse(parallel)
    extra = ElementTree.parse(retry)

    base_suite = base.getroot()
    extra_suite = extra.getroot()

    base_suite.attrib['failures'] = extra_suite.attrib['failures']
    base_suite.attrib['time'] = str(
        int(base_suite.attrib['time']) + int(extra_suite.attrib['time']))

    case_names = {element.attrib['name'] for element in extra_suite}
    base_suite[:] = [
        element
        for element in base_suite if element.attrib['name'] not in case_names
    ] + list(extra_suite)

    base.write(output, encoding="UTF-8", xml_declaration=True)


def add_parallel_info(report, parallel):
    import xml.etree.ElementTree as ElementTree
    try:
        document = ElementTree.parse(report)
    except FileNotFoundError:
        return
    root = document.getroot()
    root.attrib['parallel'] = str(parallel)
    document.write(report, encoding="UTF-8", xml_declaration=True)


def run_ctest(command: Sequence[str],
              cwd: _pl.Path,
              *,
              shell=False,
              env=None,
              timeout=None) -> None:
    override_timeout = int(_os.environ.get("CPP_TEST_TIMEOUT_OVERRIDDEN", "-1"))
    if override_timeout > 0 and (timeout is None or override_timeout > timeout):
        _logger.info(
            "Overriding the command timeout: %s (before) and %s (after)",
            timeout, override_timeout)
        timeout = override_timeout
    deadline = None
    if timeout is not None:
        deadline = datetime.datetime.now() + datetime.timedelta(seconds=timeout)
        command = list(command)
        command += ["--stop-time", deadline.strftime("%H:%M:%S")]
    try:
        _logger.info("Running: cd %s && %s", str(cwd), " ".join(command))
        _sp.check_call(command, cwd=cwd, shell=shell, env=env)
    except _sp.CalledProcessError as e:
        fuzz = datetime.timedelta(seconds=2)
        if deadline is not None and deadline - fuzz < datetime.datetime.now():
            # Detect timeout
            raise _sp.TimeoutExpired(e.cmd, timeout, e.output, e.stderr)
        raise


def parallel_run_ctest(
    command: Sequence[str],
    cwd: _pl.Path,
    *,
    shell=False,
    env=None,
    timeout=None,
    parallel=default_test_parallel,
) -> None:
    if parallel == 1:
        return run_ctest(command,
                         cwd=cwd,
                         shell=shell,
                         env=env,
                         timeout=timeout)

    env = {} if env is None else env
    env['CTEST_PARALLEL_LEVEL'] = str(parallel)

    def get_report():
        reports = glob.glob("results-*.xml", root_dir=cwd)
        if not reports:
            return ''

        return reports[0]

    report = None
    try:
        run_ctest(command, cwd=cwd, shell=shell, env=env, timeout=timeout)
    # except _sp.TimeoutExpired:
    # Deliberately let timeout propagate. We don't want to retry on timeout
    except _sp.CalledProcessError:
        report = get_report()
        if report == '':
            # Some catastrophic fail happened that there's no report generated
            raise

        # Avoid .xml extension to prevent CI from reading failures from it
        parallel_report = 'parallel-' + report + ".intermediate"
        _os.rename(cwd / report, cwd / parallel_report)

        try:
            _logger.info("Parallel test failed, retry serial on failed tests")
            del env['CTEST_PARALLEL_LEVEL']
            command = [*command, "--rerun-failed"]
            run_ctest(command, cwd=cwd, shell=shell, env=env, timeout=timeout)
        finally:
            if not _os.path.exists(cwd / report):
                # Some catastrophic fail happened that there's no report generated
                # Use parallel result as final report
                _os.rename(cwd / parallel_report, cwd / report)
            else:
                retry_report = 'retry-' + report + ".intermediate"
                _os.rename(cwd / report, cwd / retry_report)
                merge_report(cwd / parallel_report, cwd / retry_report,
                             cwd / report)
    finally:
        if report is None:
            report = get_report()
        if report:
            add_parallel_info(cwd / report, parallel)


def produce_mpirun_command(*, global_commands, nranks, local_commands,
                           leader_commands):
    l = global_commands
    for rank in range(nranks):
        l += ["-n", "1"] + local_commands + (leader_commands
                                             if rank == 0 else []) + [":"]
    return l[:-1]


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
    run_command(mpi_utils_test, cwd=tests_dir, env=cpp_env, timeout=300)

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
    run_command(cache_trans_test, cwd=tests_dir, env=new_env, timeout=300)

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
    run_command(cache_trans_test_8_proc,
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
    run_command(cache_trans_test, cwd=tests_dir, env=new_env, timeout=300)

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
    run_command(cache_trans_test_8_proc,
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
    trt_model_test = produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=4,
        local_commands=[
            "executor/executorTest",
            "--gtest_filter=*LlamaExecutorTest*LeaderMode*:*LlamaMultiExecutorTest*LeaderMode*"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"])
    # https://nvbugspro.nvidia.com/bug/5026255 disable below tests for now.
    if False:
        run_command(trt_model_test, cwd=tests_dir, env=mgpu_env, timeout=1500)

    #Executor test in orchestrator mode
    xml_output_file = build_dir / "results-multi-gpu-llama-exec-orch-mode.xml"
    trt_model_test = [
        "mpirun", "-n", "1", "--allow-run-as-root", "executor/executorTest",
        "--gtest_filter=*LlamaExecutorTest*OrchMode*",
        f"--gtest_output=xml:{xml_output_file}"
    ]
    run_command(trt_model_test, cwd=tests_dir, env=mgpu_env, timeout=1500)

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
    trt_model_test = produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=4,
        local_commands=[
            "executor/executorTest", f"--gtest_filter={gtest_filter}"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"])
    run_command(trt_model_test, cwd=tests_dir, env=mgpu_env, timeout=1500)


def run_t5_multi_gpu_tests(build_dir: _pl.Path, timeout=1500):
    tests_dir = build_dir / "tests"
    cpp_env = {**_os.environ}

    #EncDec test in leader mode
    xml_output_file = build_dir / "results-multi-gpu-t5-exec-leader-mode.xml"
    trt_model_test = produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=4,
        local_commands=[
            "executor/encDecTest",
            "--gtest_filter=T5MultiGPUTest/EncDecParamsTest.Forward*"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"],
    )
    run_command(trt_model_test, cwd=tests_dir, env=cpp_env, timeout=1500)


def run_trt_gpt_model_real_decoder_multi_gpu_tests(build_dir: _pl.Path,
                                                   timeout=1500):
    tests_dir = build_dir / "tests"
    cpp_env = {**_os.environ}

    xml_output_file = build_dir / "results-multi-gpu-real-decoder.xml"
    trt_model_test = produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=4,
        local_commands=[
            "batch_manager/trtGptModelRealDecoderTest",
            "--gtest_filter=*TP*:*PP*"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"])
    run_command(trt_model_test, cwd=tests_dir, env=cpp_env,
                timeout=timeout)  # expecting ~ 1200s


def run_disagg_multi_gpu_tests(build_dir: _pl.Path):

    tests_dir = build_dir / "tests"
    cpp_env = {**_os.environ}

    new_env = copy.copy(cpp_env)
    new_env["TRTLLM_USE_MPI_KVCACHE"] = "1"
    xml_output_file = build_dir / "results-multi-gpu-disagg-executor-2-process.xml"
    trt_model_test = produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=2,
        local_commands=[
            "executor/disaggExecutorTest",
            "--gtest_filter=*DisaggSymmetricExecutorTest*"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"])
    run_command(trt_model_test, cwd=tests_dir, env=new_env, timeout=1500)

    mgpu_env = copy.copy(cpp_env)
    mgpu_env["RUN_LLAMA_MULTI_GPU"] = "true"
    mgpu_env["TRTLLM_USE_MPI_KVCACHE"] = "1"
    xml_output_file = build_dir / "results-multi-gpu-disagg-executor-4-process.xml"
    trt_model_test = produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=4,
        local_commands=[
            "executor/disaggExecutorTest",
            "--gtest_filter=*DisaggSymmetricExecutorTest*"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"])
    # https://nvbugspro.nvidia.com/bug/5026255 disable below tests for now.
    run_command(trt_model_test, cwd=tests_dir, env=mgpu_env, timeout=1500)

    xml_output_file = build_dir / "results-multi-gpu-disagg-executor-8-process.xml"
    trt_model_test = produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=8,
        local_commands=[
            "executor/disaggExecutorTest",
            "--gtest_filter=*LlamaTP2PP2DisaggSymmetricExecutorTest*"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"])
    run_command(trt_model_test, cwd=tests_dir, env=mgpu_env, timeout=1500)

    xml_output_file = build_dir / "results-multi-gpu-disagg-asymmetric-executor-4-process.xml"
    trt_model_test = produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=4,
        local_commands=[
            "executor/disaggExecutorTest",
            "--gtest_filter=*DisaggAsymmetricExecutorTest*"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"])
    run_command(trt_model_test, cwd=tests_dir, env=mgpu_env, timeout=1500)

    xml_output_file = build_dir / "results-multi-gpu-disagg-asymmetric-executor-6-process.xml"
    trt_model_test = produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=6,
        local_commands=[
            "executor/disaggExecutorTest",
            "--gtest_filter=*DisaggAsymmetricExecutorTest*"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"])
    run_command(trt_model_test, cwd=tests_dir, env=mgpu_env, timeout=1500)

    xml_output_file = build_dir / "results-multi-gpu-disagg-asymmetric-executor-8-process.xml"
    trt_model_test = produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=8,
        local_commands=[
            "executor/disaggExecutorTest",
            "--gtest_filter=*DisaggAsymmetricExecutorTest*"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"])
    run_command(trt_model_test, cwd=tests_dir, env=mgpu_env, timeout=1500)

    xml_output_file = build_dir / "results-multi-gpu-disagg-asymmetric-orchestrator-executor-7-process.xml"
    trt_model_test = produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=7,
        local_commands=[
            "executor/disaggExecutorTest",
            "--gtest_filter=*DisaggOrchestratorParamsTest*"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"])
    run_command(trt_model_test, cwd=tests_dir, env=mgpu_env, timeout=1500)

    # UCX transceiver tests, the test may not be built if ENABLE_UCX is 0
    new_env = copy.copy(cpp_env)
    new_env["TRTLLM_USE_UCX_KVCACHE"] = "1"
    xml_output_file = build_dir / "results-multi-gpu-disagg-executor-2-process.xml"
    trt_model_test = produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=2,
        local_commands=[
            "executor/disaggExecutorTest",
            "--gtest_filter=*DisaggSymmetricExecutorTest*"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"])
    run_command(trt_model_test, cwd=tests_dir, env=new_env, timeout=1500)

    mgpu_env = copy.copy(cpp_env)
    mgpu_env["RUN_LLAMA_MULTI_GPU"] = "true"
    mgpu_env["TRTLLM_USE_UCX_KVCACHE"] = "1"
    xml_output_file = build_dir / "results-multi-gpu-disagg-executor-4-process.xml"
    trt_model_test = produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=4,
        local_commands=[
            "executor/disaggExecutorTest",
            "--gtest_filter=*DisaggSymmetricExecutorTest*"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"])
    # https://nvbugspro.nvidia.com/bug/5026255 disable below tests for now.
    run_command(trt_model_test, cwd=tests_dir, env=mgpu_env, timeout=1500)

    xml_output_file = build_dir / "results-multi-gpu-disagg-executor-8-process.xml"
    trt_model_test = produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=8,
        local_commands=[
            "executor/disaggExecutorTest",
            "--gtest_filter=*LlamaTP2PP2DisaggSymmetricExecutorTest*"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"])
    run_command(trt_model_test, cwd=tests_dir, env=mgpu_env, timeout=1500)

    xml_output_file = build_dir / "results-multi-gpu-disagg-asymmetric-executor-4-process.xml"
    trt_model_test = produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=4,
        local_commands=[
            "executor/disaggExecutorTest",
            "--gtest_filter=*DisaggAsymmetricExecutorTest*"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"])
    run_command(trt_model_test, cwd=tests_dir, env=mgpu_env, timeout=1500)

    xml_output_file = build_dir / "results-multi-gpu-disagg-asymmetric-executor-6-process.xml"
    trt_model_test = produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=6,
        local_commands=[
            "executor/disaggExecutorTest",
            "--gtest_filter=*DisaggAsymmetricExecutorTest*"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"])
    run_command(trt_model_test, cwd=tests_dir, env=mgpu_env, timeout=1500)

    xml_output_file = build_dir / "results-multi-gpu-disagg-asymmetric-executor-8-process.xml"
    trt_model_test = produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=8,
        local_commands=[
            "executor/disaggExecutorTest",
            "--gtest_filter=*DisaggAsymmetricExecutorTest*"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"])
    run_command(trt_model_test, cwd=tests_dir, env=mgpu_env, timeout=1500)

    xml_output_file = build_dir / "results-multi-gpu-disagg-asymmetric-orchestrator-executor-7-process.xml"
    trt_model_test = produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=7,
        local_commands=[
            "executor/disaggExecutorTest",
            "--gtest_filter=*DisaggOrchestratorParamsTest*"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"])
    run_command(trt_model_test, cwd=tests_dir, env=mgpu_env, timeout=1500)

    xml_output_file = build_dir / "results-multi-gpu-disagg-spawn-asymmetric-orchestrator-executor-1-process.xml"
    comms = [
        "executor/disaggExecutorTest",
        "--gtest_filter=*DisaaggSpawnOrchestrator*"
    ]
    run_command(comms, cwd=tests_dir, env=mgpu_env, timeout=1500)


def run_spec_dec_tests(build_dir: _pl.Path):
    xml_output_file = build_dir / "results-spec-dec-fast-logits.xml"
    cpp_env = {**_os.environ}
    tests_dir = build_dir / "tests"
    trt_model_test = produce_mpirun_command(
        global_commands=["mpirun", "--allow-run-as-root"],
        nranks=3,
        local_commands=[
            "executor/executorTest", "--gtest_filter=*SpecDecFastLogits*"
        ],
        leader_commands=[f"--gtest_output=xml:{xml_output_file}"])
    run_command(trt_model_test, cwd=tests_dir, env=cpp_env, timeout=1500)


def prepare_model_tests(model_name: str,
                        python_exe: str,
                        root_dir: _pl.Path,
                        resources_dir: _pl.Path,
                        model_cache_arg=[],
                        only_fp8_arg=[],
                        only_multi_gpu_arg=[]):
    scripts_dir = resources_dir / "scripts"

    model_env = {**_os.environ, "PYTHONPATH": f"examples/{model_name}"}
    enc_dec_model_name_arg = []
    beams_arg = []
    if model_name in ('bart', 't5', 'enc_dec_language_adapter'):
        enc_dec_repo_name_dict = {
            'bart': 'facebook/bart-large-cnn',
            't5': 't5-small',
            'enc_dec_language_adapter':
            'language_adapter-enc_dec_language_adapter'
        }
        enc_dec_model_name_arg = [
            '--hf_repo_name', enc_dec_repo_name_dict[model_name]
        ]
        if model_name == 't5' and (not only_multi_gpu_arg):
            beams_arg = ['--beams', '1,2']
        model_name = 'enc_dec'

    # share the same script for gpt related tests
    if model_name == 'gpt_executor' or model_name == 'gpt_session' or model_name == 'gpt_tests':
        model_name = 'gpt'

    build_engines = [
        python_exe,
        str(scripts_dir / f"build_{model_name}_engines.py")
    ] + model_cache_arg + only_fp8_arg + only_multi_gpu_arg + enc_dec_model_name_arg + beams_arg

    if model_name in ['gpt']:
        build_engines += ['--clean']
    run_command(build_engines, cwd=root_dir, env=model_env, timeout=1800)

    model_env["PYTHONPATH"] = "examples"
    generate_expected_output = [
        python_exe,
        str(scripts_dir / f"generate_expected_{model_name}_output.py")
    ] + only_fp8_arg + only_multi_gpu_arg + enc_dec_model_name_arg
    if "enc_dec" in model_name:
        generate_expected_output += model_cache_arg
        generate_expected_output += beams_arg

    if model_name in ['gpt']:
        generate_expected_output += ['--clean']

    if only_multi_gpu_arg and model_name != 'enc_dec':
        for world_size in (2, 4):
            generate_command = [
                "mpirun", "-n",
                str(world_size), "--allow-run-as-root", "--timeout", "600"
            ] + generate_expected_output
            run_command(generate_command,
                        cwd=root_dir,
                        env=model_env,
                        timeout=600)
    else:
        run_command(generate_expected_output,
                    cwd=root_dir,
                    env=model_env,
                    timeout=600)

    if model_name in ['gpt', 'llama']:
        if model_name == 'gpt':
            script_model_name = 'gpt2'
        elif model_name == 'llama':
            script_model_name = 'llama-7b-hf'
        generate_tokenizer_info = [
            python_exe, "examples/generate_xgrammar_tokenizer_info.py",
            f"--model_dir={str(resources_dir / 'models' / script_model_name)}",
            f"--output_dir={str(resources_dir / 'data' / script_model_name)}"
        ]
        run_command(generate_tokenizer_info,
                    cwd=root_dir,
                    env=model_env,
                    timeout=600)


def prepare_multi_gpu_model_tests(test_list: List[str],
                                  python_exe: str,
                                  root_dir: _pl.Path,
                                  resources_dir: _pl.Path,
                                  model_cache: Optional[str] = None):
    model_cache_arg = ["--model_cache", model_cache] if model_cache else []

    if "llama" in test_list:
        prepare_model_tests(model_name="llama",
                            python_exe=python_exe,
                            root_dir=root_dir,
                            resources_dir=resources_dir,
                            model_cache_arg=model_cache_arg,
                            only_multi_gpu_arg=["--only_multi_gpu"])

    if "t5" in test_list:
        prepare_model_tests(model_name="t5",
                            python_exe=python_exe,
                            root_dir=root_dir,
                            resources_dir=resources_dir,
                            model_cache_arg=model_cache_arg,
                            only_multi_gpu_arg=['--tp', '4', '--pp', '1'])


def run_single_gpu_tests(build_dir: _pl.Path,
                         test_list: List[str],
                         run_fp8=False,
                         timeout=3600):

    cpp_env = {**_os.environ}

    included_tests = list(generate_included_model_tests(test_list))

    fname_list = list(generate_result_file_name(test_list, run_fp8=run_fp8))
    resultFileName = "-".join(fname_list) + ".xml"

    excluded_tests = ["FP8"] if not run_fp8 else []

    excluded_tests.extend(list(generate_excluded_test_list(test_list)))

    ctest = ["ctest", "--output-on-failure", "--output-junit", resultFileName]

    if included_tests:
        ctest.extend(["-R", "|".join(included_tests)])
        if excluded_tests:
            ctest.extend(["-E", "|".join(excluded_tests)])

        gpt_tests = {"gpt", "gpt_session", "gpt_tests", "gpt_executor"}

        # gpt* tests are not parallelized as it would cause OOM because kv cache memory allocations
        # exist in multiple running tests
        if gpt_tests.intersection(test_list):
            parallel = 1
        else:
            parallel = default_test_parallel

        if parallel_override := _os.environ.get("LLM_TEST_PARALLEL_OVERRIDE",
                                                None):
            parallel = int(parallel_override)

        parallel_run_ctest(ctest,
                           cwd=build_dir,
                           env=cpp_env,
                           timeout=timeout,
                           parallel=parallel)
    if "gpt" in test_list:
        xml_output_file = build_dir / "results-single-gpu-disagg-executor_gpt.xml"
        new_env = copy.copy(cpp_env)
        new_env["TRTLLM_USE_MPI_KVCACHE"] = "1"
        trt_model_test = produce_mpirun_command(
            global_commands=["mpirun", "--allow-run-as-root"],
            nranks=2,
            local_commands=[
                "tests/executor/disaggExecutorTest",
                "--gtest_filter=*GptSingleDeviceDisaggSymmetricExecutorTest*"
            ],
            leader_commands=[f"--gtest_output=xml:{xml_output_file}"])
        run_command(trt_model_test, cwd=build_dir, env=new_env, timeout=timeout)

        run_spec_dec_tests(build_dir=build_dir)


def run_benchmarks(model_name: str, python_exe: str, root_dir: _pl.Path,
                   build_dir: _pl.Path, resources_dir: _pl.Path,
                   model_cache: str, test_gpt_session_benchmark: bool,
                   batching_types: list[str], api_types: list[str]):

    benchmark_exe_dir = build_dir / "benchmarks"
    if model_name == "gpt":
        model_engine_dir = resources_dir / "models" / "rt_engine" / "gpt2"
        tokenizer_dir = resources_dir / "models" / "gpt2"
    elif model_name in ('bart', 't5'):
        if model_name == "t5":
            hf_repo_name = "t5-small"
        elif model_name == "bart":
            hf_repo_name = "bart-large-cnn"
        model_engine_dir = resources_dir / "models" / "enc_dec" / "trt_engines" / hf_repo_name
        tokenizer_dir = model_cache + "/" + hf_repo_name
        model_engine_path = model_engine_dir / "1-gpu" / "float16" / "decoder"
        encoder_model_engine_path = model_engine_dir / "1-gpu" / "float16" / "encoder"
        model_name = "enc_dec"
    else:
        _logger.info(
            f"run_benchmark test does not support {model_name}. Skipping benchmarks"
        )
        return NotImplementedError

    if test_gpt_session_benchmark:
        if model_name == "gpt":
            pass

            # WAR: Currently importing the bindings here causes a segfault in pybind 11 during shutdown
            # As this just builds a path we hard-code for now to obviate the need for import of bindings

            # model_spec_obj = model_spec.ModelSpec(input_file, _tb.DataType.HALF)
            # model_spec_obj.set_kv_cache_type(_tb.KVCacheType.CONTINUOUS)
            # model_spec_obj.use_gpt_plugin()
            # model_engine_path = model_engine_dir / model_spec_obj.get_model_path(
            # ) / "tp1-pp1-cp1-gpu"

            model_engine_path = model_engine_dir / "fp16_plugin_continuous" / "tp1-pp1-cp1-gpu"
        else:
            _logger.info(
                f"gptSessionBenchmark test does not support {model_name}. Skipping benchmarks"
            )
            return NotImplementedError

        benchmark = [
            str(benchmark_exe_dir / "gptSessionBenchmark"), "--engine_dir",
            str(model_engine_path), "--batch_size", "8", "--input_output_len",
            "10,20", "--duration", "10"
        ]
        run_command(benchmark, cwd=root_dir, timeout=600)

    prompt_datasets_args = [{
        '--dataset-name': "cnn_dailymail",
        '--dataset-config-name': "3.0.0",
        '--dataset-split': "validation",
        '--dataset-input-key': "article",
        '--dataset-prompt': "Summarize the following article:",
        '--dataset-output-key': "highlights"
    }, {
        '--dataset-name': "Open-Orca/1million-gpt-4",
        '--dataset-split': "train",
        '--dataset-input-key': "question",
        '--dataset-prompt-key': "system_prompt",
        '--dataset-output-key': "response"
    }]
    token_files = [
        "prepared_" + s['--dataset-name'].replace('/', '_')
        for s in prompt_datasets_args
    ]
    max_input_lens = ["256", "20"]
    num_reqs = ["50", "10"]

    if model_name == "gpt":
        model_engine_path = model_engine_dir / "fp16_plugin_packed_paged" / "tp1-pp1-cp1-gpu"

        # WAR: Currently importing the bindings here causes a segfault in pybind 11 during shutdown
        # As this just builds a path we hard-code for now to obviate the need for import of bindings

        # model_spec_obj = model_spec.ModelSpec(input_file, _tb.DataType.HALF)
        # model_spec_obj.set_kv_cache_type(_tb.KVCacheType.PAGED)
        # model_spec_obj.use_gpt_plugin()
        # model_spec_obj.use_packed_input()
        # model_engine_path = model_engine_dir / model_spec_obj.get_model_path(
        # ) / "tp1-pp1-cp1-gpu"

    for prompt_ds_args, tokens_f, len, num_req in zip(prompt_datasets_args,
                                                      token_files,
                                                      max_input_lens, num_reqs):

        benchmark_src_dir = _pl.Path("benchmarks") / "cpp"
        data_dir = resources_dir / "data"
        prepare_dataset = [
            python_exe,
            str(benchmark_src_dir / "prepare_dataset.py"), "--tokenizer",
            str(tokenizer_dir), "--output",
            str(data_dir / tokens_f), "dataset", "--max-input-len", len,
            "--num-requests", num_req
        ]
        for k, v in prompt_ds_args.items():
            prepare_dataset += [k, v]
        # https://nvbugs/4658787
        # WAR before the prepare dataset can use offline cached dataset
        run_command(prepare_dataset,
                    cwd=root_dir,
                    timeout=300,
                    env={'HF_DATASETS_OFFLINE': '0'})

        for batching_type in batching_types:
            for api_type in api_types:
                benchmark = [
                    str(benchmark_exe_dir / "gptManagerBenchmark"),
                    "--engine_dir",
                    str(model_engine_path), "--type",
                    str(batching_type), "--api",
                    str(api_type), "--dataset",
                    str(data_dir / tokens_f)
                ]
                if model_name == "enc_dec":
                    benchmark += [
                        "--encoder_engine_dir",
                        str(encoder_model_engine_path)
                    ]

                run_command(benchmark, cwd=root_dir, timeout=600)
                req_rate_benchmark = benchmark + [
                    "--request_rate", "100", "--enable_exp_delays"
                ]
                run_command(req_rate_benchmark, cwd=root_dir, timeout=600)
                concurrency_benchmark = benchmark + ["--concurrency", "30"]
                run_command(concurrency_benchmark, cwd=root_dir, timeout=600)

        if "IFB" in batching_type and "executor" in api_types:
            # executor streaming test
            benchmark = [
                str(benchmark_exe_dir / "gptManagerBenchmark"), "--engine_dir",
                str(model_engine_path), "--type", "IFB", "--dataset",
                str(data_dir / tokens_f), "--api", "executor", "--streaming"
            ]
            if model_name == "enc_dec":
                benchmark += [
                    "--encoder_engine_dir",
                    str(encoder_model_engine_path)
                ]
            run_command(benchmark, cwd=root_dir, timeout=600)
