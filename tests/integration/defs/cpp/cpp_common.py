import datetime
import glob
import logging as _logger
import os as _os
import pathlib as _pl
import subprocess as _sp
import sys as _sys
from typing import Generator, List, Optional, Sequence

build_script_dir = _pl.Path(
    __file__).parent.resolve().parent.parent.parent.parent / "scripts"
assert build_script_dir.is_dir()
_sys.path.append(str(build_script_dir))

from build_wheel import get_build_dir as get_trt_llm_build_dir

default_test_parallel = 2
default_test_timeout = 3600

include_test_map = {
    "gpt": ("Gpt[^j]", ),
    "gpt_executor": ("GptExecutor", ),
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


def find_build_dir(build_type: str) -> _pl.Path:
    """Resolve the TRT-LLM C++ build directory for the given CMake build type.

    Args:
        build_type: CMake build type (e.g., "Release", "RelWithDebInfo", "Debug").

    Returns:
        Absolute path to the C++ build directory.
    """
    root_dir = find_root_dir()
    build_dir = get_trt_llm_build_dir(None, build_type)
    return build_dir if build_dir.is_absolute() else root_dir / build_dir


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
    if model_name == 'gpt_executor' or model_name == 'gpt_tests':
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
            script_model_name = 'Llama-3.2-1B'
        generate_tokenizer_info = [
            python_exe, "examples/generate_xgrammar_tokenizer_info.py",
            f"--model_dir={str(resources_dir / 'models' / script_model_name)}",
            f"--output_dir={str(resources_dir / 'data' / script_model_name)}"
        ]
        run_command(generate_tokenizer_info,
                    cwd=root_dir,
                    env=model_env,
                    timeout=600)
