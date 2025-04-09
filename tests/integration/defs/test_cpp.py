import glob
import logging as _logger
import os as _os
import pathlib as _pl
import platform
import shutil
import sys as _sys
from typing import List

import defs.cpp_common as _cpp
import pytest

build_script_dir = _pl.Path(
    __file__).parent.resolve().parent.parent.parent / "scripts"
assert build_script_dir.is_dir()
_sys.path.append(str(build_script_dir))

from build_wheel import main as build_trt_llm
from defs.conftest import llm_models_root


@pytest.fixture(scope="session")
def build_dir():
    return _cpp.find_build_dir()


@pytest.fixture(scope="session")
def cpp_resources_dir():
    return _pl.Path("cpp") / "tests" / "resources"


@pytest.fixture(scope="session")
def model_cache():
    return llm_models_root()


@pytest.fixture(scope="session")
def model_cache_arg(model_cache):
    return ["--model_cache", model_cache] if model_cache else []


@pytest.fixture(scope="session")
def python_exe():
    return _sys.executable


@pytest.fixture(scope="session")
def root_dir():
    return _cpp.find_root_dir()


@pytest.fixture(scope="session")
def lora_setup(root_dir, cpp_resources_dir, python_exe):

    cpp_script_dir = cpp_resources_dir / "scripts"
    cpp_data_dir = cpp_resources_dir / "data"

    generate_lora_data_args_tp1 = [
        python_exe,
        f"{cpp_script_dir}/generate_test_lora_weights.py",
        f"--out-dir={cpp_data_dir}/lora-test-weights-tp1",
        "--tp-size=1",
    ]

    generate_lora_data_args_tp2 = [
        python_exe,
        f"{cpp_script_dir}/generate_test_lora_weights.py",
        f"--out-dir={cpp_data_dir}/lora-test-weights-tp2",
        "--tp-size=2",
    ]

    generate_multi_lora_tp2_args = [
        python_exe,
        f"{cpp_script_dir}/generate_test_lora_weights.py",
        f"--out-dir={cpp_data_dir}/multi_lora",
        "--tp-size=2",
        "--num-loras=128",
    ]

    generate_gpt2_lora_data_args_tp1 = [
        python_exe,
        f"{cpp_script_dir}/generate_test_lora_weights.py",
        f"--out-dir={cpp_data_dir}/lora-test-weights-gpt2-tp1",
        "--tp-size=1",
        "--hidden-size=768",
        "--num-layers=12",
        "--config-ids-filter=0",
        "--no-generate-cache-pages",
    ]

    generate_lora_data_args_prefetch_task_3 = [
        python_exe,
        f"{cpp_script_dir}/generate_test_lora_weights.py",
        f"--out-dir={cpp_data_dir}/lora_prefetch/3",
        "--target-file-name=model.lora_weights.npy",
        "--config-file-name=model.lora_config.npy",
    ]

    generate_lora_data_args_prefetch_task_5 = [
        python_exe,
        f"{cpp_script_dir}/generate_test_lora_weights.py",
        f"--out-dir={cpp_data_dir}/lora_prefetch/5",
        "--target-file-name=model.lora_weights.npy",
        "--config-file-name=model.lora_config.npy",
    ]

    _cpp.run_command(generate_lora_data_args_tp1, cwd=root_dir, timeout=100)
    _cpp.run_command(generate_lora_data_args_tp2, cwd=root_dir, timeout=100)
    _cpp.run_command(generate_multi_lora_tp2_args, cwd=root_dir, timeout=100)
    _cpp.run_command(generate_gpt2_lora_data_args_tp1,
                     cwd=root_dir,
                     timeout=100)
    _cpp.run_command(generate_lora_data_args_prefetch_task_3,
                     cwd=root_dir,
                     timeout=100)
    _cpp.run_command(generate_lora_data_args_prefetch_task_5,
                     cwd=root_dir,
                     timeout=100)


@pytest.fixture(scope="session")
def install_additional_requirements(python_exe, root_dir):

    def _install(model_name: str):
        if model_name == "mamba":
            _cpp.run_command(
                [python_exe, "-m", "pip", "install", "transformers>=4.39.0"],
                cwd=root_dir,
                env=_os.environ,
                timeout=300,
            )

        elif model_name == "recurrentgemma":
            _cpp.run_command(
                [
                    python_exe,
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    "examples/recurrentgemma/requirements.txt",
                ],
                cwd=root_dir,
                env=_os.environ,
                timeout=300,
            )

    return _install


@pytest.fixture(scope="session")
def build_google_tests(request, build_dir):

    cuda_arch = f"{request.param}-real"

    print(f"Using CUDA arch: {cuda_arch}")
    build_trt_llm(
        cuda_architectures=cuda_arch,
        job_count=12,
        use_ccache=True,
        clean=True,
        trt_root="/usr/local/tensorrt",
    )

    make_google_tests = [
        "cmake",
        "--build",
        ".",
        "--config",
        "Release",
        "-j",
        "--target",
        "google-tests",
    ]

    # Build engine and generate output scripts need modelSpec
    make_modelSpec = [
        "cmake",
        "--build",
        ".",
        "--config",
        "Release",
        "-j",
        "--target",
        "modelSpec",
    ]

    _cpp.run_command(make_google_tests, cwd=build_dir, timeout=300)
    _cpp.run_command(make_modelSpec, cwd=build_dir, timeout=300)

    script_dir = (_pl.Path(__file__).parent.resolve().parent.parent.parent /
                  "cpp" / "tests" / "resources" / "scripts")

    assert script_dir.is_dir()
    _sys.path.append(str(script_dir))

    from build_engines_utils import init_model_spec_module

    init_model_spec_module(force_init_trtllm_bindings=False)


@pytest.fixture(scope="session")
def build_benchmarks(build_google_tests, build_dir):

    make_benchmarks = [
        "cmake",
        "--build",
        ".",
        "--config",
        "Release",
        "-j",
        "--target",
        "benchmarks",
    ]

    _cpp.run_command(make_benchmarks, cwd=build_dir, timeout=300)


@pytest.fixture(scope="session")
def prepare_model_multi_gpu(python_exe, root_dir, cpp_resources_dir,
                            model_cache):

    def _prepare(model_name: str):
        if platform.system() != "Windows":
            _cpp.prepare_multi_gpu_model_tests(
                test_list=[model_name],
                python_exe=python_exe,
                root_dir=root_dir,
                resources_dir=cpp_resources_dir,
                model_cache=model_cache,
            )

    return _prepare


@pytest.fixture(scope="session")
def prepare_model(
    root_dir,
    cpp_resources_dir,
    python_exe,
    model_cache_arg,
    install_additional_requirements,
):

    def _prepare(model_name: str, run_fp8=False):
        install_additional_requirements(model_name)

        _cpp.prepare_model_tests(
            model_name=model_name,
            python_exe=python_exe,
            root_dir=root_dir,
            resources_dir=cpp_resources_dir,
            model_cache_arg=model_cache_arg,
        )

    return _prepare


@pytest.fixture(scope="session")
def run_model_tests(build_dir, lora_setup):

    def _run(model_name: str, run_fp8: bool):
        _cpp.run_single_gpu_tests(
            build_dir=build_dir,
            test_list=[model_name],
            timeout=_cpp.default_test_timeout,
            run_fp8=run_fp8,
        )

    return _run


@pytest.fixture(scope="session")
def run_model_benchmarks(root_dir, build_dir, cpp_resources_dir, python_exe,
                         model_cache):

    def _run(
        model_name: str,
        test_gpt_session_benchmark: bool,
        batching_types: List[str],
        api_types: List[str],
    ):

        _cpp.run_benchmarks(
            model_name=model_name,
            python_exe=python_exe,
            root_dir=root_dir,
            build_dir=build_dir,
            resources_dir=cpp_resources_dir,
            model_cache=model_cache,
            test_gpt_session_benchmark=test_gpt_session_benchmark,
            batching_types=batching_types,
            api_types=api_types,
        )

    return _run


# Unit tests


@pytest.mark.parametrize("build_google_tests", ["80", "86", "89", "90"],
                         indirect=True)
def test_unit_tests(build_google_tests, build_dir, lora_setup):

    # Discover and run the actual gtests
    ctest_command = [
        "ctest",
        "--output-on-failure",
        "--output-junit",
        "results-unit-tests.xml",
    ]

    excluded_tests = list(_cpp.generate_excluded_model_tests())

    ctest_command.extend(["-E", "|".join(excluded_tests)])

    parallel = _cpp.default_test_parallel
    if parallel_override := _os.environ.get("LLM_TEST_PARALLEL_OVERRIDE", None):
        parallel = int(parallel_override)

    cpp_env = {**_os.environ}

    _cpp.parallel_run_ctest(ctest_command,
                            cwd=build_dir,
                            env=cpp_env,
                            timeout=2700,
                            parallel=parallel)


# Model tests


@pytest.mark.parametrize("build_google_tests", ["80", "86", "89", "90"],
                         indirect=True)
@pytest.mark.parametrize("model", [
    "bart", "chatglm", "eagle", "encoder", "enc_dec_language_adapter", "gpt",
    "llama", "mamba", "medusa", "recurrentgemma", "redrafter", "t5"
])
@pytest.mark.parametrize("run_fp8", [False, True], ids=["", "fp8"])
def test_model(build_google_tests, model, prepare_model, run_model_tests,
               run_fp8):

    if model == "recurrentgemma":
        pytest.skip(
            "TODO: fix recurrentgemma OOM with newest version of transformers")
        return

    prepare_model(model, run_fp8)

    run_model_tests(model, run_fp8)


@pytest.mark.parametrize("build_google_tests", ["80", "86", "89", "90"],
                         indirect=True)
@pytest.mark.parametrize("model", ["bart", "gpt", "t5"])
def test_benchmarks(build_benchmarks, model, prepare_model,
                    run_model_benchmarks):

    prepare_model(model)

    test_gpt_session_benchmark = True if model == "gpt" else False
    batching_types = ["IFB", "V1"] if model == "gpt" else ["IFB"]
    api_types = ["executor"]

    run_model_benchmarks(
        model_name=model,
        test_gpt_session_benchmark=test_gpt_session_benchmark,
        batching_types=batching_types,
        api_types=api_types,
    )


@pytest.mark.parametrize("build_google_tests", ["80", "86", "89", "90"],
                         indirect=True)
def test_multi_gpu_simple(build_google_tests, build_dir):

    if platform.system() != "Windows":

        _cpp.run_simple_multi_gpu_tests(build_dir=build_dir,
                                        timeout=_cpp.default_test_timeout)


@pytest.mark.parametrize("build_google_tests", ["80", "86", "89", "90"],
                         indirect=True)
def test_multi_gpu_t5(build_google_tests, prepare_model_multi_gpu, build_dir):

    if platform.system() != "Windows":
        prepare_model_multi_gpu("t5")
        _cpp.run_t5_multi_gpu_tests(build_dir=build_dir,
                                    timeout=_cpp.default_test_timeout)


@pytest.mark.parametrize("build_google_tests", ["80", "86", "89", "90"],
                         indirect=True)
def test_multi_gpu_llama_executor(build_google_tests, prepare_model_multi_gpu,
                                  lora_setup, build_dir):

    if platform.system() != "Windows":
        prepare_model_multi_gpu("llama")
        _cpp.run_llama_executor_multi_gpu_tests(
            build_dir=build_dir, timeout=_cpp.default_test_timeout)


@pytest.mark.parametrize("build_google_tests", ["80", "86", "89", "90"],
                         indirect=True)
def test_multi_gpu_trt_gpt_real_decoder(build_google_tests,
                                        prepare_model_multi_gpu, lora_setup,
                                        build_dir):

    if platform.system() != "Windows":
        prepare_model_multi_gpu("llama")
        _cpp.run_trt_gpt_model_real_decoder_multi_gpu_tests(
            build_dir=build_dir, timeout=_cpp.default_test_timeout)


@pytest.mark.parametrize("build_google_tests", ["80", "86", "89", "90"],
                         indirect=True)
def test_multi_gpu_disagg(prepare_model, prepare_model_multi_gpu,
                          build_google_tests, build_dir):

    if platform.system() != "Windows":
        # Disagg tests need single + multi GPU llama models.
        prepare_model("llama")
        prepare_model_multi_gpu("llama")

        prepare_model("gpt")

        _cpp.run_disagg_multi_gpu_tests(build_dir=build_dir)


@pytest.fixture(scope="function", autouse=True)
def keep_log_files(llm_root):
    "Backup previous cpp test results when run multiple ctest"
    results_dir = f"{llm_root}/cpp/build"

    yield

    backup_dir = f"{llm_root}/cpp/build_backup"
    _os.makedirs(backup_dir, exist_ok=True)
    # Copy XML files to backup directory
    xml_files = glob.glob(f"{results_dir}/*.xml")
    if xml_files:
        for xml_file in xml_files:
            try:
                shutil.copy(xml_file, backup_dir)
                _logger.info(f"Copied {xml_file} to {backup_dir}")
            except Exception as e:
                _logger.error(f"Error copying {xml_file}: {str(e)}")
    else:
        _logger.info("No XML files found in the build directory.")
