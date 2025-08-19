import logging as _logger
import os as _os
import pathlib as _pl
import shutil
import sys as _sys
import time

import defs.cpp.cpp_common as _cpp
import pytest

build_script_dir = _pl.Path(
    __file__).parent.resolve().parent.parent.parent.parent / "scripts"
assert build_script_dir.is_dir()
_sys.path.append(str(build_script_dir))

from build_wheel import main as build_trt_llm
from defs.conftest import llm_models_root


@pytest.fixture(scope="session")
def build_type():
    """CMake build type for C++ builds."""
    # For debugging purposes, we can use the RelWithDebInfo build type.
    return _os.environ.get("TLLM_BUILD_TYPE", "Release")


@pytest.fixture(scope="session")
def build_dir(build_type):
    """Resolved build directory for the current build_type."""
    return _cpp.find_build_dir(build_type)


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
                    "examples/models/core/recurrentgemma/requirements.txt",
                ],
                cwd=root_dir,
                env=_os.environ,
                timeout=300,
            )

    return _install


@pytest.fixture(scope="session")
def build_google_tests(request, build_type):

    cuda_arch = f"{request.param}-real"

    _logger.info(f"Using CUDA arch: {cuda_arch}")

    build_trt_llm(
        build_type=build_type,
        cuda_architectures=cuda_arch,
        job_count=12,
        use_ccache=True,
        clean=True,
        generator="Ninja",
        trt_root="/usr/local/tensorrt",
        nixl_root="/opt/nvidia/nvda_nixl",
        skip_building_wheel=True,
        extra_make_targets=["google-tests"],
    )


@pytest.fixture(scope="session")
def build_benchmarks(build_google_tests, build_dir, build_type):

    make_benchmarks = [
        "cmake",
        "--build",
        ".",
        "--config",
        build_type,
        "-j",
        "--target",
        "benchmarks",
    ]

    _cpp.run_command(make_benchmarks, cwd=build_dir, timeout=300)


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

        start_time = time.time()

        _cpp.prepare_model_tests(
            model_name=model_name,
            python_exe=python_exe,
            root_dir=root_dir,
            resources_dir=cpp_resources_dir,
            model_cache_arg=model_cache_arg,
        )

        duration = time.time() - start_time
        print(f"Built model: {model_name}")
        print(f"Duration: {duration} seconds")

    return _prepare


@pytest.fixture(scope="function", autouse=True)
def keep_log_files(build_dir):
    """Backup previous cpp test results when run multiple ctest invocations."""
    results_dir = build_dir

    yield

    build_parent_dir = build_dir.parent
    backup_dir_name = build_dir.name + "_backup"
    backup_dir = build_parent_dir / backup_dir_name
    backup_dir.mkdir(parents=True, exist_ok=True)
    # Copy XML files from all subdirectories to backup directory
    xml_files = list(results_dir.rglob("*.xml"))
    if xml_files:
        for xml_file in xml_files:
            try:
                shutil.copy(xml_file, backup_dir)
                _logger.info(f"Copied {xml_file} to {backup_dir}")
            except Exception as e:
                _logger.error(f"Error copying {xml_file}: {str(e)}")
    else:
        _logger.info("No XML files found in the build directory.")
