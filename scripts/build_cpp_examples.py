import argparse
import contextlib
import logging
import os
import platform
import shutil
import subprocess
from os import PathLike
from pathlib import Path


@contextlib.contextmanager
def working_directory(path: PathLike):
    """Changes working directory and returns to previous on exit."""
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def build_cpp_examples(build_dir: PathLike, trt_dir: PathLike,
                       enable_multi_device: str, loglevel: int) -> None:
    logging.basicConfig(level=loglevel,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    # Convert input paths to pathlib.Path objects
    build_dir = Path(build_dir)
    if trt_dir is not None:
        trt_dir = Path(trt_dir)
        if not trt_dir.is_dir():
            logging.warning(
                f"trt_dir '{trt_dir}' does not exist. "
                "Ignoring -DTensorRT_ROOT; FindTensorRT.cmake will fall back "
                "to system paths (NGC PyTorch image layout).")
            trt_dir = None

    def cmake_parse(path: PathLike) -> str:
        return str(path).replace("\\", "/")

    # Remove the build directory if it exists
    if build_dir.exists():
        logging.info(f"Removed directory: {build_dir}")
        shutil.rmtree(build_dir)

    # Create the build directory
    build_dir.mkdir(parents=True, exist_ok=True)

    # Change to the build directory
    with working_directory(build_dir):
        # Run CMake with the specified TensorRT directories
        generator = ["-GNinja"] if platform.system() == "Windows" else []
        trt_root_arg = ([f'-DTensorRT_ROOT={cmake_parse(trt_dir)}']
                        if trt_dir is not None else [])
        generate_command = [
            'cmake',
            '-S',
            '..',
            '-B',
            '.',
            f'-DENABLE_MULTI_DEVICE={enable_multi_device}',
        ] + trt_root_arg + generator
        logging.info(f"Executing {generate_command}")
        subprocess.run(generate_command, check=True)

        # Build the project using make
        build_command = ["cmake", "--build", ".", "--config", "Release"]
        logging.info(f"Executing {build_command}")
        subprocess.run(build_command, check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build C++ examples')
    parser.add_argument('--build-dir',
                        default='examples/cpp/executor/build',
                        help='Build directory path')
    parser.add_argument('--trt-dir',
                        default=None,
                        help='TensorRT directory path')
    parser.add_argument('--enable-multi-device',
                        default='ON',
                        help='Enable multi device support (requires MPI)')
    parser.add_argument('-v',
                        '--verbose',
                        help="verbose",
                        action="store_const",
                        dest="loglevel",
                        const=logging.DEBUG,
                        default=logging.INFO)
    cli = parser.parse_args()

    args = vars(cli)
    print(args)  # Log on Jenkins instance.
    build_cpp_examples(**args)
