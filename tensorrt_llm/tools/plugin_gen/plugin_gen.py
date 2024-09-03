'''
This file is a script tool for generating TensorRT plugin library for Triton.
'''
import argparse
import glob
import logging
import os
import subprocess  # nosec B404
import sys
from dataclasses import dataclass
from typing import ClassVar, Iterable, List, Optional, Tuple, Union

try:
    import triton
except ImportError:
    raise ImportError("Triton is not installed. Please install it first.")

# isort: off
from tensorrt_llm.tools.plugin_gen.core import (
    KernelMetaData, PluginCmakeCodegen, PluginCppCodegen, PluginPyCodegen,
    PluginRegistryCodegen, copy_common_files)
# isort: on

PYTHON_BIN = sys.executable

TRITON_ROOT = os.path.dirname(triton.__file__)
TRITON_COMPILE_BIN = os.path.join(TRITON_ROOT, 'tools', 'compile.py')
TRITON_LINK_BIN = os.path.join(TRITON_ROOT, 'tools', 'link.py')


@dataclass
class StageArgs:
    workspace: str  # the root directory for all the stages
    kernel_name: str

    @property
    def sub_workspace(self) -> str:
        return os.path.join(self.workspace, self.kernel_name,
                            f"_{self.stage_name}")


@dataclass
class _TritonAotArgs(StageArgs):
    stage_name: ClassVar[str] = 'triton_aot'

    @dataclass
    class _AotConfig:
        output_name: str
        num_warps: int
        num_stages: int
        signature: str

    kernel_file: str
    configs: List[_AotConfig]
    grid_dims: Tuple[str, str, str]


@dataclass
class _TritonKernelCompileArgs(StageArgs):
    stage_name: ClassVar[str] = 'compile_triton_kernel'


@dataclass
class _TrtPluginGenArgs(StageArgs):
    stage_name: ClassVar[str] = 'generate_trt_plugin'

    config: KernelMetaData


@dataclass
class _TrtPluginCompileArgs:
    workspace: str
    trt_lib_dir: Optional[str] = None
    trt_include_dir: Optional[str] = None
    trt_llm_include_dir: Optional[str] = None

    stage_name: ClassVar[str] = 'compile_trt_plugin'

    @property
    def sub_workspace(self) -> str:
        return self.workspace


@dataclass
class _CopyOutputArgs:
    so_path: str
    functional_py_path: str
    output_dir: str

    stage_name: ClassVar[str] = 'copy_output'


@dataclass
class Stage:
    '''
    Stage represents a stage in the plugin generation process. e.g. Triton AOT could be a stage.
    '''

    config: Union[_TritonAotArgs, _TritonKernelCompileArgs, _TrtPluginGenArgs,
                  _TrtPluginCompileArgs, _CopyOutputArgs]

    def run(self):
        stages = {
            _TritonAotArgs.stage_name: self.do_triton_aot,
            _TritonKernelCompileArgs.stage_name: self.do_compile_triton_kernel,
            _TrtPluginGenArgs.stage_name: self.do_generate_trt_plugin,
            _TrtPluginCompileArgs.stage_name: self.do_compile_trt_plugin,
            _CopyOutputArgs.stage_name: self.do_copy_output,
        }

        logging.info(f"Running stage {self.config.stage_name}")

        stages[self.config.stage_name]()

    def do_triton_aot(self):
        compile_dir = self.config.sub_workspace
        _clean_path(compile_dir)

        # compile each config for different hints
        for config in self.config.configs:
            command = [
                PYTHON_BIN, TRITON_COMPILE_BIN, self.config.kernel_file, '-n',
                self.config.kernel_name, '-o', f"{compile_dir}/kernel",
                '--out-name', config.output_name, '-w',
                str(config.num_warps), '-s', f"{config.signature}", '-g',
                ','.join(self.config.grid_dims), '--num-stages',
                str(config.num_stages)
            ]
            _run_command(command)

        # link and get a kernel launcher with all the configs
        h_files = glob.glob(os.path.join(compile_dir, '*.h'))
        command = [
            PYTHON_BIN,
            TRITON_LINK_BIN,
            *h_files,
            '-o',
            os.path.join(compile_dir, 'launcher'),
        ]
        _run_command(command)

    def do_compile_triton_kernel(self):
        '''
        Compile the triton kernel to library.
        '''
        #assert isinstance(self.args, _TritonKernelCompileArgs)

        from triton.common import cuda_include_dir, libcuda_dirs
        kernel_dir = os.path.join(self.config.workspace,
                                  self.config.kernel_name, '_triton_aot')
        compile_dir = self.config.sub_workspace
        _mkdir(compile_dir)
        _clean_path(compile_dir)

        c_files = glob.glob(os.path.join(os.getcwd(), kernel_dir, "*.c"))
        assert c_files
        _run_command([
            "gcc",
            "-c",
            *c_files,
            "-I",
            cuda_include_dir(),
            "-L",
            libcuda_dirs()[0],
            "-l",
            "cuda",
        ],
                     cwd=compile_dir)

        o_files = glob.glob(os.path.join(os.getcwd(), compile_dir, "*.o"))
        assert o_files
        '''
        _run_command([
            "ar", "rcs",
            f"lib{self.args.kernel_name}.a", *o_files
        ], cwd=compile_dir)
        '''

    def do_generate_trt_plugin(self):
        '''
        Generate the trt plugin from the triton kernel library.
        '''
        #assert isinstance(self.args, _TrtPluginGenArgs)
        workspace = self.config.sub_workspace
        _mkdir(workspace)
        _clean_path(workspace)

        PluginCppCodegen(output_dir=workspace,
                         meta_data=self.config.config).generate()

    def do_compile_trt_plugin(self):
        '''
        Compile the trt plugin library.
        '''

        # collect all the kernels within the workspace
        #assert isinstance(self.args, _TrtPluginCompileArgs)

        def collect_all_kernel_configs(
                workspace: str) -> Iterable[KernelMetaData]:
            ymls = glob.glob(os.path.join(workspace, '**/*.yml'),
                             recursive=True)
            for path in ymls:
                yield KernelMetaData.load_from_yaml(path)

        configs = list(collect_all_kernel_configs(self.config.workspace))

        kernel_names = [config.kernel_name for config in configs]

        PluginRegistryCodegen(out_path=os.path.join(self.config.sub_workspace,
                                                    'tritonPlugins.cpp'),
                              plugin_names=kernel_names).generate()
        copy_common_files(self.config.sub_workspace)
        PluginCmakeCodegen(out_path=os.path.join(self.config.sub_workspace,
                                                 'CMakeLists.txt'),
                           plugin_names=kernel_names,
                           kernel_names=kernel_names,
                           workspace=self.config.workspace).generate()

        functional_py = os.path.join(self.config.workspace, 'functional.py')
        plugin_lib_path = os.path.join(self.config.workspace, 'build',
                                       'libtriton_plugins.so')

        for i, kernel_config in enumerate(configs):
            PluginPyCodegen(out_path=functional_py,
                            meta_data=kernel_config,
                            add_header=i == 0,
                            plugin_lib_path=plugin_lib_path).generate()

        # create build directory and compile
        _run_command(['mkdir', '-p', 'build'], cwd=self.config.sub_workspace)
        _run_command(['rm', '-rf', 'build/*'], cwd=self.config.sub_workspace)

        cmake_cmds = ['cmake', '..']
        if self.config.trt_lib_dir is not None:
            cmake_cmds.append(f'-DTRT_LIB_DIR={self.config.trt_lib_dir}')
        if self.config.trt_include_dir:
            cmake_cmds.append(
                f'-DTRT_INCLUDE_DIR={self.config.trt_include_dir}')
        if self.config.trt_llm_include_dir is None:
            self.config.trt_llm_include_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), '../../../cpp')
        cmake_cmds.append(
            f'-DTRT_LLM_INCLUDE_DIR={self.config.trt_llm_include_dir}')
        _run_command(cmake_cmds,
                     cwd=os.path.join(self.config.sub_workspace, "build"))
        _run_command(['make', '-j'],
                     cwd=os.path.join(self.config.sub_workspace, "build"))

    def do_copy_output(self):
        '''
        Copy the output to the destination directory.
        '''
        _mkdir(self.config.output_dir)
        _clean_path(self.config.output_dir)
        # copy the so file
        _run_command(['cp', self.config.so_path, self.config.output_dir])

        # copy the functional.py
        _run_command(
            ['cp', self.config.functional_py_path, self.config.output_dir])


def gen_trt_plugins(workspace: str,
                    metas: List[KernelMetaData],
                    trt_lib_dir: Optional[str] = None,
                    trt_include_dir: Optional[str] = None,
                    trt_llm_include_dir: Optional[str] = None):
    '''
    Generate TRT plugins end-to-end.
    '''
    for meta in metas:
        Stage(meta.to_TritonAotArgs(workspace)).run()
        Stage(meta.to_TritonKernelCompileArgs(workspace)).run()
        Stage(meta.to_TrtPluginGenArgs(workspace)).run()

    # collect all the plugins
    compile_args = _TrtPluginCompileArgs(
        workspace=workspace,
        trt_lib_dir=trt_lib_dir,
        trt_include_dir=trt_include_dir,
        trt_llm_include_dir=trt_llm_include_dir,
    )
    Stage(compile_args).run()
    Stage(metas[0].to_CopyOutputArgs(workspace)).run()


def _clean_path(path: str):
    '''
    Clean the content within this directory
    '''
    _rmdir(path)
    _mkdir(path)


def _mkdir(path: str):
    '''
    mkdir if not exists
    '''
    subprocess.run(['/usr/bin/mkdir', '-p', path], check=True)


def _rmdir(path: str):
    '''
    rmdir if exists
    '''
    subprocess.run(['/usr/bin/rm', '-rf', path], check=True)


def _run_command(args, cwd=None):
    print(f"Running command: {' '.join(args)}")
    subprocess.run(args, check=True, cwd=cwd)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--workspace',
        type=str,
        required=True,
        help="The root path to store all the intermediate files")
    parser.add_argument(
        '--kernel_config',
        type=str,
        required=True,
        help=
        'The path to the kernel config file, which should be a python module '
        'containing KernelMetaData instances')
    parser.add_argument(
        '--trt_lib_dir',
        type=str,
        default=None,
        help='Directory to find TensorRT library. If None, find the library '
        'from the system path or /usr/local/tensorrt.')
    parser.add_argument(
        '--trt_include_dir',
        type=str,
        default=None,
        help='Directory to find TensorRT headers. If None, find the headers '
        'from the system path or /usr/local/tensorrt.')
    parser.add_argument('--trt_llm_include_dir',
                        type=str,
                        default=None,
                        help='Directory to find TensorRT LLM C++ headers.')
    args = parser.parse_args()

    assert args.kernel_config.endswith('.py'), \
        f"Kernel config {args.kernel_config} should be a python module"

    return args


def import_from_file(module_name, file_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)

    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


if __name__ == '__main__':
    args = parse_arguments()

    module_name = os.path.basename(args.kernel_config).replace('.py', '')
    config_module = import_from_file(module_name, args.kernel_config)
    kernel_configs: List[KernelMetaData] = config_module.KERNELS

    gen_trt_plugins(workspace=args.workspace,
                    trt_lib_dir=args.trt_lib_dir,
                    trt_include_dir=args.trt_include_dir,
                    trt_llm_include_dir=args.trt_llm_include_dir,
                    metas=kernel_configs)
