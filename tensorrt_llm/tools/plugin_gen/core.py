import glob
import os
import shutil
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import (TYPE_CHECKING, Any, ClassVar, Dict, Iterable, List, Tuple,
                    Union)

from tensorrt_llm.logger import logger

if TYPE_CHECKING:
    from tensorrt_llm.tools.plugin_gen.plugin_gen import _TritonAotArgs, _TritonKernelCompileArgs, _TrtPluginGenArgs, _CopyOutputArgs

import jinja2
import yaml

pjoin = os.path.join
cdir = Path(__file__).absolute().parent


class DType(Enum):
    FP16 = 0
    FP32 = 1
    FP64 = 2
    INT8 = 3
    INT32 = 4
    INT64 = 5

    @staticmethod
    def get_str(d: "DType"):
        assert isinstance(d, DType)
        return d.to("c")

    @staticmethod
    def get_trt_dtype(d: "DType") -> str:
        assert isinstance(d, DType)
        return d.to("trt")

    def to(self, dst: str) -> str:
        if dst == 'trt_plugin_py':
            map = DType.get_map("dtype", 'trt_plugin')
            ret = map[self]
            return ret[1:]  # skip the proceeding 'k'

        map = DType.get_map("dtype", dst)
        ret = map[self]
        return ret

    @lru_cache
    @staticmethod
    def get_map(src: str, dst: str) -> Dict[Any, Any]:
        idx = dict(dtype=0, abbr=1, trt=2, c=3, np=4, trt_plugin=5)
        return {x[idx[src]]: x[idx[dst]] for x in DType._get_dtype_strs()}

    @staticmethod
    def _get_dtype_strs() -> List[Tuple["DType", str, str, str, str, str]]:
        return [(DType.FP16, "fp16", "kHALF", "half", "float16", "kFLOAT16"),
                (DType.FP32, "fp32", "kFLOAT", "float", "float32", "kFLOAT32"),
                (DType.FP64, "fp64", "kDOUBLE", "double", "float64",
                 "kFLOAT64"),
                (DType.INT8, "i8", "kINT8", "int8_t", "int8", "kINT8"),
                (DType.INT32, "i32", "kINT32", "int32_t", "int32", "kINT32"),
                (DType.INT64, "i64", "kINT64", "int64_t", "int64", "KINT64")]


class Type:

    def __init__(self, s: str):
        to_dtype = DType.get_map("abbr", "dtype")
        is_tensor = False
        if s.startswith("tensor"):
            is_tensor = True

            s = s.split("[")[1].split("]")[0]

        self.is_tensor = is_tensor
        self.dtype = to_dtype[s]

    def to_triton_sig(self) -> str:
        dic = DType.get_map("dtype", "abbr")
        return f'*{dic[self.dtype]}' if self.is_tensor else dic[self.dtype]

    def __str__(self) -> str:
        dic = DType.get_map("dtype", "abbr")
        return f"tensor[{dic[self.dtype]}]" if self.is_tensor else dic[
            self.dtype]

    @staticmethod
    def from_str(s: str):
        return Type(s)

    @property
    def is_scalar(self) -> bool:
        return not self.is_tensor

    @staticmethod
    def tensor_ty(dtype: "DType"):
        return Type(f"tensor[{DType.get_str(dtype)}]")

    @staticmethod
    def float16() -> "Type":
        return Type("fp16")

    @staticmethod
    def float32() -> "Type":
        return Type("fp32")

    @staticmethod
    def float64() -> "Type":
        return Type("fp64")

    @staticmethod
    def int8() -> "Type":
        return Type("i8")

    @staticmethod
    def int32() -> "Type":
        return Type("i32")

    @staticmethod
    def int64() -> "Type":
        return Type("i64")


@dataclass
class Argument:
    ''' An input or output parameter of a Triton kernel.  '''

    class ArgType(Enum):
        # an input
        INPUT = 0
        # an output
        OUTPUT = 1
        # an argument that is a PluginField of a plugin
        PARAM = 2
        # an argument that is a dimension size of a tensor
        # such argument could be deduced from input tensors' shapes, so they will be hidden from plugin's input
        DIM_SIZE = 3

    name: str
    dtype: Type
    # offset of the argument in either input or output arglist
    # NOTE, both input and output offsets are counted from 0
    offset: int = 0
    hints: List[str] = field(default_factory=list)
    arg_type: "Argument.ArgType" = ArgType.INPUT

    @property
    def is_output(self) -> bool:
        return self.arg_type == Argument.ArgType.OUTPUT

    @property
    def is_input(self) -> bool:
        return self.arg_type == Argument.ArgType.INPUT

    @property
    def is_param(self) -> bool:
        return self.arg_type == Argument.ArgType.PARAM

    @property
    def is_dim_size(self) -> bool:
        return self.arg_type == Argument.ArgType.DIM_SIZE

    def to_dict(self, force_str=False) -> Dict[str, Any]:
        return dict(
            name=self.name,
            dtype=str(self.dtype),
            is_input=self.is_input,
            offset=self.offset,
            hints=self.hints,
            arg_type=self.arg_type.name,
        )

    @property
    def is_tensor(self):
        return self.dtype.is_tensor


@dataclass
class InputArg(Argument):
    '''
    Sugar for creating an input argument.
    '''
    arg_type: ClassVar = Argument.ArgType.INPUT


@dataclass
class OutputArg(Argument):
    '''
    Sugar for creating an output argument.
    '''
    arg_type: ClassVar = Argument.ArgType.OUTPUT


@dataclass
class ParamArg(Argument):
    '''
    Sugar for creating a parameter argument.

    This will generate a TensorRT PluginField.
    '''
    arg_type: ClassVar = Argument.ArgType.PARAM


@dataclass
class DimSizeArg(Argument):
    '''
    Sugar for creating a dimension size argument.

    This will generate a TensorRT PluginField.
    '''
    arg_type: ClassVar = Argument.ArgType.DIM_SIZE
    code: str = field(default_factory=str, init=False)
    dtype: ClassVar[Type] = Type("i32")  # i64?


@dataclass
class Constexpr:
    ''' tl.constexpr '''
    value: int

    def to_dict(self, force_str=False) -> Dict[str, Any]:
        return dict(value=self.value)


@dataclass
class KernelMetaData:
    '''
    All the necessary metadata of a Triton kernel.

    This acts as the core data structure for the configuration required for generating the Triton plugin.
    '''
    ArgT = Union[Constexpr, ParamArg, InputArg, OutputArg, DimSizeArg]

    kernel_name: str = ""
    ios: List[ArgT] = field(default_factory=list)
    shape_infer_rules: List[str] = field(default_factory=list)
    version: int = 1
    kernel_file: str = ""  # path to the Triton kernel file
    num_warps: int = 1
    num_stages: int = 1
    grid_dims: Tuple[str, str, str] = ("1", "1", "1")

    _name_to_arg: Dict[str, Argument] = field(default_factory=dict, init=False)

    def __post_init__(self):
        # build name_to_arg mapping
        self._name_to_arg.clear()
        for io in self.ios:
            if isinstance(io, Argument):
                self._name_to_arg[io.name] = io

        # set the argument offset
        for arg_off, arg in enumerate(self.get_inputs()):
            arg.offset = arg_off
        for arg_off, arg in enumerate(self.get_outputs()):
            arg.offset = arg_off

        self._validate()

    @property
    def arguments(self) -> Iterable[Argument]:
        return filter(lambda x: isinstance(x, Argument), self.ios)

    @staticmethod
    def load_from_yaml(yaml_path: str = "",
                       yaml_str: str = "") -> "KernelMetaData":
        assert yaml_path or yaml_str, "Either yaml_path or yaml_str should be given"
        if yaml_path:
            with open(yaml_path, "r") as f:
                yaml_str = f.read()
        yaml_data = yaml.safe_load(yaml_str)

        kernel_name = yaml_data["name"]
        ios = []
        for arg_name, arg_data in yaml_data["arguments"].items():
            if "value" in arg_data:
                ios.append(Constexpr(arg_data["value"]))
            else:
                ios.append(
                    Argument(arg_name,
                             Type(arg_data["dtype"]),
                             arg_type=Argument.ArgType[arg_data.get(
                                 "arg_type", "INPUT")]))
        shape_infer_rules = yaml_data.get("shape_infer_rules", [])
        version = yaml_data.get("version", 1)
        grid_dims = yaml_data["grid_dims"]

        # TODO: add other metadata for launching a Triton kernel

        return KernelMetaData(kernel_name=kernel_name,
                              ios=ios,
                              shape_infer_rules=shape_infer_rules,
                              version=version,
                              grid_dims=grid_dims)

    def to_yaml(self) -> str:
        ''' Convert the metadata to a YAML string. '''
        ret = dict(
            name=self.kernel_name,
            version=self.version,
            arguments=OrderedDict(),
            shape_deduce=self.shape_infer_rules,
            grid_dims=self.grid_dims,
        )

        const_count = 0
        for arg in self.ios:
            if isinstance(arg, Argument):
                ret["arguments"][arg.name] = arg.to_dict(force_str=True)
            else:
                ret["arguments"][f"constexpr_{const_count}"] = arg.to_dict()
                const_count += 1

        logger.info(f"load {self.num_inputs} inputs")
        logger.info(f"load {self.num_outputs} outputs")
        logger.info(f"load {self.num_constexprs} constexprs")

        yaml.add_representer(
            data_type=OrderedDict,
            representer=lambda dumper, data: dumper.represent_mapping(
                yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
                data.items(),
            ),
            Dumper=yaml.SafeDumper)
        return yaml.safe_dump(ret)

    def to_triton_signatures(self) -> List[str]:
        '''
        Generate the signatures for the Triton compile.py tool.
        '''
        signature = []
        hints = []
        for arg in self.ios:
            if isinstance(arg, Argument):
                sig = arg.dtype.to_triton_sig()
                signature.append(sig)
                hints.append(arg.hints)
            else:
                signature.append(str(arg.value))
                hints.append([])

        # the number of hints should be the same across the arguments those have hints
        num_hints_per_arg = 0
        for cur in hints:
            if cur and num_hints_per_arg == 0:
                num_hints_per_arg = len(cur)
            if cur:
                assert len(
                    cur
                ) == num_hints_per_arg, f"The number of hints should be the same across the arguments those have hints, get {len(cur)} mismatch with {num_hints_per_arg}"

        num_hints_per_arg = max(num_hints_per_arg, 1)
        # fill the empty hints
        for cur in hints:
            if len(cur) != num_hints_per_arg:
                cur.extend([''] * (num_hints_per_arg - len(cur)))

        signatures = []
        for sig, hints in zip(signature, hints):
            signatures.append(
                [f"{sig}:{hint}" if hint else sig for hint in hints])

        return [', '.join(sig) for sig in zip(*signatures)]

    def get_inputs(self) -> Iterable[Argument]:
        return filter(lambda x: x.is_input, self.arguments)

    def get_outputs(self) -> Iterable[Argument]:
        return filter(lambda x: x.is_output, self.arguments)

    def get_dim_size_args(self) -> Iterable[Argument]:
        return filter(lambda x: x.is_dim_size, self.arguments)

    def get_params(self) -> Iterable[Argument]:
        return filter(lambda x: x.is_param, self.arguments)

    @property
    def num_inputs(self) -> int:
        return len(list(filter(lambda x: x.is_input, self.arguments)))

    @property
    def num_outputs(self) -> int:
        return len(list(filter(lambda x: not x.is_input, self.arguments)))

    @property
    def num_constexprs(self) -> int:
        return len(self.ios) - self.num_inputs - self.num_outputs

    def to_TritonAotArgs(self, workspace: str) -> '_TritonAotArgs':
        '''
        Get a TritonAotArgs from the metadata.

        Args:
            workspace: the root directory for all the stages generations.
        '''
        from tensorrt_llm.tools.plugin_gen.plugin_gen import _TritonAotArgs
        return _TritonAotArgs(
            kernel_name=self.kernel_name,
            workspace=workspace,
            kernel_file=self.kernel_file,
            configs=[
                _TritonAotArgs._AotConfig(
                    output_name=self.kernel_name,
                    num_warps=self.num_warps,
                    num_stages=self.num_stages,
                    signature=sig,
                ) for sig in self.to_triton_signatures()
            ],
            grid_dims=self.grid_dims,
        )

    def to_TritonKernelCompileArgs(
            self, workspace: str) -> '_TritonKernelCompileArgs':
        from tensorrt_llm.tools.plugin_gen.plugin_gen import \
            _TritonKernelCompileArgs
        return _TritonKernelCompileArgs(workspace=workspace,
                                        kernel_name=self.kernel_name)

    def to_TrtPluginGenArgs(self, workspace: str) -> '_TrtPluginGenArgs':
        from tensorrt_llm.tools.plugin_gen.plugin_gen import _TrtPluginGenArgs
        return _TrtPluginGenArgs(workspace=workspace,
                                 config=self,
                                 kernel_name=self.kernel_name)

    def to_TrtPluginCompileArgs(self,
                                workspace: str) -> '_TrtPluginCompileArgs':
        from tensorrt_llm.tools.plugin_gen.plugin_gen import \
            _TrtPluginCompileArgs
        return _TrtPluginCompileArgs(workspace=workspace)

    def to_CopyOutputArgs(self, workspace: str) -> '_CopyOutputArgs':
        from tensorrt_llm.tools.plugin_gen.plugin_gen import _CopyOutputArgs
        return _CopyOutputArgs(
            so_path=pjoin(workspace, 'build', 'libtriton_plugins.so'),
            functional_py_path=pjoin(workspace, 'functional.py'),
            output_dir=pjoin(workspace, 'output'),
        )

    def _validate(self):
        assert self.num_inputs > 0, "At least one input should be given"
        assert self.num_outputs > 0, "At least one output should be given"


def _render_common_parameters():
    return dict(
        triton_aot_dir='_triton_aot',
        generate_trt_plugin_dir='_generate_trt_plugin',
        compile_trt_plugin_dir='_compile_trt_plugin',
        compile_triton_kernel_dir='_compile_triton_kernel',
    )


@dataclass
class PluginCppCodegen:
    ''' Generate the C++ code for a Triton plugin, including a xPlugin.h, xPlugin.cpp '''
    output_dir: str
    meta_data: KernelMetaData

    def __post_init__(self):
        from tensorrt_llm.tools.plugin_gen.shape_infer import CppCodeTranspiler

        # parse the rules
        transpiler = CppCodeTranspiler(self.meta_data._name_to_arg)
        self.shape_infer_code, dim_size_infer_code = transpiler(
            self.meta_data.shape_infer_rules)

        for arg in self.meta_data.get_dim_size_args():
            arg.code = dim_size_infer_code[arg.name]

    @property
    def plugin_name(self) -> str:
        return f"{self.meta_data.kernel_name}Plugin"

    def generate(self):
        file_base_name = "plugin"

        # generate header file
        with open(pjoin(self.output_dir, file_base_name + ".h"), "w") as f:
            f.write(self._render('plugin.h.tpl'))

        # generate cpp file
        with open(pjoin(self.output_dir, file_base_name + ".cpp"), "w") as f:
            f.write(self._render('plugin.cpp.tpl'))

        # dump meta_data to yaml for later collection in cmake
        with open(pjoin(self.output_dir, 'plugin.yml'), 'w') as f:
            f.write(self.meta_data.to_yaml())

    def _render(self, tpl_path: str):
        env = setup_jinja_env()

        tpl_data = dict(
            kernel_name=self.meta_data.kernel_name,
            plugin_name=f"{self.meta_data.kernel_name}Plugin",
            kernel_version=self.meta_data.version,
            construct_arg_list=self.construct_arg_list,
            getOutputDimensions_body=self.getOutputDimensions_body,
            # for supportsFormatCombination
            inputs=list(self.meta_data.get_inputs()),
            outputs=list(self.meta_data.get_outputs()),
            dim_size_args=list(self.meta_data.get_dim_size_args()),
            params=list(self.meta_data.get_params()),
            param_names=[arg.name for arg in self.meta_data.get_params()],
            io_count=self.get_io_count(),
            input_count=len(list(self.meta_data.get_inputs())),
            output_count=len(list(self.meta_data.get_outputs())),
            configurePlugin_body=self.configurePlugin_body,
            getWorkspaceSize_body=self.getWorkspaceSize_body,
            enqueue_body_arg_list=self.enqueue_body_arg_list,
            getNbOutputs_body=self.getNbOutputs_body,
            creator_constructor_body=self.creator_constructor_body,
            plugin_version='0',
            **_render_common_parameters(),
        )

        return env.get_template(tpl_path).render(tpl_data)

    def get_io_count(self) -> int:
        return len(list(self.meta_data.get_inputs())) + len(
            list(self.meta_data.get_outputs()))

    @property
    def construct_arg_list(self) -> str:
        return ", ".join("%s %s" % (arg.dtype.dtype.to("c"), arg.name)
                         for arg in self.meta_data.get_params())

    @property
    def getOutputDimensions_body(self) -> str:
        lines = [code(f"nvinfer1::DimsExprs outputDims;")
                 ] + self.shape_infer_code + [code("return outputDims;")]
        indent = 2 * ' '
        return '\n'.join(indent + line for line in lines)

    @property
    def configurePlugin_body(self) -> str:
        return ""

    @property
    def getWorkspaceSize_body(self) -> str:
        return code("return 0;")

    @property
    def enqueue_body_arg_list(self) -> str:
        # Here we add two additional arguments: stream and algo_id=0 for launching the triton kernel
        return ", ".join(["stream"] +
                         [arg.name for arg in self.meta_data.arguments] + ['0'])

    @property
    def getOutputDataType_body(self) -> str:
        outputs = filter(lambda x: x.is_output, self.meta_data.arguments)
        ret = []
        for off, out in enumerate(outputs):
            ret.append(code(f"if (index == {off}) {{"))
            ret.append(
                code(
                    f"return nvinfer1::DataType::{DType.get_trt_dtype(out.dtype.dtype)};"
                ))
            ret.append(code("}"))
        return '\n'.join(ret)

    @property
    def getNbOutputs_body(self) -> str:
        return code(f"return {self.meta_data.num_outputs};")

    @property
    def serialize_body(self) -> str:
        return ""

    @property
    def creator_constructor_body(self) -> str:
        return ""

    def getPluginVersion_body(self) -> str:
        return code(f"return {self.meta_data.version};")


@dataclass
class PluginPyCodegen:
    ''' Generate the Python functional wrapper for a Triton plugin.  '''

    out_path: str
    meta_data: KernelMetaData
    add_header: bool
    plugin_lib_path: str

    def generate(self):
        write_mode = "w" if self.add_header else "a"
        with open(self.out_path, write_mode) as f:
            env = setup_jinja_env()

            tpl_data = dict(
                metadata=self.meta_data,
                plugin_name=f"{self.meta_data.kernel_name}Plugin",
                kernel_name=self.meta_data.kernel_name,
                kernel_ret=self.kernel_ret,
                kernel_version=self.meta_data.version,
                arg_list=', '.join(arg.name for arg in self.get_arg_list()),
                add_header=self.add_header,
                params=self.meta_data.get_params(),
                inputs=[
                    dict(name=arg.name,
                         np_type=arg.dtype.dtype.to("np"),
                         trt_type=arg.dtype.dtype.to("trt")) for arg in filter(
                             lambda x: x.is_input, self.meta_data.arguments)
                ],
                input_list=', '.join(arg.name
                                     for arg in self.meta_data.get_inputs()),
                num_outputs=len(list(self.meta_data.get_outputs())),
                plugin_lib_path=self.plugin_lib_path,
            )

            content = env.get_template("functional.py.tpl").render(tpl_data)
            f.write(content)

    def get_arg_list(self) -> Iterable[Argument]:
        # NOTE: for easier argument passing, here DONT follow the argument order in the original Triton kernel
        for arg in self.meta_data.get_params():
            yield arg
        for arg in self.meta_data.get_inputs():
            yield arg

    @property
    def kernel_ret(self):
        return ', '.join(f"_create_tensor(layer.get_output({i}), layer)"
                         for i in range(self.meta_data.num_outputs))


@dataclass
class PluginRegistryCodegen:
    '''
    Generate the code for adding all the detected Triton plugins to the TensorRT registry.
    '''
    out_path: str
    plugin_names: List[str]

    def generate(self):
        with open(self.out_path, "w") as f:
            env = setup_jinja_env()

            tpl_data = dict(plugin_creators=[
                f"{plugin_name}PluginCreator"
                for plugin_name in self.plugin_names
            ],
                            headers=[
                                f"{plugin_name}/_generate_trt_plugin/plugin.h"
                                for plugin_name in self.plugin_names
                            ])

            print('tpl_data', tpl_data)

            content = env.get_template("tritonPlugins.cpp.tpl").render(tpl_data)
            f.write(content)


@dataclass
class PluginCmakeCodegen:
    ''' Generate the CMakeLists.txt for a Triton plugin.  '''

    out_path: str
    workspace: str
    plugin_names: List[str]
    kernel_names: List[str]

    def generate(self):
        with open(self.out_path, "w") as f:
            env = setup_jinja_env()

            kernel_object_files = []
            for kernel_name in self.kernel_names:
                path = f"{kernel_name}/_triton_aot/*.c"
                kernel_object_files += glob.glob(path, root_dir=self.workspace)

            assert kernel_object_files

            tpl_data = dict(plugin_lib='triton_plugins',
                            plugin_names=self.plugin_names,
                            kernel_names=self.kernel_names,
                            workspace=self.workspace,
                            kernel_object_files=kernel_object_files,
                            **_render_common_parameters())

            content = env.get_template("CMakeLists.txt.tpl").render(tpl_data)
            f.write(content)


def copy_common_files(out_path: str):
    out_path = Path(out_path)
    files_to_copy = [
        Path("templates/plugin_common.cpp"),
        Path("templates/plugin_common.h")
    ]
    for file in files_to_copy:
        shutil.copy(cdir / file, out_path)


def setup_jinja_env() -> jinja2.Environment:
    env = jinja2.Environment(loader=jinja2.PackageLoader(
        package_name="tensorrt_llm.tools.plugin_gen",
        package_path="templates",
    ),
                             undefined=jinja2.StrictUndefined,
                             autoescape=jinja2.select_autoescape())
    env.variable_start_string = '[['
    env.variable_end_string = ']]'
    return env


def code(*lines):
    return "\n".join(lines)
