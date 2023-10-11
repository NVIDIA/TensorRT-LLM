import os

from tensorrt_llm.tools.plugin_gen.core import *

from .kernel_config import get_fmha_kernel_meta_data


def _mkdir(path: str):
    '''
    mkdir if not exists
    '''
    os.makedirs(path, exist_ok=True)


def test_Type():
    t0 = Type("tensor[fp16]")
    assert t0.is_tensor and t0.dtype == DType.FP16

    t1 = Type("i32")
    assert t1.is_scalar and t1.dtype == DType.INT32


def test_KernelMetaData_to_triton_signatures():
    signature = get_fmha_kernel_meta_data().to_triton_signatures()
    assert signature == [
        '*fp16:16, *fp32:16, *fp16:16, *fp16:16, *fp16:16, *fp16:16, fp32, i32, i32, i32, 128, 64, 128',
        '*fp16:16, *fp32:16, *fp16:16, *fp16:16, *fp16:16, *fp16:16, fp32, i32, i32, i32:16, 128, 64, 128'
    ]


OUT_DIR = './output'


def test_PluginCppCodegen():
    metadata = get_fmha_kernel_meta_data()
    _mkdir(OUT_DIR)
    codegen = PluginCppCodegen(output_dir=OUT_DIR, meta_data=metadata)
    codegen.generate()


def test_PluginPyCodegen():
    metadata = get_fmha_kernel_meta_data()
    # clean the output file
    out_file = os.path.join(OUT_DIR, "functional.py")
    with open(out_file, "w") as f:
        f.write("")

    codegen = PluginPyCodegen(out_path=out_file,
                              meta_data=metadata,
                              add_header=True,
                              plugin_lib_path="libtriton_fmha.so")

    codegen.generate()


def test_PluginRegistryCodegen():
    out_path = './_plugin_registry.cc'
    codegen = PluginRegistryCodegen(
        out_path=out_path, plugin_names=['kernel0', "kernel1", "kernel2"])
    codegen.generate()

    with open(out_path, 'r') as f:
        content = f.read()
        assert 'kernel0PluginCreator' in content
        assert 'kernel1PluginCreator' in content
        assert 'kernel2PluginCreator' in content
