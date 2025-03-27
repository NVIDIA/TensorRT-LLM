from typing import Sequence

import pytest
import torch
from plugin_wrapper_utils import DummyPlugin
from python_plugin.plugin_lib import LookUpPlugin
from utils.util import create_session, run_session

import tensorrt_llm
from tensorrt_llm import PluginBase, Tensor
from tensorrt_llm._utils import (TensorWrapper, torch_dtype_to_str,
                                 torch_dtype_to_trt)
from tensorrt_llm.python_plugin import SymTensor, trtllm_plugin


@pytest.fixture(scope="function", autouse=True)
def use_cuda_as_default_device():
    old_level = tensorrt_llm.logger.level
    old_device = torch.get_default_device()
    tensorrt_llm.logger.set_level("verbose")
    torch.set_default_device("cuda")

    try:
        yield
    finally:
        torch.set_default_device(old_device)
        tensorrt_llm.logger.set_level(old_level)


@pytest.mark.parametrize("dtype",
                         [torch.float32, torch.bfloat16, torch.float16],
                         ids=torch_dtype_to_str)
@pytest.mark.parametrize("to_torch", [False, True])
@pytest.mark.parametrize("fp32_output", [False, True])
def test_triton_plugin(dtype, to_torch, fp32_output):
    # meta data
    batch_size = 10
    vocab_size = 1000
    n_embed = 1024

    # test data
    ## input index
    index_shape = (batch_size, )
    index_data = torch.randint(0, vocab_size, index_shape, dtype=torch.int32)
    weight_data = torch.rand(vocab_size, n_embed, dtype=dtype)
    embedding = torch.nn.Embedding.from_pretrained(weight_data)
    trt_plugin = LookUpPlugin(to_torch, fp32_output)

    builder = tensorrt_llm.Builder()
    builder.strongly_typed = True
    network = builder.create_network()
    with tensorrt_llm.net_guard(network):
        x = Tensor(
            name="x",
            shape=index_shape,
            dtype=tensorrt_llm.str_dtype_to_trt("int32"),
        )
        y = Tensor(name="y",
                   shape=(vocab_size, n_embed),
                   dtype=torch_dtype_to_trt(dtype))

        def lookup(x, y):
            lookup_plugin = LookUpPlugin(to_torch, fp32_output)
            return lookup_plugin(x, y)

        output = lookup(x, y)

        output.mark_output(
            "output",
            torch_dtype_to_str(dtype if not fp32_output else torch.float32))

    session = create_session(builder,
                             network,
                             precision=torch_dtype_to_str(dtype))

    input_dict = {"x": index_data, "y": weight_data}
    trt_out = run_session(session, input_dict)["output"]
    trt_plugin_out = trt_plugin(index_data, weight_data)
    torch_out = embedding(index_data)

    if fp32_output:
        torch_out = torch_out.to(torch.float32)

    torch.testing.assert_close(trt_out, torch_out)
    torch.testing.assert_close(trt_plugin_out, torch_out)


def test_redefinition():

    class Plugin(PluginBase):

        def __init__(self):
            super().__init__()

        def shape_dtype_inference(self,
                                  inputs: Sequence[SymTensor]) -> SymTensor:
            return inputs[0]

        def forward(self, inputs: Sequence[TensorWrapper],
                    outputs: Sequence[TensorWrapper]):
            pass

    trtllm_plugin("Plugin")(Plugin)

    with pytest.raises(AssertionError):
        trtllm_plugin("Plugin")(Plugin)

    with pytest.raises(AssertionError):

        @trtllm_plugin("Plugin")
        class PluginRedefine(PluginBase):

            def __init__(self):
                super().__init__()

            def shape_dtype_inference(
                    self,
                    inputs: Sequence[SymTensor]) -> tuple[SymTensor, SymTensor]:
                return inputs[0], inputs[0]

            def forward(self, inputs: Sequence[TensorWrapper],
                        outputs: Sequence[TensorWrapper]):
                pass


def test_no_register():

    class NoRegisterNoOutputNumPlugin(PluginBase):

        def __init__(self):
            super().__init__()

        def shape_dtype_inference(self,
                                  inputs: Sequence[SymTensor]) -> SymTensor:
            return inputs[0]

        def forward(self, inputs: Sequence[TensorWrapper],
                    outputs: Sequence[TensorWrapper]):
            pass

    with pytest.raises(AssertionError):
        NoRegisterNoOutputNumPlugin()

    with pytest.raises(AssertionError):

        @trtllm_plugin("UtilsPlugin")
        class NoOutputNumPlugin(PluginBase):

            def __init__(self):
                super().__init__()

            def shape_dtype_inference(self, inputs: Sequence[SymTensor]):
                return inputs[0]

            def forward(self, inputs: Sequence[TensorWrapper],
                        outputs: Sequence[TensorWrapper]):
                pass


def test_single_creator():
    a = DummyPlugin()
    b = DummyPlugin()
    assert a._plugin_creator is b._plugin_creator
