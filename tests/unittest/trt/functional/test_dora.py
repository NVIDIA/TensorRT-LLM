import itertools
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch
from parameterized import parameterized
from utils.util import create_session, run_session

import tensorrt_llm
from tensorrt_llm._common import deserialize_engine, serialize_engine
from tensorrt_llm._utils import torch_to_numpy
from tensorrt_llm.functional import Tensor, dora_plugin
from tensorrt_llm.layers.linear import Linear
from tensorrt_llm.layers.lora import Dora, Lora, LoraRuntimeParams
from tensorrt_llm.runtime.session import Session


@dataclass(repr=False, frozen=True)
class Scenario:
    num_loras: int
    num_modules: int
    dtype: str
    remove_input_padding: bool = False

    save_and_load: bool = False

    def __repr__(self) -> str:
        s = f"num_loras-{self.num_loras}_num_modules-{self.num_modules}_dtype-{self.dtype}_remove_input_padding-{self.remove_input_padding}"
        if self.save_and_load:
            s += "_save_and_load"
        return s


def get_scenarios() -> list[Scenario]:
    scenarios = []

    for num_loras, num_modules, dtype, remove_input_padding in itertools.product(
        [0, 1, 2, 3], [1, 2, 3], ["float16", "bfloat16"], [False]):
        scenarios.append(
            Scenario(num_loras=num_loras,
                     num_modules=num_modules,
                     dtype=dtype,
                     remove_input_padding=remove_input_padding))

    scenarios.append(
        Scenario(num_loras=1,
                 num_modules=1,
                 dtype="float16",
                 save_and_load=True))

    return scenarios


def name_func(testcase_func, param_num, param) -> str:
    scenario = param.args[0]
    return f"{testcase_func.__name__}_{scenario}"


class TestDoRA(unittest.TestCase):

    def create_dora_trt_session(self, batch_size: int, seq_len: int,
                                out_hidden_sizes: list[int], dtype: str,
                                remove_input_padding: bool) -> Session:
        builder = tensorrt_llm.Builder()
        network = builder.create_network()

        with tensorrt_llm.net_guard(network):
            network.plugin_config.lora_plugin = dtype
            network.plugin_config.remove_input_padding = remove_input_padding

            activations = Tensor(
                name="activations",
                shape=[batch_size, seq_len,
                       sum(out_hidden_sizes)],
                dtype=tensorrt_llm.str_dtype_to_trt(dtype))

            lora_weights_pointers = [
                Tensor(name=f"lora_weights_pointers_{i}",
                       shape=[batch_size, 3],
                       dtype=trt.int64) for i in range(len(out_hidden_sizes))
            ]

            host_request_types = Tensor(name="host_request_types",
                                        shape=[batch_size],
                                        dtype=trt.int32)

            output = dora_plugin(activations, out_hidden_sizes,
                                 lora_weights_pointers, host_request_types,
                                 None)

            output.mark_output("output", tensorrt_llm.str_dtype_to_trt(dtype))

        session = create_session(builder,
                                 network,
                                 precision=tensorrt_llm.str_dtype_to_trt(dtype))

        return session

    def create_linear_dora_trt_session(self, batch_size: int, seq_len: int,
                                       in_size: int, out_size: int,
                                       lora_rank: int, dtype: str,
                                       weight: torch.Tensor) -> Session:
        builder = tensorrt_llm.Builder()
        network = builder.create_network()

        with tensorrt_llm.net_guard(network):
            network.plugin_config.lora_plugin = dtype
            network.plugin_config.remove_input_padding = False
            network.plugin_config.dora_plugin = True

            linear = Linear(in_features=in_size,
                            out_features=out_size,
                            dtype=dtype,
                            bias=False)
            linear.lora = Lora(in_hidden_size=in_size,
                               out_hidden_sizes=[out_size],
                               max_low_rank=lora_rank)
            linear.dora = Dora(out_hidden_sizes=[out_size])
            linear.weight.value = np.ascontiguousarray(
                torch_to_numpy(weight.cpu()))

            inp = Tensor(name="input_tensor",
                         shape=[batch_size, seq_len, in_size],
                         dtype=tensorrt_llm.str_dtype_to_trt(dtype))

            lora_weights_pointers = Tensor(name=f"lora_weights_pointers",
                                           shape=[batch_size, 3],
                                           dtype=trt.int64)

            host_request_types = Tensor(name="host_request_types",
                                        shape=[batch_size],
                                        dtype=trt.int32)

            lora_ranks = Tensor(name="lora_ranks",
                                shape=(batch_size, ),
                                dtype=trt.int32)

            lora_params = LoraRuntimeParams(
                lora_ranks=[lora_ranks],
                lora_weights_pointers=[lora_weights_pointers],
                host_request_types=host_request_types,
                weight_index=0)

            output = linear(inp, lora_runtime_params=lora_params)
            output.mark_output("output", dtype)

        session = create_session(builder, network, precision=dtype)
        return session

    @parameterized.expand(get_scenarios, name_func=name_func)
    @torch.no_grad()
    def test_dora_scaling(self, scenario: Scenario) -> None:
        num_loras = scenario.num_loras
        num_modules = scenario.num_modules
        dtype = scenario.dtype
        remove_input_padding = scenario.remove_input_padding

        batch_size = 16
        seq_len = 32
        hidden_size = 1024

        activations = torch.randn(batch_size,
                                  seq_len,
                                  hidden_size * num_modules,
                                  dtype=tensorrt_llm.str_dtype_to_torch(dtype),
                                  device="cuda")

        magnitudes = torch.randn(num_loras,
                                 num_modules,
                                 hidden_size,
                                 dtype=tensorrt_llm.str_dtype_to_torch(dtype),
                                 device="cuda").unbind(dim=0)

        lora_uids = torch.randint(low=-1, high=num_loras, size=(batch_size, ))

        host_request_types = torch.zeros(batch_size, dtype=torch.int32)

        inputs = {
            "activations": activations,
            "host_request_types": host_request_types
        }

        for module_id in range(num_modules):
            module_ptrs = [
                magnitudes[lora_uid][module_id].data_ptr()
                if lora_uid >= 0 else 0 for lora_uid in lora_uids.tolist()
            ]

            inputs[f"lora_weights_pointers_{module_id}"] = torch.tensor(
                [[0, 0, ptr] for ptr in module_ptrs], dtype=torch.int64)

        session = self.create_dora_trt_session(batch_size, seq_len,
                                               [hidden_size] * num_modules,
                                               dtype, remove_input_padding)

        if scenario.save_and_load:
            # verify we can serialize and deserialize an engine that uses the dora plugin
            with tempfile.TemporaryDirectory() as temp_dir:
                engine_path = Path(temp_dir) / "engine"
                serialize_engine(session.engine, engine_path.as_posix())
                engine = deserialize_engine(engine_path.as_posix())
                session = Session.from_engine(engine)

        outputs = run_session(session, inputs)
        torch.cuda.synchronize()

        # generate ref
        expanded_magnitudes = torch.stack(list(magnitudes) + [
            torch.ones(num_modules,
                       hidden_size,
                       dtype=tensorrt_llm.str_dtype_to_torch(dtype),
                       device="cuda")
        ],
                                          dim=0)
        flat_magnitudes = expanded_magnitudes.view(num_loras + 1, -1)
        ref = activations * flat_magnitudes[lora_uids].unsqueeze(1)

        torch.testing.assert_close(outputs["output"], ref)

    @torch.no_grad()
    def test_dora_linear_layer(self) -> None:
        dtype = "float16"

        batch_size = 16
        seq_len = 32
        hidden_size = 1024
        lora_rank = 8
        save_and_load = True

        activations = torch.randn(batch_size,
                                  seq_len,
                                  hidden_size,
                                  dtype=tensorrt_llm.str_dtype_to_torch(dtype),
                                  device="cuda")

        weight = torch.randn(hidden_size,
                             hidden_size,
                             dtype=tensorrt_llm.str_dtype_to_torch(dtype),
                             device="cuda")
        A = torch.randn(lora_rank,
                        hidden_size,
                        dtype=tensorrt_llm.str_dtype_to_torch(dtype),
                        device="cuda")
        B = torch.randn(hidden_size,
                        lora_rank,
                        dtype=tensorrt_llm.str_dtype_to_torch(dtype),
                        device="cuda")

        magnitude = torch.randn(hidden_size,
                                dtype=tensorrt_llm.str_dtype_to_torch(dtype),
                                device="cuda")

        host_request_types = torch.zeros(batch_size, dtype=torch.int32)

        inputs = {
            "input_tensor": activations,
            "host_request_types": host_request_types
        }

        weights_ptrs = torch.tensor(
            [[A.data_ptr(), B.data_ptr(),
              magnitude.data_ptr()] for _ in range(batch_size)],
            dtype=torch.int64)

        inputs["lora_weights_pointers"] = weights_ptrs

        inputs["lora_ranks"] = torch.tensor([lora_rank] * batch_size,
                                            dtype=torch.int32)

        session = self.create_linear_dora_trt_session(batch_size, seq_len,
                                                      hidden_size, hidden_size,
                                                      lora_rank, dtype, weight)

        if save_and_load:
            # verify we can serialize and deserialize an engine that uses the dora plugin
            with tempfile.TemporaryDirectory() as temp_dir:
                engine_path = Path(temp_dir) / "engine"
                serialize_engine(session.engine, engine_path.as_posix())
                engine = deserialize_engine(engine_path.as_posix())
                session = Session.from_engine(engine)

        outputs = run_session(session, inputs)
        torch.cuda.synchronize()

        # generate ref
        ref = activations @ weight.T + (activations @ A.T) @ B.T
        ref = ref * magnitude

        torch.testing.assert_close(outputs["output"], ref)
