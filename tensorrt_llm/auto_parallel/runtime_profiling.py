import numpy as np
import tensorrt as trt
import torch

from tensorrt_llm.logger import logger
from tensorrt_llm.network import get_plugin_info

from .shape_info import get_per_layer_graph
from .utils import get_cache_key, get_trt_network, get_updated_plugin


class NvtxProfiler(object):

    def __init__(self, nvtx_name, enable=True):
        self.nvtx_name = nvtx_name
        self.enable = enable

    def __enter__(self):
        if self.enable:
            torch.cuda.nvtx.range_push(self.nvtx_name)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.enable:
            torch.cuda.nvtx.range_pop()


class LayerProfiler(trt.IProfiler):

    def __init__(self):
        trt.IProfiler.__init__(self)
        self.layer_count = 0
        self.time = 0

    def report_layer_time(self, layer_name, ms):
        logger.debug(f'{layer_name=}, {self.layer_count=}, time = {ms} ms')
        self.time += ms
        self.layer_count += 1


class RuntimeProfiler(object):

    def __init__(self):
        self.timing_cache = None

    def _profile(self, layer, layer_attrs, shapes, values, io_buffer_mapping):
        is_plugin = layer.type == trt.LayerType.PLUGIN_V2
        if is_plugin and len(layer_attrs) > 0:
            plugin_info = get_plugin_info(
                get_trt_network(layer),
                layer.name,
            )
            new_plugin, _ = get_updated_plugin(plugin_info, layer_attrs)
            layer_attrs = {"plugin": new_plugin}
        graph, output_mapping = get_per_layer_graph(layer, shapes, values,
                                                    layer_attrs)
        graph._io_buffer_mapping = io_buffer_mapping
        network = graph.as_trt()
        if network.num_outputs > 0 and np.all([
                network.get_output(i).is_shape_tensor
                for i in range(network.num_outputs)
        ]):
            return 0.0
        for proxy_output, output in output_mapping.items():
            shapes[proxy_output] = shapes[output]
        if not self.timing_cache:
            self.timing_cache = network.builder.create_builder_config(
            ).create_timing_cache(b"")
        runner = graph.get_runner(
            shapes,
            values,
            timing_cache=self.timing_cache,
        )
        context = runner.session.context
        context.profiler = LayerProfiler()
        runner.run()
        profiler_time_first_run = context.profiler.time
        runner.run()
        return (context.profiler.time - profiler_time_first_run) * 1000.0

    def runtime_profile(self, layer, layer_attrs, input_values, strategy,
                        device_mesh):
        logger.debug(f"start to profile layer {layer.name}")
        shapes = {}
        values = {}
        dtypes = {}
        trt_layer = layer.as_trt()

        sharding_sequences = ()
        for i in range(layer.num_inputs):
            input = trt_layer.get_input(i)
            if input is not None:
                shapes[input.name] = strategy.sharding_specs[
                    f'input{i}'].get_sharded_shape_per_device()
                dtypes[input.name] = input.dtype
                sharding_sequences += (str(
                    strategy.sharding_specs[f"input{i}"].sharding_sequence), )
                if i in input_values:
                    values[input.name] = input_values[i]
                else:
                    value = layer.get_input(i).value
                    if value is not None:
                        values[input.name] = value
            else:
                sharding_sequences += (None, )

        for i in range(layer.num_outputs):
            output = trt_layer.get_output(i)
            if f'output{i}' in strategy.communication_actions:
                shapes[output.name] = strategy.communication_actions[
                    f'output{i}'].sharding_spec.get_sharded_shape_per_device()
            else:
                shapes[output.name] = strategy.sharding_specs[
                    f'output{i}'].get_sharded_shape_per_device()
            dtypes[output.name] = output.dtype
            sharding_sequences += (str(
                strategy.sharding_specs[f"output{i}"].sharding_sequence), )
        data_key = get_cache_key(
            trt_layer,
            shapes,
            values,
            dtypes=dtypes,
            updated_attrs=layer_attrs,
        )
        data_key += (sharding_sequences, )
        elapsed_time = device_mesh.prof_database.query(
            device_mesh.cluster_key,
            data_key,
        )
        if elapsed_time:
            logger.debug(
                f'runtime profiling cache hit {data_key}: {elapsed_time} us')
            return elapsed_time
        with NvtxProfiler(f'{layer.name}_{data_key}', enable=True):
            elapsed_time = self._profile(
                layer.as_trt(),
                layer_attrs,
                shapes,
                values,
                layer.graph._io_buffer_mapping,
            )
        logger.debug(
            f'runtime profiling cache miss {data_key}: {elapsed_time} us')

        device_mesh.prof_database.update(
            device_mesh.cluster_key,
            data_key,
            (elapsed_time, strategy.alpha_beta_cost),
        )

        return elapsed_time
