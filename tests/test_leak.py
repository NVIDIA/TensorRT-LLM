import unittest

import tensorrt_llm as tllm
import tensorrt_llm.profiler as profiler

import psutil  # isort:skip


def create_model():
    ''' Lots of parameters are created here, and thus memory increases
    '''
    profiler.print_memory_usage('Before creating Module')
    # About 24GiB model size, big enough to detect leak and avoid noise and false positive
    # and small enough to make sure CI single-gpu machine can run it.
    model = tllm.models.LLaMAForCausalLM(
        num_layers=2,
        num_heads=80,
        num_kv_heads=80,
        hidden_size=12800,
        vocab_size=50000,
        hidden_act='silu',
        max_position_embeddings=2048,
        dtype='float32',
    )
    profiler.print_memory_usage('After creating Module')
    return model


def create_optimize_network():
    builder = tllm.Builder()
    model = create_model()
    network = builder.create_network()
    network.plugin_config.set_gpt_attention_plugin(dtype='float16')
    profiler.print_memory_usage('Before creating Network')
    with tllm.net_guard(network):
        # Forward
        inputs = model.prepare_inputs(max_batch_size=1,
                                      max_input_len=1024,
                                      max_seq_len=1024 + 32,
                                      use_cache=True,
                                      max_beam_width=1)
        model(*inputs)
    profiler.print_memory_usage('After creating Network')

    # When the Network has gpt attention plugin layer, graph rewriting pattern matching is triggered,
    # thus the Network._get_graph_impl will be called, and a lru_cache will be created to cache this Network object
    # and thus these registered ndarrays inside the Network, these objects are destroyed only when the cache is full or the
    # program ends
    tllm.graph_rewriting.optimize(network)


def run():
    # Create a TRT builder to warm up the memory, and avoid the noise of leak detection.
    # Builder creation will create global objects like kernels.
    _ = tllm.Builder()

    used, _, _ = profiler.host_memory_info()

    for i in range(5):
        # Ideally the memory used inside create_optimize_network will all be released after the function returns
        profiler.print_memory_usage(f'create_optimize_network {i} started')
        create_optimize_network()
        profiler.print_memory_usage(f'create_optimize_network {i} returned')

        used_after, _, _ = profiler.host_memory_info()
        mem_increase_in_gb = (used_after - used) / (1024**3)
        # The model has more than 10GB, so if there is leak, it will be absolutely bigger than 1GB
        assert mem_increase_in_gb < 1, f"Memory increased {mem_increase_in_gb} GB"


class TestHostMemLeak(unittest.TestCase):

    def test_host_mem_leak(self):
        tllm.logger.set_level('info')
        run()


if __name__ == '__main__':
    unittest.main()
