from pathlib import Path

import torch
from plugin_lib import LookUpPlugin

import tensorrt_llm
from tensorrt_llm import Tensor
from tensorrt_llm._utils import torch_dtype_to_str, torch_dtype_to_trt

if __name__ == "__main__":

    # meta data
    batch_size = 10
    vocab_size = 1000
    n_embed = 1024

    # test data
    ## input index
    index_shape = (batch_size, )
    index_data = torch.randint(0, vocab_size, index_shape,
                               dtype=torch.int32).cuda()

    def test(dtype):
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
                lookup_plugin = LookUpPlugin(False, True)
                return lookup_plugin(x, y)

            output = lookup(x, y)

            output.mark_output("output", torch_dtype_to_str(torch.float32))

        builder_config = builder.create_builder_config("float32")
        engine = builder.build_engine(network, builder_config)
        assert engine is not None

        output_dir = Path("tmp") / torch_dtype_to_str(dtype)
        output_dir.mkdir(parents=True, exist_ok=True)

        engine_path = output_dir / "lookup.engine"
        config_path = output_dir / "config.json"

        with engine_path.open("wb") as f:
            f.write(engine)
        builder.save_config(builder_config, str(config_path))

    test(torch.bfloat16)
    test(torch.float16)
    test(torch.float32)
