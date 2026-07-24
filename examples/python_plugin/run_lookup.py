from pathlib import Path

import torch

from tensorrt_llm import logger
from tensorrt_llm._utils import (torch_dtype_to_str, torch_dtype_to_trt,
                                 trt_dtype_to_torch)
from tensorrt_llm.runtime.session import Session, TensorInfo

import plugin_lib  # isort: skip

if __name__ == "__main__":

    def run_engine(dtype):
        output_dir = Path('tmp') / torch_dtype_to_str(dtype)

        engine_path = output_dir / "lookup.engine"

        with engine_path.open('rb') as f:
            session = Session.from_serialized_engine(f.read())

        # meta data
        batch_size = 10
        vocab_size = 1000
        n_embed = 1024

        # test data
        ## input index
        index_shape = (batch_size, )
        index_data = torch.randint(0,
                                   vocab_size,
                                   index_shape,
                                   dtype=torch.int32).cuda()
        weight_data = torch.rand(vocab_size, n_embed, dtype=dtype).cuda()

        inputs = {"x": index_data, "y": weight_data}

        output_info = session.infer_shapes([
            TensorInfo(name, torch_dtype_to_trt(tensor.dtype), tensor.shape)
            for name, tensor in inputs.items()
        ])
        logger.debug(f'output info {output_info}')
        outputs = {
            t.name:
            torch.empty(tuple(t.shape),
                        dtype=trt_dtype_to_torch(t.dtype),
                        device='cuda')
            for t in output_info
        }

        stream = torch.cuda.Stream()
        ok = session.run(inputs=inputs,
                         outputs=outputs,
                         stream=stream.cuda_stream)
        assert ok, 'Engine execution failed'

        embedding = torch.nn.Embedding.from_pretrained(weight_data)
        torch_out = embedding(index_data).to(torch.float32)
        trt_out = outputs['output']

        torch.testing.assert_close(trt_out, torch_out)

    run_engine(torch.bfloat16)
    run_engine(torch.float16)
    run_engine(torch.float32)
