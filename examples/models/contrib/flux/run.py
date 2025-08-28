import argparse
import json
import os
from functools import wraps

import tensorrt as trt
import torch
from cuda import cudart
from diffusers import FluxPipeline
from diffusers.models.modeling_outputs import Transformer2DModelOutput

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_torch, trt_dtype_to_torch
from tensorrt_llm.logger import logger
from tensorrt_llm.models.flux.config import FluxConfig
from tensorrt_llm.runtime.session import Session


def CUASSERT(cuda_ret):
    err = cuda_ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t"
        )
    if len(cuda_ret) > 1:
        return cuda_ret[1:]
    return None


class TllmFlux(object):

    def __init__(self,
                 config,
                 debug_mode=True,
                 stream: torch.cuda.Stream = None):
        self.dtype = config['pretrained_config']['dtype']
        self.config = FluxConfig.from_dict(config['pretrained_config'])

        rank = tensorrt_llm.mpi_rank()
        world_size = config['pretrained_config']['mapping']['world_size']
        cp_size = config['pretrained_config']['mapping']['cp_size']
        tp_size = config['pretrained_config']['mapping']['tp_size']
        pp_size = config['pretrained_config']['mapping']['pp_size']
        assert pp_size == 1
        self.mapping = tensorrt_llm.Mapping(world_size=world_size,
                                            rank=rank,
                                            cp_size=cp_size,
                                            tp_size=tp_size,
                                            pp_size=1,
                                            gpus_per_node=args.gpus_per_node)

        local_rank = rank % self.mapping.gpus_per_node
        self.device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(self.device)
        CUASSERT(cudart.cudaSetDevice(local_rank))

        self.stream = stream
        if self.stream is None:
            self.stream = torch.cuda.Stream(self.device)
        torch.cuda.set_stream(self.stream)

        engine_file = os.path.join(args.tllm_model_dir, f"rank{rank}.engine")
        logger.info(f'Loading engine from {engine_file}')
        with open(engine_file, "rb") as f:
            engine_buffer = f.read()

        assert engine_buffer is not None

        self.session = Session.from_serialized_engine(engine_buffer)

        self.debug_mode = debug_mode

        self.inputs = {}
        self.outputs = {}
        self.buffer_allocated = False

        expected_tensor_names = [
            'hidden_states', 'encoder_hidden_states', 'pooled_projections',
            'timestep', 'img_ids', 'txt_ids', 'guidance', 'output'
        ]

        found_tensor_names = [
            self.session.engine.get_tensor_name(i)
            for i in range(self.session.engine.num_io_tensors)
        ]
        if not self.debug_mode and set(expected_tensor_names) != set(
                found_tensor_names):
            logger.error(
                f"The following expected tensors are not found: {set(expected_tensor_names).difference(set(found_tensor_names))}"
            )
            logger.error(
                f"Those tensors in engine are not expected: {set(found_tensor_names).difference(set(expected_tensor_names))}"
            )
            logger.error(f"Expected tensor names: {expected_tensor_names}")
            logger.error(f"Found tensor names: {found_tensor_names}")
            raise RuntimeError(
                "Tensor names in engine are not the same as expected.")
        if self.debug_mode:
            self.debug_tensors = list(
                set(found_tensor_names) - set(expected_tensor_names))

    def _tensor_dtype(self, name):
        # return torch dtype given tensor name for convenience
        dtype = trt_dtype_to_torch(self.session.engine.get_tensor_dtype(name))
        return dtype

    def _setup(self, outputs_shape):
        for i in range(self.session.engine.num_io_tensors):
            name = self.session.engine.get_tensor_name(i)
            if self.session.engine.get_tensor_mode(
                    name) == trt.TensorIOMode.OUTPUT:
                shape = list(self.session.engine.get_tensor_shape(name))
                if self.debug_mode:
                    shape = list(self.session.engine.get_tensor_shape(name))
                    if shape[0] == -1:
                        shape[0] = outputs_shape['output'][0]
                else:
                    shape = outputs_shape[name]
                self.outputs[name] = torch.empty(shape,
                                                 dtype=self._tensor_dtype(name),
                                                 device=self.device)

        self.buffer_allocated = True

    def cuda_stream_guard(func):
        """Sync external stream and set current stream to the one bound to the session. Reset on exit.
        """

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            external_stream = torch.cuda.current_stream()
            if external_stream != self.stream:
                external_stream.synchronize()
                torch.cuda.set_stream(self.stream)
            ret = func(self, *args, **kwargs)
            if external_stream != self.stream:
                self.stream.synchronize()
                torch.cuda.set_stream(external_stream)
            return ret

        return wrapper

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @cuda_stream_guard
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs=None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ):
        self._setup(outputs_shape={'output': hidden_states.shape})
        if not self.buffer_allocated:
            raise RuntimeError('Buffer not allocated, please call setup first!')
        if controlnet_block_samples is not None or controlnet_single_block_samples is not None:
            raise NotImplementedError()
        if joint_attention_kwargs:
            raise NotImplementedError()
        if controlnet_blocks_repeat:
            raise NotImplementedError()
        inputs = {
            'hidden_states':
            hidden_states.to(str_dtype_to_torch(self.dtype)),
            'encoder_hidden_states':
            encoder_hidden_states.to(str_dtype_to_torch(self.dtype)),
            'pooled_projections':
            pooled_projections.to(str_dtype_to_torch(self.dtype)),
            'timestep':
            timestep.to(str_dtype_to_torch(self.dtype)),
            'img_ids':
            img_ids.to(str_dtype_to_torch(self.dtype)),
            'txt_ids':
            txt_ids.to(str_dtype_to_torch(self.dtype)),
            'guidance':
            guidance.float()
        }
        for k, v in inputs.items():
            inputs[k] = v.cuda().contiguous()
        self.inputs.update(**inputs)
        self.session.set_shapes(self.inputs)
        ok = self.session.run(self.inputs, self.outputs,
                              self.stream.cuda_stream)

        if not ok:
            raise RuntimeError('Executing TRT engine failed!')
        output = self.outputs['output']

        if self.debug_mode:
            torch.cuda.synchronize()
            for k, v in self.inputs.items():
                print(k, v.sum())
            for k, v in self.outputs.items():
                print(k, v.sum())
            output_np = {
                k: v.cpu().float().numpy()
                for k, v in self.outputs.items()
            }
            import numpy as np
            np.savez("tllm_output.npz", **output_np)

        if not return_dict:
            return (output, )
        else:
            return Transformer2DModelOutput(sample=output)


def main(args):
    tensorrt_llm.logger.set_level(args.log_level)

    assert torch.cuda.is_available()

    # Load model:
    config_file = os.path.join(args.tllm_model_dir, 'config.json')
    with open(config_file) as f:
        config = json.load(f)
    model = TllmFlux(config, debug_mode=args.debug_mode)

    pipe = FluxPipeline.from_pretrained("./FLUX.1-dev",
                                        torch_dtype=torch.bfloat16)

    # replace flux transformer with TRTLLM model
    del pipe.transformer
    torch.cuda.empty_cache()
    pipe.to("cuda")
    pipe.transformer = model

    image = pipe(args.prompt,
                 height=1024,
                 width=1024,
                 guidance_scale=3.5,
                 num_inference_steps=50,
                 max_sequence_length=512,
                 generator=torch.Generator("cpu").manual_seed(0)).images[0]
    if os.path.exists(args.save_path):
        os.remove(args.save_path)
    image.save("flux-dev.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tllm_model_dir",
                        type=str,
                        default='./engine_outputs/')
    parser.add_argument("--gpus_per_node", type=int, default=8)
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument('--prompt',
                        type=str,
                        default='A cat holding a sign that says hello world')
    parser.add_argument('--save_path', type=str, default='flux-dev.png')
    parser.add_argument("--debug_mode", type=bool, default=False)
    args = parser.parse_args()
    main(args)
