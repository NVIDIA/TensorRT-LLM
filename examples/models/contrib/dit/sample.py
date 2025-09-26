import argparse
import json
import os
from functools import wraps

import tensorrt as trt
import torch
from cuda import cudart
from diffusion import DiTDiffusionPipeline
from torchvision.utils import save_image

import tensorrt_llm
from tensorrt_llm._utils import str_dtype_to_torch, trt_dtype_to_torch
from tensorrt_llm.logger import logger
from tensorrt_llm.plugin.plugin import CustomAllReduceHelper
from tensorrt_llm.runtime.session import Session, TensorInfo


def CUASSERT(cuda_ret):
    err = cuda_ret[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(
            f"CUDA ERROR: {err}, error code reference: https://nvidia.github.io/cuda-python/module/cudart.html#cuda.cudart.cudaError_t"
        )
    if len(cuda_ret) > 1:
        return cuda_ret[1:]
    return None


class TllmDiT(object):

    def __init__(self,
                 config,
                 debug_mode=True,
                 stream: torch.cuda.Stream = None):
        self.dtype = config['pretrained_config']['dtype']

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

        expected_tensor_names = ['latent', 'timestep', 'label', 'output']

        if self.mapping.tp_size > 1:
            self.buffer, self.all_reduce_workspace = CustomAllReduceHelper.allocate_workspace(
                self.mapping,
                CustomAllReduceHelper.max_workspace_size_auto(
                    self.mapping.tp_size))
            self.inputs['all_reduce_workspace'] = self.all_reduce_workspace
            expected_tensor_names += ['all_reduce_workspace']

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

    def _setup(self, batch_size):
        for i in range(self.session.engine.num_io_tensors):
            name = self.session.engine.get_tensor_name(i)
            if self.session.engine.get_tensor_mode(
                    name) == trt.TensorIOMode.OUTPUT:
                shape = list(self.session.engine.get_tensor_shape(name))
                shape[0] = batch_size // 2 if name in [
                    'cond_eps', 'uncond_eps'
                ] else batch_size
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

    @cuda_stream_guard
    def forward(self, latent: torch.Tensor, timestep: torch.Tensor,
                label: torch.Tensor):
        """
        Forward pass of DiT.
        latent: (N, C, H, W)
        timestep: (N,)
        label: (N,)
        """
        self._setup(latent.shape[0])
        if not self.buffer_allocated:
            raise RuntimeError('Buffer not allocated, please call setup first!')

        inputs = {
            'latent': latent.to(str_dtype_to_torch(self.dtype)),
            "timestep": timestep.int(),
            "label": label.int()
        }
        self.inputs.update(**inputs)
        self.session.set_shapes(self.inputs)
        ok = self.session.run(self.inputs, self.outputs,
                              self.stream.cuda_stream)

        if not ok:
            raise RuntimeError('Executing TRT engine failed!')
        if self.debug_mode:
            torch.cuda.synchronize()
            for k, v in self.inputs.items():
                print(k, v.sum())
            for k, v in self.outputs.items():
                print(k, v.sum())
        return self.outputs['output']


def vae_decode(samples, engine_path):
    # Load standard plugins
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt.init_libnvinfer_plugins(TRT_LOGGER, namespace="")

    logger.info(f'Loading vae engine from {engine_path}')
    with open(engine_path, 'rb') as f:
        engine_buffer = f.read()
    logger.info(f'Creating session from engine {engine_path}')
    session_vae = Session.from_serialized_engine(engine_buffer)
    inputs = {'input': samples}
    output_info = session_vae.infer_shapes(
        [TensorInfo('input', trt.DataType.FLOAT, samples.shape)])
    outputs = {
        t.name:
        torch.empty(tuple(t.shape),
                    dtype=trt_dtype_to_torch(t.dtype),
                    device='cuda')
        for t in output_info
    }
    stream = torch.cuda.current_stream().cuda_stream
    ok = session_vae.run(inputs, outputs, stream)

    assert ok, "Runtime execution failed for vae session"

    samples = outputs['output']
    return samples


def main(args):
    tensorrt_llm.logger.set_level(args.log_level)

    torch.manual_seed(args.seed)
    assert torch.cuda.is_available()
    device = "cuda"

    # Load model:
    config_file = os.path.join(args.tllm_model_dir, 'config.json')
    with open(config_file) as f:
        config = json.load(f)
    model = TllmDiT(config, debug_mode=args.debug_mode)

    diffusion = DiTDiffusionPipeline(model.forward,
                                     timestep_respacing=args.num_sampling_steps)

    latent_size = args.image_size // 8
    latent = torch.randn(args.batch_size,
                         4,
                         latent_size,
                         latent_size,
                         device=device)
    labels = torch.randint(args.num_classes, [args.batch_size], device=device)

    latent = torch.cat([latent, latent], 0)
    labels_null = torch.tensor([1000] * args.batch_size, device=device)
    labels = torch.cat([labels, labels_null], 0)

    samples = diffusion.run(latent, labels)
    samples, _ = samples.chunk(2, dim=0)

    samples = vae_decode(samples / 0.18215, args.vae_decoder_engine)

    save_image(samples,
               "sample.png",
               nrow=4,
               normalize=True,
               value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vae_decoder_engine',
                        type=str,
                        default='vae_decoder/plan/visual_encoder_fp16.plan',
                        help='')
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size",
                        type=int,
                        choices=[256, 512],
                        default=512)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tllm_model_dir",
                        type=str,
                        default='./engine_outputs/')
    parser.add_argument("--gpus_per_node", type=int, default=8)
    parser.add_argument('--log_level', type=str, default='info')
    parser.add_argument("--debug_mode", type=bool, default=False)
    args = parser.parse_args()
    main(args)
